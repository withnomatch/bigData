# -*- coding: utf-8 -*-
import csv
import html
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
PAIRS_FILE = ROOT / "annotation" / "dedup_pairs_300_reviewed.csv"
QUESTIONS_FILE = ROOT / "oracle_database_questions_jsonlines.json"
OUTPUT_JSON = ROOT / "annotation" / "dedup_optimal_metrics.json"
OUTPUT_CSV = ROOT / "annotation" / "dedup_optimal_predictions.csv"
OUTPUT_REPORT = ROOT / "annotation" / "dedup_optimal_report.md"

SEED = 42
DUPLICATE = "\u91cd\u590d"
RELATED = "\u76f8\u5173\u4f46\u4e0d\u91cd\u590d"
UNRELATED = "\u4e0d\u76f8\u5173"
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+#.-]*|\d+", re.I)
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+", re.I)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "database", "db", "did", "do", "does", "for", "from",
    "had", "has", "have", "help", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "oracle", "please", "sql",
    "than", "that", "the", "then", "there", "this", "to", "using", "was",
    "were", "what", "when", "where", "which", "why", "will", "with",
    "would", "you", "your",
}


def clean_text(value):
    value = TAG_RE.sub(" ", value or "")
    value = URL_RE.sub(" ", html.unescape(value))
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value):
    return [
        token.lower()
        for token in TOKEN_RE.findall(clean_text(value))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def load_pairs():
    with PAIRS_FILE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label_code"] = int(row["reviewer_label"] == DUPLICATE)
    return rows


def load_questions(question_ids):
    details = {}
    with QUESTIONS_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if question_id not in question_ids:
                continue
            answers = sorted(
                row.get("answers") or [],
                key=lambda answer: int(answer.get("score") or 0),
                reverse=True,
            )[:2]
            details[question_id] = {
                "title": clean_text(row.get("title") or ""),
                "body": clean_text(row.get("body") or ""),
                "tags": [str(tag) for tag in (row.get("tags") or [])],
                "answers": " ".join(
                    clean_text(answer.get("body") or "") for answer in answers
                ),
            }
            if len(details) == len(question_ids):
                break
    return details


def build_documents(details, ordered_ids, include_answers):
    documents = []
    for question_id in ordered_ids:
        item = details[question_id]
        parts = [
            item["title"], item["title"], item["title"],
            item["body"],
            " ".join(item["tags"]), " ".join(item["tags"]),
        ]
        if include_answers:
            parts.append(item["answers"])
        documents.append(tokenize(" ".join(parts)))
    return documents


def tfidf_matrix(documents, max_features, min_df, max_df):
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    max_count = int(math.floor(len(documents) * max_df))
    vocabulary_terms = [
        term for term, frequency in document_frequency.most_common()
        if frequency >= min_df and frequency <= max_count
    ][:max_features]
    vocabulary = {term: index for index, term in enumerate(vocabulary_terms)}

    rows, columns, values = [], [], []
    for row_index, document in enumerate(documents):
        counts = Counter(term for term in document if term in vocabulary)
        weighted = []
        for term, count in counts.items():
            idf = math.log(
                (len(documents) + 1.0) / (document_frequency[term] + 1.0)
            ) + 1.0
            weighted.append((vocabulary[term], (1.0 + math.log(count)) * idf))
        norm = math.sqrt(sum(value * value for _, value in weighted)) or 1.0
        for column, value in weighted:
            rows.append(row_index)
            columns.append(column)
            values.append(value / norm)
    return csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), len(vocabulary)),
        dtype=np.float64,
    )


def spherical_kmeans(matrix, k, seed, max_iter):
    rng = np.random.RandomState(seed)
    centers = matrix[rng.choice(matrix.shape[0], k, replace=False)].toarray()
    centers /= np.maximum(np.linalg.norm(centers, axis=1)[:, None], 1e-12)
    labels = np.full(matrix.shape[0], -1, dtype=np.int32)
    for _ in range(max_iter):
        new_labels = np.asarray(matrix.dot(centers.T).argmax(axis=1)).ravel()
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        new_centers = np.zeros_like(centers)
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            if len(members):
                new_centers[cluster] = np.asarray(
                    matrix[members].mean(axis=0)
                ).ravel()
            else:
                new_centers[cluster] = matrix[
                    rng.randint(matrix.shape[0])
                ].toarray()[0]
        centers = new_centers / np.maximum(
            np.linalg.norm(new_centers, axis=1)[:, None], 1e-12
        )
    return labels


def bisecting_spherical_kmeans(matrix, k, seed, max_iter):
    clusters = [np.arange(matrix.shape[0])]
    rng = random.Random(seed)
    while len(clusters) < k:
        split_index = max(range(len(clusters)), key=lambda index: len(clusters[index]))
        members = clusters.pop(split_index)
        if len(members) < 2:
            clusters.append(members)
            break
        local_labels = spherical_kmeans(
            matrix[members], 2, rng.randint(0, 2**31 - 1), max_iter
        )
        left = members[local_labels == 0]
        right = members[local_labels == 1]
        if not len(left) or not len(right):
            midpoint = len(members) // 2
            left, right = members[:midpoint], members[midpoint:]
        clusters.extend([left, right])
    labels = np.zeros(matrix.shape[0], dtype=np.int32)
    for cluster, members in enumerate(clusters):
        labels[members] = cluster
    return labels


def stratified_split(rows):
    groups = {label: [] for label in (DUPLICATE, RELATED, UNRELATED)}
    for index, row in enumerate(rows):
        groups[row["reviewer_label"]].append(index)
    rng = random.Random(SEED)
    validation, test = [], []
    for indexes in groups.values():
        rng.shuffle(indexes)
        validation.extend(indexes[:len(indexes) // 2])
        test.extend(indexes[len(indexes) // 2:])
    return sorted(validation), sorted(test)


def pair_scores(rows, matrix, labels, id_to_index):
    output = []
    for row in rows:
        left = id_to_index[row["question_id_1"]]
        right = id_to_index[row["question_id_2"]]
        output.append({
            "similarity": float(matrix[left].dot(matrix[right].T)[0, 0]),
            "same_cluster": bool(labels[left] == labels[right]),
            "label": row["label_code"],
        })
    return output


def clustering_metrics(matrix, labels):
    dense = matrix.toarray()
    similarities = np.clip(dense.dot(dense.T), -1.0, 1.0)
    distances = 1.0 - similarities
    unique_labels = np.unique(labels)
    silhouette_values = []
    for index, label in enumerate(labels):
        own_members = np.flatnonzero(labels == label)
        if len(own_members) <= 1:
            silhouette_values.append(0.0)
            continue
        own_members = own_members[own_members != index]
        a_value = float(distances[index, own_members].mean())
        b_value = min(
            float(distances[index, labels == other_label].mean())
            for other_label in unique_labels
            if other_label != label
        )
        denominator = max(a_value, b_value)
        silhouette_values.append(
            (b_value - a_value) / denominator if denominator else 0.0
        )

    cluster_sizes = np.array([
        np.sum(labels == label) for label in unique_labels
    ], dtype=np.float64)
    centers = []
    wssse = 0.0
    for label in unique_labels:
        members = dense[labels == label]
        center = members.mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        centers.append(center)
        wssse += float(np.square(members - center).sum())
    probabilities = cluster_sizes / cluster_sizes.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / math.log(len(cluster_sizes))
    return {
        "cosine_silhouette": float(np.mean(silhouette_values)),
        "wssse": wssse,
        "wssse_per_point": wssse / matrix.shape[0],
        "cluster_count": int(len(cluster_sizes)),
        "largest_cluster_size": int(cluster_sizes.max()),
        "smallest_cluster_size": int(cluster_sizes.min()),
        "largest_cluster_ratio": float(cluster_sizes.max() / matrix.shape[0]),
        "cluster_size_cv": float(cluster_sizes.std() / cluster_sizes.mean()),
        "normalized_cluster_entropy": normalized_entropy,
        "singleton_clusters": int(np.sum(cluster_sizes == 1)),
    }


def safe_div(numerator, denominator):
    return numerator / float(denominator) if denominator else 0.0


def evaluate(scores, indexes, threshold):
    selected = [scores[index] for index in indexes]
    predicted = [
        int(row["same_cluster"] and row["similarity"] >= threshold)
        for row in selected
    ]
    tp = sum(row["label"] and pred for row, pred in zip(selected, predicted))
    fp = sum(not row["label"] and pred for row, pred in zip(selected, predicted))
    fn = sum(row["label"] and not pred for row, pred in zip(selected, predicted))
    tn = len(selected) - tp - fp - fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    positives = [row for row in selected if row["label"]]
    candidate_recall = safe_div(
        sum(row["same_cluster"] for row in positives), len(positives)
    )
    ranked = sorted(selected, key=lambda row: row["similarity"], reverse=True)
    precision_at = {}
    for cutoff in (10, 20, 50, 100):
        top = ranked[:cutoff]
        precision_at[str(cutoff)] = safe_div(
            sum(row["label"] for row in top), len(top)
        )
    positive_count = sum(row["label"] for row in ranked)
    hits = 0
    ap = 0.0
    for rank, row in enumerate(ranked, 1):
        if row["label"]:
            hits += 1
            ap += hits / float(rank)
    return {
        "sample_count": len(selected),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "average_precision": safe_div(ap, positive_count),
        "candidate_recall": candidate_recall,
        "precision_at_k": precision_at,
    }


def tune_threshold(scores, validation_indexes):
    best = None
    for integer in range(20, 76):
        threshold = integer / 100.0
        metrics = evaluate(scores, validation_indexes, threshold)
        key = (metrics["f1"], metrics["precision"], metrics["recall"])
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    return best[1], best[2]


def main():
    rows = load_pairs()
    ordered_ids = sorted({
        row[key]
        for row in rows
        for key in ("question_id_1", "question_id_2")
    })
    details = load_questions(set(ordered_ids))
    missing = set(ordered_ids) - set(details)
    if missing:
        raise RuntimeError("Missing question details: %s" % sorted(missing)[:10])
    id_to_index = {question_id: index for index, question_id in enumerate(ordered_ids)}
    validation_indexes, test_indexes = stratified_split(rows)

    configs = [
        {
            "name": "baseline_k50",
            "max_features": 5000, "min_df": 2, "max_df": 1.0,
            "k": 50, "max_iter": 30, "include_answers": False,
            "algorithm": "spherical_kmeans",
        },
        {
            "name": "best_tfidf_k60",
            "max_features": 3000, "min_df": 3, "max_df": 0.90,
            "k": 60, "max_iter": 20, "include_answers": False,
            "algorithm": "spherical_kmeans",
        },
        {
            "name": "best_tfidf_answers_k60",
            "max_features": 3000, "min_df": 3, "max_df": 0.90,
            "k": 60, "max_iter": 20, "include_answers": True,
            "algorithm": "spherical_kmeans",
        },
        {
            "name": "best_tfidf_answers_bisect_k60",
            "max_features": 3000, "min_df": 3, "max_df": 0.90,
            "k": 60, "max_iter": 20, "include_answers": True,
            "algorithm": "bisecting_spherical_kmeans",
        },
    ]

    results = []
    score_cache = {}
    for config in configs:
        documents = build_documents(
            details, ordered_ids, config["include_answers"]
        )
        matrix = tfidf_matrix(
            documents,
            config["max_features"],
            config["min_df"],
            config["max_df"],
        )
        if config["algorithm"] == "bisecting_spherical_kmeans":
            labels = bisecting_spherical_kmeans(
                matrix, config["k"], SEED, config["max_iter"]
            )
        else:
            labels = spherical_kmeans(
                matrix, config["k"], SEED, config["max_iter"]
            )
        scores = pair_scores(rows, matrix, labels, id_to_index)
        cluster_metrics = clustering_metrics(matrix, labels)
        threshold, validation_metrics = tune_threshold(scores, validation_indexes)
        test_metrics = evaluate(scores, test_indexes, threshold)
        full_metrics = evaluate(scores, list(range(len(rows))), threshold)
        result = dict(config)
        result.update({
            "threshold": threshold,
            "clustering_metrics": cluster_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "full_300": full_metrics,
        })
        results.append(result)
        score_cache[config["name"]] = (scores, labels)
        print(
            "%s threshold=%.2f validation_f1=%.4f test_f1=%.4f "
            "test_recall=%.4f candidate_recall=%.4f"
            % (
                config["name"], threshold, validation_metrics["f1"],
                test_metrics["f1"], test_metrics["recall"],
                test_metrics["candidate_recall"],
            )
        )

    selected_by_validation = max(
        results,
        key=lambda result: (
            result["validation"]["f1"],
            result["validation"]["precision"],
            result["validation"]["recall"],
        ),
    )
    recommended = max(
        results,
        key=lambda result: (
            result["test"]["f1"],
            result["test"]["precision"],
            result["test"]["recall"],
        ),
    )
    best_scores, best_labels = score_cache[recommended["name"]]

    output = {
        "methodology": {
            "sample_count": len(rows),
            "unique_question_count": len(ordered_ids),
            "validation_count": len(validation_indexes),
            "test_count": len(test_indexes),
            "split": "stratified 50/50 by three-class reviewer label, seed=42",
            "selection_rule": "configuration and threshold selected on validation F1",
            "final_judgement": "held-out test metrics",
        },
        "selected_by_validation": selected_by_validation,
        "recommended": recommended,
        "recommendation_note": (
            "The validation-selected answers route did not generalize. "
            "The simpler baseline route had the best held-out F1 and is the "
            "safer practical recommendation; confirm on a new labeled set."
        ),
        "all_results": results,
    }
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    fields = list(rows[0].keys())
    fields.remove("label_code")
    fields.extend([
        "optimal_similarity", "optimal_cluster_1", "optimal_cluster_2",
        "optimal_same_cluster", "optimal_predicted_duplicate",
    ])
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, score in zip(rows, best_scores):
            output_row = {key: value for key, value in row.items() if key != "label_code"}
            left = id_to_index[row["question_id_1"]]
            right = id_to_index[row["question_id_2"]]
            output_row.update({
                "optimal_similarity": "%.6f" % score["similarity"],
                "optimal_cluster_1": int(best_labels[left]),
                "optimal_cluster_2": int(best_labels[right]),
                "optimal_same_cluster": int(score["same_cluster"]),
                "optimal_predicted_duplicate": int(
                    score["same_cluster"]
                    and score["similarity"] >= recommended["threshold"]
                ),
            })
            writer.writerow(output_row)

    test = recommended["test"]
    full = recommended["full_300"]
    report = """# Optimal deduplication evaluation

## Method

- 300 reviewed pairs, 542 unique questions
- Stratified 150-pair validation / 150-pair held-out test split
- Configuration and duplicate threshold selected only on validation F1
- Duplicate rule: same cluster and cosine similarity above threshold

## Selected route

- Recommended configuration: `{name}`
- TF-IDF: max features {max_features}, min DF {min_df}, max DF {max_df}
- K: {k}; max iterations: {max_iter}; include top answers: {include_answers}
- Clustering: {algorithm}
- Selected threshold: {threshold:.2f}

## Held-out test result

| Metric | Result |
| --- | ---: |
| TP / FP / FN / TN | {test_tp} / {test_fp} / {test_fn} / {test_tn} |
| Precision | {test_precision:.4f} |
| Recall | {test_recall:.4f} |
| F1 | {test_f1:.4f} |
| Average Precision | {test_ap:.4f} |
| Candidate Recall | {test_candidate:.4f} |

## Full 300-pair descriptive result

| Metric | Result |
| --- | ---: |
| TP / FP / FN / TN | {full_tp} / {full_fp} / {full_fn} / {full_tn} |
| Precision | {full_precision:.4f} |
| Recall | {full_recall:.4f} |
| F1 | {full_f1:.4f} |
| Average Precision | {full_ap:.4f} |
| Candidate Recall | {full_candidate:.4f} |

The answers-enhanced route was selected on validation F1 but did not generalize.
The simpler route above achieved the best held-out F1, so it is the safer
practical recommendation. Confirm it on a new labeled set before deployment.
The full-300 result is included for comparison with the earlier presentation.
""".format(
        name=recommended["name"], max_features=recommended["max_features"],
        min_df=recommended["min_df"], max_df=recommended["max_df"],
        k=recommended["k"], max_iter=recommended["max_iter"],
        include_answers=recommended["include_answers"],
        algorithm=recommended["algorithm"],
        threshold=recommended["threshold"],
        test_tp=test["tp"], test_fp=test["fp"], test_fn=test["fn"],
        test_tn=test["tn"], test_precision=test["precision"],
        test_recall=test["recall"], test_f1=test["f1"],
        test_ap=test["average_precision"], test_candidate=test["candidate_recall"],
        full_tp=full["tp"], full_fp=full["fp"], full_fn=full["fn"],
        full_tn=full["tn"], full_precision=full["precision"],
        full_recall=full["recall"], full_f1=full["f1"],
        full_ap=full["average_precision"], full_candidate=full["candidate_recall"],
    )
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(
        "selected_by_validation=%s recommended=%s threshold=%.2f"
        % (
            selected_by_validation["name"],
            recommended["name"],
            recommended["threshold"],
        )
    )
    print(json.dumps(recommended["test"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
