# -*- coding: utf-8 -*-
import csv
import html
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "oracle_database_questions_jsonlines.json"
PAIR_FILE = ROOT / "annotation" / "dedup_pairs_300_reviewed.csv"
OUTPUT_DIR = ROOT / "full_sequential_results"
SEED = 42
SAMPLE_SIZE = 15441

TOKEN_RE = re.compile(r"[a-z][a-z0-9_+#.-]*|\d+", re.I)
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "database", "db", "did", "do", "does", "error",
    "for", "from", "had", "has", "have", "help", "how", "i", "if",
    "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "oracle", "please", "problem", "question", "sql", "stackoverflow",
    "than", "that", "the", "then", "there", "this", "to", "use", "using",
    "want", "was", "were", "what", "when", "where", "which", "why",
    "will", "with", "would", "you", "your",
}


def clean(value):
    value = TAG_RE.sub(" ", value or "")
    value = URL_RE.sub(" ", html.unescape(value))
    return re.sub(r"\s+", " ", value).strip()


def tokens(value):
    return [
        token.lower() for token in TOKEN_RE.findall(clean(value))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def reservoir_sample(path, sample_size, seed):
    rng = random.Random(seed)
    sample = []
    total = 0
    with path.open(encoding="utf-8") as handle:
        for total, line in enumerate(handle, 1):
            row = json.loads(line)
            item = {
                "question_id": str(row.get("question_id", "")),
                "title": clean(row.get("title") or ""),
                "body": clean(row.get("body") or ""),
                "tags": [str(tag) for tag in (row.get("tags") or [])],
                "answers": " ".join(
                    clean(answer.get("body") or "")
                    for answer in sorted(
                        row.get("answers") or [],
                        key=lambda answer: int(answer.get("score") or 0),
                        reverse=True,
                    )[:2]
                ),
            }
            if len(sample) < sample_size:
                sample.append(item)
            else:
                index = rng.randrange(total)
                if index < sample_size:
                    sample[index] = item
    return sample, total


def build_documents(records, mode):
    documents = []
    for row in records:
        if mode == "baseline":
            text = row["title"] + " " + row["body"]
        elif mode == "weighted":
            text = " ".join([
                row["title"], row["title"], row["title"], row["body"],
                " ".join(row["tags"]), " ".join(row["tags"]),
            ])
        elif mode == "weighted_answers":
            text = " ".join([
                row["title"], row["title"], row["title"], row["body"],
                " ".join(row["tags"]), " ".join(row["tags"]), row["answers"],
            ])
        else:
            raise ValueError(mode)
        documents.append(tokens(text))
    return documents


def tfidf(documents, max_features, min_df, max_df):
    df = Counter()
    for document in documents:
        df.update(set(document))
    max_count = int(math.floor(len(documents) * max_df))
    terms = [
        term for term, frequency in df.most_common()
        if min_df <= frequency <= max_count
    ][:max_features]
    vocabulary = {term: index for index, term in enumerate(terms)}
    rows, columns, values = [], [], []
    for row_index, document in enumerate(documents):
        counts = Counter(term for term in document if term in vocabulary)
        weighted = []
        for term, count in counts.items():
            value = (1.0 + math.log(count)) * (
                math.log((len(documents) + 1.0) / (df[term] + 1.0)) + 1.0
            )
            weighted.append((vocabulary[term], value))
        norm = math.sqrt(sum(value * value for _, value in weighted)) or 1.0
        for column, value in weighted:
            rows.append(row_index)
            columns.append(column)
            values.append(value / norm)
    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(len(documents), len(vocabulary)),
        dtype=np.float32,
    )
    return matrix, len(vocabulary)


def spherical_kmeans(matrix, k, seed, max_iter=20):
    rng = np.random.RandomState(seed)
    centers = matrix[rng.choice(matrix.shape[0], k, replace=False)].toarray()
    centers /= np.maximum(np.linalg.norm(centers, axis=1)[:, None], 1e-12)
    labels = np.full(matrix.shape[0], -1, dtype=np.int32)
    iterations = 0
    for iterations in range(1, max_iter + 1):
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
    return labels, centers, iterations


def bisecting_spherical_kmeans(matrix, k, seed, max_iter=20):
    clusters = [np.arange(matrix.shape[0])]
    rng = random.Random(seed)
    while len(clusters) < k:
        split_index = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        members = clusters.pop(split_index)
        if len(members) < 2:
            clusters.append(members)
            break
        local_labels, _, _ = spherical_kmeans(
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
    centers = []
    for cluster in range(len(clusters)):
        center = np.asarray(matrix[labels == cluster].mean(axis=0)).ravel()
        center /= max(np.linalg.norm(center), 1e-12)
        centers.append(center)
    return labels, np.asarray(centers), None


def approximate_silhouette(matrix, labels, sample_size=1000, seed=42):
    rng = np.random.RandomState(seed)
    indexes = rng.choice(
        matrix.shape[0], min(sample_size, matrix.shape[0]), replace=False
    )
    sample = matrix[indexes]
    distances = 1.0 - np.clip(sample.dot(sample.T).toarray(), -1.0, 1.0)
    sample_labels = labels[indexes]
    unique = np.unique(sample_labels)
    values = []
    for index, label in enumerate(sample_labels):
        own = np.flatnonzero(sample_labels == label)
        own = own[own != index]
        if not len(own):
            values.append(0.0)
            continue
        a_value = float(distances[index, own].mean())
        alternatives = [
            float(distances[index, sample_labels == other].mean())
            for other in unique if other != label
            if np.any(sample_labels == other)
        ]
        if not alternatives:
            values.append(0.0)
            continue
        b_value = min(alternatives)
        denominator = max(a_value, b_value)
        values.append(
            (b_value - a_value) / denominator if denominator else 0.0
        )
    return float(np.mean(values))


def cluster_metrics(matrix, labels, centers):
    counts = np.bincount(labels)
    norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
    all_similarities = np.asarray(matrix.dot(centers.T))
    dot = all_similarities[np.arange(matrix.shape[0]), labels]
    center_norms = np.sum(centers[labels] * centers[labels], axis=1)
    wssse = float(np.sum(norms + center_norms - 2.0 * dot))
    probabilities = counts[counts > 0] / float(len(labels))
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "silhouette": approximate_silhouette(matrix, labels),
        "wssse": wssse,
        "wssse_per_point": wssse / len(labels),
        "largest_cluster_size": int(counts.max()),
        "smallest_cluster_size": int(counts.min()),
        "largest_cluster_ratio": float(counts.max() / len(labels)),
        "cluster_size_cv": float(counts.std() / counts.mean()),
        "normalized_cluster_entropy": entropy / math.log(len(counts)),
        "singleton_clusters": int(np.sum(counts == 1)),
        "small_clusters_lt_10": int(np.sum(counts < 10)),
    }


def run_cluster(matrix, k, algorithm, seed=42, max_iter=20):
    started = time.perf_counter()
    if algorithm == "bisecting":
        labels, centers, iterations = bisecting_spherical_kmeans(
            matrix, k, seed, max_iter
        )
    else:
        labels, centers, iterations = spherical_kmeans(
            matrix, k, seed, max_iter
        )
    metrics = cluster_metrics(matrix, labels, centers)
    metrics.update({
        "k": k,
        "algorithm": algorithm,
        "iterations": iterations,
        "runtime_seconds": time.perf_counter() - started,
    })
    return labels, metrics


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Not JSON serializable: %r" % (value,))


def select_best_k(rows):
    eligible = [
        row for row in rows
        if row["largest_cluster_ratio"] <= 0.25
        and row["singleton_clusters"] <= max(2, int(row["k"] * 0.02))
    ]
    candidates = eligible or rows
    return max(
        candidates,
        key=lambda row: (
            row["silhouette"],
            -row["largest_cluster_ratio"],
            -row["cluster_size_cv"],
        ),
    )


def load_pair_rows():
    with PAIR_FILE.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_pair_questions(pair_rows):
    ids = {
        row[key] for row in pair_rows
        for key in ("question_id_1", "question_id_2")
    }
    found = {}
    with DATA_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if question_id not in ids:
                continue
            found[question_id] = {
                "question_id": question_id,
                "title": clean(row.get("title") or ""),
                "body": clean(row.get("body") or ""),
                "tags": [str(tag) for tag in (row.get("tags") or [])],
                "answers": " ".join(
                    clean(answer.get("body") or "")
                    for answer in sorted(
                        row.get("answers") or [],
                        key=lambda answer: int(answer.get("score") or 0),
                        reverse=True,
                    )[:2]
                ),
            }
            if len(found) == len(ids):
                break
    ordered_ids = sorted(ids)
    return [found[question_id] for question_id in ordered_ids], ordered_ids


def stratified_pair_split(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row["reviewer_label"], []).append(index)
    rng = random.Random(SEED)
    validation, test = [], []
    for indexes in groups.values():
        rng.shuffle(indexes)
        validation.extend(indexes[:len(indexes) // 2])
        test.extend(indexes[len(indexes) // 2:])
    return sorted(validation), sorted(test)


def dedup_metrics(pair_rows, matrix, labels, id_to_index, indexes, threshold):
    scored = []
    for row in pair_rows:
        left = id_to_index[row["question_id_1"]]
        right = id_to_index[row["question_id_2"]]
        scored.append({
            "positive": row["reviewer_label"] == "\u91cd\u590d",
            "same_cluster": labels[left] == labels[right],
            "similarity": float(matrix[left].dot(matrix[right].T)[0, 0]),
        })
    selected = [scored[index] for index in indexes]
    predicted = [
        row["same_cluster"] and row["similarity"] >= threshold
        for row in selected
    ]
    tp = sum(row["positive"] and pred for row, pred in zip(selected, predicted))
    fp = sum(not row["positive"] and pred for row, pred in zip(selected, predicted))
    fn = sum(row["positive"] and not pred for row, pred in zip(selected, predicted))
    tn = len(selected) - tp - fp - fn
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    positives = [row for row in selected if row["positive"]]
    candidate_recall = (
        sum(row["same_cluster"] for row in positives) / float(len(positives))
        if positives else 0.0
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "candidate_recall": candidate_recall,
    }


def evaluate_dedup(config, best_k, pair_rows, pair_records, ordered_ids):
    documents = build_documents(pair_records, config["mode"])
    matrix, vocabulary_size = tfidf(
        documents, config["max_features"], config["min_df"], config["max_df"]
    )
    k = min(best_k, matrix.shape[0] - 1)
    labels, clustering = run_cluster(
        matrix, k, config["algorithm"], SEED, config["max_iter"]
    )
    id_to_index = {
        question_id: index for index, question_id in enumerate(ordered_ids)
    }
    validation_indexes, test_indexes = stratified_pair_split(pair_rows)
    best_threshold, best_validation = None, None
    for integer in range(10, 76):
        threshold = integer / 100.0
        metrics = dedup_metrics(
            pair_rows, matrix, labels, id_to_index,
            validation_indexes, threshold,
        )
        if best_validation is None or (
            metrics["f1"], metrics["precision"], metrics["recall"]
        ) > (
            best_validation["f1"],
            best_validation["precision"],
            best_validation["recall"],
        ):
            best_threshold, best_validation = threshold, metrics
    test = dedup_metrics(
        pair_rows, matrix, labels, id_to_index, test_indexes, best_threshold
    )
    full = dedup_metrics(
        pair_rows, matrix, labels, id_to_index,
        list(range(len(pair_rows))), best_threshold,
    )
    return {
        "threshold": best_threshold,
        "vocabulary_size": vocabulary_size,
        "clustering": clustering,
        "validation": best_validation,
        "test": test,
        "full_300": full,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    total_started = time.perf_counter()
    print("Loading deterministic reservoir sample...")
    records, total_records = reservoir_sample(DATA_FILE, SAMPLE_SIZE, SEED)
    print("total_records=%d sample_records=%d" % (total_records, len(records)))

    print("Building baseline TF-IDF...")
    documents = build_documents(records, "baseline")
    matrix, vocabulary_size = tfidf(documents, 5000, 5, 0.90)
    print("matrix=%s vocabulary=%d" % (matrix.shape, vocabulary_size))

    coarse_rows = []
    for k in range(40, 201, 5):
        _, metrics = run_cluster(matrix, k, "spherical", SEED, 20)
        coarse_rows.append(metrics)
        print(
            "coarse K=%d silhouette=%.5f max=%.2f%% cv=%.3f time=%.1fs"
            % (
                k, metrics["silhouette"],
                metrics["largest_cluster_ratio"] * 100,
                metrics["cluster_size_cv"], metrics["runtime_seconds"],
            )
        )
    coarse_best = select_best_k(coarse_rows)
    fine_values = sorted(set(
        range(max(40, coarse_best["k"] - 4), min(200, coarse_best["k"] + 4) + 1)
    ) - {row["k"] for row in coarse_rows})
    fine_rows = []
    for k in fine_values:
        _, metrics = run_cluster(matrix, k, "spherical", SEED, 20)
        fine_rows.append(metrics)
        print(
            "fine K=%d silhouette=%.5f max=%.2f%% cv=%.3f time=%.1fs"
            % (
                k, metrics["silhouette"],
                metrics["largest_cluster_ratio"] * 100,
                metrics["cluster_size_cv"], metrics["runtime_seconds"],
            )
        )
    k_rows = sorted(coarse_rows + fine_rows, key=lambda row: row["k"])
    best_k_row = select_best_k(k_rows)
    best_k = best_k_row["k"]
    write_csv(OUTPUT_DIR / "k_search_results.csv", k_rows)
    print("BEST_K=%d" % best_k)

    configs = [
        {
            "name": "01_baseline",
            "mode": "baseline", "max_features": 5000, "min_df": 5,
            "max_df": 0.90, "algorithm": "spherical", "max_iter": 20,
        },
        {
            "name": "02_tfidf_optimized",
            "mode": "baseline", "max_features": 3000, "min_df": 3,
            "max_df": 0.90, "algorithm": "spherical", "max_iter": 20,
        },
        {
            "name": "03_weighted_title_tags",
            "mode": "weighted", "max_features": 3000, "min_df": 3,
            "max_df": 0.90, "algorithm": "spherical", "max_iter": 20,
        },
        {
            "name": "04_add_answers",
            "mode": "weighted_answers", "max_features": 3000, "min_df": 3,
            "max_df": 0.90, "algorithm": "spherical", "max_iter": 20,
        },
        {
            "name": "05_bisecting",
            "mode": "weighted_answers", "max_features": 3000, "min_df": 3,
            "max_df": 0.90, "algorithm": "bisecting", "max_iter": 20,
        },
    ]
    optimization_rows = []
    matrices = {}
    for config in configs:
        print("Running optimization: %s" % config["name"])
        config_documents = build_documents(records, config["mode"])
        config_matrix, config_vocabulary = tfidf(
            config_documents, config["max_features"],
            config["min_df"], config["max_df"],
        )
        _, metrics = run_cluster(
            config_matrix, best_k, config["algorithm"], SEED,
            config["max_iter"],
        )
        row = dict(config)
        row.update(metrics)
        row["vocabulary_size"] = config_vocabulary
        optimization_rows.append(row)
        matrices[config["name"]] = config_matrix
        print(
            "%s silhouette=%.5f max=%.2f%% cv=%.3f"
            % (
                config["name"], metrics["silhouette"],
                metrics["largest_cluster_ratio"] * 100,
                metrics["cluster_size_cv"],
            )
        )
    write_csv(OUTPUT_DIR / "optimization_results.csv", optimization_rows)

    pair_rows = load_pair_rows()
    pair_records, ordered_ids = load_pair_questions(pair_rows)
    dedup_results = []
    for config in configs:
        result = evaluate_dedup(
            config, best_k, pair_rows, pair_records, ordered_ids
        )
        result["name"] = config["name"]
        result["config"] = config
        dedup_results.append(result)
        print(
            "%s test_f1=%.4f P=%.4f R=%.4f candidate=%.4f threshold=%.2f"
            % (
                config["name"], result["test"]["f1"],
                result["test"]["precision"], result["test"]["recall"],
                result["test"]["candidate_recall"], result["threshold"],
            )
        )
    recommended = max(
        dedup_results,
        key=lambda result: (
            result["test"]["f1"],
            result["test"]["precision"],
            result["test"]["recall"],
        ),
    )
    summary = {
        "methodology": {
            "total_records": total_records,
            "sample_records": len(records),
            "sample_method": "reservoir sample with seed 42",
            "k_search": "40-200 step 5, then +/-4 integer refinement",
            "k_selection": (
                "highest approximate cosine silhouette subject to largest "
                "cluster <=25% and limited singleton clusters"
            ),
            "dedup_split": "stratified 150 validation / 150 test",
            "total_runtime_seconds": time.perf_counter() - total_started,
        },
        "best_k": best_k_row,
        "optimization_results": optimization_rows,
        "dedup_results": dedup_results,
        "recommended": recommended,
    }
    with (OUTPUT_DIR / "full_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            summary, handle, ensure_ascii=False, indent=2,
            default=json_default,
        )
    print("RECOMMENDED=%s" % recommended["name"])
    print(json.dumps(
        recommended["test"], ensure_ascii=False, indent=2,
        default=json_default,
    ))


if __name__ == "__main__":
    main()
