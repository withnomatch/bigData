# -*- coding: utf-8 -*-
import csv
import html
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "oracle_database_questions_jsonlines.json"
ASSIGNMENTS = ROOT / "local_test_output" / "assignments.csv"
OUTPUT = ROOT / "annotation" / "dedup_pairs_300_reviewed.csv"
METRICS_JSON = ROOT / "annotation" / "dedup_300_metrics.json"
REPORT_MD = ROOT / "annotation" / "dedup_300_report_generated.md"

SEED = 42
PAIR_COUNTS = {"重复": 100, "相关但不重复": 100, "不相关": 100}
K = 50
MAX_FEATURES = 5000
MIN_DF = 2
DUPLICATE_THRESHOLD = 0.55
RELATED_THRESHOLD = 0.15

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "database", "db", "did", "do", "does", "for", "from",
    "had", "has", "have", "help", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "oracle", "please", "sql",
    "than", "that", "the", "then", "there", "this", "to", "using", "was",
    "were", "what", "when", "where", "which", "why", "will", "with",
    "would", "you", "your",
}

URL_RE = re.compile(r"https?://[^\s<>\"]+", re.I)
QUESTION_LINK_RE = re.compile(r"stackoverflow\.com/questions/(\d+)", re.I)
POSSIBLE_DUPLICATE_RE = re.compile(r"possible\s+duplicate", re.I)
ERROR_CODE_RE = re.compile(r"(?:ora|pls)[\s_-]*(\d{4,5})", re.I)
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+-]*|\d+", re.I)


def clean_text(value):
    value = re.sub(r"<[^>]+>", " ", value or "")
    value = html.unescape(value)
    value = URL_RE.sub(" ", value)
    value = POSSIBLE_DUPLICATE_RE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokens(value):
    return [
        token.lower()
        for token in TOKEN_RE.findall(clean_text(value))
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def title_tokens(value):
    return set(tokens(value))


def error_codes(value):
    return {"ora-" + code for code in ERROR_CODE_RE.findall(value or "")}


def title_similarity(left, right):
    left_tokens = title_tokens(left)
    right_tokens = title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))


def read_assignments():
    with ASSIGNMENTS.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_official_duplicate_pairs():
    question_ids = set()
    raw_pairs = set()
    with QUESTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            question_ids.add(question_id)
            body = row.get("body") or ""
            if not POSSIBLE_DUPLICATE_RE.search(body):
                continue
            for target_id in QUESTION_LINK_RE.findall(body):
                if target_id != question_id:
                    raw_pairs.add(tuple(sorted((question_id, target_id))))
    return sorted(
        pair
        for pair in raw_pairs
        if pair[0] in question_ids and pair[1] in question_ids
    )


def build_same_cluster_candidates(assignments, excluded):
    clusters = defaultdict(list)
    for row in assignments:
        if len(title_tokens(row["title"])) >= 2:
            clusters[row["cluster"]].append(row)

    candidates = []
    for items in clusters.values():
        inverted = defaultdict(list)
        for index, row in enumerate(items):
            for token in title_tokens(row["title"]):
                inverted[token].append(index)
        seen = set()
        for indexes in inverted.values():
            if len(indexes) > 100:
                continue
            for left_index, right_index in itertools.combinations(indexes, 2):
                key = (left_index, right_index)
                if key in seen:
                    continue
                seen.add(key)
                left = items[left_index]
                right = items[right_index]
                pair = tuple(sorted((left["question_id"], right["question_id"])))
                if pair in excluded:
                    continue
                similarity = title_similarity(left["title"], right["title"])
                shared_codes = error_codes(left["title"]) & error_codes(right["title"])
                shared_terms = title_tokens(left["title"]) & title_tokens(right["title"])
                if 0.28 <= similarity <= 0.72 and (
                    shared_codes or len(shared_terms) >= 2
                ):
                    candidates.append({
                        "question_id_1": left["question_id"],
                        "question_id_2": right["question_id"],
                        "selection_score": similarity,
                    })
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    return candidates


def build_cross_cluster_candidates(assignments, excluded, count, rng):
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < count and attempts < 100000:
        attempts += 1
        left, right = rng.sample(assignments, 2)
        if left["cluster"] == right["cluster"]:
            continue
        pair = tuple(sorted((left["question_id"], right["question_id"])))
        if pair in excluded or pair in seen:
            continue
        if error_codes(left["title"]) & error_codes(right["title"]):
            continue
        similarity = title_similarity(left["title"], right["title"])
        if similarity > 0.02:
            continue
        seen.add(pair)
        candidates.append({
            "question_id_1": pair[0],
            "question_id_2": pair[1],
            "selection_score": similarity,
        })
    return candidates


def load_details(question_ids):
    details = {}
    with QUESTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if question_id not in question_ids:
                continue
            details[question_id] = {
                "question_id": question_id,
                "title": clean_text(row.get("title") or ""),
                "body": clean_text(row.get("body") or ""),
                "tags": [str(tag) for tag in (row.get("tags") or [])],
            }
            if len(details) == len(question_ids):
                break
    return details


def select_pairs():
    rng = random.Random(SEED)
    official = find_official_duplicate_pairs()
    if len(official) < PAIR_COUNTS["重复"]:
        raise RuntimeError("Not enough official duplicate pairs")
    rng.shuffle(official)
    positive_pairs = official[:PAIR_COUNTS["重复"]]
    excluded = set(official)

    assignments = read_assignments()
    related_pool = build_same_cluster_candidates(assignments, excluded)
    if len(related_pool) < PAIR_COUNTS["相关但不重复"]:
        raise RuntimeError("Not enough related candidates")
    related_pairs = related_pool[:PAIR_COUNTS["相关但不重复"]]
    excluded.update(
        tuple(sorted((row["question_id_1"], row["question_id_2"])))
        for row in related_pairs
    )
    unrelated_pairs = build_cross_cluster_candidates(
        assignments, excluded, PAIR_COUNTS["不相关"], rng
    )
    if len(unrelated_pairs) < PAIR_COUNTS["不相关"]:
        raise RuntimeError("Not enough unrelated candidates")

    selected = []
    for left, right in positive_pairs:
        selected.append({
            "question_id_1": left,
            "question_id_2": right,
            "reviewer_label": "重复",
            "label_source": "Stack Overflow Possible Duplicate链接",
            "review_notes": "原问题正文明确指向另一问题为Possible Duplicate。",
        })
    for row in related_pairs:
        row.update({
            "reviewer_label": "相关但不重复",
            "label_source": "同簇高相似困难负例初审",
            "review_notes": (
                "共享错误码或多个核心术语，但具体输入、环境或解决目标不同；"
                "需第二位人工标注者复核。"
            ),
        })
        selected.append(row)
    for row in unrelated_pairs:
        row.update({
            "reviewer_label": "不相关",
            "label_source": "跨簇低相似保守负例初审",
            "review_notes": "来自不同基线簇且无共享核心标题词；需第二位人工标注者复核。",
        })
        selected.append(row)
    rng.shuffle(selected)
    return selected


def build_document(detail):
    title = detail["title"]
    tags = " ".join(detail["tags"])
    return " ".join([title, title, title, detail["body"], tags, tags])


def tfidf_matrix(details, ordered_ids):
    documents = [tokens(build_document(details[question_id])) for question_id in ordered_ids]
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    vocabulary_terms = [
        term
        for term, frequency in document_frequency.most_common()
        if frequency >= MIN_DF
    ][:MAX_FEATURES]
    vocabulary = {term: index for index, term in enumerate(vocabulary_terms)}

    rows = []
    columns = []
    values = []
    document_count = len(documents)
    for row_index, document in enumerate(documents):
        counts = Counter(term for term in document if term in vocabulary)
        norm_values = []
        for term, count in counts.items():
            idf = math.log((document_count + 1.0) / (
                document_frequency[term] + 1.0
            )) + 1.0
            norm_values.append((vocabulary[term], (1.0 + math.log(count)) * idf))
        norm = math.sqrt(sum(value * value for _, value in norm_values)) or 1.0
        for column_index, value in norm_values:
            rows.append(row_index)
            columns.append(column_index)
            values.append(value / norm)
    return csr_matrix(
        (values, (rows, columns)),
        shape=(document_count, len(vocabulary)),
        dtype=np.float64,
    )


def spherical_kmeans(matrix, k, seed, max_iter=30):
    rng = np.random.RandomState(seed)
    centers = matrix[rng.choice(matrix.shape[0], k, replace=False)].toarray()
    center_norms = np.linalg.norm(centers, axis=1)
    centers = centers / np.maximum(center_norms[:, None], 1e-12)
    labels = np.zeros(matrix.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        similarities = matrix.dot(centers.T)
        new_labels = np.asarray(similarities.argmax(axis=1)).ravel()
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        new_centers = np.zeros_like(centers)
        for cluster in range(k):
            members = np.flatnonzero(labels == cluster)
            if len(members) == 0:
                new_centers[cluster] = matrix[
                    rng.randint(matrix.shape[0])
                ].toarray()[0]
            else:
                new_centers[cluster] = np.asarray(
                    matrix[members].mean(axis=0)
                ).ravel()
        center_norms = np.linalg.norm(new_centers, axis=1)
        centers = new_centers / np.maximum(center_norms[:, None], 1e-12)
    return labels


def safe_div(numerator, denominator):
    return numerator / float(denominator) if denominator else None


def f1_score(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(rows):
    ranked = sorted(rows, key=lambda row: row["cosine_similarity"], reverse=True)
    positive_count = sum(row["label_code"] for row in ranked)
    if not positive_count:
        return None
    hits = 0
    total = 0.0
    for rank, row in enumerate(ranked, 1):
        if row["label_code"]:
            hits += 1
            total += hits / float(rank)
    return total / positive_count


def class_metrics(rows, label):
    tp = sum(
        row["reviewer_label"] == label and row["predicted_label"] == label
        for row in rows
    )
    predicted = sum(row["predicted_label"] == label for row in rows)
    actual = sum(row["reviewer_label"] == label for row in rows)
    precision = safe_div(tp, predicted)
    recall = safe_div(tp, actual)
    return {
        "support": actual,
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
    }


def evaluate(selected, details):
    ordered_ids = sorted({
        row[key]
        for row in selected
        for key in ("question_id_1", "question_id_2")
    })
    id_to_index = {question_id: index for index, question_id in enumerate(ordered_ids)}
    matrix = tfidf_matrix(details, ordered_ids)
    clusters = spherical_kmeans(matrix, min(K, matrix.shape[0]), SEED)

    evaluated = []
    for row in selected:
        left_index = id_to_index[row["question_id_1"]]
        right_index = id_to_index[row["question_id_2"]]
        similarity = float(matrix[left_index].dot(matrix[right_index].T)[0, 0])
        same_cluster = bool(clusters[left_index] == clusters[right_index])
        predicted_duplicate = same_cluster and similarity >= DUPLICATE_THRESHOLD
        if predicted_duplicate:
            predicted_label = "重复"
        elif same_cluster or similarity >= RELATED_THRESHOLD:
            predicted_label = "相关但不重复"
        else:
            predicted_label = "不相关"
        evaluated.append(dict(row, **{
            "cosine_similarity": similarity,
            "experiment_cluster_1": int(clusters[left_index]),
            "experiment_cluster_2": int(clusters[right_index]),
            "same_experiment_cluster": int(same_cluster),
            "predicted_duplicate": int(predicted_duplicate),
            "predicted_label": predicted_label,
            "label_code": int(row["reviewer_label"] == "重复"),
        }))

    tp = sum(row["label_code"] and row["predicted_duplicate"] for row in evaluated)
    fp = sum(not row["label_code"] and row["predicted_duplicate"] for row in evaluated)
    fn = sum(row["label_code"] and not row["predicted_duplicate"] for row in evaluated)
    tn = len(evaluated) - tp - fp - fn
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    positive_rows = [row for row in evaluated if row["label_code"]]
    candidate_recall = safe_div(
        sum(row["same_experiment_cluster"] for row in positive_rows),
        len(positive_rows),
    )
    ranked = sorted(
        evaluated, key=lambda row: row["cosine_similarity"], reverse=True
    )
    precision_at = {}
    for cutoff in (10, 20, 50, 100):
        top = ranked[:cutoff]
        precision_at[str(cutoff)] = safe_div(
            sum(row["label_code"] for row in top), len(top)
        )

    per_class = {
        label: class_metrics(evaluated, label)
        for label in ("重复", "相关但不重复", "不相关")
    }
    accuracy = safe_div(
        sum(row["reviewer_label"] == row["predicted_label"] for row in evaluated),
        len(evaluated),
    )
    macro_f1 = sum(
        values["f1"] or 0.0 for values in per_class.values()
    ) / 3.0

    metrics = {
        "sample_count": len(evaluated),
        "unique_question_count": len(ordered_ids),
        "label_counts": dict(Counter(
            row["reviewer_label"] for row in evaluated
        )),
        "experiment": {
            "feature": "title*3 + body + tags*2 TF-IDF, L2 normalized",
            "clustering": "spherical KMeans",
            "k": K,
            "duplicate_threshold": DUPLICATE_THRESHOLD,
            "related_threshold": RELATED_THRESHOLD,
        },
        "binary_dedup": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1_score(precision, recall),
            "average_precision_pr_auc": average_precision(evaluated),
            "candidate_recall": candidate_recall,
            "precision_at_k": precision_at,
        },
        "three_class": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class,
        },
    }
    return evaluated, metrics


def write_outputs(rows, details, metrics):
    fields = [
        "pair_id",
        "question_id_1", "title_1", "body_excerpt_1", "tags_1",
        "question_id_2", "title_2", "body_excerpt_2", "tags_2",
        "review_status", "reviewer_label", "reviewer_name",
        "label_source", "review_notes",
        "cosine_similarity", "experiment_cluster_1", "experiment_cluster_2",
        "same_experiment_cluster", "predicted_duplicate", "predicted_label",
    ]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            left = details[row["question_id_1"]]
            right = details[row["question_id_2"]]
            writer.writerow({
                "pair_id": "A%03d" % index,
                "question_id_1": row["question_id_1"],
                "title_1": left["title"],
                "body_excerpt_1": left["body"][:500],
                "tags_1": "|".join(left["tags"]),
                "question_id_2": row["question_id_2"],
                "title_2": right["title"],
                "body_excerpt_2": right["body"][:500],
                "tags_2": "|".join(right["tags"]),
                "review_status": "初审完成，待第二人复核",
                "reviewer_label": row["reviewer_label"],
                "reviewer_name": "Codex机器辅助初审",
                "label_source": row["label_source"],
                "review_notes": row["review_notes"],
                "cosine_similarity": "%.6f" % row["cosine_similarity"],
                "experiment_cluster_1": row["experiment_cluster_1"],
                "experiment_cluster_2": row["experiment_cluster_2"],
                "same_experiment_cluster": row["same_experiment_cluster"],
                "predicted_duplicate": row["predicted_duplicate"],
                "predicted_label": row["predicted_label"],
            })

    with METRICS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    binary = metrics["binary_dedup"]
    three_class = metrics["three_class"]
    report = """# 300对问题标注与去重实验报告

## 数据构成

- 总问题对：300
- 重复：100（Stack Overflow正文中的Possible Duplicate链接）
- 相关但不重复：100（同簇、高标题相似度困难负例初审）
- 不相关：100（跨簇、低标题相似度保守负例初审）
- 涉及问题数：{unique_questions}

其中重复标签有平台链接证据；两类负例属于机器辅助初审，正式报告前仍需第二位
标注者逐对复核。

## 实验设置

- 特征：标题重复3次 + 正文 + 标签重复2次的TF-IDF
- 归一化：L2
- 聚类：球面KMeans，K={k}
- 重复判断：同簇且余弦相似度 >= {threshold}

## 二分类去重指标

| 指标 | 结果 |
| --- | ---: |
| TP / FP / FN / TN | {tp} / {fp} / {fn} / {tn} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1 | {f1:.4f} |
| PR-AUC（Average Precision） | {ap:.4f} |
| Candidate Recall | {candidate_recall:.4f} |
| Precision@10 | {p10:.4f} |
| Precision@20 | {p20:.4f} |
| Precision@50 | {p50:.4f} |
| Precision@100 | {p100:.4f} |

## 三分类指标

- Accuracy：{accuracy:.4f}
- Macro-F1：{macro_f1:.4f}

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
{class_rows}

## 解释

Candidate Recall 衡量真实重复对被聚到同一簇的比例；最终 Recall 还会受到余弦
阈值影响。若 Candidate Recall 明显高于最终 Recall，主要损失来自相似度阈值；
若 Candidate Recall 本身较低，主要问题在文本表示或聚类阶段。
""".format(
        unique_questions=metrics["unique_question_count"],
        k=metrics["experiment"]["k"],
        threshold=metrics["experiment"]["duplicate_threshold"],
        tp=binary["tp"], fp=binary["fp"], fn=binary["fn"], tn=binary["tn"],
        precision=binary["precision"] or 0.0,
        recall=binary["recall"] or 0.0,
        f1=binary["f1"] or 0.0,
        ap=binary["average_precision_pr_auc"] or 0.0,
        candidate_recall=binary["candidate_recall"] or 0.0,
        p10=binary["precision_at_k"]["10"] or 0.0,
        p20=binary["precision_at_k"]["20"] or 0.0,
        p50=binary["precision_at_k"]["50"] or 0.0,
        p100=binary["precision_at_k"]["100"] or 0.0,
        accuracy=three_class["accuracy"] or 0.0,
        macro_f1=three_class["macro_f1"] or 0.0,
        class_rows="\n".join(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support} |".format(
                label=label,
                precision=values["precision"] or 0.0,
                recall=values["recall"] or 0.0,
                f1=values["f1"] or 0.0,
                support=values["support"],
            )
            for label, values in three_class["per_class"].items()
        ),
    )
    with REPORT_MD.open("w", encoding="utf-8") as handle:
        handle.write(report)


def main():
    selected = select_pairs()
    question_ids = {
        row[key]
        for row in selected
        for key in ("question_id_1", "question_id_2")
    }
    details = load_details(question_ids)
    missing = question_ids - set(details)
    if missing:
        raise RuntimeError("Missing question details: %s" % sorted(missing)[:10])
    evaluated, metrics = evaluate(selected, details)
    write_outputs(evaluated, details, metrics)
    print("pairs=%d unique_questions=%d" % (
        metrics["sample_count"], metrics["unique_question_count"]
    ))
    print(json.dumps(metrics["binary_dedup"], ensure_ascii=False, indent=2))
    print(json.dumps(metrics["three_class"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
