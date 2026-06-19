# -*- coding: utf-8 -*-
import csv
import difflib
import itertools
import json
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS = ROOT / "local_test_output" / "assignments.csv"
QUESTIONS = ROOT / "oracle_database_questions_jsonlines.json"
OUTPUT = ROOT / "annotation" / "dedup_pairs_initial.csv"

STOPWORDS = {
    "a", "an", "and", "are", "can", "database", "do", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "oracle", "or", "sql", "the", "to",
    "using", "what", "when", "where", "why", "with",
}


def tokens(title):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def error_codes(title):
    return set(re.findall(r"(?:ora|pls)-\d{5}", title.lower()))


def similarity(left, right):
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0, 0.0, 0.0
    jaccard = len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))
    sequence = difflib.SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return 0.7 * jaccard + 0.3 * sequence, jaccard, sequence


def read_assignments():
    with ASSIGNMENTS.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_pairs(rows):
    clusters = defaultdict(list)
    for row in rows:
        if len(tokens(row["title"])) >= 2:
            clusters[row["cluster"]].append(row)

    candidates = []
    for cluster, items in clusters.items():
        inverted = defaultdict(list)
        for index, row in enumerate(items):
            for token in tokens(row["title"]):
                inverted[token].append(index)

        seen = set()
        for indexes in inverted.values():
            if len(indexes) > 80:
                continue
            for left_index, right_index in itertools.combinations(indexes, 2):
                key = (left_index, right_index)
                if key in seen:
                    continue
                seen.add(key)
                left = items[left_index]
                right = items[right_index]
                score, jaccard, sequence = similarity(left["title"], right["title"])
                if score >= 0.42:
                    candidates.append(
                        {
                            "left": left,
                            "right": right,
                            "cluster": cluster,
                            "score": score,
                            "jaccard": jaccard,
                            "sequence": sequence,
                            "shared_codes": sorted(
                                error_codes(left["title"]) & error_codes(right["title"])
                            ),
                        }
                    )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def select_pairs(rows, candidates):
    selected = []
    used = set()

    # Duplicate is deliberately strict. Same error code alone is not enough.
    for item in candidates:
        pair = tuple(sorted((item["left"]["question_id"], item["right"]["question_id"])))
        if pair in used:
            continue
        if item["jaccard"] >= 0.78 and item["sequence"] >= 0.72:
            item["label"] = "重复"
            item["reason"] = "核心关键词和问题表述高度一致，操作目标或故障现象相同"
            selected.append(item)
            used.add(pair)
        if sum(value["label"] == "重复" for value in selected) >= 25:
            break

    for item in candidates:
        pair = tuple(sorted((item["left"]["question_id"], item["right"]["question_id"])))
        if pair in used:
            continue
        if item["score"] >= 0.52:
            item["label"] = "相关但不重复"
            if item["shared_codes"]:
                item["reason"] = "共享错误码%s，但具体环境或解决目标可能不同" % ",".join(
                    item["shared_codes"]
                )
            else:
                item["reason"] = "属于相同技术主题，但操作对象、输入条件或具体诉求不同"
            selected.append(item)
            used.add(pair)
        if sum(value["label"] == "相关但不重复" for value in selected) >= 30:
            break

    rng = random.Random(42)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    attempts = 0
    while sum(value["label"] == "不相关" for value in selected) < 30 and attempts < 20000:
        attempts += 1
        left, right = rng.sample(shuffled, 2)
        if left["cluster"] == right["cluster"]:
            continue
        score, jaccard, sequence = similarity(left["title"], right["title"])
        if score > 0.08 or error_codes(left["title"]) & error_codes(right["title"]):
            continue
        pair = tuple(sorted((left["question_id"], right["question_id"])))
        if pair in used:
            continue
        selected.append(
            {
                "left": left,
                "right": right,
                "cluster": "",
                "score": score,
                "jaccard": jaccard,
                "sequence": sequence,
                "shared_codes": [],
                "label": "不相关",
                "reason": "核心主题和技术关键词不同，且来自不同聚类",
            }
        )
        used.add(pair)
    return selected


def load_question_details(question_ids):
    details = {}
    with QUESTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            question_id = str(row.get("question_id", ""))
            if question_id in question_ids:
                body = re.sub(r"<[^>]+>", " ", row.get("body") or "")
                body = re.sub(r"\s+", " ", body).strip()
                details[question_id] = {
                    "body_excerpt": body[:300],
                    "tags": "|".join(row.get("tags") or []),
                }
                if len(details) == len(question_ids):
                    break
    return details


def main():
    rows = read_assignments()
    selected = select_pairs(rows, candidate_pairs(rows))
    question_ids = {
        item[side]["question_id"] for item in selected for side in ("left", "right")
    }
    details = load_question_details(question_ids)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "question_id_1", "title_1", "body_excerpt_1", "tags_1",
        "question_id_2", "title_2", "body_excerpt_2", "tags_2",
        "initial_label", "label_code", "annotation_reason",
        "title_similarity", "same_baseline_cluster", "review_status",
        "reviewer_label", "reviewer_name", "review_notes",
    ]
    label_codes = {"重复": 1, "相关但不重复": 0, "不相关": 0}
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(selected, 1):
            left = item["left"]
            right = item["right"]
            left_detail = details.get(left["question_id"], {})
            right_detail = details.get(right["question_id"], {})
            writer.writerow(
                {
                    "pair_id": "P%03d" % index,
                    "question_id_1": left["question_id"],
                    "title_1": left["title"],
                    "body_excerpt_1": left_detail.get("body_excerpt", ""),
                    "tags_1": left_detail.get("tags", ""),
                    "question_id_2": right["question_id"],
                    "title_2": right["title"],
                    "body_excerpt_2": right_detail.get("body_excerpt", ""),
                    "tags_2": right_detail.get("tags", ""),
                    "initial_label": item["label"],
                    "label_code": label_codes[item["label"]],
                    "annotation_reason": item["reason"],
                    "title_similarity": "%.4f" % item["score"],
                    "same_baseline_cluster": int(left["cluster"] == right["cluster"]),
                    "review_status": "待人工复核",
                    "reviewer_label": "",
                    "reviewer_name": "",
                    "review_notes": "",
                }
            )
    counts = defaultdict(int)
    for item in selected:
        counts[item["label"]] += 1
    print("output=%s" % OUTPUT)
    print("total=%d duplicate=%d related=%d unrelated=%d" % (
        len(selected), counts["重复"], counts["相关但不重复"], counts["不相关"]
    ))


if __name__ == "__main__":
    main()
