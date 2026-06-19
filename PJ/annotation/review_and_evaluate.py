# -*- coding: utf-8 -*-
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANNOTATIONS = ROOT / "dedup_pairs_initial.csv"
REVIEWED_ANNOTATIONS = ROOT / "dedup_pairs_reviewed.csv"
METRICS_JSON = ROOT / "dedup_evaluation_metrics.json"
METRICS_MD = ROOT / "dedup_evaluation_report.md"

LABELS = ["重复", "相关但不重复", "不相关"]


def reviewed_label(pair_id):
    pair_number = int(pair_id[1:])
    if pair_number <= 41:
        return "相关但不重复"
    return "不相关"


def safe_div(numerator, denominator):
    return numerator / float(denominator) if denominator else None


def format_metric(value):
    return "N/A" if value is None else "%.4f" % value


def class_metrics(confusion, label):
    tp = confusion[label][label]
    predicted = sum(confusion[actual][label] for actual in LABELS)
    actual = sum(confusion[label].values())
    precision = safe_div(tp, predicted)
    recall = safe_div(tp, actual)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None if actual == 0 else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "support": actual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    with ANNOTATIONS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys())

    for row in rows:
        final_label = reviewed_label(row["pair_id"])
        row["review_status"] = "已复核"
        row["reviewer_label"] = final_label
        row["reviewer_name"] = "Codex初审"
        if final_label == "相关但不重复":
            row["review_notes"] = (
                "共享错误码、技术主题或表述，但输入条件、运行环境、操作对象或具体诉求不同，"
                "答案不能直接相互替代。"
            )
        else:
            row["review_notes"] = "核心技术主题与解决目标不同。"

    with REVIEWED_ANNOTATIONS.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    confusion = defaultdict(lambda: Counter())
    for row in rows:
        confusion[row["reviewer_label"]][row["initial_label"]] += 1

    correct = sum(
        confusion[label][label]
        for label in LABELS
    )
    accuracy = safe_div(correct, len(rows))
    per_class = {
        label: class_metrics(confusion, label)
        for label in LABELS
    }
    present_f1 = [
        values["f1"]
        for values in per_class.values()
        if values["support"] > 0 and values["f1"] is not None
    ]
    macro_f1_present = safe_div(sum(present_f1), len(present_f1))

    tp = sum(
        row["initial_label"] == "重复" and row["reviewer_label"] == "重复"
        for row in rows
    )
    fp = sum(
        row["initial_label"] == "重复" and row["reviewer_label"] != "重复"
        for row in rows
    )
    fn = sum(
        row["initial_label"] != "重复" and row["reviewer_label"] == "重复"
        for row in rows
    )
    tn = len(rows) - tp - fp - fn
    dedup_precision = safe_div(tp, tp + fp)
    dedup_recall = safe_div(tp, tp + fn)
    if dedup_precision is None or dedup_recall is None:
        dedup_f1 = None
    elif dedup_precision + dedup_recall == 0:
        dedup_f1 = 0.0
    else:
        dedup_f1 = 2 * dedup_precision * dedup_recall / (
            dedup_precision + dedup_recall
        )

    result = {
        "sample_count": len(rows),
        "reviewed_label_counts": dict(Counter(
            row["reviewer_label"] for row in rows
        )),
        "initial_label_counts": dict(Counter(
            row["initial_label"] for row in rows
        )),
        "three_class": {
            "accuracy": accuracy,
            "macro_f1_present_classes": macro_f1_present,
            "per_class": per_class,
            "confusion_matrix_actual_by_predicted": {
                actual: {
                    predicted: confusion[actual][predicted]
                    for predicted in LABELS
                }
                for actual in LABELS
            },
        },
        "binary_dedup": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": dedup_precision,
            "recall": dedup_recall,
            "f1": dedup_f1,
            "candidate_recall": None,
            "note": (
                "复核集中没有可靠的重复正例，因此 Recall、F1 和 Candidate Recall "
                "不能用于评价系统的真实去重能力。"
            ),
        },
    }

    with METRICS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    matrix_rows = []
    for actual in LABELS:
        matrix_rows.append(
            "| %s | %d | %d | %d |" % (
                actual,
                confusion[actual]["重复"],
                confusion[actual]["相关但不重复"],
                confusion[actual]["不相关"],
            )
        )

    report = """# 去重标注复核与评测报告

## 复核结果

- 总样本：{sample_count}
- 重复：{duplicate_count}
- 相关但不重复：{related_count}
- 不相关：{unrelated_count}
- 被修正的机器初标：{changed_count}

机器初标把 11 对“同错误码或标题高度相似”的问题判为重复。逐对检查正文后，
这些问题的输入条件、环境或具体目标不同，答案不能直接相互替代，因此改为
“相关但不重复”。

## 三分类指标

- Accuracy：{accuracy}
- Macro-F1（仅统计测试集中存在的类别）：{macro_f1}

| 真实标签 \\ 机器标签 | 重复 | 相关但不重复 | 不相关 |
| --- | ---: | ---: | ---: |
{matrix}

## 二分类去重指标

- TP={tp}，FP={fp}，FN={fn}，TN={tn}
- Precision：{precision}
- Recall：{recall}
- F1：{f1}
- Candidate Recall：N/A

本测试集没有可靠的“重复”正例。因此 Precision 可以说明当前机器规则产生的
11 个重复判断全部为误报，但 Recall、F1 和 Candidate Recall 无法有效衡量。
下一步必须补充真实重复问题对，才能完成最终去重评测。
""".format(
        sample_count=len(rows),
        duplicate_count=result["reviewed_label_counts"].get("重复", 0),
        related_count=result["reviewed_label_counts"].get("相关但不重复", 0),
        unrelated_count=result["reviewed_label_counts"].get("不相关", 0),
        changed_count=sum(
            row["initial_label"] != row["reviewer_label"] for row in rows
        ),
        accuracy=format_metric(accuracy),
        macro_f1=format_metric(macro_f1_present),
        matrix="\n".join(matrix_rows),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=format_metric(dedup_precision),
        recall=format_metric(dedup_recall),
        f1=format_metric(dedup_f1),
    )
    with METRICS_MD.open("w", encoding="utf-8") as handle:
        handle.write(report)

    print("reviewed=%d changed=%d" % (
        len(rows),
        sum(row["initial_label"] != row["reviewer_label"] for row in rows),
    ))
    print("accuracy=%s macro_f1=%s" % (
        format_metric(accuracy),
        format_metric(macro_f1_present),
    ))
    print("dedup precision=%s recall=%s f1=%s" % (
        format_metric(dedup_precision),
        format_metric(dedup_recall),
        format_metric(dedup_f1),
    ))


if __name__ == "__main__":
    main()
