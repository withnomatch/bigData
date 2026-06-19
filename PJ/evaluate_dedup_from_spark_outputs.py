# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, greatest, least


DUPLICATE_LABEL = u"重复"


def safe_div(numerator, denominator):
    return numerator / float(denominator) if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Spark dedup outputs with reviewed pair labels"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Evaluate_Dedup_With_Reviewed_Pairs").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        pairs = spark.read.option("multiLine", True).option(
            "escape", '"'
        ).csv(args.pairs, header=True).select(
            col("pair_id"),
            col("question_id_1").cast("string").alias("question_id_1"),
            col("question_id_2").cast("string").alias("question_id_2"),
            col("reviewer_label"),
        )
        assignments = spark.read.json(args.assignments).select(
            col("question_id").cast("string").alias("question_id"),
            col("cluster").alias("cluster"),
        )
        candidates = spark.read.json(args.candidates).select(
            least(
                col("question_id_1").cast("string"),
                col("question_id_2").cast("string"),
            ).alias("candidate_left"),
            greatest(
                col("question_id_1").cast("string"),
                col("question_id_2").cast("string"),
            ).alias("candidate_right"),
            col("similarity").alias("candidate_similarity"),
        )

        left = assignments.select(
            col("question_id").alias("question_id_1"),
            col("cluster").alias("cluster_1"),
        )
        right = assignments.select(
            col("question_id").alias("question_id_2"),
            col("cluster").alias("cluster_2"),
        )
        evaluated = pairs.join(left, "question_id_1", "left").join(
            right, "question_id_2", "left"
        ).withColumn(
            "pair_left", least(col("question_id_1"), col("question_id_2"))
        ).withColumn(
            "pair_right", greatest(col("question_id_1"), col("question_id_2"))
        ).join(
            candidates,
            (col("pair_left") == col("candidate_left"))
            & (col("pair_right") == col("candidate_right")),
            "left",
        )

        rows = evaluated.select(
            "pair_id", "question_id_1", "question_id_2", "reviewer_label",
            "cluster_1", "cluster_2", "candidate_similarity",
        ).collect()

        output_rows = []
        tp = fp = fn = tn = 0
        positive_total = 0
        positive_same_cluster = 0
        missing_assignment = 0
        for row in rows:
            actual = row["reviewer_label"] == DUPLICATE_LABEL
            same_cluster = (
                row["cluster_1"] is not None
                and row["cluster_2"] is not None
                and row["cluster_1"] == row["cluster_2"]
            )
            predicted = row["candidate_similarity"] is not None
            if row["cluster_1"] is None or row["cluster_2"] is None:
                missing_assignment += 1
            if actual:
                positive_total += 1
                if same_cluster:
                    positive_same_cluster += 1
            if actual and predicted:
                tp += 1
            elif (not actual) and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1
            output_rows.append({
                "pair_id": row["pair_id"],
                "question_id_1": row["question_id_1"],
                "question_id_2": row["question_id_2"],
                "reviewer_label": row["reviewer_label"],
                "cluster_1": row["cluster_1"],
                "cluster_2": row["cluster_2"],
                "same_cluster": same_cluster,
                "predicted_duplicate": predicted,
                "similarity": row["candidate_similarity"],
            })

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        metrics = {
            "pair_count": len(rows),
            "missing_assignment": missing_assignment,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "candidate_recall": safe_div(positive_same_cluster, positive_total),
            "positive_total": positive_total,
            "positive_same_cluster": positive_same_cluster,
            "prediction_rule": (
                "Predicted duplicate when the reviewed pair appears in "
                "duplicate_candidates generated by the Spark baseline."
            ),
        }

        payload = {"metrics": metrics, "pairs": output_rows}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        spark.sparkContext.parallelize([text], 1).saveAsTextFile(args.output)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
