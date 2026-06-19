# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col


def main():
    parser = argparse.ArgumentParser(description="Create one reusable Spark sample")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--annotation-pairs",
        help=(
            "Optional reviewed pair CSV. When provided, all question_id_1 and "
            "question_id_2 rows are forced into the reusable sample."
        ),
    )
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Create_Fixed_Clustering_Sample").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        source = spark.read.json(args.input)
        total = source.count()
        source_with_id = source.withColumn("_sample_qid", col("question_id").cast("string"))
        source_columns = source_with_id.columns
        sample = source_with_id.sample(False, args.ratio, args.seed)
        random_sample_count = sample.count()
        annotation_question_count = 0

        if args.annotation_pairs:
            pairs = spark.read.csv(args.annotation_pairs, header=True)
            pair_ids = pairs.select(col("question_id_1").alias("_sample_qid")).union(
                pairs.select(col("question_id_2").alias("_sample_qid"))
            ).distinct()
            annotation_question_count = pair_ids.count()
            annotated = source_with_id.join(pair_ids, "_sample_qid", "inner")
            sample = sample.select(source_columns).union(
                annotated.select(source_columns)
            )

        sample = sample.dropDuplicates(["_sample_qid"]).drop("_sample_qid").cache()
        sample_count = sample.count()
        sample.write.mode("overwrite").json(args.output)
        print(
            "FIXED_SAMPLE total=%d random_sample=%d annotation_questions=%d "
            "final_sample=%d ratio=%.4f seed=%d"
            % (
                total, random_sample_count, annotation_question_count,
                sample_count, args.ratio, args.seed,
            )
        )
        print("FIXED_SAMPLE_PATH %s" % args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
