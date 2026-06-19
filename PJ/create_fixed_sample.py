# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse

from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser(description="Create one reusable Spark sample")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Create_Fixed_Clustering_Sample").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        source = spark.read.json(args.input)
        total = source.count()
        sample = source.sample(False, args.ratio, args.seed).cache()
        sample_count = sample.count()
        sample.write.mode("overwrite").json(args.output)
        print("FIXED_SAMPLE total=%d sample=%d ratio=%.4f seed=%d" % (
            total, sample_count, args.ratio, args.seed
        ))
        print("FIXED_SAMPLE_PATH %s" % args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
