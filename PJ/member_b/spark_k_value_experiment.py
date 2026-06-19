# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import math
import re
import time
try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

    def handle_data(self, data):
        self.result.append(data)


def clean_html(value):
    if not value:
        return ""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(value)
    except Exception:
        return value
    return re.sub(r"\s+", " ", " ".join(extractor.result)).strip()


def preprocess_text(title, body):
    text = "%s %s" % (clean_html(title), clean_html(body))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def distribution_metrics(predictions):
    sizes = [row["count"] for row in predictions.groupBy("cluster").count().collect()]
    total = sum(sizes)
    avg = total / float(len(sizes))
    variance = sum((size - avg) ** 2 for size in sizes) / len(sizes)
    probabilities = [size / float(total) for size in sizes]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "total_points": total,
        "cluster_count": len(sizes),
        "max_cluster_size": max(sizes),
        "min_cluster_size": min(sizes),
        "largest_cluster_ratio": max(sizes) / float(total),
        "cluster_size_cv": math.sqrt(variance) / avg,
        "normalized_cluster_entropy": entropy / math.log(len(sizes)),
    }


def main():
    parser = argparse.ArgumentParser(description="Spark K=40..60 validation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--k-min", type=int, default=40)
    parser.add_argument("--k-max", type=int, default=60)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Member_B_Spark_K_Selection").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    clean = udf(preprocess_text, StringType())

    df = spark.read.json(args.input)
    if args.sample_ratio < 1.0:
        df = df.sample(False, args.sample_ratio, seed=42)
    df = (
        df.withColumn("clean_text", clean(col("title"), col("body")))
        .filter(col("clean_text") != "")
        .cache()
    )
    sample_count = df.count()
    print("Unified sample count: %d" % sample_count)

    pipeline = Pipeline(
        stages=[
            Tokenizer(inputCol="clean_text", outputCol="raw_tokens"),
            StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens"),
            CountVectorizer(
                inputCol="filtered_tokens",
                outputCol="raw_features",
                vocabSize=args.max_features,
                minDF=args.min_df,
            ),
            IDF(inputCol="raw_features", outputCol="tfidf_features"),
            Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0),
        ]
    )
    feature_start = time.time()
    features = pipeline.fit(df).transform(df).select("features").cache()
    features.count()
    feature_time = time.time() - feature_start

    evaluator = ClusteringEvaluator(
        predictionCol="cluster",
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    results = []
    for k in range(args.k_min, args.k_max + 1):
        start = time.time()
        model = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            maxIter=args.max_iter,
            seed=42,
        ).fit(features)
        predictions = model.transform(features).cache()
        predictions.count()
        training_time = time.time() - start
        metrics = distribution_metrics(predictions)
        metrics.update(
            {
                "k": k,
                "silhouette": evaluator.evaluate(predictions),
                "wssse": model.computeCost(predictions),
                "training_time_sec": training_time,
                "feature_time_sec": feature_time,
            }
        )
        results.append(metrics)
        print("K_RESULT " + json.dumps(metrics, sort_keys=True))
        predictions.unpersist()

    rows = [json.dumps(row, sort_keys=True) for row in results]
    spark.sparkContext.parallelize(rows, 1).saveAsTextFile(args.output)
    spark.stop()


if __name__ == "__main__":
    main()
