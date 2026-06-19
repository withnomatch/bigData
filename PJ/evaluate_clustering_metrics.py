# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
import math
import re
import sys
import time

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, udf
from pyspark.sql.types import StringType


DOMAIN_STOP_WORDS = [
    "oracle", "database", "db", "sql", "question", "answer",
    "thanks", "thank", "please", "help", "hello", "hi",
]


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        return " ".join(self.result)


def clean_html(value):
    if not value:
        return ""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(value)
        text = extractor.get_text()
    except Exception:
        text = value
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(value):
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"\bora[\s_-]*(\d{4,5})\b", r"ora-\1", value)
    value = re.sub(r"\bpls[\s_-]*(\d{4,5})\b", r"pls-\1", value)
    value = re.sub(r"[^a-z0-9_\-\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def top_answer_text(answers, max_answers):
    if not answers:
        return ""
    parsed = []
    for item in answers:
        try:
            body = item["body"]
            score = item["score"] or 0
        except Exception:
            body = getattr(item, "body", "")
            score = getattr(item, "score", 0) or 0
        parsed.append((score, body))
    parsed.sort(key=lambda item: item[0], reverse=True)
    return " ".join(clean_html(body) for score, body in parsed[:max_answers])


def build_document(title, body, tags, answers):
    title_text = clean_html(title)
    body_text = clean_html(body)
    tag_text = " ".join(tags) if tags else ""
    answer_text = top_answer_text(answers, 2)
    return normalize_text(" ".join([
        title_text, title_text, title_text,
        body_text,
        tag_text, tag_text,
        answer_text,
    ]))


def distribution_metrics(predictions):
    rows = predictions.groupBy("cluster").agg(count("*").alias("cluster_size")).collect()
    sizes = [row["cluster_size"] for row in rows]
    total = sum(sizes)
    cluster_count = len(sizes)
    average = total / float(cluster_count)
    variance = sum((size - average) ** 2 for size in sizes) / float(cluster_count)
    probabilities = [size / float(total) for size in sizes]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "total_points": total,
        "cluster_count": cluster_count,
        "max_cluster_size": max(sizes),
        "min_cluster_size": min(sizes),
        "average_cluster_size": average,
        "largest_cluster_ratio": max(sizes) / float(total),
        "cluster_size_std": math.sqrt(variance),
        "cluster_size_cv": math.sqrt(variance) / average,
        "normalized_cluster_entropy": entropy / math.log(cluster_count),
        "singleton_clusters": sum(1 for size in sizes if size == 1),
        "small_clusters_lt_10": sum(1 for size in sizes if size < 10),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Spark clustering metrics")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("Evaluate_Clustering_Metrics").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    started = time.time()
    try:
        build_document_udf = udf(build_document, StringType())
        df = spark.read.json(args.input).withColumn(
            "document",
            build_document_udf(col("title"), col("body"), col("tags"), col("answers")),
        ).filter((col("document").isNotNull()) & (col("document") != "")).cache()
        valid_count = df.count()

        pipeline = Pipeline(stages=[
            Tokenizer(inputCol="document", outputCol="raw_tokens"),
            StopWordsRemover(
                inputCol="raw_tokens",
                outputCol="tokens",
                stopWords=StopWordsRemover.loadDefaultStopWords("english")
                + DOMAIN_STOP_WORDS,
            ),
            CountVectorizer(
                inputCol="tokens",
                outputCol="term_features",
                vocabSize=args.max_features,
                minDF=args.min_df,
            ),
            IDF(inputCol="term_features", outputCol="tfidf_features"),
            Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0),
        ])

        feature_started = time.time()
        features = pipeline.fit(df).transform(df).select("features").cache()
        features.count()
        feature_seconds = time.time() - feature_started

        train_started = time.time()
        model = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=args.k,
            maxIter=args.max_iter,
            seed=42,
        ).fit(features)
        predictions = model.transform(features).cache()
        predictions.count()
        training_seconds = time.time() - train_started

        evaluator = ClusteringEvaluator(
            predictionCol="cluster",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )
        silhouette = evaluator.evaluate(predictions)
        metrics = distribution_metrics(predictions)
        metrics.update({
            "input": args.input,
            "k": args.k,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_iter": args.max_iter,
            "valid_records": valid_count,
            "silhouette_squared_euclidean": silhouette,
            "wssse": model.computeCost(predictions),
            "feature_time_seconds": feature_seconds,
            "training_time_seconds": training_seconds,
            "total_time_seconds": time.time() - started,
        })

        text = json.dumps(metrics, ensure_ascii=False, indent=2)
        spark.sparkContext.parallelize([text], 1).saveAsTextFile(args.output)
        print(text)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
