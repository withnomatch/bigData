# -*- coding: utf-8 -*-
"""
Basic complete workflow for StackOverflow Oracle question deduplication.

Compatible with Spark 2.0 / Python 2.7.

Outputs:
  - cluster_assignments: question -> cluster
  - cluster_summary: cluster size statistics
  - top_questions_per_cluster: representative questions
  - cluster_keywords: frequent terms in each cluster
  - duplicate_candidates: high-similarity question pairs inside clusters
"""

from __future__ import print_function

import argparse
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
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    collect_list,
    count,
    desc,
    explode,
    row_number,
    udf,
)
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.window import Window


DOMAIN_STOP_WORDS = [
    "oracle",
    "database",
    "db",
    "sql",
    "question",
    "answer",
    "thanks",
    "thank",
    "please",
    "help",
    "hello",
    "hi",
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
    # Keep Oracle error codes readable before punctuation cleanup.
    value = re.sub(r"\bora[\s_-]*(\d{4,5})\b", r"ora-\1", value)
    value = re.sub(r"\bpls[\s_-]*(\d{4,5})\b", r"pls-\1", value)
    value = re.sub(r"[^a-z0-9_\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


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

    parsed.sort(key=lambda x: x[0], reverse=True)
    return " ".join([clean_html(body) for score, body in parsed[:max_answers]])


def build_document(title, body, tags, answers):
    title_text = clean_html(title)
    body_text = clean_html(body)
    tag_text = " ".join(tags) if tags else ""
    answer_text = top_answer_text(answers, 2)

    # Repeat title and tags because they usually summarize the problem better
    # than the long body/answers.
    combined = " ".join([
        title_text,
        title_text,
        title_text,
        body_text,
        tag_text,
        tag_text,
        answer_text,
    ])
    return normalize_text(combined)


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    try:
        return float(v1.dot(v2))
    except Exception:
        return 0.0


def create_spark_session():
    spark = SparkSession.builder.appName("StackOverflow_Complete_Dedup_Workflow").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, input_path, sample_ratio):
    print("\n[Step 1] Load data")
    df = spark.read.json(input_path)
    total = df.count()
    print("Total records: %d" % total)

    if sample_ratio < 1.0:
        df = df.sample(False, sample_ratio, 42)
        print("Sampled records: %d" % df.count())

    return df


def preprocess_data(df):
    print("\n[Step 2] Build question-answer document text")
    build_document_udf = udf(build_document, StringType())

    df = df.withColumn(
        "document",
        build_document_udf(col("title"), col("body"), col("tags"), col("answers")),
    )
    df = df.filter((col("document").isNotNull()) & (col("document") != ""))
    print("Valid records: %d" % df.count())
    return df


def build_pipeline(max_features, min_df):
    print("\n[Step 3] Build TF-IDF feature pipeline")
    print("max_features=%d, min_df=%d" % (max_features, min_df))

    tokenizer = Tokenizer(inputCol="document", outputCol="raw_tokens")
    remover = StopWordsRemover(
        inputCol="raw_tokens",
        outputCol="tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + DOMAIN_STOP_WORDS,
    )
    vectorizer = CountVectorizer(
        inputCol="tokens",
        outputCol="term_features",
        vocabSize=max_features,
        minDF=min_df,
    )
    idf = IDF(inputCol="term_features", outputCol="tfidf_features")
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)
    return Pipeline(stages=[tokenizer, remover, vectorizer, idf, normalizer])


def run_clustering(processed_df, k, max_iter):
    print("\n[Step 4] Run K-Means")
    print("k=%d, max_iter=%d" % (k, max_iter))
    model = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    ).fit(processed_df)
    return model.transform(processed_df)


def save_basic_outputs(predictions, output_path):
    print("\n[Step 5] Save cluster outputs")

    predictions.select(
        col("question_id"),
        col("title"),
        col("tags"),
        col("score").alias("question_score"),
        col("cluster"),
    ).write.mode("overwrite").json(output_path + "/cluster_assignments")

    predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy("cluster").write.mode("overwrite").json(output_path + "/cluster_summary")

    window = Window.partitionBy("cluster").orderBy(desc("score"))
    predictions.withColumn("rank", row_number().over(window)) \
        .filter(col("rank") <= 10) \
        .select("cluster", "rank", "question_id", "title", "score") \
        .orderBy("cluster", "rank") \
        .write.mode("overwrite").json(output_path + "/top_questions_per_cluster")

    print("Saved cluster_assignments, cluster_summary, top_questions_per_cluster")


def save_cluster_keywords(predictions, output_path, top_terms):
    print("\n[Step 6] Save cluster keywords")

    term_counts = predictions.select("cluster", explode(col("tokens")).alias("term")) \
        .filter(col("term") != "") \
        .groupBy("cluster", "term") \
        .agg(count("*").alias("term_count"))

    window = Window.partitionBy("cluster").orderBy(desc("term_count"))
    term_counts.withColumn("rank", row_number().over(window)) \
        .filter(col("rank") <= top_terms) \
        .groupBy("cluster") \
        .agg(collect_list("term").alias("keywords")) \
        .orderBy("cluster") \
        .write.mode("overwrite").json(output_path + "/cluster_keywords")

    print("Saved cluster_keywords")


def save_duplicate_candidates(predictions, output_path, top_per_cluster, threshold):
    print("\n[Step 7] Generate duplicate candidates")
    print("top_per_cluster=%d, threshold=%.3f" % (top_per_cluster, threshold))

    window = Window.partitionBy("cluster").orderBy(desc("score"))
    candidates = predictions.withColumn("dedup_rank", row_number().over(window)) \
        .filter(col("dedup_rank") <= top_per_cluster) \
        .select("cluster", "question_id", "title", "score", "features")

    left = candidates.alias("a")
    right = candidates.alias("b")

    cosine_udf = udf(cosine_similarity, DoubleType())

    pairs = left.join(
        right,
        (col("a.cluster") == col("b.cluster")) & (col("a.question_id") < col("b.question_id")),
    ).withColumn(
        "similarity",
        cosine_udf(col("a.features"), col("b.features")),
    ).filter(
        col("similarity") >= threshold
    ).select(
        col("a.cluster").alias("cluster"),
        col("a.question_id").alias("question_id_1"),
        col("a.title").alias("title_1"),
        col("b.question_id").alias("question_id_2"),
        col("b.title").alias("title_2"),
        col("similarity"),
    ).orderBy(desc("similarity"))

    pairs = pairs.cache()
    dup_count = pairs.count()
    pairs.write.mode("overwrite").json(output_path + "/duplicate_candidates")
    print("Duplicate candidate pairs: %d" % dup_count)
    print("Saved duplicate_candidates")


def print_summary(predictions):
    print("\n[Step 8] Result summary")
    cluster_sizes = predictions.groupBy("cluster").agg(count("*").alias("cluster_size"))
    cluster_sizes.orderBy(desc("cluster_size")).show(20, truncate=False)

    stats = cluster_sizes.collect()
    sizes = [row["cluster_size"] for row in stats]
    if sizes:
        print("Total clusters: %d" % len(sizes))
        print("Max cluster size: %d" % max(sizes))
        print("Min cluster size: %d" % min(sizes))
        print("Average cluster size: %.2f" % (sum(sizes) / float(len(sizes))))


def main():
    parser = argparse.ArgumentParser(description="Complete StackOverflow dedup clustering workflow")
    parser.add_argument("--input", required=True, help="Input JSONLines path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--k", type=int, default=50, help="Number of clusters")
    parser.add_argument("--max-features", type=int, default=5000, help="CountVectorizer vocabulary size")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency")
    parser.add_argument("--sample-ratio", type=float, default=0.1, help="Sampling ratio")
    parser.add_argument("--max-iter", type=int, default=30, help="KMeans max iterations")
    parser.add_argument("--top-terms", type=int, default=10, help="Top keyword count per cluster")
    parser.add_argument("--dedup-top-per-cluster", type=int, default=80, help="Max scored questions per cluster for pair search")
    parser.add_argument("--similarity-threshold", type=float, default=0.55, help="Cosine threshold for duplicate candidates")
    args = parser.parse_args()

    start = time.time()
    print("=" * 70)
    print("StackOverflow Oracle Question Deduplication - Basic Complete Workflow")
    print("=" * 70)
    print("Input: %s" % args.input)
    print("Output: %s" % args.output)
    print("K: %d" % args.k)
    print("Sample ratio: %.3f" % args.sample_ratio)
    print("=" * 70)

    spark = create_spark_session()
    try:
        df = load_data(spark, args.input, args.sample_ratio)
        df = preprocess_data(df)

        pipeline = build_pipeline(args.max_features, args.min_df)
        pipeline_model = pipeline.fit(df)
        processed_df = pipeline_model.transform(df).cache()
        print("Feature pipeline fitted. Records: %d" % processed_df.count())

        predictions = run_clustering(processed_df, args.k, args.max_iter).cache()
        print("Predictions generated. Records: %d" % predictions.count())

        save_basic_outputs(predictions, args.output)
        save_cluster_keywords(predictions, args.output, args.top_terms)
        save_duplicate_candidates(
            predictions,
            args.output,
            args.dedup_top_per_cluster,
            args.similarity_threshold,
        )
        print_summary(predictions)

        print("\nWorkflow completed in %.2fs" % (time.time() - start))
        print("Results saved to: %s" % args.output)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
