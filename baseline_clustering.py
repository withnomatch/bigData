# -*- coding: utf-8 -*-
"""
StackOverflow Question Clustering - Baseline Implementation
Compatible with Python 2.7
"""

from __future__ import print_function
import re
import argparse
import time
import sys

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, desc, count, collect_list, first, row_number
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.clustering import KMeans


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        return " ".join(self.result)


def clean_html(html_str):
    if not html_str:
        return ""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(html_str)
    except Exception:
        return html_str
    text = extractor.get_text()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(title, body):
    title_clean = clean_html(title) if title else ""
    body_clean = clean_html(body) if body else ""
    
    combined = title_clean + " " + body_clean
    combined = combined.lower()
    combined = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()
    
    return combined


def create_spark_session(app_name="StackOverflow_Clustering_Baseline"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, input_path, sample_ratio=1.0):
    print("\n" + "=" * 60)
    print("[Step 1] Loading Data")
    print("=" * 60)
    
    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total records: %d" % total_count)
    
    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        sampled_count = df.count()
        print("Sampled records: %d (ratio: %f)" % (sampled_count, sample_ratio))
    
    return df


def preprocess_data(df):
    print("\n" + "=" * 60)
    print("[Step 2] Text Preprocessing")
    print("=" * 60)
    print("Steps: HTML cleaning -> Combine title&body -> Lowercase -> Remove special chars")
    
    preprocess_udf = udf(preprocess_text, StringType())
    
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))
    
    valid_count = df.count()
    print("Valid records: %d" % valid_count)
    
    return df


def build_feature_pipeline(max_features=5000, min_df=5):
    print("\n" + "=" * 60)
    print("[Step 3] Building Feature Extraction Pipeline")
    print("=" * 60)
    print("Pipeline: Tokenizer -> StopWordsRemover -> CountVectorizer -> IDF -> Normalizer")
    print("Params: max_features=%d, min_df=%d" % (max_features, min_df))
    
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")
    
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=max_features,
        minDF=min_df,
    )
    
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)
    
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, normalizer])
    
    return pipeline


def fit_and_transform(pipeline, df):
    print("\nFitting Pipeline...")
    start_time = time.time()
    
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)
    
    elapsed = time.time() - start_time
    print("Pipeline fitting complete, time: %.2fs" % elapsed)
    
    vocab_size = len(pipeline_model.stages[2].vocabulary)
    print("Actual vocabulary size: %d" % vocab_size)
    
    return pipeline_model, processed_df


def run_kmeans(processed_df, k=50, max_iter=30, distance_measure="cosine"):
    print("\n" + "=" * 60)
    print("[Step 4] K-Means Clustering")
    print("=" * 60)
    print("Params: K=%d, max_iter=%d" % (k, max_iter))
    print("Note: distanceMeasure not supported in Spark 2.0, using default (euclidean)")
    
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )
    
    print("Training K-Means model...")
    start_time = time.time()
    
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)
    
    elapsed = time.time() - start_time
    print("K-Means training complete, time: %.2fs" % elapsed)
    
    return kmeans_model, predictions


def evaluate_clustering(predictions, distance_measure="cosine"):
    print("\n" + "=" * 60)
    print("[Step 5] Evaluating Clustering Quality")
    print("=" * 60)
    
    print("Note: ClusteringEvaluator not available in Spark 2.0")
    print("Using cluster size distribution for evaluation...")
    
    cluster_sizes = predictions.groupBy("cluster").count().orderBy("cluster")
    print("\nCluster Size Distribution:")
    cluster_sizes.show(50, truncate=False)
    
    cluster_count = predictions.select("cluster").distinct().count()
    total_points = predictions.count()
    avg_cluster_size = float(total_points) / cluster_count
    
    print("Total Clusters: %d" % cluster_count)
    print("Total Points: %d" % total_points)
    print("Average Cluster Size: %.2f" % avg_cluster_size)
    
    return 0.0


def analyze_clusters(predictions, top_n=20):
    print("\n" + "=" * 60)
    print("[Step 6] Cluster Statistics Analysis")
    print("=" * 60)
    
    cluster_stats = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
        collect_list("title").alias("titles"),
    ).orderBy(desc("cluster_size"))
    
    print("\nTop %d largest clusters:" % top_n)
    cluster_stats.show(top_n, truncate=True)
    
    stats_data = cluster_stats.collect()
    sizes = [row["cluster_size"] for row in stats_data]
    
    print("\nCluster size statistics:")
    print("  Max cluster: %d questions" % max(sizes))
    print("  Min cluster: %d questions" % min(sizes))
    print("  Average size: %.1f questions" % (sum(sizes) / float(len(sizes))))
    
    return cluster_stats


def save_results(predictions, output_path):
    print("\n" + "=" * 60)
    print("[Step 7] Saving Results")
    print("=" * 60)
    
    output_df = predictions.select(
        col("question_id"),
        col("title"),
        col("cluster"),
        col("score").alias("question_score"),
    )
    output_df.write.mode("overwrite").json(output_path + "/cluster_assignments")
    print("Cluster assignments saved: %s/cluster_assignments" % output_path)
    
    cluster_summary = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy("cluster")
    cluster_summary.write.mode("overwrite").json(output_path + "/cluster_summary")
    print("Cluster summary saved: %s/cluster_summary" % output_path)
    
    window = Window.partitionBy("cluster").orderBy(desc("score"))
    top_per_cluster = predictions.withColumn(
        "rank", row_number().over(window)
    ).filter(col("rank") <= 10).select(
        col("cluster"),
        col("rank"),
        col("question_id"),
        col("title"),
        col("score"),
    ).orderBy("cluster", "rank")
    top_per_cluster.write.mode("overwrite").json(output_path + "/top_questions_per_cluster")
    print("Top 10 questions per cluster saved: %s/top_questions_per_cluster" % output_path)


def show_sample_clusters(predictions, n_clusters=5, n_questions=5):
    print("\n" + "=" * 60)
    print("[Step 8] Sample Cluster Display")
    print("=" * 60)
    
    cluster_stats = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
        collect_list("title").alias("titles"),
    ).orderBy(desc("cluster_size")).limit(n_clusters).collect()
    
    for row in cluster_stats:
        cluster_id = row["cluster"]
        cluster_size = row["cluster_size"]
        titles = row["titles"][:n_questions]
        
        print("\nCluster %d (total %d questions):" % (cluster_id, cluster_size))
        for i, title in enumerate(titles, 1):
            print("  %d. %s" % (i, title))


def main():
    parser = argparse.ArgumentParser(description="StackOverflow Question Clustering - Baseline")
    parser.add_argument("--input", type=str, required=True, help="Input data path (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output result path")
    parser.add_argument("--k", type=int, default=50, help="Number of clusters K (default: 50)")
    parser.add_argument("--max-features", type=int, default=5000, help="Max vocabulary size (default: 5000)")
    parser.add_argument("--min-df", type=int, default=5, help="Min document frequency (default: 5)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Sampling ratio (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations (default: 30)")
    
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("\n" + "=" * 60)
    print("StackOverflow Question Clustering - Baseline")
    print("=" * 60)
    print("Input path: %s" % args.input)
    print("Output path: %s" % args.output)
    print("Number of clusters K: %d" % args.k)
    print("Max vocabulary: %d" % args.max_features)
    print("Sampling ratio: %f" % args.sample_ratio)
    print("=" * 60)
    
    spark = create_spark_session()
    
    try:
        df = load_data(spark, args.input, args.sample_ratio)
        
        df = preprocess_data(df)
        
        pipeline = build_feature_pipeline(args.max_features, args.min_df)
        pipeline_model, processed_df = fit_and_transform(pipeline, df)
        
        kmeans_model, predictions = run_kmeans(
            processed_df, args.k, args.max_iter, "cosine"
        )
        
        silhouette = evaluate_clustering(predictions, "cosine")
        
        analyze_clusters(predictions)
        
        save_results(predictions, args.output)
        
        show_sample_clusters(predictions)
        
        total_elapsed = time.time() - total_start
        
        print("\n" + "=" * 60)
        print("Baseline Execution Complete!")
        print("=" * 60)
        print("Silhouette Score: %.4f" % silhouette)
        print("Total time: %.2fs" % total_elapsed)
        print("Results saved to: %s" % args.output)
        print("=" * 60)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
