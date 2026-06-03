import re
import json
from html.parser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, desc
from pyspark.sql.types import StringType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
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


def main():
    spark = SparkSession.builder \
        .appName("LocalTest_Clustering") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    DATA_PATH = "./StackOverFlow_Oracle_Database/oracle_database_questions.json"
    K = 30
    SAMPLE_RATIO = 0.05
    MAX_FEATURES = 3000

    print("=" * 60)
    print("LOCAL TEST - StackOverflow Question Clustering")
    print(f"Sample ratio: {SAMPLE_RATIO}")
    print(f"K: {K}")
    print("=" * 60)

    # Step 1: Load
    print("\n[Step 1] Loading data...")
    df = spark.read.json(DATA_PATH)
    total = df.count()
    print(f"Total records: {total}")

    df = df.sample(withReplacement=False, fraction=SAMPLE_RATIO, seed=42)
    sampled = df.count()
    print(f"Sampled records: {sampled}")

    # Step 2: Preprocess
    print("\n[Step 2] Preprocessing text...")
    preprocess_udf = udf(preprocess_text, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))
    print(f"Records after filtering: {df.count()}")

    # Step 3: Pipeline
    print("\n[Step 3] Building and fitting ML Pipeline...")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    cv = CountVectorizer(
        inputCol="filtered_tokens", outputCol="raw_features",
        vocabSize=MAX_FEATURES, minDF=3,
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, normalizer])
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)
    print("Pipeline fitting complete.")

    # Step 4: K-Means
    print(f"\n[Step 4] K-Means clustering (K={K})...")
    kmeans = KMeans(
        featuresCol="features", predictionCol="cluster",
        k=K, maxIter=30, seed=42, distanceMeasure="cosine",
    )
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    # Step 5: Evaluate
    print("\n[Step 5] Evaluating...")
    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="cluster",
        metricName="silhouette", distanceMeasure="cosine",
    )
    silhouette = evaluator.evaluate(predictions)
    print(f"Silhouette Score (cosine): {silhouette:.4f}")

    # Step 6: Show clusters
    print("\n[Step 6] Cluster distribution:")
    from pyspark.sql.functions import count
    cluster_stats = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy(desc("cluster_size"))
    cluster_stats.show(20, truncate=False)

    # Step 7: Show sample questions per cluster
    print("\n[Step 7] Sample questions from top clusters:")
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

    window = Window.partitionBy("cluster").orderBy(desc("score"))
    top_per_cluster = predictions.withColumn(
        "rank", row_number().over(window)
    ).filter(col("rank") <= 5).select(
        col("cluster"), col("rank"), col("question_id"), col("title"), col("score"),
    ).orderBy("cluster", "rank")

    top_per_cluster.show(50, truncate=80)

    # Save local results
    print("\n[Step 8] Saving local results...")
    output_path = "./local_test_output"
    predictions.select(
        col("question_id"), col("title"), col("cluster"), col("score"),
    ).coalesce(1).write.mode("overwrite").csv(output_path + "/assignments", header=True)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    print("LOCAL TEST COMPLETE!")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
