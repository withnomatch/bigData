import re
import argparse
from html.parser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, size, concat, lit
from pyspark.sql.types import StringType, ArrayType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    CountVectorizer,
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
    parser = argparse.ArgumentParser(description="StackOverflow Question Clustering with Spark")
    parser.add_argument("--input", type=str, required=True, help="HDFS input path for JSON data")
    parser.add_argument("--output", type=str, required=True, help="HDFS output path for clustering results")
    parser.add_argument("--k", type=int, default=50, help="Number of clusters for K-Means")
    parser.add_argument("--max-features", type=int, default=5000, help="Max vocabulary size for CountVectorizer")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency for vocabulary")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Fraction of data to use (0.0-1.0)")
    parser.add_argument("--use-hashing-tf", action="store_true", help="Use HashingTF instead of CountVectorizer")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("StackOverflow_Question_Clustering") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("StackOverflow Question Clustering - Spark Job")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"K (clusters): {args.k}")
    print(f"Max features: {args.max_features}")
    print(f"Sample ratio: {args.sample_ratio}")
    print("=" * 60)

    # ---- Step 1: Load Data ----
    print("\n[Step 1] Loading data...")
    df = spark.read.json(args.input)
    print(f"Total records loaded: {df.count()}")

    if args.sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=args.sample_ratio, seed=42)
        print(f"Sampled records: {df.count()}")

    # ---- Step 2: Preprocess Text ----
    print("\n[Step 2] Preprocessing text (cleaning HTML, combining title+body)...")
    preprocess_udf = udf(preprocess_text, StringType())

    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))
    print(f"Records after filtering empty text: {df.count()}")

    # ---- Step 3: Build ML Pipeline ----
    print("\n[Step 3] Building ML Pipeline (Tokenizer -> StopWords -> TF-IDF -> Normalizer)...")

    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")

    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")

    if args.use_hashing_tf:
        tf = HashingTF(
            inputCol="filtered_tokens",
            outputCol="raw_features",
            numFeatures=args.max_features,
        )
    else:
        tf = CountVectorizer(
            inputCol="filtered_tokens",
            outputCol="raw_features",
            vocabSize=args.max_features,
            minDF=args.min_df,
        )

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, normalizer])

    print("Fitting pipeline...")
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    print(f"Pipeline fitting complete. Features dimension: {args.max_features}")

    # ---- Step 4: K-Means Clustering ----
    print(f"\n[Step 4] Running K-Means clustering with K={args.k}...")
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=args.k,
        maxIter=30,
        seed=42,
        distanceMeasure="cosine",
    )

    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    # ---- Step 5: Evaluate ----
    print("\n[Step 5] Evaluating clustering...")
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="cosine",
    )

    silhouette = evaluator.evaluate(predictions)
    print(f"Silhouette Score (cosine): {silhouette:.4f}")

    # ---- Step 6: Analyze Clusters ----
    print("\n[Step 6] Cluster statistics:")
    from pyspark.sql.functions import count, desc, collect_list, first

    cluster_stats = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
        collect_list("title").alias("titles"),
    ).orderBy(desc("cluster_size"))

    cluster_stats.show(20, truncate=False)

    # ---- Step 7: Save Results ----
    print(f"\n[Step 7] Saving results to {args.output}...")

    output_df = predictions.select(
        col("question_id"),
        col("title"),
        col("cluster"),
        col("score").alias("question_score"),
    )

    output_df.write.mode("overwrite").json(args.output + "/cluster_assignments")

    cluster_summary = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy("cluster")

    cluster_summary.write.mode("overwrite").json(args.output + "/cluster_summary")

    # Save top questions per cluster for inspection
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

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

    top_per_cluster.write.mode("overwrite").json(args.output + "/top_questions_per_cluster")

    # ---- Step 8: Print Sample Duplicates ----
    print("\n[Step 8] Sample potential duplicate clusters (clusters with > 1 question):")
    large_clusters = cluster_stats.filter(col("cluster_size") > 1).limit(10).collect()
    for row in large_clusters:
        cluster_id = row["cluster"]
        titles = row["titles"][:5]
        print(f"\n  Cluster {cluster_id} ({row['cluster_size']} questions):")
        for t in titles:
            print(f"    - {t}")

    print("\n" + "=" * 60)
    print("Clustering job completed!")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Results saved to: {args.output}")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
