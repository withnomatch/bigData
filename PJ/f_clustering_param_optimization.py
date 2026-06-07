# -*- coding: utf-8 -*-
"""
F-task: K-Means execution parameter optimization for StackOverflow text clustering.

Compatible with Spark 2.0 / Python 2.7.
"""

from __future__ import print_function

import argparse
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
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, StopWordsRemover, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, row_number, udf
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window


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


def parse_csv(value, cast=str):
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(cast(item))
    return result


def normalize_init_mode(init_mode):
    normalized = init_mode.strip().lower()
    if normalized in ("kmeans++", "k-means++", "kmeanspp", "k-meanspp"):
        return "k-means||", "k-means||"
    if normalized in ("k-means||", "kmeans||", "parallel"):
        return "k-means||", "k-means||"
    if normalized == "random":
        return "random", "random"
    raise ValueError("Unsupported init mode: %s" % init_mode)


def normalize_distance_mode(distance_mode):
    normalized = distance_mode.strip().lower()
    if normalized in ("cos", "cosine"):
        return "cosine"
    if normalized in ("euc", "euclidean"):
        return "euclidean"
    raise ValueError("Unsupported distance mode: %s" % distance_mode)


def vector_pairs(vec):
    if hasattr(vec, "indices") and hasattr(vec, "values"):
        return zip(vec.indices, vec.values)
    return enumerate(vec)


def vector_norm2(vec):
    total = 0.0
    for _, value in vector_pairs(vec):
        total += float(value) * float(value)
    return total


def dot_with_dense(vec, dense_center):
    total = 0.0
    for index, value in vector_pairs(vec):
        total += float(value) * float(dense_center[int(index)])
    return total


def center_stats(centers):
    stats = []
    for center in centers:
        values = [float(v) for v in center]
        norm2 = sum(v * v for v in values)
        stats.append({
            "values": values,
            "norm": math.sqrt(norm2),
            "norm2": norm2,
        })
    return stats


def euclidean_distance(vec, center_stat):
    dist2 = vector_norm2(vec) + center_stat["norm2"] - 2.0 * dot_with_dense(vec, center_stat["values"])
    if dist2 < 0.0 and dist2 > -1e-9:
        dist2 = 0.0
    return math.sqrt(max(0.0, dist2))


def cosine_distance(vec, center_stat):
    vec_norm = math.sqrt(vector_norm2(vec))
    center_norm = center_stat["norm"]
    if vec_norm == 0.0 or center_norm == 0.0:
        return 1.0
    similarity = dot_with_dense(vec, center_stat["values"]) / (vec_norm * center_norm)
    if similarity > 1.0:
        similarity = 1.0
    if similarity < -1.0:
        similarity = -1.0
    return 1.0 - similarity


def approximate_center_silhouette(predictions, feature_col, centers, distance_mode, max_sample, total_points):
    if max_sample <= 0:
        return -1.0, 0

    sample_fraction = 1.0
    if total_points > max_sample:
        sample_fraction = min(1.0, float(max_sample) * 1.5 / float(total_points))

    rows = predictions.select(
        col("cluster"),
        col(feature_col),
    ).sample(False, sample_fraction, seed=42).limit(max_sample).collect()
    if not rows or len(centers) <= 1:
        return -1.0, len(rows)

    stats = center_stats(centers)
    silhouette_sum = 0.0
    used_count = 0

    for row in rows:
        cluster = int(row["cluster"])
        vec = row[feature_col]
        own_distance = None
        nearest_other = None

        for idx, center_stat in enumerate(stats):
            if distance_mode == "cosine":
                dist = cosine_distance(vec, center_stat)
            else:
                dist = euclidean_distance(vec, center_stat)

            if idx == cluster:
                own_distance = dist
            else:
                if nearest_other is None or dist < nearest_other:
                    nearest_other = dist

        if own_distance is None or nearest_other is None:
            continue

        denom = max(own_distance, nearest_other)
        if denom == 0.0:
            silhouette = 0.0
        else:
            silhouette = (nearest_other - own_distance) / denom
        silhouette_sum += silhouette
        used_count += 1

    if used_count == 0:
        return -1.0, len(rows)
    return silhouette_sum / float(used_count), used_count


def create_spark_session():
    spark = SparkSession.builder.appName("F_KMeans_Execution_Param_Optimization").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_preprocess(spark, input_path, sample_ratio):
    print("\n" + "=" * 60)
    print("[Step 1] Loading and preprocessing data")
    print("=" * 60)

    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total records: %d" % total_count)

    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        sampled_count = df.count()
        print("Sampled records: %d (ratio: %.4f)" % (sampled_count, sample_ratio))

    preprocess_udf = udf(preprocess_text, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))

    valid_count = df.count()
    print("Valid records: %d" % valid_count)
    return df


def build_feature_dataframe(df, max_features, min_df):
    print("\n" + "=" * 60)
    print("[Step 2] Building TF-IDF features")
    print("=" * 60)
    print("TF-IDF params: max_features=%d, min_df=%d" % (max_features, min_df))
    print("Feature columns:")
    print("  euclidean -> tfidf_features (raw TF-IDF)")
    print("  cosine    -> features (L2-normalized TF-IDF; cosine-equivalent in Spark 2.0)")

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

    start_time = time.time()
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    processed_df = processed_df.select(
        col("question_id"),
        col("title"),
        col("score"),
        col("tfidf_features"),
        col("features"),
    ).cache()
    feature_count = processed_df.count()

    elapsed = time.time() - start_time
    vocab_size = len(pipeline_model.stages[2].vocabulary)
    print("Feature rows cached: %d" % feature_count)
    print("Actual vocabulary size: %d" % vocab_size)
    print("Pipeline fitting + cache time: %.2fs" % elapsed)

    return pipeline_model, processed_df


def make_focused_experiments(init_modes, distance_modes, max_iters, seeds):
    experiments = []

    default_init = "k-means||" if "k-means||" in init_modes else init_modes[0]
    default_distance = "cosine" if "cosine" in distance_modes else distance_modes[0]
    default_iter = 20 if 20 in max_iters else max_iters[0]
    default_seed = seeds[0]

    for init_mode in init_modes:
        experiments.append(("init", init_mode, default_distance, default_iter, default_seed))

    for distance_mode in distance_modes:
        experiments.append(("distance", default_init, distance_mode, default_iter, default_seed))

    for max_iter in max_iters:
        experiments.append(("iteration", default_init, default_distance, max_iter, default_seed))

    for seed in seeds:
        experiments.append(("multi_seed", default_init, default_distance, default_iter, seed))

    deduped = []
    seen = set()
    for item in experiments:
        key = item[1:]
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def make_grid_experiments(init_modes, distance_modes, max_iters, seeds):
    experiments = []
    for init_mode in init_modes:
        for distance_mode in distance_modes:
            for max_iter in max_iters:
                for seed in seeds:
                    experiments.append(("grid", init_mode, distance_mode, max_iter, seed))
    return experiments


def run_single_experiment(processed_df, args, exp_id, stage, init_label, spark_init, distance_mode, max_iter, seed):
    feature_col = "features" if distance_mode == "cosine" else "tfidf_features"

    print("\n" + "-" * 60)
    print("Experiment %s | stage=%s | init=%s | distance=%s | maxIter=%d | seed=%d" % (
        exp_id, stage, init_label, distance_mode, max_iter, seed
    ))
    print("-" * 60)

    kmeans = KMeans(
        featuresCol=feature_col,
        predictionCol="cluster",
        k=args.k,
        maxIter=max_iter,
        seed=seed,
        initMode=spark_init,
    )

    train_start = time.time()
    model = kmeans.fit(processed_df)
    predictions = model.transform(processed_df)
    train_time = time.time() - train_start

    cost = -1.0
    if not args.skip_cost:
        try:
            cost = float(model.computeCost(processed_df))
        except Exception as exc:
            print("computeCost unavailable: %s" % exc)

    cluster_sizes = predictions.groupBy("cluster").agg(count("*").alias("cluster_size")).collect()
    sizes = [int(row["cluster_size"]) for row in cluster_sizes]
    total_points = sum(sizes)
    cluster_count = len(sizes)
    max_cluster_size = max(sizes) if sizes else 0
    min_cluster_size = min(sizes) if sizes else 0
    avg_cluster_size = float(total_points) / cluster_count if cluster_count else 0.0
    singleton_clusters = len([s for s in sizes if s == 1])
    largest_ratio = float(max_cluster_size) / total_points if total_points else 0.0

    silhouette, silhouette_samples = approximate_center_silhouette(
        predictions,
        feature_col,
        model.clusterCenters(),
        distance_mode,
        args.silhouette_sample,
        total_points,
    )

    result = {
        "exp_id": exp_id,
        "stage": stage,
        "init_mode": init_label,
        "spark_init_mode": spark_init,
        "distance_mode": distance_mode,
        "feature_col": feature_col,
        "max_iter": max_iter,
        "seed": seed,
        "training_time_sec": train_time,
        "training_cost": cost,
        "center_silhouette": silhouette,
        "silhouette_samples": silhouette_samples,
        "total_points": total_points,
        "cluster_count": cluster_count,
        "max_cluster_size": max_cluster_size,
        "min_cluster_size": min_cluster_size,
        "avg_cluster_size": avg_cluster_size,
        "singleton_clusters": singleton_clusters,
        "largest_cluster_ratio": largest_ratio,
    }

    print("Training time: %.2fs" % train_time)
    print("Cost: %.4f" % cost)
    print("Approx center silhouette: %.4f (sample=%d)" % (silhouette, silhouette_samples))
    print("Clusters: %d, max/min/avg size: %d/%d/%.2f, singleton=%d, largest_ratio=%.4f" % (
        cluster_count, max_cluster_size, min_cluster_size, avg_cluster_size,
        singleton_clusters, largest_ratio
    ))

    return result, predictions


def result_schema():
    return StructType([
        StructField("exp_id", StringType(), False),
        StructField("stage", StringType(), False),
        StructField("init_mode", StringType(), False),
        StructField("spark_init_mode", StringType(), False),
        StructField("distance_mode", StringType(), False),
        StructField("feature_col", StringType(), False),
        StructField("max_iter", IntegerType(), False),
        StructField("seed", IntegerType(), False),
        StructField("training_time_sec", DoubleType(), False),
        StructField("training_cost", DoubleType(), False),
        StructField("center_silhouette", DoubleType(), False),
        StructField("silhouette_samples", IntegerType(), False),
        StructField("total_points", LongType(), False),
        StructField("cluster_count", IntegerType(), False),
        StructField("max_cluster_size", LongType(), False),
        StructField("min_cluster_size", LongType(), False),
        StructField("avg_cluster_size", DoubleType(), False),
        StructField("singleton_clusters", IntegerType(), False),
        StructField("largest_cluster_ratio", DoubleType(), False),
    ])


def result_tuple(row):
    return (
        row["exp_id"],
        row["stage"],
        row["init_mode"],
        row["spark_init_mode"],
        row["distance_mode"],
        row["feature_col"],
        int(row["max_iter"]),
        int(row["seed"]),
        float(row["training_time_sec"]),
        float(row["training_cost"]),
        float(row["center_silhouette"]),
        int(row["silhouette_samples"]),
        long(row["total_points"]) if sys.version_info[0] < 3 else int(row["total_points"]),
        int(row["cluster_count"]),
        long(row["max_cluster_size"]) if sys.version_info[0] < 3 else int(row["max_cluster_size"]),
        long(row["min_cluster_size"]) if sys.version_info[0] < 3 else int(row["min_cluster_size"]),
        float(row["avg_cluster_size"]),
        int(row["singleton_clusters"]),
        float(row["largest_cluster_ratio"]),
    )


def choose_better(candidate, current):
    # Reject visibly degenerate clusterings before comparing compactness.
    # A high center-based silhouette can be misleading when nearly every point
    # collapses into one cluster, as observed in experiment F03.
    candidate_valid = (
        candidate["largest_cluster_ratio"] <= 0.50
        and candidate["singleton_clusters"] <= max(1, candidate["cluster_count"] // 10)
    )
    if not candidate_valid:
        return False
    if current is None:
        return True
    current_valid = (
        current["largest_cluster_ratio"] <= 0.50
        and current["singleton_clusters"] <= max(1, current["cluster_count"] // 10)
    )
    if not current_valid:
        return True
    if candidate["center_silhouette"] > current["center_silhouette"]:
        return True
    if candidate["center_silhouette"] == current["center_silhouette"]:
        if candidate["training_cost"] >= 0 and current["training_cost"] >= 0:
            if candidate["training_cost"] < current["training_cost"]:
                return True
        if candidate["training_time_sec"] < current["training_time_sec"]:
            return True
    return False


def print_summary(results):
    print("\n" + "=" * 100)
    print("F-task experiment summary")
    print("=" * 100)
    header = "%-6s %-10s %-10s %-10s %-7s %-7s %-10s %-10s %-10s %-10s"
    print(header % ("id", "stage", "init", "distance", "iter", "seed", "time(s)", "cost", "sil", "largest"))
    for row in results:
        print(header % (
            row["exp_id"],
            row["stage"],
            row["init_mode"],
            row["distance_mode"],
            row["max_iter"],
            row["seed"],
            "%.2f" % row["training_time_sec"],
            "%.2f" % row["training_cost"],
            "%.4f" % row["center_silhouette"],
            "%.4f" % row["largest_cluster_ratio"],
        ))
    print("=" * 100)


def save_outputs(spark, output_path, results, best_result, best_predictions):
    print("\n" + "=" * 60)
    print("[Step 4] Saving F-task outputs")
    print("=" * 60)

    schema = result_schema()
    result_df = spark.createDataFrame([result_tuple(row) for row in results], schema=schema)
    result_df.write.mode("overwrite").json(output_path + "/experiment_results")
    print("Experiment result table saved: %s/experiment_results" % output_path)

    best_df = spark.createDataFrame([result_tuple(best_result)], schema=schema)
    best_df.write.mode("overwrite").json(output_path + "/best_config")
    print("Best config saved: %s/best_config" % output_path)

    cluster_summary = best_predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy("cluster")
    cluster_summary.write.mode("overwrite").json(output_path + "/best_cluster_summary")
    print("Best cluster summary saved: %s/best_cluster_summary" % output_path)

    output_df = best_predictions.select(
        col("question_id"),
        col("title"),
        col("cluster"),
        col("score").alias("question_score"),
    )
    output_df.write.mode("overwrite").json(output_path + "/best_cluster_assignments")
    print("Best cluster assignments saved: %s/best_cluster_assignments" % output_path)

    window = Window.partitionBy("cluster").orderBy(desc("score"))
    top_per_cluster = best_predictions.withColumn(
        "rank", row_number().over(window)
    ).filter(col("rank") <= 10).select(
        col("cluster"),
        col("rank"),
        col("question_id"),
        col("title"),
        col("score"),
    ).orderBy("cluster", "rank")
    top_per_cluster.write.mode("overwrite").json(output_path + "/best_top_questions_per_cluster")
    print("Best top questions saved: %s/best_top_questions_per_cluster" % output_path)


def main():
    parser = argparse.ArgumentParser(description="F-task KMeans execution parameter optimization")
    parser.add_argument("--input", type=str, required=True, help="Input JSON/JSONLines data path")
    parser.add_argument("--output", type=str, required=True, help="Output result path")
    parser.add_argument("--k", type=int, default=50, help="Number of clusters")
    parser.add_argument("--max-features", type=int, default=5000, help="Max vocabulary size")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency")
    parser.add_argument("--sample-ratio", type=float, default=0.1, help="Sampling ratio")
    parser.add_argument("--plan", type=str, default="focused", choices=["focused", "grid"], help="Experiment plan")
    parser.add_argument("--init-modes", type=str, default="random,k-means||", help="Comma-separated init modes")
    parser.add_argument("--distance-modes", type=str, default="euclidean,cosine", help="Comma-separated distance modes")
    parser.add_argument("--max-iters", type=str, default="10,20,30", help="Comma-separated maxIter values")
    parser.add_argument("--seeds", type=str, default="42,2026,3407", help="Comma-separated seeds for multi-run tuning")
    parser.add_argument("--silhouette-sample", type=int, default=2000, help="Max rows for center-silhouette approximation")
    parser.add_argument("--skip-cost", action="store_true", help="Skip model.computeCost to reduce runtime")
    args = parser.parse_args()

    init_modes = []
    for init_mode in parse_csv(args.init_modes, str):
        spark_init, label = normalize_init_mode(init_mode)
        if label not in init_modes:
            init_modes.append(label)
    distance_modes = []
    for distance_mode in parse_csv(args.distance_modes, str):
        normalized_distance = normalize_distance_mode(distance_mode)
        if normalized_distance not in distance_modes:
            distance_modes.append(normalized_distance)
    max_iters = parse_csv(args.max_iters, int)
    seeds = parse_csv(args.seeds, int)

    if args.plan == "grid":
        experiments = make_grid_experiments(init_modes, distance_modes, max_iters, seeds)
    else:
        experiments = make_focused_experiments(init_modes, distance_modes, max_iters, seeds)

    print("\n" + "=" * 60)
    print("F-task: KMeans execution parameter optimization")
    print("=" * 60)
    print("Input: %s" % args.input)
    print("Output: %s" % args.output)
    print("K: %d" % args.k)
    print("Plan: %s, experiments: %d" % (args.plan, len(experiments)))
    print("Init modes: %s" % ",".join(init_modes))
    print("Distance modes: %s" % ",".join(distance_modes))
    print("Max iters: %s" % ",".join([str(x) for x in max_iters]))
    print("Seeds: %s" % ",".join([str(x) for x in seeds]))
    print("=" * 60)

    spark = create_spark_session()
    total_start = time.time()
    processed_df = None

    try:
        raw_df = load_and_preprocess(spark, args.input, args.sample_ratio)
        _, processed_df = build_feature_dataframe(raw_df, args.max_features, args.min_df)

        print("\n" + "=" * 60)
        print("[Step 3] Running parameter experiments")
        print("=" * 60)

        results = []
        best_result = None
        best_predictions = None

        for idx, item in enumerate(experiments, 1):
            stage, init_label, distance_mode, max_iter, seed = item
            spark_init, normalized_label = normalize_init_mode(init_label)
            exp_id = "F%02d" % idx
            result, predictions = run_single_experiment(
                processed_df,
                args,
                exp_id,
                stage,
                normalized_label,
                spark_init,
                distance_mode,
                max_iter,
                seed,
            )
            results.append(result)

            if choose_better(result, best_result):
                best_result = result
                best_predictions = predictions

        print_summary(results)
        print("\nBest config: id=%s, init=%s, distance=%s, maxIter=%d, seed=%d, sil=%.4f, time=%.2fs" % (
            best_result["exp_id"],
            best_result["init_mode"],
            best_result["distance_mode"],
            best_result["max_iter"],
            best_result["seed"],
            best_result["center_silhouette"],
            best_result["training_time_sec"],
        ))

        save_outputs(spark, args.output, results, best_result, best_predictions)

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 60)
        print("F-task optimization complete")
        print("Total time: %.2fs" % total_elapsed)
        print("Results saved to: %s" % args.output)
        print("=" * 60)

    finally:
        if processed_df is not None:
            processed_df.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
