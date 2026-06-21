# -*- coding: utf-8 -*-
"""
F-task: K-Means execution parameter optimization with 300-pair dedup evaluation.

The script is compatible with Spark 2.x and Python 2.7/3.x. It keeps the feature
pipeline close to the baseline: HTML cleaning -> TF-IDF -> L2 normalization.
For each K-Means execution setting, it evaluates both cluster distribution and
the reviewed 300 duplicate-detection pairs.
"""

from __future__ import print_function

import argparse
import csv
import math
import re
import sys
import time

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    from StringIO import StringIO
else:
    from io import StringIO

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


def to_unicode(value):
    if value is None:
        return u""
    if sys.version_info[0] < 3:
        if isinstance(value, unicode):
            return value
        return value.decode("utf-8", "ignore")
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


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


def parse_csv_arg(value, cast=str):
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
    if vec is None:
        return []
    if hasattr(vec, "indices") and hasattr(vec, "values"):
        return zip(vec.indices, vec.values)
    return enumerate(vec)


def vector_norm2(vec):
    total = 0.0
    for _, value in vector_pairs(vec):
        total += float(value) * float(value)
    return total


def dot_sparse_sparse(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    if hasattr(vec_a, "indices") and hasattr(vec_b, "indices"):
        ia = list(vec_a.indices)
        ib = list(vec_b.indices)
        va = list(vec_a.values)
        vb = list(vec_b.values)
        i = 0
        j = 0
        total = 0.0
        while i < len(ia) and j < len(ib):
            if ia[i] == ib[j]:
                total += float(va[i]) * float(vb[j])
                i += 1
                j += 1
            elif ia[i] < ib[j]:
                i += 1
            else:
                j += 1
        return total

    values_b = dict((int(idx), float(value)) for idx, value in vector_pairs(vec_b))
    total = 0.0
    for idx, value in vector_pairs(vec_a):
        total += float(value) * values_b.get(int(idx), 0.0)
    return total


def cosine_similarity(vec_a, vec_b):
    norm = math.sqrt(vector_norm2(vec_a)) * math.sqrt(vector_norm2(vec_b))
    if norm == 0.0:
        return 0.0
    sim = dot_sparse_sparse(vec_a, vec_b) / norm
    if sim > 1.0:
        return 1.0
    if sim < -1.0:
        return -1.0
    return sim


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
    return 1.0 - cosine_similarity(vec, center_stat["values"])


def approximate_center_silhouette(predictions, feature_col, centers, distance_mode, max_sample, total_points):
    if max_sample <= 0:
        return -1.0, 0

    sample_fraction = 1.0
    if total_points > max_sample:
        sample_fraction = min(1.0, float(max_sample) * 1.5 / float(total_points))

    rows = predictions.select(col("cluster"), col(feature_col)).sample(False, sample_fraction, seed=42).limit(max_sample).collect()
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
        silhouette = 0.0 if denom == 0.0 else (nearest_other - own_distance) / denom
        silhouette_sum += silhouette
        used_count += 1

    if used_count == 0:
        return -1.0, len(rows)
    return silhouette_sum / float(used_count), used_count


def create_spark_session():
    spark = SparkSession.builder.appName("F_KMeans_Execution_Dedup_Eval").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_preprocess(spark, input_path, sample_ratio):
    print("\n" + "=" * 60)
    print("[Step 1] Loading fixed evaluation sample")
    print("=" * 60)

    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total input records: %d" % total_count)

    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        print("Sampled records: %d (ratio=%.4f)" % (df.count(), sample_ratio))

    preprocess_udf = udf(preprocess_text, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))

    valid_count = df.count()
    print("Valid records after preprocessing: %d" % valid_count)
    return df


def build_feature_dataframe(df, max_features, min_df):
    print("\n" + "=" * 60)
    print("[Step 2] Building TF-IDF features once")
    print("=" * 60)
    print("TF-IDF params: max_features=%d, min_df=%d" % (max_features, min_df))

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
    processed_df = pipeline_model.transform(df).select(
        col("question_id").cast("string").alias("question_id"),
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
    return pipeline_model, processed_df, feature_count, elapsed


def load_reviewed_pairs(spark, pair_path):
    print("\n" + "=" * 60)
    print("[Step 3] Loading reviewed 300-pair dedup labels")
    print("=" * 60)
    print("Pair path: %s" % pair_path)

    lines = spark.sparkContext.textFile(pair_path).collect()
    if not lines:
        raise ValueError("Empty pair CSV: %s" % pair_path)

    if sys.version_info[0] < 3:
        csv_text = "\n".join([line.encode("utf-8") if isinstance(line, unicode) else line for line in lines])
        reader = csv.DictReader(StringIO(csv_text))
    else:
        csv_text = "\n".join(lines)
        reader = csv.DictReader(StringIO(csv_text))

    rows = []
    label_counts = {}
    for raw_row in reader:
        clean_row = {}
        for key, value in raw_row.items():
            clean_key = to_unicode(key).lstrip(u"\ufeff")
            clean_row[clean_key] = to_unicode(value)

        label = clean_row.get(u"reviewer_label", u"")
        is_duplicate = 1 if label == u"重复" else 0
        rows.append((
            clean_row.get(u"pair_id", u""),
            clean_row.get(u"question_id_1", u""),
            clean_row.get(u"question_id_2", u""),
            label,
            is_duplicate,
        ))
        label_counts[label] = label_counts.get(label, 0) + 1

    schema = StructType([
        StructField("pair_id", StringType(), False),
        StructField("question_id_1", StringType(), False),
        StructField("question_id_2", StringType(), False),
        StructField("reviewer_label", StringType(), False),
        StructField("is_duplicate", IntegerType(), False),
    ])
    pairs_df = spark.createDataFrame(rows, schema=schema).cache()
    pairs_count = pairs_df.count()
    positive_count = pairs_df.filter(col("is_duplicate") == 1).count()
    print("Reviewed pairs loaded: %d" % pairs_count)
    print("Positive duplicate pairs: %d" % positive_count)
    return pairs_df, pairs_count, positive_count


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


def evaluate_pairs(predictions, pairs_df, exp_id, threshold):
    vector_df = predictions.select(
        col("question_id").alias("qid"),
        col("cluster").alias("cluster"),
        col("features").alias("cosine_features"),
    )

    joined = pairs_df.alias("p") \
        .join(vector_df.alias("a"), col("p.question_id_1") == col("a.qid"), "left_outer") \
        .join(vector_df.alias("b"), col("p.question_id_2") == col("b.qid"), "left_outer") \
        .select(
            col("p.pair_id").alias("pair_id"),
            col("p.question_id_1").alias("question_id_1"),
            col("p.question_id_2").alias("question_id_2"),
            col("p.reviewer_label").alias("reviewer_label"),
            col("p.is_duplicate").alias("is_duplicate"),
            col("a.cluster").alias("cluster_1"),
            col("b.cluster").alias("cluster_2"),
            col("a.cosine_features").alias("features_1"),
            col("b.cosine_features").alias("features_2"),
        )

    rows = joined.collect()
    details = []
    tp = fp = tn = fn = 0
    positive_pairs = 0
    positive_same_cluster = 0
    matched_pairs = 0

    for row in rows:
        is_duplicate = int(row["is_duplicate"])
        if is_duplicate == 1:
            positive_pairs += 1

        found_both = row["features_1"] is not None and row["features_2"] is not None
        if found_both:
            matched_pairs += 1
            same_cluster = 1 if row["cluster_1"] == row["cluster_2"] else 0
            sim = cosine_similarity(row["features_1"], row["features_2"])
        else:
            same_cluster = 0
            sim = 0.0

        if is_duplicate == 1 and same_cluster == 1:
            positive_same_cluster += 1

        predicted_duplicate = 1 if same_cluster == 1 and sim >= threshold else 0

        if predicted_duplicate == 1 and is_duplicate == 1:
            tp += 1
        elif predicted_duplicate == 1 and is_duplicate == 0:
            fp += 1
        elif predicted_duplicate == 0 and is_duplicate == 1:
            fn += 1
        else:
            tn += 1

        details.append((
            exp_id,
            row["pair_id"],
            row["question_id_1"],
            row["question_id_2"],
            row["reviewer_label"],
            is_duplicate,
            int(row["cluster_1"]) if row["cluster_1"] is not None else -1,
            int(row["cluster_2"]) if row["cluster_2"] is not None else -1,
            same_cluster,
            float(sim),
            predicted_duplicate,
            found_both,
        ))

    precision = float(tp) / float(tp + fp) if (tp + fp) else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    candidate_recall = float(positive_same_cluster) / float(positive_pairs) if positive_pairs else 0.0

    metrics = {
        "eval_pairs": len(rows),
        "matched_pairs": matched_pairs,
        "positive_pairs": positive_pairs,
        "candidate_recall": candidate_recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return metrics, details


def run_single_experiment(processed_df, pairs_df, args, exp_id, stage, init_label, spark_init, distance_mode, max_iter, seed):
    feature_col = "features" if distance_mode == "cosine" else "tfidf_features"

    print("\n" + "-" * 80)
    print("Experiment %s | stage=%s | init=%s | distance=%s | maxIter=%d | seed=%d" % (
        exp_id, stage, init_label, distance_mode, max_iter, seed
    ))
    print("-" * 80)

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
    predictions = model.transform(processed_df).cache()
    predictions.count()
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

    pair_metrics, pair_details = evaluate_pairs(predictions, pairs_df, exp_id, args.threshold)

    result = {
        "exp_id": exp_id,
        "stage": stage,
        "init_mode": init_label,
        "spark_init_mode": spark_init,
        "distance_mode": distance_mode,
        "feature_col": feature_col,
        "max_iter": max_iter,
        "seed": seed,
        "threshold": args.threshold,
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
    result.update(pair_metrics)

    print("Training time: %.2fs, cost: %.4f, center silhouette: %.4f" % (train_time, cost, silhouette))
    print("Clusters: %d, max/min/avg=%d/%d/%.2f, singleton=%d, largest_ratio=%.4f" % (
        cluster_count, max_cluster_size, min_cluster_size, avg_cluster_size,
        singleton_clusters, largest_ratio
    ))
    print("Dedup eval: P=%.4f R=%.4f F1=%.4f CandidateRecall=%.4f TP/FP/FN/TN=%d/%d/%d/%d" % (
        result["precision"], result["recall"], result["f1"], result["candidate_recall"],
        result["tp"], result["fp"], result["fn"], result["tn"]
    ))

    return result, pair_details, predictions


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
        StructField("threshold", DoubleType(), False),
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
        StructField("eval_pairs", IntegerType(), False),
        StructField("matched_pairs", IntegerType(), False),
        StructField("positive_pairs", IntegerType(), False),
        StructField("candidate_recall", DoubleType(), False),
        StructField("precision", DoubleType(), False),
        StructField("recall", DoubleType(), False),
        StructField("f1", DoubleType(), False),
        StructField("tp", IntegerType(), False),
        StructField("fp", IntegerType(), False),
        StructField("tn", IntegerType(), False),
        StructField("fn", IntegerType(), False),
    ])


def result_tuple(row):
    long_type = long if sys.version_info[0] < 3 else int
    return (
        row["exp_id"],
        row["stage"],
        row["init_mode"],
        row["spark_init_mode"],
        row["distance_mode"],
        row["feature_col"],
        int(row["max_iter"]),
        int(row["seed"]),
        float(row["threshold"]),
        float(row["training_time_sec"]),
        float(row["training_cost"]),
        float(row["center_silhouette"]),
        int(row["silhouette_samples"]),
        long_type(row["total_points"]),
        int(row["cluster_count"]),
        long_type(row["max_cluster_size"]),
        long_type(row["min_cluster_size"]),
        float(row["avg_cluster_size"]),
        int(row["singleton_clusters"]),
        float(row["largest_cluster_ratio"]),
        int(row["eval_pairs"]),
        int(row["matched_pairs"]),
        int(row["positive_pairs"]),
        float(row["candidate_recall"]),
        float(row["precision"]),
        float(row["recall"]),
        float(row["f1"]),
        int(row["tp"]),
        int(row["fp"]),
        int(row["tn"]),
        int(row["fn"]),
    )


def pair_detail_schema():
    return StructType([
        StructField("exp_id", StringType(), False),
        StructField("pair_id", StringType(), False),
        StructField("question_id_1", StringType(), False),
        StructField("question_id_2", StringType(), False),
        StructField("reviewer_label", StringType(), False),
        StructField("is_duplicate", IntegerType(), False),
        StructField("cluster_1", IntegerType(), False),
        StructField("cluster_2", IntegerType(), False),
        StructField("same_cluster", IntegerType(), False),
        StructField("cosine_similarity", DoubleType(), False),
        StructField("predicted_duplicate", IntegerType(), False),
        StructField("found_both_questions", IntegerType(), False),
    ])


def normalize_pair_detail_tuple(item):
    return (
        item[0],
        item[1],
        item[2],
        item[3],
        item[4],
        int(item[5]),
        int(item[6]),
        int(item[7]),
        int(item[8]),
        float(item[9]),
        int(item[10]),
        1 if item[11] else 0,
    )


def is_degenerate(row, max_largest_ratio):
    return row is not None and row["largest_cluster_ratio"] > max_largest_ratio


def choose_better(candidate, current, max_largest_ratio):
    if current is None:
        return True
    candidate_degenerate = is_degenerate(candidate, max_largest_ratio)
    current_degenerate = is_degenerate(current, max_largest_ratio)
    if candidate_degenerate and not current_degenerate:
        return False
    if current_degenerate and not candidate_degenerate:
        return True

    eps = 1e-12
    for metric in ("f1", "candidate_recall", "precision", "center_silhouette"):
        if candidate[metric] > current[metric] + eps:
            return True
        if candidate[metric] < current[metric] - eps:
            return False
    if candidate["largest_cluster_ratio"] < current["largest_cluster_ratio"] - eps:
        return True
    if candidate["largest_cluster_ratio"] > current["largest_cluster_ratio"] + eps:
        return False
    return candidate["training_time_sec"] < current["training_time_sec"]


def print_summary(results):
    print("\n" + "=" * 120)
    print("F-task experiment summary")
    print("=" * 120)
    header = "%-5s %-10s %-10s %-9s %-4s %-5s %-8s %-8s %-8s %-8s %-8s %-8s"
    print(header % ("id", "stage", "init", "dist", "iter", "seed", "time", "sil", "maxRatio", "CandR", "P", "F1"))
    for row in results:
        print(header % (
            row["exp_id"],
            row["stage"],
            row["init_mode"],
            row["distance_mode"],
            row["max_iter"],
            row["seed"],
            "%.2f" % row["training_time_sec"],
            "%.4f" % row["center_silhouette"],
            "%.4f" % row["largest_cluster_ratio"],
            "%.4f" % row["candidate_recall"],
            "%.4f" % row["precision"],
            "%.4f" % row["f1"],
        ))
    print("=" * 120)


def save_outputs(spark, output_path, results, all_pair_details, best_result, best_pair_details, best_predictions):
    print("\n" + "=" * 60)
    print("[Step 5] Saving F-task outputs")
    print("=" * 60)

    schema = result_schema()
    result_df = spark.createDataFrame([result_tuple(row) for row in results], schema=schema)
    result_df.write.mode("overwrite").json(output_path + "/experiment_results")
    print("Experiment results saved: %s/experiment_results" % output_path)

    best_df = spark.createDataFrame([result_tuple(best_result)], schema=schema)
    best_df.write.mode("overwrite").json(output_path + "/best_config")
    print("Best config saved: %s/best_config" % output_path)

    detail_schema = pair_detail_schema()
    detail_df = spark.createDataFrame([normalize_pair_detail_tuple(row) for row in all_pair_details], schema=detail_schema)
    detail_df.write.mode("overwrite").json(output_path + "/pair_eval_details")
    print("All pair eval details saved: %s/pair_eval_details" % output_path)

    best_detail_df = spark.createDataFrame([normalize_pair_detail_tuple(row) for row in best_pair_details], schema=detail_schema)
    best_detail_df.write.mode("overwrite").json(output_path + "/best_pair_eval_details")
    print("Best pair eval details saved: %s/best_pair_eval_details" % output_path)

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
    print("Best assignments saved: %s/best_cluster_assignments" % output_path)

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
    parser = argparse.ArgumentParser(description="F-task KMeans execution parameters + dedup evaluation")
    parser.add_argument("--input", type=str, required=True, help="Input fixed JSON/JSONLines sample path")
    parser.add_argument("--pairs", type=str, required=True, help="Reviewed 300-pair CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output HDFS result path")
    parser.add_argument("--k", type=int, default=50, help="Number of clusters")
    parser.add_argument("--max-features", type=int, default=5000, help="Max vocabulary size")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum document frequency")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Optional sample ratio; keep 1.0 for fixed sample")
    parser.add_argument("--threshold", type=float, default=0.55, help="Cosine threshold for duplicate prediction")
    parser.add_argument("--max-largest-ratio", type=float, default=0.5, help="Reject configs whose largest cluster ratio is above this value when selecting the comprehensive best config")
    parser.add_argument("--plan", type=str, default="focused", choices=["focused", "grid"], help="Experiment plan")
    parser.add_argument("--init-modes", type=str, default="random,k-means||", help="Comma-separated init modes")
    parser.add_argument("--distance-modes", type=str, default="euclidean,cosine", help="Comma-separated cluster distance modes")
    parser.add_argument("--max-iters", type=str, default="10,20,30", help="Comma-separated maxIter values")
    parser.add_argument("--seeds", type=str, default="42,2026,3407", help="Comma-separated random seeds")
    parser.add_argument("--silhouette-sample", type=int, default=2000, help="Rows sampled for center-silhouette approximation")
    parser.add_argument("--skip-cost", action="store_true", help="Skip model.computeCost to reduce runtime")
    args = parser.parse_args()

    init_modes = []
    for init_mode in parse_csv_arg(args.init_modes, str):
        _, label = normalize_init_mode(init_mode)
        if label not in init_modes:
            init_modes.append(label)

    distance_modes = []
    for distance_mode in parse_csv_arg(args.distance_modes, str):
        normalized = normalize_distance_mode(distance_mode)
        if normalized not in distance_modes:
            distance_modes.append(normalized)

    max_iters = parse_csv_arg(args.max_iters, int)
    seeds = parse_csv_arg(args.seeds, int)

    experiments = make_grid_experiments(init_modes, distance_modes, max_iters, seeds) if args.plan == "grid" else make_focused_experiments(init_modes, distance_modes, max_iters, seeds)

    print("\n" + "=" * 80)
    print("F-task: KMeans execution parameter optimization with dedup evaluation")
    print("=" * 80)
    print("Input: %s" % args.input)
    print("Pairs: %s" % args.pairs)
    print("Output: %s" % args.output)
    print("K: %d, threshold: %.2f, plan: %s, experiments: %d" % (args.k, args.threshold, args.plan, len(experiments)))
    print("Init modes: %s" % ",".join(init_modes))
    print("Distance modes: %s" % ",".join(distance_modes))
    print("Max iters: %s" % ",".join([str(x) for x in max_iters]))
    print("Seeds: %s" % ",".join([str(x) for x in seeds]))
    print("=" * 80)

    spark = create_spark_session()
    total_start = time.time()
    processed_df = None
    pairs_df = None

    try:
        raw_df = load_and_preprocess(spark, args.input, args.sample_ratio)
        _, processed_df, _, _ = build_feature_dataframe(raw_df, args.max_features, args.min_df)
        pairs_df, _, _ = load_reviewed_pairs(spark, args.pairs)

        print("\n" + "=" * 60)
        print("[Step 4] Running execution-parameter experiments")
        print("=" * 60)

        results = []
        all_pair_details = []
        best_result = None
        best_pair_details = None
        best_predictions = None

        for idx, item in enumerate(experiments, 1):
            stage, init_label, distance_mode, max_iter, seed = item
            spark_init, normalized_label = normalize_init_mode(init_label)
            exp_id = "F%02d" % idx
            result, pair_details, predictions = run_single_experiment(
                processed_df,
                pairs_df,
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
            all_pair_details.extend(pair_details)

            if choose_better(result, best_result, args.max_largest_ratio):
                if best_predictions is not None:
                    best_predictions.unpersist()
                best_result = result
                best_pair_details = pair_details
                best_predictions = predictions
            else:
                predictions.unpersist()

        print_summary(results)
        print("\nBest config by comprehensive rule (largest_cluster_ratio <= %.2f first): id=%s, init=%s, distance=%s, maxIter=%d, seed=%d, F1=%.4f, CandidateRecall=%.4f, P=%.4f, maxRatio=%.4f, time=%.2fs" % (
            args.max_largest_ratio,
            best_result["exp_id"],
            best_result["init_mode"],
            best_result["distance_mode"],
            best_result["max_iter"],
            best_result["seed"],
            best_result["f1"],
            best_result["candidate_recall"],
            best_result["precision"],
            best_result["largest_cluster_ratio"],
            best_result["training_time_sec"],
        ))

        save_outputs(spark, args.output, results, all_pair_details, best_result, best_pair_details, best_predictions)

        total_elapsed = time.time() - total_start
        print("\n" + "=" * 80)
        print("F-task optimization + dedup evaluation complete")
        print("Total time: %.2fs" % total_elapsed)
        print("Results saved to: %s" % args.output)
        print("=" * 80)

    finally:
        if processed_df is not None:
            processed_df.unpersist()
        if pairs_df is not None:
            pairs_df.unpersist()
        if best_predictions is not None:
            best_predictions.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
