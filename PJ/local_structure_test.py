# -*- coding: utf-8 -*-
"""
本地测试版本 - 聚类结构优化
在本机用少量数据快速验证 BisectingKMeans vs KMeans 逻辑
Python 3 + 本地 Spark (local[*])

运行方式:
    python local_structure_test.py
"""

import re
import time
import math
from html.parser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, desc, count, collect_list, when
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# =====================================================================
# 配置（本地测试用小参数，快速出结果）
# =====================================================================
DATA_PATH     = "./StackOverFlow_Oracle_Database/oracle_database_questions.json"
SAMPLE_RATIO  = 0.05    # 只取 5% 数据，约 7600 条，本地跑几分钟
K             = 30      # 聚类数，本地用小一点
MAX_FEATURES  = 3000
MIN_DF        = 3
KM_MAX_ITER   = 20
BKM_MAX_ITER  = 15
MERGE_THRESH  = 0.1     # 小聚类阈值：avg_size * 10%
LABEL_TOP_K   = 5       # 每个聚类显示 top-5 关键词


# =====================================================================
# 文本预处理
# =====================================================================

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
    return re.sub(r"\s+", " ", extractor.get_text()).strip()


def preprocess_text(title, body):
    combined = (clean_html(title) if title else "") + " " + (clean_html(body) if body else "")
    combined = combined.lower()
    combined = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)
    return re.sub(r"\s+", " ", combined).strip()


# =====================================================================
# 分布统计
# =====================================================================

def distribution_stats(predictions, algo_name):
    rows  = predictions.groupBy("cluster").agg(count("*").alias("sz")).collect()
    sizes = [r["sz"] for r in rows]
    n     = len(sizes)
    total = sum(sizes)
    avg   = total / n if n > 0 else 0
    std   = math.sqrt(sum((s - avg) ** 2 for s in sizes) / n) if n > 0 else 0
    cv    = std / avg if avg > 0 else 0
    small = sum(1 for s in sizes if s < avg * MERGE_THRESH)

    print(f"\n  [{algo_name}]")
    print(f"    Clusters      : {n}")
    print(f"    Max / Min     : {max(sizes)} / {min(sizes)}")
    print(f"    Avg size      : {avg:.1f}  Std: {std:.1f}")
    print(f"    CV (uniformity, lower=better): {cv:.4f}")
    print(f"    Small clusters: {small} ({small/n*100:.1f}%)")
    return {"algo": algo_name, "n": n, "avg": avg, "cv": cv, "small": small,
            "max": max(sizes), "min": min(sizes), "sizes": sizes}


# =====================================================================
# 合并小聚类（join 方式）
# =====================================================================

def merge_small_clusters(spark, predictions, model, algo_name):
    size_rows  = predictions.groupBy("cluster").agg(count("*").alias("sz")).collect()
    sizes_dict = {r["cluster"]: r["sz"] for r in size_rows}
    avg        = sum(sizes_dict.values()) / len(sizes_dict)
    threshold  = avg * MERGE_THRESH

    small_ids  = {k for k, v in sizes_dict.items() if v < threshold}
    large_ids  = [k for k, v in sizes_dict.items() if v >= threshold]

    print(f"\n  [{algo_name}] Merging {len(small_ids)} small clusters "
          f"(threshold={threshold:.0f}) into {len(large_ids)} large ones")

    if not small_ids:
        print("  No small clusters, skipping.")
        return predictions

    centers = model.clusterCenters()

    def euclid(v1, v2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    mapping = []
    for sid in small_ids:
        best_lid  = min(large_ids, key=lambda lid: euclid(centers[sid], centers[lid]))
        mapping.append((int(sid), int(best_lid)))

    mapping_df = spark.createDataFrame(mapping, ["cluster", "new_cluster"])
    merged = predictions.join(mapping_df, on="cluster", how="left")
    merged = merged.withColumn(
        "cluster",
        when(col("new_cluster").isNotNull(), col("new_cluster")).otherwise(col("cluster"))
    ).drop("new_cluster")

    final_k = merged.select("cluster").distinct().count()
    print(f"  After merge: {final_k} effective clusters")
    return merged


# =====================================================================
# 生成聚类标签
# =====================================================================

def get_labels(model, vocabulary):
    centers = model.clusterCenters()
    labels  = {}
    for cid, center in enumerate(centers):
        top_idx   = sorted(range(len(center)), key=lambda i: center[i], reverse=True)[:LABEL_TOP_K]
        top_words = [vocabulary[i] for i in top_idx if i < len(vocabulary)]
        labels[cid] = " | ".join(top_words)
    return labels


# =====================================================================
# 主流程
# =====================================================================

def main():
    spark = SparkSession.builder \
        .appName("LocalTest_StructureOpt") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 65)
    print("LOCAL TEST: Structure Optimization (Member E)")
    print(f"  Data: {DATA_PATH}")
    print(f"  Sample: {SAMPLE_RATIO*100:.0f}%  K={K}  max_features={MAX_FEATURES}")
    print("=" * 65)

    # ---- 加载数据 ----
    print("\n[Step 1] Loading data...")
    df = spark.read.json(DATA_PATH)
    total = df.count()
    print(f"  Total records: {total}")
    df = df.sample(withReplacement=False, fraction=SAMPLE_RATIO, seed=42)
    sampled = df.count()
    print(f"  Sampled: {sampled}")

    # ---- 预处理 ----
    print("\n[Step 2] Preprocessing...")
    preprocess_udf = udf(preprocess_text, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))
    print(f"  Valid records: {df.count()}")

    # ---- TF-IDF ----
    print("\n[Step 3] TF-IDF feature extraction...")
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="clean_text", outputCol="raw_tokens"),
        StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens"),
        CountVectorizer(inputCol="filtered_tokens", outputCol="raw_features",
                        vocabSize=MAX_FEATURES, minDF=MIN_DF),
        IDF(inputCol="raw_features", outputCol="tfidf_features"),
        Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0),
    ])
    t0 = time.time()
    pm = pipeline.fit(df)
    processed = pm.transform(df)
    vocabulary = pm.stages[2].vocabulary
    print(f"  Vocab size: {len(vocabulary)}, time: {time.time()-t0:.1f}s")

    processed.cache()
    processed.count()

    # ---- Exp 1: K-Means ----
    print(f"\n[Exp 1] K-Means (K={K}, maxIter={KM_MAX_ITER})...")
    t0 = time.time()
    km_model = KMeans(featuresCol="features", predictionCol="cluster",
                      k=K, maxIter=KM_MAX_ITER, seed=42,
                      distanceMeasure="cosine").fit(processed)
    km_preds = km_model.transform(processed)
    km_time  = time.time() - t0
    km_wssse = km_model.computeCost(processed)

    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                    metricName="silhouette", distanceMeasure="cosine")
    km_sil = evaluator.evaluate(km_preds)
    print(f"  Time: {km_time:.1f}s  WSSSE: {km_wssse:.4f}  Silhouette: {km_sil:.4f}")

    # ---- Exp 2: BisectingKMeans ----
    print(f"\n[Exp 2] BisectingKMeans (K={K}, maxIter={BKM_MAX_ITER})...")
    t0 = time.time()
    bkm_model = BisectingKMeans(featuresCol="features", predictionCol="cluster",
                                k=K, maxIter=BKM_MAX_ITER, seed=42).fit(processed)
    bkm_preds = bkm_model.transform(processed)
    bkm_time  = time.time() - t0
    bkm_wssse = bkm_model.computeCost(processed)
    bkm_sil   = evaluator.evaluate(bkm_preds)
    print(f"  Time: {bkm_time:.1f}s  WSSSE: {bkm_wssse:.4f}  Silhouette: {bkm_sil:.4f}")

    # ---- 分布分析 ----
    print("\n[Step 4] Cluster distribution analysis:")
    km_stat  = distribution_stats(km_preds,  "K-Means")
    bkm_stat = distribution_stats(bkm_preds, "BisectingKMeans")

    # ---- 对比汇总 ----
    print("\n" + "=" * 65)
    print("COMPARISON SUMMARY")
    print("=" * 65)
    fmt = "  %-30s %12s %12s"
    print(fmt % ("Metric", "K-Means", "BisectingKMeans"))
    print("  " + "-" * 60)
    print(fmt % ("Training time (s)",      f"{km_time:.1f}",    f"{bkm_time:.1f}"))
    print(fmt % ("WSSSE",                  f"{km_wssse:.4f}",   f"{bkm_wssse:.4f}"))
    print(fmt % ("Silhouette Score",        f"{km_sil:.4f}",     f"{bkm_sil:.4f}"))
    print(fmt % ("Max cluster size",        str(km_stat["max"]), str(bkm_stat["max"])))
    print(fmt % ("Min cluster size",        str(km_stat["min"]), str(bkm_stat["min"])))
    print(fmt % ("CV (uniformity)",         f"{km_stat['cv']:.4f}", f"{bkm_stat['cv']:.4f}"))
    print(fmt % ("Small clusters",          str(km_stat["small"]), str(bkm_stat["small"])))
    print("=" * 65)

    # ---- 聚类标签 ----
    print("\n[Step 5] Generating cluster labels...")
    km_labels  = get_labels(km_model,  vocabulary)
    bkm_labels = get_labels(bkm_model, vocabulary)

    print("\n  K-Means top-5 cluster labels:")
    km_top = km_preds.groupBy("cluster").agg(count("*").alias("sz")) \
                     .orderBy(desc("sz")).limit(5).collect()
    for r in km_top:
        print(f"    Cluster {r['cluster']:3d} (size={r['sz']:4d}): [{km_labels.get(r['cluster'], '')}]")

    print("\n  BisectingKMeans top-5 cluster labels:")
    bkm_top = bkm_preds.groupBy("cluster").agg(count("*").alias("sz")) \
                       .orderBy(desc("sz")).limit(5).collect()
    for r in bkm_top:
        print(f"    Cluster {r['cluster']:3d} (size={r['sz']:4d}): [{bkm_labels.get(r['cluster'], '')}]")

    # ---- 合并小聚类 ----
    print("\n[Step 6] Merging small clusters...")
    km_merged  = merge_small_clusters(spark, km_preds,  km_model,  "K-Means")
    bkm_merged = merge_small_clusters(spark, bkm_preds, bkm_model, "BisectingKMeans")

    print("\n  Distribution after merging:")
    distribution_stats(km_merged,  "K-Means (merged)")
    distribution_stats(bkm_merged, "BisectingKMeans (merged)")

    # ---- 保存本地结果 ----
    print("\n[Step 7] Saving local results...")
    out = "./local_structure_output"
    km_merged.select("question_id", "title", "cluster") \
             .coalesce(1).write.mode("overwrite").csv(out + "/kmeans", header=True)
    bkm_merged.select("question_id", "title", "cluster") \
              .coalesce(1).write.mode("overwrite").csv(out + "/bisecting_kmeans", header=True)
    print(f"  Results saved to: {out}/")

    print("\n" + "=" * 65)
    print("LOCAL TEST COMPLETE!")
    print("=" * 65)

    spark.stop()


if __name__ == "__main__":
    main()
