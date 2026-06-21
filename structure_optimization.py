# -*- coding: utf-8 -*-
"""
StackOverflow Question Clustering - Structure Optimization
聚类结构优化

实验内容:
  1. 层次聚类对比: BisectingKMeans vs K-Means
  2. 聚类后处理:
     - 合并小聚类 (将过小聚类重新分配到最近大聚类)
     - 生成聚类标签 (基于聚类中心关键词)

兼容 Python 2.7 / Spark 2.0
"""

from __future__ import print_function
import re
import argparse
import time
import sys
import math

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, desc, count, collect_list,
    row_number, when, lit
)
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.clustering import KMeans, BisectingKMeans
try:
    from pyspark.ml.evaluation import ClusteringEvaluator
    HAS_EVALUATOR = True
except ImportError:
    HAS_EVALUATOR = False


# =====================================================================
# 文本预处理
# =====================================================================

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


# =====================================================================
# Spark 会话
# =====================================================================

def create_spark_session(app_name="StackOverflow_Structure_Optimization"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# =====================================================================
# Step 1: 数据加载与预处理
# =====================================================================

def load_and_preprocess(spark, input_path, sample_ratio=1.0):
    print("\n" + "=" * 60)
    print("[Step 1] Loading and Preprocessing Data")
    print("=" * 60)

    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total records: %d" % total_count)

    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        sampled_count = df.count()
        print("Sampled records: %d (ratio: %.2f)" % (sampled_count, sample_ratio))

    preprocess_udf = udf(preprocess_text, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))

    valid_count = df.count()
    print("Valid records after filtering: %d" % valid_count)
    return df


# =====================================================================
# Step 2: TF-IDF 特征提取
# =====================================================================

def extract_features(df, max_features=5000, min_df=5):
    print("\n" + "=" * 60)
    print("[Step 2] Feature Extraction (TF-IDF)")
    print("=" * 60)
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

    print("Fitting TF-IDF pipeline...")
    t0 = time.time()
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)
    elapsed = time.time() - t0

    vocabulary = pipeline_model.stages[2].vocabulary
    print("Vocabulary size: %d, Pipeline time: %.2fs" % (len(vocabulary), elapsed))
    return pipeline_model, processed_df, vocabulary


# =====================================================================
# Step 3a: K-Means 聚类
# =====================================================================

def run_kmeans(processed_df, k=50, max_iter=30):
    print("\n" + "=" * 60)
    print("[Experiment 1] Standard K-Means Clustering")
    print("=" * 60)
    print("K=%d, max_iter=%d" % (k, max_iter))

    # 使用与基准方案相同的欧氏距离，保证对照组条件一致
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )

    t0 = time.time()
    model = kmeans.fit(processed_df)
    predictions = model.transform(processed_df)
    elapsed = time.time() - t0

    wssse = model.computeCost(processed_df)
    sil = 0.0
    if HAS_EVALUATOR:
        evaluator = ClusteringEvaluator(
            featuresCol="features", predictionCol="cluster",
            metricName="silhouette", distanceMeasure="squaredEuclidean"
        )
        sil = evaluator.evaluate(predictions)

    print("K-Means training time: %.2fs" % elapsed)
    print("K-Means WSSSE: %.6f" % wssse)
    print("K-Means Silhouette: %.4f" % sil)
    return model, predictions, elapsed, wssse, sil


# =====================================================================
# Step 3b: BisectingKMeans 聚类
# =====================================================================

def run_bisecting_kmeans(processed_df, k=50, max_iter=20, min_divisible=1.0):
    print("\n" + "=" * 60)
    print("[Experiment 2] BisectingKMeans (Hierarchical) Clustering")
    print("=" * 60)
    print("K=%d, max_iter=%d" % (k, max_iter))

    # 使用与基准方案相同的欧氏距离，保证对照组条件一致
    bkm = BisectingKMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        minDivisibleClusterSize=min_divisible,
        seed=42,
    )

    t0 = time.time()
    model = bkm.fit(processed_df)
    predictions = model.transform(processed_df)
    elapsed = time.time() - t0

    wssse = model.computeCost(processed_df)
    sil = 0.0
    if HAS_EVALUATOR:
        evaluator = ClusteringEvaluator(
            featuresCol="features", predictionCol="cluster",
            metricName="silhouette", distanceMeasure="squaredEuclidean"
        )
        sil = evaluator.evaluate(predictions)

    print("BisectingKMeans training time: %.2fs" % elapsed)
    print("BisectingKMeans WSSSE: %.6f" % wssse)
    print("BisectingKMeans Silhouette: %.4f" % sil)
    return model, predictions, elapsed, wssse, sil


# =====================================================================
# Step 4: 聚类分布分析
# =====================================================================

def analyze_distribution(predictions, algo_name, show_detail=True):
    """
    计算聚类分布统计量：
    - 最大/最小/平均聚类大小
    - 变异系数 CV（标准差/均值，越小分布越均匀）
    - 小聚类数量及占比
    """
    stats_rows = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size")
    ).orderBy(desc("cluster_size")).collect()

    sizes = [r["cluster_size"] for r in stats_rows]
    n = len(sizes)
    total = sum(sizes)
    avg = float(total) / n if n > 0 else 0
    max_s = max(sizes) if sizes else 0
    min_s = min(sizes) if sizes else 0
    std = math.sqrt(sum((s - avg) ** 2 for s in sizes) / n) if n > 0 else 0
    cv = std / avg if avg > 0 else 0

    small_thresh = avg * 0.1
    small_cnt = sum(1 for s in sizes if s < small_thresh)
    small_ratio = float(small_cnt) / n if n > 0 else 0

    if show_detail:
        print("\n--- %s Distribution ---" % algo_name)
        print("  Clusters     : %d" % n)
        print("  Total points : %d" % total)
        print("  Max size     : %d" % max_s)
        print("  Min size     : %d" % min_s)
        print("  Avg size     : %.1f" % avg)
        print("  Std dev      : %.1f" % std)
        print("  CV (lower=more uniform): %.4f" % cv)
        print("  Small clusters (<%.0f): %d (%.1f%%)" % (
            small_thresh, small_cnt, small_ratio * 100))

    return {
        "algo": algo_name,
        "cluster_count": n,
        "total": total,
        "max_size": max_s,
        "min_size": min_s,
        "avg_size": avg,
        "std_dev": std,
        "cv": cv,
        "small_thresh": small_thresh,
        "small_cluster_count": small_cnt,
        "small_cluster_ratio": small_ratio,
        "sizes": sizes,
    }


# =====================================================================
# Step 5: 对比结果汇总打印
# =====================================================================

def print_comparison(km_stat, bkm_stat, km_wssse, bkm_wssse, km_time, bkm_time,
                     km_sil=0.0, bkm_sil=0.0):
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS: K-Means  vs  BisectingKMeans")
    print("=" * 70)
    fmt = "%-35s %15s %15s"
    sep = "-" * 70
    print(fmt % ("Metric", "K-Means", "BisectingKMeans"))
    print(sep)
    print(fmt % ("Training Time (s)",
                 "%.2f" % km_time, "%.2f" % bkm_time))
    speedup = km_time / bkm_time if bkm_time > 0 else 0
    print(fmt % ("  Speedup (BKM vs KM)",
                 "-", "%.2fx faster" % speedup if bkm_time < km_time else "%.2fx slower" % (bkm_time / km_time)))
    print(fmt % ("WSSSE",
                 "%.4f" % km_wssse, "%.4f" % bkm_wssse))
    if HAS_EVALUATOR:
        print(fmt % ("Silhouette Score (cosine)",
                     "%.4f" % km_sil, "%.4f" % bkm_sil))
    print(fmt % ("Actual Clusters",
                 str(km_stat["cluster_count"]), str(bkm_stat["cluster_count"])))
    print(fmt % ("Max Cluster Size",
                 str(km_stat["max_size"]), str(bkm_stat["max_size"])))
    print(fmt % ("Min Cluster Size",
                 str(km_stat["min_size"]), str(bkm_stat["min_size"])))
    print(fmt % ("Avg Cluster Size",
                 "%.1f" % km_stat["avg_size"], "%.1f" % bkm_stat["avg_size"]))
    print(fmt % ("Std Deviation",
                 "%.1f" % km_stat["std_dev"], "%.1f" % bkm_stat["std_dev"]))
    print(fmt % ("CV (uniformity, lower=better)",
                 "%.4f" % km_stat["cv"], "%.4f" % bkm_stat["cv"]))
    print(fmt % ("Small Clusters Count",
                 str(km_stat["small_cluster_count"]), str(bkm_stat["small_cluster_count"])))
    print(fmt % ("Small Cluster Ratio (%)",
                 "%.1f%%" % (km_stat["small_cluster_ratio"] * 100),
                 "%.1f%%" % (bkm_stat["small_cluster_ratio"] * 100)))
    print("=" * 70)

    # 分析小结
    print("\n[Analysis]")
    if bkm_time < km_time:
        print("  - BisectingKMeans is %.1fx faster than K-Means" % (km_time / bkm_time))
    else:
        print("  - K-Means is %.1fx faster than BisectingKMeans" % (bkm_time / km_time))

    if bkm_wssse < km_wssse:
        print("  - BisectingKMeans achieves lower WSSSE (better intra-cluster compactness)")
    else:
        print("  - K-Means achieves lower WSSSE (better intra-cluster compactness)")

    if bkm_stat["cv"] < km_stat["cv"]:
        print("  - BisectingKMeans produces more uniform cluster size distribution (CV: %.4f vs %.4f)"
              % (bkm_stat["cv"], km_stat["cv"]))
    else:
        print("  - K-Means produces more uniform cluster size distribution (CV: %.4f vs %.4f)"
              % (km_stat["cv"], bkm_stat["cv"]))

    if bkm_stat["small_cluster_count"] < km_stat["small_cluster_count"]:
        print("  - BisectingKMeans generates fewer small clusters (%d vs %d)"
              % (bkm_stat["small_cluster_count"], km_stat["small_cluster_count"]))
    else:
        print("  - K-Means generates fewer small clusters (%d vs %d)"
              % (km_stat["small_cluster_count"], bkm_stat["small_cluster_count"]))


# =====================================================================
# Step 6: 后处理——合并小聚类
# =====================================================================

def merge_small_clusters(spark, predictions, model, algo_name, threshold_ratio=0.1):
    """
    将 size < avg_size * threshold_ratio 的小聚类
    重新分配到距离最近的大聚类，使用 join 方式实现（避免UDF序列化问题）。
    """
    print("\n" + "=" * 60)
    print("[Post-Processing] Merge Small Clusters (%s)" % algo_name)
    print("=" * 60)

    size_rows = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size")
    ).collect()

    sizes_dict = {r["cluster"]: r["cluster_size"] for r in size_rows}
    n = len(sizes_dict)
    avg_size = float(sum(sizes_dict.values())) / n if n > 0 else 0
    threshold = avg_size * threshold_ratio

    small_ids = set(k for k, v in sizes_dict.items() if v < threshold)
    large_ids = set(k for k, v in sizes_dict.items() if v >= threshold)

    print("Avg cluster size: %.1f, threshold: %.1f (ratio=%.2f)"
          % (avg_size, threshold, threshold_ratio))
    print("Small clusters to merge: %d / %d" % (len(small_ids), n))
    print("Large clusters to keep : %d" % len(large_ids))

    if not small_ids:
        print("No small clusters found, skipping merge.")
        return predictions, n

    # 获取所有聚类中心
    centers = model.clusterCenters()

    # 为每个小聚类找最近的大聚类（欧氏距离）
    large_id_list = list(large_ids)

    def euclidean(v1, v2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    small_to_large = {}
    for sid in small_ids:
        sc = centers[sid]
        best_lid = large_id_list[0]
        best_dist = euclidean(sc, centers[best_lid])
        for lid in large_id_list[1:]:
            d = euclidean(sc, centers[lid])
            if d < best_dist:
                best_dist = d
                best_lid = lid
        small_to_large[sid] = best_lid

    # 打印前10条映射
    print("\nSmall -> Large mapping (first 10):")
    for i, (sid, lid) in enumerate(list(small_to_large.items())[:10]):
        print("  Cluster %3d (size=%4d) -> Cluster %3d (size=%4d)"
              % (sid, sizes_dict[sid], lid, sizes_dict[lid]))

    # 使用 join 操作完成重新分配（避免UDF序列化问题）
    mapping_rows = [(int(sid), int(lid)) for sid, lid in small_to_large.items()]
    mapping_df = spark.createDataFrame(mapping_rows, ["cluster", "new_cluster"])

    merged = predictions.join(mapping_df, on="cluster", how="left")
    merged = merged.withColumn(
        "cluster",
        when(col("new_cluster").isNotNull(), col("new_cluster")).otherwise(col("cluster"))
    ).drop("new_cluster")

    final_sizes = merged.groupBy("cluster").agg(count("*").alias("sz")).collect()
    final_k = len(final_sizes)
    print("\nAfter merging: %d effective clusters (was %d)" % (final_k, n))
    return merged, final_k


# =====================================================================
# Step 7: 后处理——生成聚类标签
# =====================================================================

def generate_cluster_labels(model, vocabulary, top_k=5):
    """
    基于聚类中心向量的最大权重词汇生成标签字符串。
    返回 dict: cluster_id -> label_string
    """
    print("\n" + "=" * 60)
    print("[Post-Processing] Generate Cluster Labels (top-%d keywords)" % top_k)
    print("=" * 60)

    centers = model.clusterCenters()
    vocab = vocabulary
    labels = {}

    for cid, center in enumerate(centers):
        center_list = list(center)
        # 取权重最大的 top_k 个词汇索引
        indexed = sorted(enumerate(center_list), key=lambda x: x[1], reverse=True)
        top_words = [vocab[i] for i, _ in indexed[:top_k] if i < len(vocab)]
        labels[cid] = " | ".join(top_words)

    print("Sample labels (first 10 clusters):")
    for cid in sorted(labels.keys())[:10]:
        print("  Cluster %3d: [%s]" % (cid, labels[cid]))

    return labels


# =====================================================================
# Step 8: 展示聚类示例问题
# =====================================================================

def show_sample_clusters(predictions, cluster_labels, n_clusters=5, n_questions=5, algo_name=""):
    print("\n" + "=" * 60)
    print("[Sample Clusters] %s (top %d clusters)" % (algo_name, n_clusters))
    print("=" * 60)

    top_rows = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
        collect_list("title").alias("titles"),
    ).orderBy(desc("cluster_size")).limit(n_clusters).collect()

    for row in top_rows:
        cid = row["cluster"]
        size = row["cluster_size"]
        label = cluster_labels.get(cid, "N/A")
        titles = row["titles"][:n_questions]
        print("\nCluster %d [%s] (total %d questions):" % (cid, label, size))
        for i, t in enumerate(titles, 1):
            print("  %d. %s" % (i, t))


# =====================================================================
# 实验报告打印（所有行以 [REPORT] 开头，便于从日志中提取）
# =====================================================================

def print_experiment_report(args, km_stat, bkm_stat,
                            km_wssse, bkm_wssse, km_time, bkm_time,
                            km_final_k, bkm_final_k, total_elapsed,
                            km_preds, bkm_preds, km_labels, bkm_labels):
    R = "[REPORT]"
    sep = R + " " + "=" * 62

    print("\n" + sep)
    print(R + " EXPERIMENT REPORT - Clustering Structure Optimization")
    print(R + " Member E: BisectingKMeans vs K-Means")
    print(sep)
    print(R + " Dataset    : StackOverflow Oracle Database Questions")
    print(R + " Sample     : %.0f%%" % (args.sample_ratio * 100))
    print(R + " K          : %d" % args.k)
    print(R + " Vocab size : %d" % args.max_features)
    print(sep)

    print(R + "")
    print(R + " [1] COMPARISON TABLE")
    fmt = R + "  %-32s %14s %14s"
    print(fmt % ("Metric", "K-Means", "BisectingKMeans"))
    print(R + "  " + "-" * 60)
    print(fmt % ("Training Time (s)",
                 "%.2f" % km_time, "%.2f" % bkm_time))
    print(fmt % ("WSSSE (lower=more compact)",
                 "%.2f" % km_wssse, "%.2f" % bkm_wssse))
    print(fmt % ("Cluster Count",
                 str(km_stat["cluster_count"]), str(bkm_stat["cluster_count"])))
    print(fmt % ("Max Cluster Size",
                 str(km_stat["max_size"]), str(bkm_stat["max_size"])))
    print(fmt % ("Min Cluster Size",
                 str(km_stat["min_size"]), str(bkm_stat["min_size"])))
    print(fmt % ("Avg Cluster Size",
                 "%.1f" % km_stat["avg_size"], "%.1f" % bkm_stat["avg_size"])),
    print(fmt % ("Std Deviation",
                 "%.1f" % km_stat["std_dev"], "%.1f" % bkm_stat["std_dev"]))
    print(fmt % ("CV (lower=more uniform)",
                 "%.4f" % km_stat["cv"], "%.4f" % bkm_stat["cv"]))
    print(fmt % ("Small Clusters (< avg*10%)",
                 "%d (%.0f%%)" % (km_stat["small_cluster_count"],
                                  km_stat["small_cluster_ratio"] * 100),
                 "%d (%.0f%%)" % (bkm_stat["small_cluster_count"],
                                  bkm_stat["small_cluster_ratio"] * 100)))
    print(fmt % ("Effective Clusters (after merge)",
                 str(km_final_k), str(bkm_final_k)))

    print(R + "")
    print(R + " [2] ANALYSIS")
    if bkm_time > km_time:
        print(R + "  - K-Means is %.1fx faster (%.2fs vs %.2fs)"
              % (bkm_time / km_time, km_time, bkm_time))
    else:
        print(R + "  - BisectingKMeans is %.1fx faster (%.2fs vs %.2fs)"
              % (km_time / bkm_time, bkm_time, km_time))
    if km_wssse <= bkm_wssse:
        print(R + "  - K-Means has better compactness (WSSSE %.2f vs %.2f, diff %.2f%%)"
              % (km_wssse, bkm_wssse, (bkm_wssse - km_wssse) / km_wssse * 100))
    else:
        print(R + "  - BisectingKMeans has better compactness (WSSSE %.2f vs %.2f)"
              % (bkm_wssse, km_wssse))
    cv_improve = (km_stat["cv"] - bkm_stat["cv"]) / km_stat["cv"] * 100
    print(R + "  - BisectingKMeans CV is %.1f%% lower (%.4f vs %.4f), more uniform"
          % (cv_improve, bkm_stat["cv"], km_stat["cv"]))
    print(R + "  - BisectingKMeans small clusters: %d vs K-Means: %d"
          % (bkm_stat["small_cluster_count"], km_stat["small_cluster_count"]))
    print(R + "  - Max cluster reduced by %.0f%% (%d -> %d)"
          % ((km_stat["max_size"] - bkm_stat["max_size"]) / float(km_stat["max_size"]) * 100,
             km_stat["max_size"], bkm_stat["max_size"]))

    print(R + "")
    print(R + " [3] TOP-5 CLUSTER LABELS")
    print(R + "  K-Means top clusters:")
    km_top = km_preds.groupBy("cluster").agg(count("*").alias("sz")) \
                     .orderBy(desc("sz")).limit(5).collect()
    for row in km_top:
        cid = row["cluster"]
        print(R + "    Cluster %3d (size=%4d): [%s]"
              % (cid, row["sz"], km_labels.get(cid, "")))

    print(R + "  BisectingKMeans top clusters:")
    bkm_top = bkm_preds.groupBy("cluster").agg(count("*").alias("sz")) \
                       .orderBy(desc("sz")).limit(5).collect()
    for row in bkm_top:
        cid = row["cluster"]
        print(R + "    Cluster %3d (size=%4d): [%s]"
              % (cid, row["sz"], bkm_labels.get(cid, "")))

    print(R + "")
    print(R + " [4] CONCLUSION")
    # 综合评分：WSSSE、CV、小聚类数量决定胜者
    km_score  = (1 if km_wssse  <= bkm_wssse  else 0) + \
                (1 if km_stat["cv"] <= bkm_stat["cv"] else 0) + \
                (1 if km_stat["small_cluster_count"] <= bkm_stat["small_cluster_count"] else 0)
    bkm_score = 3 - km_score
    winner    = "K-Means" if km_score >= 2 else "BisectingKMeans"
    loser     = "BisectingKMeans" if winner == "K-Means" else "K-Means"
    print(R + "  Overall winner: %s (%d/3 metrics better)" % (winner, km_score if winner == "K-Means" else bkm_score))
    print(R + "")
    if winner == "K-Means":
        print(R + "  K-Means advantages:")
        if km_wssse < bkm_wssse:
            print(R + "  + Lower WSSSE (better compactness): %.2f vs %.2f (%.1f%% better)"
                  % (km_wssse, bkm_wssse, (bkm_wssse - km_wssse) / bkm_wssse * 100))
        if km_stat["cv"] < bkm_stat["cv"]:
            print(R + "  + More uniform distribution (CV): %.4f vs %.4f"
                  % (km_stat["cv"], bkm_stat["cv"]))
        if km_stat["small_cluster_count"] <= bkm_stat["small_cluster_count"]:
            print(R + "  + Fewer small clusters: %d vs %d"
                  % (km_stat["small_cluster_count"], bkm_stat["small_cluster_count"]))
        print(R + "  + %.1fx faster training time" % (bkm_time / km_time))
        print(R + "")
        print(R + "  Note: Both algorithms use the same distance metric and parameters.")
        print(R + "  The difference is purely due to algorithm structure (flat vs hierarchical).")
    else:
        print(R + "  BisectingKMeans advantages:")
        if bkm_wssse < km_wssse:
            print(R + "  + Lower WSSSE: %.2f vs %.2f" % (bkm_wssse, km_wssse))
        if bkm_stat["cv"] < km_stat["cv"]:
            print(R + "  + More uniform distribution (CV): %.4f vs %.4f"
                  % (bkm_stat["cv"], km_stat["cv"]))
        if bkm_stat["small_cluster_count"] < km_stat["small_cluster_count"]:
            print(R + "  + Fewer small clusters: %d vs %d"
                  % (bkm_stat["small_cluster_count"], km_stat["small_cluster_count"]))
        print(R + "  - Trade-off: %.1fx slower training time" % (bkm_time / km_time))
    print(R + "")
    print(R + " Total experiment time: %.1fs (%.1f min)"
          % (total_elapsed, total_elapsed / 60))
    print(sep)


# =====================================================================
# Step 9: 保存结果
# =====================================================================

def save_comparison_table(spark, km_stat, bkm_stat,
                          km_wssse, bkm_wssse, km_time, bkm_time,
                          output_path):
    """保存两种算法的对比指标表格（CSV 格式，便于查看）"""
    rows = [
        ("training_time_s",   "%.2f" % km_time,                   "%.2f" % bkm_time),
        ("wssse",             "%.6f" % km_wssse,                   "%.6f" % bkm_wssse),
        ("cluster_count",     str(km_stat["cluster_count"]),        str(bkm_stat["cluster_count"])),
        ("max_cluster_size",  str(km_stat["max_size"]),             str(bkm_stat["max_size"])),
        ("min_cluster_size",  str(km_stat["min_size"]),             str(bkm_stat["min_size"])),
        ("avg_cluster_size",  "%.1f" % km_stat["avg_size"],         "%.1f" % bkm_stat["avg_size"]),
        ("std_deviation",     "%.1f" % km_stat["std_dev"],          "%.1f" % bkm_stat["std_dev"]),
        ("cv_uniformity",     "%.4f" % km_stat["cv"],               "%.4f" % bkm_stat["cv"]),
        ("small_cluster_cnt", str(km_stat["small_cluster_count"]),  str(bkm_stat["small_cluster_count"])),
        ("small_cluster_pct", "%.1f%%" % (km_stat["small_cluster_ratio"] * 100),
                               "%.1f%%" % (bkm_stat["small_cluster_ratio"] * 100)),
    ]
    cmp_df = spark.createDataFrame(rows, ["metric", "kmeans", "bisecting_kmeans"])
    cmp_df.coalesce(1).write.mode("overwrite").csv(output_path + "/comparison_table", header=True)
    print("Comparison table saved: %s/comparison_table" % output_path)


def save_cluster_assignments(spark, predictions, cluster_labels, output_path, prefix):
    """保存聚类分配结果（含标签列）"""
    label_rows = [(int(cid), lbl) for cid, lbl in cluster_labels.items()]
    label_df = spark.createDataFrame(label_rows, ["cluster", "cluster_label"])

    output_df = predictions.select(
        col("question_id"),
        col("title"),
        col("cluster"),
    ).join(label_df, on="cluster", how="left")

    output_df.write.mode("overwrite").json(output_path + "/%s_assignments" % prefix)
    print("Assignments saved: %s/%s_assignments" % (output_path, prefix))

    # 聚类摘要（每个聚类大小+标签）
    summary = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size")
    ).join(label_df, on="cluster", how="left").orderBy("cluster")
    summary.write.mode("overwrite").json(output_path + "/%s_summary" % prefix)
    print("Summary saved: %s/%s_summary" % (output_path, prefix))


# =====================================================================
# 主函数
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Structure Optimization: BisectingKMeans vs KMeans + Post-processing"
    )
    parser.add_argument("--input",           type=str,   required=True,  help="Input data path (JSON Lines)")
    parser.add_argument("--output",          type=str,   required=True,  help="Output result path")
    parser.add_argument("--k",               type=int,   default=50,     help="Number of clusters (default: 50)")
    parser.add_argument("--max-features",    type=int,   default=5000,   help="Max TF-IDF vocabulary size (default: 5000)")
    parser.add_argument("--min-df",          type=int,   default=5,      help="Min document frequency (default: 5)")
    parser.add_argument("--sample-ratio",    type=float, default=1.0,    help="Data sampling ratio (default: 1.0)")
    parser.add_argument("--km-max-iter",     type=int,   default=30,     help="KMeans max iterations (default: 30)")
    parser.add_argument("--bkm-max-iter",    type=int,   default=20,     help="BisectingKMeans max iterations (default: 20)")
    parser.add_argument("--merge-threshold", type=float, default=0.1,
                        help="Small cluster threshold ratio (default: 0.1 means < 10%% of avg size)")
    parser.add_argument("--label-top-k",     type=int,   default=5,      help="Top-K keywords per cluster label (default: 5)")

    args = parser.parse_args()
    total_start = time.time()

    print("\n" + "=" * 70)
    print("StackOverflow Clustering - Structure Optimization (Member E)")
    print("=" * 70)
    print("Input        : %s" % args.input)
    print("Output       : %s" % args.output)
    print("K            : %d" % args.k)
    print("Max features : %d" % args.max_features)
    print("Sample ratio : %.2f" % args.sample_ratio)
    print("=" * 70)

    spark = create_spark_session()

    try:
        # ---- Step 1: 加载预处理 ----
        df = load_and_preprocess(spark, args.input, args.sample_ratio)

        # ---- Step 2: 特征提取 ----
        pipeline_model, processed_df, vocabulary = extract_features(
            df, args.max_features, args.min_df
        )

        # 缓存特征数据，两个算法共用
        print("\nCaching feature data...")
        processed_df.cache()
        processed_df.count()
        print("Cache ready.")

        # ---- Step 3a: K-Means (欧氏距离，与基准方案相同) ----
        km_model, km_preds, km_time, km_wssse, km_sil = run_kmeans(
            processed_df, args.k, args.km_max_iter
        )

        # ---- Step 3b: BisectingKMeans (欧氏距离，仅改变算法结构) ----
        bkm_model, bkm_preds, bkm_time, bkm_wssse, bkm_sil = run_bisecting_kmeans(
            processed_df, args.k, args.bkm_max_iter
        )

        # ---- Step 4: 分布分析 ----
        print("\n" + "=" * 60)
        print("[Step 4] Cluster Distribution Analysis")
        print("=" * 60)
        km_stat  = analyze_distribution(km_preds,  "KMeans")
        bkm_stat = analyze_distribution(bkm_preds, "BisectingKMeans")

        # ---- Step 5: 对比汇总 ----
        print_comparison(km_stat, bkm_stat, km_wssse, bkm_wssse,
                         km_time, bkm_time, km_sil, bkm_sil)

        # ---- Step 6: 生成聚类标签 ----
        km_labels  = generate_cluster_labels(km_model,  vocabulary, args.label_top_k)
        bkm_labels = generate_cluster_labels(bkm_model, vocabulary, args.label_top_k)

        # ---- Step 7: 合并小聚类 ----
        km_merged,  km_final_k  = merge_small_clusters(
            spark, km_preds,  km_model,  "KMeans",          args.merge_threshold)
        bkm_merged, bkm_final_k = merge_small_clusters(
            spark, bkm_preds, bkm_model, "BisectingKMeans", args.merge_threshold)

        # ---- Step 8: 展示样例 ----
        show_sample_clusters(km_preds,  km_labels,  algo_name="KMeans")
        show_sample_clusters(bkm_preds, bkm_labels, algo_name="BisectingKMeans")

        # ---- Step 9: 保存结果 ----
        print("\n" + "=" * 60)
        print("[Step 9] Saving Results")
        print("=" * 60)
        save_comparison_table(
            spark, km_stat, bkm_stat,
            km_wssse, bkm_wssse, km_time, bkm_time,
            args.output
        )
        save_cluster_assignments(spark, km_merged,  km_labels,  args.output, "kmeans")
        save_cluster_assignments(spark, bkm_merged, bkm_labels, args.output, "bisecting_kmeans")

        # ---- 总结 ----
        total_elapsed = time.time() - total_start
        print("\n" + "=" * 70)
        print("Structure Optimization Complete!")
        print("=" * 70)
        print("K-Means        : WSSSE=%.4f, Sil=%.4f, time=%.2fs, final_k=%d"
              % (km_wssse,  km_sil,  km_time,  km_final_k))
        print("BisectingKMeans: WSSSE=%.4f, Sil=%.4f, time=%.2fs, final_k=%d"
              % (bkm_wssse, bkm_sil, bkm_time, bkm_final_k))
        print("Total time: %.2fs" % total_elapsed)
        print("Results saved to: %s" % args.output)
        print("=" * 70)

        # ---- 实验报告（纯文本，便于提取）----
        print_experiment_report(
            args, km_stat, bkm_stat,
            km_wssse, bkm_wssse, km_time, bkm_time,
            km_final_k, bkm_final_k, total_elapsed,
            km_preds, bkm_preds, km_labels, bkm_labels
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
