# -*- coding: utf-8 -*-
"""
E: 聚类结构与后处理实验
  E1: KMeans (K=45, TF-IDF C1 特征)
  E2: BisectingKMeans (K=45, TF-IDF C1 特征)
  E3: KMeans + 小簇合并
  E4: BisectingKMeans + 小簇合并
  E5: 聚类标签生成（对 E4 输出）

输入:  hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300
特征:  TF-IDF C1 (title+body, vocab=5000, min_df=5, max_df=0.9)
K:     45 (B 组最优)
输出:  每个实验单独目录，包含 cluster_assignments / duplicate_candidates
"""
from __future__ import print_function
import re, argparse, time, sys, math

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, desc, count, collect_list,
    row_number, when, lit, monotonically_increasing_id
)
from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover,
    CountVectorizer, IDF, Normalizer,
)
from pyspark.ml.clustering import KMeans, BisectingKMeans
try:
    from pyspark.ml.evaluation import ClusteringEvaluator
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False

# ─────────────────────────────────────────────────────────────────────────────
# HTML 清洗 / 文本预处理
# ─────────────────────────────────────────────────────────────────────────────

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []
    def handle_data(self, d):
        self.result.append(d)
    def get_text(self):
        return " ".join(self.result)


def clean_html(s):
    if not s:
        return ""
    ext = HTMLTextExtractor()
    try:
        ext.feed(s)
    except Exception:
        return s
    return re.sub(r"\s+", " ", ext.get_text()).strip()


def preprocess(title, body):
    t = clean_html(title) if title else ""
    b = clean_html(body)  if body  else ""
    combined = (t + " " + b).lower()
    combined = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)
    return re.sub(r"\s+", " ", combined).strip()


preprocess_udf = udf(preprocess, StringType())


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_data(spark, input_path):
    print("\n[Load] Reading data from:", input_path)
    df = spark.read.json(input_path)
    total = df.count()
    print("[Load] Total records:", total)

    df = df.withColumn("text", preprocess_udf(col("title"), col("body")))
    df = df.select("question_id", "title", "score", "text").dropna(subset=["text"])
    df = df.filter(col("text") != "")
    print("[Load] After preprocessing:", df.count())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF 特征（C1 配置）
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(df, max_features=5000, min_df=5, max_df=0.9):
    print("\n[Feature] Building TF-IDF (vocab=%d, min_df=%d, max_df=%.1f)" % (
        max_features, min_df, max_df))
    t0 = time.time()
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        CountVectorizer(inputCol="filtered", outputCol="tf",
                        vocabSize=max_features, minDF=float(min_df),
                        maxDF=max_df),
        IDF(inputCol="tf", outputCol="tfidf"),
        Normalizer(inputCol="tfidf", outputCol="features", p=2.0),
    ])
    model = pipeline.fit(df)
    processed = model.transform(df)
    vocab = model.stages[2].vocabulary
    print("[Feature] Vocab size: %d, time: %.2fs" % (len(vocab), time.time() - t0))
    return model, processed, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 聚类分布统计
# ─────────────────────────────────────────────────────────────────────────────

def distribution_stats(preds, label=""):
    rows = preds.groupBy("cluster").agg(count("*").alias("sz")) \
                .orderBy(desc("sz")).collect()
    sizes = [r["sz"] for r in rows]
    n      = len(sizes)
    total  = sum(sizes)
    mx     = max(sizes)
    mn     = min(sizes)
    avg    = float(total) / n
    std    = math.sqrt(sum((s - avg)**2 for s in sizes) / n)
    cv     = std / avg if avg > 0 else 0.0
    thresh = avg * 0.1
    small  = sum(1 for s in sizes if s < thresh)
    print("\n[Stats] %s" % label)
    print("  Clusters: %d, Max: %d (%.2f%%), Min: %d, Avg: %.1f, CV: %.4f, Small: %d" % (
        n, mx, 100.0 * mx / total, mn, avg, cv, small))
    return {
        "n": n, "total": total, "max": mx, "min": mn,
        "avg": avg, "cv": cv, "small": small,
        "max_pct": 100.0 * mx / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 合并小簇
# ─────────────────────────────────────────────────────────────────────────────

def merge_small_clusters(spark, preds, model, threshold_ratio=0.1, label=""):
    """将小于 avg*threshold_ratio 的簇并入最近大簇"""
    rows  = preds.groupBy("cluster").agg(count("*").alias("sz")).collect()
    sizes = {r["cluster"]: r["sz"] for r in rows}
    total = sum(sizes.values())
    avg   = float(total) / len(sizes) if sizes else 1.0
    thresh = avg * threshold_ratio

    small_ids = {cid for cid, sz in sizes.items() if sz < thresh}
    large_ids = [cid for cid, sz in sizes.items() if sz >= thresh]

    if not small_ids:
        print("[Merge] %s: no small clusters to merge (thresh=%.1f)" % (label, thresh))
        return preds, len(sizes)

    print("[Merge] %s: merging %d small clusters (thresh=%.1f) into %d large clusters" % (
        label, len(small_ids), thresh, len(large_ids)))

    # 获取簇中心
    try:
        centers = model.clusterCenters()
    except AttributeError:
        centers = model.clusterCenters

    def nearest_large(cid):
        if cid not in small_ids:
            return int(cid)
        sc = centers[cid]
        best, best_d = None, float("inf")
        for lid in large_ids:
            lc = centers[lid]
            d  = sum((a - b)**2 for a, b in zip(sc, lc))
            if d < best_d:
                best_d = d
                best   = lid
        return int(best) if best is not None else int(cid)

    remap = {cid: nearest_large(cid) for cid in sizes}
    remap_udf = udf(lambda c: remap.get(c, c), IntegerType())
    merged = preds.withColumn("cluster", remap_udf(col("cluster")))

    final_k = len(set(remap.values()))
    print("[Merge] %s: effective clusters after merge: %d" % (label, final_k))
    return merged, final_k


# ─────────────────────────────────────────────────────────────────────────────
# 生成聚类标签（E5）
# ─────────────────────────────────────────────────────────────────────────────

def cluster_labels(model, vocab, top_k=5):
    try:
        centers = model.clusterCenters()
    except AttributeError:
        centers = model.clusterCenters
    labels = {}
    for cid, center in enumerate(centers):
        pairs = sorted(enumerate(center), key=lambda x: -x[1])[:top_k]
        labels[cid] = " | ".join(vocab[i] for i, _ in pairs if i < len(vocab))
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# 生成重复候选对（dedup candidates）
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidates(preds, top_per_cluster=80, threshold=0.55, label=""):
    """
    每个簇内取 top_per_cluster 条（按 score），两两计算余弦相似度（L2已归一化，dot=cos），
    过滤 >= threshold 的对，输出 (question_id_1, question_id_2, similarity)。
    """
    print("\n[Candidates] %s: generating dedup candidates (top=%d, threshold=%.2f)" % (
        label, top_per_cluster, threshold))
    t0 = time.time()

    # 取每簇前 top_per_cluster 条
    w = Window.partitionBy("cluster").orderBy(desc("score"))
    top = preds.withColumn("rank", row_number().over(w)) \
               .filter(col("rank") <= top_per_cluster)

    # 自连接
    a = top.alias("a")
    b = top.alias("b")
    pairs = a.join(b,
                   (col("a.cluster") == col("b.cluster")) &
                   (col("a.question_id") < col("b.question_id")))

    # 余弦相似度（特征已 L2 归一化，dot product = cosine similarity）
    from pyspark.ml.linalg import Vectors

    def dot_product(v1, v2):
        try:
            return float(v1.dot(v2))
        except Exception:
            return 0.0

    dot_udf = udf(dot_product, DoubleType())

    cands = pairs.withColumn(
        "similarity", dot_udf(col("a.features"), col("b.features"))
    ).filter(col("similarity") >= threshold) \
     .select(
         col("a.question_id").alias("question_id_1"),
         col("b.question_id").alias("question_id_2"),
         col("a.cluster").alias("cluster"),
         col("similarity"),
     )

    n = cands.count()
    print("[Candidates] %s: %d pairs above threshold=%.2f, time=%.2fs" % (
        label, n, threshold, time.time() - t0))
    return cands, n


# ─────────────────────────────────────────────────────────────────────────────
# 保存输出
# ─────────────────────────────────────────────────────────────────────────────

def save_assignments(preds, output_dir):
    preds.select("question_id", "title", "score", "cluster") \
         .write.mode("overwrite").json(output_dir + "/cluster_assignments")
    preds.groupBy("cluster").agg(count("*").alias("cluster_size")) \
         .orderBy("cluster") \
         .write.mode("overwrite").json(output_dir + "/cluster_summary")
    print("[Save] Assignments -> %s" % output_dir)


def save_candidates(cands, output_dir):
    cands.select("question_id_1", "question_id_2", "similarity") \
         .write.mode("overwrite").json(output_dir + "/duplicate_candidates")
    print("[Save] Candidates -> %s/duplicate_candidates" % output_dir)


def save_cluster_keywords(spark, preds, labels, output_dir):
    rows = [(int(cid), lbl) for cid, lbl in labels.items()]
    spark.createDataFrame(rows, ["cluster", "cluster_label"]) \
         .write.mode("overwrite").json(output_dir + "/cluster_keywords")
    print("[Save] Keywords -> %s/cluster_keywords" % output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(exp_name, preds_raw, model, spark, vocab, args, do_merge=False):
    """运行单个实验，保存结果，返回统计字典"""
    out = "%s/%s" % (args.output, exp_name)
    print("\n" + "=" * 65)
    print("[Exp] %s" % exp_name)
    print("=" * 65)

    if do_merge:
        preds, final_k = merge_small_clusters(
            spark, preds_raw, model, args.merge_threshold, exp_name)
    else:
        preds, final_k = preds_raw, preds_raw.select("cluster").distinct().count()

    stat = distribution_stats(preds, exp_name)
    stat["final_k"] = final_k

    # 保存聚类分配
    save_assignments(preds, out)

    # 生成并保存候选对
    cands, n_cands = generate_candidates(
        preds, args.dedup_top, args.threshold, exp_name)
    save_candidates(cands, out)
    stat["n_cands"] = n_cands

    # E5: 聚类标签（对所有实验都生成）
    labels = cluster_labels(model, vocab, args.label_top_k)
    save_cluster_keywords(spark, preds, labels, out)

    return stat


def main():
    parser = argparse.ArgumentParser(description="E: Clustering Structure Experiments E1-E4")
    parser.add_argument("--input",          default="hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300")
    parser.add_argument("--output",         default="hdfs://node5:9000/user/root/output/e_structure")
    parser.add_argument("--k",              type=int,   default=45,    help="Best K from B (default: 45)")
    parser.add_argument("--max-features",   type=int,   default=5000)
    parser.add_argument("--min-df",         type=int,   default=5)
    parser.add_argument("--max-df",         type=float, default=0.9)
    parser.add_argument("--km-max-iter",    type=int,   default=30)
    parser.add_argument("--bkm-max-iter",   type=int,   default=20)
    parser.add_argument("--merge-threshold",type=float, default=0.1,
                        help="Small cluster threshold = avg_size * this (default: 0.1)")
    parser.add_argument("--dedup-top",      type=int,   default=80,
                        help="Top questions per cluster for dedup candidate generation")
    parser.add_argument("--threshold",      type=float, default=0.55,
                        help="Cosine similarity threshold for duplicate candidates")
    parser.add_argument("--label-top-k",   type=int,   default=5)
    parser.add_argument("--sample-ratio",   type=float, default=1.0)
    args = parser.parse_args()

    total_t0 = time.time()

    spark = SparkSession.builder \
        .appName("E_Structure_Experiments") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("\n" + "=" * 65)
    print("E: Clustering Structure Experiments")
    print("=" * 65)
    print("Input        :", args.input)
    print("Output base  :", args.output)
    print("K            :", args.k)
    print("Features     : TF-IDF C1 (vocab=%d, min_df=%d, max_df=%.1f)" % (
        args.max_features, args.min_df, args.max_df))
    print("Threshold    :", args.threshold)
    print("Merge thresh :", args.merge_threshold)
    print("=" * 65)

    try:
        # ── Step 1: 数据加载 ──────────────────────────────────────────────
        df = load_data(spark, args.input)
        if args.sample_ratio < 1.0:
            df = df.sample(False, args.sample_ratio, seed=42)

        # ── Step 2: 特征构建 ──────────────────────────────────────────────
        pipeline_model, processed, vocab = build_tfidf(
            df, args.max_features, args.min_df, args.max_df)

        print("\n[Cache] Caching processed data...")
        processed.cache()
        n_total = processed.count()
        print("[Cache] Ready. Records: %d" % n_total)

        # ── E1: KMeans ────────────────────────────────────────────────────
        print("\n" + "=" * 65)
        print("[E1] KMeans  K=%d" % args.k)
        t0 = time.time()
        km = KMeans(featuresCol="features", predictionCol="cluster",
                    k=args.k, maxIter=args.km_max_iter, seed=42)
        km_model = km.fit(processed)
        km_preds = km_model.transform(processed)
        km_time  = time.time() - t0
        km_wssse = km_model.computeCost(processed)
        print("[E1] Training time: %.2fs, WSSSE: %.4f" % (km_time, km_wssse))
        if HAS_EVAL:
            ev = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                     metricName="silhouette",
                                     distanceMeasure="squaredEuclidean")
            km_sil = ev.evaluate(km_preds)
            print("[E1] Silhouette: %.4f" % km_sil)

        stat_e1 = run_experiment("e1_kmeans", km_preds, km_model, spark, vocab, args)
        stat_e1["time"] = km_time;  stat_e1["wssse"] = km_wssse

        # ── E2: BisectingKMeans ───────────────────────────────────────────
        print("\n" + "=" * 65)
        print("[E2] BisectingKMeans  K=%d" % args.k)
        t0 = time.time()
        bkm = BisectingKMeans(featuresCol="features", predictionCol="cluster",
                              k=args.k, maxIter=args.bkm_max_iter,
                              minDivisibleClusterSize=1.0, seed=42)
        bkm_model = bkm.fit(processed)
        bkm_preds = bkm_model.transform(processed)
        bkm_time  = time.time() - t0
        bkm_wssse = bkm_model.computeCost(processed)
        print("[E2] Training time: %.2fs, WSSSE: %.4f" % (bkm_time, bkm_wssse))
        if HAS_EVAL:
            bkm_sil = ev.evaluate(bkm_preds)
            print("[E2] Silhouette: %.4f" % bkm_sil)

        stat_e2 = run_experiment("e2_bisecting", bkm_preds, bkm_model, spark, vocab, args)
        stat_e2["time"] = bkm_time; stat_e2["wssse"] = bkm_wssse

        # ── E3: KMeans + 小簇合并 ─────────────────────────────────────────
        stat_e3 = run_experiment("e3_kmeans_merge", km_preds, km_model,
                                 spark, vocab, args, do_merge=True)
        stat_e3["time"] = km_time; stat_e3["wssse"] = km_wssse

        # ── E4: BisectingKMeans + 小簇合并 ───────────────────────────────
        stat_e4 = run_experiment("e4_bisecting_merge", bkm_preds, bkm_model,
                                 spark, vocab, args, do_merge=True)
        stat_e4["time"] = bkm_time; stat_e4["wssse"] = bkm_wssse

        # ── E5: 聚类标签（已在各实验中生成）─────────────────────────────
        # 额外展示 E4 的 top-10 标签示例
        labels_e4 = cluster_labels(bkm_model, vocab, args.label_top_k)
        bkm_top = bkm_preds.groupBy("cluster").agg(count("*").alias("sz")) \
                            .orderBy(desc("sz")).limit(10).collect()
        print("\n[E5] BisectingKMeans+Merge cluster label examples (top-10):")
        for row in bkm_top:
            cid = row["cluster"]
            print("  Cluster %3d (size=%4d): [%s]" % (
                cid, row["sz"], labels_e4.get(cid, "")))

        # ── 最终汇总报告 ──────────────────────────────────────────────────
        total_elapsed = time.time() - total_t0
        R = "[REPORT]"
        sep = R + " " + "=" * 63
        print("\n" + sep)
        print(R + " E: Clustering Structure Experiments - Summary")
        print(sep)
        print(R + " Input     : %s" % args.input)
        print(R + " K (best)  : %d (from B)" % args.k)
        print(R + " Features  : TF-IDF C1 (vocab=%d, min_df=%d, max_df=%.1f)" % (
            args.max_features, args.min_df, args.max_df))
        print(R + " Threshold : %.2f" % args.threshold)
        print(R + " Samples   : %d" % n_total)
        print(sep)

        fmt = R + "  %-6s %-20s %-8s %5s %5s %8s %6s %6s %8s"
        print(fmt % ("Exp", "Description", "Time(s)", "MaxPct", "CV", "Small", "FinalK", "Cands", "WSSSE"))
        print(R + "  " + "-" * 78)
        for exp, stat, desc in [
            ("E1", stat_e1, "KMeans"),
            ("E2", stat_e2, "BisectingKMeans"),
            ("E3", stat_e3, "KMeans+Merge"),
            ("E4", stat_e4, "BisectingKMeans+Merge"),
        ]:
            print(fmt % (
                exp, desc,
                "%.1f" % stat["time"],
                "%.2f%%" % stat["max_pct"],
                "%.4f" % stat["cv"],
                str(stat["small"]),
                str(stat["final_k"]),
                str(stat["n_cands"]),
                "%.2f" % stat["wssse"],
            ))

        print(sep)
        print(R + " E5: Cluster labels generated for all experiments (cluster_keywords/)")
        print(R + " Total experiment time: %.1fs (%.1f min)" % (
            total_elapsed, total_elapsed / 60))
        print(R + " Output base: %s" % args.output)
        print(sep)
        print(R + " NEXT STEP: run evaluate_dedup_from_spark_outputs.py for each experiment")
        print(R + "   e1: %s/e1_kmeans/" % args.output)
        print(R + "   e2: %s/e2_bisecting/" % args.output)
        print(R + "   e3: %s/e3_kmeans_merge/" % args.output)
        print(R + "   e4: %s/e4_bisecting_merge/" % args.output)
        print(sep)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
