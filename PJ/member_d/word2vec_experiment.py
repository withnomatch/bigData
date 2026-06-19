# -*- coding: utf-8 -*-
"""
Word2Vec特征提取对比实验
三组方案:
  1. 纯TF-IDF方案 (Baseline)
  2. 纯Word2Vec方案 (简单平均)
  3. TF-IDF加权Word2Vec方案

对三组方案分别运行K-Means聚类，计算轮廓系数、聚类分布、运行时间等指标，
并进行语义分析验证。
"""

from __future__ import print_function
import re
import argparse
import json
import time
import sys
import numpy as np

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import col, udf, desc, count, collect_list, first, row_number, size as array_size
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
    Word2Vec,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, DenseVector, SparseVector, VectorUDT

try:
    from pyspark.ml.evaluation import ClusteringEvaluator
    HAS_CLUSTERING_EVALUATOR = True
except ImportError:
    HAS_CLUSTERING_EVALUATOR = False


# ============================================================
# 文本预处理工具
# ============================================================

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


# ============================================================
# 数据加载与预处理
# ============================================================

def create_spark_session(app_name="Word2Vec_Clustering_Experiment"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_preprocess(spark, input_path, sample_ratio=1.0):
    print("\n" + "=" * 60)
    print("[Step 1] Loading & Preprocessing Data")
    print("=" * 60)

    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total records: %d" % total_count)

    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        print("Sampled records: %d (ratio: %f)" % (df.count(), sample_ratio))

    preprocess_udf = udf(preprocess_text, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))

    # Tokenize + StopWords (共享步骤)
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    pipeline = Pipeline(stages=[tokenizer, remover])
    df = pipeline.fit(df).transform(df)
    df = df.filter(array_size(col("filtered_tokens")) > 0)

    valid_count = df.count()
    print("Valid records after preprocessing: %d" % valid_count)
    return df


# ============================================================
# 实验1: 纯TF-IDF方案
# ============================================================

def run_tfidf_experiment(df, k=50, max_features=5000, min_df=5, max_iter=30):
    print("\n" + "=" * 60)
    print("[Experiment 1] Pure TF-IDF + K-Means")
    print("=" * 60)

    start_time = time.time()

    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=max_features,
        minDF=min_df,
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[cv, idf, normalizer])
    print("Fitting TF-IDF pipeline...")
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    vocab_size = len(pipeline_model.stages[0].vocabulary)
    print("Vocabulary size: %d" % vocab_size)

    # K-Means
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )
    print("Training K-Means (K=%d)..." % k)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("TF-IDF experiment complete, time: %.2fs" % elapsed)

    # 提取模型组件用于后续分析
    cv_model = pipeline_model.stages[0]
    idf_model = pipeline_model.stages[1]
    vocabulary = cv_model.vocabulary

    result = {
        "method": "TF-IDF",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "pipeline_model": pipeline_model,
        "vocabulary": vocabulary,
        "elapsed": elapsed,
        "feature_dim": vocab_size,
    }
    return result


# ============================================================
# 实验2: 纯Word2Vec方案 (简单平均)
# ============================================================

def run_w2v_experiment(df, k=50, vector_size=100, min_count=5, max_iter=30):
    print("\n" + "=" * 60)
    print("[Experiment 2] Pure Word2Vec (Simple Average) + K-Means")
    print("=" * 60)

    start_time = time.time()

    w2v = Word2Vec(
        inputCol="filtered_tokens",
        outputCol="w2v_features",
        vectorSize=vector_size,
        minCount=min_count,
        windowSize=5,
        maxIter=10,
        seed=42,
    )
    normalizer = Normalizer(inputCol="w2v_features", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[w2v, normalizer])
    print("Fitting Word2Vec pipeline (vectorSize=%d, minCount=%d)..." % (vector_size, min_count))
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    w2v_model = pipeline_model.stages[0]
    vocab_size = w2v_model.getVectors().count()
    print("Word2Vec vocabulary size: %d" % vocab_size)

    # K-Means
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )
    print("Training K-Means (K=%d)..." % k)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("Word2Vec experiment complete, time: %.2fs" % elapsed)

    result = {
        "method": "Word2Vec",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "pipeline_model": pipeline_model,
        "w2v_model": w2v_model,
        "elapsed": elapsed,
        "feature_dim": vector_size,
    }
    return result


# ============================================================
# 实验3: TF-IDF加权Word2Vec方案
# ============================================================

def run_tfidf_w2v_experiment(df, spark, k=50, max_features=5000, min_df=5,
                              vector_size=100, min_count=5, max_iter=30):
    print("\n" + "=" * 60)
    print("[Experiment 3] TF-IDF Weighted Word2Vec + K-Means")
    print("=" * 60)

    start_time = time.time()

    # ---- Step A: 训练Word2Vec模型获取词向量 ----
    print("[3a] Training Word2Vec model...")
    w2v = Word2Vec(
        inputCol="filtered_tokens",
        outputCol="w2v_features",
        vectorSize=vector_size,
        minCount=min_count,
        windowSize=5,
        maxIter=10,
        seed=42,
    )
    w2v_model = w2v.fit(df)
    w2v_vocab_size = w2v_model.getVectors().count()
    print("Word2Vec vocabulary size: %d" % w2v_vocab_size)

    # 收集词向量到广播变量
    print("[3b] Collecting word vectors for broadcast...")
    word_vectors_data = w2v_model.getVectors().collect()
    word_vectors_dict = {}
    for row in word_vectors_data:
        word_vectors_dict[row.word] = row.vector.toArray().tolist()
    print("Collected %d word vectors" % len(word_vectors_dict))

    # ---- Step B: 训练TF-IDF模型获取IDF权重 ----
    print("[3c] Training TF-IDF model...")
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=max_features,
        minDF=min_df,
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    tfidf_pipeline = Pipeline(stages=[cv, idf])
    tfidf_model = tfidf_pipeline.fit(df)

    cv_model = tfidf_model.stages[0]
    idf_model = tfidf_model.stages[1]
    vocabulary = cv_model.vocabulary
    idf_values = idf_model.idf.toArray()

    # 构建word -> idf映射
    word_idf_dict = {}
    for i, word in enumerate(vocabulary):
        word_idf_dict[word] = float(idf_values[i])
    print("IDF mapping size: %d" % len(word_idf_dict))

    # ---- Step C: 计算TF-IDF加权平均词向量 ----
    print("[3d] Computing TF-IDF weighted Word2Vec features...")
    sc = spark.sparkContext
    wv_broadcast = sc.broadcast(word_vectors_dict)
    wi_broadcast = sc.broadcast(word_idf_dict)
    vs = vector_size

    def compute_weighted_w2v(tokens):
        wv = wv_broadcast.value
        wi = wi_broadcast.value
        result = [0.0] * vs
        total_weight = 0.0

        # 计算TF (词频)
        tf_counts = {}
        for t in tokens:
            tf_counts[t] = tf_counts.get(t, 0) + 1
        doc_len = len(tokens)

        for t in tokens:
            if t in wv and t in wi:
                tf = float(tf_counts[t]) / doc_len
                idf_val = wi[t]
                weight = tf * idf_val
                vec = wv[t]
                for i in range(vs):
                    result[i] += weight * vec[i]
                total_weight += weight

        if total_weight > 0:
            for i in range(vs):
                result[i] /= total_weight

        return Vectors.dense(result)

    weighted_w2v_udf = udf(compute_weighted_w2v, VectorUDT())
    processed_df = df.withColumn("features", weighted_w2v_udf(col("filtered_tokens")))

    # 归一化（Normalizer是Transformer，直接transform）
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=2.0)
    processed_df = normalizer.transform(processed_df) \
        .drop("features") \
        .withColumnRenamed("features_norm", "features")

    # 过滤零向量
    from pyspark.sql.functions import udf as udf2
    from pyspark.sql.types import BooleanType
    def is_nonzero(v):
        if v is None:
            return False
        arr = v.toArray() if hasattr(v, 'toArray') else v
        return float(np.sum(np.array(arr) ** 2)) > 1e-10
    nonzero_udf = udf2(is_nonzero, BooleanType())
    before_count = processed_df.count()
    processed_df = processed_df.filter(nonzero_udf(col("features")))
    after_count = processed_df.count()
    if before_count != after_count:
        print("Filtered %d zero-vector records (%d -> %d)" % (
            before_count - after_count, before_count, after_count))

    # ---- Step D: K-Means聚类 ----
    print("[3e] Training K-Means (K=%d)..." % k)
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("TF-IDF+Word2Vec experiment complete, time: %.2fs" % elapsed)

    result = {
        "method": "TF-IDF+Word2Vec",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "w2v_model": w2v_model,
        "vocabulary": vocabulary,
        "elapsed": elapsed,
        "feature_dim": vector_size,
    }
    return result


# ============================================================
# 评估指标
# ============================================================

def evaluate_clustering(result, k):
    """计算轮廓系数、WCSS、聚类分布等指标"""
    method = result["method"]
    predictions = result["predictions"]
    kmeans_model = result["kmeans_model"]

    print("\n" + "-" * 40)
    print("Evaluating: %s" % method)
    print("-" * 40)

    # 1. 轮廓系数
    silhouette = 0.0
    if HAS_CLUSTERING_EVALUATOR:
        try:
            evaluator = ClusteringEvaluator(
                featuresCol="features",
                predictionCol="cluster",
                metricName="silhouette",
                distanceMeasure="cosine",
            )
            silhouette = evaluator.evaluate(predictions)
            print("Silhouette Score (cosine): %.4f" % silhouette)
        except Exception as e:
            print("ClusteringEvaluator failed: %s" % str(e))
            silhouette = compute_silhouette_sampled(predictions, k)
    else:
        print("ClusteringEvaluator not available, computing sampled silhouette...")
        silhouette = compute_silhouette_sampled(predictions, k)

    # 2. WCSS (Within-Cluster Sum of Squares)
    training_cost = 0.0
    try:
        # Spark 2.0.0: computeCost方法
        training_cost = kmeans_model.computeCost(predictions)
        print("WCSS (computeCost): %.4f" % training_cost)
    except Exception:
        try:
            training_cost = kmeans_model.summary.trainingCost
            print("WCSS (Training Cost): %.4f" % training_cost)
        except Exception:
            training_cost = compute_wcss(predictions, kmeans_model)
            print("WCSS (computed): %.4f" % training_cost)

    # 3. 聚类分布统计
    cluster_sizes = predictions.groupBy("cluster").count().orderBy("cluster")
    sizes_data = cluster_sizes.collect()
    sizes = [row["count"] for row in sizes_data]
    non_empty = len([s for s in sizes if s > 0])

    print("Non-empty clusters: %d / %d" % (non_empty, k))
    print("Max cluster size: %d" % max(sizes))
    print("Min cluster size: %d" % min(sizes))
    print("Avg cluster size: %.1f" % (float(sum(sizes)) / len(sizes)))
    print("Std of cluster sizes: %.1f" % np.std(sizes))

    # 显示前10个聚类的分布
    print("\nCluster size distribution (top 10):")
    cluster_sizes.show(10, truncate=False)

    # 4. 聚类大小分布直方图信息
    size_buckets = {}
    for s in sizes:
        bucket = (s // 500) * 500
        size_buckets[bucket] = size_buckets.get(bucket, 0) + 1
    print("Cluster size histogram:")
    for bucket in sorted(size_buckets.keys()):
        print("  %d-%d: %d clusters" % (bucket, bucket + 499, size_buckets[bucket]))

    metrics = {
        "method": method,
        "silhouette": silhouette,
        "wcss": training_cost,
        "non_empty_clusters": non_empty,
        "total_clusters": k,
        "max_cluster_size": max(sizes),
        "min_cluster_size": min(sizes),
        "avg_cluster_size": float(sum(sizes)) / len(sizes),
        "std_cluster_size": float(np.std(sizes)),
        "elapsed": result["elapsed"],
        "feature_dim": result["feature_dim"],
    }
    result["metrics"] = metrics
    return metrics


def compute_silhouette_sampled(predictions, k, sample_size=3000):
    """在采样数据上计算轮廓系数（Spark 2.0兼容方案）"""
    print("Computing sampled silhouette (sample_size=%d)..." % sample_size)
    try:
        total = predictions.count()
        fraction = min(1.0, float(sample_size) / total)
        sampled = predictions.sample(False, fraction, seed=42) \
            .select("features", "cluster").collect()

        if len(sampled) < 10:
            print("Sample too small for silhouette computation")
            return 0.0

        vectors = np.array([row.features.toArray() for row in sampled])
        labels = np.array([row.cluster for row in sampled])

        # 归一化向量（用于余弦距离）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors_norm = vectors / norms

        unique_labels = set(labels)
        if len(unique_labels) < 2:
            return 0.0

        silhouette_values = []
        # 限制计算量
        compute_size = min(len(vectors_norm), 2000)
        indices = np.random.choice(len(vectors_norm), compute_size, replace=False)

        for idx in indices:
            vec = vectors_norm[idx]
            label = labels[idx]

            # a: 同簇平均距离
            same_mask = labels == label
            same_mask[idx] = False
            if np.sum(same_mask) == 0:
                continue
            same_vecs = vectors_norm[same_mask]
            a = np.mean(1.0 - np.dot(same_vecs, vec))

            # b: 最近其他簇平均距离
            b = float('inf')
            for other_label in unique_labels:
                if other_label == label:
                    continue
                other_mask = labels == other_label
                if np.sum(other_mask) == 0:
                    continue
                other_vecs = vectors_norm[other_mask]
                dist = np.mean(1.0 - np.dot(other_vecs, vec))
                b = min(b, dist)

            if max(a, b) > 0:
                silhouette_values.append((b - a) / max(a, b))

        if len(silhouette_values) == 0:
            return 0.0
        result = float(np.mean(silhouette_values))
        print("Sampled Silhouette Score: %.4f (computed on %d points)" % (result, len(silhouette_values)))
        return result
    except Exception as e:
        print("Silhouette computation failed: %s" % str(e))
        return 0.0


def compute_wcss(predictions, kmeans_model):
    """手动计算WCSS"""
    centroids = kmeans_model.clusterCenters()
    data = predictions.select("features", "cluster").collect()
    wcss = 0.0
    for row in data:
        vec = np.array(row.features.toArray())
        centroid = np.array(centroids[row.cluster])
        wcss += np.sum((vec - centroid) ** 2)
    return wcss


# ============================================================
# 语义分析验证
# ============================================================

def semantic_analysis(result, top_n_clusters=5, top_n_keywords=10, top_n_questions=5):
    """对单个实验结果进行语义分析"""
    method = result["method"]
    predictions = result["predictions"]
    kmeans_model = result["kmeans_model"]

    print("\n" + "=" * 60)
    print("[Semantic Analysis] %s" % method)
    print("=" * 60)

    # ---- 1. 提取聚类关键词 ----
    print("\n--- Cluster Keywords ---")
    cluster_keywords = extract_cluster_keywords(result, top_n_keywords)

    # ---- 2. 获取最大聚类 ----
    cluster_stats = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy(desc("cluster_size"))

    top_clusters = cluster_stats.limit(top_n_clusters).collect()

    for row in top_clusters:
        cluster_id = row["cluster"]
        cluster_size = row["cluster_size"]
        keywords = cluster_keywords.get(cluster_id, [])
        print("\nCluster %d (size=%d):" % (cluster_id, cluster_size))
        print("  Keywords: %s" % ", ".join(keywords))

    # ---- 3. 每个聚类的代表问题 ----
    print("\n--- Representative Questions ---")
    window = Window.partitionBy("cluster").orderBy(desc("score"))
    top_questions = predictions.withColumn(
        "rank", row_number().over(window)
    ).filter(col("rank") <= top_n_questions).select(
        col("cluster"),
        col("rank"),
        col("title"),
        col("score"),
    ).orderBy("cluster", "rank")

    top_q_data = top_questions.collect()
    # 只显示top_n_clusters个聚类的代表问题
    top_cluster_ids = [row["cluster"] for row in top_clusters]
    shown = 0
    for q_row in top_q_data:
        if q_row["cluster"] in top_cluster_ids:
            print("  [C%d] %s (score=%d)" % (
                q_row["cluster"], q_row["title"], q_row["score"]))
            shown += 1
            if shown >= top_n_clusters * top_n_questions:
                break

    # ---- 4. 聚类语义一致性分析 ----
    print("\n--- Semantic Coherence Analysis ---")
    coherence = compute_semantic_coherence(predictions, kmeans_model, sample_per_cluster=50)
    print("Average within-cluster cosine similarity: %.4f" % coherence)

    result["cluster_keywords"] = cluster_keywords
    result["semantic_coherence"] = coherence
    return cluster_keywords, coherence


def extract_cluster_keywords(result, top_n=10):
    """提取每个聚类的关键词"""
    method = result["method"]
    kmeans_model = result["kmeans_model"]
    centroids = kmeans_model.clusterCenters()

    cluster_keywords = {}

    if method == "TF-IDF":
        # TF-IDF空间中，质心各维度对应词汇
        vocabulary = result["vocabulary"]
        for i, centroid in enumerate(centroids):
            top_indices = centroid.argsort()[-top_n:][::-1]
            keywords = [vocabulary[j] for j in top_indices if centroid[j] > 0]
            cluster_keywords[i] = keywords

    elif method in ("Word2Vec", "TF-IDF+Word2Vec"):
        # Word2Vec空间中，找离质心最近的词
        w2v_model = result["w2v_model"]
        vectors_df = w2v_model.getVectors()
        vectors_data = vectors_df.collect()

        words = [row.word for row in vectors_data]
        word_vecs = np.array([row.vector.toArray() for row in vectors_data])

        # 归一化
        norms = np.linalg.norm(word_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        word_vecs_norm = word_vecs / norms

        for i, centroid in enumerate(centroids):
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
            similarities = np.dot(word_vecs_norm, centroid_norm)
            top_indices = similarities.argsort()[-top_n:][::-1]
            keywords = [words[j] for j in top_indices if similarities[j] > 0]
            cluster_keywords[i] = keywords

    return cluster_keywords


def compute_semantic_coherence(predictions, kmeans_model, sample_per_cluster=50):
    """计算聚类语义一致性：簇内平均余弦相似度"""
    centroids = kmeans_model.clusterCenters()

    # 采样每个聚类的部分点
    sampled = predictions.select("features", "cluster").sample(
        False, 0.1, seed=42).limit(5000).collect()

    if len(sampled) == 0:
        return 0.0

    cluster_sims = {}
    for row in sampled:
        vec = np.array(row.features.toArray())
        centroid = np.array(centroids[row.cluster])
        # 余弦相似度
        norm_v = np.linalg.norm(vec)
        norm_c = np.linalg.norm(centroid)
        if norm_v > 0 and norm_c > 0:
            sim = float(np.dot(vec, centroid) / (norm_v * norm_c))
        else:
            sim = 0.0
        cid = row.cluster
        if cid not in cluster_sims:
            cluster_sims[cid] = []
        cluster_sims[cid].append(sim)

    # 计算各簇平均相似度
    avg_sims = []
    for cid, sims in cluster_sims.items():
        if len(sims) >= 2:
            avg_sims.append(np.mean(sims))

    if len(avg_sims) == 0:
        return 0.0
    return float(np.mean(avg_sims))


# ============================================================
# 跨方法对比分析
# ============================================================

def cross_method_comparison(results):
    """跨方法对比分析"""
    print("\n" + "=" * 60)
    print("[Cross-Method Comparison]")
    print("=" * 60)

    # ---- 1. 对比表 ----
    print("\n--- Performance Comparison Table ---")
    header = "%-18s %-12s %-12s %-10s %-10s %-10s %-10s %-10s" % (
        "Method", "Silhouette", "WCSS", "Clusters", "MaxSize", "MinSize",
        "AvgSize", "Time(s)")
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print("%-18s %-12.4f %-12.2f %-10d %-10d %-10d %-10.1f %-10.2f" % (
            m["method"], m["silhouette"], m["wcss"],
            m["non_empty_clusters"], m["max_cluster_size"],
            m["min_cluster_size"], m["avg_cluster_size"], m["elapsed"]))

    # ---- 2. 语义一致性对比 ----
    print("\n--- Semantic Coherence Comparison ---")
    for r in results:
        coherence = r.get("semantic_coherence", 0.0)
        print("  %s: %.4f" % (r["method"], coherence))

    # ---- 3. 聚类关键词对比（取最大3个簇） ----
    print("\n--- Cluster Keyword Comparison (Top 3 Largest Clusters) ---")
    for r in results:
        method = r["method"]
        predictions = r["predictions"]
        cluster_keywords = r.get("cluster_keywords", {})

        cluster_sizes = predictions.groupBy("cluster").count() \
            .orderBy(desc("count")).limit(3).collect()

        print("\n  [%s]" % method)
        for row in cluster_sizes:
            cid = row["cluster"]
            csize = row["count"]
            keywords = cluster_keywords.get(cid, [])[:8]
            print("    Cluster %d (size=%d): %s" % (cid, csize, ", ".join(keywords)))

    # ---- 4. 样本问题跨方法聚类对比 ----
    print("\n--- Sample Questions: Cross-Method Cluster Assignment ---")
    # 取score最高的10个问题
    sample_questions = results[0]["predictions"] \
        .orderBy(desc("score")).limit(10) \
        .select("question_id", "title").collect()

    for sq in sample_questions:
        qid = sq["question_id"]
        title = sq["title"][:60]
        assignments = []
        for r in results:
            pred = r["predictions"].filter(col("question_id") == qid) \
                .select("cluster").collect()
            if pred:
                assignments.append("%s->C%d" % (r["method"][:6], pred[0]["cluster"]))
            else:
                assignments.append("%s->N/A" % r["method"][:6])
        print("  [%s] %s" % (qid, title))
        print("    %s" % " | ".join(assignments))


# ============================================================
# 保存结果
# ============================================================

def save_results(results, output_path):
    print("\n" + "=" * 60)
    print("[Saving Results]")
    print("=" * 60)

    for r in results:
        method = r["method"].lower().replace("+", "_").replace("-", "_")
        predictions = r["predictions"]

        # 聚类分配
        output_df = predictions.select(
            col("question_id"),
            col("title"),
            col("cluster"),
            col("score").alias("question_score"),
        )
        path = "%s/%s/cluster_assignments" % (output_path, method)
        output_df.write.mode("overwrite").json(path)
        print("Saved: %s" % path)

        # 聚类摘要
        cluster_summary = predictions.groupBy("cluster").agg(
            count("*").alias("cluster_size"),
        ).orderBy("cluster")
        path = "%s/%s/cluster_summary" % (output_path, method)
        cluster_summary.write.mode("overwrite").json(path)
        print("Saved: %s" % path)

        # 每个聚类的Top问题
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
        path = "%s/%s/top_questions_per_cluster" % (output_path, method)
        top_per_cluster.write.mode("overwrite").json(path)
        print("Saved: %s" % path)

        metrics = dict(r["metrics"])
        metrics["semantic_coherence"] = float(r.get("semantic_coherence", 0.0))
        metrics["k"] = int(metrics["total_clusters"])
        metrics_path = "%s/%s/metrics" % (output_path, method)
        predictions.rdd.context.parallelize(
            [json.dumps(metrics, sort_keys=True)], 1
        ).saveAsTextFile(metrics_path)
        print("Saved: %s" % metrics_path)

    print("All results saved to: %s" % output_path)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Word2Vec Feature Extraction Comparison Experiment")
    parser.add_argument("--input", type=str, required=True,
                        help="Input data path (JSON)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output result path")
    parser.add_argument("--k", type=int, default=50,
                        help="Number of clusters K (default: 50)")
    parser.add_argument("--max-features", type=int, default=5000,
                        help="Max vocabulary size for TF-IDF (default: 5000)")
    parser.add_argument("--min-df", type=int, default=5,
                        help="Min document frequency (default: 5)")
    parser.add_argument("--vector-size", type=int, default=100,
                        help="Word2Vec vector dimension (default: 100)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Word2Vec min word count (default: 5)")
    parser.add_argument("--sample-ratio", type=float, default=1.0,
                        help="Sampling ratio (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=30,
                        help="K-Means max iterations (default: 30)")
    parser.add_argument(
        "--methods",
        type=str,
        default="tfidf,word2vec,tfidf_word2vec",
        help="Comma-separated methods: tfidf,word2vec,tfidf_word2vec",
    )

    args = parser.parse_args()

    total_start = time.time()

    print("\n" + "=" * 60)
    print("Word2Vec Feature Extraction Comparison Experiment")
    print("=" * 60)
    print("Input: %s" % args.input)
    print("Output: %s" % args.output)
    print("K: %d" % args.k)
    print("Max features: %d" % args.max_features)
    print("Vector size: %d" % args.vector_size)
    print("Sample ratio: %f" % args.sample_ratio)
    print("Methods: %s" % args.methods)
    print("=" * 60)

    spark = create_spark_session()

    try:
        # ---- 数据加载与预处理 ----
        df = load_and_preprocess(spark, args.input, args.sample_ratio)
        df.cache()

        results = []
        methods = set(
            item.strip().lower() for item in args.methods.split(",") if item.strip()
        )
        valid_methods = {"tfidf", "word2vec", "tfidf_word2vec"}
        unknown_methods = methods - valid_methods
        if unknown_methods:
            raise ValueError("Unknown methods: %s" % sorted(unknown_methods))

        # ---- 实验1: 纯TF-IDF ----
        if "tfidf" in methods:
            r1 = run_tfidf_experiment(
                df, args.k, args.max_features, args.min_df, args.max_iter
            )
            evaluate_clustering(r1, args.k)
            semantic_analysis(r1)
            results.append(r1)

        # ---- 实验2: 纯Word2Vec ----
        if "word2vec" in methods:
            r2 = run_w2v_experiment(
                df, args.k, args.vector_size, args.min_count, args.max_iter
            )
            evaluate_clustering(r2, args.k)
            semantic_analysis(r2)
            results.append(r2)

        # ---- 实验3: TF-IDF加权Word2Vec ----
        if "tfidf_word2vec" in methods:
            r3 = run_tfidf_w2v_experiment(
                df, spark, args.k, args.max_features, args.min_df,
                args.vector_size, args.min_count, args.max_iter)
            evaluate_clustering(r3, args.k)
            semantic_analysis(r3)
            results.append(r3)

        # ---- 跨方法对比 ----
        if len(results) > 1:
            cross_method_comparison(results)

        # ---- 保存结果 ----
        save_results(results, args.output)

        total_elapsed = time.time() - total_start

        # ---- 最终总结 ----
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE!")
        print("=" * 60)
        print("Total time: %.2fs" % total_elapsed)
        print("")
        print("Summary:")
        for r in results:
            m = r["metrics"]
            c = r.get("semantic_coherence", 0.0)
            print("  %-18s: Silhouette=%.4f, WCSS=%.2f, Coherence=%.4f, Time=%.2fs" % (
                m["method"], m["silhouette"], m["wcss"], c, m["elapsed"]))
        print("")
        print("Results saved to: %s" % args.output)
        print("=" * 60)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
