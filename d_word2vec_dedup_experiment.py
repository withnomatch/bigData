# -*- coding: utf-8 -*-
"""
D: Word2Vec / 语义特征优化实验
五组对比方案:
  D1: TF-IDF 基线 (C1参数: title+body, vocab=5000, min_df=5, max_df=0.9)
  D2: Word2Vec 平均向量
  D3: TF-IDF + Word2Vec 拼接特征
  D4: TF-IDF 加权 Word2Vec
  D5: 标题向量与正文向量分权融合

统一评测流程:
  特征构建 → K-Means聚类(K=45) → 候选生成 → 300对标注数据评测

重点分析:
  - Candidate Recall 是否提升
  - 漏报案例是否减少
  - 相关但不重复问题是否更容易误报
  - 语义特征带来的运行时间成本

兼容: Python 2.7 + Spark 2.0.0
"""

from __future__ import print_function
import re
import argparse
import time
import sys
import json
import subprocess
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
from pyspark.sql.functions import (
    col, udf, desc, count, collect_list, first, row_number,
    size as array_size, lit, when, broadcast, monotonically_increasing_id
)
from pyspark.sql.types import (
    StringType, ArrayType, FloatType, DoubleType, IntegerType, BooleanType
)
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


def preprocess_text_c1(title, body):
    """C1基线预处理: 与baseline_clustering.py完全一致
    只有3步: HTML清洗 → 合并title+body → 小写+去特殊字符
    """
    title_clean = clean_html(title) if title else ""
    body_clean = clean_html(body) if body else ""
    combined = title_clean + " " + body_clean
    combined = combined.lower()
    combined = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def preprocess_title_only_c1(title):
    """C1预处理: 仅标题"""
    title_clean = clean_html(title) if title else ""
    title_clean = title_clean.lower()
    title_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", title_clean)
    title_clean = re.sub(r"\s+", " ", title_clean).strip()
    return title_clean


def preprocess_body_only_c1(body):
    """C1预处理: 仅正文"""
    body_clean = clean_html(body) if body else ""
    body_clean = body_clean.lower()
    body_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", body_clean)
    body_clean = re.sub(r"\s+", " ", body_clean).strip()
    return body_clean


# ============================================================
# Word2Vec专用数据清理（在C1预处理基础上二次清理）
# ============================================================

def clean_tokens_for_w2v(tokens):
    """在C1预处理后的filtered_tokens上做Word2Vec专用二次清理

    C1预处理流程（与complete_clustering.py/baseline_clustering.py完全一致）:
      1. clean_html: HTML标签清洗
      2. title + body合并
      3. lower(): 小写化
      4. re.sub(r"[^a-zA-Z0-9\s]", " "): 去特殊字符
      5. Tokenizer: 分词
      6. StopWordsRemover: 默认停用词过滤
      → 输出: filtered_tokens

    本函数在filtered_tokens基础上额外清理（只影响D2-D5的Word2Vec训练）:
      - 纯数字token（如"2023"、"12345"）
      - 单字母token（保留db/id/sql等缩写）
      - 超长token（>20字符，base64/hash噪声）
      - 数字占比过高的token（版本号、hash片段）

    注意: 这些额外清理不影响D1(TF-IDF)，D1直接用filtered_tokens
    """
    if tokens is None:
        return []

    filtered = []
    for t in tokens:
        # 过滤空token
        if not t or len(t) == 0:
            continue
        # 过滤单字母token（保留常见缩写）
        if len(t) == 1 and t not in ["db", "id", "sql", "pl", "jp", "odp"]:
            continue
        # 过滤纯数字
        if t.isdigit():
            continue
        # 过滤超长token（base64、hash等噪声）
        if len(t) > 20:
            continue
        # 过滤数字占比过高的token（版本号、hash片段）
        digit_count = sum(1 for c in t if c.isdigit())
        if digit_count > len(t) * 0.6 and len(t) > 5:
            continue
        filtered.append(t)

    return filtered


# ============================================================
# Spark Session
# ============================================================

def create_spark_session(app_name="D_Word2Vec_Semantic_Experiment"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# 数据加载与预处理（统一C1基线预处理）
# ============================================================

def load_and_preprocess(spark, input_path, sample_ratio=1.0):
    """数据加载与预处理，严格复用C1的预处理结果

    数据流:
      原始JSON → C1预处理(HTML清洗+title+body+小写+去特殊字符)
               → C1分词+默认停用词 → filtered_tokens  (D1 TF-IDF用)
               → W2V二次清理(去纯数字/超长token/噪声) → w2v_tokens  (D2-D5 Word2Vec用)

    关键: D1的filtered_tokens与C1完全一致，保证D1结果可与C1报告对比
          D2-D5在C1的filtered_tokens基础上做W2V专用清理，保证Word2Vec训练数据质量
    """
    print("\n" + "=" * 70)
    print("[Step 1] Loading & Preprocessing Data")
    print("  复用C1预处理: HTML清洗 → title+body合并 → 小写 → 去特殊字符 → 分词 → 停用词")
    print("  W2V二次清理: 在C1的filtered_tokens上去纯数字/超长token/噪声token")
    print("=" * 70)

    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total records: %d" % total_count)

    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        print("Sampled records: %d (ratio: %f)" % (df.count(), sample_ratio))

    # === 第1步: C1基线预处理（与baseline_clustering.py完全一致）===
    preprocess_udf = udf(preprocess_text_c1, StringType())
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))

    # D5需要: 分别处理title和body（同样用C1预处理）
    title_udf = udf(preprocess_title_only_c1, StringType())
    body_udf = udf(preprocess_body_only_c1, StringType())
    df = df.withColumn("clean_title", title_udf(col("title")))
    df = df.withColumn("clean_body", body_udf(col("body")))

    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))

    # === 第2步: C1分词 + 默认停用词（与C1完全一致）===
    # D1(TF-IDF)直接用这个filtered_tokens，保证与C1结果一致
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    pipeline = Pipeline(stages=[tokenizer, remover])
    df = pipeline.fit(df).transform(df)
    df = df.filter(array_size(col("filtered_tokens")) > 0)

    # === 第3步: Word2Vec专用二次清理 ===
    # 在C1的filtered_tokens基础上，额外清理噪声token
    # D1(TF-IDF)不用这个，D2-D5(Word2Vec)用这个
    clean_w2v_udf = udf(clean_tokens_for_w2v, ArrayType(StringType()))
    df = df.withColumn("w2v_tokens", clean_w2v_udf(col("filtered_tokens")))

    # D5需要: 分别对title和body分词（C1方式 + W2V二次清理）
    tokenizer_title = Tokenizer(inputCol="clean_title", outputCol="raw_title_tokens")
    remover_title = StopWordsRemover(inputCol="raw_title_tokens", outputCol="title_tokens_base")
    tokenizer_body = Tokenizer(inputCol="clean_body", outputCol="raw_body_tokens")
    remover_body = StopWordsRemover(inputCol="raw_body_tokens", outputCol="body_tokens_base")
    pipeline_title_body = Pipeline(stages=[tokenizer_title, remover_title, tokenizer_body, remover_body])
    df = pipeline_title_body.fit(df).transform(df)

    # W2V二次清理title和body tokens
    df = df.withColumn("title_tokens", clean_w2v_udf(col("title_tokens_base")))
    df = df.withColumn("body_tokens", clean_w2v_udf(col("body_tokens_base")))

    valid_count = df.count()
    print("Valid records after preprocessing: %d" % valid_count)

    # 打印token统计信息（验证C1预处理 + W2V清理效果）
    sample_tokens = df.select("filtered_tokens", "w2v_tokens").limit(100).collect()
    c1_tokens = []
    w2v_tokens = []
    for row in sample_tokens:
        c1_tokens.extend(row["filtered_tokens"])
        w2v_tokens.extend(row["w2v_tokens"])
    from collections import Counter
    c1_counts = Counter(c1_tokens)
    w2v_counts = Counter(w2v_tokens)
    print("\nC1 Top 15 tokens (D1 TF-IDF uses these):")
    for token, count in c1_counts.most_common(15):
        print("  %s: %d" % (token, count))
    print("\nW2V Top 15 tokens (D2-D5 Word2Vec uses these):")
    for token, count in w2v_counts.most_common(15):
        print("  %s: %d" % (token, count))

    return df


# ============================================================
# D1: TF-IDF 基线 (C1参数)
# ============================================================

def run_d1_tfidf(df, k=45, max_features=5000, min_df=5, max_df=0.9, max_iter=30):
    """D1: TF-IDF基线，使用C1完全相同的预处理和参数重新跑
    C1结果（项目总结报告.md 8.2节）:
      Candidate Recall=0.56, Precision=1.00, Recall=0.03, F1=0.0583
    D1重新跑是为了和D2-D5在相同环境下对比，结果应与C1接近
    """
    print("\n" + "=" * 70)
    print("[D1] TF-IDF Baseline (C1: title+body, vocab=%d, min_df=%d, max_df=%.1f)" % (
        max_features, min_df, max_df))
    print("  C1报告结果: Candidate Recall=0.56, Precision=1.00, Recall=0.03, F1=0.0583")
    print("=" * 70)

    start_time = time.time()

    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=max_features,
        minDF=min_df,
        # 注意: Spark 2.0.0不支持maxDF参数，C1的baseline_clustering.py也未使用
        # max_df=0.9的过滤效果由IDF权重自然实现（高频词IDF值低）
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[cv, idf, normalizer])
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    vocab_size = len(pipeline_model.stages[0].vocabulary)
    print("Vocabulary size: %d" % vocab_size)

    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, maxIter=max_iter, seed=42)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("D1 complete, time: %.2fs" % elapsed)

    return {
        "method": "D1_TF-IDF",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "pipeline_model": pipeline_model,
        "elapsed": elapsed,
        "feature_dim": vocab_size,
    }


# ============================================================
# D2: Word2Vec 平均向量
# ============================================================

def run_d2_w2v(df, k=45, vector_size=100, min_count=5, max_iter=30):
    print("\n" + "=" * 70)
    print("[D2] Word2Vec Average Vector (vectorSize=%d, minCount=%d)" % (
        vector_size, min_count))
    print("=" * 70)

    start_time = time.time()

    w2v = Word2Vec(
        inputCol="w2v_tokens",
        outputCol="w2v_features",
        vectorSize=vector_size,
        minCount=min_count,
        windowSize=5,
        maxIter=10,
        seed=42,
    )
    normalizer = Normalizer(inputCol="w2v_features", outputCol="features", p=2.0)

    pipeline = Pipeline(stages=[w2v, normalizer])
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)

    w2v_model = pipeline_model.stages[0]
    vocab_size = w2v_model.getVectors().count()
    print("Word2Vec vocabulary size: %d" % vocab_size)

    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, maxIter=max_iter, seed=42)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("D2 complete, time: %.2fs" % elapsed)

    return {
        "method": "D2_Word2Vec",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "w2v_model": w2v_model,
        "elapsed": elapsed,
        "feature_dim": vector_size,
    }


# ============================================================
# D3: TF-IDF + Word2Vec 拼接特征
# ============================================================

def run_d3_tfidf_w2v_concat(df, spark, k=45, max_features=5000, min_df=5, max_df=0.9,
                             vector_size=100, min_count=5, max_iter=30):
    print("\n" + "=" * 70)
    print("[D3] TF-IDF + Word2Vec Concatenated Feature")
    print("=" * 70)

    start_time = time.time()

    # Step A: 训练TF-IDF（用C1的filtered_tokens + C1参数）
    print("[3a] Training TF-IDF...")
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=max_features,
        minDF=min_df,
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    normalizer_tfidf = Normalizer(inputCol="tfidf_features", outputCol="tfidf_norm", p=2.0)
    tfidf_pipeline = Pipeline(stages=[cv, idf, normalizer_tfidf])
    tfidf_model = tfidf_pipeline.fit(df)
    df_with_tfidf = tfidf_model.transform(df)

    # Step B: 训练Word2Vec（用W2V清理后的w2v_tokens）
    print("[3b] Training Word2Vec...")
    w2v = Word2Vec(
        inputCol="w2v_tokens",
        outputCol="w2v_features",
        vectorSize=vector_size,
        minCount=min_count,
        windowSize=5,
        maxIter=10,
        seed=42,
    )
    normalizer_w2v = Normalizer(inputCol="w2v_features", outputCol="w2v_norm", p=2.0)
    w2v_pipeline = Pipeline(stages=[w2v, normalizer_w2v])
    w2v_model = w2v_pipeline.fit(df_with_tfidf)
    df_with_both = w2v_model.transform(df_with_tfidf)

    # Step C: 拼接TF-IDF和Word2Vec特征
    print("[3c] Concatenating TF-IDF + Word2Vec features...")
    vs_tfidf = max_features
    vs_w2v = vector_size

    def concat_features(tfidf_vec, w2v_vec):
        t = tfidf_vec.toArray() if hasattr(tfidf_vec, 'toArray') else tfidf_vec
        w = w2v_vec.toArray() if hasattr(w2v_vec, 'toArray') else w2v_vec
        return Vectors.dense(np.concatenate([t, w]))

    concat_udf = udf(concat_features, VectorUDT())
    processed_df = df_with_both.withColumn(
        "features", concat_udf(col("tfidf_norm"), col("w2v_norm"))
    )

    # Step D: K-Means
    print("[3d] Training K-Means (K=%d)..." % k)
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, maxIter=max_iter, seed=42)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("D3 complete, time: %.2fs" % elapsed)

    return {
        "method": "D3_TF-IDF+W2V_Concat",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "elapsed": elapsed,
        "feature_dim": vs_tfidf + vs_w2v,
    }


# ============================================================
# D4: TF-IDF 加权 Word2Vec
# ============================================================

def run_d4_tfidf_weighted_w2v(df, spark, k=45, max_features=5000, min_df=5, max_df=0.9,
                               vector_size=100, min_count=5, max_iter=30):
    print("\n" + "=" * 70)
    print("[D4] TF-IDF Weighted Word2Vec")
    print("=" * 70)

    start_time = time.time()

    # Step A: 训练Word2Vec（用W2V清理后的w2v_tokens）
    print("[4a] Training Word2Vec model...")
    w2v = Word2Vec(
        inputCol="w2v_tokens",
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
    print("[4b] Collecting word vectors for broadcast...")
    word_vectors_data = w2v_model.getVectors().collect()
    word_vectors_dict = {}
    for row in word_vectors_data:
        word_vectors_dict[row.word] = row.vector.toArray().tolist()
    print("Collected %d word vectors" % len(word_vectors_dict))

    # Step B: 训练TF-IDF获取IDF权重（用C1的filtered_tokens + C1参数）
    print("[4c] Training TF-IDF model...")
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

    # Step C: 计算TF-IDF加权平均词向量
    print("[4d] Computing TF-IDF weighted Word2Vec features...")
    sc = spark.sparkContext
    wv_broadcast = sc.broadcast(word_vectors_dict)
    wi_broadcast = sc.broadcast(word_idf_dict)
    vs = vector_size

    def compute_weighted_w2v(tokens):
        wv = wv_broadcast.value
        wi = wi_broadcast.value
        result = [0.0] * vs
        total_weight = 0.0

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
    processed_df = df.withColumn("features", weighted_w2v_udf(col("w2v_tokens")))

    # 归一化
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=2.0)
    processed_df = normalizer.transform(processed_df) \
        .drop("features") \
        .withColumnRenamed("features_norm", "features")

    # 过滤零向量
    def is_nonzero(v):
        if v is None:
            return False
        arr = v.toArray() if hasattr(v, 'toArray') else v
        return float(np.sum(np.array(arr) ** 2)) > 1e-10

    nonzero_udf = udf(is_nonzero, BooleanType())
    before_count = processed_df.count()
    processed_df = processed_df.filter(nonzero_udf(col("features")))
    after_count = processed_df.count()
    if before_count != after_count:
        print("Filtered %d zero-vector records (%d -> %d)" % (
            before_count - after_count, before_count, after_count))

    # Step D: K-Means
    print("[4e] Training K-Means (K=%d)..." % k)
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, maxIter=max_iter, seed=42)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("D4 complete, time: %.2fs" % elapsed)

    # 清理广播变量
    wv_broadcast.unpersist()
    wi_broadcast.unpersist()

    return {
        "method": "D4_TF-IDF_Weighted_W2V",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "elapsed": elapsed,
        "feature_dim": vector_size,
    }


# ============================================================
# D5: 标题向量与正文向量分权融合
# ============================================================

def run_d5_title_body_fusion(df, spark, k=45, vector_size=100, min_count=5,
                              title_weight=2.0, body_weight=1.0, max_iter=30):
    print("\n" + "=" * 70)
    print("[D5] Title-Body Weighted Fusion (title_w=%.1f, body_w=%.1f)" % (
        title_weight, body_weight))
    print("=" * 70)

    start_time = time.time()

    # Step A: 在全量文本上训练Word2Vec（共享词向量空间，用W2V清理后的w2v_tokens）
    print("[5a] Training shared Word2Vec model on full text...")
    w2v = Word2Vec(
        inputCol="w2v_tokens",
        outputCol="w2v_features",
        vectorSize=vector_size,
        minCount=min_count,
        windowSize=5,
        maxIter=10,
        seed=42,
    )
    w2v_model = w2v.fit(df)
    w2v_vocab_size = w2v_model.getVectors().count()
    print("Shared Word2Vec vocabulary size: %d" % w2v_vocab_size)

    # 收集词向量到广播变量
    print("[5b] Collecting word vectors for broadcast...")
    word_vectors_data = w2v_model.getVectors().collect()
    word_vectors_dict = {}
    for row in word_vectors_data:
        word_vectors_dict[row.word] = row.vector.toArray().tolist()
    print("Collected %d word vectors" % len(word_vectors_dict))

    sc = spark.sparkContext
    wv_broadcast = sc.broadcast(word_vectors_dict)
    vs = vector_size
    tw = title_weight
    bw = body_weight

    # Step B: 计算标题和正文的加权融合向量
    print("[5c] Computing title-body weighted fusion vectors...")
    print("      title_weight=%.1f, body_weight=%.1f" % (tw, bw))

    def compute_fusion_vector(title_tokens, body_tokens):
        wv = wv_broadcast.value
        title_vec = [0.0] * vs
        body_vec = [0.0] * vs
        title_count = 0
        body_count = 0

        # 标题向量（简单平均）
        for t in title_tokens:
            if t in wv:
                vec = wv[t]
                for i in range(vs):
                    title_vec[i] += vec[i]
                title_count += 1

        if title_count > 0:
            for i in range(vs):
                title_vec[i] /= title_count

        # 正文向量（简单平均）
        for t in body_tokens:
            if t in wv:
                vec = wv[t]
                for i in range(vs):
                    body_vec[i] += vec[i]
                body_count += 1

        if body_count > 0:
            for i in range(vs):
                body_vec[i] /= body_count

        # 加权融合
        result = [0.0] * vs
        for i in range(vs):
            result[i] = tw * title_vec[i] + bw * body_vec[i]

        return Vectors.dense(result)

    fusion_udf = udf(compute_fusion_vector, VectorUDT())
    processed_df = df.withColumn(
        "features",
        fusion_udf(col("title_tokens"), col("body_tokens"))
    )

    # 归一化
    normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=2.0)
    processed_df = normalizer.transform(processed_df) \
        .drop("features") \
        .withColumnRenamed("features_norm", "features")

    # 过滤零向量
    def is_nonzero(v):
        if v is None:
            return False
        arr = v.toArray() if hasattr(v, 'toArray') else v
        return float(np.sum(np.array(arr) ** 2)) > 1e-10

    nonzero_udf = udf(is_nonzero, BooleanType())
    before_count = processed_df.count()
    processed_df = processed_df.filter(nonzero_udf(col("features")))
    after_count = processed_df.count()
    if before_count != after_count:
        print("Filtered %d zero-vector records (%d -> %d)" % (
            before_count - after_count, before_count, after_count))

    # Step C: K-Means
    print("[5d] Training K-Means (K=%d)..." % k)
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=k, maxIter=max_iter, seed=42)
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)

    elapsed = time.time() - start_time
    print("D5 complete, time: %.2fs" % elapsed)

    # 清理广播变量
    wv_broadcast.unpersist()

    return {
        "method": "D5_Title_Body_Fusion",
        "predictions": predictions,
        "kmeans_model": kmeans_model,
        "elapsed": elapsed,
        "feature_dim": vector_size,
    }


# ============================================================
# 候选生成：同簇内余弦相似度计算
# ============================================================

def generate_duplicate_candidates(predictions, spark, similarity_threshold=0.55,
                                   dedup_top_per_cluster=80):
    """从聚类结果生成疑似重复问题对"""
    print("\n" + "-" * 40)
    print("Generating duplicate candidates (threshold=%.2f, top=%d)" % (
        similarity_threshold, dedup_top_per_cluster))
    print("-" * 40)

    # 获取簇信息
    cluster_sizes = predictions.groupBy("cluster").count().orderBy(desc("count"))
    cluster_data = cluster_sizes.collect()

    candidates = []

    for row in cluster_data:
        cluster_id = row["cluster"]
        cluster_size = row["count"]

        if cluster_size < 2:
            continue

        # 获取该簇的问题（限制数量避免大簇计算量过大）
        cluster_questions = predictions.filter(
            col("cluster") == cluster_id
        ).limit(dedup_top_per_cluster).select(
            "question_id", "features"
        ).collect()

        if len(cluster_questions) < 2:
            continue

        # 计算簇内问题对的余弦相似度
        qids = [r["question_id"] for r in cluster_questions]
        vecs = [r["features"] for r in cluster_questions]

        for i in range(len(qids)):
            for j in range(i + 1, len(qids)):
                if vecs[i] is None or vecs[j] is None:
                    continue
                v1 = np.array(vecs[i].toArray())
                v2 = np.array(vecs[j].toArray())

                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 == 0 or n2 == 0:
                    continue

                cos_sim = float(np.dot(v1, v2) / (n1 * n2))
                if cos_sim >= similarity_threshold:
                    candidates.append((qids[i], qids[j], cos_sim, cluster_id))

    print("Generated %d duplicate candidate pairs" % len(candidates))
    return candidates


# ============================================================
# 300对标注数据评测
# ============================================================

def evaluate_with_300_pairs(predictions, candidates, spark, pairs_path):
    """使用300对标注数据评测去重效果"""
    print("\n" + "=" * 70)
    print("[Evaluation] 300-pair Annotation Evaluation")
    print("=" * 70)

    # 读取标注数据 - 使用Python内置csv模块（集群无pandas）
    import csv
    import tempfile
    import os
    import codecs

    # 如果是HDFS路径，先下载到本地临时文件
    if pairs_path.startswith("hdfs://"):
        local_tmp = tempfile.mktemp(suffix=".csv")
        print("Downloading pairs CSV from HDFS to local: %s" % local_tmp)
        subprocess.call(["hdfs", "dfs", "-get", pairs_path, local_tmp])
        csv_path = local_tmp
    else:
        csv_path = pairs_path

    # 用csv模块读取（正确处理引号和逗号）
    # Python 2.7 csv模块不支持unicode，需要用codecs打开文件
    with codecs.open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        pairs_data = [row for row in reader]

    # 清理临时文件
    if pairs_path.startswith("hdfs://"):
        os.remove(local_tmp)

    # 检测实际列名（兼容不同格式的标注文件）
    # pairs_data是unicode字段值的dict列表（codecs.open读取）
    col_names = pairs_data[0].keys() if pairs_data else []
    print("Columns: %s" % str(col_names))

    # 确定列名映射
    # question_id_1 / question_id_2 / label 的实际列名可能不同
    qid1_col = None
    qid2_col = None
    label_col = None

    for c in col_names:
        c_lower = c.lower()
        if "question_id_1" in c_lower or c == "_c0":
            qid1_col = c
        elif "question_id_2" in c_lower or c == "_c1":
            qid2_col = c
        elif c_lower in ["label", "reviewer_label", "is_duplicate", "_c2"]:
            label_col = c

    # 如果没找到，用位置推断
    if qid1_col is None and len(col_names) >= 3:
        qid1_col = col_names[0]
    if qid2_col is None and len(col_names) >= 3:
        qid2_col = col_names[1]
    if label_col is None and len(col_names) >= 3:
        # 优先用reviewer_label
        for c in col_names:
            if "reviewer_label" in c.lower():
                label_col = c
                break
        if label_col is None:
            label_col = col_names[2]

    print("Using columns: qid1=%s, qid2=%s, label=%s" % (qid1_col, qid2_col, label_col))

    # 构建聚类分配映射
    assignments = predictions.select("question_id", "cluster").collect()
    qid_to_cluster = {}
    for row in assignments:
        qid_to_cluster[int(row["question_id"])] = row["cluster"]

    # 构建候选对集合
    candidate_set = set()
    for qid1, qid2, sim, cid in candidates:
        key1 = str(qid1) + "_" + str(qid2)
        key2 = str(qid2) + "_" + str(qid1)
        candidate_set.add(key1)
        candidate_set.add(key2)

    # pairs_data已经是unicode字段值的dict列表（codecs.open读取）
    # 不需要再从Spark DataFrame collect

    # label映射: 重复=1, 其他=0
    # 参考 PJ/PJ/evaluate_dedup_from_spark_outputs.py 的做法：直接用unicode比较
    # codecs.open(encoding='utf-8')读取后，字段值已经是unicode
    DUPLICATE_LABEL = u"\u91cd\u590d"  # "重复"

    def normalize_label(label_val):
        if label_val is None:
            return 0
        # 字段值已经是unicode（codecs.open读取）
        try:
            if isinstance(label_val, unicode):
                return 1 if label_val.strip() == DUPLICATE_LABEL else 0
            else:
                # 兜底：尝试转换为unicode
                try:
                    u = unicode(label_val).strip()
                    return 1 if u == DUPLICATE_LABEL else 0
                except Exception:
                    return 0
        except Exception:
            return 0

    # 先统计标签分布（调试）
    label_counts = {}
    for row in pairs_data:
        label_raw = row.get(label_col)
        label_norm = normalize_label(label_raw)
        key = "%s -> %d" % (repr(label_raw), label_norm)
        label_counts[key] = label_counts.get(key, 0) + 1

    print("Label distribution:")
    for key, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print("  %s: %d" % (key, count))

    # 评测
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    candidate_recall_count = 0
    total_pairs = 0
    missing = 0

    # 统计漏报和误报
    false_negative_examples = []
    false_positive_examples = []

    for row in pairs_data:
        # CSV解析可能因body中的逗号/引号导致字段错位，需要容错
        try:
            qid1 = int(row[qid1_col])
            qid2 = int(row[qid2_col])
        except (ValueError, TypeError):
            missing += 1
            continue
        label = normalize_label(row[label_col])

        total_pairs += 1

        cluster1 = qid_to_cluster.get(qid1)
        cluster2 = qid_to_cluster.get(qid2)

        if cluster1 is None or cluster2 is None:
            missing += 1
            continue

        same_cluster = (cluster1 == cluster2)
        in_candidates = (str(qid1) + "_" + str(qid2)) in candidate_set

        # Candidate Recall: 真实重复对是否同簇
        if label == 1:
            if same_cluster:
                candidate_recall_count += 1
            else:
                false_negative_examples.append((qid1, qid2, "not_same_cluster"))

        # Precision/Recall/F1
        if label == 1:
            # 正例：真实重复
            if in_candidates:
                tp += 1
            else:
                fn += 1
                if same_cluster and not in_candidates:
                    false_negative_examples.append((qid1, qid2, "same_cluster_but_below_threshold"))
        else:
            # 负例：不重复
            if in_candidates:
                fp += 1
                false_positive_examples.append((qid1, qid2, label))
            else:
                tn += 1

    # 计算指标
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 统计正例数量
    positive_count = candidate_recall_count + sum(1 for qid1, qid2, reason in false_negative_examples
                                                  if reason == "not_same_cluster")

    candidate_recall = float(candidate_recall_count) / positive_count if positive_count > 0 else 0.0

    eval_result = {
        "pair_count": total_pairs,
        "missing_assignment": missing,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate_recall": candidate_recall,
        "candidate_recall_count": candidate_recall_count,
        "positive_count": positive_count,
    }

    print("\nEvaluation Results:")
    print("  Pair Count: %d" % total_pairs)
    print("  Missing Assignment: %d" % missing)
    print("  TP: %d, FP: %d, FN: %d, TN: %d" % (tp, fp, fn, tn))
    print("  Precision: %.4f" % precision)
    print("  Recall: %.4f" % recall)
    print("  F1: %.4f" % f1)
    print("  Candidate Recall: %.4f (%d / %d)" % (
        candidate_recall, candidate_recall_count, positive_count))

    # 漏报分析
    if false_negative_examples:
        print("\n  False Negative Examples (top 5):")
        for ex in false_negative_examples[:5]:
            print("    Q%s-Q%s: %s" % ex)

    # 误报分析
    if false_positive_examples:
        print("\n  False Positive Examples (top 5):")
        for ex in false_positive_examples[:5]:
            print("    Q%s-Q%s (label=%s)" % ex)

    return eval_result


# ============================================================
# 聚类分布统计
# ============================================================

def compute_cluster_stats(predictions, k):
    """计算聚类分布统计"""
    cluster_sizes = predictions.groupBy("cluster").count().orderBy("cluster")
    sizes_data = cluster_sizes.collect()
    sizes = [row["count"] for row in sizes_data]

    if len(sizes) == 0:
        return {}

    max_size = max(sizes)
    total = sum(sizes)
    max_ratio = float(max_size) / total if total > 0 else 0.0
    std_size = float(np.std(sizes))
    mean_size = float(np.mean(sizes))
    cv = std_size / mean_size if mean_size > 0 else 0.0

    return {
        "total_clusters": len(sizes),
        "max_cluster_size": max_size,
        "min_cluster_size": min(sizes),
        "avg_cluster_size": mean_size,
        "std_cluster_size": std_size,
        "max_cluster_ratio": max_ratio,
        "cv": cv,
    }


# ============================================================
# 保存结果
# ============================================================

def save_results(predictions, candidates, output_path, method_name, spark):
    """保存聚类分配和候选对"""
    print("\nSaving results for %s to %s" % (method_name, output_path))

    # 保存聚类分配
    assignments_df = predictions.select(
        col("question_id"),
        col("title"),
        col("cluster"),
    )
    assignments_df.write.mode("overwrite").json(output_path + "/cluster_assignments")

    # 保存候选对
    if candidates:
        candidates_rows = [
            (qid1, qid2, float(sim), int(cid))
            for qid1, qid2, sim, cid in candidates
        ]
        candidates_df = spark.createDataFrame(
            candidates_rows,
            ["question_id_1", "question_id_2", "similarity", "cluster_id"]
        )
        candidates_df.write.mode("overwrite").json(output_path + "/duplicate_candidates")
    else:
        # 空候选对
        empty_df = spark.createDataFrame(
            [(0, 0, 0.0, 0)],
            ["question_id_1", "question_id_2", "similarity", "cluster_id"]
        )
        empty_df.write.mode("overwrite").json(output_path + "/duplicate_candidates")

    print("Results saved.")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="D: Word2Vec / Semantic Feature Optimization Experiment"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input data path (fixed_eval_sample_10pct_plus_300)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output result path")
    parser.add_argument("--pairs", type=str, required=True,
                        help="300-pair annotation data path")
    parser.add_argument("--k", type=int, default=45,
                        help="Number of clusters K (default: 45)")
    parser.add_argument("--max-features", type=int, default=5000,
                        help="Max vocabulary size (default: 5000)")
    parser.add_argument("--min-df", type=int, default=5,
                        help="Min document frequency (default: 5)")
    parser.add_argument("--max-df", type=float, default=0.9,
                        help="Max document frequency (default: 0.9)")
    parser.add_argument("--vector-size", type=int, default=100,
                        help="Word2Vec vector size (default: 100)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="Word2Vec minCount (default: 5)")
    parser.add_argument("--max-iter", type=int, default=30,
                        help="K-Means max iterations (default: 30)")
    parser.add_argument("--similarity-threshold", type=float, default=0.55,
                        help="Duplicate candidate similarity threshold (default: 0.55)")
    parser.add_argument("--dedup-top-per-cluster", type=int, default=80,
                        help="Max questions per cluster for dedup (default: 80)")
    parser.add_argument("--title-weight", type=float, default=2.0,
                        help="Title weight for D5 (default: 2.0)")
    parser.add_argument("--body-weight", type=float, default=1.0,
                        help="Body weight for D5 (default: 1.0)")
    parser.add_argument("--experiments", type=str, default="1,2,3,4,5",
                        help="Which experiments to run, e.g. '1,2,4' (default: '1,2,3,4,5')")

    args = parser.parse_args()

    total_start = time.time()

    print("\n" + "=" * 70)
    print("D: Word2Vec / Semantic Feature Optimization Experiment")
    print("=" * 70)
    print("Input: %s" % args.input)
    print("Output: %s" % args.output)
    print("Pairs: %s" % args.pairs)
    print("K: %d" % args.k)
    print("TF-IDF: vocab=%d, min_df=%d, max_df=%.1f" % (
        args.max_features, args.min_df, args.max_df))
    print("Word2Vec: vectorSize=%d, minCount=%d" % (args.vector_size, args.min_count))
    print("Experiments: %s" % args.experiments)
    print("=" * 70)

    spark = create_spark_session()

    try:
        # 加载数据
        df = load_and_preprocess(spark, args.input, sample_ratio=1.0)

        # 结果汇总
        all_results = []

        exp_list = [int(x.strip()) for x in args.experiments.split(",")]

        # ---- D1: TF-IDF 基线 ----
        if 1 in exp_list:
            d1_result = run_d1_tfidf(
                df, k=args.k,
                max_features=args.max_features,
                min_df=args.min_df,
                max_df=args.max_df,
                max_iter=args.max_iter,
            )
            d1_candidates = generate_duplicate_candidates(
                d1_result["predictions"], spark,
                similarity_threshold=args.similarity_threshold,
                dedup_top_per_cluster=args.dedup_top_per_cluster,
            )
            d1_eval = evaluate_with_300_pairs(
                d1_result["predictions"], d1_candidates, spark, args.pairs
            )
            d1_stats = compute_cluster_stats(d1_result["predictions"], args.k)
            save_results(
                d1_result["predictions"], d1_candidates,
                args.output + "/d1_tfidf", d1_result["method"], spark
            )
            all_results.append({
                "method": "D1_TF-IDF",
                "semantic_method": "None (TF-IDF baseline)",
                "eval": d1_eval,
                "stats": d1_stats,
                "elapsed": d1_result["elapsed"],
            })

        # ---- D2: Word2Vec 平均向量 ----
        if 2 in exp_list:
            d2_result = run_d2_w2v(
                df, k=args.k,
                vector_size=args.vector_size,
                min_count=args.min_count,
                max_iter=args.max_iter,
            )
            d2_candidates = generate_duplicate_candidates(
                d2_result["predictions"], spark,
                similarity_threshold=args.similarity_threshold,
                dedup_top_per_cluster=args.dedup_top_per_cluster,
            )
            d2_eval = evaluate_with_300_pairs(
                d2_result["predictions"], d2_candidates, spark, args.pairs
            )
            d2_stats = compute_cluster_stats(d2_result["predictions"], args.k)
            save_results(
                d2_result["predictions"], d2_candidates,
                args.output + "/d2_w2v", d2_result["method"], spark
            )
            all_results.append({
                "method": "D2_Word2Vec",
                "semantic_method": "Word2Vec average",
                "eval": d2_eval,
                "stats": d2_stats,
                "elapsed": d2_result["elapsed"],
            })

        # ---- D3: TF-IDF + Word2Vec 拼接 ----
        if 3 in exp_list:
            d3_result = run_d3_tfidf_w2v_concat(
                df, spark, k=args.k,
                max_features=args.max_features,
                min_df=args.min_df,
                max_df=args.max_df,
                vector_size=args.vector_size,
                min_count=args.min_count,
                max_iter=args.max_iter,
            )
            d3_candidates = generate_duplicate_candidates(
                d3_result["predictions"], spark,
                similarity_threshold=args.similarity_threshold,
                dedup_top_per_cluster=args.dedup_top_per_cluster,
            )
            d3_eval = evaluate_with_300_pairs(
                d3_result["predictions"], d3_candidates, spark, args.pairs
            )
            d3_stats = compute_cluster_stats(d3_result["predictions"], args.k)
            save_results(
                d3_result["predictions"], d3_candidates,
                args.output + "/d3_tfidf_w2v_concat", d3_result["method"], spark
            )
            all_results.append({
                "method": "D3_TF-IDF+W2V_Concat",
                "semantic_method": "TF-IDF + W2V concatenation",
                "eval": d3_eval,
                "stats": d3_stats,
                "elapsed": d3_result["elapsed"],
            })

        # ---- D4: TF-IDF 加权 Word2Vec ----
        if 4 in exp_list:
            d4_result = run_d4_tfidf_weighted_w2v(
                df, spark, k=args.k,
                max_features=args.max_features,
                min_df=args.min_df,
                max_df=args.max_df,
                vector_size=args.vector_size,
                min_count=args.min_count,
                max_iter=args.max_iter,
            )
            d4_candidates = generate_duplicate_candidates(
                d4_result["predictions"], spark,
                similarity_threshold=args.similarity_threshold,
                dedup_top_per_cluster=args.dedup_top_per_cluster,
            )
            d4_eval = evaluate_with_300_pairs(
                d4_result["predictions"], d4_candidates, spark, args.pairs
            )
            d4_stats = compute_cluster_stats(d4_result["predictions"], args.k)
            save_results(
                d4_result["predictions"], d4_candidates,
                args.output + "/d4_tfidf_weighted_w2v", d4_result["method"], spark
            )
            all_results.append({
                "method": "D4_TF-IDF_Weighted_W2V",
                "semantic_method": "TF-IDF weighted W2V",
                "eval": d4_eval,
                "stats": d4_stats,
                "elapsed": d4_result["elapsed"],
            })

        # ---- D5: 标题向量与正文向量分权融合 ----
        if 5 in exp_list:
            d5_result = run_d5_title_body_fusion(
                df, spark, k=args.k,
                vector_size=args.vector_size,
                min_count=args.min_count,
                title_weight=args.title_weight,
                body_weight=args.body_weight,
                max_iter=args.max_iter,
            )
            d5_candidates = generate_duplicate_candidates(
                d5_result["predictions"], spark,
                similarity_threshold=args.similarity_threshold,
                dedup_top_per_cluster=args.dedup_top_per_cluster,
            )
            d5_eval = evaluate_with_300_pairs(
                d5_result["predictions"], d5_candidates, spark, args.pairs
            )
            d5_stats = compute_cluster_stats(d5_result["predictions"], args.k)
            save_results(
                d5_result["predictions"], d5_candidates,
                args.output + "/d5_title_body_fusion", d5_result["method"], spark
            )
            all_results.append({
                "method": "D5_Title_Body_Fusion",
                "semantic_method": "Title(2.0) + Body(1.0) fusion",
                "eval": d5_eval,
                "stats": d5_stats,
                "elapsed": d5_result["elapsed"],
            })

        # ============================================================
        # 汇总输出
        # ============================================================
        total_elapsed = time.time() - total_start

        print("\n" + "=" * 70)
        print("D Experiment Summary")
        print("=" * 70)

        # 表格输出
        print("\n| 方案 | 语义特征方法 | Candidate Recall | Precision | Recall | F1 | 运行时间 |")
        print("| -- | ------ | ---------------: | --------: | -----: | -: | ---: |")
        for r in all_results:
            e = r["eval"]
            print("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.1fs |" % (
                r["method"],
                r["semantic_method"],
                e["candidate_recall"],
                e["precision"],
                e["recall"],
                e["f1"],
                r["elapsed"],
            ))

        # 聚类分布统计
        print("\n| 方案 | 最大簇占比 | 簇大小CV | 最大簇 | 最小簇 | 平均簇 |")
        print("| -- | ------: | ------: | ------: | ------: | ------: |")
        for r in all_results:
            s = r["stats"]
            print("| %s | %.2f%% | %.4f | %d | %d | %.1f |" % (
                r["method"],
                s.get("max_cluster_ratio", 0) * 100,
                s.get("cv", 0),
                s.get("max_cluster_size", 0),
                s.get("min_cluster_size", 0),
                s.get("avg_cluster_size", 0),
            ))

        # 漏报和误报分析
        print("\n" + "-" * 70)
        print("False Negative / False Positive Analysis")
        print("-" * 70)
        for r in all_results:
            e = r["eval"]
            print("\n%s:" % r["method"])
            print("  FN (missed duplicates): %d" % e["fn"])
            print("  FP (false alarms): %d" % e["fp"])
            print("  Candidate Recall: %.4f (%d / %d)" % (
                e["candidate_recall"], e["candidate_recall_count"], e["positive_count"]))

        print("\n" + "=" * 70)
        print("Total experiment time: %.2fs" % total_elapsed)
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
