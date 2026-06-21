# -*- coding: utf-8 -*-
"""
C组成员：TF-IDF与预处理优化实验

实验方案：
C1: 基线对照（title + body）
C2: 文本加权（title×2 + body）
C3: 标签加权（title×2 + body + tags×1）
C4: 答案融合（title + body + tags + answers）
C5: 术语保留（C3 + Oracle错误码 + SQL关键词）
C6系列: 参数优化对照试验（基于C3，覆盖vocab 3000-10000, min_df 3-10, max_df 0.8-0.95）
C7: 综合最优方案（C3 + C5 + 最佳C6参数）
C8系列: C1参数调优实验（基于C1，验证参数对基线方案的影响）

兼容 Spark 2.0 / Python 2.7
"""

from __future__ import print_function

import argparse
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
from pyspark.sql.functions import (
    col,
    collect_list,
    count,
    desc,
    explode,
    row_number,
    udf,
)
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.window import Window


# Oracle领域停用词
DOMAIN_STOP_WORDS = [
    "oracle",
    "database",
    "db",
    "sql",
    "question",
    "answer",
    "thanks",
    "thank",
    "please",
    "help",
    "hello",
    "hi",
]

# SQL/PLSQL关键词列表（需要保留）
SQL_KEYWORDS = [
    "select", "insert", "update", "delete", "create", "alter", "drop", 
    "truncate", "merge", "begin", "end", "if", "then", "else", "elsif",
    "loop", "for", "while", "exit", "return", "declare", "execute",
    "commit", "rollback", "savepoint", "grant", "revoke", "cursor",
    "procedure", "function", "package", "trigger", "view", "table",
    "index", "constraint", "primary", "foreign", "unique", "check",
    "not", "null", "default", "values", "into", "from", "where",
    "join", "left", "right", "inner", "outer", "on", "and", "or",
    "in", "like", "between", "exists", "case", "when", "group",
    "order", "having", "limit", "union", "intersect", "minus",
    "distinct", "all", "any", "some", "asc", "desc", "count",
    "sum", "avg", "max", "min", "rownum", "rowid", "dual",
]


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        return " ".join(self.result)


def clean_html(value):
    """清洗HTML标签"""
    if not value:
        return ""
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(value)
        text = extractor.get_text()
    except Exception:
        text = value
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(value, preserve_terms=False):
    """文本规范化，可选保留专业术语"""
    if not value:
        return ""

    value = value.lower()
    
    # 保留Oracle错误码格式
    value = re.sub(r"\bora[\s_-]*(\d{4,5})\b", r"ora-\1", value)
    value = re.sub(r"\bpls[\s_-]*(\d{4,5})\b", r"pls-\1", value)
    
    if preserve_terms:
        # 保留SQL关键词：将关键词转换为特殊格式（如 select -> _select_）
        # 这样在后续处理中不会被当作普通词过滤
        for keyword in SQL_KEYWORDS:
            value = re.sub(r"\b" + keyword + r"\b", "_" + keyword + "_", value)
    
    # 移除特殊字符（但保留下划线，用于标记关键词）
    value = re.sub(r"[^a-z0-9_\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    
    return value


def top_answer_text(answers, max_answers):
    """提取top答案文本"""
    if not answers:
        return ""

    parsed = []
    for item in answers:
        try:
            body = item["body"]
            score = item["score"] or 0
        except Exception:
            body = getattr(item, "body", "")
            score = getattr(item, "score", 0) or 0
        parsed.append((score, body))

    parsed.sort(key=lambda x: x[0], reverse=True)
    return " ".join([clean_html(body) for score, body in parsed[:max_answers]])


def build_document_c1(title, body, tags, answers):
    """C1: 基线对照 - title + body"""
    title_text = clean_html(title)
    body_text = clean_html(body)
    
    combined = " ".join([title_text, body_text])
    return normalize_text(combined, preserve_terms=False)


def build_document_c2(title, body, tags, answers):
    """C2: Title加权 - title×2 + body"""
    title_text = clean_html(title)
    body_text = clean_html(body)
    
    # title重复2次
    combined = " ".join([
        title_text,
        title_text,  # title×2
        body_text,
    ])
    return normalize_text(combined, preserve_terms=False)


def build_document_c3(title, body, tags, answers):
    """C3: Title+Tags加权 - title×2 + body + tags×1"""
    title_text = clean_html(title)
    body_text = clean_html(body)
    tag_text = " ".join(tags) if tags else ""
    
    # title重复2次，tags重复1次
    combined = " ".join([
        title_text,
        title_text,  # title×2
        body_text,
        tag_text,    # tags×1
    ])
    return normalize_text(combined, preserve_terms=False)


def build_document_c4(title, body, tags, answers):
    """C4: 答案融合 - title + body + tags + answers"""
    title_text = clean_html(title)
    body_text = clean_html(body)
    tag_text = " ".join(tags) if tags else ""
    answer_text = top_answer_text(answers, 2)
    
    combined = " ".join([
        title_text,
        body_text,
        tag_text,
        answer_text,
    ])
    return normalize_text(combined, preserve_terms=False)


def build_document_c5(title, body, tags, answers):
    """C5: 术语保留 - C3 + Oracle错误码 + SQL关键词"""
    title_text = clean_html(title)
    body_text = clean_html(body)
    tag_text = " ".join(tags) if tags else ""
    
    # title重复2次，tags重复1次
    combined = " ".join([
        title_text,
        title_text,
        body_text,
        tag_text,
    ])
    return normalize_text(combined, preserve_terms=True)


def build_document_c6(title, body, tags, answers):
    """C6系列: 参数优化 - 使用C3文本构造"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_1(title, body, tags, answers):
    """C6_1: vocab=3000"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_2(title, body, tags, answers):
    """C6_2: vocab=5000"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_3(title, body, tags, answers):
    """C6_3: vocab=8000"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_4(title, body, tags, answers):
    """C6_4: vocab=10000"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_5(title, body, tags, answers):
    """C6_5: min_df=3"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_6(title, body, tags, answers):
    """C6_6: min_df=7"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_7(title, body, tags, answers):
    """C6_7: min_df=10"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_8(title, body, tags, answers):
    """C6_8: max_df=0.8"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_9(title, body, tags, answers):
    """C6_9: max_df=0.85"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_10(title, body, tags, answers):
    """C6_10: max_df=0.95"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_11(title, body, tags, answers):
    """C6_11: Combo1 (vocab=8000, min_df=3, max_df=0.85)"""
    return build_document_c3(title, body, tags, answers)


def build_document_c6_12(title, body, tags, answers):
    """C6_12: Combo2 (vocab=10000, min_df=3, max_df=0.8)"""
    return build_document_c3(title, body, tags, answers)


def build_document_c7(title, body, tags, answers):
    """C7: 综合最优 - C3 + 术语保留"""
    return build_document_c5(title, body, tags, answers)


# C8系列：C1参数调优实验（基于C1的文本构造：title + body）
def build_document_c8_1(title, body, tags, answers):
    """C8_1: C1 + vocab=3000"""
    return build_document_c1(title, body, tags, answers)


def build_document_c8_2(title, body, tags, answers):
    """C8_2: C1 + vocab=8000"""
    return build_document_c1(title, body, tags, answers)


def build_document_c8_3(title, body, tags, answers):
    """C8_3: C1 + min_df=3"""
    return build_document_c1(title, body, tags, answers)


def build_document_c8_4(title, body, tags, answers):
    """C8_4: C1 + min_df=7"""
    return build_document_c1(title, body, tags, answers)


def build_document_c8_5(title, body, tags, answers):
    """C8_5: C1 + max_df=0.8"""
    return build_document_c1(title, body, tags, answers)


def build_document_c8_6(title, body, tags, answers):
    """C8_6: C1 + 最优参数组合（vocab=3000, min_df=3, max_df=0.85）"""
    return build_document_c1(title, body, tags, answers)


def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    if v1 is None or v2 is None:
        return 0.0
    try:
        return float(v1.dot(v2))
    except Exception:
        return 0.0


def create_spark_session(app_name):
    """创建Spark会话"""
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, input_path, sample_ratio):
    """加载数据"""
    print("\n[Step 1] Load data")
    df = spark.read.json(input_path)
    total = df.count()
    print("Total records: %d" % total)

    if sample_ratio < 1.0:
        df = df.sample(False, sample_ratio, 42)
        print("Sampled records: %d" % df.count())

    return df


def preprocess_data(df, experiment_id):
    """预处理数据，根据实验ID选择不同的文本构造方式"""
    print("\n[Step 2] Build document text for experiment %s" % experiment_id)
    
    # 选择对应的文本构造函数
    build_functions = {
        "C1": build_document_c1,
        "C2": build_document_c2,
        "C3": build_document_c3,
        "C4": build_document_c4,
        "C5": build_document_c5,
        "C6_1": build_document_c6_1,
        "C6_2": build_document_c6_2,
        "C6_3": build_document_c6_3,
        "C6_4": build_document_c6_4,
        "C6_5": build_document_c6_5,
        "C6_6": build_document_c6_6,
        "C6_7": build_document_c6_7,
        "C6_8": build_document_c6_8,
        "C6_9": build_document_c6_9,
        "C6_10": build_document_c6_10,
        "C6_11": build_document_c6_11,
        "C6_12": build_document_c6_12,
        "C7": build_document_c7,
        "C8_1": build_document_c8_1,
        "C8_2": build_document_c8_2,
        "C8_3": build_document_c8_3,
        "C8_4": build_document_c8_4,
        "C8_5": build_document_c8_5,
        "C8_6": build_document_c8_6,
    }
    
    build_func = build_functions.get(experiment_id, build_document_c1)
    build_document_udf = udf(build_func, StringType())

    df = df.withColumn(
        "document",
        build_document_udf(col("title"), col("body"), col("tags"), col("answers")),
    )
    df = df.filter((col("document").isNotNull()) & (col("document") != ""))
    print("Valid records: %d" % df.count())
    return df


def build_pipeline(max_features, min_df, max_df=0.9):
    """构建TF-IDF特征提取Pipeline"""
    print("\n[Step 3] Build TF-IDF feature pipeline")
    print("max_features=%d, min_df=%d, max_df=%.2f" % (max_features, min_df, max_df))

    tokenizer = Tokenizer(inputCol="document", outputCol="raw_tokens")
    remover = StopWordsRemover(
        inputCol="raw_tokens",
        outputCol="tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + DOMAIN_STOP_WORDS,
    )
    vectorizer = CountVectorizer(
        inputCol="tokens",
        outputCol="term_features",
        vocabSize=max_features,
        minDF=min_df,
        maxDF=max_df,  # Spark 2.4+支持maxDF
    )
    idf = IDF(inputCol="term_features", outputCol="tfidf_features")
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)
    return Pipeline(stages=[tokenizer, remover, vectorizer, idf, normalizer])


def run_clustering(processed_df, k, max_iter):
    """运行K-Means聚类"""
    print("\n[Step 4] Run K-Means clustering")
    print("k=%d, max_iter=%d" % (k, max_iter))
    
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )
    
    model = kmeans.fit(processed_df)
    predictions = model.transform(processed_df)
    
    return model, predictions


def save_basic_outputs(predictions, output_path):
    """保存基本输出"""
    print("\n[Step 5] Save cluster outputs")

    predictions.select(
        col("question_id"),
        col("title"),
        col("tags"),
        col("score").alias("question_score"),
        col("cluster"),
        col("features").alias("prediction_features"),  # 保存特征用于后续指标计算
    ).write.mode("overwrite").json(output_path + "/cluster_assignments")

    predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy("cluster").write.mode("overwrite").json(output_path + "/cluster_summary")

    window = Window.partitionBy("cluster").orderBy(desc("score"))
    predictions.withColumn("rank", row_number().over(window)) \
        .filter(col("rank") <= 10) \
        .select("cluster", "rank", "question_id", "title", "score") \
        .orderBy("cluster", "rank") \
        .write.mode("overwrite").json(output_path + "/top_questions_per_cluster")

    print("Saved cluster_assignments, cluster_summary, top_questions_per_cluster")


def save_cluster_keywords(predictions, output_path, top_terms):
    """保存簇关键词"""
    print("\n[Step 6] Save cluster keywords")

    term_counts = predictions.select("cluster", explode(col("tokens")).alias("term")) \
        .filter(col("term") != "") \
        .groupBy("cluster", "term") \
        .agg(count("*").alias("term_count"))

    window = Window.partitionBy("cluster").orderBy(desc("term_count"))
    term_counts.withColumn("rank", row_number().over(window)) \
        .filter(col("rank") <= top_terms) \
        .groupBy("cluster") \
        .agg(collect_list("term").alias("keywords")) \
        .orderBy("cluster") \
        .write.mode("overwrite").json(output_path + "/cluster_keywords")

    print("Saved cluster_keywords")


def save_duplicate_candidates(predictions, output_path, top_per_cluster, threshold):
    """生成并保存重复候选对"""
    print("\n[Step 7] Generate duplicate candidates")
    print("top_per_cluster=%d, threshold=%.3f" % (top_per_cluster, threshold))

    window = Window.partitionBy("cluster").orderBy(desc("score"))
    candidates = predictions.withColumn("dedup_rank", row_number().over(window)) \
        .filter(col("dedup_rank") <= top_per_cluster) \
        .select("cluster", "question_id", "title", "score", "features")

    left = candidates.alias("a")
    right = candidates.alias("b")

    cosine_udf = udf(cosine_similarity, DoubleType())

    pairs = left.join(
        right,
        (col("a.cluster") == col("b.cluster")) & (col("a.question_id") < col("b.question_id")),
    ).withColumn(
        "similarity",
        cosine_udf(col("a.features"), col("b.features")),
    ).filter(
        col("similarity") >= threshold
    ).select(
        col("a.cluster").alias("cluster"),
        col("a.question_id").alias("question_id_1"),
        col("a.title").alias("title_1"),
        col("b.question_id").alias("question_id_2"),
        col("b.title").alias("title_2"),
        col("similarity"),
    ).orderBy(desc("similarity"))

    pairs = pairs.cache()
    dup_count = pairs.count()
    pairs.write.mode("overwrite").json(output_path + "/duplicate_candidates")
    print("Duplicate candidate pairs: %d" % dup_count)
    print("Saved duplicate_candidates")


def print_summary(predictions):
    """打印结果摘要"""
    print("\n[Step 8] Result summary")
    cluster_sizes = predictions.groupBy("cluster").agg(count("*").alias("cluster_size"))
    cluster_sizes.orderBy(desc("cluster_size")).show(20, truncate=False)

    stats = cluster_sizes.collect()
    sizes = [row["cluster_size"] for row in stats]
    if sizes:
        print("Total clusters: %d" % len(sizes))
        print("Max cluster size: %d" % max(sizes))
        print("Min cluster size: %d" % min(sizes))
        print("Average cluster size: %.2f" % (sum(sizes) / float(len(sizes))))


def compute_cluster_metrics(predictions, model, total_samples):
    """计算聚类质量指标"""
    print("\n[Step 9] Compute cluster quality metrics")
    
    # 计算WSSSE
    wssse = model.summary.trainingCost
    wssse_per_sample = wssse / total_samples
    print("WSSSE: %.2f" % wssse)
    print("WSSSE per sample: %.4f" % wssse_per_sample)
    
    # 计算簇分布统计
    cluster_sizes = predictions.groupBy("cluster").agg(count("*").alias("cluster_size"))
    stats = cluster_sizes.collect()
    sizes = [row["cluster_size"] for row in stats]
    
    max_cluster_size = max(sizes) if sizes else 0
    max_cluster_ratio = max_cluster_size / total_samples if total_samples > 0 else 0
    min_cluster_size = min(sizes) if sizes else 0
    avg_cluster_size = sum(sizes) / len(sizes) if sizes else 0
    
    # 计算簇大小CV（变异系数）
    if len(sizes) > 1 and avg_cluster_size > 0:
        variance = sum((x - avg_cluster_size) ** 2 for x in sizes) / len(sizes)
        std_dev = variance ** 0.5
        cluster_size_cv = std_dev / avg_cluster_size
    else:
        cluster_size_cv = 0
    
    print("Max cluster size: %d" % max_cluster_size)
    print("Max cluster ratio: %.4f" % max_cluster_ratio)
    print("Min cluster size: %d" % min_cluster_size)
    print("Average cluster size: %.2f" % avg_cluster_size)
    print("Cluster size CV: %.4f" % cluster_size_cv)
    
    return {
        "wssse": wssse,
        "wssse_per_sample": wssse_per_sample,
        "max_cluster_size": max_cluster_size,
        "max_cluster_ratio": max_cluster_ratio,
        "min_cluster_size": min_cluster_size,
        "avg_cluster_size": avg_cluster_size,
        "cluster_size_cv": cluster_size_cv,
        "num_clusters": len(sizes),
    }


def save_metrics(metrics, output_path):
    """保存指标到JSON文件"""
    print("\n[Step 10] Save metrics to JSON")
    
    import json
    metrics_json = json.dumps(metrics, indent=2)
    
    # 保存到HDFS
    spark = SparkSession.builder.getOrCreate()
    metrics_df = spark.createDataFrame([(metrics_json,)], ["metrics"])
    metrics_df.write.mode("overwrite").text(output_path + "/metrics")
    
    print("Saved metrics to: %s/metrics" % output_path)
    print("Metrics content:")
    print(metrics_json)


def get_experiment_config(experiment_id):
    """获取实验配置"""
    configs = {
        # C1-C4: 项目总结报告要求的基础实验
        "C1": {
            "name": "Baseline (title + body)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C2": {
            "name": "Title Weighting (title×2 + body)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C3": {
            "name": "Title+Tags Weighting (title×2 + body + tags×1)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C4": {
            "name": "Answer Fusion (title + body + tags + answers)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.9,
        },
        # C5: 术语保留优化
        "C5": {
            "name": "Term Preservation (C3 + Oracle errors + SQL keywords)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.9,
        },
        # C6系列：参数优化对照试验（覆盖max_features 3000-10000, min_df 3-10, max_df 0.8-0.95）
        "C6_1": {
            "name": "Param Opt - vocab=3000 (min_df=5, max_df=0.9)",
            "max_features": 3000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C6_2": {
            "name": "Param Opt - vocab=5000 (min_df=5, max_df=0.9)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C6_3": {
            "name": "Param Opt - vocab=8000 (min_df=5, max_df=0.9)",
            "max_features": 8000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C6_4": {
            "name": "Param Opt - vocab=10000 (min_df=5, max_df=0.9)",
            "max_features": 10000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C6_5": {
            "name": "Param Opt - min_df=3 (vocab=5000, max_df=0.9)",
            "max_features": 5000,
            "min_df": 3,
            "max_df": 0.9,
        },
        "C6_6": {
            "name": "Param Opt - min_df=7 (vocab=5000, max_df=0.9)",
            "max_features": 5000,
            "min_df": 7,
            "max_df": 0.9,
        },
        "C6_7": {
            "name": "Param Opt - min_df=10 (vocab=5000, max_df=0.9)",
            "max_features": 5000,
            "min_df": 10,
            "max_df": 0.9,
        },
        "C6_8": {
            "name": "Param Opt - max_df=0.8 (vocab=5000, min_df=5)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.8,
        },
        "C6_9": {
            "name": "Param Opt - max_df=0.85 (vocab=5000, min_df=5)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.85,
        },
        "C6_10": {
            "name": "Param Opt - max_df=0.95 (vocab=5000, min_df=5)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.95,
        },
        "C6_11": {
            "name": "Param Opt - Combo1 (vocab=8000, min_df=3, max_df=0.85)",
            "max_features": 8000,
            "min_df": 3,
            "max_df": 0.85,
        },
        "C6_12": {
            "name": "Param Opt - Combo2 (vocab=10000, min_df=3, max_df=0.8)",
            "max_features": 10000,
            "min_df": 3,
            "max_df": 0.8,
        },
        # C7: 综合最优方案
        "C7": {
            "name": "Best Combination (C3 + C5 + Best C6)",
            "max_features": 8000,  # 根据C6结果选择最优
            "min_df": 3,
            "max_df": 0.85,
        },
        # C8系列：C1参数调优实验（验证参数对基线方案的影响）
        "C8_1": {
            "name": "C1 Param Opt - vocab=3000 (min_df=5, max_df=0.9)",
            "max_features": 3000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C8_2": {
            "name": "C1 Param Opt - vocab=8000 (min_df=5, max_df=0.9)",
            "max_features": 8000,
            "min_df": 5,
            "max_df": 0.9,
        },
        "C8_3": {
            "name": "C1 Param Opt - min_df=3 (vocab=5000, max_df=0.9)",
            "max_features": 5000,
            "min_df": 3,
            "max_df": 0.9,
        },
        "C8_4": {
            "name": "C1 Param Opt - min_df=7 (vocab=5000, max_df=0.9)",
            "max_features": 5000,
            "min_df": 7,
            "max_df": 0.9,
        },
        "C8_5": {
            "name": "C1 Param Opt - max_df=0.8 (vocab=5000, min_df=5)",
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.8,
        },
        "C8_6": {
            "name": "C1 Param Opt - Best Combo (vocab=3000, min_df=3, max_df=0.85)",
            "max_features": 3000,
            "min_df": 3,
            "max_df": 0.85,
        },
    }
    return configs.get(experiment_id, configs["C1"])


def main():
    parser = argparse.ArgumentParser(description="C Member: TF-IDF and Preprocessing Optimization")
    parser.add_argument("--input", required=True, help="Input JSONLines path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--experiment", required=True, 
                        choices=["C1", "C2", "C3", "C4", "C5", 
                                 "C6_1", "C6_2", "C6_3", "C6_4", "C6_5", "C6_6", "C6_7", "C6_8", "C6_9", "C6_10", "C6_11", "C6_12",
                                 "C7",
                                 "C8_1", "C8_2", "C8_3", "C8_4", "C8_5", "C8_6"],
                        help="Experiment ID (C1-C5, C6_1-C6_12, C7, C8_1-C8_6)")
    parser.add_argument("--k", type=int, default=45, help="Number of clusters (default: 45, from B's result)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Sampling ratio (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=30, help="KMeans max iterations")
    parser.add_argument("--top-terms", type=int, default=10, help="Top keyword count per cluster")
    parser.add_argument("--dedup-top-per-cluster", type=int, default=80, 
                        help="Max scored questions per cluster for pair search")
    parser.add_argument("--similarity-threshold", type=float, default=0.55, 
                        help="Cosine threshold for duplicate candidates")
    
    args = parser.parse_args()
    
    # 获取实验配置
    config = get_experiment_config(args.experiment)
    
    start = time.time()
    print("=" * 70)
    print("C Member Experiment: %s" % args.experiment)
    print("Description: %s" % config["name"])
    print("=" * 70)
    print("Input: %s" % args.input)
    print("Output: %s" % args.output)
    print("K: %d" % args.k)
    print("max_features: %d" % config["max_features"])
    print("min_df: %d" % config["min_df"])
    print("max_df: %.2f" % config["max_df"])
    print("Sample ratio: %.3f" % args.sample_ratio)
    print("=" * 70)

    spark = create_spark_session("C_Feature_Optimization_%s" % args.experiment)
    
    try:
        # Step 1: Load data
        df = load_data(spark, args.input, args.sample_ratio)
        total_samples = df.count()
        
        # Step 2: Preprocess with experiment-specific text construction
        df = preprocess_data(df, args.experiment)
        
        # Step 3: Build pipeline with experiment-specific parameters
        pipeline = build_pipeline(config["max_features"], config["min_df"], config["max_df"])
        pipeline_model = pipeline.fit(df)
        processed_df = pipeline_model.transform(df).cache()
        valid_samples = processed_df.count()
        print("Feature pipeline fitted. Records: %d" % valid_samples)
        
        # 获取实际词汇数
        cv_model = pipeline_model.stages[2]  # CountVectorizer model
        # Spark 2.0兼容：vocabSize是属性，不是方法
        try:
            # 尝试直接获取属性值
            actual_vocab_size = cv_model.vocabSize
            # 如果返回的是Param对象，获取其值
            if hasattr(actual_vocab_size, 'parent'):
                actual_vocab_size = cv_model._java_obj.getVocabSize()
        except:
            # 如果失败，使用Java对象方法
            actual_vocab_size = cv_model._java_obj.getVocabSize()
        print("Actual vocabulary size: %d" % actual_vocab_size)
        
        # Step 4: Run clustering
        kmeans_model, predictions = run_clustering(processed_df, args.k, args.max_iter)
        predictions = predictions.cache()
        print("Predictions generated. Records: %d" % predictions.count())
        
        # Step 5-7: Save outputs
        save_basic_outputs(predictions, args.output)
        save_cluster_keywords(predictions, args.output, args.top_terms)
        save_duplicate_candidates(
            predictions,
            args.output,
            args.dedup_top_per_cluster,
            args.similarity_threshold,
        )
        
        # Step 8: Print summary
        print_summary(predictions)
        
        # Step 9: Compute cluster quality metrics
        cluster_metrics = compute_cluster_metrics(predictions, kmeans_model, valid_samples)
        
        # Step 10: Save all metrics
        elapsed = time.time() - start
        
        # Python 2兼容的字典合并
        all_metrics = {
            "experiment_id": args.experiment,
            "experiment_name": config["name"],
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "k": args.k,
            "max_features": config["max_features"],
            "min_df": config["min_df"],
            "max_df": config["max_df"],
            "actual_vocab_size": actual_vocab_size,
            "max_iter": args.max_iter,
            "similarity_threshold": args.similarity_threshold,
            "dedup_top_per_cluster": args.dedup_top_per_cluster,
            "total_time_seconds": elapsed,
        }
        # 合并cluster_metrics（Python 2兼容）
        for key, value in cluster_metrics.items():
            all_metrics[key] = value
        
        save_metrics(all_metrics, args.output)
        
        print("\n" + "=" * 70)
        print("Experiment %s completed successfully!" % args.experiment)
        print("Total time: %.2fs" % elapsed)
        print("Results saved to: %s" % args.output)
        print("=" * 70)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()