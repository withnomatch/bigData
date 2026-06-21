#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C组最优方案：C1基线方案
特征构造：title + body
TF-IDF参数：vocab=5000, min_df=5, max_df=0.9
聚类参数：K=45
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, concat, lit
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.evaluation import ClusteringEvaluator
import argparse
import time
import re
import json

# ============================================================================
# 数据预处理函数
# ============================================================================

def clean_html(text):
    """清洗HTML标签"""
    if text is None:
        return ""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text):
    """文本规范化"""
    if text is None:
        return ""
    # 转小写
    text = text.lower()
    # 移除特殊字符（保留字母、数字、空格）
    text = re.sub(r'[^\w\s]', ' ', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_document_c1(title, body, tags, answers):
    """C1基线方案：title + body"""
    title_clean = clean_html(title) if title else ""
    body_clean = clean_html(body) if body else ""
    
    # 标题和正文拼接
    document = title_clean + " " + body_clean
    return normalize_text(document)

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='C组最优方案：C1基线')
    parser.add_argument('--input', required=True, help='输入数据路径')
    parser.add_argument('--output', required=True, help='输出目录路径')
    parser.add_argument('--annotation-pairs', required=True, help='标注数据路径')
    parser.add_argument('--k', type=int, default=45, help='聚类簇数')
    parser.add_argument('--max-features', type=int, default=5000, help='词汇表大小')
    parser.add_argument('--min-df', type=int, default=5, help='最小文档频率')
    parser.add_argument('--max-df', type=float, default=0.9, help='最大文档频率')
    parser.add_argument('--similarity-threshold', type=float, default=0.55, help='相似度阈值')
    parser.add_argument('--dedup-top-per-cluster', type=int, default=80, help='每簇候选数量上限')
    
    args = parser.parse_args()
    
    # 创建SparkSession
    spark = SparkSession.builder \
        .appName("C_Best_Solution_C1") \
        .getOrCreate()
    
    print("=" * 80)
    print("C组最优方案：C1基线方案")
    print("=" * 80)
    print("特征构造：title + body")
    print("TF-IDF参数：vocab=%d, min_df=%d, max_df=%.2f" % (args.max_features, args.min_df, args.max_df))
    print("聚类参数：K=%d" % args.k)
    print("=" * 80)
    
    start_time = time.time()
    
    # ============================================================================
    # Step 1: 加载数据
    # ============================================================================
    print("\n[Step 1] Loading data from: %s" % args.input)
    
    df = spark.read.json(args.input)
    print("Loaded %d records" % df.count())
    
    # ============================================================================
    # Step 2: 构建文档文本
    # ============================================================================
    print("\n[Step 2] Building document text (title + body)")
    
    # 注册UDF
    spark.udf.register("build_document_c1", build_document_c1, StringType())
    
    # 构建文档
    df = df.withColumn(
        "document",
        udf(build_document_c1, StringType())(
            col("title"), col("body"), col("tags"), col("answers")
        )
    )
    
    # 过滤空文档
    df = df.filter(col("document").isNotNull()).filter(col("document") != "")
    print("Valid records: %d" % df.count())
    
    # ============================================================================
    # Step 3: 分词
    # ============================================================================
    print("\n[Step 3] Tokenizing documents")
    
    tokenizer = Tokenizer(inputCol="document", outputCol="words")
    df = tokenizer.transform(df)
    
    # ============================================================================
    # Step 4: TF-IDF特征提取
    # ============================================================================
    print("\n[Step 4] Extracting TF-IDF features")
    print("Parameters: vocab_size=%d, min_df=%d, max_df=%.2f" % (args.max_features, args.min_df, args.max_df))
    
    feature_start = time.time()
    
    # 使用CountVectorizer（支持min_df和max_df）
    cv = CountVectorizer(
        inputCol="words",
        outputCol="tf_features",
        vocabSize=args.max_features,
        minDF=args.min_df,
        maxDF=args.max_df
    )
    cv_model = cv.fit(df)
    df = cv_model.transform(df)
    
    # 计算TF-IDF
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
    idf_model = idf.fit(df)
    df = idf_model.transform(df)
    
    feature_time = time.time() - feature_start
    print("Feature extraction time: %.2f seconds" % feature_time)
    
    # 获取实际词汇表大小
    vocab_size = cv_model._java_obj.getVocabSize()
    print("Actual vocabulary size: %d" % vocab_size)
    
    # ============================================================================
    # Step 5: KMeans聚类
    # ============================================================================
    print("\n[Step 5] KMeans clustering with K=%d" % args.k)
    
    cluster_start = time.time()
    
    kmeans = KMeans(
        featuresCol="tfidf_features",
        predictionCol="cluster_id",
        k=args.k,
        seed=42,
        maxIter=30,
        initMode="k-means||"
    )
    kmeans_model = kmeans.fit(df)
    df = kmeans_model.transform(df)
    
    cluster_time = time.time() - cluster_start
    print("KMeans training time: %.2f seconds" % cluster_time)
    
    # ============================================================================
    # Step 6: 计算聚类指标
    # ============================================================================
    print("\n[Step 6] Computing cluster metrics")
    
    # 计算WSSSE
    wssse = kmeans_model.computeCost(df)
    wssse_per_sample = wssse / df.count()
    print("WSSSE: %.2f" % wssse)
    print("WSSSE per sample: %.4f" % wssse_per_sample)
    
    # 计算轮廓系数
    evaluator = ClusteringEvaluator(
        featuresCol="tfidf_features",
        predictionCol="cluster_id",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean"
    )
    silhouette = evaluator.evaluate(df)
    print("Silhouette coefficient: %.4f" % silhouette)
    
    # 统计簇分布
    cluster_stats = df.groupBy("cluster_id").count().collect()
    cluster_sizes = [row["count"] for row in cluster_stats]
    max_cluster_size = max(cluster_sizes)
    min_cluster_size = min(cluster_sizes)
    avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
    max_cluster_ratio = max_cluster_size / df.count()
    
    # 计算簇大小CV（变异系数）
    import numpy as np
    cluster_size_std = np.std(cluster_sizes)
    cluster_size_cv = cluster_size_std / avg_cluster_size
    
    print("Max cluster size: %d (%.2f%%)" % (max_cluster_size, max_cluster_ratio * 100))
    print("Min cluster size: %d" % min_cluster_size)
    print("Avg cluster size: %.2f" % avg_cluster_size)
    print("Cluster size CV: %.4f" % cluster_size_cv)
    
    # ============================================================================
    # Step 7: 生成重复候选
    # ============================================================================
    print("\n[Step 7] Generating duplicate candidates")
    print("Parameters: similarity_threshold=%.2f, dedup_top=%d" % (args.similarity_threshold, args.dedup_top_per_cluster))
    
    dedup_start = time.time()
    
    # 按簇分组
    clusters = df.groupBy("cluster_id").collect()
    
    duplicate_candidates = []
    
    for cluster_row in clusters:
        cluster_id = cluster_row["cluster_id"]
        
        # 获取簇内所有问题
        cluster_questions = df.filter(col("cluster_id") == cluster_id).select(
            "question_id", "title", "tfidf_features"
        ).collect()
        
        # 如果簇太大，只处理前dedup_top_per_cluster个问题
        if len(cluster_questions) > args.dedup_top_per_cluster:
            cluster_questions = cluster_questions[:args.dedup_top_per_cluster]
        
        # 计算簇内问题对的相似度
        for i in range(len(cluster_questions)):
            for j in range(i + 1, len(cluster_questions)):
                q1 = cluster_questions[i]
                q2 = cluster_questions[j]
                
                # 计算余弦相似度
                vec1 = q1["tfidf_features"]
                vec2 = q2["tfidf_features"]
                
                # 余弦相似度 = dot(v1, v2) / (norm(v1) * norm(v2))
                dot_product = float(vec1.dot(vec2))
                norm1 = float(vec1.norm(2))
                norm2 = float(vec2.norm(2))
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    
                    # 如果相似度高于阈值，判定为疑似重复
                    if similarity >= args.similarity_threshold:
                        duplicate_candidates.append({
                            "question_id_1": q1["question_id"],
                            "question_id_2": q2["question_id"],
                            "title_1": q1["title"],
                            "title_2": q2["title"],
                            "cluster_id": cluster_id,
                            "similarity": similarity
                        })
    
    dedup_time = time.time() - dedup_start
    print("Duplicate candidates generated: %d pairs" % len(duplicate_candidates))
    print("Dedup generation time: %.2f seconds" % dedup_time)
    
    # ============================================================================
    # Step 8: 保存结果
    # ============================================================================
    print("\n[Step 8] Saving results to: %s" % args.output)
    
    # 保存聚类分配
    cluster_assignments = df.select("question_id", "title", "cluster_id")
    cluster_assignments.write.mode("overwrite").parquet(args.output + "/cluster_assignments")
    
    # 保存重复候选
    if len(duplicate_candidates) > 0:
        candidates_df = spark.createDataFrame(duplicate_candidates)
        candidates_df.write.mode("overwrite").parquet(args.output + "/duplicate_candidates")
    
    # 保存聚类摘要
    cluster_summary = df.groupBy("cluster_id").count().orderBy("cluster_id")
    cluster_summary.write.mode("overwrite").parquet(args.output + "/cluster_summary")
    
    # ============================================================================
    # Step 9: 评测去重效果
    # ============================================================================
    print("\n[Step 9] Evaluating dedup performance with 300 annotation pairs")
    
    # 加载标注数据
    annotation_df = spark.read.csv(args.annotation_pairs, header=True, inferSchema=True)
    print("Loaded %d annotation pairs" % annotation_df.count())
    
    # 统计评测指标
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    candidate_recall_count = 0
    
    annotation_pairs = annotation_df.collect()
    
    for pair in annotation_pairs:
        q1_id = pair["question_id_1"]
        q2_id = pair["question_id_2"]
        is_duplicate = pair["is_duplicate"]
        
        # 查询聚类分配
        q1_cluster = df.filter(col("question_id") == q1_id).select("cluster_id").collect()
        q2_cluster = df.filter(col("question_id") == q2_id).select("cluster_id").collect()
        
        if len(q1_cluster) > 0 and len(q2_cluster) > 0:
            q1_cluster_id = q1_cluster[0]["cluster_id"]
            q2_cluster_id = q2_cluster[0]["cluster_id"]
            
            # 判断是否同簇
            same_cluster = (q1_cluster_id == q2_cluster_id)
            
            # 判断是否在重复候选中
            in_candidates = False
            for candidate in duplicate_candidates:
                if (candidate["question_id_1"] == q1_id and candidate["question_id_2"] == q2_id) or \
                   (candidate["question_id_1"] == q2_id and candidate["question_id_2"] == q1_id):
                    in_candidates = True
                    break
            
            # 统计指标
            if is_duplicate == 1:  # 真实重复
                if same_cluster:
                    candidate_recall_count += 1
                if in_candidates:
                    tp += 1
                else:
                    fn += 1
            else:  # 不重复
                if in_candidates:
                    fp += 1
                else:
                    tn += 1
    
    # 计算评测指标
    total_duplicates = sum([1 for pair in annotation_pairs if pair["is_duplicate"] == 1])
    candidate_recall = candidate_recall_count / total_duplicates if total_duplicates > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\nEvaluation Results:")
    print("TP: %d" % tp)
    print("FP: %d" % fp)
    print("FN: %d" % fn)
    print("TN: %d" % tn)
    print("Candidate Recall: %.4f" % candidate_recall)
    print("Precision: %.4f" % precision)
    print("Recall: %.4f" % recall)
    print("F1: %.4f" % f1)
    
    # ============================================================================
    # Step 10: 保存实验报告
    # ============================================================================
    print("\n[Step 10] Saving experiment report")
    
    total_time = time.time() - start_time
    
    report = {
        "experiment_id": "C1_Best_Solution",
        "feature_construction": "title + body",
        "tfidf_params": {
            "vocab_size": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "actual_vocab_size": vocab_size
        },
        "cluster_params": {
            "k": args.k,
            "max_iter": 30,
            "init_mode": "k-means||"
        },
        "dedup_params": {
            "similarity_threshold": args.similarity_threshold,
            "dedup_top_per_cluster": args.dedup_top_per_cluster
        },
        "cluster_metrics": {
            "wssse": wssse,
            "wssse_per_sample": wssse_per_sample,
            "silhouette": silhouette,
            "max_cluster_size": max_cluster_size,
            "max_cluster_ratio": max_cluster_ratio,
            "min_cluster_size": min_cluster_size,
            "avg_cluster_size": avg_cluster_size,
            "cluster_size_cv": cluster_size_cv
        },
        "dedup_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "candidate_recall": candidate_recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "duplicate_candidates_count": len(duplicate_candidates)
        },
        "time_metrics": {
            "feature_extraction_time": feature_time,
            "kmeans_training_time": cluster_time,
            "dedup_generation_time": dedup_time,
            "total_time": total_time
        },
        "data_stats": {
            "input_records": df.count(),
            "valid_records": df.count()
        }
    }
    
    # 保存JSON报告
    import json
    report_json = json.dumps(report, indent=2)
    
    # 使用HDFS保存报告
    report_path = args.output + "/experiment_report.json"
    spark.sparkContext.parallelize([report_json]).saveAsTextFile(report_path)
    
    print("\nExperiment report saved to: %s" % report_path)
    
    # ============================================================================
    # 完成
    # ============================================================================
    print("\n" + "=" * 80)
    print("C组最优方案（C1基线）实验完成！")
    print("=" * 80)
    print("总运行时间: %.2f 秒" % total_time)
    print("Candidate Recall: %.4f" % candidate_recall)
    print("F1: %.4f" % f1)
    print("=" * 80)
    
    spark.stop()

if __name__ == "__main__":
    main()