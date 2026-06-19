# -*- coding: utf-8 -*-
"""
StackOverflow Question Clustering - Feature Extraction Optimization
Compatible with Python 2.7
"""

from __future__ import print_function
import re
import argparse
import time
import sys
import json

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding('utf-8')

try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array_distinct,
    col,
    collect_list,
    count,
    desc,
    explode,
    first,
    row_number,
    split,
    udf,
)
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# Oracle数据库专业术语列表
ORACLE_DOMAIN_TERMS = [
    'oracle', 'oracle-database', 'plsql', 'sql', 'sqlplus',
    'jdbc', 'oci', 'odbc', 'tns', 'rman', 'asm', 'rac',
    'dataguard', 'goldengate', 'exadata', 'apex', 'forms',
    'reports', 'dbms', 'schema', 'tablespace', 'segment',
    'extent', 'block', 'row', 'column', 'index', 'constraint',
    'trigger', 'procedure', 'function', 'package', 'cursor',
    'exception', 'lob', 'blob', 'clob', 'bfile', 'xmltype',
    'partition', 'subpartition', 'materialized', 'view', 'synonym',
    'sequence', 'grant', 'revoke', 'audit', 'flashback',
    'undo', 'redo', 'archive', 'controlfile', 'spfile', 'pfile',
    'listener', 'service_name', 'sid', 'instance', 'session',
    'process', 'thread', 'pga', 'sga', 'buffer_cache', 'shared_pool',
    'large_pool', 'java_pool', 'streams_pool', 'log_buffer',
    'optimizer', 'hint', 'execution_plan', 'statistics', 'histogram',
    'bind', 'peeking', 'adaptive', 'cardinality', 'cost',
    'parallel', 'dml', 'ddl', 'dcl', 'tcl', 'commit', 'rollback',
    'savepoint', 'isolation', 'level', 'lock', 'deadlock',
    'wait', 'event', 'latch', 'mutex', 'enqueue',
    'awr', 'ash', 'addm', 'statspack', 'trace', 'tkprof',
    'sqlt', 'sqlhc', 'sqlpa', 'spa', 'spm', 'baseline',
    'profile', 'patch', 'hint', 'sql_profile', 'sql_patch',
    'sql_plan_baseline', 'sql_patch', 'sql_profile',
]

# 定制停用词列表（扩展标准停用词）
CUSTOM_STOPWORDS = [
    # 标准停用词
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    # StackOverflow特定停用词
    'question', 'answer', 'post', 'comment', 'user', 'stackoverflow',
    'thanks', 'thank', 'please', 'help', 'hi', 'hello', 'ok', 'okay',
    'yes', 'no', 'maybe', 'like', 'want', 'need', 'try', 'tried',
    'trying', 'use', 'used', 'using', 'get', 'got', 'getting',
    'work', 'worked', 'working', 'works', 'fix', 'fixed', 'fixing',
    'issue', 'issues', 'problem', 'problems', 'error', 'errors',
    'code', 'example', 'sample', 'test', 'testing', 'tested',
    'file', 'files', 'way', 'ways', 'method', 'methods', 'approach',
    'approaches', 'solution', 'solutions', 'result', 'results',
    'output', 'outputs', 'input', 'inputs', 'value', 'values',
    'data', 'information', 'info', 'details', 'detail', 'description',
    'following', 'below', 'above', 'here', 'there', 'where', 'when',
    'could', 'would', 'should', 'might', 'may', 'must', 'shall',
    'cannot', 'can', 'able', 'unable', 'possible', 'impossible',
    'better', 'best', 'good', 'bad', 'right', 'wrong', 'correct',
    'incorrect', 'true', 'false', 'null', 'empty', 'full',
]


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        return " ".join(self.result)


def clean_html(html_str):
    """清洗HTML标签"""
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


def simple_lemmatize(word):
    """简单的词形还原（基于规则）"""
    # 复数形式
    if word.endswith('ies'):
        return word[:-3] + 'y'
    elif word.endswith('es'):
        return word[:-2]
    elif word.endswith('s') and len(word) > 3:
        return word[:-1]
    
    # 过去式
    if word.endswith('ed'):
        if len(word) > 4 and word[-3] == word[-4]:
            return word[:-3]  # 如 stopped -> stop
        return word[:-2]
    
    # 进行时
    if word.endswith('ing'):
        if len(word) > 5 and word[-4] == word[-5]:
            return word[:-4]  # 如 running -> run
        return word[:-3]
    
    # 其他常见变化
    if word.endswith('ly'):
        return word[:-2]
    
    return word


def preprocess_text_simplified(title, body):
    # 1. HTML清洗
    title_clean = clean_html(title) if title else ""
    body_clean = clean_html(body) if body else ""
    
    # 2. 合并标题和正文
    combined = title_clean + " " + body_clean
    
    # 3. 转小写
    combined = combined.lower()
    
    # 4. 移除特殊字符
    combined = re.sub(r"[^a-zA-Z0-9\s]", " ", combined)
    combined = re.sub(r"\s+", " ", combined).strip()
    
    # 5. 过滤短词（长度小于3）
    words = combined.split()
    filtered_words = [word for word in words if len(word) > 2]
    
    return " ".join(filtered_words)


def create_spark_session(app_name="StackOverflow_Clustering_Optimized"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark, input_path, sample_ratio=1.0):
    print("\n" + "=" * 60)
    print("[Step 1] Loading Data")
    print("=" * 60)
    
    df = spark.read.json(input_path)
    total_count = df.count()
    print("Total records: %d" % total_count)
    
    if sample_ratio < 1.0:
        df = df.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        sampled_count = df.count()
        print("Sampled records: %d (ratio: %f)" % (sampled_count, sample_ratio))
    
    return df


def preprocess_data_optimized(df):
    print("\n" + "=" * 60)
    print("[Step 2] Simplified Text Preprocessing")
    print("=" * 60)
    print("Steps: HTML cleaning -> Lowercase -> Remove special chars -> Filter short words")
    print("Note: Custom stopwords will be removed in Pipeline (Spark built-in function)")
    
    preprocess_udf = udf(preprocess_text_simplified, StringType())
    
    df = df.withColumn("clean_text", preprocess_udf(col("title"), col("body")))
    df = df.filter((col("clean_text").isNotNull()) & (col("clean_text") != ""))
    
    valid_count = df.count()
    print("Valid records: %d" % valid_count)
    
    return df


def find_high_frequency_terms(df, max_df):
    """Return terms present in more than max_df of documents."""
    total_documents = df.count()
    threshold = float(total_documents) * max_df
    rows = (
        df.select(explode(array_distinct(split(col("clean_text"), r"\s+"))).alias("term"))
        .filter(col("term") != "")
        .groupBy("term")
        .count()
        .filter(col("count") > threshold)
        .collect()
    )
    terms = [row["term"] for row in rows]
    print(
        "maxDF=%.2f threshold=%d documents, filtered high-frequency terms=%d"
        % (max_df, threshold, len(terms))
    )
    return terms


def build_optimized_pipeline(
    max_features=5000, min_df=5, max_df=0.9, high_frequency_terms=None
):
    """构建优化的特征提取Pipeline"""
    print("\n" + "=" * 60)
    print("[Step 3] Building Optimized Feature Extraction Pipeline")
    print("=" * 60)
    print("Pipeline: Tokenizer -> StopWordsRemover -> CountVectorizer -> IDF -> Normalizer")
    print("Params: max_features=%d, min_df=%d, max_df=%.2f" % (max_features, min_df, max_df))
    
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="raw_tokens")
    
    # Spark CountVectorizer has no maxDF parameter. Terms above maxDF are
    # computed from document frequency and removed before vectorization.
    stop_words = list(CUSTOM_STOPWORDS)
    if high_frequency_terms:
        stop_words.extend(high_frequency_terms)

    remover = StopWordsRemover(
        inputCol="raw_tokens",
        outputCol="filtered_tokens",
        stopWords=sorted(set(stop_words)),
    )
    
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=max_features,
        minDF=min_df,
    )
    
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    
    normalizer = Normalizer(inputCol="tfidf_features", outputCol="features", p=2.0)
    
    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, normalizer])
    
    return pipeline


def fit_and_transform(pipeline, df):
    print("\nFitting Pipeline...")
    start_time = time.time()
    
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)
    
    elapsed = time.time() - start_time
    print("Pipeline fitting complete, time: %.2fs" % elapsed)
    
    vocab_size = len(pipeline_model.stages[2].vocabulary)
    print("Actual vocabulary size: %d" % vocab_size)
    
    return pipeline_model, processed_df, elapsed, vocab_size


def run_kmeans(processed_df, k=50, max_iter=30):
    print("\n" + "=" * 60)
    print("[Step 4] K-Means Clustering")
    print("=" * 60)
    print("Params: K=%d, max_iter=%d" % (k, max_iter))
    
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        maxIter=max_iter,
        seed=42,
    )
    
    print("Training K-Means model...")
    start_time = time.time()
    
    kmeans_model = kmeans.fit(processed_df)
    predictions = kmeans_model.transform(processed_df)
    
    elapsed = time.time() - start_time
    print("K-Means training complete, time: %.2fs" % elapsed)
    
    return kmeans_model, predictions, elapsed


def evaluate_clustering(kmeans_model, predictions):
    """评估聚类质量"""
    print("\n" + "=" * 60)
    print("[Step 5] Evaluating Clustering Quality")
    print("=" * 60)
    
    cluster_sizes = predictions.groupBy("cluster").count().orderBy("cluster")
    
    cluster_count = predictions.select("cluster").distinct().count()
    total_points = predictions.count()
    avg_cluster_size = float(total_points) / cluster_count
    
    # 计算聚类大小分布统计
    sizes_data = cluster_sizes.collect()
    sizes = [row["count"] for row in sizes_data]
    
    max_size = max(sizes)
    min_size = min(sizes)
    std_size = (sum([(s - avg_cluster_size) ** 2 for s in sizes]) / len(sizes)) ** 0.5
    
    print("Total Clusters: %d" % cluster_count)
    print("Total Points: %d" % total_points)
    print("Average Cluster Size: %.2f" % avg_cluster_size)
    print("Max Cluster Size: %d" % max_size)
    print("Min Cluster Size: %d" % min_size)
    print("Std Deviation: %.2f" % std_size)
    
    evaluator = ClusteringEvaluator(
        predictionCol="cluster",
        featuresCol="features",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    silhouette = evaluator.evaluate(predictions)
    wssse = kmeans_model.computeCost(predictions)
    largest_ratio = max_size / float(total_points)
    cluster_size_cv = std_size / avg_cluster_size if avg_cluster_size > 0 else 0.0

    print("Silhouette Score: %.6f" % silhouette)
    print("WSSSE: %.6f" % wssse)
    print("Largest Cluster Ratio: %.6f" % largest_ratio)
    print("Cluster Size CV: %.6f" % cluster_size_cv)

    return {
        "silhouette": silhouette,
        "wssse": wssse,
        "total_points": total_points,
        "cluster_count": cluster_count,
        "avg_cluster_size": avg_cluster_size,
        "max_cluster_size": max_size,
        "min_cluster_size": min_size,
        "std_cluster_size": std_size,
        "cluster_size_cv": cluster_size_cv,
        "largest_cluster_ratio": largest_ratio,
    }


def save_results(predictions, output_path):
    print("\n" + "=" * 60)
    print("[Step 6] Saving Results")
    print("=" * 60)
    
    output_df = predictions.select(
        col("question_id"),
        col("title"),
        col("cluster"),
        col("score").alias("question_score"),
    )
    output_df.write.mode("overwrite").json(output_path + "/cluster_assignments")
    print("Cluster assignments saved: %s/cluster_assignments" % output_path)
    
    cluster_summary = predictions.groupBy("cluster").agg(
        count("*").alias("cluster_size"),
    ).orderBy("cluster")
    cluster_summary.write.mode("overwrite").json(output_path + "/cluster_summary")
    print("Cluster summary saved: %s/cluster_summary" % output_path)


def run_single_experiment(df, params, k=50, max_iter=30, high_frequency_terms=None):
    """运行单个参数组合的实验"""
    max_features = params['max_features']
    min_df = params['min_df']
    max_df = params['max_df']
    
    print("\n" + "=" * 60)
    print("Experiment: vocabSize=%d, minDF=%d, maxDF=%.2f" % (max_features, min_df, max_df))
    print("=" * 60)
    
    start_time = time.time()
    
    # 构建Pipeline
    if high_frequency_terms is None:
        high_frequency_terms = find_high_frequency_terms(df, max_df)
    pipeline = build_optimized_pipeline(
        max_features, min_df, max_df, high_frequency_terms
    )
    
    # 训练Pipeline
    pipeline_model, processed_df, pipeline_time, vocab_size = fit_and_transform(pipeline, df)
    
    # K-Means聚类
    processed_df = processed_df.cache()
    processed_df.count()
    kmeans_model, predictions, kmeans_time = run_kmeans(
        processed_df, k=k, max_iter=max_iter
    )
    
    # 评估
    metrics = evaluate_clustering(kmeans_model, predictions)
    
    total_time = time.time() - start_time
    
    result = {
        'params': params,
        'vocab_size': vocab_size,
        'pipeline_time': pipeline_time,
        'kmeans_time': kmeans_time,
        'total_time': total_time,
        'silhouette': metrics["silhouette"],
        'wssse': metrics["wssse"],
        'total_points': metrics["total_points"],
        'cluster_count': metrics["cluster_count"],
        'avg_cluster_size': metrics["avg_cluster_size"],
        'max_cluster_size': metrics["max_cluster_size"],
        'min_cluster_size': metrics["min_cluster_size"],
        'std_cluster_size': metrics["std_cluster_size"],
        'cluster_size_cv': metrics["cluster_size_cv"],
        'largest_cluster_ratio': metrics["largest_cluster_ratio"],
    }
    
    print("\nExperiment Result:")
    print("  Vocabulary Size: %d" % vocab_size)
    print("  Pipeline Time: %.2fs" % pipeline_time)
    print("  K-Means Time: %.2fs" % kmeans_time)
    print("  Total Time: %.2fs" % total_time)
    print("  Silhouette: %.6f" % metrics["silhouette"])
    print("  WSSSE: %.6f" % metrics["wssse"])
    print("  Avg Cluster Size: %.2f" % metrics["avg_cluster_size"])
    print("  Std Cluster Size: %.2f" % metrics["std_cluster_size"])

    processed_df.unpersist()
    return result, predictions


def generate_param_grid():
    """生成参数网格"""
    vocab_sizes = [3000, 5000, 7000, 10000]
    min_dfs = [3, 5, 7, 10]
    max_dfs = [0.8, 0.85, 0.9, 0.95]
    
    param_grid = []
    for vocab in vocab_sizes:
        for min_df in min_dfs:
            for max_df in max_dfs:
                param_grid.append({
                    'max_features': vocab,
                    'min_df': min_df,
                    'max_df': max_df,
                })
    
    return param_grid


def save_comparison_results(results, output_path):
    """保存对比实验结果"""
    print("\n" + "=" * 60)
    print("Saving Comparison Results")
    print("=" * 60)
    
    result_lines = [json.dumps(result, sort_keys=True) for result in results]

    if output_path.startswith('hdfs://'):
        hdfs_path = output_path + "/comparison_results"
        spark = SparkSession.builder.getOrCreate()
        spark.createDataFrame(
            [(line,) for line in result_lines], ["value"]
        ).coalesce(1).write.mode("overwrite").text(hdfs_path)
        print("Comparison results saved to HDFS: %s" % hdfs_path)
    else:
        import os
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        final_file = output_path + "/comparison_results.json"
        with open(final_file, "w") as handle:
            handle.write("\n".join(result_lines))
        print("Comparison results saved: %s" % final_file)
    
    # 打印结果表格
    print("\n" + "=" * 80)
    print("Comparison Results Summary")
    print("=" * 80)
    print("%-15s %-10s %-10s %-12s %-12s %-12s %-12s" % (
        "vocabSize", "minDF", "maxDF", "vocab", "time", "silhouette", "avg_size"
    ))
    print("-" * 80)
    
    for result in results:
        params = result['params']
        print("%-15d %-10d %-10.2f %-12d %-12.2f %-12.4f %-12.2f" % (
            params['max_features'],
            params['min_df'],
            params['max_df'],
            result['vocab_size'],
            result['total_time'],
            result['silhouette'],
            result['avg_cluster_size'],
        ))
    
    # 找出最佳参数组合
    best_result = max(results, key=lambda x: x['silhouette'])
    print("\n" + "=" * 80)
    print("Best Parameter Combination:")
    print("=" * 80)
    print("vocabSize: %d" % best_result['params']['max_features'])
    print("minDF: %d" % best_result['params']['min_df'])
    print("maxDF: %.2f" % best_result['params']['max_df'])
    print("Silhouette Score: %.6f" % best_result['silhouette'])
    print("Total Time: %.2fs" % best_result['total_time'])


def main():
    parser = argparse.ArgumentParser(description="StackOverflow Question Clustering - Feature Extraction Optimization")
    parser.add_argument("--input", type=str, required=True, help="Input data path (JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output result path")
    parser.add_argument("--k", type=int, default=50, help="Number of clusters K (default: 50)")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Sampling ratio (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=30, help="Max iterations (default: 30)")
    parser.add_argument("--grid-search", action="store_true", help="Run grid search for parameter optimization")
    parser.add_argument("--vocab-size", type=int, default=5000, help="Vocabulary size (if not grid search)")
    parser.add_argument("--min-df", type=int, default=5, help="Min document frequency (if not grid search)")
    parser.add_argument("--max-df", type=float, default=0.9, help="Max document frequency (if not grid search)")
    
    args = parser.parse_args()
    
    total_start = time.time()
    
    print("\n" + "=" * 60)
    print("StackOverflow Question Clustering - Feature Extraction Optimization")
    print("=" * 60)
    print("Input path: %s" % args.input)
    print("Output path: %s" % args.output)
    print("Number of clusters K: %d" % args.k)
    print("Sampling ratio: %f" % args.sample_ratio)
    print("Grid search: %s" % ("Yes" if args.grid_search else "No"))
    print("=" * 60)
    
    spark = create_spark_session()
    
    try:
        # 加载和预处理数据
        df = load_data(spark, args.input, args.sample_ratio)
        df = preprocess_data_optimized(df)
        df = df.cache()
        df.count()
        
        if args.grid_search:
            # 参数网格搜索
            print("\n" + "=" * 60)
            print("Running Parameter Grid Search")
            print("=" * 60)
            
            param_grid = generate_param_grid()
            print("Total parameter combinations: %d" % len(param_grid))
            
            all_results = []
            max_df_terms = {}
            for max_df in sorted(set(item["max_df"] for item in param_grid)):
                max_df_terms[max_df] = find_high_frequency_terms(df, max_df)
            
            for i, params in enumerate(param_grid):
                print("\n[%d/%d] Running experiment..." % (i+1, len(param_grid)))
                
                result, predictions = run_single_experiment(
                    df,
                    params,
                    args.k,
                    args.max_iter,
                    max_df_terms[params["max_df"]],
                )
                all_results.append(result)
                
                # 保存中间结果
                if i % 5 == 0:
                    save_comparison_results(all_results, args.output)
            
            # 保存最终对比结果
            save_comparison_results(all_results, args.output)
            
            # 使用最佳参数保存最终聚类结果
            best_result = max(all_results, key=lambda x: x['silhouette'])
            best_params = best_result['params']
            
            print("\n" + "=" * 60)
            print("Running Final Clustering with Best Parameters")
            print("=" * 60)
            
            pipeline = build_optimized_pipeline(
                best_params['max_features'],
                best_params['min_df'],
                best_params['max_df'],
                max_df_terms[best_params["max_df"]],
            )
            pipeline_model, processed_df, _, _ = fit_and_transform(pipeline, df)
            kmeans_model, predictions, _ = run_kmeans(
                processed_df, args.k, args.max_iter
            )
            
            save_results(predictions, args.output + "/best_clustering")
            
        else:
            # 单次运行
            print("\n" + "=" * 60)
            print("Running Single Experiment")
            print("=" * 60)
            print("Parameters: vocabSize=%d, minDF=%d, maxDF=%.2f" % (
                args.vocab_size, args.min_df, args.max_df
            ))
            
            params = {
                'max_features': args.vocab_size,
                'min_df': args.min_df,
                'max_df': args.max_df,
            }
            
            result, predictions = run_single_experiment(
                df, params, args.k, args.max_iter
            )
            
            save_results(predictions, args.output)
            
            # 保存单次结果
            save_comparison_results([result], args.output)
        
        total_elapsed = time.time() - total_start
        
        print("\n" + "=" * 60)
        print("Optimization Execution Complete!")
        print("=" * 60)
        print("Total time: %.2fs" % total_elapsed)
        print("Results saved to: %s" % args.output)
        print("=" * 60)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
