#!/bin/bash

# Word2Vec特征提取对比实验 - 集群提交脚本
# 使用方法: ./run_word2vec_experiment.sh

# ==================== 配置参数 ====================
MASTER_NODE="10.176.62.230"
HDFS_INPUT="/user/root/data/oracle_database_questions_jsonlines.json"
HDFS_OUTPUT="/user/root/output/word2vec_experiment"

K=50
MAX_FEATURES=5000
MIN_DF=5
VECTOR_SIZE=100
MIN_COUNT=5
MAX_ITER=30

SPARK_MASTER="spark://${MASTER_NODE}:7077"
DRIVER_MEMORY="4g"
EXECUTOR_MEMORY="4g"
EXECUTOR_CORES="2"
NUM_EXECUTORS="3"

# ==================== 打印配置 ====================
echo "============================================================"
echo "Word2Vec Feature Extraction Comparison Experiment"
echo "============================================================"
echo "集群Master: ${SPARK_MASTER}"
echo "输入路径: hdfs://${MASTER_NODE}:9000${HDFS_INPUT}"
echo "输出路径: hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo "聚类数量K: ${K}"
echo "最大词汇表: ${MAX_FEATURES}"
echo "Word2Vec向量维度: ${VECTOR_SIZE}"
echo "Word2Vec最小词频: ${MIN_COUNT}"
echo "============================================================"

# ==================== 检查输入数据 ====================
echo ""
echo "[检查] 验证输入数据..."
hdfs dfs -test -e ${HDFS_INPUT}
if [ $? -ne 0 ]; then
    echo "错误: 输入数据不存在: ${HDFS_INPUT}"
    echo "请先运行 ./upload_data.sh 上传数据"
    exit 1
fi

INPUT_SIZE=$(hdfs dfs -du -h ${HDFS_INPUT} | awk '{print $1}')
echo "输入数据大小: ${INPUT_SIZE}"

# ==================== 清理旧输出 ====================
echo ""
echo "[清理] 删除旧的输出目录..."
hdfs dfs -test -e ${HDFS_OUTPUT}
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r ${HDFS_OUTPUT}
    echo "已删除旧输出目录"
fi

# ==================== 提交Spark任务 ====================
echo ""
echo "[提交] 提交Spark任务..."
echo "============================================================"

spark-submit \
    --master ${SPARK_MASTER} \
    --driver-memory ${DRIVER_MEMORY} \
    --executor-memory ${EXECUTOR_MEMORY} \
    --executor-cores ${EXECUTOR_CORES} \
    --num-executors ${NUM_EXECUTORS} \
    --conf spark.sql.shuffle.partitions=24 \
    --conf spark.default.parallelism=24 \
    --conf spark.network.timeout=300 \
    --conf spark.executor.heartbeatInterval=30 \
    word2vec_experiment.py \
    --input ${HDFS_INPUT} \
    --output ${HDFS_OUTPUT} \
    --k ${K} \
    --max-features ${MAX_FEATURES} \
    --min-df ${MIN_DF} \
    --vector-size ${VECTOR_SIZE} \
    --min-count ${MIN_COUNT} \
    --max-iter ${MAX_ITER}

# ==================== 检查结果 ====================
echo ""
echo "============================================================"
echo "[验证] 检查输出结果..."
echo "============================================================"

for method in tfidf word2vec tfidf_word2vec; do
    hdfs dfs -test -e ${HDFS_OUTPUT}/${method}/cluster_assignments
    if [ $? -eq 0 ]; then
        echo "✓ ${method} 聚类分配结果已生成"
        COUNT=$(hdfs dfs -cat ${HDFS_OUTPUT}/${method}/cluster_assignments/* | wc -l)
        echo "  问题分配数: ${COUNT}"
    fi

    hdfs dfs -test -e ${HDFS_OUTPUT}/${method}/cluster_summary
    if [ $? -eq 0 ]; then
        echo "✓ ${method} 聚类摘要已生成"
    fi

    hdfs dfs -test -e ${HDFS_OUTPUT}/${method}/top_questions_per_cluster
    if [ $? -eq 0 ]; then
        echo "✓ ${method} Top问题列表已生成"
    fi
done

echo ""
echo "============================================================"
echo "Word2Vec对比实验执行完成!"
echo "============================================================"
echo "结果路径: hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo ""
echo "查看结果命令:"
echo "  hdfs dfs -ls ${HDFS_OUTPUT}"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/tfidf/cluster_summary/*"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/word2vec/cluster_summary/*"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/tfidf_word2vec/cluster_summary/*"
echo "============================================================"
