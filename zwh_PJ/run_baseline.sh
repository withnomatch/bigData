#!/bin/bash

# StackOverflow 运维文本聚类 - 基准方案集群提交脚本
# 使用方法: ./run_baseline.sh

# ==================== 配置参数 ====================
MASTER_NODE="10.176.62.230"
HDFS_INPUT="/user/root/data/oracle_database_questions.json"
HDFS_OUTPUT="/user/root/output/baseline_clustering"

K=50
MAX_FEATURES=5000
MIN_DF=5
MAX_ITER=30

SPARK_MASTER="spark://${MASTER_NODE}:7077"
DRIVER_MEMORY="4g"
EXECUTOR_MEMORY="4g"
EXECUTOR_CORES="2"
NUM_EXECUTORS="3"

# ==================== 打印配置 ====================
echo "============================================================"
echo "StackOverflow 运维文本聚类 - 基准方案"
echo "============================================================"
echo "集群Master: ${SPARK_MASTER}"
echo "输入路径: hdfs://${MASTER_NODE}:9000${HDFS_INPUT}"
echo "输出路径: hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo "聚类数量K: ${K}"
echo "最大词汇表: ${MAX_FEATURES}"
echo "最小文档频率: ${MIN_DF}"
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
    baseline_clustering.py \
    --input ${HDFS_INPUT} \
    --output ${HDFS_OUTPUT} \
    --k ${K} \
    --max-features ${MAX_FEATURES} \
    --min-df ${MIN_DF} \
    --max-iter ${MAX_ITER}

# ==================== 检查结果 ====================
echo ""
echo "============================================================"
echo "[验证] 检查输出结果..."
echo "============================================================"

hdfs dfs -test -e ${HDFS_OUTPUT}/cluster_assignments
if [ $? -eq 0 ]; then
    echo "✓ 聚类分配结果已生成"
    ASSIGNMENTS_COUNT=$(hdfs dfs -cat ${HDFS_OUTPUT}/cluster_assignments/* | wc -l)
    echo "  问题分配数: ${ASSIGNMENTS_COUNT}"
fi

hdfs dfs -test -e ${HDFS_OUTPUT}/cluster_summary
if [ $? -eq 0 ]; then
    echo "✓ 聚类摘要已生成"
    echo ""
    echo "聚类摘要预览:"
    hdfs dfs -cat ${HDFS_OUTPUT}/cluster_summary/* | head -10
fi

hdfs dfs -test -e ${HDFS_OUTPUT}/top_questions_per_cluster
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Top问题列表已生成"
fi

echo ""
echo "============================================================"
echo "基准方案执行完成!"
echo "============================================================"
echo "结果路径: hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo ""
echo "查看结果命令:"
echo "  hdfs dfs -ls ${HDFS_OUTPUT}"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/cluster_summary/*"
echo "============================================================"
