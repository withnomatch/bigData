#!/bin/bash

# StackOverflow 聚类结构优化 - 集群提交脚本
# BisectingKMeans vs KMeans 对比 + 聚类后处理
# 使用方法: ./run_structure_opt.sh

# ==================== 环境变量 ====================
export HADOOP_HOME=/opt/hadoop
export SPARK_HOME=/opt/spark          # 实际运行版本: Spark 2.4.8
export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$PATH

# ==================== 配置参数 ====================
MASTER_NODE="10.176.62.230"
HDFS_INPUT="/user/root/data/oracle_database_questions_jsonlines.json"
HDFS_OUTPUT="/user/root/output/structure_optimization"

K=50
MAX_FEATURES=5000
MIN_DF=5
KM_MAX_ITER=30
BKM_MAX_ITER=20
SAMPLE_RATIO=0.1       # 先用 10% 数据测试，全量改为 1.0
MERGE_THRESHOLD=0.1    # 小聚类阈值（平均大小的 10%）
LABEL_TOP_K=5          # 每个聚类标签关键词数

SPARK_MASTER="spark://${MASTER_NODE}:7077"
DRIVER_MEMORY="4g"
EXECUTOR_MEMORY="4g"
EXECUTOR_CORES="2"
NUM_EXECUTORS="3"

# ==================== 打印配置 ====================
echo "============================================================"
echo "聚类结构优化 - BisectingKMeans vs KMeans"
echo "聚类结构优化 + 后处理"
echo "============================================================"
echo "集群Master  : ${SPARK_MASTER}"
echo "输入路径    : hdfs://${MASTER_NODE}:9000${HDFS_INPUT}"
echo "输出路径    : hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo "聚类数 K    : ${K}"
echo "采样比例    : ${SAMPLE_RATIO}"
echo "最大词汇表  : ${MAX_FEATURES}"
echo "合并阈值    : ${MERGE_THRESHOLD} (avg_size 的 ${MERGE_THRESHOLD} 倍)"
echo "============================================================"

# ==================== 检查输入数据 ====================
echo ""
echo "[检查] 验证输入数据..."
hdfs dfs -test -e ${HDFS_INPUT}
if [ $? -ne 0 ]; then
    echo "错误: 输入数据不存在: ${HDFS_INPUT}"
    echo "请先运行 ./upload_baseline_data.sh 上传数据"
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

# ==================== 提交 Spark 任务 ====================
echo ""
echo "[提交] 提交 Spark 任务..."
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
    structure_optimization.py \
    --input  hdfs://${MASTER_NODE}:9000${HDFS_INPUT} \
    --output hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT} \
    --k ${K} \
    --max-features ${MAX_FEATURES} \
    --min-df ${MIN_DF} \
    --sample-ratio ${SAMPLE_RATIO} \
    --km-max-iter ${KM_MAX_ITER} \
    --bkm-max-iter ${BKM_MAX_ITER} \
    --merge-threshold ${MERGE_THRESHOLD} \
    --label-top-k ${LABEL_TOP_K}

EXIT_CODE=$?

# ==================== 检查结果 ====================
echo ""
echo "============================================================"
echo "[验证] 检查输出结果..."
echo "============================================================"

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "错误: Spark 任务执行失败 (exit code: ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi

for dir in comparison_table kmeans_assignments bisecting_kmeans_assignments \
           kmeans_summary bisecting_kmeans_summary; do
    hdfs dfs -test -e ${HDFS_OUTPUT}/${dir}
    if [ $? -eq 0 ]; then
        echo "  ✓ ${dir}"
    else
        echo "  ✗ ${dir} (未生成)"
    fi
done

echo ""
echo "对比表格预览:"
hdfs dfs -cat ${HDFS_OUTPUT}/comparison_table/*.csv 2>/dev/null | head -20

echo ""
echo "============================================================"
echo "结构优化实验执行完成!"
echo "============================================================"
echo "结果路径: hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo ""

# ==================== 提取干净的实验报告 ====================
echo "[提取] 从日志中提取实验报告..."
grep "^\[REPORT\]" structure_opt.log | sed 's/^\[REPORT\] \?//' > experiment_report.txt

if [ -s experiment_report.txt ]; then
    echo "✓ 实验报告已保存至: experiment_report.txt"
    echo ""
    echo "====== 实验报告预览 ======"
    cat experiment_report.txt
else
    echo "⚠ 未能提取实验报告，请手动查看 structure_opt.log"
fi

echo ""
echo "查看完整结果:"
echo "  cat experiment_report.txt"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/comparison_table/*.csv"
echo "============================================================"
