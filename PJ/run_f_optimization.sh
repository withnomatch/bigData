#!/bin/bash

# F-task: K-Means execution parameter optimization.
# Run this script on node5 after the Spark/HDFS services are started.

MASTER_NODE="10.176.62.230"
HDFS_INPUT="/user/root/data/oracle_database_questions_jsonlines.json"
HDFS_OUTPUT="/user/root/output/f_param_optimization"
FULL_INPUT="hdfs://${MASTER_NODE}:9000${HDFS_INPUT}"
FULL_OUTPUT="hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"

K=50
MAX_FEATURES=5000
MIN_DF=5
SAMPLE_RATIO=0.1

PLAN="focused"
INIT_MODES="random,k-means||"
DISTANCE_MODES="euclidean,cosine"
MAX_ITERS="10,20,30"
SEEDS="42,2026,3407"
SILHOUETTE_SAMPLE=2000

SPARK_MASTER="spark://${MASTER_NODE}:7077"
DRIVER_MEMORY="4g"
EXECUTOR_MEMORY="4g"
EXECUTOR_CORES="2"
NUM_EXECUTORS="2"

echo "============================================================"
echo "F-task: K-Means execution parameter optimization"
echo "============================================================"
echo "Spark master: ${SPARK_MASTER}"
echo "Input: ${FULL_INPUT}"
echo "Output: ${FULL_OUTPUT}"
echo "K: ${K}"
echo "Sample ratio: ${SAMPLE_RATIO}"
echo "Plan: ${PLAN}"
echo "Init modes: ${INIT_MODES}"
echo "Distance modes: ${DISTANCE_MODES}"
echo "Max iterations: ${MAX_ITERS}"
echo "Seeds: ${SEEDS}"
echo "============================================================"

echo ""
echo "[Check] Verifying input data..."
hdfs dfs -test -e ${HDFS_INPUT}
if [ $? -ne 0 ]; then
    echo "Error: input data does not exist: ${HDFS_INPUT}"
    echo "Please upload oracle_database_questions_jsonlines.json to HDFS first."
    exit 1
fi

INPUT_SIZE=$(hdfs dfs -du -h ${HDFS_INPUT} | awk '{print $1}')
echo "Input size: ${INPUT_SIZE}"

echo ""
echo "[Clean] Removing old output..."
hdfs dfs -test -e ${HDFS_OUTPUT}
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r ${HDFS_OUTPUT}
fi

echo ""
echo "[Submit] Running F-task Spark job..."
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
    f_clustering_param_optimization.py \
    --input ${FULL_INPUT} \
    --output ${FULL_OUTPUT} \
    --k ${K} \
    --max-features ${MAX_FEATURES} \
    --min-df ${MIN_DF} \
    --sample-ratio ${SAMPLE_RATIO} \
    --plan ${PLAN} \
    --init-modes "${INIT_MODES}" \
    --distance-modes "${DISTANCE_MODES}" \
    --max-iters "${MAX_ITERS}" \
    --seeds "${SEEDS}" \
    --silhouette-sample ${SILHOUETTE_SAMPLE}

SPARK_STATUS=$?
if [ ${SPARK_STATUS} -ne 0 ]; then
    echo "Error: spark-submit failed with status ${SPARK_STATUS}"
    exit ${SPARK_STATUS}
fi

echo ""
echo "============================================================"
echo "[Verify] F-task outputs"
echo "============================================================"

hdfs dfs -test -e ${HDFS_OUTPUT}/experiment_results
if [ $? -eq 0 ]; then
    echo "Experiment result table:"
    hdfs dfs -cat ${HDFS_OUTPUT}/experiment_results/* | head -20
fi

hdfs dfs -test -e ${HDFS_OUTPUT}/best_config
if [ $? -eq 0 ]; then
    echo ""
    echo "Best config:"
    hdfs dfs -cat ${HDFS_OUTPUT}/best_config/*
fi

hdfs dfs -test -e ${HDFS_OUTPUT}/best_cluster_summary
if [ $? -eq 0 ]; then
    echo ""
    echo "Best cluster summary preview:"
    hdfs dfs -cat ${HDFS_OUTPUT}/best_cluster_summary/* | head -10
fi

echo ""
echo "============================================================"
echo "F-task optimization finished."
echo "Result path: ${FULL_OUTPUT}"
echo "============================================================"
