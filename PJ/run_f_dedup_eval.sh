#!/bin/bash

# F-task: K-Means execution parameter optimization with reviewed 300-pair
# duplicate-detection evaluation. Run this script on node5.

MASTER_NODE="10.176.62.230"
SPARK_MASTER="spark://${MASTER_NODE}:7077"
SPARK_HOME="/opt/spark-2.0.0-bin-hadoop2.7"
SPARK_SUBMIT="${SPARK_HOME}/bin/spark-submit"

HDFS_INPUT="/user/root/data/fixed_eval_sample_10pct_plus_300"
HDFS_PAIRS="/user/root/data/dedup_pairs_300_reviewed.csv"
HDFS_OUTPUT="/user/root/output/f_execution_param_dedup_eval"

FULL_INPUT="hdfs://${MASTER_NODE}:9000${HDFS_INPUT}"
FULL_PAIRS="hdfs://${MASTER_NODE}:9000${HDFS_PAIRS}"
FULL_OUTPUT="hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"

K=50
MAX_FEATURES=5000
MIN_DF=5
THRESHOLD=0.55
MAX_LARGEST_RATIO=0.50

PLAN="focused"
INIT_MODES="random,k-means||"
DISTANCE_MODES="euclidean,cosine"
MAX_ITERS="10,20,30"
SEEDS="42,2026,3407"
SILHOUETTE_SAMPLE=2000

DRIVER_MEMORY="4g"
EXECUTOR_MEMORY="4g"
EXECUTOR_CORES="2"
NUM_EXECUTORS="2"

echo "============================================================"
echo "F-task: execution params + 300-pair dedup evaluation"
echo "============================================================"
echo "Spark master: ${SPARK_MASTER}"
echo "Spark submit: ${SPARK_SUBMIT}"
echo "Input: ${FULL_INPUT}"
echo "Pairs: ${FULL_PAIRS}"
echo "Output: ${FULL_OUTPUT}"
echo "K=${K}, threshold=${THRESHOLD}"
echo "Max largest-cluster ratio for comprehensive recommendation=${MAX_LARGEST_RATIO}"
echo "Plan=${PLAN}"
echo "============================================================"

echo ""
echo "[Check] Verifying HDFS inputs..."
hdfs dfs -test -e "${HDFS_INPUT}"
if [ $? -ne 0 ]; then
    echo "Error: fixed sample does not exist: ${HDFS_INPUT}"
    exit 1
fi

hdfs dfs -test -e "${HDFS_PAIRS}"
if [ $? -ne 0 ]; then
    echo "Error: reviewed pair CSV does not exist: ${HDFS_PAIRS}"
    exit 1
fi

echo ""
echo "[Clean] Removing old output..."
hdfs dfs -test -e "${HDFS_OUTPUT}"
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r "${HDFS_OUTPUT}"
fi

echo ""
echo "[Submit] Running Spark job..."
export SPARK_HOME
"${SPARK_SUBMIT}" \
    --master "${SPARK_MASTER}" \
    --driver-memory "${DRIVER_MEMORY}" \
    --executor-memory "${EXECUTOR_MEMORY}" \
    --executor-cores "${EXECUTOR_CORES}" \
    --num-executors "${NUM_EXECUTORS}" \
    --conf spark.sql.shuffle.partitions=24 \
    --conf spark.default.parallelism=24 \
    --conf spark.network.timeout=300 \
    --conf spark.executor.heartbeatInterval=30 \
    f_execution_param_dedup_eval.py \
    --input "${FULL_INPUT}" \
    --pairs "${FULL_PAIRS}" \
    --output "${FULL_OUTPUT}" \
    --k "${K}" \
    --max-features "${MAX_FEATURES}" \
    --min-df "${MIN_DF}" \
    --threshold "${THRESHOLD}" \
    --max-largest-ratio "${MAX_LARGEST_RATIO}" \
    --plan "${PLAN}" \
    --init-modes "${INIT_MODES}" \
    --distance-modes "${DISTANCE_MODES}" \
    --max-iters "${MAX_ITERS}" \
    --seeds "${SEEDS}" \
    --silhouette-sample "${SILHOUETTE_SAMPLE}"

SPARK_STATUS=$?
if [ ${SPARK_STATUS} -ne 0 ]; then
    echo "Error: spark-submit failed with status ${SPARK_STATUS}"
    exit ${SPARK_STATUS}
fi

echo ""
echo "============================================================"
echo "[Verify] Experiment result preview"
echo "============================================================"
hdfs dfs -cat "${HDFS_OUTPUT}/experiment_results/"* | head -20

echo ""
echo "Best config:"
hdfs dfs -cat "${HDFS_OUTPUT}/best_config/"*

echo ""
echo "============================================================"
echo "F-task finished."
echo "Result path: ${FULL_OUTPUT}"
echo "============================================================"
