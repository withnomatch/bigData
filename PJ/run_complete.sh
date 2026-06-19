#!/bin/bash

# Basic complete workflow:
# JSONLines data -> QA document text -> TF-IDF -> KMeans -> duplicate candidates.

MASTER_NODE="10.176.62.230"
SPARK_MASTER="spark://${MASTER_NODE}:7077"

HDFS_INPUT="/user/root/data/oracle_database_questions_jsonlines.json"
HDFS_PAIRS="/user/root/data/dedup_pairs_300_reviewed.csv"
HDFS_FIXED_SAMPLE="/user/root/data/fixed_eval_sample_10pct_plus_300"
HDFS_OUTPUT="/user/root/output/complete_clustering"

K=50
MAX_FEATURES=5000
MIN_DF=5
MAX_ITER=30
SAMPLE_RATIO=0.1
DEDUP_TOP_PER_CLUSTER=80
SIMILARITY_THRESHOLD=0.55

DRIVER_MEMORY="2g"
EXECUTOR_MEMORY="4g"
EXECUTOR_CORES="2"
NUM_EXECUTORS="2"

echo "============================================================"
echo "StackOverflow Oracle Question Deduplication"
echo "============================================================"
echo "Spark master: ${SPARK_MASTER}"
echo "Input: hdfs://${MASTER_NODE}:9000${HDFS_INPUT}"
echo "Annotation pairs: hdfs://${MASTER_NODE}:9000${HDFS_PAIRS}"
echo "Fixed eval sample: hdfs://${MASTER_NODE}:9000${HDFS_FIXED_SAMPLE}"
echo "Output: hdfs://${MASTER_NODE}:9000${HDFS_OUTPUT}"
echo "K: ${K}"
echo "Sample ratio: ${SAMPLE_RATIO}"
echo "Duplicate threshold: ${SIMILARITY_THRESHOLD}"
echo "============================================================"

hdfs dfs -test -e ${HDFS_INPUT}
if [ $? -ne 0 ]; then
    echo "ERROR: input data does not exist: ${HDFS_INPUT}"
    echo "Upload oracle_database_questions_jsonlines.json to HDFS first."
    exit 1
fi

hdfs dfs -test -e ${HDFS_PAIRS}
if [ $? -ne 0 ]; then
    echo "ERROR: annotation pairs do not exist: ${HDFS_PAIRS}"
    echo "Upload annotation/dedup_pairs_300_reviewed.csv to HDFS first."
    exit 1
fi

hdfs dfs -rm -r -f ${HDFS_FIXED_SAMPLE}
spark-submit \
    --master ${SPARK_MASTER} \
    --driver-memory ${DRIVER_MEMORY} \
    --executor-memory ${EXECUTOR_MEMORY} \
    --executor-cores ${EXECUTOR_CORES} \
    --num-executors ${NUM_EXECUTORS} \
    --conf spark.sql.shuffle.partitions=24 \
    --conf spark.default.parallelism=24 \
    create_fixed_sample.py \
    --input hdfs://node5:9000${HDFS_INPUT} \
    --annotation-pairs hdfs://node5:9000${HDFS_PAIRS} \
    --output hdfs://node5:9000${HDFS_FIXED_SAMPLE} \
    --ratio ${SAMPLE_RATIO} \
    --seed 42

hdfs dfs -test -e ${HDFS_OUTPUT}
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r ${HDFS_OUTPUT}
fi

spark-submit \
    --master ${SPARK_MASTER} \
    --driver-memory ${DRIVER_MEMORY} \
    --executor-memory ${EXECUTOR_MEMORY} \
    --executor-cores ${EXECUTOR_CORES} \
    --num-executors ${NUM_EXECUTORS} \
    --conf spark.sql.shuffle.partitions=24 \
    --conf spark.default.parallelism=24 \
    complete_clustering.py \
    --input hdfs://node5:9000${HDFS_FIXED_SAMPLE} \
    --output hdfs://node5:9000${HDFS_OUTPUT} \
    --k ${K} \
    --max-features ${MAX_FEATURES} \
    --min-df ${MIN_DF} \
    --max-iter ${MAX_ITER} \
    --sample-ratio 1.0 \
    --dedup-top-per-cluster ${DEDUP_TOP_PER_CLUSTER} \
    --similarity-threshold ${SIMILARITY_THRESHOLD}

echo "============================================================"
echo "Result paths:"
echo "  ${HDFS_OUTPUT}/cluster_assignments"
echo "  ${HDFS_OUTPUT}/cluster_summary"
echo "  ${HDFS_OUTPUT}/top_questions_per_cluster"
echo "  ${HDFS_OUTPUT}/cluster_keywords"
echo "  ${HDFS_OUTPUT}/duplicate_candidates"
echo "============================================================"
