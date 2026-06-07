#!/bin/bash
# run_clustering.sh - Submit Spark clustering job to YARN

HDFS_INPUT="/stackoverflow_oracle/input/oracle_database_questions.json"
HDFS_OUTPUT="/stackoverflow_oracle/output"
K=50
MAX_FEATURES=5000

echo "============================================"
echo "Submitting Text Clustering Job to Spark/YARN"
echo "============================================"

spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name "StackOverflow_Question_Clustering" \
    --driver-memory 4g \
    --executor-memory 4g \
    --executor-cores 2 \
    --num-executors 4 \
    --conf spark.sql.shuffle.partitions=32 \
    --conf spark.driver.maxResultSize=2g \
    text_clustering.py \
    --input ${HDFS_INPUT} \
    --output ${HDFS_OUTPUT} \
    --k ${K} \
    --max-features ${MAX_FEATURES} \
    --min-df 5

echo ""
echo "Job submitted! Check YARN UI for progress."
echo "Results will be at: ${HDFS_OUTPUT}"
echo ""
echo "To view results:"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/cluster_summary/* | head -50"
echo "  hdfs dfs -cat ${HDFS_OUTPUT}/top_questions_per_cluster/* | head -100"
