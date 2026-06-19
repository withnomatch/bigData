#!/bin/bash
set -e

export HADOOP_HOME=/opt/hadoop
export SPARK_HOME=/opt/spark
export PATH="$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH"

MASTER="spark://10.176.62.230:7077"
INPUT="hdfs://10.176.62.230:9000/user/root/data/oracle_database_questions_jsonlines.json"
OUTPUT_ROOT="hdfs://10.176.62.230:9000/user/root/output/unified_team_20260608"
COMMON_ARGS=(
  --master "$MASTER"
  --driver-memory 4g
  --executor-memory 4g
  --executor-cores 2
  --total-executor-cores 8
  --conf spark.sql.shuffle.partitions=24
  --conf spark.default.parallelism=24
  --conf spark.network.timeout=300
  --conf spark.executor.heartbeatInterval=30
)

hdfs dfs -rm -r -f /user/root/output/unified_team_20260608 >/dev/null 2>&1 || true

echo "=== MEMBER B: K=40..60 ==="
spark-submit "${COMMON_ARGS[@]}" /root/team_validation/spark_k_value_experiment.py \
  --input "$INPUT" \
  --output "$OUTPUT_ROOT/member_b_k_values" \
  --sample-ratio 0.1 \
  --max-features 5000 \
  --min-df 5 \
  --max-iter 30 \
  --k-min 40 \
  --k-max 60

echo "=== MEMBER C: 64 TF-IDF combinations ==="
spark-submit "${COMMON_ARGS[@]}" /root/team_validation/optimized_clustering.py \
  --input "$INPUT" \
  --output "$OUTPUT_ROOT/member_c_feature_grid" \
  --sample-ratio 0.1 \
  --k 50 \
  --max-iter 30 \
  --grid-search

echo "=== UNIFIED VALIDATION COMPLETE ==="
