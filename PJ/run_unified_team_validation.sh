#!/bin/bash
set -e

export HADOOP_HOME=/opt/hadoop
export SPARK_HOME=/opt/spark
export PATH="$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH"

MASTER="spark://10.176.62.230:7077"
INPUT="hdfs://10.176.62.230:9000/user/root/data/oracle_database_questions_jsonlines.json"
ANNOTATION_PAIRS="hdfs://10.176.62.230:9000/user/root/data/dedup_pairs_300_reviewed.csv"
FIXED_SAMPLE="hdfs://10.176.62.230:9000/user/root/data/fixed_eval_sample_10pct_plus_300"
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
hdfs dfs -rm -r -f /user/root/data/fixed_eval_sample_10pct_plus_300 >/dev/null 2>&1 || true

echo "=== BUILD FIXED SAMPLE: 10% random + 300-pair annotation questions ==="
spark-submit "${COMMON_ARGS[@]}" /root/team_validation/create_fixed_sample.py \
  --input "$INPUT" \
  --annotation-pairs "$ANNOTATION_PAIRS" \
  --output "$FIXED_SAMPLE" \
  --ratio 0.1 \
  --seed 42

echo "=== MEMBER B: K=40..60 ==="
spark-submit "${COMMON_ARGS[@]}" /root/team_validation/spark_k_value_experiment.py \
  --input "$FIXED_SAMPLE" \
  --output "$OUTPUT_ROOT/member_b_k_values" \
  --sample-ratio 1.0 \
  --max-features 5000 \
  --min-df 5 \
  --max-iter 30 \
  --k-min 40 \
  --k-max 60

echo "=== MEMBER C: 64 TF-IDF combinations ==="
spark-submit "${COMMON_ARGS[@]}" /root/team_validation/optimized_clustering.py \
  --input "$FIXED_SAMPLE" \
  --output "$OUTPUT_ROOT/member_c_feature_grid" \
  --sample-ratio 1.0 \
  --k 50 \
  --max-iter 30 \
  --grid-search

echo "=== UNIFIED VALIDATION COMPLETE ==="
