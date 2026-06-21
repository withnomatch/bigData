#!/bin/bash
# =============================================================================
# 聚类结构实验 一键运行脚本
# 运行 E1-E4 并对每个实验调用评测脚本
# =============================================================================

export HADOOP_HOME=/opt/hadoop
export SPARK_HOME=/opt/spark-2.4.8-bin-hadoop2.7
export PATH=$SPARK_HOME/bin:$HADOOP_HOME/bin:$PATH

SCRIPT_DIR=/root
DATA=hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300
PAIRS=hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv
OUT_BASE=hdfs://node5:9000/user/root/output/e_structure
LOG_BASE=$SCRIPT_DIR

echo "============================================================"
echo "E: Clustering Structure Experiments"
echo "Input : $DATA"
echo "Output: $OUT_BASE"
echo "Time  : $(date)"
echo "============================================================"

# ── Step 1: 运行 E1-E4 主实验 ─────────────────────────────────────────────
echo ""
echo "[Step 1] Running E1-E4 clustering experiments..."

spark-submit \
  --master spark://node5:7077 \
  --executor-memory 4g \
  --driver-memory 2g \
  --conf spark.executor.cores=2 \
  $SCRIPT_DIR/structure_experiments.py \
  --input "$DATA" \
  --output "$OUT_BASE" \
  --k 45 \
  --max-features 5000 \
  --min-df 5 \
  --max-df 0.9 \
  --km-max-iter 30 \
  --bkm-max-iter 20 \
  --merge-threshold 0.1 \
  --dedup-top 80 \
  --threshold 0.55 \
  --sample-ratio 1.0 \
  2>&1 | tee $LOG_BASE/e_structure.log

echo ""
echo "[Step 1] Done. Log -> $LOG_BASE/e_structure.log"

# ── Step 2: 对每个实验调用评测脚本 ──────────────────────────────────────────
# 需要 evaluate_dedup_from_spark_outputs.py 在 $SCRIPT_DIR 下

EVAL_SCRIPT=$SCRIPT_DIR/evaluate_dedup_from_spark_outputs.py

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "[WARN] evaluate_dedup_from_spark_outputs.py not found at $EVAL_SCRIPT"
  echo "       Skipping evaluation. Run manually using commands below."
else
  for EXP in e1_kmeans e2_bisecting e3_kmeans_merge e4_bisecting_merge; do
    echo ""
    echo "[Step 2] Evaluating $EXP ..."
    spark-submit \
      --master spark://node5:7077 \
      --executor-memory 2g \
      --driver-memory 1g \
      $EVAL_SCRIPT \
      --pairs "$PAIRS" \
      --assignments "$OUT_BASE/$EXP/cluster_assignments" \
      --candidates  "$OUT_BASE/$EXP/duplicate_candidates" \
      --output      "$OUT_BASE/${EXP}_eval" \
      2>&1 | tee -a $LOG_BASE/e_structure.log
    echo "[Step 2] $EXP evaluation done."
  done
fi

# ── Step 3: 提取实验报告 ───────────────────────────────────────────────────
echo ""
echo "[Step 3] Extracting experiment summary..."
grep "^\[REPORT\]" $LOG_BASE/e_structure.log \
  | sed 's/^\[REPORT\] *//' \
  > $LOG_BASE/e_experiment_report.txt

echo ""
echo "============================================================"
echo "All done. $(date)"
echo "  Main log      : $LOG_BASE/e_structure.log"
echo "  Summary report: $LOG_BASE/e_experiment_report.txt"
echo ""
echo "  Manual evaluation commands (if Step 2 was skipped):"
for EXP in e1_kmeans e2_bisecting e3_kmeans_merge e4_bisecting_merge; do
  echo ""
  echo "  spark-submit $EVAL_SCRIPT \\"
  echo "    --pairs $PAIRS \\"
  echo "    --assignments $OUT_BASE/$EXP/cluster_assignments \\"
  echo "    --candidates  $OUT_BASE/$EXP/duplicate_candidates \\"
  echo "    --output      $OUT_BASE/${EXP}_eval"
done
echo "============================================================"
