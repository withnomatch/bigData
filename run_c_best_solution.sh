#!/bin/bash

# C组最优方案（C1基线）运行脚本

echo "========================================================================"
echo "C组最优方案：C1基线方案"
echo "========================================================================"
echo "特征构造：title + body"
echo "TF-IDF参数：vocab=5000, min_df=5, max_df=0.9"
echo "聚类参数：K=45"
echo "========================================================================"

# Spark配置
SPARK_MASTER="local[4]"
DRIVER_MEMORY="2g"

# 输入输出路径
INPUT_PATH="hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300"
OUTPUT_PATH="hdfs://node5:9000/user/root/output/c_best_solution_c1"
ANNOTATION_PATH="hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv"

# 删除已存在的输出目录
echo "删除已存在的输出目录..."
hdfs dfs -rm -r "$OUTPUT_PATH" 2>/dev/null

# 运行最优方案
echo "运行C组最优方案..."
spark-submit \
    --master $SPARK_MASTER \
    --driver-memory $DRIVER_MEMORY \
    c_best_solution.py \
    --input $INPUT_PATH \
    --output $OUTPUT_PATH \
    --annotation-pairs $ANNOTATION_PATH \
    --k 45 \
    --max-features 5000 \
    --min-df 5 \
    --max-df 0.9 \
    --similarity-threshold 0.55 \
    --dedup-top-per-cluster 80

echo "========================================================================"
echo "实验完成！"
echo "========================================================================"

# 查看实验报告
echo "查看实验报告..."
hdfs dfs -cat "$OUTPUT_PATH/experiment_report.json"

echo "========================================================================"
echo "查看聚类分配..."
hdfs dfs -ls "$OUTPUT_PATH/cluster_assignments"

echo "========================================================================"
echo "查看重复候选..."
hdfs dfs -ls "$OUTPUT_PATH/duplicate_candidates"

echo "========================================================================"
echo "查看聚类摘要..."
hdfs dfs -ls "$OUTPUT_PATH/cluster_summary"

echo "========================================================================"
echo "全部完成！"
echo "========================================================================"