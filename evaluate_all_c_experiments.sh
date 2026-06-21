#!/bin/bash
# C组成员批量评测脚本
# 对所有实验结果进行去重效果评测

# 配置参数
ANNOTATION_PATH="hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv"
BASE_OUTPUT="hdfs://node5:9000/user/root/output/c_feature_optimization"
EVAL_BASE="hdfs://node5:9000/user/root/output/c_feature_optimization_eval"

echo "=========================================="
echo "C Member: Batch Evaluation for All Experiments"
echo "=========================================="
echo "Annotation pairs: $ANNOTATION_PATH"
echo "Base output: $BASE_OUTPUT"
echo "=========================================="

# 实验列表（C1-C7 + C6系列参数优化，与项目总结报告编号一致）
EXPERIMENTS=("C1" "C2" "C3" "C4" "C5" "C6_1" "C6_2" "C6_3" "C6_4" "C6_5" "C6_6" "C6_7" "C6_8" "C6_9" "C6_10" "C6_11" "C6_12" "C7")

# 创建结果汇总文件
SUMMARY_FILE="c_experiment_results_summary.txt"
echo "Experiment Results Summary" > $SUMMARY_FILE
echo "===========================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# 评测每个实验
for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Evaluating Experiment: $exp"
    echo "=========================================="
    
    OUTPUT_PATH="${BASE_OUTPUT}/${exp}_results"
    EVAL_OUTPUT="${EVAL_BASE}/${exp}_eval"
    
    # 运行评测脚本
    spark-submit \
        --master local[4] \
        --driver-memory 2g \
        evaluate_dedup_from_spark_outputs.py \
        --pairs $ANNOTATION_PATH \
        --assignments ${OUTPUT_PATH}/cluster_assignments \
        --candidates ${OUTPUT_PATH}/duplicate_candidates \
        --output $EVAL_OUTPUT
    
    echo "Evaluation for $exp completed!"
    echo "Eval results saved to: $EVAL_OUTPUT"
    
    # 提取关键指标并汇总
    echo "" >> $SUMMARY_FILE
    echo "Experiment: $exp" >> $SUMMARY_FILE
    hdfs dfs -cat ${EVAL_OUTPUT}/metrics.json >> $SUMMARY_FILE 2>/dev/null || echo "Metrics not available" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    
    echo ""
    sleep 2
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo "Summary file: $SUMMARY_FILE"
echo ""

# 显示汇总结果
cat $SUMMARY_FILE

echo ""
echo "Next step: Fill the comparison table in the project report"