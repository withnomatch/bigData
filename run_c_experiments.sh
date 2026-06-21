#!/bin/bash
# C组成员完整流程脚本
# 包含：实验运行 + 评测 + 结果汇总

# 配置参数
INPUT_PATH="hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300"
BASE_OUTPUT="hdfs://node5:9000/user/root/output/c_feature_optimization"
EVAL_OUTPUT="hdfs://node5:9000/user/root/output/c_feature_optimization_eval"
ANNOTATION_PATH="hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv"
K=45  # 使用成员B确定的最优K值
SAMPLE_RATIO=1.0  # 使用固定样本，不重新抽样
MAX_ITER=30
DEDUP_TOP=80
SIM_THRESHOLD=0.55

# Spark配置（使用local模式，更稳定）
SPARK_MASTER="local[4]"
DRIVER_MEMORY="2g"

# 计数器
total_exp=0
success_exp=0
failed_exp=0
total_eval=0
success_eval=0
failed_eval=0

echo "======================================================================"
echo "C组实验完整流程：实验 + 评测 + 结果汇总"
echo "======================================================================"
echo ""
echo "配置信息："
echo "  输入数据: $INPUT_PATH"
echo "  实验输出: $BASE_OUTPUT"
echo "  评测输出: $EVAL_OUTPUT"
echo "  标注数据: $ANNOTATION_PATH"
echo "  K值: $K"
echo "  样本比例: $SAMPLE_RATIO"
echo "======================================================================"
echo ""

# 实验列表（C1-C7 + C6系列参数优化，与项目总结报告编号一致）
EXPERIMENTS=("C1" "C2" "C3" "C4" "C5" "C6_1" "C6_2" "C6_3" "C6_4" "C6_5" "C6_6" "C6_7" "C6_8" "C6_9" "C6_10" "C6_11" "C6_12" "C7")

echo "======================================================================"
echo "第一阶段：运行所有实验（共17个）"
echo "======================================================================"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Running Experiment: $exp"
    echo "----------------------------------------------------------------------"
    
    total_exp=$((total_exp + 1))
    OUTPUT_PATH="${BASE_OUTPUT}/${exp}_results"
    
    # 运行实验
    spark-submit \
        --master $SPARK_MASTER \
        --driver-memory $DRIVER_MEMORY \
        c_feature_optimization.py \
        --input $INPUT_PATH \
        --output $OUTPUT_PATH \
        --experiment $exp \
        --k $K \
        --sample-ratio $SAMPLE_RATIO \
        --max-iter $MAX_ITER \
        --dedup-top-per-cluster $DEDUP_TOP \
        --similarity-threshold $SIM_THRESHOLD
    
    # 检查是否成功
    if hdfs dfs -test -e ${OUTPUT_PATH}/metrics; then
        echo "✓ Experiment $exp completed successfully"
        success_exp=$((success_exp + 1))
    else
        echo "✗ Experiment $exp failed"
        failed_exp=$((failed_exp + 1))
    fi
    
    echo ""
    sleep 10
done

echo "======================================================================"
echo "第一阶段完成：$success_exp/$total_exp 实验成功，$failed_exp 失败"
echo "======================================================================"
echo ""

echo "======================================================================"
echo "第二阶段：评测所有实验（共17个）"
echo "======================================================================"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Evaluating Experiment: $exp"
    echo "----------------------------------------------------------------------"
    
    total_eval=$((total_eval + 1))
    
    # 运行评测
    spark-submit \
        --master $SPARK_MASTER \
        --driver-memory $DRIVER_MEMORY \
        evaluate_dedup_from_spark_outputs.py \
        --pairs $ANNOTATION_PATH \
        --assignments ${BASE_OUTPUT}/${exp}_results/cluster_assignments \
        --candidates ${BASE_OUTPUT}/${exp}_results/duplicate_candidates \
        --output ${EVAL_OUTPUT}/${exp}_eval
    
    # 检查是否成功
    if hdfs dfs -test -e ${EVAL_OUTPUT}/${exp}_eval; then
        echo "✓ Evaluation $exp completed successfully"
        success_eval=$((success_eval + 1))
    else
        echo "✗ Evaluation $exp failed"
        failed_eval=$((failed_eval + 1))
    fi
    
    echo ""
    sleep 5
done

echo "======================================================================"
echo "第二阶段完成：$success_eval/$total_eval 评测成功，$failed_eval 失败"
echo "======================================================================"
echo ""

echo "======================================================================"
echo "第三阶段：提取并汇总所有结果"
echo "======================================================================"
echo ""

# 创建结果汇总文件
SUMMARY_FILE="all_results_summary.txt"
echo "C组实验结果汇总" > $SUMMARY_FILE
echo "生成时间：$(date)" >> $SUMMARY_FILE
echo "========================================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# 创建CSV格式汇总文件（用于导入Excel）
CSV_FILE="all_results_summary.csv"
echo "实验编号,实验名称,词汇表大小,最小文档频率,最大文档频率,运行时间(秒),WSSSE/样本,最大簇占比,簇大小CV,Candidate Recall,Precision,Recall,F1" > $CSV_FILE

for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------" >> $SUMMARY_FILE
    echo "实验：$exp" >> $SUMMARY_FILE
    echo "----------------------------------------------------------------------" >> $SUMMARY_FILE
    
    # 提取聚类指标
    echo "聚类指标：" >> $SUMMARY_FILE
    hdfs dfs -cat ${BASE_OUTPUT}/${exp}_results/metrics/*.json 2>/dev/null >> $SUMMARY_FILE || echo "无数据" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    
    # 提取评测指标
    echo "评测指标：" >> $SUMMARY_FILE
    hdfs dfs -cat ${EVAL_OUTPUT}/${exp}_eval/*.json 2>/dev/null >> $SUMMARY_FILE || echo "无数据" >> $SUMMARY_FILE
    echo "" >> $SUMMARY_FILE
    
    # 提取CSV数据（需要解析JSON）
    # 这里简化处理，实际运行后可以手动整理
done

echo "" >> $SUMMARY_FILE
echo "========================================================================" >> $SUMMARY_FILE
echo "统计信息：" >> $SUMMARY_FILE
echo "实验成功：$success_exp/$total_exp" >> $SUMMARY_FILE
echo "评测成功：$success_eval/$total_eval" >> $SUMMARY_FILE
echo "========================================================================" >> $SUMMARY_FILE

echo "结果汇总已保存到：$SUMMARY_FILE"
echo "CSV汇总已保存到：$CSV_FILE"
echo ""

echo "======================================================================"
echo "全部完成！"
echo "======================================================================"
echo ""
echo "统计信息："
echo "  实验成功：$success_exp/$total_exp"
echo "  评测成功：$success_eval/$total_eval"
echo ""
echo "输出文件："
echo "  结果汇总：$SUMMARY_FILE"
echo "  CSV汇总：$CSV_FILE"
echo ""
echo "查看结果："
echo "  cat $SUMMARY_FILE"
echo "  cat $CSV_FILE"
echo ""
echo "下载到本地："
echo "  scp root@10.176.62.230:/root/$SUMMARY_FILE ."
echo "  scp root@10.176.62.230:/root/$CSV_FILE ."
echo "======================================================================"
echo ""

# 显示汇总结果
cat $SUMMARY_FILE