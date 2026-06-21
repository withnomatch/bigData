#!/bin/bash

# C8系列实验一键运行脚本
# 包含实验运行 + 评测 + 结果汇总

# 实验列表
EXPERIMENTS=("C8_1" "C8_2" "C8_3" "C8_4" "C8_5" "C8_6")

# Spark配置
SPARK_MASTER="local[4]"
DRIVER_MEMORY="2g"

# 输入输出路径
INPUT_PATH="hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300"
BASE_OUTPUT="hdfs://node5:9000/user/root/output/c_feature_optimization"
ANNOTATION_PATH="hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv"
EVAL_OUTPUT="hdfs://node5:9000/user/root/output/c_feature_optimization_eval"

# K值和其他参数
K=45
SAMPLE_RATIO=1.0
MAX_ITER=30
DEDUP_TOP=80
SIM_THRESHOLD=0.55

# 统计变量
total_exp=0
total_eval=0
success_exp=0
success_eval=0

echo "======================================================================"
echo "C8系列实验一键运行脚本"
echo "实验目标：验证C1参数调优效果"
echo "======================================================================"
echo "实验列表："
for exp in "${EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo "======================================================================"
echo ""

# 第一阶段：运行所有实验
echo "======================================================================"
echo "第一阶段：运行所有C8实验"
echo "======================================================================"

for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Running Experiment: $exp"
    echo "----------------------------------------------------------------------"
    
    total_exp=$((total_exp + 1))
    OUTPUT_PATH="${BASE_OUTPUT}/${exp}_results"
    
    # 删除已存在的输出目录（避免FileAlreadyExistsException）
    hdfs dfs -rm -r "$OUTPUT_PATH" 2>/dev/null
    
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
    if [ $? -eq 0 ]; then
        echo "✓ Experiment $exp completed successfully"
        success_exp=$((success_exp + 1))
    else
        echo "✗ Experiment $exp failed"
    fi
    
    echo ""
done

echo "======================================================================"
echo "第一阶段完成：运行了 $total_exp 个实验，成功 $success_exp 个"
echo "======================================================================"
echo ""

# 第二阶段：评测所有实验
echo "======================================================================"
echo "第二阶段：评测所有C8实验"
echo "======================================================================"

for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Evaluating Experiment: $exp"
    echo "----------------------------------------------------------------------"
    
    total_eval=$((total_eval + 1))
    
    # 删除已存在的评测目录
    hdfs dfs -rm -r "${EVAL_OUTPUT}/${exp}_eval" 2>/dev/null
    
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
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation $exp completed successfully"
        success_eval=$((success_eval + 1))
    else
        echo "✗ Evaluation $exp failed"
    fi
    
    echo ""
done

echo "======================================================================"
echo "第二阶段完成：评测了 $total_eval 个实验，成功 $success_eval 个"
echo "======================================================================"
echo ""

# 第三阶段：汇总结果
echo "======================================================================"
echo "第三阶段：汇总C8实验结果"
echo "======================================================================"

# 创建临时结果文件
TEMP_RESULTS="/tmp/c8_results.txt"
echo "C8系列实验结果汇总" > $TEMP_RESULTS
echo "生成时间：$(date)" >> $TEMP_RESULTS
echo "=======================================================================" >> $TEMP_RESULTS
echo "" >> $TEMP_RESULTS

# 提取每个实验的聚类指标和评测指标
for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------" >> $TEMP_RESULTS
    echo "实验：$exp" >> $TEMP_RESULTS
    echo "----------------------------------------------------------------------" >> $TEMP_RESULTS
    
    # 提取聚类指标
    CLUSTER_METRICS="${BASE_OUTPUT}/${exp}_results/metrics.json"
    if hdfs dfs -test -e "$CLUSTER_METRICS"; then
        echo "聚类指标：" >> $TEMP_RESULTS
        hdfs dfs -cat "$CLUSTER_METRICS" >> $TEMP_RESULTS
        echo "" >> $TEMP_RESULTS
    else
        echo "聚类指标：无数据" >> $TEMP_RESULTS
        echo "" >> $TEMP_RESULTS
    fi
    
    # 提取评测指标
    EVAL_METRICS="${EVAL_OUTPUT}/${exp}_eval/metrics.json"
    if hdfs dfs -test -e "$EVAL_METRICS"; then
        echo "评测指标：" >> $TEMP_RESULTS
        hdfs dfs -cat "$EVAL_METRICS" >> $TEMP_RESULTS
        echo "" >> $TEMP_RESULTS
    else
        echo "评测指标：无数据" >> $TEMP_RESULTS
        echo "" >> $TEMP_RESULTS
    fi
    
    echo "" >> $TEMP_RESULTS
done

echo "----------------------------------------------------------------------" >> $TEMP_RESULTS
echo "统计信息：" >> $TEMP_RESULTS
echo "----------------------------------------------------------------------" >> $TEMP_RESULTS
echo "总实验数：$total_exp" >> $TEMP_RESULTS
echo "成功实验数：$success_exp" >> $TEMP_RESULTS
echo "总评测数：$total_eval" >> $TEMP_RESULTS
echo "成功评测数：$success_eval" >> $TEMP_RESULTS
echo "=======================================================================" >> $TEMP_RESULTS

# 显示汇总结果
cat $TEMP_RESULTS

echo ""
echo "======================================================================"
echo "所有C8实验已完成！"
echo "======================================================================"
echo "结果汇总文件：$TEMP_RESULTS"
echo ""
echo "下一步："
echo "1. 查看 $TEMP_RESULTS 文件中的详细结果"
echo "2. 对比C8系列与C1的结果（C1的F1为0.0583）"
echo "3. 如果C8_6的F1显著高于C1，考虑更新最优方案选择"
echo "======================================================================"

# 保存结果到本地文件
LOCAL_RESULTS="c8_results_summary.txt"
cat $TEMP_RESULTS > $LOCAL_RESULTS
echo "结果已保存到本地文件：$LOCAL_RESULTS"

# 清理临时文件
rm $TEMP_RESULTS