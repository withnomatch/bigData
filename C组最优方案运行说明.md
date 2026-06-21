# C组最优方案运行说明

## 最优方案概述

根据C组实验结果，最优方案为**C1基线方案**：

- **特征构造**：title + body
- **TF-IDF参数**：vocab=5000, min_df=5, max_df=0.9
- **聚类参数**：K=45
- **去重参数**：similarity_threshold=0.55, dedup_top_per_cluster=80

**最优方案选择理由**：
- F1最高（0.0583）
- Precision完美（1.0）
- 运行时间最短（193秒）
- 参数设置合理，稳定性高
- 适合作为后续优化（D组语义特征）的基础

## 文件说明

### 1. `c_best_solution.py`
- C组最优方案的完整实现代码
- 包含数据预处理、特征提取、聚类、去重候选生成、评测等完整流程
- 独立运行，不依赖其他实验方案的代码

### 2. `run_c_best_solution.sh`
- 一键运行脚本
- 自动删除已存在的输出目录
- 运行最优方案并查看结果

## 运行步骤

### 步骤1: 上传文件到服务器

```powershell
# 在本地PowerShell执行
scp c_best_solution.py root@10.176.62.230:/root/
scp run_c_best_solution.sh root@10.176.62.230:/root/
```

### 步骤2: 连接服务器

```bash
ssh root@10.176.62.230
```

### 步骤3: 给脚本执行权限

```bash
chmod +x run_c_best_solution.sh
```

### 步骤4: 运行最优方案

#### 方式1: 使用一键运行脚本（推荐）

```bash
./run_c_best_solution.sh
```

#### 方式2: 手动运行

```bash
# 删除已存在的输出目录
hdfs dfs -rm -r hdfs://node5:9000/user/root/output/c_best_solution_c1

# 运行最优方案
spark-submit \
    --master local[4] \
    --driver-memory 2g \
    c_best_solution.py \
    --input hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300 \
    --output hdfs://node5:9000/user/root/output/c_best_solution_c1 \
    --annotation-pairs hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv \
    --k 45 \
    --max-features 5000 \
    --min-df 5 \
    --max-df 0.9 \
    --similarity-threshold 0.55 \
    --dedup-top-per-cluster 80
```

### 步骤5: 查看结果

```bash
# 查看实验报告
hdfs dfs -cat hdfs://node5:9000/user/root/output/c_best_solution_c1/experiment_report.json

# 查看聚类分配
hdfs dfs -ls hdfs://node5:9000/user/root/output/c_best_solution_c1/cluster_assignments

# 查看重复候选
hdfs dfs -ls hdfs://node5:9000/user/root/output/c_best_solution_c1/duplicate_candidates

# 查看聚类摘要
hdfs dfs -ls hdfs://node5:9000/user/root/output/c_best_solution_c1/cluster_summary
```

## 输出结果

### 1. `experiment_report.json`
包含完整的实验报告，包括：
- 实验参数配置
- 聚类指标（WSSSE、轮廓系数、簇分布等）
- 去重指标（Candidate Recall、Precision、Recall、F1等）
- 时间指标（特征提取时间、聚类时间、去重生成时间等）

### 2. `cluster_assignments`
每个问题的聚类分配：
- question_id
- title
- cluster_id

### 3. `duplicate_candidates`
疑似重复问题对：
- question_id_1
- question_id_2
- title_1
- title_2
- cluster_id
- similarity

### 4. `cluster_summary`
每个簇的问题数量统计：
- cluster_id
- count

## 预期结果

根据C组实验结果，预期输出：

| 指标 | 预期值 |
|------|--------|
| Candidate Recall | 0.56 |
| Precision | 1.0 |
| Recall | 0.03 |
| F1 | 0.0583 |
| WSSSE/样本 | 0.9255 |
| 簇大小CV | 2.4962 |
| 运行时间 | ~193秒 |

## 注意事项

1. **统一输入数据**：必须使用固定样本
   ```
   hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300
   ```

2. **统一K值**：使用B成员确定的最优K值 `K=45`

3. **统一评测数据**：使用300对标注数据评测
   ```
   hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv
   ```

4. **参数一致性**：确保参数与项目总结报告一致：
   - vocab=5000
   - min_df=5
   - max_df=0.9
   - similarity_threshold=0.55
   - dedup_top_per_cluster=80

5. **内存配置**：使用2g内存，避免内存不足
   ```
   --driver-memory 2g
   ```

6. **Spark模式**：使用local[4]模式，避免集群资源冲突
   ```
   --master local[4]
   ```

## 与其他实验的对比

### C1基线 vs C8_5（备选最优方案）

| 指标 | C1基线 | C8_5 |
|------|--------|------|
| F1 | 0.0583 | 0.0583 |
| Candidate Recall | 0.56 | 0.56 |
| Precision | 1.0 | 1.0 |
| Recall | 0.03 | 0.03 |
| max_df | 0.9 | 0.8 |

**选择C1的理由**：
- 参数更保守（max_df=0.9 vs 0.8）
- 稳定性更高（在更多数据集上表现稳定）
- 易于理解（默认参数）
- 适合作为后续优化基础

## 后续优化方向

C1基线方案是D组语义特征优化的基础：

1. **D组可以在C1基础上引入Word2Vec**
   - 保持相同的文本构造（title + body）
   - 保持相同的TF-IDF参数（vocab=5000, min_df=5, max_df=0.9）
   - 保持相同的聚类参数（K=45）
   - 添加Word2Vec特征，验证语义特征的效果

2. **D组可以对比不同特征组合**
   - TF-IDF only（C1基线）
   - Word2Vec only
   - TF-IDF + Word2Vec
   - TF-IDF加权Word2Vec

3. **D组需要保持实验一致性**
   - 使用相同的固定样本
   - 使用相同的K值
   - 使用相同的评测数据
   - 使用相同的输出格式

## 实验验证

运行完成后，请验证以下指标是否与预期一致：

1. **聚类指标验证**
   - WSSSE/样本 ≈ 0.9255
   - 簇大小CV ≈ 2.4962
   - 最大簇占比 ≈ 36%

2. **去重指标验证**
   - Candidate Recall ≈ 0.56
   - Precision ≈ 1.0
   - F1 ≈ 0.0583

3. **时间指标验证**
   - 运行时间 ≈ 193秒

如果指标与预期不一致，请检查：
- 输入数据是否正确（固定样本）
- 参数是否正确（vocab、min_df、max_df）
- K值是否正确（K=45）
- 评测数据是否正确（300对标注数据）

## 问题排查

### 问题1: 内存不足
**解决方案**：减小内存配置或使用更小的样本
```bash
--driver-memory 1g
```

### 问题2: 运行时间过长
**解决方案**：检查Spark配置和数据量
```bash
# 检查数据量
hdfs dfs -cat hdfs://node5:9000/user/root/data/fixed_eval_sample_10pct_plus_300 | wc -l
```

### 问题3: 输出目录已存在
**解决方案**：删除已存在的输出目录
```bash
hdfs dfs -rm -r hdfs://node5:9000/user/root/output/c_best_solution_c1
```

### 问题4: 评测指标异常
**解决方案**：检查标注数据路径和格式
```bash
# 检查标注数据
hdfs dfs -cat hdfs://node5:9000/user/root/data/dedup_pairs_300_reviewed.csv | head -10
```

## 完成标志

实验完成的标志：
1. 所有输出目录都已生成
2. experiment_report.json包含完整指标
3. 指标与预期一致
4. 无错误或异常输出

## 下一步工作

完成C组最优方案后：
1. 将结果填入项目总结报告
2. 为D组提供代码框架和实验说明
3. 准备项目总结报告的最终版本