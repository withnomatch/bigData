# StackOverflow 运维文本聚类 - 基准方案

## 方案概述

基准方案采用 **TF-IDF + K-Means** 实现StackOverflow运维文本聚类。

```
数据加载 → 文本预处理 → 特征提取(TF-IDF) → 聚类(K-Means) → 评估
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `baseline_clustering.py` | 基准方案主程序 |
| `run_baseline.sh` | 集群提交脚本 |
| `upload_baseline_data.sh` | 数据上传脚本 |

## 运行步骤

### 步骤1：上传数据到HDFS

```bash
./upload_baseline_data.sh
```

### 步骤2：提交Spark任务

```bash
./run_baseline.sh
```

### 步骤3：查看结果

```bash
hdfs dfs -ls /user/root/output/baseline_clustering
hdfs dfs -cat /user/root/output/baseline_clustering/cluster_summary/*
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--k` | 50 | 聚类数量 |
| `--max-features` | 5000 | 最大词汇表大小 |
| `--min-df` | 5 | 最小文档频率 |
| `--max-iter` | 30 | K-Means最大迭代次数 |

## 输出结果

| 文件 | 说明 |
|------|------|
| `cluster_assignments/` | 每个问题的聚类分配 |
| `cluster_summary/` | 聚类大小统计 |
| `top_questions_per_cluster/` | 每个聚类的Top10问题 |

## 评估指标

轮廓系数 (Silhouette Score): 范围 -1 到 1，越接近1聚类效果越好
- > 0.5: 优秀
- 0.3-0.5: 良好  
- 0.1-0.3: 一般
- < 0.1: 较差
