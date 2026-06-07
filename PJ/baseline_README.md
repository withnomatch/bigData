# StackOverflow Oracle 问题聚类基准方案

## 方案概述

基准方案采用 **TF-IDF + KMeans**：

```text
数据加载 → 文本预处理 → TF-IDF → L2 归一化 → KMeans → 评价与输出
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `baseline_clustering.py` | Spark 2.0 基准聚类程序 |
| `convert_json.py` | JSON 数组转 JSONLines |
| `upload_baseline_data.sh` | 数据上传脚本 |
| `run_baseline.sh` | 集群提交与结果检查脚本 |
| `基准实验报告.md` | 实验方法、结果与结论 |
| `baseline_experiment_summary.csv` | 实验指标汇总 |

## 运行步骤

```bash
python convert_json.py \
  StackOverFlow_Oracle_Database/oracle_database_questions.json \
  oracle_database_questions_jsonlines.json

./upload_baseline_data.sh
./run_baseline.sh
```

## 默认参数

| 参数 | 默认值 |
| --- | ---: |
| `--k` | 50 |
| `--max-features` | 5000 |
| `--min-df` | 5 |
| `--max-iter` | 30 |
| `--sample-ratio` | 0.1（提交脚本） |
| 随机种子 | 42 |

## 输出结果

| 目录 | 内容 |
| --- | --- |
| `cluster_assignments` | 问题与聚类编号 |
| `cluster_summary` | 各簇问题数量 |
| `top_questions_per_cluster` | 每簇分数最高的 10 个问题 |
| `metrics` | WSSSE、簇大小 CV、归一化簇熵等指标 |

Spark 2.0 不包含 `ClusteringEvaluator`，因此本方案不伪造轮廓系数。评价以 WSSSE、最大簇占比、簇大小 CV、归一化簇熵和运行时间为主。
