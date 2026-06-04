# F：聚类执行参数优化

## 任务目标

本部分负责在基准方案 `TF-IDF + K-Means` 的基础上，只调整 K-Means 执行参数，重点比较：

| 优化维度 | 对比方案 | 说明 |
| --- | --- | --- |
| 初始化方法 | `random` vs `k-means||` | Spark ML 中的 `k-means||` 是面向分布式环境的 K-Means++ 可扩展初始化方式 |
| 距离度量 | 欧氏距离 vs 余弦距离等价方案 | Spark 2.0 的 KMeans 不直接支持 `distanceMeasure` 参数，因此用原始 TF-IDF 表示欧氏距离，用 L2 归一化 TF-IDF 上的欧氏距离近似余弦距离 |
| 迭代次数 | `maxIter=10/20/30` | 观察聚类质量提升与运行时间增长是否匹配 |
| 运行次数 | 多随机种子 `42/2026/3407` | Spark ML 无独立 `runs` 参数，用多 seed 重复运行模拟多次初始化，选择效果最好的结果 |

## 实现文件

| 文件 | 作用 |
| --- | --- |
| `f_clustering_param_optimization.py` | F 任务主程序，复用基准预处理与 TF-IDF 特征，输出参数对比结果 |
| `run_f_optimization.sh` | 集群提交脚本，默认使用 10% 样本、K=50、聚焦实验计划 |

## 实验设计

脚本采用“聚焦实验”而不是全量网格搜索，避免组合爆炸导致运行时间过长。默认会运行：

1. 初始化方法对比：固定 `distance=cosine`、`maxIter=20`、`seed=42`，比较 `random` 与 `k-means||`。
2. 距离方式对比：固定 `init=k-means||`、`maxIter=20`、`seed=42`，比较欧氏距离与余弦等价方案。
3. 迭代次数对比：固定 `init=k-means||`、`distance=cosine`、`seed=42`，比较 10、20、30 次迭代。
4. 多次运行对比：固定推荐配置，使用多个随机种子重复运行，选择近似轮廓系数更高且耗时可接受的配置。

## 评价指标

| 指标 | 含义 |
| --- | --- |
| `training_time_sec` | K-Means 训练耗时，用于回答助教关于运行时间的问题 |
| `training_cost` | Spark KMeans 的类内平方误差，越低表示簇内越紧凑 |
| `center_silhouette` | 基于样本到聚类中心距离的近似轮廓系数，越高表示区分度越好 |
| `largest_cluster_ratio` | 最大簇占比，用于观察是否出现过大的“杂项簇” |
| `singleton_clusters` | 单样本簇数量，过多说明聚类过碎 |

说明：原基准日志显示 Spark 2.0 不提供 `ClusteringEvaluator`，所以基准中的 `Silhouette Score` 实际为 `0.0000` 占位值。本实验脚本补充了中心近似轮廓系数，便于横向比较参数。

## 运行命令

在 node5 上启动 HDFS 与 Spark 后，上传 `f_clustering_param_optimization.py` 和 `run_f_optimization.sh` 到同一目录，然后执行：

```bash
chmod +x run_f_optimization.sh
./run_f_optimization.sh
```

若只想快速验证脚本，可以降低样本比例与特征规模：

```bash
spark-submit \
    --master spark://10.176.62.230:7077 \
    --driver-memory 2g \
    --executor-memory 4g \
    --executor-cores 2 \
    --num-executors 2 \
    f_clustering_param_optimization.py \
    --input /user/root/data/oracle_database_questions_jsonlines.json \
    --output /user/root/output/f_param_optimization_quick \
    --k 50 \
    --max-features 3000 \
    --min-df 5 \
    --sample-ratio 0.03 \
    --plan focused \
    --silhouette-sample 1000 \
    --skip-cost
```

## 输出结果

| HDFS 输出目录 | 内容 |
| --- | --- |
| `experiment_results/` | 每组参数的运行时间、代价、近似轮廓系数、聚类分布指标 |
| `best_config/` | 根据近似轮廓系数优先、耗时和 cost 辅助选择出的最佳配置 |
| `best_cluster_summary/` | 最佳配置下每个簇的大小 |
| `best_cluster_assignments/` | 最佳配置下每个问题的聚类编号 |
| `best_top_questions_per_cluster/` | 最佳配置下每个簇的 Top 问题，便于人工分析语义一致性 |

## 运行验证记录

已在 node5 的 Spark 2.0 环境上完成小样本 smoke test，验证脚本可以读取 HDFS 数据、完成 TF-IDF 特征构建、运行 K-Means 参数实验并写回 HDFS。

验证命令使用 `sample-ratio=0.001`、`k=5`、`max-features=300`、`maxIter=1`、`--skip-cost`。实际读取总记录数 152758，抽样后有效记录 141；TF-IDF 构建与缓存耗时约 179.76 秒，单组 K-Means 训练耗时约 0.54 秒，结果已写入 `hdfs://node5:9000/user/root/output/f_param_optimization_smoke2`。

## 可用于 PPT 的结论表述

基准方案在 10% 样本、K=50、`maxIter=30` 下，K-Means 训练耗时约 1463 秒，总耗时约 1894 秒，且最大簇包含 5874/15344 个问题，占比约 38.3%。这说明基准方案可以跑通，但存在运行时间较长、簇分布不均衡的问题。

F 部分将聚类执行优化拆成初始化、距离表示、迭代次数、多次运行四个可控变量，并缓存 TF-IDF 特征，避免每组实验重复做文本特征转换。这样既能比较聚类质量，也能量化运行时间开销。

建议最终采用：`k-means||` 初始化、L2 归一化 TF-IDF 的余弦等价距离、`maxIter=20~30`、多 seed 取最优。若 `maxIter=20` 与 `30` 的近似轮廓系数差距很小，则优先选择 `20`，因为它能显著减少运行时间，更适合课程项目中的多组对比实验。

## PPT 页建议

1. 原理页：说明 KMeans 执行参数会影响初始中心、收敛速度和最终局部最优；Spark 2.0 中用 `k-means||` 作为分布式 K-Means++ 初始化。
2. 实验页：放 `experiment_results` 中的核心表格，列出 init、distance、maxIter、seed、time、cost、center_silhouette、largest_cluster_ratio。
3. 分析页：强调运行时间与聚类效果的折中，说明最终选择不是单纯追求最多迭代，而是在效果接近时选择更快配置。
