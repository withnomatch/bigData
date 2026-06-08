# 成员B：K值选择实验说明

本部分负责在 `K=40` 到 `K=60` 范围内选择相对合适的聚类数，用于支持基准方案和后续优化方案对比。

## 目标

- 对 StackOverflow 问题文本进行 TF-IDF 向量化。
- 在 `K=40..60` 范围内分别运行 K-Means。
- 记录每个 K 的轮廓系数、SSE/Inertia、运行时间、聚类规模分布。
- 根据轮廓系数、肘部趋势和聚类分布稳定性选择推荐 K。

## 输入数据格式

脚本支持 `.csv`、`.jsonl`、`.parquet` 三种格式。推荐 CSV 至少包含以下列之一：

- `text`
- `body`
- `title`
- `question`

如果同时存在 `title` 和 `body`，脚本会自动拼接为问题文本。

示例：

```csv
id,title,body,tags
1,How to fix Java NullPointerException?,<p>I got an error...</p>,java
```

## 运行方式

数据压缩包解压后路径示例：

```
PJ_data/PJ/oracle_database_questions_jsonlines.json
```

```powershell
cd D:\louis\fd\BDT\pj
python member_b/k_value_experiment.py `
  --input PJ_data/PJ/oracle_database_questions_jsonlines.json `
  --output member_b/outputs `
  --sample-ratio 0.1 `
  --silhouette-sample 3000
```

默认参数已与 `baseline_clustering.py` 对齐：`max_features=5000`，`min_df=5`，`max_iter=30`，`sample_ratio=0.1`。

## 输出文件

- `k_value_results.csv`：每个 K 的指标结果。
- `k_value_metrics.png`：轮廓系数、SSE、运行时间趋势图。
- `best_k_summary.txt`：推荐 K 值和解释。

## PPT建议结论写法

如果实验结果中某个 K 的轮廓系数最高，并且聚类规模没有严重失衡，可以选它作为推荐 K。

如果多个 K 的轮廓系数差异很小，优先选择更小的 K，这样聚类结构更简洁，后续解释和去重更方便。
