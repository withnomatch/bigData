# 聚类结构优化

## 任务说明

聚类结构优化，具体包括：

1. **层次聚类对比**：BisectingKMeans vs K-Means（算法结构对比）
2. **聚类后处理**：合并小聚类、生成聚类标签

> **注**：距离度量优化（欧氏 vs 余弦）、初始化方法、迭代次数调优属于成员F（聚类执行参数优化）的范围，本模块不涉及。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `structure_optimization.py` | 本模块主脚本（集群版，Spark 2.4.8） |
| `run_structure_opt.sh` | 集群一键提交脚本，运行完自动生成 `experiment_report.txt` |

---

## 与基准方案的关系

### 基准方案做了什么

```
原始数据 (HDFS)
  → 文本预处理（HTML清洗、分词、去停用词）
  → TF-IDF 特征提取（词汇量5000，IDF降权，L2归一化）
  → K-Means 聚类（K=50，欧氏距离，随机初始化）
  → 输出聚类分配结果
```

**基准方案问题**：聚类分布极度不均，最大聚类 5,874 条（占总量 38%），存在大量"垃圾桶聚类"。

### 本模块在哪里改进

**保持不变**：数据读取、文本预处理、TF-IDF特征提取、K=50、欧氏距离。

**改变的只有**：**算法结构**——将 K-Means 替换为 BisectingKMeans，并增加聚类后处理步骤。

这样对比的就是**纯算法结构差异**（扁平聚类 vs 层次聚类），排除了其他变量的干扰。

---

## 实现原理

### 实验1：K-Means（对照组，复现基准方案）

随机选取 K 个初始中心，反复迭代：
- 将每个点分配到最近中心（欧氏距离）
- 重新计算每个聚类的均值中心
- 直到收敛

**缺陷**：对初始中心敏感，全局最优迭代方式容易形成"一个大聚类吸收大量样本"的局面。

### 实验2：BisectingKMeans（实验组，层次聚类）

```
初始状态: 所有数据 = 1 个大聚类
    ↓ 从当前所有聚类中选出体积最大的一个
    ↓ 对该聚类执行 2-Means，一分为二
    ↓ 重复上述步骤，直到聚类数量达到 K
```

**原理优势**：自顶向下的层次划分，每次切分都只处理最不均匀的那个聚类，**天然保证结果分布更均匀**。

### 聚类后处理

#### ① 生成聚类标签

```
每个聚类有一个中心向量（TF-IDF 空间的质心）
中心向量各维度对应词汇表中的一个词，值越大越重要
取权重最大的 Top-5 词拼接成语义标签
示例: "date | month | convert | timestamp | format"
```

#### ② 合并小聚类

```
计算全部聚类的平均大小 avg_size
小聚类阈值 = avg_size × 10%
对每个小聚类:
    计算其中心到所有大聚类中心的欧氏距离
    将其所有问题重新分配到最近的大聚类
    从聚类列表中移除该小聚类
```

---

## 运行方法

### 集群运行（正式结果）

```bash
# 上传脚本到集群
scp structure_optimization.py run_structure_opt.sh root@10.176.62.230:/root/

# 登录集群
ssh root@10.176.62.230

# 后台运行
chmod +x run_structure_opt.sh
nohup ./run_structure_opt.sh > structure_opt.log 2>&1 &

# 查看进度
tail -f structure_opt.log

# 完成后查看纯净实验报告
cat experiment_report.txt
```

### 本地测试（快速验证逻辑）

```bash
python local_structure_test.py
```

---

## 实验结果

> **实验条件**：采样比 10%（约 15,441 条），K=50，TF-IDF 词汇表 5000，**两种算法均使用欧氏距离**（与基准方案一致，控制变量，仅对比算法结构），总耗时 2.5 分钟

### 算法结构对比（核心结果）

| 指标 | K-Means（对照组）| BisectingKMeans（实验组）| 说明 |
|------|:---:|:---:|------|
| 训练时间 | 15.97s | 39.09s | K-Means 快 2.4 倍 |
| WSSSE | **14,180.97** | 14,201.62 | 相差仅 0.15%，基本持平 |
| 最大聚类大小 | 5,129 | **2,236** | BisectingKMeans 缩小 56% |
| 最小聚类大小 | 1 | **43** | BisectingKMeans 无单样本聚类 |
| 标准差 | 745.0 | **411.0** | BisectingKMeans 更集中 |
| 变异系数 CV | 2.4124 | **1.3310** | BisectingKMeans 均匀度提升 **44.8%** |
| 小聚类数（< avg×10%）| 14 个（占 28%）| **0 个** | BisectingKMeans 完全消除 |
| 合并后有效聚类数 | **36**（14个被合并）| **50**（无需合并）| BisectingKMeans 无损失 |

**综合结论：BisectingKMeans 在分布均匀性上全面优于 K-Means（CV 低 44.8%、无小聚类），代价是训练时间多 2.4 倍，WSSSE 几乎持平（0.15%差异可忽略）。**

### 与基准方案的改进

| 指标 | 基准方案（K-Means）| 结构优化后（BisectingKMeans）| 提升 |
|------|:---:|:---:|:---:|
| 最大聚类大小 | 5,874 | **2,236** | 缩小 **62%** |
| 小聚类数量 | 多个 | **0 个** | 完全消除 |
| 有效聚类数 | < 50（含小聚类）| **50** | 无损失 |

*基准方案数据来自 `log.txt`（基准实验日志）。*

### 聚类标签示例（后处理生成）

**K-Means 前5大聚类：**

| 聚类 | 关键词标签 | 大小 | 语义解读 |
|------|-----------|:---:|---------|
| Cluster 36 | `id \| query \| table \| data \| sql` | 5,129 | 通用SQL查询（过于宽泛）|
| Cluster 44 | `column \| name \| table \| value \| insert` | 1,494 | 列操作类 |
| Cluster 6 | `connect \| server \| database \| client \| connection` | 1,015 | 数据库连接类 |
| Cluster 28 | `1 \| 2 \| 01 \| 3 \| 0` | 907 | 噪声聚类（数字混杂）|
| Cluster 10 | `connection \| string \| cmd \| jdbc \| new` | 594 | JDBC连接类 |

**BisectingKMeans 前5大聚类：**

| 聚类 | 关键词标签 | 大小 | 语义解读 |
|------|-----------|:---:|---------|
| Cluster 0 | `query \| select \| count \| table \| 1` | 2,236 | SELECT查询类 |
| Cluster 26 | `database \| data \| db \| oracle \| sql` | 1,338 | 数据库综合类 |
| Cluster 27 | `sql \| error \| data \| oracle \| query` | 1,301 | SQL报错类 |
| Cluster 1 | `column \| 1 \| value \| 2 \| table` | 1,269 | 列值操作类 |
| Cluster 3 | `id \| p \| join \| name \| 1` | 1,032 | JOIN关联查询类 |

> 观察：BisectingKMeans 最大聚类仅 2,236 条，而 K-Means 最大聚类达 5,129 条（集中了全部数据的 33%），语义更聚焦。

---

## 分析结论

### 主要发现

1. **算法结构是影响聚类分布的核心因素**  
   在完全相同的参数条件下（欧氏距离、K=50、同一数据集），BisectingKMeans 的聚类分布均匀度（CV）比 K-Means 提升 **44.8%**，最大聚类缩小 **56%**，小聚类数量从 14 个降至 0。这直接证明了层次结构（自顶向下切分）在文本聚类中的结构性优势。

2. **K-Means 的"垃圾桶聚类"问题**  
   K-Means 产生 14 个小聚类（占比 28%），合并后有效聚类数仅剩 36，实际上没有充分利用 K=50 的设定。BisectingKMeans 50 个聚类全部保留，质量更高。

3. **WSSSE 几乎持平，不能作为结构优劣的唯一标准**  
   K-Means 的 WSSSE 略低（14,180 vs 14,201，差 0.15%），但这是以牺牲分布均匀性为代价的——大聚类压低了整体误差，却掩盖了聚类质量问题。

4. **聚类后处理效果**  
   - **标签生成**：每个聚类自动提取 Top-5 关键词，语义清晰可读（如"日期时间类"、"JDBC连接类"）  
   - **小聚类合并**：K-Means 的 14 个小聚类被合并到语义最近的大聚类，BisectingKMeans 无需合并

### 在项目整体框架中的定位

```
聚类优化框架
├── 结构层面（本模块，成员E）
│   ├── 算法选择：BisectingKMeans vs K-Means
│   └── 聚类后处理：合并小聚类、生成聚类标签
└── 执行层面（成员F）
    ├── 初始化方法：随机 vs K-Means++
    ├── 距离度量：欧氏 vs 余弦
    └── 迭代次数调优
```

结构优化与执行优化正交分离，便于独立分析和组合对比。

---

## 输出文件说明（HDFS）

```
/user/root/output/structure_optimization/
├── comparison_table/          # 两种算法对比指标（CSV）
├── kmeans_assignments/        # K-Means 聚类分配结果（含标签）
├── kmeans_summary/            # K-Means 聚类摘要
├── bisecting_kmeans_assignments/   # BisectingKMeans 聚类分配结果（含标签）
└── bisecting_kmeans_summary/       # BisectingKMeans 聚类摘要
```
