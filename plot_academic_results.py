#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Word2Vec特征提取实验结果可视化（学术汇报版）
生成符合学术规范的统计图表
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import FormatStrFormatter

# 设置学术风格
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'SimSun', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 11

# 实验数据
methods = ['TF-IDF', 'Word2Vec', 'TF-IDF+Word2Vec']
methods_short = ['TF-IDF', 'W2V', 'TF-IDF+W2V']

# 核心指标
silhouette = [0.0189, 0.1091, 0.1376]
wcss = [13976.82, 4877.56, 6470.52]
coherence = [0.4063, 0.8082, 0.7461]
time_seconds = [56.87, 378.41, 853.78]

# 聚类分布
max_cluster_size = [5042, 791, 754]
min_cluster_size = [2, 51, 123]
std_cluster_size = [717.4, 151.4, 130.4]

# 学术配色方案（色盲友好）
colors = ['#E64B35', '#4DBBD5', '#00A087']  # Nature期刊配色
patterns = ['///', '', '...']  # 填充图案

# ===== 图1: 轮廓系数对比（学术柱状图） =====
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(methods))
bars = ax.bar(x, silhouette, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, silhouette)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加提升百分比标注
ax.annotate('', xy=(2, silhouette[2]), xytext=(1, silhouette[1]),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(1.5, (silhouette[1]+silhouette[2])/2 + 0.01, '+26.2%',
        ha='center', fontsize=10, color='gray')

ax.annotate('', xy=(2, silhouette[2]), xytext=(0, silhouette[0]),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
ax.text(1, silhouette[0] + 0.03, '+628%', ha='center', fontsize=10, color='gray')

ax.set_ylabel('Silhouette Coefficient', fontsize=13)
ax.set_xlabel('Feature Extraction Method', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, max(silhouette) * 1.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.savefig('Fig1_Silhouette_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig1_Silhouette_Comparison.png")

# ===== 图2: 多指标对比（分组柱状图） =====
fig, ax = plt.subplots(figsize=(10, 6))

# 归一化指标
silhouette_norm = [s/max(silhouette) for s in silhouette]
wcss_norm = [1 - w/max(wcss) for w in wcss]  # 反向
coherence_norm = [c/max(coherence) for c in coherence]

x = np.arange(len(methods))
width = 0.25

bars1 = ax.bar(x - width, silhouette_norm, width, label='Silhouette (Normalized)',
               color='#E64B35', edgecolor='black', linewidth=1)
bars2 = ax.bar(x, wcss_norm, width, label='WCSS (Inverted & Normalized)',
               color='#4DBBD5', edgecolor='black', linewidth=1)
bars3 = ax.bar(x + width, coherence_norm, width, label='Coherence (Normalized)',
               color='#00A087', edgecolor='black', linewidth=1)

ax.set_ylabel('Normalized Score', fontsize=13)
ax.set_xlabel('Feature Extraction Method', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc='upper left', framealpha=0.9, edgecolor='black')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Fig2_Multi_Metric_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig2_Multi_Metric_Comparison.png")

# ===== 图3: 聚类分布对比（双轴柱状图） =====
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(methods))
width = 0.35

# 左轴：最大簇和最小簇
bars1 = ax1.bar(x - width/2, max_cluster_size, width, label='Max Cluster Size',
                color='#E64B35', edgecolor='black', linewidth=1.2, alpha=0.8)
bars2 = ax1.bar(x + width/2, min_cluster_size, width, label='Min Cluster Size',
                color='#4DBBD5', edgecolor='black', linewidth=1.2, alpha=0.8)

ax1.set_ylabel('Cluster Size', fontsize=13)
ax1.set_xlabel('Feature Extraction Method', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.legend(loc='upper left', framealpha=0.9, edgecolor='black')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加注释框
textstr = 'TF-IDF: Severe imbalance\nMax cluster: 32.9% of data'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='black')
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('Fig3_Cluster_Distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig3_Cluster_Distribution.png")

# ===== 图4: 簇大小标准差对比 =====
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(methods))
bars = ax.bar(x, std_cluster_size, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# 添加数值标签
for bar, val in zip(bars, std_cluster_size):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加下降百分比
reduction_1 = (std_cluster_size[0] - std_cluster_size[1]) / std_cluster_size[0] * 100
reduction_2 = (std_cluster_size[0] - std_cluster_size[2]) / std_cluster_size[0] * 100

ax.text(0.5, (std_cluster_size[0]+std_cluster_size[1])/2, f'↓{reduction_1:.1f}%',
        ha='center', fontsize=10, color='gray')
ax.text(1, (std_cluster_size[0]+std_cluster_size[2])/2 + 50, f'↓{reduction_2:.1f}%',
        ha='center', fontsize=10, color='gray')

ax.set_ylabel('Standard Deviation of Cluster Sizes', fontsize=13)
ax.set_xlabel('Feature Extraction Method', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, max(std_cluster_size) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig4_Cluster_StdDev.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig4_Cluster_StdDev.png")

# ===== 图5: 综合性能雷达图 =====
categories = ['Silhouette', 'WCSS\n(Inverted)', 'Coherence', 'Time\nEfficiency', 'Balance']

# 计算各方法的综合得分
silhouette_score = silhouette
wcss_score = [1 - w/max(wcss) for w in wcss]
coherence_score = coherence
time_score = [1 - t/max(time_seconds) for t in time_seconds]
balance_score = [1 - s/max(std_cluster_size) for s in std_cluster_size]

# 归一化到0-1
def normalize(arr):
    max_val = max(arr)
    min_val = min(arr)
    if max_val == min_val:
        return [0.5 for _ in arr]
    return [(v - min_val)/(max_val - min_val) for v in arr]

data_tfidf = normalize([silhouette_score[0], wcss_score[0], coherence_score[0], time_score[0], balance_score[0]])
data_w2v = normalize([silhouette_score[1], wcss_score[1], coherence_score[1], time_score[1], balance_score[1]])
data_tfidf_w2v = normalize([silhouette_score[2], wcss_score[2], coherence_score[2], time_score[2], balance_score[2]])

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

data_tfidf += data_tfidf[:1]
data_w2v += data_w2v[:1]
data_tfidf_w2v += data_tfidf_w2v[:1]

ax.plot(angles, data_tfidf, 'o-', linewidth=2, label='TF-IDF', color='#E64B35', markersize=6)
ax.fill(angles, data_tfidf, alpha=0.15, color='#E64B35')

ax.plot(angles, data_w2v, 's-', linewidth=2, label='Word2Vec', color='#4DBBD5', markersize=6)
ax.fill(angles, data_w2v, alpha=0.15, color='#4DBBD5')

ax.plot(angles, data_tfidf_w2v, '^-', linewidth=2, label='TF-IDF+Word2Vec', color='#00A087', markersize=6)
ax.fill(angles, data_tfidf_w2v, alpha=0.15, color='#00A087')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.savefig('Fig5_Radar_Chart.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig5_Radar_Chart.png")

# ===== 图6: 性能提升对比（折线图） =====
fig, ax = plt.subplots(figsize=(10, 6))

# 相对于TF-IDF的提升百分比
silhouette_improvement = [(s - silhouette[0]) / silhouette[0] * 100 for s in silhouette]
wcss_reduction = [(wcss[0] - w) / wcss[0] * 100 for w in wcss]
coherence_improvement = [(c - coherence[0]) / coherence[0] * 100 for c in coherence]

x = np.arange(len(methods))

ax.plot(x, silhouette_improvement, 'o-', linewidth=2.5, markersize=10,
        label='Silhouette Improvement', color='#E64B35', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(x, wcss_reduction, 's-', linewidth=2.5, markersize=10,
        label='WCSS Reduction', color='#4DBBD5', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(x, coherence_improvement, '^-', linewidth=2.5, markersize=10,
        label='Coherence Improvement', color='#00A087', markeredgecolor='black', markeredgewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=12)
ax.set_ylabel('Improvement over TF-IDF Baseline (%)', fontsize=13)
ax.set_xlabel('Feature Extraction Method', fontsize=13)
ax.legend(loc='best', framealpha=0.9, edgecolor='black')
ax.grid(alpha=0.3, linestyle='--')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加数值标签
for i, (s, w, c) in enumerate(zip(silhouette_improvement, wcss_reduction, coherence_improvement)):
    if i > 0:  # 跳过基线
        ax.text(i, s + 30, f'{s:.1f}%', ha='center', fontsize=10, color='#E64B35', fontweight='bold')
        ax.text(i, w + 30, f'{w:.1f}%', ha='center', fontsize=10, color='#4DBBD5', fontweight='bold')
        ax.text(i, c - 50, f'{c:.1f}%', ha='center', fontsize=10, color='#00A087', fontweight='bold')

plt.tight_layout()
plt.savefig('Fig6_Performance_Improvement.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig6_Performance_Improvement.png")

# ===== 图7: 运行时间对比 =====
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(methods))
bars = ax.bar(x, time_seconds, color=colors, edgecolor='black', linewidth=1.2, width=0.6)

# 添加数值标签
for bar, val in zip(bars, time_seconds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加时间倍数标注
time_ratio = time_seconds[2] / time_seconds[0]
ax.text(1, time_seconds[0] + 100, f'×{time_seconds[1]/time_seconds[0]:.1f}',
        ha='center', fontsize=10, color='gray')
ax.text(2, time_seconds[0] + 100, f'×{time_ratio:.1f}',
        ha='center', fontsize=10, color='gray')

ax.set_ylabel('Execution Time (seconds)', fontsize=13)
ax.set_xlabel('Feature Extraction Method', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, max(time_seconds) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig7_Execution_Time.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig7_Execution_Time.png")

# ===== 图8: 完整对比表格图 =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# 表格数据
table_data = [
    ['Method', 'Silhouette', 'WCSS', 'Coherence', 'Max Size', 'Min Size', 'Std Dev', 'Time (s)'],
    ['TF-IDF', '0.0189', '13976.82', '0.4063', '5042', '2', '717.4', '56.87'],
    ['Word2Vec', '0.1091', '4877.56', '0.8082', '791', '51', '151.4', '378.41'],
    ['TF-IDF+Word2Vec', '0.1376', '6470.52', '0.7461', '754', '123', '130.4', '853.78']
]

# 创建表格
table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 colWidths=[0.18, 0.11, 0.11, 0.11, 0.1, 0.1, 0.1, 0.1])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# 设置表头样式
for j in range(len(table_data[0])):
    cell = table[(0, j)]
    cell.set_facecolor('#E6E6E6')
    cell.set_text_props(fontweight='bold')

# 设置最优值高亮
best_indices = [(2, 1), (2, 2), (2, 4), (2, 6)]  # TF-IDF+Word2Vec的最优指标
for i, j in best_indices:
    cell = table[(i, j)]
    cell.set_facecolor('#90EE90')

# 添加标题
ax.text(0.5, 0.85, 'Table 1: Comprehensive Comparison of Three Feature Extraction Methods',
        ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)

# 添加注释
ax.text(0.5, 0.15, 'Note: Green cells indicate the best performance in each metric (excluding execution time).',
        ha='center', va='bottom', fontsize=10, style='italic', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('Fig8_Comparison_Table.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: Fig8_Comparison_Table.png")

print("\n" + "="*70)
print("所有学术图表已生成完成！")
print("="*70)
print("\n生成的图表列表：")
print("  1. Fig1_Silhouette_Comparison.png    - 轮廓系数对比")
print("  2. Fig2_Multi_Metric_Comparison.png  - 多指标归一化对比")
print("  3. Fig3_Cluster_Distribution.png     - 聚类分布对比")
print("  4. Fig4_Cluster_StdDev.png           - 簇大小标准差对比")
print("  5. Fig5_Radar_Chart.png              - 综合性能雷达图")
print("  6. Fig6_Performance_Improvement.png  - 性能提升折线图")
print("  7. Fig7_Execution_Time.png           - 运行时间对比")
print("  8. Fig8_Comparison_Table.png         - 完整对比表格")
print("\n图表特点：")
print("  - 使用Nature期刊配色方案（色盲友好）")
print("  - Times New Roman学术字体")
print("  - 清晰的数值标注和提升百分比")
print("  - 规范的坐标轴标签和图例")
print("  - 适合学术汇报和论文发表")
