#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Word2Vec特征提取实验结果可视化
生成柱状图、折线图等统计图表
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 实验数据
methods = ['TF-IDF', 'Word2Vec', 'TF-IDF+Word2Vec']

# 核心指标
silhouette = [0.0189, 0.1091, 0.1376]
wcss = [13976.82, 4877.56, 6470.52]
coherence = [0.4063, 0.8082, 0.7461]
time_seconds = [56.87, 378.41, 853.78]

# 聚类分布
max_cluster_size = [5042, 791, 754]
min_cluster_size = [2, 51, 123]
std_cluster_size = [717.4, 151.4, 130.4]

# 颜色方案
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# ===== 图1: 轮廓系数对比柱状图 =====
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, silhouette, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('轮廓系数 (Silhouette)', fontsize=14)
plt.xlabel('特征提取方法', fontsize=14)
plt.title('三种方案的轮廓系数对比', fontsize=16, fontweight='bold')
plt.ylim(0, max(silhouette) * 1.2)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, silhouette):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('图1_轮廓系数对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图1_轮廓系数对比.png")

# ===== 图2: WCSS对比柱状图 =====
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, wcss, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('WCSS (簇内平方和)', fontsize=14)
plt.xlabel('特征提取方法', fontsize=14)
plt.title('三种方案的WCSS对比', fontsize=16, fontweight='bold')
plt.ylim(0, max(wcss) * 1.1)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, wcss):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('图2_WCSS对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图2_WCSS对比.png")

# ===== 图3: 语义一致性对比柱状图 =====
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, coherence, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('语义一致性 (Coherence)', fontsize=14)
plt.xlabel('特征提取方法', fontsize=14)
plt.title('三种方案的语义一致性对比', fontsize=16, fontweight='bold')
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, coherence):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('图3_语义一致性对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图3_语义一致性对比.png")

# ===== 图4: 运行时间对比柱状图 =====
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, time_seconds, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('运行时间 (秒)', fontsize=14)
plt.xlabel('特征提取方法', fontsize=14)
plt.title('三种方案的运行时间对比', fontsize=16, fontweight='bold')
plt.ylim(0, max(time_seconds) * 1.1)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, time_seconds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('图4_运行时间对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图4_运行时间对比.png")

# ===== 图5: 聚类分布对比（最大/最小簇大小） =====
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, max_cluster_size, width, label='最大簇大小',
               color='#FF6B6B', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, min_cluster_size, width, label='最小簇大小',
               color='#4ECDC4', edgecolor='black', linewidth=1.5)

ax.set_ylabel('簇大小', fontsize=14)
ax.set_xlabel('特征提取方法', fontsize=14)
ax.set_title('三种方案的聚类分布对比', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('图5_聚类分布对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图5_聚类分布对比.png")

# ===== 图6: 簇大小标准差对比 =====
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, std_cluster_size, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('簇大小标准差', fontsize=14)
plt.xlabel('特征提取方法', fontsize=14)
plt.title('三种方案的簇大小标准差对比（越小越均衡）', fontsize=16, fontweight='bold')
plt.ylim(0, max(std_cluster_size) * 1.2)
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, std_cluster_size):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('图6_簇大小标准差对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图6_簇大小标准差对比.png")

# ===== 图7: 综合指标对比（归一化） =====
# 归一化处理（越高越好）
silhouette_norm = [s/max(silhouette) for s in silhouette]
wcss_norm = [1 - w/max(wcss) for w in wcss]  # WCSS越小越好，反转
coherence_norm = [c/max(coherence) for c in coherence]
time_norm = [1 - t/max(time_seconds) for t in time_seconds]  # 时间越短越好，反转

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(methods))
width = 0.2

bars1 = ax.bar(x - 1.5*width, silhouette_norm, width, label='轮廓系数',
               color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x - 0.5*width, wcss_norm, width, label='WCSS (反向)',
               color='#4ECDC4', edgecolor='black')
bars3 = ax.bar(x + 0.5*width, coherence_norm, width, label='语义一致性',
               color='#45B7D1', edgecolor='black')
bars4 = ax.bar(x + 1.5*width, time_norm, width, label='运行时间 (反向)',
               color='#96CEB4', edgecolor='black')

ax.set_ylabel('归一化指标值', fontsize=14)
ax.set_xlabel('特征提取方法', fontsize=14)
ax.set_title('三种方案的综合指标对比（归一化，越高越好）', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图7_综合指标对比.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图7_综合指标对比.png")

# ===== 图8: 雷达图 =====
# 准备雷达图数据
categories = ['轮廓系数', 'WCSS(反向)', '语义一致性', '时间效率', '聚类均衡性']

# 计算各方法的综合得分
silhouette_score = silhouette
wcss_score = [1 - w/max(wcss) for w in wcss]
coherence_score = coherence
time_score = [1 - t/max(time_seconds) for t in time_seconds]
balance_score = [1 - s/max(std_cluster_size) for s in std_cluster_size]

# 归一化到0-1
def normalize(arr):
    max_val = max(arr)
    return [v/max_val for v in arr]

data_tfidf = normalize([silhouette_score[0], wcss_score[0], coherence_score[0], time_score[0], balance_score[0]])
data_w2v = normalize([silhouette_score[1], wcss_score[1], coherence_score[1], time_score[1], balance_score[1]])
data_tfidf_w2v = normalize([silhouette_score[2], wcss_score[2], coherence_score[2], time_score[2], balance_score[2]])

# 绘制雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

data_tfidf += data_tfidf[:1]
data_w2v += data_w2v[:1]
data_tfidf_w2v += data_tfidf_w2v[:1]

ax.plot(angles, data_tfidf, 'o-', linewidth=2, label='TF-IDF', color='#FF6B6B')
ax.fill(angles, data_tfidf, alpha=0.15, color='#FF6B6B')

ax.plot(angles, data_w2v, 'o-', linewidth=2, label='Word2Vec', color='#4ECDC4')
ax.fill(angles, data_w2v, alpha=0.15, color='#4ECDC4')

ax.plot(angles, data_tfidf_w2v, 'o-', linewidth=2, label='TF-IDF+Word2Vec', color='#45B7D1')
ax.fill(angles, data_tfidf_w2v, alpha=0.15, color='#45B7D1')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title('三种方案的综合性能雷达图', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('图8_综合性能雷达图.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图8_综合性能雷达图.png")

# ===== 图9: 性能提升折线图 =====
fig, ax = plt.subplots(figsize=(12, 7))

# 相对于TF-IDF的提升百分比
silhouette_improvement = [(s - silhouette[0]) / silhouette[0] * 100 for s in silhouette]
wcss_improvement = [(wcss[0] - w) / wcss[0] * 100 for w in wcss]  # WCSS降低是提升
coherence_improvement = [(c - coherence[0]) / coherence[0] * 100 for c in coherence]

x = np.arange(len(methods))

ax.plot(x, silhouette_improvement, 'o-', linewidth=3, markersize=10,
        label='轮廓系数提升', color='#FF6B6B')
ax.plot(x, wcss_improvement, 's-', linewidth=3, markersize=10,
        label='WCSS降低', color='#4ECDC4')
ax.plot(x, coherence_improvement, '^-', linewidth=3, markersize=10,
        label='语义一致性提升', color='#45B7D1')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=13)
ax.set_ylabel('相对于TF-IDF的提升百分比 (%)', fontsize=14)
ax.set_xlabel('特征提取方法', fontsize=14)
ax.set_title('各方案相对于TF-IDF的性能提升', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# 添加数值标签
for i, (s, w, c) in enumerate(zip(silhouette_improvement, wcss_improvement, coherence_improvement)):
    ax.text(i, s + 30, f'{s:.1f}%', ha='center', fontsize=11, color='#FF6B6B', fontweight='bold')
    ax.text(i, w + 30, f'{w:.1f}%', ha='center', fontsize=11, color='#4ECDC4', fontweight='bold')
    ax.text(i, c - 50, f'{c:.1f}%', ha='center', fontsize=11, color='#45B7D1', fontweight='bold')

plt.tight_layout()
plt.savefig('图9_性能提升折线图.png', dpi=300, bbox_inches='tight')
plt.close()
print("已生成: 图9_性能提升折线图.png")

print("\n" + "="*60)
print("所有图表已生成完成！")
print("="*60)
print("\n生成的图表列表：")
print("  1. 图1_轮廓系数对比.png")
print("  2. 图2_WCSS对比.png")
print("  3. 图3_语义一致性对比.png")
print("  4. 图4_运行时间对比.png")
print("  5. 图5_聚类分布对比.png")
print("  6. 图6_簇大小标准差对比.png")
print("  7. 图7_综合指标对比.png")
print("  8. 图8_综合性能雷达图.png")
print("  9. 图9_性能提升折线图.png")
