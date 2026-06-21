#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析StackOverflow Oracle Database数据集的主题（标签）分布
"""

import json
from collections import Counter

def analyze_tags(file_path):
    print("="*70)
    print("分析StackOverflow Oracle Database数据集的主题分布")
    print("="*70)
    print()

    # 读取数据
    print("正在加载数据...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    total_count = len(data)
    print(f"总记录数: {total_count:,}")
    print()

    # 收集所有标签
    all_tags = []
    for item in data:
        tags = item.get('tags', [])
        all_tags.extend(tags)

    # 统计标签频率
    tag_counter = Counter(all_tags)
    unique_tags = len(tag_counter)
    total_tag_occurrences = len(all_tags)

    print("-"*70)
    print("[1] 标签总体统计")
    print("-"*70)
    print(f"唯一标签数: {unique_tags:,}")
    print(f"标签总出现次数: {total_tag_occurrences:,}")
    print(f"平均每条问题的标签数: {total_tag_occurrences/total_count:.2f}")
    print()

    # Top 50 最常见标签
    print("-"*70)
    print("[2] Top 50 最常见标签")
    print("-"*70)
    top_50_tags = tag_counter.most_common(50)
    for i, (tag, count) in enumerate(top_50_tags, 1):
        percentage = count / total_count * 100
        print(f"{i:2d}. {tag:30s} {count:6d} ({percentage:5.2f}%)")
    print()

    # 标签频率分布
    print("-"*70)
    print("[3] 标签频率分布")
    print("-"*70)
    freq_bins = {
        '>10000': 0,
        '5000-10000': 0,
        '1000-5000': 0,
        '500-1000': 0,
        '100-500': 0,
        '50-100': 0,
        '10-50': 0,
        '<10': 0
    }
    for tag, count in tag_counter.items():
        if count > 10000:
            freq_bins['>10000'] += 1
        elif count >= 5000:
            freq_bins['5000-10000'] += 1
        elif count >= 1000:
            freq_bins['1000-5000'] += 1
        elif count >= 500:
            freq_bins['500-1000'] += 1
        elif count >= 100:
            freq_bins['100-500'] += 1
        elif count >= 50:
            freq_bins['50-100'] += 1
        elif count >= 10:
            freq_bins['10-50'] += 1
        else:
            freq_bins['<10'] += 1

    for freq_range, count in freq_bins.items():
        print(f"{freq_range:15s}: {count:4d} 个标签")
    print()

    # 估计主要主题数量
    print("-"*70)
    print("[4] 主要主题估计")
    print("-"*70)

    # 方法1: 出现次数>100的标签
    major_tags_100 = sum(1 for tag, count in tag_counter.items() if count >= 100)
    print(f"出现次数≥100的标签数: {major_tags_100}")

    # 方法2: 出现次数>500的标签
    major_tags_500 = sum(1 for tag, count in tag_counter.items() if count >= 500)
    print(f"出现次数≥500的标签数: {major_tags_500}")

    # 方法3: 出现次数>1000的标签
    major_tags_1000 = sum(1 for tag, count in tag_counter.items() if count >= 1000)
    print(f"出现次数≥1000的标签数: {major_tags_1000}")

    # 方法4: 覆盖80%问题的标签数
    sorted_tags = tag_counter.most_common()
    cumulative = 0
    tags_for_80 = 0
    for tag, count in sorted_tags:
        cumulative += count
        tags_for_80 += 1
        if cumulative >= total_tag_occurrences * 0.8:
            break
    print(f"覆盖80%标签出现次数的标签数: {tags_for_80}")

    print()

    # 核心Oracle相关标签
    print("-"*70)
    print("[5] 核心Oracle相关标签")
    print("-"*70)
    oracle_tags = [
        'oracle', 'oracle-database', 'oracle11g', 'oracle10g', 'oracle12c',
        'plsql', 'pl/sql', 'sql', 'sqlplus',
        'jdbc', 'odp.net', 'oracle-jdbc',
        'ora-error', 'oracle-apex', 'oracle-sql-developer'
    ]
    for tag in oracle_tags:
        if tag in tag_counter:
            count = tag_counter[tag]
            percentage = count / total_count * 100
            print(f"{tag:25s}: {count:6d} ({percentage:5.2f}%)")
    print()

    # 总结
    print("="*70)
    print("总结")
    print("="*70)
    print()
    print(f"数据集包含 {unique_tags} 个不同的标签")
    print(f"其中主要标签（出现≥100次）: {major_tags_100} 个")
    print(f"其中核心标签（出现≥500次）: {major_tags_500} 个")
    print(f"其中高频标签（出现≥1000次）: {major_tags_1000} 个")
    print()
    print("【建议】")
    if major_tags_500 > 0:
        print(f"  基于标签分析，数据集的主要主题数量约为 {major_tags_500}-{major_tags_100} 个")
        print(f"  推荐K值范围: {max(20, major_tags_500//2)}-{min(100, major_tags_100)}")
    print()

    return {
        'unique_tags': unique_tags,
        'major_tags_100': major_tags_100,
        'major_tags_500': major_tags_500,
        'major_tags_1000': major_tags_1000
    }

if __name__ == "__main__":
    file_path = r"d:\OneDrive - shoutoutuoadi325\桌面\大数据\PJ\oracle_database_questions_jsonlines.json"
    result = analyze_tags(file_path)
