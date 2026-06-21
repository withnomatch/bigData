#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查StackOverflow Oracle Database数据的重复情况
"""

import json
from collections import Counter

def check_duplicates(file_path):
    print("="*70)
    print("检查StackOverflow Oracle Database数据重复情况")
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

    # 1. 检查question_id重复
    print("-"*70)
    print("[1] 检查question_id重复")
    print("-"*70)
    question_ids = [item.get('question_id') for item in data]
    id_counter = Counter(question_ids)
    id_duplicates = {k: v for k, v in id_counter.items() if v > 1}
    id_dup_count = len(id_duplicates)
    id_dup_records = sum(v - 1 for v in id_duplicates.values())

    print(f"唯一question_id数: {len(id_counter):,}")
    print(f"重复的question_id数: {id_dup_count:,}")
    print(f"重复记录数: {id_dup_records:,}")
    print(f"重复率: {id_dup_records / total_count * 100:.4f}%")

    if id_dup_count > 0:
        print(f"\n重复最多的question_id (Top 5):")
        for qid, count in sorted(id_duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  question_id={qid}: 重复{count}次")

    # 2. 检查title重复
    print()
    print("-"*70)
    print("[2] 检查title重复")
    print("-"*70)
    titles = [item.get('title', '') for item in data]
    title_counter = Counter(titles)
    title_duplicates = {k: v for k, v in title_counter.items() if v > 1}
    title_dup_count = len(title_duplicates)
    title_dup_records = sum(v - 1 for v in title_duplicates.values())

    print(f"唯一title数: {len(title_counter):,}")
    print(f"重复的title数: {title_dup_count:,}")
    print(f"重复记录数: {title_dup_records:,}")
    print(f"重复率: {title_dup_records / total_count * 100:.4f}%")

    if title_dup_count > 0:
        print(f"\n重复最多的title (Top 5):")
        for title, count in sorted(title_duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  \"{title[:60]}...\": 重复{count}次")

    # 3. 检查内容（title+body）重复
    print()
    print("-"*70)
    print("[3] 检查内容（title+body）重复")
    print("-"*70)
    contents = []
    for item in data:
        title = item.get('title', '')
        body = item.get('body', '')
        # 取前500字符避免过长
        content = (title + ' ' + body)[:500]
        contents.append(content)

    content_counter = Counter(contents)
    content_duplicates = {k: v for k, v in content_counter.items() if v > 1}
    content_dup_count = len(content_duplicates)
    content_dup_records = sum(v - 1 for v in content_duplicates.values())

    print(f"唯一内容数: {len(content_counter):,}")
    print(f"重复的内容数: {content_dup_count:,}")
    print(f"重复记录数: {content_dup_records:,}")
    print(f"重复率: {content_dup_records / total_count * 100:.4f}%")

    # 4. 汇总与建议
    print()
    print("="*70)
    print("汇总与建议")
    print("="*70)
    print()
    print(f"总记录数: {total_count:,}")
    print(f"question_id重复记录: {id_dup_records:,} ({id_dup_records/total_count*100:.4f}%)")
    print(f"title重复记录: {title_dup_records:,} ({title_dup_records/total_count*100:.4f}%)")
    print(f"内容重复记录: {content_dup_records:,} ({content_dup_records/total_count*100:.4f}%)")
    print()

    # 建议
    if id_dup_records / total_count > 0.01:
        print("【建议】question_id重复率 > 1%，建议基于question_id去重")
        print("  去重后记录数: {:,}".format(total_count - id_dup_records))
    elif id_dup_records / total_count > 0.001:
        print("【建议】question_id重复率在0.1%-1%之间，可选去重")
    else:
        print("【建议】question_id重复率 < 0.1%，无需去重")

    print()

    if title_dup_records / total_count > 0.05:
        print("【注意】title重复率 > 5%，但不建议基于title去重")
        print("  原因：不同用户可能用相同标题问不同问题，或用不同标题问相同问题")
        print("  建议：仅基于question_id去重")

    print()
    print("="*70)

    return {
        'total': total_count,
        'id_dup': id_dup_records,
        'title_dup': title_dup_records,
        'content_dup': content_dup_records
    }

if __name__ == "__main__":
    file_path = r"d:\OneDrive - shoutoutuoadi325\桌面\大数据\PJ\oracle_database_questions_jsonlines.json"
    result = check_duplicates(file_path)
