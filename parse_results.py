#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
解析运行结果文件，提取所有实验数据并生成汇总文件
"""

import re
import json

def parse_results(input_file, output_txt, output_csv):
    """解析运行结果文件"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建数据字典
    experiments_data = {}
    
    # 提取所有实验的聚类指标 - 使用更精确的模式
    # 模式：从"C Member Experiment: XX"开始，到下一个"Metrics content:"结束
    cluster_sections = re.split(r'Running Experiment:', content)
    
    for section in cluster_sections[1:]:  # 跳过第一个空section
        # 提取实验编号
        exp_match = re.search(r'(\w+)\n', section)
        if not exp_match:
            continue
        exp_id = exp_match.group(1)
        
        # 提取聚类指标JSON
        metrics_match = re.search(r'Metrics content:\s*\n\s*{\s*\n(.*?)\n\s*}', section, re.DOTALL)
        if metrics_match:
            try:
                metrics_json = '{' + metrics_match.group(1) + '}'
                metrics = json.loads(metrics_json)
                experiments_data[exp_id] = {
                    'cluster_metrics': metrics,
                    'eval_metrics': None
                }
            except Exception as e:
                print(f"Error parsing cluster metrics for {exp_id}: {e}")
                experiments_data[exp_id] = {
                    'cluster_metrics': None,
                    'eval_metrics': None
                }
    
    # 提取所有实验的评测指标
    eval_sections = re.split(r'Evaluating Experiment:', content)
    
    for section in eval_sections[1:]:  # 跳过第一个空section
        # 提取实验编号
        exp_match = re.search(r'(\w+)\n', section)
        if not exp_match:
            continue
        exp_id = exp_match.group(1)
        
        # 提取评测指标JSON
        eval_match = re.search(r'{\s*\n(.*?)\n\s*}\s*\n✓', section, re.DOTALL)
        if eval_match:
            try:
                eval_json = '{' + eval_match.group(1) + '}'
                eval_metrics = json.loads(eval_json)
                if exp_id in experiments_data:
                    experiments_data[exp_id]['eval_metrics'] = eval_metrics
                else:
                    experiments_data[exp_id] = {
                        'cluster_metrics': None,
                        'eval_metrics': eval_metrics
                    }
            except Exception as e:
                print(f"Error parsing eval metrics for {exp_id}: {e}")
    
    # 生成文本汇总文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("C组实验结果汇总\n")
        f.write("生成时间：2026-06-20\n")
        f.write("=" * 70 + "\n\n")
        
        for exp_id in sorted(experiments_data.keys()):
            f.write("-" * 70 + "\n")
            f.write(f"实验：{exp_id}\n")
            f.write("-" * 70 + "\n")
            
            # 聚类指标
            f.write("聚类指标：\n")
            if experiments_data[exp_id]['cluster_metrics']:
                f.write(json.dumps(experiments_data[exp_id]['cluster_metrics'], indent=2, ensure_ascii=False) + "\n")
            else:
                f.write("无数据\n")
            f.write("\n")
            
            # 评测指标
            f.write("评测指标：\n")
            if experiments_data[exp_id]['eval_metrics']:
                f.write(json.dumps(experiments_data[exp_id]['eval_metrics'], indent=2, ensure_ascii=False) + "\n")
            else:
                f.write("无数据\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write(f"统计信息：\n")
        f.write(f"实验总数：{len(experiments_data)}\n")
        f.write("=" * 70 + "\n")
    
    # 生成CSV汇总文件
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("实验编号,实验名称,词汇表大小,最小文档频率,最大文档频率,运行时间(秒),WSSSE/样本,最大簇占比,簇大小CV,Candidate Recall,Precision,Recall,F1\n")
        
        for exp_id in sorted(experiments_data.keys()):
            data = experiments_data[exp_id]
            
            # 从聚类指标提取数据
            cluster = data['cluster_metrics'] or {}
            eval_metrics = data['eval_metrics'] or {}
            
            exp_name = cluster.get('experiment_name', '')
            vocab_size = cluster.get('actual_vocab_size', cluster.get('max_features', ''))
            min_df = cluster.get('min_df', '')
            max_df = cluster.get('max_df', '')
            time_seconds = cluster.get('total_time_seconds', '')
            wssse_per_sample = cluster.get('wssse_per_sample', '')
            max_cluster_ratio = cluster.get('max_cluster_ratio', '')
            cluster_size_cv = cluster.get('cluster_size_cv', '')
            
            # 从评测指标提取数据
            candidate_recall = eval_metrics.get('candidate_recall', '')
            precision = eval_metrics.get('precision', '')
            recall = eval_metrics.get('recall', '')
            f1 = eval_metrics.get('f1', '')
            
            f.write(f"{exp_id},{exp_name},{vocab_size},{min_df},{max_df},{time_seconds},{wssse_per_sample},{max_cluster_ratio},{cluster_size_cv},{candidate_recall},{precision},{recall},{f1}\n")
    
    print(f"解析完成！")
    print(f"文本汇总文件：{output_txt}")
    print(f"CSV汇总文件：{output_csv}")
    print(f"实验总数：{len(experiments_data)}")
    
    # 打印调试信息
    print("\n实验数据检查：")
    for exp_id in sorted(experiments_data.keys()):
        data = experiments_data[exp_id]
        cluster_name = data['cluster_metrics'].get('experiment_name', 'N/A') if data['cluster_metrics'] else 'N/A'
        eval_recall = data['eval_metrics'].get('candidate_recall', 'N/A') if data['eval_metrics'] else 'N/A'
        print(f"{exp_id}: 聚类名称={cluster_name}, Candidate Recall={eval_recall}")

if __name__ == '__main__':
    parse_results(
        'c:\\Users\\Administrator\\Desktop\\PJ\\运行结果.md',
        'c:\\Users\\Administrator\\Desktop\\PJ\\PJ\\all_results_summary.txt',
        'c:\\Users\\Administrator\\Desktop\\PJ\\PJ\\all_results_summary.csv'
    )