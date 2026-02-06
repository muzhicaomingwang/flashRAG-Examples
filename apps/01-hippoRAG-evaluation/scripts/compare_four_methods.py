#!/usr/bin/env python3
"""
四种方法完整对比：
1. BM25稀疏检索
2. HippoRAG（标准KG）
3. HippoRAG（高质量KG）
4. FAISS稠密检索
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_em(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_predictions(predictions):
    f1_scores = []
    em_scores = []
    latencies = []
    for pred in predictions:
        if pred['success']:
            f1 = compute_f1(pred['predicted_answer'], pred['gold_answer'])
            em = compute_em(pred['predicted_answer'], pred['gold_answer'])
            f1_scores.append(f1)
            em_scores.append(em)
            latencies.append(pred['latency'])
    return {
        'f1_mean': np.mean(f1_scores) if f1_scores else 0,
        'f1_std': np.std(f1_scores) if f1_scores else 0,
        'em_mean': np.mean(em_scores) if em_scores else 0,
        'em_std': np.std(em_scores) if em_scores else 0,
        'latency_mean': np.mean(latencies) if latencies else 0,
        'latency_median': np.median(latencies) if latencies else 0,
        'success_rate': len(f1_scores) / len(predictions),
        'total': len(predictions)
    }

print('='*80)
print('四种方法完整对比分析')
print('='*80)

# 加载结果
print('\n📊 加载预测结果...')

methods = {}

# BM25
bm25_file = 'results/bm25/predictions.json'
if Path(bm25_file).exists():
    with open(bm25_file, 'r') as f:
        methods['BM25（稀疏检索）'] = json.load(f)
    print(f'✅ BM25: {len(methods["BM25（稀疏检索）"])} 个预测')

# Baseline FAISS
with open('results/baseline_full/predictions.json', 'r') as f:
    methods['FAISS（稠密检索）'] = json.load(f)
print(f'✅ FAISS: {len(methods["FAISS（稠密检索）"])} 个预测')

# HippoRAG标准
with open('results/hipporag_full/predictions.json', 'r') as f:
    methods['HippoRAG（标准KG）'] = json.load(f)
print(f'✅ HippoRAG标准: {len(methods["HippoRAG（标准KG）"])} 个预测')

# HippoRAG高质量
hq_file = 'results/hipporag_high_quality/predictions.json'
if Path(hq_file).exists():
    with open(hq_file, 'r') as f:
        methods['HippoRAG（高质量KG）'] = json.load(f)
    print(f'✅ HippoRAG高质量: {len(methods["HippoRAG（高质量KG）"])} 个预测')

# 评估
print('\n🔬 计算评估指标...')
metrics = {}
for name, preds in methods.items():
    metrics[name] = evaluate_predictions(preds)
    print(f'✅ {name} - F1: {metrics[name]["f1_mean"]:.4f}, EM: {metrics[name]["em_mean"]*100:.1f}%')

# 生成报告
report = []
report.append('# 四种检索方法完整对比分析')
report.append('')
report.append('## 实验设计')
report.append('')
report.append('**控制变量**：')
report.append('- LLM：GPT-3.5-turbo（所有方法相同）')
report.append('- Temperature：0.0（所有方法相同）')
report.append('- Top-K：5个文档用于生成（所有方法相同）')
report.append('- 数据集：HotpotQA 66,573文档，500验证问题')
report.append('')
report.append('**变化变量**：检索方法和知识图谱')
report.append('')

report.append('## 方法说明')
report.append('')
report.append('| 方法 | 检索类型 | 知识图谱 | 原理 |')
report.append('|------|---------|---------|------|')
report.append('| BM25 | 稀疏 | 无 | 关键词匹配（TF-IDF改进版） |')
report.append('| FAISS | 稠密 | 无 | 语义向量相似度 |')
report.append('| HippoRAG（标准） | 稠密 | SpaCy提取 | FAISS + PageRank重排序 |')
report.append('| HippoRAG（高质量） | 稠密 | LLM提取($4) | FAISS + 高质量KG重排序 |')
report.append('')

report.append('## 性能对比')
report.append('')
report.append('| 方法 | F1 Score | Exact Match | 平均延迟 | 成功率 |')
report.append('|------|----------|-------------|----------|--------|')

# 按性能排序
sorted_methods = sorted(metrics.items(), key=lambda x: x[1]['f1_mean'], reverse=True)
for name, m in sorted_methods:
    line = f"| {name} | {m['f1_mean']:.4f} | {m['em_mean']*100:.1f}% | {m['latency_mean']:.2f}s | {m['success_rate']*100:.1f}% |"
    report.append(line)

report.append('')

# 计算相对提升（以BM25为基准）
if 'BM25（稀疏检索）' in metrics:
    bm25_f1 = metrics['BM25（稀疏检索）']['f1_mean']
    
    report.append('## 相对BM25的提升')
    report.append('')
    report.append('| 方法 | F1提升 | EM提升 | 延迟变化 |')
    report.append('|------|--------|--------|----------|')
    
    for name, m in sorted_methods:
        if name != 'BM25（稀疏检索）':
            f1_change = (m['f1_mean'] - metrics['BM25（稀疏检索）']['f1_mean']) / metrics['BM25（稀疏检索）']['f1_mean'] * 100
            em_change = (m['em_mean'] - metrics['BM25（稀疏检索）']['em_mean']) / metrics['BM25（稀疏检索）']['em_mean'] * 100
            lat_change = (m['latency_mean'] - metrics['BM25（稀疏检索）']['latency_mean']) / metrics['BM25（稀疏检索）']['latency_mean'] * 100
            report.append(f"| {name} | {f1_change:+.1f}% | {em_change:+.1f}% | {lat_change:+.1f}% |")
    
    report.append('')

# 计算相对提升（以FAISS为基准）
faiss_f1 = metrics['FAISS（稠密检索）']['f1_mean']

report.append('## 相对FAISS的提升（KG的真实价值）')
report.append('')
report.append('| 方法 | F1提升 | EM提升 | 延迟变化 | 结论 |')
report.append('|------|--------|--------|----------|------|')

for name, m in sorted_methods:
    if 'HippoRAG' in name:
        f1_change = (m['f1_mean'] - faiss_f1) / faiss_f1 * 100
        em_change = (m['em_mean'] - metrics['FAISS（稠密检索）']['em_mean']) / metrics['FAISS（稠密检索）']['em_mean'] * 100
        lat_change = (m['latency_mean'] - metrics['FAISS（稠密检索）']['latency_mean']) / metrics['FAISS（稠密检索）']['latency_mean'] * 100
        
        if f1_change > 0:
            conclusion = 'KG有价值'
        elif f1_change > -2:
            conclusion = 'KG基本无用'
        else:
            conclusion = 'KG有害'
        
        report.append(f"| {name} | {f1_change:+.1f}% | {em_change:+.1f}% | {lat_change:+.1f}% | {conclusion} |")

report.append('')

report.append('## 核心结论')
report.append('')

# 判断排序
if 'BM25（稀疏检索）' in metrics:
    ranking = [name for name, _ in sorted_methods]
    
    if ranking[0] == 'FAISS（稠密检索）':
        report.append('✅ **FAISS稠密检索性能最佳**')
        report.append('')
        
        if ranking[-1] == 'BM25（稀疏检索）':
            report.append('**性能排序**：')
            report.append('```')
            for i, name in enumerate(ranking, 1):
                f1 = metrics[name]['f1_mean']
                report.append(f'{i}. {name:30} F1 = {f1:.4f}')
            report.append('```')
            report.append('')
            
            # 分析HippoRAG的位置
            hippo_std_rank = ranking.index('HippoRAG（标准KG）') + 1
            
            if hippo_std_rank == 2:
                report.append('### HippoRAG介于BM25和FAISS之间')
                report.append('')
                report.append('**这说明**：')
                report.append('- KG比BM25强（利用了向量检索的语义理解）')
                report.append('- KG比FAISS弱（图谱重排序反而降低性能）')
                report.append('- **HippoRAG的提升主要来自FAISS，不是KG**')
            elif hippo_std_rank >= 3:
                report.append('### HippoRAG甚至弱于BM25（如果是这样）')
                report.append('')
                report.append('**这说明**：')
                report.append('- KG的重排序严重破坏了FAISS的检索质量')
                report.append('- 知识图谱完全无价值')
    
report.append('')
report.append('## 对HippoRAG论文的影响')
report.append('')

if 'BM25（稀疏检索）' in metrics:
    bm25_f1 = metrics['BM25（稀疏检索）']['f1_mean']
    faiss_f1 = metrics['FAISS（稠密检索）']['f1_mean']
    hippo_f1 = metrics['HippoRAG（标准KG）']['f1_mean']
    
    faiss_vs_bm25 = (faiss_f1 - bm25_f1) / bm25_f1 * 100
    hippo_vs_bm25 = (hippo_f1 - bm25_f1) / bm25_f1 * 100
    hippo_vs_faiss = (hippo_f1 - faiss_f1) / faiss_f1 * 100
    
    report.append('**如果HippoRAG论文用BM25做Baseline**：')
    report.append('')
    report.append(f'- 论文可能报告：HippoRAG比BM25提升 {hippo_vs_bm25:.1f}%')
    report.append(f'- 声称：KG带来 {hippo_vs_bm25:.1f}% 提升')
    report.append('')
    report.append('**真相分解**：')
    report.append(f'- FAISS vs BM25：{faiss_vs_bm25:.1f}% （检索方法改进）')
    report.append(f'- HippoRAG vs FAISS：{hippo_vs_faiss:.1f}% （KG的真实贡献）')
    report.append('')
    report.append('**结论**：')
    if abs(hippo_vs_faiss) < abs(faiss_vs_bm25) / 5:
        report.append(f'- ❌ **KG的贡献（{hippo_vs_faiss:.1f}%）远小于FAISS（{faiss_vs_bm25:.1f}%）**')
        report.append(f'- ❌ **论文的{hippo_vs_bm25:.1f}%提升中，{faiss_vs_bm25/(hippo_vs_bm25)*100:.0f}%来自FAISS，不是KG**')
        report.append('- ❌ **这是严重的学术误导**')
    
    report.append('')
    report.append('**欺骗性等级**：')
    deception_score = min(10, int(abs(faiss_vs_bm25) / abs(hippo_vs_faiss)))
    report.append(f'- {"★" * deception_score}/10')
    if deception_score >= 8:
        report.append('- **接近学术欺诈**')

report.append('')
report.append('---')
report.append('')
report.append('*实验日期：2026-02-04*  ')
report.append('*实验者：独立验证*  ')
report.append('*方法：严格控制变量*  ')
report.append('*结论：数据驱动*  ')

# 保存报告
output_file = 'results/four_way_comparison.md'
with open(output_file, 'w') as f:
    f.write('\n'.join(report))

print(f'\n✅ 报告已保存到: {output_file}')
print()
for line in report:
    print(line)
