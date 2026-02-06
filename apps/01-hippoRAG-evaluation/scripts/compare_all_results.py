#!/usr/bin/env python3
"""
对比三种方法的效果：
1. Baseline RAG
2. HippoRAG（标准KG）
3. HippoRAG（高质量KG）
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

def normalize_answer(s):
    """标准化答案"""
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
    """计算F1分数"""
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
    """计算精确匹配"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate_predictions(predictions):
    """评估预测结果"""
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
print('三种方法完整对比分析')
print('='*80)

# 加载结果
print('\n📊 加载预测结果...')

with open('results/baseline_full/predictions.json', 'r') as f:
    baseline = json.load(f)
print(f'✅ Baseline RAG: {len(baseline)} 个预测')

with open('results/hipporag_full/predictions.json', 'r') as f:
    hipporag_standard = json.load(f)
print(f'✅ HippoRAG (标准KG): {len(hipporag_standard)} 个预测')

hq_file = 'results/hipporag_high_quality/predictions.json'
if Path(hq_file).exists():
    with open(hq_file, 'r') as f:
        hipporag_hq = json.load(f)
    print(f'✅ HippoRAG (高质量KG): {len(hipporag_hq)} 个预测')
    has_hq = True
else:
    print(f'⚠️  高质量KG结果不存在')
    has_hq = False

# 评估
print('\n🔬 计算评估指标...')
baseline_metrics = evaluate_predictions(baseline)
standard_metrics = evaluate_predictions(hipporag_standard)

if has_hq:
    hq_metrics = evaluate_predictions(hipporag_hq)

# 生成报告
report = []
report.append('# 三种方法完整对比分析')
report.append('')
report.append('## 知识图谱质量对比')
report.append('')

with open('data/knowledge_graphs/hotpotqa_kg_full_stats.json', 'r') as f:
    standard_stats = json.load(f)

report.append('| KG版本 | 总节点 | 总边 | Entity→Entity | 占比 | 图密度 |')
report.append('|--------|--------|------|--------------|------|--------|')
report.append(f"| 标准KG | {standard_stats['num_nodes']:,} | {standard_stats['num_edges']:,} | {standard_stats['num_entity_relations']:,} | 1.7% | 0.000008 |")

if has_hq:
    with open('data/knowledge_graphs/hotpotqa_kg_high_quality_stats.json', 'r') as f:
        hq_stats = json.load(f)
    e2e_pct = hq_stats['num_entity_relations'] / hq_stats['num_edges'] * 100
    report.append(f"| 高质量KG | {hq_stats['num_nodes']:,} | {hq_stats['num_edges']:,} | {hq_stats['num_entity_relations']:,} | {e2e_pct:.1f}% | {hq_stats['graph_density']:.6f} |")

report.append('')

report.append('## 性能对比')
report.append('')
report.append('| 方法 | F1 Score | Exact Match | 平均延迟 |')
report.append('|------|----------|-------------|----------|')
report.append(f"| Baseline RAG | {baseline_metrics['f1_mean']:.4f} | {baseline_metrics['em_mean']*100:.1f}% | {baseline_metrics['latency_mean']:.2f}s |")
report.append(f"| HippoRAG (标准KG) | {standard_metrics['f1_mean']:.4f} | {standard_metrics['em_mean']*100:.1f}% | {standard_metrics['latency_mean']:.2f}s |")

if has_hq:
    report.append(f"| HippoRAG (高质量KG) | {hq_metrics['f1_mean']:.4f} | {hq_metrics['em_mean']*100:.1f}% | {hq_metrics['latency_mean']:.2f}s |")

report.append('')

report.append('## 提升幅度')
report.append('')
report.append('| 对比 | F1提升 | EM提升 | 延迟变化 |')
report.append('|------|--------|--------|----------|')

standard_f1_change = (standard_metrics['f1_mean'] - baseline_metrics['f1_mean']) / baseline_metrics['f1_mean'] * 100
standard_em_change = (standard_metrics['em_mean'] - baseline_metrics['em_mean']) / baseline_metrics['em_mean'] * 100
standard_lat_change = (standard_metrics['latency_mean'] - baseline_metrics['latency_mean']) / baseline_metrics['latency_mean'] * 100

report.append(f"| 标准KG vs Baseline | {standard_f1_change:+.1f}% | {standard_em_change:+.1f}% | {standard_lat_change:+.1f}% |")

if has_hq:
    hq_f1_change = (hq_metrics['f1_mean'] - baseline_metrics['f1_mean']) / baseline_metrics['f1_mean'] * 100
    hq_em_change = (hq_metrics['em_mean'] - baseline_metrics['em_mean']) / baseline_metrics['em_mean'] * 100
    hq_lat_change = (hq_metrics['latency_mean'] - baseline_metrics['latency_mean']) / baseline_metrics['latency_mean'] * 100

    report.append(f"| 高质量KG vs Baseline | {hq_f1_change:+.1f}% | {hq_em_change:+.1f}% | {hq_lat_change:+.1f}% |")
    
    kg_improvement = (hq_metrics['f1_mean'] - standard_metrics['f1_mean']) / standard_metrics['f1_mean'] * 100
    report.append(f"| 高质量KG vs 标准KG | {kg_improvement:+.1f}% | {(hq_metrics['em_mean'] - standard_metrics['em_mean']) / standard_metrics['em_mean'] * 100:+.1f}% | {(hq_metrics['latency_mean'] - standard_metrics['latency_mean']) / standard_metrics['latency_mean'] * 100:+.1f}% |")

report.append('')

report.append('## 结论')
report.append('')

if has_hq:
    if hq_f1_change > 5:
        report.append(f'✅ **高质量KG证明了HippoRAG的价值**，F1提升{hq_f1_change:.1f}%，在真实多跳场景有显著优势。')
    elif hq_f1_change > 0 and hq_f1_change <= 5:
        report.append(f'⚠️ **高质量KG带来小幅提升**，F1提升{hq_f1_change:.1f}%，但考虑成本可能不值得。')
        report.append('')
        report.append(f'KG质量改进后，Entity→Entity关系提升{hq_stats["num_entity_relations"]/standard_stats["num_entity_relations"]:.1f}倍，但F1只提升{hq_f1_change:.1f}%，说明：')
        report.append('- 即使有密集的实体关系，HippoRAG的收益仍然有限')
        report.append('- 现代Baseline (向量检索 + LLM) 已经足够强大')
        report.append('- 对通用RAG场景，KG的复杂度不值得')
    else:
        report.append(f'❌ **即使使用高质量KG，HippoRAG仍未超过Baseline**，F1变化{hq_f1_change:.1f}%。')
        report.append('')
        report.append('**这是决定性的证据**，证明：')
        report.append('1. KG质量不是唯一瓶颈')
        report.append('2. HippoRAG的核心假设在通用场景不成立')
        report.append('3. 简单的向量检索 + LLM推理已经足够好')
        report.append('4. 知识图谱在RAG中的价值被严重高估')
        report.append('')
        report.append(f'投入{hq_stats["total_cost"]:.2f}美元构建高质量KG，仍然获得{hq_f1_change:.1f}%的负收益。')

output_file = 'results/three_way_comparison.md'
with open(output_file, 'w') as f:
    f.write('\n'.join(report))

print(f'\n✅ 报告已保存到: {output_file}')
print()
for line in report:
    print(line)
