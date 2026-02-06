#!/usr/bin/env python3
"""
简单的评估脚本
计算F1、EM、平均延迟等指标，生成对比报告
"""

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np


def normalize_answer(s):
    """标准化答案用于比较"""
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


def compute_exact_match(prediction, ground_truth):
    """计算精确匹配"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate_predictions(predictions):
    """评估预测结果"""
    f1_scores = []
    em_scores = []
    latencies = []
    successes = []

    for pred in predictions:
        if pred['success']:
            f1 = compute_f1(pred['predicted_answer'], pred['gold_answer'])
            em = compute_exact_match(pred['predicted_answer'], pred['gold_answer'])

            f1_scores.append(f1)
            em_scores.append(em)
            latencies.append(pred['latency'])
            successes.append(True)
        else:
            successes.append(False)

    return {
        'f1_mean': np.mean(f1_scores) if f1_scores else 0,
        'f1_std': np.std(f1_scores) if f1_scores else 0,
        'em_mean': np.mean(em_scores) if em_scores else 0,
        'em_std': np.std(em_scores) if em_scores else 0,
        'latency_mean': np.mean(latencies) if latencies else 0,
        'latency_std': np.std(latencies) if latencies else 0,
        'latency_median': np.median(latencies) if latencies else 0,
        'success_rate': sum(successes) / len(successes) if successes else 0,
        'total': len(predictions),
        'successful': sum(successes)
    }


def generate_comparison_report(baseline_metrics, hipporag_metrics):
    """生成对比报告"""
    report = []

    report.append("# HippoRAG优化版实验对比报告")
    report.append("")
    report.append("## 实验配置")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|------|---|")
    report.append("| 文档数 | 66,573 (完整) |")
    report.append("| KG覆盖率 | 70% |")
    report.append("| 验证集大小 | 500 问题 |")
    report.append("| Embedding模型 | text-embedding-3-small |")
    report.append("| LLM模型 | gpt-3.5-turbo |")
    report.append("")

    report.append("## 整体性能对比")
    report.append("")
    report.append("| 方法 | F1 Score | Exact Match | 平均延迟 (s) | 成功率 |")
    report.append("|------|----------|-------------|-------------|--------|")

    baseline_line = f"| Baseline RAG | {baseline_metrics['f1_mean']:.4f} ± {baseline_metrics['f1_std']:.4f} | "
    baseline_line += f"{baseline_metrics['em_mean']*100:.1f}% ± {baseline_metrics['em_std']*100:.1f}% | "
    baseline_line += f"{baseline_metrics['latency_mean']:.2f} ± {baseline_metrics['latency_std']:.2f} | "
    baseline_line += f"{baseline_metrics['success_rate']*100:.1f}% |"
    report.append(baseline_line)

    hipporag_line = f"| HippoRAG | {hipporag_metrics['f1_mean']:.4f} ± {hipporag_metrics['f1_std']:.4f} | "
    hipporag_line += f"{hipporag_metrics['em_mean']*100:.1f}% ± {hipporag_metrics['em_std']*100:.1f}% | "
    hipporag_line += f"{hipporag_metrics['latency_mean']:.2f} ± {hipporag_metrics['latency_std']:.2f} | "
    hipporag_line += f"{hipporag_metrics['success_rate']*100:.1f}% |"
    report.append(hipporag_line)

    report.append("")

    # 计算提升
    f1_improvement = (hipporag_metrics['f1_mean'] - baseline_metrics['f1_mean']) / baseline_metrics['f1_mean'] * 100
    em_improvement = (hipporag_metrics['em_mean'] - baseline_metrics['em_mean']) / baseline_metrics['em_mean'] * 100
    latency_change = (hipporag_metrics['latency_mean'] - baseline_metrics['latency_mean']) / baseline_metrics['latency_mean'] * 100

    report.append("## 性能提升")
    report.append("")
    report.append("| 指标 | 提升 |")
    report.append("|------|------|")
    report.append(f"| F1 Score | {f1_improvement:+.1f}% |")
    report.append(f"| Exact Match | {em_improvement:+.1f}% |")
    report.append(f"| 平均延迟 | {latency_change:+.1f}% |")
    report.append("")

    # 详细统计
    report.append("## 详细统计")
    report.append("")
    report.append("### Baseline RAG")
    report.append("")
    report.append(f"- 总问题数: {baseline_metrics['total']}")
    report.append(f"- 成功数: {baseline_metrics['successful']}")
    report.append(f"- 成功率: {baseline_metrics['success_rate']*100:.1f}%")
    report.append(f"- F1: {baseline_metrics['f1_mean']:.4f} ± {baseline_metrics['f1_std']:.4f}")
    report.append(f"- EM: {baseline_metrics['em_mean']*100:.1f}% ± {baseline_metrics['em_std']*100:.1f}%")
    report.append(f"- 平均延迟: {baseline_metrics['latency_mean']:.2f}s")
    report.append(f"- 中位延迟: {baseline_metrics['latency_median']:.2f}s")
    report.append("")

    report.append("### HippoRAG")
    report.append("")
    report.append(f"- 总问题数: {hipporag_metrics['total']}")
    report.append(f"- 成功数: {hipporag_metrics['successful']}")
    report.append(f"- 成功率: {hipporag_metrics['success_rate']*100:.1f}%")
    report.append(f"- F1: {hipporag_metrics['f1_mean']:.4f} ± {hipporag_metrics['f1_std']:.4f}")
    report.append(f"- EM: {hipporag_metrics['em_mean']*100:.1f}% ± {hipporag_metrics['em_std']*100:.1f}%")
    report.append(f"- 平均延迟: {hipporag_metrics['latency_mean']:.2f}s")
    report.append(f"- 中位延迟: {hipporag_metrics['latency_median']:.2f}s")
    report.append("")

    # 结论
    report.append("## 结论")
    report.append("")
    if f1_improvement > 10:
        report.append(f"✅ **HippoRAG显著优于Baseline**，F1提升{f1_improvement:.1f}%，验证了知识图谱在多跳推理中的价值。")
    elif f1_improvement > 5:
        report.append(f"✓ **HippoRAG略优于Baseline**，F1提升{f1_improvement:.1f}%，在多跳推理场景有一定优势。")
    elif f1_improvement > -5:
        report.append(f"➖ **HippoRAG与Baseline基本持平**，F1变化{f1_improvement:.1f}%，可能原因：")
        report.append("- 验证集包含较多单跳问题")
        report.append("- 实体提取不够准确")
        report.append("- PPR权重需要调优")
    else:
        report.append(f"⚠️ **HippoRAG弱于Baseline**，F1下降{abs(f1_improvement):.1f}%，需要进一步分析原因。")

    report.append("")

    return '\n'.join(report)


def main():
    """主函数"""
    print('='*70)
    print('评估实验结果')
    print('='*70)

    # 加载结果
    print('\n📊 加载预测结果...')

    baseline_file = 'results/baseline_full/predictions.json'
    hipporag_file = 'results/hipporag_full/predictions.json'

    if not Path(baseline_file).exists():
        print(f'✗ Baseline结果文件不存在: {baseline_file}')
        return 1

    if not Path(hipporag_file).exists():
        print(f'✗ HippoRAG结果文件不存在: {hipporag_file}')
        return 1

    with open(baseline_file, 'r') as f:
        baseline_predictions = json.load(f)
    print(f'✅ Baseline: {len(baseline_predictions)} 个预测')

    with open(hipporag_file, 'r') as f:
        hipporag_predictions = json.load(f)
    print(f'✅ HippoRAG: {len(hipporag_predictions)} 个预测')

    # 评估
    print('\n🔬 计算评估指标...')

    baseline_metrics = evaluate_predictions(baseline_predictions)
    print(f'✅ Baseline - F1: {baseline_metrics["f1_mean"]:.4f}, EM: {baseline_metrics["em_mean"]*100:.1f}%')

    hipporag_metrics = evaluate_predictions(hipporag_predictions)
    print(f'✅ HippoRAG - F1: {hipporag_metrics["f1_mean"]:.4f}, EM: {hipporag_metrics["em_mean"]*100:.1f}%')

    # 生成报告
    print('\n📝 生成对比报告...')

    report = generate_comparison_report(baseline_metrics, hipporag_metrics)

    # 保存报告
    output_file = 'results/comparison_report.md'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report)

    print(f'✅ 报告已保存到: {output_file}')

    # 保存JSON指标
    metrics_file = 'results/evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'baseline': baseline_metrics,
            'hipporag': hipporag_metrics
        }, f, indent=2)

    print(f'✅ 指标已保存到: {metrics_file}')

    print('\n' + '='*70)
    print('✅ 评估完成！')
    print('='*70)

    # 显示简要结果
    print('\n' + report)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
