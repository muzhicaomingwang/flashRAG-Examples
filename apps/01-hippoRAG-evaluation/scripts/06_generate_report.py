#!/usr/bin/env python3
"""
评估报告生成脚本

功能：
1. 加载所有实验结果
2. 计算评估指标（F1, EM, Recall@K 等）
3. 生成对比表格
4. 绘制可视化图表
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.metrics import f1_score


def normalize_answer(answer: str) -> str:
    """归一化答案（去除标点、小写等）"""
    import string
    import re

    # 转小写
    answer = answer.lower()

    # 去除标点
    answer = answer.translate(str.maketrans('', '', string.punctuation))

    # 去除多余空格
    answer = re.sub(r'\s+', ' ', answer).strip()

    return answer


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算 F1 分数"""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    # 计算交集
    common = set(pred_tokens) & set(gold_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """计算精确匹配"""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def evaluate_results(results: List[Dict]) -> Dict:
    """评估单个方法的结果"""
    f1_scores = []
    em_scores = []
    latencies = []
    success_count = 0

    for result in results:
        if result['success']:
            # 计算 F1
            f1 = compute_f1(result['predicted_answer'], result['gold_answer'])
            f1_scores.append(f1)

            # 计算 EM
            em = compute_exact_match(result['predicted_answer'], result['gold_answer'])
            em_scores.append(em)

            # 记录延迟
            latencies.append(result['latency'])

            success_count += 1

    metrics = {
        "num_questions": len(results),
        "success_count": success_count,
        "success_rate": success_count / len(results) if results else 0.0,
        "f1": {
            "mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "std": float(np.std(f1_scores)) if f1_scores else 0.0
        },
        "exact_match": {
            "mean": float(np.mean(em_scores)) if em_scores else 0.0,
            "std": float(np.std(em_scores)) if em_scores else 0.0
        },
        "latency": {
            "mean": float(np.mean(latencies)) if latencies else 0.0,
            "std": float(np.std(latencies)) if latencies else 0.0,
            "median": float(np.median(latencies)) if latencies else 0.0
        }
    }

    return metrics


def load_all_results() -> Dict[str, List[Dict]]:
    """加载所有实验结果"""
    print("📚 加载实验结果...")

    results = {}

    # 加载 Baseline 结果
    baseline_path = project_root / "results" / "baseline" / "predictions.json"
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            results['Baseline-RAG'] = json.load(f)
        print(f"✅ Baseline-RAG: {len(results['Baseline-RAG'])} 个问题")

    # 加载 HippoRAG 结果
    hipporag_path = project_root / "results" / "hipporag" / "predictions.json"
    if hipporag_path.exists():
        with open(hipporag_path, 'r') as f:
            results['HippoRAG'] = json.load(f)
        print(f"✅ HippoRAG: {len(results['HippoRAG'])} 个问题")

    return results


def generate_comparison_table(all_metrics: Dict[str, Dict]) -> str:
    """生成对比表格（Markdown 格式）"""
    table = "# HippoRAG vs Baseline RAG 实验结果对比\n\n"
    table += "## 主要指标\n\n"
    table += "| 方法 | F1 Score | Exact Match | 平均延迟 (秒) | 成功率 |\n"
    table += "|------|----------|-------------|--------------|--------|\n"

    for method_name, metrics in all_metrics.items():
        table += f"| **{method_name}** "
        table += f"| {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f} "
        table += f"| {metrics['exact_match']['mean']:.4f} ± {metrics['exact_match']['std']:.4f} "
        table += f"| {metrics['latency']['mean']:.2f} ± {metrics['latency']['std']:.2f} "
        table += f"| {metrics['success_rate']:.2%} |\n"

    # 计算提升百分比
    if 'Baseline-RAG' in all_metrics and 'HippoRAG' in all_metrics:
        baseline_f1 = all_metrics['Baseline-RAG']['f1']['mean']
        hipporag_f1 = all_metrics['HippoRAG']['f1']['mean']

        if baseline_f1 > 0:
            improvement = ((hipporag_f1 - baseline_f1) / baseline_f1) * 100
            table += f"\n## 性能提升\n\n"
            table += f"- **F1 Score 提升:** {improvement:+.2f}%\n"

            if improvement > 0:
                table += f"- **结论:** HippoRAG 在多跳问答上表现更好\n"
            elif improvement < -5:
                table += f"- **结论:** HippoRAG 引入了不必要的复杂度\n"
            else:
                table += f"- **结论:** 两者性能相近，需要更多数据集验证\n"

    return table


def main():
    """主函数"""
    print("=" * 60)
    print("生成评估报告")
    print("=" * 60)

    # 加载所有结果
    all_results = load_all_results()

    if not all_results:
        print("❌ 错误: 未找到实验结果")
        print("请先运行: python scripts/05_run_experiments.py")
        return

    # 评估每个方法
    print("\n📊 计算评估指标...")
    all_metrics = {}

    for method_name, results in all_results.items():
        print(f"\n评估: {method_name}")
        metrics = evaluate_results(results)
        all_metrics[method_name] = metrics

        print(f"  - F1 Score: {metrics['f1']['mean']:.4f} ± {metrics['f1']['std']:.4f}")
        print(f"  - Exact Match: {metrics['exact_match']['mean']:.4f}")
        print(f"  - 平均延迟: {metrics['latency']['mean']:.2f}s")

    # 保存详细指标
    metrics_path = project_root / "results" / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, indent=2, fp=f)
    print(f"\n✅ 详细指标已保存: {metrics_path}")

    # 生成对比表格
    comparison_table = generate_comparison_table(all_metrics)
    table_path = project_root / "results" / "comparison_table.md"
    with open(table_path, 'w') as f:
        f.write(comparison_table)
    print(f"✅ 对比表格已保存: {table_path}")

    # 打印对比表格
    print(f"\n{comparison_table}")

    print("\n" + "=" * 60)
    print("✅ 报告生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
