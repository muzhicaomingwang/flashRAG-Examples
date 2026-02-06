#!/usr/bin/env python3
"""
评估报告生成脚本（简化版）
"""

import json
import string
import re
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent


def normalize_answer(answer: str) -> str:
    """归一化答案"""
    answer = answer.lower()
    answer = answer.translate(str.maketrans('', '', string.punctuation))
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算F1分数"""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

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


def evaluate_results(results: list) -> dict:
    """评估结果"""
    f1_scores = []
    em_scores = []
    latencies = []
    success_count = 0

    for result in results:
        if result['success']:
            f1 = compute_f1(result['predicted_answer'], result['gold_answer'])
            f1_scores.append(f1)

            em = compute_exact_match(result['predicted_answer'], result['gold_answer'])
            em_scores.append(em)

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


def main():
    print("=" * 60)
    print("生成评估报告")
    print("=" * 60)

    # 加载结果
    print("\n📚 加载实验结果...")

    baseline_path = project_root / "results" / "baseline" / "predictions.json"
    hipporag_path = project_root / "results" / "hipporag" / "predictions.json"

    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    print(f"✅ Baseline: {len(baseline_results)} 问题")

    with open(hipporag_path, 'r') as f:
        hipporag_results = json.load(f)
    print(f"✅ HippoRAG: {len(hipporag_results)} 问题")

    # 评估
    print("\n📊 计算评估指标...")

    baseline_metrics = evaluate_results(baseline_results)
    hipporag_metrics = evaluate_results(hipporag_results)

    print(f"\nBaseline RAG:")
    print(f"  - F1: {baseline_metrics['f1']['mean']:.4f} ± {baseline_metrics['f1']['std']:.4f}")
    print(f"  - EM: {baseline_metrics['exact_match']['mean']:.4f}")
    print(f"  - 延迟: {baseline_metrics['latency']['mean']:.2f}s")

    print(f"\nHippoRAG:")
    print(f"  - F1: {hipporag_metrics['f1']['mean']:.4f} ± {hipporag_metrics['f1']['std']:.4f}")
    print(f"  - EM: {hipporag_metrics['exact_match']['mean']:.4f}")
    print(f"  - 延迟: {hipporag_metrics['latency']['mean']:.2f}s")

    # 计算提升
    f1_improvement = ((hipporag_metrics['f1']['mean'] - baseline_metrics['f1']['mean']) / baseline_metrics['f1']['mean']) * 100 if baseline_metrics['f1']['mean'] > 0 else 0

    print(f"\n📈 性能提升:")
    print(f"  - F1提升: {f1_improvement:+.2f}%")

    # 保存指标
    all_metrics = {
        'Baseline-RAG': baseline_metrics,
        'HippoRAG': hipporag_metrics
    }

    metrics_path = project_root / "results" / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, indent=2, fp=f)
    print(f"\n✅ 指标已保存: {metrics_path}")

    # 生成对比表格
    table = "# HippoRAG vs Baseline RAG 实验结果\n\n"
    table += "## 性能对比\n\n"
    table += "| 方法 | F1 Score | Exact Match | 平均延迟 (秒) | 成功率 |\n"
    table += "|------|----------|-------------|--------------|--------|\n"
    table += f"| **Baseline-RAG** | {baseline_metrics['f1']['mean']:.4f} ± {baseline_metrics['f1']['std']:.4f} | {baseline_metrics['exact_match']['mean']:.4f} | {baseline_metrics['latency']['mean']:.2f} | {baseline_metrics['success_rate']:.2%} |\n"
    table += f"| **HippoRAG** | {hipporag_metrics['f1']['mean']:.4f} ± {hipporag_metrics['f1']['std']:.4f} | {hipporag_metrics['exact_match']['mean']:.4f} | {hipporag_metrics['latency']['mean']:.2f} | {hipporag_metrics['success_rate']:.2%} |\n"

    table += f"\n## 性能提升\n\n"
    table += f"- **F1 Score提升:** {f1_improvement:+.2f}%\n"

    if f1_improvement > 10:
        table += f"- **结论:** ✅ HippoRAG在多跳问答上有显著提升\n"
    elif f1_improvement > 0:
        table += f"- **结论:** ✅ HippoRAG有适度提升\n"
    else:
        table += f"- **结论:** ⚠️ HippoRAG未带来明显改进\n"

    table_path = project_root / "results" / "comparison_table.md"
    with open(table_path, 'w') as f:
        f.write(table)
    print(f"✅ 对比表格已保存: {table_path}")

    print("\n" + "=" * 60)
    print("✅ 报告生成完成！")
    print("=" * 60)
    print(f"\n{table}")


if __name__ == "__main__":
    main()
