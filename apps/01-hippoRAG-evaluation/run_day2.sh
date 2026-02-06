#!/bin/bash

# Day 2 一键运行脚本
# 功能：构建 HippoRAG + 运行实验 + 生成报告

set -e

echo "=========================================="
echo "Day 2: HippoRAG 实验"
echo "=========================================="
echo ""

# 激活虚拟环境
source venv/bin/activate

# 检查 Day 1 的输出是否存在
if [ ! -f data/indices/faiss/hotpotqa.index ]; then
    echo "❌ 错误: 未找到 Baseline RAG 索引"
    echo "请先运行: ./run_day1.sh"
    exit 1
fi

# Step 1: 构建 HippoRAG 知识图谱
echo "Step 1/3: 构建 HippoRAG 知识图谱..."
echo "⏱️  预计时间: 20-30 分钟"
echo "💰 预计成本: ~$0（使用 SpaCy，不调用 LLM）"
echo ""

python scripts/04_build_hipporag.py

if [ $? -ne 0 ]; then
    echo "❌ HippoRAG 构建失败"
    exit 1
fi

# Step 2: 运行实验
echo ""
echo "Step 2/3: 运行对比实验..."
echo "⏱️  预计时间: 10-15 分钟"
echo "💰 预计成本: ~$4.35（500 问题 × 2 组）"
echo ""

python scripts/05_run_experiments.py

if [ $? -ne 0 ]; then
    echo "❌ 实验运行失败"
    exit 1
fi

# Step 3: 生成报告
echo ""
echo "Step 3/3: 生成评估报告..."
python scripts/06_generate_report.py

if [ $? -ne 0 ]; then
    echo "❌ 报告生成失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Day 2 完成！"
echo "=========================================="
echo ""
echo "📊 查看结果:"
echo "  - results/evaluation_metrics.json"
echo "  - results/comparison_table.md"
echo ""
echo "🎉 实验全部完成！"
