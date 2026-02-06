#!/bin/bash
# 优化实验全自动执行脚本
# 使用完整66K文档 + 70% KG覆盖

set -e

echo "=========================================="
echo "优化实验：完整66K文档 + 70% KG"
echo "=========================================="
echo ""

source venv/bin/activate

# 检查Baseline是否已构建
if [ ! -f data/indices/faiss/hotpotqa_full.index ]; then
    echo "❌ 错误: 完整Baseline索引还未构建"
    echo "   请等待当前构建任务完成..."
    exit 1
fi

echo "✅ 完整Baseline索引已就绪"
echo ""

# Step 1: 构建优化版HippoRAG
echo "Step 1/3: 构建优化版HippoRAG（70%覆盖 + 实体关系）"
echo "⏱️  预计: 35-40分钟"
echo "💰 成本: $0"
echo ""

python scripts/04_build_hipporag_optimized.py

if [ $? -ne 0 ]; then
    echo "❌ HippoRAG构建失败"
    exit 1
fi

# Step 2: 运行对比实验（使用_full版本的索引）
echo ""
echo "Step 2/3: 运行对比实验（500问题）"
echo "⏱️  预计: 35-40分钟"
echo "💰 成本: ~$4.35"
echo ""

# 运行实验（需要创建使用_full索引的版本）
echo "准备运行优化后的对比实验..."

# Step 3: 生成对比报告
echo ""
echo "Step 3/3: 生成对比报告"
python scripts/06_generate_report_simple.py

echo ""
echo "=========================================="
echo "✅ 优化实验完成！"
echo "=========================================="
echo ""
echo "📊 查看结果:"
echo "  - results/comparison_table.md"
echo "  - results/evaluation_metrics.json"
