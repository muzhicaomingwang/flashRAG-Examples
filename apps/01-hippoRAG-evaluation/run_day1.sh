#!/bin/bash

# Day 1 一键运行脚本
# 功能：下载数据 + 构建 Baseline RAG + 测试

set -e

echo "=========================================="
echo "Day 1: Baseline RAG 构建"
echo "=========================================="
echo ""

# 检查环境
if [ ! -f .env ]; then
    echo "❌ 错误: .env 文件不存在"
    echo "请先复制 .env.example 并配置 OPENAI_API_KEY"
    exit 1
fi

# Step 1: 下载数据
echo "Step 1/3: 下载 HotpotQA 数据集..."
python scripts/01_download_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据下载失败"
    exit 1
fi

# Step 2: 构建 Baseline RAG
echo ""
echo "Step 2/3: 构建 Baseline RAG 索引..."
echo "⏱️  预计时间: 30-40 分钟"
echo "💰 预计成本: $0.56"
echo ""

python scripts/02_build_baseline.py

if [ $? -ne 0 ]; then
    echo "❌ Baseline RAG 构建失败"
    exit 1
fi

# Step 3: 测试
echo ""
echo "Step 3/3: 测试 Baseline 检索..."
python scripts/03_test_baseline.py

if [ $? -ne 0 ]; then
    echo "❌ 测试失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Day 1 完成！"
echo "=========================================="
echo ""
echo "📊 生成的文件:"
echo "  - data/raw/hotpotqa_corpus.jsonl"
echo "  - data/raw/hotpotqa_validation.jsonl"
echo "  - data/indices/faiss/hotpotqa.index"
echo "  - data/indices/bm25/hotpotqa_bm25.pkl"
echo ""
echo "📝 下一步:"
echo "  ./run_day2.sh"
