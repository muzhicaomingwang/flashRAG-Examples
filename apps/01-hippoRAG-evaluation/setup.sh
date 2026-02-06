#!/bin/bash

# HippoRAG 实验环境快速安装脚本

set -e

echo "🚀 开始安装 HippoRAG 实验环境..."

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误: 需要 Python >= 3.9，当前版本: $python_version"
    exit 1
fi

echo "✅ Python 版本检查通过: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ 虚拟环境已激活"
fi

# 升级 pip
echo "📦 升级 pip..."
pip install --upgrade pip

# 安装依赖
echo "📦 安装项目依赖..."
pip install -r requirements.txt

# 下载 SpaCy 模型
echo "📦 下载 SpaCy 英文模型..."
python3 -m spacy download en_core_web_sm

# 创建 .env 文件
if [ ! -f .env ]; then
    echo "📝 创建 .env 文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件，填入你的 OPENAI_API_KEY"
fi

# 验证安装
echo ""
echo "🧪 验证安装..."
python3 -c "
import flashrag
import faiss
import spacy
import networkx
import openai
print('✅ 所有依赖安装成功！')
"

echo ""
echo "🎉 安装完成！"
echo ""
echo "📝 下一步："
echo "1. 编辑 .env 文件，填入你的 OPENAI_API_KEY"
echo "2. 运行: python scripts/01_download_data.py"
echo "3. 查看 README.md 了解更多信息"
