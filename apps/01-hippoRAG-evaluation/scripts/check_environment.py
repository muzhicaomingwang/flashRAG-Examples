#!/usr/bin/env python3
"""
环境检查脚本
验证所有运行条件是否满足
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("✓ Python版本检查")
    version = sys.version_info
    print(f"  当前版本: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 9):
        print("  ⚠️  建议使用Python 3.9或更高版本")
        return False

    if version >= (3, 13):
        print("  ⚠️  Python版本过新，可能存在兼容性问题")
        print("     建议使用Python 3.9-3.12")

    return True

def check_memory():
    """检查可用内存"""
    print("\n✓ 内存检查")
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)

        print(f"  总内存: {total_gb:.1f} GB")
        print(f"  可用内存: {available_gb:.1f} GB")

        if available_gb < 2:
            print("  ⚠️  可用内存不足2GB，实验可能会失败")
            return False
        elif available_gb < 4:
            print("  ⚠️  可用内存不足4GB，建议释放一些内存")

        return True
    except ImportError:
        print("  ⚠️  psutil未安装，跳过内存检查")
        print("     提示: pip install psutil")
        return True

def check_openai_key():
    """检查OpenAI API Key"""
    print("\n✓ OpenAI API Key检查")
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        print("  ✗ OPENAI_API_KEY未设置")
        print("     请设置环境变量: export OPENAI_API_KEY='your-key'")
        return False

    print(f"  已设置: {api_key[:20]}...{api_key[-4:]}")
    return True

def check_data_files():
    """检查所有必需的数据文件"""
    print("\n✓ 数据文件检查")

    required_files = [
        "data/indices/faiss/hotpotqa_full.index",
        "data/indices/faiss/hotpotqa_full_docs.pkl",
        "data/knowledge_graphs/hotpotqa_kg_full.gpickle",
        "data/knowledge_graphs/hotpotqa_pagerank_full.pkl",
        "data/raw/hotpotqa_validation.jsonl"
    ]

    all_exist = True
    total_size = 0

    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            total_size += size_mb
            print(f"  ✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {file_path} 不存在")
            all_exist = False

    if all_exist:
        print(f"\n  总数据大小: {total_size:.1f} MB")

    return all_exist

def check_dependencies():
    """检查Python依赖包"""
    print("\n✓ 依赖包检查")

    required_packages = [
        ('openai', 'OpenAI API客户端'),
        ('faiss', 'FAISS向量索引'),
        ('spacy', 'SpaCy NLP库'),
        ('networkx', 'NetworkX图库'),
        ('numpy', 'NumPy数值计算'),
    ]

    all_installed = True

    for package, description in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} - {description}")
        except ImportError:
            print(f"  ✗ {package} 未安装 - {description}")
            all_installed = False

    # 检查SpaCy模型
    try:
        import spacy
        spacy.load('en_core_web_sm')
        print(f"  ✓ en_core_web_sm - SpaCy英文模型")
    except OSError:
        print(f"  ✗ en_core_web_sm 未安装")
        print(f"     运行: python -m spacy download en_core_web_sm")
        all_installed = False

    return all_installed

def main():
    """主函数"""
    print("="*70)
    print("HippoRAG实验环境检查")
    print("="*70)

    checks = [
        ("Python版本", check_python_version()),
        ("内存", check_memory()),
        ("OpenAI API Key", check_openai_key()),
        ("数据文件", check_data_files()),
        ("依赖包", check_dependencies()),
    ]

    print("\n" + "="*70)
    print("检查结果汇总")
    print("="*70)

    all_passed = True
    for name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✅ 所有检查通过！可以运行实验。")
        return 0
    else:
        print("\n❌ 部分检查未通过，请修复后再运行实验。")
        return 1

if __name__ == '__main__':
    sys.exit(main())
