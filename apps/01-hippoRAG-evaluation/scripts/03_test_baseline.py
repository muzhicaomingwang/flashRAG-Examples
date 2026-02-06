#!/usr/bin/env python3
"""
Baseline RAG 功能测试脚本

功能：
1. 加载 FAISS 和 BM25 索引
2. 测试检索功能
3. 运行几个示例问题
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def load_config() -> Dict:
    """加载配置文件"""
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_indices():
    """加载索引"""
    print("📚 加载索引...")

    # 加载 FAISS 索引
    faiss_index_path = project_root / "data" / "indices" / "faiss" / "hotpotqa.index"
    faiss_index = faiss.read_index(str(faiss_index_path))
    print(f"✅ FAISS 索引加载完成: {faiss_index.ntotal} 个向量")

    # 加载文档映射
    doc_mapping_path = project_root / "data" / "indices" / "faiss" / "hotpotqa_docs.pkl"
    with open(doc_mapping_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"✅ 文档映射加载完成: {len(chunks)} 个块")

    # 加载 BM25 索引
    bm25_path = project_root / "data" / "indices" / "bm25" / "hotpotqa_bm25.pkl"
    with open(bm25_path, 'rb') as f:
        bm25_index = pickle.load(f)
    print(f"✅ BM25 索引加载完成")

    return faiss_index, bm25_index, chunks


def retrieve_faiss(query: str, faiss_index, chunks: List[Dict], client, config: Dict, top_k: int = 5):
    """使用 FAISS 检索"""
    # 向量化查询
    response = client.embeddings.create(
        model=config['embedding']['model'],
        input=[query]
    )
    query_vector = np.array([response.data[0].embedding], dtype='float32')

    # 归一化
    if config['faiss']['normalize_vectors']:
        faiss.normalize_L2(query_vector)

    # 检索
    distances, indices = faiss_index.search(query_vector, top_k)

    # 获取文档
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        chunk = chunks[idx]
        results.append({
            "chunk_id": chunk['chunk_id'],
            "title": chunk['title'],
            "text": chunk['text'],
            "score": float(distance)
        })

    return results


def retrieve_bm25(query: str, bm25_index, chunks: List[Dict], top_k: int = 5):
    """使用 BM25 检索"""
    # Tokenize 查询
    tokenized_query = query.lower().split()

    # 检索
    scores = bm25_index.get_scores(tokenized_query)

    # 获取 top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    # 获取文档
    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "chunk_id": chunk['chunk_id'],
            "title": chunk['title'],
            "text": chunk['text'][:200] + "...",  # 只显示前200字符
            "score": float(scores[idx])
        })

    return results


def test_retrieval():
    """测试检索功能"""
    print("\n" + "=" * 60)
    print("测试检索功能")
    print("=" * 60)

    # 加载配置和索引
    config = load_config()
    faiss_index, bm25_index, chunks = load_indices()

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 测试问题
    test_questions = [
        "Who is the president of the United States?",
        "What is the capital of France?",
        "When was the Eiffel Tower built?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}: {question}")
        print(f"{'='*60}")

        # FAISS 检索
        print("\n🔍 FAISS 检索结果 (Top-3):")
        faiss_results = retrieve_faiss(question, faiss_index, chunks, client, config, top_k=3)
        for j, result in enumerate(faiss_results, 1):
            print(f"\n  [{j}] {result['title']} (Score: {result['score']:.4f})")
            print(f"      {result['text'][:150]}...")

        # BM25 检索
        print("\n🔍 BM25 检索结果 (Top-3):")
        bm25_results = retrieve_bm25(question, bm25_index, chunks, top_k=3)
        for j, result in enumerate(bm25_results, 1):
            print(f"\n  [{j}] {result['title']} (Score: {result['score']:.4f})")
            print(f"      {result['text'][:150]}...")


def main():
    """主函数"""
    print("=" * 60)
    print("Baseline RAG 功能测试")
    print("=" * 60)

    test_retrieval()

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    print("\n📝 如果检索结果正常，下一步:")
    print("   python scripts/04_build_hipporag.py")


if __name__ == "__main__":
    main()
