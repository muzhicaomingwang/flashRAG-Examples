#!/usr/bin/env python3
"""
HNSW 索引构建脚本

功能：
1. 文档分块（与 Baseline 保持一致）
2. 构建 FAISS HNSW 向量索引
3. 保存索引到磁盘（独立路径）
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def load_config() -> Dict:
    """加载配置文件"""
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_corpus() -> List[Dict]:
    """加载文档语料库（使用采样版本以节省内存）"""
    print("📚 加载文档语料库...")
    sampled_path = project_root / "data" / "raw" / "hotpotqa_corpus_sampled.jsonl"
    full_path = project_root / "data" / "raw" / "hotpotqa_corpus.jsonl"

    corpus_path = sampled_path if sampled_path.exists() else full_path

    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"✅ 加载完成！总计 {len(corpus)} 个文档")
    if corpus_path == sampled_path:
        print("   （使用采样版本以节省内存）")
    return corpus


def load_existing_flat_index(config: Dict):
    """加载现有 FlatL2 索引与文档映射（用于复用向量）"""
    faiss_index_path = project_root / config['faiss']['persist_path']
    doc_mapping_path = project_root / config['faiss']['doc_mapping_path']

    if not faiss_index_path.exists() or not doc_mapping_path.exists():
        return None, None

    print("📦 发现现有 FlatL2 索引，直接复用向量...")
    faiss_index = faiss.read_index(str(faiss_index_path))
    with open(doc_mapping_path, 'rb') as f:
        chunks = pickle.load(f)

    return faiss_index, chunks


def chunk_documents(corpus: List[Dict], config: Dict) -> List[Dict]:
    """文档分块（优化版：使用字符数近似代替token数）"""
    print("\n✂️  文档分块...")

    chunk_config = config['document_processing']
    chunk_size_chars = chunk_config['chunk_size'] * 4
    chunk_overlap_chars = chunk_config['chunk_overlap'] * 4

    chunks = []

    for doc in tqdm(corpus, desc="分块进度"):
        doc_text = doc['text']
        text_length = len(doc_text)

        if text_length <= chunk_size_chars:
            chunks.append({
                "chunk_id": f"{doc['id']}_0",
                "doc_id": doc['id'],
                "title": doc['title'],
                "text": doc_text,
                "chunk_index": 0
            })
            continue

        start = 0
        chunk_index = 0

        while start < text_length:
            end = min(start + chunk_size_chars, text_length)
            chunk_text = doc_text[start:end]

            chunks.append({
                "chunk_id": f"{doc['id']}_{chunk_index}",
                "doc_id": doc['id'],
                "title": doc['title'],
                "text": chunk_text,
                "chunk_index": chunk_index
            })

            start = end - chunk_overlap_chars
            chunk_index += 1

    print(f"✅ 分块完成！总计 {len(chunks)} 个块")
    print(f"   平均每文档 {len(chunks) / len(corpus):.1f} 个块")

    return chunks


def build_hnsw_index(chunks: List[Dict], config: Dict):
    """构建 FAISS HNSW 索引（优先复用 FlatL2 向量）"""
    print("\n🔨 构建 FAISS HNSW 向量索引...")

    flat_index, flat_chunks = load_existing_flat_index(config)
    if flat_index is not None and flat_chunks is not None:
        chunks = flat_chunks
        embeddings_array = flat_index.reconstruct_n(0, flat_index.ntotal)
        print(f"✅ 复用向量完成！维度: {embeddings_array.shape}")
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        embedding_config = config['embedding']
        model = embedding_config['model']
        batch_size = embedding_config['batch_size']

        texts = [chunk['text'] for chunk in chunks]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="向量化进度"):
            batch_texts = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        embeddings_array = np.array(all_embeddings, dtype='float32')

        if config['faiss']['normalize_vectors']:
            faiss.normalize_L2(embeddings_array)

        print(f"✅ 向量化完成！维度: {embeddings_array.shape}")

    hnsw_cfg = config['faiss_hnsw']
    dimension = embeddings_array.shape[1]
    m = int(hnsw_cfg['m'])

    print(f"🔨 构建 FAISS IndexHNSWFlat (M={m})...")
    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = int(hnsw_cfg['ef_construction'])
    index.add(embeddings_array)

    print(f"✅ HNSW 索引构建完成！索引包含 {index.ntotal} 个向量")

    return index, chunks


def save_indices(faiss_index, chunks: List[Dict], config: Dict) -> None:
    """保存索引到磁盘"""
    print("\n💾 保存索引...")

    faiss_dir = project_root / "data" / "indices" / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    faiss_index_path = project_root / config['faiss_hnsw']['persist_path']
    faiss.write_index(faiss_index, str(faiss_index_path))
    print(f"✅ HNSW 索引已保存: {faiss_index_path}")

    doc_mapping_path = project_root / config['faiss_hnsw']['doc_mapping_path']
    with open(doc_mapping_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ 文档映射已保存: {doc_mapping_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("HNSW 索引构建")
    print("=" * 60)

    config = load_config()

    flat_index, flat_chunks = load_existing_flat_index(config)
    if flat_chunks is not None:
        chunks = flat_chunks
    else:
        corpus = load_corpus()
        chunks = chunk_documents(corpus, config)

    faiss_index, _ = build_hnsw_index(chunks, config)
    save_indices(faiss_index, chunks, config)

    print("\n" + "=" * 60)
    print("✅ HNSW 索引构建完成！")
    print("=" * 60)
    print("\n📝 下一步:")
    print("   python scripts/05_run_experiments_hnsw.py")


if __name__ == "__main__":
    main()
