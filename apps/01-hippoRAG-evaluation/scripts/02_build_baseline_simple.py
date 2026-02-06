#!/usr/bin/env python3
"""
Baseline RAG 索引构建脚本（简化版，无tqdm）
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def load_config() -> Dict:
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_corpus() -> List[Dict]:
    print("📚 加载文档语料库...")
    sampled_path = project_root / "data" / "raw" / "hotpotqa_corpus_sampled.jsonl"
    full_path = project_root / "data" / "raw" / "hotpotqa_corpus.jsonl"

    corpus_path = sampled_path if sampled_path.exists() else full_path

    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"✅ 加载完成！总计 {len(corpus)} 个文档")
    return corpus


def chunk_documents(corpus: List[Dict], config: Dict) -> List[Dict]:
    print("\n✂️  文档分块...")

    chunk_config = config['document_processing']
    chunk_size_chars = chunk_config['chunk_size'] * 4
    chunk_overlap_chars = chunk_config['chunk_overlap'] * 4

    chunks = []

    for i, doc in enumerate(corpus):
        if i % 1000 == 0:
            print(f"   处理进度: {i}/{len(corpus)}")

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


def build_faiss_index(chunks: List[Dict], config: Dict) -> tuple:
    print("\n🔨 构建 FAISS 向量索引...")
    print(f"   处理 {len(chunks)} 个文档块...")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    embedding_config = config['embedding']
    model = embedding_config['model']
    batch_size = embedding_config['batch_size']

    texts = [chunk['text'] for chunk in chunks]
    all_embeddings = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"   总批次数: {total_batches}")

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        print(f"   向量化批次 {batch_num}/{total_batches}")

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

    print("🔨 构建 FAISS IndexFlatL2...")
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    print(f"✅ FAISS 索引构建完成！索引包含 {index.ntotal} 个向量")

    return index, chunks


def build_bm25_index(chunks: List[Dict], config: Dict) -> BM25Okapi:
    print("\n🔨 构建 BM25 索引...")

    tokenized_corpus = [chunk['text'].lower().split() for chunk in chunks]

    bm25_config = config['bm25']
    bm25 = BM25Okapi(
        tokenized_corpus,
        k1=bm25_config['k1'],
        b=bm25_config['b']
    )

    print(f"✅ BM25 索引构建完成！")

    return bm25


def save_indices(faiss_index, bm25_index, chunks: List[Dict], config: Dict) -> None:
    print("\n💾 保存索引...")

    faiss_dir = project_root / "data" / "indices" / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    faiss_index_path = faiss_dir / "hotpotqa.index"
    faiss.write_index(faiss_index, str(faiss_index_path))
    print(f"✅ FAISS 索引已保存: {faiss_index_path}")

    doc_mapping_path = faiss_dir / "hotpotqa_docs.pkl"
    with open(doc_mapping_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✅ 文档映射已保存: {doc_mapping_path}")

    bm25_dir = project_root / "data" / "indices" / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = bm25_dir / "hotpotqa_bm25.pkl"
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_index, f)
    print(f"✅ BM25 索引已保存: {bm25_path}")

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = processed_dir / "hotpotqa_chunks.jsonl"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"✅ 分块文档已保存: {chunks_path}")


def main():
    print("=" * 60)
    print("Baseline RAG 索引构建（简化版）")
    print("=" * 60)

    config = load_config()
    corpus = load_corpus()
    chunks = chunk_documents(corpus, config)
    faiss_index, chunks_with_embeddings = build_faiss_index(chunks, config)
    bm25_index = build_bm25_index(chunks, config)
    save_indices(faiss_index, bm25_index, chunks, config)

    print("\n" + "=" * 60)
    print("✅ Baseline RAG 构建完成！")
    print("=" * 60)
    print("\n📝 下一步:")
    print("   python scripts/03_test_baseline.py")


if __name__ == "__main__":
    main()
