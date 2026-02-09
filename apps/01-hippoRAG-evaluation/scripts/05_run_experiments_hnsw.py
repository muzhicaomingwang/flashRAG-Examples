#!/usr/bin/env python3
"""
HNSW 对比实验运行脚本

功能：
1. 运行 Baseline RAG（HNSW 检索）
2. 运行 HippoRAG（HNSW 初检索 + KG/PPR 重排）
3. 保存结果到独立目录
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import faiss
import spacy
import networkx as nx
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def load_config() -> Dict:
    """加载配置文件"""
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_validation_set() -> List[Dict]:
    """加载验证集"""
    val_path = project_root / "data" / "raw" / "hotpotqa_validation.jsonl"

    validation = []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 兼容 FlashRAG dev 格式：golden_answers -> answer
            if 'answer' not in item and 'golden_answers' in item:
                ga = item.get('golden_answers') or []
                item['answer'] = ga[0] if ga else ""
            validation.append(item)

    return validation


def load_hnsw_indices(config: Dict):
    """加载 HNSW 索引"""
    # 加载 FAISS HNSW 索引
    faiss_index_path = project_root / config['faiss_hnsw']['persist_path']
    faiss_index = faiss.read_index(str(faiss_index_path))

    # 设置 HNSW 搜索参数
    faiss_index.hnsw.efSearch = int(config['faiss_hnsw']['ef_search'])

    # 加载文档映射
    doc_mapping_path = project_root / config['faiss_hnsw']['doc_mapping_path']
    with open(doc_mapping_path, 'rb') as f:
        chunks = pickle.load(f)

    # 兼容旧映射：补齐 title/doc_id/chunk_index
    for chunk in chunks:
        if 'chunk_id' in chunk:
            doc_id, _, idx_str = chunk['chunk_id'].rpartition('_')
            if not doc_id:
                doc_id = chunk['chunk_id']
                idx_str = ""
            if 'doc_id' not in chunk:
                chunk['doc_id'] = doc_id
            if 'title' not in chunk:
                chunk['title'] = doc_id.replace('_', ' ')
            if 'chunk_index' not in chunk and idx_str.isdigit():
                chunk['chunk_index'] = int(idx_str)

    return faiss_index, chunks


def load_hipporag_kg():
    """加载 HippoRAG 知识图谱"""
    kg_path = project_root / "data" / "knowledge_graphs" / "hotpotqa_kg.gpickle"
    if hasattr(nx, "read_gpickle"):
        kg = nx.read_gpickle(str(kg_path))
    else:
        with open(kg_path, 'rb') as f:
            kg = pickle.load(f)

    pr_path = project_root / "data" / "knowledge_graphs" / "hotpotqa_pagerank.pkl"
    with open(pr_path, 'rb') as f:
        pagerank_scores = pickle.load(f)

    return kg, pagerank_scores


def extract_entities_spacy(text: str, nlp) -> List[Tuple[str, str]]:
    """使用 SpaCy 提取实体"""
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        normalized_name = ent.text.strip().lower()
        entities.append((normalized_name, ent.label_))

    return entities


class BaselineRAG:
    """Baseline RAG 系统"""

    def __init__(self, faiss_index, chunks, client, config):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.client = client
        self.config = config

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        response = self.client.embeddings.create(
            model=self.config['embedding']['model'],
            input=[query]
        )
        query_vector = np.array([response.data[0].embedding], dtype='float32')

        if self.config['faiss']['normalize_vectors']:
            faiss.normalize_L2(query_vector)

        distances, indices = self.faiss_index.search(query_vector, top_k)
        results = [self.chunks[idx] for idx in indices[0]]
        return results

    def answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """基于检索文档生成答案"""
        context = "\n\n".join([
            f"Document {i+1} ({doc['title']}):\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer (be concise):"""

        response = self.client.chat.completions.create(
            model=self.config['baseline_rag']['llm_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['baseline_rag']['llm_temperature'],
            max_completion_tokens=self.config['baseline_rag']['llm_max_tokens']
        )

        return response.choices[0].message.content.strip()


class HippoRAG:
    """HippoRAG 系统"""

    def __init__(self, faiss_index, chunks, kg, pagerank_scores, client, config, nlp):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.kg = kg
        self.pagerank_scores = pagerank_scores
        self.client = client
        self.config = config
        self.nlp = nlp

        self.chunk_id_to_idx = {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """使用 KG + PPR 检索"""
        initial_k = self.config['hipporag']['retrieval']['initial_k']

        response = self.client.embeddings.create(
            model=self.config['embedding']['model'],
            input=[query]
        )
        query_vector = np.array([response.data[0].embedding], dtype='float32')

        if self.config['faiss']['normalize_vectors']:
            faiss.normalize_L2(query_vector)

        distances, indices = self.faiss_index.search(query_vector, initial_k)

        query_entities = extract_entities_spacy(query, self.nlp)
        query_entity_ids = [f"entity_{name.replace(' ', '_')}" for name, _ in query_entities]
        query_entity_ids = [eid for eid in query_entity_ids if self.kg.has_node(eid)]

        if query_entity_ids:
            personalization = {node: 0.0 for node in self.kg.nodes}
            for eid in query_entity_ids:
                personalization[eid] = 1.0 / len(query_entity_ids)

            ppr_scores = nx.pagerank(
                self.kg,
                alpha=self.config['hipporag']['pagerank']['damping_factor'],
                personalization=personalization,
                max_iter=self.config['hipporag']['pagerank']['max_iterations']
            )
        else:
            ppr_scores = self.pagerank_scores

        candidate_chunks = [self.chunks[idx] for idx in indices[0]]

        reranked = []
        for chunk, distance in zip(candidate_chunks, distances[0]):
            chunk_id = chunk['chunk_id']
            ppr_score = ppr_scores.get(chunk_id, 0.0)
            retrieval_score = 1.0 / (1.0 + float(distance))
            combined_score = 0.5 * ppr_score + 0.5 * retrieval_score

            reranked.append({
                "chunk": chunk,
                "ppr_score": ppr_score,
                "retrieval_score": retrieval_score,
                "combined_score": combined_score
            })

        reranked.sort(key=lambda x: x['combined_score'], reverse=True)
        return [item['chunk'] for item in reranked[:top_k]]

    def answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """基于检索文档生成答案（与 Baseline 相同）"""
        context = "\n\n".join([
            f"Document {i+1} ({doc['title']}):\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer (be concise):"""

        response = self.client.chat.completions.create(
            model=self.config['hipporag']['retrieval']['llm_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['hipporag']['retrieval']['llm_temperature'],
            max_completion_tokens=self.config['hipporag']['retrieval']['llm_max_tokens']
        )

        return response.choices[0].message.content.strip()


def run_single_experiment(method_name: str, retriever, question: Dict) -> Dict:
    """运行单个问题的实验"""
    query = question['question']
    gold_answer = question['answer']

    start_time = time.time()

    try:
        retrieved_docs = retriever.retrieve(query)
        predicted_answer = retriever.answer(query, retrieved_docs)
        latency = time.time() - start_time

        return {
            "question_id": question['id'],
            "question": query,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "retrieved_docs": [doc['chunk_id'] for doc in retrieved_docs],
            "latency": latency,
            "success": True,
            "error": None
        }

    except Exception as e:
        return {
            "question_id": question['id'],
            "question": query,
            "gold_answer": gold_answer,
            "predicted_answer": "",
            "retrieved_docs": [],
            "latency": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def _load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    return []


def _save_checkpoint(checkpoint_path: Path, results: List[Dict]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'w') as f:
        json.dump(results, indent=2, fp=f)


def run_experiment(
    method_name: str,
    retriever,
    validation_set: List[Dict],
    config: Dict,
    checkpoint_path: Path,
) -> List[Dict]:
    """运行完整实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {method_name}")
    print(f"{'='*60}")

    results = _load_checkpoint(checkpoint_path)
    processed_ids = {r.get('question_id') for r in results if r.get('question_id')}

    max_workers = config['api']['max_concurrent_requests']
    total = len(validation_set)
    pending = [q for q in validation_set if q.get('id') not in processed_ids]
    if not pending:
        print(f"✅ 已完成，无需重复运行: {method_name}")
        return results

    checkpoint_every = 20

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for question in pending:
            future = executor.submit(run_single_experiment, method_name, retriever, question)
            futures.append(future)

        with tqdm(total=total, initial=len(processed_ids), desc=f"{method_name} 进度") as pbar:
            last_percent = int((pbar.n / pbar.total) * 100) if pbar.total else 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

                if len(results) % checkpoint_every == 0:
                    _save_checkpoint(checkpoint_path, results)

                percent = int((pbar.n / pbar.total) * 100) if pbar.total else 0
                if percent > last_percent:
                    print(f"{method_name} 进度: {percent}% ({pbar.n}/{pbar.total})")
                    last_percent = percent

    success_count = sum(1 for r in results if r['success'])
    print(f"\n✅ 实验完成！成功率: {success_count}/{len(results)}")

    _save_checkpoint(checkpoint_path, results)
    return results


def main():
    """主函数"""
    print("=" * 80)
    print("HippoRAG 对比实验（HNSW）")
    print("=" * 80)

    config = load_config()

    print("\n📚 加载数据...")
    validation_set = load_validation_set()
    print(f"✅ 验证集: {len(validation_set)} 个问题")

    is_full = len(validation_set) > 500
    suffix = "_full" if is_full else ""

    baseline_checkpoint = project_root / "results" / f"checkpoint_baseline_hnsw_{len(validation_set)}.json"
    hipporag_checkpoint = project_root / "results" / f"checkpoint_hipporag_hnsw_{len(validation_set)}.json"

    print("\n📚 加载索引...")
    faiss_index, chunks = load_hnsw_indices(config)
    print(f"✅ HNSW 索引: {faiss_index.ntotal:,} 个向量")

    kg, pagerank_scores = load_hipporag_kg()
    print(f"✅ 知识图谱: {kg.number_of_nodes():,} 个节点")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    nlp = spacy.load("en_core_web_sm")

    print("\n" + "="*80)
    print("实验 1/2: Baseline RAG（HNSW）")
    print("="*80)

    baseline_rag = BaselineRAG(faiss_index, chunks, client, config)
    baseline_results = run_experiment(
        "Baseline-RAG-HNSW",
        baseline_rag,
        validation_set,
        config,
        baseline_checkpoint,
    )

    results_dir = project_root / "results" / f"baseline_hnsw{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "predictions.json", 'w') as f:
        json.dump(baseline_results, indent=2, fp=f)

    print("\n" + "="*80)
    print("实验 2/2: HippoRAG（HNSW）")
    print("="*80)

    hipporag = HippoRAG(faiss_index, chunks, kg, pagerank_scores, client, config, nlp)
    hipporag_results = run_experiment(
        "HippoRAG-HNSW",
        hipporag,
        validation_set,
        config,
        hipporag_checkpoint,
    )

    results_dir = project_root / "results" / f"hipporag_hnsw{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "predictions.json", 'w') as f:
        json.dump(hipporag_results, indent=2, fp=f)

    print("\n" + "="*80)
    print("✅ HNSW 实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()
