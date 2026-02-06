#!/usr/bin/env python3
"""
完整实验运行脚本

功能：
1. 运行 Baseline RAG
2. 运行 HippoRAG-Simple
3. 运行 HippoRAG-Full
4. 评估并保存结果
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict
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
            validation.append(json.loads(line))

    return validation


def load_baseline_indices():
    """加载 Baseline RAG 索引"""
    # 加载 FAISS 索引
    faiss_index_path = project_root / "data" / "indices" / "faiss" / "hotpotqa.index"
    faiss_index = faiss.read_index(str(faiss_index_path))

    # 加载文档映射
    doc_mapping_path = project_root / "data" / "indices" / "faiss" / "hotpotqa_docs.pkl"
    with open(doc_mapping_path, 'rb') as f:
        chunks = pickle.load(f)

    return faiss_index, chunks


def load_hipporag_kg():
    """加载 HippoRAG 知识图谱"""
    kg_path = project_root / "data" / "knowledge_graphs" / "hotpotqa_kg.gpickle"
    kg = nx.read_gpickle(str(kg_path))

    pr_path = project_root / "data" / "knowledge_graphs" / "hotpotqa_pagerank.pkl"
    with open(pr_path, 'rb') as f:
        pagerank_scores = pickle.load(f)

    return kg, pagerank_scores


class BaselineRAG:
    """Baseline RAG 系统"""

    def __init__(self, faiss_index, chunks, client, config):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.client = client
        self.config = config

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        # 向量化查询
        response = self.client.embeddings.create(
            model=self.config['embedding']['model'],
            input=[query]
        )
        query_vector = np.array([response.data[0].embedding], dtype='float32')

        # 归一化
        if self.config['faiss']['normalize_vectors']:
            faiss.normalize_L2(query_vector)

        # 检索
        distances, indices = self.faiss_index.search(query_vector, top_k)

        # 获取文档
        results = [self.chunks[idx] for idx in indices[0]]
        return results

    def answer(self, query: str, retrieved_docs: List[Dict]) -> str:
        """基于检索文档生成答案"""
        # 构建 prompt
        context = "\n\n".join([
            f"Document {i+1} ({doc['title']}):\n{doc['text']}"
            for i, doc in enumerate(retrieved_docs)
        ])

        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer (be concise):"""

        # 调用 LLM
        response = self.client.chat.completions.create(
            model=self.config['baseline_rag']['llm_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config['baseline_rag']['llm_temperature'],
            max_tokens=self.config['baseline_rag']['llm_max_tokens']
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

        # 构建 chunk_id -> index 的映射
        self.chunk_id_to_idx = {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """使用 KG + PPR 检索"""
        # Step 1: 初检索（获取候选集）
        initial_k = self.config['hipporag']['retrieval']['initial_k']

        response = self.client.embeddings.create(
            model=self.config['embedding']['model'],
            input=[query]
        )
        query_vector = np.array([response.data[0].embedding], dtype='float32')

        if self.config['faiss']['normalize_vectors']:
            faiss.normalize_L2(query_vector)

        distances, indices = self.faiss_index.search(query_vector, initial_k)

        # Step 2: 从查询中提取实体
        query_entities = extract_entities_spacy(query, self.nlp)
        query_entity_ids = [f"entity_{name.replace(' ', '_')}" for name, _ in query_entities]

        # 过滤：只保留图中存在的实体
        query_entity_ids = [eid for eid in query_entity_ids if self.kg.has_node(eid)]

        # Step 3: 个性化 PageRank
        if query_entity_ids:
            # 构建个性化向量（查询实体的权重更高）
            personalization = {node: 0.0 for node in self.kg.nodes}
            for eid in query_entity_ids:
                personalization[eid] = 1.0 / len(query_entity_ids)

            # 计算个性化 PageRank
            ppr_scores = nx.pagerank(
                self.kg,
                alpha=self.config['hipporag']['pagerank']['damping_factor'],
                personalization=personalization,
                max_iter=self.config['hipporag']['pagerank']['max_iterations']
            )
        else:
            # 如果没有找到实体，使用全局 PageRank
            ppr_scores = self.pagerank_scores

        # Step 4: 对候选文档块重新排序
        candidate_chunks = [self.chunks[idx] for idx in indices[0]]

        # 计算综合得分（PPR + 初检索距离）
        reranked = []
        for chunk, distance in zip(candidate_chunks, distances[0]):
            chunk_id = chunk['chunk_id']

            # PPR 分数（如果chunk在图中）
            ppr_score = ppr_scores.get(chunk_id, 0.0)

            # 初检索分数（距离越小越好，转换为相似度）
            retrieval_score = 1.0 / (1.0 + float(distance))

            # 综合得分
            combined_score = 0.5 * ppr_score + 0.5 * retrieval_score

            reranked.append({
                "chunk": chunk,
                "ppr_score": ppr_score,
                "retrieval_score": retrieval_score,
                "combined_score": combined_score
            })

        # 按综合得分排序
        reranked.sort(key=lambda x: x['combined_score'], reverse=True)

        # 返回 top-k
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
            max_tokens=self.config['hipporag']['retrieval']['llm_max_tokens']
        )

        return response.choices[0].message.content.strip()


def run_single_experiment(method_name: str, retriever, question: Dict) -> Dict:
    """运行单个问题的实验"""
    query = question['question']
    gold_answer = question['answer']

    # 记录开始时间
    start_time = time.time()

    try:
        # 检索
        retrieved_docs = retriever.retrieve(query)

        # 生成答案
        predicted_answer = retriever.answer(query, retrieved_docs)

        # 记录时间
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


def run_experiment(method_name: str, retriever, validation_set: List[Dict], config: Dict) -> List[Dict]:
    """运行完整实验"""
    print(f"\n{'='*60}")
    print(f"运行实验: {method_name}")
    print(f"{'='*60}")

    results = []

    # 并行运行（加速）
    max_workers = config['api']['max_concurrent_requests']

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for question in validation_set:
            future = executor.submit(run_single_experiment, method_name, retriever, question)
            futures.append(future)

        # 使用进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{method_name} 进度"):
            result = future.result()
            results.append(result)

    # 统计成功率
    success_count = sum(1 for r in results if r['success'])
    print(f"\n✅ 实验完成！成功率: {success_count}/{len(results)}")

    return results


def main():
    """主函数"""
    print("=" * 80)
    print("HippoRAG 对比实验")
    print("=" * 80)

    # 加载配置
    config = load_config()

    # 加载验证集
    print("\n📚 加载数据...")
    validation_set = load_validation_set()
    print(f"✅ 验证集: {len(validation_set)} 个问题")

    # 加载索引
    print("\n📚 加载索引...")
    faiss_index, chunks = load_baseline_indices()
    print(f"✅ FAISS 索引: {faiss_index.ntotal:,} 个向量")

    kg, pagerank_scores = load_hipporag_kg()
    print(f"✅ 知识图谱: {kg.number_of_nodes():,} 个节点")

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 加载 SpaCy
    nlp = spacy.load("en_core_web_sm")

    # 实验 1: Baseline RAG
    print("\n" + "="*80)
    print("实验 1/3: Baseline RAG")
    print("="*80)

    baseline_rag = BaselineRAG(faiss_index, chunks, client, config)
    baseline_results = run_experiment("Baseline-RAG", baseline_rag, validation_set, config)

    # 保存中间结果
    results_dir = project_root / "results" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "predictions.json", 'w') as f:
        json.dump(baseline_results, indent=2, fp=f)

    # 实验 2 & 3: HippoRAG
    print("\n" + "="*80)
    print("实验 2/3: HippoRAG")
    print("="*80)

    hipporag = HippoRAG(faiss_index, chunks, kg, pagerank_scores, client, config, nlp)
    hipporag_results = run_experiment("HippoRAG", hipporag, validation_set, config)

    # 保存中间结果
    results_dir = project_root / "results" / "hipporag"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "predictions.json", 'w') as f:
        json.dump(hipporag_results, indent=2, fp=f)

    print("\n" + "="*80)
    print("✅ 所有实验完成！")
    print("="*80)
    print("\n📝 下一步:")
    print("   python scripts/06_generate_report.py")


if __name__ == "__main__":
    main()
