#!/usr/bin/env python3
"""
优化版对比实验（安全版本）
使用完整66K文档索引和70%覆盖的KG

改进:
1. PPR缓存避免重复计算
2. 断点续传机制
3. API重试机制
4. 内存管理
5. 详细进度显示
"""

import json
import pickle
import time
import os
import gc
from pathlib import Path
import numpy as np
import faiss
import spacy
import networkx as nx
from openai import OpenAI


class PPRCache:
    """
    Personalized PageRank缓存类
    避免对相同实体组合重复计算PPR
    """

    def __init__(self, kg, global_pagerank):
        self.kg = kg
        self.global_pr = global_pagerank
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_ppr(self, query_entity_ids):
        """获取PPR分数，使用缓存"""
        if not query_entity_ids:
            return self.global_pr

        # 使用frozenset作为缓存key（顺序无关）
        cache_key = frozenset(query_entity_ids)

        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        # 缓存未命中，计算PPR
        self.misses += 1
        personalization = {node: 0.0 for node in self.kg.nodes}
        for eid in query_entity_ids:
            personalization[eid] = 1.0 / len(query_entity_ids)

        ppr_scores = nx.pagerank(
            self.kg,
            alpha=0.85,
            personalization=personalization,
            max_iter=100
        )

        # 缓存结果
        self.cache[cache_key] = ppr_scores
        return ppr_scores

    def get_stats(self):
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


def load_checkpoint(checkpoint_file):
    """加载checkpoint"""
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return []


def save_checkpoint(results, checkpoint_file):
    """保存checkpoint"""
    Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)


def call_openai_with_retry(func, max_retries=3, backoff=2):
    """
    调用OpenAI API with重试机制
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = backoff ** attempt
            print(f'   ⚠️  API error, retry {attempt+1}/{max_retries} in {wait_time}s: {str(e)[:50]}')
            time.sleep(wait_time)


def print_progress(current, total, start_time, experiment_name=""):
    """打印进度信息"""
    if current == 0:
        return

    percent = current / total * 100
    elapsed = time.time() - start_time
    rate = elapsed / current
    remaining = rate * (total - current)
    remaining_min = remaining / 60

    print(f'   [{current}/{total}] {percent:.1f}% - '
          f'用时: {elapsed/60:.1f}分钟 - '
          f'预计剩余: {remaining_min:.1f}分钟 - '
          f'平均: {rate:.1f}秒/问题')


def run_baseline_experiment(validation, faiss_index, chunks, client, checkpoint_file):
    """运行Baseline RAG实验"""
    print('\n' + '='*80)
    print('实验 1/2: Baseline RAG（66K文档）')
    print('='*80)

    # 加载checkpoint
    results = load_checkpoint(checkpoint_file)
    start_idx = len(results)

    if start_idx > 0:
        print(f'📋 从checkpoint恢复，已完成: {start_idx}/{len(validation)}')

    start_time = time.time()

    for i in range(start_idx, len(validation)):
        question = validation[i]

        # 显示进度
        if i % 10 == 0 or i == start_idx:
            print_progress(i - start_idx, len(validation) - start_idx, start_time, "Baseline")

        query = question['question']
        gold_answer = question['answer']
        q_start_time = time.time()

        try:
            # 向量化查询（with重试）
            resp = call_openai_with_retry(
                lambda: client.embeddings.create(model='text-embedding-3-small', input=[query])
            )
            q_vec = np.array([resp.data[0].embedding], dtype='float32')
            faiss.normalize_L2(q_vec)

            # FAISS检索
            distances, indices = faiss_index.search(q_vec, 5)
            retrieved_docs = [chunks[idx] for idx in indices[0]]

            # 构建context
            context = '\n\n'.join([f"Doc {j+1}: {doc['text']}" for j, doc in enumerate(retrieved_docs)])
            prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {query}

Answer (be concise):"""

            # 生成答案（with重试）
            answer_resp = call_openai_with_retry(
                lambda: client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=256
                )
            )

            predicted = answer_resp.choices[0].message.content.strip()

            results.append({
                'question_id': question['id'],
                'question': query,
                'gold_answer': gold_answer,
                'predicted_answer': predicted,
                'latency': time.time() - q_start_time,
                'success': True
            })

        except Exception as e:
            print(f'   ✗ Question {i} failed: {str(e)[:100]}')
            results.append({
                'question_id': question['id'],
                'question': query,
                'gold_answer': gold_answer,
                'predicted_answer': '',
                'latency': time.time() - q_start_time,
                'success': False,
                'error': str(e)
            })

        # 每50个问题保存checkpoint
        if (i + 1) % 50 == 0:
            save_checkpoint(results, checkpoint_file)
            print(f'   💾 Checkpoint saved at {i+1}/{len(validation)}')

        # 每100个问题清理内存
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f'   🧹 Memory cleanup at {i+1}')

    # 最终保存
    save_checkpoint(results, checkpoint_file)

    success = sum(1 for r in results if r['success'])
    print(f'\n✅ Baseline完成: {success}/{len(results)} ({success/len(results)*100:.1f}%)')

    return results


def run_hipporag_experiment(validation, faiss_index, chunks, kg, pagerank_scores,
                            client, nlp, checkpoint_file):
    """运行HippoRAG实验"""
    print('\n' + '='*80)
    print('实验 2/2: HippoRAG（66K文档，70% KG）')
    print('='*80)

    # 初始化PPR缓存
    ppr_cache = PPRCache(kg, pagerank_scores)
    print(f'🔄 PPR缓存已初始化（全局PageRank包含 {len(pagerank_scores):,} 节点）')

    # 加载checkpoint
    results = load_checkpoint(checkpoint_file)
    start_idx = len(results)

    if start_idx > 0:
        print(f'📋 从checkpoint恢复，已完成: {start_idx}/{len(validation)}')

    start_time = time.time()

    for i in range(start_idx, len(validation)):
        question = validation[i]

        # 显示进度
        if i % 10 == 0 or i == start_idx:
            print_progress(i - start_idx, len(validation) - start_idx, start_time, "HippoRAG")

        query = question['question']
        gold_answer = question['answer']
        q_start_time = time.time()

        try:
            # 向量化查询（with重试）
            resp = call_openai_with_retry(
                lambda: client.embeddings.create(model='text-embedding-3-small', input=[query])
            )
            q_vec = np.array([resp.data[0].embedding], dtype='float32')
            faiss.normalize_L2(q_vec)

            # FAISS检索（top 20）
            distances, indices = faiss_index.search(q_vec, 20)

            # 提取查询实体
            query_doc = nlp(query)
            query_entities = [ent.text.lower().strip() for ent in query_doc.ents]
            query_entity_ids = [f"entity_{e.replace(' ', '_')}" for e in query_entities]
            query_entity_ids = [eid for eid in query_entity_ids if kg.has_node(eid)]

            # 使用PPR缓存获取分数
            ppr_scores = ppr_cache.get_ppr(query_entity_ids)

            # 重新排序
            candidate_chunks = [chunks[idx] for idx in indices[0]]
            reranked = []

            for chunk, distance in zip(candidate_chunks, distances[0]):
                chunk_id = chunk['chunk_id']
                ppr_score = ppr_scores.get(chunk_id, 0.0)
                retrieval_score = 1.0 / (1.0 + float(distance))
                combined_score = 0.5 * ppr_score + 0.5 * retrieval_score
                reranked.append({'chunk': chunk, 'combined_score': combined_score})

            reranked.sort(key=lambda x: x['combined_score'], reverse=True)
            retrieved_docs = [item['chunk'] for item in reranked[:5]]

            # 构建context
            context = '\n\n'.join([f"Doc {j+1}: {doc['text']}" for j, doc in enumerate(retrieved_docs)])
            prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {query}

Answer (be concise):"""

            # 生成答案（with重试）
            answer_resp = call_openai_with_retry(
                lambda: client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=256
                )
            )

            predicted = answer_resp.choices[0].message.content.strip()

            results.append({
                'question_id': question['id'],
                'question': query,
                'gold_answer': gold_answer,
                'predicted_answer': predicted,
                'latency': time.time() - q_start_time,
                'query_entities': query_entities,
                'ppr_cache_hit': len(query_entity_ids) > 0 and ppr_cache.hits > 0,
                'success': True
            })

        except Exception as e:
            print(f'   ✗ Question {i} failed: {str(e)[:100]}')
            results.append({
                'question_id': question['id'],
                'question': query,
                'gold_answer': gold_answer,
                'predicted_answer': '',
                'latency': time.time() - q_start_time,
                'success': False,
                'error': str(e)
            })

        # 每50个问题保存checkpoint和显示缓存统计
        if (i + 1) % 50 == 0:
            save_checkpoint(results, checkpoint_file)
            stats = ppr_cache.get_stats()
            print(f'   💾 Checkpoint saved at {i+1}/{len(validation)}')
            print(f'   📊 PPR缓存: 命中率 {stats["hit_rate"]*100:.1f}%, '
                  f'命中 {stats["hits"]}, 未命中 {stats["misses"]}, '
                  f'缓存大小 {stats["cache_size"]}')

        # 每100个问题清理内存
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f'   🧹 Memory cleanup at {i+1}')

    # 最终保存
    save_checkpoint(results, checkpoint_file)

    success = sum(1 for r in results if r['success'])
    print(f'\n✅ HippoRAG完成: {success}/{len(results)} ({success/len(results)*100:.1f}%)')

    # 显示最终缓存统计
    stats = ppr_cache.get_stats()
    print(f'\n📊 PPR缓存统计:')
    print(f'   命中率: {stats["hit_rate"]*100:.1f}%')
    print(f'   总命中: {stats["hits"]}')
    print(f'   总未命中: {stats["misses"]}')
    print(f'   缓存大小: {stats["cache_size"]}')

    return results


def main():
    """主函数"""
    print('='*80)
    print('优化版HippoRAG对比实验（66K文档）- 安全版本')
    print('='*80)

    # 加载数据
    print('\n📚 加载数据和索引...')

    validation = []
    with open('data/raw/hotpotqa_validation.jsonl', 'r') as f:
        for line in f:
            validation.append(json.loads(line))
    print(f'✅ 验证集: {len(validation)} 问题')

    # 使用_full版本的索引
    faiss_index = faiss.read_index('data/indices/faiss/hotpotqa_full.index')
    with open('data/indices/faiss/hotpotqa_full_docs.pkl', 'rb') as f:
        chunks = pickle.load(f)
    print(f'✅ FAISS: {faiss_index.ntotal:,} 向量')

    # 使用_full版本的KG
    with open('data/knowledge_graphs/hotpotqa_kg_full.gpickle', 'rb') as f:
        kg = pickle.load(f)
    with open('data/knowledge_graphs/hotpotqa_pagerank_full.pkl', 'rb') as f:
        pagerank_scores = pickle.load(f)
    print(f'✅ 知识图谱: {kg.number_of_nodes():,} 节点, {kg.number_of_edges():,} 边')

    # 初始化客户端
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    nlp = spacy.load('en_core_web_sm')
    print('✅ OpenAI客户端和SpaCy已初始化')

    # 运行实验1: Baseline
    baseline_results = run_baseline_experiment(
        validation,
        faiss_index,
        chunks,
        client,
        'results/checkpoint_baseline.json'
    )

    # 保存最终结果
    Path('results/baseline_full').mkdir(parents=True, exist_ok=True)
    with open('results/baseline_full/predictions.json', 'w') as f:
        json.dump(baseline_results, indent=2, fp=f)

    # 运行实验2: HippoRAG
    hipporag_results = run_hipporag_experiment(
        validation,
        faiss_index,
        chunks,
        kg,
        pagerank_scores,
        client,
        nlp,
        'results/checkpoint_hipporag.json'
    )

    # 保存最终结果
    Path('results/hipporag_full').mkdir(parents=True, exist_ok=True)
    with open('results/hipporag_full/predictions.json', 'w') as f:
        json.dump(hipporag_results, indent=2, fp=f)

    print('\n' + '='*80)
    print('✅ 优化版实验完成！')
    print('='*80)
    print(f'\n结果已保存到:')
    print(f'  - results/baseline_full/predictions.json')
    print(f'  - results/hipporag_full/predictions.json')


if __name__ == '__main__':
    main()
