#!/usr/bin/env python3
"""
使用高质量KG运行HippoRAG实验
对比标准KG vs 高质量KG的效果
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

print('='*80)
print('HippoRAG实验 - 高质量知识图谱版本')
print('='*80)

# 加载数据
print('\n📚 加载数据和索引...')

validation = []
with open('data/raw/hotpotqa_validation.jsonl', 'r') as f:
    for line in f:
        validation.append(json.loads(line))
print(f'✅ 验证集: {len(validation)} 问题')

# 使用_full版本的索引（保持与之前一致）
faiss_index = faiss.read_index('data/indices/faiss/hotpotqa_full.index')
with open('data/indices/faiss/hotpotqa_full_docs.pkl', 'rb') as f:
    chunks = pickle.load(f)
print(f'✅ FAISS: {faiss_index.ntotal:,} 向量')

# 使用高质量版本的KG
print('\n📊 加载高质量KG...')
with open('data/knowledge_graphs/hotpotqa_kg_high_quality.gpickle', 'rb') as f:
    kg = pickle.load(f)
with open('data/knowledge_graphs/hotpotqa_pagerank_high_quality.pkl', 'rb') as f:
    pagerank_scores = pickle.load(f)

# 统计KG
entity_to_entity = sum(1 for u, v in kg.edges() if u.startswith('entity_') and v.startswith('entity_'))
print(f'✅ 知识图谱: {kg.number_of_nodes():,} 节点, {kg.number_of_edges():,} 边')
print(f'   Entity→Entity关系: {entity_to_entity:,} ({entity_to_entity/kg.number_of_edges()*100:.1f}%)')

# 初始化
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
nlp = spacy.load('en_core_web_sm')

# PPR缓存类（与之前相同）
class PPRCache:
    def __init__(self, kg, global_pagerank):
        self.kg = kg
        self.global_pr = global_pagerank
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_ppr(self, query_entity_ids):
        if not query_entity_ids:
            return self.global_pr

        cache_key = frozenset(query_entity_ids)
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1
        personalization = {node: 0.0 for node in self.kg.nodes}
        for eid in query_entity_ids:
            personalization[eid] = 1.0 / len(query_entity_ids)

        ppr_scores = nx.pagerank(self.kg, alpha=0.85, personalization=personalization, max_iter=100)
        self.cache[cache_key] = ppr_scores
        return ppr_scores

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

# 检查点函数
def load_checkpoint(checkpoint_file):
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return []

def save_checkpoint(results, checkpoint_file):
    Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)

def print_progress(current, total, start_time):
    if current == 0:
        return
    percent = current / total * 100
    elapsed = time.time() - start_time
    rate = elapsed / current
    remaining_min = (rate * (total - current)) / 60
    print(f'   [{current}/{total}] {percent:.1f}% - '
          f'用时: {elapsed/60:.1f}分钟 - '
          f'预计剩余: {remaining_min:.1f}分钟')

# 运行HippoRAG实验
print('\n' + '='*80)
print('运行 HippoRAG（高质量KG）')
print('='*80)

ppr_cache = PPRCache(kg, pagerank_scores)
results = load_checkpoint('results/checkpoint_hipporag_hq.json')
start_idx = len(results)

if start_idx > 0:
    print(f'📋 从checkpoint恢复，已完成: {start_idx}/{len(validation)}')

start_time = time.time()

for i in range(start_idx, len(validation)):
    question = validation[i]

    if i % 10 == 0 or i == start_idx:
        print_progress(i - start_idx, len(validation) - start_idx, start_time)

    query = question['question']
    gold_answer = question['answer']
    q_start_time = time.time()

    try:
        # 向量化查询
        resp = client.embeddings.create(model='text-embedding-3-small', input=[query])
        q_vec = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(q_vec)

        # FAISS检索（top 20）
        distances, indices = faiss_index.search(q_vec, 20)

        # 提取查询实体
        query_doc = nlp(query)
        query_entities = [ent.text.lower().strip() for ent in query_doc.ents]
        query_entity_ids = [f"entity_{e.replace(' ', '_')}" for e in query_entities]
        query_entity_ids = [eid for eid in query_entity_ids if kg.has_node(eid)]

        # 使用PPR缓存
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

        # 生成答案
        context = '\n\n'.join([f"Doc {j+1}: {doc['text']}" for j, doc in enumerate(retrieved_docs)])
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {query}

Answer (be concise):"""

        answer_resp = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=256
        )

        predicted = answer_resp.choices[0].message.content.strip()

        results.append({
            'question_id': question['id'],
            'question': query,
            'gold_answer': gold_answer,
            'predicted_answer': predicted,
            'latency': time.time() - q_start_time,
            'query_entities': query_entities,
            'success': True
        })

    except Exception as e:
        print(f'\n   ✗ Question {i} failed: {str(e)[:100]}')
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
        save_checkpoint(results, 'results/checkpoint_hipporag_hq.json')
        stats = ppr_cache.get_stats()
        print(f'\n   💾 Checkpoint saved at {i+1}/{len(validation)}')
        print(f'   📊 PPR缓存: 命中率 {stats["hit_rate"]*100:.1f}%, 缓存 {stats["cache_size"]}')

    # 每100个问题清理内存
    if (i + 1) % 100 == 0:
        gc.collect()

# 最终保存
save_checkpoint(results, 'results/checkpoint_hipporag_hq.json')

success = sum(1 for r in results if r['success'])
print(f'\n✅ HippoRAG (高质量KG) 完成: {success}/{len(results)} ({success/len(results)*100:.1f}%)')

# 显示缓存统计
stats = ppr_cache.get_stats()
print(f'\n📊 PPR缓存统计:')
print(f'   命中率: {stats["hit_rate"]*100:.1f}%')
print(f'   总命中: {stats["hits"]}')
print(f'   总未命中: {stats["misses"]}')
print(f'   缓存大小: {stats["cache_size"]}')

# 保存结果
Path('results/hipporag_high_quality').mkdir(parents=True, exist_ok=True)
with open('results/hipporag_high_quality/predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✅ 结果已保存到: results/hipporag_high_quality/predictions.json')
print()
print('下一步: 对比分析三种方法的效果')
print('运行: python scripts/compare_all_results.py')
