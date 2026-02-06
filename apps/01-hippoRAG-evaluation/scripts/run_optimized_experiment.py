#!/usr/bin/env python3
"""
优化版对比实验
使用完整66K文档索引和70%覆盖的KG
"""

import json
import pickle
import time
import os
from pathlib import Path
import numpy as np
import faiss
import spacy
import networkx as nx
from openai import OpenAI

print('='*80)
print('优化版HippoRAG对比实验（66K文档）')
print('='*80)

# 加载
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

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
nlp = spacy.load('en_core_web_sm')

chunk_id_to_idx = {chunk['chunk_id']: i for i, chunk in enumerate(chunks)}

# 实验1: Baseline
print('\n' + '='*80)
print('实验 1/2: Baseline RAG（66K文档）')
print('='*80)

baseline_results = []

for i, question in enumerate(validation):
    if i % 50 == 0:
        print(f'   进度: {i}/{len(validation)}')
    
    query = question['question']
    gold_answer = question['answer']
    start_time = time.time()
    
    try:
        resp = client.embeddings.create(model='text-embedding-3-small', input=[query])
        q_vec = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(q_vec)
        
        distances, indices = faiss_index.search(q_vec, 5)
        retrieved_docs = [chunks[idx] for idx in indices[0]]
        
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
        
        baseline_results.append({
            'question_id': question['id'],
            'question': query,
            'gold_answer': gold_answer,
            'predicted_answer': predicted,
            'latency': time.time() - start_time,
            'success': True
        })
        
    except Exception as e:
        baseline_results.append({
            'question_id': question['id'],
            'question': query,
            'gold_answer': gold_answer,
            'predicted_answer': '',
            'latency': time.time() - start_time,
            'success': False,
            'error': str(e)
        })

success = sum(1 for r in baseline_results if r['success'])
print(f'\n✅ Baseline完成: {success}/{len(baseline_results)}')

Path('results/baseline_full').mkdir(parents=True, exist_ok=True)
with open('results/baseline_full/predictions.json', 'w') as f:
    json.dump(baseline_results, indent=2, fp=f)

# 实验2: HippoRAG
print('\n' + '='*80)
print('实验 2/2: HippoRAG（66K文档，70% KG）')
print('='*80)

hipporag_results = []

for i, question in enumerate(validation):
    if i % 50 == 0:
        print(f'   进度: {i}/{len(validation)}')
    
    query = question['question']
    gold_answer = question['answer']
    start_time = time.time()
    
    try:
        resp = client.embeddings.create(model='text-embedding-3-small', input=[query])
        q_vec = np.array([resp.data[0].embedding], dtype='float32')
        faiss.normalize_L2(q_vec)
        
        distances, indices = faiss_index.search(q_vec, 20)
        
        query_doc = nlp(query)
        query_entities = [ent.text.lower().strip() for ent in query_doc.ents]
        query_entity_ids = [f"entity_{e.replace(' ', '_')}" for e in query_entities]
        query_entity_ids = [eid for eid in query_entity_ids if kg.has_node(eid)]
        
        if query_entity_ids:
            personalization = {node: 0.0 for node in kg.nodes}
            for eid in query_entity_ids:
                personalization[eid] = 1.0 / len(query_entity_ids)
            ppr_scores = nx.pagerank(kg, alpha=0.85, personalization=personalization, max_iter=100)
        else:
            ppr_scores = pagerank_scores
        
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
        
        hipporag_results.append({
            'question_id': question['id'],
            'question': query,
            'gold_answer': gold_answer,
            'predicted_answer': predicted,
            'latency': time.time() - start_time,
            'success': True
        })
        
    except Exception as e:
        hipporag_results.append({
            'question_id': question['id'],
            'question': query,
            'gold_answer': gold_answer,
            'predicted_answer': '',
            'latency': time.time() - start_time,
            'success': False,
            'error': str(e)
        })

success = sum(1 for r in hipporag_results if r['success'])
print(f'\n✅ HippoRAG完成: {success}/{len(hipporag_results)}')

Path('results/hipporag_full').mkdir(parents=True, exist_ok=True)
with open('results/hipporag_full/predictions.json', 'w') as f:
    json.dump(hipporag_results, indent=2, fp=f)

print('\n' + '='*80)
print('✅ 优化版实验完成！')
print('='*80)
