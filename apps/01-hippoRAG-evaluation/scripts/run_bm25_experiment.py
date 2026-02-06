#!/usr/bin/env python3
"""
BM25稀疏检索实验
对比BM25、HippoRAG、FAISS的性能
"""

import json
import pickle
import time
import os
import gc
from pathlib import Path
from rank_bm25 import BM25Okapi
from openai import OpenAI

print('='*80)
print('BM25稀疏检索实验')
print('='*80)

# 加载数据
print('\n📚 加载数据...')

validation = []
with open('data/raw/hotpotqa_validation.jsonl', 'r') as f:
    for line in f:
        validation.append(json.loads(line))
print(f'✅ 验证集: {len(validation)} 问题')

# 加载BM25索引
with open('data/indices/bm25/hotpotqa_full_bm25.pkl', 'rb') as f:
    bm25_index = pickle.load(f)
print(f'✅ BM25索引: {len(bm25_index.doc_len):,} 文档')

# 加载chunks（与FAISS共用）
with open('data/indices/faiss/hotpotqa_full_docs.pkl', 'rb') as f:
    chunks = pickle.load(f)
print(f'✅ Chunks: {len(chunks):,}')

# 初始化
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# 辅助函数
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

def call_openai_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

# 运行实验
print('\n' + '='*80)
print('BM25稀疏检索实验')
print('='*80)

checkpoint_file = 'results/checkpoint_bm25.json'
results = load_checkpoint(checkpoint_file)
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
        # BM25检索（基于关键词）
        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)

        # 获取top-5
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        retrieved_docs = [chunks[idx] for idx in top_indices]

        # 构建context
        context = '\n\n'.join([f"Doc {j+1}: {doc['text']}" for j, doc in enumerate(retrieved_docs)])
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {query}

Answer (be concise):"""

        # 生成答案
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
        save_checkpoint(results, checkpoint_file)
        print(f'\n   💾 Checkpoint saved at {i+1}/{len(validation)}')

    # 每100个问题清理内存
    if (i + 1) % 100 == 0:
        gc.collect()

# 最终保存
save_checkpoint(results, checkpoint_file)

success = sum(1 for r in results if r['success'])
print(f'\n✅ BM25实验完成: {success}/{len(results)} ({success/len(results)*100:.1f}%)')

# 保存最终结果
Path('results/bm25').mkdir(parents=True, exist_ok=True)
with open('results/bm25/predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✅ 结果已保存到: results/bm25/predictions.json')
print()
print('下一步: 生成四方对比报告')
print('运行: python scripts/compare_four_methods.py')
