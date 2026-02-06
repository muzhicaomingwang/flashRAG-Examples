#!/usr/bin/env python3
"""
HippoRAG 知识图谱构建脚本（优化版）

改进：
1. 70%覆盖率（而非30%）
2. 添加实体间关系（SpaCy依存句法）
3. 优化处理速度
"""

import json
import pickle
import random
from pathlib import Path
import spacy
import networkx as nx

print('='*60)
print('HippoRAG 知识图谱构建（优化版）')
print('='*60)

# 1. 加载文档块
print('\n📚 步骤1/5: 加载文档块')
chunks = []
chunks_path = Path('data/processed/hotpotqa_full_chunks.jsonl')

if not chunks_path.exists():
    print('⚠️  完整chunks文件不存在，使用10K版本')
    chunks_path = Path('data/processed/hotpotqa_chunks.jsonl')

with open(chunks_path, 'r') as f:
    for line in f:
        chunks.append(json.loads(line))

print(f'✅ 加载 {len(chunks)} 个块')

# 2. 采样70%（优化）
print('\n🎲 步骤2/5: 采样70%（优化from30%）')
random.seed(42)
sampling_ratio = 0.7
n_samples = int(len(chunks) * sampling_ratio)
sampled_chunks = random.sample(chunks, n_samples)
print(f'✅ 采样: {len(chunks)} → {len(sampled_chunks)} 块（{sampling_ratio*100:.0f}%）')

# 3. 加载SpaCy
print('\n🔧 步骤3/5: 加载SpaCy模型')
nlp = spacy.load('en_core_web_sm')
print('✅ SpaCy就绪')

# 4. 提取实体和关系
print('\n🔨 步骤4/5: 提取实体和关系')
kg = nx.DiGraph()
entity_counts = {}
relation_counts = {}

for i, chunk in enumerate(sampled_chunks):
    if i % 1000 == 0:
        print(f'   进度: {i}/{len(sampled_chunks)}')

    chunk_id = chunk['chunk_id']
    text = chunk['text']

    # 添加chunk节点
    kg.add_node(chunk_id, type='chunk', text=text[:500])  # 只保存前500字符节省内存

    # 提取实体
    doc = nlp(text)
    entities = [(ent.text.lower().strip(), ent.label_) for ent in doc.ents]
    unique_entities = list(set(entities))[:10]

    # 添加实体节点
    for entity_name, entity_type in unique_entities:
        entity_id = f"entity_{entity_name.replace(' ', '_')}"

        if not kg.has_node(entity_id):
            kg.add_node(entity_id, type='entity', name=entity_name, entity_type=entity_type)

        kg.add_edge(chunk_id, entity_id, relation='contains')
        entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1

    # 提取实体间关系（优化：使用依存句法）
    for token in doc:
        if token.dep_ in ['nsubj', 'nsubjpass'] and token.head.pos_ == 'VERB':
            # 主语
            subj = token.text.lower()
            verb = token.head.lemma_

            # 查找宾语
            for child in token.head.children:
                if child.dep_ in ['dobj', 'attr', 'pobj']:
                    obj = child.text.lower()

                    subj_id = f"entity_{subj.replace(' ', '_')}"
                    obj_id = f"entity_{obj.replace(' ', '_')}"

                    # 只连接图中已存在的实体
                    if kg.has_node(subj_id) and kg.has_node(obj_id):
                        kg.add_edge(subj_id, obj_id, relation=verb)
                        relation_counts[verb] = relation_counts.get(verb, 0) + 1

num_chunks = len([n for n in kg.nodes if kg.nodes[n]['type'] == 'chunk'])
num_entities = len([n for n in kg.nodes if kg.nodes[n]['type'] == 'entity'])

print(f'\n✅ 知识图谱构建完成')
print(f'   - 节点总数: {kg.number_of_nodes():,}')
print(f'   - 边总数: {kg.number_of_edges():,}')
print(f'   - Chunk节点: {num_chunks:,}')
print(f'   - 实体节点: {num_entities:,}')
print(f'   - 实体间关系: {sum(relation_counts.values()):,}')

# 5. PageRank
print('\n🔨 步骤5/5: 计算PageRank')
pagerank_scores = nx.pagerank(kg, alpha=0.85, max_iter=100)
print(f'✅ PageRank计算完成')

# Top实体
print(f'\n📊 Top-10 最重要实体:')
entity_scores = {n: s for n, s in pagerank_scores.items() if kg.nodes[n]['type'] == 'entity'}
for node_id, score in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    entity_name = kg.nodes[node_id]['name']
    print(f'   - {entity_name}: {score:.6f}')

# 保存
print('\n💾 保存知识图谱...')
Path('data/knowledge_graphs').mkdir(parents=True, exist_ok=True)

with open('data/knowledge_graphs/hotpotqa_kg_full.gpickle', 'wb') as f:
    pickle.dump(kg, f)
print(f'✅ 图谱已保存: hotpotqa_kg_full.gpickle')

with open('data/knowledge_graphs/hotpotqa_pagerank_full.pkl', 'wb') as f:
    pickle.dump(pagerank_scores, f)
print(f'✅ PageRank已保存')

# 统计
stats = {
    'num_nodes': kg.number_of_nodes(),
    'num_edges': kg.number_of_edges(),
    'num_chunks': num_chunks,
    'num_entities': num_entities,
    'num_entity_relations': sum(relation_counts.values()),
    'sampling_ratio': sampling_ratio,
    'top_entities': dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
    'top_relations': dict(sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:20])
}

with open('data/knowledge_graphs/hotpotqa_kg_full_stats.json', 'w') as f:
    json.dump(stats, indent=2, fp=f)
print(f'✅ 统计信息已保存')

print('\n' + '='*60)
print('🎉 优化版HippoRAG知识图谱构建成功！')
print('='*60)
print(f'\n改进点：')
print(f'  - 覆盖率: 30% → 70%')
print(f'  - 添加实体间关系: {sum(relation_counts.values()):,}条')
print(f'  - 图更密集，连接更丰富')
