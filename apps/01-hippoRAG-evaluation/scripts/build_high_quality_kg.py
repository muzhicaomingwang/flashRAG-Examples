#!/usr/bin/env python3
"""
构建高质量知识图谱
使用LLM提取实体间的关系，增加图密度
"""

import json
import pickle
import os
from pathlib import Path
import networkx as nx
from openai import OpenAI
import spacy
from tqdm import tqdm
import time

print('='*80)
print('构建高质量知识图谱（增加实体间关系）')
print('='*80)

# 初始化
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
nlp = spacy.load('en_core_web_sm')

# 加载chunks
print('\n📚 加载数据...')
with open('data/indices/faiss/hotpotqa_full_docs.pkl', 'rb') as f:
    chunks = pickle.load(f)

# 采样70%用于KG构建（与之前保持一致）
import random
random.seed(42)
sampled_chunks = random.sample(chunks, int(len(chunks) * 0.7))
print(f'✅ 加载了 {len(chunks):,} chunks')
print(f'✅ 采样 {len(sampled_chunks):,} chunks (70%) 用于KG构建')

# 创建新的KG
kg = nx.Graph()
print('\n🔨 构建知识图谱...')

# 第一阶段：添加chunk和实体节点
print('\n第一阶段：提取实体并建立Chunk→Entity连接')
chunk_entities_map = {}

for i, chunk in enumerate(tqdm(sampled_chunks, desc='提取实体')):
    chunk_id = chunk['chunk_id']
    text = chunk['text']

    # 添加chunk节点
    kg.add_node(chunk_id, type='chunk', text=text[:200])

    # 提取实体
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entity_text = ent.text.lower().strip()
        entity_id = f"entity_{entity_text.replace(' ', '_')}"
        entities.append(entity_id)

        # 添加实体节点
        if not kg.has_node(entity_id):
            kg.add_node(entity_id, type='entity', name=entity_text, label=ent.label_)

        # 添加chunk→entity边
        kg.add_edge(chunk_id, entity_id, relation='contains')

    chunk_entities_map[chunk_id] = entities

print(f'\n✅ 第一阶段完成:')
print(f'   节点数: {kg.number_of_nodes():,}')
print(f'   边数: {kg.number_of_edges():,}')

# 第二阶段：使用LLM提取实体间关系
print('\n第二阶段：使用LLM提取实体间关系')
print('这将调用OpenAI API，预计成本: $2-3, 时间: 30-40分钟')
print()

# 只对包含2个以上实体的chunk提取关系
chunks_with_multiple_entities = [
    (cid, ents) for cid, ents in chunk_entities_map.items()
    if len(ents) >= 2
]

print(f'有 {len(chunks_with_multiple_entities):,} 个chunks包含2+实体')
print(f'将为这些chunks提取实体间关系...')
print()

# 采样策略：随机选择5000个chunks进行关系提取（平衡成本和质量）
MAX_CHUNKS_FOR_RELATION = 5000
if len(chunks_with_multiple_entities) > MAX_CHUNKS_FOR_RELATION:
    chunks_to_process = random.sample(chunks_with_multiple_entities, MAX_CHUNKS_FOR_RELATION)
    print(f'⚠️  为控制成本，采样 {MAX_CHUNKS_FOR_RELATION:,} 个chunks进行关系提取')
else:
    chunks_to_process = chunks_with_multiple_entities

total_relations_added = 0
api_calls = 0
total_cost = 0

# 批处理：每次处理一个chunk
for i, (chunk_id, entities) in enumerate(tqdm(chunks_to_process, desc='提取关系')):
    if len(entities) < 2:
        continue

    # 获取chunk文本
    chunk_text = next(c['text'] for c in sampled_chunks if c['chunk_id'] == chunk_id)

    # 提取实体名称（去掉entity_前缀）
    entity_names = [e.replace('entity_', '').replace('_', ' ') for e in entities]

    # 构建prompt
    prompt = f"""Extract relationships between entities in this text. Only include direct, explicit relationships.

Text: {chunk_text[:500]}

Entities: {', '.join(entity_names[:10])}

Output format (JSON):
{{"relations": [
  {{"subject": "entity1", "relation": "verb/relationship", "object": "entity2"}},
  ...
]}}

Rules:
- Only include relationships explicitly stated in the text
- Use simple verbs (e.g., "founded", "located_in", "part_of", "won")
- Maximum 5 relations per text
- Return empty list if no clear relationships"""

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=300
        )

        api_calls += 1
        cost = response.usage.total_tokens * 0.000002
        total_cost += cost

        # 解析结果
        result_text = response.choices[0].message.content

        try:
            # 尝试解析JSON
            result = json.loads(result_text)
            relations = result.get('relations', [])

            # 添加关系到KG
            for rel in relations[:5]:  # 最多5个关系
                subj = rel.get('subject', '').lower().strip().replace(' ', '_')
                relation = rel.get('relation', '').lower().strip()
                obj = rel.get('object', '').lower().strip().replace(' ', '_')

                if subj and obj and relation:
                    subj_id = f"entity_{subj}"
                    obj_id = f"entity_{obj}"

                    # 只添加KG中已存在的实体间的关系
                    if kg.has_node(subj_id) and kg.has_node(obj_id):
                        kg.add_edge(subj_id, obj_id, relation=relation, source='llm')
                        total_relations_added += 1

        except json.JSONDecodeError:
            # LLM返回格式不正确，跳过
            pass

        # 每100次显示进度
        if (i + 1) % 100 == 0:
            print(f'\n   处理: {i+1}/{len(chunks_to_process)}')
            print(f'   API调用: {api_calls}, 成本: ${total_cost:.2f}')
            print(f'   已添加关系: {total_relations_added}')

        # 限流（避免超过API速率限制）
        if (i + 1) % 50 == 0:
            time.sleep(1)

    except Exception as e:
        print(f'\n   ⚠️  错误: {str(e)[:100]}')
        continue

print(f'\n✅ 第二阶段完成:')
print(f'   API调用: {api_calls}')
print(f'   总成本: ${total_cost:.2f}')
print(f'   添加的实体间关系: {total_relations_added:,}')

# 统计最终KG
chunk_nodes = [n for n in kg.nodes if not n.startswith('entity_')]
entity_nodes = [n for n in kg.nodes if n.startswith('entity_')]

chunk_to_entity_edges = 0
entity_to_entity_edges = 0

for u, v in kg.edges():
    if (not u.startswith('entity_')) and v.startswith('entity_'):
        chunk_to_entity_edges += 1
    elif u.startswith('entity_') and v.startswith('entity_'):
        entity_to_entity_edges += 1

print(f'\n📊 最终知识图谱统计:')
print(f'   总节点: {kg.number_of_nodes():,}')
print(f'   - Chunk节点: {len(chunk_nodes):,}')
print(f'   - 实体节点: {len(entity_nodes):,}')
print(f'   总边: {kg.number_of_edges():,}')
print(f'   - Chunk→Entity: {chunk_to_entity_edges:,} ({chunk_to_entity_edges/kg.number_of_edges()*100:.1f}%)')
print(f'   - Entity→Entity: {entity_to_entity_edges:,} ({entity_to_entity_edges/kg.number_of_edges()*100:.1f}%)')

density = kg.number_of_edges() / (kg.number_of_nodes() * (kg.number_of_nodes() - 1)) if kg.number_of_nodes() > 1 else 0
print(f'   图密度: {density:.6f}')

# 对比改进
print(f'\n📈 与之前KG对比:')
print(f'   Entity→Entity边: 6,956 (1.7%) → {entity_to_entity_edges:,} ({entity_to_entity_edges/kg.number_of_edges()*100:.1f}%)')
print(f'   提升: {entity_to_entity_edges/6956:.1f}x')

# 保存KG
output_dir = Path('data/knowledge_graphs')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'hotpotqa_kg_high_quality.gpickle', 'wb') as f:
    pickle.dump(kg, f, protocol=pickle.HIGHEST_PROTOCOL)

# 计算并保存PageRank
print('\n🔄 计算PageRank...')
pagerank_scores = nx.pagerank(kg, alpha=0.85, max_iter=100)

with open(output_dir / 'hotpotqa_pagerank_high_quality.pkl', 'wb') as f:
    pickle.dump(pagerank_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

# 保存统计信息
stats = {
    'num_nodes': kg.number_of_nodes(),
    'num_edges': kg.number_of_edges(),
    'num_chunks': len(chunk_nodes),
    'num_entities': len(entity_nodes),
    'num_entity_relations': entity_to_entity_edges,
    'chunk_entity_edges': chunk_to_entity_edges,
    'graph_density': density,
    'sampling_ratio': 0.7,
    'llm_extraction': True,
    'api_calls': api_calls,
    'total_cost': total_cost,
}

with open(output_dir / 'hotpotqa_kg_high_quality_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f'\n✅ 高质量KG构建完成！')
print(f'   保存位置: {output_dir}/')
print(f'   - hotpotqa_kg_high_quality.gpickle')
print(f'   - hotpotqa_pagerank_high_quality.pkl')
print(f'   - hotpotqa_kg_high_quality_stats.json')
print(f'\n💰 总成本: ${total_cost:.2f}')
print()
print('下一步: 使用高质量KG重新运行HippoRAG实验')
print('运行: python scripts/run_experiment_high_quality_kg.py')
