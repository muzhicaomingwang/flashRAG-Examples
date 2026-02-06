#!/usr/bin/env python3
"""
HippoRAG 知识图谱构建脚本

功能：
1. 加载分块后的文档
2. 使用 SpaCy 进行实体识别（NER）
3. 使用依存句法分析提取关系（RE）
4. 构建知识图谱（NetworkX）
5. 计算 Personalized PageRank
"""

import os
import sys
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import spacy
import networkx as nx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def load_config() -> Dict:
    """加载配置文件"""
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_chunks() -> List[Dict]:
    """加载分块后的文档"""
    print("📚 加载文档块...")
    chunks_path = project_root / "data" / "processed" / "hotpotqa_chunks.jsonl"

    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"✅ 加载完成！总计 {len(chunks)} 个块")
    return chunks


def sample_chunks(chunks: List[Dict], sampling_ratio: float, random_seed: int = 42) -> List[Dict]:
    """采样文档块（节省成本）"""
    print(f"\n🎲 采样 {sampling_ratio*100:.0f}% 的文档块...")

    random.seed(random_seed)
    n_samples = int(len(chunks) * sampling_ratio)

    sampled_chunks = random.sample(chunks, n_samples)

    print(f"✅ 采样完成！{len(chunks)} → {len(sampled_chunks)} 个块")

    return sampled_chunks


def extract_entities_spacy(text: str, nlp) -> List[Tuple[str, str]]:
    """使用 SpaCy 提取实体"""
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        # 归一化实体名称
        normalized_name = ent.text.strip().lower()
        entities.append((normalized_name, ent.label_))

    return entities


def extract_relations_spacy(text: str, nlp) -> List[Tuple[str, str, str]]:
    """使用 SpaCy 依存句法分析提取关系"""
    doc = nlp(text)

    relations = []

    for token in doc:
        # 提取主谓宾关系
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subject = token.text.lower()
            verb = token.head.lemma_
            # 查找宾语
            for child in token.head.children:
                if child.dep_ in ["dobj", "attr", "prep"]:
                    obj = child.text.lower()
                    relations.append((subject, verb, obj))

        # 提取修饰关系
        elif token.dep_ in ["amod", "compound"]:
            relations.append((token.head.text.lower(), "is_a", token.text.lower()))

    return relations


def build_knowledge_graph(chunks: List[Dict], config: Dict) -> nx.DiGraph:
    """构建知识图谱"""
    print("\n🔨 构建知识图谱...")

    # 加载 SpaCy 模型
    nlp = spacy.load(config['hipporag']['entity_extraction']['spacy_model'])

    # 初始化图
    kg = nx.DiGraph()

    # 实体统计
    entity_counts = defaultdict(int)
    relation_counts = defaultdict(int)

    # 处理每个文档块
    for chunk in tqdm(chunks, desc="提取实体和关系"):
        chunk_id = chunk['chunk_id']
        text = chunk['text']

        # 添加文档块节点
        kg.add_node(chunk_id, type="chunk", text=text, title=chunk['title'])

        # 提取实体
        entities = extract_entities_spacy(text, nlp)

        # 去重
        unique_entities = list(set(entities))

        # 限制每个块的最大实体数
        max_entities = config['hipporag']['entity_extraction']['max_entities_per_chunk']
        if len(unique_entities) > max_entities:
            unique_entities = unique_entities[:max_entities]

        # 添加实体节点和chunk-entity边
        for entity_name, entity_type in unique_entities:
            entity_id = f"entity_{entity_name.replace(' ', '_')}"

            # 添加实体节点（如果不存在）
            if not kg.has_node(entity_id):
                kg.add_node(entity_id, type="entity", name=entity_name, entity_type=entity_type)

            # 添加 chunk -> entity 边
            kg.add_edge(chunk_id, entity_id, relation="contains")

            # 统计
            entity_counts[entity_name] += 1

        # 提取关系
        relations = extract_relations_spacy(text, nlp)

        # 限制每个块的最大关系数
        max_relations = config['hipporag']['relation_extraction']['max_relations_per_chunk']
        if len(relations) > max_relations:
            relations = relations[:max_relations]

        # 添加实体间关系
        for subj, rel, obj in relations:
            subj_id = f"entity_{subj.replace(' ', '_')}"
            obj_id = f"entity_{obj.replace(' ', '_')}"

            # 只添加图中已存在实体的关系
            if kg.has_node(subj_id) and kg.has_node(obj_id):
                kg.add_edge(subj_id, obj_id, relation=rel)
                relation_counts[rel] += 1

    print(f"\n✅ 知识图谱构建完成！")
    print(f"   - 节点总数: {kg.number_of_nodes():,}")
    print(f"   - 边总数: {kg.number_of_edges():,}")
    print(f"   - 唯一实体数: {len([n for n in kg.nodes if kg.nodes[n]['type'] == 'entity']):,}")
    print(f"   - 唯一关系类型: {len(relation_counts):,}")

    # 显示最常见的实体和关系
    print(f"\n📊 Top-10 最常见实体:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {entity}: {count}")

    print(f"\n📊 Top-10 最常见关系:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   - {rel}: {count}")

    return kg


def compute_pagerank(kg: nx.DiGraph, config: Dict) -> Dict[str, float]:
    """计算 Personalized PageRank"""
    print("\n🔨 计算 Personalized PageRank...")

    pr_config = config['hipporag']['pagerank']

    # 计算 PageRank（无个性化，先计算全局重要性）
    pagerank_scores = nx.pagerank(
        kg,
        alpha=pr_config['damping_factor'],
        max_iter=pr_config['max_iterations'],
        tol=pr_config['convergence_threshold']
    )

    print(f"✅ PageRank 计算完成！")

    # 显示最重要的节点
    print(f"\n📊 Top-10 最重要的实体:")
    entity_scores = {n: s for n, s in pagerank_scores.items() if kg.nodes[n]['type'] == 'entity'}
    for node_id, score in sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
        entity_name = kg.nodes[node_id]['name']
        print(f"   - {entity_name}: {score:.6f}")

    return pagerank_scores


def save_knowledge_graph(kg: nx.DiGraph, pagerank_scores: Dict, config: Dict) -> None:
    """保存知识图谱"""
    print("\n💾 保存知识图谱...")

    # 创建输出目录
    kg_dir = project_root / "data" / "knowledge_graphs"
    kg_dir.mkdir(parents=True, exist_ok=True)

    # 保存图谱
    kg_path = kg_dir / "hotpotqa_kg.gpickle"
    nx.write_gpickle(kg, str(kg_path))
    print(f"✅ 知识图谱已保存: {kg_path}")

    # 保存 PageRank 分数
    pr_path = kg_dir / "hotpotqa_pagerank.pkl"
    with open(pr_path, 'wb') as f:
        pickle.dump(pagerank_scores, f)
    print(f"✅ PageRank 分数已保存: {pr_path}")

    # 保存图谱统计信息
    stats = {
        "num_nodes": kg.number_of_nodes(),
        "num_edges": kg.number_of_edges(),
        "num_chunks": len([n for n in kg.nodes if kg.nodes[n]['type'] == 'chunk']),
        "num_entities": len([n for n in kg.nodes if kg.nodes[n]['type'] == 'entity']),
        "avg_degree": sum(dict(kg.degree()).values()) / kg.number_of_nodes(),
        "density": nx.density(kg)
    }

    stats_path = kg_dir / "hotpotqa_kg_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, indent=2, fp=f)
    print(f"✅ 统计信息已保存: {stats_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("HippoRAG 知识图谱构建")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 加载文档块
    chunks = load_chunks()

    # 采样文档块（节省成本）
    sampling_ratio = config['hipporag']['sampling_ratio']
    sampled_chunks = sample_chunks(chunks, sampling_ratio)

    # 构建知识图谱
    kg = build_knowledge_graph(sampled_chunks, config)

    # 计算 PageRank
    pagerank_scores = compute_pagerank(kg, config)

    # 保存知识图谱
    save_knowledge_graph(kg, pagerank_scores, config)

    print("\n" + "=" * 60)
    print("✅ HippoRAG 知识图谱构建完成！")
    print("=" * 60)
    print("\n📝 下一步:")
    print("   python scripts/05_run_experiments.py")


if __name__ == "__main__":
    main()
