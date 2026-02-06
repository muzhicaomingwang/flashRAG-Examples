#!/usr/bin/env python3
"""
HotpotQA 数据集下载和预处理脚本

功能：
1. 从 HuggingFace datasets 下载 HotpotQA
2. 提取 corpus（文档集合）
3. 采样验证集（500个问题）
4. 保存到 data/raw 目录
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from datasets import load_dataset


def load_config() -> Dict:
    """加载配置文件"""
    config_path = project_root / "configs" / "experiment_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_hotpotqa(config: Dict) -> None:
    """下载 HotpotQA 数据集"""
    print("📥 下载 HotpotQA 数据集...")

    # 加载数据集（只加载 validation split）
    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")

    print(f"✅ 下载完成！总计 {len(dataset)} 个样本")

    return dataset


def extract_corpus(dataset) -> List[Dict]:
    """从数据集中提取文档语料库"""
    print("\n📚 提取文档语料库...")

    corpus = {}

    for example in tqdm(dataset, desc="处理样本"):
        # HotpotQA 的 context 包含多个段落
        # context 格式: [title, [sent1, sent2, ...]]
        contexts = example['context']

        for title, sentences in zip(contexts['title'], contexts['sentences']):
            # 使用 title 作为文档 ID
            doc_id = title.replace(" ", "_")

            # 如果文档已存在，跳过
            if doc_id in corpus:
                continue

            # 合并句子成完整文档
            full_text = " ".join(sentences)

            corpus[doc_id] = {
                "id": doc_id,
                "title": title,
                "text": full_text
            }

    corpus_list = list(corpus.values())
    print(f"✅ 提取完成！总计 {len(corpus_list)} 个唯一文档")

    return corpus_list


def sample_validation_set(dataset, config: Dict) -> List[Dict]:
    """采样验证集"""
    print("\n🎲 采样验证集...")

    max_samples = config['dataset']['max_samples']
    random_seed = config['dataset']['random_seed']

    # 设置随机种子
    random.seed(random_seed)

    # 转换为列表并采样
    all_examples = list(dataset)

    if len(all_examples) > max_samples:
        sampled = random.sample(all_examples, max_samples)
    else:
        sampled = all_examples

    # 转换为标准格式
    validation_set = []
    for idx, example in enumerate(sampled):
        validation_set.append({
            "id": f"hotpotqa_dev_{idx}",
            "question": example['question'],
            "answer": example['answer'],
            "type": example['type'],
            "level": example['level'],
            "supporting_facts": example['supporting_facts']
        })

    print(f"✅ 采样完成！验证集大小: {len(validation_set)}")

    return validation_set


def save_data(corpus: List[Dict], validation_set: List[Dict]) -> None:
    """保存数据到文件"""
    print("\n💾 保存数据...")

    # 创建输出目录
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 保存 corpus
    corpus_path = raw_dir / "hotpotqa_corpus.jsonl"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"✅ Corpus 已保存: {corpus_path}")

    # 保存 validation set
    val_path = raw_dir / "hotpotqa_validation.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for example in validation_set:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✅ 验证集已保存: {val_path}")

    # 保存统计信息
    stats = {
        "corpus_size": len(corpus),
        "validation_size": len(validation_set),
        "avg_doc_length": sum(len(doc['text'].split()) for doc in corpus) / len(corpus),
        "avg_question_length": sum(len(q['question'].split()) for q in validation_set) / len(validation_set)
    }

    stats_path = raw_dir / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, indent=2, fp=f)

    print(f"✅ 统计信息已保存: {stats_path}")
    print("\n📊 数据集统计:")
    print(f"  - 文档数量: {stats['corpus_size']:,}")
    print(f"  - 验证集大小: {stats['validation_size']}")
    print(f"  - 平均文档长度: {stats['avg_doc_length']:.1f} 词")
    print(f"  - 平均问题长度: {stats['avg_question_length']:.1f} 词")


def main():
    """主函数"""
    print("=" * 60)
    print("HotpotQA 数据集下载和预处理")
    print("=" * 60)

    # 加载配置
    config = load_config()

    # 下载数据集
    dataset = download_hotpotqa(config)

    # 提取语料库
    corpus = extract_corpus(dataset)

    # 采样验证集
    validation_set = sample_validation_set(dataset, config)

    # 保存数据
    save_data(corpus, validation_set)

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print("=" * 60)
    print("\n📝 下一步:")
    print("   python scripts/02_build_baseline.py")


if __name__ == "__main__":
    main()
