# HNSW vs FlatL2 对比报告 (HotpotQA dev, 500 questions)
## 结果总览
| 方法 | F1 | EM | 平均延迟(s) | 中位延迟(s) | 成功率 |
|---|---:|---:|---:|---:|---:|
| Baseline + FlatL2 | 0.2901 | 0.160 | 2.587 | 2.144 | 100.0% |
| HippoRAG + FlatL2 | 0.2898 | 0.164 | 3.035 | 2.378 | 100.0% |
| Baseline + HNSW | 0.2553 | 0.126 | 1.242 | 1.097 | 100.0% |
| HippoRAG + HNSW | 0.2519 | 0.126 | 1.268 | 1.144 | 100.0% |

## 结论摘要
- Baseline：HNSW 相比 FlatL2，F1 下降 -0.0348，EM 下降 -0.034，平均延迟降低 -1.346s。- HippoRAG：HNSW 相比 FlatL2，F1 下降 -0.0379，EM 下降 -0.038，平均延迟降低 -1.767s。- FlatL2：HippoRAG 相比 Baseline，F1 变化 -0.0003，EM 变化 0.004，平均延迟增加 0.448s。- HNSW：HippoRAG 相比 Baseline，F1 变化 -0.0033，EM 变化 0.000，平均延迟增加 0.027s。
## 锁定条件 vs 不同点
| 项目 | 锁定条件 | 不同点 |
|---|---|---|
| 数据集 | HotpotQA dev, 500 questions | 无 |
| 文档语料 | hotpotqa_corpus_sampled.jsonl (10,043 chunks) | 无 |
| Chunking | 512 tokens, overlap 50, separators一致 | 无 |
| Embedding | text-embedding-3-small | 无 |
| Normalization | 向量 L2 归一化 | 无 |
| Retriever top-k | Baseline top_k=5, HippoRAG initial_k=20, rerank_k=5 | 无 |
| LLM | gpt-3.5-turbo, temperature=0, max_tokens=256 | 无 |
| HippoRAG KG | networkx KG + PageRank 参数固定 | 无 |
| Faiss 索引 | IndexFlatL2 vs HNSW (M=32, efC=200, efS=64) | 仅索引类型与参数不同 |

## 数据集与指标说明
- 数据集：HotpotQA 以多跳问答为主，适合检验检索与知识图谱重排的收益。这里使用 dev 集 500 条样本，保证可重复且成本可控。
- 指标：F1 和 EM 衡量答案匹配质量；平均/中位延迟衡量响应成本；成功率衡量系统稳定性。多跳任务下 F1 比 EM 更能反映部分正确的回答质量，因此二者同时给出。
