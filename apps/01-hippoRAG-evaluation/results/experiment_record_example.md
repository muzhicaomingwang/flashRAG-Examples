# Experiment Record (Example)

## Run Metadata
- Date (YYYY-MM-DD): 2026-02-07 <!-- 运行日期，用于对齐时间与配置 -->
- Operator: Example User <!-- 运行/记录者姓名 -->
- Purpose / Hypothesis: Validate HippoRAG improvements with GPT-5.2 <!-- 本次实验目标或假设 -->
- Dataset: hotpotqa <!-- 数据集名称，与配置保持一致 -->
- Split: dev <!-- 数据集划分，如 dev/test/train -->
- Sample size (max_samples): 7405 <!-- 本次评测的样本数量上限 -->
- Random seed: 42 <!-- 控制抽样/随机性的种子 -->

## Code & Environment
- Repo path: /Users/qitmac001395/workspace/QAL/flashRAG-Examples/apps/01-hippoRAG-evaluation <!-- 仓库本地路径 -->
- Git branch: main <!-- 代码分支名 -->
- Git commit: abcdef1 <!-- 代码提交哈希 -->
- Dirty worktree (yes/no): yes <!-- 是否存在未提交改动 -->
- Python version: 3.11.6 <!-- Python 运行时版本 -->
- OS: macOS <!-- 操作系统 -->

## Model & API
- LLM (generation): baseline=gpt-5.2; hipporag=gpt-5.2 <!-- 生成模型配置（baseline 与 hipporag） -->
- Embedding model: text-embedding-3-small <!-- 向量化模型 -->
- Temperature: 0.0 <!-- 生成温度（随机性） -->
- Max tokens: 256 <!-- 单次生成的最大 token 数 -->
- API provider: OpenAI <!-- API 提供方 -->
- Rate limit / concurrency: 10 <!-- 并发或限速设置 -->

## Retrieval & Index
- Baseline top_k: 5 <!-- baseline 检索返回条数 -->
- HippoRAG initial_k: 20 <!-- hipporag 初检索候选数量 -->
- HippoRAG rerank_k: 5 <!-- hipporag 重排序后返回数量 -->
- FAISS index type: IndexFlatL2 <!-- 向量索引类型 -->
- Normalize vectors: true <!-- 是否对向量做 L2 归一化 -->

## Document Processing
- Chunk size: 512 <!-- 文本切块大小 -->
- Chunk overlap: 50 <!-- 相邻块重叠大小 -->
- Separators: ["\n\n", "\n", ". ", " ", ""] <!-- 切分优先级分隔符 -->
- Tokenizer: cl100k_base <!-- 分词器名称 -->

## HippoRAG Graph
- Sampling ratio: 0.3 <!-- 图构建时采样比例 -->
- NER method: spacy <!-- 实体识别方法 -->
- RE method: spacy_dep <!-- 关系抽取方法 -->
- PageRank damping factor: 0.85 <!-- PageRank 阻尼系数 -->
- PageRank max iterations: 100 <!-- PageRank 最大迭代次数 -->

## Runs
- Baseline RAG: results/baseline/predictions.json <!-- baseline 预测结果文件 -->
- HippoRAG: results/hipporag/predictions.json <!-- hipporag 预测结果文件 -->

## Metrics
- Baseline F1: 0.5500 <!-- baseline 的 F1 分数 -->
- Baseline EM: 0.2100 <!-- baseline 的 EM 分数 -->
- HippoRAG F1: 0.7000 <!-- hipporag 的 F1 分数 -->
- HippoRAG EM: 0.2800 <!-- hipporag 的 EM 分数 -->

## Notes / Anomalies
- Example only. Replace with real run info. <!-- 记录异常与注意事项 -->

## Comparison
- Against prior run (commit/date): 2026-02-01 / 1234567 <!-- 对比基准（提交或日期） -->
- Key deltas: +15% F1, similar latency <!-- 关键变化点总结 -->
