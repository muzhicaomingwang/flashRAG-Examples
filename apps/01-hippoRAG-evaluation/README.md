# HippoRAG 效果验证实验

## 1. 实验背景

HippoRAG 是一种受神经生物学启发的检索增强生成(RAG)框架，发表于 NeurIPS 2024。它模仿人类海马体索引理论，通过协同使用大语言模型(LLM)、知识图谱和 Personalized PageRank 算法来实现跨文档的深度知识整合。

传统 RAG 方法的局限性：
- 每个文档片段独立编码，无法跨文档边界整合知识
- 在需要多步推理的任务上表现不佳
- 依赖迭代检索时成本高、速度慢

HippoRAG 的核心创新：
- 模拟新皮层和海马体在人类记忆中的不同角色
- 使用知识图谱构建文档间的关联关系
- 通过 Personalized PageRank 实现高效的相关性排序

## 2. 实验目标

本实验旨在使用 flashRAG 框架验证 HippoRAG 在多跳问答任务上的性能提升，并与标准 RAG 方法进行对比。

**核心研究问题：**
1. HippoRAG 在多跳推理任务上是否优于标准 RAG？
2. HippoRAG 的检索效率（速度和成本）优势有多大？
3. 知识图谱的引入对检索质量有何影响？

## 3. 实验设计

### 3.1 对比方法

| 方法名称 | 描述 | 检索策略 |
|---------|------|---------|
| **Baseline: Naive RAG** | flashRAG 标准 RAG 实现 | Dense retrieval (DPR/BM25) |
| **HippoRAG** | 基于知识图谱 + PPR 的检索 | KG-based + Personalized PageRank |
| **Iterative RAG** | 迭代检索方法（如 IRCoT） | 多轮检索 + 推理 |

### 3.2 评估数据集

使用 flashRAG 提供的预处理数据集：

| 数据集 | 任务类型 | 样本数 | 复杂度 |
|-------|---------|-------|--------|
| **HotpotQA** | 多跳问答 | 7,405 (dev) | 高 |
| **2WikiMultihopQA** | 多跳推理 | 5,000+ | 高 |
| **MuSiQue** | 多跳推理 | 2,417 (dev) | 高 |
| **NQ (Natural Questions)** | 单跳问答 | 3,610 (dev) | 中 |

**优先级：** HotpotQA > 2WikiMultihopQA > MuSiQue > NQ

### 3.3 评估指标

| 指标类别 | 指标名称 | 说明 |
|---------|---------|------|
| **准确性** | F1 Score | 答案文本重叠度 |
|  | Exact Match (EM) | 精确匹配率 |
| **检索质量** | Recall@K | Top-K 检索召回率 |
|  | Precision@K | Top-K 检索精确率 |
| **效率** | Latency | 平均响应时间 (秒) |
|  | Cost | API 调用成本 ($) |
|  | Speedup | 相对于 Iterative RAG 的加速比 |

### 3.4 实验参数

```yaml
# 通用参数
llm_model: "gpt-3.5-turbo"
embedding_model: "text-embedding-ada-002"
top_k: 5
max_tokens: 512
temperature: 0.0

# HippoRAG 特定参数
kg_construction:
  relation_extraction: "llm-based"  # 使用 LLM 提取实体关系
  entity_linking: true              # 实体链接
ppr_parameters:
  damping_factor: 0.85
  max_iterations: 100
  convergence_threshold: 1e-6
```

## 4. 实验步骤

### Phase 1: 环境准备
1. 安装 flashRAG 框架
   ```bash
   pip install flashrag
   ```

2. 下载预处理数据集
   ```bash
   python scripts/download_datasets.py --datasets hotpotqa 2wikimultihop musique nq
   ```

3. 安装 HippoRAG 依赖
   ```bash
   pip install -r requirements.txt
   ```

### Phase 2: 数据准备
1. 加载并预处理数据集
2. 构建文档索引（BM25 + Dense Embedding）
3. 为 HippoRAG 构建知识图谱
   - 实体识别与链接
   - 关系抽取
   - 图谱构建与存储

### Phase 3: 基线实验
1. 运行 Naive RAG (BM25)
   ```bash
   python scripts/run_baseline.py --method bm25 --dataset hotpotqa
   ```

2. 运行 Naive RAG (Dense Retrieval)
   ```bash
   python scripts/run_baseline.py --method dense --dataset hotpotqa
   ```

3. 运行 Iterative RAG (IRCoT)
   ```bash
   python scripts/run_baseline.py --method ircot --dataset hotpotqa
   ```

### Phase 4: HippoRAG 实验
1. 运行 HippoRAG
   ```bash
   python scripts/run_hipporag.py --dataset hotpotqa --top_k 5
   ```

2. 进行消融实验
   - 不使用知识图谱（仅 PPR）
   - 不使用 PPR（仅知识图谱）
   - 完整 HippoRAG

### Phase 5: 结果分析
1. 收集所有方法的评估结果
2. 生成对比图表
3. 进行统计显著性测试
4. 分析典型案例

## 5. 预期结果

基于论文报告，预期观察到：

| 指标 | Naive RAG | Iterative RAG | HippoRAG | 提升幅度 |
|-----|----------|--------------|----------|---------|
| **F1 (HotpotQA)** | ~55% | ~60% | ~70%+ | +20% |
| **EM (HotpotQA)** | ~45% | ~50% | ~60%+ | +15% |
| **Latency** | 2s | 15s | 2.5s | 6-13x faster |
| **Cost** | $0.01/query | $0.30/query | $0.01/query | 10-30x cheaper |

## 6. 目录结构

```
01-hippoRAG-evaluation/
├── README.md                    # 本文档：实验设计说明
├── requirements.txt             # Python 依赖
├── configs/                     # 实验配置文件
│   ├── baseline_config.yaml    # 基线方法配置
│   ├── hipporag_config.yaml    # HippoRAG 配置
│   └── datasets_config.yaml    # 数据集配置
├── scripts/                     # 实验脚本
│   ├── download_datasets.py    # 下载数据集
│   ├── build_kg.py             # 构建知识图谱
│   ├── run_baseline.py         # 运行基线实验
│   ├── run_hipporag.py         # 运行 HippoRAG
│   ├── evaluate.py             # 评估脚本
│   └── visualize_results.py    # 结果可视化
├── data/                        # 数据目录
│   ├── datasets/               # 原始数据集
│   ├── indices/                # 检索索引
│   └── knowledge_graphs/       # 知识图谱
└── results/                     # 实验结果
    ├── baseline/               # 基线结果
    ├── hipporag/               # HippoRAG 结果
    ├── analysis/               # 分析报告
    └── visualizations/         # 可视化图表
```

## 7. 实验执行计划

### 阶段一：准备工作（预计 1-2 天）
- [ ] 配置 flashRAG 环境
- [ ] 下载并预处理 HotpotQA 数据集
- [ ] 验证数据集格式和完整性

### 阶段二：基线实验（预计 2-3 天）
- [ ] 运行 BM25 baseline
- [ ] 运行 Dense Retrieval baseline
- [ ] 运行 Iterative RAG (IRCoT)
- [ ] 收集基线性能数据

### 阶段三：HippoRAG 实验（预计 3-4 天）
- [ ] 构建知识图谱（实体识别 + 关系抽取）
- [ ] 实现 Personalized PageRank 模块
- [ ] 运行完整 HippoRAG 实验
- [ ] 运行消融实验

### 阶段四：结果分析（预计 1-2 天）
- [ ] 生成对比表格
- [ ] 绘制性能曲线图
- [ ] 分析典型案例
- [ ] 撰写实验报告

## 8. 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据集
python scripts/download_datasets.py --datasets hotpotqa

# 3. 运行基线实验
python scripts/run_baseline.py --method bm25 --dataset hotpotqa

# 4. 构建知识图谱
python scripts/build_kg.py --dataset hotpotqa

# 5. 运行 HippoRAG
python scripts/run_hipporag.py --dataset hotpotqa

# 6. 生成结果报告
python scripts/evaluate.py --compare baseline hipporag
```

## 9. 关键技术挑战

1. **知识图谱构建质量**
   - 实体识别准确率
   - 关系抽取质量
   - 图谱完整性

2. **PageRank 参数调优**
   - Damping factor 选择
   - 收敛阈值设置
   - 计算效率优化

3. **评估公平性**
   - 确保检索文档数量一致
   - 控制 LLM API 调用次数
   - 统一评估指标计算方式

## 10. 参考资料

### HippoRAG 官方资源
- **arXiv 摘要页：** https://arxiv.org/abs/2405.14831
- **NeurIPS 2024 官方会议版 PDF：** https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf
- **代码仓库：** https://github.com/OSU-NLP-Group/HippoRAG
- **OpenReview 讨论页：** https://openreview.net/forum?id=hkujvAPVsg

### flashRAG 框架资源
- **flashRAG GitHub：** https://github.com/RUC-NLPIR/FlashRAG
- **flashRAG 数据集 (HuggingFace)：** https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
- **flashRAG 论文：** https://arxiv.org/abs/2405.13576

## 11. 实验日志

| 日期 | 阶段 | 完成内容 | 备注 |
|-----|------|---------|------|
| 2026-02-03 | 准备 | 创建实验目录，完成实验设计 | 初始化 |

---

**实验负责人：** 王植萌
**创建时间：** 2026-02-03
**框架版本：** flashRAG v1.0, HippoRAG (NeurIPS 2024)
