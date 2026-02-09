# 基于 FlashRAG 的 RAG 实验验证（20 分钟逐字稿）

> 面向技术听众的逐字稿。可直接照读或轻微口语化调整。

---

## 0. FlashRAG 系统简介（约 1:30）

在进入实验之前，我先用 1 分钟系统性介绍一下 FlashRAG。

FlashRAG 是一个面向 RAG 研究的模块化工具包，目标是让 **复现与对比实验更轻量、更标准**。

**项目地址（GitHub）：**
```
https://github.com/RUC-NLPIR/FlashRAG
```

**数据集（Hugging Face）：**
```
https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets
```

**相关论文：**
- 论文题目：FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research
- arXiv：2405.13576
- WWW 2025 Resource Track 录用（ACM Web Conference 2025 资源轨）
- DOI：10.1145/3701716.3715313

**演进版本（按时间线）：**
- 2024-09：引入轻量 BM25s 作为 Pyserini 替代。
- 2024-09/10：推出 MindSpore / Paddle 版本以适配特定硬件平台。
- 2025-01：论文被 WWW 2025 Resource Track 录用，并推出 FlashRAG-UI。
- 2025-02：新增多模态 RAG 支持（MLLM + 多模态检索）。
- 2025-03：扩展到 23 个 RAG 方法，包含推理型方法。
- 2025-08：支持 Reasoning Pipeline，并在多跳任务上给出结果。
- 2025-11：新增基于 Web Search Engine 的检索器（可用 Serper API）。

---

## 1. 标题页（约 0:30）

大家好，我今天分享的主题是 **“基于 FlashRAG 的 RAG 实验验证”**，
副标题是：**以 HippoRAG 复现为例**。

这次分享我主要想回答三个问题：
1）为什么要做实验验证；
2）如何按步骤验证一个 RAG 方法；
3）FlashRAG 在这个过程中提供了哪些关键能力。

我们最后会看一个基于 HippoRAG 的实验结果，并总结 FlashRAG 对 RAG 实验的贡献。

---

## 2. 实验验证的重要性（约 1:30）

在 RAG 领域，**“能跑”不代表“可信”**。

原因有三点：
- 第一，RAG 的系统复杂度高，细节很多。比如 chunking 的大小、检索的 top-k、rerank 的策略、生成模型的温度，都能显著影响指标。
- 第二，很多论文和工程实践里，**实现细节并不公开**。即使方法一样，细节不同，结果可能差很多。
- 第三，缺乏统一的验证条件，会导致 **“假提升”**：看似方法提升了，实际上是因为数据、设置或评价标准不一致。

所以实验验证的核心目标是：
**同数据、同指标、同设置**，在可控条件下比较方法差异。

如果没有这个前提，任何“提升”其实都很难被相信。

---

## 3. 实验验证的步骤（以 HippoRAG 为例）（约 2:00）

我们以 HippoRAG 为例，实验验证通常包括以下步骤：

**第一步：明确任务与数据集。**
我们选择 HotpotQA，这是一个典型的多跳 QA 数据集，适合验证 KG 在推理上的作用。

**第二步：确定基线方法。**
最基础的 baseline 是 Dense Retrieval RAG，使用 FAISS IndexFlatL2。
为了对比，我们还可加入 BM25 作为稀疏检索的基线。

**第三步：实现核心创新模块。**
HippoRAG 的核心是 **构建知识图谱 + PPR rerank**。
所以我们复现 KG 构建流程，并确保它与 baseline 共享其他模块。

**第四步：统一评估标准。**
使用相同验证集、相同指标，比如 EM/F1、Recall@k、Latency。

**第五步：记录并对比结果。**
输出预测、指标、日志与 checkpoint，保证实验可追踪。

---

## 4. 实验验证的几个要素：FlashRAG 提供了什么（约 3:00）

FlashRAG 在实验验证中提供了四类关键能力。

### 4.1 标准数据与验证集
FlashRAG 已经整理了多种任务类型的数据集：
- 单跳 QA、多跳 QA、长答案 QA、判断题、多选题、事实核查等。
- 数据以统一格式提供，并区分 train / dev / test。

这能保证“不同方法比较的是同一份数据”。

### 4.2 验证标准
FlashRAG 对不同任务提供标准指标：
- QA：EM / F1
- 多跳 QA：Answer EM/F1 + Supporting Facts EM/F1
- 长答案：ROUGE 或 BERTScore
- 多选题：Accuracy

这样可以避免“指标口径不一致”。

### 4.3 标准实现
FlashRAG 提供了 **可复用的 pipeline**：
- chunking、retrieval、rerank、generation 的标准实现
- 统一的配置文件
- 统一的输出格式

这非常关键，因为它让我们能在“只改一个模块”的情况下进行对比，避免系统层面的干扰。

### 4.4 其他工程支持
- 支持日志、checkpoint、中断恢复
- 多实验对比脚本
- 统一数据格式，易于扩展新模型

总结一下：FlashRAG 解决的是 **“RAG 实验的可复现性和可对比性”**。

---

## 5. 实验过程：如何构建 HippoRAG 的 KG（约 3:00）

下面是我们复现 HippoRAG 的关键流程。

**第一步：文档分块。**
我们对 HotpotQA 的 corpus 进行 chunking，保证检索和 KG 都基于同样的文本片段。

**第二步：实体识别。**
使用 NER（如 SpaCy）识别文本中的实体。

**第三步：关系抽取。**
基于依存关系提取实体间的关系（非 LLM 方式），构建关系边。

**第四步：构建知识图谱。**
我们用 NetworkX 构建图结构，并对图执行 PageRank。

**第五步：PPR rerank。**
在检索阶段先通过 FAISS 获取候选，再用 PPR 重排，使 KG 信息融入排序。

这样，HippoRAG 的创新点就体现在 **“rerank 阶段融合结构化信息”** 上。

---

## 6. 实验结果（约 2:30）

接下来展示实验结果。

> 【此页请填真实结果】

推荐展示方式：
- 一张对比表格：Baseline vs HippoRAG（可加 BM25）
- 指标包含：EM / F1 / Latency / Recall@k

**讲述建议：**
- 先讲 baseline 表现
- 再讲 HippoRAG 相比 baseline 的变化
- 如果差距不大，也要如实说明

例如：
“在同一验证集下，HippoRAG 的 F1 变化为 X，EM 变化为 Y，延迟增加/下降为 Z。”

这部分的重点是：**结论可信，因为实验设置一致**。

---

## 7. 结果解读（约 1:30）

如果 HippoRAG 指标有提升，说明 KG rerank 在多跳任务中有作用。

如果提升有限，可能原因包括：
- KG 质量不足
- rerank 权重设置不合理
- LLM 生成端仍然是瓶颈

这提醒我们：
**结构化信息的加入并不一定带来稳定提升，必须通过实验验证。**

---

## 8. 总结：FlashRAG 对 RAG 实验的贡献（约 1:30）

最后总结 FlashRAG 的贡献：

1）**降低复现成本**：数据集、脚本、流程都已有标准化支持。
2）**提升对比公平性**：同格式、同指标、同设置，避免“假提升”。
3）**帮助定位提升来源**：可以清晰看到改动带来的真实增益。

一句话总结：
**FlashRAG 让 RAG 实验从“能跑”走向“可复现、可对比、可解释”。**

---

## 9. Q&A（约 0:30）

以上是我的分享，感谢大家。
欢迎提问。

---

# 附：建议的结果页模板（可复制到 PPT）

**标题：实验结果对比（HotpotQA Dev）**

| 方法 | EM | F1 | Recall@5 | Latency | 备注 |
| --- | --- | --- | --- | --- | --- |
| BM25 |  |  |  |  | 稀疏检索基线 |
| Baseline RAG (FAISS) |  |  |  |  | Dense 检索 |
| HippoRAG (KG + PPR) |  |  |  |  | Rerank 加入 KG |

**讲述句式建议：**
- “在同一数据集和评估设置下，HippoRAG 相比 Baseline 在 F1 上提升/下降了 X。”
- “延迟方面的变化为 Y，说明 KG rerank 的代价/收益是 Z。”
