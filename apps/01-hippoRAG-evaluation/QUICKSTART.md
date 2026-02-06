# 🚀 HippoRAG 实验快速开始指南

## 📋 前置要求

- Python 3.9+
- OpenAI API Key
- 至少 8GB RAM
- 至少 5GB 磁盘空间

## ⚡ 快速开始（2天方案）

### Day 1: 环境准备 + Baseline RAG（预计 4 小时）

#### Step 1: 安装依赖

```bash
# 运行安装脚本
./setup.sh

# 或手动安装
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Step 2: 配置 API Key

```bash
# 已经自动创建了 .env 文件
# 确认你的 API key 已经配置
cat .env
```

#### Step 3: 下载数据集

```bash
# 下载 HotpotQA 数据集（约 10-20 分钟）
python scripts/01_download_data.py
```

**预期输出：**
- `data/raw/hotpotqa_corpus.jsonl` - 文档语料库
- `data/raw/hotpotqa_validation.jsonl` - 500 个验证集问题
- `data/raw/dataset_stats.json` - 数据集统计

#### Step 4: 构建 Baseline RAG

```bash
# 构建 FAISS + BM25 索引（约 30-40 分钟）
python scripts/02_build_baseline.py
```

**这一步会：**
1. 将文档分块（~110K chunks）
2. 调用 OpenAI Embedding API 向量化
3. 构建 FAISS 索引
4. 构建 BM25 索引

**预期成本：** ~$0.56（使用 text-embedding-3-small）

#### Step 5: 测试 Baseline

```bash
# 测试检索功能（约 1 分钟）
python scripts/03_test_baseline.py
```

**预期输出：** 会运行 3 个示例问题，展示 FAISS 和 BM25 的检索结果

---

### Day 2: HippoRAG + 实验（预计 8 小时）

#### Step 6: 构建 HippoRAG 知识图谱

```bash
# 构建知识图谱（约 20-30 分钟）
python scripts/04_build_hipporag.py
```

**这一步会：**
1. 使用 SpaCy 进行实体识别
2. 使用依存句法分析提取关系
3. 构建 NetworkX 图
4. 计算 Personalized PageRank

**只处理 30% 的文档块（采样策略）**

#### Step 7: 运行完整实验

```bash
# 运行 3 组对比实验（约 10-15 分钟）
python scripts/05_run_experiments.py
```

**实验组：**
1. **Baseline** - Dense Retrieval (FAISS)
2. **HippoRAG-Simple** - KG + SpaCy RE + PPR
3. **HippoRAG-Full** - 完整方法

**预期成本：** ~$4.35（500 问题 × 3 组）

#### Step 8: 生成评估报告

```bash
# 计算指标并生成报告（约 2 分钟）
python scripts/06_generate_report.py
```

**输出：**
- `results/evaluation_metrics.json` - 详细指标
- `results/comparison_table.md` - 对比表格
- `results/plots/` - 可视化图表

---

## 🎯 一键运行脚本

如果你想自动运行整个 Day 1：

```bash
# Day 1 一键运行
./run_day1.sh
```

如果你想自动运行整个 Day 2：

```bash
# Day 2 一键运行
./run_day2.sh
```

---

## 📊 预期成本和时间

| 阶段 | 机器时间 | 人工时间 | API 成本 |
|------|---------|---------|---------|
| Day 1 | ~40 分钟 | ~1 小时（等待 + 验证） | ~$0.56 |
| Day 2 | ~30 分钟 | ~2 小时（分析结果） | ~$12.00 |
| **总计** | ~1.2 小时 | ~3 小时 | **~$12.56** |

---

## 🛠️ 故障排除

### 问题 1: OpenAI API 速率限制

```
错误: RateLimitError: Rate limit exceeded
```

**解决方案：** 在 `experiment_config.yaml` 中降低 `max_concurrent_requests`

### 问题 2: 内存不足

```
错误: MemoryError
```

**解决方案：**
- 减少 `max_samples`（从 500 降到 200）
- 或增加系统内存

### 问题 3: FAISS 索引加载失败

```
错误: Index file not found
```

**解决方案：** 确保已运行 `02_build_baseline.py`

---

## 📝 下一步

完成 2 天实验后：

1. **分析结果** - 查看 `results/` 目录的评估报告
2. **调整参数** - 修改 `configs/experiment_config.yaml`
3. **扩展数据集** - 添加更多数据集（TriviaQA, 2WikiMultihopQA 等）
4. **深入消融** - 运行完整的消融实验

---

## 💡 提示

- 建议在运行实验前先运行测试脚本验证环境
- 可以使用 `tqdm` 进度条监控长时间运行的任务
- 所有中间结果都会保存，可以随时重新运行评估

---

## 🆘 获取帮助

如果遇到问题：

1. 查看日志输出
2. 检查 `data/` 目录的中间文件
3. 查阅 `EXPERIMENT_DESIGN.md` 了解实验设计
4. 查阅 `FAISS_GUIDE.md` 了解 FAISS 配置
