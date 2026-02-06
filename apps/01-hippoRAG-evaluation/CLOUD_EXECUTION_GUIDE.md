# ☁️ 云端执行指南（推荐方案）

由于本地环境内存限制，建议在云端执行完整实验。本指南提供3种云端执行方案。

---

## 🆓 方案 1：Google Colab（推荐，免费）

### 优点
- ✅ 完全免费
- ✅ 12GB RAM（足够运行完整实验）
- ✅ 可选GPU加速
- ✅ 无需配置服务器

### 执行步骤

#### Step 1: 创建 Colab Notebook

我已经将所有代码模块化，你可以创建一个新的 Colab notebook：

```python
# Cell 1: 安装依赖
!pip install datasets openai tiktoken faiss-cpu rank-bm25 spacy networkx scikit-learn pyyaml python-dotenv -q
!python -m spacy download en_core_web_sm

# Cell 2: 上传配置文件
from google.colab import files
import os

# 设置 API Key
os.environ['OPENAI_API_KEY'] = 'your_api_key_here'

# Cell 3-8: 复制各个脚本的代码
# （将 scripts/ 中的 Python 代码粘贴到各个 cell）

# Cell 9: 运行完整实验
# 执行所有步骤
```

#### Step 2: 上传项目文件

**方式 A：直接上传**
```python
# 在 Colab 中运行
from google.colab import files
uploaded = files.upload()  # 上传整个项目 ZIP
```

**方式 B：从 GitHub**
```bash
# 先将项目推送到 GitHub
git add . && git commit -m "HippoRAG实验代码"
git push

# 在 Colab 中 clone
!git clone https://github.com/your-repo/flashRAG-Examples.git
%cd flashRAG-Examples/apps/01-hippoRAG-evaluation
```

#### Step 3: 配置并运行

```python
# 在 Colab 中执行
!./setup.sh
!./run_day1.sh
!./run_day2.sh
```

#### Step 4: 下载结果

```python
# 下载结果到本地
from google.colab import files
files.download('results/evaluation_metrics.json')
files.download('results/comparison_table.md')
```

### 预计时间和成本

| 项目 | 时间 | Colab成本 | OpenAI成本 |
|------|------|-----------|-----------|
| Day 1 | 40分钟 | $0 | $0.22 |
| Day 2 | 30分钟 | $0 | $4.35 |
| **总计** | **~1.2小时** | **$0** | **$4.57** |

---

## ☁️ 方案 2：AWS EC2

### 推荐实例类型

| 实例类型 | RAM | vCPU | 价格/小时 | 总成本 |
|---------|-----|------|----------|--------|
| t3.medium | 4GB | 2 | $0.0416 | $0.05 |
| t3.large | 8GB | 2 | $0.0832 | $0.10 |
| t3.xlarge | 16GB | 4 | $0.1664 | $0.20 |

**推荐：** t3.medium（4GB足够，只需$0.05）

### 快速启动命令

```bash
# 1. 启动 EC2 实例（使用 AWS CLI）
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key

# 2. SSH 登录
ssh -i your-key.pem ec2-user@<instance-ip>

# 3. 安装Python 3.9
sudo yum install python39 -y

# 4. 上传项目
scp -i your-key.pem -r apps/01-hippoRAG-evaluation ec2-user@<instance-ip>:~/

# 5. 运行实验
cd 01-hippoRAG-evaluation
./setup.sh
./run_day1.sh
./run_day2.sh

# 6. 下载结果
scp -i your-key.pem -r ec2-user@<instance-ip>:~/01-hippoRAG-evaluation/results ./

# 7. 终止实例（节省成本）
aws ec2 terminate-instances --instance-ids <instance-id>
```

### 总成本

| 项目 | 成本 |
|------|------|
| EC2运行时间（~2小时） | $0.08 |
| OpenAI API | $4.57 |
| **总计** | **$4.65** |

---

## 💻 方案 3：本地其他机器

如果你有其他内存充足的机器（>=8GB RAM）：

### Mac/Linux

```bash
# 1. 复制项目
cp -r apps/01-hippoRAG-evaluation /path/to/target/

# 2. 运行
cd /path/to/target/01-hippoRAG-evaluation
./setup.sh
./run_day1.sh
./run_day2.sh
```

### Windows（WSL）

```bash
# 1. 在 WSL 中安装 Python 3.9
sudo apt update
sudo apt install python3.9 python3.9-venv -y

# 2. 运行项目
cd /mnt/c/path/to/01-hippoRAG-evaluation
bash setup.sh
bash run_day1.sh
bash run_day2.sh
```

---

## 🔧 内存优化建议

如果必须在本地运行，可以进一步优化：

### 1. 减少文档数量

在 `scripts/02_build_baseline.py` 中：

```python
# 修改 load_corpus() 函数
def load_corpus() -> List[Dict]:
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # 只加载1K文档
                break
            corpus.append(json.loads(line))
    return corpus
```

### 2. 减少验证集大小

在 `configs/experiment_config.yaml` 中：

```yaml
dataset:
  max_samples: 50  # 从500降到50
```

### 3. 分批处理向量化

在 `scripts/02_build_baseline.py` 的 `build_faiss_index()` 中：

```python
# 修改 batch_size
batch_size = 10  # 从100降到10
```

---

## 🎯 我的建议

**最佳方案：Google Colab**

理由：
1. ✅ 完全免费
2. ✅ 内存充足（12GB）
3. ✅ 无需服务器配置
4. ✅ 可以复用所有代码

**执行流程：**
1. 我帮你创建 Colab notebook
2. 你上传到 Google Drive
3. 运行 notebook（约1小时）
4. 下载结果到本地

**要我现在创建 Colab notebook 吗？**

---

## 📦 项目文件清单

所有代码已保存在：
```
/Users/qitmac001395/workspace/QAL/flashRAG-Examples/apps/01-hippoRAG-evaluation/
```

**可以直接打包发送到云端：**

```bash
# 创建ZIP包
cd /Users/qitmac001395/workspace/QAL/flashRAG-Examples
tar -czf hipporag-experiment.tar.gz apps/01-hippoRAG-evaluation/

# 查看大小
ls -lh hipporag-experiment.tar.gz
```

**预计包大小：** ~15MB（不包含data/目录）

---

## 🚀 快速启动命令（云端）

无论你选择哪个云端平台，核心命令相同：

```bash
# 1. 进入项目目录
cd 01-hippoRAG-evaluation

# 2. 查看README
cat QUICKSTART.md

# 3. 一键运行
./run_day1.sh && ./run_day2.sh

# 4. 查看结果
cat results/comparison_table.md
```

**就这么简单！** 🎉

---

需要我创建 Colab notebook 或其他云端执行方案吗？
