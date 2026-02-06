# 🔧 故障排除指南

## ❌ 常见问题及解决方案

### 问题 1: 进程被强制终止（Exit Code 137）

**症状：**
```
/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

**原因：** 系统内存不足，操作系统强制杀死进程

**解决方案：**

#### A. 减少文档数量（快速修复）

编辑 `scripts/02_build_baseline.py`：

```python
def load_corpus() -> List[Dict]:
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # 只加载1K文档（原本66K）
                break
            corpus.append(json.loads(line))
    return corpus
```

#### B. 分批处理向量化

```python
def build_faiss_index_streaming(chunks, config):
    """流式处理，避免一次性加载所有向量"""
    all_embeddings = []

    # 每次只处理1000个chunks
    for start in range(0, len(chunks), 1000):
        end = min(start + 1000, len(chunks))
        batch_chunks = chunks[start:end]

        # 向量化
        embeddings = vectorize_batch(batch_chunks)
        all_embeddings.extend(embeddings)

        # 释放内存
        del embeddings
        import gc
        gc.collect()
```

#### C. 切换到云端（推荐）

参考 `CLOUD_EXECUTION_GUIDE.md`

---

### 问题 2: OpenAI API 速率限制

**症状：**
```
RateLimitError: Rate limit exceeded for requests
```

**原因：** 并发请求过多，超过API配额

**解决方案：**

编辑 `configs/experiment_config.yaml`：

```yaml
api:
  max_concurrent_requests: 5  # 从10降到5
  rate_limit_buffer: 0.5      # 增加请求间隔
```

或使用更高级别的API key（付费账户）

---

### 问题 3: FAISS 索引加载失败

**症状：**
```
FileNotFoundError: data/indices/faiss/hotpotqa.index
```

**原因：** 索引文件未生成或路径错误

**解决方案：**

```bash
# 检查文件是否存在
ls -lh data/indices/faiss/

# 重新构建索引
python scripts/02_build_baseline.py

# 确认生成
ls -lh data/indices/faiss/hotpotqa.index
```

---

### 问题 4: HuggingFace 数据集下载失败

**症状：**
```
ConnectionError: Couldn't reach https://huggingface.co/datasets
```

**原因：** 网络连接问题或需要认证

**解决方案：**

#### A. 使用镜像站（国内用户）

```python
# 在脚本开头添加
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

#### B. 手动下载

```bash
# 从HuggingFace手动下载
wget https://huggingface.co/datasets/hotpot_qa/resolve/main/data.zip

# 解压并放到正确位置
unzip data.zip -d data/raw/
```

#### C. 使用认证token

```bash
# 登录 HuggingFace
huggingface-cli login

# 或设置环境变量
export HF_TOKEN=your_token_here
```

---

### 问题 5: SpaCy 模型加载失败

**症状：**
```
OSError: Can't find model 'en_core_web_sm'
```

**原因：** 模型未下载或路径问题

**解决方案：**

```bash
# 重新下载模型
python -m spacy download en_core_web_sm

# 验证安装
python -c "import spacy; spacy.load('en_core_web_sm')"

# 如果还是失败，手动下载
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

---

### 问题 6: 向量化速度太慢

**症状：** 向量化超过1小时仍未完成

**原因：**
- 网络延迟高
- 文档数量太多
- batch_size太小

**解决方案：**

编辑 `configs/experiment_config.yaml`：

```yaml
embedding:
  batch_size: 200  # 从100增加到200
  max_retries: 5
  timeout: 60  # 增加超时时间
```

或使用更快的embedding模型：

```yaml
embedding:
  model: "text-embedding-3-small"  # 更快更便宜
```

---

### 问题 7: 知识图谱构建失败

**症状：**
```
KeyError: 'chunk_id'
NetworkXError: Node not found
```

**原因：** chunk ID 映射错误或图结构问题

**解决方案：**

检查中间文件：

```bash
# 检查chunks文件
head -5 data/processed/hotpotqa_chunks.jsonl

# 检查chunk ID格式
python -c "
import json
with open('data/processed/hotpotqa_chunks.jsonl') as f:
    chunk = json.loads(f.readline())
    print(chunk.keys())
    print(chunk['chunk_id'])
"
```

---

### 问题 8: 实验运行时 LLM 超时

**症状：**
```
TimeoutError: Request timed out
```

**原因：** OpenAI API 响应慢或网络问题

**解决方案：**

编辑 `scripts/05_run_experiments.py`：

```python
# 增加超时时间
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    timeout=60  # 增加到60秒
)
```

或降低并发数：

```yaml
api:
  max_concurrent_requests: 3  # 从10降到3
```

---

### 问题 9: 评估指标计算错误

**症状：**
```
ValueError: empty sequence
ZeroDivisionError: division by zero
```

**原因：** 某些实验组没有成功的结果

**解决方案：**

检查实验结果：

```bash
# 查看成功率
python -c "
import json
with open('results/baseline/predictions.json') as f:
    results = json.load(f)
    success = sum(1 for r in results if r['success'])
    print(f'成功率: {success}/{len(results)}')

    # 查看失败原因
    for r in results:
        if not r['success']:
            print(f'失败: {r[\"error\"]}'
            break
"
```

---

### 问题 10: Python 版本兼容性

**症状：**
```
pydantic.v1.errors.ConfigError: unable to infer type
```

**原因：** Python 3.14 太新，SpaCy 不兼容

**解决方案：**

使用 Python 3.9-3.11：

```bash
# 使用系统 Python 3.9
rm -rf venv
/usr/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🆘 仍然无法解决？

### 检查系统资源

```bash
# 查看可用内存
free -h  # Linux
vm_stat  # macOS

# 查看进程内存占用
top -o MEM

# 查看Python进程
ps aux | grep python
```

### 启用详细日志

在脚本开头添加：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 最后手段：逐步调试

```bash
# 1. 只测试数据加载
python -c "from scripts.01_download_data import *; load_config()"

# 2. 只测试分块
python -c "from scripts.02_build_baseline import *; corpus = load_corpus(); print(len(corpus))"

# 3. 只测试向量化（1个文档）
python -c "
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.embeddings.create(
    model='text-embedding-3-small',
    input=['test text']
)
print('✅ OpenAI API 正常')
"
```

---

## 💡 预防措施

### 运行前检查清单

- [ ] 可用内存 >= 4GB
- [ ] Python 版本 3.9-3.11
- [ ] OpenAI API Key 已配置
- [ ] HuggingFace 可访问（或使用镜像）
- [ ] 磁盘空间 >= 5GB

### 监控运行状态

```bash
# 实时监控内存
watch -n 5 'free -h'  # Linux
watch -n 5 'vm_stat'  # macOS

# 监控进程
watch -n 5 'ps aux | grep python | grep -v grep'
```

---

## 🎯 总结

大多数问题都与**系统资源限制**有关。最稳妥的方案是：

**使用 Google Colab 或云服务器执行实验** ☁️

这样可以避免99%的故障，专注于实验结果而非环境问题。
