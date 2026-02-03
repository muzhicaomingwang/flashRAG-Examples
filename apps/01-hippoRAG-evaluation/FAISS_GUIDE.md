# FAISS 向量数据库使用指南

## 1. FAISS 简介

**FAISS (Facebook AI Similarity Search)** 是 Meta AI 开发的开源向量相似度搜索库，专为高维向量的高效检索而设计。

### 核心特点
- **极致性能：** 优化的 C++ 实现，支持 GPU 加速（5-10倍速度提升）
- **内存高效：** 多种索引类型，灵活权衡速度和内存占用
- **可扩展：** 支持数十亿级别的向量检索
- **开源免费：** MIT 协议，无使用限制

### FAISS vs 完整向量数据库的区别

| 特性 | FAISS | Chroma/Pinecone |
|-----|-------|----------------|
| **定位** | 向量索引库 | 完整向量数据库 |
| **核心功能** | 相似度搜索 | 搜索 + 存储 + 元数据管理 |
| **持久化** | ❌ 需手动实现 | ✅ 内置 |
| **元数据** | ❌ 不支持 | ✅ 支持 |
| **性能** | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐ 需要封装 | ⭐⭐⭐⭐⭐ |

**结论：** FAISS 是纯粹的检索引擎，需要自己管理数据持久化和元数据，但性能无可匹敌。

---

## 2. FAISS 在 RAG 中的使用

### 2.1 基础工作流

```python
import faiss
import numpy as np
from openai import OpenAI

# 1. 准备文档和向量
documents = [
    "巴黎是法国的首都",
    "伦敦是英国的首都",
    "北京是中国的首都"
]

# 2. 向量化文档
client = OpenAI()
embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=doc
    )
    embeddings.append(response.data[0].embedding)

# 转换为 numpy 数组
vectors = np.array(embeddings, dtype=np.float32)  # FAISS 要求 float32
dimension = vectors.shape[1]  # 1536（OpenAI embedding 维度）

# 3. 创建 FAISS 索引
index = faiss.IndexFlatL2(dimension)  # L2 距离（欧氏距离）
index.add(vectors)  # 添加向量到索引

# 4. 检索
query = "法国的首都在哪里？"
query_vector = client.embeddings.create(
    model="text-embedding-ada-002",
    input=query
).data[0].embedding

query_vector = np.array([query_vector], dtype=np.float32)

# 搜索 top-5 最相似的文档
distances, indices = index.search(query_vector, k=5)

# 5. 获取结果
print(f"最相似的文档: {documents[indices[0][0]]}")
print(f"距离: {distances[0][0]}")
```

### 2.2 持久化方案（重要！）

FAISS 本身不提供持久化，需要自己实现：

```python
import faiss
import pickle

# ========== 保存 ==========
# 1. 保存 FAISS 索引
faiss.write_index(index, "hotpotqa.index")

# 2. 保存文档映射（id -> 文档内容和元数据）
doc_mapping = {
    i: {
        "content": documents[i],
        "metadata": {"source": "wikipedia", "chunk_id": i}
    }
    for i in range(len(documents))
}
with open("hotpotqa_docs.pkl", "wb") as f:
    pickle.dump(doc_mapping, f)

# ========== 加载 ==========
# 1. 加载 FAISS 索引
index = faiss.read_index("hotpotqa.index")

# 2. 加载文档映射
with open("hotpotqa_docs.pkl", "rb") as f:
    doc_mapping = pickle.load(f)

# 3. 检索并返回完整文档
distances, indices = index.search(query_vector, k=5)
results = [doc_mapping[idx] for idx in indices[0]]
```

---

## 3. FAISS 索引类型选择

FAISS 提供多种索引类型，适用于不同场景：

### 3.1 常用索引类型

| 索引类型 | 特点 | 速度 | 内存 | 精度 | 适用场景 |
|---------|------|------|------|------|---------|
| **IndexFlatL2** | 精确搜索，暴力计算 | ⭐⭐⭐ | ⭐⭐ | 100% | 小规模（<100K），实验 |
| **IndexFlatIP** | 精确搜索，内积距离 | ⭐⭐⭐ | ⭐⭐ | 100% | Cosine 相似度（需归一化） |
| **IndexIVFFlat** | 倒排索引，快速近似 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 95%+ | 中大规模（100K-10M） |
| **IndexHNSWFlat** | 图索引，极快检索 | ⭐⭐⭐⭐⭐ | ⭐ | 98%+ | 实时检索，内存充足 |

### 3.2 推荐选择：IndexFlatL2

**理由：**
- ✅ **精确搜索：** 100% 准确，无近似误差
- ✅ **实验友好：** 结果可复现，易于调试
- ✅ **规模合适：** 我们的实验数据量（几千到几万文档块）完全够用
- ✅ **简单可靠：** 无需调参，开箱即用

```python
# 创建 IndexFlatL2
import faiss

dimension = 1536  # text-embedding-ada-002 的维度
index = faiss.IndexFlatL2(dimension)

# 添加向量
index.add(vectors)  # vectors: np.array, shape=(n, 1536)

# 检索
distances, indices = index.search(query_vector, k=5)
```

### 3.3 如果数据量很大（可选优化）

如果某个数据集超过 10 万个文档块，可以使用 IVF 索引：

```python
# IndexIVFFlat: 倒排索引 + 精确搜索
nlist = 100  # 聚类中心数量（通常设为 sqrt(n)）
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 训练索引（找聚类中心）
index.train(vectors)

# 添加向量
index.add(vectors)

# 检索时设置探查的聚类数
index.nprobe = 10  # 探查 10 个聚类（权衡速度和精度）
distances, indices = index.search(query_vector, k=5)
```

---

## 4. FAISS 在本实验中的完整实现

### 4.1 构建统一的 FAISS 知识库

```python
# scripts/build_knowledge_base.py
import faiss
import numpy as np
import pickle
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FAISSKnowledgeBase:
    """统一的 FAISS 知识库构建器"""

    def __init__(self, dataset_name, persist_dir="./data/indices/faiss"):
        self.dataset_name = dataset_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.dimension = 1536

    def build_from_documents(self, documents):
        """从文档构建 FAISS 索引"""

        # 1. 统一分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)
        print(f"分块完成: {len(chunks)} 个文档块")

        # 2. 向量化（批量处理）
        texts = [chunk.page_content for chunk in chunks]
        vectors = []

        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = self.embeddings.embed_documents(batch_texts)
            vectors.extend(batch_vectors)
            print(f"向量化进度: {min(i+batch_size, len(texts))}/{len(texts)}")

        vectors = np.array(vectors, dtype=np.float32)

        # 3. 创建 FAISS 索引
        index = faiss.IndexFlatL2(self.dimension)
        index.add(vectors)
        print(f"FAISS 索引构建完成: {index.ntotal} 个向量")

        # 4. 保存索引
        index_path = self.persist_dir / f"{self.dataset_name}.index"
        faiss.write_index(index, str(index_path))

        # 5. 保存文档映射
        doc_mapping = {
            i: {
                "content": chunks[i].page_content,
                "metadata": chunks[i].metadata,
            }
            for i in range(len(chunks))
        }
        mapping_path = self.persist_dir / f"{self.dataset_name}_docs.pkl"
        with open(mapping_path, "wb") as f:
            pickle.dump(doc_mapping, f)

        print(f"知识库已保存到: {self.persist_dir}")

        return index, doc_mapping

    def load(self):
        """加载已有的 FAISS 索引"""
        index_path = self.persist_dir / f"{self.dataset_name}.index"
        mapping_path = self.persist_dir / f"{self.dataset_name}_docs.pkl"

        index = faiss.read_index(str(index_path))
        with open(mapping_path, "rb") as f:
            doc_mapping = pickle.load(f)

        print(f"已加载 {index.ntotal} 个向量")
        return index, doc_mapping

    def search(self, query, k=5):
        """检索相似文档"""
        # 加载索引
        index, doc_mapping = self.load()

        # 向量化查询
        query_vector = np.array([self.embeddings.embed_query(query)], dtype=np.float32)

        # 检索
        distances, indices = index.search(query_vector, k)

        # 返回结果
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            results.append({
                "rank": i + 1,
                "content": doc_mapping[idx]["content"],
                "metadata": doc_mapping[idx]["metadata"],
                "distance": float(dist),
                "similarity": 1 / (1 + float(dist))  # 距离转相似度
            })

        return results
```

### 4.2 使用示例

```python
# 构建知识库
kb = FAISSKnowledgeBase(dataset_name="hotpotqa")
index, doc_mapping = kb.build_from_documents(documents)

# 检索
results = kb.search("Who is the president of France?", k=5)

# 查看结果
for result in results:
    print(f"Rank {result['rank']}: {result['content'][:100]}...")
    print(f"Similarity: {result['similarity']:.3f}\n")
```

---

## 5. FAISS + BM25 混合检索

### 5.1 混合检索策略

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """FAISS (Dense) + BM25 (Sparse) 混合检索器"""

    def __init__(self, faiss_kb, bm25_index, doc_mapping):
        self.faiss_kb = faiss_kb
        self.bm25_index = bm25_index
        self.doc_mapping = doc_mapping

    def search(self, query, k=5, alpha=0.5):
        """
        混合检索

        Args:
            query: 查询文本
            k: 返回文档数
            alpha: FAISS 权重（1-alpha 为 BM25 权重）
        """
        # 1. FAISS 检索（语义相似度）
        faiss_results = self.faiss_kb.search(query, k=k*2)
        faiss_scores = {
            i: res["similarity"]
            for i, res in enumerate(faiss_results)
        }

        # 2. BM25 检索（关键词匹配）
        query_tokens = query.split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # 归一化 BM25 分数到 [0, 1]
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

        # 3. 混合评分
        hybrid_scores = {}
        for doc_id in range(len(self.doc_mapping)):
            faiss_score = faiss_scores.get(doc_id, 0)
            bm25_score = bm25_scores[doc_id]

            hybrid_scores[doc_id] = alpha * faiss_score + (1 - alpha) * bm25_score

        # 4. 排序并返回 top-k
        top_k_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:k]

        results = []
        for rank, doc_id in enumerate(top_k_ids):
            results.append({
                "rank": rank + 1,
                "content": self.doc_mapping[doc_id]["content"],
                "hybrid_score": hybrid_scores[doc_id],
                "faiss_score": faiss_scores.get(doc_id, 0),
                "bm25_score": bm25_scores[doc_id],
            })

        return results
```

### 5.2 混合检索参数调优

```python
# 实验不同的权重
alphas = [0.3, 0.5, 0.7, 1.0]  # 1.0 表示纯 FAISS

for alpha in alphas:
    results = hybrid_retriever.search(query, k=5, alpha=alpha)
    # 评估结果...
```

---

## 6. FAISS 性能优化技巧

### 6.1 批量操作

```python
# ❌ 不推荐：逐个添加向量
for vector in vectors:
    index.add(vector)

# ✅ 推荐：批量添加
index.add(vectors)  # 一次性添加所有向量
```

### 6.2 GPU 加速（可选）

```python
import faiss

# CPU 索引
cpu_index = faiss.IndexFlatL2(dimension)

# 转移到 GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 0 表示 GPU 0

# 使用 GPU 索引检索（速度提升 5-10 倍）
distances, indices = gpu_index.search(query_vector, k=5)
```

### 6.3 向量归一化（Cosine 相似度）

```python
# 如果想使用 Cosine 相似度而非 L2 距离
import faiss
import numpy as np

# 归一化向量
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

vectors_normalized = normalize_vectors(vectors)

# 使用 IndexFlatIP (内积)
index = faiss.IndexFlatIP(dimension)
index.add(vectors_normalized)

# 检索时也需要归一化
query_normalized = normalize_vectors(query_vector)
distances, indices = index.search(query_normalized, k=5)
# distances 现在是 cosine 相似度（值越大越相似）
```

---

## 7. FAISS 在本实验中的配置

### 7.1 目录结构

```
data/indices/faiss/
├── hotpotqa.index           # FAISS 索引文件
├── hotpotqa_docs.pkl        # 文档映射（id -> 内容）
├── nq.index
├── nq_docs.pkl
├── triviaqa.index
├── triviaqa_docs.pkl
└── ...（其他数据集）
```

### 7.2 配置文件

```yaml
# configs/knowledge_base_config.yaml

# 统一文档处理配置
document_processing:
  chunk_size: 512
  chunk_overlap: 50
  separators: ["\n\n", "\n", ". ", " ", ""]
  tokenizer: "tiktoken"

# 向量化配置
embedding:
  model: "text-embedding-ada-002"
  dimension: 1536
  batch_size: 100

# FAISS 配置
faiss:
  index_type: "IndexFlatL2"      # 精确搜索
  normalize_vectors: false        # L2 距离不需要归一化
  persist_directory: "./data/indices/faiss"

  # 如果数据量大，可启用 IVF
  # index_type: "IndexIVFFlat"
  # nlist: 100
  # nprobe: 10

# BM25 配置
bm25:
  k1: 1.5
  b: 0.75
  persist_directory: "./data/indices/bm25"
```

---

## 8. FAISS 常见问题

### Q1: FAISS 索引文件有多大？
```python
# 计算索引大小
n_vectors = 10000  # 1万个向量
dimension = 1536
index_size_mb = (n_vectors * dimension * 4) / (1024 * 1024)  # float32 = 4 bytes
print(f"索引大小: {index_size_mb:.2f} MB")  # 约 58.6 MB

# HotpotQA 约 50K 文档块 → 索引约 293 MB
```

### Q2: FAISS 检索速度有多快？
```python
import time

# 测试检索速度
start = time.time()
for _ in range(1000):
    distances, indices = index.search(query_vector, k=5)
end = time.time()

print(f"平均检索时间: {(end - start) / 1000 * 1000:.2f} ms")
# IndexFlatL2: 约 1-5 ms（取决于索引大小）
```

### Q3: 如何处理元数据过滤？
```python
# FAISS 不支持元数据过滤，需要后处理
def search_with_filter(query, k=5, filter_func=None):
    # 1. 检索更多候选（k * 3）
    distances, indices = index.search(query_vector, k=k*3)

    # 2. 应用过滤
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc = doc_mapping[idx]
        if filter_func is None or filter_func(doc["metadata"]):
            results.append({"content": doc["content"], "distance": dist})
            if len(results) >= k:
                break

    return results

# 使用示例
results = search_with_filter(
    query="法国首都",
    k=5,
    filter_func=lambda meta: meta.get("source") == "wikipedia"
)
```

---

## 9. 与 LangChain 集成

LangChain 提供了 FAISS 的封装，简化使用：

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 创建
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# 保存
vectorstore.save_local("data/indices/faiss/hotpotqa")

# 加载
vectorstore = FAISS.load_local(
    "data/indices/faiss/hotpotqa",
    embeddings,
    allow_dangerous_deserialization=True
)

# 检索
results = vectorstore.similarity_search(query, k=5)

# 带分数检索
results_with_scores = vectorstore.similarity_search_with_score(query, k=5)
for doc, score in results_with_scores:
    print(f"距离: {score:.3f}, 内容: {doc.page_content[:50]}...")
```

**推荐：** 使用 LangChain 的 FAISS 封装，它已经处理了持久化和文档映射。

---

## 10. 快速测试脚本

### 测试 FAISS 基础功能

```python
# test_faiss.py
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 测试文档
documents = [
    "The capital of France is Paris.",
    "London is the capital of the United Kingdom.",
    "Beijing is the capital of China.",
    "Tokyo is the capital of Japan.",
    "Berlin is the capital of Germany.",
]

# 使用 LangChain 封装
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_texts(documents, embeddings)

# 检索
query = "What is the capital of France?"
results = vectorstore.similarity_search_with_score(query, k=3)

print(f"查询: {query}\n")
for i, (doc, score) in enumerate(results):
    print(f"{i+1}. 距离: {score:.3f}")
    print(f"   内容: {doc.page_content}\n")

# 保存和加载测试
vectorstore.save_local("./test_faiss")
loaded_store = FAISS.load_local(
    "./test_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)
print("✓ 保存和加载测试成功")
```

---

## 11. FAISS vs Chroma 在本实验中的权衡

| 考虑因素 | FAISS ⭐ | Chroma |
|---------|---------|--------|
| **检索速度** | 极快（1-5ms） | 中等（10-50ms） |
| **内存效率** | 高 | 中 |
| **持久化** | 需手动实现（2行代码） | 自动 |
| **元数据过滤** | 需后处理 | 原生支持 |
| **GPU 支持** | ✅ | ❌ |
| **学术认可度** | 极高（广泛引用） | 中等（较新） |
| **实验复现性** | 极高（确定性） | 高 |

**你的选择（FAISS）的优势：**
- ✅ 性能最优，适合大规模实验
- ✅ Meta AI 官方维护，久经考验
- ✅ 学术界标准选择，论文中常用
- ✅ GPU 加速选项（如果需要）

**需要注意：**
- 需要手动管理持久化（不过只需 2 行代码）
- 元数据过滤需要后处理

---

## 12. 下一步

FAISS 配置已更新完毕。需要我创建一个 FAISS 快速测试脚本让你验证一下吗？

或者你想直接开始实现实验代码的哪个部分？
