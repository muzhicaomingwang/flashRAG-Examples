# Experiment Record Template

## Run Metadata
- Date (YYYY-MM-DD):
- Operator:
- Purpose / Hypothesis:
- Dataset: HotpotQA
- Split: dev
- Sample size (max_samples): 7405
- Random seed: 42

## Code & Environment
- Repo path:
- Git branch:
- Git commit:
- Dirty worktree (yes/no):
- Python version:
- OS:

## Model & API
- LLM (generation): gpt-5.2
- Embedding model:
- Temperature: 0.0
- Max tokens: 256
- API provider:
- Rate limit / concurrency:

## Retrieval & Index
- Baseline top_k: 5
- HippoRAG initial_k: 20
- HippoRAG rerank_k: 5
- FAISS index type: IndexFlatL2
- Normalize vectors: true

## Document Processing
- Chunk size: 512
- Chunk overlap: 50
- Separators: ["\n\n", "\n", ". ", " ", ""]
- Tokenizer: cl100k_base

## HippoRAG Graph
- Sampling ratio: 0.3
- NER method: spacy (en_core_web_sm)
- RE method: spacy_dep
- PageRank damping factor: 0.85
- PageRank max iterations: 100

## Runs
- Baseline RAG: (path to predictions.json)
- HippoRAG: (path to predictions.json)

## Metrics
- F1:
- EM:
- Recall@5:
- Precision@5:

## Notes / Anomalies
- 

## Comparison
- Against prior run (commit/date):
- Key deltas:
