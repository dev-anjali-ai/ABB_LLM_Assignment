# SEC 10-K RAG (Apple 2024 + Tesla 2023) — vLLM + FAISS + Reranker

This repository implements the complete RAG pipeline required by the assignment:
- PDF ingestion (page-aware, section-aware)
- Embeddings (open model) + FAISS vector index
- Retrieve top-5 chunks + rerank (cross-encoder)
- Answer generation using a local/open-access LLM via **vLLM**
- Strict grounding rules + out-of-scope refusal + required JSON output

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
# on GPU, also:
pip install -r requirements-gpu-vllm.txt
```

### 2) Put PDFs in `data/`
- `data/10-Q4-2024-As-Filed.pdf`
- `data/tsla-20231231-gen.pdf`

### 3) Build index
```bash
python -m rag_sec.cli ingest --data-dir data --index-dir index
```

### 4) Run evaluation (writes `outputs/predictions.json`)
```bash
python -m rag_sec.cli eval --index-dir index --out outputs/predictions.json
```

### 5) Serve an API (optional)
```bash
uvicorn rag_sec.api:app --host 0.0.0.0 --port 8000
```

POST `{"query":"...question..."}` to `/answer`.

## Notes
- Default embedder: `BAAI/bge-small-en-v1.5` (fast + good). Change in `rag_sec/config.py`.
- Default reranker: `BAAI/bge-reranker-base` (strong). Change in `rag_sec/config.py`.
- Default generator model is configured by env var `RAG_LLM_MODEL` (path or HF id).
  Example:
  ```bash
  export RAG_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
  ```

