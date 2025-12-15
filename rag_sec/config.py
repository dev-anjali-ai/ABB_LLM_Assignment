from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional

class RAGConfig(BaseModel):
    # Input PDFs -> index
    chunk_tokens: int = 900
    chunk_overlap: int = 120
    min_chunk_chars: int = 200

    # Embeddings
    embed_model: str = "BAAI/bge-small-en-v1.5"   # good default speed/quality
    embed_batch_size: int = 64
    embed_device: str = "cuda"                   # "cpu" if needed
    normalize_embeddings: bool = True

    # Vector search
    top_k: int = 5

    # Reranking
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_device: str = "cuda"                  # "cpu" if needed
    rerank_batch_size: int = 16

    # LLM (vLLM)
    llm_model: str = "microsoft/Phi-3-mini-4k-instruct"  # can be overridden by env
    llm_max_new_tokens: int = 256
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0

    # Guardrails
    rerank_min_score: float = 0.15  # tune per reranker; low threshold to avoid false negatives
    evidence_must_match: bool = True
    max_sources: int = 1            # return the best single citation (3 strings)

