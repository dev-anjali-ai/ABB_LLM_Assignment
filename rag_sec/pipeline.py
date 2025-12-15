from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import os
import re

from .config import RAGConfig
from .extract import iter_pdf_pages
from .chunking import TokenChunker
from .embeddings import Embedder
from .vector_store import FaissStore
from .rerank import Reranker
from .llm_vllm import VLLMGenerator
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .guards import ScopeGate, OUT_OF_SCOPE_MSG, NOT_SPECIFIED_MSG
from .utils import dedupe_keep_order

def _format_source(meta: Dict[str, Any]) -> List[str]:
    doc = meta.get("doc_name", "Unknown")
    item = meta.get("item", "Unknown")
    # Prefer printed report page if available; else pdf page
    page = meta.get("page_report") or meta.get("page_pdf")
    return [doc, item, f"p. {page}"]

def _compact_context(metas_texts: List[Tuple[Dict[str, Any], str]], max_chars: int = 9000) -> List[str]:
    # Keep a few chunks; add metadata header lines to help the LLM.
    blocks = []
    total = 0
    for meta, txt in metas_texts:
        page = meta.get("page_report") or meta.get("page_pdf")
        header = f"[DOC={meta.get('doc_name')}] [SECTION={meta.get('item')}] [PAGE={page}]"
        block = header + "\n" + txt.strip()
        if total + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        total += len(block)
    return blocks

def _evidence_matches(context_blocks: List[str], evidence: List[str]) -> bool:
    ctx = "\n".join(context_blocks)
    for ev in evidence:
        if not ev:
            continue
        if ev not in ctx:
            return False
    return True

@dataclass
class RAGPipeline:
    config: RAGConfig
    store: FaissStore
    embedder: Embedder
    reranker: Reranker
    llm: VLLMGenerator
    scope_gate: ScopeGate

    @classmethod
    def from_index(cls, index_dir: Path, config: RAGConfig) -> "RAGPipeline":
        store = FaissStore.load(index_dir)
        embedder = Embedder(config.embed_model, device=config.embed_device, batch_size=config.embed_batch_size, normalize=config.normalize_embeddings)
        reranker = Reranker(config.rerank_model, device=config.rerank_device, batch_size=config.rerank_batch_size)
        llm = VLLMGenerator(
            model=os.environ.get("RAG_LLM_MODEL", config.llm_model),
            max_new_tokens=config.llm_max_new_tokens,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
        )
        return cls(config=config, store=store, embedder=embedder, reranker=reranker, llm=llm, scope_gate=ScopeGate())

    def retrieve_top5(self, query: str) -> List[Tuple[Dict[str, Any], str, float]]:
        qvec = self.embedder.encode([query])[0]
        hits = self.store.search(qvec, self.config.top_k)
        metas_texts = [(self.store.metas[i], self.store.texts[i], score) for i, score in hits]
        return metas_texts

    def rerank_top5(self, query: str, retrieved: List[Tuple[Dict[str, Any], str, float]]) -> List[Tuple[Dict[str, Any], str, float]]:
        passages = [t for (_, t, _) in retrieved]
        order = self.reranker.rerank(query, passages)  # (idx_in_list, rerank_score)
        reranked = [(retrieved[i][0], retrieved[i][1], float(score)) for i, score in order]
        return reranked

    def answer(self, query: str) -> Dict[str, Any]:
        # Out-of-scope shortcut (required exact refusal)
        if self.scope_gate.is_out_of_scope(query):
            return {"answer": OUT_OF_SCOPE_MSG, "sources": []}

        retrieved = self.retrieve_top5(query)
        if not retrieved:
            return {"answer": NOT_SPECIFIED_MSG, "sources": []}

        reranked = self.rerank_top5(query, retrieved)
        best_meta, best_text, best_score = reranked[0]

        # Rerank confidence gate
        if best_score < self.config.rerank_min_score:
            return {"answer": NOT_SPECIFIED_MSG, "sources": []}

        context_blocks = _compact_context([(m, t) for (m, t, _) in reranked])

        user_prompt = build_user_prompt(query, context_blocks)
        try:
            result = self.llm.generate_json(SYSTEM_PROMPT, user_prompt)
        except Exception:
            # If the LLM fails, we degrade safely without hallucinating.
            return {"answer": NOT_SPECIFIED_MSG, "sources": []}

        answer = (result.get("answer") or "").strip()
        answerable = bool(result.get("answerable", False))
        evidence = result.get("evidence") or []

        if answer == OUT_OF_SCOPE_MSG:
            return {"answer": OUT_OF_SCOPE_MSG, "sources": []}

        if (not answerable) or (answer == NOT_SPECIFIED_MSG):
            return {"answer": NOT_SPECIFIED_MSG, "sources": []}

        if self.config.evidence_must_match and (not _evidence_matches(context_blocks, evidence)):
            return {"answer": NOT_SPECIFIED_MSG, "sources": []}

        sources = _format_source(best_meta)
        return {"answer": answer, "sources": sources}

def build_index(data_dir: Path, index_dir: Path, config: RAGConfig) -> None:
    pdfs = sorted([p for p in data_dir.glob("*.pdf")])
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    pages = []
    for pdf in pdfs:
        pages.extend(list(iter_pdf_pages(pdf)))

    chunker = TokenChunker(config.embed_model, config.chunk_tokens, config.chunk_overlap, min_chars=config.min_chunk_chars)
    chunks = chunker.build_chunks(pages)
    texts = [c.text for c in chunks]
    metas = [c.meta for c in chunks]

    embedder = Embedder(config.embed_model, device=config.embed_device, batch_size=config.embed_batch_size, normalize=config.normalize_embeddings)
    embs = embedder.encode(texts)

    store = FaissStore.build(embeddings=embs, texts=texts, metas=metas)
    store.save(index_dir)

# Required interface (assignment)
_PIPELINE_SINGLETON: RAGPipeline | None = None

def answer_question(query: str) -> Dict[str, Any]:
    """Answers a question using the RAG pipeline.

    Returns:
      {
        "answer": "...",
        "sources": ["Apple 10-K", "Item 8", "p. 28"]   # Empty list if refused/out-of-scope
      }
    """
    global _PIPELINE_SINGLETON
    index_dir = Path(os.environ.get("RAG_INDEX_DIR", "index"))
    config = RAGConfig(llm_model=os.environ.get("RAG_LLM_MODEL", RAGConfig().llm_model))
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = RAGPipeline.from_index(index_dir=index_dir, config=config)
    return _PIPELINE_SINGLETON.answer(query)
