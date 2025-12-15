from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import AutoTokenizer

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: Dict

def _make_chunk_id(doc_name: str, page_pdf: int, idx: int) -> str:
    return f"{doc_name}::p{page_pdf}::c{idx}"

class TokenChunker:
    def __init__(self, tokenizer_name: str, chunk_tokens: int, overlap: int, min_chars: int = 200):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.chunk_tokens = int(chunk_tokens)
        self.overlap = int(overlap)
        self.min_chars = int(min_chars)

    def chunk_page(self, page_text: str) -> List[str]:
        # Chunk within a page to preserve page citations.
        ids = self.tok.encode(page_text, add_special_tokens=False)
        if not ids:
            return []
        out = []
        start = 0
        n = len(ids)
        while start < n:
            end = min(n, start + self.chunk_tokens)
            piece_ids = ids[start:end]
            text = self.tok.decode(piece_ids, skip_special_tokens=True).strip()
            if len(text) >= self.min_chars:
                out.append(text)
            if end >= n:
                break
            start = max(0, end - self.overlap)
        return out

    def build_chunks(self, pages) -> List[Chunk]:
        chunks: List[Chunk] = []
        for p in pages:
            pieces = self.chunk_page(p.text)
            for j, piece in enumerate(pieces):
                meta = {
                    "company": p.company,
                    "doc_name": p.doc_name,
                    "item": p.item or "Cover Page",
                    "item_title": p.item_title or "",
                    "page_pdf": p.page_pdf,
                    "page_report": p.page_report,
                    "pdf_path": p.pdf_path,
                }
                cid = _make_chunk_id(p.doc_name, p.page_pdf, j)
                chunks.append(Chunk(chunk_id=cid, text=piece, meta=meta))
        return chunks
