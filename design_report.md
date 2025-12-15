# Design Report (1 page)

## 1) Chunking & Metadata
- **Extraction:** PyMuPDF extracts text per PDF page. We keep both (a) PDF page number and (b) any **printed report page** detected from footers.
- **Sectioning:** We detect SEC headings using case-insensitive regex for `ITEM 1`, `ITEM 1A`, ... `ITEM 16` and carry forward the current Item across pages.
- **Chunking:** Page-aware token chunking (default ~900 tokens with ~120 token overlap) so citations stay stable.
- **Metadata preserved per chunk:** `{company, doc_name, item, item_title, page_pdf, page_report, chunk_id}`.

## 2) Retrieval & Re-ranking
- **Vector retrieval:** FAISS inner-product on normalized embeddings; returns top-5 chunks.
- **Re-ranker:** Cross-encoder (default `BAAI/bge-reranker-base`) scores (query, chunk) pairs and reorders the top-5.
  Justification: rerankers are strong at precision over small candidate sets, especially for numeric/finance questions where lexical signals matter.

## 3) LLM Choice & Prompting
- **LLM:** Any open-access instruct model via vLLM (e.g., Llama 3, Mistral, Phi-3). vLLM provides high-throughput batching and KV-cache efficiency.
- **Prompt:** Strict grounding policy:
  - Use ONLY provided context.
  - Return `Not specified in the document.` if answer cannot be supported.
  - Refuse out-of-scope questions with the exact required sentence.

## 4) Out-of-scope Handling
- A deterministic gate flags questions about:
  - future forecasts (e.g., stock price forecasts)
  - “as of 2025” info beyond filing period
  - non-filing trivia (paint color, etc.)
- For these, system returns the refusal sentence and `sources=[]` without calling the LLM.

