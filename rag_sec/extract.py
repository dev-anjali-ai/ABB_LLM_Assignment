from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List, Dict
import fitz  # PyMuPDF
from .utils import clean_text

ITEM_RE = re.compile(r"(?im)^\s*item\s+(\d{1,2}[a-z]?)\.?\s*(.*)$")
# Footer style examples:
# "Apple Inc. | 2024 Form 10-K | 56"
# "Tesla, Inc. | 2023 Form 10-K | 4"
FOOTER_PAGE_RE = re.compile(r"\|\s*(\d{1,4})\s*$")

@dataclass(frozen=True)
class PageDoc:
    doc_name: str
    company: str
    pdf_path: str
    page_pdf: int          # 1-indexed
    page_report: Optional[int]  # printed report page if detected
    item: Optional[str]    # e.g. "Item 7", "Item 1A"
    item_title: Optional[str]
    text: str

def _infer_company_and_name(pdf_path: Path) -> tuple[str, str]:
    name = pdf_path.stem
    lower = name.lower()
    if "tsla" in lower or "tesla" in lower:
        return "Tesla", "Tesla 10-K"
    if "apple" in lower or "10-q4-2024" in lower or "q4-2024" in lower:
        return "Apple", "Apple 10-K"
    # fallback: use filename
    return "Unknown", name

def _detect_report_page(page_text: str) -> Optional[int]:
    # look for a footer ending with "| <number>"
    # prefer last non-empty line
    lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
    if not lines:
        return None
    tail = lines[-1]
    m = FOOTER_PAGE_RE.search(tail)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None

def iter_pdf_pages(pdf_path: Path) -> Iterator[PageDoc]:
    company, doc_name = _infer_company_and_name(pdf_path)
    doc = fitz.open(str(pdf_path))
    current_item = None
    current_title = None

    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw = page.get_text("text") or ""
        txt = clean_text(raw)
        if not txt:
            continue

        # Update item if this page contains an ITEM heading.
        # We take the first match on the page as the "current item" for the page;
        # headings may be repeated in TOCs, but reranker + embedding usually resolve that.
        m = ITEM_RE.search(txt)
        if m:
            num = m.group(1).upper()
            title = (m.group(2) or "").strip()
            current_item = f"Item {num}"
            current_title = title if title else None

        page_report = _detect_report_page(raw)

        yield PageDoc(
            doc_name=doc_name,
            company=company,
            pdf_path=str(pdf_path),
            page_pdf=i + 1,
            page_report=page_report,
            item=current_item,
            item_title=current_title,
            text=txt,
        )
