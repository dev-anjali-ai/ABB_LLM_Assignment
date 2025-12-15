from __future__ import annotations
import re
from typing import Iterable, List

_whitespace_re = re.compile(r"[ \t\u00A0]+")
_multiline_blank_re = re.compile(r"\n{3,}")

def clean_text(text: str) -> str:
    # normalize whitespace + fix hyphen line breaks
    text = text.replace("\r", "")
    # join hyphenated line breaks: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # convert single newlines inside paragraphs into spaces (keep blank lines)
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            out.append("")
            continue
        out.append(_whitespace_re.sub(" ", line))
    text2 = "\n".join(out)
    text2 = _multiline_blank_re.sub("\n\n", text2)
    return text2.strip()

def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
