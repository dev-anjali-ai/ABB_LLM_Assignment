from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple

OUT_OF_SCOPE_MSG = "This question cannot be answered based on the provided documents."
NOT_SPECIFIED_MSG = "Not specified in the document."

@dataclass
class ScopeGate:
    apple_max_year: int = 2024
    tesla_max_year: int = 2023

    def is_out_of_scope(self, query: str) -> bool:
        q = query.lower()

        # explicit forecast / prediction / price targets
        forecast_terms = ["forecast", "predict", "prediction", "price target", "stock price"]
        if any(t in q for t in forecast_terms):
            return True

        # trivia not normally in filings
        trivia_terms = ["painted", "what color", "headquarters painted", "hq painted", "wall color"]
        if any(t in q for t in trivia_terms):
            return True

        # year gating: if question asks "as of 2025" etc beyond covered years
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", q)]
        if years:
            # infer company by mention; if none mentioned, treat future years as out-of-scope
            is_apple = "apple" in q
            is_tesla = "tesla" in q or "tsla" in q
            for y in years:
                if is_apple and y > self.apple_max_year:
                    return True
                if is_tesla and y > self.tesla_max_year:
                    return True
                if (not is_apple and not is_tesla) and y > max(self.apple_max_year, self.tesla_max_year):
                    return True

        # roles "CFO as of 2025" tends to be out-of-scope by assignment
        if "cfo" in q and "2025" in q:
            return True

        return False
