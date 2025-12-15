from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from sentence_transformers import CrossEncoder

@dataclass
class Reranker:
    model_name: str
    device: str = "cuda"
    batch_size: int = 16

    def __post_init__(self):
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(self, query: str, passages: List[str]) -> List[Tuple[int, float]]:
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scored = list(enumerate([float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
