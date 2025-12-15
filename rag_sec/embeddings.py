from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Embedder:
    model_name: str
    device: str = "cuda"
    batch_size: int = 64
    normalize: bool = True

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)
