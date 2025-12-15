from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
import orjson

@dataclass
class FaissStore:
    dim: int
    index: faiss.Index
    texts: List[str]
    metas: List[Dict]

    @classmethod
    def build(cls, embeddings: np.ndarray, texts: List[str], metas: List[Dict]) -> "FaissStore":
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [N, D]")
        n, d = embeddings.shape
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        return cls(dim=d, index=index, texts=texts, metas=metas)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idxs = self.index.search(query_vec.astype(np.float32), top_k)
        out = []
        for i, s in zip(idxs[0].tolist(), scores[0].tolist()):
            if i == -1:
                continue
            out.append((int(i), float(s)))
        return out

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "faiss.index"))
        (out_dir / "store.jsonl").write_bytes(
            b"\n".join(orjson.dumps({"text": t, "meta": m}) for t, m in zip(self.texts, self.metas))
        )

        meta = {"dim": self.dim, "count": len(self.texts)}
        (out_dir / "manifest.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))

    @classmethod
    def load(cls, out_dir: Path) -> "FaissStore":
        idx = faiss.read_index(str(out_dir / "faiss.index"))
        texts = []
        metas = []
        with (out_dir / "store.jsonl").open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                texts.append(obj["text"])
                metas.append(obj["meta"])
        dim = idx.d
        return cls(dim=dim, index=idx, texts=texts, metas=metas)
