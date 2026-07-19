"""Hybrid dense + sparse retrieval index with reciprocal-rank fusion (F03).

The engine ranks chunks by combining two complementary signals:

* **Dense** — cosine similarity over embedding vectors. Uses a FAISS
  ``IndexFlatIP`` when FAISS is installed (rows are L2-normalized, so inner
  product is cosine); otherwise a NumPy brute-force fallback with identical
  results. Captures paraphrase / semantic overlap.
* **Sparse** — BM25 over tokenized chunk text (pure Python, no deps). Captures
  exact term / rare-keyword matches the dense signal can miss.

The two ranked lists are merged with **Reciprocal Rank Fusion** (RRF), which
is scale-free (it needs only ranks, not comparable scores) and robust — the
standard way to fuse dense and sparse retrievers.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

try:
    import faiss  # type: ignore[import-untyped]

    _FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover - env-dependent
    _FAISS_AVAILABLE = False


class DenseIndex:
    """Cosine-similarity index over L2-normalized vectors.

    Ids are arbitrary integers supplied by the caller (the engine uses the
    chunk's ordinal). Backed by FAISS when available, else NumPy.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._ids: list[int] = []
        self._matrix: np.ndarray = np.zeros((0, self.dimension), dtype=np.float32)
        self._faiss = faiss.IndexFlatIP(self.dimension) if _FAISS_AVAILABLE else None

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"vector dim {vectors.shape[1]} != index dim {self.dimension}")
        self._ids.extend(ids)
        self._matrix = np.vstack([self._matrix, vectors]) if self.size else vectors
        if self._faiss is not None:
            self._faiss.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.size == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(query.astype(np.float32).reshape(1, -1))
        k = min(k, self.size)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return [
                (self._ids[i], float(s)) for s, i in zip(scores[0], idx[0], strict=False) if i != -1
            ]
        sims = self._matrix @ q[0]
        top = np.argsort(-sims)[:k]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in top]


class BM25SparseIndex:
    """Okapi BM25 sparse index over tokenized documents (pure Python)."""

    _WORD = __import__("re").compile(r"[a-z0-9]+")

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[int] = []
        self._docs_tokens: list[list[str]] = []
        self._df: dict[str, int] = defaultdict(int)
        self._doc_len: list[int] = []
        self._avg_len = 0.0

    @property
    def size(self) -> int:
        return len(self._ids)

    def _tokenize(self, text: str) -> list[str]:
        return self._WORD.findall(text.lower())

    def add(self, doc_id: int, text: str) -> None:
        toks = self._tokenize(text)
        self._ids.append(doc_id)
        self._docs_tokens.append(toks)
        self._doc_len.append(len(toks))
        for term in set(toks):
            self._df[term] += 1
        total = sum(self._doc_len)
        self._avg_len = total / len(self._doc_len) if self._doc_len else 0.0

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.size == 0 or k <= 0:
            return []
        q_terms = self._tokenize(query)
        n = self.size
        scores: list[tuple[int, float]] = []
        for pos, doc_id in enumerate(self._ids):
            toks = self._docs_tokens[pos]
            if not toks:
                continue
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            dl = self._doc_len[pos]
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                # BM25 idf (with +0.5 smoothing; clamped non-negative).
                idf = max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1.0))
                score += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
            if score > 0.0:
                scores.append((doc_id, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], *, k: int = 60, top_k: int | None = None
) -> list[tuple[int, float]]:
    """Fuse several ranked ``(id, score)`` lists into one via RRF.

    RRF score for an id is ``sum(1 / (k + rank))`` over the lists it appears
    in (rank is 1-based). ``k`` damps the influence of low ranks; 60 is the
    canonical default. Returns ids sorted by fused score descending.
    """
    fused: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    out = sorted(fused.items(), key=lambda x: -x[1])
    return out[:top_k] if top_k is not None else out
