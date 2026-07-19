"""On-device text embedders for the retrieval engine (F03).

Two backends, selected at construction time with a dependency-free default so
the base install stays lean and fully offline:

* :class:`HashingEmbedder` — pure-Python + NumPy. Deterministic feature
  hashing over word unigrams/bigrams and character n-grams with sublinear TF
  weighting, L2-normalized. No model download, no network, no heavy deps —
  matching Missy's secure-by-default, self-hosted posture. This is the
  default and is always available.
* :class:`SentenceTransformerEmbedder` — used only when
  ``sentence-transformers`` is installed (the ``[retrieval]`` extra). Produces
  genuinely semantic dense vectors from a local model.

Both expose the same tiny interface (:class:`Embedder`): ``embed(texts)`` →
an ``(n, dim)`` float32 NumPy array whose rows are L2-normalized, plus
``dimension`` and ``name`` properties. Normalized rows make cosine similarity
a plain inner product, which both the dense index and RRF fusion rely on.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used across the retrieval engine."""

    @property
    def dimension(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dimension)`` float32, L2-normalized array."""
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder.

    Features are word unigrams, word bigrams, and character 3–5 grams, each
    hashed into a fixed-dimension vector with a signed hash (to reduce
    collision bias) and sublinear term-frequency weighting. The result is L2
    normalized so cosine similarity is an inner product.

    It is not a neural embedder, but it captures real lexical/sub-lexical
    overlap (including morphology and typos via char n-grams), is fully
    offline, and is stable across processes — which is exactly what the dense
    half of the hybrid index needs when no model is installed.
    """

    def __init__(self, dimension: int = 384, *, char_ngrams: tuple[int, int] = (3, 5)) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = int(dimension)
        self._char_lo, self._char_hi = char_ngrams

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"hashing-{self._dimension}"

    def _tokens(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        feats: list[str] = list(words)
        # Word bigrams.
        feats += [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        # Character n-grams over the normalized, space-joined word stream.
        joined = " ".join(words)
        for n in range(self._char_lo, self._char_hi + 1):
            if len(joined) < n:
                continue
            feats += [f"#{joined[i : i + n]}" for i in range(len(joined) - n + 1)]
        return feats

    @staticmethod
    def _hash(feature: str) -> tuple[int, float]:
        # blake2b keyed digest → (bucket, sign). Splitting the digest gives an
        # independent sign bit so collisions partially cancel instead of
        # always adding constructively.
        import hashlib

        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        val = int.from_bytes(h, "big")
        sign = 1.0 if (val & 1) else -1.0
        return val, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, float] = {}
            signs: dict[int, float] = {}
            for feat in self._tokens(text):
                val, sign = self._hash(feat)
                bucket = val % self._dimension
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
                signs[bucket] = sign
            for bucket, tf in counts.items():
                # Sublinear TF weighting damps very frequent features.
                mat[row, bucket] = signs[bucket] * (1.0 + np.log(tf))
        return _l2_normalize(mat)


class SentenceTransformerEmbedder:
    """Semantic embedder backed by a locally-installed sentence-transformers model.

    Constructed lazily; raises at construction if the library is unavailable
    so callers can fall back to :class:`HashingEmbedder`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only when installed
            raise RuntimeError(
                "sentence-transformers is not installed; install the [retrieval] "
                "extra or use HashingEmbedder"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        # sentence-transformers >=5.x renamed get_sentence_embedding_dimension()
        # to get_embedding_dimension(); prefer the new name, fall back for
        # older installs, and derive from a probe encode as a last resort.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        if get_dim is not None:
            self._dimension = int(get_dim())
        else:  # pragma: no cover - defensive
            self._dimension = int(
                np.asarray(self._model.encode(["x"], convert_to_numpy=True)).shape[-1]
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"sbert:{self._model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs model
        vecs = np.asarray(self._model.encode(texts, convert_to_numpy=True), dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return _l2_normalize(vecs)


def get_default_embedder(dimension: int = 384) -> Embedder:
    """Return the best available embedder without requiring optional deps.

    Prefers a local sentence-transformers model when installed; otherwise the
    always-available :class:`HashingEmbedder`.
    """
    try:
        import sentence_transformers  # noqa: F401

        return SentenceTransformerEmbedder()
    except Exception:  # ImportError, or model load failure — degrade gracefully.
        return HashingEmbedder(dimension=dimension)
