"""Optional FAISS-based vector memory store for semantic search.

Provides a :class:`VectorMemoryStore` that indexes text entries using
TF-IDF vectors and retrieves them via approximate nearest-neighbour
search backed by FAISS.

When ``faiss`` is not installed, the store degrades gracefully — all
methods become no-ops that return empty results and log a debug message.

Example::

    from missy.memory.vector_store import VectorMemoryStore

    store = VectorMemoryStore()
    store.add("The deployment failed due to a missing env var", {"category": "solution"})
    results = store.search("environment variable error", top_k=3)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FAISS import
# ---------------------------------------------------------------------------

try:
    import faiss  # type: ignore[import-untyped]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Simple vectorizer (no sklearn dependency required)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return _WORD_RE.findall(text.lower())


class SimpleVectorizer:
    """Minimal TF-IDF-like vectorizer using fixed-dimension hashing.

    Maps each token to a bucket via hash, accumulates term frequencies,
    and normalises to unit length.  This avoids needing scikit-learn or
    any external embedding model.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        """Encode *text* into a float vector of length :attr:`dimension`."""
        tokens = _tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        counts: Counter[int] = Counter()
        for token in tokens:
            bucket = hash(token) % self.dimension
            counts[bucket] += 1

        vec = [0.0] * self.dimension
        for bucket, count in counts.items():
            vec[bucket] = float(count)

        # L2 normalise
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


# ---------------------------------------------------------------------------
# VectorMemoryStore
# ---------------------------------------------------------------------------


class VectorMemoryStore:
    """FAISS-backed semantic search over text entries.

    Parameters:
        dimension: Vector dimensionality for the FAISS index.
        index_path: File path to persist/load the FAISS index.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: str = "~/.missy/memory.faiss",
    ) -> None:
        self.dimension = dimension
        self.index_path = str(Path(index_path).expanduser())
        self._metadata_path = self.index_path + ".meta"
        self._vectorizer = SimpleVectorizer(dimension=dimension)
        self._entries: list[dict] = []  # parallel to FAISS index rows
        self._index: faiss.IndexFlatL2 | None = None  # type: ignore[name-defined]

        if not _FAISS_AVAILABLE:
            logger.debug("faiss not installed — VectorMemoryStore is a no-op")
            return

        self._index = faiss.IndexFlatL2(dimension)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Add a text entry to the index.

        Args:
            text: The text content to index.
            metadata: Optional metadata dict (e.g. ``{"category": "solution"}``).
        """
        if not _FAISS_AVAILABLE or self._index is None:
            logger.debug("faiss not available — add() is a no-op")
            return

        import numpy as np

        vec = self._vectorizer.encode(text)
        arr = np.array([vec], dtype=np.float32)
        self._index.add(arr)
        entry = {"text": text, "metadata": metadata or {}}
        self._entries.append(entry)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for the *top_k* most similar entries.

        Args:
            query: The query text.
            top_k: Maximum number of results to return.

        Returns:
            A list of dicts, each with ``"text"``, ``"metadata"``, and
            ``"score"`` keys, ordered by descending similarity (lower
            distance = better match).
        """
        if not _FAISS_AVAILABLE or self._index is None:
            logger.debug("faiss not available — search() returns empty")
            return []

        if self._index.ntotal == 0:
            return []

        import numpy as np

        vec = self._vectorizer.encode(query)
        arr = np.array([vec], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(arr, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self._entries):
                continue
            entry = self._entries[idx]
            results.append(
                {
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "score": float(dist),
                }
            )
        return results

    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if not _FAISS_AVAILABLE or self._index is None:
            logger.debug("faiss not available — save() is a no-op")
            return

        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, self.index_path)

        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f)

        logger.info("Vector index saved: %d entries", len(self._entries))

    def load(self) -> None:
        """Load a previously saved FAISS index and metadata from disk."""
        if not _FAISS_AVAILABLE:
            logger.debug("faiss not available — load() is a no-op")
            return

        index_file = Path(self.index_path)
        meta_file = Path(self._metadata_path)

        if not index_file.exists() or not meta_file.exists():
            logger.debug("No saved vector index found at %s", self.index_path)
            return

        self._index = faiss.read_index(self.index_path)

        with open(self._metadata_path, encoding="utf-8") as f:
            self._entries = json.load(f)

        logger.info("Vector index loaded: %d entries", len(self._entries))

    def count(self) -> int:
        """Return the number of entries in the index."""
        if not _FAISS_AVAILABLE or self._index is None:
            return 0
        return self._index.ntotal
