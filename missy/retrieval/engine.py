"""On-device retrieval engine: chunk → embed → hybrid-rank → cite (F03).

``RetrievalEngine`` is the public core. It indexes documents by chunking them
(span-preserving), embedding each chunk with a local :class:`Embedder`, and
registering the chunk in both a dense and a sparse index. Queries fuse the two
signals with reciprocal-rank fusion and return results carrying a precise
``source_span`` back into the original document — citation-grounded recall,
fully offline.

Indexing is incremental: re-indexing an existing ``doc_id`` replaces that
document's chunks (and rebuilds the dense index, since the flat indexes don't
support in-place deletion). State persists to a directory as JSON (chunk
records + metadata) plus a NumPy ``.npy`` of the vectors, so an index survives
process restarts and can be rebuilt if the embedder changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import Chunk, chunk_text
from .embedder import Embedder, get_default_embedder
from .hybrid_index import BM25SparseIndex, DenseIndex, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = "~/.missy/retrieval"


@dataclass
class RetrievalResult:
    """A single ranked chunk returned from a query.

    Attributes:
        doc_id: Source document identifier.
        text: The chunk text (the citation body).
        source_span: ``(start, end)`` character offsets in the source document.
        score: Fused RRF score (higher is better).
        chunk_index: The chunk's ordinal within its document.
        metadata: Per-document metadata carried onto the chunk.
    """

    doc_id: str
    text: str
    source_span: tuple[int, int]
    score: float
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        """Return a compact human-readable citation string."""
        return f"{self.doc_id}[{self.source_span[0]}:{self.source_span[1]}]"


class RetrievalEngine:
    """Hybrid dense+sparse retrieval over locally-embedded document chunks."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        index_dir: str | Path | None = None,
        max_chars: int = 800,
        overlap: int = 150,
    ) -> None:
        self.embedder: Embedder = embedder or get_default_embedder()
        self.index_dir = Path(index_dir).expanduser() if index_dir else None
        self.max_chars = max_chars
        self.overlap = overlap
        # Ordinal-keyed chunk store; ordinal is the id used in both sub-indexes.
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, self.embedder.dimension), dtype=np.float32)
        self._dense = DenseIndex(self.embedder.dimension)
        self._sparse = BM25SparseIndex()
        if self.index_dir and (self.index_dir / "chunks.json").exists():
            self.load()

    # -- indexing ---------------------------------------------------------
    def index_document(
        self, doc_id: str, text: str, *, metadata: dict[str, Any] | None = None
    ) -> int:
        """Index (or re-index) a document. Returns the number of chunks added.

        Re-indexing an existing ``doc_id`` first removes its prior chunks.
        """
        if any(c.doc_id == doc_id for c in self._chunks):
            self.remove_document(doc_id, _rebuild=False)
        new_chunks = chunk_text(
            text,
            doc_id=doc_id,
            max_chars=self.max_chars,
            overlap=self.overlap,
            metadata=metadata,
        )
        if not new_chunks:
            self._rebuild()
            return 0
        vecs = self.embedder.embed([c.text for c in new_chunks])
        self._chunks.extend(new_chunks)
        self._vectors = np.vstack([self._vectors, vecs]) if len(self._vectors) else vecs
        self._rebuild()
        return len(new_chunks)

    def index_file(self, path: str | Path, *, doc_id: str | None = None) -> int:
        """Index a text file. ``doc_id`` defaults to the absolute path."""
        p = Path(path).expanduser()
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.index_document(doc_id or str(p.resolve()), text)

    def remove_document(self, doc_id: str, *, _rebuild: bool = True) -> int:
        """Remove all chunks for ``doc_id``. Returns the count removed."""
        keep_idx = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep_idx)
        if removed:
            self._chunks = [self._chunks[i] for i in keep_idx]
            self._vectors = self._vectors[keep_idx] if len(self._vectors) else self._vectors
        if _rebuild:
            self._rebuild()
        return removed

    def _rebuild(self) -> None:
        """Rebuild both sub-indexes from the current chunk/vector store."""
        self._dense = DenseIndex(self.embedder.dimension)
        self._sparse = BM25SparseIndex()
        if self._chunks:
            self._dense.add(list(range(len(self._chunks))), self._vectors)
            for i, c in enumerate(self._chunks):
                self._sparse.add(i, c.text)
        if self.index_dir:
            self.save()

    # -- querying ---------------------------------------------------------
    def query(
        self, text: str, *, top_k: int = 5, candidate_k: int | None = None
    ) -> list[RetrievalResult]:
        """Return the top-``k`` chunks for ``text`` via hybrid RRF."""
        if not self._chunks or top_k <= 0:
            return []
        candidate_k = candidate_k or max(top_k * 4, 20)
        qvec = self.embedder.embed([text])[0]
        dense = self._dense.search(qvec, candidate_k)
        sparse = self._sparse.search(text, candidate_k)
        fused = reciprocal_rank_fusion([dense, sparse], top_k=top_k)
        results: list[RetrievalResult] = []
        for ordinal, score in fused:
            c = self._chunks[ordinal]
            results.append(
                RetrievalResult(
                    doc_id=c.doc_id,
                    text=c.text,
                    source_span=(c.start, c.end),
                    score=float(score),
                    chunk_index=c.index,
                    metadata=dict(c.metadata),
                )
            )
        return results

    # -- introspection ----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        doc_ids = sorted({c.doc_id for c in self._chunks})
        return {
            "embedder": self.embedder.name,
            "dimension": self.embedder.dimension,
            "documents": len(doc_ids),
            "chunks": len(self._chunks),
            "doc_ids": doc_ids,
            "index_dir": str(self.index_dir) if self.index_dir else None,
        }

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        if not self.index_dir:
            return
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": self.embedder.name,
            "dimension": self.embedder.dimension,
            "chunks": [asdict(c) for c in self._chunks],
        }
        (self.index_dir / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")
        np.save(self.index_dir / "vectors.npy", self._vectors)

    def load(self) -> None:
        if not self.index_dir:
            return
        chunks_path = self.index_dir / "chunks.json"
        vectors_path = self.index_dir / "vectors.npy"
        if not chunks_path.exists():
            return
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        if payload.get("dimension") != self.embedder.dimension:
            logger.warning(
                "retrieval index dim %s != embedder dim %s; ignoring stale index",
                payload.get("dimension"),
                self.embedder.dimension,
            )
            return
        self._chunks = [
            Chunk(
                text=c["text"],
                start=c["start"],
                end=c["end"],
                doc_id=c["doc_id"],
                index=c["index"],
                metadata=c.get("metadata", {}),
            )
            for c in payload.get("chunks", [])
        ]
        if vectors_path.exists():
            self._vectors = np.load(vectors_path).astype(np.float32)
        else:  # pragma: no cover - re-embed if vectors missing
            self._vectors = (
                self.embedder.embed([c.text for c in self._chunks])
                if self._chunks
                else np.zeros((0, self.embedder.dimension), dtype=np.float32)
            )
        # Rebuild in-memory sub-indexes without re-persisting.
        self._dense = DenseIndex(self.embedder.dimension)
        self._sparse = BM25SparseIndex()
        if self._chunks:
            self._dense.add(list(range(len(self._chunks))), self._vectors)
            for i, c in enumerate(self._chunks):
                self._sparse.add(i, c.text)
