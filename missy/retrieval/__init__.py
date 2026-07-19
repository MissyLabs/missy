"""On-device retrieval engine (F03): local embeddings + hybrid RAG.

A first-class, fully-offline retrieval core — chunking, on-device embeddings,
hybrid dense+sparse ranking with reciprocal-rank fusion, and citation-grounded
results — matching Missy's secure-by-default, self-hosted posture. No cloud
embedding calls; the default :class:`HashingEmbedder` needs no optional deps,
while the ``[retrieval]`` extra enables a local sentence-transformers model.

Public surface::

    from missy.retrieval import RetrievalEngine, get_default_embedder

    engine = RetrievalEngine(index_dir="~/.missy/retrieval")
    engine.index_document("notes.md", "... long text ...")
    for hit in engine.query("how do I rotate keys?", top_k=3):
        print(hit.citation(), hit.text)
"""

from __future__ import annotations

from .chunking import Chunk, chunk_text
from .embedder import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    get_default_embedder,
)
from .engine import DEFAULT_INDEX_DIR, RetrievalEngine, RetrievalResult
from .hybrid_index import (
    BM25SparseIndex,
    DenseIndex,
    reciprocal_rank_fusion,
)

__all__ = [
    "BM25SparseIndex",
    "Chunk",
    "DEFAULT_INDEX_DIR",
    "DenseIndex",
    "Embedder",
    "HashingEmbedder",
    "RetrievalEngine",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "chunk_text",
    "get_default_embedder",
    "reciprocal_rank_fusion",
]
