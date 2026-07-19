"""Tests for the RetrievalEngine (F03)."""

from __future__ import annotations

import numpy as np

from missy.retrieval.embedder import HashingEmbedder
from missy.retrieval.engine import RetrievalEngine, RetrievalResult


def _engine(tmp_path=None):
    return RetrievalEngine(
        embedder=HashingEmbedder(dimension=256),
        index_dir=str(tmp_path) if tmp_path else None,
        max_chars=120,
        overlap=20,
    )


class TestIndexingAndQuery:
    def test_query_returns_citation_grounded_results(self) -> None:
        eng = _engine()
        eng.index_document(
            "keys.md",
            "To rotate API keys, run the vault set command and restart the gateway. "
            "The default key rotation strategy is failover.",
        )
        eng.index_document(
            "net.md",
            "The network policy engine enforces CIDR blocks and domain suffix "
            "matching with per-category host allowlists.",
        )
        results = eng.query("how do I rotate credentials", top_k=2)
        assert results
        top = results[0]
        assert isinstance(top, RetrievalResult)
        assert top.doc_id == "keys.md"
        # source_span cites back into the original document text.
        start, end = top.source_span
        assert 0 <= start < end

    def test_source_span_matches_indexed_text(self) -> None:
        eng = _engine()
        src = "Alpha content. " * 30 + "unique_marker_token appears here."
        eng.index_document("d", src)
        results = eng.query("unique_marker_token", top_k=1)
        s, e = results[0].source_span
        assert src[s:e] == results[0].text

    def test_empty_index_returns_empty(self) -> None:
        assert _engine().query("anything") == []

    def test_top_k_zero_returns_empty(self) -> None:
        eng = _engine()
        eng.index_document("d", "some content here")
        assert eng.query("content", top_k=0) == []


class TestIncrementalReindex:
    def test_reindex_replaces_document_chunks(self) -> None:
        eng = _engine()
        eng.index_document("d", "original content about vaults " * 10)
        first = eng.stats()["chunks"]
        assert first > 0
        eng.index_document("d", "short new content")
        assert eng.stats()["documents"] == 1
        assert eng.stats()["chunks"] == 1  # replaced, not appended

    def test_remove_document(self) -> None:
        eng = _engine()
        eng.index_document("a", "first doc")
        eng.index_document("b", "second doc")
        removed = eng.remove_document("a")
        assert removed >= 1
        assert eng.stats()["doc_ids"] == ["b"]
        assert all(r.doc_id == "b" for r in eng.query("doc", top_k=5))

    def test_empty_document_indexes_zero_chunks(self) -> None:
        eng = _engine()
        assert eng.index_document("blank", "   ") == 0
        assert eng.stats()["chunks"] == 0


class TestPersistence:
    def test_save_and_reload_round_trips(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng.index_document("k.md", "vault key rotation and secret storage details")
        eng.index_document("n.md", "network cidr policy allowlist configuration")
        stats_before = eng.stats()

        reloaded = _engine(tmp_path)
        assert reloaded.stats()["chunks"] == stats_before["chunks"]
        assert reloaded.stats()["doc_ids"] == stats_before["doc_ids"]
        # queries work on the reloaded index
        assert reloaded.query("rotate key", top_k=1)[0].doc_id == "k.md"

    def test_stale_dimension_index_is_ignored(self, tmp_path) -> None:
        eng = RetrievalEngine(embedder=HashingEmbedder(256), index_dir=str(tmp_path))
        eng.index_document("d", "content to persist")
        # Reload with a different embedding dimension -> stale, ignored.
        other = RetrievalEngine(embedder=HashingEmbedder(128), index_dir=str(tmp_path))
        assert other.stats()["chunks"] == 0

    def test_no_index_dir_does_not_persist(self, tmp_path) -> None:
        eng = _engine()  # index_dir=None
        eng.index_document("d", "content")
        eng.save()  # no-op, must not raise
        assert not list(tmp_path.iterdir())


class TestIndexFile:
    def test_index_file_uses_path_as_default_doc_id(self, tmp_path) -> None:
        p = tmp_path / "notes.txt"
        p.write_text("retrieval engine indexes local files with citations")
        eng = _engine()
        n = eng.index_file(p)
        assert n >= 1
        results = eng.query("index local files", top_k=1)
        assert results[0].doc_id == str(p.resolve())


class TestResultHelpers:
    def test_citation_format(self) -> None:
        r = RetrievalResult(doc_id="doc", text="x", source_span=(3, 9), score=0.5)
        assert r.citation() == "doc[3:9]"

    def test_vectors_stay_float32(self, tmp_path) -> None:
        eng = _engine(tmp_path)
        eng.index_document("d", "content here for embedding")
        assert eng._vectors.dtype == np.float32
