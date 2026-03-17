"""Tests for the FAISS-based vector memory store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from missy.memory.vector_store import (
    _FAISS_AVAILABLE,
    SimpleVectorizer,
    VectorMemoryStore,
)

_HAS_FAISS = _FAISS_AVAILABLE

needs_faiss = pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")


# ---------------------------------------------------------------------------
# SimpleVectorizer
# ---------------------------------------------------------------------------


class TestSimpleVectorizer:
    def test_encode_returns_correct_dimension(self) -> None:
        vec = SimpleVectorizer(dimension=128)
        result = vec.encode("hello world")
        assert len(result) == 128

    def test_encode_empty_string(self) -> None:
        vec = SimpleVectorizer(dimension=64)
        result = vec.encode("")
        assert len(result) == 64
        assert all(v == 0.0 for v in result)

    def test_encode_is_normalised(self) -> None:
        vec = SimpleVectorizer(dimension=384)
        result = vec.encode("the quick brown fox jumps over the lazy dog")
        norm = sum(v * v for v in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_similar_texts_closer(self) -> None:
        vec = SimpleVectorizer(dimension=384)
        v1 = vec.encode("python programming language")
        v2 = vec.encode("python programming syntax")
        v3 = vec.encode("cooking recipes for dinner")

        # Distance between v1 and v2 should be less than v1 and v3
        d12 = sum((a - b) ** 2 for a, b in zip(v1, v2, strict=True)) ** 0.5
        d13 = sum((a - b) ** 2 for a, b in zip(v1, v3, strict=True)) ** 0.5
        assert d12 < d13


# ---------------------------------------------------------------------------
# VectorMemoryStore (with FAISS)
# ---------------------------------------------------------------------------


@needs_faiss
class TestVectorMemoryStoreWithFaiss:
    def test_add_and_search(self, tmp_path: Path) -> None:
        store = VectorMemoryStore(
            dimension=128,
            index_path=str(tmp_path / "test.faiss"),
        )
        store.add("deployment failed due to missing environment variable")
        store.add("database connection timeout after migration")
        store.add("CSS styling issue on the login page")

        results = store.search("environment variable error", top_k=2)
        assert len(results) == 2
        # The most relevant result should mention environment
        assert "environment" in results[0]["text"].lower()

    def test_search_ranking(self, tmp_path: Path) -> None:
        store = VectorMemoryStore(
            dimension=128,
            index_path=str(tmp_path / "test.faiss"),
        )
        store.add("python error traceback exception handling")
        store.add("javascript react component rendering")
        store.add("python debugging exception stacktrace")

        results = store.search("python exception error", top_k=3)
        assert len(results) == 3
        # Both python entries should score better than the JS one
        texts = [r["text"] for r in results]
        js_idx = next(i for i, t in enumerate(texts) if "javascript" in t)
        # JS entry should not be first
        assert js_idx > 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        index_path = str(tmp_path / "persist.faiss")

        store1 = VectorMemoryStore(dimension=64, index_path=index_path)
        store1.add("first entry about networking")
        store1.add("second entry about databases")
        store1.save()

        store2 = VectorMemoryStore(dimension=64, index_path=index_path)
        store2.load()

        assert store2.count() == 2
        results = store2.search("network", top_k=1)
        assert len(results) == 1
        assert "networking" in results[0]["text"]

    def test_empty_store_search(self, tmp_path: Path) -> None:
        store = VectorMemoryStore(
            dimension=64,
            index_path=str(tmp_path / "empty.faiss"),
        )
        results = store.search("anything")
        assert results == []

    def test_categories(self, tmp_path: Path) -> None:
        store = VectorMemoryStore(
            dimension=64,
            index_path=str(tmp_path / "cat.faiss"),
        )
        store.add("user asked about weather", {"category": "conversation"})
        store.add("fixed the timeout bug", {"category": "solution"})
        store.add("code snippet for parsing", {"category": "fragment"})

        results = store.search("bug fix timeout", top_k=3)
        assert len(results) == 3
        # All results should have metadata preserved
        for r in results:
            assert "category" in r["metadata"]
        # The solution entry should rank first
        assert results[0]["metadata"]["category"] == "solution"

    def test_count(self, tmp_path: Path) -> None:
        store = VectorMemoryStore(
            dimension=64,
            index_path=str(tmp_path / "count.faiss"),
        )
        assert store.count() == 0
        store.add("entry one")
        assert store.count() == 1
        store.add("entry two")
        assert store.count() == 2

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = VectorMemoryStore(
            dimension=64,
            index_path=str(tmp_path / "nope.faiss"),
        )
        store.load()  # Should not raise
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Graceful degradation without FAISS
# ---------------------------------------------------------------------------


class TestGracefulWithoutFaiss:
    """Test that the store degrades gracefully when faiss is not importable."""

    def test_graceful_without_faiss(self, tmp_path: Path) -> None:
        """Simulate faiss not being available by patching the module flag."""
        with patch("missy.memory.vector_store._FAISS_AVAILABLE", False):
            store = VectorMemoryStore(
                dimension=64,
                index_path=str(tmp_path / "nofaiss.faiss"),
            )
            # Force index to None as it would be without faiss
            store._index = None

            # All methods should be no-ops
            store.add("test entry")
            assert store.count() == 0
            results = store.search("test")
            assert results == []
            store.save()  # Should not raise
            store.load()  # Should not raise
