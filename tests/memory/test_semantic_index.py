"""Tests for F12 — semantic conversation memory index."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from missy.memory.semantic_index import ConversationSemanticIndex
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore

faiss = pytest.importorskip("faiss")  # F12 requires the [vector] extra


def _index() -> ConversationSemanticIndex:
    d = tempfile.mkdtemp()
    return ConversationSemanticIndex(index_path=os.path.join(d, "sem.faiss"))


class TestIndexTurn:
    def test_available(self) -> None:
        assert _index().available is True

    def test_index_and_dedup(self) -> None:
        idx = _index()
        t = ConversationTurn.new("s1", "user", "hiking in the mountains")
        assert idx.index_turn(t) is True
        assert idx.index_turn(t) is False  # same id -> deduped

    def test_blank_content_skipped(self) -> None:
        idx = _index()
        assert idx.index_turn(ConversationTurn.new("s1", "user", "   ")) is False

    def test_search_finds_indexed(self) -> None:
        idx = _index()
        idx.index_turn(ConversationTurn.new("s1", "user", "my favorite food is pizza"))
        idx.index_turn(ConversationTurn.new("s1", "user", "the weather is sunny today"))
        results = idx.search("pizza", top_k=2)
        assert results
        assert any("pizza" in r["text"] for r in results)

    def test_session_filter(self) -> None:
        idx = _index()
        idx.index_turn(ConversationTurn.new("s1", "user", "apples and oranges"))
        idx.index_turn(ConversationTurn.new("s2", "user", "apples and bananas"))
        results = idx.search("apples", top_k=10, session_id="s2")
        assert results
        assert all((r["metadata"].get("session_id") == "s2") for r in results)

    def test_empty_query_returns_empty(self) -> None:
        idx = _index()
        idx.index_turn(ConversationTurn.new("s1", "user", "hello world"))
        assert idx.search("  ", top_k=5) == []


class TestReindex:
    def test_reindex_from_store(self, tmp_path) -> None:
        store = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"))
        for i in range(5):
            store.add_turn(ConversationTurn.new("s1", "user", f"message number {i}"))
        idx = _index()
        count = idx.reindex(store)
        assert count == 5
        # Reindexing again de-dups (same turn ids).
        assert idx.reindex(store) == 0

    def test_reindex_session_filter(self, tmp_path) -> None:
        store = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"))
        store.add_turn(ConversationTurn.new("s1", "user", "one"))
        store.add_turn(ConversationTurn.new("s2", "user", "two"))
        idx = _index()
        assert idx.reindex(store, session_id="s1") == 1


class TestUnavailableBackend:
    def test_no_backend_is_noop(self) -> None:
        idx = ConversationSemanticIndex.__new__(ConversationSemanticIndex)
        idx._store = None  # simulate faiss unavailable
        assert idx.available is False
        assert idx.index_turn(ConversationTurn.new("s", "user", "x")) is False
        assert idx.search("x") == []
        assert idx.reindex(MagicMock()) == 0


class TestSleeptimeWiring:
    def test_worker_indexes_batch(self) -> None:
        from missy.agent.sleeptime import SleeptimeWorker

        sem = MagicMock()
        sem.index_turn.return_value = True
        worker = SleeptimeWorker(memory_store=None, semantic_index=sem)
        batch = [ConversationTurn.new("s1", "user", f"m{i}") for i in range(3)]
        assert worker._index_semantic(batch) == 3
        assert sem.index_turn.call_count == 3

    def test_worker_no_index_is_noop(self) -> None:
        from missy.agent.sleeptime import SleeptimeWorker

        worker = SleeptimeWorker(memory_store=None, semantic_index=None)
        assert worker._index_semantic([ConversationTurn.new("s", "user", "x")]) == 0

    def test_worker_index_failure_isolated(self) -> None:
        from missy.agent.sleeptime import SleeptimeWorker

        sem = MagicMock()
        sem.index_turn.side_effect = RuntimeError("index down")
        worker = SleeptimeWorker(memory_store=None, semantic_index=sem)
        # Must swallow and return 0, not raise.
        assert worker._index_semantic([ConversationTurn.new("s", "user", "x")]) == 0
