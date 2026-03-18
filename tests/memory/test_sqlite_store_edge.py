"""Tests for sqlite_store.py — targeting remaining uncovered lines.

Lines 848 (get_summaries without depth), 883 (empty summary_ids guard),
940-942 (FTS5 operational error).
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from missy.memory.sqlite_store import SQLiteMemoryStore, SummaryRecord


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    return SQLiteMemoryStore(db_path)


def _make_summary(session_id="sess1", content="Test", depth=1, **kwargs):
    import uuid

    return SummaryRecord(
        id=f"sum_{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        depth=depth,
        content=content,
        token_estimate=kwargs.get("token_estimate", 50),
        source_turn_ids=kwargs.get("source_turn_ids", []),
        source_summary_ids=kwargs.get("source_summary_ids", []),
    )


class TestGetSummariesWithoutDepth:
    """Line 848: get_summaries called without depth argument."""

    def test_get_summaries_no_depth(self, store):
        store.add_summary(_make_summary(content="Test summary", source_turn_ids=["t1"]))
        results = store.get_summaries("sess1")
        assert len(results) == 1
        assert results[0].content == "Test summary"

    def test_get_summaries_mixed_depths_no_filter(self, store):
        store.add_summary(_make_summary(content="D1", depth=1))
        store.add_summary(_make_summary(content="D2", depth=2))
        results = store.get_summaries("sess1")
        assert len(results) == 2


class TestMarkSummaryCompactedEmpty:
    """Line 883: empty summary_ids returns early."""

    def test_empty_ids_noop(self, store):
        store.mark_summary_compacted([], "parent_1")


class _FailingConn:
    """Wrapper that raises OperationalError on FTS queries."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, sql, params=()):
        if "summaries_fts MATCH" in sql:
            raise sqlite3.OperationalError("fts5 syntax error")
        return self._real.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestSearchSummariesFTSError:
    """Lines 940-942: FTS5 operational error returns empty list."""

    def test_fts_error_returns_empty(self, store):
        store.add_summary(_make_summary(content="Some content"))

        original_conn = store._conn

        def patched_conn():
            return _FailingConn(original_conn())

        with patch.object(store, "_conn", patched_conn):
            results = store.search_summaries("test")
        assert results == []

    def test_fts_error_with_session_returns_empty(self, store):
        store.add_summary(_make_summary(content="Hello test"))

        original_conn = store._conn

        def patched_conn():
            return _FailingConn(original_conn())

        with patch.object(store, "_conn", patched_conn):
            results = store.search_summaries("test", session_id="sess1")
        assert results == []
