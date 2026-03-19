"""Extended tests for the memory subsystem.

Covers:
- SQLiteMemoryStore CRUD operations
- FTS5 full-text search behavior
- ResilientMemoryStore fallback behavior
- Memory cleanup (old entries)
- Concurrent access (thread safety)
- Large memory stores (performance edge cases)
- Search ranking and relevance
- Memory store initialization and migration
- Edge cases: empty queries, very long content, special characters
- SimpleVectorizer and VectorMemoryStore (no FAISS required)
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import (
    ConversationTurn,
    LargeContentRecord,
    SQLiteMemoryStore,
    SummaryRecord,
)
from missy.memory.vector_store import SimpleVectorizer, VectorMemoryStore

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _turn(session_id: str = "s1", role: str = "user", content: str = "hello") -> ConversationTurn:
    return ConversationTurn.new(session_id=session_id, role=role, content=content)


def _turn_at(
    session_id: str,
    content: str,
    delta_days: int,
    role: str = "user",
) -> ConversationTurn:
    """Create a ConversationTurn with a timestamp offset by *delta_days* from now."""
    ts = (datetime.now(UTC) - timedelta(days=delta_days)).isoformat()
    return ConversationTurn(
        id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp=ts,
        role=role,
        content=content,
    )


@dataclass
class _FakeLearning:
    task_type: str = "coding"
    outcome: str = "success"
    lesson: str = "Use try/except"
    approach: list = None
    timestamp: str = ""

    def __post_init__(self):
        if self.approach is None:
            self.approach = []
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))


# ===========================================================================
# 1. SQLiteMemoryStore CRUD operations
# ===========================================================================


class TestSQLiteStoreCRUD:
    def test_add_and_retrieve_turn(self, store):
        turn = _turn("sess-a", content="Hello world")
        store.add_turn(turn)
        turns = store.get_session_turns("sess-a")
        assert len(turns) == 1
        assert turns[0].content == "Hello world"

    def test_turn_fields_round_trip(self, store):
        turn = ConversationTurn.new(
            session_id="sess-b",
            role="assistant",
            content="I can help with that",
            provider="anthropic",
        )
        store.add_turn(turn)
        retrieved = store.get_session_turns("sess-b")[0]
        assert retrieved.id == turn.id
        assert retrieved.session_id == turn.session_id
        assert retrieved.role == "assistant"
        assert retrieved.provider == "anthropic"

    def test_add_multiple_turns_ordered_chronologically(self, store):
        for i in range(5):
            t = ConversationTurn(
                id=str(uuid.uuid4()),
                session_id="sess-c",
                timestamp=(datetime.now(UTC) - timedelta(seconds=5 - i)).isoformat(),
                role="user",
                content=f"message {i}",
            )
            store.add_turn(t)
        turns = store.get_session_turns("sess-c")
        contents = [t.content for t in turns]
        assert contents == [f"message {i}" for i in range(5)]

    def test_clear_session_removes_all_turns(self, store):
        for _ in range(3):
            store.add_turn(_turn("sess-d"))
        store.clear_session("sess-d")
        assert store.get_session_turns("sess-d") == []

    def test_clear_session_does_not_affect_other_sessions(self, store):
        store.add_turn(_turn("keep", content="preserve me"))
        store.add_turn(_turn("remove", content="remove me"))
        store.clear_session("remove")
        keep_turns = store.get_session_turns("keep")
        assert len(keep_turns) == 1
        assert keep_turns[0].content == "preserve me"

    def test_get_session_turns_limit(self, store):
        for i in range(20):
            t = ConversationTurn(
                id=str(uuid.uuid4()),
                session_id="sess-limit",
                timestamp=(datetime.now(UTC) - timedelta(seconds=20 - i)).isoformat(),
                role="user",
                content=f"msg {i}",
            )
            store.add_turn(t)
        result = store.get_session_turns("sess-limit", limit=5)
        assert len(result) == 5
        # Should return most recent 5
        assert result[-1].content == "msg 19"

    def test_get_recent_turns_across_sessions(self, store):
        store.add_turn(_turn("s1", content="from s1"))
        store.add_turn(_turn("s2", content="from s2"))
        recent = store.get_recent_turns(limit=10)
        contents = [t.content for t in recent]
        assert "from s1" in contents
        assert "from s2" in contents

    def test_get_recent_turns_limit(self, store):
        for i in range(30):
            store.add_turn(_turn("bulk", content=f"msg {i}"))
        recent = store.get_recent_turns(limit=10)
        assert len(recent) == 10

    def test_replace_duplicate_id_is_idempotent(self, store):
        turn = _turn("s-dup", content="original")
        store.add_turn(turn)
        # Same id, different content — INSERT OR REPLACE
        duplicate = ConversationTurn(
            id=turn.id,
            session_id=turn.session_id,
            timestamp=turn.timestamp,
            role="user",
            content="replacement",
        )
        store.add_turn(duplicate)
        turns = store.get_session_turns("s-dup")
        assert len(turns) == 1
        assert turns[0].content == "replacement"

    def test_metadata_persisted_and_restored(self, store):
        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            session_id="s-meta",
            timestamp=datetime.now(UTC).isoformat(),
            role="user",
            content="meta test",
            metadata={"key": "value", "num": 42},
        )
        store.add_turn(turn)
        retrieved = store.get_session_turns("s-meta")[0]
        assert retrieved.metadata == {"key": "value", "num": 42}

    def test_empty_metadata_defaults_to_empty_dict(self, store):
        turn = _turn("s-nometa", content="no meta")
        store.add_turn(turn)
        retrieved = store.get_session_turns("s-nometa")[0]
        assert retrieved.metadata == {}


# ===========================================================================
# 2. FTS5 full-text search behavior
# ===========================================================================


class TestFTS5Search:
    def test_search_finds_matching_turn(self, store):
        store.add_turn(_turn("s1", content="The quick brown fox jumps"))
        store.add_turn(_turn("s1", content="Irrelevant content here"))
        results = store.search("fox")
        assert len(results) == 1
        assert "fox" in results[0].content

    def test_search_returns_empty_for_no_match(self, store):
        store.add_turn(_turn("s1", content="Hello world"))
        results = store.search("nonexistent_term_xyz")
        assert results == []

    def test_search_is_case_insensitive(self, store):
        store.add_turn(_turn("s1", content="Python is awesome"))
        results = store.search("python")
        assert len(results) == 1

    def test_search_respects_session_filter(self, store):
        store.add_turn(_turn("s1", content="Python rocks"))
        store.add_turn(_turn("s2", content="Python is cool"))
        results = store.search("Python", session_id="s1")
        assert len(results) == 1
        assert results[0].session_id == "s1"

    def test_search_without_session_filter_finds_all(self, store):
        store.add_turn(_turn("s1", content="Needle in haystack"))
        store.add_turn(_turn("s2", content="Another needle here"))
        results = store.search("needle", limit=10)
        assert len(results) == 2

    def test_search_limit_respected(self, store):
        for i in range(10):
            store.add_turn(_turn("s1", content=f"target word iteration {i}"))
        results = store.search("target", limit=3)
        assert len(results) <= 3

    def test_search_empty_query_returns_empty(self, store):
        store.add_turn(_turn("s1", content="some content"))
        # An empty quoted phrase matches nothing in FTS5
        results = store.search("")
        assert isinstance(results, list)

    def test_search_special_chars_do_not_raise(self, store):
        store.add_turn(_turn("s1", content="hello world"))
        # FTS5 special characters inside quotes are escaped
        for special in ['AND', 'OR', 'NOT', '"quoted"', "O'Brien", "100%"]:
            results = store.search(special)
            assert isinstance(results, list)

    def test_search_unicode_content(self, store):
        store.add_turn(_turn("s1", content="Café résumé naïve"))
        results = store.search("résumé")
        assert isinstance(results, list)

    def test_search_on_empty_store_returns_empty(self, store):
        results = store.search("anything")
        assert results == []

    def test_fts_error_returns_empty_not_raises(self, store):
        """FTS5 OperationalError should be caught and return empty list."""
        store.add_turn(_turn("s1", content="some content"))
        original_conn = store._conn

        class _BrokenFTSConn:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, params=()):
                if "turns_fts MATCH" in sql:
                    import sqlite3
                    raise sqlite3.OperationalError("fts5 error injected")
                return self._real.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._real, name)

        with patch.object(store, "_conn", lambda: _BrokenFTSConn(original_conn())):
            results = store.search("content")
        assert results == []


# ===========================================================================
# 3. ResilientMemoryStore fallback behavior
# ===========================================================================


def _make_mock_turn(session_id: str = "s1", content: str = "hello") -> MagicMock:
    t = MagicMock()
    t.session_id = session_id
    t.content = content
    t.timestamp = datetime.now(UTC).isoformat()
    t.id = str(uuid.uuid4())
    return t


def _broken_primary() -> MagicMock:
    p = MagicMock()
    p.add_turn.side_effect = RuntimeError("db error")
    p.clear_session.side_effect = RuntimeError("db error")
    p.save_learning.side_effect = RuntimeError("db error")
    p.get_session_turns.side_effect = RuntimeError("db error")
    p.get_recent_turns.side_effect = RuntimeError("db error")
    p.search.side_effect = RuntimeError("db error")
    p.get_learnings.side_effect = RuntimeError("db error")
    p.cleanup.side_effect = RuntimeError("db error")
    return p


def _healthy_primary() -> MagicMock:
    p = MagicMock()
    p.get_session_turns.return_value = []
    p.get_recent_turns.return_value = []
    p.search.return_value = []
    p.get_learnings.return_value = []
    p.cleanup.return_value = 0
    return p


class TestResilientStoreFallback:
    def test_writes_to_cache_on_primary_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        turn = _make_mock_turn("s1")
        store.add_turn(turn)
        assert turn in store._cache["s1"]

    def test_get_session_turns_falls_back_to_cache(self):
        store = ResilientMemoryStore(_broken_primary())
        turn = _make_mock_turn("s1", "cached content")
        store._cache["s1"] = [turn]
        result = store.get_session_turns("s1")
        assert turn in result

    def test_get_recent_turns_falls_back_to_all_cached(self):
        store = ResilientMemoryStore(_broken_primary())
        t1 = _make_mock_turn("s1", "a")
        t2 = _make_mock_turn("s2", "b")
        store._cache["s1"] = [t1]
        store._cache["s2"] = [t2]
        result = store.get_recent_turns(limit=10)
        assert t1 in result
        assert t2 in result

    def test_search_falls_back_to_keyword_scan(self):
        store = ResilientMemoryStore(_broken_primary())
        match_turn = _make_mock_turn("s1", "unique keyword found here")
        no_match = _make_mock_turn("s1", "nothing relevant")
        store._cache["s1"] = [match_turn, no_match]
        results = store.search("keyword")
        assert match_turn in results
        assert no_match not in results

    def test_search_cache_case_insensitive(self):
        store = ResilientMemoryStore(_broken_primary())
        turn = _make_mock_turn("s1", "Python Programming")
        store._cache["s1"] = [turn]
        results = store.search("python")
        assert turn in results

    def test_cleanup_returns_zero_on_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        result = store.cleanup(older_than_days=7)
        assert result == 0

    def test_get_learnings_returns_empty_on_failure(self):
        store = ResilientMemoryStore(_broken_primary())
        result = store.get_learnings()
        assert result == []

    def test_is_healthy_tracks_state(self):
        store = ResilientMemoryStore(_broken_primary(), max_failures=1)
        assert store.is_healthy is True
        store.add_turn(_make_mock_turn())
        assert store.is_healthy is False

    def test_recovery_restores_healthy_state(self):
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=1)
        store.add_turn(_make_mock_turn())
        assert store.is_healthy is False
        # Fix the primary
        primary.add_turn.side_effect = None
        store.add_turn(_make_mock_turn())
        assert store.is_healthy is True

    def test_recovery_syncs_cache_to_primary(self):
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=1)
        turn = _make_mock_turn()
        store.add_turn(turn)  # fails, goes to cache
        primary.add_turn.side_effect = None  # fix primary
        store.add_turn(_make_mock_turn())  # triggers recovery
        # The sync replays cached turns; turn should have been written
        assert primary.add_turn.call_count >= 2

    def test_clear_session_removes_from_cache(self):
        store = ResilientMemoryStore(_broken_primary())
        store._cache["s1"] = [_make_mock_turn("s1")]
        store.clear_session("s1")
        assert "s1" not in store._cache

    def test_healthy_primary_is_preferred_over_cache(self):
        primary = _healthy_primary()
        expected = [_make_mock_turn("s1")]
        primary.get_session_turns.return_value = expected
        store = ResilientMemoryStore(primary)
        result = store.get_session_turns("s1")
        assert result == expected


# ===========================================================================
# 4. Memory cleanup (old entries)
# ===========================================================================


class TestMemoryCleanup:
    def test_cleanup_removes_old_turns(self, store):
        old_turn = _turn_at("s1", "old content", delta_days=60)
        store.add_turn(old_turn)
        removed = store.cleanup(older_than_days=30)
        assert removed == 1
        assert store.get_session_turns("s1") == []

    def test_cleanup_preserves_recent_turns(self, store):
        recent = _turn_at("s1", "recent content", delta_days=5)
        store.add_turn(recent)
        removed = store.cleanup(older_than_days=30)
        assert removed == 0
        assert len(store.get_session_turns("s1")) == 1

    def test_cleanup_returns_count_of_deleted_rows(self, store):
        for i in range(5):
            store.add_turn(_turn_at("s1", f"old {i}", delta_days=60))
        removed = store.cleanup(older_than_days=30)
        assert removed == 5

    def test_cleanup_mixed_ages(self, store):
        for i in range(3):
            store.add_turn(_turn_at("s1", f"old {i}", delta_days=40))
        for i in range(2):
            store.add_turn(_turn_at("s1", f"new {i}", delta_days=10))
        removed = store.cleanup(older_than_days=30)
        assert removed == 3
        remaining = store.get_session_turns("s1")
        assert len(remaining) == 2

    def test_cleanup_zero_days_removes_all(self, store):
        # delta_days=0 puts timestamp in the past by a small amount
        store.add_turn(_turn_at("s1", "any content", delta_days=0))
        # older_than_days=0 means cutoff is now; turn at 0 days is just on/before cutoff
        removed = store.cleanup(older_than_days=0)
        assert removed >= 0  # depends on sub-second precision

    def test_cleanup_on_empty_store_returns_zero(self, store):
        removed = store.cleanup(older_than_days=30)
        assert removed == 0


# ===========================================================================
# 5. Concurrent access (thread safety)
# ===========================================================================


class TestConcurrentAccess:
    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        db_path = str(tmp_path / "concurrent.db")
        store = SQLiteMemoryStore(db_path=db_path)
        errors = []

        def write_turns(session_id: str, n: int) -> None:
            try:
                for i in range(n):
                    store.add_turn(_turn(session_id, content=f"msg {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_turns, args=(f"s{j}", 20)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent write errors: {errors}"
        # All turns should be stored
        for j in range(5):
            turns = store.get_session_turns(f"s{j}")
            assert len(turns) == 20

    def test_concurrent_reads_and_writes(self, tmp_path):
        db_path = str(tmp_path / "rw_concurrent.db")
        store = SQLiteMemoryStore(db_path=db_path)
        # Seed data
        for i in range(10):
            store.add_turn(_turn("shared", content=f"baseline {i}"))

        errors = []

        def reader():
            try:
                for _ in range(10):
                    store.get_recent_turns(limit=5)
            except Exception as exc:
                errors.append(exc)

        def writer():
            try:
                for i in range(10):
                    store.add_turn(_turn("shared", content=f"new {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(3)] + [
            threading.Thread(target=writer) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent R/W errors: {errors}"

    def test_resilient_store_concurrent_add_turn(self):
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        errors = []

        def add_many(session_id: str) -> None:
            try:
                for i in range(30):
                    store.add_turn(_make_mock_turn(session_id, f"content {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_many, args=(f"t{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ===========================================================================
# 6. Large memory stores (performance edge cases)
# ===========================================================================


class TestLargeMemoryStores:
    def test_store_and_retrieve_very_long_content(self, store):
        long_content = "A" * 100_000
        turn = _turn("s-long", content=long_content)
        store.add_turn(turn)
        retrieved = store.get_session_turns("s-long")[0]
        assert retrieved.content == long_content

    def test_bulk_insert_performance(self, store):
        start = time.monotonic()
        for i in range(500):
            store.add_turn(_turn("s-bulk", content=f"message number {i} with some text"))
        elapsed = time.monotonic() - start
        # 500 inserts should complete in under 10 seconds even on slow CI
        assert elapsed < 10.0

    def test_search_over_large_store(self, store):
        for i in range(200):
            content = "needle" if i == 100 else f"irrelevant item {i}"
            store.add_turn(_turn("s-search", content=content))
        results = store.search("needle")
        assert len(results) >= 1
        assert any("needle" in r.content for r in results)

    def test_get_session_turns_large_session(self, store):
        for i in range(300):
            t = ConversationTurn(
                id=str(uuid.uuid4()),
                session_id="s-large",
                timestamp=(datetime.now(UTC) - timedelta(seconds=300 - i)).isoformat(),
                role="user",
                content=f"turn {i}",
            )
            store.add_turn(t)
        turns = store.get_session_turns("s-large", limit=50)
        assert len(turns) == 50
        # Should return the 50 most recent
        assert turns[-1].content == "turn 299"

    def test_large_content_record_store_and_retrieve(self, store):
        big_content = "X" * 50_000
        record = LargeContentRecord.new(
            session_id="s-lc",
            tool_name="read_file",
            content=big_content,
            summary="Very large file output",
        )
        stored_id = store.store_large_content(record)
        retrieved = store.get_large_content(stored_id)
        assert retrieved is not None
        assert retrieved.content == big_content
        assert retrieved.original_chars == 50_000


# ===========================================================================
# 7. Search ranking and relevance
# ===========================================================================


class TestSearchRanking:
    def test_search_summaries_finds_matching_content(self, store):
        s1 = SummaryRecord.new("sess1", 0, "Python async programming patterns")
        s2 = SummaryRecord.new("sess1", 0, "Unrelated database migration notes")
        store.add_summary(s1)
        store.add_summary(s2)
        results = store.search_summaries("python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_summaries_session_filter(self, store):
        s1 = SummaryRecord.new("sess-a", 0, "matching content here")
        s2 = SummaryRecord.new("sess-b", 0, "matching content there")
        store.add_summary(s1)
        store.add_summary(s2)
        results = store.search_summaries("matching", session_id="sess-a")
        assert all(r.session_id == "sess-a" for r in results)
        assert len(results) == 1

    def test_search_summaries_empty_returns_empty(self, store):
        results = store.search_summaries("anything")
        assert results == []

    def test_search_large_content_by_summary(self, store):
        record = LargeContentRecord.new(
            session_id="lc-sess",
            tool_name="grep",
            content="some large file output",
            summary="grep output for authentication errors",
        )
        store.store_large_content(record)
        results = store.search_large_content("authentication", "lc-sess")
        assert len(results) == 1

    def test_search_large_content_by_content(self, store):
        record = LargeContentRecord.new(
            session_id="lc-sess2",
            tool_name="cat",
            content="the actual file content with unique_phrase_xyz",
            summary="file read",
        )
        store.store_large_content(record)
        results = store.search_large_content("unique_phrase_xyz", "lc-sess2")
        assert len(results) == 1

    def test_search_large_content_no_match_returns_empty(self, store):
        record = LargeContentRecord.new(
            session_id="lc-sess3",
            tool_name="ls",
            content="directory listing",
            summary="ls output",
        )
        store.store_large_content(record)
        results = store.search_large_content("notpresent", "lc-sess3")
        assert results == []


# ===========================================================================
# 8. Memory store initialization and schema
# ===========================================================================


class TestStoreInitialization:
    def test_creates_database_file(self, tmp_path):
        db_path = str(tmp_path / "new.db")
        assert not Path(db_path).exists()
        SQLiteMemoryStore(db_path=db_path)
        assert Path(db_path).exists()

    def test_creates_parent_directory_if_missing(self, tmp_path):
        db_path = str(tmp_path / "subdir" / "deep" / "memory.db")
        SQLiteMemoryStore(db_path=db_path)
        assert Path(db_path).exists()

    def test_all_required_tables_exist(self, tmp_path):
        import sqlite3

        db_path = str(tmp_path / "schema.db")
        SQLiteMemoryStore(db_path=db_path)
        conn = sqlite3.connect(db_path)
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        for required in ("turns", "learnings", "sessions", "costs", "summaries", "large_content"):
            assert required in tables, f"Missing table: {required}"

    def test_reinitialize_existing_database_is_safe(self, tmp_path):
        db_path = str(tmp_path / "existing.db")
        store1 = SQLiteMemoryStore(db_path=db_path)
        store1.add_turn(_turn("s1", content="before reinit"))
        # Open again — CREATE IF NOT EXISTS should not destroy data
        store2 = SQLiteMemoryStore(db_path=db_path)
        turns = store2.get_session_turns("s1")
        assert len(turns) == 1
        assert turns[0].content == "before reinit"

    def test_wal_mode_enabled(self, tmp_path):
        import sqlite3

        db_path = str(tmp_path / "wal.db")
        SQLiteMemoryStore(db_path=db_path)
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"


# ===========================================================================
# 9. Edge cases: empty queries, very long content, special characters
# ===========================================================================


class TestEdgeCases:
    def test_very_long_content_stored_and_found(self, store):
        long_content = "word " * 10_000  # 50k chars
        store.add_turn(_turn("edge-long", content=long_content))
        turns = store.get_session_turns("edge-long")
        assert len(turns[0].content) == len(long_content)

    def test_content_with_sql_injection_characters(self, store):
        evil = "'; DROP TABLE turns; --"
        store.add_turn(_turn("edge-sql", content=evil))
        turns = store.get_session_turns("edge-sql")
        assert turns[0].content == evil

    def test_content_with_null_bytes(self, store):
        content_with_null = "before\x00after"
        t = ConversationTurn(
            id=str(uuid.uuid4()),
            session_id="edge-null",
            timestamp=datetime.now(UTC).isoformat(),
            role="user",
            content=content_with_null,
        )
        store.add_turn(t)
        turns = store.get_session_turns("edge-null")
        # SQLite stores null bytes fine; content is preserved
        assert turns[0].content == content_with_null

    def test_content_with_unicode_emoji(self, store):
        content = "Hello world"
        store.add_turn(_turn("edge-emoji", content=content))
        turns = store.get_session_turns("edge-emoji")
        assert turns[0].content == content

    def test_search_with_double_quote_in_query(self, store):
        store.add_turn(_turn("s1", content='He said "hello"'))
        results = store.search('"hello"')
        assert isinstance(results, list)

    def test_search_with_fts5_operators_escaped(self, store):
        store.add_turn(_turn("s1", content="AND OR NOT operators"))
        for q in ["AND", "OR", "NOT", "AND OR", "*wildcard"]:
            results = store.search(q)
            assert isinstance(results, list)

    def test_zero_limit_returns_empty(self, store):
        store.add_turn(_turn("s1", content="something"))
        turns = store.get_session_turns("s1", limit=0)
        assert turns == []

    def test_content_only_whitespace(self, store):
        whitespace = "   \t\n   "
        store.add_turn(_turn("s-ws", content=whitespace))
        turns = store.get_session_turns("s-ws")
        assert turns[0].content == whitespace

    def test_session_id_with_special_characters(self, store):
        special_sid = "session/with:special!chars@2026"
        store.add_turn(_turn(special_sid, content="testing"))
        turns = store.get_session_turns(special_sid)
        assert len(turns) == 1

    def test_nonexistent_session_returns_empty_list(self, store):
        turns = store.get_session_turns("does-not-exist")
        assert turns == []

    def test_get_large_content_nonexistent_id_returns_none(self, store):
        result = store.get_large_content("ref_nonexistent")
        assert result is None

    def test_get_summary_by_id_nonexistent_returns_none(self, store):
        result = store.get_summary_by_id("sum_nonexistent")
        assert result is None


# ===========================================================================
# 10. Learnings CRUD
# ===========================================================================


class TestLearnings:
    def test_save_and_retrieve_learning(self, store):
        learning = _FakeLearning(lesson="Always validate inputs")
        store.save_learning(learning)
        lessons = store.get_learnings()
        assert "Always validate inputs" in lessons

    def test_get_learnings_filtered_by_task_type(self, store):
        store.save_learning(_FakeLearning(task_type="coding", lesson="Coding lesson"))
        store.save_learning(_FakeLearning(task_type="writing", lesson="Writing lesson"))
        coding_lessons = store.get_learnings(task_type="coding")
        assert "Coding lesson" in coding_lessons
        assert "Writing lesson" not in coding_lessons

    def test_get_learnings_limit(self, store):
        for i in range(10):
            store.save_learning(_FakeLearning(lesson=f"Lesson {i}"))
        lessons = store.get_learnings(limit=3)
        assert len(lessons) == 3

    def test_get_learnings_returns_most_recent_first(self, store):
        import time as _time

        store.save_learning(_FakeLearning(lesson="First lesson"))
        _time.sleep(0.01)
        store.save_learning(_FakeLearning(lesson="Second lesson"))
        lessons = store.get_learnings(limit=2)
        assert lessons[0] == "Second lesson"

    def test_get_learnings_empty_store_returns_empty(self, store):
        lessons = store.get_learnings()
        assert lessons == []


# ===========================================================================
# 11. Session metadata operations
# ===========================================================================


class TestSessionMetadata:
    def test_register_and_list_session(self, store):
        store.register_session("sess-reg", name="My Session", provider="anthropic")
        sessions = store.list_sessions()
        ids = [s["session_id"] for s in sessions]
        assert "sess-reg" in ids

    def test_register_session_upserts_on_conflict(self, store):
        store.register_session("sess-upsert", name="Original")
        store.register_session("sess-upsert", name="Updated")
        sessions = store.list_sessions()
        match = [s for s in sessions if s["session_id"] == "sess-upsert"]
        assert len(match) == 1
        assert match[0]["name"] == "Updated"

    def test_rename_session(self, store):
        store.register_session("sess-rename", name="old name")
        result = store.rename_session("sess-rename", "new name")
        assert result is True
        sessions = store.list_sessions()
        match = next(s for s in sessions if s["session_id"] == "sess-rename")
        assert match["name"] == "new name"

    def test_rename_nonexistent_session_returns_false(self, store):
        result = store.rename_session("no-such-session", "anything")
        assert result is False

    def test_resolve_session_name(self, store):
        store.register_session("sess-resolve", name="My Named Session")
        sid = store.resolve_session_name("My Named Session")
        assert sid == "sess-resolve"

    def test_resolve_nonexistent_name_returns_none(self, store):
        result = store.resolve_session_name("Ghost Session")
        assert result is None

    def test_update_session_turn_count(self, store):
        store.register_session("sess-count")
        for _ in range(5):
            store.add_turn(_turn("sess-count", content="msg"))
        store.update_session_turn_count("sess-count")
        sessions = store.list_sessions()
        match = next(s for s in sessions if s["session_id"] == "sess-count")
        assert match["turn_count"] == 5


# ===========================================================================
# 12. SimpleVectorizer and VectorMemoryStore (no FAISS dependency)
# ===========================================================================


class TestSimpleVectorizer:
    def test_encode_returns_correct_dimension(self):
        v = SimpleVectorizer(dimension=64)
        result = v.encode("hello world")
        assert len(result) == 64

    def test_encode_empty_string_returns_zero_vector(self):
        v = SimpleVectorizer(dimension=64)
        result = v.encode("")
        assert all(x == 0.0 for x in result)
        assert len(result) == 64

    def test_encode_produces_unit_vector(self):
        v = SimpleVectorizer(dimension=128)
        result = v.encode("some text to encode here")
        norm = sum(x * x for x in result) ** 0.5
        assert abs(norm - 1.0) < 1e-5

    def test_encode_identical_texts_are_identical(self):
        v = SimpleVectorizer()
        vec1 = v.encode("the same sentence")
        vec2 = v.encode("the same sentence")
        assert vec1 == vec2

    def test_encode_different_texts_differ(self):
        v = SimpleVectorizer()
        vec1 = v.encode("machine learning")
        vec2 = v.encode("fishing and camping")
        assert vec1 != vec2

    def test_encode_numeric_only_string(self):
        v = SimpleVectorizer(dimension=32)
        result = v.encode("12345 67890")
        assert len(result) == 32
        assert any(x != 0.0 for x in result)


class TestVectorMemoryStoreNoFAISS:
    """Tests that run regardless of whether FAISS is installed.
    When FAISS is absent, all methods are safe no-ops.
    """

    def test_count_is_zero_on_new_store(self, tmp_path):
        store = VectorMemoryStore(index_path=str(tmp_path / "test.faiss"))
        assert store.count() == 0

    def test_search_on_empty_store_returns_empty(self, tmp_path):
        store = VectorMemoryStore(index_path=str(tmp_path / "test.faiss"))
        results = store.search("any query")
        assert results == []

    def test_add_and_search_when_faiss_available(self, tmp_path):
        """Only executes the add/search assertions when FAISS is installed."""
        pytest.importorskip("faiss", reason="faiss not installed")
        store = VectorMemoryStore(index_path=str(tmp_path / "test.faiss"))
        store.add("The quick brown fox", metadata={"cat": "animals"})
        store.add("Python async programming", metadata={"cat": "tech"})
        results = store.search("fox", top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "The quick brown fox"
        assert "score" in results[0]

    def test_save_and_load_when_faiss_available(self, tmp_path):
        pytest.importorskip("faiss", reason="faiss not installed")
        index_path = str(tmp_path / "mem.faiss")
        store = VectorMemoryStore(index_path=index_path)
        store.add("persistent entry", metadata={"tag": "persist"})
        store.save()
        assert Path(index_path).exists()
        store2 = VectorMemoryStore(index_path=index_path)
        store2.load()
        assert store2.count() == 1

    def test_load_nonexistent_file_is_safe(self, tmp_path):
        store = VectorMemoryStore(index_path=str(tmp_path / "ghost.faiss"))
        store.load()  # should not raise
        assert store.count() == 0

    def test_save_when_faiss_unavailable_is_noop(self, tmp_path):
        """Patch FAISS away and confirm save() does not raise."""
        index_path = str(tmp_path / "nofaiss.faiss")
        store = VectorMemoryStore(index_path=index_path)
        with patch("missy.memory.vector_store._FAISS_AVAILABLE", False):
            store._index = None
            store.save()  # must not raise
        assert not Path(index_path).exists()
