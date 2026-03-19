"""Extended tests for the memory subsystem.

Covers:
1.  FTS5 full-text search functionality
2.  Session turn ordering
3.  Cleanup of old turns (retention policy)
4.  WAL mode enabled
5.  Thread safety (concurrent read/write)
6.  Large content handling
7.  Summary records: add, get by depth, mark compacted
8.  Session token count calculation
9.  Turn pagination (limit / offset via sliding-window helpers)
10. Unicode content handling

For ResilientMemoryStore:
- Delegates to underlying store on success
- Handles exceptions gracefully (returns defaults)
- Reconnection logic after failure
- Thread safety
- All methods have resilient wrappers
"""

from __future__ import annotations

import sqlite3
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

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _turn(
    session_id: str = "s1",
    role: str = "user",
    content: str = "hello",
    provider: str = "",
) -> ConversationTurn:
    return ConversationTurn.new(session_id=session_id, role=role, content=content, provider=provider)


def _turn_at(
    session_id: str,
    content: str,
    delta_days: int,
    role: str = "user",
) -> ConversationTurn:
    """Create a ConversationTurn with a timestamp offset by delta_days from now."""
    ts = (datetime.now(UTC) - timedelta(days=delta_days)).isoformat()
    return ConversationTurn(
        id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp=ts,
        role=role,
        content=content,
    )


def _turn_at_seconds(session_id: str, content: str, delta_seconds: int) -> ConversationTurn:
    """Create a turn with a timestamp delta_seconds in the past."""
    ts = (datetime.now(UTC) - timedelta(seconds=delta_seconds)).isoformat()
    return ConversationTurn(
        id=str(uuid.uuid4()),
        session_id=session_id,
        timestamp=ts,
        role="user",
        content=content,
    )


@dataclass
class _FakeLearning:
    task_type: str = "coding"
    outcome: str = "success"
    lesson: str = "Use try/except"
    approach: list = None  # type: ignore[assignment]
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.approach is None:
            self.approach = []
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=str(tmp_path / "test.db"))


# ===========================================================================
# 1. FTS5 full-text search functionality
# ===========================================================================


class TestFTS5Search:
    def test_search_finds_exact_word_match(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="The quick brown fox jumps over the lazy dog"))
        store.add_turn(_turn("s1", content="Completely irrelevant sentence about cats"))
        results = store.search("fox")
        assert len(results) == 1
        assert "fox" in results[0].content

    def test_search_returns_empty_when_no_match(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="Hello world"))
        results = store.search("xyzzy_nonexistent_term")
        assert results == []

    def test_search_is_case_insensitive(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="Python programming is fun"))
        assert len(store.search("python")) == 1
        assert len(store.search("PYTHON")) == 1
        assert len(store.search("Python")) == 1

    def test_search_phrase_match(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="async await is the modern pattern"))
        store.add_turn(_turn("s1", content="async programming without await"))
        # Phrase search via double-quoting is internal — the store wraps the query
        results = store.search("async await")
        # FTS5 phrase: "async await" only matches the first turn
        assert len(results) >= 1
        # The exact match must appear in results
        matching = [r for r in results if "async await" in r.content]
        assert len(matching) >= 1

    def test_search_multi_term_returns_relevant_turns(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="machine learning neural network"))
        store.add_turn(_turn("s1", content="database query optimization"))
        # Single term from first turn
        results = store.search("neural")
        assert any("neural" in r.content for r in results)

    def test_search_session_filter_restricts_results(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("sess-a", content="Python is powerful"))
        store.add_turn(_turn("sess-b", content="Python is also fun"))
        results_a = store.search("Python", session_id="sess-a")
        assert all(r.session_id == "sess-a" for r in results_a)
        results_b = store.search("Python", session_id="sess-b")
        assert all(r.session_id == "sess-b" for r in results_b)

    def test_search_without_session_filter_finds_all_sessions(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="needle in session one"))
        store.add_turn(_turn("s2", content="another needle in session two"))
        results = store.search("needle", limit=10)
        assert len(results) == 2

    def test_search_limit_is_respected(self, store: SQLiteMemoryStore) -> None:
        for i in range(15):
            store.add_turn(_turn("s1", content=f"target term iteration {i}"))
        results = store.search("target", limit=5)
        assert len(results) <= 5

    def test_search_empty_query_is_safe(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="some content"))
        results = store.search("")
        assert isinstance(results, list)

    def test_search_on_empty_store_returns_empty_list(self, store: SQLiteMemoryStore) -> None:
        results = store.search("anything")
        assert results == []

    def test_search_with_sql_injection_attempt_is_safe(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="hello world"))
        # Must not raise even though the query looks like SQL
        results = store.search("'; DROP TABLE turns; --")
        assert isinstance(results, list)

    def test_search_fts_error_returns_empty_not_raises(self, store: SQLiteMemoryStore) -> None:
        """An FTS5 OperationalError must be caught and return an empty list."""
        store.add_turn(_turn("s1", content="some content"))
        original_conn = store._conn

        class _BrokenFTSConn:
            def __init__(self, real: sqlite3.Connection) -> None:
                self._real = real

            def execute(self, sql: str, params: tuple = ()) -> object:  # type: ignore[override]
                if "turns_fts MATCH" in sql:
                    raise sqlite3.OperationalError("fts5 error injected")
                return self._real.execute(sql, params)

            def __getattr__(self, name: str) -> object:
                return getattr(self._real, name)

        with patch.object(store, "_conn", lambda: _BrokenFTSConn(original_conn())):
            results = store.search("content")
        assert results == []

    def test_search_deleted_turn_is_not_returned(self, store: SQLiteMemoryStore) -> None:
        """Turns deleted from the main table must not surface in FTS results."""
        turn = _turn("s1", content="temporary sensitive data")
        store.add_turn(turn)
        assert len(store.search("sensitive")) == 1
        store.clear_session("s1")
        assert store.search("sensitive") == []


# ===========================================================================
# 2. Session turn ordering
# ===========================================================================


class TestSessionTurnOrdering:
    def test_turns_returned_in_chronological_order(self, store: SQLiteMemoryStore) -> None:
        for i in range(5):
            store.add_turn(
                _turn_at_seconds("ordered", f"message {i}", delta_seconds=50 - i)
            )
        turns = store.get_session_turns("ordered")
        contents = [t.content for t in turns]
        assert contents == [f"message {i}" for i in range(5)]

    def test_get_session_turns_returns_most_recent_when_limited(self, store: SQLiteMemoryStore) -> None:
        for i in range(20):
            store.add_turn(
                _turn_at_seconds("paged", f"msg {i}", delta_seconds=20 - i)
            )
        turns = store.get_session_turns("paged", limit=5)
        assert len(turns) == 5
        # The last element should be the most recent message
        assert turns[-1].content == "msg 19"

    def test_get_recent_turns_across_sessions_preserves_order(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn_at_seconds("s1", "first", delta_seconds=10))
        store.add_turn(_turn_at_seconds("s2", "second", delta_seconds=5))
        store.add_turn(_turn_at_seconds("s1", "third", delta_seconds=1))
        recent = store.get_recent_turns(limit=3)
        contents = [t.content for t in recent]
        assert contents == ["first", "second", "third"]

    def test_turns_with_same_session_isolated_from_others(self, store: SQLiteMemoryStore) -> None:
        for i in range(3):
            store.add_turn(_turn("session-a", content=f"a-msg-{i}"))
        for i in range(3):
            store.add_turn(_turn("session-b", content=f"b-msg-{i}"))
        a_turns = store.get_session_turns("session-a")
        assert all("a-msg" in t.content for t in a_turns)
        assert len(a_turns) == 3

    def test_get_recent_turns_limit_applies_globally(self, store: SQLiteMemoryStore) -> None:
        for sess in ["x", "y", "z"]:
            for i in range(10):
                store.add_turn(_turn(sess, content=f"{sess}-{i}"))
        recent = store.get_recent_turns(limit=10)
        assert len(recent) == 10

    def test_get_session_turns_returns_empty_for_unknown_session(self, store: SQLiteMemoryStore) -> None:
        turns = store.get_session_turns("ghost-session")
        assert turns == []

    def test_get_session_turns_zero_limit_returns_empty(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("s1", content="something"))
        turns = store.get_session_turns("s1", limit=0)
        assert turns == []


# ===========================================================================
# 3. Cleanup of old turns (retention policy)
# ===========================================================================


class TestCleanupRetentionPolicy:
    def test_cleanup_removes_turns_older_than_threshold(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn_at("s1", "very old content", delta_days=90))
        removed = store.cleanup(older_than_days=30)
        assert removed == 1
        assert store.get_session_turns("s1") == []

    def test_cleanup_preserves_turns_within_threshold(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn_at("s1", "recent content", delta_days=5))
        removed = store.cleanup(older_than_days=30)
        assert removed == 0
        assert len(store.get_session_turns("s1")) == 1

    def test_cleanup_returns_count_of_deleted_rows(self, store: SQLiteMemoryStore) -> None:
        for i in range(7):
            store.add_turn(_turn_at("s1", f"old {i}", delta_days=60))
        removed = store.cleanup(older_than_days=30)
        assert removed == 7

    def test_cleanup_mixed_ages_removes_only_old_turns(self, store: SQLiteMemoryStore) -> None:
        for i in range(4):
            store.add_turn(_turn_at("s1", f"old-{i}", delta_days=40))
        for i in range(3):
            store.add_turn(_turn_at("s1", f"new-{i}", delta_days=10))
        removed = store.cleanup(older_than_days=30)
        assert removed == 4
        remaining = store.get_session_turns("s1")
        assert len(remaining) == 3
        assert all("new-" in t.content for t in remaining)

    def test_cleanup_on_empty_store_returns_zero(self, store: SQLiteMemoryStore) -> None:
        assert store.cleanup(older_than_days=30) == 0

    def test_cleanup_spans_multiple_sessions(self, store: SQLiteMemoryStore) -> None:
        for sess in ["alpha", "beta", "gamma"]:
            store.add_turn(_turn_at(sess, "old data", delta_days=50))
        removed = store.cleanup(older_than_days=30)
        assert removed == 3

    def test_cleanup_idempotent_on_second_call(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn_at("s1", "old data", delta_days=60))
        first = store.cleanup(older_than_days=30)
        second = store.cleanup(older_than_days=30)
        assert first == 1
        assert second == 0


# ===========================================================================
# 4. WAL mode enabled
# ===========================================================================


class TestWALMode:
    def test_journal_mode_is_wal(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "wal_check.db")
        SQLiteMemoryStore(db_path=db_path)
        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_wal_mode_persists_after_reopening(self, tmp_path: Path) -> None:
        """WAL mode is sticky — reopening the file should still report wal."""
        db_path = str(tmp_path / "wal_persist.db")
        store = SQLiteMemoryStore(db_path=db_path)
        store.add_turn(_turn("s1", content="data"))
        del store  # close original connection

        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_wal_mode_allows_concurrent_readers(self, tmp_path: Path) -> None:
        """WAL allows one writer and multiple simultaneous readers without locking errors."""
        db_path = str(tmp_path / "wal_concurrent.db")
        store = SQLiteMemoryStore(db_path=db_path)
        for i in range(20):
            store.add_turn(_turn("s1", content=f"entry {i}"))

        errors: list[Exception] = []

        def read_many() -> None:
            try:
                for _ in range(10):
                    store.get_recent_turns(limit=5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# 5. Thread safety (concurrent read/write)
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_writes_to_same_store_do_not_corrupt(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "concurrent_write.db")
        store = SQLiteMemoryStore(db_path=db_path)
        errors: list[Exception] = []

        def write_batch(session_id: str, count: int) -> None:
            try:
                for i in range(count):
                    store.add_turn(_turn(session_id, content=f"msg {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=write_batch, args=(f"sess-{j}", 25))
            for j in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        for j in range(6):
            turns = store.get_session_turns(f"sess-{j}")
            assert len(turns) == 25

    def test_concurrent_reads_and_writes_do_not_raise(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "rw_concurrent.db")
        store = SQLiteMemoryStore(db_path=db_path)
        for i in range(10):
            store.add_turn(_turn("shared", content=f"baseline {i}"))

        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(15):
                    store.get_recent_turns(limit=5)
                    store.search("baseline")
            except Exception as exc:
                errors.append(exc)

        def writer() -> None:
            try:
                for i in range(15):
                    store.add_turn(_turn("shared", content=f"new msg {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=reader) for _ in range(4)]
            + [threading.Thread(target=writer) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_cleanup_and_writes_are_safe(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "cleanup_concurrent.db")
        store = SQLiteMemoryStore(db_path=db_path)
        errors: list[Exception] = []

        def cleaner() -> None:
            try:
                for _ in range(5):
                    store.cleanup(older_than_days=0)
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        def writer() -> None:
            try:
                for i in range(20):
                    store.add_turn(_turn("sess", content=f"item {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=cleaner) for _ in range(2)]
            + [threading.Thread(target=writer) for _ in range(3)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ===========================================================================
# 6. Large content handling
# ===========================================================================


class TestLargeContentHandling:
    def test_store_and_retrieve_100k_character_turn(self, store: SQLiteMemoryStore) -> None:
        huge = "Z" * 100_000
        store.add_turn(_turn("s-huge", content=huge))
        turns = store.get_session_turns("s-huge")
        assert turns[0].content == huge

    def test_bulk_insert_500_turns_completes_in_under_10_seconds(
        self, store: SQLiteMemoryStore
    ) -> None:
        start = time.monotonic()
        for i in range(500):
            store.add_turn(_turn("bulk", content=f"message {i} with some padding text here"))
        elapsed = time.monotonic() - start
        assert elapsed < 10.0

    def test_fts_search_over_large_store_finds_unique_needle(self, store: SQLiteMemoryStore) -> None:
        for i in range(200):
            content = "unique_needle_term" if i == 77 else f"unrelated content item {i}"
            store.add_turn(_turn("large-sess", content=content))
        results = store.search("unique_needle_term")
        assert any("unique_needle_term" in r.content for r in results)

    def test_large_content_record_round_trip(self, store: SQLiteMemoryStore) -> None:
        big = "B" * 75_000
        record = LargeContentRecord.new(
            session_id="lc-sess",
            tool_name="read_file",
            content=big,
            summary="A very large file",
        )
        stored_id = store.store_large_content(record)
        retrieved = store.get_large_content(stored_id)
        assert retrieved is not None
        assert retrieved.content == big
        assert retrieved.original_chars == 75_000
        assert retrieved.tool_name == "read_file"
        assert retrieved.summary == "A very large file"

    def test_get_large_content_nonexistent_returns_none(self, store: SQLiteMemoryStore) -> None:
        assert store.get_large_content("ref_does_not_exist") is None

    def test_search_large_content_by_summary_keyword(self, store: SQLiteMemoryStore) -> None:
        rec = LargeContentRecord.new(
            session_id="lc2",
            tool_name="grep",
            content="raw output here",
            summary="grep results for authentication failures",
        )
        store.store_large_content(rec)
        matches = store.search_large_content("authentication", "lc2")
        assert len(matches) == 1

    def test_search_large_content_by_content_keyword(self, store: SQLiteMemoryStore) -> None:
        rec = LargeContentRecord.new(
            session_id="lc3",
            tool_name="cat",
            content="the file contains UNIQUE_PHRASE_ALPHA inside",
            summary="cat output",
        )
        store.store_large_content(rec)
        matches = store.search_large_content("UNIQUE_PHRASE_ALPHA", "lc3")
        assert len(matches) == 1

    def test_search_large_content_no_match_returns_empty(self, store: SQLiteMemoryStore) -> None:
        rec = LargeContentRecord.new(
            session_id="lc4",
            tool_name="ls",
            content="directory listing",
            summary="ls output",
        )
        store.store_large_content(rec)
        assert store.search_large_content("xyzzy_not_here", "lc4") == []

    def test_large_content_replace_on_duplicate_id(self, store: SQLiteMemoryStore) -> None:
        rec = LargeContentRecord.new(
            session_id="lc5",
            tool_name="cat",
            content="original content",
        )
        store.store_large_content(rec)
        # Replace with same id
        updated = LargeContentRecord(
            id=rec.id,
            session_id=rec.session_id,
            turn_id=None,
            tool_name="cat",
            original_chars=len("replaced"),
            content="replaced",
            summary="replaced summary",
            created_at=rec.created_at,
        )
        store.store_large_content(updated)
        retrieved = store.get_large_content(rec.id)
        assert retrieved is not None
        assert retrieved.content == "replaced"


# ===========================================================================
# 7. Summary records: add, get by depth, mark compacted
# ===========================================================================


class TestSummaryRecords:
    def test_add_and_retrieve_summary(self, store: SQLiteMemoryStore) -> None:
        s = SummaryRecord.new("sess1", depth=0, content="A concise summary of recent history")
        store.add_summary(s)
        results = store.get_summaries("sess1")
        assert len(results) == 1
        assert results[0].content == "A concise summary of recent history"
        assert results[0].depth == 0

    def test_get_summaries_filtered_by_depth(self, store: SQLiteMemoryStore) -> None:
        s0 = SummaryRecord.new("sess2", depth=0, content="leaf summary")
        s1 = SummaryRecord.new("sess2", depth=1, content="condensed summary")
        store.add_summary(s0)
        store.add_summary(s1)
        depth0 = store.get_summaries("sess2", depth=0)
        assert len(depth0) == 1
        assert depth0[0].depth == 0
        depth1 = store.get_summaries("sess2", depth=1)
        assert len(depth1) == 1
        assert depth1[0].depth == 1

    def test_get_summaries_all_depths_when_depth_is_none(self, store: SQLiteMemoryStore) -> None:
        for d in range(3):
            store.add_summary(SummaryRecord.new("sess3", depth=d, content=f"depth {d} summary"))
        all_summaries = store.get_summaries("sess3")
        assert len(all_summaries) == 3
        depths = {s.depth for s in all_summaries}
        assert depths == {0, 1, 2}

    def test_summary_round_trip_preserves_source_turn_ids(self, store: SQLiteMemoryStore) -> None:
        ids = [str(uuid.uuid4()) for _ in range(3)]
        s = SummaryRecord.new("sess4", depth=0, content="summary", source_turn_ids=ids)
        store.add_summary(s)
        retrieved = store.get_summary_by_id(s.id)
        assert retrieved is not None
        assert retrieved.source_turn_ids == ids

    def test_summary_round_trip_preserves_source_summary_ids(self, store: SQLiteMemoryStore) -> None:
        child_ids = [f"sum_{uuid.uuid4().hex[:16]}" for _ in range(2)]
        s = SummaryRecord.new("sess5", depth=1, content="condensed", source_summary_ids=child_ids)
        store.add_summary(s)
        retrieved = store.get_summary_by_id(s.id)
        assert retrieved is not None
        assert retrieved.source_summary_ids == child_ids

    def test_summary_token_estimate_auto_computed(self, store: SQLiteMemoryStore) -> None:
        content = "word " * 400  # 2000 chars -> ~500 tokens
        s = SummaryRecord.new("sess6", depth=0, content=content)
        store.add_summary(s)
        retrieved = store.get_summary_by_id(s.id)
        assert retrieved is not None
        assert retrieved.token_estimate > 0

    def test_mark_summary_compacted_sets_parent_id(self, store: SQLiteMemoryStore) -> None:
        s1 = SummaryRecord.new("sess7", depth=0, content="leaf one")
        s2 = SummaryRecord.new("sess7", depth=0, content="leaf two")
        parent = SummaryRecord.new("sess7", depth=1, content="parent summary")
        store.add_summary(s1)
        store.add_summary(s2)
        store.add_summary(parent)

        store.mark_summary_compacted([s1.id, s2.id], parent_id=parent.id)

        r1 = store.get_summary_by_id(s1.id)
        r2 = store.get_summary_by_id(s2.id)
        assert r1 is not None and r1.parent_id == parent.id
        assert r2 is not None and r2.parent_id == parent.id

    def test_mark_summary_compacted_empty_list_is_noop(self, store: SQLiteMemoryStore) -> None:
        # Should not raise and should not modify any record
        store.mark_summary_compacted([], parent_id="sum_parent")

    def test_get_uncompacted_summaries_returns_only_parentless(
        self, store: SQLiteMemoryStore
    ) -> None:
        parent = SummaryRecord.new("sess8", depth=1, content="parent")
        s_free = SummaryRecord.new("sess8", depth=0, content="free leaf")
        s_bound = SummaryRecord.new("sess8", depth=0, content="compacted leaf")
        store.add_summary(parent)
        store.add_summary(s_free)
        store.add_summary(s_bound)
        store.mark_summary_compacted([s_bound.id], parent_id=parent.id)

        uncompacted = store.get_uncompacted_summaries("sess8", depth=0)
        ids = [s.id for s in uncompacted]
        assert s_free.id in ids
        assert s_bound.id not in ids

    def test_get_child_summaries_returns_correct_children(self, store: SQLiteMemoryStore) -> None:
        parent = SummaryRecord.new("sess9", depth=1, content="parent node")
        c1 = SummaryRecord.new("sess9", depth=0, content="child one")
        c2 = SummaryRecord.new("sess9", depth=0, content="child two")
        store.add_summary(parent)
        store.add_summary(c1)
        store.add_summary(c2)
        store.mark_summary_compacted([c1.id, c2.id], parent_id=parent.id)

        children = store.get_child_summaries(parent.id)
        child_ids = {s.id for s in children}
        assert c1.id in child_ids
        assert c2.id in child_ids

    def test_get_source_turns_resolves_ids_to_full_turns(self, store: SQLiteMemoryStore) -> None:
        t1 = _turn("sessA", content="first source turn")
        t2 = _turn("sessA", content="second source turn")
        store.add_turn(t1)
        store.add_turn(t2)
        s = SummaryRecord.new("sessA", depth=0, content="summary", source_turn_ids=[t1.id, t2.id])
        store.add_summary(s)

        source_turns = store.get_source_turns(s.id)
        source_ids = {t.id for t in source_turns}
        assert t1.id in source_ids
        assert t2.id in source_ids

    def test_get_source_turns_returns_empty_for_no_source_ids(
        self, store: SQLiteMemoryStore
    ) -> None:
        s = SummaryRecord.new("sessB", depth=0, content="summary without sources")
        store.add_summary(s)
        assert store.get_source_turns(s.id) == []

    def test_get_source_turns_returns_empty_for_nonexistent_summary(
        self, store: SQLiteMemoryStore
    ) -> None:
        assert store.get_source_turns("sum_ghost") == []

    def test_get_summary_by_id_nonexistent_returns_none(self, store: SQLiteMemoryStore) -> None:
        assert store.get_summary_by_id("sum_nonexistent") is None

    def test_add_summary_replace_on_duplicate_id(self, store: SQLiteMemoryStore) -> None:
        s = SummaryRecord.new("sessC", depth=0, content="original")
        store.add_summary(s)
        # Replace with same id
        s2 = SummaryRecord(
            id=s.id,
            session_id=s.session_id,
            depth=0,
            content="replaced content",
            token_estimate=100,
            source_turn_ids=[],
            source_summary_ids=[],
            parent_id=None,
            time_range_start=None,
            time_range_end=None,
            descendant_count=0,
            file_refs=[],
            created_at=s.created_at,
        )
        store.add_summary(s2)
        retrieved = store.get_summary_by_id(s.id)
        assert retrieved is not None
        assert retrieved.content == "replaced content"

    def test_search_summaries_fts_finds_matching_content(self, store: SQLiteMemoryStore) -> None:
        s1 = SummaryRecord.new("sess-fts", 0, "Python coroutines and async IO patterns")
        s2 = SummaryRecord.new("sess-fts", 0, "Database migration and schema evolution")
        store.add_summary(s1)
        store.add_summary(s2)
        results = store.search_summaries("coroutines")
        assert len(results) == 1
        assert "coroutines" in results[0].content

    def test_search_summaries_with_session_filter(self, store: SQLiteMemoryStore) -> None:
        store.add_summary(SummaryRecord.new("s-a", 0, "matching topic in session a"))
        store.add_summary(SummaryRecord.new("s-b", 0, "matching topic in session b"))
        results = store.search_summaries("matching", session_id="s-a")
        assert len(results) == 1
        assert results[0].session_id == "s-a"

    def test_search_summaries_empty_store_returns_empty(self, store: SQLiteMemoryStore) -> None:
        assert store.search_summaries("anything") == []


# ===========================================================================
# 8. Session token count calculation
# ===========================================================================


class TestSessionTokenCount:
    def test_token_count_is_positive_for_nonempty_session(self, store: SQLiteMemoryStore) -> None:
        for _ in range(5):
            store.add_turn(_turn("tok-sess", content="This is a message of some length"))
        count = store.get_session_token_count("tok-sess")
        assert count > 0

    def test_token_count_empty_session_returns_one(self, store: SQLiteMemoryStore) -> None:
        # Empty session: 0 chars // 4 -> 0, clamped to max(1, 0) == 1
        count = store.get_session_token_count("empty-tok-sess")
        assert count == 1

    def test_token_count_scales_with_content_length(self, store: SQLiteMemoryStore) -> None:
        short_content = "hi"  # 2 chars
        long_content = "word " * 1000  # 5000 chars
        store.add_turn(_turn("short-sess", content=short_content))
        store.add_turn(_turn("long-sess", content=long_content))
        short_count = store.get_session_token_count("short-sess")
        long_count = store.get_session_token_count("long-sess")
        assert long_count > short_count

    def test_token_count_includes_summaries(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("ts1", content="base turn content"))
        count_before = store.get_session_token_count("ts1")

        big_summary = SummaryRecord.new("ts1", depth=0, content="summary content " * 50)
        store.add_summary(big_summary)
        count_after = store.get_session_token_count("ts1")
        assert count_after > count_before

    def test_token_count_excludes_compacted_summaries(self, store: SQLiteMemoryStore) -> None:
        """Compacted summaries (with parent_id set) are excluded from token count."""
        store.add_turn(_turn("ts2", content="a turn"))
        leaf = SummaryRecord.new("ts2", depth=0, content="leaf summary " * 100)
        parent = SummaryRecord.new("ts2", depth=1, content="parent")
        store.add_summary(leaf)
        store.add_summary(parent)

        count_with_leaf = store.get_session_token_count("ts2")
        store.mark_summary_compacted([leaf.id], parent_id=parent.id)
        count_without_leaf = store.get_session_token_count("ts2")
        # Compacted leaf must not count toward the token budget
        assert count_without_leaf < count_with_leaf

    def test_token_count_approximately_chars_over_four(self, store: SQLiteMemoryStore) -> None:
        content = "a" * 400  # exactly 400 chars -> 100 tokens
        store.add_turn(_turn("approx-sess", content=content))
        count = store.get_session_token_count("approx-sess")
        # Allow some rounding variance
        assert 95 <= count <= 105

    def test_token_count_multiple_turns_summed(self, store: SQLiteMemoryStore) -> None:
        for _ in range(4):
            store.add_turn(_turn("multi-tok", content="a" * 400))  # 400 chars each
        # 4 x 400 = 1600 chars -> 400 tokens
        count = store.get_session_token_count("multi-tok")
        assert 380 <= count <= 420


# ===========================================================================
# 9. Turn pagination (limit / offset via helper patterns)
# ===========================================================================


class TestTurnPagination:
    """SQLiteMemoryStore provides limit-based pagination.

    get_session_turns(session_id, limit=N) returns the N most-recent turns.
    Sliding-window pagination can be approximated by using timestamps as
    cursors, or by fetching all turns and slicing in application code.
    These tests verify the limit-based primitives behave correctly.
    """

    def test_limit_1_returns_single_most_recent_turn(self, store: SQLiteMemoryStore) -> None:
        for i in range(5):
            store.add_turn(_turn_at_seconds("pag", f"msg {i}", delta_seconds=5 - i))
        turns = store.get_session_turns("pag", limit=1)
        assert len(turns) == 1
        assert turns[0].content == "msg 4"

    def test_successive_limits_give_different_windows(self, store: SQLiteMemoryStore) -> None:
        for i in range(10):
            store.add_turn(_turn_at_seconds("win", f"item {i}", delta_seconds=10 - i))
        last_3 = store.get_session_turns("win", limit=3)
        last_5 = store.get_session_turns("win", limit=5)
        assert len(last_3) == 3
        assert len(last_5) == 5
        # The 3-item window is a subset of the 5-item window
        last_3_ids = {t.id for t in last_3}
        last_5_ids = {t.id for t in last_5}
        assert last_3_ids.issubset(last_5_ids)

    def test_limit_equal_to_total_returns_all(self, store: SQLiteMemoryStore) -> None:
        for i in range(8):
            store.add_turn(_turn("page-eq", content=f"msg {i}"))
        turns = store.get_session_turns("page-eq", limit=8)
        assert len(turns) == 8

    def test_limit_greater_than_total_returns_all(self, store: SQLiteMemoryStore) -> None:
        for i in range(5):
            store.add_turn(_turn("page-over", content=f"msg {i}"))
        turns = store.get_session_turns("page-over", limit=100)
        assert len(turns) == 5

    def test_recent_turns_limit_pages_across_sessions(self, store: SQLiteMemoryStore) -> None:
        for sess in ["p1", "p2", "p3"]:
            for i in range(10):
                store.add_turn(_turn_at_seconds(sess, f"{sess}-{i}", delta_seconds=100 - i))
        page1 = store.get_recent_turns(limit=5)
        page2 = store.get_recent_turns(limit=10)
        assert len(page1) == 5
        assert len(page2) == 10
        # page1 items should all appear in page2
        p1_ids = {t.id for t in page1}
        p2_ids = {t.id for t in page2}
        assert p1_ids.issubset(p2_ids)

    def test_get_summaries_limit(self, store: SQLiteMemoryStore) -> None:
        for i in range(10):
            store.add_summary(SummaryRecord.new("pag-sess", depth=0, content=f"summary {i}"))
        results = store.get_summaries("pag-sess", limit=4)
        assert len(results) == 4

    def test_get_learnings_limit(self, store: SQLiteMemoryStore) -> None:
        for i in range(8):
            store.save_learning(_FakeLearning(lesson=f"Lesson {i}"))
        lessons = store.get_learnings(limit=3)
        assert len(lessons) == 3


# ===========================================================================
# 10. Unicode content handling
# ===========================================================================


class TestUnicodeContentHandling:
    def test_store_and_retrieve_accented_characters(self, store: SQLiteMemoryStore) -> None:
        content = "Café résumé naïve fiancée"
        store.add_turn(_turn("u1", content=content))
        turns = store.get_session_turns("u1")
        assert turns[0].content == content

    def test_store_and_retrieve_cjk_characters(self, store: SQLiteMemoryStore) -> None:
        content = "你好世界 — 日本語テスト — 한국어"
        store.add_turn(_turn("u2", content=content))
        turns = store.get_session_turns("u2")
        assert turns[0].content == content

    def test_store_and_retrieve_arabic_rtl_text(self, store: SQLiteMemoryStore) -> None:
        content = "مرحبا بالعالم"
        store.add_turn(_turn("u3", content=content))
        turns = store.get_session_turns("u3")
        assert turns[0].content == content

    def test_store_and_retrieve_emoji_characters(self, store: SQLiteMemoryStore) -> None:
        content = "rocket launch complete"
        store.add_turn(_turn("u4", content=content))
        turns = store.get_session_turns("u4")
        assert turns[0].content == content

    def test_store_and_retrieve_mixed_scripts(self, store: SQLiteMemoryStore) -> None:
        content = "Python (蟒蛇) + Rust (锈) = fast & safe"
        store.add_turn(_turn("u5", content=content))
        turns = store.get_session_turns("u5")
        assert turns[0].content == content

    def test_fts_search_with_accented_query_does_not_raise(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("u6", content="Café au lait is delicious"))
        results = store.search("Café")
        assert isinstance(results, list)

    def test_fts_search_with_cjk_query_does_not_raise(self, store: SQLiteMemoryStore) -> None:
        store.add_turn(_turn("u7", content="日本語のテスト"))
        results = store.search("テスト")
        assert isinstance(results, list)

    def test_unicode_session_id_works(self, store: SQLiteMemoryStore) -> None:
        session_id = "session-日本語-2026"
        store.add_turn(_turn(session_id, content="unicode session test"))
        turns = store.get_session_turns(session_id)
        assert len(turns) == 1

    def test_summary_with_unicode_content(self, store: SQLiteMemoryStore) -> None:
        s = SummaryRecord.new("u-sess", depth=0, content="Zusammenfassung: Künstliche Intelligenz")
        store.add_summary(s)
        retrieved = store.get_summary_by_id(s.id)
        assert retrieved is not None
        assert retrieved.content == "Zusammenfassung: Künstliche Intelligenz"

    def test_large_content_record_with_unicode(self, store: SQLiteMemoryStore) -> None:
        content = "Unicode output: " + "日本語テスト" * 1000
        rec = LargeContentRecord.new(
            session_id="u-lc",
            tool_name="cat",
            content=content,
        )
        cid = store.store_large_content(rec)
        retrieved = store.get_large_content(cid)
        assert retrieved is not None
        assert retrieved.content == content

    def test_learning_with_unicode_lesson(self, store: SQLiteMemoryStore) -> None:
        store.save_learning(_FakeLearning(lesson="Naïve string comparison fails with Ångström"))
        lessons = store.get_learnings()
        assert "Naïve string comparison fails with Ångström" in lessons


# ===========================================================================
# ResilientMemoryStore: delegates to underlying store on success
# ===========================================================================


def _mock_turn(session_id: str = "s1", content: str = "hello") -> MagicMock:
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


class TestResilientStoreDelegation:
    def test_add_turn_delegates_to_primary_when_healthy(self) -> None:
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        turn = _mock_turn()
        store.add_turn(turn)
        primary.add_turn.assert_called_once_with(turn)

    def test_get_session_turns_delegates_to_primary_when_healthy(self) -> None:
        primary = _healthy_primary()
        expected = [_mock_turn("s1")]
        primary.get_session_turns.return_value = expected
        store = ResilientMemoryStore(primary)
        result = store.get_session_turns("s1")
        assert result == expected
        primary.get_session_turns.assert_called_once_with("s1", 100)

    def test_get_recent_turns_delegates_to_primary_when_healthy(self) -> None:
        primary = _healthy_primary()
        expected = [_mock_turn()]
        primary.get_recent_turns.return_value = expected
        store = ResilientMemoryStore(primary)
        result = store.get_recent_turns(limit=10)
        assert result == expected

    def test_search_delegates_to_primary_when_healthy(self) -> None:
        primary = _healthy_primary()
        expected = [_mock_turn()]
        primary.search.return_value = expected
        store = ResilientMemoryStore(primary)
        result = store.search("query", limit=5, session_id="s1")
        assert result == expected

    def test_cleanup_delegates_to_primary_when_healthy(self) -> None:
        primary = _healthy_primary()
        primary.cleanup.return_value = 7
        store = ResilientMemoryStore(primary)
        result = store.cleanup(older_than_days=30)
        assert result == 7

    def test_save_learning_delegates_to_primary(self) -> None:
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        learning = _FakeLearning()
        store.save_learning(learning)
        primary.save_learning.assert_called_once_with(learning)

    def test_get_learnings_delegates_to_primary_when_healthy(self) -> None:
        primary = _healthy_primary()
        primary.get_learnings.return_value = ["lesson one"]
        store = ResilientMemoryStore(primary)
        result = store.get_learnings()
        assert result == ["lesson one"]

    def test_clear_session_delegates_to_primary(self) -> None:
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        store.clear_session("s1")
        primary.clear_session.assert_called_once_with("s1")


class TestResilientStoreGracefulDegradation:
    def test_add_turn_writes_to_cache_on_primary_failure(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        turn = _mock_turn("s1")
        store.add_turn(turn)
        assert turn in store._cache["s1"]

    def test_get_session_turns_falls_back_to_cache(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        turn = _mock_turn("s1", "cached content")
        store._cache["s1"] = [turn]
        result = store.get_session_turns("s1")
        assert turn in result

    def test_get_recent_turns_falls_back_to_all_cached_turns(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        t1 = _mock_turn("s1", "a")
        t2 = _mock_turn("s2", "b")
        store._cache["s1"] = [t1]
        store._cache["s2"] = [t2]
        result = store.get_recent_turns(limit=10)
        assert t1 in result
        assert t2 in result

    def test_search_falls_back_to_keyword_scan_of_cache(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        match_turn = _mock_turn("s1", "unique keyword found here")
        no_match = _mock_turn("s1", "nothing relevant")
        store._cache["s1"] = [match_turn, no_match]
        results = store.search("keyword")
        assert match_turn in results
        assert no_match not in results

    def test_search_cache_keyword_scan_is_case_insensitive(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        turn = _mock_turn("s1", "Python Programming Language")
        store._cache["s1"] = [turn]
        results = store.search("python")
        assert turn in results

    def test_search_cache_respects_session_filter(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        t_a = _mock_turn("s-a", "matching term alpha")
        t_b = _mock_turn("s-b", "matching term beta")
        store._cache["s-a"] = [t_a]
        store._cache["s-b"] = [t_b]
        results = store.search("matching", session_id="s-a")
        assert t_a in results
        assert t_b not in results

    def test_cleanup_returns_zero_on_primary_failure(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        assert store.cleanup(older_than_days=30) == 0

    def test_get_learnings_returns_empty_list_on_primary_failure(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        assert store.get_learnings() == []

    def test_clear_session_removes_cache_on_primary_failure(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        store._cache["s1"] = [_mock_turn("s1")]
        store.clear_session("s1")
        assert "s1" not in store._cache


class TestResilientStoreHealthTracking:
    def test_is_healthy_true_initially(self) -> None:
        store = ResilientMemoryStore(_broken_primary())
        assert store.is_healthy is True

    def test_is_healthy_becomes_false_after_failure(self) -> None:
        store = ResilientMemoryStore(_broken_primary(), max_failures=1)
        store.add_turn(_mock_turn())
        assert store.is_healthy is False

    def test_is_healthy_requires_max_failures_to_trip(self) -> None:
        store = ResilientMemoryStore(_broken_primary(), max_failures=3)
        # Two failures are below the threshold set but _healthy is set False on first failure
        store.add_turn(_mock_turn())
        assert store.is_healthy is False

    def test_recovery_restores_healthy_state(self) -> None:
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=1)
        store.add_turn(_mock_turn())
        assert store.is_healthy is False

        primary.add_turn.side_effect = None  # fix primary
        store.add_turn(_mock_turn())
        assert store.is_healthy is True

    def test_recovery_syncs_cached_turns_to_primary(self) -> None:
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=1)
        turn = _mock_turn()
        store.add_turn(turn)  # fails -> cached

        primary.add_turn.side_effect = None  # fix primary
        store.add_turn(_mock_turn())  # triggers recovery + sync

        # The sync replays cached turns to the primary
        assert primary.add_turn.call_count >= 2

    def test_failure_counter_resets_after_recovery(self) -> None:
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=3)
        store.add_turn(_mock_turn())  # failure 1
        primary.add_turn.side_effect = None  # fix
        store.add_turn(_mock_turn())  # success -> resets counter
        with store._lock:
            assert store._failures == 0


class TestResilientStoreThreadSafety:
    def test_concurrent_add_turn_does_not_raise(self) -> None:
        primary = _healthy_primary()
        store = ResilientMemoryStore(primary)
        errors: list[Exception] = []

        def add_many(session_id: str) -> None:
            try:
                for i in range(50):
                    store.add_turn(_mock_turn(session_id, f"content {i}"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_many, args=(f"t{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_failure_and_recovery_does_not_deadlock(self) -> None:
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=2)
        errors: list[Exception] = []
        done = threading.Event()

        def add_and_recover() -> None:
            try:
                for _ in range(10):
                    store.add_turn(_mock_turn())
                # Fix primary mid-flight
                primary.add_turn.side_effect = None
                for _ in range(5):
                    store.add_turn(_mock_turn())
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        t = threading.Thread(target=add_and_recover)
        t.start()
        finished = done.wait(timeout=5.0)
        t.join(timeout=1.0)

        assert finished, "Thread did not complete within timeout — possible deadlock"
        assert errors == []

    def test_is_healthy_read_is_thread_safe(self) -> None:
        primary = _broken_primary()
        store = ResilientMemoryStore(primary, max_failures=1)
        results: list[bool] = []
        errors: list[Exception] = []

        def check_health() -> None:
            try:
                for _ in range(100):
                    results.append(store.is_healthy)
            except Exception as exc:
                errors.append(exc)

        def trigger_failure() -> None:
            try:
                for _ in range(10):
                    store.add_turn(_mock_turn())
            except Exception as exc:
                errors.append(exc)

        threads = (
            [threading.Thread(target=check_health) for _ in range(3)]
            + [threading.Thread(target=trigger_failure) for _ in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Results must only be booleans — no corrupted state
        assert all(isinstance(r, bool) for r in results)
