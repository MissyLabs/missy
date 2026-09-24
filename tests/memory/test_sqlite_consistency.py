"""DATA-04: memory.db deletes keep sessions.turn_count, large_content,
summaries and FTS indexes consistent."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from missy.memory.sqlite_store import (
    ConversationTurn,
    LargeContentRecord,
    SQLiteMemoryStore,
    SummaryRecord,
)


@pytest.fixture
def store(tmp_path):
    return SQLiteMemoryStore(str(tmp_path / "memory.db"))


def _count(store, sid):
    row = (
        store._conn()
        .execute("SELECT turn_count FROM sessions WHERE session_id = ?", (sid,))
        .fetchone()
    )
    return row["turn_count"]


def _seed(store, sid="s1", n=3, *, old=False):
    store.register_session(sid, name="t")
    turns = []
    for i in range(n):
        t = ConversationTurn.new(sid, "user", f"message {i}")
        if old:
            t.timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        store.add_turn(t)
        turns.append(t)
    store.update_session_turn_count(sid)
    return turns


def _large(store, sid, turn_id, *, old=False):
    rec = LargeContentRecord.new(session_id=sid, turn_id=turn_id, tool_name="t", content="x" * 100)
    if old:
        rec.created_at = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    store.store_large_content(rec)
    return rec


def test_delete_turn_refreshes_count_and_large_content(store):
    turns = _seed(store)
    rec = _large(store, "s1", turns[0].id)
    assert store.delete_turn(turns[0].id) is True
    assert _count(store, "s1") == 2
    assert store.get_large_content(rec.id) is None


def test_clear_session_full_cleans_everything(store):
    turns = _seed(store)
    rec = _large(store, "s1", turns[1].id)
    result = store.clear_session_full("s1")
    assert result == {"turns": 3, "summaries": 0}
    assert _count(store, "s1") == 0
    assert store.get_large_content(rec.id) is None


def test_clear_session_refreshes_count(store):
    _seed(store)
    store.clear_session("s1")
    assert _count(store, "s1") == 0


def test_cleanup_refreshes_counts_and_prunes_orphans(store):
    old = _seed(store, "s1", 2, old=True)
    _seed(store, "s2", 1)
    rec = _large(store, "s1", old[0].id, old=True)
    summary = SummaryRecord.new(
        session_id="s1",
        depth=0,
        content="summary of old turns",
        source_turn_ids=[t.id for t in old],
    )
    summary.created_at = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    store.add_summary(summary)

    assert store.cleanup(older_than_days=30, include_summaries=True) == 2
    assert _count(store, "s1") == 0
    assert _count(store, "s2") == 1
    assert store.get_large_content(rec.id) is None
    assert store.get_summary_by_id(summary.id) is None


def test_cleanup_keeps_summaries_by_default(store):
    old = _seed(store, "s1", 1, old=True)
    summary = SummaryRecord.new(
        session_id="s1", depth=0, content="keep", source_turn_ids=[old[0].id]
    )
    summary.created_at = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    store.add_summary(summary)
    store.cleanup(older_than_days=30)
    assert store.get_summary_by_id(summary.id) is not None


def test_summary_update_keeps_fts_in_sync(store):
    summary = SummaryRecord.new(session_id="s1", depth=0, content="alpha", source_turn_ids=[])
    store.add_summary(summary)
    store._conn().execute("UPDATE summaries SET content = ? WHERE id = ?", ("bravo", summary.id))
    store._conn().commit()
    assert store.fts_integrity_check() == {"turns_fts": "ok", "summaries_fts": "ok"}
    assert [s.id for s in store.search_summaries("bravo")] == [summary.id]
    assert store.search_summaries("alpha") == []


def test_source_ids_are_json(store):
    summary = SummaryRecord.new(session_id="s", depth=0, content="c", source_turn_ids=["a"])
    store.add_summary(summary)
    row = (
        store._conn()
        .execute("SELECT source_turn_ids FROM summaries WHERE id = ?", (summary.id,))
        .fetchone()
    )
    assert json.loads(row["source_turn_ids"]) == ["a"]
