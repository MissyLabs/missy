"""Tests for SQLiteMemoryStore cost tracking."""

import pytest

from missy.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    return SQLiteMemoryStore(db_path=db_path)


class TestRecordCost:
    def test_record_and_retrieve(self, store):
        store.record_cost(
            session_id="sess-1",
            model="claude-sonnet-4-20250514",
            prompt_tokens=500,
            completion_tokens=200,
            cost_usd=0.0045,
        )
        costs = store.get_session_costs("sess-1")
        assert len(costs) == 1
        assert costs[0]["model"] == "claude-sonnet-4-20250514"
        assert costs[0]["prompt_tokens"] == 500
        assert costs[0]["completion_tokens"] == 200
        assert costs[0]["cost_usd"] == pytest.approx(0.0045)

    def test_multiple_records(self, store):
        store.record_cost("sess-1", "claude-sonnet-4", 100, 50, 0.001)
        store.record_cost("sess-1", "claude-sonnet-4", 200, 100, 0.002)
        store.record_cost("sess-1", "claude-haiku-4", 300, 150, 0.0005)
        costs = store.get_session_costs("sess-1")
        assert len(costs) == 3

    def test_different_sessions_isolated(self, store):
        store.record_cost("sess-1", "model-a", 100, 50, 0.001)
        store.record_cost("sess-2", "model-b", 200, 100, 0.002)
        assert len(store.get_session_costs("sess-1")) == 1
        assert len(store.get_session_costs("sess-2")) == 1

    def test_empty_session_returns_empty(self, store):
        assert store.get_session_costs("nonexistent") == []

    def test_timestamp_is_set(self, store):
        store.record_cost("sess-1", "model", 100, 50, 0.001)
        costs = store.get_session_costs("sess-1")
        assert costs[0]["timestamp"] is not None
        assert len(costs[0]["timestamp"]) > 10  # ISO-8601


class TestGetTotalCosts:
    def test_aggregates_by_session(self, store):
        store.record_cost("sess-1", "model-a", 100, 50, 0.001)
        store.record_cost("sess-1", "model-a", 200, 100, 0.002)
        store.record_cost("sess-2", "model-b", 300, 150, 0.003)

        totals = store.get_total_costs()
        assert len(totals) == 2

        # Find sess-1 summary
        sess1 = next(t for t in totals if t["session_id"] == "sess-1")
        assert sess1["call_count"] == 2
        assert sess1["total_prompt_tokens"] == 300
        assert sess1["total_completion_tokens"] == 150
        assert sess1["total_cost_usd"] == pytest.approx(0.003)

    def test_respects_limit(self, store):
        for i in range(5):
            store.record_cost(f"sess-{i}", "model", 100, 50, 0.001)
        totals = store.get_total_costs(limit=3)
        assert len(totals) == 3

    def test_empty_returns_empty(self, store):
        assert store.get_total_costs() == []

    def test_ordered_by_most_recent(self, store):
        import time

        store.record_cost("sess-old", "model", 100, 50, 0.001)
        time.sleep(0.01)
        store.record_cost("sess-new", "model", 200, 100, 0.002)
        totals = store.get_total_costs()
        assert totals[0]["session_id"] == "sess-new"


class TestCostsTableCreation:
    def test_costs_table_exists(self, store):
        conn = store._conn()
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='costs'")
        assert cursor.fetchone() is not None

    def test_costs_indexes_exist(self, store):
        conn = store._conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_costs%'"
        ).fetchall()
        index_names = {r["name"] for r in indexes}
        assert "idx_costs_session" in index_names
        assert "idx_costs_timestamp" in index_names
