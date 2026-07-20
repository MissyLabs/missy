"""Run-18 recovery journal regressions (COREVAL-029/030)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from missy.memory.resilient import ResilientMemoryStore


@dataclass
class _Turn:
    id: str
    session_id: str
    content: str = "content"
    timestamp: int = 0


class _Primary:
    def __init__(self) -> None:
        self.turns: dict[str, _Turn] = {}
        self.down = False
        self.add_calls: list[str] = []

    def _check(self) -> None:
        if self.down:
            raise RuntimeError("primary down")

    def add_turn(self, turn: _Turn) -> None:
        self.add_calls.append(turn.id)
        self._check()
        self.turns[turn.id] = turn

    def clear_session(self, session_id: str) -> None:
        self._check()
        self.turns = {
            key: turn for key, turn in self.turns.items() if turn.session_id != session_id
        }

    clear_session_full = clear_session

    def delete_turn(self, turn_id: str) -> bool:
        self._check()
        return self.turns.pop(turn_id, None) is not None

    def get_session_turns(self, session_id: str, limit: int) -> list[_Turn]:
        self._check()
        return [turn for turn in self.turns.values() if turn.session_id == session_id][-limit:]


def test_coreval_029_recovery_replays_only_failed_writes_once() -> None:
    primary = _Primary()
    store = ResilientMemoryStore(primary, max_failures=1)
    store.add_turn(_Turn("old", "s"))
    primary.down = True
    store.add_turn(_Turn("pending", "s"))
    primary.down = False

    store.get_session_turns("s")  # successful read triggers pending convergence
    store.get_session_turns("s")  # repeated recovery must be idempotent

    assert list(primary.turns) == ["old", "pending"]
    assert primary.add_calls.count("old") == 1
    assert primary.add_calls.count("pending") == 2  # failed attempt + one replay
    assert store._pending_ops == []
    assert store.is_healthy


def test_coreval_030_failed_clear_then_new_turn_replays_in_order_without_resurrection() -> None:
    primary = _Primary()
    store = ResilientMemoryStore(primary, max_failures=1)
    store.add_turn(_Turn("old", "s"))
    primary.down = True
    store.clear_session_full("s")
    store.add_turn(_Turn("new", "s"))
    primary.down = False

    store.get_session_turns("s")
    recovered = store.get_session_turns("s")

    assert [turn.id for turn in recovered] == ["new"]
    assert "old" not in primary.turns
    assert store._pending_ops == []


def test_coreval_031_fallback_reads_preserve_scope_order_limits_and_status() -> None:
    primary = _Primary()
    store = ResilientMemoryStore(primary, max_failures=1)
    store.add_turn(_Turn("b", "session", content="Straße marker", timestamp=10))
    store.add_turn(_Turn("a", "session", content="other", timestamp=10))
    store.add_turn(_Turn("outside", "other", content="Straße marker", timestamp=1))
    primary.down = True

    assert [turn.id for turn in store.get_session_turns("session", limit=10)] == ["a", "b"]
    assert [turn.id for turn in store.search("STRASSE", session_id="session")] == ["b"]
    assert store.read_status["get_session_turns"] == "fallback"
    assert store.read_status["search"] == "fallback"

    with pytest.raises(ValueError, match="limit"):
        store.get_session_turns("session", limit=0)
    with pytest.raises(ValueError, match="limit"):
        store.search("marker", limit=1001)

    assert store.get_learnings() == []
    assert store.get_cost_totals()["total_cost_usd"] == 0.0
    assert store.read_status["get_learnings"] == "unavailable"
    assert store.read_status["get_cost_totals"] == "unavailable"
