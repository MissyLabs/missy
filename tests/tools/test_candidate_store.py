"""Tests for missy.tools.intelligence.candidate_store."""

from __future__ import annotations

from pathlib import Path

import pytest

from missy.tools.intelligence.candidate_store import (
    BenchmarkSummary,
    CandidateStore,
    ToolCandidate,
    ToolLifecycleState,
    is_valid_transition,
)


@pytest.fixture
def store(tmp_path: Path) -> CandidateStore:
    return CandidateStore(db_path=tmp_path / "test_candidates.db")


def _make_candidate(name: str = "test_tool") -> ToolCandidate:
    return ToolCandidate.create(
        name=name,
        description="A test tool",
        schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        permissions={"filesystem_read": True},
        provenance="unit test",
        tags=["test"],
    )


class TestCandidateStoreAdd:
    def test_add_returns_candidate(self, store):
        c = _make_candidate()
        result = store.add(c)
        assert result.id == c.id

    def test_add_persists(self, store):
        c = _make_candidate()
        c.implementation = {"type": "delegated_tool", "tool": "file_read"}
        store.add(c)
        loaded = store.get(c.id)
        assert loaded is not None
        assert loaded.name == c.name
        assert loaded.implementation == {"type": "delegated_tool", "tool": "file_read"}

    def test_initial_state_is_proposed(self, store):
        c = _make_candidate()
        store.add(c)
        loaded = store.get(c.id)
        assert loaded.state == ToolLifecycleState.PROPOSED

    def test_count_increments(self, store):
        assert store.count() == 0
        store.add(_make_candidate("a"))
        store.add(_make_candidate("b"))
        assert store.count() == 2


class TestCandidateStoreGet:
    def test_get_returns_none_for_missing(self, store):
        assert store.get("nonexistent") is None

    def test_get_by_name(self, store):
        c = _make_candidate("my_tool")
        store.add(c)
        result = store.get_by_name("my_tool")
        assert result is not None
        assert result.name == "my_tool"

    def test_get_by_name_missing(self, store):
        assert store.get_by_name("ghost") is None


class TestCandidateStoreGetByPatternKey:
    def test_returns_none_when_absent(self, store):
        assert store.get_by_pattern_key("no-such-pattern") is None

    def test_returns_none_for_empty_key(self, store):
        assert store.get_by_pattern_key("") is None

    def test_finds_candidate_by_pattern_key(self, store):
        c = ToolCandidate.create(
            name="pattern_tool",
            description="from a pattern",
            schema={"type": "object", "properties": {}, "required": []},
            pattern_key="pattern-abc",
        )
        store.add(c)
        found = store.get_by_pattern_key("pattern-abc")
        assert found is not None
        assert found.name == "pattern_tool"

    def test_returns_most_recently_updated_for_duplicate_pattern(self, store):
        first = ToolCandidate.create(
            name="first_attempt",
            description="d",
            schema={"type": "object", "properties": {}, "required": []},
            pattern_key="shared-pattern",
        )
        store.add(first)
        second = ToolCandidate.create(
            name="second_attempt",
            description="d",
            schema={"type": "object", "properties": {}, "required": []},
            pattern_key="shared-pattern",
        )
        # Force a strictly later timestamp so ordering is deterministic even
        # if both candidates are created within the same microsecond.
        second.updated_at = first.updated_at + "1"
        store.add(second)
        found = store.get_by_pattern_key("shared-pattern")
        assert found is not None
        assert found.name == "second_attempt"


class TestCandidateStoreListAll:
    def test_list_returns_all(self, store):
        store.add(_make_candidate("a"))
        store.add(_make_candidate("b"))
        all_items = store.list_all()
        assert len(all_items) == 2

    def test_list_filters_by_state(self, store):
        c = _make_candidate()
        store.add(c)
        proposed = store.list_all(state=ToolLifecycleState.PROPOSED)
        assert len(proposed) == 1
        approved = store.list_all(state=ToolLifecycleState.APPROVED)
        assert len(approved) == 0

    def test_list_filters_by_owner(self, store):
        c = ToolCandidate.create(
            name="owned", description="d", schema={}, provenance="", owner="alice"
        )
        store.add(c)
        store.add(_make_candidate())
        owned = store.list_all(owner="alice")
        assert len(owned) == 1
        assert owned[0].owner == "alice"

    def test_limit_respected(self, store):
        for i in range(5):
            store.add(_make_candidate(f"t{i}"))
        result = store.list_all(limit=3)
        assert len(result) <= 3


class TestTransition:
    def test_transition_updates_state(self, store):
        c = _make_candidate()
        store.add(c)
        updated = store.transition(c.id, ToolLifecycleState.EXPERIMENTAL)
        assert updated.state == ToolLifecycleState.EXPERIMENTAL

    def test_transition_persists(self, store):
        c = _make_candidate()
        store.add(c)
        store.update_benchmark(c.id, BenchmarkSummary(provider="mock", composite=0.9))
        store.transition(c.id, ToolLifecycleState.APPROVED)
        store.transition(c.id, ToolLifecycleState.ENABLED)
        loaded = store.get(c.id)
        assert loaded.state == ToolLifecycleState.ENABLED

    def test_transition_stores_notes(self, store):
        c = _make_candidate()
        store.add(c)
        store.transition(c.id, ToolLifecycleState.DISABLED, notes="too dangerous")
        loaded = store.get(c.id)
        assert loaded.notes == "too dangerous"

    def test_transition_missing_returns_none(self, store):
        result = store.transition("ghost", ToolLifecycleState.APPROVED)
        assert result is None

    def test_transition_rejects_skipping_benchmark_and_approval(self, store):
        c = _make_candidate()
        store.add(c)
        with pytest.raises(ValueError, match="proposed -> enabled"):
            store.transition(c.id, ToolLifecycleState.ENABLED, actor="operator")
        loaded = store.get(c.id)
        assert loaded.state == ToolLifecycleState.PROPOSED

    def test_transition_rejects_approving_before_benchmark(self, store):
        c = _make_candidate()
        store.add(c)
        with pytest.raises(ValueError, match="proposed -> approved"):
            store.transition(c.id, ToolLifecycleState.APPROVED, actor="operator")
        loaded = store.get(c.id)
        assert loaded.state == ToolLifecycleState.PROPOSED

    def test_transition_rejects_disabled_resurrection(self, store):
        c = _make_candidate()
        store.add(c)
        store.transition(c.id, ToolLifecycleState.DISABLED, notes="unsafe")
        with pytest.raises(ValueError, match="disabled -> experimental"):
            store.transition(c.id, ToolLifecycleState.EXPERIMENTAL, actor="operator")

    def test_deprecated_can_be_rolled_forward_or_disabled(self, store):
        c = _make_candidate()
        store.add(c)
        store.update_benchmark(c.id, BenchmarkSummary(provider="mock", composite=0.9))
        store.transition(c.id, ToolLifecycleState.APPROVED)
        store.transition(c.id, ToolLifecycleState.ENABLED)
        deprecated = store.transition(c.id, ToolLifecycleState.DEPRECATED)
        assert deprecated.state == ToolLifecycleState.DEPRECATED
        restored = store.transition(c.id, ToolLifecycleState.ENABLED)
        assert restored.state == ToolLifecycleState.ENABLED

    def test_noop_transition_is_allowed(self, store):
        c = _make_candidate()
        store.add(c)
        updated = store.transition(c.id, ToolLifecycleState.PROPOSED, notes="reviewed")
        assert updated.state == ToolLifecycleState.PROPOSED
        assert updated.notes == "reviewed"


class TestUpdateBenchmark:
    def test_attaches_benchmark_summary(self, store):
        c = _make_candidate()
        store.add(c)
        summary = BenchmarkSummary(
            provider="anthropic",
            correctness=0.9,
            latency_ms=300.0,
            cost_usd=0.0001,
            reliability=1.0,
            safety=1.0,
            schema_score=1.0,
            composite=0.85,
            run_at="2026-01-01T00:00:00",
        )
        updated = store.update_benchmark(c.id, summary)
        assert updated is not None
        assert len(updated.benchmark_scores) == 1
        assert updated.benchmark_scores[0].provider == "anthropic"

    def test_replaces_same_provider_summary(self, store):
        c = _make_candidate()
        store.add(c)
        s1 = BenchmarkSummary(provider="anthropic", composite=0.5, run_at="")
        s2 = BenchmarkSummary(provider="anthropic", composite=0.9, run_at="")
        store.update_benchmark(c.id, s1)
        store.update_benchmark(c.id, s2)
        loaded = store.get(c.id)
        assert len(loaded.benchmark_scores) == 1
        assert loaded.benchmark_scores[0].composite == 0.9

    def test_transitions_to_benchmarked(self, store):
        c = _make_candidate()
        store.add(c)
        s = BenchmarkSummary(provider="openai", composite=0.7, run_at="")
        updated = store.update_benchmark(c.id, s)
        assert updated.state == ToolLifecycleState.BENCHMARKED

    def test_updates_provider_enabled_map(self, store):
        c = _make_candidate()
        store.add(c)
        s = BenchmarkSummary(provider="anthropic", composite=0.8, run_at="")
        updated = store.update_benchmark(c.id, s, provider_enabled={"anthropic": True})
        assert updated.provider_enabled.get("anthropic") is True


class TestDelete:
    def test_delete_removes_candidate(self, store):
        c = _make_candidate()
        store.add(c)
        assert store.delete(c.id) is True
        assert store.get(c.id) is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("ghost") is False


class TestBenchmarkSummary:
    def test_round_trip(self):
        s = BenchmarkSummary(
            provider="anthropic",
            correctness=0.9,
            latency_ms=150.0,
            cost_usd=0.0001,
            reliability=1.0,
            safety=1.0,
            schema_score=0.95,
            composite=0.88,
            run_at="2026-01-01T00:00:00",
        )
        d = s.to_dict()
        s2 = BenchmarkSummary.from_dict(d)
        assert s2.provider == "anthropic"
        assert s2.composite == 0.88


class TestToolLifecycleState:
    def test_active_states(self):
        active = ToolLifecycleState.active_states()
        assert ToolLifecycleState.ENABLED in active
        assert ToolLifecycleState.PROPOSED not in active

    def test_terminal_states(self):
        terminal = ToolLifecycleState.terminal_states()
        assert ToolLifecycleState.DISABLED in terminal


class TestTransitionRules:
    def test_valid_review_path(self):
        assert is_valid_transition(ToolLifecycleState.PROPOSED, ToolLifecycleState.EXPERIMENTAL)
        assert is_valid_transition(ToolLifecycleState.EXPERIMENTAL, ToolLifecycleState.BENCHMARKED)
        assert is_valid_transition(ToolLifecycleState.BENCHMARKED, ToolLifecycleState.APPROVED)
        assert is_valid_transition(ToolLifecycleState.APPROVED, ToolLifecycleState.ENABLED)

    def test_invalid_direct_enable(self):
        assert not is_valid_transition(ToolLifecycleState.PROPOSED, ToolLifecycleState.ENABLED)

    def test_disabled_is_terminal(self):
        assert not is_valid_transition(ToolLifecycleState.DISABLED, ToolLifecycleState.PROPOSED)
