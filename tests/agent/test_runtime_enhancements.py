"""Tests for session-4 runtime enhancements: budget enforcement, checkpoint
recovery scan, max_spend_usd config flow."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.agent.cost_tracker import BudgetExceededError
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.providers import registry as registry_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(name="fake", reply="ok"):
    provider = MagicMock()
    provider.name = name
    provider.is_available.return_value = True

    from missy.providers.base import CompletionResponse

    provider.complete.return_value = CompletionResponse(
        content=reply,
        model="fake-model-1",
        provider=name,
        usage={"prompt_tokens": 5, "completion_tokens": 3},
        raw={},
    )
    return provider


def _make_registry(providers=None):
    reg = MagicMock()
    providers = providers or {}
    reg.get.side_effect = lambda n: providers.get(n)
    reg.get_available.side_effect = lambda: [p for p in providers.values() if p.is_available()]
    return reg


@pytest.fixture(autouse=True)
def reset_singleton():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# AgentConfig.max_spend_usd
# ---------------------------------------------------------------------------


class TestAgentConfigMaxSpend:
    def test_default_is_zero(self):
        cfg = AgentConfig()
        assert cfg.max_spend_usd == 0.0

    def test_custom_value(self):
        cfg = AgentConfig(max_spend_usd=5.0)
        assert cfg.max_spend_usd == 5.0


# ---------------------------------------------------------------------------
# CostTracker creation with budget
# ---------------------------------------------------------------------------


class TestCostTrackerCreation:
    def test_cost_tracker_inherits_budget(self):
        cfg = AgentConfig(max_spend_usd=2.50)
        runtime = AgentRuntime(cfg)
        tracker = runtime._get_cost_tracker("")
        assert tracker is not None
        assert tracker.max_spend_usd == 2.50

    def test_cost_tracker_default_unlimited(self):
        cfg = AgentConfig()
        runtime = AgentRuntime(cfg)
        tracker = runtime._get_cost_tracker("")
        assert tracker is not None
        assert tracker.max_spend_usd == 0.0


# ---------------------------------------------------------------------------
# Budget enforcement (_check_budget)
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_check_budget_no_tracker(self):
        cfg = AgentConfig()
        runtime = AgentRuntime(cfg)
        runtime._cost_tracking_enabled = False
        # Should not raise
        runtime._check_budget()

    def test_check_budget_under_limit(self):
        cfg = AgentConfig(max_spend_usd=10.0)
        runtime = AgentRuntime(cfg)
        runtime._get_cost_tracker("").record(
            "claude-sonnet-4", prompt_tokens=100, completion_tokens=50
        )
        # Should not raise
        runtime._check_budget()

    def test_check_budget_over_limit_raises(self):
        cfg = AgentConfig(max_spend_usd=0.001)
        runtime = AgentRuntime(cfg)
        # Record enough to exceed $0.001, against the same session the
        # check below queries.
        runtime._get_cost_tracker("test").record(
            "claude-opus-4", prompt_tokens=10000, completion_tokens=10000
        )
        with pytest.raises(BudgetExceededError):
            runtime._check_budget(session_id="test", task_id="t1")

    def test_check_budget_emits_audit_event(self):
        cfg = AgentConfig(max_spend_usd=0.001)
        runtime = AgentRuntime(cfg)
        runtime._get_cost_tracker("s1").record(
            "claude-opus-4", prompt_tokens=10000, completion_tokens=10000
        )

        events = []
        runtime._emit_event = lambda **kwargs: events.append(kwargs)

        with pytest.raises(BudgetExceededError):
            runtime._check_budget(session_id="s1", task_id="t1")
        # Audit event should have been emitted before the raise
        assert len(events) == 1
        assert events[0]["event_type"] == "agent.budget.exceeded"
        assert events[0]["result"] == "deny"


# ---------------------------------------------------------------------------
# SR-3.4: budget must be checked BEFORE a paid provider call, not only
# after -- otherwise every request past the cap incurs another paid
# provider call before being denied (a billing-control bug worse than a
# clean lockout).
# ---------------------------------------------------------------------------


class TestBudgetCheckedBeforePaidCall:
    def test_tool_loop_denies_before_paid_call_when_already_over_budget(self):
        """The core SR-3.4 reproduction: once accumulated cost from prior
        calls has already crossed max_spend_usd, the NEXT provider call
        must never happen at all."""
        provider = _make_provider()
        provider.accepts_message_dicts = False
        provider.complete_with_tools = MagicMock()

        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_iterations=5, max_spend_usd=0.01)
        runtime = AgentRuntime(cfg)
        # Simulate: budget was already exhausted by prior calls in this
        # session (CostTracker persists across calls within a session).
        runtime._get_cost_tracker("s").record(
            "claude-opus-4", prompt_tokens=10_000, completion_tokens=10_000
        )

        with pytest.raises(BudgetExceededError):
            runtime._tool_loop(
                provider=provider,
                tools=[],
                system_prompt="",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s",
                task_id="t",
            )

        provider.complete_with_tools.assert_not_called()

    def test_single_turn_denies_before_paid_call_when_already_over_budget(self):
        """_single_turn() previously never checked budget at all, in
        either direction -- used both directly and as _tool_loop's
        fallback when a provider doesn't implement complete_with_tools."""
        provider = _make_provider()
        provider.complete = MagicMock()

        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_spend_usd=0.01)
        runtime = AgentRuntime(cfg)
        runtime._get_cost_tracker("s").record(
            "claude-opus-4", prompt_tokens=10_000, completion_tokens=10_000
        )

        with pytest.raises(BudgetExceededError):
            runtime._single_turn(
                provider=provider,
                system_prompt="",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s",
                task_id="t",
            )

        provider.complete.assert_not_called()

    def test_tool_loop_proceeds_normally_under_budget(self):
        from missy.providers.base import CompletionResponse

        provider = _make_provider()
        provider.accepts_message_dicts = False
        provider.complete_with_tools = MagicMock(
            return_value=CompletionResponse(
                content="done",
                model="m",
                provider="fake",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
                raw={},
                finish_reason="stop",
                tool_calls=None,
            )
        )

        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_iterations=3, max_spend_usd=100.0)
        runtime = AgentRuntime(cfg)

        result_text, tools_used = runtime._tool_loop(
            provider=provider,
            tools=[],
            system_prompt="",
            messages=[{"role": "user", "content": "hi"}],
            session_id="s",
            task_id="t",
        )

        assert result_text == "done"
        provider.complete_with_tools.assert_called_once()

    def test_single_turn_proceeds_normally_under_budget(self):
        provider = _make_provider(reply="hello there")

        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_spend_usd=100.0)
        runtime = AgentRuntime(cfg)

        result = runtime._single_turn(
            provider=provider,
            system_prompt="",
            messages=[{"role": "user", "content": "hi"}],
            session_id="s",
            task_id="t",
        )

        assert result.content == "hello there"
        provider.complete.assert_called_once()

    def test_second_sessions_paid_call_proceeds_despite_first_sessions_exhausted_budget(
        self,
    ):
        """End-to-end through the real dispatch path (not just
        _check_budget directly): session A's exhausted budget must not
        block session B's provider call on the same shared runtime."""
        provider = _make_provider(reply="session B response")
        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_spend_usd=0.01)
        runtime = AgentRuntime(cfg)
        runtime._get_cost_tracker("session-a").record(
            "claude-opus-4", prompt_tokens=10_000, completion_tokens=10_000
        )

        with pytest.raises(BudgetExceededError):
            runtime._single_turn(
                provider=provider,
                system_prompt="",
                messages=[{"role": "user", "content": "hi"}],
                session_id="session-a",
                task_id="t",
            )

        result = runtime._single_turn(
            provider=provider,
            system_prompt="",
            messages=[{"role": "user", "content": "hi"}],
            session_id="session-b",
            task_id="t",
        )
        assert result.content == "session B response"
        provider.complete.assert_called_once()

    def test_unlimited_budget_never_blocks(self):
        """max_spend_usd=0.0 (the default) means unlimited -- no check
        should ever deny regardless of accumulated cost."""
        provider = _make_provider()
        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_spend_usd=0.0)
        runtime = AgentRuntime(cfg)
        runtime._get_cost_tracker("s").record(
            "claude-opus-4", prompt_tokens=1_000_000, completion_tokens=1_000_000
        )

        result = runtime._single_turn(
            provider=provider,
            system_prompt="",
            messages=[{"role": "user", "content": "hi"}],
            session_id="s",
            task_id="t",
        )
        provider.complete.assert_called_once()
        assert result is not None


# ---------------------------------------------------------------------------
# SR-3.4 (cross-session-aggregation sub-finding): max_spend_usd is
# documented as a per-session cap, so a single shared AgentRuntime serving
# many logically distinct sessions (every Discord user, every Web API
# session) must not let one session's spend count against another
# session's budget.
# ---------------------------------------------------------------------------


class TestCostTrackerCrossSessionIsolation:
    def test_one_sessions_spend_does_not_deny_another_session(self):
        """The core cross-session reproduction: session A exhausts its
        budget; session B (sharing the same runtime instance) must still
        be allowed to proceed."""
        cfg = AgentConfig(max_spend_usd=0.01)
        runtime = AgentRuntime(cfg)

        runtime._get_cost_tracker("session-a").record(
            "claude-opus-4", prompt_tokens=10_000, completion_tokens=10_000
        )

        with pytest.raises(BudgetExceededError):
            runtime._check_budget(session_id="session-a", task_id="t1")

        # Session B has recorded nothing and must not be denied.
        runtime._check_budget(session_id="session-b", task_id="t2")  # must not raise

    def test_sessions_have_independent_total_cost(self):
        cfg = AgentConfig(max_spend_usd=0.0)
        runtime = AgentRuntime(cfg)

        runtime._get_cost_tracker("session-a").record(
            "claude-opus-4", prompt_tokens=1000, completion_tokens=1000
        )
        runtime._get_cost_tracker("session-b").record(
            "claude-haiku-4", prompt_tokens=10, completion_tokens=10
        )

        assert runtime._get_cost_tracker("session-a").total_cost_usd > 0
        assert runtime._get_cost_tracker("session-b").total_cost_usd > 0
        assert (
            runtime._get_cost_tracker("session-a").total_cost_usd
            != runtime._get_cost_tracker("session-b").total_cost_usd
        )

    def test_same_session_id_returns_same_tracker_instance(self):
        """Repeated lookups for the same session must accumulate onto the
        same tracker, not silently reset."""
        cfg = AgentConfig(max_spend_usd=0.0)
        runtime = AgentRuntime(cfg)

        tracker1 = runtime._get_cost_tracker("session-a")
        tracker1.record("claude-opus-4", prompt_tokens=1000, completion_tokens=1000)
        tracker2 = runtime._get_cost_tracker("session-a")

        assert tracker1 is tracker2
        assert tracker2.total_cost_usd > 0

    def test_peek_cost_tracker_does_not_create_entry(self):
        """_peek_cost_tracker() must not fabricate a tracker for a session
        that never recorded any cost -- it's a read-only lookup used by
        display/audit sites."""
        cfg = AgentConfig(max_spend_usd=0.0)
        runtime = AgentRuntime(cfg)

        assert runtime._peek_cost_tracker("never-used-session") is None
        # Confirm the peek truly didn't create an entry.
        assert "never-used-session" not in runtime._cost_trackers

    def test_disabling_cost_tracking_applies_to_all_sessions(self):
        cfg = AgentConfig(max_spend_usd=1.0)
        runtime = AgentRuntime(cfg)
        runtime._cost_tracking_enabled = False

        assert runtime._get_cost_tracker("session-a") is None
        assert runtime._get_cost_tracker("session-b") is None
        # Should not raise even under a configured budget, since tracking
        # is fully disabled.
        runtime._check_budget(session_id="session-a")

    def test_tracked_session_count_is_bounded(self):
        """A long-running shared runtime must not grow its per-session
        tracker dict without bound."""
        cfg = AgentConfig(max_spend_usd=0.0)
        runtime = AgentRuntime(cfg)
        runtime._MAX_TRACKED_SESSIONS = 5

        for i in range(10):
            runtime._get_cost_tracker(f"session-{i}")

        assert len(runtime._cost_trackers) <= 5
        # The most recently created session must still be present.
        assert "session-9" in runtime._cost_trackers


# ---------------------------------------------------------------------------
# Checkpoint recovery scan
# ---------------------------------------------------------------------------


class TestCheckpointRecoveryScan:
    def test_scan_returns_empty_when_no_db(self):
        """When checkpoint module is available but DB doesn't exist, returns []."""
        results = AgentRuntime._scan_checkpoints()
        assert isinstance(results, list)

    def test_pending_recovery_property(self):
        cfg = AgentConfig()
        runtime = AgentRuntime(cfg)
        # Should return a list (likely empty in test env)
        assert isinstance(runtime.pending_recovery, list)

    def test_pending_recovery_is_copy(self):
        cfg = AgentConfig()
        runtime = AgentRuntime(cfg)
        r1 = runtime.pending_recovery
        r2 = runtime.pending_recovery
        assert r1 is not r2  # Should be a copy


# ---------------------------------------------------------------------------
# MissyConfig.max_spend_usd
# ---------------------------------------------------------------------------


class TestMissyConfigMaxSpend:
    def test_default_config_has_max_spend(self):
        from missy.config.settings import get_default_config

        cfg = get_default_config()
        assert cfg.max_spend_usd == 0.0

    def test_load_config_parses_max_spend(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "network:\n  default_deny: true\n"
            "filesystem:\n  allowed_read_paths: []\n  allowed_write_paths: []\n"
            "shell:\n  enabled: false\n  allowed_commands: []\n"
            "plugins:\n  enabled: false\n  allowed_plugins: []\n"
            "providers: {}\n"
            "workspace_path: /tmp\n"
            "audit_log_path: /tmp/audit.jsonl\n"
            "max_spend_usd: 3.50\n"
        )
        from missy.config.settings import load_config

        cfg = load_config(str(config_file))
        assert cfg.max_spend_usd == 3.50
