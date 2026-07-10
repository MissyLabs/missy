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
        assert runtime._cost_tracker is not None
        assert runtime._cost_tracker.max_spend_usd == 2.50

    def test_cost_tracker_default_unlimited(self):
        cfg = AgentConfig()
        runtime = AgentRuntime(cfg)
        assert runtime._cost_tracker is not None
        assert runtime._cost_tracker.max_spend_usd == 0.0


# ---------------------------------------------------------------------------
# Budget enforcement (_check_budget)
# ---------------------------------------------------------------------------


class TestCheckBudget:
    def test_check_budget_no_tracker(self):
        cfg = AgentConfig()
        runtime = AgentRuntime(cfg)
        runtime._cost_tracker = None
        # Should not raise
        runtime._check_budget()

    def test_check_budget_under_limit(self):
        cfg = AgentConfig(max_spend_usd=10.0)
        runtime = AgentRuntime(cfg)
        runtime._cost_tracker.record("claude-sonnet-4", prompt_tokens=100, completion_tokens=50)
        # Should not raise
        runtime._check_budget()

    def test_check_budget_over_limit_raises(self):
        cfg = AgentConfig(max_spend_usd=0.001)
        runtime = AgentRuntime(cfg)
        # Record enough to exceed $0.001
        runtime._cost_tracker.record("claude-opus-4", prompt_tokens=10000, completion_tokens=10000)
        with pytest.raises(BudgetExceededError):
            runtime._check_budget(session_id="test", task_id="t1")

    def test_check_budget_emits_audit_event(self):
        cfg = AgentConfig(max_spend_usd=0.001)
        runtime = AgentRuntime(cfg)
        runtime._cost_tracker.record("claude-opus-4", prompt_tokens=10000, completion_tokens=10000)

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
        # session/runtime (CostTracker persists across calls).
        runtime._cost_tracker.record(
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
        runtime._cost_tracker.record(
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

    def test_unlimited_budget_never_blocks(self):
        """max_spend_usd=0.0 (the default) means unlimited -- no check
        should ever deny regardless of accumulated cost."""
        provider = _make_provider()
        reg = _make_registry({"fake": provider})
        registry_module._registry = reg

        cfg = AgentConfig(provider="fake", max_spend_usd=0.0)
        runtime = AgentRuntime(cfg)
        runtime._cost_tracker.record(
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
