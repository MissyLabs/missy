"""F19 — global budget singleton + runtime enforcement wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.cost_tracker import BudgetExceededError
from missy.agent.global_budget import (
    get_global_budget,
    init_global_budget,
)
from missy.agent.runtime import AgentConfig, AgentRuntime


@pytest.fixture(autouse=True)
def _reset_singleton():
    # Ensure a clean global-budget singleton around each test.
    import missy.agent.global_budget as gb

    gb._ACTIVE = None
    yield
    gb._ACTIVE = None


def _make_runtime() -> AgentRuntime:
    from missy.providers.base import CompletionResponse

    provider = MagicMock()
    provider.name = "fake"
    provider.is_available.return_value = True
    provider.complete.return_value = CompletionResponse(
        content="ok", model="m", provider="fake", usage={}, raw={}, finish_reason="stop"
    )
    reg = MagicMock()
    reg.get.return_value = provider
    reg.get_available.return_value = [provider]
    with (
        patch("missy.agent.runtime.get_registry", return_value=reg),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(AgentConfig(provider="fake"))
    return runtime


class TestSingleton:
    def test_default_is_disabled_noop(self) -> None:
        b = get_global_budget()
        assert b.enabled is False

    def test_init_installs_active(self, tmp_path: Path) -> None:
        init_global_budget(1.0, path=str(tmp_path / "gb.json"))
        assert get_global_budget().enabled is True

    def test_reinit_replaces(self, tmp_path: Path) -> None:
        init_global_budget(1.0, path=str(tmp_path / "gb.json"))
        init_global_budget(0.0, path=str(tmp_path / "gb.json"))
        assert get_global_budget().enabled is False


class TestRuntimeEnforcement:
    def test_check_budget_raises_on_global_breach(self, tmp_path: Path) -> None:
        p = str(tmp_path / "gb.json")
        b = init_global_budget(0.01, path=p)
        b.record(0.02)  # over the ceiling
        runtime = _make_runtime()
        runtime._memory_store = None
        with pytest.raises(BudgetExceededError):
            runtime._check_budget(session_id="s1")

    def test_check_budget_ok_below_global(self, tmp_path: Path) -> None:
        p = str(tmp_path / "gb.json")
        b = init_global_budget(1.0, path=p)
        b.record(0.1)
        runtime = _make_runtime()
        runtime._memory_store = None
        runtime._cost_tracking_enabled = False
        runtime._check_budget(session_id="s1")  # no raise

    def test_disabled_global_never_blocks(self, tmp_path: Path) -> None:
        # No init -> disabled singleton -> _check_budget must not raise on it.
        runtime = _make_runtime()
        runtime._memory_store = None
        runtime._cost_tracking_enabled = False
        runtime._check_budget(session_id="s1")

    def test_record_cost_feeds_global(self, tmp_path: Path) -> None:
        p = str(tmp_path / "gb.json")
        init_global_budget(1.0, path=p)
        runtime = _make_runtime()
        runtime._memory_store = None
        # A cost tracker whose record_from_response yields a rec with cost.
        rec = MagicMock()
        rec.cost_usd = 0.25
        rec.model = "m"
        rec.prompt_tokens = 1
        rec.completion_tokens = 1
        tracker = MagicMock()
        tracker.record_from_response.return_value = rec
        with patch.object(runtime, "_get_cost_tracker", return_value=tracker):
            runtime._record_cost(MagicMock(), session_id="s1")
        assert get_global_budget().total_spent() == pytest.approx(0.25)
