"""F06 — heartbeat gateway wiring seam.

Verifies the integration contract gateway_start() uses to wire HeartbeatRunner
to the main AgentRuntime: a run callback of the shape
``lambda prompt: str(runtime.run(prompt))`` correctly drives a real agent run
with the HEARTBEAT.md checklist and captures the result.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.agent.heartbeat import HEARTBEAT_FILE, HeartbeatRunner
from missy.agent.runtime import AgentConfig, AgentRuntime


def _make_runtime() -> AgentRuntime:
    from missy.providers.base import CompletionResponse

    provider = MagicMock()
    provider.name = "fake"
    provider.is_available.return_value = True
    provider.complete.return_value = CompletionResponse(
        content="heartbeat done", model="m", provider="fake", usage={}, raw={}, finish_reason="stop"
    )
    reg = MagicMock()
    reg.get.return_value = provider
    reg.get_available.return_value = [provider]
    with (
        patch("missy.agent.runtime.get_registry", return_value=reg),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(AgentConfig(provider="fake", max_iterations=1))
    runtime._memory_store = None
    runtime._cost_tracking_enabled = False
    return runtime


class TestHeartbeatRunCallback:
    def test_callback_drives_agent_run(self, tmp_path: Path) -> None:
        # A real runtime whose .run is stubbed to a string (its full provider
        # machinery isn't needed to validate the wiring seam).
        runtime = _make_runtime()
        runtime.run = MagicMock(return_value="heartbeat done")  # type: ignore[method-assign]
        captured: dict = {}

        def _heartbeat_run(prompt: str) -> str:
            captured["prompt"] = prompt
            return str(runtime.run(prompt))

        (tmp_path / HEARTBEAT_FILE).write_text("- check disk usage\n- verify services")
        results: list[str] = []
        runner = HeartbeatRunner(
            agent_run_fn=_heartbeat_run,
            interval_seconds=9999,
            workspace=str(tmp_path),
            report_fn=results.append,
        )
        runner._fire()

        # The agent ran with the heartbeat checklist embedded in the prompt.
        assert "check disk usage" in captured["prompt"]
        assert "[HEARTBEAT CHECK]" in captured["prompt"]
        # And the run's result (a string) was captured/reported.
        assert results and isinstance(results[0], str)
        assert runner._runs == 1

    def test_callback_returns_string_for_report(self, tmp_path: Path) -> None:
        # str(runtime.run(...)) must always yield a str even if run returns a
        # non-str — the report_fn/HeartbeatRunner expect a string.
        runtime = MagicMock()
        runtime.run.return_value = 12345  # deliberately not a string

        def _heartbeat_run(prompt: str) -> str:
            return str(runtime.run(prompt))

        (tmp_path / HEARTBEAT_FILE).write_text("do the thing")
        reported: list[str] = []
        runner = HeartbeatRunner(
            agent_run_fn=_heartbeat_run,
            interval_seconds=9999,
            workspace=str(tmp_path),
            report_fn=reported.append,
        )
        runner._fire()
        assert reported == ["12345"]

    def test_disabled_by_default_config(self) -> None:
        # The gateway gate reads heartbeat.enabled; the default is False.
        from missy.config.settings import HeartbeatConfig

        assert HeartbeatConfig().enabled is False
