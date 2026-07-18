"""F10 — CondenserPipeline wiring into context building."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.agent.runtime import AgentConfig, AgentRuntime


def _make_runtime(**config_kwargs) -> AgentRuntime:
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
        runtime = AgentRuntime(AgentConfig(provider="fake", **config_kwargs))
    runtime._memory_store = None
    return runtime


def _msgs(n: int) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


class TestConfigDefaults:
    def test_flag_defaults_false(self) -> None:
        assert AgentConfig().condenser_pipeline_enabled is False
        assert AgentConfig().condenser_min_messages == 30


class TestMaybeCondense:
    def test_disabled_returns_unchanged(self) -> None:
        runtime = _make_runtime()  # flag off
        msgs = _msgs(100)
        sys, out = runtime._maybe_condense_context("sys", msgs)
        assert out is msgs  # untouched

    def test_short_conversation_not_condensed(self) -> None:
        runtime = _make_runtime(condenser_pipeline_enabled=True, condenser_min_messages=30)
        msgs = _msgs(10)  # below threshold
        _, out = runtime._maybe_condense_context("sys", msgs)
        assert out is msgs

    def test_long_conversation_is_condensed(self) -> None:
        runtime = _make_runtime(condenser_pipeline_enabled=True, condenser_min_messages=20)
        msgs = _msgs(200)
        _, out = runtime._maybe_condense_context("sys", msgs)
        # The window stage caps the list, so it must shrink.
        assert len(out) < len(msgs)

    def test_condenser_failure_falls_back(self) -> None:
        runtime = _make_runtime(condenser_pipeline_enabled=True, condenser_min_messages=1)
        msgs = _msgs(50)
        with patch(
            "missy.agent.condensers.create_default_pipeline", side_effect=RuntimeError("boom")
        ):
            _, out = runtime._maybe_condense_context("sys", msgs)
        assert out is msgs  # unchanged on failure

    def test_system_prompt_preserved(self) -> None:
        runtime = _make_runtime(condenser_pipeline_enabled=True, condenser_min_messages=20)
        sys, _ = runtime._maybe_condense_context("IMPORTANT SYSTEM", _msgs(200))
        assert sys == "IMPORTANT SYSTEM"


class TestWrapperIntegration:
    def test_build_context_messages_applies_condense(self) -> None:
        runtime = _make_runtime(condenser_pipeline_enabled=True, condenser_min_messages=20)
        # Stub the raw builder to return a long list.
        long = _msgs(200)
        with patch.object(runtime, "_build_context_messages_raw", return_value=("sys", long)):
            _, out = runtime._build_context_messages("hi", [])
        assert len(out) < len(long)

    def test_build_context_messages_noop_when_disabled(self) -> None:
        runtime = _make_runtime()  # disabled
        long = _msgs(200)
        with patch.object(runtime, "_build_context_messages_raw", return_value=("sys", long)):
            _, out = runtime._build_context_messages("hi", [])
        assert out is long
