"""Tests for F05 — complexity-based model-tier routing via ModelRouter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.config.settings import ProviderConfig


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
    cfg = AgentConfig(provider="fake", **config_kwargs)
    with (
        patch("missy.agent.runtime.get_registry", return_value=reg),
        patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError("no tools")),
        patch("missy.agent.runtime.get_message_bus", side_effect=RuntimeError("no bus")),
    ):
        runtime = AgentRuntime(cfg)
    runtime._memory_store = None
    runtime._cost_tracking_enabled = False
    return runtime


def _registry_with_config(pconfig: ProviderConfig) -> MagicMock:
    reg = MagicMock()
    reg.get_config.return_value = pconfig
    return reg


class TestConfigDefault:
    def test_flag_defaults_false(self) -> None:
        assert AgentConfig().model_routing_enabled is False


class TestDisabled:
    def test_returns_config_model_unchanged_when_disabled(self) -> None:
        runtime = _make_runtime(model="claude-primary")  # routing off
        assert runtime._route_model("anything", 0, 0) == "claude-primary"

    def test_returns_none_when_no_config_model_and_disabled(self) -> None:
        runtime = _make_runtime()  # model=None, routing off
        assert runtime._route_model("x") is None


class TestEnabledRouting:
    def _run_route(self, runtime, prompt, **kw):
        pconfig = ProviderConfig(
            name="fake",
            model="primary-model",
            fast_model="fast-model",
            premium_model="premium-model",
        )
        with patch(
            "missy.providers.registry.get_registry", return_value=_registry_with_config(pconfig)
        ):
            return runtime._route_model(prompt, **kw)

    def test_fast_tier_for_short_indicator_prompt(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        # Short + a FAST_INDICATOR word ("what") + no tools -> fast tier.
        assert self._run_route(runtime, "what is it?") == "fast-model"

    def test_premium_tier_for_complex_prompt(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        # A PREMIUM_KEYWORD ("refactor") -> premium tier.
        assert self._run_route(runtime, "please refactor this module") == "premium-model"

    def test_premium_tier_for_deep_history(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        assert self._run_route(runtime, "continue", history_length=15) == "premium-model"

    def test_primary_tier_for_ordinary_prompt(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        assert self._run_route(runtime, "summarize the meeting notes for me") == "primary-model"

    def test_falls_back_to_primary_when_no_tiered_models(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        pconfig = ProviderConfig(name="fake", model="primary-model")  # no fast/premium
        with patch(
            "missy.providers.registry.get_registry",
            return_value=_registry_with_config(pconfig),
        ):
            # Even a "fast" prompt gets the primary model (select_model fallback).
            assert runtime._route_model("what is it?") == "primary-model"

    def test_missing_provider_config_falls_back_to_default(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        reg = MagicMock()
        reg.get_config.return_value = None
        with patch("missy.providers.registry.get_registry", return_value=reg):
            assert runtime._route_model("what is it?") == "primary-model"

    def test_routing_error_falls_back_to_default(self) -> None:
        runtime = _make_runtime(model="primary-model", model_routing_enabled=True)
        with patch("missy.providers.registry.get_registry", side_effect=RuntimeError("boom")):
            assert runtime._route_model("what is it?") == "primary-model"
