"""Tests for missy.agent.runtime.AgentRuntime and AgentConfig."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.core.events import event_bus
from missy.core.exceptions import ProviderError
from missy.providers import registry as registry_module
from missy.providers.base import CompletionResponse, Message

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_provider(
    name: str = "fake",
    available: bool = True,
    reply: str = "test reply",
) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.is_available.return_value = available
    provider.complete.return_value = CompletionResponse(
        content=reply,
        model="fake-model-1",
        provider=name,
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        raw={},
    )
    return provider


def _make_registry(providers: dict | None = None) -> MagicMock:
    """Build a mock ProviderRegistry."""
    registry = MagicMock()
    providers = providers or {}

    def _get(name):
        return providers.get(name)

    def _get_available():
        return [p for p in providers.values() if p.is_available()]

    registry.get.side_effect = _get
    registry.get_available.side_effect = _get_available
    return registry


@pytest.fixture(autouse=True)
def clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture(autouse=True)
def reset_singleton():
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# AgentConfig defaults
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_default_provider_is_anthropic(self):
        config = AgentConfig()
        assert config.provider == "anthropic"

    def test_default_model_is_none(self):
        config = AgentConfig()
        assert config.model is None

    def test_default_system_prompt_is_set(self):
        config = AgentConfig()
        assert "Missy" in config.system_prompt

    def test_default_max_iterations(self):
        config = AgentConfig()
        assert config.max_iterations == 10

    def test_default_temperature(self):
        config = AgentConfig()
        assert config.temperature == 0.7

    def test_custom_values(self):
        config = AgentConfig(
            provider="openai",
            model="gpt-4o",
            system_prompt="Be terse.",
            max_iterations=5,
            temperature=0.0,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.system_prompt == "Be terse."
        assert config.max_iterations == 5
        assert config.temperature == 0.0


# ---------------------------------------------------------------------------
# AgentRuntime.run – success path
# ---------------------------------------------------------------------------


class TestAgentRuntimeRun:
    def test_run_returns_string_response(self):
        provider = _make_provider(reply="Hello, world!")
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            result = runtime.run("Hi")

        assert result == "Hello, world!"

    def test_run_returns_plain_string(self):
        provider = _make_provider(reply="response text")
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            result = runtime.run("question")

        assert isinstance(result, str)

    def test_run_emits_start_event(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime.run("Hello")

        events = event_bus.get_events(event_type="agent.run.start")
        assert len(events) >= 1

    def test_run_emits_complete_event(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime.run("Hello")

        events = event_bus.get_events(event_type="agent.run.complete")
        assert len(events) >= 1

    def test_run_passes_system_prompt(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake", system_prompt="Custom prompt."))
            runtime.run("question")

        call_args = provider.complete.call_args
        messages: list[Message] = call_args[0][0]
        system_messages = [m for m in messages if m.role == "system"]
        assert len(system_messages) == 1
        assert system_messages[0].content == "Custom prompt."

    def test_run_includes_user_input_in_messages(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime.run("user question here")

        call_args = provider.complete.call_args
        messages: list[Message] = call_args[0][0]
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "user question here"

    def test_run_forwards_temperature(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake", temperature=0.3))
            runtime.run("hi")

        call_kwargs = provider.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.3

    def test_run_forwards_model_when_set(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake", model="gpt-4o"))
            runtime.run("hi")

        call_kwargs = provider.complete.call_args[1]
        assert call_kwargs.get("model") == "gpt-4o"

    def test_run_does_not_set_model_when_none(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake", model=None))
            runtime.run("hi")

        call_kwargs = provider.complete.call_args[1]
        assert "model" not in call_kwargs

    def test_run_reuses_existing_thread_session(self):
        provider = _make_provider()
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            runtime.run("first")
            runtime.run("second")

        # Both calls share the same session (session events should reference same sid)
        start_events = event_bus.get_events(event_type="agent.run.start")
        assert len(start_events) == 2
        assert start_events[0].session_id == start_events[1].session_id


# ---------------------------------------------------------------------------
# AgentRuntime – provider resolution fallback
# ---------------------------------------------------------------------------


class TestProviderResolution:
    def test_fallback_to_available_provider(self):
        """If configured provider unavailable, fallback to any available one."""
        primary = _make_provider("primary", available=False)
        fallback = _make_provider("fallback", available=True, reply="fallback reply")
        mock_registry = _make_registry({"primary": primary, "fallback": fallback})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="primary"))
            result = runtime.run("hi")

        assert result == "fallback reply"

    def test_no_providers_available_raises_provider_error(self):
        mock_registry = _make_registry(providers={})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="missing"))
            with pytest.raises(ProviderError):
                runtime.run("hi")

    def test_no_providers_emits_error_event(self):
        mock_registry = _make_registry(providers={})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="missing"))
            with pytest.raises(ProviderError):
                runtime.run("hi")

        events = event_bus.get_events(event_type="agent.run.error")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# AgentRuntime – provider completion errors
# ---------------------------------------------------------------------------


class TestCompletionErrors:
    def test_provider_error_propagates(self):
        provider = _make_provider()
        provider.complete.side_effect = ProviderError("API down")
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            with pytest.raises(ProviderError, match="API down"):
                runtime.run("hi")

    def test_provider_error_emits_error_event(self):
        provider = _make_provider()
        provider.complete.side_effect = ProviderError("API down")
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            with pytest.raises(ProviderError):
                runtime.run("hi")

        events = event_bus.get_events(event_type="agent.run.error")
        assert len(events) >= 1

    def test_unexpected_exception_wrapped_in_provider_error(self):
        provider = _make_provider()
        provider.complete.side_effect = ValueError("unexpected crash")
        mock_registry = _make_registry({"fake": provider})

        with patch("missy.agent.runtime.get_registry", return_value=mock_registry):
            runtime = AgentRuntime(AgentConfig(provider="fake"))
            with pytest.raises(ProviderError, match="Unexpected"):
                runtime.run("hi")
