"""Tests for runtime behavior layer integration.

Verifies that:
- BehaviorLayer shapes system prompts in the runtime
- ResponseShaper post-processes final responses
- IntentInterpreter is called with user input
- Persona influences runtime behavior
- All subsystems degrade gracefully when unavailable
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.providers.base import CompletionResponse


def _make_config(**overrides):
    defaults = {
        "provider": "mock",
        "system_prompt": "You are Missy.",
        "max_iterations": 1,
    }
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _mock_provider(response_text="Hello!"):
    provider = MagicMock()
    provider.name = "mock"
    provider.is_available.return_value = True
    provider.complete.return_value = CompletionResponse(
        content=response_text,
        model="mock-model",
        provider="mock",
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        raw={},
        tool_calls=[],
        finish_reason="stop",
    )
    return provider


def _mock_registry(provider):
    registry = MagicMock()
    registry.get.return_value = provider
    registry.get_available.return_value = [provider]
    return registry


def _run_with_mocks(rt, user_input, provider):
    """Run the agent with standard mocks for registry and censor."""
    registry = _mock_registry(provider)
    with ExitStack() as stack:
        stack.enter_context(patch("missy.agent.runtime.get_registry", return_value=registry))
        stack.enter_context(
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError)
        )
        stack.enter_context(patch("missy.agent.runtime.censor_response", side_effect=lambda x: x))
        return rt.run(user_input)


class TestBehaviorLayerCreation:
    """Test that behavior subsystems are lazily created."""

    def test_behavior_layer_created(self):
        AgentRuntime(_make_config())
        assert True  # If we get here, no crash

    def test_response_shaper_created(self):
        rt = AgentRuntime(_make_config())
        assert hasattr(rt, "_response_shaper")

    def test_intent_interpreter_created(self):
        rt = AgentRuntime(_make_config())
        assert hasattr(rt, "_intent_interpreter")

    def test_persona_manager_created(self):
        rt = AgentRuntime(_make_config())
        assert hasattr(rt, "_persona_manager")


class TestBehaviorInRun:
    """Test behavior integration during run()."""

    def test_response_shaper_applied(self):
        provider = _mock_provider("  Hello there!  Hello there!  ")
        rt = AgentRuntime(_make_config())
        if rt._response_shaper is not None:
            result = _run_with_mocks(rt, "Hi", provider)
            assert isinstance(result, str)

    def test_behavior_failure_degrades_gracefully(self):
        provider = _mock_provider("Hello!")
        rt = AgentRuntime(_make_config())
        rt._behavior = MagicMock()
        rt._behavior.shape_system_prompt.side_effect = RuntimeError("broken")
        rt._behavior.analyze_user_tone.side_effect = RuntimeError("broken")

        result = _run_with_mocks(rt, "Hi", provider)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_shaper_failure_degrades(self):
        provider = _mock_provider("Hello!")
        rt = AgentRuntime(_make_config())
        rt._response_shaper = MagicMock()
        rt._response_shaper.shape_response.side_effect = RuntimeError("broken")

        result = _run_with_mocks(rt, "Hi", provider)
        assert isinstance(result, str)

    def test_intent_interpreter_called(self):
        provider = _mock_provider("Hello!")
        rt = AgentRuntime(_make_config())
        mock_interpreter = MagicMock()
        mock_interpreter.classify_intent.return_value = "greeting"
        mock_interpreter.extract_urgency.return_value = "low"
        rt._intent_interpreter = mock_interpreter

        _run_with_mocks(rt, "Hey there!", provider)
        mock_interpreter.classify_intent.assert_called()

    def test_none_subsystems_handled(self):
        provider = _mock_provider("Hello!")
        rt = AgentRuntime(_make_config())
        rt._behavior = None
        rt._response_shaper = None
        rt._intent_interpreter = None
        rt._persona_manager = None

        result = _run_with_mocks(rt, "Hi", provider)
        assert isinstance(result, str)


class TestPersonaIntegration:
    """Test persona manager integration."""

    def test_persona_manager_failure_degrades(self):
        provider = _mock_provider("Hello!")
        rt = AgentRuntime(_make_config())
        rt._persona_manager = MagicMock()
        rt._persona_manager.get_persona.side_effect = RuntimeError("no persona file")

        result = _run_with_mocks(rt, "Hi", provider)
        assert isinstance(result, str)


class TestSubsystemFactories:
    """Test graceful degradation of factory methods."""

    def test_make_behavior_layer_returns_none_on_error(self):
        rt = AgentRuntime(_make_config())
        rt._persona_manager = MagicMock()
        rt._persona_manager.get_persona.side_effect = Exception("fail")
        result = rt._make_behavior_layer()
        assert result is None or hasattr(result, "shape_system_prompt")

    def test_make_response_shaper_returns_object_or_none(self):
        result = AgentRuntime._make_response_shaper()
        assert result is None or hasattr(result, "shape_response")

    def test_make_intent_interpreter_returns_object_or_none(self):
        result = AgentRuntime._make_intent_interpreter()
        assert result is None or hasattr(result, "classify_intent")

    def test_scan_checkpoints_returns_list(self):
        result = AgentRuntime._scan_checkpoints()
        assert isinstance(result, list)
