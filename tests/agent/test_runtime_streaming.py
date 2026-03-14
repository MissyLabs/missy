"""Tests for AgentRuntime streaming and rate limiting integration."""

from unittest.mock import MagicMock, patch

import pytest

from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.providers.base import CompletionResponse


@pytest.fixture
def mock_registry():
    """Set up a mock provider registry."""
    provider = MagicMock()
    provider.name = "test"
    provider.is_available.return_value = True
    provider.complete.return_value = CompletionResponse(
        content="Hello!",
        model="test-model",
        provider="test",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        raw={},
    )
    provider.stream.return_value = iter(["Hel", "lo", "!"])

    registry = MagicMock()
    registry.get.return_value = provider
    registry.get_available.return_value = [provider]

    return registry, provider


class TestRunStream:
    def test_run_stream_yields_chunks(self, mock_registry):
        registry, provider = mock_registry
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            agent = AgentRuntime(AgentConfig(provider="test"))
            chunks = list(agent.run_stream("Hello"))
            assert len(chunks) == 3
            assert "".join(chunks) == "Hello!"

    def test_run_stream_falls_back_on_error(self, mock_registry):
        registry, provider = mock_registry
        provider.stream.side_effect = Exception("Stream failed")
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            agent = AgentRuntime(AgentConfig(provider="test"))
            chunks = list(agent.run_stream("Hello"))
            assert len(chunks) == 1
            assert chunks[0] == "Hello!"

    def test_run_stream_falls_back_to_run_with_tools(self, mock_registry):
        registry, provider = mock_registry
        tool_registry = MagicMock()
        tool_registry.list_tools.return_value = ["calculator"]
        tool = MagicMock()
        tool.name = "calculator"
        tool_registry.get.return_value = tool
        provider.complete_with_tools.return_value = CompletionResponse(
            content="Result: 4",
            model="test-model",
            provider="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw={},
            finish_reason="stop",
        )

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", return_value=tool_registry),
        ):
            agent = AgentRuntime(AgentConfig(provider="test", max_iterations=5))
            chunks = list(agent.run_stream("What is 2+2?"))
            # Falls back to run() which returns full response as single chunk
            assert len(chunks) == 1
            assert "Result: 4" in chunks[0]


class TestRateLimitIntegration:
    def test_rate_limiter_created(self, mock_registry):
        registry, _ = mock_registry
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            agent = AgentRuntime(AgentConfig(provider="test"))
            assert agent._rate_limiter is not None

    def test_rate_limiter_called_before_completion(self, mock_registry):
        registry, provider = mock_registry
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            agent = AgentRuntime(AgentConfig(provider="test"))
            rl_mock = MagicMock()
            agent._rate_limiter = rl_mock
            agent.run("Hello")
            rl_mock.acquire.assert_called()


class TestCostPersistence:
    def test_record_cost_passes_session_id(self, mock_registry):
        registry, provider = mock_registry
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.get_tool_registry", side_effect=RuntimeError),
        ):
            agent = AgentRuntime(AgentConfig(provider="test"))
            memory_store = MagicMock()
            memory_store.get_session_turns.return_value = []
            agent._memory_store = memory_store

            response = CompletionResponse(
                content="test",
                model="test-model",
                provider="test",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                raw={},
            )
            agent._record_cost(response, session_id="sess-123")
            # Cost tracker should have recorded
            assert agent._cost_tracker.call_count == 1
