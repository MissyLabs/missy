"""Tests for missy.tools.benchmark.mock_provider."""

from __future__ import annotations

from missy.providers.base import Message
from missy.tools.benchmark.mock_provider import MockToolProvider
from missy.tools.builtin.calculator import CalculatorTool


class TestMockToolProviderBasics:
    def test_is_available(self) -> None:
        assert MockToolProvider().is_available() is True

    def test_name(self) -> None:
        assert MockToolProvider().name == "mock"

    def test_complete_echoes_input(self) -> None:
        provider = MockToolProvider()
        response = provider.complete([Message(role="user", content="hello there")])
        assert "hello there" in response.content
        assert response.provider == "mock"
        assert response.tool_calls == []


class TestMockToolProviderWithTools:
    def test_calls_the_only_tool_offered(self) -> None:
        provider = MockToolProvider()
        response = provider.complete_with_tools(
            [Message(role="user", content="please compute '2 + 2'")],
            tools=[CalculatorTool()],
        )
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "calculator"

    def test_extracts_quoted_string_argument(self) -> None:
        provider = MockToolProvider()
        response = provider.complete_with_tools(
            [Message(role="user", content="please compute '3 * 9'")],
            tools=[CalculatorTool()],
        )
        assert response.tool_calls[0].arguments["expression"] == "3 * 9"

    def test_no_tools_offered_returns_no_call(self) -> None:
        provider = MockToolProvider()
        response = provider.complete_with_tools(
            [Message(role="user", content="do something")],
            tools=[],
        )
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_call_tool_false_never_calls(self) -> None:
        provider = MockToolProvider(call_tool=False)
        response = provider.complete_with_tools(
            [Message(role="user", content="please compute '2 + 2'")],
            tools=[CalculatorTool()],
        )
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_usage_reflects_prompt_length(self) -> None:
        provider = MockToolProvider()
        response = provider.complete_with_tools(
            [Message(role="user", content="one two three four five")],
            tools=[CalculatorTool()],
            system="",
        )
        assert response.usage["prompt_tokens"] == 5
