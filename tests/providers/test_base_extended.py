"""Tests for missy.providers.base covering uncovered paths."""

from __future__ import annotations

from missy.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    ToolCall,
    ToolResult,
)


class StubProvider(BaseProvider):
    """Minimal provider for testing base class methods."""

    name = "stub"

    def __init__(self, response_text="hello"):
        self._response_text = response_text

    def complete(self, messages, **kwargs):
        return CompletionResponse(
            content=self._response_text,
            model="stub-1",
            provider="stub",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw={},
        )

    def is_available(self):
        return True


class TestMessage:
    def test_message_fields(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"


class TestToolCall:
    def test_tool_call_fields(self):
        tc = ToolCall(id="tc1", name="calculator", arguments={"expr": "1+1"})
        assert tc.id == "tc1"
        assert tc.name == "calculator"
        assert tc.arguments == {"expr": "1+1"}


class TestToolResult:
    def test_default_is_error_false(self):
        tr = ToolResult(tool_call_id="tc1", name="calc", content="2")
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(tool_call_id="tc1", name="calc", content="err", is_error=True)
        assert tr.is_error is True


class TestCompletionResponse:
    def test_default_finish_reason(self):
        cr = CompletionResponse(content="x", model="m", provider="p", usage={}, raw={})
        assert cr.finish_reason == "stop"
        assert cr.tool_calls == []

    def test_with_tool_calls(self):
        tc = ToolCall(id="1", name="t", arguments={})
        cr = CompletionResponse(
            content="",
            model="m",
            provider="p",
            usage={},
            raw={},
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        assert len(cr.tool_calls) == 1
        assert cr.finish_reason == "tool_calls"


class TestBaseProviderDefaultMethods:
    def test_get_tool_schema(self):
        class FakeTool:
            name = "echo"
            description = "Echo text"

            def get_schema(self):
                return {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                }

        provider = StubProvider()
        schemas = provider.get_tool_schema([FakeTool()])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"
        assert "text" in schemas[0]["parameters"]["properties"]

    def test_get_tool_schema_no_get_schema_method(self):
        class BareObj:
            name = "bare"
            description = "No schema"

        provider = StubProvider()
        schemas = provider.get_tool_schema([BareObj()])
        assert schemas[0]["name"] == "bare"
        assert schemas[0]["parameters"] == {}

    def test_complete_with_tools_defaults_to_complete(self):
        provider = StubProvider(response_text="fallback")
        messages = [Message(role="user", content="hi")]
        result = provider.complete_with_tools(messages, tools=[])
        assert result.content == "fallback"

    def test_stream_yields_full_content(self):
        provider = StubProvider(response_text="streamed")
        messages = [Message(role="user", content="hi")]
        chunks = list(provider.stream(messages))
        assert chunks == ["streamed"]

    def test_repr(self):
        provider = StubProvider()
        assert "stub" in repr(provider)
