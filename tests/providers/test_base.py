"""Tests for missy.providers.base – Message, CompletionResponse, BaseProvider."""

from __future__ import annotations

import pytest

from missy.providers.base import BaseProvider, CompletionResponse, Message

# ---------------------------------------------------------------------------
# Concrete stub for abstract base
# ---------------------------------------------------------------------------


class StubProvider(BaseProvider):
    """Minimal concrete provider used to test the abstract base."""

    name = "stub"

    def __init__(self, available: bool = True, reply: str = "stub reply") -> None:
        self._available = available
        self._reply = reply

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        return CompletionResponse(
            content=self._reply,
            model="stub-model",
            provider=self.name,
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            raw={},
        )

    def is_available(self) -> bool:
        return self._available


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_create_user_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_system_message(self):
        msg = Message(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"

    def test_create_assistant_message(self):
        msg = Message(role="assistant", content="Sure!")
        assert msg.role == "assistant"

    def test_message_equality(self):
        m1 = Message(role="user", content="Hi")
        m2 = Message(role="user", content="Hi")
        assert m1 == m2

    def test_message_inequality_on_content(self):
        m1 = Message(role="user", content="Hi")
        m2 = Message(role="user", content="Bye")
        assert m1 != m2


# ---------------------------------------------------------------------------
# CompletionResponse
# ---------------------------------------------------------------------------


class TestCompletionResponse:
    def test_create_completion_response(self):
        resp = CompletionResponse(
            content="Hello!",
            model="gpt-4o",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            raw={"id": "chatcmpl-123"},
        )
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4o"
        assert resp.provider == "openai"
        assert resp.usage["total_tokens"] == 15
        assert resp.raw["id"] == "chatcmpl-123"

    def test_completion_response_equality(self):
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        r1 = CompletionResponse(
            content="X", model="m", provider="p", usage=usage, raw={}
        )
        r2 = CompletionResponse(
            content="X", model="m", provider="p", usage=usage, raw={}
        )
        assert r1 == r2


# ---------------------------------------------------------------------------
# BaseProvider (via StubProvider)
# ---------------------------------------------------------------------------


class TestBaseProvider:
    def test_complete_returns_completion_response(self):
        provider = StubProvider(reply="test response")
        messages = [Message(role="user", content="Hi")]
        resp = provider.complete(messages)
        assert isinstance(resp, CompletionResponse)
        assert resp.content == "test response"

    def test_is_available_returns_true(self):
        provider = StubProvider(available=True)
        assert provider.is_available() is True

    def test_is_available_returns_false(self):
        provider = StubProvider(available=False)
        assert provider.is_available() is False

    def test_repr_includes_name(self):
        provider = StubProvider()
        assert "stub" in repr(provider)

    def test_complete_accepts_multiple_messages(self):
        provider = StubProvider()
        messages = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="Tell me more."),
        ]
        resp = provider.complete(messages)
        assert resp.provider == "stub"

    def test_complete_passes_kwargs(self):
        """kwargs are accepted without error (concrete providers may use them)."""
        provider = StubProvider()
        messages = [Message(role="user", content="Hi")]
        resp = provider.complete(messages, temperature=0.5, max_tokens=100)
        assert resp is not None

    def test_cannot_instantiate_base_provider_directly(self):
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]
