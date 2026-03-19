"""Tests for missy.providers.anthropic_provider.AnthropicProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.anthropic_provider import AnthropicProvider
from missy.providers.base import Message


def _make_config(**overrides) -> ProviderConfig:
    defaults = {"name": "anthropic", "model": "claude-sonnet-4-6", "api_key": "sk-ant-api03-test"}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestAnthropicInit:
    """Initialization and setup-token rejection."""

    def test_normal_api_key(self):
        p = AnthropicProvider(_make_config())
        assert p._api_key == "sk-ant-api03-test"
        assert p._model == "claude-sonnet-4-6"

    def test_setup_token_rejected(self):
        p = AnthropicProvider(_make_config(api_key="sk-ant-oat01-shortlived"))
        assert p._api_key is None

    def test_no_api_key(self):
        p = AnthropicProvider(_make_config(api_key=None))
        assert p._api_key is None

    def test_default_model_when_empty(self):
        p = AnthropicProvider(_make_config(model=""))
        assert p._model == "claude-sonnet-4-6"

    def test_custom_timeout(self):
        p = AnthropicProvider(_make_config(timeout=60))
        assert p._timeout == 60


class TestAnthropicAvailability:
    """Tests for is_available()."""

    def test_available_with_key(self):
        p = AnthropicProvider(_make_config())
        assert p.is_available() is True

    def test_unavailable_without_key(self):
        p = AnthropicProvider(_make_config(api_key=None))
        assert p.is_available() is False

    def test_unavailable_with_setup_token(self):
        p = AnthropicProvider(_make_config(api_key="sk-ant-oat01-xxx"))
        assert p.is_available() is False


class TestAnthropicComplete:
    """Tests for complete() with mocked SDK."""

    def _mock_response(self, text="Hello!", model="claude-sonnet-4-6"):
        content_block = SimpleNamespace(text=text)
        usage = SimpleNamespace(input_tokens=10, output_tokens=20)
        return SimpleNamespace(
            content=[content_block],
            model=model,
            usage=usage,
            model_dump=dict,
        )

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_successful_completion(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response()
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        msgs = [Message(role="user", content="Hi")]
        resp = p.complete(msgs)
        assert resp.content == "Hello!"
        assert resp.provider == "anthropic"
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 20

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_system_message_extracted(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response()
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        msgs = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hi"),
        ]
        p.complete(msgs)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_timeout_error(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _APITimeoutError("timed out")
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        with pytest.raises(ProviderError, match="timed out"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_auth_error(self, mock_sdk):
        # Build a proper exception hierarchy
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _AuthenticationError("bad key")
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        with pytest.raises(ProviderError, match="authentication failed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_api_error(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _APIError("rate limited")
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        with pytest.raises(ProviderError, match="API error"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", False)
    def test_sdk_not_installed(self):
        p = AnthropicProvider(_make_config())
        with pytest.raises(ProviderError, match="not installed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_empty_content(self, mock_sdk):
        resp = SimpleNamespace(
            content=[],
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(input_tokens=5, output_tokens=0),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        result = p.complete([Message(role="user", content="Hi")])
        assert result.content == ""

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_model_override(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response()
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        p.complete([Message(role="user", content="Hi")], model="claude-haiku-4-5")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5"


class TestAnthropicToolSchema:
    """Tests for get_tool_schema()."""

    def test_schema_conversion(self):
        p = AnthropicProvider(_make_config())
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search the web"
        tool.get_schema.return_value = {
            "parameters": {
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }
        }
        schemas = p.get_tool_schema([tool])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "search"
        assert schemas[0]["input_schema"]["properties"]["query"]["type"] == "string"

    def test_tool_without_schema(self):
        p = AnthropicProvider(_make_config())
        tool = MagicMock(spec=[])  # No get_schema
        tool.name = "simple"
        tool.description = "Simple tool"
        schemas = p.get_tool_schema([tool])
        assert schemas[0]["input_schema"]["properties"] == {}


class TestAnthropicCompleteWithTools:
    """Tests for complete_with_tools() with mocked SDK."""

    @patch("missy.providers.anthropic_provider._anthropic_sdk")
    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", True)
    def test_tool_call_response(self, mock_sdk):
        text_block = SimpleNamespace(type="text", text="Let me search")
        tool_block = SimpleNamespace(
            type="tool_use", id="tc_1", name="search", input={"query": "missy"}
        )
        resp = SimpleNamespace(
            content=[text_block, tool_block],
            model="claude-sonnet-4-6",
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp
        mock_sdk.Anthropic.return_value = mock_client

        p = AnthropicProvider(_make_config())
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search"
        tool.get_schema.return_value = {"parameters": {"properties": {}, "required": []}}

        result = p.complete_with_tools([Message(role="user", content="search")], [tool])
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    @patch("missy.providers.anthropic_provider._ANTHROPIC_AVAILABLE", False)
    def test_sdk_not_installed(self):
        p = AnthropicProvider(_make_config())
        with pytest.raises(ProviderError, match="not installed"):
            p.complete_with_tools([Message(role="user", content="Hi")], [])
