"""Tests for missy.providers.openai_provider.OpenAIProvider."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.base import Message
from missy.providers.openai_provider import OpenAIProvider


def _make_config(**overrides) -> ProviderConfig:
    defaults = {"name": "openai", "model": "gpt-4o", "api_key": "sk-test-key"}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestOpenAIInit:
    def test_defaults(self):
        p = OpenAIProvider(_make_config())
        assert p._model == "gpt-4o"
        assert p._api_key == "sk-test-key"
        assert p._base_url is None

    def test_custom_base_url(self):
        p = OpenAIProvider(_make_config(base_url="http://local:8080/v1"))
        assert p._base_url == "http://local:8080/v1"

    def test_default_model_when_empty(self):
        p = OpenAIProvider(_make_config(model=""))
        assert p._model == "gpt-4o"


class TestOpenAIAvailability:
    def test_available(self):
        p = OpenAIProvider(_make_config())
        assert p.is_available() is True

    def test_unavailable_no_key(self):
        p = OpenAIProvider(_make_config(api_key=None))
        assert p.is_available() is False


class TestOpenAIComplete:
    def _mock_response(self, text="Hello!", model="gpt-4o"):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=text, tool_calls=None),
            finish_reason="stop",
        )
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
            model_dump=lambda: {},
        )

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_successful_completion(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Hello!"
        assert resp.provider == "openai"
        assert resp.usage["total_tokens"] == 30

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_timeout_error(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _APITimeoutError("timeout")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="timed out"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_auth_error(self, mock_sdk):
        _APIError = type("APIError", (Exception,), {})
        _APITimeoutError = type("APITimeoutError", (_APIError,), {})
        _AuthenticationError = type("AuthenticationError", (_APIError,), {})
        mock_sdk.APIError = _APIError
        mock_sdk.APITimeoutError = _APITimeoutError
        mock_sdk.AuthenticationError = _AuthenticationError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _AuthenticationError("bad")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="authentication failed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", False)
    def test_sdk_not_installed(self):
        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="not installed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_empty_choices(self, mock_sdk):
        resp = SimpleNamespace(
            choices=[],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=0, total_tokens=5),
            model_dump=lambda: {},
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        result = p.complete([Message(role="user", content="Hi")])
        assert result.content == ""

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_base_url_passed_to_client(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(base_url="http://local:8080"))
        p.complete([Message(role="user", content="Hi")])
        call_kwargs = mock_sdk.OpenAI.call_args[1]
        assert call_kwargs["base_url"] == "http://local:8080"


class TestOpenAIToolSchema:
    def test_function_format(self):
        p = OpenAIProvider(_make_config())
        tool = MagicMock()
        tool.name = "calc"
        tool.description = "Calculator"
        tool.get_schema.return_value = {
            "parameters": {
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            }
        }
        schemas = p.get_tool_schema([tool])
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "calc"


class TestOpenAICompleteWithTools:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_tool_call_parsing(self, mock_sdk):
        tc = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(name="calc", arguments='{"expr": "1+1"}'),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="Calculating", tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = SimpleNamespace(
            choices=[choice],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model_dump=lambda: {},
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        tool = MagicMock()
        tool.name = "calc"
        tool.description = "Calculator"
        tool.get_schema.return_value = {"parameters": {"properties": {}, "required": []}}

        result = p.complete_with_tools([Message(role="user", content="calc")], [tool])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calc"
        assert result.tool_calls[0].arguments == {"expr": "1+1"}

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_malformed_tool_args(self, mock_sdk):
        tc = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(name="calc", arguments="not json"),
        )
        choice = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=[tc]),
            finish_reason="tool_calls",
        )
        resp = SimpleNamespace(
            choices=[choice],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model_dump=lambda: {},
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        result = p.complete_with_tools(
            [Message(role="user", content="x")],
            [MagicMock(name="calc", description="c", get_schema=MagicMock(return_value={"parameters": {"properties": {}, "required": []}}))],
        )
        # Malformed args should be empty dict
        assert result.tool_calls[0].arguments == {}

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_system_prompt_injection(self, mock_sdk):
        mock_client = MagicMock()
        resp = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model_dump=lambda: {},
        )
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        p.complete_with_tools(
            [Message(role="user", content="hi")],
            [],
            system="Be helpful",
        )
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "Be helpful"

    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", False)
    def test_sdk_not_installed(self):
        p = OpenAIProvider(_make_config())
        with pytest.raises(ProviderError, match="not installed"):
            p.complete_with_tools([Message(role="user", content="Hi")], [])
