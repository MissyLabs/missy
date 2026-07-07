"""Tests for missy.providers.openai_provider.OpenAIProvider."""

from __future__ import annotations

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
        assert p._model == "auto"

    def test_api_key_setter_resets_cached_client(self):
        p = OpenAIProvider(_make_config())
        p._client = object()
        p._resolved_model = "gpt-5.4"
        p.api_key = "sk-new-key"
        assert p._client is None
        assert p._resolved_model is None
        assert p.api_key == "sk-new-key"


class TestOpenAIAvailability:
    def test_available(self):
        p = OpenAIProvider(_make_config())
        assert p.is_available() is True

    def test_unavailable_no_key(self):
        p = OpenAIProvider(_make_config(api_key=None))
        assert p.is_available() is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"})
    def test_available_with_env_key(self):
        p = OpenAIProvider(_make_config(api_key=None))
        assert p.is_available() is True


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
            model_dump=dict,
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
            model_dump=dict,
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

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_auto_model_uses_best_available_model(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-4o"),
                SimpleNamespace(id="gpt-5.4-mini"),
                SimpleNamespace(id="gpt-5.5"),
            ]
        )
        mock_client.chat.completions.create.return_value = self._mock_response(model="gpt-5.5")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="auto"))
        p.complete([Message(role="user", content="Hi")])
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-5.5"

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_gpt5_omits_custom_temperature_and_maps_max_tokens(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._mock_response(model="gpt-5.5")
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config(model="gpt-5.5"))
        p.complete([Message(role="user", content="Hi")], temperature=0.2, max_tokens=7)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "temperature" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 7
        assert "max_tokens" not in call_kwargs


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


class TestOpenAIMessagePayload:
    def test_runtime_tool_messages_are_converted_to_openai_shape(self):
        p = OpenAIProvider(_make_config())
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "use a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "calc", "arguments": {"expr": "1+1"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "calc",
                "content": "2",
                "is_error": False,
            },
        ]
        payload = p._messages_to_chat_payload(messages)
        assert payload[2]["tool_calls"][0] == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calc", "arguments": '{"expr": "1+1"}'},
        }
        assert payload[3] == {"role": "tool", "tool_call_id": "call_1", "content": "2"}


class TestOpenAICompleteWithTools:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_native_dict_tool_messages_do_not_break_token_estimate(self, mock_sdk):
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        messages = [
            {"role": "user", "content": "use a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "name": "calc", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "2"},
        ]

        result = p.complete_with_tools(messages, [])

        assert result.content == "done"
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "2",
        }
        assert "tools" not in call_kwargs

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
            model_dump=dict,
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
            model_dump=dict,
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        result = p.complete_with_tools(
            [Message(role="user", content="x")],
            [
                MagicMock(
                    name="calc",
                    description="c",
                    get_schema=MagicMock(
                        return_value={"parameters": {"properties": {}, "required": []}}
                    ),
                )
            ],
        )
        # Malformed args should be empty dict
        assert result.tool_calls[0].arguments == {}

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_system_prompt_injection(self, mock_sdk):
        mock_client = MagicMock()
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            model="gpt-4o",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            model_dump=dict,
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
