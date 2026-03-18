"""Tests for missy.providers.ollama_provider.OllamaProvider."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.base import Message
from missy.providers.ollama_provider import OllamaProvider


def _make_config(**overrides) -> ProviderConfig:
    defaults = {"name": "ollama", "model": "llama3.2", "api_key": None}
    defaults.update(overrides)
    return ProviderConfig(**defaults)


class TestOllamaInit:
    def test_defaults(self):
        p = OllamaProvider(_make_config())
        assert p._model == "llama3.2"
        assert p._base_url == "http://localhost:11434"

    def test_custom_base_url(self):
        p = OllamaProvider(_make_config(base_url="http://gpu:11434/"))
        assert p._base_url == "http://gpu:11434"  # trailing slash stripped

    def test_default_model_when_empty(self):
        p = OllamaProvider(_make_config(model=""))
        assert p._model == "llama3.2"


class TestOllamaAvailability:
    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_available(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client_cls.return_value.get.return_value = mock_resp

        p = OllamaProvider(_make_config())
        assert p.is_available() is True

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_unavailable_non_200(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_client_cls.return_value.get.return_value = mock_resp

        p = OllamaProvider(_make_config())
        assert p.is_available() is False

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_unavailable_exception(self, mock_client_cls):
        mock_client_cls.return_value.get.side_effect = ConnectionError("refused")
        p = OllamaProvider(_make_config())
        assert p.is_available() is False


class TestOllamaComplete:
    def _mock_response(self, content="Hello!", model="llama3.2"):
        resp = MagicMock()
        resp.json.return_value = {
            "model": model,
            "message": {"role": "assistant", "content": content},
            "prompt_eval_count": 10,
            "eval_count": 20,
        }
        resp.raise_for_status.return_value = None
        return resp

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_successful_completion(self, mock_client_cls):
        mock_client_cls.return_value.post.return_value = self._mock_response()

        p = OllamaProvider(_make_config())
        resp = p.complete([Message(role="user", content="Hi")])
        assert resp.content == "Hello!"
        assert resp.provider == "ollama"
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 20

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_network_error(self, mock_client_cls):
        mock_client_cls.return_value.post.side_effect = ConnectionError("refused")

        p = OllamaProvider(_make_config())
        with pytest.raises(ProviderError, match="Ollama request failed"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_invalid_json_response(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_client_cls.return_value.post.return_value = mock_resp

        p = OllamaProvider(_make_config())
        with pytest.raises(ProviderError, match="invalid JSON"):
            p.complete([Message(role="user", content="Hi")])

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_empty_message_in_response(self, mock_client_cls):
        resp = MagicMock()
        resp.json.return_value = {"model": "llama3.2", "message": None}
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value.post.return_value = resp

        p = OllamaProvider(_make_config())
        result = p.complete([Message(role="user", content="Hi")])
        assert result.content == ""

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_temperature_option(self, mock_client_cls):
        mock_client_cls.return_value.post.return_value = self._mock_response()

        p = OllamaProvider(_make_config())
        p.complete([Message(role="user", content="Hi")], temperature=0.7)
        call_kwargs = mock_client_cls.return_value.post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["options"]["temperature"] == 0.7


class TestOllamaToolSchema:
    def test_schema_format(self):
        p = OllamaProvider(_make_config())
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search"
        tool.get_schema.return_value = {
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            }
        }
        schemas = p.get_tool_schema([tool])
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"

    def test_schema_missing_type(self):
        """Parameters without 'type' key get wrapped."""
        p = OllamaProvider(_make_config())
        tool = MagicMock()
        tool.name = "t"
        tool.description = "d"
        tool.get_schema.return_value = {
            "parameters": {"q": {"type": "string"}}
        }
        schemas = p.get_tool_schema([tool])
        assert schemas[0]["function"]["parameters"]["type"] == "object"


class TestOllamaCompleteWithTools:
    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_tool_call_response(self, mock_client_cls):
        resp = MagicMock()
        resp.json.return_value = {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {"name": "search", "arguments": {"q": "test"}},
                        "id": "tc1",
                    }
                ],
            },
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value.post.return_value = resp

        p = OllamaProvider(_make_config())
        tool = MagicMock()
        tool.name = "search"
        tool.description = "Search"
        tool.get_schema.return_value = {"parameters": {"type": "object", "properties": {}}}

        result = p.complete_with_tools([Message(role="user", content="find")], [tool])
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_no_tool_calls(self, mock_client_cls):
        resp = MagicMock()
        resp.json.return_value = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "No tools needed"},
            "prompt_eval_count": 5,
            "eval_count": 10,
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value.post.return_value = resp

        p = OllamaProvider(_make_config())
        result = p.complete_with_tools([Message(role="user", content="hi")], [])
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_tool_call_string_args_handled(self, mock_client_cls):
        """If arguments is a string instead of dict, treat as empty."""
        resp = MagicMock()
        resp.json.return_value = {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "t", "arguments": "not a dict"}}
                ],
            },
            "prompt_eval_count": 5,
            "eval_count": 5,
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value.post.return_value = resp

        p = OllamaProvider(_make_config())
        result = p.complete_with_tools([Message(role="user", content="x")], [])
        assert result.tool_calls[0].arguments == {}


class TestOllamaStream:
    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_stream_tokens(self, mock_client_cls):
        lines = [
            json.dumps({"message": {"content": "Hello"}, "done": False}),
            json.dumps({"message": {"content": " world"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True}),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(lines)
        mock_client_cls.return_value.post.return_value = mock_resp

        p = OllamaProvider(_make_config())
        tokens = list(p.stream([Message(role="user", content="Hi")]))
        assert tokens == ["Hello", " world"]

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_stream_network_error(self, mock_client_cls):
        mock_client_cls.return_value.post.side_effect = ConnectionError("refused")

        p = OllamaProvider(_make_config())
        with pytest.raises(ProviderError, match="Ollama stream failed"):
            list(p.stream([Message(role="user", content="Hi")]))

    @patch("missy.providers.ollama_provider.PolicyHTTPClient")
    def test_stream_malformed_json_skipped(self, mock_client_cls):
        lines = [
            "not json",
            json.dumps({"message": {"content": "ok"}, "done": True}),
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(lines)
        mock_client_cls.return_value.post.return_value = mock_resp

        p = OllamaProvider(_make_config())
        tokens = list(p.stream([Message(role="user", content="Hi")]))
        assert tokens == ["ok"]
