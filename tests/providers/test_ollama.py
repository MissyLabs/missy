"""Tests for missy.providers.ollama_provider.OllamaProvider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.events import event_bus
from missy.core.exceptions import ProviderError
from missy.providers.base import CompletionResponse, Message
from missy.providers.ollama_provider import OllamaProvider


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(
    base_url: str | None = None,
    model: str | None = None,
    timeout: int = 30,
) -> ProviderConfig:
    return ProviderConfig(
        name="ollama",
        model=model or "llama3.2",
        base_url=base_url,
        timeout=timeout,
    )


def _make_http_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


_VALID_CHAT_RESPONSE = {
    "model": "llama3.2",
    "message": {"role": "assistant", "content": "Hello from Ollama!"},
    "prompt_eval_count": 10,
    "eval_count": 5,
}


@pytest.fixture(autouse=True)
def clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture()
def provider() -> OllamaProvider:
    return OllamaProvider(_make_config())


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_true_when_api_responds_200(self, provider):
        mock_resp = _make_http_response(200)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.get.return_value = mock_resp
            assert provider.is_available() is True

    def test_returns_false_when_api_responds_non_200(self, provider):
        mock_resp = _make_http_response(503)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.get.return_value = mock_resp
            assert provider.is_available() is False

    def test_returns_false_when_connection_error(self, provider):
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.get.side_effect = ConnectionError("refused")
            assert provider.is_available() is False

    def test_is_available_uses_api_tags_endpoint(self, provider):
        mock_resp = _make_http_response(200)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.get.return_value = mock_resp
            provider.is_available()
            call_args = MockClient.return_value.get.call_args
            assert "/api/tags" in call_args[0][0]


# ---------------------------------------------------------------------------
# complete – success path
# ---------------------------------------------------------------------------


class TestComplete:
    def test_complete_returns_completion_response(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            messages = [Message(role="user", content="Hello")]
            result = provider.complete(messages)
        assert isinstance(result, CompletionResponse)

    def test_complete_parses_content(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            result = provider.complete([Message(role="user", content="Hi")])
        assert result.content == "Hello from Ollama!"

    def test_complete_parses_model(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            result = provider.complete([Message(role="user", content="Hi")])
        assert result.model == "llama3.2"

    def test_complete_parses_usage(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            result = provider.complete([Message(role="user", content="Hi")])
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    def test_complete_sets_provider_name(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            result = provider.complete([Message(role="user", content="Hi")])
        assert result.provider == "ollama"

    def test_complete_emits_allow_event(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            provider.complete([Message(role="user", content="Hi")])
        events = event_bus.get_events(event_type="provider_invoke", result="allow")
        assert len(events) >= 1

    def test_complete_sends_stream_false(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            provider.complete([Message(role="user", content="Hi")])
            call_kwargs = MockClient.return_value.post.call_args[1]
            assert call_kwargs["json"]["stream"] is False

    def test_complete_forwards_temperature(self, provider):
        mock_resp = _make_http_response(200, _VALID_CHAT_RESPONSE)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            provider.complete(
                [Message(role="user", content="Hi")], temperature=0.9
            )
            payload = MockClient.return_value.post.call_args[1]["json"]
            assert payload["options"]["temperature"] == 0.9

    def test_complete_empty_message_gives_empty_content(self, provider):
        data = {"model": "llama3.2", "message": {}, "prompt_eval_count": 0, "eval_count": 0}
        mock_resp = _make_http_response(200, data)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            result = provider.complete([Message(role="user", content="Hi")])
        assert result.content == ""


# ---------------------------------------------------------------------------
# complete – error paths
# ---------------------------------------------------------------------------


class TestCompleteErrors:
    def test_http_error_raises_provider_error(self, provider):
        mock_resp = _make_http_response(500)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            with pytest.raises(ProviderError, match="Ollama"):
                provider.complete([Message(role="user", content="Hi")])

    def test_connection_error_raises_provider_error(self, provider):
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.side_effect = ConnectionError("refused")
            with pytest.raises(ProviderError, match="Ollama"):
                provider.complete([Message(role="user", content="Hi")])

    def test_invalid_json_raises_provider_error(self, provider):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("not JSON")
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            with pytest.raises(ProviderError, match="JSON"):
                provider.complete([Message(role="user", content="Hi")])

    def test_http_error_emits_error_event(self, provider):
        mock_resp = _make_http_response(500)
        with patch(
            "missy.providers.ollama_provider.PolicyHTTPClient"
        ) as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            with pytest.raises(ProviderError):
                provider.complete([Message(role="user", content="Hi")])
        events = event_bus.get_events(event_type="provider_invoke", result="error")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_default_base_url_is_localhost(self):
        provider = OllamaProvider(_make_config(base_url=None))
        assert "localhost" in provider._base_url

    def test_custom_base_url_is_used(self):
        provider = OllamaProvider(_make_config(base_url="http://remote:11434"))
        assert "remote" in provider._base_url

    def test_trailing_slash_stripped_from_base_url(self):
        provider = OllamaProvider(_make_config(base_url="http://localhost:11434/"))
        assert not provider._base_url.endswith("/")

    def test_name_attribute(self, provider):
        assert provider.name == "ollama"
