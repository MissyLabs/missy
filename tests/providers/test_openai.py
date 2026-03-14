"""Tests for missy.providers.openai_provider.OpenAIProvider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.config.settings import ProviderConfig
from missy.core.events import event_bus
from missy.core.exceptions import ProviderError
from missy.providers.base import CompletionResponse, Message

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(
    api_key: str | None = "sk-test",
    model: str = "gpt-4o",
    base_url: str | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        name="openai",
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=30,
    )


def _make_sdk_module() -> MagicMock:
    sdk = MagicMock()
    sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
    sdk.AuthenticationError = type("AuthenticationError", (Exception,), {})
    sdk.APIError = type("APIError", (Exception,), {})
    return sdk


def _make_sdk_response(content: str = "Hello!", model: str = "gpt-4o") -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    resp.model = model
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    resp.model_dump.return_value = {"model": model}
    return resp


@pytest.fixture(autouse=True)
def clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_true_when_sdk_installed_and_api_key_set(self):
        import missy.providers.openai_provider as mod

        original = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = True
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config(api_key="sk-key"))
            assert provider.is_available() is True
        finally:
            mod._OPENAI_AVAILABLE = original

    def test_returns_false_when_no_api_key(self):
        import missy.providers.openai_provider as mod

        original = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = True
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config(api_key=None))
            assert provider.is_available() is False
        finally:
            mod._OPENAI_AVAILABLE = original

    def test_returns_false_when_sdk_not_installed(self):
        import missy.providers.openai_provider as mod

        original = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = False
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config())
            assert provider.is_available() is False
        finally:
            mod._OPENAI_AVAILABLE = original


# ---------------------------------------------------------------------------
# complete – success path
# ---------------------------------------------------------------------------


class TestComplete:
    def _run_complete(self, messages, config=None, **kwargs):
        """Patch the OpenAI SDK module and run complete()."""
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_resp = _make_sdk_response()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = sdk_resp

        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            mod._openai_sdk = sdk_mock
            mod._OPENAI_AVAILABLE = True
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(config or _make_config())
            return provider.complete(messages, **kwargs), sdk_mock
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail

    def test_complete_returns_completion_response(self):
        result, _ = self._run_complete([Message(role="user", content="Hi")])
        assert isinstance(result, CompletionResponse)

    def test_complete_maps_content(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = _make_sdk_response(
            content="GPT says hello!"
        )
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            mod._openai_sdk = sdk_mock
            mod._OPENAI_AVAILABLE = True
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config())
            result = provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail
        assert result.content == "GPT says hello!"

    def test_complete_maps_usage(self):
        result, _ = self._run_complete([Message(role="user", content="Hi")])
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    def test_complete_sets_provider_name(self):
        result, _ = self._run_complete([Message(role="user", content="Hi")])
        assert result.provider == "openai"

    def test_complete_forwards_temperature(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = _make_sdk_response()
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            mod._openai_sdk = sdk_mock
            mod._OPENAI_AVAILABLE = True
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config())
            provider.complete([Message(role="user", content="Hi")], temperature=0.2)
            call_kwargs = sdk_mock.OpenAI.return_value.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.2
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail

    def test_complete_emits_allow_event(self):
        self._run_complete([Message(role="user", content="Hi")])
        events = event_bus.get_events(event_type="provider_invoke", result="allow")
        assert len(events) >= 1

    def test_complete_raises_when_sdk_unavailable(self):
        import missy.providers.openai_provider as mod

        original = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = False
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config())
            with pytest.raises(ProviderError, match="not installed"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._OPENAI_AVAILABLE = original

    def test_complete_empty_choices_returns_empty_content(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        empty_resp = MagicMock()
        empty_resp.choices = []
        empty_resp.model = "gpt-4o"
        empty_resp.usage.prompt_tokens = 5
        empty_resp.usage.completion_tokens = 0
        empty_resp.usage.total_tokens = 5
        empty_resp.model_dump.return_value = {}
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = empty_resp

        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            mod._openai_sdk = sdk_mock
            mod._OPENAI_AVAILABLE = True
            from missy.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(_make_config())
            result = provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail
        assert result.content == ""


# ---------------------------------------------------------------------------
# complete – error paths
# ---------------------------------------------------------------------------


class TestCompleteErrors:
    def _provider_with_sdk(self, sdk_mock):
        import missy.providers.openai_provider as mod

        mod._openai_sdk = sdk_mock
        mod._OPENAI_AVAILABLE = True
        from missy.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(_make_config())

    def test_authentication_error_raises_provider_error(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = (
            sdk_mock.AuthenticationError("bad key")
        )
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="authentication"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail

    def test_api_error_raises_provider_error(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APIError(
            "server error"
        )
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="API error"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail

    def test_timeout_error_raises_provider_error(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APITimeoutError(
            "timeout"
        )
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="timed out"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail

    def test_unexpected_exception_raises_provider_error(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError("boom")
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="Unexpected"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail

    def test_error_emits_error_event(self):
        import missy.providers.openai_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APIError("boom")
        original_sdk = mod._openai_sdk
        original_avail = mod._OPENAI_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._openai_sdk = original_sdk
            mod._OPENAI_AVAILABLE = original_avail
        events = event_bus.get_events(event_type="provider_invoke", result="error")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name_attribute(self):
        from missy.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(_make_config())
        assert provider.name == "openai"
