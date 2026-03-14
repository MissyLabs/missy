"""Tests for missy.providers.anthropic_provider.AnthropicProvider."""

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


def _make_config(api_key: str | None = "sk-ant-test") -> ProviderConfig:
    return ProviderConfig(
        name="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        timeout=30,
    )


def _make_sdk_response(content: str = "Hello!", model: str = "claude-3-5-sonnet-20241022") -> MagicMock:
    """Build a mock Anthropic SDK response object."""
    resp = MagicMock()
    block = MagicMock()
    block.text = content
    resp.content = [block]
    resp.model = model
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    resp.model_dump.return_value = {"model": model, "content": content}
    return resp


def _make_sdk_module() -> MagicMock:
    """Return a mock of the anthropic SDK module."""
    sdk = MagicMock()
    sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
    sdk.AuthenticationError = type("AuthenticationError", (Exception,), {})
    sdk.APIError = type("APIError", (Exception,), {})
    return sdk


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
        import missy.providers.anthropic_provider as mod

        original = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = True
            from missy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(_make_config(api_key="sk-ant-key"))
            assert provider.is_available() is True
        finally:
            mod._ANTHROPIC_AVAILABLE = original

    def test_returns_false_when_no_api_key(self):
        import missy.providers.anthropic_provider as mod

        original = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = True
            from missy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(_make_config(api_key=None))
            assert provider.is_available() is False
        finally:
            mod._ANTHROPIC_AVAILABLE = original

    def test_returns_false_when_sdk_not_installed(self):
        import missy.providers.anthropic_provider as mod

        original = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = False
            from missy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(_make_config(api_key="sk-ant-key"))
            assert provider.is_available() is False
        finally:
            mod._ANTHROPIC_AVAILABLE = original


# ---------------------------------------------------------------------------
# complete – success path
# ---------------------------------------------------------------------------


class TestComplete:
    def _run_complete(self, messages, config=None, **kwargs):
        """Helper that patches the SDK and runs complete()."""
        import missy.providers.anthropic_provider as mod
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_sdk_module()
        sdk_resp = _make_sdk_response()
        sdk_mock.Anthropic.return_value.messages.create.return_value = sdk_resp

        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            mod._anthropic_sdk = sdk_mock
            mod._ANTHROPIC_AVAILABLE = True
            provider = AnthropicProvider(config or _make_config())
            return provider.complete(messages, **kwargs), sdk_mock
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

    def test_complete_returns_completion_response(self):
        messages = [Message(role="user", content="Hello")]
        result, _ = self._run_complete(messages)
        assert isinstance(result, CompletionResponse)

    def test_complete_maps_content(self):
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.return_value = (
            _make_sdk_response(content="Claude says hi!")
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            mod._anthropic_sdk = sdk_mock
            mod._ANTHROPIC_AVAILABLE = True
            from missy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(_make_config())
            result = provider.complete([Message(role="user", content="Hello")])
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

        assert result.content == "Claude says hi!"

    def test_complete_maps_usage(self):
        messages = [Message(role="user", content="Hello")]
        result, _ = self._run_complete(messages)
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 15

    def test_complete_sets_provider_name(self):
        messages = [Message(role="user", content="Hello")]
        result, _ = self._run_complete(messages)
        assert result.provider == "anthropic"

    def test_complete_extracts_system_message(self):
        """System messages must be sent via the 'system' kwarg to the SDK."""
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.return_value = (
            _make_sdk_response()
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            mod._anthropic_sdk = sdk_mock
            mod._ANTHROPIC_AVAILABLE = True
            from missy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(_make_config())
            messages = [
                Message(role="system", content="Be concise."),
                Message(role="user", content="Hello"),
            ]
            provider.complete(messages)
            call_kwargs = sdk_mock.Anthropic.return_value.messages.create.call_args[1]
            assert call_kwargs["system"] == "Be concise."
            # System message should NOT appear in the messages list
            for m in call_kwargs["messages"]:
                assert m["role"] != "system"
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

    def test_complete_emits_allow_event(self):
        messages = [Message(role="user", content="Hello")]
        self._run_complete(messages)
        events = event_bus.get_events(event_type="provider_invoke", result="allow")
        assert len(events) >= 1

    def test_complete_raises_when_sdk_unavailable(self):
        import missy.providers.anthropic_provider as mod

        original = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = False
            from missy.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(_make_config())
            with pytest.raises(ProviderError, match="not installed"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._ANTHROPIC_AVAILABLE = original


# ---------------------------------------------------------------------------
# complete – error paths
# ---------------------------------------------------------------------------


class TestCompleteErrors:
    def _provider_with_sdk(self, sdk_mock):
        """Return a configured AnthropicProvider using a mock SDK."""
        import missy.providers.anthropic_provider as mod

        mod._anthropic_sdk = sdk_mock
        mod._ANTHROPIC_AVAILABLE = True
        from missy.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(_make_config())

    def test_authentication_error_raises_provider_error(self):
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = (
            sdk_mock.AuthenticationError("bad key")
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="authentication"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

    def test_api_error_raises_provider_error(self):
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = (
            sdk_mock.APIError("server error")
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="API error"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

    def test_timeout_error_raises_provider_error(self):
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = (
            sdk_mock.APITimeoutError("timed out")
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="timed out"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

    def test_unexpected_exception_raises_provider_error(self):
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = RuntimeError(
            "unexpected"
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError, match="Unexpected"):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail

    def test_error_emits_error_event(self):
        import missy.providers.anthropic_provider as mod

        sdk_mock = _make_sdk_module()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = (
            sdk_mock.APIError("boom")
        )
        original_sdk = mod._anthropic_sdk
        original_avail = mod._ANTHROPIC_AVAILABLE
        try:
            provider = self._provider_with_sdk(sdk_mock)
            with pytest.raises(ProviderError):
                provider.complete([Message(role="user", content="Hi")])
        finally:
            mod._anthropic_sdk = original_sdk
            mod._ANTHROPIC_AVAILABLE = original_avail
        events = event_bus.get_events(event_type="provider_invoke", result="error")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_name_attribute(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_make_config())
        assert provider.name == "anthropic"
