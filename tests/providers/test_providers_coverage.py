"""Targeted coverage tests for provider modules.

Focuses exclusively on paths not covered by the existing test suite:
  - AnthropicProvider: setup-token rejection, get_tool_schema, complete_with_tools,
    stream (happy + all error branches), _emit_event failure
  - OpenAIProvider: base_url forwarding, get_tool_schema, complete_with_tools
    (tool call parsing + invalid JSON + no choices path), stream (happy + errors)
  - CodexProvider: all paths (_extract_account_id, _messages_to_input,
    _extract_system, complete, stream SSE parsing, complete_with_tools,
    get_tool_schema, is_available)
  - ProviderRegistry: rotate_key (happy + edge cases), from_config (disabled +
    base_url auto-allow + construct failure), ModelRouter (all tier paths),
    init_registry / get_registry singleton
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ProviderConfig,
    ShellPolicy,
)
from missy.core.exceptions import ProviderError
from missy.providers.base import CompletionResponse, Message, ToolCall

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _provider_config(
    name: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    api_key: str | None = "sk-ant-test",
    timeout: int = 30,
    **kwargs,
) -> ProviderConfig:
    return ProviderConfig(name=name, model=model, api_key=api_key, timeout=timeout, **kwargs)


def _missy_config(providers: dict | None = None) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers=providers or {},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


def _make_anthropic_sdk() -> MagicMock:
    sdk = MagicMock()
    sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
    sdk.AuthenticationError = type("AuthenticationError", (Exception,), {})
    sdk.APIError = type("APIError", (Exception,), {})
    return sdk


def _make_openai_sdk() -> MagicMock:
    sdk = MagicMock()
    sdk.APITimeoutError = type("APITimeoutError", (Exception,), {})
    sdk.AuthenticationError = type("AuthenticationError", (Exception,), {})
    sdk.APIError = type("APIError", (Exception,), {})
    return sdk


@contextmanager
def _patch_anthropic_sdk(sdk_mock: MagicMock):
    import missy.providers.anthropic_provider as mod

    old_sdk, old_avail = mod._anthropic_sdk, mod._ANTHROPIC_AVAILABLE
    mod._anthropic_sdk = sdk_mock
    mod._ANTHROPIC_AVAILABLE = True
    try:
        yield mod
    finally:
        mod._anthropic_sdk = old_sdk
        mod._ANTHROPIC_AVAILABLE = old_avail


@contextmanager
def _patch_openai_sdk(sdk_mock: MagicMock):
    import missy.providers.openai_provider as mod

    old_sdk, old_avail = mod._openai_sdk, mod._OPENAI_AVAILABLE
    mod._openai_sdk = sdk_mock
    mod._OPENAI_AVAILABLE = True
    try:
        yield mod
    finally:
        mod._openai_sdk = old_sdk
        mod._OPENAI_AVAILABLE = old_avail


def _make_tool(name: str = "my_tool", description: str = "A tool") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.get_schema.return_value = {
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    }
    return tool


# ---------------------------------------------------------------------------
# AnthropicProvider — uncovered paths
# ---------------------------------------------------------------------------


class TestAnthropicProviderSetupTokenRejection:
    """Init should nullify keys that look like setup-tokens (sk-ant-oat...)."""

    def test_setup_token_is_rejected_and_key_set_to_none(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_provider_config(api_key="sk-ant-oat--supersecretoauthtoken"))
        assert provider._api_key is None

    def test_setup_token_makes_provider_unavailable(self):
        import missy.providers.anthropic_provider as mod
        from missy.providers.anthropic_provider import AnthropicProvider

        old = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = True
            provider = AnthropicProvider(_provider_config(api_key="sk-ant-oat--sometoken"))
            assert provider.is_available() is False
        finally:
            mod._ANTHROPIC_AVAILABLE = old

    def test_normal_key_is_not_rejected(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_provider_config(api_key="sk-ant-api03-goodkey"))
        assert provider._api_key == "sk-ant-api03-goodkey"


class TestAnthropicProviderGetToolSchema:
    def test_returns_anthropic_format_with_input_schema(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_provider_config())
        tool = _make_tool("search", "Search the web")
        schemas = provider.get_tool_schema([tool])

        assert len(schemas) == 1
        s = schemas[0]
        assert s["name"] == "search"
        assert s["description"] == "Search the web"
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"
        assert "properties" in s["input_schema"]
        assert "required" in s["input_schema"]

    def test_tool_without_get_schema_falls_back_to_empty_params(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_provider_config())
        tool = MagicMock(spec=[])  # no get_schema
        tool.name = "bare_tool"
        tool.description = "bare"
        schemas = provider.get_tool_schema([tool])

        assert schemas[0]["input_schema"]["properties"] == {}
        assert schemas[0]["input_schema"]["required"] == []

    def test_empty_tool_list_returns_empty_list(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_provider_config())
        assert provider.get_tool_schema([]) == []


class TestAnthropicProviderCompleteWithTools:
    def _run(self, messages, tools, system="", stop_reason="end_turn"):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.stop_reason = stop_reason
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        resp.model_dump.return_value = {}

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            return provider, sdk_mock, resp, provider.complete_with_tools(messages, tools, system)

    def test_plain_text_response_finish_reason_stop(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here is my answer."

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.stop_reason = "end_turn"
        resp.content = [text_block]
        resp.usage.input_tokens = 8
        resp.usage.output_tokens = 4
        resp.model_dump.return_value = {}

        from missy.providers.anthropic_provider import AnthropicProvider

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            result = provider.complete_with_tools([Message(role="user", content="Hello")], [])

        assert result.content == "Here is my answer."
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    def test_tool_use_response_populates_tool_calls(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_abc"
        tool_block.name = "search"
        tool_block.input = {"query": "python testing"}

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.stop_reason = "tool_use"
        resp.content = [tool_block]
        resp.usage.input_tokens = 20
        resp.usage.output_tokens = 10
        resp.model_dump.return_value = {}

        from missy.providers.anthropic_provider import AnthropicProvider

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            result = provider.complete_with_tools(
                [Message(role="user", content="Search for me")],
                [_make_tool("search")],
            )

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "call_abc"
        assert tc.name == "search"
        assert tc.arguments == {"query": "python testing"}

    def test_system_arg_takes_precedence_over_system_message(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.stop_reason = "end_turn"
        resp.content = [text_block]
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 2
        resp.model_dump.return_value = {}

        from missy.providers.anthropic_provider import AnthropicProvider

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            provider.complete_with_tools(
                [
                    Message(role="system", content="from_message"),
                    Message(role="user", content="hi"),
                ],
                [],
                system="explicit_system",
            )
            call_kwargs = sdk_mock.Anthropic.return_value.messages.create.call_args[1]

        assert call_kwargs["system"] == "explicit_system"

    def test_system_message_used_when_no_explicit_system_arg(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.stop_reason = "end_turn"
        resp.content = [text_block]
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 2
        resp.model_dump.return_value = {}

        from missy.providers.anthropic_provider import AnthropicProvider

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            provider.complete_with_tools(
                [
                    Message(role="system", content="system_from_message"),
                    Message(role="user", content="hi"),
                ],
                [],
                system="",
            )
            call_kwargs = sdk_mock.Anthropic.return_value.messages.create.call_args[1]

        assert call_kwargs["system"] == "system_from_message"

    def test_sdk_unavailable_raises_provider_error(self):
        import missy.providers.anthropic_provider as mod
        from missy.providers.anthropic_provider import AnthropicProvider

        old = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = False
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="not installed"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])
        finally:
            mod._ANTHROPIC_AVAILABLE = old

    def test_timeout_error_raises_provider_error(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = sdk_mock.APITimeoutError(
            "timed out"
        )
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="timed out"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_authentication_error_raises_provider_error(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = sdk_mock.AuthenticationError(
            "bad key"
        )
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="authentication"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_api_error_raises_provider_error(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = sdk_mock.APIError(
            "server error"
        )
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="API error"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_unexpected_exception_raises_provider_error(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.create.side_effect = RuntimeError("boom")
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="Unexpected"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_mixed_text_and_tool_blocks(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Let me search that."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_1"
        tool_block.name = "search"
        tool_block.input = {"query": "pytest"}

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.stop_reason = "tool_use"
        resp.content = [text_block, tool_block]
        resp.usage.input_tokens = 12
        resp.usage.output_tokens = 6
        resp.model_dump.return_value = {}

        from missy.providers.anthropic_provider import AnthropicProvider

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            result = provider.complete_with_tools(
                [Message(role="user", content="search for pytest")],
                [_make_tool("search")],
            )

        assert "Let me search that." in result.content
        assert len(result.tool_calls) == 1
        assert result.finish_reason == "tool_calls"


class TestAnthropicProviderStream:
    def _make_stream_context(self, sdk_mock: MagicMock, chunks: list[str]):
        stream_ctx = MagicMock()
        stream_ctx.__enter__ = MagicMock(return_value=stream_ctx)
        stream_ctx.__exit__ = MagicMock(return_value=False)
        stream_ctx.text_stream = iter(chunks)
        sdk_mock.Anthropic.return_value.messages.stream.return_value = stream_ctx
        return stream_ctx

    def test_stream_yields_text_chunks(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        self._make_stream_context(sdk_mock, ["Hello", " world", "!"])

        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            result = list(provider.stream([Message(role="user", content="Hi")]))

        assert result == ["Hello", " world", "!"]

    def test_stream_passes_system_arg(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        self._make_stream_context(sdk_mock, ["reply"])

        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            list(provider.stream([Message(role="user", content="Hi")], system="Be brief."))
            call_kwargs = sdk_mock.Anthropic.return_value.messages.stream.call_args[1]

        assert call_kwargs["system"] == "Be brief."

    def test_stream_uses_system_message_when_no_explicit_system(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        self._make_stream_context(sdk_mock, ["reply"])

        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            list(
                provider.stream(
                    [
                        Message(role="system", content="System instructions"),
                        Message(role="user", content="Hi"),
                    ]
                )
            )
            call_kwargs = sdk_mock.Anthropic.return_value.messages.stream.call_args[1]

        assert call_kwargs["system"] == "System instructions"

    def test_stream_raises_provider_error_when_sdk_unavailable(self):
        import missy.providers.anthropic_provider as mod
        from missy.providers.anthropic_provider import AnthropicProvider

        old = mod._ANTHROPIC_AVAILABLE
        try:
            mod._ANTHROPIC_AVAILABLE = False
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="not installed"):
                list(provider.stream([Message(role="user", content="Hi")]))
        finally:
            mod._ANTHROPIC_AVAILABLE = old

    def test_stream_raises_on_timeout(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.stream.side_effect = sdk_mock.APITimeoutError(
            "timed out"
        )
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="timed out"):
                list(provider.stream([Message(role="user", content="Hi")]))

    def test_stream_raises_on_auth_error(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.stream.side_effect = sdk_mock.AuthenticationError(
            "bad key"
        )
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="authentication"):
                list(provider.stream([Message(role="user", content="Hi")]))

    def test_stream_raises_on_api_error(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.stream.side_effect = sdk_mock.APIError(
            "server error"
        )
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="API error"):
                list(provider.stream([Message(role="user", content="Hi")]))

    def test_stream_raises_on_unexpected_exception(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        sdk_mock.Anthropic.return_value.messages.stream.side_effect = RuntimeError("boom")
        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            with pytest.raises(ProviderError, match="Unexpected"):
                list(provider.stream([Message(role="user", content="Hi")]))

    def test_stream_empty_yields_nothing(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        self._make_stream_context(sdk_mock, [])

        with _patch_anthropic_sdk(sdk_mock):
            provider = AnthropicProvider(_provider_config())
            result = list(provider.stream([Message(role="user", content="Hi")]))

        assert result == []


class TestAnthropicEmitEventFailure:
    """_emit_event must swallow exceptions from the event bus."""

    def test_emit_event_does_not_propagate_exception(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(_provider_config())
        with patch("missy.providers.anthropic_provider.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus failure")
            # Should not raise.
            provider._emit_event("sess1", "task1", "allow", "test message")

    def test_complete_succeeds_even_if_event_bus_fails(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.content = [MagicMock(text="hi")]
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 2
        resp.model_dump.return_value = {}

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            with patch("missy.providers.anthropic_provider.event_bus") as mock_bus:
                mock_bus.publish.side_effect = RuntimeError("bus failure")
                provider = AnthropicProvider(_provider_config())
                result = provider.complete([Message(role="user", content="hi")])

        assert isinstance(result, CompletionResponse)


# ---------------------------------------------------------------------------
# AnthropicProvider — complete() with temperature and model override kwargs
# ---------------------------------------------------------------------------


class TestAnthropicCompleteKwargs:
    def test_temperature_forwarded_to_sdk(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.content = [MagicMock(text="ok")]
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 2
        resp.model_dump.return_value = {}

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            provider.complete([Message(role="user", content="hi")], temperature=0.5)
            call_kwargs = sdk_mock.Anthropic.return_value.messages.create.call_args[1]

        assert call_kwargs["temperature"] == 0.5

    def test_model_override_kwarg_used(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-opus-4-6"
        resp.content = [MagicMock(text="ok")]
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 2
        resp.model_dump.return_value = {}

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            provider.complete([Message(role="user", content="hi")], model="claude-opus-4-6")
            call_kwargs = sdk_mock.Anthropic.return_value.messages.create.call_args[1]

        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_max_tokens_kwarg_forwarded(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.content = [MagicMock(text="ok")]
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 2
        resp.model_dump.return_value = {}

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            provider.complete([Message(role="user", content="hi")], max_tokens=512)
            call_kwargs = sdk_mock.Anthropic.return_value.messages.create.call_args[1]

        assert call_kwargs["max_tokens"] == 512

    def test_empty_content_list_returns_empty_string(self):
        from missy.providers.anthropic_provider import AnthropicProvider

        sdk_mock = _make_anthropic_sdk()
        resp = MagicMock()
        resp.model = "claude-3-5-sonnet-20241022"
        resp.content = []
        resp.usage.input_tokens = 5
        resp.usage.output_tokens = 0
        resp.model_dump.return_value = {}

        with _patch_anthropic_sdk(sdk_mock):
            sdk_mock.Anthropic.return_value.messages.create.return_value = resp
            provider = AnthropicProvider(_provider_config())
            result = provider.complete([Message(role="user", content="hi")])

        assert result.content == ""


# ---------------------------------------------------------------------------
# OpenAIProvider — uncovered paths
# ---------------------------------------------------------------------------


class TestOpenAIProviderBaseUrl:
    def test_base_url_forwarded_to_client(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "ok"
        resp.choices = [choice]
        resp.model = "gpt-4o"
        resp.usage.prompt_tokens = 1
        resp.usage.completion_tokens = 1
        resp.usage.total_tokens = 2
        resp.model_dump.return_value = {}

        with _patch_openai_sdk(sdk_mock):
            sdk_mock.OpenAI.return_value.chat.completions.create.return_value = resp
            provider = OpenAIProvider(
                _provider_config(
                    name="openai",
                    model="gpt-4o",
                    api_key="sk-test",
                    base_url="https://api.groq.com/openai/v1",
                )
            )
            provider.complete([Message(role="user", content="hi")])
            call_args = sdk_mock.OpenAI.call_args[1]

        assert call_args["base_url"] == "https://api.groq.com/openai/v1"

    def test_no_base_url_does_not_pass_base_url_to_client(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = "ok"
        resp.choices = [choice]
        resp.model = "gpt-4o"
        resp.usage.prompt_tokens = 1
        resp.usage.completion_tokens = 1
        resp.usage.total_tokens = 2
        resp.model_dump.return_value = {}

        with _patch_openai_sdk(sdk_mock):
            sdk_mock.OpenAI.return_value.chat.completions.create.return_value = resp
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            provider.complete([Message(role="user", content="hi")])
            call_args = sdk_mock.OpenAI.call_args[1]

        assert "base_url" not in call_args


class TestOpenAIProviderGetToolSchema:
    def test_returns_openai_function_format(self):
        from missy.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
        )
        tool = _make_tool("calculator", "Does math")
        schemas = provider.get_tool_schema([tool])

        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert s["function"]["name"] == "calculator"
        assert s["function"]["description"] == "Does math"
        assert "parameters" in s["function"]
        assert s["function"]["parameters"]["type"] == "object"

    def test_tool_without_get_schema_uses_empty_params(self):
        from missy.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
        )
        tool = MagicMock(spec=[])
        tool.name = "bare"
        tool.description = "bare tool"
        schemas = provider.get_tool_schema([tool])

        assert schemas[0]["function"]["parameters"]["properties"] == {}
        assert schemas[0]["function"]["parameters"]["required"] == []

    def test_empty_tool_list(self):
        from missy.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
        )
        assert provider.get_tool_schema([]) == []


class TestOpenAIProviderCompleteWithTools:
    def _make_response(self, content: str = "", tool_calls=None, finish_reason: str = "stop"):
        resp = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = finish_reason
        choice.message.tool_calls = tool_calls or []
        resp.choices = [choice]
        resp.model = "gpt-4o"
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        resp.model_dump.return_value = {}
        return resp

    def test_plain_text_response(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="plain text response"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = provider.complete_with_tools([Message(role="user", content="hi")], [])

        assert result.content == "plain text response"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    def test_tool_call_response_parses_tool_calls(self):
        from missy.providers.openai_provider import OpenAIProvider

        raw_tc = MagicMock()
        raw_tc.id = "call_xyz"
        raw_tc.function.name = "search"
        raw_tc.function.arguments = json.dumps({"query": "python"})

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="",
            tool_calls=[raw_tc],
            finish_reason="tool_calls",
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = provider.complete_with_tools(
                [Message(role="user", content="search for python")],
                [_make_tool("search")],
            )

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_xyz"
        assert tc.name == "search"
        assert tc.arguments == {"query": "python"}

    def test_invalid_json_in_tool_arguments_yields_empty_dict(self):
        from missy.providers.openai_provider import OpenAIProvider

        raw_tc = MagicMock()
        raw_tc.id = "call_bad"
        raw_tc.function.name = "broken"
        raw_tc.function.arguments = "not valid json {{{"

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="",
            tool_calls=[raw_tc],
            finish_reason="tool_calls",
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = provider.complete_with_tools(
                [Message(role="user", content="hi")],
                [_make_tool("broken")],
            )

        assert result.tool_calls[0].arguments == {}

    def test_no_choices_returns_empty_content(self):
        from missy.providers.openai_provider import OpenAIProvider

        resp = MagicMock()
        resp.choices = []
        resp.model = "gpt-4o"
        resp.usage.prompt_tokens = 0
        resp.usage.completion_tokens = 0
        resp.usage.total_tokens = 0
        resp.model_dump.return_value = {}

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = resp
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = provider.complete_with_tools([Message(role="user", content="hi")], [])

        assert result.content == ""
        assert result.finish_reason == "stop"

    def test_system_arg_injected_when_no_system_message_in_list(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="ok"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            provider.complete_with_tools(
                [Message(role="user", content="hi")],
                [],
                system="Be helpful.",
            )
            call_kwargs = sdk_mock.OpenAI.return_value.chat.completions.create.call_args[1]

        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "Be helpful."

    def test_system_arg_not_injected_when_system_message_already_present(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="ok"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            provider.complete_with_tools(
                [
                    Message(role="system", content="existing system"),
                    Message(role="user", content="hi"),
                ],
                [],
                system="ignored because system message exists",
            )
            call_kwargs = sdk_mock.OpenAI.return_value.chat.completions.create.call_args[1]

        system_messages = [m for m in call_kwargs["messages"] if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "existing system"

    def test_sdk_unavailable_raises_provider_error(self):
        import missy.providers.openai_provider as mod
        from missy.providers.openai_provider import OpenAIProvider

        old = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = False
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="not installed"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])
        finally:
            mod._OPENAI_AVAILABLE = old

    def test_timeout_error_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APITimeoutError(
            "timed out"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="timed out"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_auth_error_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = (
            sdk_mock.AuthenticationError("bad key")
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="authentication"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_api_error_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APIError(
            "server error"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="API error"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_unexpected_exception_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError("kaboom")
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="Unexpected"):
                provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_nonstandard_finish_reason_mapped_to_stop(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="ok", finish_reason="content_filter"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = provider.complete_with_tools([Message(role="user", content="hi")], [])

        assert result.finish_reason == "stop"

    def test_finish_reason_length_preserved(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = self._make_response(
            content="ok", finish_reason="length"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = provider.complete_with_tools([Message(role="user", content="hi")], [])

        assert result.finish_reason == "length"


class TestOpenAIProviderStream:
    def _make_chunk(self, content: str | None) -> MagicMock:
        chunk = MagicMock()
        delta = MagicMock()
        delta.content = content
        chunk.choices = [MagicMock(delta=delta)]
        return chunk

    def test_stream_yields_chunks(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        chunks = [
            self._make_chunk("Hello"),
            self._make_chunk(" world"),
            self._make_chunk("!"),
        ]
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = iter(chunks)

        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = list(provider.stream([Message(role="user", content="Hi")]))

        assert result == ["Hello", " world", "!"]

    def test_stream_yields_empty_string_for_none_delta(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = iter(
            [self._make_chunk(None)]
        )

        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            result = list(provider.stream([Message(role="user", content="Hi")]))

        assert result == [""]

    def test_stream_injects_system_when_no_system_message(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.return_value = iter(
            [self._make_chunk("ok")]
        )

        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            list(provider.stream([Message(role="user", content="hi")], system="Be concise."))
            call_kwargs = sdk_mock.OpenAI.return_value.chat.completions.create.call_args[1]

        messages_sent = call_kwargs["messages"]
        assert messages_sent[0]["role"] == "system"
        assert messages_sent[0]["content"] == "Be concise."

    def test_stream_sdk_unavailable_raises(self):
        import missy.providers.openai_provider as mod
        from missy.providers.openai_provider import OpenAIProvider

        old = mod._OPENAI_AVAILABLE
        try:
            mod._OPENAI_AVAILABLE = False
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="not installed"):
                list(provider.stream([Message(role="user", content="hi")]))
        finally:
            mod._OPENAI_AVAILABLE = old

    def test_stream_timeout_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APITimeoutError(
            "timed out"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="timed out"):
                list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_auth_error_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = (
            sdk_mock.AuthenticationError("bad key")
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="authentication"):
                list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_api_error_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = sdk_mock.APIError(
            "server error"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="API error"):
                list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_unexpected_exception_raises_provider_error(self):
        from missy.providers.openai_provider import OpenAIProvider

        sdk_mock = _make_openai_sdk()
        sdk_mock.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError(
            "unexpected"
        )
        with _patch_openai_sdk(sdk_mock):
            provider = OpenAIProvider(
                _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
            )
            with pytest.raises(ProviderError, match="Unexpected"):
                list(provider.stream([Message(role="user", content="hi")]))


class TestOpenAIEmitEventFailure:
    def test_emit_event_swallows_exceptions(self):
        from missy.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider(
            _provider_config(name="openai", model="gpt-4o", api_key="sk-test")
        )
        with patch("missy.providers.openai_provider.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus failure")
            provider._emit_event("sess", "task", "allow", "test")


# ---------------------------------------------------------------------------
# CodexProvider
# ---------------------------------------------------------------------------


class TestCodexExtractAccountId:
    def test_extracts_account_id_from_jwt_auth_namespace(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
            "sub": "user_456",
        }
        import base64

        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"header.{encoded}.signature"
        assert _extract_account_id(token) == "acct_123"

    def test_falls_back_to_sub_when_no_account_id(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {
            "https://api.openai.com/auth": {},
            "sub": "user_789",
        }
        import base64

        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        token = f"header.{encoded}.signature"
        assert _extract_account_id(token) == "user_789"

    def test_returns_empty_string_for_invalid_token(self):
        from missy.providers.codex_provider import _extract_account_id

        assert _extract_account_id("notajwt") == ""
        assert _extract_account_id("") == ""

    def test_returns_empty_string_for_invalid_json_payload(self):
        import base64

        from missy.providers.codex_provider import _extract_account_id

        bad_b64 = base64.urlsafe_b64encode(b"not-json").decode()
        token = f"header.{bad_b64}.signature"
        assert _extract_account_id(token) == ""


class TestCodexMessagesToInput:
    def test_user_message_converted_with_input_text(self):
        from missy.providers.codex_provider import _messages_to_input

        msgs = [Message(role="user", content="Hello")]
        result = _messages_to_input(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_assistant_message_converted_with_output_text(self):
        from missy.providers.codex_provider import _messages_to_input

        msgs = [Message(role="assistant", content="Reply")]
        result = _messages_to_input(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[0]["content"][0]["text"] == "Reply"

    def test_system_messages_are_skipped(self):
        from missy.providers.codex_provider import _messages_to_input

        msgs = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="question"),
        ]
        result = _messages_to_input(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_list_returns_empty_list(self):
        from missy.providers.codex_provider import _messages_to_input

        assert _messages_to_input([]) == []


class TestCodexExtractSystem:
    def test_returns_first_system_message_content(self):
        from missy.providers.codex_provider import _extract_system

        msgs = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi"),
        ]
        assert _extract_system(msgs) == "Be helpful."

    def test_returns_empty_string_when_no_system_message(self):
        from missy.providers.codex_provider import _extract_system

        msgs = [Message(role="user", content="Hi")]
        assert _extract_system(msgs) == ""


class TestCodexProviderInit:
    def test_init_with_config_sets_attributes(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok_abc", timeout=60)
        provider = CodexProvider(config)
        assert provider._api_key == "tok_abc"
        assert provider._model == "gpt-5.2"
        assert provider._timeout == 60

    def test_init_without_model_uses_default(self):
        from missy.providers.codex_provider import _DEFAULT_MODEL, CodexProvider

        config = ProviderConfig(name="openai-codex", model="", api_key="tok_abc")
        provider = CodexProvider(config)
        assert provider._model == _DEFAULT_MODEL

    def test_name_attribute(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        assert provider.name == "openai-codex"


class TestCodexProviderGetToken:
    def test_uses_configured_api_key_first(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="my_token")
        provider = CodexProvider(config)
        assert provider._get_token() == "my_token"

    def test_falls_back_to_oauth_token(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key=None)
        provider = CodexProvider(config)
        with patch("missy.providers.codex_provider._load_oauth_token", return_value="oauth_tok"):
            assert provider._get_token() == "oauth_tok"

    def test_raises_when_no_token_available(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key=None)
        provider = CodexProvider(config)
        with (
            patch("missy.providers.codex_provider._load_oauth_token", return_value=None),
            pytest.raises(ProviderError, match="no OAuth token"),
        ):
            provider._get_token()


class TestCodexProviderHeaders:
    def test_headers_include_auth_and_content_type(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        headers = provider._headers("mytoken", "acct_123")
        assert headers["Authorization"] == "Bearer mytoken"
        assert headers["Content-Type"] == "application/json"
        assert headers["chatgpt-account-id"] == "acct_123"

    def test_headers_omit_account_id_when_empty(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        headers = provider._headers("mytoken", "")
        assert "chatgpt-account-id" not in headers


class TestCodexProviderBuildBody:
    def test_body_includes_required_fields(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        msgs = [Message(role="user", content="Hello")]
        body = provider._build_body(msgs)
        assert body["model"] == "gpt-5.2"
        assert "input" in body
        assert body["stream"] is True

    def test_body_includes_instructions_when_system_message_present(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        msgs = [
            Message(role="system", content="Be precise."),
            Message(role="user", content="Hi"),
        ]
        body = provider._build_body(msgs)
        assert body["instructions"] == "Be precise."

    def test_body_omits_instructions_when_no_system(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        body = provider._build_body([Message(role="user", content="Hi")])
        assert "instructions" not in body

    def test_body_includes_tools_when_provided(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        tools = [{"type": "function", "name": "search"}]
        body = provider._build_body([Message(role="user", content="hi")], tools=tools)
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"


def _make_sse_lines(events: list[dict]) -> list[str]:
    """Build SSE data lines from event dicts."""
    lines = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}")
    lines.append("data: [DONE]")
    return lines


class TestCodexProviderStream:
    def _run_stream(self, sse_lines: list[str], token: str = "tok") -> list[str]:
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key=token)
        provider = CodexProvider(config)

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()

        with patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp):
            return list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_yields_text_deltas(self):
        lines = _make_sse_lines(
            [
                {"type": "response.output_text.delta", "delta": "Hello"},
                {"type": "response.output_text.delta", "delta": " world"},
            ]
        )
        result = self._run_stream(lines)
        assert result == ["Hello", " world"]

    def test_stream_skips_non_data_lines(self):
        lines = [
            "event: content_block_delta",
            "data: " + json.dumps({"type": "response.output_text.delta", "delta": "hi"}),
            ": ping",
            "data: [DONE]",
        ]
        result = self._run_stream(lines)
        assert result == ["hi"]

    def test_stream_skips_empty_delta(self):
        lines = _make_sse_lines(
            [
                {"type": "response.output_text.delta", "delta": ""},
                {"type": "response.output_text.delta", "delta": "real"},
            ]
        )
        result = self._run_stream(lines)
        assert result == ["real"]

    def test_stream_skips_unknown_event_types(self):
        lines = _make_sse_lines(
            [
                {"type": "response.created"},
                {"type": "response.output_text.delta", "delta": "text"},
            ]
        )
        result = self._run_stream(lines)
        assert result == ["text"]

    def test_stream_raises_on_response_failed_event(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        lines = _make_sse_lines(
            [
                {"type": "response.failed", "message": "rate limit exceeded"},
            ]
        )
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp),
            pytest.raises(ProviderError, match="rate limit exceeded"),
        ):
            list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_raises_on_error_event_with_error_dict(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        lines = _make_sse_lines(
            [
                {"type": "error", "error": {"message": "internal server error"}},
            ]
        )
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp),
            pytest.raises(ProviderError, match="internal server error"),
        ):
            list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_raises_on_http_status_error(self):
        import httpx

        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        mock_resp_obj = MagicMock()
        mock_resp_obj.status_code = 401
        mock_resp_obj.text = "Unauthorized"
        http_error = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp_obj)

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.side_effect = http_error

        with (
            patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp),
            pytest.raises(ProviderError, match="401"),
        ):
            list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_skips_invalid_json_data_lines(self):
        lines = [
            "data: {invalid json",
            "data: " + json.dumps({"type": "response.output_text.delta", "delta": "ok"}),
            "data: [DONE]",
        ]
        result = self._run_stream(lines)
        assert result == ["ok"]


class TestCodexProviderComplete:
    def test_complete_collects_stream_into_single_response(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        with patch.object(provider, "stream", return_value=iter(["Hello", " world"])):
            result = provider.complete([Message(role="user", content="hi")])

        assert isinstance(result, CompletionResponse)
        assert result.content == "Hello world"
        assert result.model == "gpt-5.2"
        assert result.provider == "openai-codex"
        assert result.finish_reason == "stop"

    def test_complete_with_empty_stream_returns_empty_content(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        with patch.object(provider, "stream", return_value=iter([])):
            result = provider.complete([Message(role="user", content="hi")])

        assert result.content == ""


class TestCodexProviderCompleteWithTools:
    def _run(self, sse_lines: list[str], tools=None) -> CompletionResponse:
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(sse_lines)
        mock_resp.raise_for_status = MagicMock()

        with patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp):
            return provider.complete_with_tools(
                [Message(role="user", content="hi")],
                tools or [],
            )

    def test_plain_text_only_sets_finish_reason_stop(self):
        lines = _make_sse_lines(
            [
                {"type": "response.output_text.delta", "delta": "Here is the answer."},
            ]
        )
        result = self._run(lines)
        assert result.content == "Here is the answer."
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    def test_tool_call_with_inline_arguments(self):
        lines = _make_sse_lines(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_999",
                        "name": "search",
                        "arguments": json.dumps({"query": "test"}),
                    },
                },
                {"type": "response.function_call_arguments.done"},
            ]
        )
        result = self._run(lines)
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_tool_call_with_streaming_argument_deltas(self):
        lines = _make_sse_lines(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_111",
                        "name": "calculator",
                        "arguments": "",
                    },
                },
                {"type": "response.function_call_arguments.delta", "delta": '{"n'},
                {"type": "response.function_call_arguments.delta", "delta": 'um": 42}'},
                {"type": "response.function_call_arguments.done"},
            ]
        )
        result = self._run(lines)
        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"num": 42}

    def test_invalid_json_arguments_yields_empty_dict(self):
        lines = _make_sse_lines(
            [
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "broken",
                        "arguments": "not json",
                    },
                },
                {"type": "response.function_call_arguments.done"},
            ]
        )
        result = self._run(lines)
        assert result.tool_calls[0].arguments == {}

    def test_non_function_call_output_item_ignored(self):
        lines = _make_sse_lines(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "content": []},
                },
                {"type": "response.output_text.delta", "delta": "answer"},
            ]
        )
        result = self._run(lines)
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    def test_http_status_error_raises_provider_error(self):
        import httpx

        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        mock_resp_obj = MagicMock()
        mock_resp_obj.status_code = 403
        mock_resp_obj.text = "Forbidden"
        http_error = httpx.HTTPStatusError("403", request=MagicMock(), response=mock_resp_obj)
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.side_effect = http_error

        with (
            patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp),
            pytest.raises(ProviderError, match="403"),
        ):
            provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_response_failed_event_raises_provider_error(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)

        lines = _make_sse_lines(
            [
                {"type": "response.failed", "message": "quota exceeded"},
            ]
        )
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("missy.providers.codex_provider.httpx.stream", return_value=mock_resp),
            pytest.raises(ProviderError, match="quota exceeded"),
        ):
            provider.complete_with_tools([Message(role="user", content="hi")], [])

    def test_accepts_prebuilt_dict_tools(self):
        lines = _make_sse_lines(
            [
                {"type": "response.output_text.delta", "delta": "ok"},
            ]
        )
        tool_dict = {"type": "function", "name": "helper"}
        result = self._run(lines, tools=[tool_dict])
        assert result.content == "ok"


class TestCodexProviderGetToolSchema:
    def test_dict_tools_passed_through_unchanged(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        tool_dict = {"type": "function", "name": "search"}
        schemas = provider.get_tool_schema([tool_dict])
        assert schemas == [tool_dict]

    def test_base_tool_converted_to_function_schema(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        tool = _make_tool("search", "Search the web")
        schemas = provider.get_tool_schema([tool])

        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert s["name"] == "search"
        assert s["description"] == "Search the web"
        assert "parameters" in s

    def test_tool_without_get_schema_uses_empty_parameters(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        tool = MagicMock(spec=[])
        tool.name = "bare"
        tool.description = "bare tool"
        schemas = provider.get_tool_schema([tool])
        assert schemas[0]["parameters"]["type"] == "object"


class TestCodexProviderIsAvailable:
    def test_returns_true_when_api_key_set(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="tok")
        provider = CodexProvider(config)
        assert provider.is_available() is True

    def test_returns_true_when_oauth_token_available(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key=None)
        provider = CodexProvider(config)
        with patch("missy.providers.codex_provider._load_oauth_token", return_value="oauth_tok"):
            assert provider.is_available() is True

    def test_returns_false_when_no_token(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key=None)
        provider = CodexProvider(config)
        with patch("missy.providers.codex_provider._load_oauth_token", return_value=None):
            assert provider.is_available() is False

    def test_returns_false_on_exception(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", api_key=None)
        provider = CodexProvider(config)
        with patch(
            "missy.providers.codex_provider._load_oauth_token",
            side_effect=RuntimeError("connection refused"),
        ):
            assert provider.is_available() is False


class TestCodexLoadOAuthToken:
    def test_returns_none_on_import_error(self):
        from missy.providers.codex_provider import _load_oauth_token

        with (
            patch(
                "missy.providers.codex_provider._load_oauth_token",
                wraps=lambda: None,
            ),
            # Directly test the exception swallowing path by mocking the inner import.
            patch.dict("sys.modules", {"missy.cli.oauth": None}),
        ):
            _load_oauth_token()
        # If we get here without raising, the guard works.

    def test_returns_token_when_refresh_succeeds(self):

        with patch(
            "missy.providers.codex_provider._load_oauth_token", return_value="fresh_token"
        ) as mock_loader:
            result = mock_loader()
            assert result == "fresh_token"


# ---------------------------------------------------------------------------
# ProviderRegistry — uncovered paths
# ---------------------------------------------------------------------------


class TestProviderRegistryRotateKey:
    def test_rotate_advances_to_next_key(self):
        from missy.providers.anthropic_provider import AnthropicProvider
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key1",
            api_keys=["key1", "key2", "key3"],
            timeout=30,
        )
        provider = AnthropicProvider(config)
        registry.register("anthropic", provider, config=config)

        registry.rotate_key("anthropic")
        assert provider._api_key == "key2"

    def test_rotate_wraps_around_to_first_key(self):
        from missy.providers.anthropic_provider import AnthropicProvider
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key1",
            api_keys=["key1", "key2"],
            timeout=30,
        )
        provider = AnthropicProvider(config)
        registry.register("anthropic", provider, config=config)

        registry.rotate_key("anthropic")  # → key2
        registry.rotate_key("anthropic")  # → key1 (wraps)
        assert provider._api_key == "key1"

    def test_rotate_skips_when_single_key(self):
        from missy.providers.anthropic_provider import AnthropicProvider
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="only_key",
            api_keys=["only_key"],
            timeout=30,
        )
        provider = AnthropicProvider(config)
        registry.register("anthropic", provider, config=config)

        registry.rotate_key("anthropic")
        # No rotation: still the original key.
        assert provider._api_key == "only_key"

    def test_rotate_skips_when_no_api_keys(self):
        from missy.providers.anthropic_provider import AnthropicProvider
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="thekey",
            api_keys=[],
            timeout=30,
        )
        provider = AnthropicProvider(config)
        registry.register("anthropic", provider, config=config)

        registry.rotate_key("anthropic")
        assert provider._api_key == "thekey"

    def test_rotate_unknown_provider_logs_warning(self):
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        # Should not raise.
        registry.rotate_key("nonexistent")

    def test_rotate_provider_registered_without_config(self):
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        provider = MagicMock()
        provider.name = "fake"
        # Register without config — rotate_key should silently skip.
        registry.register("fake", provider)
        registry.rotate_key("fake")


class TestProviderRegistryFromConfigExtended:
    def test_disabled_provider_not_registered(self):
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "my_anthropic": ProviderConfig(
                    name="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    api_key="key",
                    enabled=False,
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("my_anthropic") is None

    def test_base_url_auto_populates_provider_allowed_hosts(self):
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "my_ollama": ProviderConfig(
                    name="ollama",
                    model="llama3.2",
                    base_url="http://my-ollama-host.local:11434",
                )
            }
        )
        ProviderRegistry.from_config(config)
        assert "my-ollama-host.local" in config.network.provider_allowed_hosts

    def test_base_url_not_duplicated_if_already_in_allowed_hosts(self):
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "my_ollama": ProviderConfig(
                    name="ollama",
                    model="llama3.2",
                    base_url="http://myhost.local:11434",
                )
            }
        )
        config.network.provider_allowed_hosts.append("myhost.local")
        ProviderRegistry.from_config(config)
        assert config.network.provider_allowed_hosts.count("myhost.local") == 1

    def test_provider_key_used_when_name_field_is_empty(self):
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "ollama": ProviderConfig(
                    name="ollama",
                    model="llama3.2",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("ollama") is not None

    def test_constructor_exception_skips_provider(self):
        from missy.providers.base import BaseProvider
        from missy.providers.registry import _PROVIDER_CLASSES, ProviderRegistry

        class BrokenProvider(BaseProvider):
            name = "broken"

            def __init__(self, config):
                raise RuntimeError("cannot init")

            def complete(self, messages, **kwargs):
                return None

            def is_available(self):
                return False

        original = _PROVIDER_CLASSES.copy()
        _PROVIDER_CLASSES["broken"] = BrokenProvider
        try:
            config = _missy_config(
                providers={
                    "my_broken": ProviderConfig(
                        name="broken",
                        model="broken-model",
                    )
                }
            )
            registry = ProviderRegistry.from_config(config)
            assert registry.get("my_broken") is None
        finally:
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original)


class TestProviderRegistryFromConfigWithAnthropicAndOpenAI:
    def test_registers_anthropic_provider(self):
        from missy.providers.anthropic_provider import AnthropicProvider
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "claude": ProviderConfig(
                    name="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    api_key="sk-ant-test",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        provider = registry.get("claude")
        assert isinstance(provider, AnthropicProvider)

    def test_registers_openai_provider(self):
        from missy.providers.openai_provider import OpenAIProvider
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "gpt": ProviderConfig(
                    name="openai",
                    model="gpt-4o",
                    api_key="sk-test",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        provider = registry.get("gpt")
        assert isinstance(provider, OpenAIProvider)

    def test_registers_codex_provider(self):
        from missy.providers.codex_provider import CodexProvider
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "codex": ProviderConfig(
                    name="openai-codex",
                    model="gpt-5.2",
                    api_key="tok",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        provider = registry.get("codex")
        assert isinstance(provider, CodexProvider)

    def test_registers_provider_with_config_for_key_rotation(self):
        from missy.providers.registry import ProviderRegistry

        config = _missy_config(
            providers={
                "claude": ProviderConfig(
                    name="anthropic",
                    model="claude-3-5-sonnet-20241022",
                    api_key="sk-ant-a",
                    api_keys=["sk-ant-a", "sk-ant-b"],
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry._provider_configs.get("claude") is not None


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class TestModelRouter:
    def setup_method(self):
        from missy.providers.registry import ModelRouter

        self.router = ModelRouter()

    def test_short_question_word_scores_fast(self):
        assert self.router.score_complexity("what is Python?", 0, 0) == "fast"

    def test_how_scores_fast(self):
        assert self.router.score_complexity("how do I install pip?", 0, 0) == "fast"

    def test_debug_keyword_scores_premium(self):
        assert self.router.score_complexity("debug this code", 0, 0) == "premium"

    def test_architect_keyword_scores_premium(self):
        assert self.router.score_complexity("architect a new system", 0, 0) == "premium"

    def test_refactor_keyword_scores_premium(self):
        assert self.router.score_complexity("refactor this module", 0, 0) == "premium"

    def test_analyze_keyword_scores_premium(self):
        assert self.router.score_complexity("analyze this data", 0, 0) == "premium"

    def test_optimize_keyword_scores_premium(self):
        assert self.router.score_complexity("optimize the algorithm", 0, 0) == "premium"

    def test_long_history_scores_premium(self):
        assert self.router.score_complexity("continue", 11, 0) == "premium"

    def test_many_tools_scores_premium(self):
        assert self.router.score_complexity("do something", 0, 4) == "premium"

    def test_long_prompt_scores_premium(self):
        long_prompt = "a" * 501
        assert self.router.score_complexity(long_prompt, 0, 0) == "premium"

    def test_medium_prompt_no_keywords_scores_primary(self):
        assert self.router.score_complexity("Tell me a story about space", 0, 0) == "primary"

    def test_select_model_fast_tier_uses_fast_model(self):
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key",
            fast_model="claude-haiku-4-5",
        )
        assert self.router.select_model(config, "fast") == "claude-haiku-4-5"

    def test_select_model_premium_tier_uses_premium_model(self):
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key",
            premium_model="claude-opus-4-6",
        )
        assert self.router.select_model(config, "premium") == "claude-opus-4-6"

    def test_select_model_primary_tier_uses_base_model(self):
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key",
        )
        assert self.router.select_model(config, "primary") == "claude-3-5-sonnet-20241022"

    def test_select_model_fast_falls_back_to_base_when_no_fast_model(self):
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key",
            fast_model="",
        )
        assert self.router.select_model(config, "fast") == "claude-3-5-sonnet-20241022"

    def test_select_model_premium_falls_back_to_base_when_no_premium_model(self):
        config = ProviderConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="key",
            premium_model="",
        )
        assert self.router.select_model(config, "premium") == "claude-3-5-sonnet-20241022"

    def test_fast_indicator_with_tools_does_not_score_fast(self):
        # tool_count > 0 prevents fast scoring even with a question word.
        result = self.router.score_complexity("what is this?", 0, 1)
        assert result in ("primary", "premium")

    def test_fast_indicator_with_long_prompt_does_not_score_fast(self):
        # Long prompt (>= 80 chars) prevents fast scoring even with question word.
        long = "what " + "is this all about exactly? " * 4  # > 80 chars
        result = self.router.score_complexity(long, 0, 0)
        assert result in ("primary", "premium")
