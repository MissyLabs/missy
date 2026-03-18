"""Comprehensive unit tests for missy/providers/codex_provider.py.

Covers all public and private surfaces:

  Module-level helpers:
    - _extract_account_id  — JWT payload extraction, fallbacks, invalid tokens
    - _load_oauth_token    — happy path, None from refresh, import failure
    - _messages_to_input   — user/assistant/system roles
    - _extract_system      — first system message, no system message

  CodexProvider class:
    - __init__             — attribute storage, defaults
    - is_available()       — key present, OAuth token, no token, exception
    - _get_token()         — api_key priority, OAuth fallback, raises ProviderError
    - _headers()           — Auth header, account-id conditional
    - _build_body()        — required fields, instructions, tools
    - _extract_text_from_response() — all output shapes, fallbacks
    - stream()             — SSE parsing (deltas, skips, errors, HTTP errors)
    - complete()           — delegates to stream, returns CompletionResponse
    - complete_with_tools()— tool call accumulation (inline args, delta args,
                             done event, text-only, error events, HTTP errors,
                             non-data lines skipped, invalid JSON skipped,
                             dict tools vs BaseTool instances)
    - get_tool_schema()    — dict pass-through, BaseTool conversion, no get_schema
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.base import Message

# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------


def _make_config(
    api_key: str | None = "tok-test",
    model: str = "gpt-5.2",
    timeout: int = 30,
) -> ProviderConfig:
    return ProviderConfig(name="openai-codex", model=model, api_key=api_key, timeout=timeout)


def _make_messages(*pairs: tuple[str, str]) -> list[Message]:
    """Build a Message list from (role, content) pairs."""
    return [Message(role=r, content=c) for r, c in pairs]


def _make_jwt_token(payload: dict) -> str:
    """Construct a syntactically valid (but unsigned) JWT string."""
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    body_bytes = json.dumps(payload).encode()
    body = base64.urlsafe_b64encode(body_bytes).rstrip(b"=").decode()
    return f"{header}.{body}.fakesig"


@contextmanager
def _mock_sse_stream(lines: list[str]):
    """Patch httpx.stream to return an iterator over *lines*."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(lines))
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_resp)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    with patch("httpx.stream", return_value=mock_ctx):
        yield mock_ctx


def _sse(*events: dict, done: bool = True) -> list[str]:
    """Build SSE line list from event dicts."""
    lines = [f"data: {json.dumps(e)}" for e in events]
    if done:
        lines.append("data: [DONE]")
    return lines


# ===========================================================================
# Module-level helpers
# ===========================================================================


class TestExtractAccountId:
    """Tests for _extract_account_id."""

    def test_extracts_from_auth_namespace(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        token = _make_jwt_token(payload)
        assert _extract_account_id(token) == "acct-123"

    def test_falls_back_to_sub_when_no_account_id_in_namespace(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {
            "https://api.openai.com/auth": {},
            "sub": "user-abc",
        }
        token = _make_jwt_token(payload)
        assert _extract_account_id(token) == "user-abc"

    def test_empty_auth_namespace_uses_sub(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {"sub": "user-xyz"}
        token = _make_jwt_token(payload)
        assert _extract_account_id(token) == "user-xyz"

    def test_returns_empty_string_for_token_with_one_part(self):
        from missy.providers.codex_provider import _extract_account_id

        assert _extract_account_id("notajwt") == ""

    def test_returns_empty_string_for_invalid_base64_payload(self):
        from missy.providers.codex_provider import _extract_account_id

        assert _extract_account_id("header.!!!invalid!!!.sig") == ""

    def test_returns_empty_string_for_non_json_payload(self):
        from missy.providers.codex_provider import _extract_account_id

        bad_payload = base64.urlsafe_b64encode(b"not-json").decode()
        token = f"header.{bad_payload}.sig"
        assert _extract_account_id(token) == ""

    def test_returns_empty_string_when_payload_has_no_relevant_keys(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {"iss": "openai.com", "exp": 9999999999}
        token = _make_jwt_token(payload)
        assert _extract_account_id(token) == ""

    def test_account_id_preferred_over_sub(self):
        from missy.providers.codex_provider import _extract_account_id

        payload = {
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct-preferred"},
            "sub": "user-fallback",
        }
        token = _make_jwt_token(payload)
        assert _extract_account_id(token) == "acct-preferred"


class TestLoadOAuthToken:
    """Tests for _load_oauth_token."""

    def test_returns_token_when_refresh_succeeds(self):
        from missy.providers.codex_provider import _load_oauth_token

        with patch("missy.cli.oauth.refresh_token_if_needed", return_value="oauth-fresh-token"):
            result = _load_oauth_token()

        assert result == "oauth-fresh-token"

    def test_returns_none_when_refresh_returns_none(self):
        from missy.providers.codex_provider import _load_oauth_token

        with patch("missy.cli.oauth.refresh_token_if_needed", return_value=None):
            result = _load_oauth_token()

        assert result is None

    def test_returns_none_on_import_error(self):
        from missy.providers.codex_provider import _load_oauth_token

        with patch("missy.cli.oauth.refresh_token_if_needed", side_effect=ImportError("no module")):
            result = _load_oauth_token()

        assert result is None

    def test_returns_none_on_arbitrary_exception(self):
        from missy.providers.codex_provider import _load_oauth_token

        with patch(
            "missy.cli.oauth.refresh_token_if_needed", side_effect=RuntimeError("disk error")
        ):
            result = _load_oauth_token()

        assert result is None


class TestMessagesToInput:
    """Tests for _messages_to_input."""

    def test_user_message_uses_input_text_type(self):
        from missy.providers.codex_provider import _messages_to_input

        result = _messages_to_input([Message(role="user", content="hello")])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "hello"

    def test_assistant_message_uses_output_text_type(self):
        from missy.providers.codex_provider import _messages_to_input

        result = _messages_to_input([Message(role="assistant", content="world")])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[0]["content"][0]["text"] == "world"

    def test_system_messages_are_excluded(self):
        from missy.providers.codex_provider import _messages_to_input

        msgs = [
            Message(role="system", content="be helpful"),
            Message(role="user", content="hi"),
        ]
        result = _messages_to_input(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_empty_list_returns_empty_list(self):
        from missy.providers.codex_provider import _messages_to_input

        assert _messages_to_input([]) == []

    def test_mixed_roles_ordered_correctly(self):
        from missy.providers.codex_provider import _messages_to_input

        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
            Message(role="user", content="followup"),
        ]
        result = _messages_to_input(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"


class TestExtractSystem:
    """Tests for _extract_system."""

    def test_returns_first_system_message_content(self):
        from missy.providers.codex_provider import _extract_system

        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="hi"),
        ]
        assert _extract_system(msgs) == "You are helpful."

    def test_returns_empty_string_when_no_system_message(self):
        from missy.providers.codex_provider import _extract_system

        msgs = [Message(role="user", content="hi")]
        assert _extract_system(msgs) == ""

    def test_returns_empty_string_for_empty_list(self):
        from missy.providers.codex_provider import _extract_system

        assert _extract_system([]) == ""

    def test_returns_first_system_when_multiple_present(self):
        from missy.providers.codex_provider import _extract_system

        msgs = [
            Message(role="system", content="first system"),
            Message(role="system", content="second system"),
        ]
        assert _extract_system(msgs) == "first system"


# ===========================================================================
# CodexProvider.__init__
# ===========================================================================


class TestCodexProviderInit:
    def test_stores_api_key(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key="my-key"))
        assert p._api_key == "my-key"

    def test_stores_model(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(model="gpt-4o"))
        assert p._model == "gpt-4o"

    def test_uses_default_model_when_config_model_empty(self):
        from missy.providers.codex_provider import _DEFAULT_MODEL, CodexProvider

        p = CodexProvider(_make_config(model=""))
        assert p._model == _DEFAULT_MODEL

    def test_stores_timeout(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(timeout=120))
        assert p._timeout == 120

    def test_default_timeout_when_none(self):
        from missy.providers.codex_provider import CodexProvider

        config = ProviderConfig(name="openai-codex", model="gpt-5.2", timeout=0)
        p = CodexProvider(config)
        # timeout=0 is falsy, so falls back to 60
        assert p._timeout == 60

    def test_name_class_attribute(self):
        from missy.providers.codex_provider import CodexProvider

        assert CodexProvider.name == "openai-codex"

    def test_none_api_key_stored_as_none(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        assert p._api_key is None


# ===========================================================================
# CodexProvider.is_available
# ===========================================================================


class TestCodexProviderIsAvailable:
    def test_returns_true_when_api_key_set(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key="tok-abc"))
        assert p.is_available() is True

    def test_returns_true_when_oauth_token_loadable(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        with patch(
            "missy.providers.codex_provider._load_oauth_token", return_value="oauth-tok"
        ):
            assert p.is_available() is True

    def test_returns_false_when_no_key_and_no_oauth_token(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        with patch("missy.providers.codex_provider._load_oauth_token", return_value=None):
            assert p.is_available() is False

    def test_returns_false_on_exception_from_load_oauth(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        with patch(
            "missy.providers.codex_provider._load_oauth_token",
            side_effect=RuntimeError("disk error"),
        ):
            assert p.is_available() is False


# ===========================================================================
# CodexProvider._get_token
# ===========================================================================


class TestCodexProviderGetToken:
    def test_returns_configured_api_key_without_calling_oauth(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key="direct-key"))
        with patch(
            "missy.providers.codex_provider._load_oauth_token"
        ) as mock_load:
            token = p._get_token()

        assert token == "direct-key"
        mock_load.assert_not_called()

    def test_falls_back_to_oauth_when_no_api_key(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        with patch(
            "missy.providers.codex_provider._load_oauth_token", return_value="oauth-token"
        ):
            token = p._get_token()

        assert token == "oauth-token"

    def test_raises_provider_error_when_no_token_available(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        with patch("missy.providers.codex_provider._load_oauth_token", return_value=None):
            with pytest.raises(ProviderError, match="no OAuth token"):
                p._get_token()

    def test_error_message_mentions_missy_setup(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(api_key=None))
        with patch("missy.providers.codex_provider._load_oauth_token", return_value=None):
            with pytest.raises(ProviderError, match="missy setup"):
                p._get_token()


# ===========================================================================
# CodexProvider._headers
# ===========================================================================


class TestCodexProviderHeaders:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config())

    def test_includes_bearer_authorization(self):
        headers = self.provider._headers("my-token", "")
        assert headers["Authorization"] == "Bearer my-token"

    def test_includes_content_type_json(self):
        headers = self.provider._headers("my-token", "")
        assert headers["Content-Type"] == "application/json"

    def test_includes_account_id_when_provided(self):
        headers = self.provider._headers("my-token", "acct-xyz")
        assert headers["chatgpt-account-id"] == "acct-xyz"

    def test_omits_account_id_when_empty_string(self):
        headers = self.provider._headers("my-token", "")
        assert "chatgpt-account-id" not in headers

    def test_omits_account_id_when_none(self):
        # None is falsy, treated like empty
        headers = self.provider._headers("my-token", None)  # type: ignore[arg-type]
        assert "chatgpt-account-id" not in headers


# ===========================================================================
# CodexProvider._build_body
# ===========================================================================


class TestCodexProviderBuildBody:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config(model="gpt-5.2"))

    def test_body_contains_model(self):
        msgs = _make_messages(("user", "hello"))
        body = self.provider._build_body(msgs)
        assert body["model"] == "gpt-5.2"

    def test_body_contains_input_list(self):
        msgs = _make_messages(("user", "hello"))
        body = self.provider._build_body(msgs)
        assert "input" in body
        assert isinstance(body["input"], list)

    def test_body_always_sets_stream_true(self):
        msgs = _make_messages(("user", "hi"))
        body = self.provider._build_body(msgs, stream=False)
        # Codex endpoint requires stream=true always
        assert body["stream"] is True

    def test_body_store_is_false(self):
        msgs = _make_messages(("user", "hi"))
        body = self.provider._build_body(msgs)
        assert body["store"] is False

    def test_body_includes_include_field(self):
        msgs = _make_messages(("user", "hi"))
        body = self.provider._build_body(msgs)
        assert "include" in body
        assert "reasoning.encrypted_content" in body["include"]

    def test_body_includes_instructions_when_system_message_present(self):
        msgs = _make_messages(("system", "be concise"), ("user", "hello"))
        body = self.provider._build_body(msgs)
        assert body.get("instructions") == "be concise"

    def test_body_omits_instructions_when_no_system_message(self):
        msgs = _make_messages(("user", "hello"))
        body = self.provider._build_body(msgs)
        assert "instructions" not in body

    def test_body_includes_tools_when_provided(self):
        msgs = _make_messages(("user", "use a tool"))
        tools = [{"type": "function", "name": "do_thing"}]
        body = self.provider._build_body(msgs, tools=tools)
        assert body["tools"] == tools
        assert body["tool_choice"] == "auto"

    def test_body_omits_tools_when_none(self):
        msgs = _make_messages(("user", "no tools"))
        body = self.provider._build_body(msgs, tools=None)
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_body_text_verbosity_field_present(self):
        msgs = _make_messages(("user", "hi"))
        body = self.provider._build_body(msgs)
        assert body.get("text") == {"verbosity": "medium"}


# ===========================================================================
# CodexProvider._extract_text_from_response
# ===========================================================================


class TestCodexExtractTextFromResponse:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config())

    def test_extracts_output_text_from_message_item(self):
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello from Codex"}],
                }
            ]
        }
        assert self.provider._extract_text_from_response(data) == "Hello from Codex"

    def test_returns_first_output_text_part(self):
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "first"},
                        {"type": "output_text", "text": "second"},
                    ],
                }
            ]
        }
        assert self.provider._extract_text_from_response(data) == "first"

    def test_skips_non_message_output_items(self):
        data = {
            "output": [
                {"type": "function_call", "content": [{"type": "output_text", "text": "skip"}]}
            ]
        }
        assert self.provider._extract_text_from_response(data) == ""

    def test_skips_non_output_text_content_parts(self):
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "input_text", "text": "wrong type"}],
                }
            ]
        }
        assert self.provider._extract_text_from_response(data) == ""

    def test_fallback_to_text_field(self):
        data = {"output": [], "text": "fallback text"}
        assert self.provider._extract_text_from_response(data) == "fallback text"

    def test_fallback_to_content_field_when_no_text(self):
        data = {"output": [], "content": "content fallback"}
        assert self.provider._extract_text_from_response(data) == "content fallback"

    def test_empty_response_returns_empty_string(self):
        assert self.provider._extract_text_from_response({}) == ""

    def test_empty_output_list_no_fallback_returns_empty_string(self):
        data = {"output": []}
        assert self.provider._extract_text_from_response(data) == ""


# ===========================================================================
# CodexProvider.stream
# ===========================================================================


class TestCodexProviderStream:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config(api_key="tok-stream"))

    def _messages(self) -> list[Message]:
        return _make_messages(("user", "stream this"))

    def test_yields_text_delta_tokens(self):
        lines = _sse(
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
        )
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == ["Hello", " world"]

    def test_skips_non_data_lines(self):
        lines = [
            "event: response.created",
            ": keep-alive",
            'data: {"type": "response.output_text.delta", "delta": "hi"}',
            "data: [DONE]",
        ]
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == ["hi"]

    def test_skips_done_sentinel(self):
        lines = _sse({"type": "response.output_text.delta", "delta": "ok"})
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert "[DONE]" not in result

    def test_skips_empty_data_payloads(self):
        lines = [
            "data: ",
            'data: {"type": "response.output_text.delta", "delta": "x"}',
            "data: [DONE]",
        ]
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == ["x"]

    def test_skips_invalid_json_data_lines(self):
        lines = [
            "data: {{{not json",
            'data: {"type": "response.output_text.delta", "delta": "y"}',
            "data: [DONE]",
        ]
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == ["y"]

    def test_skips_events_with_empty_delta(self):
        lines = _sse(
            {"type": "response.output_text.delta", "delta": ""},
            {"type": "response.output_text.delta", "delta": "real"},
        )
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == ["real"]

    def test_skips_unrecognized_event_types(self):
        lines = _sse(
            {"type": "response.created", "response": {}},
            {"type": "response.output_text.delta", "delta": "token"},
        )
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == ["token"]

    def test_raises_provider_error_on_response_failed_event(self):
        lines = _sse({"type": "response.failed", "message": "rate limit exceeded"})
        with _mock_sse_stream(lines):
            with pytest.raises(ProviderError, match="rate limit exceeded"):
                list(self.provider.stream(self._messages()))

    def test_raises_provider_error_on_error_event(self):
        lines = _sse({"type": "error", "error": {"message": "internal error"}})
        with _mock_sse_stream(lines), pytest.raises(ProviderError, match="internal error"):
            list(self.provider.stream(self._messages()))

    def test_raises_provider_error_on_http_status_error(self):
        import httpx

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        http_error = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp)

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=http_error)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("httpx.stream", return_value=mock_ctx):
            with pytest.raises(ProviderError, match="401"):
                list(self.provider.stream(self._messages()))

    def test_empty_stream_yields_nothing(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = list(self.provider.stream(self._messages()))

        assert result == []

    def test_stream_calls_get_token(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            with patch.object(self.provider, "_get_token", return_value="tok-test") as mock_tok:
                list(self.provider.stream(self._messages()))

        mock_tok.assert_called_once()


# ===========================================================================
# CodexProvider.complete
# ===========================================================================


class TestCodexProviderComplete:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config(api_key="tok-complete"))

    def _messages(self) -> list[Message]:
        return _make_messages(("user", "complete this"))

    def test_returns_completion_response_type(self):
        from missy.providers.base import CompletionResponse

        lines = _sse({"type": "response.output_text.delta", "delta": "hi"})
        with _mock_sse_stream(lines):
            result = self.provider.complete(self._messages())

        assert isinstance(result, CompletionResponse)

    def test_content_is_concatenated_stream_output(self):
        lines = _sse(
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": ", world"},
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete(self._messages())

        assert result.content == "Hello, world"

    def test_provider_name_is_openai_codex(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete(self._messages())

        assert result.provider == "openai-codex"

    def test_model_reflects_config(self):
        from missy.providers.codex_provider import CodexProvider

        p = CodexProvider(_make_config(model="gpt-4o", api_key="k"))
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = p.complete(_make_messages(("user", "hi")))

        assert result.model == "gpt-4o"

    def test_finish_reason_is_stop(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete(self._messages())

        assert result.finish_reason == "stop"

    def test_usage_contains_expected_keys(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete(self._messages())

        assert "prompt_tokens" in result.usage
        assert "completion_tokens" in result.usage
        assert "total_tokens" in result.usage

    def test_empty_stream_returns_empty_content(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete(self._messages())

        assert result.content == ""

    def test_delegates_to_stream(self):
        """complete() should call self.stream(), not make its own HTTP request."""
        with patch.object(
            self.provider, "stream", return_value=iter(["abc", "def"])
        ) as mock_stream:
            result = self.provider.complete(self._messages())

        mock_stream.assert_called_once()
        assert result.content == "abcdef"


# ===========================================================================
# CodexProvider.complete_with_tools
# ===========================================================================


class TestCodexProviderCompleteWithTools:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config(api_key="tok-tools"))

    def _messages(self) -> list[Message]:
        return _make_messages(("user", "call a tool"))

    def test_returns_completion_response_type(self):
        from missy.providers.base import CompletionResponse

        with _mock_sse_stream(["data: [DONE]"]):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert isinstance(result, CompletionResponse)

    def test_text_only_stream_finish_reason_is_stop(self):
        lines = _sse({"type": "response.output_text.delta", "delta": "Just text"})
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.finish_reason == "stop"
        assert result.content == "Just text"
        assert result.tool_calls == []

    def test_tool_call_with_inline_arguments(self):
        lines = _sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call-001",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            },
            {
                "type": "response.function_call_arguments.done",
            },
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "London"}
        assert tc.id == "call-001"

    def test_tool_call_with_delta_argument_accumulation(self):
        lines = _sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call-002",
                    "name": "calc",
                    "arguments": "",
                },
            },
            {"type": "response.function_call_arguments.delta", "delta": '{"expr"'},
            {"type": "response.function_call_arguments.delta", "delta": ': "1+1"}'},
            {"type": "response.function_call_arguments.done"},
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"expr": "1+1"}

    def test_malformed_tool_arguments_become_empty_dict(self):
        lines = _sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call-003",
                    "name": "bad_tool",
                    "arguments": "not json",
                },
            },
            {"type": "response.function_call_arguments.done"},
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.tool_calls[0].arguments == {}

    def test_output_item_added_for_non_function_call_is_ignored(self):
        lines = _sse(
            {
                "type": "response.output_item.added",
                "item": {"type": "message", "content": []},
            },
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.tool_calls == []

    def test_non_data_lines_are_silently_skipped(self):
        lines = [
            "event: start",
            ": keep-alive comment",
            "data: [DONE]",
        ]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.finish_reason == "stop"
        assert result.content == ""

    def test_invalid_json_data_lines_are_silently_skipped(self):
        lines = [
            "data: {{{not valid json",
            "data: also bad",
            "data: [DONE]",
        ]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.tool_calls == []

    def test_done_sentinel_skipped_cleanly(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.finish_reason == "stop"

    def test_empty_data_payload_skipped(self):
        lines = ["data: ", "data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.finish_reason == "stop"

    def test_response_failed_event_raises_provider_error(self):
        lines = _sse({"type": "response.failed", "message": "quota exceeded"})
        with _mock_sse_stream(lines), pytest.raises(ProviderError, match="quota exceeded"):
            self.provider.complete_with_tools(self._messages(), tools=[])

    def test_error_event_raises_provider_error(self):
        lines = _sse({"type": "error", "error": {"message": "server crash"}})
        with _mock_sse_stream(lines), pytest.raises(ProviderError, match="server crash"):
            self.provider.complete_with_tools(self._messages(), tools=[])

    def test_http_status_error_raises_provider_error(self):
        import httpx

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        http_error = httpx.HTTPStatusError("403", request=MagicMock(), response=mock_resp)

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=http_error)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("httpx.stream", return_value=mock_ctx):
            with pytest.raises(ProviderError, match="403"):
                self.provider.complete_with_tools(self._messages(), tools=[])

    def test_accepts_prebuilt_dict_tools(self):
        dict_tool = {"type": "function", "name": "echo", "description": "echoes input"}
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[dict_tool])

        assert result.finish_reason == "stop"

    def test_accepts_base_tool_instances(self):
        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.description = "does stuff"
        mock_tool.get_schema = MagicMock(
            return_value={"parameters": {"type": "object", "properties": {}}}
        )
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[mock_tool])

        assert result.finish_reason == "stop"

    def test_multiple_tool_calls_accumulated(self):
        lines = _sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "tool_a",
                    "arguments": '{"x": 1}',
                },
            },
            {"type": "response.function_call_arguments.done"},
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "c2",
                    "name": "tool_b",
                    "arguments": '{"y": 2}',
                },
            },
            {"type": "response.function_call_arguments.done"},
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert len(result.tool_calls) == 2
        names = {tc.name for tc in result.tool_calls}
        assert names == {"tool_a", "tool_b"}

    def test_provider_name_in_response(self):
        lines = ["data: [DONE]"]
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.provider == "openai-codex"

    def test_done_event_without_name_does_not_create_tool_call(self):
        """If function_call_arguments.done fires with no current_fn name, skip."""
        lines = _sse(
            {"type": "response.function_call_arguments.done"},
        )
        with _mock_sse_stream(lines):
            result = self.provider.complete_with_tools(self._messages(), tools=[])

        assert result.tool_calls == []


# ===========================================================================
# CodexProvider.get_tool_schema
# ===========================================================================


class TestCodexProviderGetToolSchema:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config())

    def test_dict_tools_passed_through_unchanged(self):
        tool = {"type": "function", "name": "foo", "description": "bar"}
        result = self.provider.get_tool_schema([tool])
        assert result == [tool]

    def test_base_tool_converted_to_function_schema(self):
        mock_tool = MagicMock()
        mock_tool.name = "my_func"
        mock_tool.description = "Does something"
        mock_tool.get_schema = MagicMock(
            return_value={
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                    "required": ["arg"],
                }
            }
        )
        schemas = self.provider.get_tool_schema([mock_tool])
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["name"] == "my_func"
        assert schema["description"] == "Does something"
        assert schema["parameters"]["properties"]["arg"]["type"] == "string"

    def test_tool_without_get_schema_uses_empty_parameters(self):
        mock_tool = MagicMock(spec=[])  # no get_schema attribute
        mock_tool.name = "no_schema"
        mock_tool.description = "bare"
        schemas = self.provider.get_tool_schema([mock_tool])
        assert schemas[0]["parameters"]["type"] == "object"
        assert schemas[0]["parameters"]["properties"] == {}

    def test_empty_tool_list_returns_empty_list(self):
        assert self.provider.get_tool_schema([]) == []

    def test_mixed_dict_and_base_tool(self):
        dict_tool = {"type": "function", "name": "d_tool"}
        mock_tool = MagicMock()
        mock_tool.name = "b_tool"
        mock_tool.description = "base tool"
        mock_tool.get_schema = MagicMock(return_value={"parameters": {}})
        schemas = self.provider.get_tool_schema([dict_tool, mock_tool])
        assert len(schemas) == 2
        assert schemas[0] == dict_tool
        assert schemas[1]["name"] == "b_tool"
