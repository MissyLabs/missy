"""Targeted coverage tests for missy/providers/codex_provider.py.

Covers the remaining uncovered lines:
- Line 69: _load_oauth_token happy path (refresh_token_if_needed returns a value)
- Lines 172-178: _extract_text_from_response — all branches
- Line 251: complete_with_tools — non-data line skipped
- Lines 257-258: complete_with_tools — invalid JSON skipped
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from missy.config.settings import ProviderConfig


def _make_config(api_key: str | None = "tok-test", model: str = "gpt-5.2") -> ProviderConfig:
    return ProviderConfig(name="openai-codex", model=model, api_key=api_key, timeout=30)


# ---------------------------------------------------------------------------
# _load_oauth_token — happy path (line 69)
# ---------------------------------------------------------------------------


class TestLoadOAuthTokenHappyPath:
    def test_returns_token_from_refresh(self):
        """_load_oauth_token should return the token when refresh_token_if_needed succeeds."""
        from missy.providers.codex_provider import _load_oauth_token

        with patch("missy.cli.oauth.refresh_token_if_needed", return_value="oauth-token-123"):
            result = _load_oauth_token()

        assert result == "oauth-token-123"

    def test_returns_none_when_refresh_returns_none(self):
        """_load_oauth_token should propagate None from refresh_token_if_needed."""
        from missy.providers.codex_provider import _load_oauth_token

        with patch("missy.cli.oauth.refresh_token_if_needed", return_value=None):
            result = _load_oauth_token()

        assert result is None


# ---------------------------------------------------------------------------
# _extract_text_from_response — all branches (lines 172-178)
# ---------------------------------------------------------------------------


class TestExtractTextFromResponse:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config())

    def test_extracts_output_text_from_message_item(self):
        """Should return text from output[].content[type=output_text]."""
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello from Codex"},
                    ],
                }
            ]
        }
        result = self.provider._extract_text_from_response(data)
        assert result == "Hello from Codex"

    def test_skips_non_message_output_items(self):
        """Items with type != 'message' should be skipped; falls back to empty."""
        data = {
            "output": [
                {"type": "function_call", "content": [{"type": "output_text", "text": "skip"}]},
            ]
        }
        result = self.provider._extract_text_from_response(data)
        assert result == ""

    def test_skips_non_output_text_content_parts(self):
        """Content parts with type != 'output_text' should be skipped."""
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "input_text", "text": "should not appear"},
                    ],
                }
            ]
        }
        result = self.provider._extract_text_from_response(data)
        assert result == ""

    def test_fallback_to_text_field(self):
        """When output list is empty, fall back to data['text']."""
        data = {"output": [], "text": "fallback text"}
        result = self.provider._extract_text_from_response(data)
        assert result == "fallback text"

    def test_fallback_to_content_field(self):
        """When output list is empty and no 'text', fall back to data['content']."""
        data = {"output": [], "content": "content fallback"}
        result = self.provider._extract_text_from_response(data)
        assert result == "content fallback"

    def test_empty_response_returns_empty_string(self):
        """When nothing matches, return empty string."""
        result = self.provider._extract_text_from_response({})
        assert result == ""

    def test_multiple_output_items_returns_first_match(self):
        """Returns text from the first matching output_text content part."""
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
        result = self.provider._extract_text_from_response(data)
        assert result == "first"


# ---------------------------------------------------------------------------
# complete_with_tools — non-data line skip and invalid JSON skip
# (lines 251, 257-258)
# ---------------------------------------------------------------------------


def _make_sse_lines(*events) -> list[str]:
    """Build a list of SSE lines from event dicts (plus a [DONE] terminator)."""
    lines = [f"data: {json.dumps(e)}" for e in events]
    lines.append("data: [DONE]")
    return lines


@contextmanager
def _mock_stream(lines: list[str]):
    """Context manager that patches PolicyHTTPClient.post to return given SSE lines."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(lines))
    mock_resp.close = MagicMock()

    with patch("missy.providers.codex_provider.PolicyHTTPClient") as mock_cls:
        mock_cls.return_value.post.return_value = mock_resp
        yield


class TestCompleteWithToolsSkipBranches:
    def setup_method(self):
        from missy.providers.codex_provider import CodexProvider

        self.provider = CodexProvider(_make_config())

    def _messages(self):
        from missy.providers.base import Message

        return [Message(role="user", content="call a tool")]

    def test_non_data_lines_are_skipped(self):
        """Lines not starting with 'data: ' must be silently ignored (line 251)."""
        lines = [
            "event: start",  # non-data line — must be skipped
            ": keep-alive",  # another non-data line
            "data: [DONE]",
        ]
        with _mock_stream(lines):
            response = self.provider.complete_with_tools(self._messages(), tools=[])

        # No tool calls, no text — just a clean stop result.
        assert response.finish_reason == "stop"
        assert response.content == ""

    def test_invalid_json_data_lines_are_skipped(self):
        """Data lines with non-JSON payloads must be silently ignored (lines 257-258)."""
        lines = [
            "data: this is not json {{{",  # invalid JSON — must be skipped
            "data: also bad",
            "data: [DONE]",
        ]
        with _mock_stream(lines):
            response = self.provider.complete_with_tools(self._messages(), tools=[])

        assert response.finish_reason == "stop"
        assert response.tool_calls == []

    def test_done_and_empty_data_lines_skipped(self):
        """Empty data payloads and [DONE] must be skipped cleanly."""
        lines = [
            "data: ",  # empty payload
            "data: [DONE]",
        ]
        with _mock_stream(lines):
            response = self.provider.complete_with_tools(self._messages(), tools=[])

        assert response.finish_reason == "stop"

    def test_complete_with_tools_accepts_dict_tools_and_non_dict(self):
        """Verifies the isinstance(tools[0], dict) guard (tool schema conversion)."""
        from missy.providers.codex_provider import CodexProvider

        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        mock_tool.description = "does stuff"
        mock_tool.get_schema = MagicMock(
            return_value={"parameters": {"type": "object", "properties": {}}}
        )

        provider = CodexProvider(_make_config())

        lines = ["data: [DONE]"]
        with _mock_stream(lines):
            response = provider.complete_with_tools(self._messages(), tools=[mock_tool])

        assert response.finish_reason == "stop"
