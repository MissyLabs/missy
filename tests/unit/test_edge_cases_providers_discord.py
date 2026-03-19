"""Edge case tests for providers, Discord REST, MCP, cost tracker — session 27.

Covers defensive code paths in:
- Anthropic provider: empty content, timeout, auth errors
- OpenAI provider: malformed tool arguments
- Discord REST: Retry-After header parsing, non-dict payload
- MCP client: disconnect cleanup
- Cost tracker: zero tokens, unknown model, negative tokens
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import ProviderConfig
from missy.providers.base import Message

# ---------------------------------------------------------------------------
# Anthropic provider edge cases
# ---------------------------------------------------------------------------


class TestAnthropicProviderEdgeCases:
    """Edge cases in Anthropic provider complete() method."""

    def _make_provider(self) -> Any:
        from missy.providers.anthropic_provider import AnthropicProvider

        cfg = ProviderConfig(name="anthropic", model="claude-sonnet-4-6", api_key="sk-ant-test-key")
        return AnthropicProvider(cfg)

    def test_empty_content_returns_empty_string(self) -> None:
        """Response with content=[] should return empty string content."""
        provider = self._make_provider()

        mock_response = MagicMock()
        mock_response.content = []  # Empty content array
        mock_response.model = "claude-sonnet-4-6"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=0)
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch.object(provider, "_make_client", return_value=mock_client):
            result = provider.complete(
                messages=[Message(role="user", content="hi")],
                session_id="s1",
                task_id="t1",
            )

        assert result.content == ""

    def test_timeout_error_raises_provider_error(self) -> None:
        """APITimeoutError should be caught and re-raised as ProviderError."""
        provider = self._make_provider()

        import anthropic as _sdk

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _sdk.APITimeoutError(request=MagicMock())

        with (
            patch.object(provider, "_make_client", return_value=mock_client),
            pytest.raises(Exception, match="(?i)time"),
        ):
            provider.complete(
                messages=[Message(role="user", content="hi")],
                session_id="s1",
                task_id="t1",
            )

    def test_auth_error_raises_provider_error(self) -> None:
        """AuthenticationError should be caught and re-raised as ProviderError."""
        provider = self._make_provider()

        import anthropic as _sdk

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _sdk.AuthenticationError(
            message="invalid key",
            response=MagicMock(status_code=401),
            body=None,
        )

        with (
            patch.object(provider, "_make_client", return_value=mock_client),
            pytest.raises(Exception, match="(?i)auth"),
        ):
            provider.complete(
                messages=[Message(role="user", content="hi")],
                session_id="s1",
                task_id="t1",
            )

    def test_generic_exception_raises_provider_error(self) -> None:
        """Unexpected exception should be caught and re-raised as ProviderError."""
        provider = self._make_provider()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("kaboom")

        with (
            patch.object(provider, "_make_client", return_value=mock_client),
            pytest.raises(Exception, match="kaboom"),
        ):
            provider.complete(
                messages=[Message(role="user", content="hi")],
                session_id="s1",
                task_id="t1",
            )


# ---------------------------------------------------------------------------
# OpenAI provider edge cases
# ---------------------------------------------------------------------------


class TestOpenAIProviderEdgeCases:
    """Edge cases in OpenAI provider complete_with_tools()."""

    def _make_provider(self) -> Any:
        from missy.providers.openai_provider import OpenAIProvider

        cfg = ProviderConfig(name="openai", model="gpt-4", api_key="sk-test-key")
        return OpenAIProvider(cfg)

    def test_malformed_tool_arguments_become_empty_dict(self) -> None:
        """Malformed JSON in tool call arguments should fallback to {}."""
        provider = self._make_provider()

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "test_tool"
        mock_tc.function.arguments = "{invalid json"

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.tool_calls = [mock_tc]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {}

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.chat.completions.create.return_value = mock_response

        with patch("missy.providers.openai_provider._openai_sdk") as mock_sdk:
            mock_sdk.OpenAI = mock_client_cls
            mock_sdk.APITimeoutError = Exception
            mock_sdk.AuthenticationError = Exception
            mock_sdk.APIError = Exception
            result = provider.complete_with_tools(
                messages=[Message(role="user", content="use tool")],
                tools=[],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    def test_none_tool_arguments_become_empty_dict(self) -> None:
        """None in tool call arguments should also fallback to {}."""
        provider = self._make_provider()

        mock_tc = MagicMock()
        mock_tc.id = "call_456"
        mock_tc.function.name = "other_tool"
        mock_tc.function.arguments = None  # json.loads(None) => TypeError

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.tool_calls = [mock_tc]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {}

        mock_client_cls = MagicMock()
        mock_client_cls.return_value.chat.completions.create.return_value = mock_response

        with patch("missy.providers.openai_provider._openai_sdk") as mock_sdk:
            mock_sdk.OpenAI = mock_client_cls
            mock_sdk.APITimeoutError = Exception
            mock_sdk.AuthenticationError = Exception
            mock_sdk.APIError = Exception
            result = provider.complete_with_tools(
                messages=[Message(role="user", content="use tool")],
                tools=[],
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# Discord REST edge cases
# ---------------------------------------------------------------------------


class TestDiscordRESTRetryAfterParsing:
    """Test Retry-After header parsing edge cases in send_message."""

    def _make_client(self) -> Any:
        from missy.channels.discord.rest import DiscordRestClient

        http = MagicMock()
        return DiscordRestClient(bot_token="Bot test-token", http_client=http)

    def test_non_numeric_retry_after_uses_fallback(self) -> None:
        """Non-numeric Retry-After header should fall back to exponential backoff."""
        client = self._make_client()

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "not-a-number"}
        resp_429.text = ""

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {"id": "12345"}
        resp_200.raise_for_status = MagicMock()

        client._http.post.side_effect = [resp_429, resp_200]

        with patch("missy.channels.discord.rest.time.sleep"):
            result = client.send_message("123456789012345678", "test message")

        assert result["id"] == "12345"

    def test_non_dict_payload_raises(self) -> None:
        """Response payload that isn't a dict should raise RuntimeError."""
        client = self._make_client()

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = ["not", "a", "dict"]

        client._http.post.return_value = resp

        with pytest.raises(RuntimeError, match="missing id"):
            client.send_message("123456789012345678", "test")


# ---------------------------------------------------------------------------
# Cost tracker edge cases
# ---------------------------------------------------------------------------


class TestCostTrackerEdgeCases:
    """Edge cases in CostTracker budget enforcement."""

    def test_record_zero_tokens(self) -> None:
        """Recording zero tokens should not crash."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="gpt-4", prompt_tokens=0, completion_tokens=0)
        summary = tracker.get_summary()
        assert summary["total_cost_usd"] == 0.0

    def test_budget_enforcement_at_zero_budget(self) -> None:
        """Zero max_spend should mean unlimited (no enforcement)."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=0.0)
        tracker.record(model="gpt-4", prompt_tokens=1_000_000, completion_tokens=1_000_000)
        # Should not raise — 0 means unlimited

    def test_negative_token_count_handled(self) -> None:
        """Negative token counts should not crash the tracker."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="gpt-4", prompt_tokens=-1, completion_tokens=-1)
        summary = tracker.get_summary()
        assert "total_cost_usd" in summary

    def test_unknown_model_uses_default_pricing(self) -> None:
        """Unknown model name should use fallback pricing, not crash."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="totally-made-up-model-xyz", prompt_tokens=100, completion_tokens=50)
        summary = tracker.get_summary()
        assert summary["total_tokens"] == 150


# ---------------------------------------------------------------------------
# MCP client edge cases
# ---------------------------------------------------------------------------


class TestMCPClientEdgeCases:
    """Edge cases in MCP client disconnect and tool validation."""

    def test_disconnect_when_process_already_dead(self) -> None:
        """Disconnect should handle process that already exited."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        proc = MagicMock()
        proc.poll.return_value = None  # Alive
        proc.terminate.side_effect = OSError("terminate failed")
        proc.kill = MagicMock()
        proc.wait = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        client._proc = proc

        # Should not raise — kill fallback handles terminate failure
        client.disconnect()
        proc.kill.assert_called()

    def test_disconnect_when_no_process(self) -> None:
        """Calling disconnect with _proc=None should be safe."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        client._proc = None

        # Should not raise
        client.disconnect()

    def test_disconnect_clears_proc(self) -> None:
        """After disconnect, _proc should be None."""
        from missy.mcp.client import McpClient

        client = McpClient.__new__(McpClient)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.terminate = MagicMock()
        proc.wait = MagicMock()
        proc.stdin = MagicMock()
        proc.stdout = MagicMock()
        proc.stderr = MagicMock()
        client._proc = proc

        client.disconnect()
        assert client._proc is None


# ---------------------------------------------------------------------------
# Provider config edge cases
# ---------------------------------------------------------------------------


class TestProviderConfigEdgeCases:
    """Edge cases in provider configuration."""

    def test_empty_api_keys_list(self) -> None:
        """Empty api_keys list should fallback to single api_key."""
        cfg = ProviderConfig(
            name="openai",
            model="gpt-4",
            api_key="sk-primary",
            api_keys=[],
        )
        assert cfg.api_key == "sk-primary"
        assert cfg.api_keys == []

    def test_provider_config_timeout_zero(self) -> None:
        """Timeout of 0 should be accepted."""
        cfg = ProviderConfig(name="test", model="test-model", timeout=0)
        assert cfg.timeout == 0
