"""Tests for Discord credential-detection and message-deletion (gap #25).

Covers:
- DiscordRestClient.delete_message returns True on HTTP 204.
- DiscordRestClient.delete_message returns False on HTTP 403/404.
- DiscordRestClient.delete_message returns False and logs on unexpected exceptions.
- DiscordChannel._handle_message drops messages containing secrets, emits audit
  event, attempts deletion, and sends a warning reply.
- Non-secret messages are processed normally (not dropped).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy
from missy.channels.discord.rest import DiscordRestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account(dm_policy: DiscordDMPolicy = DiscordDMPolicy.OPEN) -> DiscordAccountConfig:
    return DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        dm_policy=dm_policy,
    )


def _make_channel(dm_policy: DiscordDMPolicy = DiscordDMPolicy.OPEN) -> DiscordChannel:
    ch = DiscordChannel(account_config=_make_account(dm_policy))
    # Pre-set bot_user_id so own-message filter knows who the bot is.
    ch._bot_user_id = "99999"
    return ch


def _message_payload(
    content: str,
    author_id: str = "11111",
    channel_id: str = "22222",
    message_id: str = "33333",
) -> dict[str, Any]:
    return {
        "id": message_id,
        "channel_id": channel_id,
        "content": content,
        "author": {"id": author_id, "bot": False},
    }


# ---------------------------------------------------------------------------
# DiscordRestClient.delete_message
# ---------------------------------------------------------------------------


class TestDeleteMessage:
    """Unit tests for DiscordRestClient.delete_message."""

    def _make_rest(self) -> DiscordRestClient:
        mock_http = MagicMock()
        return DiscordRestClient(bot_token="Bot TOKEN", http_client=mock_http)

    def test_returns_true_on_204(self) -> None:
        rest = self._make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("httpx.delete", return_value=mock_response):
            result = rest.delete_message("chan123", "msg456")

        assert result is True

    def test_returns_false_on_403(self) -> None:
        rest = self._make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch("httpx.delete", return_value=mock_response):
            result = rest.delete_message("chan123", "msg456")

        assert result is False

    def test_returns_false_on_404(self) -> None:
        rest = self._make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.delete", return_value=mock_response):
            result = rest.delete_message("chan123", "msg456")

        assert result is False

    def test_returns_false_on_exception(self) -> None:
        rest = self._make_rest()

        with patch("httpx.delete", side_effect=OSError("network error")):
            result = rest.delete_message("chan123", "msg456")

        assert result is False

    def test_raises_for_status_on_unexpected_code(self) -> None:
        """A 5xx triggers raise_for_status, which is then caught and returns False."""
        rest = self._make_rest()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("server error")

        with patch("httpx.delete", return_value=mock_response):
            result = rest.delete_message("chan123", "msg456")

        assert result is False


# ---------------------------------------------------------------------------
# DiscordChannel._handle_message credential detection
# ---------------------------------------------------------------------------


class TestHandleMessageCredentialDetection:
    """Integration-level tests for credential detection in _handle_message."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_secret_message_is_dropped(self) -> None:
        """A message containing an AWS key should not be enqueued."""
        ch = _make_channel()

        with (
            patch.object(ch._rest, "delete_message", return_value=True),
            patch.object(ch._rest, "send_message", return_value={}),
            patch.object(ch, "_emit_audit") as mock_audit,
        ):
            self._run(
                ch._handle_message(_message_payload("My AWS key is AKIAIOSFODNN7EXAMPLE123456"))
            )

        assert ch._queue.empty(), "Secret message must not be enqueued"
        # Audit event for credential detection must have been emitted.
        event_types = [call.args[0] for call in mock_audit.call_args_list]
        assert "discord.channel.credential_detected" in event_types

    def test_secret_message_triggers_delete(self) -> None:
        """delete_message is called with the correct channel/message IDs."""
        ch = _make_channel()

        with (
            patch.object(ch._rest, "delete_message", return_value=True) as mock_del,
            patch.object(ch._rest, "send_message", return_value={}),
            patch.object(ch, "_emit_audit"),
        ):
            self._run(
                ch._handle_message(
                    _message_payload(
                        "token: ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890",
                        channel_id="CH1",
                        message_id="MSG1",
                    )
                )
            )

        mock_del.assert_called_once_with("CH1", "MSG1")

    def test_secret_message_sends_warning_reply(self) -> None:
        """A warning message is sent back to the channel after detection."""
        ch = _make_channel()

        with (
            patch.object(ch._rest, "delete_message", return_value=True),
            patch.object(ch._rest, "send_message", return_value={}) as mock_send,
            patch.object(ch, "_emit_audit"),
        ):
            self._run(
                ch._handle_message(_message_payload("AKIA1234567890ABCDEF", channel_id="CH2"))
            )

        # send_message should have been called with a warning.
        assert mock_send.call_count >= 1
        warning_call_args = mock_send.call_args_list[-1]
        sent_content: str = warning_call_args.args[1] if warning_call_args.args else ""
        assert "credentials" in sent_content.lower() or "secrets" in sent_content.lower()

    def test_audit_event_contains_message_deleted_true(self) -> None:
        """Audit detail contains message_deleted=True when deletion succeeds."""
        ch = _make_channel()
        captured: list[dict] = []

        def _capture_audit(event_type, result, detail):
            captured.append({"event_type": event_type, "result": result, "detail": detail})

        with (
            patch.object(ch._rest, "delete_message", return_value=True),
            patch.object(ch._rest, "send_message", return_value={}),
            patch.object(ch, "_emit_audit", side_effect=_capture_audit),
        ):
            self._run(ch._handle_message(_message_payload("sk_live_abcdefghijklmnopqrstuvwx")))

        cred_events = [
            e for e in captured if e["event_type"] == "discord.channel.credential_detected"
        ]
        assert cred_events, "credential_detected audit event must be emitted"
        assert cred_events[0]["detail"]["message_deleted"] is True

    def test_audit_event_message_deleted_false_on_delete_failure(self) -> None:
        """Audit detail contains message_deleted=False when deletion fails."""
        ch = _make_channel()
        captured: list[dict] = []

        def _capture_audit(event_type, result, detail):
            captured.append({"event_type": event_type, "result": result, "detail": detail})

        with (
            patch.object(ch._rest, "delete_message", return_value=False),
            patch.object(ch._rest, "send_message", return_value={}),
            patch.object(ch, "_emit_audit", side_effect=_capture_audit),
        ):
            self._run(ch._handle_message(_message_payload("AKIAIOSFODNN7EXAMPLE")))

        cred_events = [
            e for e in captured if e["event_type"] == "discord.channel.credential_detected"
        ]
        assert cred_events
        assert cred_events[0]["detail"]["message_deleted"] is False

    def test_non_secret_message_is_enqueued_normally(self) -> None:
        """A message without secrets passes through and is enqueued."""
        ch = _make_channel()

        self._run(ch._handle_message(_message_payload("Hello, how are you?")))

        assert not ch._queue.empty()

    def test_secrets_detection_error_does_not_drop_message(self) -> None:
        """If SecretsDetector raises unexpectedly, the message is not silently dropped."""
        ch = _make_channel()

        with patch(
            "missy.security.secrets.SecretsDetector.has_secrets",
            side_effect=RuntimeError("unexpected"),
        ):
            self._run(ch._handle_message(_message_payload("normal message")))

        # Message should still be enqueued (detection error logged, not re-raised).
        assert not ch._queue.empty()

    def test_own_bot_message_is_filtered_before_credential_check(self) -> None:
        """Own-bot messages are dropped before credential scanning."""
        ch = _make_channel()

        with patch.object(ch._rest, "delete_message") as mock_del:
            # Send a message from the bot itself with an AWS key.
            self._run(
                ch._handle_message(
                    _message_payload(
                        "AKIAIOSFODNN7EXAMPLE",
                        author_id=ch._bot_user_id,
                    )
                )
            )

        # delete_message must NOT be called for own messages.
        mock_del.assert_not_called()
        assert ch._queue.empty()
