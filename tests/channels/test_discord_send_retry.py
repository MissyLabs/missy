"""Tests for Discord message send retry with exponential backoff.

Covers:
- send_to raises DiscordSendError on failure
- send_with_retry retries on failure with backoff
- send_with_retry succeeds after transient failures
- send_with_retry gives up after max_attempts
- send_with_retry respects max_total_seconds
- DiscordSendError carries channel_id and original_error
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from missy.channels.discord.channel import DiscordChannel, DiscordSendError
from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_rest() -> MagicMock:
    rest = MagicMock()
    rest.trigger_typing.return_value = None
    rest.send_message.return_value = {"id": "msg-001"}
    return rest


@pytest.fixture()
def channel(mock_rest: MagicMock) -> DiscordChannel:
    acct = DiscordAccountConfig(
        token_env_var="DISCORD_BOT_TOKEN",
        account_id="bot-001",
        dm_policy=DiscordDMPolicy.OPEN,
    )
    ch = DiscordChannel(account_config=acct)
    ch._rest = mock_rest
    ch._bot_user_id = "bot-001"
    return ch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# DiscordSendError
# ---------------------------------------------------------------------------


class TestDiscordSendError:
    def test_carries_channel_id(self):
        err = DiscordSendError("boom", channel_id="ch-42")
        assert err.channel_id == "ch-42"
        assert str(err) == "boom"

    def test_carries_original_error(self):
        orig = RuntimeError("inner")
        err = DiscordSendError("outer", original_error=orig)
        assert err.original_error is orig

    def test_defaults(self):
        err = DiscordSendError("msg")
        assert err.channel_id == ""
        assert err.original_error is None


# ---------------------------------------------------------------------------
# send_to raises on failure
# ---------------------------------------------------------------------------


class TestSendToRaises:
    def test_raises_discord_send_error(self, channel: DiscordChannel, mock_rest: MagicMock):
        mock_rest.send_message.side_effect = Exception("network down")
        with pytest.raises(DiscordSendError) as exc_info:
            _run(channel.send_to("ch-1", "hello"))
        assert "ch-1" in str(exc_info.value)
        assert exc_info.value.channel_id == "ch-1"
        assert exc_info.value.original_error is not None

    def test_raises_on_missing_message_id(self, channel: DiscordChannel, mock_rest: MagicMock):
        mock_rest.send_message.return_value = {}
        with pytest.raises(DiscordSendError):
            _run(channel.send_to("ch-1", "hello"))

    def test_success_returns_message_id(self, channel: DiscordChannel, mock_rest: MagicMock):
        result = _run(channel.send_to("ch-1", "hello"))
        assert result == "msg-001"


# ---------------------------------------------------------------------------
# send_with_retry
# ---------------------------------------------------------------------------


class TestSendWithRetry:
    def test_succeeds_first_try(self, channel: DiscordChannel, mock_rest: MagicMock):
        result = _run(channel.send_with_retry("ch-1", "hello", max_attempts=3))
        assert result == "msg-001"
        assert mock_rest.send_message.call_count == 1

    def test_succeeds_after_transient_failure(self, channel: DiscordChannel, mock_rest: MagicMock):
        call_count = 0

        def _fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("transient")
            return {"id": "msg-recovered"}

        mock_rest.send_message.side_effect = _fail_then_succeed
        result = _run(
            channel.send_with_retry(
                "ch-1",
                "hello",
                max_attempts=5,
                max_total_seconds=30.0,
            )
        )
        assert result == "msg-recovered"
        assert call_count == 3

    def test_gives_up_after_max_attempts(self, channel: DiscordChannel, mock_rest: MagicMock):
        mock_rest.send_message.side_effect = Exception("permanent")
        with pytest.raises(DiscordSendError) as exc_info:
            _run(
                channel.send_with_retry(
                    "ch-1",
                    "hello",
                    max_attempts=3,
                    max_total_seconds=60.0,
                )
            )
        assert "3 attempts" in str(exc_info.value)
        assert exc_info.value.channel_id == "ch-1"
        # Should have tried exactly 3 times.
        assert mock_rest.send_message.call_count == 3

    def test_respects_max_total_seconds(self, channel: DiscordChannel, mock_rest: MagicMock):
        mock_rest.send_message.side_effect = Exception("fail")
        start = time.monotonic()
        with pytest.raises(DiscordSendError):
            _run(
                channel.send_with_retry(
                    "ch-1",
                    "hello",
                    max_attempts=20,
                    max_total_seconds=3.0,
                )
            )
        elapsed = time.monotonic() - start
        # Should give up around the 3s mark, not run all 20 attempts.
        assert elapsed < 10.0

    def test_emits_retry_audit_events(self, channel: DiscordChannel, mock_rest: MagicMock):
        audit_events = []
        original_emit = channel._emit_audit

        def _capture_audit(event_type, outcome, details):
            if "retry" in event_type or "failed" in event_type:
                audit_events.append((event_type, outcome, details))
            original_emit(event_type, outcome, details)

        channel._emit_audit = _capture_audit
        mock_rest.send_message.side_effect = Exception("fail")

        with pytest.raises(DiscordSendError):
            _run(
                channel.send_with_retry(
                    "ch-1",
                    "hello",
                    max_attempts=3,
                    max_total_seconds=30.0,
                )
            )

        retry_events = [e for e in audit_events if e[0] == "discord.channel.send_retry"]
        failed_events = [e for e in audit_events if e[0] == "discord.channel.send_failed"]
        # 2 retries (attempt 1 and 2 fail, attempt 3 is the final failure).
        assert len(retry_events) == 2
        assert len(failed_events) == 1
        assert failed_events[0][1] == "error"
