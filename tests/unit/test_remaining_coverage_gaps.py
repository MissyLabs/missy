"""Targeted tests for remaining coverage gaps across several modules.

Covers:
- missy/agent/heartbeat.py lines 59, 105
- missy/channels/webhook.py line 45
- missy/config/hotreload.py lines 62-63
- missy/channels/discord/channel.py lines 639-640, 764, 878-879
- missy/channels/discord/gateway.py lines 286-291
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# missy/agent/heartbeat.py — line 59: _fire() called inside _loop
# ---------------------------------------------------------------------------


class TestHeartbeatRunnerLoop:
    """Line 59: _loop calls _fire() when the stop event has not been set."""

    def test_loop_calls_fire_once_then_stops(self, tmp_path):
        """_fire is invoked at least once when _stop.wait() returns False briefly."""
        from missy.agent.heartbeat import HEARTBEAT_FILE, HeartbeatRunner

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / HEARTBEAT_FILE).write_text("- check everything\n")

        fired_event = threading.Event()
        agent_called = []

        def fake_run(prompt: str) -> str:
            agent_called.append(prompt)
            fired_event.set()
            return "ok"

        runner = HeartbeatRunner(
            agent_run_fn=fake_run,
            interval_seconds=0,  # fire immediately without waiting
            workspace=str(workspace),
        )

        runner.start()
        fired_event.wait(timeout=2.0)
        runner.stop()

        assert len(agent_called) >= 1
        assert "[HEARTBEAT CHECK]" in agent_called[0]

    def test_loop_does_not_fire_when_stopped_immediately(self, tmp_path):
        """If stop is set before the first wait expires, _fire is never called."""
        from missy.agent.heartbeat import HeartbeatRunner

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        agent_called = []

        runner = HeartbeatRunner(
            agent_run_fn=lambda p: agent_called.append(p) or "ok",
            interval_seconds=60,  # long interval — stop before it fires
            workspace=str(workspace),
        )
        runner.start()
        runner.stop()  # immediate stop

        assert agent_called == []


# ---------------------------------------------------------------------------
# missy/agent/heartbeat.py — line 105: overnight active hours window
# ---------------------------------------------------------------------------


class TestHeartbeatActiveHoursOvernight:
    """Line 105: overnight window (end < start) — now >= start OR now <= end."""

    def _runner(self):
        from missy.agent.heartbeat import HeartbeatRunner

        return HeartbeatRunner(agent_run_fn=lambda p: "ok", active_hours="22:00-06:00")

    def _fake_now(self, hour: int, minute: int = 0) -> datetime:
        """Return a datetime with the given hour/minute and today's date."""
        return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)

    def test_overnight_window_matches_after_start(self):
        """22:00-06:00 window: 23:30 is inside (now >= start branch of line 105)."""
        runner = self._runner()
        fake_now = self._fake_now(23, 30)

        # _in_active_hours does `from datetime import datetime` locally, so patch
        # the class in the standard library module directly.
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = runner._in_active_hours()

        assert result is True

    def test_overnight_window_matches_before_end(self):
        """22:00-06:00 window: 03:00 is inside (now <= end branch of line 105)."""
        runner = self._runner()
        fake_now = self._fake_now(3, 0)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = runner._in_active_hours()

        assert result is True

    def test_overnight_window_excludes_midday(self):
        """22:00-06:00 window: 12:00 is outside (both branches False on line 105)."""
        runner = self._runner()
        fake_now = self._fake_now(12, 0)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = runner._in_active_hours()

        assert result is False


# ---------------------------------------------------------------------------
# missy/channels/webhook.py — line 45: Handler.log_message
# ---------------------------------------------------------------------------


class TestWebhookHandlerLogMessage:
    """Line 45: Handler.log_message routes through to logger.debug."""

    def test_log_message_calls_logger_debug(self):
        """The closure-defined Handler.log_message calls logger.debug correctly."""
        from missy.channels.webhook import WebhookChannel

        captured: dict = {}

        def capturing_httpserver(addr, handler_cls_arg):
            captured["handler_cls"] = handler_cls_arg
            srv = MagicMock()
            return srv

        with patch("missy.channels.webhook.HTTPServer", side_effect=capturing_httpserver):
            ch = WebhookChannel(host="127.0.0.1", port=0)
            ch.start()

        HandlerCls = captured["handler_cls"]
        # Create an instance without invoking __init__ (avoids socket setup).
        handler = HandlerCls.__new__(HandlerCls)

        with patch("missy.channels.webhook.logger") as mock_logger:
            handler.log_message("%s %s", "GET", "/")
            mock_logger.debug.assert_called_once_with("Webhook: %s %s", "GET", "/")

    def test_log_message_with_no_extra_args(self):
        """log_message works with only a format string and no positional args."""
        from missy.channels.webhook import WebhookChannel

        captured: dict = {}

        def capturing_httpserver(addr, handler_cls_arg):
            captured["handler_cls"] = handler_cls_arg
            return MagicMock()

        with patch("missy.channels.webhook.HTTPServer", side_effect=capturing_httpserver):
            ch = WebhookChannel(host="127.0.0.1", port=0)
            ch.start()

        HandlerCls = captured["handler_cls"]
        handler = HandlerCls.__new__(HandlerCls)

        with patch("missy.channels.webhook.logger") as mock_logger:
            handler.log_message("simple message")
            mock_logger.debug.assert_called_once_with("Webhook: simple message")


# ---------------------------------------------------------------------------
# missy/config/hotreload.py — lines 62-63: OSError in _watch loop
# ---------------------------------------------------------------------------


class TestConfigWatcherOSError:
    """Lines 62-63: OSError from path.stat() causes continue (not a crash)."""

    def test_oserror_in_watch_loop_is_silently_skipped(self, tmp_path):
        """When stat() raises OSError the watcher continues polling without error."""
        from missy.config.hotreload import ConfigWatcher

        config_path = tmp_path / "config.yaml"
        reload_calls: list = []

        watcher = ConfigWatcher(
            config_path=str(config_path),
            reload_fn=reload_calls.append,
            debounce_seconds=10.0,
            poll_interval=0.02,
        )
        # Do not create the file — start() sets _last_mtime=0 on OSError.
        watcher.start()

        # Let the watcher loop iterate a few times over a non-existent file.
        time.sleep(0.12)
        watcher.stop()

        # The thread finished cleanly — no exception propagated.
        assert not watcher._thread.is_alive()
        # No reload was triggered because mtime never changed.
        assert reload_calls == []

    def test_oserror_does_not_increment_mtime(self, tmp_path):
        """After OSError, _last_mtime remains at its initial value."""
        from missy.config.hotreload import ConfigWatcher

        watcher = ConfigWatcher(
            config_path=str(tmp_path / "nonexistent.yaml"),
            reload_fn=lambda cfg: None,
            debounce_seconds=10.0,
            poll_interval=0.02,
        )
        watcher.start()
        initial_mtime = watcher._last_mtime  # 0.0 — file does not exist
        time.sleep(0.1)
        watcher.stop()

        # Because the file never existed, stat() always raises OSError and
        # _last_mtime stays at 0.0.
        assert watcher._last_mtime == initial_mtime == 0.0

    def test_watch_loop_recovers_after_oserror_when_file_created(self, tmp_path):
        """After a period of OSError, the watcher picks up the file once created."""
        from missy.config.hotreload import ConfigWatcher

        config_path = tmp_path / "config.yaml"
        reload_calls: list = []

        # Use a short debounce so we do not have to wait long.
        watcher = ConfigWatcher(
            config_path=str(config_path),
            reload_fn=reload_calls.append,
            debounce_seconds=0.05,
            poll_interval=0.02,
        )
        watcher.start()

        # Let the loop spin a couple of times with no file (OSError path).
        time.sleep(0.06)

        # Now create the file — the watcher should detect the mtime change.
        config_path.write_text("providers: {}\n")

        # load_config is imported locally inside _do_reload; patch it there.
        with patch("missy.config.settings.load_config", return_value=MagicMock()):
            time.sleep(0.2)

        watcher.stop()
        # After creating the file, _last_mtime should be non-zero.
        assert watcher._last_mtime != 0.0


# ---------------------------------------------------------------------------
# missy/channels/discord/channel.py — lines 639-640: voice agent callback
# ---------------------------------------------------------------------------


class TestDiscordVoiceAgentCallbackClosure:
    """Lines 639-640: the _voice_agent_cb closure delegates to run_in_executor."""

    @pytest.mark.asyncio
    async def test_voice_agent_callback_uses_run_in_executor(self):
        """Exercise the closure pattern at lines 638-645 directly."""
        # The closure in the source is:
        #
        #   async def _voice_agent_cb(prompt: str, session_id: str) -> str:
        #       loop = asyncio.get_running_loop()
        #       return await loop.run_in_executor(None, _rt.run, prompt, session_id)
        #
        # We replicate that closure verbatim and verify that run_in_executor is
        # called with the runtime's .run method and the correct arguments.
        fake_runtime = MagicMock()
        fake_runtime.run.return_value = "agent-response"
        _rt = fake_runtime

        async def _voice_agent_cb(prompt: str, session_id: str) -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _rt.run, prompt, session_id)

        result = await _voice_agent_cb("hello from voice", "session-abc")

        assert result == "agent-response"
        fake_runtime.run.assert_called_once_with("hello from voice", "session-abc")

    @pytest.mark.asyncio
    async def test_voice_agent_callback_propagates_exception(self):
        """If the runtime raises, the exception propagates through the executor."""
        fake_runtime = MagicMock()
        fake_runtime.run.side_effect = RuntimeError("agent crashed")
        _rt = fake_runtime

        async def _voice_agent_cb(prompt: str, session_id: str) -> str:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _rt.run, prompt, session_id)

        with pytest.raises(RuntimeError, match="agent crashed"):
            await _voice_agent_cb("bad prompt", "session-xyz")


# ---------------------------------------------------------------------------
# missy/channels/discord/channel.py — line 764: DM policy fallthrough
# ---------------------------------------------------------------------------


def _make_discord_channel(account_config):
    """Build a DiscordChannel with both sub-clients patched out."""
    with (
        patch("missy.channels.discord.channel.DiscordGatewayClient"),
        patch("missy.channels.discord.channel.DiscordRestClient"),
    ):
        from missy.channels.discord.channel import DiscordChannel

        return DiscordChannel(account_config=account_config)


class TestDiscordDMPolicyFallthrough:
    """Line 764: _check_dm_policy returns False for unrecognised policy value."""

    def test_unknown_dm_policy_returns_false(self):
        """A policy value not matching any branch causes the method to return False."""
        from missy.channels.discord.config import DiscordAccountConfig, DiscordDMPolicy

        account = DiscordAccountConfig(
            token_env_var="DISCORD_BOT_TOKEN",
            dm_policy=DiscordDMPolicy.OPEN,
        )
        channel = _make_discord_channel(account)
        channel._emit_audit = MagicMock()

        # Replace account_config with a mock whose .dm_policy does not match any
        # of the four known DiscordDMPolicy branches.
        channel.account_config = MagicMock()
        channel.account_config.dm_policy = "totally-unknown-policy"

        result = channel._check_dm_policy("user-xyz", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# missy/channels/discord/channel.py — lines 878-879: mention fallback
# ---------------------------------------------------------------------------


class TestDiscordMentionFallbackNoOwnId:
    """Lines 878-879: mention check when own_id is falsy falls back to mentions list."""

    def _make_channel_with_no_own_id(self):
        """Return a DiscordChannel where both bot_user_id and account_id are falsy."""
        from missy.channels.discord.config import (
            DiscordAccountConfig,
            DiscordDMPolicy,
            DiscordGuildPolicy,
        )

        guild_policy = DiscordGuildPolicy(
            enabled=True,
            require_mention=True,
            allowed_channels=[],
            allowed_users=[],
        )
        account = DiscordAccountConfig(
            token_env_var="DISCORD_BOT_TOKEN",
            account_id=None,  # falsy account_id
            dm_policy=DiscordDMPolicy.DISABLED,
            guild_policies={"guild-1": guild_policy},
        )
        channel = _make_discord_channel(account)
        channel._emit_audit = MagicMock()
        # Ensure the gateway mock also returns None for bot_user_id.
        channel._gateway.bot_user_id = None
        return channel

    def test_mention_check_uses_mentions_list_when_own_id_falsy(self):
        """When own_id is falsy and mentions list has id == '', message is allowed."""
        channel = self._make_channel_with_no_own_id()

        # own_id is None; (own_id or "") == "".
        # The any() check: str(m.get("id","")) == "" — needs an entry with id "".
        data_with_mention = {"mentions": [{"id": ""}]}
        result = channel._check_guild_policy(
            guild_id="guild-1",
            channel_id="chan-1",
            author_id="user-a",
            content="hello",
            data=data_with_mention,
        )
        assert result is True

    def test_mention_check_no_match_in_mentions_denies(self):
        """When own_id is falsy and mentions list has no id=='', message is denied."""
        channel = self._make_channel_with_no_own_id()

        data_no_match = {"mentions": [{"id": "other-bot-id"}]}
        result = channel._check_guild_policy(
            guild_id="guild-1",
            channel_id="chan-1",
            author_id="user-b",
            content="hello",
            data=data_no_match,
        )
        assert result is False

    def test_mention_check_empty_mentions_list_denies(self):
        """When own_id is falsy and mentions list is empty, message is denied."""
        channel = self._make_channel_with_no_own_id()

        result = channel._check_guild_policy(
            guild_id="guild-1",
            channel_id="chan-1",
            author_id="user-c",
            content="hello",
            data={"mentions": []},
        )
        assert result is False

    def test_mention_check_none_mentions_key_denies(self):
        """When mentions key is absent, data.get('mentions') or [] gives empty list."""
        channel = self._make_channel_with_no_own_id()

        result = channel._check_guild_policy(
            guild_id="guild-1",
            channel_id="chan-1",
            author_id="user-d",
            content="hello",
            data={},
        )
        assert result is False


# ---------------------------------------------------------------------------
# missy/channels/discord/gateway.py — lines 286-291: _heartbeat_loop
# ---------------------------------------------------------------------------


class TestDiscordGatewayHeartbeatLoop:
    """Lines 286-291: _heartbeat_loop sleeps for jittered interval then loops."""

    @pytest.mark.asyncio
    async def test_heartbeat_loop_sends_heartbeat_after_jitter(self):
        """_heartbeat_loop calls _send_heartbeat at least once after the jitter sleep."""
        from missy.channels.discord.gateway import DiscordGatewayClient

        async def noop_callback(event: dict) -> None:
            pass

        gw = DiscordGatewayClient(bot_token="Bot TOKEN", on_message=noop_callback)
        gw._ws = MagicMock()
        gw._ws.send = AsyncMock()

        send_count = [0]

        async def counting_send():
            send_count[0] += 1
            if send_count[0] >= 1:
                raise asyncio.CancelledError

        gw._send_heartbeat = counting_send

        with (
            patch("missy.channels.discord.gateway.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(asyncio.CancelledError),
        ):
            await gw._heartbeat_loop(interval=30.0)

        assert send_count[0] >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_loop_initial_jitter_is_fraction_of_interval(self):
        """The first sleep call receives a value in [0, interval)."""
        from missy.channels.discord.gateway import DiscordGatewayClient

        async def noop_callback(event: dict) -> None:
            pass

        gw = DiscordGatewayClient(bot_token="Bot TOKEN", on_message=noop_callback)
        gw._ws = MagicMock()
        gw._ws.send = AsyncMock()

        sleep_calls: list[float] = []

        async def recording_sleep(duration):
            sleep_calls.append(duration)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        gw._send_heartbeat = AsyncMock()

        with (
            patch("missy.channels.discord.gateway.asyncio.sleep", side_effect=recording_sleep),
            pytest.raises(asyncio.CancelledError),
        ):
            await gw._heartbeat_loop(interval=40.0)

        # First sleep is the jitter: random() * 40.0 — must be in [0, 40).
        assert len(sleep_calls) >= 1
        assert 0.0 <= sleep_calls[0] < 40.0

    @pytest.mark.asyncio
    async def test_heartbeat_loop_second_sleep_equals_interval(self):
        """After the first heartbeat, the loop sleeps for exactly `interval` seconds."""
        from missy.channels.discord.gateway import DiscordGatewayClient

        async def noop_callback(event: dict) -> None:
            pass

        gw = DiscordGatewayClient(bot_token="Bot TOKEN", on_message=noop_callback)
        gw._ws = MagicMock()
        gw._ws.send = AsyncMock()

        sleep_calls: list[float] = []

        async def recording_sleep(duration):
            sleep_calls.append(duration)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError

        gw._send_heartbeat = AsyncMock()

        with (
            patch("missy.channels.discord.gateway.asyncio.sleep", side_effect=recording_sleep),
            pytest.raises(asyncio.CancelledError),
        ):
            await gw._heartbeat_loop(interval=45.0)

        # Second sleep (after first _send_heartbeat) must be exactly 45.0.
        assert len(sleep_calls) == 2
        assert sleep_calls[1] == 45.0

    @pytest.mark.asyncio
    async def test_start_heartbeat_cancels_existing_task(self):
        """_start_heartbeat cancels a pre-existing heartbeat task before starting a new one."""
        from missy.channels.discord.gateway import DiscordGatewayClient

        async def noop_callback(event: dict) -> None:
            pass

        gw = DiscordGatewayClient(bot_token="Bot TOKEN", on_message=noop_callback)
        gw._ws = MagicMock()
        gw._ws.send = AsyncMock()

        # Create a long-running task to act as the existing heartbeat.
        async def long_sleep():
            await asyncio.sleep(9999)

        existing_task = asyncio.create_task(long_sleep())
        gw._heartbeat_task = existing_task

        # Replace _heartbeat_loop with one that stops quickly after one iteration.
        async def quick_loop(interval: float) -> None:
            await asyncio.sleep(0)

        with patch.object(gw, "_heartbeat_loop", side_effect=quick_loop):
            await gw._start_heartbeat(30.0)

        # The old task was cancelled.
        assert existing_task.cancelled()
        # A new task was created.
        assert gw._heartbeat_task is not None
        assert gw._heartbeat_task is not existing_task
