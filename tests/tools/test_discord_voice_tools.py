"""Tests for the discord_voice_* built-in tools."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from missy.channels.discord import voice_binding
from missy.tools.builtin.discord_voice import (
    DiscordVoiceJoinTool,
    DiscordVoiceLeaveTool,
    DiscordVoiceSayTool,
    DiscordVoiceStatusTool,
)

# ---------------------------------------------------------------------------
# Background asyncio loop fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def loop_thread():
    """Run a dedicated asyncio loop in a background thread.

    The tools call ``asyncio.run_coroutine_threadsafe(..., loop)`` from the
    executor thread, so the loop must already be running.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _runner() -> None:
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    ready.wait()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2.0)
        loop.close()


@pytest.fixture
def fake_manager():
    """Async-aware DiscordVoiceManager double."""
    mgr = MagicMock()
    mgr.is_ready = True
    mgr.can_listen = True
    mgr.can_speak = True

    async def _join(guild_id, **kwargs):
        mgr.join_calls.append((guild_id, kwargs))
        return "General"

    async def _leave(guild_id):
        mgr.leave_calls.append(guild_id)
        return "General"

    async def _say(guild_id, text):
        mgr.say_calls.append((guild_id, text))

    mgr.join_calls = []
    mgr.leave_calls = []
    mgr.say_calls = []
    mgr.join = _join
    mgr.leave = _leave
    mgr.say = _say
    mgr.is_connected = MagicMock(return_value=True)
    mgr.is_listening = MagicMock(return_value=True)
    mgr.current_channel_name = MagicMock(return_value="General")
    mgr.list_voice_channels = MagicMock(return_value=["General", "Music"])
    return mgr


@pytest.fixture
def bound(fake_manager, loop_thread):
    voice_binding.set_voice_binding(fake_manager, loop_thread)
    try:
        yield fake_manager
    finally:
        voice_binding.clear_voice_binding()


# ---------------------------------------------------------------------------
# Binding errors
# ---------------------------------------------------------------------------


def test_voice_tools_declare_discord_network_permissions():
    for tool in (
        DiscordVoiceJoinTool(),
        DiscordVoiceLeaveTool(),
        DiscordVoiceSayTool(),
        DiscordVoiceStatusTool(),
    ):
        assert tool.permissions.network is True
        assert "discord.com" in tool.permissions.allowed_hosts
        assert "gateway.discord.gg" in tool.permissions.allowed_hosts


def test_join_no_binding_returns_error():
    voice_binding.clear_voice_binding()
    result = DiscordVoiceJoinTool().execute(guild_id="123", user_id="456")
    assert result.success is False
    assert "not active" in result.error


def test_join_manager_not_ready(loop_thread, fake_manager):
    fake_manager.is_ready = False
    voice_binding.set_voice_binding(fake_manager, loop_thread)
    try:
        result = DiscordVoiceJoinTool().execute(guild_id="123", user_id="456")
    finally:
        voice_binding.clear_voice_binding()
    assert result.success is False
    assert "starting up" in result.error


# ---------------------------------------------------------------------------
# discord_voice_join
# ---------------------------------------------------------------------------


def test_join_requires_guild_id(bound):
    result = DiscordVoiceJoinTool().execute(guild_id="", channel="General")
    assert result.success is False
    assert "guild_id" in result.error


def test_join_by_channel_name(bound):
    result = DiscordVoiceJoinTool().execute(guild_id="42", channel="General")
    assert result.success is True
    assert "General" in result.output
    assert bound.join_calls == [(42, {"channel_name": "General"})]


def test_join_by_channel_id_string(bound):
    result = DiscordVoiceJoinTool().execute(guild_id="42", channel="98765")
    assert result.success is True
    assert bound.join_calls == [(42, {"channel_id": 98765})]


def test_join_by_user_id_follows_user(bound):
    result = DiscordVoiceJoinTool().execute(guild_id="42", user_id="11111")
    assert result.success is True
    assert bound.join_calls == [(42, {"user_id": 11111})]


def test_join_without_channel_or_user_errors(bound):
    result = DiscordVoiceJoinTool().execute(guild_id="42")
    assert result.success is False
    assert "channel" in result.error.lower()


def test_join_capabilities_reported(bound):
    bound.can_speak = False
    result = DiscordVoiceJoinTool().execute(guild_id="42", channel="General")
    assert "listening" in result.output
    assert "speaking" not in result.output


def test_join_propagates_manager_errors(bound):
    async def _bad(*_a, **_kw):
        raise RuntimeError("boom")

    bound.join = _bad
    result = DiscordVoiceJoinTool().execute(guild_id="42", channel="General")
    assert result.success is False
    assert "boom" in result.error


# ---------------------------------------------------------------------------
# discord_voice_leave
# ---------------------------------------------------------------------------


def test_leave_requires_guild_id(bound):
    result = DiscordVoiceLeaveTool().execute(guild_id="")
    assert result.success is False
    assert "guild_id" in result.error


def test_leave_returns_channel(bound):
    result = DiscordVoiceLeaveTool().execute(guild_id="42")
    assert result.success is True
    assert "General" in result.output
    assert bound.leave_calls == [42]


def test_leave_when_not_connected(bound):
    async def _noop(_gid):
        return None

    bound.leave = _noop
    result = DiscordVoiceLeaveTool().execute(guild_id="42")
    assert result.success is True
    assert "not in" in result.output.lower()


# ---------------------------------------------------------------------------
# discord_voice_say
# ---------------------------------------------------------------------------


def test_say_requires_guild_and_text(bound):
    no_gid = DiscordVoiceSayTool().execute(guild_id="", text="hi")
    assert no_gid.success is False

    no_text = DiscordVoiceSayTool().execute(guild_id="42", text="   ")
    assert no_text.success is False


def test_say_dispatches(bound):
    result = DiscordVoiceSayTool().execute(guild_id="42", text="Hello world")
    assert result.success is True
    assert bound.say_calls == [(42, "Hello world")]


# ---------------------------------------------------------------------------
# discord_voice_status
# ---------------------------------------------------------------------------


def test_status_returns_struct(bound):
    result = DiscordVoiceStatusTool().execute(guild_id="42")
    assert result.success is True
    out = result.output
    assert out["connected"] is True
    assert out["current_channel"] == "General"
    assert out["listening"] is True
    assert out["available_channels"] == ["General", "Music"]
    assert out["can_listen"] is True
    assert out["can_speak"] is True


def test_status_requires_guild_id(bound):
    result = DiscordVoiceStatusTool().execute(guild_id="")
    assert result.success is False
