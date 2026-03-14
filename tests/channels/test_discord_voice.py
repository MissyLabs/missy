"""Tests for Discord voice manager and voice commands.

Covers:
- DiscordVoiceManager channel resolution (by user, by name, by ID)
- join / leave / say lifecycle
- Natural command parsing (!join, !join <name>, !leave, !say)
- Error paths (not connected, channel not found, etc.)
"""

from __future__ import annotations

import asyncio
import struct
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord.voice import DiscordVoiceError, DiscordVoiceManager, _make_sink_class
from missy.channels.discord.voice_commands import (
    maybe_handle_voice_command,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_voice_channel(*, channel_id: int = 100, name: str = "General"):
    ch = MagicMock()
    ch.id = channel_id
    ch.name = name
    ch.connect = AsyncMock(return_value=MagicMock(
        is_connected=MagicMock(return_value=True),
        channel=ch,
        disconnect=AsyncMock(),
        move_to=AsyncMock(),
        is_playing=MagicMock(return_value=False),
        play=MagicMock(),
        stop=MagicMock(),
    ))
    return ch


def _make_guild(
    guild_id: int = 999,
    voice_channels: Optional[list] = None,
    members: Optional[dict] = None,
):
    guild = MagicMock()
    guild.id = guild_id
    channels = voice_channels or []
    guild.voice_channels = channels

    channel_map = {ch.id: ch for ch in channels}

    def get_channel(cid):
        return channel_map.get(cid)

    guild.get_channel = get_channel

    member_map = members or {}

    def get_member(uid):
        return member_map.get(uid)

    guild.get_member = get_member
    return guild


@pytest.fixture()
def manager():
    """Create a DiscordVoiceManager with mocked internals."""
    with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
        mgr = DiscordVoiceManager()
    mgr._ready.set()

    # Mock discord module.
    mock_discord = MagicMock()
    mock_discord.VoiceChannel = type("VoiceChannel", (), {})
    # Make isinstance checks work for voice channels.
    mock_discord.VoiceChannel = MagicMock
    mgr._discord = mock_discord

    # Mock voice_recv module so join() can use VoiceRecvClient.
    mock_voice_recv = MagicMock()
    mgr._voice_recv = mock_voice_recv

    return mgr


def _setup_client(mgr, guild):
    """Wire a mock client with the given guild."""
    client = MagicMock()
    client.get_guild = MagicMock(return_value=guild)
    client.user = MagicMock(id=999)
    mgr._client = client


# ---------------------------------------------------------------------------
# DiscordVoiceManager — channel resolution
# ---------------------------------------------------------------------------


class TestFindVoiceChannel:
    def test_find_by_exact_name(self, manager):
        ch = _make_voice_channel(name="Music Room")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        result = manager.find_voice_channel(999, "Music Room")
        assert result is ch

    def test_find_by_case_insensitive_name(self, manager):
        ch = _make_voice_channel(name="Gaming")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        result = manager.find_voice_channel(999, "gaming")
        assert result is ch

    def test_find_by_partial_name(self, manager):
        ch = _make_voice_channel(name="Music Room")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        result = manager.find_voice_channel(999, "music")
        assert result is ch

    def test_find_by_id(self, manager):
        ch = _make_voice_channel(channel_id=42, name="VC")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        result = manager.find_voice_channel(999, "42")
        assert result is ch

    def test_returns_none_when_not_found(self, manager):
        guild = _make_guild(voice_channels=[])
        _setup_client(manager, guild)

        result = manager.find_voice_channel(999, "nonexistent")
        assert result is None

    def test_list_voice_channels(self, manager):
        ch1 = _make_voice_channel(name="A")
        ch2 = _make_voice_channel(name="B")
        guild = _make_guild(voice_channels=[ch1, ch2])
        _setup_client(manager, guild)

        names = manager.list_voice_channels(999)
        assert names == ["A", "B"]


class TestGetUserVoiceChannel:
    def test_user_in_voice(self, manager):
        ch = _make_voice_channel(name="Lounge")
        member = MagicMock()
        member.voice = MagicMock(channel=ch)
        guild = _make_guild(members={42: member})
        _setup_client(manager, guild)

        result = manager.get_user_voice_channel(999, 42)
        assert result is ch

    def test_user_not_in_voice(self, manager):
        member = MagicMock()
        member.voice = None
        guild = _make_guild(members={42: member})
        _setup_client(manager, guild)

        result = manager.get_user_voice_channel(999, 42)
        assert result is None

    def test_user_not_in_guild(self, manager):
        guild = _make_guild(members={})
        _setup_client(manager, guild)

        result = manager.get_user_voice_channel(999, 42)
        assert result is None


# ---------------------------------------------------------------------------
# DiscordVoiceManager — join / leave
# ---------------------------------------------------------------------------


class TestJoin:
    def test_join_by_user_location(self, manager):
        ch = _make_voice_channel(name="Lounge")
        member = MagicMock()
        member.voice = MagicMock(channel=ch)
        guild = _make_guild(voice_channels=[ch], members={42: member})
        _setup_client(manager, guild)

        name = _run(manager.join(999, user_id=42))
        assert name == "Lounge"
        ch.connect.assert_awaited_once()

    def test_join_by_channel_name(self, manager):
        ch = _make_voice_channel(name="Gaming")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        name = _run(manager.join(999, channel_name="Gaming"))
        assert name == "Gaming"

    def test_join_by_channel_id(self, manager):
        ch = _make_voice_channel(channel_id=77, name="Music")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        name = _run(manager.join(999, channel_id=77))
        assert name == "Music"

    def test_join_user_not_in_voice_raises(self, manager):
        member = MagicMock()
        member.voice = None
        guild = _make_guild(members={42: member})
        _setup_client(manager, guild)

        with pytest.raises(DiscordVoiceError, match="not in a voice channel"):
            _run(manager.join(999, user_id=42))

    def test_join_unknown_channel_name_raises(self, manager):
        guild = _make_guild(voice_channels=[])
        _setup_client(manager, guild)

        with pytest.raises(DiscordVoiceError, match="No voice channel"):
            _run(manager.join(999, channel_name="nonexistent"))

    def test_join_no_args_raises(self, manager):
        guild = _make_guild()
        _setup_client(manager, guild)

        with pytest.raises(DiscordVoiceError, match="Specify a voice channel"):
            _run(manager.join(999))


class TestLeave:
    def test_leave_connected(self, manager):
        ch = _make_voice_channel(name="Lounge")
        vc = ch.connect.return_value
        manager._guild_states[999] = MagicMock(
            voice_client=vc, lock=asyncio.Lock(),
        )

        name = _run(manager.leave(999))
        assert name == "Lounge"
        vc.disconnect.assert_awaited_once()

    def test_leave_not_connected(self, manager):
        name = _run(manager.leave(999))
        assert name is None


class TestIsConnected:
    def test_connected(self, manager):
        vc = MagicMock(is_connected=MagicMock(return_value=True))
        manager._guild_states[999] = MagicMock(voice_client=vc)
        assert manager.is_connected(999) is True

    def test_not_connected(self, manager):
        assert manager.is_connected(999) is False


# ---------------------------------------------------------------------------
# Voice commands
# ---------------------------------------------------------------------------


class TestVoiceCommands:
    def _make_voice(self) -> MagicMock:
        voice = MagicMock(spec=DiscordVoiceManager)
        voice.is_ready = True
        voice.can_listen = True
        voice.can_speak = True
        voice.join = AsyncMock(return_value="General")
        voice.leave = AsyncMock(return_value="General")
        voice.say = AsyncMock()
        return voice

    def test_join_no_args_follows_user(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!join", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "General" in result.reply
        voice.join.assert_awaited_once_with(999, user_id=42)

    def test_join_by_name(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!join Music Room", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "Music Room" not in result.reply or "General" in result.reply
        voice.join.assert_awaited_once_with(999, channel_name="Music Room")

    def test_join_by_id(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!join 12345", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        voice.join.assert_awaited_once_with(999, channel_id=12345)

    def test_join_error_shown(self):
        voice = self._make_voice()
        voice.join = AsyncMock(side_effect=DiscordVoiceError("bad channel"))
        result = _run(maybe_handle_voice_command(
            content="!join nonexistent", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "bad channel" in result.reply

    def test_leave(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!leave", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "General" in result.reply
        voice.leave.assert_awaited_once_with(999)

    def test_leave_not_connected(self):
        voice = self._make_voice()
        voice.leave = AsyncMock(return_value=None)
        result = _run(maybe_handle_voice_command(
            content="!leave", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "not in" in result.reply.lower()

    def test_say(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!say hello world", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert result.reply is None  # no text reply on success
        voice.say.assert_awaited_once_with(999, "hello world")

    def test_say_no_text(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!say", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "usage" in result.reply.lower()

    def test_say_error(self):
        voice = self._make_voice()
        voice.say = AsyncMock(side_effect=DiscordVoiceError("not connected"))
        result = _run(maybe_handle_voice_command(
            content="!say hi", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "not connected" in result.reply

    def test_non_voice_command_ignored(self):
        result = _run(maybe_handle_voice_command(
            content="!foo", channel_id="ch-1", guild_id="999",
            author_id="42", voice=None,
        ))
        assert result.handled is False

    def test_dm_rejected(self):
        voice = self._make_voice()
        result = _run(maybe_handle_voice_command(
            content="!join", channel_id="ch-1", guild_id=None,
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "servers" in result.reply.lower()

    def test_voice_not_enabled(self):
        result = _run(maybe_handle_voice_command(
            content="!join", channel_id="ch-1", guild_id="999",
            author_id="42", voice=None,
        ))
        assert result.handled is True
        assert "not enabled" in result.reply.lower()

    def test_voice_not_ready(self):
        voice = self._make_voice()
        voice.is_ready = False
        result = _run(maybe_handle_voice_command(
            content="!join", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert result.handled is True
        assert "starting up" in result.reply.lower()

    def test_plain_message_ignored(self):
        result = _run(maybe_handle_voice_command(
            content="hello world", channel_id="ch-1", guild_id="999",
            author_id="42", voice=None,
        ))
        assert result.handled is False

    def test_join_shows_capabilities(self):
        voice = self._make_voice()
        voice.can_listen = True
        voice.can_speak = True
        result = _run(maybe_handle_voice_command(
            content="!join", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert "listening" in result.reply.lower()
        assert "speaking" in result.reply.lower()

    def test_join_no_capabilities(self):
        voice = self._make_voice()
        voice.can_listen = False
        voice.can_speak = False
        result = _run(maybe_handle_voice_command(
            content="!join", channel_id="ch-1", guild_id="999",
            author_id="42", voice=voice,
        ))
        assert "listening" not in result.reply.lower()
        assert "speaking" not in result.reply.lower()
        assert "General" in result.reply


# ---------------------------------------------------------------------------
# Audio resampling
# ---------------------------------------------------------------------------


class TestResamplePcm:
    def test_same_rate_passthrough(self):
        from missy.channels.discord.voice import _resample_pcm

        pcm = struct.pack("<4h", 100, 200, 300, 400)
        result = _resample_pcm(pcm, 16000, 16000)
        assert result == pcm

    def test_downsample_produces_fewer_samples(self):
        from missy.channels.discord.voice import _resample_pcm

        # 960 stereo samples at 48kHz = 20ms of audio.
        n = 960
        samples = [1000] * (n * 2)  # stereo
        pcm = struct.pack(f"<{n * 2}h", *samples)
        result = _resample_pcm(pcm, 48000, 16000)
        out_count = len(result) // 2
        # Should be roughly 960 / (48000/16000) / 2 (stereo→mono) ≈ 160 samples.
        # But our function does stereo→mono first then resample.
        assert out_count > 0
        assert out_count < n

    def test_empty_input(self):
        from missy.channels.discord.voice import _resample_pcm

        result = _resample_pcm(b"", 48000, 16000)
        assert result == b""


# ---------------------------------------------------------------------------
# SpeechCollectorSink
# ---------------------------------------------------------------------------


class TestSpeechCollectorSink:
    def _make_sink_cls(self):
        """Create a sink class with a mock AudioSink base."""
        mock_voice_recv = MagicMock()
        mock_voice_recv.AudioSink = type("AudioSink", (), {
            "__init__": lambda self: None,
        })
        return _make_sink_class(mock_voice_recv)

    def test_buffers_audio_per_user(self):
        SinkClass = self._make_sink_cls()
        collected = []
        loop = asyncio.new_event_loop()

        sink = SinkClass(
            loop=loop,
            on_speech_done=lambda uid, pcm, sr: collected.append((uid, pcm, sr)),
            bot_user_id=999,
        )

        user = MagicMock(id=42)
        data = MagicMock(pcm=b"\x00\x01\x02\x03")
        sink.write(user, data)

        assert 42 in sink._buffers
        assert len(sink._buffers[42]) == 1
        sink.cleanup()
        loop.close()

    def test_ignores_bot_audio(self):
        SinkClass = self._make_sink_cls()
        loop = asyncio.new_event_loop()
        sink = SinkClass(
            loop=loop,
            on_speech_done=lambda uid, pcm, sr: None,
            bot_user_id=999,
        )

        bot_user = MagicMock(id=999)
        data = MagicMock(pcm=b"\x00\x01")
        sink.write(bot_user, data)

        assert 999 not in sink._buffers
        sink.cleanup()
        loop.close()

    def test_cleanup_clears_state(self):
        SinkClass = self._make_sink_cls()
        loop = asyncio.new_event_loop()
        sink = SinkClass(
            loop=loop,
            on_speech_done=lambda uid, pcm, sr: None,
            bot_user_id=0,
        )

        user = MagicMock(id=42)
        data = MagicMock(pcm=b"\x00\x01")
        sink.write(user, data)
        sink.cleanup()

        assert len(sink._buffers) == 0
        assert len(sink._timers) == 0
        loop.close()


# ---------------------------------------------------------------------------
# _clean_for_speech
# ---------------------------------------------------------------------------


class TestCleanForSpeech:
    def test_strips_code_blocks(self):
        from missy.channels.discord.voice import _clean_for_speech

        text = "Here is code:\n```python\nprint('hi')\n```\nDone."
        assert "```" not in _clean_for_speech(text)
        assert "Done." in _clean_for_speech(text)

    def test_strips_inline_code(self):
        from missy.channels.discord.voice import _clean_for_speech

        assert _clean_for_speech("Run `pip install foo` now") == "Run pip install foo now"

    def test_strips_markdown_bold(self):
        from missy.channels.discord.voice import _clean_for_speech

        assert _clean_for_speech("This is **important**") == "This is important"

    def test_strips_markdown_headers(self):
        from missy.channels.discord.voice import _clean_for_speech

        result = _clean_for_speech("## Title\nSome text")
        assert "##" not in result
        assert "Title" in result

    def test_strips_urls(self):
        from missy.channels.discord.voice import _clean_for_speech

        result = _clean_for_speech("Check https://example.com for details")
        assert "https://" not in result
        assert "Check" in result

    def test_truncates_long_responses(self):
        from missy.channels.discord.voice import _clean_for_speech

        long_text = "This is a sentence. " * 50
        result = _clean_for_speech(long_text)
        assert len(result) <= 601  # 600 + possible period

    def test_empty_string(self):
        from missy.channels.discord.voice import _clean_for_speech

        assert _clean_for_speech("") == ""
        assert _clean_for_speech("   ") == ""

    def test_preserves_normal_text(self):
        from missy.channels.discord.voice import _clean_for_speech

        text = "Hello, how are you doing today?"
        assert _clean_for_speech(text) == text

    def test_strips_markdown_links(self):
        from missy.channels.discord.voice import _clean_for_speech

        result = _clean_for_speech("See [the docs](https://example.com) for more")
        assert "the docs" in result
        assert "https://" not in result
        assert "[" not in result


# ---------------------------------------------------------------------------
# Listen watchdog
# ---------------------------------------------------------------------------


class TestListenWatchdog:
    def test_watchdog_restarts_dead_router(self):
        """Watchdog detects dead PacketRouter thread and re-attaches sink."""
        from missy.channels.discord.voice import (
            DiscordVoiceManager,
            _GuildVoiceState,
        )

        mgr = DiscordVoiceManager.__new__(DiscordVoiceManager)
        mgr._loop = asyncio.new_event_loop()
        mgr._client = MagicMock()
        mgr._client.user = MagicMock(id=999)
        mgr._stt_engine = MagicMock()
        mgr._tts_engine = MagicMock()
        mgr._agent_callback = None

        # Build a mock voice_recv module.
        mock_voice_recv = MagicMock()
        mock_voice_recv.AudioSink = type("AudioSink", (), {
            "__init__": lambda self: None,
        })
        mgr._voice_recv = mock_voice_recv

        # Mock voice client with dead router thread.
        vc = MagicMock()
        vc.is_connected.return_value = True
        dead_router = MagicMock()
        dead_router.is_alive.return_value = False
        vc._packet_router = dead_router

        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states = {1: state}

        # Run the watchdog for one tick.
        async def _run_one_tick():
            import missy.channels.discord.voice as vmod
            original = vmod._WATCHDOG_INTERVAL_S
            vmod._WATCHDOG_INTERVAL_S = 0.01
            task = asyncio.ensure_future(mgr._listen_watchdog(1))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            vmod._WATCHDOG_INTERVAL_S = original

        mgr._loop.run_until_complete(_run_one_tick())

        # The watchdog should have called stop_listening then listen(new_sink).
        vc.stop_listening.assert_called()
        vc.listen.assert_called()
        mgr._loop.close()

    def test_watchdog_ignores_alive_router(self):
        """Watchdog does nothing when the PacketRouter is alive."""
        from missy.channels.discord.voice import (
            DiscordVoiceManager,
            _GuildVoiceState,
        )

        mgr = DiscordVoiceManager.__new__(DiscordVoiceManager)
        mgr._loop = asyncio.new_event_loop()
        mgr._client = MagicMock()
        mgr._client.user = MagicMock(id=999)
        mgr._stt_engine = MagicMock()
        mgr._tts_engine = MagicMock()
        mgr._agent_callback = None
        mgr._voice_recv = MagicMock()

        vc = MagicMock()
        vc.is_connected.return_value = True
        alive_router = MagicMock()
        alive_router.is_alive.return_value = True
        vc._packet_router = alive_router

        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states = {1: state}

        async def _run_one_tick():
            import missy.channels.discord.voice as vmod
            original = vmod._WATCHDOG_INTERVAL_S
            vmod._WATCHDOG_INTERVAL_S = 0.01
            task = asyncio.ensure_future(mgr._listen_watchdog(1))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            vmod._WATCHDOG_INTERVAL_S = original

        mgr._loop.run_until_complete(_run_one_tick())

        # Should NOT have restarted.
        vc.stop_listening.assert_not_called()
        mgr._loop.close()
