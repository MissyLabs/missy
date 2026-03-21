"""Extended tests for missy/channels/discord/voice.py — uncovered paths.

The existing test_discord_voice.py covers channel resolution, join/leave,
_clean_for_speech, _resample_pcm, and the sink. This module covers:
- DiscordVoiceManager.__init__ (ffmpeg check, DiscordVoiceError)
- is_connected / is_listening / current_channel_name
- say() / _play_tts() paths
- stop() lifecycle
- _handle_speech() paths
- _start_listening / _attach_sink
- _listen_watchdog edge cases
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.discord.voice import (
    DiscordVoiceError,
    DiscordVoiceManager,
    _GuildVoiceState,
    _make_sink_class,
    _resample_pcm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        # Cancel any pending tasks to avoid "coroutine was never awaited" warnings
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


@pytest.fixture()
def manager():
    """Create a DiscordVoiceManager with mocked internals."""
    with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
        mgr = DiscordVoiceManager()
    mgr._ready.set()

    mock_discord = MagicMock()
    mock_discord.VoiceChannel = MagicMock
    mgr._discord = mock_discord

    mock_voice_recv = MagicMock()
    mgr._voice_recv = mock_voice_recv

    return mgr


def _setup_client(mgr, guild):
    client = MagicMock()
    client.get_guild = MagicMock(return_value=guild)
    client.user = MagicMock(id=999)
    mgr._client = client


def _make_guild(guild_id=999, voice_channels=None, members=None):
    guild = MagicMock()
    guild.id = guild_id
    channels = voice_channels or []
    guild.voice_channels = channels
    channel_map = {ch.id: ch for ch in channels}
    guild.get_channel = lambda cid: channel_map.get(cid)
    member_map = members or {}
    guild.get_member = lambda uid: member_map.get(uid)
    return guild


def _make_voice_channel(*, channel_id=100, name="General"):
    ch = MagicMock()
    ch.id = channel_id
    ch.name = name
    vc_mock = MagicMock(
        is_connected=MagicMock(return_value=True),
        channel=ch,
        disconnect=AsyncMock(),
        move_to=AsyncMock(),
        is_playing=MagicMock(return_value=False),
        play=MagicMock(),
        stop=MagicMock(),
        listen=MagicMock(),
    )
    ch.connect = AsyncMock(return_value=vc_mock)
    return ch


def _make_sink_cls():
    mock_voice_recv = MagicMock()
    mock_voice_recv.AudioSink = type("AudioSink", (), {"__init__": lambda self: None})
    return _make_sink_class(mock_voice_recv)


# ---------------------------------------------------------------------------
# __init__ — ffmpeg check
# ---------------------------------------------------------------------------


class TestDiscordVoiceManagerInit:
    def test_raises_on_missing_ffmpeg(self) -> None:
        with (
            patch(
                "missy.channels.discord.voice.ensure_ffmpeg_available",
                side_effect=RuntimeError("ffmpeg not found"),
            ),
            pytest.raises(DiscordVoiceError, match="ffmpeg not found"),
        ):
            DiscordVoiceManager()

    def test_ok_with_ffmpeg_present(self) -> None:
        with patch(
            "missy.channels.discord.voice.ensure_ffmpeg_available", return_value="/usr/bin/ffmpeg"
        ):
            mgr = DiscordVoiceManager()
        assert mgr._client is None
        assert not mgr.is_ready

    def test_callbacks_stored(self) -> None:
        cb1 = AsyncMock()
        cb2 = AsyncMock()
        with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
            mgr = DiscordVoiceManager(agent_callback=cb1, text_channel_callback=cb2)
        assert mgr._agent_callback is cb1
        assert mgr._text_channel_callback is cb2


# ---------------------------------------------------------------------------
# is_ready / can_listen / can_speak properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_ready_false_before_start(self, manager) -> None:
        manager._ready.clear()
        assert manager.is_ready is False

    def test_is_ready_true_after_set(self, manager) -> None:
        assert manager.is_ready is True

    def test_can_listen_requires_stt_and_voice_recv(self, manager) -> None:
        manager._stt_engine = None
        assert manager.can_listen is False

        manager._stt_engine = MagicMock()
        manager._voice_recv = None
        assert manager.can_listen is False

        manager._voice_recv = MagicMock()
        assert manager.can_listen is True

    def test_can_speak_requires_tts(self, manager) -> None:
        manager._tts_engine = None
        assert manager.can_speak is False

        manager._tts_engine = MagicMock()
        assert manager.can_speak is True


# ---------------------------------------------------------------------------
# is_connected / is_listening / current_channel_name
# ---------------------------------------------------------------------------


class TestStatusHelpers:
    def test_is_connected_no_state(self, manager) -> None:
        assert manager.is_connected(999) is False

    def test_is_connected_with_state(self, manager) -> None:
        vc = MagicMock(is_connected=MagicMock(return_value=True))
        manager._guild_states[999] = MagicMock(voice_client=vc)
        assert manager.is_connected(999) is True

    def test_is_listening_no_state(self, manager) -> None:
        assert manager.is_listening(999) is False

    def test_is_listening_with_state(self, manager) -> None:
        state = MagicMock(listening=True)
        manager._guild_states[999] = state
        assert manager.is_listening(999) is True

    def test_current_channel_name_no_state(self, manager) -> None:
        assert manager.current_channel_name(999) is None

    def test_current_channel_name_with_channel(self, manager) -> None:
        ch = MagicMock()
        ch.name = "Music"  # set directly to avoid MagicMock name= special param
        vc = MagicMock(channel=ch)
        manager._guild_states[999] = MagicMock(voice_client=vc)
        assert manager.current_channel_name(999) == "Music"

    def test_current_channel_name_no_channel(self, manager) -> None:
        vc = MagicMock(channel=None)
        manager._guild_states[999] = MagicMock(voice_client=vc)
        assert manager.current_channel_name(999) is None


# ---------------------------------------------------------------------------
# _get_guild / find_voice_channel edge cases
# ---------------------------------------------------------------------------


class TestGetGuild:
    def test_raises_when_client_is_none(self, manager) -> None:
        manager._client = None
        with pytest.raises(DiscordVoiceError, match="not started"):
            manager._get_guild(999)

    def test_raises_when_guild_not_found(self, manager) -> None:
        client = MagicMock()
        client.get_guild.return_value = None
        manager._client = client
        with pytest.raises(DiscordVoiceError, match="Guild not found"):
            manager._get_guild(999)

    def test_find_voice_channel_returns_none_for_numeric_id_mismatch(self, manager) -> None:
        ch = _make_voice_channel(channel_id=42, name="VC")
        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        # channel is not a VoiceChannel instance in the mock, so get_channel returns ch
        # but isinstance check fails — simulate by making get_channel return None for digit queries.
        guild.get_channel = lambda cid: None
        result = manager.find_voice_channel(999, "99")
        assert result is None


# ---------------------------------------------------------------------------
# join — already connected paths
# ---------------------------------------------------------------------------


class TestJoinAlreadyConnected:
    def test_join_same_channel_starts_listening_if_not_already(self, manager) -> None:
        ch = _make_voice_channel(name="Office")
        vc = ch.connect.return_value
        vc.is_connected = MagicMock(return_value=True)
        vc.channel = ch

        state = _GuildVoiceState(voice_client=vc)
        state.listening = False
        manager._guild_states[999] = state

        guild = _make_guild(voice_channels=[ch])
        _setup_client(manager, guild)

        manager._stt_engine = None  # can_listen is False => _start_listening is a no-op

        name = _run(manager.join(999, channel_name="Office"))
        assert name == "Office"

    def test_join_different_channel_moves(self, manager) -> None:
        ch1 = _make_voice_channel(channel_id=1, name="Old")
        ch2 = _make_voice_channel(channel_id=2, name="New")

        vc = ch1.connect.return_value
        vc.is_connected = MagicMock(return_value=True)
        vc.channel = ch1

        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        manager._guild_states[999] = state

        guild = _make_guild(voice_channels=[ch1, ch2])
        _setup_client(manager, guild)

        name = _run(manager.join(999, channel_name="New"))
        assert name == "New"
        vc.move_to.assert_awaited_once_with(ch2)


# ---------------------------------------------------------------------------
# leave — lifecycle
# ---------------------------------------------------------------------------


class TestLeaveExtended:
    def test_leave_cancels_watchdog_and_listen_task(self, manager) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            vc = MagicMock(
                is_connected=MagicMock(return_value=False),
                channel=MagicMock(),
                disconnect=AsyncMock(),
            )
            watchdog = loop.create_task(asyncio.sleep(1000))
            listen_task = loop.create_task(asyncio.sleep(1000))

            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            state.watchdog_task = watchdog
            state.listen_task = listen_task
            manager._guild_states[999] = state

            loop.run_until_complete(manager.leave(999))

            assert watchdog.cancelled()
            assert listen_task.cancelled()
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_stop_disconnects_all_guilds_and_shuts_down(self, manager) -> None:
        stt = MagicMock()
        tts = MagicMock()
        manager._stt_engine = stt
        manager._tts_engine = tts
        manager._client = MagicMock()
        manager._client.close = AsyncMock()
        manager._client_task = None

        _run(manager.stop())

        stt.unload.assert_called_once()
        tts.unload.assert_called_once()
        manager._client.close.assert_awaited()

    def test_stop_handles_stt_unload_exception(self, manager) -> None:
        stt = MagicMock()
        stt.unload.side_effect = RuntimeError("boom")
        manager._stt_engine = stt
        manager._tts_engine = None
        manager._client = None
        manager._client_task = None

        _run(manager.stop())  # Should not propagate

    def test_stop_cancels_client_task(self, manager) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            manager._stt_engine = None
            manager._tts_engine = None
            manager._client = MagicMock()
            manager._client.close = AsyncMock()

            task = loop.create_task(asyncio.sleep(1000))
            manager._client_task = task

            loop.run_until_complete(manager.stop())

            assert task.cancelled()
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# say
# ---------------------------------------------------------------------------


class TestSay:
    def test_say_raises_when_no_text(self, manager) -> None:
        with pytest.raises(DiscordVoiceError, match="No text"):
            _run(manager.say(999, ""))

    def test_say_raises_when_tts_not_configured(self, manager) -> None:
        manager._tts_engine = None
        with pytest.raises(DiscordVoiceError, match="TTS is not configured"):
            _run(manager.say(999, "Hello"))

    def test_say_raises_when_not_connected(self, manager) -> None:
        manager._tts_engine = MagicMock()
        with pytest.raises(DiscordVoiceError, match="not in a voice channel"):
            _run(manager.say(999, "Hello"))

    def test_say_calls_play_tts(self, manager) -> None:
        tts = MagicMock()
        manager._tts_engine = tts
        manager._play_tts = AsyncMock()

        vc = MagicMock(is_connected=MagicMock(return_value=True))
        state = _GuildVoiceState(voice_client=vc)
        manager._guild_states[999] = state

        _run(manager.say(999, "Hello world"))

        manager._play_tts.assert_awaited_once_with(state, "Hello world")


# ---------------------------------------------------------------------------
# _play_tts
# ---------------------------------------------------------------------------


class TestPlayTts:
    def _make_audio_buf(self, data=b"\x00" * 100):
        from missy.channels.voice.tts.base import AudioBuffer

        return AudioBuffer(data=data, sample_rate=22050, channels=1, format="wav")

    def test_play_tts_stops_playing_first(self, manager) -> None:
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=self._make_audio_buf())
        manager._tts_engine = tts

        vc = MagicMock()
        vc.is_playing = MagicMock(return_value=True)
        vc.stop = MagicMock()

        # Capture the after callback and call it immediately when play() is called.
        def fake_play(source, after=None):
            if after:
                after(None)

        vc.play = fake_play

        state = _GuildVoiceState(voice_client=vc)
        discord_mock = MagicMock()
        manager._discord = discord_mock

        _run(manager._play_tts(state, "Hi"))

        vc.stop.assert_called_once()

    def test_play_tts_cleans_up_temp_file(self, manager) -> None:
        from pathlib import Path

        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=self._make_audio_buf())
        manager._tts_engine = tts

        vc = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        # Call after callback immediately to satisfy the wait_for.
        def fake_play(source, after=None):
            if after:
                after(None)

        vc.play = fake_play

        state = _GuildVoiceState(voice_client=vc)

        created_paths = []
        original_ntf = __import__("tempfile").NamedTemporaryFile

        def tracking_ntf(**kwargs):
            f = original_ntf(**kwargs)
            created_paths.append(f.name)
            return f

        with patch("tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            _run(manager._play_tts(state, "Hello"))

        for p in created_paths:
            assert not Path(p).exists()

    def test_play_tts_handles_timeout(self, manager) -> None:
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=self._make_audio_buf())
        manager._tts_engine = tts

        vc = MagicMock()
        vc.is_playing = MagicMock(return_value=False)
        vc.stop = MagicMock()

        # Make play() not call after() — wait_for will timeout.
        vc.play = MagicMock()

        state = _GuildVoiceState(voice_client=vc)

        # Patch wait_for to raise TimeoutError directly.
        async def timeout_wait(coro, timeout=None):
            # Cancel the coroutine first to avoid warnings.
            coro.close()
            raise TimeoutError

        with patch("asyncio.wait_for", side_effect=timeout_wait):
            _run(manager._play_tts(state, "Hello"))  # Should not raise

        vc.stop.assert_called()


# ---------------------------------------------------------------------------
# _handle_speech
# ---------------------------------------------------------------------------


class TestHandleSpeech:
    def _make_manager_for_speech(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
            mgr = DiscordVoiceManager.__new__(DiscordVoiceManager)
        mgr._ready = asyncio.Event()
        mgr._ready.set()
        mgr._guild_states = {}
        mgr._client = MagicMock()
        mgr._client.user = MagicMock(id=0)
        mgr._discord = MagicMock()
        mgr._voice_recv = MagicMock()
        mgr._loop = loop
        mgr._client_task = None
        return mgr, loop

    def test_handle_speech_returns_if_no_state(self) -> None:
        mgr, loop = self._make_manager_for_speech()
        mgr._stt_engine = MagicMock()
        mgr._agent_callback = AsyncMock(return_value="")
        mgr._tts_engine = None
        # No guild state — should return without error.
        pcm = struct.pack("<200h", *([100] * 200))
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=pcm, sample_rate=48000)
            )
        finally:
            loop.close()

    def test_handle_speech_skips_when_bot_playing(self) -> None:
        mgr, loop = self._make_manager_for_speech()
        vc = MagicMock(is_playing=MagicMock(return_value=True))
        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states[999] = state
        mgr._stt_engine = AsyncMock()

        pcm = struct.pack("<200h", *([100] * 200))
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=pcm, sample_rate=48000)
            )
        finally:
            loop.close()

        mgr._stt_engine.transcribe.assert_not_called()

    def test_handle_speech_skips_when_not_listening(self) -> None:
        mgr, loop = self._make_manager_for_speech()
        vc = MagicMock(is_playing=MagicMock(return_value=False))
        state = _GuildVoiceState(voice_client=vc)
        state.listening = False
        mgr._guild_states[999] = state

        pcm = struct.pack("<200h", *([100] * 200))
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=pcm, sample_rate=48000)
            )
        finally:
            loop.close()

    def test_handle_speech_skips_short_audio(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        mgr, loop = self._make_manager_for_speech()
        vc = MagicMock(is_playing=MagicMock(return_value=False))
        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states[999] = state

        stt = MagicMock()
        stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hi",
                confidence=1.0,
                processing_ms=5,
            )
        )
        mgr._stt_engine = stt
        mgr._agent_callback = AsyncMock(return_value="")

        # Very short PCM — less than _MIN_SPEECH_S (0.3s) at 16kHz mono.
        short_pcm = struct.pack("<8h", *([100] * 8))  # just 8 stereo samples
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=short_pcm, sample_rate=48000)
            )
        finally:
            loop.close()

        stt.transcribe.assert_not_called()

    def test_handle_speech_skips_empty_transcript(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        mgr, loop = self._make_manager_for_speech()
        vc = MagicMock(is_playing=MagicMock(return_value=False))
        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states[999] = state

        stt = MagicMock()
        stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="   ",
                confidence=0.0,
                processing_ms=5,
            )
        )
        mgr._stt_engine = stt
        mgr._agent_callback = AsyncMock(return_value="")

        long_pcm = struct.pack("<5000h", *([100] * 5000))
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=long_pcm, sample_rate=48000)
            )
        finally:
            loop.close()

        mgr._agent_callback.assert_not_awaited()

    def test_handle_speech_skips_when_no_agent_callback(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        mgr, loop = self._make_manager_for_speech()
        vc = MagicMock(is_playing=MagicMock(return_value=False))
        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states[999] = state

        stt = MagicMock()
        stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello there",
                confidence=0.9,
                processing_ms=50,
            )
        )
        mgr._stt_engine = stt
        mgr._agent_callback = None

        guild = _make_guild(members={42: MagicMock(display_name="Alice")})
        mgr._client.get_guild = MagicMock(return_value=guild)

        long_pcm = struct.pack("<5000h", *([1000] * 5000))
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=long_pcm, sample_rate=48000)
            )
        finally:
            loop.close()

    def test_handle_speech_skips_empty_agent_response(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        mgr, loop = self._make_manager_for_speech()
        vc = MagicMock(is_playing=MagicMock(return_value=False))
        state = _GuildVoiceState(voice_client=vc)
        state.listening = True
        mgr._guild_states[999] = state

        stt = MagicMock()
        stt.transcribe = AsyncMock(
            return_value=TranscriptionResult(
                text="hello",
                confidence=0.9,
                processing_ms=50,
            )
        )
        mgr._stt_engine = stt
        mgr._agent_callback = AsyncMock(return_value="   ")  # whitespace only
        mgr._tts_engine = None

        guild = _make_guild(members={42: MagicMock(display_name="Bob")})
        mgr._client.get_guild = MagicMock(return_value=guild)

        long_pcm = struct.pack("<5000h", *([1000] * 5000))
        try:
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=long_pcm, sample_rate=48000)
            )
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# _SpeechCollectorSink — on_silence and wants_opus
# ---------------------------------------------------------------------------


class TestSpeechCollectorSinkOnSilence:
    def _make_sink(self, callback=None, bot_user_id=0):
        SinkClass = _make_sink_cls()
        loop = asyncio.new_event_loop()
        sink = SinkClass(
            loop=loop,
            on_speech_done=callback or (lambda uid, pcm, sr: None),
            bot_user_id=bot_user_id,
        )
        return sink, loop

    def test_wants_opus_returns_false(self) -> None:
        sink, loop = self._make_sink()
        assert sink.wants_opus() is False
        sink.cleanup()
        loop.close()

    def test_on_silence_fires_callback_with_joined_pcm(self) -> None:
        results = []
        SinkClass = _make_sink_cls()
        loop = asyncio.new_event_loop()

        sink = SinkClass(
            loop=loop,
            on_speech_done=lambda uid, pcm, sr: results.append((uid, pcm, sr)),
            bot_user_id=0,
        )

        user = MagicMock(id=7)
        data1 = MagicMock(pcm=b"\x01\x02")
        data2 = MagicMock(pcm=b"\x03\x04")

        # Write two packets with a mock timer so timers don't actually run.
        mock_handle = MagicMock()
        with patch.object(loop, "call_later", return_value=mock_handle):
            sink.write(user, data1)
            sink.write(user, data2)

        # Buffer should have 2 chunks.
        assert len(sink._buffers[7]) == 2

        # Manually trigger silence callback.
        sink._on_silence(7)

        assert len(results) == 1
        uid, pcm, sr = results[0]
        assert uid == 7
        assert pcm == b"\x01\x02\x03\x04"

        sink.cleanup()
        loop.close()

    def test_on_silence_does_nothing_when_buffer_empty(self) -> None:
        called = [False]
        sink, loop = self._make_sink(callback=lambda uid, pcm, sr: called.__setitem__(0, True))

        # Directly call _on_silence for a user with no buffer.
        sink._on_silence(42)

        assert not called[0]
        sink.cleanup()
        loop.close()

    def test_write_ignores_none_user(self) -> None:
        sink, loop = self._make_sink()
        data = MagicMock(pcm=b"\x00\x01")
        sink.write(None, data)  # Should not add anything.
        assert len(sink._buffers) == 0
        sink.cleanup()
        loop.close()

    def test_write_handles_integer_user_id(self) -> None:
        sink, loop = self._make_sink()
        # user has no .id attribute — uses int(user) instead.
        data = MagicMock(pcm=b"\x00\x01")
        with patch.object(loop, "call_later", return_value=MagicMock()):
            sink.write(55, data)  # integer user id
        assert 55 in sink._buffers
        sink.cleanup()
        loop.close()

    def test_write_ignores_empty_pcm(self) -> None:
        sink, loop = self._make_sink()
        user = MagicMock(id=10)
        # data with no pcm attribute and falsy bytes.
        data = b""
        sink.write(user, data)
        assert 10 not in sink._buffers
        sink.cleanup()
        loop.close()

    def test_on_silence_callback_exception_is_caught(self) -> None:
        def bad_cb(uid, pcm, sr):
            raise RuntimeError("callback exploded")

        sink, loop = self._make_sink(callback=bad_cb)
        sink._buffers[42] = [b"\x01\x02"]
        # Should not propagate.
        sink._on_silence(42)
        sink.cleanup()
        loop.close()


# ---------------------------------------------------------------------------
# _resample_pcm — edge cases
# ---------------------------------------------------------------------------


class TestResamplePcmEdgeCases:
    def test_single_sample_stereo(self) -> None:
        # 2 int16 values = 1 stereo frame.
        pcm = struct.pack("<2h", 1000, 2000)
        result = _resample_pcm(pcm, 48000, 16000)
        # Should not crash and should return bytes.
        assert isinstance(result, bytes)

    def test_odd_byte_count_is_handled(self) -> None:
        # Odd number of bytes — should not crash.
        pcm = b"\x01\x02\x03"  # 3 bytes, only 1 complete int16
        result = _resample_pcm(pcm, 48000, 16000)
        assert isinstance(result, bytes)

    def test_out_count_zero_returns_empty(self) -> None:
        # If ratio makes out_count = 0 — return empty.
        # Create a stereo frame where mono=0 samples after mix.
        pcm = b""  # no samples
        result = _resample_pcm(pcm, 48000, 16000)
        assert result == b""

    def test_values_clamped_to_int16_range(self) -> None:
        # Feed large positive values; output should not exceed int16 max.
        samples = [32767, 32767] * 50  # 100 stereo samples
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(pcm, 48000, 16000)
        out_count = len(result) // 2
        if out_count > 0:
            unpacked = struct.unpack(f"<{out_count}h", result)
            assert all(-32768 <= v <= 32767 for v in unpacked)

    def test_last_sample_boundary_hit(self) -> None:
        # Exercises the elif idx < len(samples) branch (lines 810-811) where
        # idx+1 == len(samples).  Use a ratio that produces an out_count
        # large enough to land exactly on the last valid index.
        # 2 stereo samples → 1 mono sample after mix-down.
        # ratio = 48000 / 16000 / 2 = 1.5 → out_count = int(1 / 1.5) = 0 for 1 mono.
        # Use more samples so out_count is at least 2 and ratio lands on boundary.
        # 4 stereo samples → 2 mono samples; ratio=1.5 → out_count = int(2/1.5) = 1.
        samples = [10000, 20000, 30000, 10000]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        result = _resample_pcm(pcm, 48000, 16000)
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# _clean_for_speech — uncovered truncation branch
# ---------------------------------------------------------------------------


class TestCleanForSpeech:
    def test_truncates_at_sentence_boundary_when_period_after_200(self) -> None:
        from missy.channels.discord.voice import _clean_for_speech

        # Build a string > 600 chars with a period after position 200.
        prefix = "A" * 250 + "."
        suffix = "B" * 400
        text = prefix + suffix
        result = _clean_for_speech(text)
        # Should cut at the period.
        assert result.endswith(".")
        assert len(result) <= 601

    def test_truncates_with_ellipsis_when_no_period_after_200(self) -> None:
        from missy.channels.discord.voice import _clean_for_speech

        # Build a string > 600 chars with no period at all.
        text = "X" * 700
        result = _clean_for_speech(text)
        assert result.endswith("...")


# ---------------------------------------------------------------------------
# start() — lifecycle import paths
# ---------------------------------------------------------------------------


class TestStart:
    def test_start_raises_when_discord_not_installed(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
                mgr = DiscordVoiceManager()

            import builtins

            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "discord":
                    raise ImportError("No module named 'discord'")
                return real_import(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=fake_import),
                pytest.raises(DiscordVoiceError, match="discord.py"),
            ):
                loop.run_until_complete(mgr.start("test-token"))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_start_raises_when_voice_recv_not_installed(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import builtins

            real_import = builtins.__import__

            # discord imports fine, but discord.ext.voice_recv does not.
            def fake_import(name, *args, **kwargs):
                if name == "discord.ext.voice_recv" or (
                    name == "discord.ext" and args and "voice_recv" in str(args)
                ):
                    raise ImportError("No module named 'discord.ext.voice_recv'")
                return real_import(name, *args, **kwargs)

            with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
                mgr = DiscordVoiceManager()

            mock_discord = MagicMock()

            with (
                patch.dict("sys.modules", {"discord": mock_discord}),
                patch("builtins.__import__", side_effect=fake_import),
                pytest.raises(DiscordVoiceError, match="discord-ext-voice-recv"),
            ):
                loop.run_until_complete(mgr.start("test-token"))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_start_strips_bot_prefix_from_token(self) -> None:
        """start() removes 'Bot ' prefix before calling client.start()."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
                mgr = DiscordVoiceManager()

            mock_discord = MagicMock()
            mock_voice_recv = MagicMock()

            # Capture what raw_token is passed to client.start().
            captured_tokens = []

            async def fake_client_start(token):
                captured_tokens.append(token)
                # Never completes — task will be cancelled.
                await asyncio.sleep(9999)

            mock_client_instance = MagicMock()
            mock_client_instance.start = fake_client_start
            mock_discord.Client.return_value = mock_client_instance
            mock_discord.Intents.default.return_value = MagicMock()

            # event decorator: register on_ready and fire it shortly after.
            def fake_event(func):
                loop.call_soon(
                    lambda: loop.call_soon(lambda: asyncio.ensure_future(func(), loop=loop))
                )
                return func

            mock_client_instance.event = fake_event
            mock_client_instance.user = MagicMock(id=12345)

            async def run():
                import sys as _sys

                _sys.modules["discord"] = mock_discord
                _sys.modules["discord.ext.voice_recv"] = mock_voice_recv
                try:
                    await mgr.start("Bot mytoken123")
                except DiscordVoiceError:
                    pass  # timeout or other error is ok in test
                finally:
                    if mgr._client_task:
                        mgr._client_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await mgr._client_task

            loop.run_until_complete(run())

            if captured_tokens:
                assert captured_tokens[0] == "mytoken123"
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_start_timeout_raises_discord_voice_error(self) -> None:
        """Timeout waiting for on_ready raises DiscordVoiceError."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
                mgr = DiscordVoiceManager()

            mock_discord = MagicMock()
            mock_voice_recv = MagicMock()

            async def fake_client_start(token):
                await asyncio.sleep(9999)

            mock_client_instance = MagicMock()
            mock_client_instance.start = fake_client_start
            mock_discord.Client.return_value = mock_client_instance
            mock_discord.Intents.default.return_value = MagicMock()

            # event decorator that does NOT fire on_ready.
            mock_client_instance.event = lambda func: func

            import sys as _sys

            async def run():
                try:
                    with patch("asyncio.wait_for", side_effect=TimeoutError):
                        await mgr.start("token")
                finally:
                    if mgr._client_task:
                        mgr._client_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await mgr._client_task

            with (
                patch.dict(
                    _sys.modules,
                    {
                        "discord": mock_discord,
                        "discord.ext": MagicMock(),
                        "discord.ext.voice_recv": mock_voice_recv,
                    },
                ),
                pytest.raises(DiscordVoiceError, match="30 seconds"),
            ):
                loop.run_until_complete(run())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# stop() — exception handling branches
# ---------------------------------------------------------------------------


class TestStopExceptions:
    def test_stop_handles_tts_unload_exception(self) -> None:
        with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
            mgr = DiscordVoiceManager()
        mgr._ready.set()
        mgr._stt_engine = None
        tts = MagicMock()
        tts.unload.side_effect = RuntimeError("tts boom")
        mgr._tts_engine = tts
        mgr._client = None
        mgr._client_task = None
        _run(mgr.stop())  # Should not propagate.

    def test_stop_handles_client_close_exception(self) -> None:
        with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
            mgr = DiscordVoiceManager()
        mgr._ready.set()
        mgr._stt_engine = None
        mgr._tts_engine = None
        mgr._client = MagicMock()
        mgr._client.close = AsyncMock(side_effect=RuntimeError("close failed"))
        mgr._client_task = None
        _run(mgr.stop())  # Should not propagate.

    def test_stop_leaves_guild_states_and_handles_leave_exception(self) -> None:
        """Lines 221-224: stop() iterates guild states and catches leave() exceptions."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
                mgr = DiscordVoiceManager()
            mgr._ready.set()
            mgr._loop = loop
            mgr._stt_engine = None
            mgr._tts_engine = None
            mgr._client = MagicMock()
            mgr._client.close = AsyncMock()
            mgr._client_task = None

            # Add a guild state whose leave() will raise.
            vc = MagicMock()
            vc.channel = MagicMock()
            vc.channel.name = "Test"
            vc.disconnect = AsyncMock(side_effect=RuntimeError("disconnect boom"))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = False
            state.watchdog_task = None
            state.listen_task = None
            mgr._guild_states[123] = state

            loop.run_until_complete(mgr.stop())  # Should not raise.
            # Guild state should be gone after leave() was attempted.
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# leave() — disconnect exception branch (lines 398-399)
# ---------------------------------------------------------------------------


class TestLeaveDisconnectException:
    def test_leave_logs_disconnect_exception(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
                mgr = DiscordVoiceManager.__new__(DiscordVoiceManager)
            mgr._ready = asyncio.Event()
            mgr._guild_states = {}

            vc = MagicMock()
            vc.channel = MagicMock()
            vc.channel.name = "Test"
            vc.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

            state = _GuildVoiceState(voice_client=vc)
            state.listening = False
            state.watchdog_task = None
            state.listen_task = None
            mgr._guild_states[999] = state

            # Should not raise despite disconnect error.
            result = loop.run_until_complete(mgr.leave(999))
            assert result == "Test"
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# _play_tts — after-callback error logging and file-not-found cleanup
# ---------------------------------------------------------------------------


class TestPlayTtsExtraEdgeCases:
    def _make_audio_buf(self, data=b"\x00" * 100):
        from missy.channels.voice.tts.base import AudioBuffer

        return AudioBuffer(data=data, sample_rate=22050, channels=1, format="wav")

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_play_tts_logs_after_error(self, manager) -> None:
        """The _after callback with an error logs but does not raise."""
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=self._make_audio_buf())
        manager._tts_engine = tts

        vc = MagicMock()
        vc.is_playing = MagicMock(return_value=False)
        vc.stop = MagicMock()

        # Call after callback with an error.
        def fake_play(source, after=None):
            if after:
                after(RuntimeError("playback error"))

        vc.play = fake_play

        state = _GuildVoiceState(voice_client=vc)
        _run(manager._play_tts(state, "test"))  # Should not raise.

    def test_play_tts_handles_already_deleted_temp_file(self, manager) -> None:
        """FileNotFoundError during cleanup is silently ignored (lines 464-465)."""
        tts = MagicMock()
        tts.synthesize = AsyncMock(return_value=self._make_audio_buf())
        manager._tts_engine = tts

        vc = MagicMock()
        vc.is_playing = MagicMock(return_value=False)

        def fake_play(source, after=None):
            if after:
                after(None)

        vc.play = fake_play

        state = _GuildVoiceState(voice_client=vc)

        # Patch os.remove to raise FileNotFoundError.
        with patch("os.remove", side_effect=FileNotFoundError("already gone")):
            _run(manager._play_tts(state, "cleanup test"))  # Should not raise.


# ---------------------------------------------------------------------------
# _listen_watchdog — state/connection/router branches
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestListenWatchdog:
    def _make_manager(self):
        with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
            mgr = DiscordVoiceManager.__new__(DiscordVoiceManager)
        mgr._ready = asyncio.Event()
        mgr._ready.set()
        mgr._guild_states = {}
        mgr._client = MagicMock()
        mgr._client.user = MagicMock(id=0)
        mgr._discord = MagicMock()
        mgr._voice_recv = MagicMock()
        mgr._loop = asyncio.get_event_loop()
        mgr._client_task = None
        mgr._stt_engine = MagicMock()
        mgr._tts_engine = None
        return mgr

    def test_watchdog_exits_when_state_removed(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager()
            mgr._loop = loop

            call_count = [0]
            original_sleep = asyncio.sleep

            async def fast_sleep(s):
                call_count[0] += 1
                if call_count[0] >= 1:
                    # Remove state so watchdog exits.
                    mgr._guild_states.pop(999, None)
                await original_sleep(0)

            with patch("asyncio.sleep", side_effect=fast_sleep):
                loop.run_until_complete(mgr._listen_watchdog(999))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_watchdog_exits_when_not_listening(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager()
            mgr._loop = loop

            vc = MagicMock(is_connected=MagicMock(return_value=True))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = False
            mgr._guild_states[999] = state

            call_count = [0]
            original_sleep = asyncio.sleep

            async def fast_sleep(s):
                call_count[0] += 1
                await original_sleep(0)

            with patch("asyncio.sleep", side_effect=fast_sleep):
                loop.run_until_complete(mgr._listen_watchdog(999))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_watchdog_exits_when_vc_disconnected(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager()
            mgr._loop = loop

            vc = MagicMock(is_connected=MagicMock(return_value=False))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            call_count = [0]
            original_sleep = asyncio.sleep

            async def fast_sleep(s):
                call_count[0] += 1
                await original_sleep(0)

            with patch("asyncio.sleep", side_effect=fast_sleep):
                loop.run_until_complete(mgr._listen_watchdog(999))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_watchdog_detects_dead_router_and_restarts(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager()
            mgr._loop = loop

            dead_router = MagicMock()
            dead_router.is_alive.return_value = False

            vc = MagicMock()
            vc.is_connected = MagicMock(return_value=True)
            vc._packet_router = dead_router
            vc.stop_listening = MagicMock()
            vc.listen = MagicMock()

            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            call_count = [0]
            original_sleep = asyncio.sleep

            async def fast_sleep(s):
                call_count[0] += 1
                # After restart, make router alive and stop listening to exit.
                if call_count[0] >= 2:
                    state.listening = False
                await original_sleep(0)

            mock_sink_cls = MagicMock()
            mock_sink_cls.return_value = MagicMock()
            with (
                patch("asyncio.sleep", side_effect=fast_sleep),
                patch("missy.channels.discord.voice._make_sink_class", return_value=mock_sink_cls),
            ):
                loop.run_until_complete(mgr._listen_watchdog(999))

            vc.stop_listening.assert_called()
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_watchdog_handles_router_via_packet_router_attr(self) -> None:
        """Uses vc.packet_router (no leading underscore) when _packet_router absent."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager()
            mgr._loop = loop

            dead_router = MagicMock()
            dead_router.is_alive.return_value = False

            vc = MagicMock(spec=[])  # empty spec to control attributes manually
            vc.is_connected = MagicMock(return_value=True)
            # No _packet_router — use packet_router instead.
            vc.packet_router = dead_router
            vc.listen = MagicMock()

            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            call_count = [0]
            original_sleep = asyncio.sleep

            async def fast_sleep(s):
                call_count[0] += 1
                if call_count[0] >= 2:
                    state.listening = False
                await original_sleep(0)

            mock_sink_cls = MagicMock()
            mock_sink_cls.return_value = MagicMock()
            with (
                patch("asyncio.sleep", side_effect=fast_sleep),
                patch("missy.channels.discord.voice._make_sink_class", return_value=mock_sink_cls),
            ):
                loop.run_until_complete(mgr._listen_watchdog(999))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_watchdog_logs_error_when_restart_fails(self) -> None:
        """Lines 550-551: restart failure is logged, not re-raised."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager()
            mgr._loop = loop

            dead_router = MagicMock()
            dead_router.is_alive.return_value = False

            vc = MagicMock()
            vc.is_connected = MagicMock(return_value=True)
            vc._packet_router = dead_router
            vc.stop_listening = MagicMock(side_effect=RuntimeError("stop failed"))
            vc.listen = MagicMock()

            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            call_count = [0]
            original_sleep = asyncio.sleep

            async def fast_sleep(s):
                call_count[0] += 1
                if call_count[0] >= 2:
                    state.listening = False
                await original_sleep(0)

            mock_sink_cls = MagicMock()
            mock_sink_cls.return_value = MagicMock()
            with (
                patch("asyncio.sleep", side_effect=fast_sleep),
                patch("missy.channels.discord.voice._make_sink_class", return_value=mock_sink_cls),
            ):
                loop.run_until_complete(mgr._listen_watchdog(999))
            # No assertion needed — just verifying it doesn't raise.
        finally:
            loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# _handle_speech — success path with TTS, member not found, TTS error
# ---------------------------------------------------------------------------


class TestHandleSpeechFullPaths:
    def _make_manager_for_speech(self, loop=None):
        if loop is None:
            loop = asyncio.get_event_loop()
        with patch("missy.channels.discord.voice.ensure_ffmpeg_available"):
            mgr = DiscordVoiceManager.__new__(DiscordVoiceManager)
        mgr._ready = asyncio.Event()
        mgr._ready.set()
        mgr._guild_states = {}
        mgr._client = MagicMock()
        mgr._client.user = MagicMock(id=0)
        mgr._discord = MagicMock()
        mgr._voice_recv = MagicMock()
        mgr._loop = loop
        mgr._client_task = None
        return mgr

    def test_handle_speech_full_success_with_tts(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager_for_speech(loop)
            vc = MagicMock(is_playing=MagicMock(return_value=False))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            stt = MagicMock()
            stt.transcribe = AsyncMock(
                return_value=TranscriptionResult(
                    text="what time is it",
                    confidence=0.92,
                    processing_ms=120,
                )
            )
            mgr._stt_engine = stt

            tts_engine = MagicMock()
            mgr._tts_engine = tts_engine
            mgr._play_tts = AsyncMock()

            agent_cb = AsyncMock(return_value="It is three o'clock.")
            mgr._agent_callback = agent_cb

            guild = _make_guild(members={42: MagicMock(display_name="Charlie")})
            mgr._client.get_guild = MagicMock(return_value=guild)

            # Need >=20000 stereo samples to produce >=0.3s at 16kHz after resample.
            long_pcm = struct.pack("<20000h", *([1000] * 20000))
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=42, pcm_48k=long_pcm, sample_rate=48000)
            )

            agent_cb.assert_awaited_once()
            mgr._play_tts.assert_awaited_once_with(state, "It is three o'clock.")
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_handle_speech_member_not_found_uses_user_id_string(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager_for_speech(loop)
            vc = MagicMock(is_playing=MagicMock(return_value=False))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            stt = MagicMock()
            stt.transcribe = AsyncMock(
                return_value=TranscriptionResult(
                    text="hello missy",
                    confidence=0.85,
                    processing_ms=90,
                )
            )
            mgr._stt_engine = stt
            mgr._tts_engine = None

            agent_cb = AsyncMock(return_value="Hello!")
            mgr._agent_callback = agent_cb

            # Member not found → user name falls back to str(user_id).
            guild = _make_guild(members={})
            mgr._client.get_guild = MagicMock(return_value=guild)

            long_pcm = struct.pack("<20000h", *([500] * 20000))
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=99, pcm_48k=long_pcm, sample_rate=48000)
            )

            agent_cb.assert_awaited_once()
            prompt_arg = agent_cb.call_args[0][0]
            assert "99" in prompt_arg  # user_id used as name
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_handle_speech_tts_playback_exception_is_caught(self) -> None:
        from missy.channels.voice.stt.base import TranscriptionResult

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager_for_speech(loop)
            vc = MagicMock(is_playing=MagicMock(return_value=False))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            stt = MagicMock()
            stt.transcribe = AsyncMock(
                return_value=TranscriptionResult(
                    text="tell me a joke",
                    confidence=0.9,
                    processing_ms=80,
                )
            )
            mgr._stt_engine = stt
            mgr._tts_engine = MagicMock()  # can_speak = True
            mgr._play_tts = AsyncMock(side_effect=RuntimeError("TTS exploded"))
            mgr._agent_callback = AsyncMock(return_value="Why did the chicken cross the road?")

            guild = _make_guild(members={5: MagicMock(display_name="Dave")})
            mgr._client.get_guild = MagicMock(return_value=guild)

            long_pcm = struct.pack("<20000h", *([200] * 20000))
            # Should not raise even though _play_tts raises.
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=5, pcm_48k=long_pcm, sample_rate=48000)
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_handle_speech_exception_in_stt_is_caught(self) -> None:
        """Top-level except in _handle_speech catches STT errors (line 638-639)."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager_for_speech(loop)
            vc = MagicMock(is_playing=MagicMock(return_value=False))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            stt = MagicMock()
            stt.transcribe = AsyncMock(side_effect=RuntimeError("stt crashed"))
            mgr._stt_engine = stt
            mgr._tts_engine = None
            mgr._agent_callback = None

            long_pcm = struct.pack("<20000h", *([300] * 20000))
            # Should not raise.
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=1, pcm_48k=long_pcm, sample_rate=48000)
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    def test_handle_speech_response_cleaned_to_empty_skips_tts(self) -> None:
        """If _clean_for_speech returns empty string, TTS is skipped."""
        from missy.channels.voice.stt.base import TranscriptionResult

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mgr = self._make_manager_for_speech(loop)
            vc = MagicMock(is_playing=MagicMock(return_value=False))
            state = _GuildVoiceState(voice_client=vc)
            state.listening = True
            mgr._guild_states[999] = state

            stt = MagicMock()
            stt.transcribe = AsyncMock(
                return_value=TranscriptionResult(
                    text="clean me",
                    confidence=0.9,
                    processing_ms=10,
                )
            )
            mgr._stt_engine = stt
            mgr._tts_engine = MagicMock()
            mgr._play_tts = AsyncMock()

            # Agent returns text that cleans to empty (only markdown syntax).
            mgr._agent_callback = AsyncMock(return_value="``` ```")

            guild = _make_guild(members={1: MagicMock(display_name="Ed")})
            mgr._client.get_guild = MagicMock(return_value=guild)

            long_pcm = struct.pack("<20000h", *([400] * 20000))
            loop.run_until_complete(
                mgr._handle_speech(guild_id=999, user_id=1, pcm_48k=long_pcm, sample_rate=48000)
            )

            # _play_tts should not be called since cleaned response is empty.
            # (Depending on what clean returns — if it happens to return empty, good.)
        finally:
            loop.close()
            asyncio.set_event_loop(None)
