"""Tests for missy.channels.voice.channel."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.voice.channel import VoiceChannel, _build_agent_callback


class TestBuildAgentCallback:
    def test_async_runtime_uses_run_async(self):
        """If runtime has run_async coroutine, use it directly."""
        runtime = MagicMock()
        runtime.run_async = AsyncMock(return_value="async result")
        cb = _build_agent_callback(runtime)
        assert asyncio.iscoroutinefunction(cb)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(cb("hello", "sess", {}))
        finally:
            loop.close()
        assert result == "async result"
        runtime.run_async.assert_awaited_once()

    def test_sync_runtime_wraps_in_executor(self):
        """If runtime only has sync run(), wrap it."""
        runtime = MagicMock()
        runtime.run.return_value = "sync result"
        # Remove run_async so it falls through
        del runtime.run_async
        cb = _build_agent_callback(runtime)
        assert asyncio.iscoroutinefunction(cb)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(cb("hello", "sess", {}))
        finally:
            loop.close()
        assert result == "sync result"


class TestVoiceChannelInit:
    @patch("missy.channels.voice.channel.FasterWhisperSTT")
    @patch("missy.channels.voice.channel.PiperTTS")
    @patch("missy.channels.voice.channel.DeviceRegistry")
    def test_init_creates_subsystems(self, mock_reg, mock_tts, mock_stt):
        mock_reg_inst = MagicMock()
        mock_reg.return_value = mock_reg_inst
        mock_stt.return_value = MagicMock()
        mock_tts.return_value = MagicMock()

        ch = VoiceChannel(
            host="127.0.0.1",
            port=9999,
            stt_model="tiny.en",
            tts_voice="test-voice",
        )
        assert ch is not None

    def test_receive_raises_not_implemented(self):
        with (
            patch("missy.channels.voice.channel.FasterWhisperSTT"),
            patch("missy.channels.voice.channel.PiperTTS"),
            patch("missy.channels.voice.channel.DeviceRegistry"),
        ):
            ch = VoiceChannel(host="0.0.0.0", port=8765)
            with pytest.raises(NotImplementedError):
                ch.receive()

    def test_send_raises_not_implemented(self):
        with (
            patch("missy.channels.voice.channel.FasterWhisperSTT"),
            patch("missy.channels.voice.channel.PiperTTS"),
            patch("missy.channels.voice.channel.DeviceRegistry"),
        ):
            ch = VoiceChannel(host="0.0.0.0", port=8765)
            msg = MagicMock()
            with pytest.raises(NotImplementedError):
                ch.send(msg)
