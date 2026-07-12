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


class TestBuildAgentCallbackSafeChatRouting:
    """Regression: policy_mode was read in exactly one place in the whole
    voice subsystem (the "muted" check at auth time) -- a node explicitly
    configured with `missy devices policy <id> --mode safe-chat` still got
    full, unrestricted tool access identical to "full" mode, since nothing
    ever routed it to a capability-restricted runtime. _build_agent_callback()
    must dispatch to a dedicated safe-chat runtime when metadata declares
    policy_mode="safe-chat", and fail closed (refuse, not fall back to full
    access) when no such runtime is configured.
    """

    def _run(self, cb, prompt="hello", session_id="s1", metadata=None):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(cb(prompt, session_id, metadata or {}))
        finally:
            loop.close()

    def test_full_policy_mode_uses_default_runtime(self):
        full_runtime = MagicMock()
        full_runtime.run.return_value = "full-response"
        del full_runtime.run_async
        safe_runtime = MagicMock()
        safe_runtime.run.return_value = "safe-response"
        del safe_runtime.run_async

        cb = _build_agent_callback(full_runtime, safe_runtime)
        result = self._run(cb, metadata={"policy_mode": "full"})

        assert result == "full-response"
        full_runtime.run.assert_called_once()
        safe_runtime.run.assert_not_called()

    def test_safe_chat_policy_mode_uses_restricted_runtime(self):
        full_runtime = MagicMock()
        full_runtime.run.return_value = "full-response"
        del full_runtime.run_async
        safe_runtime = MagicMock()
        safe_runtime.run.return_value = "safe-response"
        del safe_runtime.run_async

        cb = _build_agent_callback(full_runtime, safe_runtime)
        result = self._run(cb, metadata={"policy_mode": "safe-chat"})

        assert result == "safe-response"
        full_runtime.run.assert_not_called()
        safe_runtime.run.assert_called_once()

    def test_safe_chat_without_restricted_runtime_fails_closed(self):
        """No safe-chat runtime configured must refuse the request outright
        rather than silently falling back to the full-access runtime.
        """
        full_runtime = MagicMock()
        full_runtime.run.return_value = "full-response"
        del full_runtime.run_async

        cb = _build_agent_callback(full_runtime)  # no safe_chat_agent_runtime
        result = self._run(cb, metadata={"policy_mode": "safe-chat"})

        assert "restricted" in result.lower() or "safe-chat" in result.lower()
        full_runtime.run.assert_not_called()


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
