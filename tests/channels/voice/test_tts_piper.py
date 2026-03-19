"""Tests for PiperTTS engine.

Tests lifecycle, model resolution, subprocess synthesis, PCM-to-WAV conversion,
and environment sanitization.
"""

from __future__ import annotations

import struct
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.voice.tts.base import AudioBuffer, TTSEngine
from missy.channels.voice.tts.piper import (
    PiperTTS,
    _pcm_to_wav,
    _piper_subprocess_env,
)

# ---------------------------------------------------------------------------
# _pcm_to_wav helper
# ---------------------------------------------------------------------------


class TestPcmToWav:
    def test_valid_wav_output(self):
        pcm = struct.pack("<100h", *([1000] * 100))
        wav_data = _pcm_to_wav(pcm, 22050, 1)
        # Should be parseable as WAV
        buf = BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 100

    def test_stereo_wav(self):
        pcm = struct.pack("<200h", *([500] * 200))
        wav_data = _pcm_to_wav(pcm, 44100, 2)
        buf = BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getnframes() == 100  # 200 samples / 2 channels

    def test_empty_pcm(self):
        wav_data = _pcm_to_wav(b"", 22050, 1)
        buf = BytesIO(wav_data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0


# ---------------------------------------------------------------------------
# _piper_subprocess_env
# ---------------------------------------------------------------------------


class TestPiperSubprocessEnv:
    def test_only_safe_vars(self):
        with patch.dict("os.environ", {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "DATABASE_URL": "postgres://secret",
        }, clear=True):
            env = _piper_subprocess_env()
            assert "PATH" in env
            assert "HOME" in env
            assert "ANTHROPIC_API_KEY" not in env
            assert "AWS_SECRET_ACCESS_KEY" not in env
            assert "DATABASE_URL" not in env

    def test_ld_library_path_set(self):
        with patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            env = _piper_subprocess_env()
            assert "LD_LIBRARY_PATH" in env


# ---------------------------------------------------------------------------
# PiperTTS lifecycle
# ---------------------------------------------------------------------------


class TestPiperTTSLifecycle:
    def test_load_raises_if_binary_not_found(self):
        with patch("shutil.which", return_value=None), patch.object(Path, "is_file", return_value=False):
            engine = PiperTTS(piper_bin="nonexistent_piper")
            with pytest.raises(RuntimeError, match="not found"):
                engine.load()

    def test_load_success_with_model(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")
        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()
            assert engine.is_loaded()

    def test_load_idempotent(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")
        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()
            engine.load()  # Should not raise
            assert engine.is_loaded()

    def test_unload(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")
        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()
            engine.unload()
            assert not engine.is_loaded()

    def test_load_model_not_found(self):
        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path="/nonexistent/path/model.onnx")
            with pytest.raises(RuntimeError, match="not found"):
                engine.load()


# ---------------------------------------------------------------------------
# PiperTTS model resolution
# ---------------------------------------------------------------------------


class TestPiperTTSModelResolution:
    def test_resolve_from_voices_dir(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        model = voices_dir / "test-voice.onnx"
        model.write_bytes(b"model data")

        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", voices_dir), patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(voice="test-voice")
            engine.load()
            assert engine.is_loaded()
            assert engine._model_file == model

    def test_resolve_model_not_in_voices_dir(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", voices_dir), patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(voice="nonexistent-voice")
            with pytest.raises(RuntimeError, match="not found"):
                engine.load()


# ---------------------------------------------------------------------------
# PiperTTS list_voices
# ---------------------------------------------------------------------------


class TestPiperTTSListVoices:
    def test_list_voices_from_dir(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        (voices_dir / "en_US-lessac-medium.onnx").write_bytes(b"m1")
        (voices_dir / "de_DE-thorsten-high.onnx").write_bytes(b"m2")
        (voices_dir / "not-a-model.json").write_bytes(b"skip")

        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", voices_dir):
            engine = PiperTTS()
            voices = engine.list_voices()
            assert "en_US-lessac-medium" in voices
            assert "de_DE-thorsten-high" in voices
            assert "not-a-model" not in voices
            assert voices == sorted(voices)

    def test_list_voices_no_dir(self, tmp_path):
        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path / "nope"):
            engine = PiperTTS()
            assert engine.list_voices() == []


# ---------------------------------------------------------------------------
# PiperTTS synthesis
# ---------------------------------------------------------------------------


class TestPiperTTSSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_raises_if_not_loaded(self):
        engine = PiperTTS()
        with pytest.raises(RuntimeError, match="load"):
            await engine.synthesize("hello")

    @pytest.mark.asyncio
    async def test_synthesize_success(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")

        # Mock subprocess to produce PCM audio
        pcm_output = struct.pack("<1000h", *([500] * 1000))

        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(pcm_output, b""))
            mock_proc.returncode = 0

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                buf = await engine.synthesize("Hello world")
                assert isinstance(buf, AudioBuffer)
                assert buf.format == "wav"
                assert buf.sample_rate == 22050
                assert len(buf.data) > 0

    @pytest.mark.asyncio
    async def test_synthesize_piper_error(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")

        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: model invalid"))
            mock_proc.returncode = 1

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc), pytest.raises(RuntimeError, match="code 1"):
                await engine.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_empty_output(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")

        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc), pytest.raises(RuntimeError, match="no audio"):
                await engine.synthesize("Hello")

    @pytest.mark.asyncio
    async def test_synthesize_timeout(self, tmp_path):
        model_file = tmp_path / "voice.onnx"
        model_file.write_bytes(b"fake model")

        with patch("shutil.which", return_value="/usr/bin/piper"):
            engine = PiperTTS(model_path=str(model_file))
            engine.load()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=TimeoutError())
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc), patch("asyncio.wait_for", side_effect=TimeoutError()), pytest.raises(RuntimeError, match="timed out"):
                await engine.synthesize("Hello")


# ---------------------------------------------------------------------------
# TTSEngine abstract base class
# ---------------------------------------------------------------------------


class TestTTSEngineAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            TTSEngine()

    def test_audio_buffer_duration(self):
        # 22050 samples at 22050 Hz = 1 second = 1000ms
        data = b"\x00\x00" * 22050
        buf = AudioBuffer(data=data, sample_rate=22050, channels=1, format="pcm_s16le")
        assert buf.duration_ms == 1000

    def test_audio_buffer_stereo_duration(self):
        # 22050 stereo frames = 44100 samples = 1 second
        data = b"\x00\x00" * 44100
        buf = AudioBuffer(data=data, sample_rate=22050, channels=2, format="pcm_s16le")
        assert buf.duration_ms == 1000
