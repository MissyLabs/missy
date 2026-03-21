"""Tests for FasterWhisperSTT engine.

Tests lifecycle, device resolution, transcription, and audit event emission
using mocked faster_whisper and numpy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.voice.stt.base import STTEngine, TranscriptionResult

# ---------------------------------------------------------------------------
# Mock segment for Whisper output
# ---------------------------------------------------------------------------


@dataclass
class MockSegment:
    text: str = " Hello world."
    no_speech_prob: float = 0.05


@dataclass
class MockTranscribeInfo:
    language: str = "en"


# ---------------------------------------------------------------------------
# FasterWhisperSTT tests
# ---------------------------------------------------------------------------


class TestFasterWhisperSTTLifecycle:
    def test_load_raises_without_faster_whisper(self):
        with patch.dict("sys.modules", {"faster_whisper": None}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT()
            with pytest.raises(ImportError, match="faster-whisper"):
                engine.load()

    def test_load_success(self):
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
            engine.load()
            assert engine.is_loaded()
            mock_whisper_module.WhisperModel.assert_called_once_with(
                "tiny", device="cpu", compute_type="int8"
            )

    def test_load_idempotent(self):
        mock_whisper_module = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT(device="cpu", compute_type="int8")
            engine.load()
            engine.load()  # Should not raise or load twice
            assert mock_whisper_module.WhisperModel.call_count == 1

    def test_unload(self):
        mock_whisper_module = MagicMock()
        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT(device="cpu", compute_type="int8")
            engine.load()
            engine.unload()
            assert not engine.is_loaded()


class TestFasterWhisperSTTDeviceResolution:
    def test_auto_cpu_fallback(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT

        with patch.dict("sys.modules", {"torch": None, "ctranslate2": None}):
            engine = FasterWhisperSTT(device="auto")
            device = engine._detect_device()
            assert device == "cpu"

    def test_auto_compute_type_cpu(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT

        engine = FasterWhisperSTT(device="cpu", compute_type="auto")
        device, ct = engine._resolve_device_and_compute()
        assert device == "cpu"
        assert ct == "int8"

    def test_auto_compute_type_cuda(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT

        engine = FasterWhisperSTT(device="cuda", compute_type="auto")
        device, ct = engine._resolve_device_and_compute()
        assert device == "cuda"
        assert ct == "float16"

    def test_explicit_device_and_compute(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT

        engine = FasterWhisperSTT(device="cpu", compute_type="float32")
        device, ct = engine._resolve_device_and_compute()
        assert device == "cpu"
        assert ct == "float32"


class TestFasterWhisperSTTTranscribe:
    @pytest.mark.asyncio
    async def test_transcribe_raises_if_not_loaded(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT

        engine = FasterWhisperSTT()
        with pytest.raises(RuntimeError, match="load"):
            await engine.transcribe(b"\x00" * 100)

    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        pytest.importorskip("numpy")
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        segments = [MockSegment(text=" Hello world.")]
        info = MockTranscribeInfo(language="en")
        mock_model.transcribe.return_value = (iter(segments), info)
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT(device="cpu", compute_type="int8")
            engine.load()

            # Create valid PCM data (16-bit samples)
            import struct

            pcm = struct.pack("<100h", *([0] * 100))

            result = await engine.transcribe(pcm, sample_rate=16000)
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Hello world."
            assert result.language == "en"
            assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_transcribe_no_speech_confidence(self):
        pytest.importorskip("numpy")
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        segments = [MockSegment(text=" noise", no_speech_prob=0.9)]
        info = MockTranscribeInfo(language="en")
        mock_model.transcribe.return_value = (iter(segments), info)
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT(device="cpu", compute_type="int8")
            engine.load()

            import struct

            pcm = struct.pack("<100h", *([0] * 100))
            result = await engine.transcribe(pcm)
            assert result.confidence == pytest.approx(0.1, abs=0.01)

    @pytest.mark.asyncio
    async def test_transcribe_multichannel(self):
        pytest.importorskip("numpy")
        mock_whisper_module = MagicMock()
        mock_model = MagicMock()
        segments = [MockSegment(text=" stereo")]
        info = MockTranscribeInfo()
        mock_model.transcribe.return_value = (iter(segments), info)
        mock_whisper_module.WhisperModel.return_value = mock_model

        with patch.dict("sys.modules", {"faster_whisper": mock_whisper_module}):
            from missy.channels.voice.stt.whisper import FasterWhisperSTT

            engine = FasterWhisperSTT(device="cpu", compute_type="int8")
            engine.load()

            import struct

            pcm = struct.pack("<200h", *([0] * 200))
            result = await engine.transcribe(pcm, channels=2)
            assert result.text == "stereo"


class TestSTTEngineIsAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            STTEngine()

    def test_name_attribute(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT

        assert FasterWhisperSTT.name == "faster-whisper"
