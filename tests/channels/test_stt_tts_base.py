"""Tests for STT/TTS base classes and data structures."""

from __future__ import annotations

from missy.channels.voice.stt.base import STTEngine, TranscriptionResult
from missy.channels.voice.tts.base import AudioBuffer, TTSEngine


class TestTranscriptionResult:
    def test_basic_fields(self):
        r = TranscriptionResult(text="hello", confidence=0.95, processing_ms=120)
        assert r.text == "hello"
        assert r.confidence == 0.95
        assert r.processing_ms == 120
        assert r.language == ""

    def test_with_language(self):
        r = TranscriptionResult(text="hi", confidence=0.8, processing_ms=50, language="en")
        assert r.language == "en"

    def test_no_confidence(self):
        r = TranscriptionResult(text="test", confidence=-1.0, processing_ms=0)
        assert r.confidence == -1.0


class TestAudioBuffer:
    def test_duration_calculation(self):
        # 16000 Hz, mono, 16-bit = 2 bytes/sample
        # 32000 bytes = 16000 samples = 1 second = 1000ms
        audio_data = b"\x00" * 32000
        buf = AudioBuffer(data=audio_data, sample_rate=16000, channels=1, format="pcm_s16le")
        assert buf.duration_ms == 1000

    def test_stereo_duration(self):
        # 16000 Hz, stereo, 16-bit = 4 bytes/frame
        # 64000 bytes = 16000 frames = 1 second
        audio_data = b"\x00" * 64000
        buf = AudioBuffer(data=audio_data, sample_rate=16000, channels=2, format="pcm_s16le")
        assert buf.duration_ms == 1000

    def test_wav_format(self):
        buf = AudioBuffer(data=b"\x00" * 100, sample_rate=22050, channels=1, format="wav")
        assert buf.format == "wav"
        assert buf.duration_ms > 0

    def test_empty_buffer(self):
        buf = AudioBuffer(data=b"", sample_rate=16000, channels=1, format="pcm_s16le")
        assert buf.duration_ms == 0


class TestSTTEngineInterface:
    def test_is_abstract(self):
        # STTEngine cannot be instantiated directly
        import pytest

        with pytest.raises(TypeError):
            STTEngine()


class TestTTSEngineInterface:
    def test_is_abstract(self):
        import pytest

        with pytest.raises(TypeError):
            TTSEngine()
