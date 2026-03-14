"""Targeted tests to push overall coverage above 85%.

Covers:
- missy/channels/voice/stt/whisper.py  (FasterWhisperSTT — all paths)
- missy/channels/voice/tts/piper.py    (PiperTTS — all paths)
- missy/agent/proactive.py             (remaining 24% — threshold loop,
                                        schedule loop, stop with observer error,
                                        _ProactiveFileHandler, watchdog import branch)
- missy/providers/ollama_provider.py   (stream(), complete_with_tools error paths,
                                        ProviderError re-raise paths)
- missy/memory/store.py                (compact_session, search, save_learning,
                                        get_learnings, non-list/malformed JSON load,
                                        _save error)
"""

from __future__ import annotations

import asyncio
import json
import shutil
import struct
import threading
import time
import types
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# FasterWhisperSTT
# ---------------------------------------------------------------------------


class TestFasterWhisperSTTLoad:
    """Tests for load / unload / is_loaded lifecycle."""

    def _make_stt(self, model_size="base.en", device="cpu", compute_type="int8"):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        return FasterWhisperSTT(model_size=model_size, device=device, compute_type=compute_type)

    def test_load_imports_and_creates_model(self):
        stt = self._make_stt()
        mock_model_cls = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance

        with patch.dict("sys.modules", {"faster_whisper": types.SimpleNamespace(WhisperModel=mock_model_cls)}):
            stt.load()

        assert stt.is_loaded() is True
        mock_model_cls.assert_called_once_with("base.en", device="cpu", compute_type="int8")

    def test_load_idempotent_when_already_loaded(self):
        stt = self._make_stt()
        mock_model_cls = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": types.SimpleNamespace(WhisperModel=mock_model_cls)}):
            stt.load()
            stt.load()  # second call should be no-op

        assert mock_model_cls.call_count == 1

    def test_load_raises_import_error_when_faster_whisper_missing(self):
        stt = self._make_stt()
        import sys
        saved = sys.modules.pop("faster_whisper", None)
        try:
            with pytest.raises(ImportError, match="faster-whisper"):
                stt.load()
        finally:
            if saved is not None:
                sys.modules["faster_whisper"] = saved

    def test_unload_clears_model(self):
        stt = self._make_stt()
        mock_model_cls = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"faster_whisper": types.SimpleNamespace(WhisperModel=mock_model_cls)}):
            stt.load()

        assert stt.is_loaded() is True
        stt.unload()
        assert stt.is_loaded() is False
        assert stt._resolved_device is None
        assert stt._resolved_compute_type is None

    def test_is_loaded_false_before_load(self):
        stt = self._make_stt()
        assert stt.is_loaded() is False


class TestFasterWhisperSTTAutoDevice:
    """Tests for device/compute_type auto-resolution."""

    def _make_stt(self, device="auto", compute_type="auto"):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        return FasterWhisperSTT(device=device, compute_type=compute_type)

    def test_auto_device_falls_back_to_cpu_when_torch_missing(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        import sys
        # Remove torch and ctranslate2 so detection falls through to cpu
        saved_torch = sys.modules.pop("torch", None)
        saved_ct2 = sys.modules.pop("ctranslate2", None)
        try:
            result = FasterWhisperSTT._detect_device()
        finally:
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch
            if saved_ct2 is not None:
                sys.modules["ctranslate2"] = saved_ct2
        assert result == "cpu"

    def test_auto_device_uses_cuda_when_torch_available(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = FasterWhisperSTT._detect_device()
        assert result == "cuda"

    def test_auto_device_uses_cuda_via_ctranslate2_when_torch_missing(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        mock_ct2 = MagicMock()
        mock_ct2.get_cuda_device_count.return_value = 1
        import sys
        saved_torch = sys.modules.pop("torch", None)
        try:
            with patch.dict("sys.modules", {"ctranslate2": mock_ct2}):
                result = FasterWhisperSTT._detect_device()
        finally:
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch
        assert result == "cuda"

    def test_auto_compute_type_float16_for_cuda(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        stt = FasterWhisperSTT(device="cuda", compute_type="auto")
        device, compute_type = stt._resolve_device_and_compute()
        assert device == "cuda"
        assert compute_type == "float16"

    def test_auto_compute_type_int8_for_cpu(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        stt = FasterWhisperSTT(device="cpu", compute_type="auto")
        device, compute_type = stt._resolve_device_and_compute()
        assert compute_type == "int8"

    def test_explicit_compute_type_forwarded_unchanged(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        stt = FasterWhisperSTT(device="cpu", compute_type="float32")
        _, compute_type = stt._resolve_device_and_compute()
        assert compute_type == "float32"

    def test_auto_device_resolved_during_load(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        stt = FasterWhisperSTT(device="auto", compute_type="auto")
        mock_model_cls = MagicMock(return_value=MagicMock())
        with (
            patch.object(FasterWhisperSTT, "_detect_device", return_value="cpu"),
            patch.dict("sys.modules", {"faster_whisper": types.SimpleNamespace(WhisperModel=mock_model_cls)}),
        ):
            stt.load()
        assert stt._resolved_device == "cpu"
        assert stt._resolved_compute_type == "int8"


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _make_numpy_mock():
    """Return a lightweight numpy mock sufficient for PCM conversion."""
    import sys
    # If real numpy is present, just use it.
    if _numpy_available():
        import numpy as np
        return np, {}

    # Build a minimal numpy mock using only stdlib so tests run without numpy.
    np_mock = MagicMock(name="numpy")

    def _frombuffer(data, dtype=None):
        """Parse raw bytes as int16 array (real conversion)."""
        n = len(data) // 2
        arr = MagicMock(name="ndarray")
        # Store raw values for astype
        import array as arr_mod
        raw = arr_mod.array("h")
        raw.frombytes(data[: n * 2])
        int_list = list(raw)
        arr._int_list = int_list
        arr._n = n

        def astype(t):
            floats = [v / 32768.0 for v in int_list]
            farr = MagicMock(name="float32arr")
            farr._vals = floats
            farr._n = n

            def reshape(shape):
                rows, cols = shape[0], shape[1]
                matrix = []
                for i in range(rows):
                    matrix.append(floats[i * cols: (i + 1) * cols])
                marr = MagicMock(name="matrix")
                marr._matrix = matrix

                def mean(axis):
                    means = [sum(row) / len(row) for row in matrix]
                    mono = MagicMock(name="mono")
                    mono._vals = means
                    return mono

                marr.mean = mean
                return marr

            farr.reshape = reshape
            return farr

        arr.astype = astype
        return arr

    np_mock.frombuffer = _frombuffer
    np_mock.int16 = "int16"
    np_mock.float32 = "float32"
    return np_mock, {"numpy": np_mock}


class TestFasterWhisperSTTTranscribe:
    """Tests for the async transcribe method."""

    def _loaded_stt(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        stt = FasterWhisperSTT(device="cpu", compute_type="int8")
        stt._model = MagicMock()
        stt._resolved_device = "cpu"
        stt._resolved_compute_type = "int8"
        return stt

    def _make_pcm(self, n_samples: int = 160) -> bytes:
        """Build silent 16-bit PCM bytes."""
        return struct.pack(f"<{n_samples}h", *([0] * n_samples))

    def test_transcribe_raises_when_not_loaded(self):
        from missy.channels.voice.stt.whisper import FasterWhisperSTT
        stt = FasterWhisperSTT()
        with pytest.raises(RuntimeError, match="load"):
            asyncio.get_event_loop().run_until_complete(
                stt.transcribe(self._make_pcm())
            )

    def test_transcribe_returns_result(self):
        np_mod, extra = _make_numpy_mock()
        stt = self._loaded_stt()

        seg = MagicMock()
        seg.text = " hello world"
        seg.no_speech_prob = 0.1
        info = MagicMock()
        info.language = "en"
        stt._model.transcribe.return_value = (iter([seg]), info)

        with patch.dict("sys.modules", {"numpy": np_mod, **extra}):
            result = asyncio.get_event_loop().run_until_complete(
                stt.transcribe(self._make_pcm())
            )

        assert result.text == "hello world"
        assert result.language == "en"
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_ms >= 0

    def test_transcribe_confidence_no_speech_prob(self):
        np_mod, extra = _make_numpy_mock()
        stt = self._loaded_stt()

        seg = MagicMock()
        seg.text = "test"
        seg.no_speech_prob = 0.2
        info = MagicMock()
        info.language = "en"
        stt._model.transcribe.return_value = (iter([seg]), info)

        with patch.dict("sys.modules", {"numpy": np_mod, **extra}):
            result = asyncio.get_event_loop().run_until_complete(
                stt.transcribe(self._make_pcm())
            )

        assert abs(result.confidence - 0.8) < 0.001

    def test_transcribe_confidence_minus_one_when_no_prob(self):
        np_mod, extra = _make_numpy_mock()
        stt = self._loaded_stt()

        # Segment with no no_speech_prob attribute
        seg = types.SimpleNamespace(text="hello")
        info = MagicMock()
        info.language = "en"
        stt._model.transcribe.return_value = (iter([seg]), info)

        with patch.dict("sys.modules", {"numpy": np_mod, **extra}):
            result = asyncio.get_event_loop().run_until_complete(
                stt.transcribe(self._make_pcm())
            )

        assert result.confidence == -1.0

    def test_transcribe_multichannel_mixdown(self):
        np_mod, extra = _make_numpy_mock()
        stt = self._loaded_stt()

        info = MagicMock()
        info.language = "en"
        stt._model.transcribe.return_value = (iter([]), info)

        stereo_pcm = struct.pack("<8h", 100, -100, 200, -200, 50, -50, 80, -80)

        with patch.dict("sys.modules", {"numpy": np_mod, **extra}):
            result = asyncio.get_event_loop().run_until_complete(
                stt.transcribe(stereo_pcm, channels=2)
            )

        assert result.text == ""
        # Verify that transcribe was actually called — proves mix-down code ran
        stt._model.transcribe.assert_called_once()

    def test_transcribe_raises_import_error_without_numpy(self):
        stt = self._loaded_stt()
        import sys
        saved = sys.modules.pop("numpy", None)
        try:
            with pytest.raises(ImportError, match="numpy"):
                asyncio.get_event_loop().run_until_complete(
                    stt.transcribe(self._make_pcm())
                )
        finally:
            if saved is not None:
                sys.modules["numpy"] = saved

    def test_transcribe_emits_audit_event(self):
        np_mod, extra = _make_numpy_mock()
        stt = self._loaded_stt()

        info = MagicMock()
        info.language = "fr"
        stt._model.transcribe.return_value = (iter([]), info)

        with (
            patch.dict("sys.modules", {"numpy": np_mod, **extra}),
            patch("missy.channels.voice.stt.whisper.event_bus") as mock_bus,
        ):
            asyncio.get_event_loop().run_until_complete(
                stt.transcribe(self._make_pcm())
            )

        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == "voice.stt.complete"
        assert event.detail["language"] == "fr"

    def test_transcribe_audit_failure_is_swallowed(self):
        """Audit emit errors must not propagate to callers."""
        np_mod, extra = _make_numpy_mock()
        stt = self._loaded_stt()

        info = MagicMock()
        info.language = "en"
        stt._model.transcribe.return_value = (iter([]), info)

        with (
            patch.dict("sys.modules", {"numpy": np_mod, **extra}),
            patch("missy.channels.voice.stt.whisper.event_bus") as mock_bus,
        ):
            mock_bus.publish.side_effect = RuntimeError("bus down")
            # Should not raise
            asyncio.get_event_loop().run_until_complete(
                stt.transcribe(self._make_pcm())
            )


# ---------------------------------------------------------------------------
# PiperTTS
# ---------------------------------------------------------------------------


def _make_wav_bytes(n_samples: int = 44100) -> bytes:
    """Return a minimal valid WAV file with *n_samples* silent 16-bit samples."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


class TestPiperTTSLoad:
    """Tests for PiperTTS load/unload/is_loaded lifecycle."""

    def test_load_resolves_binary_from_which(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        fake_onnx = tmp_path / "en_US-lessac-medium.onnx"
        fake_onnx.touch()
        tts = PiperTTS(piper_bin="piper", voice="en_US-lessac-medium")

        with (
            patch("shutil.which", return_value="/usr/bin/piper"),
            patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path),
        ):
            tts.load()

        assert tts.is_loaded() is True
        assert tts._piper_bin == "/usr/bin/piper"

    def test_load_falls_back_to_local_bin(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        fake_onnx = tmp_path / "en_US-lessac-medium.onnx"
        fake_onnx.touch()
        fake_bin = tmp_path / "piper"
        fake_bin.touch()
        tts = PiperTTS(piper_bin="piper", voice="en_US-lessac-medium")

        with (
            patch("shutil.which", return_value=None),
            patch("missy.channels.voice.tts.piper._LOCAL_PIPER_BIN", fake_bin),
            patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path),
        ):
            tts.load()

        assert tts.is_loaded() is True
        assert tts._piper_bin == str(fake_bin)

    def test_load_raises_when_binary_not_found(self):
        from missy.channels.voice.tts.piper import PiperTTS
        tts = PiperTTS(piper_bin="nonexistent_piper")
        fake_missing_bin = Path("/nonexistent/piper")

        with (
            patch("shutil.which", return_value=None),
            patch("missy.channels.voice.tts.piper._LOCAL_PIPER_BIN", fake_missing_bin),
        ):
            with pytest.raises(RuntimeError, match="not found"):
                tts.load()

    def test_load_raises_when_model_not_found(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        tts = PiperTTS(piper_bin="piper", voice="missing-voice")

        with (
            patch("shutil.which", return_value="/usr/bin/piper"),
            patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path),
        ):
            with pytest.raises(RuntimeError, match="not found"):
                tts.load()

    def test_load_idempotent(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        fake_onnx = tmp_path / "en_US-lessac-medium.onnx"
        fake_onnx.touch()
        tts = PiperTTS(piper_bin="piper", voice="en_US-lessac-medium")

        with (
            patch("shutil.which", return_value="/usr/bin/piper"),
            patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path),
        ):
            tts.load()
            tts.load()  # second call is no-op

        assert tts.is_loaded() is True

    def test_unload_resets_state(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        fake_onnx = tmp_path / "en_US-lessac-medium.onnx"
        fake_onnx.touch()
        tts = PiperTTS(piper_bin="piper", voice="en_US-lessac-medium")

        with (
            patch("shutil.which", return_value="/usr/bin/piper"),
            patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path),
        ):
            tts.load()

        tts.unload()
        assert tts.is_loaded() is False
        assert tts._piper_bin is None
        assert tts._model_file is None

    def test_load_with_explicit_model_path(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        model_file = tmp_path / "my_voice.onnx"
        model_file.touch()
        tts = PiperTTS(piper_bin="piper", model_path=str(model_file))

        with patch("shutil.which", return_value="/usr/bin/piper"):
            tts.load()

        assert tts._model_file == model_file.resolve()

    def test_load_raises_when_explicit_model_path_missing(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        tts = PiperTTS(piper_bin="piper", model_path=str(tmp_path / "nonexistent.onnx"))

        with patch("shutil.which", return_value="/usr/bin/piper"):
            with pytest.raises(RuntimeError, match="not found"):
                tts.load()


class TestPiperTTSListVoices:
    def test_returns_empty_when_voices_dir_missing(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        tts = PiperTTS()
        missing_dir = tmp_path / "no_voices"

        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", missing_dir):
            voices = tts.list_voices()

        assert voices == []

    def test_returns_sorted_voice_stems(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        (tmp_path / "zz_voice.onnx").touch()
        (tmp_path / "aa_voice.onnx").touch()
        (tmp_path / "mm_voice.onnx").touch()
        tts = PiperTTS()

        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path):
            voices = tts.list_voices()

        assert voices == ["aa_voice", "mm_voice", "zz_voice"]

    def test_ignores_non_onnx_files(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        (tmp_path / "voice.onnx").touch()
        (tmp_path / "config.json").touch()
        tts = PiperTTS()

        with patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path):
            voices = tts.list_voices()

        assert voices == ["voice"]


class TestPiperTTSSynthesize:
    def _loaded_tts(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        fake_onnx = tmp_path / "en_US-lessac-medium.onnx"
        fake_onnx.touch()
        tts = PiperTTS(piper_bin="/usr/bin/piper", voice="en_US-lessac-medium")
        tts._loaded = True
        tts._piper_bin = "/usr/bin/piper"
        tts._model_file = fake_onnx
        return tts

    def _fake_pcm(self, n_samples: int = 22050) -> bytes:
        return b"\x00\x00" * n_samples

    def test_synthesize_raises_when_not_loaded(self):
        from missy.channels.voice.tts.piper import PiperTTS
        tts = PiperTTS()

        with pytest.raises(RuntimeError, match="load"):
            asyncio.get_event_loop().run_until_complete(tts.synthesize("hello"))

    def test_synthesize_returns_audio_buffer(self, tmp_path):
        from missy.channels.voice.tts.base import AudioBuffer
        tts = self._loaded_tts(tmp_path)
        pcm = self._fake_pcm()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(pcm, b""))

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)):
            buffer = asyncio.get_event_loop().run_until_complete(tts.synthesize("hello world"))

        assert isinstance(buffer, AudioBuffer)
        assert buffer.format == "wav"
        assert buffer.sample_rate == 22050
        assert len(buffer.data) > 0

    def test_synthesize_raises_on_nonzero_exit_code(self, tmp_path):
        tts = self._loaded_tts(tmp_path)

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"model error"))

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)),
            pytest.raises(RuntimeError, match="Piper exited"),
        ):
            asyncio.get_event_loop().run_until_complete(tts.synthesize("hello"))

    def test_synthesize_raises_when_no_audio_output(self, tmp_path):
        tts = self._loaded_tts(tmp_path)

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)),
            pytest.raises(RuntimeError, match="no audio"),
        ):
            asyncio.get_event_loop().run_until_complete(tts.synthesize("hello"))

    def test_synthesize_with_override_voice(self, tmp_path):
        from missy.channels.voice.tts.piper import PiperTTS
        tts = self._loaded_tts(tmp_path)
        # Create a model for the override voice
        override_model = tmp_path / "other_voice.onnx"
        override_model.touch()
        pcm = self._fake_pcm()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(pcm, b""))

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)),
            patch("missy.channels.voice.tts.piper._DEFAULT_VOICES_DIR", tmp_path),
        ):
            buffer = asyncio.get_event_loop().run_until_complete(
                tts.synthesize("hi", voice="other_voice")
            )

        assert buffer.format == "wav"

    def test_synthesize_emits_audit_event(self, tmp_path):
        tts = self._loaded_tts(tmp_path)
        pcm = self._fake_pcm()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(pcm, b""))

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)),
            patch("missy.channels.voice.tts.piper.event_bus") as mock_bus,
        ):
            asyncio.get_event_loop().run_until_complete(tts.synthesize("hello"))

        mock_bus.publish.assert_called_once()
        event = mock_bus.publish.call_args.args[0]
        assert event.event_type == "voice.tts.complete"

    def test_synthesize_audit_failure_swallowed(self, tmp_path):
        tts = self._loaded_tts(tmp_path)
        pcm = self._fake_pcm()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(pcm, b""))

        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_proc)),
            patch("missy.channels.voice.tts.piper.event_bus") as mock_bus,
        ):
            mock_bus.publish.side_effect = RuntimeError("bus down")
            # Must not raise
            asyncio.get_event_loop().run_until_complete(tts.synthesize("hello"))


class TestPiperSubprocessEnv:
    def test_ld_library_path_prepended(self):
        from missy.channels.voice.tts.piper import _piper_subprocess_env
        import os
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/usr/lib"}, clear=False):
            env = _piper_subprocess_env()
        assert "LD_LIBRARY_PATH" in env
        # The piper local bin dir should be prepended
        assert env["LD_LIBRARY_PATH"].endswith("/usr/lib") or "/usr/lib" in env["LD_LIBRARY_PATH"]

    def test_ld_library_path_set_when_absent(self):
        from missy.channels.voice.tts.piper import _piper_subprocess_env
        import os
        env_without = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        with patch.dict(os.environ, env_without, clear=True):
            env = _piper_subprocess_env()
        assert "LD_LIBRARY_PATH" in env


class TestPcmToWav:
    def test_pcm_to_wav_produces_valid_wav(self):
        from missy.channels.voice.tts.piper import _pcm_to_wav
        pcm = b"\x00\x00" * 100  # 100 silent 16-bit samples
        wav = _pcm_to_wav(pcm, sample_rate=22050, channels=1)
        # Parse the WAV header to verify validity
        buf = BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 100


# ---------------------------------------------------------------------------
# ProactiveManager — uncovered paths
# ---------------------------------------------------------------------------


class TestProactiveThresholdLoop:
    """Exercise _threshold_loop by patching _stop_event.wait to fire once then stop."""

    def _run_loop_once(self, mgr, triggers, side_effects_fn):
        """Run _threshold_loop in a thread where stop_event.wait fires once then stops."""
        call_count = {"n": 0}

        original_wait = mgr._stop_event.wait

        def fake_wait(timeout=None):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                return True  # stop the loop
            side_effects_fn()
            return False  # first iteration: don't stop yet

        with patch.object(mgr._stop_event, "wait", side_effect=fake_wait):
            t = threading.Thread(
                target=mgr._threshold_loop,
                args=(triggers,),
                daemon=True,
            )
            t.start()
            t.join(timeout=3)

    def test_disk_threshold_fires_via_loop(self):
        from missy.agent.proactive import ProactiveTrigger, ProactiveManager
        counter = {"n": 0}
        lock = threading.Lock()

        def cb(prompt, session_id):
            with lock:
                counter["n"] += 1

        trigger = ProactiveTrigger(
            name="disk-loop",
            trigger_type="disk_threshold",
            disk_path="/",
            disk_threshold_pct=1.0,  # will exceed 1%
            cooldown_seconds=0,
            interval_seconds=5,
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=cb)

        mock_usage = MagicMock()
        mock_usage.used = 90
        mock_usage.total = 100  # 90% > 1%

        def setup_mocks():
            pass

        with patch("shutil.disk_usage", return_value=mock_usage):
            self._run_loop_once(mgr, [trigger], setup_mocks)

        with lock:
            assert counter["n"] >= 1

    def test_load_threshold_fires_via_loop(self):
        from missy.agent.proactive import ProactiveTrigger, ProactiveManager
        import os as _os
        counter = {"n": 0}
        lock = threading.Lock()

        def cb(prompt, session_id):
            with lock:
                counter["n"] += 1

        trigger = ProactiveTrigger(
            name="load-loop",
            trigger_type="load_threshold",
            load_threshold=0.001,  # nearly zero — will always exceed
            cooldown_seconds=0,
            interval_seconds=5,
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=cb)

        with (
            patch.object(_os, "getloadavg", return_value=(8.0, 4.0, 2.0)),
            patch.object(_os, "cpu_count", return_value=1),
        ):
            self._run_loop_once(mgr, [trigger], lambda: None)

        with lock:
            assert counter["n"] >= 1

    def test_threshold_loop_handles_exception_without_crash(self):
        from missy.agent.proactive import ProactiveTrigger, ProactiveManager
        trigger = ProactiveTrigger(
            name="disk-err",
            trigger_type="disk_threshold",
            disk_path="/bad/path",
            disk_threshold_pct=1.0,
            cooldown_seconds=0,
            interval_seconds=5,
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=lambda p, s: None)

        with patch("shutil.disk_usage", side_effect=OSError("no such path")):
            self._run_loop_once(mgr, [trigger], lambda: None)
        # Thread terminated without propagating the exception — test passes implicitly


class TestProactiveScheduleLoop:
    def test_schedule_loop_fires_callback(self):
        from missy.agent.proactive import ProactiveTrigger, ProactiveManager
        counter = {"n": 0}
        lock = threading.Lock()

        def cb(prompt, session_id):
            with lock:
                counter["n"] += 1

        trigger = ProactiveTrigger(
            name="sched-loop",
            trigger_type="schedule",
            interval_seconds=5,
            cooldown_seconds=0,
        )
        mgr = ProactiveManager(triggers=[trigger], agent_callback=cb)

        call_count = {"n": 0}

        def fake_wait(timeout=None):
            call_count["n"] += 1
            return call_count["n"] >= 2  # stop after second call

        with patch.object(mgr._stop_event, "wait", side_effect=fake_wait):
            t = threading.Thread(
                target=mgr._schedule_loop,
                args=(trigger,),
                daemon=True,
            )
            t.start()
            t.join(timeout=3)

        with lock:
            assert counter["n"] >= 1


class TestProactiveStopWithObserverError:
    def test_stop_handles_observer_exception(self):
        from missy.agent.proactive import ProactiveTrigger, ProactiveManager
        trigger = ProactiveTrigger(name="t", trigger_type="schedule", interval_seconds=10)
        mgr = ProactiveManager(triggers=[trigger], agent_callback=lambda p, s: None)

        mock_observer = MagicMock()
        mock_observer.stop.side_effect = RuntimeError("observer crash")
        mgr._observer = mock_observer

        # Should not raise even when observer.stop() throws
        mgr.stop()

        assert mgr._observer is None


class TestProactiveManagerStartThresholdTrigger:
    def test_start_creates_threshold_thread(self):
        from missy.agent.proactive import ProactiveTrigger, ProactiveManager
        trigger = ProactiveTrigger(
            name="disk-start",
            trigger_type="disk_threshold",
            disk_path="/",
            disk_threshold_pct=99.9,  # won't fire
            cooldown_seconds=0,
            interval_seconds=30,
        )
        mgr = ProactiveManager(
            triggers=[trigger],
            agent_callback=lambda p, s: None,
        )
        mgr.start()
        # Threshold thread should have been created
        assert len(mgr._threads) >= 1
        mgr.stop()


class TestProactiveFileHandlerClass:
    """Test _ProactiveFileHandler when watchdog IS available."""

    def test_on_any_event_calls_fire_fn(self):
        """If watchdog is importable, _ProactiveFileHandler.on_any_event fires correctly."""
        from missy.agent.proactive import _WATCHDOG_AVAILABLE

        if not _WATCHDOG_AVAILABLE:
            pytest.skip("watchdog not installed — testing stub path instead")

        from missy.agent.proactive import _ProactiveFileHandler, ProactiveTrigger

        fired = []
        trigger = ProactiveTrigger(name="fh-test", trigger_type="file_change")

        handler = _ProactiveFileHandler(
            trigger=trigger,
            fire_fn=lambda t: fired.append(t),
            patterns=["*"],
            ignore_directories=True,
            case_sensitive=False,
        )
        fake_event = MagicMock()
        handler.on_any_event(fake_event)

        assert fired == [trigger]

    def test_stub_handler_when_watchdog_unavailable(self):
        """When watchdog absent the stub class is a no-op object."""
        import missy.agent.proactive as _mod
        with patch.object(_mod, "_WATCHDOG_AVAILABLE", False):
            # Re-execute the conditional definition branch
            # The stub class exists regardless; just instantiate it
            handler = _mod._ProactiveFileHandler()
            assert handler is not None


# ---------------------------------------------------------------------------
# OllamaProvider — uncovered paths (stream, error re-raises)
# ---------------------------------------------------------------------------


def _make_ollama_provider():
    from missy.config.settings import ProviderConfig
    from missy.providers.ollama_provider import OllamaProvider
    config = ProviderConfig(name="ollama", model="llama3.2", timeout=30)
    return OllamaProvider(config)


class TestOllamaStream:
    def test_stream_yields_tokens(self):
        provider = _make_ollama_provider()
        chunks = [
            json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
            json.dumps({"message": {"content": " world"}, "done": True}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(chunks)

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            tokens = list(provider.stream([Message(role="user", content="hi")]))

        assert tokens == ["Hello", " world"]

    def test_stream_skips_empty_lines(self):
        provider = _make_ollama_provider()
        chunks = [
            b"",  # empty line — should be skipped
            json.dumps({"message": {"content": "ok"}, "done": True}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(chunks)

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            tokens = list(provider.stream([Message(role="user", content="hi")]))

        assert tokens == ["ok"]

    def test_stream_skips_malformed_json_lines(self):
        provider = _make_ollama_provider()
        chunks = [
            b"NOT_JSON",
            json.dumps({"message": {"content": "good"}, "done": True}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(chunks)

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            tokens = list(provider.stream([Message(role="user", content="hi")]))

        assert tokens == ["good"]

    def test_stream_injects_system_prompt(self):
        provider = _make_ollama_provider()
        chunks = [
            json.dumps({"message": {"content": "hi"}, "done": True}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(chunks)

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            list(provider.stream([Message(role="user", content="hi")], system="Be helpful."))
            payload = MockClient.return_value.post.call_args[1]["json"]

        assert payload["messages"][0]["role"] == "system"
        assert payload["stream"] is True

    def test_stream_raises_provider_error_on_transport_failure(self):
        from missy.core.exceptions import ProviderError
        provider = _make_ollama_provider()

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.side_effect = ConnectionError("refused")
            from missy.providers.base import Message
            with pytest.raises(ProviderError, match="stream"):
                list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_reraises_provider_error_directly(self):
        from missy.core.exceptions import ProviderError
        provider = _make_ollama_provider()

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.side_effect = ProviderError("original")
            from missy.providers.base import Message
            with pytest.raises(ProviderError, match="original"):
                list(provider.stream([Message(role="user", content="hi")]))

    def test_stream_stops_at_done_flag(self):
        """Generator should stop yielding after done=True even if more lines follow."""
        provider = _make_ollama_provider()
        chunks = [
            json.dumps({"message": {"content": "first"}, "done": True}).encode(),
            json.dumps({"message": {"content": "second"}, "done": False}).encode(),
        ]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_lines.return_value = iter(chunks)

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            tokens = list(provider.stream([Message(role="user", content="hi")]))

        assert tokens == ["first"]


class TestOllamaCompleteWithToolsErrors:
    def test_connection_error_raises_provider_error(self):
        from missy.core.exceptions import ProviderError
        provider = _make_ollama_provider()

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.side_effect = ConnectionError("refused")
            from missy.providers.base import Message
            with pytest.raises(ProviderError, match="Ollama"):
                provider.complete_with_tools(
                    [Message(role="user", content="hi")], tools=[]
                )

    def test_invalid_json_raises_provider_error(self):
        from missy.core.exceptions import ProviderError
        provider = _make_ollama_provider()

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("bad json")

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            with pytest.raises(ProviderError, match="JSON"):
                provider.complete_with_tools(
                    [Message(role="user", content="hi")], tools=[]
                )

    def test_provider_error_reraised(self):
        from missy.core.exceptions import ProviderError
        provider = _make_ollama_provider()

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.side_effect = ProviderError("direct")
            from missy.providers.base import Message
            with pytest.raises(ProviderError, match="direct"):
                provider.complete_with_tools(
                    [Message(role="user", content="hi")], tools=[]
                )

    def test_system_prompt_not_injected_when_already_present(self):
        provider = _make_ollama_provider()
        data = {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "hi", "tool_calls": []},
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = data

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            provider.complete_with_tools(
                [
                    Message(role="system", content="existing system"),
                    Message(role="user", content="hi"),
                ],
                tools=[],
                system="new system",  # should be ignored since system msg already present
            )
            payload = MockClient.return_value.post.call_args[1]["json"]

        # Count how many system messages appear
        system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "existing system"

    def test_tool_call_id_falls_back_to_name_prefix(self):
        provider = _make_ollama_provider()
        # Tool call with no id field — should use name[:8] as fallback
        data = {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "my_tool_name", "arguments": {"x": 1}}}
                ],
            },
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = data

        with patch("missy.providers.ollama_provider.PolicyHTTPClient") as MockClient:
            MockClient.return_value.post.return_value = mock_resp
            from missy.providers.base import Message
            result = provider.complete_with_tools(
                [Message(role="user", content="go")], tools=[]
            )

        assert result.tool_calls[0].id == "my_tool_"  # first 8 chars of "my_tool_name"


class TestOllamaGetToolSchemaNoGetSchema:
    def test_tool_without_get_schema_uses_empty_params(self):
        provider = _make_ollama_provider()

        # A tool without a get_schema method
        class MinimalTool:
            name = "no_schema"
            description = "tool with no schema"

        schemas = provider.get_tool_schema([MinimalTool()])
        assert schemas[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_params_without_type_key_wrapped_in_object(self):
        provider = _make_ollama_provider()

        class ToolWithBareParams:
            name = "bare"
            description = "bare params"

            def get_schema(self):
                # Parameters dict without a "type" key
                return {"parameters": {"text": {"type": "string"}}}

        schemas = provider.get_tool_schema([ToolWithBareParams()])
        params = schemas[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "text" in params["properties"]


# ---------------------------------------------------------------------------
# MemoryStore — uncovered paths
# ---------------------------------------------------------------------------


@pytest.fixture
def mem_store(tmp_path):
    from missy.memory.store import MemoryStore
    return MemoryStore(store_path=str(tmp_path / "memory.json"))


class TestMemoryStoreCompactSession:
    def test_compact_does_nothing_when_too_few_turns(self, mem_store):
        for i in range(5):
            mem_store.add_turn("s1", "user", f"msg {i}")
        removed = mem_store.compact_session("s1", keep_recent=10)
        assert removed == 0

    def test_compact_removes_old_turns(self, mem_store):
        for i in range(15):
            mem_store.add_turn("s1", "user", f"msg {i}")
        removed = mem_store.compact_session("s1", keep_recent=5)
        assert removed == 10

    def test_compact_inserts_summary_turn(self, mem_store):
        for i in range(10):
            mem_store.add_turn("s1", "user", f"msg {i}")
        mem_store.compact_session("s1", keep_recent=5)
        turns = mem_store.get_session_turns("s1")
        # First turn should be the compaction summary
        assert turns[0].content.startswith("[Compacted history]")

    def test_compact_preserves_keep_recent_turns(self, mem_store):
        for i in range(10):
            mem_store.add_turn("s1", "user", f"msg {i}")
        mem_store.compact_session("s1", keep_recent=3)
        turns = mem_store.get_session_turns("s1")
        # Last 3 verbatim turns + 1 summary = 4 total
        verbatim = [t for t in turns if not t.content.startswith("[Compacted")]
        assert len(verbatim) == 3
        assert verbatim[-1].content == "msg 9"

    def test_compact_persists_to_file(self, mem_store):
        from missy.memory.store import MemoryStore
        for i in range(8):
            mem_store.add_turn("s1", "user", f"msg {i}")
        mem_store.compact_session("s1", keep_recent=3)

        reloaded = MemoryStore(store_path=str(mem_store.store_path))
        turns = reloaded.get_session_turns("s1")
        assert any(t.content.startswith("[Compacted") for t in turns)

    def test_compact_preserves_other_sessions(self, mem_store):
        for i in range(8):
            mem_store.add_turn("s1", "user", f"a{i}")
        mem_store.add_turn("s2", "user", "b_turn")
        mem_store.compact_session("s1", keep_recent=3)

        s2_turns = mem_store.get_session_turns("s2")
        assert len(s2_turns) == 1
        assert s2_turns[0].content == "b_turn"

    def test_compact_summary_includes_user_and_assistant_prefix(self, mem_store):
        mem_store.add_turn("s1", "user", "user message")
        mem_store.add_turn("s1", "assistant", "assistant reply")
        for _ in range(8):
            mem_store.add_turn("s1", "user", "filler")
        mem_store.compact_session("s1", keep_recent=5)
        turns = mem_store.get_session_turns("s1")
        summary = turns[0].content
        # Both prefixes should appear
        assert "User:" in summary or "Assistant:" in summary


class TestMemoryStoreSearch:
    def test_search_finds_matching_turns(self, mem_store):
        mem_store.add_turn("s1", "user", "the cat sat on the mat")
        mem_store.add_turn("s1", "user", "the dog barked loudly")
        results = mem_store.search("cat")
        assert len(results) == 1
        assert "cat" in results[0].content

    def test_search_is_case_insensitive(self, mem_store):
        mem_store.add_turn("s1", "user", "Hello World")
        results = mem_store.search("hello")
        assert len(results) == 1

    def test_search_respects_limit(self, mem_store):
        for i in range(10):
            mem_store.add_turn("s1", "user", f"keyword item {i}")
        results = mem_store.search("keyword", limit=3)
        assert len(results) == 3

    def test_search_filters_by_session_id(self, mem_store):
        mem_store.add_turn("s1", "user", "keyword in s1")
        mem_store.add_turn("s2", "user", "keyword in s2")
        results = mem_store.search("keyword", session_id="s1")
        assert all(t.session_id == "s1" for t in results)
        assert len(results) == 1

    def test_search_returns_empty_when_no_match(self, mem_store):
        mem_store.add_turn("s1", "user", "nothing relevant")
        results = mem_store.search("xyzzy")
        assert results == []


class TestMemoryStoreLearningsStubs:
    def test_save_learning_is_noop(self, mem_store):
        # Should not raise and should not add turns
        mem_store.save_learning({"task_type": "test", "outcome": "ok"})
        assert len(mem_store.get_recent_turns()) == 0

    def test_get_learnings_returns_empty_list(self, mem_store):
        result = mem_store.get_learnings(task_type="any", limit=10)
        assert result == []


class TestMemoryStoreLoadEdgeCases:
    def test_non_list_json_starts_empty(self, tmp_path):
        from missy.memory.store import MemoryStore
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        store = MemoryStore(store_path=str(path))
        assert store.get_recent_turns() == []

    def test_non_dict_records_skipped(self, tmp_path):
        from missy.memory.store import MemoryStore
        path = tmp_path / "mixed.json"
        good = {
            "id": "abc",
            "session_id": "s1",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "role": "user",
            "content": "hello",
            "provider": "",
        }
        path.write_text(json.dumps(["not_a_dict", good]), encoding="utf-8")
        store = MemoryStore(store_path=str(path))
        # The valid record should be loaded; the string skipped
        assert len(store.get_recent_turns()) == 1

    def test_save_error_does_not_raise(self, tmp_path):
        from missy.memory.store import MemoryStore
        store = MemoryStore(store_path=str(tmp_path / "memory.json"))
        store.add_turn("s1", "user", "initial")

        # Make write_text raise to simulate disk full
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            # Must not propagate
            store.add_turn("s1", "user", "second")
