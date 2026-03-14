"""Coverage-gap tests for missy/tools/builtin/tts_speak.py.

Targets uncovered lines:
  49-50  : _ensure_runtime_dir — XDG_RUNTIME_DIR absent → uid-based fallback
  66-78  : _find_piper_model — voices-dir present; exact match, subdir match, glob match, none
  99     : _synth_piper — speed != 1.0 appends --length_scale flag
  114    : _synth_piper — piper succeeds but wav file is empty
  270-273: TTSSpeakTool.execute — subprocess.TimeoutExpired and FileNotFoundError branches
  426-427: AudioSetVolumeTool.execute — ValueError on bad volume string
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.tools.builtin.tts_speak import (
    AudioSetVolumeTool,
    TTSSpeakTool,
    _ensure_runtime_dir,
    _find_piper_model,
    _synth_piper,
)

# ---------------------------------------------------------------------------
# _ensure_runtime_dir
# ---------------------------------------------------------------------------


class TestEnsureRuntimeDir:
    """Lines 48-50: fallback sets XDG_RUNTIME_DIR from uid when missing."""

    def test_preserves_existing_xdg_runtime_dir(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/already/set")
        env = _ensure_runtime_dir()
        assert env["XDG_RUNTIME_DIR"] == "/run/user/already/set"

    def test_sets_xdg_runtime_dir_when_absent(self, monkeypatch):
        """Lines 49-50: when the key is absent, fall back to /run/user/<uid>."""
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        uid = os.getuid()
        env = _ensure_runtime_dir()
        assert env["XDG_RUNTIME_DIR"] == f"/run/user/{uid}"

    def test_returned_env_contains_other_vars(self, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_VAR", "custom_value")
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        env = _ensure_runtime_dir()
        assert env.get("MY_CUSTOM_VAR") == "custom_value"


# ---------------------------------------------------------------------------
# _find_piper_model
# ---------------------------------------------------------------------------


class TestFindPiperModel:
    """Lines 66-78: voice model lookup in the voices directory."""

    def _make_voices_dir(self, tmp_path: Path) -> Path:
        voices_dir = tmp_path / "piper-voices"
        voices_dir.mkdir()
        return voices_dir

    def test_returns_none_when_voices_dir_missing(self, tmp_path):
        """Line 66-67: voices dir not a directory → None."""
        nonexistent = tmp_path / "no-voices"
        with patch("missy.tools.builtin.tts_speak._PIPER_VOICES_DIR", nonexistent):
            result = _find_piper_model("en_US-lessac-medium")
        assert result is None

    def test_finds_exact_onnx_at_root(self, tmp_path):
        """Line 70: <voice>.onnx exists directly under voices dir."""
        voices_dir = self._make_voices_dir(tmp_path)
        model = voices_dir / "en_US-lessac-medium.onnx"
        model.write_bytes(b"fake-onnx")
        with patch("missy.tools.builtin.tts_speak._PIPER_VOICES_DIR", voices_dir):
            result = _find_piper_model("en_US-lessac-medium")
        assert result == model

    def test_finds_onnx_in_subdirectory(self, tmp_path):
        """Line 71: <voice>/<voice>.onnx exists in a voice-name subdirectory."""
        voices_dir = self._make_voices_dir(tmp_path)
        subdir = voices_dir / "myvoice"
        subdir.mkdir()
        model = subdir / "myvoice.onnx"
        model.write_bytes(b"fake-onnx")
        with patch("missy.tools.builtin.tts_speak._PIPER_VOICES_DIR", voices_dir):
            result = _find_piper_model("myvoice")
        assert result == model

    def test_finds_via_glob_fallback(self, tmp_path):
        """Lines 76-77: no exact match; glob finds a partial-name match."""
        voices_dir = self._make_voices_dir(tmp_path)
        # Name does not exactly match but contains the voice substring.
        model = voices_dir / "prefix_myvoice_suffix.onnx"
        model.write_bytes(b"fake-onnx")
        with patch("missy.tools.builtin.tts_speak._PIPER_VOICES_DIR", voices_dir):
            result = _find_piper_model("myvoice")
        assert result is not None
        assert result.name == "prefix_myvoice_suffix.onnx"

    def test_returns_none_when_no_match(self, tmp_path):
        """Line 78: glob finds nothing → None."""
        voices_dir = self._make_voices_dir(tmp_path)
        # Put an unrelated .onnx file there.
        (voices_dir / "other_voice.onnx").write_bytes(b"fake")
        with patch("missy.tools.builtin.tts_speak._PIPER_VOICES_DIR", voices_dir):
            result = _find_piper_model("totally_unknown_voice")
        assert result is None


# ---------------------------------------------------------------------------
# _synth_piper
# ---------------------------------------------------------------------------


class TestSynthPiper:
    """Lines 99 and 114."""

    def _fake_piper_bin(self, tmp_path: Path) -> Path:
        """Create a dummy piper binary file."""
        bin_path = tmp_path / "piper"
        bin_path.write_bytes(b"\x7fELF")  # minimal content; is_file() returns True
        return bin_path

    def _fake_model(self, tmp_path: Path) -> Path:
        model = tmp_path / "voice.onnx"
        model.write_bytes(b"fake")
        return model

    def test_speed_not_one_adds_length_scale_flag(self, tmp_path):
        """Line 99: speed != 1.0 extends cmd with --length_scale."""
        piper_bin = self._fake_piper_bin(tmp_path)
        model = self._fake_model(tmp_path)

        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = b""
        # Make the wav appear non-empty.
        wav_path = str(tmp_path / "out.wav")
        Path(wav_path).write_bytes(b"\x00" * 100)

        with (
            patch("missy.tools.builtin.tts_speak._PIPER_BIN", piper_bin),
            patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=model),
            patch("subprocess.run", return_value=completed) as mock_run,
        ):
            err = _synth_piper("hello", wav_path, "voice", speed=2.0)

        assert err is None
        cmd_used = mock_run.call_args.args[0]
        assert "--length_scale" in cmd_used
        # At speed=2.0, length_scale = 1/2.0 = 0.5
        length_scale_idx = cmd_used.index("--length_scale")
        assert float(cmd_used[length_scale_idx + 1]) == pytest.approx(0.5)

    def test_speed_exactly_one_omits_length_scale(self, tmp_path):
        """Confirm --length_scale is NOT added when speed == 1.0."""
        piper_bin = self._fake_piper_bin(tmp_path)
        model = self._fake_model(tmp_path)

        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = b""
        wav_path = str(tmp_path / "out.wav")
        Path(wav_path).write_bytes(b"\x00" * 100)

        with (
            patch("missy.tools.builtin.tts_speak._PIPER_BIN", piper_bin),
            patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=model),
            patch("subprocess.run", return_value=completed) as mock_run,
        ):
            err = _synth_piper("hello", wav_path, "voice", speed=1.0)

        assert err is None
        cmd_used = mock_run.call_args.args[0]
        assert "--length_scale" not in cmd_used

    def test_empty_wav_returns_error(self, tmp_path):
        """Line 114: piper exits 0 but produces a zero-byte wav → error string."""
        piper_bin = self._fake_piper_bin(tmp_path)
        model = self._fake_model(tmp_path)

        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = b""
        # Create the wav path but leave it empty (0 bytes).
        wav_path = str(tmp_path / "empty.wav")
        Path(wav_path).write_bytes(b"")

        with (
            patch("missy.tools.builtin.tts_speak._PIPER_BIN", piper_bin),
            patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=model),
            patch("subprocess.run", return_value=completed),
        ):
            err = _synth_piper("hello", wav_path, "voice", speed=1.0)

        assert err == "piper produced no audio"

    def test_nonexistent_wav_returns_error(self, tmp_path):
        """Line 112-113: piper exits 0 but wav was never written → error string."""
        piper_bin = self._fake_piper_bin(tmp_path)
        model = self._fake_model(tmp_path)

        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = b""
        wav_path = str(tmp_path / "never_created.wav")
        # Do NOT write the wav file.

        with (
            patch("missy.tools.builtin.tts_speak._PIPER_BIN", piper_bin),
            patch("missy.tools.builtin.tts_speak._find_piper_model", return_value=model),
            patch("subprocess.run", return_value=completed),
        ):
            err = _synth_piper("hello", wav_path, "voice", speed=1.0)

        assert err == "piper produced no audio"


# ---------------------------------------------------------------------------
# TTSSpeakTool.execute — exception branches
# ---------------------------------------------------------------------------


class TestTTSSpeakToolExceptionBranches:
    """Lines 270-273: subprocess.TimeoutExpired and FileNotFoundError in execute()."""

    def test_timeout_expired_returns_failure(self):
        """Line 270-271: subprocess.TimeoutExpired in execute → ToolResult with 'timed out'."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", side_effect=subprocess.TimeoutExpired("cmd", 60)),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_file_not_found_returns_failure(self):
        """Line 272-273: FileNotFoundError in execute → ToolResult mentioning the binary."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", side_effect=FileNotFoundError("piper")),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_empty_text_returns_failure(self):
        """Guard at line 232-233: no text → immediate failure."""
        result = TTSSpeakTool().execute(text="   ")
        assert result.success is False
        assert "No text provided" in result.error

    def test_execute_piper_fails_espeak_also_fails(self):
        """Both synthesis paths fail → TTS synthesis failed in error."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper not found"),
            patch("missy.tools.builtin.tts_speak._synth_espeak", return_value="espeak not installed"),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "TTS synthesis failed" in result.error

    def test_execute_piper_fails_espeak_succeeds_play_fails(self):
        """Piper fails, espeak works, but playback fails."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper not found"),
            patch("missy.tools.builtin.tts_speak._synth_espeak", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value="gst-launch-1.0 not installed. Install with: sudo apt install gstreamer1.0-tools"),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world")

        assert result.success is False
        assert "gst-launch" in result.error

    def test_execute_full_success_path(self):
        """Happy path: piper + playback both succeed."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="hello world", speed=1.0, voice="en_US-lessac-medium")

        assert result.success is True
        assert "2" in result.output  # 2 words spoken
        assert "engine=piper" in result.output

    def test_execute_espeak_fallback_reports_engine(self):
        """When piper fails but espeak succeeds, output mentions espeak-ng."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value="piper binary not found"),
            patch("missy.tools.builtin.tts_speak._synth_espeak", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="test text here", voice="en")

        assert result.success is True
        assert "engine=espeak-ng" in result.output

    def test_execute_clamps_speed_high(self):
        """Speed above 4.0 is clamped to 4.0 (no error)."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="fast", speed=99.0)

        assert result.success is True

    def test_execute_clamps_speed_low(self):
        """Speed below 0.25 is clamped to 0.25 (no error)."""
        tool = TTSSpeakTool()
        with (
            patch("missy.tools.builtin.tts_speak._synth_piper", return_value=None),
            patch("missy.tools.builtin.tts_speak._play_wav", return_value=None),
            patch("missy.tools.builtin.tts_speak._piper_env", return_value={}),
        ):
            result = tool.execute(text="slow", speed=0.0)

        assert result.success is True


# ---------------------------------------------------------------------------
# AudioSetVolumeTool — ValueError branch
# ---------------------------------------------------------------------------


class TestAudioSetVolumeToolValueError:
    """Lines 426-427: ValueError on a non-numeric volume string."""

    def test_invalid_volume_string_returns_error(self):
        """Lines 426-427: ValueError raised inside the try block → handled by except ValueError."""
        tool = AudioSetVolumeTool()
        # Trigger ValueError inside the try block (subprocess.run) by patching it directly.
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=ValueError("bad volume conversion")),
        ):
            result = tool.execute(volume="75%")

        assert result.success is False
        assert "Invalid volume value" in result.error
        assert "75%" in result.error

    def test_valid_absolute_volume_succeeds(self):
        """Ensure wpctl happy path for '75%' works (regression guard)."""
        tool = AudioSetVolumeTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""
        mock_get = MagicMock()
        mock_get.returncode = 0
        mock_get.stdout = "Volume: 0.75"

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=[mock_result, mock_get]),
        ):
            result = tool.execute(volume="75%")

        assert result.success is True
        assert "0.75" in result.output

    def test_mute_command(self):
        """'mute' routes to wpctl set-mute 1."""
        tool = AudioSetVolumeTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_get = MagicMock()
        mock_get.returncode = 0
        mock_get.stdout = "Volume: 0.50 [MUTED]"

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=[mock_result, mock_get]) as mock_run,
        ):
            result = tool.execute(volume="mute")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        assert "set-mute" in first_cmd
        assert "1" in first_cmd

    def test_unmute_command(self):
        """'unmute' routes to wpctl set-mute 0."""
        tool = AudioSetVolumeTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_get = MagicMock()
        mock_get.returncode = 0
        mock_get.stdout = "Volume: 0.50"

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=[mock_result, mock_get]) as mock_run,
        ):
            result = tool.execute(volume="unmute")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        assert "set-mute" in first_cmd
        assert "0" in first_cmd

    def test_relative_volume_increase(self):
        """'+10%' maps to wpctl set-volume with positive relative format."""
        tool = AudioSetVolumeTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_get = MagicMock()
        mock_get.returncode = 0
        mock_get.stdout = "Volume: 0.80"

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=[mock_result, mock_get]) as mock_run,
        ):
            result = tool.execute(volume="+10%")

        assert result.success is True
        first_cmd = mock_run.call_args_list[0].args[0]
        # wpctl relative volume arg ends with '+'
        assert any(arg.endswith("+") for arg in first_cmd)

    def test_wpctl_failure_returns_error(self):
        """When wpctl exits non-zero, return failure result."""
        tool = AudioSetVolumeTool()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error msg"
        mock_result.stdout = ""

        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = tool.execute(volume="50%")

        assert result.success is False
        assert "wpctl failed" in result.error

    def test_file_not_found_returns_install_hint(self):
        """When wpctl is not installed, return a helpful error with install hint."""
        tool = AudioSetVolumeTool()
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=FileNotFoundError("wpctl")),
        ):
            result = tool.execute(volume="50%")

        assert result.success is False
        assert "wpctl not found" in result.error

    def test_timeout_returns_error(self):
        """TimeoutExpired → 'timed out' in error."""
        tool = AudioSetVolumeTool()
        with (
            patch("missy.tools.builtin.tts_speak._ensure_runtime_dir", return_value={}),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("wpctl", 5)),
        ):
            result = tool.execute(volume="50%")

        assert result.success is False
        assert "timed out" in result.error.lower()
