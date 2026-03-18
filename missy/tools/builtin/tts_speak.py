"""Text-to-speech tool for Missy.

Synthesizes text to speech using Piper neural TTS (primary) or espeak-ng
(fallback) and plays it through the system's default audio output via
GStreamer + PipeWire.  Works over SSH sessions when ``XDG_RUNTIME_DIR``
is set correctly.

Prerequisites::

    # Piper (primary — high-quality neural TTS)
    # Install from https://github.com/rhasspy/piper
    # Binary: ~/.local/bin/piper
    # Voices: ~/.local/share/piper-voices/

    # espeak-ng (fallback — robotic but always available)
    sudo apt install espeak-ng gstreamer1.0-tools gstreamer1.0-plugins-base

Example::

    from missy.tools.builtin.tts_speak import TTSSpeakTool

    tool = TTSSpeakTool()
    result = tool.execute(text="Hello, I am Missy.")
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)

# Piper paths (user-local install)
_PIPER_BIN = Path.home() / ".local" / "bin" / "piper"
_PIPER_VOICES_DIR = Path.home() / ".local" / "share" / "piper-voices"
_PIPER_DEFAULT_VOICE = "en_US-lessac-medium"


#: Environment variables safe to pass to TTS/audio subprocesses.
#: Prevents API key leakage to espeak-ng, piper, gst-launch, etc.
_SAFE_TTS_ENV_VARS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE", "LANGUAGE",
    "TERM", "XDG_RUNTIME_DIR", "XDG_DATA_HOME", "XDG_CONFIG_HOME",
    "TMPDIR", "TMP", "TEMP", "DISPLAY", "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS", "LD_LIBRARY_PATH",
    "PULSE_SERVER", "PIPEWIRE_REMOTE",
})


def _ensure_runtime_dir() -> dict[str, str]:
    """Return a sanitized environment dict with XDG_RUNTIME_DIR set for PipeWire access.

    Only safe variables are passed to prevent API key leakage to
    TTS/audio subprocesses.
    """
    env = {k: v for k, v in os.environ.items() if k in _SAFE_TTS_ENV_VARS}
    if "XDG_RUNTIME_DIR" not in env:
        uid = os.getuid()
        env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    return env


def _piper_env() -> dict[str, str]:
    """Return env dict with LD_LIBRARY_PATH set for Piper shared libs."""
    env = _ensure_runtime_dir()
    piper_lib = str(_PIPER_BIN.parent)
    existing = env.get("LD_LIBRARY_PATH", "")
    if piper_lib not in existing:
        env["LD_LIBRARY_PATH"] = f"{piper_lib}:{existing}" if existing else piper_lib
    return env


def _find_piper_model(voice: str) -> Path | None:
    """Locate a Piper .onnx voice model file."""
    if not _PIPER_VOICES_DIR.is_dir():
        return None
    # Try exact name first, then with .onnx suffix
    for candidate in [
        _PIPER_VOICES_DIR / f"{voice}.onnx",
        _PIPER_VOICES_DIR / voice / f"{voice}.onnx",
    ]:
        if candidate.is_file():
            return candidate
    # Search for any matching .onnx
    for onnx in _PIPER_VOICES_DIR.glob(f"*{voice}*.onnx"):
        return onnx
    return None


def _synth_piper(text: str, wav_path: str, voice: str, speed: float) -> str | None:
    """Synthesize with Piper. Returns None on success, error string on failure."""
    if not _PIPER_BIN.is_file():
        return "piper binary not found"

    model = _find_piper_model(voice)
    if model is None:
        return f"piper voice model not found: {voice}"

    env = _piper_env()
    cmd = [
        str(_PIPER_BIN),
        "--model",
        str(model),
        "--output_file",
        wav_path,
    ]
    if speed != 1.0:
        cmd.extend(["--length_scale", str(1.0 / speed)])

    try:
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            env=env,
            timeout=60,
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace").strip()
            return f"piper failed: {err}"
        if not Path(wav_path).exists() or Path(wav_path).stat().st_size == 0:
            return "piper produced no audio"
        return None
    except FileNotFoundError:
        return "piper binary not found"
    except subprocess.TimeoutExpired:
        return "piper timed out"


def _synth_espeak(
    text: str, wav_path: str, speed: int, pitch: int, voice: str, env: dict
) -> str | None:
    """Synthesize with espeak-ng. Returns None on success, error string on failure."""
    try:
        synth = subprocess.run(
            [
                "espeak-ng",
                "--stdout",
                "-s",
                str(speed),
                "-p",
                str(pitch),
                "-v",
                voice,
                text,
            ],
            capture_output=True,
            env=env,
            timeout=30,
        )
        if synth.returncode != 0:
            err = synth.stderr.decode("utf-8", errors="replace").strip()
            return f"espeak-ng failed: {err}"
        if not synth.stdout:
            return "espeak-ng produced no audio"
        fd = os.open(wav_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(synth.stdout)
        return None
    except FileNotFoundError:
        return "espeak-ng not installed"
    except subprocess.TimeoutExpired:
        return "espeak-ng timed out"


def _play_wav(wav_path: str, env: dict) -> str | None:
    """Play a WAV file via GStreamer + PipeWire. Returns None on success, error on failure."""
    try:
        play = subprocess.run(
            [
                "gst-launch-1.0",
                "filesrc",
                f"location={wav_path}",
                "!",
                "wavparse",
                "!",
                "audioconvert",
                "!",
                "audioresample",
                "!",
                "pipewiresink",
            ],
            capture_output=True,
            env=env,
            timeout=60,
        )
        if play.returncode != 0:
            err = play.stderr.decode("utf-8", errors="replace").strip()
            if "command not found" in err or play.returncode == 127:
                return "gst-launch-1.0 not installed. Install with: sudo apt install gstreamer1.0-tools"
            return f"audio playback failed: {err}"
        return None
    except FileNotFoundError:
        return "gst-launch-1.0 not found"
    except subprocess.TimeoutExpired:
        return "audio playback timed out"


class TTSSpeakTool(BaseTool):
    """Speak text aloud through the system audio output.

    Uses Piper neural TTS for high-quality speech synthesis, with espeak-ng
    as a fallback.  Playback uses GStreamer through PipeWire.
    """

    name = "tts_speak"
    description = (
        "Speak text aloud through the USB speaker or default audio output. "
        "Use this to give voice responses, read text aloud, or announce information."
    )
    permissions = ToolPermissions(shell=True)

    parameters: dict[str, Any] = {
        "text": {
            "type": "string",
            "description": "The text to speak aloud.",
            "required": True,
        },
        "speed": {
            "type": "number",
            "description": "Speech speed multiplier (default 1.0; >1 = faster, <1 = slower).",
            "default": 1.0,
        },
        "voice": {
            "type": "string",
            "description": (
                "Voice name. For Piper: 'en_US-lessac-medium' (default). "
                "For espeak-ng fallback: 'en', 'en+f3', etc."
            ),
            "default": "en_US-lessac-medium",
        },
    }

    def execute(
        self,
        *,
        text: str,
        speed: float = 1.0,
        voice: str = "en_US-lessac-medium",
        **_: Any,
    ) -> ToolResult:
        if not text.strip():
            return ToolResult(success=False, output=None, error="No text provided.")

        speed = max(0.25, min(4.0, float(speed)))
        env = _piper_env()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            # 1. Try Piper first (high-quality neural TTS).
            engine = "piper"
            piper_voice = voice if voice != "en" else _PIPER_DEFAULT_VOICE
            err = _synth_piper(text, wav_path, piper_voice, speed)

            if err is not None:
                # 2. Fall back to espeak-ng.
                engine = "espeak-ng"
                logger.info("Piper unavailable (%s), falling back to espeak-ng", err)
                espeak_voice = voice if not voice.startswith("en_US") else "en"
                espeak_speed = max(80, min(450, int(160 * speed)))
                err = _synth_espeak(text, wav_path, espeak_speed, 50, espeak_voice, env)
                if err is not None:
                    return ToolResult(
                        success=False, output=None, error=f"TTS synthesis failed: {err}"
                    )

            # 3. Play via GStreamer + PipeWire.
            play_err = _play_wav(wav_path, env)
            if play_err is not None:
                return ToolResult(success=False, output=None, error=play_err)

            word_count = len(text.split())
            return ToolResult(
                success=True,
                output=f"Spoke {word_count} words aloud ({len(text)} chars, engine={engine}, voice={voice}).",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="TTS timed out.")
        except FileNotFoundError as exc:
            return ToolResult(success=False, output=None, error=f"Required binary not found: {exc}")
        finally:
            import contextlib

            with contextlib.suppress(OSError):
                os.unlink(wav_path)


class AudioListDevicesTool(BaseTool):
    """List available audio output devices via PipeWire/ALSA."""

    name = "audio_list_devices"
    description = (
        "List available audio playback devices (speakers, headphones, HDMI, USB). "
        "Useful for checking which audio output is active."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {}

    def execute(self, **_: Any) -> ToolResult:
        env = _ensure_runtime_dir()

        # Try wpctl first (PipeWire), fall back to aplay -l (ALSA).
        try:
            result = subprocess.run(
                ["wpctl", "status"],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )
            if result.returncode == 0:
                # Extract just the Audio section.
                lines = result.stdout.splitlines()
                audio_lines = []
                in_audio = False
                for line in lines:
                    if line.strip().startswith("Audio"):
                        in_audio = True
                    elif (
                        in_audio
                        and line.strip()
                        and not line.startswith(" ")
                        and not line.startswith("│")
                        and not line.startswith("├")
                        and not line.startswith("└")
                    ):
                        break
                    if in_audio:
                        audio_lines.append(line)
                if audio_lines:
                    return ToolResult(success=True, output="\n".join(audio_lines))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: ALSA.
        try:
            result = subprocess.run(
                ["aplay", "-l"],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )
            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return ToolResult(
            success=False, output=None, error="No audio listing tools available (wpctl or aplay)."
        )


class AudioSetVolumeTool(BaseTool):
    """Set the volume of an audio sink via PipeWire."""

    name = "audio_set_volume"
    description = (
        "Set the playback volume for the default or specified audio device. "
        "Volume is a percentage (0-100) or a relative change like '+5%' or '-10%'."
    )
    permissions = ToolPermissions(shell=True)

    parameters: dict[str, Any] = {
        "volume": {
            "type": "string",
            "description": "Volume level: '75%', '+10%', '-5%', or 'mute'/'unmute'.",
            "required": True,
        },
        "device_id": {
            "type": "string",
            "description": "PipeWire sink ID (default: '@DEFAULT_SINK@').",
            "default": "@DEFAULT_SINK@",
        },
    }

    def execute(
        self,
        *,
        volume: str,
        device_id: str = "@DEFAULT_SINK@",
        **_: Any,
    ) -> ToolResult:
        env = _ensure_runtime_dir()

        if volume.lower() == "mute":
            cmd = ["wpctl", "set-mute", device_id, "1"]
        elif volume.lower() == "unmute":
            cmd = ["wpctl", "set-mute", device_id, "0"]
        else:
            # Normalize: "75%" → "0.75", "+10%" → "0.10+"
            vol_str = volume.strip().rstrip("%")
            if vol_str.startswith(("+", "-")):
                sign = vol_str[0]
                val = float(vol_str[1:]) / 100.0
                wpctl_vol = f"{val}{sign}"
            else:
                val = float(vol_str) / 100.0
                wpctl_vol = str(val)
            cmd = ["wpctl", "set-volume", device_id, wpctl_vol]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )
            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip()
                return ToolResult(success=False, output=None, error=f"wpctl failed: {err}")

            # Get current volume for confirmation.
            get_vol = subprocess.run(
                ["wpctl", "get-volume", device_id],
                capture_output=True,
                text=True,
                env=env,
                timeout=5,
            )
            current = get_vol.stdout.strip() if get_vol.returncode == 0 else "unknown"
            return ToolResult(success=True, output=f"Volume set. Current: {current}")

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output=None,
                error="wpctl not found. Install wireplumber: sudo apt install wireplumber",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Volume command timed out.")
        except ValueError:
            return ToolResult(success=False, output=None, error=f"Invalid volume value: {volume}")
