"""Text-to-speech tool for Missy.

Synthesizes text to speech using espeak-ng and plays it through the
system's default audio output via GStreamer + PipeWire.  Works over SSH
sessions when ``XDG_RUNTIME_DIR`` is set correctly.

Prerequisites::

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
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult

logger = logging.getLogger(__name__)


def _ensure_runtime_dir() -> dict[str, str]:
    """Return an environment dict with XDG_RUNTIME_DIR set for PipeWire access."""
    env = {**os.environ}
    if "XDG_RUNTIME_DIR" not in env:
        uid = os.getuid()
        env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    return env


class TTSSpeakTool(BaseTool):
    """Speak text aloud through the system audio output.

    Uses espeak-ng for synthesis and GStreamer for playback through
    PipeWire.  The Jabra or default audio sink is used automatically.
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
            "type": "integer",
            "description": "Speech rate in words per minute (default 160, range 80-450).",
            "default": 160,
        },
        "pitch": {
            "type": "integer",
            "description": "Voice pitch (default 50, range 0-99).",
            "default": 50,
        },
        "voice": {
            "type": "string",
            "description": "espeak-ng voice name (default 'en', try 'en+f3' for female).",
            "default": "en",
        },
    }

    def execute(
        self,
        *,
        text: str,
        speed: int = 160,
        pitch: int = 50,
        voice: str = "en",
        **_: Any,
    ) -> ToolResult:
        if not text.strip():
            return ToolResult(success=False, output=None, error="No text provided.")

        speed = max(80, min(450, speed))
        pitch = max(0, min(99, pitch))
        env = _ensure_runtime_dir()

        # 1. Synthesize to a temp WAV file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            synth = subprocess.run(
                [
                    "espeak-ng",
                    "--stdout",
                    "-s", str(speed),
                    "-p", str(pitch),
                    "-v", voice,
                    text,
                ],
                capture_output=True,
                env=env,
                timeout=30,
            )
            if synth.returncode != 0:
                err = synth.stderr.decode("utf-8", errors="replace").strip()
                if "command not found" in err or synth.returncode == 127:
                    return ToolResult(
                        success=False, output=None,
                        error="espeak-ng not installed. Install with: sudo apt install espeak-ng",
                    )
                return ToolResult(success=False, output=None, error=f"espeak-ng failed: {err}")

            if not synth.stdout:
                return ToolResult(success=False, output=None, error="espeak-ng produced no audio.")

            with open(wav_path, "wb") as f:
                f.write(synth.stdout)

            # 2. Play via GStreamer + PipeWire.
            play = subprocess.run(
                [
                    "gst-launch-1.0",
                    "filesrc", f"location={wav_path}",
                    "!", "wavparse",
                    "!", "audioconvert",
                    "!", "audioresample",
                    "!", "pipewiresink",
                ],
                capture_output=True,
                env=env,
                timeout=30,
            )
            if play.returncode != 0:
                err = play.stderr.decode("utf-8", errors="replace").strip()
                if "command not found" in err or play.returncode == 127:
                    return ToolResult(
                        success=False, output=None,
                        error="gst-launch-1.0 not installed. Install with: sudo apt install gstreamer1.0-tools",
                    )
                return ToolResult(success=False, output=None, error=f"Audio playback failed: {err}")

            word_count = len(text.split())
            return ToolResult(
                success=True,
                output=f"Spoke {word_count} words aloud ({len(text)} chars, voice={voice}, speed={speed}wpm).",
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="TTS timed out after 30 seconds.")
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
                    elif in_audio and line.strip() and not line.startswith(" ") and not line.startswith("│") and not line.startswith("├") and not line.startswith("└"):
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

        return ToolResult(success=False, output=None, error="No audio listing tools available (wpctl or aplay).")


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
                success=False, output=None,
                error="wpctl not found. Install wireplumber: sudo apt install wireplumber",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Volume command timed out.")
        except ValueError:
            return ToolResult(success=False, output=None, error=f"Invalid volume value: {volume}")
