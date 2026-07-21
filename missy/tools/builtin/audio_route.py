"""Audio routing tools: bridge Missy's TTS output into OBS/VTube Studio.

Creates a PipeWire virtual sink (via ``pactl``, which PipeWire's
``pipewire-pulse`` compatibility layer implements) that Missy's TTS output
can be pointed at. OBS ("Audio Input Capture" source) and VTube Studio
(its own built-in microphone-based lip sync, Settings -> Input) then both
select that sink's *monitor* source as their input -- this is the
recommended way to get audio-driven mouth movement (see
``vtube_tools.py``'s module docstring for why a PipeWire route is
preferred over Missy computing per-frame lip-sync parameters itself).

Safety
------
- Never changes the operator's system-wide default audio sink unless
  ``set_default=True`` is explicitly passed (default ``False``) --
  creating a virtual sink for OBS/VTS to *listen to* doesn't require also
  making it where the operator's own desktop audio plays.
- The virtual sink's volume is capped at :data:`_SAFE_DEFAULT_VOLUME_PCT`
  (70%) on creation, never left at PipeWire's sometimes-100%+ default --
  "prevent accidental loud output."
- Uses ``pactl``/``espeak-ng`` via argv lists (no shell string), same
  non-shell-injection posture as ``desktop_tools.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import check_rate_limit

_SINK_NAME = "missy_tts_out"
_SAFE_DEFAULT_VOLUME_PCT = 70
#: No dedicated config section exists for audio routing (unlike obs/vtube/
#: desktop), so this is a fixed default rather than operator-configurable --
#: still a real guardrail against a runaway loop repeatedly recreating/
#: testing the sink.
_AUDIO_RATE_LIMIT_PER_MINUTE = 20


def _pw_env() -> dict[str, str]:
    safe_vars = ("PATH", "HOME", "USER", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")
    return {k: os.environ[k] for k in safe_vars if k in os.environ}


def _run(cmd: list[str], *, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=_pw_env(), timeout=timeout)


def _sink_exists(sink_name: str) -> bool:
    result = _run(["pactl", "list", "short", "sinks"])
    return any(
        line.split("\t")[1] == sink_name for line in result.stdout.splitlines() if "\t" in line
    )


class AudioRouteTtsTool(BaseTool):
    """Create/ensure a PipeWire virtual sink for OBS/VTube Studio to capture Missy's TTS."""

    name = "audio_route_tts"
    description = (
        "Create (idempotently) a PipeWire virtual audio sink that OBS or VTube "
        "Studio can capture as an input, so Missy's TTS speech drives lip sync "
        "and/or is heard on stream. Returns the monitor source name to select "
        "in OBS/VTube Studio's input settings."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "sink_name": {
            "type": "string",
            "description": "Name for the virtual sink.",
            "default": _SINK_NAME,
        },
        "set_default": {
            "type": "boolean",
            "description": (
                "Also make this sink the system default output (changes where ALL "
                "desktop audio plays, not just Missy's TTS). Default False -- leave "
                "the operator's normal audio output alone."
            ),
            "default": False,
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "pactl"

    def execute(
        self, *, sink_name: str = _SINK_NAME, set_default: bool = False, **_: Any
    ) -> ToolResult:
        if rate_error := check_rate_limit(self.name, _AUDIO_RATE_LIMIT_PER_MINUTE):
            return ToolResult(success=False, output=None, error=rate_error)
        if not shutil.which("pactl"):
            return ToolResult(
                success=False,
                output=None,
                error="pactl is not available. Install PipeWire's pulse-compat layer: "
                "sudo apt install pipewire-pulse",
            )

        created = False
        if not _sink_exists(sink_name):
            result = _run(
                [
                    "pactl",
                    "load-module",
                    "module-null-sink",
                    f"sink_name={sink_name}",
                    f"sink_properties=device.description={sink_name}",
                ]
            )
            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip()
                return ToolResult(success=False, output=None, error=f"Failed to create sink: {err}")
            created = True

        # Cap volume so this route can never surprise-blast at 100%+.
        _run(["pactl", "set-sink-volume", sink_name, f"{_SAFE_DEFAULT_VOLUME_PCT}%"])
        _run(["pactl", "set-sink-mute", sink_name, "0"])

        if set_default:
            default_result = _run(["pactl", "set-default-sink", sink_name])
            if default_result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Sink ready but set-default-sink failed: {default_result.stderr.strip()}",
                )

        return ToolResult(
            success=True,
            output={
                "sink_name": sink_name,
                "monitor_source": f"{sink_name}.monitor",
                "created": created,
                "volume_pct": _SAFE_DEFAULT_VOLUME_PCT,
                "set_as_default": set_default,
                "next_steps": (
                    f"In OBS: add an Audio Input Capture source, device = "
                    f"'{sink_name}.monitor'. In VTube Studio: Settings -> Input, select "
                    f"'{sink_name}.monitor' as the microphone for lip sync."
                ),
            },
        )


class AudioTestRouteTool(BaseTool):
    """Verify the TTS route is healthy and optionally play an audible test phrase through it."""

    name = "audio_test_route"
    description = (
        "Check that the audio_route_tts virtual sink exists, is unmuted, and has a "
        "safe volume; optionally play a short test phrase through it to confirm "
        "OBS/VTube Studio are receiving audio."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "sink_name": {
            "type": "string",
            "description": "Sink to test.",
            "default": _SINK_NAME,
        },
        "play_test_phrase": {
            "type": "boolean",
            "description": "Also play a short audible test phrase through the sink.",
            "default": True,
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "pactl && espeak-ng"

    def execute(
        self, *, sink_name: str = _SINK_NAME, play_test_phrase: bool = True, **_: Any
    ) -> ToolResult:
        if rate_error := check_rate_limit(self.name, _AUDIO_RATE_LIMIT_PER_MINUTE):
            return ToolResult(success=False, output=None, error=rate_error)
        if not shutil.which("pactl"):
            return ToolResult(success=False, output=None, error="pactl is not available.")
        if not _sink_exists(sink_name):
            return ToolResult(
                success=False,
                output=None,
                error=f"Sink {sink_name!r} does not exist. Run audio_route_tts first.",
            )

        vol_result = _run(["pactl", "get-sink-volume", sink_name])
        mute_result = _run(["pactl", "get-sink-mute", sink_name])
        muted = "yes" in mute_result.stdout.lower()

        played = False
        play_error: str | None = None
        if play_test_phrase:
            if not shutil.which("espeak-ng"):
                play_error = "espeak-ng not installed; skipped audible test."
            else:
                env = _pw_env()
                env["PULSE_SINK"] = sink_name
                try:
                    result = subprocess.run(
                        ["espeak-ng", "Testing Missy's audio route."],
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=10,
                    )
                    played = result.returncode == 0
                    if not played:
                        play_error = result.stderr.strip() or "espeak-ng failed"
                except subprocess.TimeoutExpired:
                    play_error = "espeak-ng timed out"

        return ToolResult(
            success=True,
            output={
                "sink_name": sink_name,
                "exists": True,
                "muted": muted,
                "volume": vol_result.stdout.strip(),
                "test_phrase_played": played,
                "play_error": play_error,
            },
        )
