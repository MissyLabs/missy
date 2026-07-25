"""VTube Studio integration for Missy via its public WebSocket API.

See https://github.com/DenchiSoft/VTubeStudio for the protocol. Uses the
``websockets`` package already a core Missy dependency -- no new required
dependency. Each tool call opens a fresh connection, authenticates with the
persisted token, does one request/response, and closes.

Auth flow
---------
VTube Studio requires a one-time interactive authorization: the plugin
requests a token (``AuthenticationTokenRequest``), the human clicks
"Allow" in a pop-up inside the VTube Studio app itself, and VTS returns a
persistent token. Every later connection re-authenticates with that same
token (``AuthenticationRequest``) -- no further human interaction needed
once granted.

:func:`_vtube_authenticate` handles both paths: if ``vtube.auth_token`` is
already configured, it just re-authenticates; if not, it requests a new
token (blocking up to :data:`_AUTH_POPUP_TIMEOUT_SECONDS` for the human to
click Allow in VTS) and, on success, saves it directly to the encrypted
:class:`~missy.security.vault.Vault` -- the token is **never** returned in
any tool output, matching the OAuth flow's own persist-don't-print pattern
(see ``missy/cli/oauth.py``).

Known VTube Studio API gap
---------------------------
The feedback that motivated this integration asked for "start/stop
tracking" and audio-driven mouth movement. VTube Studio's public API has
no direct "start/stop face tracking" request -- tracking is controlled by
the app's own webcam/tracker settings or a hotkey the operator has bound
in the VTS UI, reachable here only via :class:`VtubeTriggerHotkeyTool`.
:class:`VtubeSetParameterTool` still exists for simple scripted/discrete
puppeting (e.g. one-shot expression parameters).

Audio-synced lip sync: direct parameter injection, not VTS's own mic
--------------------------------------------------------------------
An earlier version of this module preferred routing TTS audio into VTube
Studio's own built-in microphone-based lip sync (via a PipeWire virtual
sink from ``audio_route_tts``) over Missy computing and streaming
synthetic ``InjectParameterDataRequest`` calls herself, reasoning that a
real microphone input would be more reliable than a hand-rolled
alternative. Live testing found the opposite on a Proton/Wine VTube
Studio install: no evidence VTS ever captured the routed PipeWire monitor
as a microphone input at all (no corresponding audio client ever appeared
against the sink), and the one-time GUI toggle needed to pick that input
device (VTS's "Show controls" setting) turned out to have no discoverable
persisted state anywhere (config JSON, Wine registry, Steam Cloud) and no
bound hotkey to restore it once hidden -- an unreliable, hard-to-recover
dependency on a Windows app's audio-input handling under Wine.
:class:`VtubeSpeakTool` instead synthesizes speech, computes its
amplitude envelope directly, and streams ``MouthOpen`` (VTS's own
standard input parameter, present for any model) over one persistent
authenticated connection timed to playback -- confirmed live to produce
real, continuously-varying mouth movement, with no dependency on VTS's
own audio-input capture at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
import wave
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import check_rate_limit, load_missy_config

logger = logging.getLogger(__name__)

_WS_TIMEOUT_SECONDS = 10.0
_AUTH_POPUP_TIMEOUT_SECONDS = 30.0
_API_NAME = "VTubeStudioPublicAPI"
_API_VERSION = "1.0"


class VtubeError(Exception):
    """Raised for any VTube Studio connection, auth, or request failure."""


def _vtube_config():
    """Return the configured :class:`~missy.config.settings.VtubeConfig`, or ``None``."""
    cfg = load_missy_config()
    return cfg.vtube if cfg is not None else None


def _check_rate_limit(tool_name: str) -> str | None:
    """Rate-limit *tool_name* against ``vtube.rate_limit_per_minute``."""
    from missy.config.settings import VtubeConfig

    config = _vtube_config() or VtubeConfig()
    return check_rate_limit(tool_name, config.rate_limit_per_minute)


def _vtube_host() -> str:
    config = _vtube_config()
    return config.host if config is not None else ""


# ---------------------------------------------------------------------------
# VTube Studio API client
# ---------------------------------------------------------------------------


async def _vts_send(ws: Any, message_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Send one VTS request and return its ``data`` payload."""
    import json

    request_id = str(uuid.uuid4())
    await ws.send(
        json.dumps(
            {
                "apiName": _API_NAME,
                "apiVersion": _API_VERSION,
                "requestID": request_id,
                "messageType": message_type,
                "data": data,
            }
        )
    )
    raw = await ws.recv()
    response = json.loads(raw)
    if response.get("messageType") == "APIError":
        err = response.get("data", {})
        raise VtubeError(f"VTube Studio error {err.get('errorID')}: {err.get('message')}")
    return response.get("data", {})


async def _vts_authenticate(
    ws: Any, *, plugin_name: str, plugin_developer: str, token: str | None
) -> str:
    """Authenticate this connection; acquire a new token if none is configured.

    Returns the token that was used (freshly acquired or pre-configured).
    A freshly-acquired token is the caller's responsibility to persist --
    this function only authenticates, it never writes to Vault itself, so
    it stays a pure protocol helper with no I/O side effects.

    Raises:
        VtubeError: If a fresh token is required and the human doesn't
            approve the pop-up within :data:`_AUTH_POPUP_TIMEOUT_SECONDS`,
            or if authentication is rejected.
    """
    if not token:
        token_resp = await asyncio.wait_for(
            _vts_send(
                ws,
                "AuthenticationTokenRequest",
                {
                    "pluginName": plugin_name,
                    "pluginDeveloper": plugin_developer,
                },
            ),
            timeout=_AUTH_POPUP_TIMEOUT_SECONDS,
        )
        token = token_resp.get("authenticationToken")
        if not token:
            raise VtubeError(
                "VTube Studio did not issue an authentication token "
                "(the user may have clicked Deny in the VTS pop-up)."
            )

    auth_resp = await _vts_send(
        ws,
        "AuthenticationRequest",
        {
            "pluginName": plugin_name,
            "pluginDeveloper": plugin_developer,
            "authenticationToken": token,
        },
    )
    if not auth_resp.get("authenticated"):
        raise VtubeError(f"VTube Studio authentication rejected: {auth_resp.get('reason')}")
    return token


async def _vtube_request_async(
    message_type: str,
    data: dict[str, Any],
    *,
    host: str,
    port: int,
    token: str | None,
    plugin_name: str,
    plugin_developer: str,
) -> tuple[dict[str, Any], str]:
    """Open a connection, authenticate, perform one request, and close.

    Returns ``(response_data, token_used)`` -- the caller persists
    ``token_used`` to Vault when it's newly acquired (differs from the
    ``token`` passed in).
    """
    import websockets

    uri = f"ws://{host}:{port}"
    try:
        async with asyncio.timeout(_WS_TIMEOUT_SECONDS + _AUTH_POPUP_TIMEOUT_SECONDS):
            async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
                token_used = await _vts_authenticate(
                    ws, plugin_name=plugin_name, plugin_developer=plugin_developer, token=token
                )
                response_data = await _vts_send(ws, message_type, data)
                return response_data, token_used
    except TimeoutError as exc:
        raise VtubeError(
            f"Timed out talking to VTube Studio at {uri} "
            "(if this is a first-time authorization, check for a pop-up in the VTS app)."
        ) from exc
    except OSError as exc:
        raise VtubeError(f"Could not connect to VTube Studio at {uri}: {exc}") from exc


def _vtube_request(message_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sync wrapper: validate config/policy, authenticate, perform one request.

    Persists a freshly-acquired auth token to Vault (never returned to the
    caller) so subsequent calls skip the pop-up.

    Raises:
        VtubeError: When VTube Studio integration is disabled, or on any
            connection/auth/request failure.
    """
    config = _vtube_config()
    if config is None or not config.enabled:
        raise VtubeError(
            "VTube Studio integration is disabled. Set vtube.enabled: true in "
            "config.yaml (see docs/desktop_obs_vtube.md for setup)."
        )

    from missy.policy.engine import get_policy_engine

    get_policy_engine().check_network(config.host, category="tool")

    response_data, token_used = asyncio.run(
        _vtube_request_async(
            message_type,
            data or {},
            host=config.host,
            port=config.port,
            token=config.auth_token,
            plugin_name=config.plugin_name,
            plugin_developer=config.plugin_developer,
        )
    )

    if token_used != config.auth_token:
        _persist_token(token_used)

    return response_data


def _persist_token(token: str) -> None:
    """Save a freshly-acquired VTube Studio auth token to the encrypted vault.

    Never logged, never returned in any :class:`~missy.tools.base.ToolResult`.
    """
    try:
        from missy.security.vault import Vault

        Vault().set("vtube_studio_token", token)
        logger.info(
            "vtube_tools: new auth token saved to vault as 'vtube_studio_token'. "
            "Set vtube.auth_token: vault://vtube_studio_token in config.yaml to use it."
        )
    except Exception:
        logger.warning(
            "vtube_tools: acquired a new auth token but could not save it to vault; "
            "the VTS approval pop-up will be needed again next time.",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class VtubeStatusTool(BaseTool):
    """Report VTube Studio connection state and the currently loaded model."""

    name = "vtube_status"
    description = (
        "Get VTube Studio's current status: connection health and the currently "
        "loaded Live2D model. First call may block waiting for a one-time "
        "approval pop-up inside the VTube Studio app."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _vtube_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        try:
            data = _vtube_request("CurrentModelRequest")
        except VtubeError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(
            success=True,
            output={
                "connected": True,
                "model_loaded": data.get("modelLoaded"),
                "model_name": data.get("modelName"),
                "model_id": data.get("modelID"),
                "live2d_model_name": data.get("live2DModelName"),
            },
        )


class VtubeLoadModelTool(BaseTool):
    """Load a Live2D model in VTube Studio by name."""

    name = "vtube_load_model"
    description = (
        "Load a Live2D model in VTube Studio by its display name (see "
        "AvailableModelsRequest names, or ask the operator which models are imported)."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {
        "model_name": {
            "type": "string",
            "description": "Exact model name as shown in VTube Studio.",
            "required": True,
        },
    }

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _vtube_host()
        return [host] if host else []

    def execute(self, *, model_name: str, **_: Any) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        try:
            models = _vtube_request("AvailableModelsRequest")
            model_id = next(
                (
                    m["modelID"]
                    for m in models.get("availableModels", [])
                    if m.get("modelName") == model_name
                ),
                None,
            )
            if model_id is None:
                available = [m.get("modelName") for m in models.get("availableModels", [])]
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Model {model_name!r} not found. Available: {available}",
                )
            _vtube_request("ModelLoadRequest", {"modelID": model_id})
        except VtubeError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(success=True, output={"model_name": model_name, "model_id": model_id})


class VtubeTriggerHotkeyTool(BaseTool):
    """Trigger a VTube Studio hotkey (expressions, animations, tracking toggles, etc.)."""

    name = "vtube_trigger_hotkey"
    description = (
        "Trigger a VTube Studio hotkey by name (expressions, animations, or any "
        "action the operator has bound to a hotkey in VTS, including toggling "
        "face tracking if bound)."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {
        "hotkey_name": {
            "type": "string",
            "description": "Exact hotkey name as configured in VTube Studio.",
            "required": True,
        },
    }

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _vtube_host()
        return [host] if host else []

    def execute(self, *, hotkey_name: str, **_: Any) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        try:
            hotkeys = _vtube_request("HotkeysInCurrentModelRequest")
            hotkey_id = next(
                (
                    h["hotkeyID"]
                    for h in hotkeys.get("availableHotkeys", [])
                    if h.get("name") == hotkey_name
                ),
                None,
            )
            if hotkey_id is None:
                available = [h.get("name") for h in hotkeys.get("availableHotkeys", [])]
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Hotkey {hotkey_name!r} not found. Available: {available}",
                )
            _vtube_request("HotkeyTriggerRequest", {"hotkeyID": hotkey_id})
        except VtubeError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(success=True, output={"hotkey_name": hotkey_name, "triggered": True})


class VtubeSetParameterTool(BaseTool):
    """Set a Live2D model parameter (e.g. for scripted/discrete puppeting).

    For continuous audio-synced mouth movement, use :class:`VtubeSpeakTool`
    instead of scripting this per-frame -- see the module docstring's
    "Audio-synced lip sync" section.
    """

    name = "vtube_set_parameter"
    description = (
        "Set a Live2D model parameter's value (e.g. 'MouthOpen', 'MouthSmile', "
        "'EyeOpenLeft'). For speech with synced mouth movement, use vtube_speak "
        "instead of scripting this per-frame."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {
        "parameter_id": {
            "type": "string",
            "description": "Live2D parameter ID, e.g. 'MouthOpen'.",
            "required": True,
        },
        "value": {
            "type": "number",
            "description": "Value to set (parameter-defined range, commonly 0.0-1.0).",
            "required": True,
        },
        "weight": {
            "type": "number",
            "description": "Blend weight against other inputs to this parameter (0.0-1.0).",
            "default": 1.0,
        },
    }

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _vtube_host()
        return [host] if host else []

    def execute(
        self, *, parameter_id: str, value: float, weight: float = 1.0, **_: Any
    ) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        try:
            _vtube_request(
                "InjectParameterDataRequest",
                {
                    "faceFound": False,
                    "mode": "set",
                    "parameterValues": [
                        {"id": parameter_id, "value": float(value), "weight": float(weight)}
                    ],
                },
            )
        except VtubeError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(
            success=True, output={"parameter_id": parameter_id, "value": float(value)}
        )


class VtubeListModelsTool(BaseTool):
    """List every Live2D model VTube Studio currently has imported.

    Answers "locating Live2D models" -- this is read-only discovery, not
    import: VTube Studio has no API to import a new model file, only to
    load one it already knows about (see module docstring).
    """

    name = "vtube_list_models"
    description = (
        "List every Live2D model currently imported into VTube Studio (name, ID, "
        "and whether it's the one currently loaded). Use this to find a model_name "
        "for vtube_load_model."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _vtube_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        try:
            models = _vtube_request("AvailableModelsRequest")
        except VtubeError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        model_list = [
            {
                "model_name": m.get("modelName"),
                "model_id": m.get("modelID"),
                "is_loaded": bool(m.get("modelLoaded")),
            }
            for m in models.get("availableModels", [])
        ]
        return ToolResult(success=True, output={"models": model_list, "count": len(model_list)})


# ---------------------------------------------------------------------------
# VtubeSpeakTool -- speech with directly-injected, amplitude-driven lip sync
# ---------------------------------------------------------------------------

_ENVELOPE_CHUNK_MS = 50
_LIPSYNC_PARAMETER_ID = "MouthOpen"
_MOUTH_CURVE_EXPONENT = 0.6  # compress peaks so speech doesn't slam to 1.0
_MOUTH_PEAK_SCALE = 0.85
_SINK_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _wav_envelope(wav_path: str, chunk_ms: int) -> tuple[list[float], float]:
    """Return (per-chunk RMS levels, total duration seconds) for a WAV file."""
    try:
        import numpy as np
    except ImportError as exc:
        raise VtubeError(
            "numpy is required for vtube_speak's amplitude envelope. Install with: "
            'pip install -e ".[voice]" (or [vision]/[retrieval], any of which include numpy)'
        ) from exc

    with wave.open(wav_path, "rb") as w:
        rate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
        sampwidth = w.getsampwidth()
        channels = w.getnchannels()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
    if dtype is None:
        raise VtubeError(f"Unsupported WAV sample width: {sampwidth} bytes")
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    maxval = float(np.iinfo(dtype).max)
    chunk_len = max(1, int(rate * chunk_ms / 1000))
    n_chunks = max(1, len(samples) // chunk_len) if len(samples) else 0
    levels: list[float] = []
    for i in range(n_chunks):
        chunk = samples[i * chunk_len : (i + 1) * chunk_len]
        rms = float(np.sqrt(np.mean(chunk**2))) / maxval if len(chunk) else 0.0
        levels.append(rms)
    duration = n_frames / rate if rate else 0.0
    return levels, duration


def _normalize_envelope(levels: list[float]) -> list[float]:
    """Scale raw RMS levels to a natural-looking MouthOpen range (0.0-0.85)."""
    peak = max(levels) if levels else 0.0
    if peak <= 0:
        return [0.0 for _ in levels]
    return [
        min(1.0, (level / peak) ** _MOUTH_CURVE_EXPONENT) * _MOUTH_PEAK_SCALE for level in levels
    ]


async def _stream_mouth_envelope(
    levels: list[float],
    chunk_ms: int,
    *,
    host: str,
    port: int,
    token: str | None,
    plugin_name: str,
    plugin_developer: str,
    parameter_id: str,
) -> str:
    """Stream ``levels`` to *parameter_id* over one persistent connection.

    Returns the token used (so a freshly-acquired one can be persisted by
    the caller, matching :func:`_vtube_request`'s contract).
    """
    import websockets

    uri = f"ws://{host}:{port}"
    async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
        token_used = await _vts_authenticate(
            ws, plugin_name=plugin_name, plugin_developer=plugin_developer, token=token
        )
        start = time.monotonic()
        for i, level in enumerate(levels):
            target_t = start + i * chunk_ms / 1000
            now = time.monotonic()
            if target_t > now:
                await asyncio.sleep(target_t - now)
            await _vts_send(
                ws,
                "InjectParameterDataRequest",
                {
                    "faceFound": False,
                    "mode": "set",
                    "parameterValues": [{"id": parameter_id, "value": level}],
                },
            )
        # Always close the mouth on completion, even for an empty envelope.
        await _vts_send(
            ws,
            "InjectParameterDataRequest",
            {
                "faceFound": False,
                "mode": "set",
                "parameterValues": [{"id": parameter_id, "value": 0.0}],
            },
        )
        return token_used


class VtubeSpeakTool(BaseTool):
    """Speak text aloud and drive VTube Studio's mouth parameter directly.

    Synthesizes with the same Piper/espeak-ng engines as ``tts_speak``,
    plays the result through a PipeWire sink (so OBS/humans still hear it),
    and concurrently streams the audio's own amplitude envelope to VTS's
    ``MouthOpen`` input over one persistent connection -- see the module
    docstring's "Audio-synced lip sync" section for why this replaced an
    earlier attempt to route TTS into VTube Studio's own microphone input.
    """

    name = "vtube_speak"
    description = (
        "Speak text aloud with VTube Studio's avatar mouth moving in sync, by "
        "injecting the synthesized audio's own amplitude directly into VTS's "
        "MouthOpen parameter. Use this instead of tts_speak when the VTuber "
        "avatar should visibly talk."
    )
    permissions = ToolPermissions(shell=True, network=True)
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
        "output_sink": {
            "type": "string",
            "description": (
                "PipeWire sink to play through -- default matches audio_route_tts's "
                "sink so OBS/humans hear it without touching the system default."
            ),
            "default": "missy_tts_out",
        },
        "parameter_id": {
            "type": "string",
            "description": "VTS input parameter to drive from audio amplitude.",
            "default": _LIPSYNC_PARAMETER_ID,
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        from missy.tools.builtin.tts_speak import _PIPER_BIN

        return f"{_PIPER_BIN} && espeak-ng && gst-launch-1.0"

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _vtube_host()
        return [host] if host else []

    def execute(
        self,
        *,
        text: str,
        speed: float = 1.0,
        voice: str = "en_US-lessac-medium",
        output_sink: str = "missy_tts_out",
        parameter_id: str = _LIPSYNC_PARAMETER_ID,
        **_: Any,
    ) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        if not text.strip():
            return ToolResult(success=False, output=None, error="No text provided.")

        if not _SINK_NAME_RE.fullmatch(output_sink):
            return ToolResult(
                success=False,
                output=None,
                error=f"Invalid output_sink {output_sink!r}: must be a plain PipeWire sink name.",
            )

        config = _vtube_config()
        if config is None or not config.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="VTube Studio integration is disabled. Set vtube.enabled: true in config.yaml.",
            )

        from missy.policy.engine import get_policy_engine

        try:
            get_policy_engine().check_network(config.host, category="tool")
        except Exception as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        from missy.tools.builtin.tts_speak import (
            _PIPER_DEFAULT_VOICE,
            _piper_env,
            _synth_espeak,
            _synth_piper,
        )

        speed = max(0.25, min(4.0, float(speed)))
        env = _piper_env()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            engine = "piper"
            piper_voice = voice if voice != "en" else _PIPER_DEFAULT_VOICE
            synth_err = _synth_piper(text, wav_path, piper_voice, speed)
            if synth_err is not None:
                engine = "espeak-ng"
                logger.info("Piper unavailable (%s), falling back to espeak-ng", synth_err)
                espeak_voice = voice if not voice.startswith("en_US") else "en"
                espeak_speed = max(80, min(450, int(160 * speed)))
                synth_err = _synth_espeak(text, wav_path, espeak_speed, 50, espeak_voice, env)
                if synth_err is not None:
                    return ToolResult(
                        success=False, output=None, error=f"TTS synthesis failed: {synth_err}"
                    )

            try:
                levels, duration = _wav_envelope(wav_path, _ENVELOPE_CHUNK_MS)
            except VtubeError as exc:
                return ToolResult(success=False, output=None, error=str(exc))
            levels = _normalize_envelope(levels)

            play_env = dict(env)
            play_proc = subprocess.Popen(
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
                    f"target-object={output_sink}",
                ],
                env=play_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                token_used = asyncio.run(
                    _stream_mouth_envelope(
                        levels,
                        _ENVELOPE_CHUNK_MS,
                        host=config.host,
                        port=config.port,
                        token=config.auth_token,
                        plugin_name=config.plugin_name,
                        plugin_developer=config.plugin_developer,
                        parameter_id=parameter_id,
                    )
                )
            except VtubeError as exc:
                play_proc.wait(timeout=10)
                return ToolResult(success=False, output=None, error=str(exc))

            if token_used != config.auth_token:
                _persist_token(token_used)

            play_returncode = play_proc.wait(timeout=10)
            if play_returncode != 0:
                stderr = (play_proc.stderr.read() if play_proc.stderr else b"").decode(
                    "utf-8", errors="replace"
                )
                return ToolResult(
                    success=False, output=None, error=f"audio playback failed: {stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error="Playback did not finish in time.")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(wav_path)

        word_count = len(text.split())
        return ToolResult(
            success=True,
            output=(
                f"Spoke {word_count} words with synced mouth movement "
                f"({duration:.2f}s, engine={engine}, voice={voice}, "
                f"output_sink={output_sink}, parameter_id={parameter_id})."
            ),
        )
