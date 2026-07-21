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
Real per-frame audio-synced lip flap is also not attempted here: VTube
Studio has its own built-in microphone-based lip sync, and pointing it at
a PipeWire virtual sink that Missy's TTS output is routed into (see
``audio_route_tts`` in ``audio_route.py``) is far more reliable than Missy
computing and streaming synthetic ``InjectParameterDataRequest`` calls in
realtime against TTS playback timing. :class:`VtubeSetParameterTool` still
exists for simple scripted/discrete puppeting (e.g. one-shot expression
parameters), just not true audio-synced mouth flap.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import load_missy_config

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

    Not intended for real-time audio-synced mouth movement -- see the
    module docstring's "Known VTube Studio API gap" section for why
    routing TTS audio into VTube Studio's own built-in lip sync (via
    ``audio_route_tts``) is the recommended approach for that instead.
    """

    name = "vtube_set_parameter"
    description = (
        "Set a Live2D model parameter's value (e.g. 'MouthOpen', 'MouthSmile', "
        "'EyeOpenLeft'). For continuous audio-synced lip sync, use VTube Studio's "
        "own microphone lip sync fed by audio_route_tts instead of scripting this "
        "per-frame."
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
