"""OBS Studio integration for Missy via the obs-websocket v5 protocol.

Talks directly to OBS's built-in WebSocket server (obs-websocket v5,
bundled with OBS Studio 28+) -- no extra OBS plugin needed beyond enabling
"WebSocket Server" in OBS's Tools menu. Uses the ``websockets`` package
already a core Missy dependency, so no new required dependency is added.

Protocol (see https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md)::

    connect -> server sends Hello (op 0, may include an auth challenge)
    client sends Identify (op 1, with computed auth response if required)
    server sends Identified (op 2)
    client sends Request (op 6) -> server sends RequestResponse (op 7)

Each tool call opens a fresh connection, does exactly one request/response,
and closes -- no persistent connection is kept between calls, mirroring how
other Missy tools (e.g. ``video_generate``) treat an external service as
stateless per call.

Security model
---------------
- The OBS password never appears in any :class:`~missy.tools.base.ToolResult`
  output, log line, or audit event -- only used transiently to compute the
  auth response hash for the handshake.
- ``obs.enabled`` must be explicitly set in config; every tool here fails
  closed (denied) otherwise, matching ``ShellPolicy.enabled``'s contract.
- ``obs_switch_scene`` only requires approval when the target scene is
  outside ``obs.scene_allowlist`` (empty allowlist = every scene allowed).
- ``obs_start_streaming_confirmed``/``obs_stop_streaming_confirmed`` ALWAYS
  require :class:`~missy.agent.approval.ApprovalGate` confirmation --
  unconditionally, with no allowlist bypass -- and fail closed (deny) when
  no gate is configured for the session, mirroring
  ``McpManager``'s ``requires_approval`` handling exactly (see
  ``missy/mcp/manager.py``). Stream keys are never touched by Missy at
  all: OBS holds them internally, and no obs-websocket request used here
  ever returns one.
- The real OBS host is declared via :meth:`resolve_network_hosts` so the
  network policy engine enforces the actual configured target, not just a
  static declaration.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import uuid
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import load_missy_config, require_approval

logger = logging.getLogger(__name__)

_WS_TIMEOUT_SECONDS = 10.0


class ObsError(Exception):
    """Raised for any obs-websocket connection, auth, or request failure."""


def _obs_config():
    """Return the configured :class:`~missy.config.settings.ObsConfig`, or ``None``."""
    cfg = load_missy_config()
    return cfg.obs if cfg is not None else None


# ---------------------------------------------------------------------------
# obs-websocket v5 client
# ---------------------------------------------------------------------------


def _compute_auth_response(password: str, challenge: str, salt: str) -> str:
    """Compute the obs-websocket v5 authentication response string.

    Per the protocol spec::

        base64_secret = base64(sha256(password + salt))
        auth_response = base64(sha256(base64_secret + challenge))
    """
    base64_secret = base64.b64encode(
        hashlib.sha256((password + salt).encode("utf-8")).digest()
    ).decode("utf-8")
    auth_response = base64.b64encode(
        hashlib.sha256((base64_secret + challenge).encode("utf-8")).digest()
    ).decode("utf-8")
    return auth_response


async def _obs_request_async(
    request_type: str,
    request_data: dict[str, Any] | None,
    *,
    host: str,
    port: int,
    password: str | None,
) -> dict[str, Any]:
    """Open a connection, perform one obs-websocket request, and close.

    Returns the ``responseData`` dict (``{}`` if the request has none).

    Raises:
        ObsError: On connection failure, auth failure, or a non-success
            ``requestStatus``.
    """
    import json

    import websockets

    uri = f"ws://{host}:{port}"
    try:
        async with asyncio.timeout(_WS_TIMEOUT_SECONDS):
            async with websockets.connect(uri, max_size=8 * 1024 * 1024) as ws:
                hello_raw = await ws.recv()
                hello = json.loads(hello_raw)
                hello_data = hello.get("d", {})

                identify: dict[str, Any] = {"rpcVersion": 1, "eventSubscriptions": 0}
                auth = hello_data.get("authentication")
                if auth:
                    if not password:
                        raise ObsError(
                            "OBS requires a password but none is configured "
                            "(set obs.password in config.yaml, e.g. vault://obs_password)."
                        )
                    identify["authentication"] = _compute_auth_response(
                        password, auth["challenge"], auth["salt"]
                    )

                await ws.send(json.dumps({"op": 1, "d": identify}))
                identified_raw = await ws.recv()
                identified = json.loads(identified_raw)
                if identified.get("op") != 2:
                    raise ObsError(
                        f"Unexpected response during identify (op={identified.get('op')}); "
                        "check the OBS password."
                    )

                request_id = str(uuid.uuid4())
                await ws.send(
                    json.dumps(
                        {
                            "op": 6,
                            "d": {
                                "requestType": request_type,
                                "requestId": request_id,
                                "requestData": request_data or {},
                            },
                        }
                    )
                )
                response_raw = await ws.recv()
                response = json.loads(response_raw)
                d = response.get("d", {})
                status = d.get("requestStatus", {})
                if not status.get("result"):
                    raise ObsError(
                        f"OBS request {request_type!r} failed "
                        f"(code={status.get('code')}): {status.get('comment', 'no detail')}"
                    )
                return d.get("responseData", {}) or {}
    except TimeoutError as exc:
        raise ObsError(f"Timed out talking to OBS at {uri} after {_WS_TIMEOUT_SECONDS}s") from exc
    except OSError as exc:
        raise ObsError(f"Could not connect to OBS at {uri}: {exc}") from exc


def _obs_request(request_type: str, request_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Sync wrapper: validate config/policy, then perform one OBS request.

    Raises:
        ObsError: When OBS integration is disabled, or on any connection/
            request failure (see :func:`_obs_request_async`).
    """
    config = _obs_config()
    if config is None or not config.enabled:
        raise ObsError(
            "OBS integration is disabled. Set obs.enabled: true in config.yaml "
            "(see docs/desktop_obs_vtube.md for setup)."
        )

    from missy.policy.engine import get_policy_engine

    get_policy_engine().check_network(config.host, category="tool")

    return asyncio.run(
        _obs_request_async(
            request_type,
            request_data,
            host=config.host,
            port=config.port,
            password=config.password,
        )
    )


def _obs_host() -> str:
    """Return the configured OBS host, or ``""`` if unconfigured."""
    config = _obs_config()
    return config.host if config is not None else ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ObsStatusTool(BaseTool):
    """Report OBS connection, current scene, and streaming/recording state."""

    name = "obs_status"
    description = (
        "Get OBS Studio's current status: connection health, active scene, "
        "and whether streaming/recording is active. Read-only."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        try:
            version = _obs_request("GetVersion")
            scene = _obs_request("GetCurrentProgramScene")
            stream_status = _obs_request("GetStreamStatus")
            record_status = _obs_request("GetRecordStatus")
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(
            success=True,
            output={
                "connected": True,
                "obs_version": version.get("obsVersion"),
                "current_scene": scene.get("currentProgramSceneName"),
                "streaming": bool(stream_status.get("outputActive")),
                "stream_timecode": stream_status.get("outputTimecode"),
                "recording": bool(record_status.get("outputActive")),
                "recording_paused": bool(record_status.get("outputPaused")),
                "record_timecode": record_status.get("outputTimecode"),
            },
        )


class ObsListScenesTool(BaseTool):
    """List every scene and, for the current scene, its sources."""

    name = "obs_list_scenes"
    description = (
        "List all OBS scenes and the current scene's sources. Read-only. "
        "Use this to find valid scene/source names for other obs_* tools."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        try:
            scene_list = _obs_request("GetSceneList")
            current = scene_list.get("currentProgramSceneName", "")
            items = _obs_request("GetSceneItemList", {"sceneName": current}) if current else {}
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        scenes = [s.get("sceneName") for s in scene_list.get("scenes", [])]
        sources = [
            {
                "name": item.get("sourceName"),
                "visible": item.get("sceneItemEnabled"),
                "id": item.get("sceneItemId"),
            }
            for item in items.get("sceneItems", [])
        ]
        return ToolResult(
            success=True,
            output={"scenes": scenes, "current_scene": current, "current_scene_sources": sources},
        )


class ObsSwitchSceneTool(BaseTool):
    """Switch OBS's active program scene.

    Requires approval when the target scene is outside
    ``obs.scene_allowlist`` (empty allowlist = every scene allowed).
    """

    name = "obs_switch_scene"
    description = (
        "Switch the active OBS scene by name. Requires human approval if the "
        "scene isn't on the configured allowlist."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {
        "scene_name": {
            "type": "string",
            "description": "Exact scene name (see obs_list_scenes).",
            "required": True,
        },
    }

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, *, scene_name: str, **_: Any) -> ToolResult:
        config = _obs_config()
        if (
            config is not None
            and config.scene_allowlist
            and scene_name not in config.scene_allowlist
        ):
            denial = require_approval(
                action=f"Switch OBS scene to {scene_name!r}",
                reason="Scene is not on the configured obs.scene_allowlist.",
                risk="medium",
            )
            if denial:
                return ToolResult(success=False, output=None, error=denial)

        try:
            _obs_request("SetCurrentProgramScene", {"sceneName": scene_name})
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(success=True, output={"current_scene": scene_name})


class ObsSetSourceVisibilityTool(BaseTool):
    """Show or hide a source within a given scene."""

    name = "obs_set_source_visibility"
    description = "Show or hide a source within a given OBS scene."
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {
        "scene_name": {
            "type": "string",
            "description": "Scene containing the source (see obs_list_scenes).",
            "required": True,
        },
        "source_name": {
            "type": "string",
            "description": "Name of the source within that scene.",
            "required": True,
        },
        "visible": {
            "type": "boolean",
            "description": "True to show, False to hide.",
            "required": True,
        },
    }

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, *, scene_name: str, source_name: str, visible: bool, **_: Any) -> ToolResult:
        try:
            items = _obs_request("GetSceneItemList", {"sceneName": scene_name})
            item_id = next(
                (
                    i["sceneItemId"]
                    for i in items.get("sceneItems", [])
                    if i.get("sourceName") == source_name
                ),
                None,
            )
            if item_id is None:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Source {source_name!r} not found in scene {scene_name!r}.",
                )
            _obs_request(
                "SetSceneItemEnabled",
                {
                    "sceneName": scene_name,
                    "sceneItemId": item_id,
                    "sceneItemEnabled": bool(visible),
                },
            )
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))

        return ToolResult(
            success=True,
            output={"scene_name": scene_name, "source_name": source_name, "visible": bool(visible)},
        )


class ObsStartRecordingTool(BaseTool):
    """Start local OBS recording (no public-facing risk; no approval required)."""

    name = "obs_start_recording"
    description = "Start local OBS recording."
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        try:
            _obs_request("StartRecord")
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
        return ToolResult(success=True, output={"recording": True})


class ObsStopRecordingTool(BaseTool):
    """Stop local OBS recording."""

    name = "obs_stop_recording"
    description = "Stop local OBS recording."
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        try:
            result = _obs_request("StopRecord")
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
        return ToolResult(
            success=True, output={"recording": False, "output_path": result.get("outputPath")}
        )


class ObsStartStreamingConfirmedTool(BaseTool):
    """Go live on OBS's configured streaming target -- ALWAYS requires approval.

    Stream keys are never touched here: OBS holds and uses its own
    configured stream destination/key internally. This tool only issues
    obs-websocket's ``StartStream`` request, which contains and returns no
    credential material.
    """

    name = "obs_start_streaming_confirmed"
    description = (
        "Start OBS streaming (go live) using OBS's own configured stream "
        "destination. ALWAYS requires human approval before executing -- "
        "there is no allowlist bypass for this action."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        denial = require_approval(
            action="Start OBS streaming (go live)",
            reason="Public-facing action; always confirmed regardless of any allowlist.",
            risk="high",
        )
        if denial:
            return ToolResult(success=False, output=None, error=denial)

        try:
            _obs_request("StartStream")
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
        return ToolResult(success=True, output={"streaming": True})


class ObsStopStreamingConfirmedTool(BaseTool):
    """Stop OBS streaming -- ALWAYS requires approval."""

    name = "obs_stop_streaming_confirmed"
    description = (
        "Stop OBS streaming. ALWAYS requires human approval before executing "
        "-- there is no allowlist bypass for this action."
    )
    permissions = ToolPermissions(network=True)
    parameters: dict[str, Any] = {}

    def resolve_network_hosts(self, kwargs: dict[str, Any]) -> list[str]:
        host = _obs_host()
        return [host] if host else []

    def execute(self, **_: Any) -> ToolResult:
        denial = require_approval(
            action="Stop OBS streaming",
            reason="Public-facing action; always confirmed regardless of any allowlist.",
            risk="high",
        )
        if denial:
            return ToolResult(success=False, output=None, error=denial)

        try:
            _obs_request("StopStream")
        except ObsError as exc:
            return ToolResult(success=False, output=None, error=str(exc))
        return ToolResult(success=True, output={"streaming": False})
