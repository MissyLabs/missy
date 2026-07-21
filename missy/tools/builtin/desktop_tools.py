"""Desktop session detection, safe app launch, and mouse-drag gap-fillers.

``x11_screenshot``, ``x11_click``, ``x11_type``, ``x11_key``, and
``x11_window_list`` (see ``missy/tools/builtin/x11_tools.py``) already
implement screenshot/click/type/hotkey/window-listing for X11 sessions --
this module only adds what those don't cover: session detection
(X11/Wayland/GNOME, and which automation binaries are actually present),
a standalone focus-window tool, mouse move/drag, and an *allowlisted,
confirmation-gated* app launcher.

Wayland limitation (documented, not silently papered over): ``xdotool``/
``wmctrl``/``scrot`` are X11-only. GNOME's Mutter compositor does not
implement the wlroots protocols ``wtype``/``ydotool`` need either, so
under GNOME on native Wayland there is currently no working click/type/key
path here -- :class:`DesktopStatusTool` reports this honestly rather than
claiming a capability that would silently fail. The documented future
path (not implemented in this phase) is GNOME's Remote Desktop portal
(``org.gnome.Mutter.RemoteDesktop`` over D-Bus) or falling back to X11 via
XWayland for legacy apps specifically.

Security model
---------------
- ``desktop.enabled`` must be explicitly set; :class:`DesktopLaunchAppTool`
  fails closed otherwise.
- :class:`DesktopLaunchAppTool` never uses a shell string -- it always
  execs an argv list directly (``subprocess.Popen([binary, *args], ...)``,
  no ``shell=True``), so there is no shell-metacharacter injection surface
  regardless of what arguments are requested.
- An app not on ``desktop.app_allowlist`` (and ``desktop.unrestricted`` not
  set) requires :class:`~missy.agent.approval.ApprovalGate` confirmation,
  failing closed when no gate is configured -- same posture as the
  ``obs_*``/``vtube_*`` confirmation gates.
- ``ToolPermissions(shell=True)`` is still declared (and the real binary
  reported via :meth:`resolve_shell_command`) so the existing global
  ``ShellPolicy`` allowlist also applies as defense in depth -- this is
  strictly *more* restrictive than bare ``shell_exec`` (arbitrary
  commands), not a bypass of it.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import load_missy_config, require_approval

logger = logging.getLogger(__name__)

#: Automation binaries relevant to desktop control, and which session type
#: each one targets (for DesktopStatusTool's capability report).
_CAPABILITY_BINARIES: dict[str, str] = {
    "xdotool": "x11",
    "wmctrl": "x11",
    "scrot": "x11",
    "wtype": "wayland",
    "ydotool": "wayland",
    "grim": "wayland",
    "slurp": "wayland",
}


def _display_env() -> dict[str, str]:
    """Return a minimal environment dict safe to pass to desktop subprocesses.

    Only passes through essential variables to avoid leaking secrets (API
    keys, tokens, etc.) into subprocess environments -- same allowlist
    convention as ``x11_tools.py``'s ``_display_env()``.
    """
    safe_vars = (
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "TERM",
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE",
        "XDG_CURRENT_DESKTOP",
        "XAUTHORITY",
        "DBUS_SESSION_BUS_ADDRESS",
        "WAYLAND_DISPLAY",
    )
    env = {k: os.environ[k] for k in safe_vars if k in os.environ}
    env.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))
    return env


def _desktop_config():
    """Return the configured :class:`~missy.config.settings.DesktopConfig`, or ``None``."""
    cfg = load_missy_config()
    return cfg.desktop if cfg is not None else None


class DesktopStatusTool(BaseTool):
    """Detect the desktop session type and which automation binaries are usable."""

    name = "desktop_status"
    description = (
        "Detect the current desktop session (X11/Wayland), the desktop "
        "environment, and which GUI automation capabilities are actually "
        "available. Call this before other desktop_*/x11_* tools to know "
        "what will work."
    )
    permissions = ToolPermissions()
    parameters: dict[str, Any] = {}

    def execute(self, **_: Any) -> ToolResult:
        session_type = os.environ.get("XDG_SESSION_TYPE", "unknown")
        desktop_env = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")
        has_display = bool(os.environ.get("DISPLAY"))
        has_wayland_display = bool(os.environ.get("WAYLAND_DISPLAY"))

        binaries = {name: shutil.which(name) is not None for name in _CAPABILITY_BINARIES}
        x11_capable = has_display and any(
            binaries[b] for b, kind in _CAPABILITY_BINARIES.items() if kind == "x11"
        )
        wayland_capable = has_wayland_display and any(
            binaries[b] for b, kind in _CAPABILITY_BINARIES.items() if kind == "wayland"
        )

        notes: list[str] = []
        if (
            session_type == "wayland"
            and desktop_env.lower().startswith("gnome")
            and not wayland_capable
        ):
            notes.append(
                "GNOME on native Wayland: xdotool/wmctrl/scrot won't work (X11-only), and "
                "wtype/ydotool need a wlroots compositor GNOME's Mutter doesn't provide. "
                "click/type/key/window-list/screenshot are not currently functional in this "
                "session. See desktop_tools.py's module docstring for the documented future "
                "path (GNOME Remote Desktop portal over D-Bus)."
            )
        elif session_type == "wayland" and has_display:
            notes.append(
                "Wayland session with an XWayland DISPLAY also present -- x11_* tools may "
                "work for XWayland-backed windows only, not native Wayland clients."
            )

        return ToolResult(
            success=True,
            output={
                "session_type": session_type,
                "desktop_environment": desktop_env,
                "has_x11_display": has_display,
                "has_wayland_display": has_wayland_display,
                "binaries_available": binaries,
                "x11_automation_usable": x11_capable,
                "wayland_automation_usable": wayland_capable,
                "notes": notes,
            },
        )


class DesktopFocusWindowTool(BaseTool):
    """Focus a window by name pattern (standalone; x11_click/type/key also accept this inline)."""

    name = "desktop_focus_window"
    description = "Focus a window by name pattern, bringing it to the foreground. X11 only."
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "window_name": {
            "type": "string",
            "description": "Window name pattern to search for and focus.",
            "required": True,
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "xdotool"

    def execute(self, *, window_name: str, **_: Any) -> ToolResult:
        result = subprocess.run(
            ["xdotool", "search", "--name", window_name, "windowfocus"],
            capture_output=True,
            text=True,
            env=_display_env(),
            timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "No such file or directory" in err or not shutil.which("xdotool"):
                return ToolResult(
                    success=False,
                    output=None,
                    error="xdotool is not installed. Install it with: sudo apt install xdotool",
                )
            return ToolResult(
                success=False,
                output=None,
                error=f"No window matching {window_name!r} found, or focus failed: {err}",
            )
        return ToolResult(success=True, output={"focused_window": window_name})


class DesktopMouseDragTool(BaseTool):
    """Press, drag, and release the mouse from one point to another. X11 only."""

    name = "desktop_mouse_drag"
    description = (
        "Drag the mouse from (start_x, start_y) to (end_x, end_y), e.g. to move a "
        "window, resize, or drag-select. X11 only."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "start_x": {"type": "integer", "description": "Starting X coordinate.", "required": True},
        "start_y": {"type": "integer", "description": "Starting Y coordinate.", "required": True},
        "end_x": {"type": "integer", "description": "Ending X coordinate.", "required": True},
        "end_y": {"type": "integer", "description": "Ending Y coordinate.", "required": True},
        "button": {
            "type": "string",
            "description": "Mouse button to hold during the drag: 'left', 'right', 'middle'.",
            "default": "left",
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "xdotool"

    def execute(
        self,
        *,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: str = "left",
        **_: Any,
    ) -> ToolResult:
        start_x, start_y, end_x, end_y = int(start_x), int(start_y), int(end_x), int(end_y)
        btn_num = {"left": "1", "middle": "2", "right": "3"}.get(button.lower(), "1")

        cmd = (
            f"xdotool mousemove {start_x} {start_y} "
            f"mousedown {btn_num} "
            f"mousemove {end_x} {end_y} "
            f"mouseup {btn_num}"
        )
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, env=_display_env(), timeout=15
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "command not found" in err or not shutil.which("xdotool"):
                return ToolResult(
                    success=False,
                    output=None,
                    error="xdotool is not installed. Install it with: sudo apt install xdotool",
                )
            return ToolResult(success=False, output=None, error=f"xdotool drag failed: {err}")

        return ToolResult(
            success=True,
            output={"start": [start_x, start_y], "end": [end_x, end_y], "button": button},
        )


class DesktopLaunchAppTool(BaseTool):
    """Launch a GUI application under the current desktop session.

    Never shells out via a shell string -- always an argv exec. Requires
    approval when the app isn't on ``desktop.app_allowlist`` (unless
    ``desktop.unrestricted`` is set). See module docstring for the full
    security model.
    """

    name = "desktop_launch_app"
    description = (
        "Launch a GUI application by binary name under the current desktop "
        "session's DISPLAY/D-Bus environment. Apps not on the configured "
        "allowlist require human approval before launching."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "app": {
            "type": "string",
            "description": "Executable name or absolute path, e.g. 'firefox', 'obs'.",
            "required": True,
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional command-line arguments.",
            "default": [],
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        app = str(kwargs.get("app", "")).strip()
        return app or "true"

    def execute(self, *, app: str, args: list[str] | None = None, **_: Any) -> ToolResult:
        args = list(args or [])
        config = _desktop_config()
        if config is None or not config.enabled:
            return ToolResult(
                success=False,
                output=None,
                error="Desktop app launching is disabled. Set desktop.enabled: true in config.yaml.",
            )

        app_binary = os.path.basename(app)
        allowlisted = config.unrestricted or app_binary in config.app_allowlist
        if not allowlisted:
            denial = require_approval(
                action=f"Launch desktop app: {app}" + (f" {shlex.join(args)}" if args else ""),
                reason=f"{app_binary!r} is not on desktop.app_allowlist.",
                risk="medium",
            )
            if denial:
                return ToolResult(success=False, output=None, error=denial)

        resolved = shutil.which(app) or (
            app if os.path.isabs(app) and os.path.exists(app) else None
        )
        if resolved is None:
            return ToolResult(success=False, output=None, error=f"Executable not found: {app!r}")

        try:
            proc = subprocess.Popen(
                [resolved, *args],
                env=_display_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return ToolResult(success=False, output=None, error=f"Failed to launch {app!r}: {exc}")

        return ToolResult(
            success=True,
            output={"app": app, "resolved_path": resolved, "pid": proc.pid, "args": args},
        )
