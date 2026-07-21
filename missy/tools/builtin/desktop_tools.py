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
- ``desktop.enabled`` must be explicitly set; :class:`DesktopLaunchAppTool`/
  :class:`InstallSoftwareConfirmedTool` fail closed otherwise.
- :class:`DesktopLaunchAppTool`/:class:`InstallSoftwareConfirmedTool` never
  use a shell string -- both always exec an argv list directly
  (``subprocess.Popen([binary, *args], ...)``/``subprocess.run([...])``, no
  ``shell=True``), so there is no shell-metacharacter injection surface
  regardless of what arguments are requested.
- An app not on ``desktop.app_allowlist`` (and ``desktop.unrestricted`` not
  set) requires :class:`~missy.agent.approval.ApprovalGate` confirmation,
  failing closed when no gate is configured -- same posture as the
  ``obs_*``/``vtube_*`` confirmation gates. A window name not on
  ``desktop.window_allowlist`` requires the same for
  :class:`DesktopFocusWindowTool`.
  :class:`InstallSoftwareConfirmedTool` *always* requires approval
  (no allowlist bypass -- same posture as OBS's streaming start/stop) and
  additionally needs ``desktop.allow_software_install: true``, since
  installing packages is a materially larger blast radius than launching
  an already-installed GUI app.
- ``ToolPermissions(shell=True)`` is still declared (and the real binary
  reported via :meth:`resolve_shell_command`) so the existing global
  ``ShellPolicy`` allowlist also applies as defense in depth -- this is
  strictly *more* restrictive than bare ``shell_exec`` (arbitrary
  commands), not a bypass of it.
- Every tool in this module checks ``desktop.rate_limit_per_minute``
  (default 30/tool/60s) before doing anything else, as a guardrail against
  a runaway loop hammering desktop actions.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Any

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.builtin._desktop_shared import (
    check_rate_limit,
    check_window_allowed,
    load_missy_config,
    require_approval,
)

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


def _probe_x11_display() -> bool:
    """Return True if an X11 display is actually reachable right now.

    Every *other* tool in this module (and x11_tools.py) runs commands
    against ``_display_env()``'s DISPLAY, which defaults to ``:0`` when
    the env var is unset -- so an unset ``DISPLAY`` does NOT mean X11
    automation is unavailable, it means the default target is in play.
    A bare ``bool(os.environ.get("DISPLAY"))`` check would therefore
    misreport a perfectly working ":0" default session as unavailable
    (observed in practice: a gateway process launched from an SSH/screen
    session with no DISPLAY of its own, automating a real GNOME session
    on :0). Actually probing connectivity -- rather than trusting either
    the raw env var or the default -- is the only way to report this
    honestly in both directions.
    """
    if shutil.which("xdotool") is None:
        return False
    try:
        result = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            capture_output=True,
            text=True,
            env=_display_env(),
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def _desktop_config():
    """Return the configured :class:`~missy.config.settings.DesktopConfig`, or ``None``."""
    cfg = load_missy_config()
    return cfg.desktop if cfg is not None else None


def _check_rate_limit(tool_name: str) -> str | None:
    """Rate-limit *tool_name* against ``desktop.rate_limit_per_minute``.

    Falls back to the field's default (30/min) when no config loads at
    all, consistent with :func:`~missy.tools.builtin._desktop_shared.check_window_allowed`
    treating an unloadable config as the safe default rather than "no limit."
    """
    from missy.config.settings import DesktopConfig

    config = _desktop_config() or DesktopConfig()
    return check_rate_limit(tool_name, config.rate_limit_per_minute)


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
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        session_type = os.environ.get("XDG_SESSION_TYPE", "unknown")
        desktop_env = os.environ.get("XDG_CURRENT_DESKTOP", "unknown")
        has_display = _probe_x11_display()
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
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)
        if window_error := check_window_allowed(window_name):
            return ToolResult(success=False, output=None, error=window_error)

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
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

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
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

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


class DesktopMouseMoveTool(BaseTool):
    """Move the mouse cursor to an x,y coordinate without clicking. X11 only."""

    name = "desktop_mouse_move"
    description = (
        "Move the mouse cursor to an x,y coordinate without clicking, e.g. to "
        "hover for a tooltip or reposition before a separate click. X11 only."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "x": {"type": "integer", "description": "X coordinate.", "required": True},
        "y": {"type": "integer", "description": "Y coordinate.", "required": True},
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "xdotool"

    def execute(self, *, x: int, y: int, **_: Any) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        x, y = int(x), int(y)
        result = subprocess.run(
            ["xdotool", "mousemove", str(x), str(y)],
            capture_output=True,
            text=True,
            env=_display_env(),
            timeout=10,
        )
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if "command not found" in err or not shutil.which("xdotool"):
                return ToolResult(
                    success=False,
                    output=None,
                    error="xdotool is not installed. Install it with: sudo apt install xdotool",
                )
            return ToolResult(success=False, output=None, error=f"xdotool mousemove failed: {err}")

        return ToolResult(success=True, output={"x": x, "y": y})


#: Package names/specs must look like real apt package identifiers -- this
#: is defense in depth, not the injection boundary itself (argv exec already
#: means no shell metacharacter can do anything): it stops a value that's
#: obviously not a package name (e.g. a stray "-y" or "--reinstall" flag
#: smuggled in as the "package") from being accepted as one, since argv
#: position alone doesn't stop apt from interpreting a leading-dash argument
#: as a flag rather than a package name.
_PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9+.\-]*(=[a-zA-Z0-9+.\-:~]+)?$")


class InstallSoftwareConfirmedTool(BaseTool):
    """Install an apt package. ALWAYS requires approval; needs a separate opt-in.

    Distinct from :class:`DesktopLaunchAppTool`'s allowlist model: there is
    no "trusted package" allowlist here, because unlike launching an
    already-installed binary, installing a *new* package is inherently
    supply-chain-sensitive (a typo-squatted or malicious package name is a
    real risk apt itself won't catch). Every call requires human approval
    regardless of what's being installed, mirroring
    ``obs_start_streaming_confirmed``'s no-bypass posture, and additionally
    needs ``desktop.allow_software_install: true`` (a separate opt-in from
    ``desktop.enabled``) before it's reachable at all.
    """

    name = "install_software_confirmed"
    description = (
        "Install a package via apt. ALWAYS requires human approval -- there is no "
        "allowlist bypass for this action, and it must be separately enabled via "
        "desktop.allow_software_install in config.yaml."
    )
    permissions = ToolPermissions(shell=True)
    parameters: dict[str, Any] = {
        "package": {
            "type": "string",
            "description": "apt package name, e.g. 'obs-studio' (optionally 'name=version').",
            "required": True,
        },
    }

    def resolve_shell_command(self, kwargs: dict[str, Any]) -> str:
        return "sudo && apt-get"

    def execute(self, *, package: str, **_: Any) -> ToolResult:
        if rate_error := _check_rate_limit(self.name):
            return ToolResult(success=False, output=None, error=rate_error)

        config = _desktop_config()
        if config is None or not config.enabled or not config.allow_software_install:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    "Software installation is disabled. Set both desktop.enabled: true "
                    "and desktop.allow_software_install: true in config.yaml."
                ),
            )

        package = package.strip()
        if not _PACKAGE_NAME_RE.match(package):
            return ToolResult(
                success=False,
                output=None,
                error=f"{package!r} doesn't look like a valid apt package name/spec.",
            )

        denial = require_approval(
            action=f"Install software package: {package}",
            reason="Package installation always requires approval regardless of allowlists.",
            risk="high",
        )
        if denial:
            return ToolResult(success=False, output=None, error=denial)

        try:
            result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", package],
                capture_output=True,
                text=True,
                env=_display_env(),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, output=None, error=f"Installation of {package!r} timed out."
            )

        if result.returncode != 0:
            err = (result.stderr.strip() or result.stdout.strip())[-2000:]
            return ToolResult(
                success=False, output=None, error=f"Installation of {package!r} failed: {err}"
            )

        return ToolResult(
            success=True,
            output={"package": package, "log_tail": result.stdout.strip()[-1000:]},
        )
