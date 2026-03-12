"""Built-in tool: launch an X11 application and wait for its window."""
from __future__ import annotations

import os
import subprocess
import time
from missy.tools.base import BaseTool, ToolPermissions, ToolResult


def _display_env() -> dict:
    return {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}


class X11LaunchTool(BaseTool):
    """Launch an X11 application and wait for its window to appear."""

    name = "x11_launch"
    description = (
        "Launch an X11 application and wait for its window to appear. "
        "Returns the window ID and name so you can target it with x11_click. "
        "Example: command='firefox', command='gedit /tmp/file.txt', command='gnome-calculator'."
    )
    permissions = ToolPermissions(shell=True)
    parameters = {
        "command": {
            "type": "string",
            "description": "The application command to run, e.g. 'firefox' or 'gedit /tmp/notes.txt'.",
            "required": True,
        },
        "window_name_hint": {
            "type": "string",
            "description": "Partial window name to wait for, e.g. 'Firefox' or 'gedit'. "
                           "Defaults to the first word of the command.",
        },
        "wait_seconds": {
            "type": "integer",
            "description": "How long to wait for the window to appear (default 10, max 30).",
        },
    }

    def execute(
        self,
        *,
        command: str,
        window_name_hint: str = "",
        wait_seconds: int = 10,
        **_kwargs,
    ) -> ToolResult:
        if not command.strip():
            return ToolResult(success=False, output=None, error="command must not be empty")

        wait_seconds = min(int(wait_seconds), 30)
        hint = window_name_hint.strip() or command.split()[0]
        env = _display_env()

        # Launch the app detached from this process.
        try:
            subprocess.Popen(
                command,
                shell=True,
                executable="/bin/bash",
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            return ToolResult(success=False, output=None, error=f"Failed to launch: {exc}")

        # Poll for the window to appear.
        deadline = time.time() + wait_seconds
        window_id = ""
        window_name = ""
        while time.time() < deadline:
            time.sleep(0.5)
            result = subprocess.run(
                f"xdotool search --onlyvisible --name {hint!r}",
                shell=True,
                capture_output=True,
                text=True,
                env=env,
            )
            ids = result.stdout.strip().splitlines()
            if ids:
                window_id = ids[0].strip()
                # Get the window name.
                name_result = subprocess.run(
                    f"xdotool getwindowname {window_id}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                window_name = name_result.stdout.strip()
                break

        if window_id:
            return ToolResult(
                success=True,
                output=f"Launched '{command}'. Window ID: {window_id}, Name: '{window_name}'",
                error=None,
            )
        else:
            return ToolResult(
                success=True,
                output=(
                    f"Launched '{command}' but no window matching '{hint}' appeared "
                    f"within {wait_seconds}s. App may still be loading — "
                    f"try x11_window_list to find it."
                ),
                error=None,
            )
