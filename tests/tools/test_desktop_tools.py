"""Tests for missy.tools.builtin.desktop_tools.

DesktopLaunchAppTool gets the deepest coverage here: it's the one tool in
this module that spawns a real host process, so its argv-only-exec
(never a shell string), allowlist, and fail-closed-approval behavior are
all safety-relevant and get tested directly rather than through
end-to-end subprocess mocking alone.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import DesktopConfig, MissyConfig
from missy.tools.builtin.desktop_tools import (
    DesktopFocusWindowTool,
    DesktopLaunchAppTool,
    DesktopMouseDragTool,
    DesktopStatusTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_missy_config(desktop: DesktopConfig) -> MissyConfig:
    return MagicMock(desktop=desktop)


def _mock_config(**overrides) -> DesktopConfig:
    defaults = {"enabled": True, "app_allowlist": [], "unrestricted": False}
    defaults.update(overrides)
    return DesktopConfig(**defaults)


# ---------------------------------------------------------------------------
# DesktopStatusTool
# ---------------------------------------------------------------------------


class TestDesktopStatusTool:
    def test_detects_x11_session(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        with patch(
            "shutil.which",
            side_effect=lambda b: f"/usr/bin/{b}" if b in ("xdotool", "wmctrl", "scrot") else None,
        ):
            result = DesktopStatusTool().execute()

        assert result.success is True
        assert result.output["session_type"] == "x11"
        assert result.output["has_x11_display"] is True
        assert result.output["x11_automation_usable"] is True
        assert result.output["wayland_automation_usable"] is False
        assert result.output["notes"] == []

    def test_detects_gnome_wayland_with_no_working_automation(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        with patch("shutil.which", return_value=None):
            result = DesktopStatusTool().execute()

        assert result.success is True
        assert result.output["session_type"] == "wayland"
        assert result.output["x11_automation_usable"] is False
        assert result.output["wayland_automation_usable"] is False
        assert any("GNOME" in note for note in result.output["notes"])

    def test_wayland_with_xwayland_display_present(self, monkeypatch):
        # A non-GNOME desktop so this hits the generic Wayland+DISPLAY note
        # rather than the GNOME-specific one covered above.
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        with patch(
            "shutil.which", side_effect=lambda b: f"/usr/bin/{b}" if b == "xdotool" else None
        ):
            result = DesktopStatusTool().execute()

        assert result.output["x11_automation_usable"] is True
        assert any("XWayland" in note for note in result.output["notes"])

    def test_binaries_available_reports_each_checked_binary(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with patch("shutil.which", return_value=None):
            result = DesktopStatusTool().execute()
        assert set(result.output["binaries_available"]) == {
            "xdotool",
            "wmctrl",
            "scrot",
            "wtype",
            "ydotool",
            "grim",
            "slurp",
        }

    def test_requires_no_permissions(self):
        assert DesktopStatusTool().permissions.shell is False
        assert DesktopStatusTool().permissions.network is False


# ---------------------------------------------------------------------------
# DesktopFocusWindowTool
# ---------------------------------------------------------------------------


class TestDesktopFocusWindowTool:
    def test_focuses_matching_window(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = DesktopFocusWindowTool().execute(window_name="Firefox")

        assert result.success is True
        assert result.output["focused_window"] == "Firefox"
        args = mock_run.call_args[0][0]
        assert args == ["xdotool", "search", "--name", "Firefox", "windowfocus"]

    def test_xdotool_missing_reports_install_hint(self):
        with (
            patch("subprocess.run") as mock_run,
            patch("shutil.which", return_value=None),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
            result = DesktopFocusWindowTool().execute(window_name="Firefox")

        assert result.success is False
        assert "xdotool" in result.error.lower()

    def test_no_matching_window_returns_error(self):
        with (
            patch("subprocess.run") as mock_run,
            patch("shutil.which", return_value="/usr/bin/xdotool"),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
            result = DesktopFocusWindowTool().execute(window_name="Ghost")

        assert result.success is False


# ---------------------------------------------------------------------------
# DesktopMouseDragTool
# ---------------------------------------------------------------------------


class TestDesktopMouseDragTool:
    def test_drags_from_start_to_end(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = DesktopMouseDragTool().execute(start_x=10, start_y=20, end_x=100, end_y=200)

        assert result.success is True
        assert result.output["start"] == [10, 20]
        assert result.output["end"] == [100, 200]
        cmd = mock_run.call_args[0][0]
        assert "mousemove 10 20" in cmd
        assert "mousedown 1" in cmd
        assert "mousemove 100 200" in cmd
        assert "mouseup 1" in cmd

    def test_right_button_drag(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            DesktopMouseDragTool().execute(start_x=0, start_y=0, end_x=1, end_y=1, button="right")
        cmd = mock_run.call_args[0][0]
        assert "mousedown 3" in cmd
        assert "mouseup 3" in cmd

    def test_failure_reports_error(self):
        with (
            patch("subprocess.run") as mock_run,
            patch("shutil.which", return_value="/usr/bin/xdotool"),
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="boom")
            result = DesktopMouseDragTool().execute(start_x=0, start_y=0, end_x=1, end_y=1)
        assert result.success is False
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# DesktopLaunchAppTool -- the security-sensitive one
# ---------------------------------------------------------------------------


class TestDesktopLaunchAppToolGating:
    def test_disabled_fails_closed(self):
        with patch(
            "missy.tools.builtin.desktop_tools._desktop_config",
            return_value=_mock_config(enabled=False),
        ):
            result = DesktopLaunchAppTool().execute(app="firefox")
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_no_config_fails_closed(self):
        with patch("missy.tools.builtin.desktop_tools._desktop_config", return_value=None):
            result = DesktopLaunchAppTool().execute(app="firefox")
        assert result.success is False

    def test_allowlisted_app_skips_approval(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(app_allowlist=["firefox"]),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval") as mock_approval,
            patch("shutil.which", return_value="/usr/bin/firefox"),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value = MagicMock(pid=1234)
            result = DesktopLaunchAppTool().execute(app="firefox")

        assert result.success is True
        mock_approval.assert_not_called()

    def test_non_allowlisted_app_requires_approval(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(app_allowlist=["firefox"]),
            ),
            patch(
                "missy.tools.builtin.desktop_tools.require_approval",
                return_value="denied by operator",
            ) as mock_approval,
        ):
            result = DesktopLaunchAppTool().execute(app="gimp")

        assert result.success is False
        assert "denied" in result.error
        mock_approval.assert_called_once()

    def test_approved_non_allowlisted_app_launches(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(app_allowlist=["firefox"]),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval", return_value=None),
            patch("shutil.which", return_value="/usr/bin/gimp"),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value = MagicMock(pid=5678)
            result = DesktopLaunchAppTool().execute(app="gimp")

        assert result.success is True
        assert result.output["pid"] == 5678

    def test_unrestricted_skips_allowlist_and_approval(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(app_allowlist=[], unrestricted=True),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval") as mock_approval,
            patch("shutil.which", return_value="/usr/bin/anything"),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value = MagicMock(pid=1)
            result = DesktopLaunchAppTool().execute(app="anything")

        assert result.success is True
        mock_approval.assert_not_called()

    def test_empty_allowlist_and_not_unrestricted_requires_approval_for_everything(self):
        """Fail-closed default: no allowlist configured means every launch
        needs a human, mirroring ShellPolicy's empty-allow-list contract."""
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(app_allowlist=[], unrestricted=False),
            ),
            patch(
                "missy.tools.builtin.desktop_tools.require_approval",
                return_value="no gate configured",
            ) as mock_approval,
        ):
            result = DesktopLaunchAppTool().execute(app="firefox")

        assert result.success is False
        mock_approval.assert_called_once()

    def test_executable_not_found_returns_error(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(unrestricted=True),
            ),
            patch("shutil.which", return_value=None),
        ):
            result = DesktopLaunchAppTool().execute(app="nonexistent-binary")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestDesktopLaunchAppToolExecArgv:
    def test_never_uses_a_shell_string(self):
        """subprocess.Popen must be called with an argv list, never shell=True."""
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(unrestricted=True),
            ),
            patch("shutil.which", return_value="/usr/bin/firefox"),
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value = MagicMock(pid=1)
            DesktopLaunchAppTool().execute(app="firefox", args=["--private-window"])

        call_args, call_kwargs = mock_popen.call_args
        assert isinstance(call_args[0], list)
        assert call_args[0] == ["/usr/bin/firefox", "--private-window"]
        assert call_kwargs.get("shell", False) is False

    def test_launch_failure_reports_oserror(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(unrestricted=True),
            ),
            patch("shutil.which", return_value="/usr/bin/firefox"),
            patch("subprocess.Popen", side_effect=OSError("permission denied")),
        ):
            result = DesktopLaunchAppTool().execute(app="firefox")

        assert result.success is False
        assert "permission denied" in result.error

    def test_resolve_shell_command_reports_app_name(self):
        tool = DesktopLaunchAppTool()
        assert tool.resolve_shell_command({"app": "firefox"}) == "firefox"

    def test_resolve_shell_command_defaults_when_no_app(self):
        tool = DesktopLaunchAppTool()
        assert tool.resolve_shell_command({}) == "true"


# ---------------------------------------------------------------------------
# Permission declarations
# ---------------------------------------------------------------------------


class TestDesktopToolPermissions:
    @pytest.mark.parametrize(
        "tool_cls", [DesktopFocusWindowTool, DesktopMouseDragTool, DesktopLaunchAppTool]
    )
    def test_declares_shell_permission(self, tool_cls):
        assert tool_cls().permissions.shell is True
