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
    DesktopMouseMoveTool,
    DesktopStatusTool,
    InstallSoftwareConfirmedTool,
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
        with (
            patch(
                "shutil.which",
                side_effect=lambda b: (
                    f"/usr/bin/{b}" if b in ("xdotool", "wmctrl", "scrot") else None
                ),
            ),
            patch("missy.tools.builtin.desktop_tools._probe_x11_display", return_value=True),
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
        with (
            patch("shutil.which", return_value=None),
            patch("missy.tools.builtin.desktop_tools._probe_x11_display", return_value=False),
        ):
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
        with (
            patch(
                "shutil.which", side_effect=lambda b: f"/usr/bin/{b}" if b == "xdotool" else None
            ),
            patch("missy.tools.builtin.desktop_tools._probe_x11_display", return_value=True),
        ):
            result = DesktopStatusTool().execute()

        assert result.output["x11_automation_usable"] is True
        assert any("XWayland" in note for note in result.output["notes"])

    def test_binaries_available_reports_each_checked_binary(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        with (
            patch("shutil.which", return_value=None),
            patch("missy.tools.builtin.desktop_tools._probe_x11_display", return_value=False),
        ):
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

    def test_display_env_var_unset_but_probe_succeeds_is_still_usable(self, monkeypatch):
        """Regression: a gateway process launched with no DISPLAY of its
        own (e.g. from an SSH/screen session) must not be misreported as
        automation-unusable when the default ':0' target is actually
        reachable -- this is exactly the discrepancy that made a real
        desktop_status call report x11_automation_usable=False while
        x11_screenshot/x11_click were actually succeeding."""
        monkeypatch.setenv("XDG_SESSION_TYPE", "tty")
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        with (
            patch(
                "shutil.which", side_effect=lambda b: f"/usr/bin/{b}" if b == "xdotool" else None
            ),
            patch("missy.tools.builtin.desktop_tools._probe_x11_display", return_value=True),
        ):
            result = DesktopStatusTool().execute()

        assert result.output["has_x11_display"] is True
        assert result.output["x11_automation_usable"] is True


class TestProbeX11Display:
    def test_returns_false_when_xdotool_missing(self):
        from missy.tools.builtin.desktop_tools import _probe_x11_display

        with patch("shutil.which", return_value=None):
            assert _probe_x11_display() is False

    def test_returns_true_when_probe_succeeds(self):
        from missy.tools.builtin.desktop_tools import _probe_x11_display

        with (
            patch("shutil.which", return_value="/usr/bin/xdotool"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            assert _probe_x11_display() is True
            args, kwargs = mock_run.call_args
            assert args[0] == ["xdotool", "getdisplaygeometry"]
            assert kwargs["timeout"] == 3

    def test_returns_false_when_probe_fails(self):
        from missy.tools.builtin.desktop_tools import _probe_x11_display

        with (
            patch("shutil.which", return_value="/usr/bin/xdotool"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1)
            assert _probe_x11_display() is False

    def test_returns_false_on_exception(self):
        from missy.tools.builtin.desktop_tools import _probe_x11_display

        with (
            patch("shutil.which", return_value="/usr/bin/xdotool"),
            patch("subprocess.run", side_effect=OSError("boom")),
        ):
            assert _probe_x11_display() is False


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


# ---------------------------------------------------------------------------
# DesktopMouseMoveTool
# ---------------------------------------------------------------------------


class TestDesktopMouseMoveTool:
    def test_moves_cursor(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = DesktopMouseMoveTool().execute(x=100, y=200)

        assert result.success is True
        assert result.output == {"x": 100, "y": 200}
        args = mock_run.call_args[0][0]
        assert args == ["xdotool", "mousemove", "100", "200"]

    def test_does_not_click(self):
        """Regression guard: must never issue a click, only a move."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            DesktopMouseMoveTool().execute(x=0, y=0)
        args = mock_run.call_args[0][0]
        assert "click" not in " ".join(args)

    def test_xdotool_missing_reports_install_hint(self):
        with patch("subprocess.run") as mock_run, patch("shutil.which", return_value=None):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
            result = DesktopMouseMoveTool().execute(x=0, y=0)
        assert result.success is False
        assert "xdotool" in result.error.lower()

    def test_requires_shell_permission(self):
        assert DesktopMouseMoveTool().permissions.shell is True


# ---------------------------------------------------------------------------
# InstallSoftwareConfirmedTool
# ---------------------------------------------------------------------------


class TestInstallSoftwareConfirmedTool:
    def test_disabled_by_default_even_when_desktop_enabled(self):
        """desktop.enabled alone is not enough -- needs the separate
        allow_software_install opt-in."""
        with patch(
            "missy.tools.builtin.desktop_tools._desktop_config",
            return_value=_mock_config(enabled=True, allow_software_install=False),
        ):
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio")
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_disabled_when_desktop_not_enabled_even_if_install_flag_set(self):
        with patch(
            "missy.tools.builtin.desktop_tools._desktop_config",
            return_value=_mock_config(enabled=False, allow_software_install=True),
        ):
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio")
        assert result.success is False

    def test_invalid_package_name_rejected(self):
        with patch(
            "missy.tools.builtin.desktop_tools._desktop_config",
            return_value=_mock_config(enabled=True, allow_software_install=True),
        ):
            result = InstallSoftwareConfirmedTool().execute(package="--reinstall")
        assert result.success is False
        assert "valid apt package" in result.error

    def test_always_requires_approval(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(enabled=True, allow_software_install=True),
            ),
            patch(
                "missy.tools.builtin.desktop_tools.require_approval",
                return_value="denied by operator",
            ) as mock_approval,
        ):
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio")
        assert result.success is False
        assert "denied" in result.error
        mock_approval.assert_called_once()

    def test_approved_install_runs_apt_get(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(enabled=True, allow_software_install=True),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Setting up obs-studio", stderr=""
            )
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio")

        assert result.success is True
        assert result.output["package"] == "obs-studio"
        args = mock_run.call_args[0][0]
        assert args == ["sudo", "apt-get", "install", "-y", "obs-studio"]

    def test_failed_install_reports_error(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(enabled=True, allow_software_install=True),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="E: Unable to locate package bogus-pkg"
            )
            result = InstallSoftwareConfirmedTool().execute(package="bogus-pkg")

        assert result.success is False
        assert "Unable to locate package" in result.error

    def test_install_timeout_reports_error(self):
        import subprocess as _subprocess

        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(enabled=True, allow_software_install=True),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval", return_value=None),
            patch(
                "subprocess.run",
                side_effect=_subprocess.TimeoutExpired(cmd="apt-get", timeout=300),
            ),
        ):
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio")

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_never_uses_a_shell_string(self):
        """subprocess.run must be called with an argv list, never shell=True."""
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(enabled=True, allow_software_install=True),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            InstallSoftwareConfirmedTool().execute(package="obs-studio")

        call_args, call_kwargs = mock_run.call_args
        assert isinstance(call_args[0], list)
        assert call_kwargs.get("shell", False) is False

    def test_package_with_version_spec_accepted(self):
        with (
            patch(
                "missy.tools.builtin.desktop_tools._desktop_config",
                return_value=_mock_config(enabled=True, allow_software_install=True),
            ),
            patch("missy.tools.builtin.desktop_tools.require_approval", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio=30.0.2")

        assert result.success is True

    def test_resolve_shell_command_declares_sudo_and_apt(self):
        tool = InstallSoftwareConfirmedTool()
        assert tool.resolve_shell_command({}) == "sudo && apt-get"


# ---------------------------------------------------------------------------
# Rate limiting integration (mechanics tested directly in
# test_desktop_shared.py; these confirm each tool actually wires it in)
# ---------------------------------------------------------------------------


class TestDesktopToolsRateLimiting:
    def test_launch_app_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.desktop_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = DesktopLaunchAppTool().execute(app="firefox")
        assert result.success is False
        assert "Rate limit" in result.error

    def test_focus_window_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.desktop_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = DesktopFocusWindowTool().execute(window_name="Firefox")
        assert result.success is False

    def test_mouse_drag_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.desktop_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = DesktopMouseDragTool().execute(start_x=0, start_y=0, end_x=1, end_y=1)
        assert result.success is False

    def test_mouse_move_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.desktop_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = DesktopMouseMoveTool().execute(x=0, y=0)
        assert result.success is False

    def test_status_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.desktop_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = DesktopStatusTool().execute()
        assert result.success is False

    def test_install_software_denied_when_rate_limited(self):
        with patch(
            "missy.tools.builtin.desktop_tools._check_rate_limit",
            return_value="Rate limit exceeded",
        ):
            result = InstallSoftwareConfirmedTool().execute(package="obs-studio")
        assert result.success is False


# ---------------------------------------------------------------------------
# DesktopFocusWindowTool window allowlist integration
# ---------------------------------------------------------------------------


class TestDesktopFocusWindowToolAllowlist:
    def test_denied_when_window_not_allowed(self):
        with patch(
            "missy.tools.builtin.desktop_tools.check_window_allowed",
            return_value="requires approval",
        ):
            result = DesktopFocusWindowTool().execute(window_name="Secret App")
        assert result.success is False
        assert "requires approval" in result.error

    def test_proceeds_when_window_allowed(self):
        with (
            patch("missy.tools.builtin.desktop_tools.check_window_allowed", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = DesktopFocusWindowTool().execute(window_name="Firefox")
        assert result.success is True
