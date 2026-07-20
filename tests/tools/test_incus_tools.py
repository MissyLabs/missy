"""Tests for Incus container/VM management tools."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.policy.engine import init_policy_engine
from missy.tools.builtin.incus_tools import (
    IncusConfigTool,
    IncusCopyMoveTool,
    IncusDeviceTool,
    IncusExecTool,
    IncusFileTool,
    IncusImageTool,
    IncusInfoTool,
    IncusInstanceActionTool,
    IncusLaunchTool,
    IncusListTool,
    IncusNetworkTool,
    IncusProfileTool,
    IncusProjectTool,
    IncusSnapshotTool,
    IncusStorageTool,
    _run_incus,
)
from missy.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_proc(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["incus"],
        returncode=returncode,
        stdout=stdout.encode(),
        stderr=stderr.encode(),
    )


def _json_proc(data: Any) -> subprocess.CompletedProcess:
    return _make_proc(stdout=json.dumps(data))


# ---------------------------------------------------------------------------
# _run_incus helper
# ---------------------------------------------------------------------------
class TestRunIncus:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="ok\n")
        result = _run_incus(["list"])
        assert result.success
        assert result.output == "ok\n"
        assert result.error is None

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_json_output_parsed(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([{"name": "test"}])
        result = _run_incus(["list", "--format", "json"])
        assert result.success
        assert isinstance(result.output, list)
        assert result.output[0]["name"] == "test"

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_failure_exit_code(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stderr="Error: not found", returncode=1)
        result = _run_incus(["info", "missing"])
        assert not result.success
        assert "Exit code 1" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="incus", timeout=60)
        result = _run_incus(["exec", "test", "--", "sleep", "999"])
        assert not result.success
        assert "timed out" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_binary_not_found(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError()
        result = _run_incus(["list"])
        assert not result.success
        assert "not found" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_output_truncation(self, mock_run: MagicMock) -> None:
        big = "x" * 100_000
        mock_run.return_value = _make_proc(stdout=big)
        result = _run_incus(["list"])
        assert result.success
        assert "[Output truncated]" in result.output

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_stdin_data(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="ok")
        _run_incus(["profile", "edit", "default"], stdin_data="config: {}")
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("input") == b"config: {}"

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_timeout_clamped(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        _run_incus(["list"], timeout=9999)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["timeout"] == 600


# ---------------------------------------------------------------------------
# IncusListTool
# ---------------------------------------------------------------------------
class TestIncusListTool:
    def setup_method(self) -> None:
        self.tool = IncusListTool()

    def test_schema(self) -> None:
        s = self.tool.get_schema()
        assert s["name"] == "incus_list"
        assert "properties" in s["parameters"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_basic(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute()
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "list", "--format", "json"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_with_project(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        self.tool.execute(project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd
        assert "dev" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_all_projects(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        self.tool.execute(all_projects=True)
        cmd = mock_run.call_args[0][0]
        assert "--all-projects" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_by_type(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        self.tool.execute(instance_type="container")
        cmd = mock_run.call_args[0][0]
        assert "--type" not in cmd
        assert "type=container" in cmd


# ---------------------------------------------------------------------------
# IncusLaunchTool
# ---------------------------------------------------------------------------
class TestIncusLaunchTool:
    def setup_method(self) -> None:
        self.tool = IncusLaunchTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_launch_basic(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="Creating test\nStarting test\n")
        result = self.tool.execute(image="images:ubuntu/24.04")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "launch", "images:ubuntu/24.04"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_launch_named_vm(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(image="images:debian/12", name="myvm", vm=True)
        cmd = mock_run.call_args[0][0]
        assert "myvm" in cmd
        assert "--vm" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_launch_with_config(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            image="images:alpine/3.19",
            config={"limits.cpu": "2", "limits.memory": "4GiB"},
        )
        cmd = mock_run.call_args[0][0]
        assert "--config" in cmd
        assert "limits.cpu=2" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_launch_with_profiles(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(image="images:ubuntu/24.04", profiles=["default", "gpu"])
        cmd = mock_run.call_args[0][0]
        assert cmd.count("--profile") == 2

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_launch_ephemeral(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(image="images:ubuntu/24.04", ephemeral=True)
        cmd = mock_run.call_args[0][0]
        assert "--ephemeral" in cmd


# ---------------------------------------------------------------------------
# IncusInstanceActionTool
# ---------------------------------------------------------------------------
class TestIncusInstanceActionTool:
    def setup_method(self) -> None:
        self.tool = IncusInstanceActionTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_start(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(instance="test", action="start")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "start", "test"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_stop_force(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="stop", force=True)
        cmd = mock_run.call_args[0][0]
        assert "--force" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="delete")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "delete", "test"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_rename(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="old", action="rename", new_name="new")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "rename", "old", "new"]

    def test_rename_missing_new_name(self) -> None:
        result = self.tool.execute(instance="old", action="rename")
        assert not result.success
        assert "new_name" in result.error

    def test_invalid_action(self) -> None:
        result = self.tool.execute(instance="test", action="explode")
        assert not result.success
        assert "Invalid action" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_force_ignored_for_start(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="start", force=True)
        cmd = mock_run.call_args[0][0]
        assert "--force" not in cmd


# ---------------------------------------------------------------------------
# IncusInstanceActionTool -- post-timeout state recheck (INCUS-006,
# prompt.md line 91: "On timeout, mark pending effects unknown, perform
# a fresh read-only state check before retrying or reporting status").
# A client-side subprocess timeout says nothing about whether the Incus
# daemon actually completed a mutating action server-side -- reporting
# only "timed out" leaves the caller with no way to distinguish "nothing
# happened" from "it happened anyway, just slowly." Live-verified this
# session against a real Incus container with an artificially tiny
# timeout that genuinely triggered subprocess.TimeoutExpired.
# ---------------------------------------------------------------------------
class TestIncusInstanceActionTimeoutRecheck:
    def setup_method(self) -> None:
        self.tool = IncusInstanceActionTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_timeout_triggers_readonly_state_recheck(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="incus", timeout=1),
            _json_proc([{"name": "test", "status": "Running"}]),
        ]
        result = self.tool.execute(instance="test", action="restart", timeout=1)

        assert result.success is False
        assert "timed out after 1s" in result.error
        assert "unknown at the moment of timeout" in result.error
        assert "currently 'Running'" in result.error
        # The recheck must be a real, separate, read-only `incus list`
        # call -- not just reusing/guessing from the timed-out action.
        assert mock_run.call_count == 2
        recheck_cmd = mock_run.call_args_list[1][0][0]
        assert recheck_cmd == ["incus", "list", "test", "--format", "json"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_timeout_recheck_reports_instance_gone(self, mock_run: MagicMock) -> None:
        """A `delete` that races past the client-side timeout may have
        actually completed server-side -- the recheck must say so
        plainly rather than implying the instance still exists."""
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="incus", timeout=1),
            _json_proc([]),  # incus list returns an empty array: gone
        ]
        result = self.tool.execute(instance="test", action="delete", timeout=1)

        assert result.success is False
        assert "no longer exists" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_timeout_recheck_itself_failing_is_reported_honestly(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="incus", timeout=1),
            subprocess.TimeoutExpired(cmd="incus", timeout=30),  # recheck also times out
        ]
        result = self.tool.execute(instance="test", action="stop", timeout=1)

        assert result.success is False
        assert "could not be determined" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_rename_timeout_does_not_attempt_recheck(self, mock_run: MagicMock) -> None:
        """After a rename times out, the instance could be under either
        the old or the new name -- guessing which one to recheck could
        itself misreport state, so rename is deliberately excluded."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="incus", timeout=1)
        result = self.tool.execute(instance="old", action="rename", new_name="new", timeout=1)

        assert result.success is False
        assert result.error == "Command timed out after 1s"
        assert mock_run.call_count == 1  # no recheck attempted

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_non_timeout_failure_does_not_trigger_recheck(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stderr="Error: not found", returncode=1)
        result = self.tool.execute(instance="test", action="start")

        assert result.success is False
        assert "Exit code 1" in result.error
        assert mock_run.call_count == 1  # no recheck for an ordinary exit-code failure

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_project_scope_carried_into_recheck(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="incus", timeout=1),
            _json_proc([{"name": "test", "status": "Stopped"}]),
        ]
        self.tool.execute(instance="test", action="stop", project="myproj", timeout=1)

        recheck_cmd = mock_run.call_args_list[1][0][0]
        assert recheck_cmd == ["incus", "list", "test", "--format", "json", "--project", "myproj"]


# ---------------------------------------------------------------------------
# IncusInfoTool
# ---------------------------------------------------------------------------
class TestIncusInfoTool:
    def setup_method(self) -> None:
        self.tool = IncusInfoTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_info_basic(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="Name: test\nStatus: Running\n")
        result = self.tool.execute(instance="test")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_info_resources(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", show_resources=True)
        cmd = mock_run.call_args[0][0]
        assert "--resources" in cmd


# ---------------------------------------------------------------------------
# IncusExecTool
# ---------------------------------------------------------------------------
class TestIncusExecTool:
    def setup_method(self) -> None:
        self.tool = IncusExecTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_exec_string_command(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="hello\n")
        result = self.tool.execute(instance="test", command="echo hello")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "exec", "test", "--", "bash", "-c", "echo hello"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_exec_list_command(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", command=["ls", "-la"])
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "exec", "test", "--", "ls", "-la"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_exec_with_env(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", command="env", env={"FOO": "bar"})
        cmd = mock_run.call_args[0][0]
        assert "--env" in cmd
        assert "FOO=bar" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_exec_with_cwd_and_user(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", command="pwd", cwd="/tmp", user=1000)
        cmd = mock_run.call_args[0][0]
        assert "--cwd" in cmd
        assert "/tmp" in cmd
        assert "--user" in cmd
        assert "1000" in cmd


# ---------------------------------------------------------------------------
# IncusFileTool
# ---------------------------------------------------------------------------
class TestIncusFileTool:
    def setup_method(self) -> None:
        self.tool = IncusFileTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_push(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(
            action="push",
            instance="test",
            instance_path="/root/file.txt",
            host_path="/tmp/file.txt",
        )
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "file", "push", "/tmp/file.txt", "test/root/file.txt"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_pull(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            action="pull",
            instance="test",
            instance_path="/var/log/syslog",
            host_path="/tmp/syslog",
        )
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "file", "pull", "test/var/log/syslog", "/tmp/syslog"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_push_recursive(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            action="push",
            instance="test",
            instance_path="/opt/app/",
            host_path="/src/app/",
            recursive=True,
            create_dirs=True,
        )
        cmd = mock_run.call_args[0][0]
        assert "--create-dirs" in cmd
        assert "--recursive" in cmd

    def test_invalid_action(self) -> None:
        result = self.tool.execute(
            action="delete",
            instance="test",
            instance_path="/foo",
            host_path="/bar",
        )
        assert not result.success


# ---------------------------------------------------------------------------
# IncusSnapshotTool
# ---------------------------------------------------------------------------
class TestIncusSnapshotTool:
    def setup_method(self) -> None:
        self.tool = IncusSnapshotTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(
            instance="test",
            action="create",
            snapshot_name="snap1",
        )
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "snapshot", "create", "test", "snap1"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create_stateful(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            instance="test",
            action="create",
            snapshot_name="snap1",
            stateful=True,
        )
        cmd = mock_run.call_args[0][0]
        assert "--stateful" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_restore(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="restore", snapshot_name="snap1")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "snapshot", "restore", "test", "snap1"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="delete", snapshot_name="snap1")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "snapshot", "delete", "test", "snap1"]

    def test_create_missing_name(self) -> None:
        result = self.tool.execute(instance="test", action="create")
        assert not result.success
        assert "snapshot_name" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="{}")
        self.tool.execute(instance="test", action="list")
        cmd = mock_run.call_args[0][0]
        # Implementation may use 'info' or 'query' API
        assert "info" in cmd or "query" in cmd or "snapshots" in " ".join(cmd)


# ---------------------------------------------------------------------------
# IncusConfigTool
# ---------------------------------------------------------------------------
class TestIncusConfigTool:
    def setup_method(self) -> None:
        self.tool = IncusConfigTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_show(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="config:\n  limits.cpu: '2'\n")
        result = self.tool.execute(instance="test")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_get(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="2\n")
        result = self.tool.execute(instance="test", action="get", key="limits.cpu")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "config", "get", "test", "limits.cpu"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_set(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="set", key="limits.cpu", value="4")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "config", "set", "test", "limits.cpu", "4"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_unset(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="unset", key="limits.cpu")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "config", "unset", "test", "limits.cpu"]

    def test_get_missing_key(self) -> None:
        result = self.tool.execute(instance="test", action="get")
        assert not result.success

    def test_set_missing_value(self) -> None:
        result = self.tool.execute(instance="test", action="set", key="limits.cpu")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusImageTool
# ---------------------------------------------------------------------------
class TestIncusImageTool:
    def setup_method(self) -> None:
        self.tool = IncusImageTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_local(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="list")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "image" in cmd
        assert "--format" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_remote(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        self.tool.execute(action="list", remote="images")
        cmd = mock_run.call_args[0][0]
        assert "images:" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_copy(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="copy", image="images:ubuntu/24.04", alias="ubuntu")
        cmd = mock_run.call_args[0][0]
        assert "--alias" in cmd

    def test_info_missing_image(self) -> None:
        result = self.tool.execute(action="info")
        assert not result.success

    def test_invalid_action(self) -> None:
        result = self.tool.execute(action="nuke")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusNetworkTool
# ---------------------------------------------------------------------------
class TestIncusNetworkTool:
    def setup_method(self) -> None:
        self.tool = IncusNetworkTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="list")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="create", name="br0", network_type="bridge")
        cmd = mock_run.call_args[0][0]
        assert "create" in cmd
        assert "--type" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="delete", name="br0")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "network", "delete", "br0"]

    def test_create_missing_name(self) -> None:
        result = self.tool.execute(action="create")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_attach(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="attach", name="br0 mycontainer")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "network", "attach", "br0", "mycontainer"]


# ---------------------------------------------------------------------------
# IncusStorageTool
# ---------------------------------------------------------------------------
class TestIncusStorageTool:
    def setup_method(self) -> None:
        self.tool = IncusStorageTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="list")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create_pool(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="create", pool="fast", driver="zfs", size="50GiB")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "storage", "create", "fast", "zfs", "size=50GiB"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_volume_create(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="volume-create", pool="default", volume="data")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "storage", "volume", "create", "default", "data"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_volume_attach(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            action="volume-attach",
            pool="default",
            volume="data",
            instance="test",
        )
        cmd = mock_run.call_args[0][0]
        assert "attach" in cmd

    def test_create_missing_driver(self) -> None:
        result = self.tool.execute(action="create", pool="fast")
        assert not result.success

    def test_volume_create_missing_volume(self) -> None:
        result = self.tool.execute(action="volume-create", pool="default")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusProfileTool
# ---------------------------------------------------------------------------
class TestIncusProfileTool:
    def setup_method(self) -> None:
        self.tool = IncusProfileTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="list")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="create", name="gpu")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "profile", "create", "gpu"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_edit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        yaml = "config:\n  limits.cpu: '4'\n"
        self.tool.execute(action="edit", name="default", yaml_content=yaml)
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("input") == yaml.encode("utf-8")

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_set(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            action="set",
            name="default",
            config={"limits.cpu": "2"},
        )
        cmd = mock_run.call_args[0][0]
        assert "limits.cpu" in cmd

    def test_show_missing_name(self) -> None:
        result = self.tool.execute(action="show")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusProjectTool
# ---------------------------------------------------------------------------
class TestIncusProjectTool:
    def setup_method(self) -> None:
        self.tool = IncusProjectTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="list")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="create", name="dev")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "project", "create", "dev"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_switch(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="switch", name="dev")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "project", "switch", "dev"]

    def test_create_missing_name(self) -> None:
        result = self.tool.execute(action="create")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusDeviceTool
# ---------------------------------------------------------------------------
class TestIncusDeviceTool:
    def setup_method(self) -> None:
        self.tool = IncusDeviceTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list(self, mock_run: MagicMock) -> None:
        # `incus config device list` has no --format flag (unlike most
        # other incus subcommands) and always prints plain text, one
        # device name per line -- found via live validation against a
        # real Incus instance (task #10): the tool previously requested
        # `--format json`, which `incus` rejects with "Error: unknown
        # flag: --format" on every real call. Mocking a JSON response
        # here would have masked that bug forever, so this asserts the
        # real argv instead of just the mocked return value.
        mock_run.return_value = _make_proc(stdout="eth0\n")
        result = self.tool.execute(instance="test", action="list")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "config", "device", "list", "test"]
        assert "--format" not in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_add_gpu(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            instance="test",
            action="add",
            device_name="mygpu",
            device_type="gpu",
        )
        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "incus",
            "config",
            "device",
            "add",
            "test",
            "mygpu",
            "gpu",
        ]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_add_proxy(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            instance="test",
            action="add",
            device_name="http",
            device_type="proxy",
            config={
                "listen": "tcp:0.0.0.0:8080",
                "connect": "tcp:127.0.0.1:80",
            },
        )
        cmd = mock_run.call_args[0][0]
        assert "listen=tcp:0.0.0.0:8080" in cmd
        assert "connect=tcp:127.0.0.1:80" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_remove(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(instance="test", action="remove", device_name="mygpu")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "config", "device", "remove", "test", "mygpu"]

    def test_add_missing_type(self) -> None:
        result = self.tool.execute(
            instance="test",
            action="add",
            device_name="foo",
        )
        assert not result.success


# ---------------------------------------------------------------------------
# IncusCopyMoveTool
# ---------------------------------------------------------------------------
class TestIncusCopyMoveTool:
    def setup_method(self) -> None:
        self.tool = IncusCopyMoveTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_copy(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(source="test", destination="test-copy")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "copy", "test", "test-copy"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_move(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(source="test", destination="test-new", action="move")
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "move", "test", "test-new"]

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_copy_to_different_project(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            source="test",
            destination="test-copy",
            target_project="staging",
        )
        cmd = mock_run.call_args[0][0]
        assert "--target-project" in cmd

    def test_invalid_action(self) -> None:
        result = self.tool.execute(source="a", destination="b", action="teleport")
        assert not result.success


# ---------------------------------------------------------------------------
# Schema and permissions tests
# ---------------------------------------------------------------------------
class TestAllToolsCommon:
    """Verify all Incus tools have consistent schema and permissions."""

    ALL_TOOLS = [
        IncusListTool,
        IncusLaunchTool,
        IncusInstanceActionTool,
        IncusInfoTool,
        IncusExecTool,
        IncusFileTool,
        IncusSnapshotTool,
        IncusConfigTool,
        IncusImageTool,
        IncusNetworkTool,
        IncusStorageTool,
        IncusProfileTool,
        IncusProjectTool,
        IncusDeviceTool,
        IncusCopyMoveTool,
    ]

    @pytest.mark.parametrize(
        "tool_cls",
        ALL_TOOLS,
        ids=lambda c: c.__name__,
    )
    def test_has_schema(self, tool_cls: type) -> None:
        tool = tool_cls()
        schema = tool.get_schema()
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"

    @pytest.mark.parametrize(
        "tool_cls",
        ALL_TOOLS,
        ids=lambda c: c.__name__,
    )
    def test_requires_shell_permission(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.permissions.shell is True

    @pytest.mark.parametrize(
        "tool_cls",
        ALL_TOOLS,
        ids=lambda c: c.__name__,
    )
    def test_name_starts_with_incus(self, tool_cls: type) -> None:
        tool = tool_cls()
        assert tool.name.startswith("incus_")


# ---------------------------------------------------------------------------
# FX-C: structured Incus list/network output must be preserved exactly
# through response construction -- the validation harness observed a run
# where Incus was reported as having an invented "lo" network and an
# incorrect bridge address. The tool layer itself must be a deterministic
# passthrough of real `incus ... --format json` output (no LLM-based
# resummarization at this layer) so any downstream fabrication is
# unambiguously a delegate/model behavior issue, not a tool-layer one.
# ---------------------------------------------------------------------------


class TestIncusListExactRowPreservation:
    """incus_list: the parsed JSON returned as tool output must be the
    exact same rows/fields the `incus` binary produced -- no rows added,
    removed, or fields altered."""

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_multi_instance_payload_passes_through_unmodified(self, mock_run: MagicMock) -> None:
        payload = [
            {
                "name": "web-01",
                "status": "Running",
                "type": "container",
                "state": {"network": {"eth0": {"addresses": [{"address": "10.0.0.5"}]}}},
            },
            {
                "name": "db-01",
                "status": "Stopped",
                "type": "virtual-machine",
                "state": None,
            },
        ]
        mock_run.return_value = _json_proc(payload)
        tool = IncusListTool()
        result = tool.execute()

        assert result.success
        assert result.output == payload
        assert len(result.output) == 2
        assert result.output[0]["name"] == "web-01"
        assert result.output[1]["name"] == "db-01"

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_empty_list_stays_empty_not_padded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _json_proc([])
        tool = IncusListTool()
        result = tool.execute()

        assert result.success
        assert result.output == []

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_no_extra_fields_are_synthesized(self, mock_run: MagicMock) -> None:
        # A minimal, realistic row with only a few fields -- the tool must
        # not add fields (e.g. a fabricated IP) that weren't in the real
        # incus output.
        payload = [{"name": "sparse-instance", "status": "Running"}]
        mock_run.return_value = _json_proc(payload)
        tool = IncusListTool()
        result = tool.execute()

        assert result.output == payload
        assert set(result.output[0].keys()) == {"name", "status"}


class TestIncusNetworkListExactRowPreservation:
    """incus_network(action='list'): same exact-passthrough guarantee for
    network definitions, including addresses -- the harness's specific
    observed failure was an invented 'lo' network and a wrong bridge
    address."""

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_real_bridge_network_payload_passes_through_unmodified(
        self, mock_run: MagicMock
    ) -> None:
        payload = [
            {
                "name": "incusbr0",
                "type": "bridge",
                "config": {"ipv4.address": "10.153.226.1/24"},
                "managed": True,
            },
        ]
        mock_run.return_value = _json_proc(payload)
        tool = IncusNetworkTool()
        result = tool.execute(action="list")

        assert result.success
        assert result.output == payload
        # Exactly one network -- no "lo" or any other network fabricated.
        assert len(result.output) == 1
        assert result.output[0]["name"] == "incusbr0"
        assert result.output[0]["config"]["ipv4.address"] == "10.153.226.1/24"

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_no_loopback_network_invented_when_absent_from_real_output(
        self, mock_run: MagicMock
    ) -> None:
        # Real `incus network list` output containing no "lo" entry.
        payload = [{"name": "incusbr0", "type": "bridge", "config": {}}]
        mock_run.return_value = _json_proc(payload)
        tool = IncusNetworkTool()
        result = tool.execute(action="list")

        names = [n["name"] for n in result.output]
        assert "lo" not in names

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_bridge_address_field_is_exact_string_not_reformatted(
        self, mock_run: MagicMock
    ) -> None:
        payload = [
            {"name": "incusbr0", "type": "bridge", "config": {"ipv4.address": "192.0.2.1/24"}}
        ]
        mock_run.return_value = _json_proc(payload)
        tool = IncusNetworkTool()
        result = tool.execute(action="list")

        assert result.output[0]["config"]["ipv4.address"] == "192.0.2.1/24"

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_network_show_returns_raw_command_output_not_reparsed(
        self, mock_run: MagicMock
    ) -> None:
        # `incus network show <name>` has no --format json flag (YAML by
        # default) -- confirm the tool sends the plain "show" command and
        # doesn't attempt to reshape the output.
        mock_run.return_value = _make_proc(stdout="name: incusbr0\ntype: bridge\n")
        tool = IncusNetworkTool()
        result = tool.execute(action="show", name="incusbr0")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "network", "show", "incusbr0"]
        assert result.output == "name: incusbr0\ntype: bridge\n"


# ---------------------------------------------------------------------------
# SR-1.5: registry+policy integration -- Incus tools must be gated on the
# real host command/paths they invoke, not a declaration/dispatch mismatch.
# ---------------------------------------------------------------------------
def _init_policy(
    allowed_commands: list[str] | None = None,
    allowed_write_paths: list[str] | None = None,
    allowed_read_paths: list[str] | None = None,
) -> None:
    init_policy_engine(
        MissyConfig(
            network=NetworkPolicy(),
            filesystem=FilesystemPolicy(
                allowed_write_paths=allowed_write_paths or [],
                allowed_read_paths=allowed_read_paths or [],
            ),
            shell=ShellPolicy(enabled=True, allowed_commands=allowed_commands or []),
            plugins=PluginPolicy(),
            providers={},
            workspace_path="/tmp/incus-test-ws",
            audit_log_path="/tmp/incus-test-audit.jsonl",
        )
    )


class TestSR15ShellPolicyGatesRealHostCommand:
    """Every Incus tool always runs the ``incus`` binary on the host --
    that must be the command checked, not a dummy default or (for
    incus_exec) the command run inside the guest."""

    def test_incus_denied_when_only_unrelated_command_allowed(self) -> None:
        _init_policy(allowed_commands=["git"])
        registry = ToolRegistry()
        registry.register(IncusInstanceActionTool())
        result = registry.execute(
            "incus_instance_action",
            instance="victim",
            action="delete",
            force=True,
            session_id="s",
            task_id="t",
        )
        assert result.success is False
        assert "incus" in result.error
        assert "not in the allowed commands list" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_incus_allowed_when_incus_explicitly_allowlisted(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="")
        _init_policy(allowed_commands=["incus"])
        registry = ToolRegistry()
        registry.register(IncusInstanceActionTool())
        result = registry.execute(
            "incus_instance_action",
            instance="victim",
            action="delete",
            force=True,
            session_id="s",
            task_id="t",
        )
        assert result.success is True
        mock_run.assert_called_once()

    def test_incus_exec_checked_against_host_binary_not_guest_command(self) -> None:
        """An operator who allowlists 'bash' -- intending only to let the
        agent run bash scripts inside a sandboxed guest -- must NOT
        thereby authorize the host `incus` binary itself. Before the
        fix, the registry checked kwargs["command"] (the guest command),
        so command="bash" slipped straight past the shell policy and the
        real host `incus exec ...` call executed unauthorized."""
        _init_policy(allowed_commands=["bash"])
        registry = ToolRegistry()
        registry.register(IncusExecTool())
        result = registry.execute(
            "incus_exec",
            instance="victim-container",
            command="bash",
            session_id="s",
            task_id="t",
        )
        assert result.success is False
        assert "incus" in result.error
        assert "not in the allowed commands list" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_incus_exec_allowed_when_incus_explicitly_allowlisted(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _make_proc(stdout="")
        _init_policy(allowed_commands=["incus"])
        registry = ToolRegistry()
        registry.register(IncusExecTool())
        result = registry.execute(
            "incus_exec",
            instance="victim-container",
            command="rm -rf /",
            session_id="s",
            task_id="t",
        )
        # The host command itself (incus) is authorized; the registry does
        # not additionally gate the guest-side command string -- that is
        # the operator's explicit tradeoff in allowlisting "incus" at all.
        assert result.success is True
        mock_run.assert_called_once()

    def test_all_incus_tools_resolve_shell_command_to_incus(self) -> None:
        tool_classes = [
            IncusListTool,
            IncusLaunchTool,
            IncusInstanceActionTool,
            IncusInfoTool,
            IncusExecTool,
            IncusFileTool,
            IncusSnapshotTool,
            IncusConfigTool,
            IncusImageTool,
            IncusNetworkTool,
            IncusStorageTool,
            IncusProfileTool,
            IncusProjectTool,
            IncusDeviceTool,
            IncusCopyMoveTool,
        ]
        for cls in tool_classes:
            tool = cls()
            assert tool.resolve_shell_command({}) == "incus", (
                f"{cls.__name__} must resolve its host command to 'incus'"
            )


class TestSR15IncusFileFilesystemEnforcement:
    """incus_file must enforce the real host_path against the filesystem
    policy -- previously it declared shell=True only, so host_path was
    never checked against allowed_read_paths/allowed_write_paths at all."""

    def test_pull_denied_when_write_path_not_allowlisted(self) -> None:
        _init_policy(allowed_commands=["incus"], allowed_write_paths=["/safe/downloads"])
        registry = ToolRegistry()
        registry.register(IncusFileTool())
        result = registry.execute(
            "incus_file",
            action="pull",
            instance="victim",
            instance_path="/etc/shadow",
            host_path="/tmp/exfiltrated-secrets",
            session_id="s",
            task_id="t",
        )
        assert result.success is False
        assert "Filesystem write denied" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_pull_allowed_when_write_path_allowlisted(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(stdout="")
        _init_policy(allowed_commands=["incus"], allowed_write_paths=["/safe/downloads"])
        registry = ToolRegistry()
        registry.register(IncusFileTool())
        result = registry.execute(
            "incus_file",
            action="pull",
            instance="victim",
            instance_path="/data/report.txt",
            host_path="/safe/downloads/report.txt",
            session_id="s",
            task_id="t",
        )
        assert result.success is True

    def test_push_denied_when_read_path_not_allowlisted(self) -> None:
        _init_policy(allowed_commands=["incus"], allowed_read_paths=["/safe/uploads"])
        registry = ToolRegistry()
        registry.register(IncusFileTool())
        result = registry.execute(
            "incus_file",
            action="push",
            instance="victim",
            instance_path="/tmp/payload",
            host_path="/etc/passwd",
            session_id="s",
            task_id="t",
        )
        assert result.success is False
        assert "Filesystem read denied" in result.error

    def test_resolve_filesystem_targets_push_is_read_only(self) -> None:
        tool = IncusFileTool()
        reads, writes = tool.resolve_filesystem_targets(
            {"action": "push", "host_path": "/some/path"}
        )
        assert reads == ["/some/path"]
        assert writes == []

    def test_resolve_filesystem_targets_pull_is_write_only(self) -> None:
        tool = IncusFileTool()
        reads, writes = tool.resolve_filesystem_targets(
            {"action": "pull", "host_path": "/some/path"}
        )
        assert reads == []
        assert writes == ["/some/path"]

    def test_resolve_filesystem_targets_no_host_path_is_empty(self) -> None:
        tool = IncusFileTool()
        reads, writes = tool.resolve_filesystem_targets({"action": "push"})
        assert reads == []
        assert writes == []
