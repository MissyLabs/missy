"""Tests for Incus container/VM management tools."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
        assert "--type" in cmd
        assert "container" in cmd


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
        assert "info" in cmd


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
        mock_run.return_value = _json_proc({})
        result = self.tool.execute(instance="test", action="list")
        assert result.success

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
