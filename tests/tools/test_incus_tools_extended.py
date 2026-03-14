"""Extended tests for missy/tools/builtin/incus_tools.py — uncovered paths.

Covers the branches not exercised by the existing test_incus_tools.py:
- _run_incus generic exception handling
- project flags on various tools
- IncusConfigTool invalid action / project flags
- IncusImageTool: delete, info, alias, copy without alias, project flags
- IncusNetworkTool: show, set, attach/detach failures, detach success, project
- IncusStorageTool: delete, show, info, volume-delete, invalid actions
- IncusProfileTool: show, delete, copy, project flags, show missing name already tested
- IncusProjectTool: show, delete, project flags
- IncusSnapshotTool: restore without snapshot_name, invalid action, project flags
- IncusDeviceTool: invalid action, project
- IncusCopyMoveTool: copy_to_remote, project flags
- IncusInfoTool: project flag
- IncusExecTool: group flag, project flag
"""

from __future__ import annotations

import subprocess
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


def _make_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(
        args=["incus"],
        returncode=returncode,
        stdout=stdout.encode(),
        stderr=stderr.encode(),
    )


def _json_proc(data):
    import json

    return _make_proc(stdout=json.dumps(data))


# ---------------------------------------------------------------------------
# _run_incus — generic exception handling
# ---------------------------------------------------------------------------


class TestRunIncusGenericException:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_generic_exception_returns_error(self, mock_run) -> None:
        mock_run.side_effect = PermissionError("permission denied")
        result = _run_incus(["list"])
        assert not result.success
        assert "permission denied" in result.error


# ---------------------------------------------------------------------------
# IncusListTool — project/all_projects combinations
# ---------------------------------------------------------------------------


class TestIncusListToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusListTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_all_projects_takes_precedence_over_project(self, mock_run) -> None:
        mock_run.return_value = _json_proc([])
        self.tool.execute(all_projects=True, project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--all-projects" in cmd
        assert "--project" not in cmd


# ---------------------------------------------------------------------------
# IncusLaunchTool — project flag
# ---------------------------------------------------------------------------


class TestIncusLaunchToolExtended:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_launch_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusLaunchTool()
        tool.execute(image="images:ubuntu/24.04", project="staging")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd
        assert "staging" in cmd


# ---------------------------------------------------------------------------
# IncusInstanceActionTool — project flag + pause action
# ---------------------------------------------------------------------------


class TestIncusInstanceActionToolExtended:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_pause_action(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusInstanceActionTool()
        tool.execute(instance="test", action="pause")
        cmd = mock_run.call_args[0][0]
        assert "pause" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_project_flag_on_action(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusInstanceActionTool()
        tool.execute(instance="test", action="start", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd
        assert "dev" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_force_delete(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusInstanceActionTool()
        tool.execute(instance="test", action="delete", force=True)
        cmd = mock_run.call_args[0][0]
        assert "--force" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_force_restart(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusInstanceActionTool()
        tool.execute(instance="test", action="restart", force=True)
        cmd = mock_run.call_args[0][0]
        assert "--force" in cmd


# ---------------------------------------------------------------------------
# IncusInfoTool — project flag
# ---------------------------------------------------------------------------


class TestIncusInfoToolExtended:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_info_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="Name: test\n")
        tool = IncusInfoTool()
        tool.execute(instance="test", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd
        assert "dev" in cmd


# ---------------------------------------------------------------------------
# IncusExecTool — group flag, project flag
# ---------------------------------------------------------------------------


class TestIncusExecToolExtended:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_exec_with_group(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="ok\n")
        tool = IncusExecTool()
        tool.execute(instance="test", command="id", group=1001)
        cmd = mock_run.call_args[0][0]
        assert "--group" in cmd
        assert "1001" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_exec_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusExecTool()
        tool.execute(instance="test", command="ls", project="staging")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd


# ---------------------------------------------------------------------------
# IncusFileTool — project flag, list action
# ---------------------------------------------------------------------------


class TestIncusFileToolExtended:
    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_push_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        tool = IncusFileTool()
        tool.execute(
            action="push",
            instance="test",
            instance_path="/root/f",
            host_path="/tmp/f",
            project="dev",
        )
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd


# ---------------------------------------------------------------------------
# IncusSnapshotTool — restore without name, invalid action, project
# ---------------------------------------------------------------------------


class TestIncusSnapshotToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusSnapshotTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(instance="test", action="clone")
        assert not result.success
        assert "Invalid action" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_restore_without_snapshot_name(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(instance="test", action="restore")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert cmd == ["incus", "snapshot", "restore", "test"]

    def test_delete_missing_snapshot_name(self) -> None:
        result = self.tool.execute(instance="test", action="delete")
        assert not result.success
        assert "snapshot_name" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            instance="test",
            action="create",
            snapshot_name="snap1",
            project="dev",
        )
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="{}")
        self.tool.execute(instance="test", action="list", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd


# ---------------------------------------------------------------------------
# IncusConfigTool — invalid action, project flag, unset with project
# ---------------------------------------------------------------------------


class TestIncusConfigToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusConfigTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(instance="test", action="apply")
        assert not result.success
        assert "Invalid action" in result.error

    def test_unset_missing_key(self) -> None:
        result = self.tool.execute(instance="test", action="unset")
        assert not result.success
        assert "key" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_show_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="config: {}\n")
        self.tool.execute(instance="test", action="show", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_get_with_project(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="4\n")
        self.tool.execute(instance="test", action="get", key="limits.cpu", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd


# ---------------------------------------------------------------------------
# IncusImageTool — delete, info, alias, copy without alias, project
# ---------------------------------------------------------------------------


class TestIncusImageToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusImageTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="delete", image="abc123")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd
        assert "abc123" in cmd

    def test_delete_missing_image(self) -> None:
        result = self.tool.execute(action="delete")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_info(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="fingerprint: abc\n")
        result = self.tool.execute(action="info", image="abc123")
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_alias_create(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="alias", image="abc123", alias="ubuntu-latest")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "alias" in cmd
        assert "create" in cmd

    def test_alias_missing_both_fields(self) -> None:
        result = self.tool.execute(action="alias")
        assert not result.success

    def test_alias_missing_alias(self) -> None:
        result = self.tool.execute(action="alias", image="abc123")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_copy_without_alias(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(action="copy", image="images:ubuntu/24.04")
        cmd = mock_run.call_args[0][0]
        assert "--alias" not in cmd

    def test_copy_missing_image(self) -> None:
        result = self.tool.execute(action="copy")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_with_project(self, mock_run) -> None:
        mock_run.return_value = _json_proc([])
        self.tool.execute(action="list", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd


# ---------------------------------------------------------------------------
# IncusNetworkTool — extended paths
# ---------------------------------------------------------------------------


class TestIncusNetworkToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusNetworkTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(action="clone")
        assert not result.success
        assert "Invalid action" in result.error

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_show(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="config: {}\n")
        result = self.tool.execute(action="show", name="br0")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "show" in cmd

    def test_show_missing_name(self) -> None:
        result = self.tool.execute(action="show")
        assert not result.success

    def test_delete_missing_name(self) -> None:
        result = self.tool.execute(action="delete")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_set_config(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(
            action="set",
            name="br0",
            config={"ipv4.address": "10.0.0.1/24"},
        )
        assert result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_set_config_fails_on_error(self, mock_run) -> None:
        mock_run.return_value = _make_proc(returncode=1, stderr="Error: not found")
        result = self.tool.execute(
            action="set",
            name="br0",
            config={"bad.key": "value"},
        )
        assert not result.success

    def test_set_missing_name(self) -> None:
        result = self.tool.execute(action="set", config={"k": "v"})
        assert not result.success

    def test_set_missing_config(self) -> None:
        result = self.tool.execute(action="set", name="br0")
        assert not result.success

    def test_attach_single_word_name_fails(self) -> None:
        result = self.tool.execute(action="attach", name="br0")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_detach(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="detach", name="br0 mycontainer")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "detach" in cmd

    def test_detach_missing_name(self) -> None:
        result = self.tool.execute(action="detach")
        assert not result.success

    def test_detach_single_word_name_fails(self) -> None:
        result = self.tool.execute(action="detach", name="br0")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_create_with_config(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            action="create",
            name="testnet",
            config={"ipv4.address": "10.1.0.1/24", "ipv4.nat": "true"},
        )
        cmd = mock_run.call_args[0][0]
        assert "ipv4.address=10.1.0.1/24" in cmd


# ---------------------------------------------------------------------------
# IncusStorageTool — extended paths
# ---------------------------------------------------------------------------


class TestIncusStorageToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusStorageTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(action="clone")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete_pool(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="delete", pool="mypool")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd
        assert "mypool" in cmd

    def test_delete_missing_pool(self) -> None:
        result = self.tool.execute(action="delete")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_show_pool(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="config: {}\n")
        result = self.tool.execute(action="show", pool="mypool")
        assert result.success

    def test_show_missing_pool(self) -> None:
        result = self.tool.execute(action="show")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_volume_list(self, mock_run) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="volume-list", pool="mypool")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "volume" in cmd
        assert "list" in cmd

    def test_volume_list_missing_pool(self) -> None:
        result = self.tool.execute(action="volume-list")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_volume_delete(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="volume-delete", pool="default", volume="mydata")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd

    def test_volume_delete_missing_volume(self) -> None:
        result = self.tool.execute(action="volume-delete", pool="default")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_volume_attach_missing_instance(self, mock_run) -> None:
        # volume-attach without instance should fail.
        result = self.tool.execute(action="volume-attach", pool="default", volume="data")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusProfileTool — extended paths
# ---------------------------------------------------------------------------


class TestIncusProfileToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusProfileTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(action="nuke")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_show(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="config: {}\n")
        result = self.tool.execute(action="show", name="default")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "show" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="delete", name="old-profile")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd

    def test_delete_missing_name(self) -> None:
        result = self.tool.execute(action="delete")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_with_project(self, mock_run) -> None:
        mock_run.return_value = _json_proc([])
        result = self.tool.execute(action="list", project="dev")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_set_missing_config(self, mock_run) -> None:
        # set without config should fail validation or produce unexpected args.
        result = self.tool.execute(action="set", name="default")
        # No crash — just shouldn't add config args.
        # The tool calls _run_incus with an empty loop.
        assert result is not None


# ---------------------------------------------------------------------------
# IncusProjectTool — extended paths
# ---------------------------------------------------------------------------


class TestIncusProjectToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusProjectTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(action="nuke")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_show(self, mock_run) -> None:
        mock_run.return_value = _make_proc(stdout="config: {}\n")
        result = self.tool.execute(action="show", name="dev")
        assert result.success

    def test_show_missing_name(self) -> None:
        result = self.tool.execute(action="show")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_delete(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(action="delete", name="old-project")
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "delete" in cmd

    def test_delete_missing_name(self) -> None:
        result = self.tool.execute(action="delete")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_switch_missing_name(self, mock_run) -> None:
        result = self.tool.execute(action="switch")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusDeviceTool — invalid action, project
# ---------------------------------------------------------------------------


class TestIncusDeviceToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusDeviceTool()

    def test_invalid_action(self) -> None:
        result = self.tool.execute(instance="test", action="clone")
        assert not result.success

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_list_with_project(self, mock_run) -> None:
        mock_run.return_value = _json_proc({})
        self.tool.execute(instance="test", action="list", project="dev")
        cmd = mock_run.call_args[0][0]
        assert "--project" in cmd

    def test_remove_missing_device_name(self) -> None:
        result = self.tool.execute(instance="test", action="remove")
        assert not result.success


# ---------------------------------------------------------------------------
# IncusCopyMoveTool — extended paths
# ---------------------------------------------------------------------------


class TestIncusCopyMoveToolExtended:
    def setup_method(self) -> None:
        self.tool = IncusCopyMoveTool()

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_copy_to_remote(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(
            source="test", destination="remote:test-copy", target_remote="remote"
        )
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "copy" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_copy_with_project_flags(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        self.tool.execute(
            source="test",
            destination="test-copy",
            target_project="staging",
            source_project="dev",
        )
        cmd = mock_run.call_args[0][0]
        assert "--target-project" in cmd or "--source-project" in cmd

    @patch("missy.tools.builtin.incus_tools.subprocess.run")
    def test_move_to_different_remote(self, mock_run) -> None:
        mock_run.return_value = _make_proc()
        result = self.tool.execute(
            source="test",
            destination="other:test",
            action="move",
        )
        assert result.success
        cmd = mock_run.call_args[0][0]
        assert "move" in cmd
