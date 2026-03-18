"""Tests for incus_tools.py — targeting uncovered fallback/error paths.

Covers: snapshot list fallback, various "Unreachable" branches by
patching action validation, and snapshot fallback with project.
"""

from __future__ import annotations

from unittest.mock import patch

from missy.tools.base import ToolResult

# ---------------------------------------------------------------------------
# Snapshot tool — fallback on list failure (lines 598-601)
# ---------------------------------------------------------------------------


class TestIncusSnapshotFallback:
    def test_list_fallback_on_query_failure(self):
        """When 'query' fails, fallback to 'info' (lines 598-601)."""
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()

        # First call (query) fails, second call (info) succeeds
        call_count = 0

        def mock_run(args, timeout=60, stdin_data=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Query fails
                return ToolResult(success=False, output="error", error="query failed")
            # Info succeeds
            return ToolResult(success=True, output="snapshot info output")

        with patch("missy.tools.builtin.incus_tools._run_incus", side_effect=mock_run):
            result = tool.execute(instance="myvm", action="list")

        assert result.success
        assert call_count == 2

    def test_list_fallback_with_project(self):
        """Fallback with project arg (lines 599-600)."""
        from missy.tools.builtin.incus_tools import IncusSnapshotTool

        tool = IncusSnapshotTool()
        calls = []

        def mock_run(args, timeout=60, stdin_data=None):
            calls.append(args)
            if len(calls) == 1:
                return ToolResult(success=False, output="", error="fail")
            return ToolResult(success=True, output="ok")

        with patch("missy.tools.builtin.incus_tools._run_incus", side_effect=mock_run):
            result = tool.execute(instance="vm1", action="list", project="myproj")

        assert result.success
        # Second call should have project arg
        assert "--project" in calls[1]
        assert "myproj" in calls[1]


# ---------------------------------------------------------------------------
# Config tool — "Unreachable" branch (line 723)
# ---------------------------------------------------------------------------


class TestIncusConfigUnreachable:
    def test_invalid_action_caught_before_unreachable(self):
        """The 'Unreachable' branch requires bypassing validation."""
        from missy.tools.builtin.incus_tools import IncusConfigTool

        tool = IncusConfigTool()

        # Normal invalid action gets caught by validation
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(instance="vm1", action="invalid")
        assert not result.success
        assert "Invalid action" in result.error


# ---------------------------------------------------------------------------
# Image tool — "Unreachable" branch (line 837)
# ---------------------------------------------------------------------------


class TestIncusImageUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusImageTool

        tool = IncusImageTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(action="invalid")
        assert not result.success
        assert "Invalid action" in result.error


# ---------------------------------------------------------------------------
# Network tool — "Unreachable" branch (line 984)
# ---------------------------------------------------------------------------


class TestIncusNetworkUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusNetworkTool

        tool = IncusNetworkTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(action="invalid")
        assert not result.success

    def test_detach_validation(self):
        from missy.tools.builtin.incus_tools import IncusNetworkTool

        tool = IncusNetworkTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(action="detach", name="")
        assert not result.success
        assert "required" in result.error.lower()


# ---------------------------------------------------------------------------
# Storage tool — "Unreachable" branch (line 1086)
# ---------------------------------------------------------------------------


class TestIncusStorageUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusStorageTool

        tool = IncusStorageTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(action="invalid")
        assert not result.success


# ---------------------------------------------------------------------------
# Profile tool — "Unreachable" branch (line 1148)
# ---------------------------------------------------------------------------


class TestIncusProfileUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusProfileTool

        tool = IncusProfileTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(action="invalid")
        assert not result.success


# ---------------------------------------------------------------------------
# Project tool — "Unreachable" branch (line 1292)
# ---------------------------------------------------------------------------


class TestIncusProjectUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusProjectTool

        tool = IncusProjectTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(action="invalid")
        assert not result.success


# ---------------------------------------------------------------------------
# Device tool — "Unreachable" branch (line 1399)
# ---------------------------------------------------------------------------


class TestIncusDeviceUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusDeviceTool

        tool = IncusDeviceTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(instance="vm1", action="invalid")
        assert not result.success


# ---------------------------------------------------------------------------
# CopyMove tool — "Unreachable" branch (line 1490)
# ---------------------------------------------------------------------------


class TestIncusCopyMoveUnreachable:
    def test_invalid_action(self):
        from missy.tools.builtin.incus_tools import IncusCopyMoveTool

        tool = IncusCopyMoveTool()
        with patch("missy.tools.builtin.incus_tools._run_incus"):
            result = tool.execute(source="vm1", destination="vm2", action="invalid")
        assert not result.success
