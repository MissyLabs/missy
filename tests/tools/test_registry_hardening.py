"""Hardening tests for ToolRegistry: execution paths, policy checks, audit events.

Tests the execute() method's error handling, permission checking logic,
audit event emission, and singleton management.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.core.exceptions import PolicyViolationError
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


class EchoTool(BaseTool):
    name = "echo"
    description = "Returns input text"
    permissions = ToolPermissions()

    def execute(self, *, text: str = "") -> ToolResult:
        return ToolResult(success=True, output=text)


class NetworkTool(BaseTool):
    name = "network_tool"
    description = "Makes network requests"
    permissions = ToolPermissions(network=True, allowed_hosts=["api.example.com"])

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="network result")


class FileReadTool(BaseTool):
    name = "file_reader"
    description = "Reads files"
    permissions = ToolPermissions(filesystem_read=True, allowed_paths=["/tmp"])

    def execute(self, *, path: str = "/tmp/test") -> ToolResult:
        return ToolResult(success=True, output=f"read {path}")


class FileWriteTool(BaseTool):
    name = "file_writer"
    description = "Writes files"
    permissions = ToolPermissions(filesystem_write=True, allowed_paths=["/tmp"])

    def execute(self, *, path: str = "/tmp/test", content: str = "") -> ToolResult:
        return ToolResult(success=True, output=f"wrote {path}")


class ShellTool(BaseTool):
    name = "shell"
    description = "Runs shell commands"
    permissions = ToolPermissions(shell=True)

    def execute(self, *, command: str = "echo hi") -> ToolResult:
        return ToolResult(success=True, output="shell output")


class CrashingTool(BaseTool):
    name = "crasher"
    description = "Always crashes"
    permissions = ToolPermissions()

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("Tool crashed!")


class FailingTool(BaseTool):
    name = "failing"
    description = "Returns failure"
    permissions = ToolPermissions()

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=False, output=None, error="Something went wrong")


# ---------------------------------------------------------------------------
# Registry basic operations
# ---------------------------------------------------------------------------


class TestRegistryBasics:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = EchoTool()
        reg.register(tool)
        assert reg.get("echo") is tool

    def test_get_nonexistent(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_list_tools_sorted(self):
        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.register(EchoTool())
        reg.register(NetworkTool())
        names = reg.list_tools()
        assert names == sorted(names)
        assert "echo" in names
        assert "shell" in names

    def test_register_replaces_existing(self):
        reg = ToolRegistry()
        tool1 = EchoTool()
        tool2 = EchoTool()
        reg.register(tool1)
        reg.register(tool2)
        assert reg.get("echo") is tool2

    def test_list_tools_empty(self):
        reg = ToolRegistry()
        assert reg.list_tools() == []


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestRegistryExecution:
    def test_execute_simple_tool(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = reg.execute("echo", text="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_execute_nonexistent_raises_keyerror(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="ghost"):
            reg.execute("ghost")

    def test_execute_crashing_tool_returns_error(self):
        reg = ToolRegistry()
        reg.register(CrashingTool())
        result = reg.execute("crasher")
        assert result.success is False
        assert "crashed" in result.error

    def test_execute_failing_tool(self):
        reg = ToolRegistry()
        reg.register(FailingTool())
        result = reg.execute("failing")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_execute_strips_session_task_kwargs(self):
        """session_id and task_id should not be passed to tool.execute()."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        # Should not raise TypeError about unexpected kwargs
        result = reg.execute("echo", session_id="s1", task_id="t1", text="test")
        assert result.success is True
        assert result.output == "test"


# ---------------------------------------------------------------------------
# Permission checking
# ---------------------------------------------------------------------------


class TestPermissionChecking:
    def test_no_permissions_skips_policy(self):
        """Tools with no permissions should not trigger policy checks."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        # Even without policy engine, this should work
        with patch("missy.tools.registry.get_policy_engine", side_effect=RuntimeError("no engine")):
            result = reg.execute("echo", text="safe")
            assert result.success is True

    def test_network_permission_denied(self):
        """Network tool denied by policy returns failure result."""
        reg = ToolRegistry()
        reg.register(NetworkTool())

        mock_engine = MagicMock()
        mock_engine.check_network.side_effect = PolicyViolationError(
            "host denied", category="network", detail="api.example.com"
        )
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            result = reg.execute("network_tool")
            assert result.success is False
            assert "denied" in result.error.lower()

    def test_filesystem_read_checks_actual_path(self):
        """Policy engine should check the actual path from kwargs."""
        reg = ToolRegistry()
        reg.register(FileReadTool())

        mock_engine = MagicMock()
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            reg.execute("file_reader", path="/etc/passwd")
            # Should check both the declared allowed_paths AND the actual path
            check_read_calls = mock_engine.check_read.call_args_list
            paths_checked = [c[0][0] for c in check_read_calls]
            assert "/tmp" in paths_checked
            assert "/etc/passwd" in paths_checked

    def test_filesystem_write_checks_actual_path(self):
        """Policy engine checks actual write path from kwargs."""
        reg = ToolRegistry()
        reg.register(FileWriteTool())

        mock_engine = MagicMock()
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            reg.execute("file_writer", path="/etc/shadow")
            check_write_calls = mock_engine.check_write.call_args_list
            paths_checked = [c[0][0] for c in check_write_calls]
            assert "/etc/shadow" in paths_checked

    def test_shell_permission_check(self):
        """Shell commands are checked against policy."""
        reg = ToolRegistry()
        reg.register(ShellTool())

        mock_engine = MagicMock()
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            reg.execute("shell", command="rm -rf /")
            mock_engine.check_shell.assert_called_once()
            assert "rm -rf /" in mock_engine.check_shell.call_args[0][0]

    def test_shell_list_command_joined(self):
        """List commands should be joined to string for policy check."""
        reg = ToolRegistry()
        reg.register(ShellTool())

        mock_engine = MagicMock()
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            reg.execute("shell", command=["ls", "-la", "/tmp"])
            mock_engine.check_shell.assert_called_once()
            assert "ls -la /tmp" in mock_engine.check_shell.call_args[0][0]

    def test_policy_engine_not_initialized_fails_closed(self):
        """When policy engine isn't initialized, tools with permissions are denied."""
        reg = ToolRegistry()
        reg.register(NetworkTool())

        with patch("missy.tools.registry.get_policy_engine", side_effect=RuntimeError("not init")):
            result = reg.execute("network_tool")
            assert result.success is False
            assert "not initialised" in result.error.lower()

    def test_multiple_path_kwarg_names_checked(self):
        """Check all common path parameter names."""
        reg = ToolRegistry()

        class MultiPathTool(BaseTool):
            name = "multi_path"
            description = "test"
            permissions = ToolPermissions(filesystem_read=True)
            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output="ok")

        reg.register(MultiPathTool())
        mock_engine = MagicMock()
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            reg.execute("multi_path", file_path="/a", target="/b", destination="/c")
            paths = [c[0][0] for c in mock_engine.check_read.call_args_list]
            assert "/a" in paths
            assert "/b" in paths
            assert "/c" in paths


# ---------------------------------------------------------------------------
# Audit event emission
# ---------------------------------------------------------------------------


class TestAuditEvents:
    def test_success_emits_allow(self):
        reg = ToolRegistry()
        reg.register(EchoTool())

        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("echo", session_id="s1", task_id="t1", text="hi")
            mock_bus.publish.assert_called()
            event = mock_bus.publish.call_args[0][0]
            assert event.result == "allow"

    def test_failure_emits_error(self):
        reg = ToolRegistry()
        reg.register(FailingTool())

        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("failing", session_id="s1", task_id="t1")
            mock_bus.publish.assert_called()
            event = mock_bus.publish.call_args[0][0]
            assert event.result == "error"

    def test_crash_emits_error(self):
        reg = ToolRegistry()
        reg.register(CrashingTool())

        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("crasher")
            mock_bus.publish.assert_called()
            event = mock_bus.publish.call_args[0][0]
            assert event.result == "error"

    def test_policy_denial_emits_deny(self):
        reg = ToolRegistry()
        reg.register(NetworkTool())

        with patch("missy.tools.registry.event_bus") as mock_bus:
            mock_engine = MagicMock()
            mock_engine.check_network.side_effect = PolicyViolationError(
                "denied", category="network", detail="host blocked"
            )
            with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
                reg.execute("network_tool")
                mock_bus.publish.assert_called()
                event = mock_bus.publish.call_args[0][0]
                assert event.result == "deny"

    def test_audit_event_failure_does_not_crash(self):
        """If audit event emission fails, execute still returns normally."""
        reg = ToolRegistry()
        reg.register(EchoTool())

        with patch("missy.tools.registry.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("event bus broken")
            result = reg.execute("echo", text="hello")
            assert result.success is True


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_init_returns_registry(self):
        reg = init_tool_registry()
        assert isinstance(reg, ToolRegistry)

    def test_get_after_init(self):
        reg = init_tool_registry()
        assert get_tool_registry() is reg

    def test_init_replaces_existing(self):
        reg1 = init_tool_registry()
        reg2 = init_tool_registry()
        assert reg1 is not reg2
        assert get_tool_registry() is reg2
