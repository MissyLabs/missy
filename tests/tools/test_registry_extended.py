"""Extended tests for missy.tools.registry covering uncovered paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo"
    permissions = ToolPermissions()

    def execute(self, *, text: str = "", **kwargs):
        return ToolResult(success=True, output=text)


class CrashTool(BaseTool):
    name = "crash"
    description = "Always crashes"
    permissions = ToolPermissions()

    def execute(self, **kwargs):
        raise RuntimeError("boom")


class NetworkTool(BaseTool):
    name = "net_tool"
    description = "Needs network"
    permissions = ToolPermissions(network=True, allowed_hosts=["api.example.com"])

    def execute(self, **kwargs):
        return ToolResult(success=True, output="fetched")


class ShellTool(BaseTool):
    name = "shell_tool"
    description = "Needs shell"
    permissions = ToolPermissions(shell=True)

    def execute(self, *, command: str = "", **kwargs):
        return ToolResult(success=True, output=command)


class FSTool(BaseTool):
    name = "fs_tool"
    description = "Needs filesystem"
    permissions = ToolPermissions(
        filesystem_read=True, filesystem_write=True,
        allowed_paths=["/tmp/test"]
    )

    def execute(self, **kwargs):
        return ToolResult(success=True, output="ok")


class TestToolRegistryExecute:
    def test_execute_missing_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="No tool registered"):
            reg.execute("nonexistent")

    def test_execute_success(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = reg.execute("echo", text="hello")
        assert result.success
        assert result.output == "hello"

    def test_execute_tool_exception(self):
        reg = ToolRegistry()
        reg.register(CrashTool())
        result = reg.execute("crash")
        assert not result.success
        assert "boom" in result.error

    def test_execute_strips_session_task_keys(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = reg.execute("echo", session_id="s1", task_id="t1", text="hi")
        assert result.success
        assert result.output == "hi"


class TestToolRegistryPolicyChecks:
    def test_policy_not_initialized_skips_gracefully(self):
        """When PolicyEngine is not initialized, execution proceeds."""
        import missy.policy.engine as pe
        old = pe._engine
        pe._engine = None
        try:
            reg = ToolRegistry()
            reg.register(NetworkTool())
            result = reg.execute("net_tool")
            assert result.success
        finally:
            pe._engine = old

    @patch("missy.tools.registry.get_policy_engine")
    def test_policy_violation_returns_failure(self, mock_get_engine):
        from missy.core.exceptions import PolicyViolationError
        engine = MagicMock()
        engine.check_network.side_effect = PolicyViolationError("blocked", category="network", detail="host not allowed")
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(NetworkTool())
        result = reg.execute("net_tool")
        assert not result.success
        assert "blocked" in result.error

    @patch("missy.tools.registry.get_policy_engine")
    def test_shell_policy_check(self, mock_get_engine):
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.execute("shell_tool", command="ls -la")
        engine.check_shell.assert_called_once()

    @patch("missy.tools.registry.get_policy_engine")
    def test_fs_read_write_policy_checks(self, mock_get_engine):
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(FSTool())
        reg.execute("fs_tool")
        engine.check_read.assert_called_once_with("/tmp/test", session_id="", task_id="")
        engine.check_write.assert_called_once_with("/tmp/test", session_id="", task_id="")


class TestToolRegistryQueries:
    def test_list_tools_sorted(self):
        reg = ToolRegistry()
        reg.register(CrashTool())
        reg.register(EchoTool())
        assert reg.list_tools() == ["crash", "echo"]

    def test_get_returns_none_for_missing(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_register_replaces_existing(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        new_echo = EchoTool()
        reg.register(new_echo)
        assert reg.get("echo") is new_echo


class TestSingleton:
    def test_init_and_get(self):
        reg = init_tool_registry()
        assert get_tool_registry() is reg

    def test_get_without_init_raises(self):
        import missy.tools.registry as mod
        with mod._lock:
            old = mod._registry
            mod._registry = None
        try:
            with pytest.raises(RuntimeError, match="not been initialised"):
                get_tool_registry()
        finally:
            with mod._lock:
                mod._registry = old


class TestEmitEvent:
    @patch("missy.tools.registry.event_bus.publish", side_effect=Exception("bus down"))
    def test_emit_event_failure_does_not_crash(self, mock_pub):
        reg = ToolRegistry()
        reg.register(EchoTool())
        result = reg.execute("echo", text="hi")
        assert result.success  # should still succeed
