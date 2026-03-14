"""Tests for missy.tools.registry.ToolRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from missy.core.events import event_bus
from missy.tools import registry as registry_module
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "echo", succeed: bool = True) -> BaseTool:
    """Build a minimal concrete BaseTool for testing."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = f"Test tool: {name}"
    tool.permissions = ToolPermissions()
    if succeed:
        tool.execute.return_value = ToolResult(success=True, output=f"{name} result")
    else:
        tool.execute.return_value = ToolResult(
            success=False, output=None, error="deliberate failure"
        )
    return tool


@pytest.fixture(autouse=True)
def clear_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the module-level registry singleton is reset between tests."""
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# ToolRegistry.register and get
# ---------------------------------------------------------------------------


class TestRegisterAndGet:
    def test_register_then_get_returns_tool(self):
        registry = ToolRegistry()
        tool = _make_tool("echo")
        registry.register(tool)
        assert registry.get("echo") is tool

    def test_get_unknown_name_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_register_replaces_existing(self):
        registry = ToolRegistry()
        first = _make_tool("echo")
        second = _make_tool("echo")
        registry.register(first)
        registry.register(second)
        assert registry.get("echo") is second


# ---------------------------------------------------------------------------
# ToolRegistry.list_tools
# ---------------------------------------------------------------------------


class TestListTools:
    def test_empty_registry_returns_empty_list(self):
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_single_tool_listed(self):
        registry = ToolRegistry()
        registry.register(_make_tool("alpha"))
        assert registry.list_tools() == ["alpha"]

    def test_multiple_tools_sorted_alphabetically(self):
        registry = ToolRegistry()
        for name in ["zebra", "alpha", "mango"]:
            registry.register(_make_tool(name))
        assert registry.list_tools() == ["alpha", "mango", "zebra"]


# ---------------------------------------------------------------------------
# ToolRegistry.execute – success path
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    def test_execute_known_tool_returns_success(self):
        registry = ToolRegistry()
        registry.register(_make_tool("calc"))
        result = registry.execute("calc")
        assert result.success is True
        assert result.output == "calc result"

    def test_execute_passes_kwargs_to_tool(self):
        registry = ToolRegistry()
        tool = _make_tool("calc")
        tool.execute.return_value = ToolResult(success=True, output=42)
        registry.register(tool)
        registry.execute("calc", expression="1+1")
        tool.execute.assert_called_once_with(expression="1+1")

    def test_execute_emits_allow_audit_event(self):
        registry = ToolRegistry()
        registry.register(_make_tool("calc"))
        registry.execute("calc", session_id="s1", task_id="t1")
        events = event_bus.get_events(event_type="tool_execute")
        assert len(events) >= 1
        assert events[0].result == "allow"

    def test_execute_failure_result_emits_error_event(self):
        registry = ToolRegistry()
        registry.register(_make_tool("bad", succeed=False))
        result = registry.execute("bad")
        assert result.success is False
        events = event_bus.get_events(event_type="tool_execute")
        assert any(e.result == "error" for e in events)


# ---------------------------------------------------------------------------
# ToolRegistry.execute – unknown tool
# ---------------------------------------------------------------------------


class TestExecuteUnknown:
    def test_execute_unknown_tool_raises_key_error(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="ghost"):
            registry.execute("ghost")


# ---------------------------------------------------------------------------
# ToolRegistry.execute – tool raises exception
# ---------------------------------------------------------------------------


class TestExecuteException:
    def test_unhandled_exception_returns_failure_result(self):
        registry = ToolRegistry()
        tool = _make_tool("crash")
        tool.execute.side_effect = RuntimeError("boom")
        registry.register(tool)
        result = registry.execute("crash")
        assert result.success is False
        assert "boom" in (result.error or "")

    def test_exception_emits_error_audit_event(self):
        registry = ToolRegistry()
        tool = _make_tool("crash")
        tool.execute.side_effect = ValueError("oops")
        registry.register(tool)
        registry.execute("crash")
        events = event_bus.get_events(event_type="tool_execute")
        assert any(e.result == "error" for e in events)


# ---------------------------------------------------------------------------
# Module-level singleton: init_tool_registry / get_tool_registry
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_init_tool_registry_returns_tool_registry(self):
        registry = init_tool_registry()
        assert isinstance(registry, ToolRegistry)

    def test_get_tool_registry_before_init_raises(self):
        registry_module._registry = None
        with pytest.raises(RuntimeError, match="not been initialised"):
            get_tool_registry()

    def test_get_tool_registry_after_init_returns_same_instance(self):
        registry = init_tool_registry()
        retrieved = get_tool_registry()
        assert retrieved is registry

    def test_second_init_replaces_registry(self):
        first = init_tool_registry()
        second = init_tool_registry()
        assert get_tool_registry() is second
        assert first is not second
