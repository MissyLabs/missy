"""Requirement-focused tool checks from validation run 19."""

from __future__ import annotations

import threading
from unittest.mock import patch

from missy.core.events import event_bus
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry


class TestTdeep044RegistryReplacement:
    def test_inflight_call_finishes_on_a_and_next_call_uses_b(self):
        entered = threading.Event()
        release = threading.Event()

        class VersionA(BaseTool):
            name = "versioned_tool"
            description = "Version A"
            permissions = ToolPermissions()

            def execute(self, **kwargs):
                entered.set()
                release.wait(2)
                return ToolResult(success=True, output="A")

        class VersionB(BaseTool):
            name = "versioned_tool"
            description = "Version B"
            permissions = ToolPermissions()

            def execute(self, **kwargs):
                return ToolResult(success=True, output="B")

        event_bus.clear()
        registry_a = ToolRegistry()
        registry_a.register(VersionA())
        holder = {}
        with patch("missy.tools.registry._registry", registry_a):
            thread = threading.Thread(
                target=lambda: holder.setdefault(
                    "result_a",
                    registry_a.execute("versioned_tool", session_id="s", task_id="a"),
                )
            )
            thread.start()
            assert entered.wait(2)

            registry_b = init_tool_registry()
            registry_b.register(VersionB())
            assert get_tool_registry() is registry_b
            result_b = registry_b.execute("versioned_tool", session_id="s", task_id="b")
            stale = registry_a.execute("versioned_tool")

            release.set()
            thread.join(2)

        assert holder["result_a"].output == "A"
        assert result_b.output == "B"
        assert not stale.success
        assert "replaced" in stale.error
        task_ids = {
            event.task_id
            for event in event_bus.get_events(event_type="tool_execute")
            if event.detail.get("tool") == "versioned_tool"
        }
        assert {"a", "b"}.issubset(task_ids)
