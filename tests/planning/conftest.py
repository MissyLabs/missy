"""Shared fakes for planning-kernel tests (F02)."""

from __future__ import annotations

import threading
import time

import pytest

from missy.tools.base import ToolResult


class FakeRegistry:
    """A minimal stand-in for ToolRegistry with a fixed set of test tools."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def list_tools(self) -> list[str]:
        return ["upper", "concat", "slow", "fail", "raise", "echo", "number"]

    def execute(self, name, /, session_id="", task_id="", **kwargs) -> ToolResult:
        with self._lock:
            self.calls.append((name, dict(kwargs)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return self._dispatch(name, kwargs)
        finally:
            with self._lock:
                self.active -= 1

    def _dispatch(self, name, kwargs) -> ToolResult:
        if name == "upper":
            return ToolResult(success=True, output=str(kwargs.get("text", "")).upper())
        if name == "concat":
            return ToolResult(success=True, output=f"{kwargs.get('a')}|{kwargs.get('b')}")
        if name == "echo":
            return ToolResult(success=True, output=kwargs.get("value"))
        if name == "number":
            return ToolResult(success=True, output=int(kwargs.get("n", 0)))
        if name == "slow":
            time.sleep(0.05)
            return ToolResult(success=True, output=kwargs.get("v"))
        if name == "fail":
            return ToolResult(success=False, output=None, error="boom")
        if name == "raise":
            raise RuntimeError("tool exploded")
        return ToolResult(success=False, output=None, error=f"unknown tool {name}")


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry()
