"""Run-18 tool reference-monitor and registry-integrity regressions."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from missy.core.events import event_bus
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry


class _SnapshotTool(BaseTool):
    name = "snapshot_tool"
    description = "Validate immutable argument snapshots"
    permissions = ToolPermissions(filesystem_read=True)

    def __init__(self, entered=None, release=None):
        self.entered = entered
        self.release = release
        self.executed_path = None

    def resolve_filesystem_targets(self, kwargs):
        if self.entered:
            self.entered.set()
            self.release.wait(2)
        return [kwargs["nested"]["path"]], []

    def execute(self, **kwargs):
        self.executed_path = kwargs["nested"]["path"]
        return ToolResult(success=True, output=self.executed_path)


def test_tdeep_049_policy_and_execution_use_one_public_argument_snapshot() -> None:
    entered = threading.Event()
    release = threading.Event()
    tool = _SnapshotTool(entered, release)
    registry = ToolRegistry()
    registry.register(tool)
    policy = MagicMock()
    original = {"nested": {"path": "/safe/input"}}
    holder = {}
    with patch("missy.tools.registry.get_policy_engine", return_value=policy):
        thread = threading.Thread(
            target=lambda: holder.setdefault(
                "result", registry.execute("snapshot_tool", **original)
            )
        )
        thread.start()
        assert entered.wait(2)
        original["nested"]["path"] = "/outside/secret"
        release.set()
        thread.join(2)
    policy.check_read.assert_called_once_with("/safe/input", session_id="", task_id="")
    assert tool.executed_path == "/safe/input"
    assert holder["result"].success


def test_tdeep_050_censor_failure_never_falls_back_to_raw_tool_error() -> None:
    class Failing(BaseTool):
        name = "failing_tool"
        description = "Fail"
        permissions = ToolPermissions()

        def execute(self, **kwargs):
            raise RuntimeError("sk-test-abcdefghijklmnopqrstuvwxyz")

    event_bus.clear()
    registry = ToolRegistry()
    registry.register(Failing())
    with patch("missy.security.censor.censor_response", side_effect=RuntimeError("censor down")):
        result = registry.execute("failing_tool")
    assert "sk-test" not in result.error
    event = event_bus.get_events(event_type="tool_execute")[-1]
    assert "sk-test" not in str(event.detail)


def test_toolschema_013_collision_is_atomic_and_non_shadowing() -> None:
    registry = ToolRegistry()
    tools = [_SnapshotTool() for _ in range(20)]
    successes = []
    failures = []

    def register(tool):
        try:
            registry.register(tool)
            successes.append(tool)
        except ValueError:
            failures.append(tool)

    threads = [threading.Thread(target=register, args=(tool,)) for tool in tools]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert len(successes) == 1
    assert len(failures) == 19
    assert registry.get("snapshot_tool") is successes[0]


def test_toolschema_015_unknown_tool_is_never_reported_enabled() -> None:
    assert not ToolRegistry().is_enabled("unknown")


def test_tdeep_051_registry_replacement_revokes_stale_dispatch_only() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Blocking(BaseTool):
        name = "blocking"
        description = "Block one in-flight invocation"
        permissions = ToolPermissions()

        def execute(self, **kwargs):
            entered.set()
            release.wait(2)
            return ToolResult(success=True, output="old-finished")

    old = ToolRegistry()
    old.register(Blocking())
    holder = {}
    with patch("missy.tools.registry._registry", old):
        thread = threading.Thread(
            target=lambda: holder.setdefault("result", old.execute("blocking"))
        )
        thread.start()
        assert entered.wait(2)
        current = init_tool_registry()
        assert get_tool_registry() is current
        stale = old.execute("blocking")
        assert not stale.success
        assert "replaced" in stale.error
        release.set()
        thread.join(2)
        assert holder["result"].output == "old-finished"


def test_tdeep_055_resolvers_are_single_call_typed_and_isolated() -> None:
    class Resolving(BaseTool):
        name = "resolving"
        description = "Resolve one network target"
        permissions = ToolPermissions(network=True)

        def __init__(self):
            self.calls = 0
            self.executed = False

        def resolve_network_hosts(self, kwargs):
            self.calls += 1
            kwargs["nested"]["host"] = "mutated.example"
            return ["safe.example"]

        def execute(self, **kwargs):
            self.executed = True
            return ToolResult(success=True, output="unexpected")

    tool = Resolving()
    registry = ToolRegistry()
    registry.register(tool)
    with patch("missy.tools.registry.get_policy_engine", return_value=MagicMock()):
        result = registry.execute("resolving", nested={"host": "safe.example"})
    assert not result.success
    assert not tool.executed
    assert tool.calls == 1


def test_tdeep_055_resolver_generators_and_hangs_are_bounded() -> None:
    class Resolving(BaseTool):
        name = "resolving"
        description = "Return a selected invalid resolver output"
        permissions = ToolPermissions(network=True)

        def __init__(self, mode):
            self.mode = mode
            self.executed = False

        def resolve_network_hosts(self, kwargs):
            if self.mode == "generator":
                return (host for host in ["safe.example"])
            time.sleep(1)
            return ["safe.example"]

        def execute(self, **kwargs):
            self.executed = True
            return ToolResult(success=True, output="unexpected")

    for mode in ("generator", "hang"):
        tool = Resolving(mode)
        registry = ToolRegistry()
        registry.register(tool)
        started = time.monotonic()
        with patch("missy.tools.registry.get_policy_engine", return_value=MagicMock()):
            result = registry.execute("resolving")
        assert time.monotonic() - started < 0.8
        assert not result.success
        assert not tool.executed
