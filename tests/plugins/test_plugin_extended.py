"""Extended tests for missy.plugins — loader, base, lifecycle, thread safety, and edge cases.

Covers areas not addressed by test_base.py and test_loader.py:
- Plugin loading from multiple sources / replacement behaviour
- Permission declaration completeness and manifest accuracy
- Full lifecycle: load → enable → disable → reload
- Plugin isolation: one crash does not affect siblings
- Config validation edge cases (empty allowed list, wildcard-like names)
- Duplicate name handling
- Concurrent load and execute (thread safety of the singleton)
- Audit event fields (session_id / task_id forwarding)
- execute() keyword argument forwarding
- get_plugin() contract after failed initialisation
- list_plugins() ordering and content
- init_plugin_loader() singleton replacement atomicity
- Policy error messages contain the plugin name
- Plugin with all permissions declared
- Plugin with filesystem permissions
- Plugin with shell permission
- execute() start event is emitted before result event
- execute() with no extra kwargs
- load_plugin() crash init does NOT leave plugin in registry
- load_plugin() bad-init does NOT leave plugin in registry
- Plugin version attribute propagation into manifest
- Plugin description propagation into manifest
- Manifest permissions dict has all expected keys
- Re-loading same plugin name after previous failure
- Loading two distinct plugins into one loader
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.plugins.base import BasePlugin, PluginPermissions
from missy.plugins.loader import PluginLoader, get_plugin_loader, init_plugin_loader

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _cfg(
    enabled: bool = True,
    allowed: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(enabled=enabled, allowed_plugins=allowed or []),
        providers={},
        workspace_path=".",
        audit_log_path="~/.missy/audit.log",
    )


def _loader(enabled: bool = True, allowed: list[str] | None = None) -> PluginLoader:
    return PluginLoader(_cfg(enabled=enabled, allowed=allowed))


# ---------------------------------------------------------------------------
# Concrete plugin fixtures used across multiple test classes
# ---------------------------------------------------------------------------


class AlphaPlugin(BasePlugin):
    name = "alpha"
    description = "Alpha plugin"
    version = "1.2.3"
    permissions = PluginPermissions(network=True, allowed_hosts=["alpha.example.com"])

    def initialize(self) -> bool:
        return True

    def execute(self, *, multiplier: int = 2) -> int:
        return 10 * multiplier


class BetaPlugin(BasePlugin):
    name = "beta"
    description = "Beta plugin"
    permissions = PluginPermissions(filesystem_read=True, allowed_paths=["/tmp/beta"])

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        return "beta-result"


class FullPermsPlugin(BasePlugin):
    name = "fullperms"
    description = "Plugin with every permission set"
    permissions = PluginPermissions(
        network=True,
        filesystem_read=True,
        filesystem_write=True,
        shell=True,
        allowed_hosts=["svc.example.com"],
        allowed_paths=["/data", "/var/log"],
    )

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> bool:
        return True


class StatefulPlugin(BasePlugin):
    """Records calls so tests can assert on them."""

    name = "stateful"
    description = "Tracks init/exec calls"
    permissions = PluginPermissions()

    def __init__(self) -> None:
        super().__init__()
        self.init_called: int = 0
        self.exec_called: int = 0

    def initialize(self) -> bool:
        self.init_called += 1
        return True

    def execute(self, *, payload: str = "") -> str:
        self.exec_called += 1
        return payload.upper()


class SlowInitPlugin(BasePlugin):
    """initialize() sleeps briefly — used for concurrency tests."""

    name = "slow"
    description = "Slow initialiser"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        time.sleep(0.05)
        return True

    def execute(self, **kwargs: Any) -> str:
        return "done"


class CrashExecPlugin(BasePlugin):
    name = "crashexec"
    description = "execute() always raises"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> None:
        raise ValueError("intentional execute crash")


class BadInitPlugin(BasePlugin):
    name = "badinit"
    description = "initialize() returns False"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return False

    def execute(self, **kwargs: Any) -> None:
        pass


class CrashInitPlugin(BasePlugin):
    name = "crashinit"
    description = "initialize() raises"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        raise OSError("disk full")

    def execute(self, **kwargs: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# 1. Plugin loading from multiple sources / distinct plugins in one loader
# ---------------------------------------------------------------------------


class TestMultiplePluginLoading:
    def test_two_plugins_loaded_into_one_loader(self):
        ldr = _loader(allowed=["alpha", "beta"])
        assert ldr.load_plugin(AlphaPlugin()) is True
        assert ldr.load_plugin(BetaPlugin()) is True
        names = {m["name"] for m in ldr.list_plugins()}
        assert names == {"alpha", "beta"}

    def test_second_plugin_independent_of_first(self):
        ldr = _loader(allowed=["alpha", "beta"])
        ldr.load_plugin(AlphaPlugin())
        # Beta should load even though alpha is already present
        result = ldr.load_plugin(BetaPlugin())
        assert result is True

    def test_execute_correct_plugin_when_multiple_loaded(self):
        ldr = _loader(allowed=["alpha", "beta"])
        ldr.load_plugin(AlphaPlugin())
        ldr.load_plugin(BetaPlugin())
        assert ldr.execute("alpha", multiplier=3) == 30
        assert ldr.execute("beta") == "beta-result"


# ---------------------------------------------------------------------------
# 2. Permission declaration and manifest accuracy
# ---------------------------------------------------------------------------


class TestPermissionManifest:
    def test_manifest_permissions_contains_all_keys(self):
        plugin = AlphaPlugin()
        perms = plugin.get_manifest()["permissions"]
        expected_keys = {"network", "filesystem_read", "filesystem_write", "shell",
                         "allowed_hosts", "allowed_paths"}
        assert expected_keys == set(perms.keys())

    def test_full_permissions_plugin_manifest_all_true(self):
        plugin = FullPermsPlugin()
        perms = plugin.get_manifest()["permissions"]
        assert perms["network"] is True
        assert perms["filesystem_read"] is True
        assert perms["filesystem_write"] is True
        assert perms["shell"] is True
        assert perms["allowed_hosts"] == ["svc.example.com"]
        assert perms["allowed_paths"] == ["/data", "/var/log"]

    def test_filesystem_plugin_permissions_in_manifest(self):
        plugin = BetaPlugin()
        perms = plugin.get_manifest()["permissions"]
        assert perms["filesystem_read"] is True
        assert perms["network"] is False
        assert perms["allowed_paths"] == ["/tmp/beta"]

    def test_network_false_plugin_has_empty_allowed_hosts(self):
        plugin = StatefulPlugin()
        perms = plugin.get_manifest()["permissions"]
        assert perms["network"] is False
        assert perms["allowed_hosts"] == []

    def test_manifest_version_propagates_custom_version(self):
        plugin = AlphaPlugin()
        assert plugin.get_manifest()["version"] == "1.2.3"

    def test_manifest_description_propagates(self):
        plugin = BetaPlugin()
        assert plugin.get_manifest()["description"] == "Beta plugin"

    def test_manifest_enabled_reflects_current_state(self):
        ldr = _loader(allowed=["alpha"])
        plugin = AlphaPlugin()
        assert plugin.get_manifest()["enabled"] is False
        ldr.load_plugin(plugin)
        assert plugin.get_manifest()["enabled"] is True


# ---------------------------------------------------------------------------
# 3. Plugin lifecycle: load, disable, reload
# ---------------------------------------------------------------------------


class TestPluginLifecycle:
    def test_plugin_starts_disabled(self):
        plugin = StatefulPlugin()
        assert plugin.enabled is False

    def test_load_sets_enabled(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        assert plugin.enabled is True

    def test_manual_disable_after_load_prevents_execute(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        plugin.enabled = False
        with pytest.raises(PolicyViolationError, match="not enabled"):
            ldr.execute("stateful")

    def test_re_enable_after_manual_disable_allows_execute(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        plugin.enabled = False
        plugin.enabled = True
        result = ldr.execute("stateful", payload="hello")
        assert result == "HELLO"

    def test_initialize_called_exactly_once_on_load(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        assert plugin.init_called == 1

    def test_reload_same_name_replaces_registry_entry(self):
        ldr = _loader(allowed=["stateful"])
        plugin_a = StatefulPlugin()
        ldr.load_plugin(plugin_a)
        plugin_b = StatefulPlugin()
        ldr.load_plugin(plugin_b)
        # The registry should now point to the second instance
        assert ldr.get_plugin("stateful") is plugin_b

    def test_execute_tracks_call_count(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        ldr.execute("stateful", payload="a")
        ldr.execute("stateful", payload="b")
        assert plugin.exec_called == 2


# ---------------------------------------------------------------------------
# 4. Plugin isolation and error handling
# ---------------------------------------------------------------------------


class TestPluginIsolation:
    def test_crash_in_one_plugin_does_not_affect_sibling(self):
        ldr = _loader(allowed=["crashexec", "alpha"])
        ldr.load_plugin(CrashExecPlugin())
        ldr.load_plugin(AlphaPlugin())
        with pytest.raises(ValueError):
            ldr.execute("crashexec")
        # alpha must still be callable
        assert ldr.execute("alpha", multiplier=1) == 10

    def test_crash_init_does_not_block_other_plugin_load(self):
        ldr = _loader(allowed=["crashinit", "alpha"])
        ldr.load_plugin(CrashInitPlugin())  # returns False, no raise
        result = ldr.load_plugin(AlphaPlugin())
        assert result is True

    def test_bad_init_plugin_not_in_registry(self):
        ldr = _loader(allowed=["badinit"])
        ldr.load_plugin(BadInitPlugin())
        assert ldr.get_plugin("badinit") is None

    def test_crash_init_plugin_not_in_registry(self):
        ldr = _loader(allowed=["crashinit"])
        ldr.load_plugin(CrashInitPlugin())
        assert ldr.get_plugin("crashinit") is None

    def test_execute_crash_propagates_original_exception_type(self):
        ldr = _loader(allowed=["crashexec"])
        ldr.load_plugin(CrashExecPlugin())
        with pytest.raises(ValueError, match="intentional execute crash"):
            ldr.execute("crashexec")


# ---------------------------------------------------------------------------
# 5. Config validation edge cases
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    def test_empty_allowed_list_blocks_all_plugins(self):
        ldr = _loader(enabled=True, allowed=[])
        with pytest.raises(PolicyViolationError, match="not in allowed_plugins list"):
            ldr.load_plugin(AlphaPlugin())

    def test_plugin_name_must_match_exactly(self):
        ldr = _loader(enabled=True, allowed=["alph"])  # one char short
        with pytest.raises(PolicyViolationError):
            ldr.load_plugin(AlphaPlugin())

    def test_case_sensitive_name_matching(self):
        ldr = _loader(enabled=True, allowed=["Alpha"])  # wrong case
        with pytest.raises(PolicyViolationError):
            ldr.load_plugin(AlphaPlugin())

    def test_plugins_disabled_overrides_allowed_list(self):
        ldr = _loader(enabled=False, allowed=["alpha"])
        with pytest.raises(PolicyViolationError, match="plugins are disabled"):
            ldr.load_plugin(AlphaPlugin())

    def test_error_message_contains_plugin_name_on_deny(self):
        ldr = _loader(enabled=True, allowed=[])
        with pytest.raises(PolicyViolationError) as exc_info:
            ldr.load_plugin(AlphaPlugin())
        assert "alpha" in str(exc_info.value)

    def test_execute_error_message_contains_plugin_name(self):
        ldr = _loader(allowed=[])
        with pytest.raises(PolicyViolationError) as exc_info:
            ldr.execute("alpha")
        assert "alpha" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Duplicate name handling
# ---------------------------------------------------------------------------


class TestDuplicateNames:
    def test_loading_duplicate_name_replaces_silently(self):
        ldr = _loader(allowed=["stateful"])
        p1 = StatefulPlugin()
        p2 = StatefulPlugin()
        ldr.load_plugin(p1)
        ldr.load_plugin(p2)
        assert ldr.get_plugin("stateful") is p2

    def test_list_plugins_shows_only_one_entry_after_duplicate_load(self):
        ldr = _loader(allowed=["stateful"])
        ldr.load_plugin(StatefulPlugin())
        ldr.load_plugin(StatefulPlugin())
        assert len(ldr.list_plugins()) == 1

    def test_execute_after_duplicate_load_uses_latest_instance(self):
        ldr = _loader(allowed=["stateful"])
        p1 = StatefulPlugin()
        ldr.load_plugin(p1)
        p2 = StatefulPlugin()
        ldr.load_plugin(p2)
        ldr.execute("stateful", payload="x")
        assert p1.exec_called == 0
        assert p2.exec_called == 1


# ---------------------------------------------------------------------------
# 7. Plugin discovery / list_plugins
# ---------------------------------------------------------------------------


class TestListPlugins:
    def test_list_returns_dicts_not_plugin_instances(self):
        ldr = _loader(allowed=["alpha"])
        ldr.load_plugin(AlphaPlugin())
        manifests = ldr.list_plugins()
        assert all(isinstance(m, dict) for m in manifests)

    def test_list_includes_enabled_true_after_load(self):
        ldr = _loader(allowed=["alpha"])
        ldr.load_plugin(AlphaPlugin())
        manifest = ldr.list_plugins()[0]
        assert manifest["enabled"] is True

    def test_list_returns_correct_count_with_multiple_plugins(self):
        ldr = _loader(allowed=["alpha", "beta", "stateful"])
        ldr.load_plugin(AlphaPlugin())
        ldr.load_plugin(BetaPlugin())
        ldr.load_plugin(StatefulPlugin())
        assert len(ldr.list_plugins()) == 3

    def test_list_plugins_empty_when_all_inits_fail(self):
        ldr = _loader(allowed=["badinit", "crashinit"])
        ldr.load_plugin(BadInitPlugin())
        ldr.load_plugin(CrashInitPlugin())
        assert ldr.list_plugins() == []


# ---------------------------------------------------------------------------
# 8. Audit event fields (session_id / task_id forwarding)
# ---------------------------------------------------------------------------


class TestAuditEventFields:
    def test_execute_success_event_carries_session_and_task_ids(self):
        event_bus.clear()
        ldr = _loader(allowed=["alpha"])
        ldr.load_plugin(AlphaPlugin())
        ldr.execute("alpha", session_id="sess-42", task_id="task-99", multiplier=1)
        events = event_bus.get_events(event_type="plugin.execute", result="allow")
        assert len(events) == 1
        ev = events[0]
        assert ev.session_id == "sess-42"
        assert ev.task_id == "task-99"

    def test_execute_start_event_emitted_before_result_event(self):
        event_bus.clear()
        ldr = _loader(allowed=["alpha"])
        ldr.load_plugin(AlphaPlugin())
        ldr.execute("alpha", session_id="s", task_id="t")
        all_events = event_bus.get_events()
        types = [e.event_type for e in all_events]
        start_idx = types.index("plugin.execute.start")
        result_idx = types.index("plugin.execute")
        assert start_idx < result_idx

    def test_execute_error_event_carries_session_and_task_ids(self):
        event_bus.clear()
        ldr = _loader(allowed=["crashexec"])
        ldr.load_plugin(CrashExecPlugin())
        with pytest.raises(ValueError):
            ldr.execute("crashexec", session_id="s1", task_id="t1")
        events = event_bus.get_events(event_type="plugin.execute", result="error")
        assert events[0].session_id == "s1"
        assert events[0].task_id == "t1"

    def test_deny_event_on_load_contains_reason_plugins_disabled(self):
        event_bus.clear()
        ldr = _loader(enabled=False)
        with pytest.raises(PolicyViolationError):
            ldr.load_plugin(AlphaPlugin())
        events = event_bus.get_events(event_type="plugin.load", result="deny")
        assert events[0].detail["reason"] == "plugins_disabled"

    def test_deny_event_on_load_contains_reason_not_in_allowed_list(self):
        event_bus.clear()
        ldr = _loader(enabled=True, allowed=[])
        with pytest.raises(PolicyViolationError):
            ldr.load_plugin(AlphaPlugin())
        events = event_bus.get_events(event_type="plugin.load", result="deny")
        assert events[0].detail["reason"] == "not_in_allowed_list"

    def test_load_allow_event_detail_contains_manifest(self):
        event_bus.clear()
        ldr = _loader(allowed=["alpha"])
        ldr.load_plugin(AlphaPlugin())
        events = event_bus.get_events(event_type="plugin.load", result="allow")
        assert "manifest" in events[0].detail
        assert events[0].detail["manifest"]["name"] == "alpha"


# ---------------------------------------------------------------------------
# 9. Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_loads_all_succeed(self):
        """Multiple threads loading distinct plugins into the same loader must all succeed."""
        plugins = {
            "alpha": AlphaPlugin,
            "beta": BetaPlugin,
            "stateful": StatefulPlugin,
            "fullperms": FullPermsPlugin,
        }
        ldr = _loader(allowed=list(plugins.keys()))
        errors: list[Exception] = []

        def load(plugin_cls: type) -> None:
            try:
                ldr.load_plugin(plugin_cls())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=load, args=(cls,)) for cls in plugins.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors during concurrent load: {errors}"
        assert len(ldr.list_plugins()) == len(plugins)

    def test_concurrent_executes_on_same_plugin(self):
        """Multiple threads executing the same stateful plugin should not crash."""
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)

        results: list[str] = []
        lock = threading.Lock()

        def run() -> None:
            res = ldr.execute("stateful", payload="ping")
            with lock:
                results.append(res)

        threads = [threading.Thread(target=run) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == ["PING"] * 10
        assert plugin.exec_called == 10

    def test_init_plugin_loader_singleton_replacement_is_atomic(self):
        """Rapid re-initialisation from two threads should leave a valid singleton."""
        cfg = _cfg(allowed=["alpha"])
        results: list[PluginLoader] = []
        lock = threading.Lock()

        def reinit() -> None:
            ldr = init_plugin_loader(cfg)
            with lock:
                results.append(ldr)

        threads = [threading.Thread(target=reinit) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # The global singleton should be one of the loaders we created.
        final = get_plugin_loader()
        assert final in results


# ---------------------------------------------------------------------------
# 10. Execute keyword argument forwarding
# ---------------------------------------------------------------------------


class TestExecuteKwargForwarding:
    def test_execute_forwards_all_kwargs_to_plugin(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        result = ldr.execute("stateful", payload="world")
        assert result == "WORLD"

    def test_execute_with_no_extra_kwargs_uses_plugin_defaults(self):
        ldr = _loader(allowed=["stateful"])
        plugin = StatefulPlugin()
        ldr.load_plugin(plugin)
        result = ldr.execute("stateful")
        # Default payload="" → "".upper() == ""
        assert result == ""

    def test_execute_with_numeric_kwarg(self):
        ldr = _loader(allowed=["alpha"])
        ldr.load_plugin(AlphaPlugin())
        assert ldr.execute("alpha", multiplier=5) == 50
