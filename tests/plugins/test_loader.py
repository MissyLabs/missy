"""Tests for missy.plugins.loader."""

from __future__ import annotations

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
# Helpers
# ---------------------------------------------------------------------------


def _make_config(enabled: bool = True, allowed: list[str] | None = None) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(enabled=enabled, allowed_plugins=allowed or []),
        providers={},
        workspace_path=".",
        audit_log_path="~/.missy/audit.log",
    )


class GoodPlugin(BasePlugin):
    name = "good"
    description = "Always initialises"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, *, x: int = 1) -> int:
        return x * 3


class BadInitPlugin(BasePlugin):
    name = "bad_init"
    description = "initialize() returns False"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return False

    def execute(self, **kwargs):
        pass


class CrashInitPlugin(BasePlugin):
    name = "crash_init"
    description = "initialize() raises"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        raise RuntimeError("init crash")

    def execute(self, **kwargs):
        pass


class CrashExecutePlugin(BasePlugin):
    name = "crash_exec"
    description = "execute() raises"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs):
        raise RuntimeError("exec crash")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allowed_loader() -> PluginLoader:
    return PluginLoader(_make_config(enabled=True, allowed=["good"]))


@pytest.fixture
def disabled_loader() -> PluginLoader:
    return PluginLoader(_make_config(enabled=False))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadPlugin:
    def test_load_succeeds_when_allowed(self, allowed_loader: PluginLoader):
        result = allowed_loader.load_plugin(GoodPlugin())
        assert result is True

    def test_load_sets_enabled_true(self, allowed_loader: PluginLoader):
        plugin = GoodPlugin()
        allowed_loader.load_plugin(plugin)
        assert plugin.enabled is True

    def test_load_denied_when_plugins_globally_disabled(self, disabled_loader: PluginLoader):
        with pytest.raises(PolicyViolationError, match="plugins are disabled"):
            disabled_loader.load_plugin(GoodPlugin())

    def test_load_denied_when_not_in_allowed_list(self):
        loader = PluginLoader(_make_config(enabled=True, allowed=["other"]))
        with pytest.raises(PolicyViolationError, match="not in allowed_plugins list"):
            loader.load_plugin(GoodPlugin())

    def test_load_returns_false_when_init_fails(self):
        loader = PluginLoader(_make_config(enabled=True, allowed=["bad_init"]))
        result = loader.load_plugin(BadInitPlugin())
        assert result is False

    def test_load_returns_false_when_init_crashes(self):
        loader = PluginLoader(_make_config(enabled=True, allowed=["crash_init"]))
        result = loader.load_plugin(CrashInitPlugin())
        assert result is False

    def test_load_denied_emits_deny_audit_event(self, disabled_loader: PluginLoader):
        event_bus.clear()
        with pytest.raises(PolicyViolationError):
            disabled_loader.load_plugin(GoodPlugin())
        events = event_bus.get_events(event_type="plugin.load", result="deny")
        assert len(events) == 1

    def test_load_success_emits_allow_audit_event(self, allowed_loader: PluginLoader):
        event_bus.clear()
        allowed_loader.load_plugin(GoodPlugin())
        events = event_bus.get_events(event_type="plugin.load", result="allow")
        assert len(events) == 1
        assert events[0].detail["plugin"] == "good"

    def test_load_bad_init_emits_error_audit_event(self):
        event_bus.clear()
        loader = PluginLoader(_make_config(enabled=True, allowed=["bad_init"]))
        loader.load_plugin(BadInitPlugin())
        events = event_bus.get_events(event_type="plugin.load", result="error")
        assert len(events) == 1


class TestListPlugins:
    def test_empty_before_any_loads(self, allowed_loader: PluginLoader):
        assert allowed_loader.list_plugins() == []

    def test_returns_manifest_after_load(self, allowed_loader: PluginLoader):
        allowed_loader.load_plugin(GoodPlugin())
        manifests = allowed_loader.list_plugins()
        assert len(manifests) == 1
        assert manifests[0]["name"] == "good"

    def test_failed_load_not_in_list(self):
        loader = PluginLoader(_make_config(enabled=True, allowed=["bad_init"]))
        loader.load_plugin(BadInitPlugin())
        assert loader.list_plugins() == []


class TestGetPlugin:
    def test_returns_plugin_after_load(self, allowed_loader: PluginLoader):
        plugin = GoodPlugin()
        allowed_loader.load_plugin(plugin)
        assert allowed_loader.get_plugin("good") is plugin

    def test_returns_none_when_not_loaded(self, allowed_loader: PluginLoader):
        assert allowed_loader.get_plugin("nonexistent") is None


class TestExecutePlugin:
    def test_execute_returns_result(self, allowed_loader: PluginLoader):
        allowed_loader.load_plugin(GoodPlugin())
        result = allowed_loader.execute("good", x=7)
        assert result == 21

    def test_execute_unloaded_raises_policy_violation(self, allowed_loader: PluginLoader):
        with pytest.raises(PolicyViolationError, match="not loaded"):
            allowed_loader.execute("ghost")

    def test_execute_raises_if_plugin_not_enabled(self, allowed_loader: PluginLoader):
        plugin = GoodPlugin()
        allowed_loader.load_plugin(plugin)
        plugin.enabled = False  # Force disable after load
        with pytest.raises(PolicyViolationError, match="not enabled"):
            allowed_loader.execute("good")

    def test_execute_crashing_plugin_propagates_exception(self):
        loader = PluginLoader(_make_config(enabled=True, allowed=["crash_exec"]))
        loader.load_plugin(CrashExecutePlugin())
        with pytest.raises(RuntimeError, match="exec crash"):
            loader.execute("crash_exec")

    def test_execute_emits_allow_event_on_success(self, allowed_loader: PluginLoader):
        event_bus.clear()
        allowed_loader.load_plugin(GoodPlugin())
        allowed_loader.execute("good", session_id="s", task_id="t", x=1)
        events = event_bus.get_events(event_type="plugin.execute", result="allow")
        assert len(events) == 1

    def test_execute_emits_error_event_on_crash(self):
        event_bus.clear()
        loader = PluginLoader(_make_config(enabled=True, allowed=["crash_exec"]))
        loader.load_plugin(CrashExecutePlugin())
        with pytest.raises(RuntimeError):
            loader.execute("crash_exec")
        events = event_bus.get_events(event_type="plugin.execute", result="error")
        assert len(events) == 1

    def test_execute_emits_deny_event_when_not_loaded(self, allowed_loader: PluginLoader):
        event_bus.clear()
        with pytest.raises(PolicyViolationError):
            allowed_loader.execute("ghost")
        events = event_bus.get_events(event_type="plugin.execute", result="deny")
        assert len(events) == 1


class TestSingleton:
    def test_init_returns_loader(self):
        config = _make_config()
        loader = init_plugin_loader(config)
        assert isinstance(loader, PluginLoader)

    def test_get_returns_same_instance(self):
        config = _make_config()
        loader = init_plugin_loader(config)
        assert get_plugin_loader() is loader

    def test_get_before_init_raises(self, monkeypatch):
        import missy.plugins.loader as mod
        monkeypatch.setattr(mod, "_loader", None)
        with pytest.raises(RuntimeError, match="PluginLoader has not been initialised"):
            get_plugin_loader()

    def test_second_init_replaces_first(self):
        c = _make_config()
        l1 = init_plugin_loader(c)
        l2 = init_plugin_loader(c)
        assert l1 is not l2
        assert get_plugin_loader() is l2
