"""Hardening tests for plugin loader: edge cases, concurrency, error paths."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from missy.core.exceptions import PolicyViolationError
from missy.plugins.base import BasePlugin, PluginPermissions
from missy.plugins.loader import PluginLoader, get_plugin_loader, init_plugin_loader


@dataclass
class _PluginsConfig:
    enabled: bool = False
    allowed_plugins: list[str] = field(default_factory=list)


@dataclass
class _MockConfig:
    plugins: _PluginsConfig = field(default_factory=_PluginsConfig)


class _GoodPlugin(BasePlugin):
    name = "good_plugin"
    description = "A test plugin"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs):
        return {"result": "ok", **kwargs}


class _BadInitPlugin(BasePlugin):
    name = "bad_init"
    description = "Fails init"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return False

    def execute(self, **kwargs):
        return None


class _ExplodingPlugin(BasePlugin):
    name = "exploding"
    description = "Raises on init"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        raise RuntimeError("kaboom")

    def execute(self, **kwargs):
        raise RuntimeError("execution kaboom")


class _ExplodingExecutePlugin(BasePlugin):
    name = "exploding_exec"
    description = "Raises on execute"
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs):
        raise ValueError("exec error")


class TestPluginLoaderPolicyEnforcement:
    """Tests for plugin policy enforcement edge cases."""

    def test_plugins_disabled_raises(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=False))
        loader = PluginLoader(config)
        with pytest.raises(PolicyViolationError, match="plugins are disabled"):
            loader.load_plugin(_GoodPlugin())

    def test_plugin_not_in_allowlist_raises(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["other"]))
        loader = PluginLoader(config)
        with pytest.raises(PolicyViolationError, match="not in allowed_plugins"):
            loader.load_plugin(_GoodPlugin())

    def test_successful_load(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        result = loader.load_plugin(_GoodPlugin())
        assert result is True

    def test_init_failure_returns_false(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["bad_init"]))
        loader = PluginLoader(config)
        result = loader.load_plugin(_BadInitPlugin())
        assert result is False

    def test_init_exception_returns_false(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["exploding"]))
        loader = PluginLoader(config)
        result = loader.load_plugin(_ExplodingPlugin())
        assert result is False

    def test_execute_not_loaded_raises(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True))
        loader = PluginLoader(config)
        with pytest.raises(PolicyViolationError, match="not loaded"):
            loader.execute("nonexistent")

    def test_execute_disabled_plugin_raises(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        plugin = _GoodPlugin()
        loader.load_plugin(plugin)
        plugin.enabled = False
        with pytest.raises(PolicyViolationError, match="not enabled"):
            loader.execute("good_plugin")

    def test_execute_success(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        loader.load_plugin(_GoodPlugin())
        result = loader.execute("good_plugin", key="val")
        assert result == {"result": "ok", "key": "val"}

    def test_execute_exception_propagates(self) -> None:
        config = _MockConfig(
            plugins=_PluginsConfig(enabled=True, allowed_plugins=["exploding_exec"])
        )
        loader = PluginLoader(config)
        loader.load_plugin(_ExplodingExecutePlugin())
        with pytest.raises(ValueError, match="exec error"):
            loader.execute("exploding_exec")


class TestPluginLoaderQueries:
    """Tests for plugin listing and retrieval."""

    def test_list_plugins_empty(self) -> None:
        config = _MockConfig()
        loader = PluginLoader(config)
        assert loader.list_plugins() == []

    def test_list_plugins_with_loaded(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        loader.load_plugin(_GoodPlugin())
        plugins = loader.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "good_plugin"

    def test_get_plugin_exists(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        loader.load_plugin(_GoodPlugin())
        assert loader.get_plugin("good_plugin") is not None

    def test_get_plugin_missing(self) -> None:
        config = _MockConfig()
        loader = PluginLoader(config)
        assert loader.get_plugin("nonexistent") is None


class TestPluginSingleton:
    """Tests for module-level singleton."""

    def test_init_and_get(self) -> None:
        config = _MockConfig()
        loader = init_plugin_loader(config)
        assert get_plugin_loader() is loader

    def test_get_before_init_raises(self) -> None:
        import missy.plugins.loader as mod

        old = mod._loader
        try:
            mod._loader = None
            with pytest.raises(RuntimeError, match="not been initialised"):
                get_plugin_loader()
        finally:
            mod._loader = old

    def test_concurrent_init(self) -> None:
        results = []
        errors = []

        def init_loader(idx):
            try:
                config = _MockConfig()
                loader = init_plugin_loader(config)
                results.append(loader)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=init_loader, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(results) == 10


class TestPluginAuditEvents:
    """Test that plugin operations emit audit events."""

    def test_load_denied_emits_event(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=False))
        loader = PluginLoader(config)
        events = []
        with patch("missy.plugins.loader.event_bus") as mock_bus:
            mock_bus.publish.side_effect = lambda e: events.append(e)
            with pytest.raises(PolicyViolationError):
                loader.load_plugin(_GoodPlugin())
        assert len(events) == 1
        assert events[0].result == "deny"

    def test_load_success_emits_event(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        events = []
        with patch("missy.plugins.loader.event_bus") as mock_bus:
            mock_bus.publish.side_effect = lambda e: events.append(e)
            loader.load_plugin(_GoodPlugin())
        assert any(e.result == "allow" for e in events)

    def test_execute_emits_start_and_end(self) -> None:
        config = _MockConfig(plugins=_PluginsConfig(enabled=True, allowed_plugins=["good_plugin"]))
        loader = PluginLoader(config)
        events = []
        with patch("missy.plugins.loader.event_bus") as mock_bus:
            mock_bus.publish.side_effect = lambda e: events.append(e)
            loader.load_plugin(_GoodPlugin())
            events.clear()
            loader.execute("good_plugin")
        assert len(events) == 2  # start + end
        types = [e.event_type for e in events]
        assert "plugin.execute.start" in types
        assert "plugin.execute" in types
