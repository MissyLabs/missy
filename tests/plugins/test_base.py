"""Tests for missy.plugins.base."""

from __future__ import annotations

import pytest

from missy.plugins.base import BasePlugin, PluginPermissions


class ConcretePlugin(BasePlugin):
    name = "concrete"
    description = "A test plugin"
    permissions = PluginPermissions(network=True, allowed_hosts=["api.example.com"])

    def initialize(self) -> bool:
        return True

    def execute(self, *, value: int = 0):
        return value + 1


class TestPluginPermissionsDefaults:
    def test_all_flags_default_false(self):
        p = PluginPermissions()
        assert p.network is False
        assert p.filesystem_read is False
        assert p.filesystem_write is False
        assert p.shell is False

    def test_allowed_hosts_defaults_empty(self):
        p = PluginPermissions()
        assert p.allowed_hosts == []

    def test_allowed_paths_defaults_empty(self):
        p = PluginPermissions()
        assert p.allowed_paths == []

    def test_can_set_all_fields(self):
        p = PluginPermissions(
            network=True,
            filesystem_read=True,
            allowed_hosts=["api.x.com"],
            allowed_paths=["/tmp"],
        )
        assert p.network is True
        assert p.filesystem_read is True
        assert p.allowed_hosts == ["api.x.com"]
        assert p.allowed_paths == ["/tmp"]


class TestBasePlugin:
    def test_default_enabled_is_false(self):
        plugin = ConcretePlugin()
        assert plugin.enabled is False

    def test_default_version(self):
        assert ConcretePlugin.version == "0.1.0"

    def test_initialize_returns_bool(self):
        plugin = ConcretePlugin()
        result = plugin.initialize()
        assert isinstance(result, bool)
        assert result is True

    def test_execute_returns_value(self):
        plugin = ConcretePlugin()
        assert plugin.execute(value=10) == 11

    def test_get_manifest_keys(self):
        plugin = ConcretePlugin()
        manifest = plugin.get_manifest()
        assert set(manifest.keys()) == {"name", "version", "description", "permissions", "enabled"}

    def test_get_manifest_values(self):
        plugin = ConcretePlugin()
        manifest = plugin.get_manifest()
        assert manifest["name"] == "concrete"
        assert manifest["version"] == "0.1.0"
        assert manifest["description"] == "A test plugin"
        assert manifest["enabled"] is False

    def test_get_manifest_permissions_dict(self):
        plugin = ConcretePlugin()
        manifest = plugin.get_manifest()
        perms = manifest["permissions"]
        assert isinstance(perms, dict)
        assert perms["network"] is True
        assert perms["allowed_hosts"] == ["api.example.com"]

    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BasePlugin()  # type: ignore[abstract]
