"""Tests for missy.config.settings — configuration loading and policy dataclasses."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ProviderConfig,
    ShellPolicy,
    get_default_config,
    load_config,
)
from missy.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, content: str) -> str:
    """Write *content* to a temporary YAML file and return its string path."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(textwrap.dedent(content), encoding="utf-8")
    return str(cfg_file)


# ---------------------------------------------------------------------------
# get_default_config
# ---------------------------------------------------------------------------


class TestGetDefaultConfig:
    def test_returns_missy_config_instance(self):
        cfg = get_default_config()
        assert isinstance(cfg, MissyConfig)

    def test_network_default_deny_is_true(self):
        cfg = get_default_config()
        assert cfg.network.default_deny is True

    def test_network_allow_lists_are_empty(self):
        cfg = get_default_config()
        assert cfg.network.allowed_cidrs == []
        assert cfg.network.allowed_domains == []
        assert cfg.network.allowed_hosts == []

    def test_shell_disabled_by_default(self):
        cfg = get_default_config()
        assert cfg.shell.enabled is False

    def test_shell_allowed_commands_empty(self):
        cfg = get_default_config()
        assert cfg.shell.allowed_commands == []

    def test_plugins_disabled_by_default(self):
        cfg = get_default_config()
        assert cfg.plugins.enabled is False

    def test_plugins_allowed_list_empty(self):
        cfg = get_default_config()
        assert cfg.plugins.allowed_plugins == []

    def test_filesystem_paths_empty(self):
        cfg = get_default_config()
        assert cfg.filesystem.allowed_read_paths == []
        assert cfg.filesystem.allowed_write_paths == []

    def test_providers_empty(self):
        cfg = get_default_config()
        assert cfg.providers == {}

    def test_workspace_path_is_string(self):
        cfg = get_default_config()
        assert isinstance(cfg.workspace_path, str)
        assert len(cfg.workspace_path) > 0

    def test_audit_log_path_is_string(self):
        cfg = get_default_config()
        assert isinstance(cfg.audit_log_path, str)
        assert "audit" in cfg.audit_log_path.lower()


# ---------------------------------------------------------------------------
# load_config — minimal file
# ---------------------------------------------------------------------------


class TestLoadConfigMinimal:
    def test_load_minimal_yaml(self, tmp_path: Path):
        """A YAML file with only top-level keys should load without error."""
        path = _write_yaml(
            tmp_path,
            """
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )
        cfg = load_config(path)
        assert isinstance(cfg, MissyConfig)

    def test_minimal_yaml_workspace_path(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )
        cfg = load_config(path)
        assert cfg.workspace_path == "/tmp/workspace"

    def test_minimal_yaml_audit_log_path(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )
        cfg = load_config(path)
        assert cfg.audit_log_path == "/tmp/audit.log"

    def test_minimal_yaml_defaults_to_secure_posture(self, tmp_path: Path):
        """Missing policy sections default to a secure-by-default posture."""
        path = _write_yaml(tmp_path, "workspace_path: /tmp\n")
        cfg = load_config(path)
        assert cfg.network.default_deny is True
        assert cfg.shell.enabled is False
        assert cfg.plugins.enabled is False


# ---------------------------------------------------------------------------
# load_config — network fields
# ---------------------------------------------------------------------------


class TestLoadConfigNetwork:
    def test_default_deny_false(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            network:
              default_deny: false
            """,
        )
        cfg = load_config(path)
        assert cfg.network.default_deny is False

    def test_allowed_domains_loaded(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            network:
              allowed_domains:
                - "api.anthropic.com"
                - "api.openai.com"
            """,
        )
        cfg = load_config(path)
        assert "api.anthropic.com" in cfg.network.allowed_domains
        assert "api.openai.com" in cfg.network.allowed_domains

    def test_allowed_cidrs_loaded(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            network:
              allowed_cidrs:
                - "10.0.0.0/8"
                - "192.168.1.0/24"
            """,
        )
        cfg = load_config(path)
        assert "10.0.0.0/8" in cfg.network.allowed_cidrs
        assert "192.168.1.0/24" in cfg.network.allowed_cidrs

    def test_allowed_hosts_loaded(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            network:
              allowed_hosts:
                - "localhost:8080"
                - "intranet.local:443"
            """,
        )
        cfg = load_config(path)
        assert "localhost:8080" in cfg.network.allowed_hosts
        assert "intranet.local:443" in cfg.network.allowed_hosts

    def test_all_network_fields_together(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            network:
              default_deny: true
              allowed_cidrs:
                - "172.16.0.0/12"
              allowed_domains:
                - "example.com"
              allowed_hosts:
                - "proxy.corp:3128"
            """,
        )
        cfg = load_config(path)
        assert cfg.network.default_deny is True
        assert cfg.network.allowed_cidrs == ["172.16.0.0/12"]
        assert cfg.network.allowed_domains == ["example.com"]
        assert cfg.network.allowed_hosts == ["proxy.corp:3128"]


# ---------------------------------------------------------------------------
# load_config — filesystem fields
# ---------------------------------------------------------------------------


class TestLoadConfigFilesystem:
    def test_allowed_read_paths_loaded(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            filesystem:
              allowed_read_paths:
                - "/home/user/workspace"
                - "/tmp"
            """,
        )
        cfg = load_config(path)
        assert "/home/user/workspace" in cfg.filesystem.allowed_read_paths
        assert "/tmp" in cfg.filesystem.allowed_read_paths

    def test_allowed_write_paths_loaded(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            filesystem:
              allowed_write_paths:
                - "/home/user/output"
            """,
        )
        cfg = load_config(path)
        assert "/home/user/output" in cfg.filesystem.allowed_write_paths

    def test_filesystem_defaults_to_empty_lists(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "workspace_path: /tmp\n")
        cfg = load_config(path)
        assert cfg.filesystem.allowed_read_paths == []
        assert cfg.filesystem.allowed_write_paths == []

    def test_both_filesystem_fields_set(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            filesystem:
              allowed_read_paths:
                - "/data/in"
              allowed_write_paths:
                - "/data/out"
            """,
        )
        cfg = load_config(path)
        assert cfg.filesystem.allowed_read_paths == ["/data/in"]
        assert cfg.filesystem.allowed_write_paths == ["/data/out"]


# ---------------------------------------------------------------------------
# load_config — providers
# ---------------------------------------------------------------------------


class TestLoadConfigProviders:
    def test_single_provider_loaded(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
                timeout: 30
            """,
        )
        cfg = load_config(path)
        assert "anthropic" in cfg.providers

    def test_provider_model_field(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["anthropic"].model == "claude-3-5-sonnet-20241022"

    def test_provider_name_field(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["anthropic"].name == "anthropic"

    def test_provider_timeout_field(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai:
                name: openai
                model: "gpt-4o"
                timeout: 60
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["openai"].timeout == 60

    def test_provider_default_timeout(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["anthropic"].timeout == 30

    def test_provider_base_url_optional(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              ollama:
                name: ollama
                model: "llama3"
                base_url: "http://localhost:11434"
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["ollama"].base_url == "http://localhost:11434"

    def test_provider_base_url_defaults_to_none(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["anthropic"].base_url is None

    def test_multiple_providers(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
              openai:
                name: openai
                model: "gpt-4o"
            """,
        )
        cfg = load_config(path)
        assert "anthropic" in cfg.providers
        assert "openai" in cfg.providers

    def test_provider_missing_model_raises_configuration_error(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              broken:
                name: broken
            """,
        )
        with pytest.raises(ConfigurationError, match="model"):
            load_config(path)

    def test_provider_not_a_mapping_raises_configuration_error(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              bad_provider: "not-a-mapping"
            """,
        )
        with pytest.raises(ConfigurationError):
            load_config(path)

    def test_no_providers_section_gives_empty_dict(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "workspace_path: /tmp\n")
        cfg = load_config(path)
        assert cfg.providers == {}


# ---------------------------------------------------------------------------
# load_config — error cases
# ---------------------------------------------------------------------------


class TestLoadConfigErrors:
    def test_missing_file_raises_configuration_error(self, tmp_path: Path):
        missing = str(tmp_path / "nonexistent.yaml")
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(missing)

    def test_bad_yaml_raises_configuration_error(self, tmp_path: Path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("key: [\n  unclosed bracket\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="[Ii]nvalid YAML"):
            load_config(str(bad_file))

    def test_yaml_not_a_mapping_raises_configuration_error(self, tmp_path: Path):
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config(str(list_file))

    def test_error_message_includes_file_path(self, tmp_path: Path):
        missing = str(tmp_path / "missing.yaml")
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(missing)
        assert "missing.yaml" in str(exc_info.value)

    def test_path_that_is_a_directory_raises_configuration_error(self, tmp_path: Path):
        with pytest.raises(ConfigurationError):
            load_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_network_policy_defaults(self):
        policy = NetworkPolicy()
        assert policy.default_deny is True
        assert policy.allowed_cidrs == []
        assert policy.allowed_domains == []
        assert policy.allowed_hosts == []

    def test_network_policy_custom_values(self):
        policy = NetworkPolicy(
            default_deny=False,
            allowed_cidrs=["10.0.0.0/8"],
            allowed_domains=["example.com"],
            allowed_hosts=["proxy:3128"],
        )
        assert policy.default_deny is False
        assert policy.allowed_cidrs == ["10.0.0.0/8"]
        assert policy.allowed_domains == ["example.com"]
        assert policy.allowed_hosts == ["proxy:3128"]

    def test_filesystem_policy_defaults(self):
        policy = FilesystemPolicy()
        assert policy.allowed_write_paths == []
        assert policy.allowed_read_paths == []

    def test_filesystem_policy_custom_values(self):
        policy = FilesystemPolicy(
            allowed_read_paths=["/data/in"],
            allowed_write_paths=["/data/out"],
        )
        assert policy.allowed_read_paths == ["/data/in"]
        assert policy.allowed_write_paths == ["/data/out"]

    def test_shell_policy_defaults(self):
        policy = ShellPolicy()
        assert policy.enabled is False
        assert policy.allowed_commands == []

    def test_shell_policy_custom_values(self):
        policy = ShellPolicy(enabled=True, allowed_commands=["ls", "cat"])
        assert policy.enabled is True
        assert policy.allowed_commands == ["ls", "cat"]

    def test_plugin_policy_defaults(self):
        policy = PluginPolicy()
        assert policy.enabled is False
        assert policy.allowed_plugins == []

    def test_plugin_policy_custom_values(self):
        policy = PluginPolicy(enabled=True, allowed_plugins=["my-plugin"])
        assert policy.enabled is True
        assert policy.allowed_plugins == ["my-plugin"]

    def test_provider_config_required_fields(self):
        provider = ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022")
        assert provider.name == "anthropic"
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_provider_config_optional_defaults(self):
        provider = ProviderConfig(name="openai", model="gpt-4o")
        assert provider.api_key is None
        assert provider.base_url is None
        assert provider.timeout == 30

    def test_provider_config_all_fields(self):
        provider = ProviderConfig(
            name="ollama",
            model="llama3",
            api_key="sk-secret",
            base_url="http://localhost:11434",
            timeout=120,
        )
        assert provider.api_key == "sk-secret"
        assert provider.base_url == "http://localhost:11434"
        assert provider.timeout == 120

    def test_network_policy_list_independence(self):
        """Mutating one instance's lists must not affect another instance."""
        p1 = NetworkPolicy()
        p2 = NetworkPolicy()
        p1.allowed_domains.append("example.com")
        assert p2.allowed_domains == []

    def test_shell_policy_list_independence(self):
        p1 = ShellPolicy()
        p2 = ShellPolicy()
        p1.allowed_commands.append("ls")
        assert p2.allowed_commands == []
