"""Tests for missy.config.settings — configuration loading and policy dataclasses."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from missy.config.settings import (
    AgentPolicyConfig,
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ProviderConfig,
    ShellPolicy,
    ToolIntelligenceConfig,
    ToolPolicyConfig,
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

    def test_shell_unrestricted_disabled_by_default(self):
        cfg = get_default_config()
        assert cfg.shell.unrestricted is False

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


class TestLoadConfigToolPolicy:
    def test_default_tool_policy_is_full(self):
        cfg = get_default_config()
        assert isinstance(cfg.tools, ToolPolicyConfig)
        assert cfg.tools.profile == "full"
        assert cfg.agents == {}

    def test_default_disabled_tools_is_empty(self):
        cfg = get_default_config()
        assert cfg.tools.disabled_tools == []

    def test_loads_disabled_tools(self, tmp_path: Path):
        """Regression: ToolRegistry.disable()/is_enabled() (an
        execute()-level kill switch stronger than tools.deny, which only
        narrows what's offered to the model per turn) was fully built and
        tested but had zero callers anywhere in the codebase -- an
        operator had no way to actually disable a tool via any
        first-party surface. tools.disabled_tools makes this reachable.
        """
        path = _write_yaml(
            tmp_path,
            """
            tools:
              disabled_tools: ["shell_exec", "file_write"]
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )
        cfg = load_config(str(path))
        assert cfg.tools.disabled_tools == ["shell_exec", "file_write"]

    def test_loads_global_and_agent_tool_policy(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            tools:
              profile: coding
              allow: ["group:project", "-shell_exec"]
              deny: ["browser_*"]
              alsoAllow: ["calculator"]
              groups:
                project: ["file_read", "file_write"]
              byProvider:
                anthropic:
                  deny: ["vision_*"]
                  byModel:
                    claude-haiku-*:
                      allow: ["calculator"]
            agents:
              analyst:
                tools:
                  profile: minimal
                  deny: ["file_write"]
                subagents:
                  tools:
                    deny: ["sessions_*"]
            sandbox:
              enabled: true
              tools:
                allow: ["calculator"]
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        cfg = load_config(path)

        assert cfg.tools.profile == "coding"
        assert cfg.tools.allow == ["group:project", "-shell_exec"]
        assert cfg.tools.also_allow == ["calculator"]
        assert cfg.tools.groups["project"] == ["file_read", "file_write"]
        assert cfg.tools.by_provider["anthropic"]["by_model"]["claude-haiku-*"]["allow"] == [
            "calculator"
        ]
        assert isinstance(cfg.agents["analyst"], AgentPolicyConfig)
        assert cfg.agents["analyst"].tools.profile == "minimal"
        assert cfg.agents["analyst"].subagent_tools.deny == ["sessions_*"]
        assert cfg.sandbox is not None
        assert cfg.sandbox.tools == {"allow": ["calculator"]}

    def test_invalid_tool_profile_raises(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            tools:
              profile: unsafe
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        with pytest.raises(ConfigurationError, match="tools.profile"):
            load_config(path)


class TestLoadConfigToolIntelligence:
    def test_default_is_disabled(self):
        cfg = get_default_config()
        assert isinstance(cfg.tool_intelligence, ToolIntelligenceConfig)
        assert cfg.tool_intelligence.candidate_generation_enabled is False
        assert cfg.tool_intelligence.provider_gating_enabled is False
        assert cfg.tool_intelligence.candidate_runtime_loading_enabled is False

    def test_loads_candidate_generation_provider_gating_and_runtime(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            tool_intelligence:
              candidate_generation:
                enabled: true
                min_pattern_count: 7
                allow_shell: true
                check_every_n_requests: 2
              provider_gating:
                enabled: true
                min_samples: 5
                min_composite: 0.6
              candidate_runtime:
                enabled: true
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        cfg = load_config(path)

        assert cfg.tool_intelligence.candidate_generation_enabled is True
        assert cfg.tool_intelligence.min_pattern_count == 7
        assert cfg.tool_intelligence.allow_shell is True
        assert cfg.tool_intelligence.check_every_n_requests == 2
        assert cfg.tool_intelligence.provider_gating_enabled is True
        assert cfg.tool_intelligence.provider_gating_min_samples == 5
        assert cfg.tool_intelligence.provider_gating_min_composite == 0.6
        assert cfg.tool_intelligence.candidate_runtime_loading_enabled is True

    def test_missing_section_uses_defaults(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        cfg = load_config(path)

        assert cfg.tool_intelligence == ToolIntelligenceConfig()

    def test_non_mapping_section_raises(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            tool_intelligence: "yes please"
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        with pytest.raises(ConfigurationError, match="tool_intelligence"):
            load_config(path)

    def test_non_mapping_candidate_generation_raises(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            tool_intelligence:
              candidate_generation: "yes"
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        with pytest.raises(ConfigurationError, match="candidate_generation"):
            load_config(path)

    def test_non_mapping_candidate_runtime_raises(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            tool_intelligence:
              candidate_runtime: "yes"
            workspace_path: "/tmp/workspace"
            audit_log_path: "/tmp/audit.log"
            """,
        )

        with pytest.raises(ConfigurationError, match="candidate_runtime"):
            load_config(path)


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


class TestLoadConfigShellUnrestricted:
    def test_unrestricted_defaults_to_false(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: true
              allowed_commands: ["ls"]
            """,
        )
        cfg = load_config(path)
        assert cfg.shell.unrestricted is False

    def test_unrestricted_true_loaded_with_empty_allowed_commands(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: true
              unrestricted: true
            """,
        )
        cfg = load_config(path)
        assert cfg.shell.enabled is True
        assert cfg.shell.allowed_commands == []
        assert cfg.shell.unrestricted is True

    def test_quoted_true_unrestricted_is_unrestricted(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: true
              unrestricted: "true"
            """,
        )
        cfg = load_config(path)
        assert cfg.shell.unrestricted is True

    def test_quoted_false_unrestricted_stays_restricted(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: true
              allowed_commands: ["ls"]
              unrestricted: "false"
            """,
        )
        cfg = load_config(path)
        assert cfg.shell.unrestricted is False


# ---------------------------------------------------------------------------
# Config-hygiene gap: unrecognized keys in a policy section are silently
# dropped, with no signal to the operator that a typo or a stale/renamed
# field means the config isn't doing what they think it's doing.
# ---------------------------------------------------------------------------


class TestUnknownConfigKeyWarnings:
    def test_shell_typo_key_warns(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: true
              allowed_commands: ["ls"]
              allowd_commands: ["typo"]
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            cfg = load_config(path)

        assert cfg.shell.enabled is True
        assert cfg.shell.allowed_commands == ["ls"]
        assert any("shell" in r.message and "allowd_commands" in r.message for r in caplog.records)

    def test_network_unknown_key_warns(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            network:
              default_deny: true
              allowed_domain: "api.example.com"
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)

        # "allowed_domain" (singular, a plausible typo for the real
        # "allowed_domains") must be flagged, not silently accepted as
        # if it configured anything.
        assert any("network" in r.message and "allowed_domain" in r.message for r in caplog.records)

    def test_filesystem_unknown_key_warns(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            filesystem:
              allowed_read_paths: ["/tmp"]
              readonly_paths: ["/etc"]
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)

        assert any(
            "filesystem" in r.message and "readonly_paths" in r.message for r in caplog.records
        )

    def test_plugins_unknown_key_warns(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            plugins:
              enabled: false
              whitelist: ["foo"]
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)

        assert any("plugins" in r.message and "whitelist" in r.message for r in caplog.records)

    def test_no_warning_for_recognized_keys_only(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            network:
              default_deny: true
              allowed_domains: ["api.example.com"]
            shell:
              enabled: true
              allowed_commands: ["ls"]
            filesystem:
              allowed_read_paths: ["/tmp"]
              allowed_write_paths: ["/tmp"]
            plugins:
              enabled: false
              allowed_plugins: []
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)

        assert not any("unrecognized key" in r.message for r in caplog.records)

    def test_unknown_keys_are_ignored_not_fatal(self, tmp_path: Path):
        """The warning is visibility-only -- an unrecognized key must
        never raise or otherwise fail config loading, since that would
        be a breaking change for any operator with genuinely-extra keys
        (e.g. comments-as-keys, keys meant for a newer/older version)."""
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: true
              allowed_commands: ["ls"]
              unrestricted: true
            """,
        )
        cfg = load_config(path)  # must not raise
        assert isinstance(cfg, MissyConfig)


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

    def test_provider_circuit_breaker_tunables_default(self, tmp_path: Path):
        """SR-4.8 residual: per-provider CircuitBreaker cooldown config.
        Defaults must match CircuitBreaker's own hardcoded defaults
        (threshold=5, base_timeout=60.0) exactly, so a config that
        doesn't set these fields at all behaves identically to before
        this option existed."""
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
        assert cfg.providers["anthropic"].circuit_breaker_threshold == 5
        assert cfg.providers["anthropic"].circuit_breaker_cooldown_seconds == 60.0

    def test_provider_circuit_breaker_tunables_explicit(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              flaky-provider:
                name: flaky-provider
                model: "some-model"
                circuit_breaker_threshold: 2
                circuit_breaker_cooldown_seconds: 15.0
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["flaky-provider"].circuit_breaker_threshold == 2
        assert cfg.providers["flaky-provider"].circuit_breaker_cooldown_seconds == 15.0

    def test_provider_unknown_key_warns(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              anthropic:
                name: anthropic
                model: "claude-3-5-sonnet-20241022"
                circuit_breaker_threshhold: 2
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)

        assert any(
            "providers.anthropic" in r.message and "circuit_breaker_threshhold" in r.message
            for r in caplog.records
        )

    def test_provider_key_rotation_strategy_default(self, tmp_path: Path):
        """Providers without the field must default to 'failover' -- the
        original single-sticky-key-with-reactive-rotation behavior."""
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai:
                name: openai
                model: "gpt-5.5"
                api_keys: ["key-a", "key-b"]
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["openai"].key_rotation_strategy == "failover"

    def test_provider_key_rotation_strategy_round_robin(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai:
                name: openai
                model: "gpt-5.5"
                api_keys: ["key-a", "key-b"]
                key_rotation_strategy: round_robin
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["openai"].key_rotation_strategy == "round_robin"

    def test_provider_key_rotation_strategy_invalid_value_raises(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai:
                name: openai
                model: "gpt-5.5"
                api_keys: ["key-a", "key-b"]
                key_rotation_strategy: yolo
            """,
        )
        with pytest.raises(ConfigurationError, match="key_rotation_strategy"):
            load_config(path)

    def test_provider_oauth_accounts_default_empty(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai-codex:
                name: openai-codex
                model: "gpt-5.2"
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["openai-codex"].oauth_accounts == []

    def test_provider_oauth_accounts_parsed(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai-codex:
                name: openai-codex
                model: "gpt-5.2"
                oauth_accounts: ["work", "personal"]
                key_rotation_strategy: round_robin
            """,
        )
        cfg = load_config(path)
        assert cfg.providers["openai-codex"].oauth_accounts == ["work", "personal"]
        assert cfg.providers["openai-codex"].key_rotation_strategy == "round_robin"

    def test_provider_oauth_accounts_unknown_key_not_warned(self, tmp_path: Path, caplog):
        """oauth_accounts is a real ProviderConfig field, not a typo — must
        not trip the unknown-provider-key warning."""
        path = _write_yaml(
            tmp_path,
            """
            providers:
              openai-codex:
                name: openai-codex
                model: "gpt-5.2"
                oauth_accounts: ["work", "personal"]
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)

        assert not any(
            "oauth_accounts" in r.message and "providers.openai-codex" in r.message
            for r in caplog.records
        )

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
# load_config — vault:// reference resolution against a custom vault_dir
# ---------------------------------------------------------------------------


class TestLoadConfigVaultResolutionCustomDir:
    """Regression tests: a vault:// reference must resolve against the
    SAME vault.vault_dir the config file itself declares, not always
    Vault()'s hardcoded ~/.missy/secrets default.

    _resolve_vault_ref() and DiscordAccountConfig.resolve_token() both
    previously called the bare Vault() constructor, silently ignoring
    any custom vault.vault_dir -- a provider api_key or Discord bot
    token configured as a vault:// reference alongside a non-default
    vault_dir resolved to the literal, unresolved reference string
    (an unusable "secret") instead of raising a clear error, with no
    diagnostic beyond a logging.debug() call.
    """

    def test_provider_api_key_resolves_against_custom_vault_dir(self, tmp_path: Path):
        pytest.importorskip("cryptography")
        from missy.security.vault import Vault

        vault_dir = tmp_path / "custom_vault_location"
        Vault(str(vault_dir)).set("OPENAI_API_KEY", "sk-REAL-SECRET-VALUE")

        path = _write_yaml(
            tmp_path,
            f"""
            vault:
              vault_dir: "{vault_dir}"
            providers:
              openai:
                model: "gpt-4"
                api_key: "vault://OPENAI_API_KEY"
            """,
        )
        cfg = load_config(path)

        assert cfg.providers["openai"].api_key == "sk-REAL-SECRET-VALUE"

    def test_discord_token_resolves_against_custom_vault_dir(self, tmp_path: Path):
        pytest.importorskip("cryptography")
        from missy.security.vault import Vault

        vault_dir = tmp_path / "custom_vault_location"
        Vault(str(vault_dir)).set("DISCORD_BOT_TOKEN", "discord-real-secret-token")

        path = _write_yaml(
            tmp_path,
            f"""
            vault:
              vault_dir: "{vault_dir}"
            discord:
              enabled: true
              accounts:
                - token: "vault://DISCORD_BOT_TOKEN"
                  application_id: "123"
            """,
        )
        cfg = load_config(path)

        assert cfg.discord is not None
        assert cfg.discord.accounts[0].resolve_token() == "discord-real-secret-token"


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
        assert policy.unrestricted is False

    def test_shell_policy_custom_values(self):
        policy = ShellPolicy(enabled=True, allowed_commands=["ls", "cat"])
        assert policy.enabled is True
        assert policy.allowed_commands == ["ls", "cat"]

    def test_shell_policy_unrestricted(self):
        policy = ShellPolicy(enabled=True, unrestricted=True)
        assert policy.unrestricted is True

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


# ---------------------------------------------------------------------------
# _coerce_bool -- quoted-string YAML boolean values must not silently invert
# ---------------------------------------------------------------------------


class TestCoerceBool:
    """Regression: bool(data.get(key, default)) treats ANY non-empty string
    as truthy in Python, so a quoted YAML boolean like enabled: "false"
    parses as the string "false" and bool("false") is True -- silently
    inverting a security-relevant flag the operator explicitly tried to
    disable, with no error, warning, or log line anywhere. _coerce_bool()
    must recognize common human-readable string forms and fail loud on
    anything genuinely ambiguous, rather than falling back to Python's
    truthiness rules.
    """

    def test_none_returns_default(self):
        from missy.config.settings import _coerce_bool

        assert _coerce_bool(None, True) is True
        assert _coerce_bool(None, False) is False

    def test_real_bool_passed_through(self):
        from missy.config.settings import _coerce_bool

        assert _coerce_bool(True, False) is True
        assert _coerce_bool(False, True) is False

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "yes", "on", "1"])
    def test_truthy_strings(self, value: str):
        from missy.config.settings import _coerce_bool

        assert _coerce_bool(value, False) is True

    @pytest.mark.parametrize("value", ["false", "False", "FALSE", "no", "off", "0"])
    def test_falsy_strings(self, value: str):
        from missy.config.settings import _coerce_bool

        assert _coerce_bool(value, True) is False

    def test_ambiguous_string_raises(self):
        from missy.config.settings import _coerce_bool

        with pytest.raises(ConfigurationError, match="Cannot interpret"):
            _coerce_bool("maybe", False)

    def test_empty_string_raises(self):
        from missy.config.settings import _coerce_bool

        with pytest.raises(ConfigurationError, match="Cannot interpret"):
            _coerce_bool("", False)


class TestQuotedBooleanConfigValues:
    """End-to-end regression: a quoted-string boolean in the actual YAML
    config file must not silently invert the resulting policy flag.
    """

    def test_quoted_false_shell_enabled_stays_disabled(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: "false"
            """,
        )
        cfg = load_config(path)
        assert cfg.shell.enabled is False

    def test_quoted_false_plugins_enabled_stays_disabled(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            plugins:
              enabled: "false"
            """,
        )
        cfg = load_config(path)
        assert cfg.plugins.enabled is False

    def test_quoted_true_shell_enabled_is_enabled(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: "true"
            """,
        )
        cfg = load_config(path)
        assert cfg.shell.enabled is True

    def test_ambiguous_string_value_raises_configuration_error(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            shell:
              enabled: "maybe"
            """,
        )
        with pytest.raises(ConfigurationError, match="Cannot interpret"):
            load_config(path)


# ---------------------------------------------------------------------------
# Desktop / OBS / VTube Studio config sections
# ---------------------------------------------------------------------------


class TestObsConfigParsing:
    def test_defaults_when_section_absent(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "providers: {}\n")
        cfg = load_config(path)
        assert cfg.obs.enabled is False
        assert cfg.obs.host == "127.0.0.1"
        assert cfg.obs.port == 4455
        assert cfg.obs.password is None
        assert cfg.obs.scene_allowlist == []

    def test_parses_explicit_values(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers: {}
            obs:
              enabled: true
              host: 192.168.1.50
              port: 4456
              scene_allowlist: ["Main", "BRB"]
            """,
        )
        cfg = load_config(path)
        assert cfg.obs.enabled is True
        assert cfg.obs.host == "192.168.1.50"
        assert cfg.obs.port == 4456
        assert cfg.obs.scene_allowlist == ["Main", "BRB"]

    def test_password_resolves_vault_reference(self, tmp_path: Path):
        vault_dir = tmp_path / "secrets"
        vault_dir.mkdir()
        from missy.security.vault import Vault

        Vault(str(vault_dir)).set("obs_pw", "hunter2")

        path = _write_yaml(
            tmp_path,
            f"""
            providers: {{}}
            vault:
              vault_dir: {vault_dir}
            obs:
              enabled: true
              password: vault://obs_pw
            """,
        )
        cfg = load_config(path)
        assert cfg.obs.password == "hunter2"

    def test_unknown_key_warns(self, tmp_path: Path, caplog):
        path = _write_yaml(
            tmp_path,
            """
            providers: {}
            obs:
              enabled: true
              typo_field: oops
            """,
        )
        with caplog.at_level("WARNING", logger="missy.config.settings"):
            load_config(path)
        assert any("typo_field" in r.message and "obs" in r.message for r in caplog.records)


class TestVtubeConfigParsing:
    def test_defaults_when_section_absent(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "providers: {}\n")
        cfg = load_config(path)
        assert cfg.vtube.enabled is False
        assert cfg.vtube.host == "127.0.0.1"
        assert cfg.vtube.port == 8001
        assert cfg.vtube.auth_token is None
        assert cfg.vtube.plugin_name == "Missy"
        assert cfg.vtube.plugin_developer == "MissyLabs"

    def test_parses_explicit_values(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers: {}
            vtube:
              enabled: true
              host: 127.0.0.1
              port: 8002
              plugin_name: MissyStream
              plugin_developer: TestDev
            """,
        )
        cfg = load_config(path)
        assert cfg.vtube.enabled is True
        assert cfg.vtube.port == 8002
        assert cfg.vtube.plugin_name == "MissyStream"
        assert cfg.vtube.plugin_developer == "TestDev"

    def test_auth_token_resolves_vault_reference(self, tmp_path: Path):
        vault_dir = tmp_path / "secrets"
        vault_dir.mkdir()
        from missy.security.vault import Vault

        Vault(str(vault_dir)).set("vtube_studio_token", "tok-abc")

        path = _write_yaml(
            tmp_path,
            f"""
            providers: {{}}
            vault:
              vault_dir: {vault_dir}
            vtube:
              enabled: true
              auth_token: vault://vtube_studio_token
            """,
        )
        cfg = load_config(path)
        assert cfg.vtube.auth_token == "tok-abc"


class TestDesktopConfigParsing:
    def test_defaults_when_section_absent(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "providers: {}\n")
        cfg = load_config(path)
        assert cfg.desktop.enabled is False
        assert cfg.desktop.app_allowlist == []
        assert cfg.desktop.unrestricted is False

    def test_parses_explicit_values(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers: {}
            desktop:
              enabled: true
              app_allowlist: ["firefox", "obs"]
              unrestricted: false
            """,
        )
        cfg = load_config(path)
        assert cfg.desktop.enabled is True
        assert cfg.desktop.app_allowlist == ["firefox", "obs"]
        assert cfg.desktop.unrestricted is False

    def test_unrestricted_true(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers: {}
            desktop:
              enabled: true
              unrestricted: true
            """,
        )
        cfg = load_config(path)
        assert cfg.desktop.unrestricted is True

    def test_ambiguous_enabled_string_raises(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """
            providers: {}
            desktop:
              enabled: "maybe"
            """,
        )
        with pytest.raises(ConfigurationError, match="Cannot interpret"):
            load_config(path)
