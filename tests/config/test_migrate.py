"""Tests for config migration (old format → presets + version stamp)."""

from __future__ import annotations

import textwrap

import yaml

from missy.config.migrate import (
    CURRENT_CONFIG_VERSION,
    detect_presets,
    migrate_config,
    needs_migration,
)


class TestNeedsMigration:
    """Tests for needs_migration."""

    def test_no_version_needs_migration(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")
        assert needs_migration(str(cfg)) is True

    def test_version_1_needs_migration(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: 1\nnetwork:\n  default_deny: true\n")
        assert needs_migration(str(cfg)) is True

    def test_current_version_no_migration(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"config_version: {CURRENT_CONFIG_VERSION}\n")
        assert needs_migration(str(cfg)) is False

    def test_missing_file_no_migration(self, tmp_path):
        assert needs_migration(str(tmp_path / "nonexistent.yaml")) is False

    def test_empty_file_no_migration(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        # Empty YAML parses to None, not a dict
        assert needs_migration(str(cfg)) is False


class TestDetectPresets:
    """Tests for detect_presets."""

    def test_detects_anthropic_preset(self):
        network = {"allowed_hosts": ["api.anthropic.com"], "allowed_domains": ["anthropic.com"]}
        detected, remaining_hosts, remaining_domains, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "api.anthropic.com" not in remaining_hosts
        assert "anthropic.com" not in remaining_domains

    def test_detects_multiple_presets(self):
        network = {
            "allowed_hosts": [
                "api.anthropic.com",
                "discord.com",
                "gateway.discord.gg",
            ],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "discord" in detected
        assert remaining_hosts == []

    def test_partial_match_not_detected(self):
        # Discord requires both discord.com AND gateway.discord.gg
        network = {"allowed_hosts": ["discord.com"], "allowed_domains": []}
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "discord" not in detected
        assert "discord.com" in remaining_hosts

    def test_remaining_hosts_preserved(self):
        network = {
            "allowed_hosts": ["api.anthropic.com", "custom.example.com"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "custom.example.com" in remaining_hosts
        assert "api.anthropic.com" not in remaining_hosts

    def test_empty_network(self):
        detected, remaining_hosts, remaining_domains, remaining_cidrs = detect_presets({})
        assert detected == []
        assert remaining_hosts == []
        assert remaining_domains == []
        assert remaining_cidrs == []

    def test_existing_presets_merged(self):
        network = {
            "presets": ["anthropic"],
            "allowed_hosts": ["discord.com", "gateway.discord.gg"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "discord" in detected
        assert remaining_hosts == []

    def test_hosts_detect_without_domains(self):
        """Preset detected by hosts alone, even if domains weren't listed."""
        network = {"allowed_hosts": ["api.anthropic.com"], "allowed_domains": []}
        detected, _, _, _ = detect_presets(network)
        assert "anthropic" in detected


class TestMigrateConfig:
    """Tests for migrate_config."""

    def test_full_migration(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
                - "api.openai.com"
                - "auth.openai.com"
                - "chatgpt.com"
              allowed_domains: []
              allowed_cidrs: []

            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"
        """)
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        assert "anthropic" in result["presets_detected"]
        assert "openai" in result["presets_detected"]
        assert result["backup_path"] is not None

        # Verify the written file
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION
        assert "anthropic" in data["network"]["presets"]
        assert "openai" in data["network"]["presets"]
        # Manual hosts should be removed since they're covered by presets
        assert data["network"]["allowed_hosts"] == []

    def test_idempotent(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"

            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"
        """)
        )

        result1 = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))
        assert result1["migrated"] is True

        result2 = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))
        assert result2["migrated"] is False

    def test_preserves_other_fields(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
              allowed_domains: []
              allowed_cidrs: []

            filesystem:
              allowed_write_paths:
                - "~/workspace"

            shell:
              enabled: true
              allowed_commands:
                - "ls"
                - "cat"

            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"
                timeout: 30

            workspace_path: "~/workspace"
        """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        assert data["shell"]["enabled"] is True
        assert data["shell"]["allowed_commands"] == ["ls", "cat"]
        assert data["filesystem"]["allowed_write_paths"] == ["~/workspace"]
        assert data["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"
        assert data["workspace_path"] == "~/workspace"

    def test_keeps_extra_hosts(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
                - "my-custom-server.local"
              allowed_domains: []
              allowed_cidrs: []
        """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        assert "anthropic" in data["network"]["presets"]
        assert "my-custom-server.local" in data["network"]["allowed_hosts"]
        assert "api.anthropic.com" not in data["network"]["allowed_hosts"]

    def test_creates_backup(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")
        backup_dir = tmp_path / "backups"

        result = migrate_config(str(cfg), backup_dir=str(backup_dir))

        assert result["backup_path"] is not None
        assert backup_dir.exists()
        backups = list(backup_dir.iterdir())
        assert len(backups) >= 1

    def test_missing_network_section(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"
        """)
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION
        assert data["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"

    def test_empty_config(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("{}\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION

    def test_category_overrides_not_touched(self, tmp_path):
        """provider_allowed_hosts etc. should not be migrated."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
              provider_allowed_hosts:
                - "api.anthropic.com"
              tool_allowed_hosts:
                - "some-tool.example.com"
              discord_allowed_hosts:
                - "discord.com"
              allowed_domains: []
              allowed_cidrs: []
        """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        # Category overrides preserved verbatim
        assert "api.anthropic.com" in data["network"]["provider_allowed_hosts"]
        assert "some-tool.example.com" in data["network"]["tool_allowed_hosts"]
        assert "discord.com" in data["network"]["discord_allowed_hosts"]
        # But allowed_hosts was migrated
        assert "api.anthropic.com" not in data["network"]["allowed_hosts"]
        assert "anthropic" in data["network"]["presets"]

    def test_atomic_write_valid_yaml(self, tmp_path):
        """After migration, the file must be valid YAML."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
                - "discord.com"
                - "gateway.discord.gg"
        """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        # Must parse without error
        data = yaml.safe_load(cfg.read_text())
        assert isinstance(data, dict)
        assert data["config_version"] == CURRENT_CONFIG_VERSION


class TestLoadConfigWithVersion:
    """Tests that config_version is available after load_config."""

    def test_config_version_parsed(self, tmp_path):
        from missy.config.settings import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            config_version: 2

            network:
              default_deny: true
              presets:
                - anthropic

            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"

            workspace_path: "~/workspace"
            audit_log_path: "~/.missy/audit.jsonl"
        """)
        )

        cfg = load_config(str(cfg_file))
        assert cfg.config_version == 2

    def test_config_version_defaults_to_zero(self, tmp_path):
        from missy.config.settings import load_config

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true

            providers:
              anthropic:
                name: anthropic
                model: "claude-sonnet-4-6"

            workspace_path: "~/workspace"
            audit_log_path: "~/.missy/audit.jsonl"
        """)
        )

        cfg = load_config(str(cfg_file))
        assert cfg.config_version == 0
