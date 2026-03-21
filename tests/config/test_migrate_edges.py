"""Comprehensive tests for missy/config/migrate.py.


Covers needs_migration, detect_presets, migrate_config, CURRENT_CONFIG_VERSION,
and edge cases not addressed by the primary test suites.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import yaml

from missy.config.migrate import (
    CURRENT_CONFIG_VERSION,
    _atomic_write_yaml,
    detect_presets,
    migrate_config,
    needs_migration,
)
from missy.policy.presets import PRESETS

# ---------------------------------------------------------------------------
# Sentinel: CURRENT_CONFIG_VERSION
# ---------------------------------------------------------------------------


class TestCurrentConfigVersion:
    """Verify the module-level version constant."""

    def test_current_config_version_is_two(self):
        assert CURRENT_CONFIG_VERSION == 2

    def test_current_config_version_is_integer(self):
        assert isinstance(CURRENT_CONFIG_VERSION, int)

    def test_current_config_version_positive(self):
        assert CURRENT_CONFIG_VERSION > 0


# ---------------------------------------------------------------------------
# needs_migration — happy paths
# ---------------------------------------------------------------------------


class TestNeedsMigrationHappyPaths:
    """Core positive / negative cases for needs_migration."""

    def test_nonexistent_file_returns_false(self, tmp_path):
        """A path that does not exist must return False without raising."""
        assert needs_migration(str(tmp_path / "no_such_file.yaml")) is False

    def test_directory_path_returns_false(self, tmp_path):
        """Passing a directory (not a file) must return False."""
        assert needs_migration(str(tmp_path)) is False

    def test_corrupt_yaml_returns_false(self, tmp_path):
        """Malformed YAML must not raise — must return False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("key: :\n  bad: [unclosed\n")
        assert needs_migration(str(cfg)) is False

    def test_yaml_list_not_dict_returns_false(self, tmp_path):
        """YAML that parses to a list (not dict) must return False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- first\n- second\n- third\n")
        assert needs_migration(str(cfg)) is False

    def test_no_config_version_key_returns_true(self, tmp_path):
        """Missing config_version defaults to 0, which triggers migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")
        assert needs_migration(str(cfg)) is True

    def test_config_version_1_returns_true(self, tmp_path):
        """Version 1 is below CURRENT_CONFIG_VERSION and must trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: 1\n")
        assert needs_migration(str(cfg)) is True

    def test_config_version_current_returns_false(self, tmp_path):
        """Config stamped with CURRENT_CONFIG_VERSION must not trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"config_version: {CURRENT_CONFIG_VERSION}\n")
        assert needs_migration(str(cfg)) is False

    def test_config_version_future_returns_false(self, tmp_path):
        """A version higher than CURRENT_CONFIG_VERSION must return False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"config_version: {CURRENT_CONFIG_VERSION + 5}\n")
        assert needs_migration(str(cfg)) is False

    def test_non_integer_version_string_returns_true(self, tmp_path):
        """A non-integer version string must return True (triggers migration)."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: 'alpha'\n")
        assert needs_migration(str(cfg)) is True

    def test_empty_file_returns_false(self, tmp_path):
        """An empty file parses to None (not a dict) — must return False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        assert needs_migration(str(cfg)) is False


# ---------------------------------------------------------------------------
# needs_migration — additional edge cases
# ---------------------------------------------------------------------------


class TestNeedsMigrationEdges:
    """Further edge cases not covered by the base suite."""

    def test_yaml_scalar_string_returns_false(self, tmp_path):
        """YAML that is a plain string at root (not a dict) returns False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("just a string\n")
        assert needs_migration(str(cfg)) is False

    def test_yaml_integer_root_returns_false(self, tmp_path):
        """YAML that is just an integer at root returns False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("42\n")
        assert needs_migration(str(cfg)) is False

    def test_config_version_null_returns_true(self, tmp_path):
        """config_version: null — int(None) raises TypeError → returns True."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: null\n")
        assert needs_migration(str(cfg)) is True

    def test_config_version_float_triggers_migration(self, tmp_path):
        """A float version like 1.5 converts cleanly to int(1) < 2 → True."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: 1.5\n")
        assert needs_migration(str(cfg)) is True

    def test_config_version_zero_returns_true(self, tmp_path):
        """Explicit config_version: 0 is below current → must trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: 0\n")
        assert needs_migration(str(cfg)) is True

    def test_whitespace_only_file_returns_false(self, tmp_path):
        """A file with only whitespace parses to None → returns False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("   \n\n\t\n")
        assert needs_migration(str(cfg)) is False

    def test_config_with_only_comments_returns_false(self, tmp_path):
        """YAML consisting only of comments parses to None → returns False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("# This is a comment\n# Another comment\n")
        assert needs_migration(str(cfg)) is False

    def test_tilde_path_nonexistent_returns_false(self):
        """A tilde-prefixed path that does not exist must return False, not crash."""
        result = needs_migration("~/definitely-does-not-exist-missy-test-XYZ.yaml")
        assert result is False

    def test_version_exactly_one_below_current_returns_true(self, tmp_path):
        """Boundary: version CURRENT-1 must trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"config_version: {CURRENT_CONFIG_VERSION - 1}\n")
        assert needs_migration(str(cfg)) is True

    def test_returns_bool_not_truthy(self, tmp_path):
        """Return value must be the bool True or False, not just truthy/falsy."""
        cfg_true = tmp_path / "needs.yaml"
        cfg_true.write_text("network:\n  default_deny: true\n")
        result_true = needs_migration(str(cfg_true))
        assert result_true is True

        cfg_false = tmp_path / "current.yaml"
        cfg_false.write_text(f"config_version: {CURRENT_CONFIG_VERSION}\n")
        result_false = needs_migration(str(cfg_false))
        assert result_false is False


# ---------------------------------------------------------------------------
# detect_presets — happy paths
# ---------------------------------------------------------------------------


class TestDetectPresetsHappyPaths:
    """Core detection scenarios for detect_presets."""

    def test_empty_network_data_returns_empty_tuples(self):
        detected, hosts, domains, cidrs = detect_presets({})
        assert detected == []
        assert hosts == []
        assert domains == []
        assert cidrs == []

    def test_detects_anthropic_by_host(self):
        """anthropic preset has one required host: api.anthropic.com."""
        network = {"allowed_hosts": ["api.anthropic.com"], "allowed_domains": []}
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "api.anthropic.com" not in remaining_hosts

    def test_detects_github_preset(self):
        """github preset requires both api.github.com and github.com."""
        network = {
            "allowed_hosts": ["api.github.com", "github.com"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "github" in detected
        assert "api.github.com" not in remaining_hosts
        assert "github.com" not in remaining_hosts

    def test_detects_multiple_presets_simultaneously(self):
        """Both anthropic and github can be detected from a single host list."""
        network = {
            "allowed_hosts": [
                "api.anthropic.com",
                "api.github.com",
                "github.com",
            ],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "github" in detected
        assert remaining_hosts == []

    def test_partial_match_not_detected(self):
        """github requires BOTH hosts; a single one must not trigger detection."""
        network = {
            "allowed_hosts": ["api.github.com"],  # missing github.com
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "github" not in detected
        assert "api.github.com" in remaining_hosts

    def test_existing_presets_in_config_preserved(self):
        """presets already listed in the config must appear in the result."""
        network = {
            "presets": ["anthropic"],
            "allowed_hosts": [],
            "allowed_domains": [],
        }
        detected, _, _, _ = detect_presets(network)
        assert "anthropic" in detected

    def test_existing_preset_hosts_removed_from_remaining(self):
        """Hosts belonging to an already-listed preset are stripped from remaining."""
        network = {
            "presets": ["anthropic"],
            "allowed_hosts": ["api.anthropic.com", "custom.host.local"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "api.anthropic.com" not in remaining_hosts
        assert "custom.host.local" in remaining_hosts

    def test_non_preset_hosts_preserved_in_remaining(self):
        """Hosts that do not belong to any preset must survive in remaining_hosts."""
        network = {
            "allowed_hosts": ["api.anthropic.com", "my-private-server.example.com"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "my-private-server.example.com" in remaining_hosts

    def test_preset_domains_removed_from_remaining(self):
        """Detected preset domains must not appear in remaining_domains."""
        network = {
            "allowed_hosts": ["api.anthropic.com"],
            "allowed_domains": ["anthropic.com", "extra-domain.example.com"],
        }
        detected, _, remaining_domains, _ = detect_presets(network)
        assert "anthropic" in detected
        assert "anthropic.com" not in remaining_domains
        assert "extra-domain.example.com" in remaining_domains

    def test_preset_with_cidr_removes_cidr(self):
        """ollama preset has cidrs; detecting it must strip its CIDR from remaining."""
        network = {
            "allowed_hosts": ["localhost:11434", "127.0.0.1:11434"],
            "allowed_domains": [],
            "allowed_cidrs": ["127.0.0.0/8", "10.0.0.0/8"],
        }
        detected, _, _, remaining_cidrs = detect_presets(network)
        assert "ollama" in detected
        assert "127.0.0.0/8" not in remaining_cidrs
        assert "10.0.0.0/8" in remaining_cidrs


# ---------------------------------------------------------------------------
# detect_presets — additional edge cases
# ---------------------------------------------------------------------------


class TestDetectPresetsEdges:
    """Edge cases for detect_presets."""

    def test_none_allowed_hosts_treated_as_empty(self):
        """allowed_hosts: null must not raise — treated as empty list."""
        network = {
            "allowed_hosts": None,
            "allowed_domains": None,
            "allowed_cidrs": None,
        }
        detected, remaining_hosts, remaining_domains, remaining_cidrs = detect_presets(network)
        assert isinstance(detected, list)
        assert remaining_hosts == []
        assert remaining_domains == []
        assert remaining_cidrs == []

    def test_fully_unknown_hosts_no_presets_detected(self):
        """Hosts that belong to no preset should yield an empty detected list."""
        network = {
            "allowed_hosts": ["unknown-1.example.com", "unknown-2.example.com"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert detected == []
        assert "unknown-1.example.com" in remaining_hosts
        assert "unknown-2.example.com" in remaining_hosts

    def test_openai_requires_all_three_hosts(self):
        """openai preset has 3 required hosts; only 2 present must not detect it."""
        network = {
            "allowed_hosts": ["api.openai.com", "auth.openai.com"],
            # missing chatgpt.com
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "openai" not in detected
        assert "api.openai.com" in remaining_hosts
        assert "auth.openai.com" in remaining_hosts

    def test_all_presets_detectable_from_their_own_hosts(self):
        """Every PRESETS entry must be detectable by supplying its own host list."""
        for name, preset in PRESETS.items():
            preset_hosts = preset.get("hosts", [])
            if not preset_hosts:
                continue
            network = {"allowed_hosts": list(preset_hosts), "allowed_domains": []}
            detected, _, _, _ = detect_presets(network)
            assert name in detected, (
                f"Preset '{name}' should be detectable from its own hosts {preset_hosts!r}"
            )

    def test_return_value_is_four_tuple(self):
        """detect_presets must always return exactly a 4-tuple."""
        result = detect_presets({})
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_detected_list_contains_only_strings(self):
        """All entries in the detected list must be strings."""
        network = {
            "allowed_hosts": ["api.anthropic.com"],
            "allowed_domains": [],
        }
        detected, _, _, _ = detect_presets(network)
        assert all(isinstance(p, str) for p in detected)

    def test_discord_preset_requires_both_hosts(self):
        """discord requires discord.com AND gateway.discord.gg."""
        # Only one host present — must not detect
        network_partial = {"allowed_hosts": ["discord.com"], "allowed_domains": []}
        detected_partial, remaining_partial, _, _ = detect_presets(network_partial)
        assert "discord" not in detected_partial
        assert "discord.com" in remaining_partial

        # Both hosts present — must detect
        network_full = {
            "allowed_hosts": ["discord.com", "gateway.discord.gg"],
            "allowed_domains": [],
        }
        detected_full, remaining_full, _, _ = detect_presets(network_full)
        assert "discord" in detected_full
        assert remaining_full == []

    def test_extra_hosts_beyond_preset_remain(self):
        """If extra hosts are present alongside a full preset match, they stay."""
        network = {
            "allowed_hosts": [
                "api.github.com",
                "github.com",
                "custom-ci.example.internal",
            ],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "github" in detected
        assert "custom-ci.example.internal" in remaining_hosts
        assert "api.github.com" not in remaining_hosts


# ---------------------------------------------------------------------------
# migrate_config — happy paths
# ---------------------------------------------------------------------------


class TestMigrateConfigHappyPaths:
    """Core migration scenarios."""

    def test_migrates_version_0_config(self, tmp_path):
        """A config with no version field must be migrated and version stamped."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION

    def test_idempotent_second_call_is_noop(self, tmp_path):
        """Running migrate_config twice must not change the file on the second call."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "network:\n  default_deny: true\n  allowed_hosts:\n    - api.anthropic.com\n"
        )
        backup_dir = str(tmp_path / "backups")

        result1 = migrate_config(str(cfg), backup_dir=backup_dir)
        content_after_first = cfg.read_text()

        result2 = migrate_config(str(cfg), backup_dir=backup_dir)
        content_after_second = cfg.read_text()

        assert result1["migrated"] is True
        assert result2["migrated"] is False
        assert content_after_first == content_after_second

    def test_result_contains_all_required_keys(self, tmp_path):
        """The result dict must always carry migrated, backup_path, presets_detected, version."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        for key in ("migrated", "backup_path", "presets_detected", "version"):
            assert key in result, f"Key '{key}' missing from migrate_config result"

    def test_result_version_equals_current(self, tmp_path):
        """result['version'] must equal CURRENT_CONFIG_VERSION regardless of input."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["version"] == CURRENT_CONFIG_VERSION

    def test_migrated_file_is_valid_yaml(self, tmp_path):
        """The file written during migration must parse as valid YAML."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - api.anthropic.com
                - custom.internal.host
              allowed_domains:
                - extra.example.com
              allowed_cidrs:
                - 192.168.0.0/16
            """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        parsed = yaml.safe_load(cfg.read_text())
        assert isinstance(parsed, dict)

    def test_detects_anthropic_preset_in_migration(self, tmp_path):
        """Migration injects the anthropic preset when its host is present."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "network:\n  allowed_hosts:\n    - api.anthropic.com\n  allowed_domains: []\n  allowed_cidrs: []\n"
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert "anthropic" in result["presets_detected"]
        data = yaml.safe_load(cfg.read_text())
        assert "anthropic" in data["network"]["presets"]
        assert "api.anthropic.com" not in data["network"]["allowed_hosts"]

    def test_custom_hosts_survive_migration(self, tmp_path):
        """Hosts that do not match any preset must remain in allowed_hosts after migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              allowed_hosts:
                - api.anthropic.com
                - private-api.mycompany.internal
              allowed_domains: []
              allowed_cidrs: []
            """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        assert "private-api.mycompany.internal" in data["network"]["allowed_hosts"]
        assert "api.anthropic.com" not in data["network"]["allowed_hosts"]

    def test_backup_created_during_migration(self, tmp_path):
        """A backup file must be created and referenced in result['backup_path']."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")
        backup_dir = tmp_path / "backups"

        result = migrate_config(str(cfg), backup_dir=str(backup_dir))

        assert result["backup_path"] is not None
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()

    def test_backup_contains_premigration_content(self, tmp_path):
        """The backup file must contain the original, unmigrated YAML content."""
        original_content = "network:\n  default_deny: true\n"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(original_content)
        backup_dir = tmp_path / "backups"

        result = migrate_config(str(cfg), backup_dir=str(backup_dir))

        backup_path = Path(result["backup_path"])
        assert backup_path.read_text() == original_content

    def test_missing_file_returns_migrated_false(self, tmp_path):
        """migrate_config on a non-existent path must return migrated=False."""
        result = migrate_config(
            str(tmp_path / "nonexistent.yaml"),
            backup_dir=str(tmp_path / "backups"),
        )
        assert result["migrated"] is False
        assert result["backup_path"] is None


# ---------------------------------------------------------------------------
# migrate_config — additional edge cases
# ---------------------------------------------------------------------------


class TestMigrateConfigEdges:
    """Edge cases and error paths for migrate_config."""

    def test_config_with_no_network_section_migrated(self, tmp_path):
        """A config without a network key must still be migrated and version-stamped."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            providers:
              anthropic:
                name: anthropic
                model: claude-sonnet-4-6
            """)
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION

    def test_non_network_sections_preserved(self, tmp_path):
        """shell, filesystem, providers sections must survive migration intact."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts: []
            shell:
              enabled: true
              allowed_commands:
                - git
                - python3
            filesystem:
              allowed_read_paths:
                - /home/user/workspace
            providers:
              anthropic:
                name: anthropic
                model: claude-sonnet-4-6
            workspace_path: /home/user/workspace
            """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        assert data["shell"]["enabled"] is True
        assert "git" in data["shell"]["allowed_commands"]
        assert "/home/user/workspace" in data["filesystem"]["allowed_read_paths"]
        assert data["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"
        assert data["workspace_path"] == "/home/user/workspace"

    def test_v1_config_fully_migrated_to_v2(self, tmp_path):
        """A config stamped config_version: 1 must be upgraded to CURRENT."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            config_version: 1
            network:
              default_deny: true
              allowed_hosts:
                - api.github.com
                - github.com
              allowed_domains: []
              allowed_cidrs: []
            """)
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        assert "github" in result["presets_detected"]
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION
        assert "github" in data["network"]["presets"]

    def test_already_migrated_config_file_unchanged(self, tmp_path):
        """A config at CURRENT_CONFIG_VERSION must not be modified at all."""
        original = f"config_version: {CURRENT_CONFIG_VERSION}\nnetwork:\n  default_deny: true\n"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(original)

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is False
        assert cfg.read_text() == original

    def test_no_backup_when_no_migration_needed(self, tmp_path):
        """backup_path must be None when the file is already at current version."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"config_version: {CURRENT_CONFIG_VERSION}\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["backup_path"] is None

    def test_backup_failure_does_not_abort_migration(self, tmp_path):
        """If the backup call raises, migration must still complete."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        with patch("missy.config.plan.backup_config", side_effect=OSError("disk full")):
            result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        assert result["backup_path"] is None
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION

    def test_category_overrides_untouched(self, tmp_path):
        """provider_allowed_hosts, tool_allowed_hosts, discord_allowed_hosts must not be altered."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - api.anthropic.com
              provider_allowed_hosts:
                - api.anthropic.com
              tool_allowed_hosts:
                - tool.internal.example
              discord_allowed_hosts:
                - discord.com
              allowed_domains: []
              allowed_cidrs: []
            """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        assert "api.anthropic.com" in data["network"]["provider_allowed_hosts"]
        assert "tool.internal.example" in data["network"]["tool_allowed_hosts"]
        assert "discord.com" in data["network"]["discord_allowed_hosts"]

    def test_no_presets_detected_when_hosts_dont_match(self, tmp_path):
        """Migration where no hosts match any preset results in empty presets list."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - totally.custom.internal
                - another.private.host
              allowed_domains: []
              allowed_cidrs: []
            """)
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        assert result["presets_detected"] == []
        data = yaml.safe_load(cfg.read_text())
        assert "totally.custom.internal" in data["network"]["allowed_hosts"]
        assert "another.private.host" in data["network"]["allowed_hosts"]
        # No presets key (or empty) when nothing was detected
        presets = data.get("network", {}).get("presets", [])
        assert "anthropic" not in presets
        assert "github" not in presets


# ---------------------------------------------------------------------------
# _atomic_write_yaml
# ---------------------------------------------------------------------------


class TestAtomicWriteYaml:
    """Tests for the internal _atomic_write_yaml helper."""

    def test_writes_valid_yaml(self, tmp_path):
        """Written file must parse back to the original dict."""
        target = tmp_path / "out.yaml"
        data = {"config_version": 2, "network": {"default_deny": True}}
        _atomic_write_yaml(target, data)
        parsed = yaml.safe_load(target.read_text())
        assert parsed == data

    def test_target_file_created(self, tmp_path):
        """The target path must exist after the call."""
        target = tmp_path / "out.yaml"
        _atomic_write_yaml(target, {"key": "value"})
        assert target.exists()

    def test_overwrites_existing_file(self, tmp_path):
        """Calling _atomic_write_yaml a second time replaces the old content."""
        target = tmp_path / "out.yaml"
        _atomic_write_yaml(target, {"version": 1})
        _atomic_write_yaml(target, {"version": 2})
        parsed = yaml.safe_load(target.read_text())
        assert parsed["version"] == 2

    def test_unicode_preserved(self, tmp_path):
        """Unicode values must survive the round-trip through YAML."""
        target = tmp_path / "out.yaml"
        data = {"label": "caf\u00e9 na\u00efve r\u00e9sum\u00e9"}
        _atomic_write_yaml(target, data)
        parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
        assert parsed["label"] == data["label"]
