"""Edge case tests for config migration and config plan systems.

These tests focus on boundary conditions, error paths, and behaviours that
are not covered by the primary test suites in test_migrate.py and test_plan.py.
"""

from __future__ import annotations

import os
import stat
import textwrap
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from missy.config.migrate import (
    CURRENT_CONFIG_VERSION,
    detect_presets,
    migrate_config,
    needs_migration,
)
from missy.config.plan import (
    MAX_BACKUPS,
    backup_config,
    diff_configs,
    list_backups,
    rollback,
)


# ---------------------------------------------------------------------------
# Migration edge cases
# ---------------------------------------------------------------------------


class TestNeedsMigrationEdgeCases:
    """Edge cases for needs_migration."""

    def test_already_migrated_returns_false(self, tmp_path):
        """Config stamped with CURRENT_CONFIG_VERSION must not trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"config_version: {CURRENT_CONFIG_VERSION}\n"
            "network:\n  default_deny: true\n"
        )
        assert needs_migration(str(cfg)) is False

    def test_future_version_returns_false(self, tmp_path):
        """A config with a version number higher than current is not migrated."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"config_version: {CURRENT_CONFIG_VERSION + 10}\n")
        assert needs_migration(str(cfg)) is False

    def test_corrupted_yaml_returns_false(self, tmp_path):
        """Unparseable YAML should not raise — it should return False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network: :\n  bad: [unclosed\n")
        assert needs_migration(str(cfg)) is False

    def test_non_dict_yaml_returns_false(self, tmp_path):
        """YAML that parses to a non-dict (e.g. a plain string) is not migrated."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("- item1\n- item2\n")
        assert needs_migration(str(cfg)) is False

    def test_missing_file_returns_false(self, tmp_path):
        """A path that does not exist should return False, not raise."""
        assert needs_migration(str(tmp_path / "does_not_exist.yaml")) is False

    def test_path_is_directory_returns_false(self, tmp_path):
        """Passing a directory path should return False, not raise."""
        assert needs_migration(str(tmp_path)) is False

    def test_non_integer_version_triggers_migration(self, tmp_path):
        """A non-integer version value (e.g. a string) should trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: 'not-a-number'\n")
        assert needs_migration(str(cfg)) is True

    def test_null_version_triggers_migration(self, tmp_path):
        """config_version: null (YAML null) should trigger migration."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("config_version: null\n")
        assert needs_migration(str(cfg)) is True

    def test_empty_file_returns_false(self, tmp_path):
        """An empty file parses to None — not a dict — should return False."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        assert needs_migration(str(cfg)) is False

    def test_tilde_in_path_resolved(self, tmp_path):
        """needs_migration must handle paths with a leading tilde gracefully."""
        # We can't easily write to ~/ in tests; instead verify that a
        # non-existent tilde path returns False rather than crashing.
        assert needs_migration("~/this-path-definitely-does-not-exist.yaml") is False


class TestDetectPresetsEdgeCases:
    """Edge cases for detect_presets."""

    def test_unknown_hosts_preserved_as_remaining(self):
        """Hosts that do not match any preset should be left in remaining_hosts."""
        network = {
            "allowed_hosts": ["totally.unknown.example.com", "another.unknown.net"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert detected == []
        assert "totally.unknown.example.com" in remaining_hosts
        assert "another.unknown.net" in remaining_hosts

    def test_partial_preset_hosts_not_consumed(self):
        """Partial match hosts stay in remaining_hosts when preset is not detected."""
        # Anthropic requires only api.anthropic.com; that alone IS a full match.
        # Use a multi-host preset (openai: 3 hosts) with only 2 present.
        network = {
            "allowed_hosts": ["api.openai.com", "auth.openai.com"],
            "allowed_domains": [],
        }
        detected, remaining_hosts, _, _ = detect_presets(network)
        assert "openai" not in detected
        # Both partial hosts must remain since the preset was not detected
        assert "api.openai.com" in remaining_hosts
        assert "auth.openai.com" in remaining_hosts

    def test_preset_with_cidr_removes_cidr_from_remaining(self):
        """Detected preset CIDRs are removed from remaining_cidrs."""
        # ollama preset has cidrs: ["127.0.0.0/8"]
        network = {
            "allowed_hosts": ["localhost:11434", "127.0.0.1:11434"],
            "allowed_domains": [],
            "allowed_cidrs": ["127.0.0.0/8", "10.0.0.0/8"],
        }
        detected, _, _, remaining_cidrs = detect_presets(network)
        assert "ollama" in detected
        assert "127.0.0.0/8" not in remaining_cidrs
        # Non-preset CIDR must survive
        assert "10.0.0.0/8" in remaining_cidrs

    def test_already_listed_preset_avoids_duplicate(self):
        """A preset already in config presets is not added twice."""
        network = {
            "presets": ["anthropic", "anthropic"],
            "allowed_hosts": [],
            "allowed_domains": [],
        }
        detected, _, _, _ = detect_presets(network)
        assert detected.count("anthropic") == 2  # existing dupes preserved as-is

    def test_none_values_treated_as_empty(self):
        """None values for allowed_hosts / allowed_domains must not raise."""
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

    def test_all_known_presets_detectable(self):
        """Every preset in PRESETS should be detectable given its full host list."""
        from missy.policy.presets import PRESETS

        for name, preset in PRESETS.items():
            preset_hosts = preset.get("hosts", [])
            if not preset_hosts:
                continue  # skip presets with no required hosts
            network = {"allowed_hosts": list(preset_hosts), "allowed_domains": []}
            detected, _, _, _ = detect_presets(network)
            assert name in detected, f"Preset '{name}' should be detectable from its own hosts"


class TestMigrateConfigEdgeCases:
    """Edge cases for migrate_config."""

    def test_already_migrated_config_unchanged(self, tmp_path):
        """A config at CURRENT_CONFIG_VERSION must be returned without modification."""
        cfg = tmp_path / "config.yaml"
        original_text = (
            f"config_version: {CURRENT_CONFIG_VERSION}\n"
            "network:\n  default_deny: true\n"
        )
        cfg.write_text(original_text)

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is False
        assert result["backup_path"] is None
        assert result["presets_detected"] == []
        # File must not have changed
        assert cfg.read_text() == original_text

    def test_backup_created_before_migration(self, tmp_path):
        """A backup file must exist before the migrated config is written."""
        cfg = tmp_path / "config.yaml"
        original_content = "network:\n  default_deny: true\n"
        cfg.write_text(original_content)
        backup_dir = tmp_path / "backups"

        result = migrate_config(str(cfg), backup_dir=str(backup_dir))

        assert result["migrated"] is True
        assert result["backup_path"] is not None
        backup_path = Path(result["backup_path"])
        assert backup_path.exists()
        # Backup should contain the pre-migration content
        assert backup_path.read_text() == original_content

    def test_migration_idempotent_second_call_noop(self, tmp_path):
        """Running migrate_config twice must produce identical file content."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
              allowed_domains: []
              allowed_cidrs: []
        """)
        )
        backup_dir = tmp_path / "backups"

        result1 = migrate_config(str(cfg), backup_dir=str(backup_dir))
        content_after_first = cfg.read_text()

        result2 = migrate_config(str(cfg), backup_dir=str(backup_dir))
        content_after_second = cfg.read_text()

        assert result1["migrated"] is True
        assert result2["migrated"] is False
        assert content_after_first == content_after_second

    def test_corrupted_yaml_raises_on_migrate(self, tmp_path):
        """Corrupted YAML is not migrated (needs_migration returns False)."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network: :\n  bad: [unclosed\n")
        backup_dir = tmp_path / "backups"

        result = migrate_config(str(cfg), backup_dir=str(backup_dir))

        # needs_migration returns False for bad YAML, so migrated is False
        assert result["migrated"] is False
        assert result["backup_path"] is None

    def test_empty_yaml_file_not_migrated(self, tmp_path):
        """Truly empty file (YAML None) is not migrated."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("")
        backup_dir = tmp_path / "backups"

        result = migrate_config(str(cfg), backup_dir=str(backup_dir))

        assert result["migrated"] is False

    def test_missing_config_file_returns_no_migration(self, tmp_path):
        """migrate_config on a non-existent path must return migrated=False."""
        result = migrate_config(
            str(tmp_path / "nonexistent.yaml"),
            backup_dir=str(tmp_path / "backups"),
        )
        assert result["migrated"] is False
        assert result["backup_path"] is None

    def test_unknown_hosts_not_migrated_to_preset(self, tmp_path):
        """Hosts that do not match any preset must remain in allowed_hosts."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "totally.custom.internal"
                - "another.private.host"
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

    def test_config_version_stamped_after_migration(self, tmp_path):
        """After migration the file must carry config_version == CURRENT_CONFIG_VERSION."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION

    def test_migrated_file_is_valid_yaml(self, tmp_path):
        """The atomically written file must always parse as valid YAML."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - "api.anthropic.com"
                - "my-custom-host.example.com"
              allowed_domains:
                - "extra.domain.example"
              allowed_cidrs:
                - "192.168.1.0/24"
        """)
        )

        migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        parsed = yaml.safe_load(cfg.read_text())
        assert isinstance(parsed, dict)

    def test_v1_config_with_preset_hosts_fully_migrated(self, tmp_path):
        """v1 config where all hosts match a preset gets that preset injected."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            textwrap.dedent("""\
            config_version: 1
            network:
              default_deny: true
              allowed_hosts:
                - "api.github.com"
                - "github.com"
              allowed_domains: []
              allowed_cidrs: []
        """)
        )

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["migrated"] is True
        assert "github" in result["presets_detected"]
        data = yaml.safe_load(cfg.read_text())
        assert "github" in data["network"]["presets"]
        # Preset hosts must be removed from allowed_hosts
        assert "api.github.com" not in data["network"]["allowed_hosts"]
        assert "github.com" not in data["network"]["allowed_hosts"]

    def test_backup_failure_does_not_abort_migration(self, tmp_path):
        """If backup raises, migration still proceeds."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        def _raise(*args, **kwargs):
            raise OSError("disk full")

        with patch("missy.config.plan.backup_config", side_effect=_raise):
            result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        # Migration must still complete despite backup failure
        assert result["migrated"] is True
        assert result["backup_path"] is None
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION

    def test_migrate_preserves_non_network_sections(self, tmp_path):
        """Non-network sections (shell, filesystem, providers) survive migration."""
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

    def test_migrate_result_keys_always_present(self, tmp_path):
        """The result dict must always contain all documented keys."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        for key in ("migrated", "backup_path", "presets_detected", "version"):
            assert key in result, f"Expected key '{key}' missing from result"

    def test_result_version_is_current(self, tmp_path):
        """result['version'] must always equal CURRENT_CONFIG_VERSION."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("network:\n  default_deny: true\n")

        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "backups"))

        assert result["version"] == CURRENT_CONFIG_VERSION


# ---------------------------------------------------------------------------
# Config plan edge cases
# ---------------------------------------------------------------------------


class TestBackupConfigEdgeCases:
    """Edge cases for backup_config."""

    def test_sixth_backup_prunes_oldest(self, tmp_path):
        """Creating MAX_BACKUPS + 1 backups must prune the oldest exactly once.

        backup_config uses time.strftime for the filename, which has one-second
        resolution.  Patching it ensures distinct filenames even when the loop
        completes in under a second.
        """
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        timestamps = [f"20260101_00000{i}" for i in range(MAX_BACKUPS + 1)]
        names_created: list[str] = []

        for i, ts in enumerate(timestamps):
            config_file.write_text(f"iteration: {i}\n")
            with patch("missy.config.plan.time") as mock_time:
                mock_time.strftime.return_value = ts
                bkp = backup_config(config_file, backup_dir)
            names_created.append(bkp.name)

        remaining = list_backups(backup_dir)
        assert len(remaining) == MAX_BACKUPS

        # The very first backup (oldest by mtime) must have been pruned
        remaining_names = {p.name for p in remaining}
        assert names_created[0] not in remaining_names

    def test_backup_dir_created_if_missing(self, tmp_path):
        """backup_config must create the backup directory when it does not exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: true\n")
        backup_dir = tmp_path / "deep" / "nested" / "config.d"

        assert not backup_dir.exists()
        backup_config(config_file, backup_dir)
        assert backup_dir.exists()

    def test_backup_preserves_exact_content(self, tmp_path):
        """The backup file must be a byte-for-byte copy of the original."""
        config_file = tmp_path / "config.yaml"
        content = "network:\n  default_deny: true\nunicode: \u00e9\u00e0\u00fc\n"
        config_file.write_text(content, encoding="utf-8")
        backup_dir = tmp_path / "config.d"

        bkp = backup_config(config_file, backup_dir)

        assert bkp.read_text(encoding="utf-8") == content

    def test_backup_filename_pattern(self, tmp_path):
        """Backup files must be named config.yaml.<timestamp>."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("x: 1\n")
        backup_dir = tmp_path / "config.d"

        bkp = backup_config(config_file, backup_dir)

        assert bkp.name.startswith("config.yaml.")
        suffix = bkp.name[len("config.yaml."):]
        # Timestamp portion must be non-empty digits/underscores
        assert all(c.isdigit() or c == "_" for c in suffix)

    def test_multiple_backups_at_max_keeps_exactly_max(self, tmp_path):
        """Creating exactly MAX_BACKUPS entries should not prune any.

        Uses patched timestamps to guarantee distinct filenames within the
        same wall-clock second.
        """
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        for i in range(MAX_BACKUPS):
            config_file.write_text(f"v: {i}\n")
            with patch("missy.config.plan.time") as mock_time:
                mock_time.strftime.return_value = f"20260101_10000{i}"
                backup_config(config_file, backup_dir)

        assert len(list_backups(backup_dir)) == MAX_BACKUPS

    def test_backup_dir_has_restrictive_permissions(self, tmp_path):
        """The backup directory should be created with mode 0o700."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: true\n")
        backup_dir = tmp_path / "config.d"

        backup_config(config_file, backup_dir)

        mode = stat.S_IMODE(backup_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"


class TestListBackupsEdgeCases:
    """Edge cases for list_backups."""

    def test_no_backups_returns_empty_list(self, tmp_path):
        """list_backups on a non-existent directory must return [] not raise."""
        result = list_backups(tmp_path / "nonexistent")
        assert result == []

    def test_empty_backup_dir_returns_empty_list(self, tmp_path):
        """An existing but empty backup directory yields an empty list."""
        backup_dir = tmp_path / "config.d"
        backup_dir.mkdir()
        assert list_backups(backup_dir) == []

    def test_non_backup_files_ignored(self, tmp_path):
        """Files not matching config.yaml.* are excluded from the listing."""
        backup_dir = tmp_path / "config.d"
        backup_dir.mkdir()
        (backup_dir / "unrelated.txt").write_text("noise\n")
        (backup_dir / "config.yaml").write_text("current\n")  # no timestamp suffix
        (backup_dir / "config.yaml.20240101_120000").write_text("backup\n")

        backups = list_backups(backup_dir)
        assert len(backups) == 1
        assert backups[0].name == "config.yaml.20240101_120000"

    def test_backups_ordered_oldest_first(self, tmp_path):
        """list_backups must return files sorted oldest mtime first."""
        backup_dir = tmp_path / "config.d"
        backup_dir.mkdir()
        config_file = tmp_path / "config.yaml"

        paths: list[Path] = []
        for i in range(3):
            config_file.write_text(f"i: {i}\n")
            time.sleep(0.02)
            paths.append(backup_config(config_file, backup_dir))

        backups = list_backups(backup_dir)
        mtimes = [p.stat().st_mtime for p in backups]
        assert mtimes == sorted(mtimes)


class TestRollbackEdgeCases:
    """Edge cases for rollback."""

    def test_rollback_no_backups_returns_none(self, tmp_path):
        """rollback must return None (not raise) when no backups exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("current: true\n")
        backup_dir = tmp_path / "config.d"

        result = rollback(config_file, backup_dir)

        assert result is None
        # Current config must be untouched
        assert config_file.read_text() == "current: true\n"

    def test_rollback_restores_latest_backup(self, tmp_path):
        """rollback must write the content of the most-recent backup to the config."""
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        config_file.write_text("version: 1\n")
        time.sleep(0.02)
        backup_config(config_file, backup_dir)

        config_file.write_text("version: 2\n")
        time.sleep(0.02)
        backup_config(config_file, backup_dir)

        # Now overwrite with something else
        config_file.write_text("version: 3\n")

        rolled_back = rollback(config_file, backup_dir)

        assert rolled_back is not None
        # The active config should now hold "version: 2" (the latest backup)
        assert "version: 2" in config_file.read_text()

    def test_rollback_backs_up_current_before_restoring(self, tmp_path):
        """rollback must create an extra backup of the current config before overwriting.

        Uses patched timestamps to guarantee the initial backup and the rollback
        safety backup get distinct filenames even within the same wall-clock second.
        """
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        config_file.write_text("original: true\n")
        with patch("missy.config.plan.time") as mock_time:
            mock_time.strftime.return_value = "20260101_100000"
            backup_config(config_file, backup_dir)

        config_file.write_text("current: true\n")
        backups_before = len(list_backups(backup_dir))

        with patch("missy.config.plan.time") as mock_time:
            mock_time.strftime.return_value = "20260101_100001"
            rollback(config_file, backup_dir)

        backups_after = len(list_backups(backup_dir))

        # An additional backup of "current: true" must have been created
        assert backups_after > backups_before

    def test_rollback_when_config_file_missing(self, tmp_path):
        """rollback when the active config does not exist must still restore."""
        backup_dir = tmp_path / "config.d"
        config_file = tmp_path / "config.yaml"

        # Create a backup without the config currently existing
        config_file.write_text("backed_up: true\n")
        backup_config(config_file, backup_dir)
        config_file.unlink()

        rolled_back = rollback(config_file, backup_dir)

        assert rolled_back is not None
        assert config_file.exists()
        assert "backed_up: true" in config_file.read_text()

    def test_rollback_returns_path_to_restored_backup(self, tmp_path):
        """rollback return value must be a Path pointing to the source backup."""
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        config_file.write_text("original: true\n")
        created_backup = backup_config(config_file, backup_dir)
        config_file.write_text("changed: true\n")

        result = rollback(config_file, backup_dir)

        assert result is not None
        assert isinstance(result, Path)
        assert result.exists()
        assert result.name == created_backup.name


class TestDiffConfigsEdgeCases:
    """Edge cases for diff_configs."""

    def test_identical_files_produce_empty_diff(self, tmp_path):
        """diff_configs on two identical files must return an empty string."""
        file_a = tmp_path / "a.yaml"
        file_a.write_text("network:\n  default_deny: true\n")
        file_b = tmp_path / "b.yaml"
        file_b.write_text("network:\n  default_deny: true\n")

        assert diff_configs(file_a, file_b) == ""

    def test_diff_shows_added_key(self, tmp_path):
        """A new key added in file_b must appear as a '+' line in the diff."""
        file_a = tmp_path / "a.yaml"
        file_a.write_text("network:\n  default_deny: true\n")
        file_b = tmp_path / "b.yaml"
        file_b.write_text("network:\n  default_deny: true\n  allowed_hosts: []\n")

        diff = diff_configs(file_a, file_b)
        assert "+" in diff
        assert "allowed_hosts" in diff

    def test_diff_shows_removed_key(self, tmp_path):
        """A key removed in file_b must appear as a '-' line in the diff."""
        file_a = tmp_path / "a.yaml"
        file_a.write_text("network:\n  default_deny: true\n  allowed_hosts: []\n")
        file_b = tmp_path / "b.yaml"
        file_b.write_text("network:\n  default_deny: true\n")

        diff = diff_configs(file_a, file_b)
        assert "-" in diff
        assert "allowed_hosts" in diff

    def test_diff_same_file_reference_is_empty(self, tmp_path):
        """Diffing a file against itself must return an empty string."""
        file_a = tmp_path / "a.yaml"
        file_a.write_text("key: value\n")

        assert diff_configs(file_a, file_a) == ""

    def test_diff_multiline_content(self, tmp_path):
        """Diff works correctly for files with many lines."""
        lines_a = "\n".join(f"line{i}: {i}" for i in range(50)) + "\n"
        lines_b = "\n".join(f"line{i}: {i * 2}" for i in range(50)) + "\n"
        file_a = tmp_path / "a.yaml"
        file_b = tmp_path / "b.yaml"
        file_a.write_text(lines_a)
        file_b.write_text(lines_b)

        diff = diff_configs(file_a, file_b)
        assert diff != ""
        assert "-" in diff
        assert "+" in diff


class TestPlanShowsChanges:
    """Integration-style tests verifying the plan/diff workflow."""

    def test_diff_between_current_and_latest_backup(self, tmp_path):
        """Simulate the 'missy config diff' workflow: diff current vs latest backup."""
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        config_file.write_text("network:\n  default_deny: true\n")
        backup_config(config_file, backup_dir)

        # Simulate a change made after the backup
        config_file.write_text(
            "network:\n  default_deny: true\n  allowed_hosts:\n    - api.anthropic.com\n"
        )

        latest_backup = list_backups(backup_dir)[-1]
        diff = diff_configs(latest_backup, config_file)

        assert "allowed_hosts" in diff
        assert "api.anthropic.com" in diff

    def test_no_diff_when_unchanged_since_backup(self, tmp_path):
        """If the config has not changed since the backup, the diff is empty."""
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        config_file.write_text("network:\n  default_deny: true\n")
        backup_config(config_file, backup_dir)

        latest_backup = list_backups(backup_dir)[-1]
        diff = diff_configs(latest_backup, config_file)

        assert diff == ""

    def test_migrate_then_diff_shows_migration_changes(self, tmp_path):
        """After migrating, diff between backup and current shows the preset injection."""
        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        original = textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - api.anthropic.com
              allowed_domains: []
              allowed_cidrs: []
        """)
        config_file.write_text(original)

        result = migrate_config(str(config_file), backup_dir=str(backup_dir))

        assert result["migrated"] is True
        backup_path = Path(result["backup_path"])
        diff = diff_configs(backup_path, config_file)

        # Diff must show the preset being added and the host being removed
        assert "presets" in diff or "anthropic" in diff
        assert "config_version" in diff
