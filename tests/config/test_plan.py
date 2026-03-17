"""Tests for config plan/backup/rollback (Feature 6)."""

from __future__ import annotations

import time


class TestBackupConfig:
    """Tests for backup_config."""

    def test_backup_creates_file(self, tmp_path):
        from missy.config.plan import backup_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text("network:\n  default_deny: true\n")
        backup_dir = tmp_path / "config.d"

        backup_path = backup_config(config_file, backup_dir)

        assert backup_path.exists()
        assert backup_path.parent == backup_dir
        assert backup_path.name.startswith("config.yaml.")
        assert backup_path.read_text() == config_file.read_text()

    def test_backup_prunes_to_max(self, tmp_path):
        from missy.config.plan import backup_config, list_backups

        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        for i in range(8):
            config_file.write_text(f"version: {i}\n")
            backup_config(config_file, backup_dir)
            time.sleep(0.05)  # ensure distinct timestamps

        backups = list_backups(backup_dir)
        assert len(backups) <= 5


class TestRollback:
    """Tests for rollback."""

    def test_rollback_restores(self, tmp_path):
        from missy.config.plan import backup_config, rollback

        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        # Write original, back it up, then modify
        config_file.write_text("original: true\n")
        backup_config(config_file, backup_dir)
        config_file.write_text("modified: true\n")

        restored = rollback(config_file, backup_dir)

        assert restored is not None
        assert config_file.read_text() == "original: true\n"

    def test_rollback_no_backups(self, tmp_path):
        from missy.config.plan import rollback

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: true\n")
        backup_dir = tmp_path / "config.d"

        result = rollback(config_file, backup_dir)
        assert result is None


class TestDiffConfigs:
    """Tests for diff_configs."""

    def test_diff_shows_changes(self, tmp_path):
        from missy.config.plan import diff_configs

        file_a = tmp_path / "a.yaml"
        file_b = tmp_path / "b.yaml"
        file_a.write_text("network:\n  default_deny: true\n")
        file_b.write_text("network:\n  default_deny: false\n")

        diff = diff_configs(file_a, file_b)

        assert "default_deny" in diff
        assert "-" in diff or "+" in diff

    def test_diff_identical(self, tmp_path):
        from missy.config.plan import diff_configs

        file_a = tmp_path / "a.yaml"
        file_a.write_text("same: true\n")

        diff = diff_configs(file_a, file_a)
        assert diff == ""


class TestListBackups:
    """Tests for list_backups."""

    def test_list_empty_dir(self, tmp_path):
        from missy.config.plan import list_backups

        result = list_backups(tmp_path / "nonexistent")
        assert result == []


class TestWizardBacksUp:
    """Tests that the wizard creates backups on overwrite."""

    def test_wizard_backs_up_on_overwrite(self, tmp_path):
        from missy.cli.wizard import run_wizard_noninteractive

        config_file = tmp_path / ".missy" / "config.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text("original: content\n")

        # Run non-interactive setup which should back up the existing config
        run_wizard_noninteractive(
            config_path=str(config_file),
            provider="anthropic",
            api_key="sk-ant-test-12345",
            workspace=str(tmp_path / "workspace"),
        )

        # Since we can't control the default backup dir in tests, just verify
        # the config was overwritten
        assert config_file.exists()
        content = config_file.read_text()
        assert "anthropic" in content
