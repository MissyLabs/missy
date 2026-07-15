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

    def test_two_backups_within_the_same_second_do_not_collide(self, tmp_path, monkeypatch):
        """Regression: two backup_config() calls within the same wall-clock
        second (the timestamp's resolution) previously produced the
        identical filename, and shutil.copy2() silently overwrote the first
        backup with the second's content -- no error, no warning, the first
        backup's data simply gone.
        """
        from missy.config.plan import backup_config

        # Freeze the timestamp so both calls fall in the "same second"
        # deterministically, rather than relying on real clock timing.
        monkeypatch.setattr("missy.config.plan.time.strftime", lambda fmt: "20260101_120000")

        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"

        config_file.write_text("version: 1\n")
        path1 = backup_config(config_file, backup_dir)

        config_file.write_text("version: 2\n")
        path2 = backup_config(config_file, backup_dir)

        assert path1 != path2, "same-second backups must not share a filename"
        assert path1.exists()
        assert path2.exists()
        assert path1.read_text() == "version: 1\n"
        assert path2.read_text() == "version: 2\n"

    def test_ordering_survives_tied_mtimes_from_unchanged_source(self, tmp_path, monkeypatch):
        """Regression: list_backups()/rollback()/_prune_backups() sorted by
        stat().st_mtime, but shutil.copy2() (used by backup_config())
        preserves the *source* config file's mtime on the copy, not the
        time the backup was actually made. Two backups of an unchanged
        source file get IDENTICAL mtimes, even when their filenames
        (and the disambiguating _N suffix backup_config() already adds
        for same-second collisions) correctly encode true creation order.
        Sorting must use the filename, not mtime, so ties can't scramble
        backup ordering.
        """
        from missy.config.plan import backup_config, list_backups

        config_file = tmp_path / "config.yaml"
        backup_dir = tmp_path / "config.d"
        config_file.write_text("version: 1\n")

        # Two backups of the SAME unchanged content: shutil.copy2() gives
        # both copies the identical source mtime, regardless of real
        # wall-clock time between the two calls.
        path1 = backup_config(config_file, backup_dir)
        path2 = backup_config(config_file, backup_dir)

        assert path1.stat().st_mtime == path2.stat().st_mtime
        assert path1 != path2

        backups = list_backups(backup_dir)
        assert backups[-1] == path2, (
            "the true most-recent backup must sort last despite tied mtimes"
        )
        assert backups[0] == path1


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
        from missy.config.plan import list_backups

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

        assert config_file.exists()
        content = config_file.read_text()
        assert "anthropic" in content

        # CFGPLAN-001 (6th tool-specific validation run): backup_config()
        # used to default to the hardcoded, absolute ~/.missy/config.d
        # regardless of *this* tmp_path config file's own location, so
        # every test run silently polluted the real operator's config
        # backup history with fake fixture content -- and could evict
        # genuine backups out of the retained max-5 window. The backup
        # must now land in a config.d directory alongside this tmp_path
        # config file, not the real one.
        backups = list_backups(config_path=config_file)
        assert len(backups) == 1
        assert backups[0].parent == config_file.parent / "config.d"
        assert backups[0].read_text() == "original: content\n"
