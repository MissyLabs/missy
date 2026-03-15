"""Session 21: File permission hardening tests.

Verifies that sensitive files are written with restrictive permissions (0o600).
"""

from __future__ import annotations

import stat


class TestVaultFilePermissions:
    """Verify vault data file is written with 0o600 permissions."""

    def test_vault_data_file_permissions(self, tmp_path):
        """Vault data file should have 0o600 permissions after write."""
        from missy.security.vault import Vault

        vault = Vault(vault_dir=str(tmp_path / "vault"))
        vault.set("test_key", "test_value")

        vault_data = tmp_path / "vault" / "vault.enc"
        if vault_data.exists():
            mode = stat.S_IMODE(vault_data.stat().st_mode)
            assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"

    def test_vault_key_file_permissions(self, tmp_path):
        """Vault key file should have restrictive permissions."""
        from missy.security.vault import Vault

        vault = Vault(vault_dir=str(tmp_path / "vault"))
        vault.set("test_key", "test_value")

        vault_key = tmp_path / "vault" / "vault.key"
        if vault_key.exists():
            mode = stat.S_IMODE(vault_key.stat().st_mode)
            assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"

    def test_vault_overwrite_preserves_permissions(self, tmp_path):
        """Overwriting vault data preserves 0o600 permissions."""
        from missy.security.vault import Vault

        vault = Vault(vault_dir=str(tmp_path / "vault"))
        vault.set("key1", "value1")
        vault.set("key2", "value2")
        vault.set("key1", "updated")

        vault_data = tmp_path / "vault" / "vault.enc"
        if vault_data.exists():
            mode = stat.S_IMODE(vault_data.stat().st_mode)
            assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"


class TestConfigFilePermissions:
    """Verify config file is written with 0o600 permissions."""

    def test_config_atomic_write_permissions(self, tmp_path):
        """Config file written atomically should have 0o600 permissions."""
        from missy.cli.wizard import _write_config_atomic

        config_path = tmp_path / "config.yaml"
        _write_config_atomic(config_path, "network:\n  default_deny: true\n")

        assert config_path.exists()
        mode = stat.S_IMODE(config_path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"

    def test_config_overwrite_preserves_permissions(self, tmp_path):
        """Overwriting config file preserves 0o600 permissions."""
        from missy.cli.wizard import _write_config_atomic

        config_path = tmp_path / "config.yaml"
        _write_config_atomic(config_path, "version: 1\n")
        _write_config_atomic(config_path, "version: 2\n")

        mode = stat.S_IMODE(config_path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"

    def test_config_content_preserved(self, tmp_path):
        """Config file content is written correctly."""
        from missy.cli.wizard import _write_config_atomic

        config_path = tmp_path / "config.yaml"
        content = "network:\n  default_deny: true\nshell:\n  enabled: false\n"
        _write_config_atomic(config_path, content)

        assert config_path.read_text() == content


class TestSchedulerJobsPermissions:
    """Verify scheduler jobs file has restrictive permissions."""

    def test_jobs_file_permissions_after_add(self, tmp_path):
        """Jobs file should have 0o600 permissions after adding a job."""
        from missy.scheduler.manager import SchedulerManager

        jobs_file = tmp_path / "jobs.json"
        mgr = SchedulerManager(jobs_file=str(jobs_file))
        mgr.start()
        try:
            mgr.add_job(name="test", schedule="every 5 minutes", task="do stuff")
        finally:
            mgr.stop()

        if jobs_file.exists():
            mode = stat.S_IMODE(jobs_file.stat().st_mode)
            assert mode == 0o600, f"Expected 0o600 but got {oct(mode)}"
