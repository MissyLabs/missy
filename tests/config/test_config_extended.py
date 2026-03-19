"""Extended tests for the config subsystem.

Covers gaps not addressed by the primary test suites:

- Config migration: version upgrades, preset expansion, backup creation
- Hot reload: file change detection, callback invocation, thread safety
- Config plan: backup management, diff, rollback, pruning
- Settings loading: defaults, overrides, env var substitution
- VisionConfig: all fields, defaults, partial overrides
- Network config: preset expansion, category overrides, deduplication
- Edge cases: missing file, malformed YAML, empty config
- Backup pruning at MAX_BACKUPS boundary
- Thread safety during concurrent reload requests
- Config validation error messages
"""

from __future__ import annotations

import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from missy.config.migrate import (
    CURRENT_CONFIG_VERSION,
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
from missy.config.settings import (
    VisionConfig,
    get_default_config,
    load_config,
)
from missy.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cfg(tmp_path: Path, content: str) -> Path:
    """Write dedented *content* to tmp_path/config.yaml and return the path."""
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 1. Config migration — version upgrades
# ---------------------------------------------------------------------------


class TestMigrationVersionUpgrades:
    """Version-specific upgrade paths through migrate_config."""

    def test_version_zero_triggers_migration(self, tmp_path):
        """A config stamped config_version: 0 must be migrated."""
        cfg = _write_cfg(tmp_path, "config_version: 0\nnetwork:\n  default_deny: true\n")
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert result["migrated"] is True
        assert yaml.safe_load(cfg.read_text())["config_version"] == CURRENT_CONFIG_VERSION

    def test_version_one_triggers_migration(self, tmp_path):
        """A config stamped config_version: 1 must be migrated to CURRENT."""
        cfg = _write_cfg(tmp_path, "config_version: 1\nnetwork:\n  default_deny: true\n")
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert result["migrated"] is True

    def test_result_dict_has_all_keys_on_noop(self, tmp_path):
        """Even when no migration occurs, all result keys must be present."""
        cfg = _write_cfg(tmp_path, f"config_version: {CURRENT_CONFIG_VERSION}\nnetwork: {{}}\n")
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert result["migrated"] is False
        assert "backup_path" in result
        assert "presets_detected" in result
        assert "version" in result

    def test_no_network_section_still_stamps_version(self, tmp_path):
        """Config without a network section gets version stamp and no preset crash."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert result["migrated"] is True
        data = yaml.safe_load(cfg.read_text())
        assert data["config_version"] == CURRENT_CONFIG_VERSION


# ---------------------------------------------------------------------------
# 2. Config migration — preset expansion
# ---------------------------------------------------------------------------


class TestMigrationPresetExpansion:
    """Preset detection and expansion during migration."""

    def test_github_preset_detected_and_hosts_removed(self, tmp_path):
        """GitHub hosts are collapsed into the 'github' preset."""
        from missy.policy.presets import PRESETS

        github_hosts = PRESETS["github"]["hosts"]
        cfg = _write_cfg(
            tmp_path,
            "network:\n  default_deny: true\n  allowed_hosts:\n"
            + "".join(f"    - {h}\n" for h in github_hosts)
            + "  allowed_domains: []\n  allowed_cidrs: []\n",
        )
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert result["migrated"] is True
        assert "github" in result["presets_detected"]
        data = yaml.safe_load(cfg.read_text())
        for host in github_hosts:
            assert host not in data["network"]["allowed_hosts"]

    def test_custom_host_plus_preset_host_split_correctly(self, tmp_path):
        """Custom host stays in allowed_hosts; preset host is consumed."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            network:
              default_deny: true
              allowed_hosts:
                - api.anthropic.com
                - internal.corp.example.com
              allowed_domains: []
              allowed_cidrs: []
            """),
        )
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert "anthropic" in result["presets_detected"]
        data = yaml.safe_load(cfg.read_text())
        assert "internal.corp.example.com" in data["network"]["allowed_hosts"]
        assert "api.anthropic.com" not in data["network"]["allowed_hosts"]

    def test_domains_from_detected_preset_removed_from_remaining(self, tmp_path):
        """Domains belonging to a detected preset disappear from allowed_domains."""
        from missy.policy.presets import PRESETS

        preset_domains = PRESETS.get("anthropic", {}).get("domains", [])
        if not preset_domains:
            pytest.skip("anthropic preset has no domains")
        cfg = _write_cfg(
            tmp_path,
            "network:\n  default_deny: true\n  allowed_hosts:\n    - api.anthropic.com\n"
            "  allowed_domains:\n"
            + "".join(f"    - {d}\n" for d in preset_domains)
            + "  allowed_cidrs: []\n",
        )
        migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        data = yaml.safe_load(cfg.read_text())
        for domain in preset_domains:
            assert domain not in data["network"]["allowed_domains"]


# ---------------------------------------------------------------------------
# 3. Config migration — backup creation
# ---------------------------------------------------------------------------


class TestMigrationBackupCreation:
    """Backup behaviour during migration."""

    def test_backup_contains_pre_migration_content(self, tmp_path):
        """The backup must hold the original content, not the migrated version."""
        original = "network:\n  default_deny: true\n"
        cfg = _write_cfg(tmp_path, original)
        backup_dir = tmp_path / "bk"
        result = migrate_config(str(cfg), backup_dir=str(backup_dir))
        assert result["backup_path"] is not None
        backup = Path(result["backup_path"])
        assert backup.read_text(encoding="utf-8") == original

    def test_migration_without_backup_dir_still_succeeds(self, tmp_path):
        """migrate_config accepts no backup_dir and completes successfully."""
        cfg = _write_cfg(tmp_path, "network:\n  default_deny: true\n")
        # Patch backup_config so it doesn't try to write to ~/.missy
        with patch("missy.config.plan.backup_config", return_value=tmp_path / "fake.bk"):
            result = migrate_config(str(cfg))
        assert result["migrated"] is True

    def test_migration_result_backup_path_is_str(self, tmp_path):
        """backup_path in the result must be a string, not a Path object."""
        cfg = _write_cfg(tmp_path, "network:\n  default_deny: true\n")
        result = migrate_config(str(cfg), backup_dir=str(tmp_path / "bk"))
        assert isinstance(result["backup_path"], str)


# ---------------------------------------------------------------------------
# 4. Hot reload — file change detection and callback invocation
# ---------------------------------------------------------------------------


class TestHotReloadCallback:
    """ConfigWatcher callback behaviour."""

    @patch("missy.config.settings.load_config")
    def test_callback_receives_loaded_config(self, mock_load, tmp_path):
        """The callback is invoked with the object returned by load_config."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        cfg.chmod(0o600)
        sentinel = MagicMock(name="config-sentinel")
        mock_load.return_value = sentinel
        received = []

        def cb(c):
            received.append(c)

        w = ConfigWatcher(str(cfg), cb)
        w._do_reload()
        assert received == [sentinel]

    @patch("missy.config.settings.load_config")
    def test_multiple_direct_reloads_invoke_callback_each_time(self, mock_load, tmp_path):
        """Each _do_reload() call fires the callback once."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()
        cb = MagicMock()

        w = ConfigWatcher(str(cfg), cb)
        w._do_reload()
        w._do_reload()
        w._do_reload()
        assert cb.call_count == 3

    def test_reload_blocked_for_world_readable_only_writable(self, tmp_path):
        """World-writable (o+w) file is rejected by _check_file_safety."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        cfg.chmod(0o602)  # owner=rw, world=w
        w = ConfigWatcher(str(cfg), MagicMock())
        assert w._check_file_safety() is False

    def test_reload_accepted_for_0o644_without_write_bits_for_others(self, tmp_path):
        """A 0o644 file (read-only for group/world) passes safety check."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        cfg.chmod(0o644)
        w = ConfigWatcher(str(cfg), MagicMock())
        # 0o644 has no write bits for group or world — must be safe
        assert w._check_file_safety() is True


# ---------------------------------------------------------------------------
# 5. Hot reload — thread safety
# ---------------------------------------------------------------------------


class TestHotReloadThreadSafety:
    """Concurrent _do_reload calls are safe."""

    @patch("missy.config.settings.load_config")
    def test_concurrent_reloads_all_invoke_callback(self, mock_load, tmp_path):
        """Multiple threads calling _do_reload simultaneously all fire the callback."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        cfg.chmod(0o600)
        mock_load.return_value = MagicMock()

        results = []
        lock = threading.Lock()

        def cb(c):
            with lock:
                results.append(c)

        w = ConfigWatcher(str(cfg), cb)
        threads = [threading.Thread(target=w._do_reload) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10

    def test_watcher_daemon_thread_exits_when_main_exits(self, tmp_path):
        """The watcher thread is a daemon thread (won't block process exit)."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        w = ConfigWatcher(str(cfg), MagicMock(), poll_interval=0.1)
        w.start()
        assert w._thread is not None
        assert w._thread.daemon is True
        w.stop()

    def test_watcher_thread_name(self, tmp_path):
        """The watcher thread carries the expected name for diagnostics."""
        from missy.config.hotreload import ConfigWatcher

        cfg = tmp_path / "config.yaml"
        cfg.write_text("v: 1")
        w = ConfigWatcher(str(cfg), MagicMock(), poll_interval=0.1)
        w.start()
        assert w._thread is not None
        assert "hotreload" in w._thread.name or "missy" in w._thread.name
        w.stop()


# ---------------------------------------------------------------------------
# 6. Config plan — backup management
# ---------------------------------------------------------------------------


class TestPlanBackupManagement:
    """Backup creation, listing, and metadata."""

    def test_backup_path_contains_timestamp_digits(self, tmp_path):
        """Backup filename suffix must contain only digits and underscores."""
        cfg = tmp_path / "config.yaml"
        cfg.write_text("x: 1")
        bd = tmp_path / "bk"
        bkp = backup_config(cfg, bd)
        suffix = bkp.name[len("config.yaml.") :]
        assert all(c.isdigit() or c == "_" for c in suffix)

    def test_backup_dir_is_created_with_0o700(self, tmp_path):
        """backup_config creates the backup directory with mode 0o700."""
        import stat

        cfg = tmp_path / "config.yaml"
        cfg.write_text("x: 1")
        bd = tmp_path / "new_bk_dir"
        assert not bd.exists()
        backup_config(cfg, bd)
        mode = stat.S_IMODE(bd.stat().st_mode)
        assert mode == 0o700

    def test_backup_content_matches_source_exactly(self, tmp_path):
        """Backup must be an exact copy of the source file."""
        content = "network:\n  default_deny: true\n  presets: [anthropic]\n"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(content, encoding="utf-8")
        bkp = backup_config(cfg, tmp_path / "bk")
        assert bkp.read_text(encoding="utf-8") == content

    def test_list_backups_sorted_oldest_first(self, tmp_path):
        """list_backups must return paths sorted by ascending mtime."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        paths = []
        for i in range(3):
            cfg.write_text(f"i: {i}")
            time.sleep(0.03)
            paths.append(backup_config(cfg, bd))
        listed = list_backups(bd)
        mtimes = [p.stat().st_mtime for p in listed]
        assert mtimes == sorted(mtimes)

    def test_list_backups_ignores_files_without_timestamp_suffix(self, tmp_path):
        """Files named exactly 'config.yaml' (no dot-suffix) are excluded."""
        bd = tmp_path / "bk"
        bd.mkdir()
        (bd / "config.yaml").write_text("no suffix")
        (bd / "config.yaml.20260101_000001").write_text("with suffix")
        backups = list_backups(bd)
        assert len(backups) == 1
        assert backups[0].name == "config.yaml.20260101_000001"


# ---------------------------------------------------------------------------
# 7. Config plan — backup pruning (max 5)
# ---------------------------------------------------------------------------


class TestPlanBackupPruning:
    """MAX_BACKUPS enforcement."""

    def test_creating_six_backups_leaves_exactly_five(self, tmp_path):
        """After MAX_BACKUPS + 1 backup_config calls at most MAX_BACKUPS remain."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        timestamps = [f"2026010{i}_000000" for i in range(MAX_BACKUPS + 1)]
        for i, ts in enumerate(timestamps):
            cfg.write_text(f"v: {i}")
            with patch("missy.config.plan.time") as mt:
                mt.strftime.return_value = ts
                backup_config(cfg, bd)
        assert len(list_backups(bd)) == MAX_BACKUPS

    def test_oldest_backup_is_pruned_first(self, tmp_path):
        """The oldest backup (lowest mtime) is the one removed during pruning."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        timestamps = [f"2026010{i}_000000" for i in range(MAX_BACKUPS + 1)]
        names = []
        for i, ts in enumerate(timestamps):
            cfg.write_text(f"v: {i}")
            with patch("missy.config.plan.time") as mt:
                mt.strftime.return_value = ts
                names.append(backup_config(cfg, bd).name)
        remaining = {p.name for p in list_backups(bd)}
        # The first (oldest) backup must have been pruned
        assert names[0] not in remaining

    def test_exactly_max_backups_no_pruning(self, tmp_path):
        """Exactly MAX_BACKUPS backups must not trigger any pruning."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        for i in range(MAX_BACKUPS):
            cfg.write_text(f"v: {i}")
            with patch("missy.config.plan.time") as mt:
                mt.strftime.return_value = f"2026020{i}_000000"
                backup_config(cfg, bd)
        assert len(list_backups(bd)) == MAX_BACKUPS


# ---------------------------------------------------------------------------
# 8. Config plan — diff and rollback
# ---------------------------------------------------------------------------


class TestPlanDiffAndRollback:
    """diff_configs and rollback edge cases."""

    def test_diff_reports_filenames_in_header(self, tmp_path):
        """Unified diff output must include both file paths in the header."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("k: 1\n")
        b.write_text("k: 2\n")
        diff = diff_configs(a, b)
        assert str(a) in diff
        assert str(b) in diff

    def test_rollback_restores_most_recent_of_multiple_backups(self, tmp_path):
        """When multiple backups exist, rollback uses the most recent one."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        for i in range(3):
            cfg.write_text(f"version: {i}\n")
            time.sleep(0.03)
            backup_config(cfg, bd)
        cfg.write_text("version: modified\n")
        rollback(cfg, bd)
        # The latest backup was for version 2
        assert "version: 2" in cfg.read_text()

    def test_rollback_creates_additional_backup_of_current(self, tmp_path):
        """rollback must back up the current config before overwriting it."""
        cfg = tmp_path / "config.yaml"
        bd = tmp_path / "bk"
        cfg.write_text("original: true\n")
        with patch("missy.config.plan.time") as mt:
            mt.strftime.return_value = "20260101_000000"
            backup_config(cfg, bd)
        count_before = len(list_backups(bd))
        cfg.write_text("current: true\n")
        with patch("missy.config.plan.time") as mt:
            mt.strftime.return_value = "20260101_000001"
            rollback(cfg, bd)
        count_after = len(list_backups(bd))
        assert count_after > count_before

    def test_rollback_of_non_existent_config_creates_file(self, tmp_path):
        """If the config file does not exist when rollback is called, it is created."""
        bd = tmp_path / "bk"
        cfg = tmp_path / "config.yaml"
        cfg.write_text("from_backup: true\n")
        backup_config(cfg, bd)
        cfg.unlink()
        rollback(cfg, bd)
        assert cfg.exists()
        assert "from_backup: true" in cfg.read_text()

    def test_diff_empty_strings_for_identical_content(self, tmp_path):
        """diff_configs returns empty string when both files have the same bytes."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        content = "network:\n  default_deny: true\n"
        a.write_text(content)
        b.write_text(content)
        assert diff_configs(a, b) == ""


# ---------------------------------------------------------------------------
# 9. Settings loading — defaults and overrides
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    """Default field values when sections are absent from the YAML."""

    def test_scheduling_defaults_when_section_absent(self, tmp_path):
        """Missing scheduling section yields SchedulingPolicy with safe defaults."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.scheduling.enabled is True
        assert c.scheduling.max_jobs == 0
        assert c.scheduling.active_hours == ""

    def test_heartbeat_defaults_when_section_absent(self, tmp_path):
        """Missing heartbeat section yields HeartbeatConfig with safe defaults."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.heartbeat.enabled is False
        assert c.heartbeat.interval_seconds == 1800

    def test_observability_defaults_when_section_absent(self, tmp_path):
        """Missing observability section yields ObservabilityConfig defaults."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.observability.otel_enabled is False
        assert c.observability.otel_endpoint == "http://localhost:4317"
        assert c.observability.log_level == "warning"

    def test_vault_defaults_when_section_absent(self, tmp_path):
        """Missing vault section yields VaultConfig with vault disabled."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.vault.enabled is False
        assert "secrets" in c.vault.vault_dir

    def test_max_spend_defaults_to_zero(self, tmp_path):
        """max_spend_usd defaults to 0.0 (unlimited) when absent."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.max_spend_usd == 0.0

    def test_max_spend_parsed_as_float(self, tmp_path):
        """max_spend_usd is parsed as a float from YAML."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\nmax_spend_usd: 5\n")
        c = load_config(str(cfg))
        assert isinstance(c.max_spend_usd, float)
        assert c.max_spend_usd == 5.0

    def test_config_version_zero_when_absent(self, tmp_path):
        """config_version defaults to 0 when the field is missing."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.config_version == 0

    def test_workspace_path_defaults_to_dot(self, tmp_path):
        """workspace_path defaults to '.' when absent from config."""
        cfg = _write_cfg(tmp_path, "network: {}\n")
        c = load_config(str(cfg))
        assert c.workspace_path == "."


# ---------------------------------------------------------------------------
# 10. Settings loading — environment variable substitution for API keys
# ---------------------------------------------------------------------------


class TestSettingsEnvVarSubstitution:
    """Provider API key resolution from environment variables."""

    def test_api_key_read_from_env_var(self, tmp_path, monkeypatch):
        """When api_key is absent in YAML, the env var ANTHROPIC_API_KEY is used."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        cfg = _write_cfg(
            tmp_path,
            "providers:\n  anthropic:\n    name: anthropic\n    model: claude-test\n",
        )
        c = load_config(str(cfg))
        assert c.providers["anthropic"].api_key == "sk-from-env"

    def test_explicit_api_key_takes_precedence_over_env(self, tmp_path, monkeypatch):
        """An explicit api_key in YAML overrides the environment variable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-should-not-be-used")
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            providers:
              anthropic:
                name: anthropic
                model: claude-test
                api_key: sk-explicit
            """),
        )
        c = load_config(str(cfg))
        assert c.providers["anthropic"].api_key == "sk-explicit"

    def test_api_keys_list_fallback_when_api_key_absent(self, tmp_path, monkeypatch):
        """When api_key is absent but api_keys has entries, first entry is used."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            providers:
              anthropic:
                name: anthropic
                model: claude-test
                api_keys:
                  - sk-rotation-first
                  - sk-rotation-second
            """),
        )
        c = load_config(str(cfg))
        assert c.providers["anthropic"].api_key == "sk-rotation-first"

    def test_fast_model_and_premium_model_parsed(self, tmp_path):
        """fast_model and premium_model fields are parsed correctly."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            providers:
              anthropic:
                name: anthropic
                model: claude-sonnet-4-6
                fast_model: claude-haiku-4-5
                premium_model: claude-opus-4-6
            """),
        )
        c = load_config(str(cfg))
        assert c.providers["anthropic"].fast_model == "claude-haiku-4-5"
        assert c.providers["anthropic"].premium_model == "claude-opus-4-6"

    def test_provider_enabled_false(self, tmp_path):
        """A provider with enabled: false is loaded with enabled=False."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            providers:
              anthropic:
                name: anthropic
                model: claude-sonnet-4-6
                enabled: false
            """),
        )
        c = load_config(str(cfg))
        assert c.providers["anthropic"].enabled is False


# ---------------------------------------------------------------------------
# 11. Vision config — all fields, defaults, partial overrides
# ---------------------------------------------------------------------------


class TestVisionConfig:
    """VisionConfig dataclass defaults and YAML parsing."""

    def test_vision_defaults_from_dataclass(self):
        """All VisionConfig defaults match the documented values."""
        v = VisionConfig()
        assert v.enabled is True
        assert v.preferred_device == ""
        assert v.capture_width == 1920
        assert v.capture_height == 1080
        assert v.warmup_frames == 5
        assert v.max_retries == 3
        assert v.auto_activate_threshold == pytest.approx(0.80)
        assert v.scene_memory_max_frames == 20
        assert v.scene_memory_max_sessions == 5

    def test_vision_section_absent_gives_defaults(self, tmp_path):
        """Missing vision section in YAML yields all VisionConfig defaults."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\n")
        c = load_config(str(cfg))
        assert c.vision.enabled is True
        assert c.vision.capture_width == 1920
        assert c.vision.capture_height == 1080

    def test_vision_enabled_false(self, tmp_path):
        """vision.enabled: false disables vision subsystem."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\nvision:\n  enabled: false\n")
        c = load_config(str(cfg))
        assert c.vision.enabled is False

    def test_vision_custom_resolution(self, tmp_path):
        """Custom capture_width and capture_height are parsed."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            workspace_path: /tmp
            vision:
              capture_width: 1280
              capture_height: 720
            """),
        )
        c = load_config(str(cfg))
        assert c.vision.capture_width == 1280
        assert c.vision.capture_height == 720

    def test_vision_preferred_device_set(self, tmp_path):
        """preferred_device is parsed from YAML."""
        cfg = _write_cfg(
            tmp_path,
            "workspace_path: /tmp\nvision:\n  preferred_device: /dev/video2\n",
        )
        c = load_config(str(cfg))
        assert c.vision.preferred_device == "/dev/video2"

    def test_vision_warmup_frames_override(self, tmp_path):
        """warmup_frames override is applied."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\nvision:\n  warmup_frames: 10\n")
        c = load_config(str(cfg))
        assert c.vision.warmup_frames == 10

    def test_vision_auto_activate_threshold_override(self, tmp_path):
        """auto_activate_threshold is parsed as float."""
        cfg = _write_cfg(
            tmp_path,
            "workspace_path: /tmp\nvision:\n  auto_activate_threshold: 0.95\n",
        )
        c = load_config(str(cfg))
        assert c.vision.auto_activate_threshold == pytest.approx(0.95)

    def test_vision_scene_memory_overrides(self, tmp_path):
        """scene_memory_max_frames and scene_memory_max_sessions are parsed."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            workspace_path: /tmp
            vision:
              scene_memory_max_frames: 50
              scene_memory_max_sessions: 10
            """),
        )
        c = load_config(str(cfg))
        assert c.vision.scene_memory_max_frames == 50
        assert c.vision.scene_memory_max_sessions == 10

    def test_vision_max_retries_override(self, tmp_path):
        """max_retries is parsed from YAML."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\nvision:\n  max_retries: 7\n")
        c = load_config(str(cfg))
        assert c.vision.max_retries == 7


# ---------------------------------------------------------------------------
# 12. Network config — preset expansion and category overrides
# ---------------------------------------------------------------------------


class TestNetworkPresetExpansionInSettings:
    """Preset expansion in load_config (settings.py _parse_network)."""

    def test_anthropic_preset_expands_allowed_hosts(self, tmp_path):
        """The 'anthropic' preset adds api.anthropic.com to allowed_hosts."""
        cfg = _write_cfg(
            tmp_path,
            "network:\n  presets:\n    - anthropic\n  default_deny: true\n",
        )
        c = load_config(str(cfg))
        assert "api.anthropic.com" in c.network.allowed_hosts

    def test_preset_hosts_deduplicated_with_explicit(self, tmp_path):
        """If a preset host is also listed explicitly, it appears only once."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            network:
              presets:
                - anthropic
              allowed_hosts:
                - api.anthropic.com
              default_deny: true
            """),
        )
        c = load_config(str(cfg))
        count = c.network.allowed_hosts.count("api.anthropic.com")
        assert count == 1

    def test_category_override_hosts_loaded(self, tmp_path):
        """provider_allowed_hosts, tool_allowed_hosts, discord_allowed_hosts are parsed."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            network:
              provider_allowed_hosts:
                - provider-host.example.com
              tool_allowed_hosts:
                - tool-host.example.com
              discord_allowed_hosts:
                - discord-host.example.com
            """),
        )
        c = load_config(str(cfg))
        assert "provider-host.example.com" in c.network.provider_allowed_hosts
        assert "tool-host.example.com" in c.network.tool_allowed_hosts
        assert "discord-host.example.com" in c.network.discord_allowed_hosts

    def test_rest_policies_parsed(self, tmp_path):
        """rest_policies list is parsed from YAML."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            network:
              rest_policies:
                - host: api.github.com
                  method: GET
                  path: /repos/**
                  action: allow
            """),
        )
        c = load_config(str(cfg))
        assert len(c.network.rest_policies) == 1
        assert c.network.rest_policies[0]["host"] == "api.github.com"

    def test_multiple_presets_expand_all(self, tmp_path):
        """Multiple presets each contribute their hosts to allowed_hosts."""
        cfg = _write_cfg(
            tmp_path,
            "network:\n  presets:\n    - anthropic\n    - github\n  default_deny: true\n",
        )
        c = load_config(str(cfg))
        assert "api.anthropic.com" in c.network.allowed_hosts
        # GitHub preset must contribute at least one host
        from missy.policy.presets import PRESETS

        for h in PRESETS["github"]["hosts"]:
            assert h in c.network.allowed_hosts


# ---------------------------------------------------------------------------
# 13. Edge cases: missing file, malformed YAML, empty config
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for load_config."""

    def test_missing_file_error_message_contains_path(self, tmp_path):
        """ConfigurationError from a missing file must mention the path."""
        path = str(tmp_path / "no_such.yaml")
        with pytest.raises(ConfigurationError) as exc:
            load_config(path)
        assert "no_such.yaml" in str(exc.value)

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        """Tab-invalid YAML raises ConfigurationError with YAML in the message."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [\n  unclosed\n")
        with pytest.raises(ConfigurationError):
            load_config(str(bad))

    def test_yaml_scalar_not_mapping_raises(self, tmp_path):
        """A YAML file whose top-level value is a scalar raises ConfigurationError."""
        f = tmp_path / "scalar.yaml"
        f.write_text("just a string\n")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config(str(f))

    def test_yaml_list_not_mapping_raises(self, tmp_path):
        """A YAML file whose top-level value is a list raises ConfigurationError."""
        f = tmp_path / "list.yaml"
        f.write_text("- a\n- b\n")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_config(str(f))

    def test_empty_yaml_raises_configuration_error(self, tmp_path):
        """An empty YAML file (parses to None, not dict) raises ConfigurationError."""
        f = tmp_path / "empty.yaml"
        f.write_bytes(b"")
        with pytest.raises(ConfigurationError):
            load_config(str(f))

    def test_entirely_commented_yaml_raises_configuration_error(self, tmp_path):
        """A YAML file that is only comments (parses to None) raises ConfigurationError."""
        f = tmp_path / "comments.yaml"
        f.write_text("# just a comment\n# nothing else\n")
        with pytest.raises(ConfigurationError):
            load_config(str(f))

    def test_path_is_directory_raises_configuration_error(self, tmp_path):
        """Passing a directory path raises ConfigurationError mentioning the path."""
        with pytest.raises(ConfigurationError):
            load_config(str(tmp_path))

    def test_shell_section_null_uses_defaults(self, tmp_path):
        """A shell section that is null (not a dict) is treated as empty."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\nshell: ~\n")
        # Should not raise; null is coerced to {}
        c = load_config(str(cfg))
        assert c.shell.enabled is False

    def test_network_section_null_uses_defaults(self, tmp_path):
        """A null network section is treated as an empty mapping."""
        cfg = _write_cfg(tmp_path, "workspace_path: /tmp\nnetwork: ~\n")
        c = load_config(str(cfg))
        assert c.network.default_deny is True

    def test_provider_non_dict_raises_configuration_error(self, tmp_path):
        """A provider entry that is a scalar (not a mapping) raises ConfigurationError."""
        cfg = _write_cfg(
            tmp_path,
            "providers:\n  anthropic: just-a-string\n",
        )
        with pytest.raises(ConfigurationError):
            load_config(str(cfg))

    def test_provider_missing_model_error_message_mentions_provider(self, tmp_path):
        """ConfigurationError for missing model must mention the provider name."""
        cfg = _write_cfg(tmp_path, "providers:\n  myprovider:\n    name: myprovider\n")
        with pytest.raises(ConfigurationError) as exc:
            load_config(str(cfg))
        assert "myprovider" in str(exc.value)


# ---------------------------------------------------------------------------
# 14. needs_migration — additional edge cases
# ---------------------------------------------------------------------------


class TestNeedsMigrationExtra:
    """Additional edge cases for needs_migration not in existing test files."""

    def test_boolean_version_triggers_migration(self, tmp_path):
        """config_version: true should trigger migration (non-integer)."""
        cfg = _write_cfg(tmp_path, "config_version: true\n")
        # True == 1 in Python, so int(True) < CURRENT_CONFIG_VERSION — should migrate
        result = needs_migration(str(cfg))
        assert result is True

    def test_zero_float_version_triggers_migration(self, tmp_path):
        """config_version: 0.0 is treated as version 0 and triggers migration."""
        cfg = _write_cfg(tmp_path, "config_version: 0.0\n")
        assert needs_migration(str(cfg)) is True

    def test_large_version_does_not_migrate(self, tmp_path):
        """A very large config_version (future) never triggers migration."""
        cfg = _write_cfg(tmp_path, f"config_version: {CURRENT_CONFIG_VERSION + 999}\n")
        assert needs_migration(str(cfg)) is False


# ---------------------------------------------------------------------------
# 15. get_default_config
# ---------------------------------------------------------------------------


class TestGetDefaultConfigExtended:
    """Additional coverage for get_default_config."""

    def test_default_config_scheduling_enabled(self):
        """Default scheduling policy has enabled=True."""
        cfg = get_default_config()
        assert cfg.scheduling.enabled is True

    def test_default_config_heartbeat_disabled(self):
        """Default heartbeat has enabled=False."""
        cfg = get_default_config()
        assert cfg.heartbeat.enabled is False

    def test_default_config_vault_disabled(self):
        """Default vault has enabled=False."""
        cfg = get_default_config()
        assert cfg.vault.enabled is False

    def test_default_config_discord_is_none(self):
        """Default discord is None (not configured)."""
        cfg = get_default_config()
        assert cfg.discord is None

    def test_default_config_config_version_zero(self):
        """Default config_version is 0 (pre-migration)."""
        cfg = get_default_config()
        assert cfg.config_version == 0

    def test_default_config_proactive_disabled(self):
        """Default proactive config has enabled=False."""
        cfg = get_default_config()
        assert cfg.proactive.enabled is False
        assert cfg.proactive.triggers == []

    def test_default_config_max_spend_zero(self):
        """Default max_spend_usd is 0.0."""
        cfg = get_default_config()
        assert cfg.max_spend_usd == 0.0

    def test_two_default_configs_are_independent(self):
        """Mutating one default config's lists must not affect another."""
        cfg1 = get_default_config()
        cfg2 = get_default_config()
        cfg1.network.allowed_hosts.append("leak.example.com")
        assert "leak.example.com" not in cfg2.network.allowed_hosts


# ---------------------------------------------------------------------------
# 16. Scheduling and heartbeat full YAML parsing
# ---------------------------------------------------------------------------


class TestSchedulingAndHeartbeatParsing:
    """Scheduling and heartbeat section parsing from YAML."""

    def test_scheduling_all_fields(self, tmp_path):
        """All scheduling fields are parsed from YAML."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            workspace_path: /tmp
            scheduling:
              enabled: false
              max_jobs: 10
              active_hours: "09:00-17:00"
            """),
        )
        c = load_config(str(cfg))
        assert c.scheduling.enabled is False
        assert c.scheduling.max_jobs == 10
        assert c.scheduling.active_hours == "09:00-17:00"

    def test_heartbeat_all_fields(self, tmp_path):
        """All heartbeat fields are parsed from YAML."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            workspace_path: /tmp
            heartbeat:
              enabled: true
              interval_seconds: 900
              workspace: /custom/workspace
              active_hours: "08:00-22:00"
            """),
        )
        c = load_config(str(cfg))
        assert c.heartbeat.enabled is True
        assert c.heartbeat.interval_seconds == 900
        assert c.heartbeat.workspace == "/custom/workspace"
        assert c.heartbeat.active_hours == "08:00-22:00"

    def test_observability_all_fields(self, tmp_path):
        """All observability fields are parsed from YAML."""
        cfg = _write_cfg(
            tmp_path,
            textwrap.dedent("""\
            workspace_path: /tmp
            observability:
              otel_enabled: true
              otel_endpoint: http://collector:4317
              otel_protocol: http/protobuf
              otel_service_name: my-agent
              log_level: debug
            """),
        )
        c = load_config(str(cfg))
        assert c.observability.otel_enabled is True
        assert c.observability.otel_endpoint == "http://collector:4317"
        assert c.observability.otel_protocol == "http/protobuf"
        assert c.observability.otel_service_name == "my-agent"
        assert c.observability.log_level == "debug"
