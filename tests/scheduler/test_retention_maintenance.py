"""SCHED-04: daily retention maintenance."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from missy.config.settings import RetentionConfig
from missy.scheduler.maintenance import (
    MAINTENANCE_JOB_ID,
    prune_directory,
    register_maintenance_job,
    run_retention,
)


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


class TestPruneDirectory:
    def test_prunes_old_files_and_empty_dirs_only(self, tmp_path):
        old = tmp_path / "a" / "old.jpg"
        old.parent.mkdir()
        old.write_bytes(b"x")
        _age(old, 30)
        fresh = tmp_path / "fresh.jpg"
        fresh.write_bytes(b"x")
        assert prune_directory(tmp_path, 14) == 1
        assert not old.exists() and not old.parent.exists()
        assert fresh.exists() and tmp_path.exists()

    def test_never_follows_or_deletes_symlinks(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "precious.txt"
        target.write_text("keep")
        _age(target, 100)
        base = tmp_path / "captures"
        base.mkdir()
        (base / "link_dir").symlink_to(outside, target_is_directory=True)
        (base / "link_file").symlink_to(target)
        assert prune_directory(base, 1) == 0
        assert target.exists()
        assert (base / "link_file").is_symlink()

    def test_excluded_top_level_skipped(self, tmp_path):
        inbound = tmp_path / "discord_inbound" / "x.png"
        inbound.parent.mkdir()
        inbound.write_bytes(b"x")
        _age(inbound, 100)
        assert prune_directory(tmp_path, 1, exclude_top_level=["discord_inbound"]) == 0
        assert inbound.exists()

    def test_zero_days_disabled_and_missing_dir_ok(self, tmp_path):
        assert prune_directory(tmp_path, 0) == 0
        assert prune_directory(tmp_path / "nope", 5) == 0


class TestRunRetention:
    def test_disabled_does_nothing(self, tmp_path):
        assert run_retention(RetentionConfig(enabled=False), captures_dir=str(tmp_path)) == {}

    def test_runs_enabled_pruners_and_audits(self, tmp_path):
        cap = tmp_path / "vision.jpg"
        cap.write_bytes(b"x")
        _age(cap, 30)
        zip_file = tmp_path / "discord_inbound_zips" / "m_0_a" / "evil.txt"
        zip_file.parent.mkdir(parents=True)
        zip_file.write_text("untrusted")
        _age(zip_file, 5)
        retention = RetentionConfig(
            memory_days=90, request_tracker_days=0, checkpoints_days=7, graph_memory_days=0
        )
        with (
            patch("missy.memory.sqlite_store.SQLiteMemoryStore") as store_cls,
            patch("missy.agent.checkpoint.CheckpointManager") as cp_cls,
            patch("missy.core.events.event_bus.publish") as publish,
        ):
            store_cls.return_value.cleanup.return_value = 4
            cp_cls.return_value.cleanup.return_value = 2
            results = run_retention(
                retention,
                captures_dir=str(tmp_path),
                devices_path=str(tmp_path / "devices.json"),
            )
        store_cls.return_value.cleanup.assert_called_once_with(
            older_than_days=90, include_summaries=True
        )
        assert results["memory_turns"] == 4
        assert results["checkpoints"] == 2
        assert results["capture_files"] == 1
        assert results["inbound_attachment_files"] == 1
        assert not zip_file.exists()
        assert publish.call_args.args[0].event_type == "maintenance.retention"

    def test_failing_pruner_does_not_stop_others(self, tmp_path):
        with patch("missy.agent.checkpoint.CheckpointManager", side_effect=RuntimeError("x")):
            results = run_retention(
                RetentionConfig(
                    captures_days=0, inbound_attachments_days=0, request_tracker_days=0
                ),
                captures_dir=str(tmp_path),
                devices_path=str(tmp_path / "devices.json"),
            )
        assert str(results["checkpoints"]).startswith("error")


def test_register_uses_fresh_config_each_run():
    scheduler = MagicMock()
    loader = MagicMock(return_value=SimpleNamespace(retention=RetentionConfig(enabled=False)))
    assert register_maintenance_job(scheduler, loader) is True
    kwargs = scheduler.add_job.call_args.kwargs
    assert kwargs["id"] == MAINTENANCE_JOB_ID and kwargs["trigger"] == "cron"
    kwargs["func"]()
    kwargs["func"]()
    assert loader.call_count == 2


def test_voice_step_expires_pending_pairings(tmp_path):
    from missy.channels.voice.pairing import PairingManager
    from missy.channels.voice.registry import DeviceRegistry

    devices = tmp_path / "devices.json"
    reg = DeviceRegistry(str(devices))
    reg.load()
    node_id = PairingManager(reg).initiate_pairing("", "pi", "den", "1.2.3.4", {})
    reg.update_node(node_id, requested_at=time.time() - 3 * 86400)
    results = run_retention(
        RetentionConfig(
            checkpoints_days=0,
            captures_days=0,
            inbound_attachments_days=0,
            request_tracker_days=0,
            pending_pairing_hours=24,
        ),
        captures_dir=str(tmp_path),
        devices_path=str(devices),
    )
    assert results["voice_audio_and_pending_pairings"] == 1
    reg2 = DeviceRegistry(str(devices))
    reg2.load()
    assert reg2.get_node(node_id) is None
