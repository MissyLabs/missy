"""Coverage-gap tests for missy/channels/voice/registry.py.

Targets uncovered lines:
  191-195  : save() — os.replace fails, temp-file cleanup, exception re-raised
  520      : purge_audio_logs() — entry.is_file() returns False (directory entry skipped)
  523-524  : purge_audio_logs() — entry.stat() raises OSError (file skipped)
  530-531  : purge_audio_logs() — entry.unlink() raises OSError (warning logged, continue)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.channels.voice.registry import DeviceRegistry, EdgeNode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry(tmp_path: Path) -> DeviceRegistry:
    """Fresh DeviceRegistry backed by a temp file."""
    reg = DeviceRegistry(registry_path=str(tmp_path / "devices.json"))
    reg.load()
    return reg


@pytest.fixture()
def sample_node() -> EdgeNode:
    return EdgeNode(
        node_id="node-1",
        friendly_name="Living Room",
        room="Living Room",
        ip_address="192.168.1.10",
    )


def _audio_node(
    node_id: str,
    log_dir: str,
    retention_days: int = 7,
) -> EdgeNode:
    """Return an EdgeNode with audio_logging enabled at *log_dir*."""
    return EdgeNode(
        node_id=node_id,
        friendly_name=f"Node {node_id}",
        room="Room",
        ip_address="10.0.0.1",
        audio_logging=True,
        audio_log_dir=log_dir,
        audio_log_retention_days=retention_days,
    )


# ---------------------------------------------------------------------------
# Lines 191-195: save() atomic write failure — temp file cleanup
# ---------------------------------------------------------------------------


class TestSaveAtomicWriteFailure:
    """Lines 191-195: when os.replace raises, the temp file is cleaned up and the error re-raised."""

    def test_replace_failure_reraises_exception(self, registry: DeviceRegistry, sample_node: EdgeNode):
        """An OSError from os.replace propagates to the caller after temp-file cleanup."""
        registry.add_node(sample_node)  # Populates _nodes but save() is called inside.

        # We need to force a second save() to hit the failure path.
        with patch("os.replace", side_effect=OSError("replace failed")), pytest.raises(OSError, match="replace failed"):
            registry.save()

    def test_replace_failure_cleans_up_temp_file(self, tmp_path: Path):
        """After os.replace raises, the temporary file is deleted."""
        reg = DeviceRegistry(registry_path=str(tmp_path / "dev.json"))
        reg.load()
        reg._nodes["n1"] = EdgeNode(
            node_id="n1",
            friendly_name="N",
            room="R",
            ip_address="1.1.1.1",
        )

        temp_files_created: list[str] = []
        real_mkstemp = __import__("tempfile").mkstemp

        def _capturing_mkstemp(**kwargs):
            fd, path = real_mkstemp(**kwargs)
            temp_files_created.append(path)
            return fd, path

        with (
            patch("tempfile.mkstemp", side_effect=_capturing_mkstemp),
            patch("os.replace", side_effect=OSError("atomic rename failed")),pytest.raises(OSError)
        ):
            reg.save()

        # Every temp file created during the failing save must no longer exist.
        for tmp in temp_files_created:
            assert not os.path.exists(tmp), f"Temp file not cleaned up: {tmp}"

    def test_replace_failure_suppress_unlink_oserror(self, tmp_path: Path):
        """If os.unlink also raises during cleanup, no additional exception propagates."""
        reg = DeviceRegistry(registry_path=str(tmp_path / "dev2.json"))
        reg.load()
        reg._nodes["n2"] = EdgeNode(
            node_id="n2",
            friendly_name="N2",
            room="R",
            ip_address="2.2.2.2",
        )

        with (
            patch("os.replace", side_effect=OSError("rename failed")),
            patch("os.unlink", side_effect=OSError("unlink also failed")),
            pytest.raises(OSError, match="rename failed"),
        ):
            reg.save()

    def test_successful_save_does_not_raise(self, registry: DeviceRegistry, sample_node: EdgeNode):
        """Baseline: a normal save() completes without raising and persists data."""
        registry._nodes["node-1"] = sample_node
        registry.save()  # Must not raise.

        reg2 = DeviceRegistry(registry_path=str(registry._path))
        reg2.load()
        assert reg2.get_node("node-1") is not None

    def test_save_failure_does_not_corrupt_existing_file(self, tmp_path: Path):
        """When save fails, the original registry file is left untouched."""
        reg = DeviceRegistry(registry_path=str(tmp_path / "dev3.json"))
        reg.load()
        original_node = EdgeNode(
            node_id="orig",
            friendly_name="Original",
            room="R",
            ip_address="1.2.3.4",
        )
        reg._nodes["orig"] = original_node
        reg.save()  # First successful save.

        # Read back original content.
        original_content = (tmp_path / "dev3.json").read_text()

        # Now force a failure.
        reg._nodes["new"] = EdgeNode(
            node_id="new",
            friendly_name="New",
            room="R",
            ip_address="5.6.7.8",
        )
        with patch("os.replace", side_effect=OSError("failed")), pytest.raises(OSError):
            reg.save()

        # Original file must be unchanged.
        assert (tmp_path / "dev3.json").read_text() == original_content


# ---------------------------------------------------------------------------
# Line 520: purge_audio_logs() — non-file entries are skipped
# ---------------------------------------------------------------------------


class TestPurgeAudioLogsNonFileEntry:
    """Line 520: entries that are not regular files are skipped by purge_audio_logs."""

    def test_subdirectory_inside_log_dir_is_skipped(self, registry: DeviceRegistry, tmp_path: Path):
        """A subdirectory inside the audio_log_dir is iterated but not deleted."""
        log_dir = tmp_path / "audio"
        log_dir.mkdir()

        # Create a subdirectory (not a file).
        sub = log_dir / "subdir"
        sub.mkdir()

        # Also create an old regular file that will be deleted.
        old_file = log_dir / "old.wav"
        old_file.write_bytes(b"audio")
        old_time = time.time() - 86_400 * 30
        os.utime(old_file, (old_time, old_time))

        node = _audio_node("n-subdir", str(log_dir), retention_days=7)
        registry.add_node(node)

        deleted = registry.purge_audio_logs()

        assert deleted == 1  # Only the file, not the directory.
        assert sub.exists()  # Directory must still be present.
        assert not old_file.exists()

    def test_symlink_to_dir_not_counted_as_file(self, registry: DeviceRegistry, tmp_path: Path):
        """A symlink that points to a directory is not treated as a regular file."""
        log_dir = tmp_path / "audio_sym"
        log_dir.mkdir()
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        link = log_dir / "link_to_dir"
        link.symlink_to(target_dir)

        # Also add a genuinely old regular file.
        old_file = log_dir / "old.wav"
        old_file.write_bytes(b"data")
        old_time = time.time() - 86_400 * 30
        os.utime(old_file, (old_time, old_time))

        node = _audio_node("n-symlink", str(log_dir), retention_days=7)
        registry.add_node(node)

        deleted = registry.purge_audio_logs()

        # Exactly the regular file is deleted; the symlink-to-dir is not.
        assert deleted == 1
        assert link.exists()

    def test_only_files_not_dirs_are_candidates_for_deletion(self, registry: DeviceRegistry, tmp_path: Path):
        """With only directories in log_dir, purge deletes nothing."""
        log_dir = tmp_path / "only_dirs"
        log_dir.mkdir()
        (log_dir / "subdir1").mkdir()
        (log_dir / "subdir2").mkdir()

        node = _audio_node("n-dirs", str(log_dir), retention_days=0)
        registry.add_node(node)

        deleted = registry.purge_audio_logs()
        assert deleted == 0


# ---------------------------------------------------------------------------
# Lines 523-524: purge_audio_logs() — stat() raises OSError
# ---------------------------------------------------------------------------


class TestPurgeAudioLogsStatError:
    """Lines 523-524: when entry.stat() raises OSError the entry is skipped (continue)."""

    def test_stat_oserror_skips_entry_without_raising(self, registry: DeviceRegistry, tmp_path: Path):
        """An OSError from entry.stat() causes that entry to be skipped; purge continues."""
        log_dir = tmp_path / "audio_stat_err"
        log_dir.mkdir()

        # Create a real old file for the "good" entry that will be deleted.
        old_file = log_dir / "old.wav"
        old_file.write_bytes(b"data")
        old_time = time.time() - 86_400 * 30
        os.utime(old_file, (old_time, old_time))

        node = _audio_node("n-stat-err", str(log_dir), retention_days=7)
        registry.add_node(node)

        # Build a fake entry whose is_file() returns True but stat() raises OSError.
        broken_entry = MagicMock(spec=Path)
        broken_entry.is_file.return_value = True
        broken_entry.stat.side_effect = OSError("permission denied")
        broken_entry.name = "broken.wav"

        # Wrap the real old_file entry so stat() works normally.
        real_old_entry = old_file  # actual Path; stat() and unlink() work as-is.

        real_iterdir = Path.iterdir

        def _patched_iterdir(self_path):
            if self_path == log_dir:
                return iter([broken_entry, real_old_entry])
            return real_iterdir(self_path)

        with patch.object(Path, "iterdir", _patched_iterdir):
            deleted = registry.purge_audio_logs()

        # broken_entry stat failed — skipped; old_file is deleted normally.
        assert deleted == 1
        assert not old_file.exists()

    def test_stat_oserror_on_all_entries_returns_zero(self, registry: DeviceRegistry, tmp_path: Path):
        """If every entry's stat raises, purge returns 0 without raising."""
        log_dir = tmp_path / "audio_all_stat_err"
        log_dir.mkdir()

        node = _audio_node("n-all-stat", str(log_dir), retention_days=0)
        registry.add_node(node)

        # Build two fake entries: both is_file()=True, both stat() raise OSError.
        def _make_broken(name: str) -> MagicMock:
            e = MagicMock(spec=Path)
            e.is_file.return_value = True
            e.stat.side_effect = OSError("permission denied")
            e.name = name
            return e

        fake_entries = [_make_broken("a.wav"), _make_broken("b.wav")]

        real_iterdir = Path.iterdir

        def _patched_iterdir(self_path):
            if self_path == log_dir:
                return iter(fake_entries)
            return real_iterdir(self_path)

        with patch.object(Path, "iterdir", _patched_iterdir):
            deleted = registry.purge_audio_logs()

        assert deleted == 0


# ---------------------------------------------------------------------------
# Lines 530-531: purge_audio_logs() — unlink() raises OSError
# ---------------------------------------------------------------------------


class TestPurgeAudioLogsUnlinkError:
    """Lines 530-531: when entry.unlink() raises OSError a warning is logged and purge continues."""

    def test_unlink_oserror_does_not_propagate(self, registry: DeviceRegistry, tmp_path: Path):
        """An OSError from unlink is swallowed; purge returns normally."""
        log_dir = tmp_path / "audio_unlink_err"
        log_dir.mkdir()

        old_file = log_dir / "locked.wav"
        old_file.write_bytes(b"data")
        old_time = time.time() - 86_400 * 30
        os.utime(old_file, (old_time, old_time))

        node = _audio_node("n-unlink", str(log_dir), retention_days=7)
        registry.add_node(node)

        with patch.object(Path, "unlink", side_effect=OSError("locked")):
            # Must not raise.
            deleted = registry.purge_audio_logs()

        # File could not be deleted but count is not incremented.
        assert deleted == 0

    def test_unlink_oserror_logs_warning(self, registry: DeviceRegistry, tmp_path: Path):
        """A warning log entry is emitted when unlink fails."""
        log_dir = tmp_path / "audio_warn"
        log_dir.mkdir()

        old_file = log_dir / "warn.wav"
        old_file.write_bytes(b"data")
        old_time = time.time() - 86_400 * 30
        os.utime(old_file, (old_time, old_time))

        node = _audio_node("n-warn", str(log_dir), retention_days=7)
        registry.add_node(node)

        with (
            patch.object(Path, "unlink", side_effect=OSError("locked")),
            patch("missy.channels.voice.registry.logger") as mock_log,
        ):
            registry.purge_audio_logs()

        warning_calls = [str(c) for c in mock_log.warning.call_args_list]
        assert any("could not delete" in w or "warn.wav" in w for w in warning_calls)

    def test_unlink_error_on_one_file_does_not_prevent_deleting_others(
        self, registry: DeviceRegistry, tmp_path: Path
    ):
        """If one unlink fails, subsequent eligible files in the same dir are still attempted."""
        log_dir = tmp_path / "audio_partial"
        log_dir.mkdir()

        # Two old files: first will fail to unlink, second should succeed.
        file_a = log_dir / "a.wav"
        file_b = log_dir / "b.wav"
        for f in (file_a, file_b):
            f.write_bytes(b"data")
            old_time = time.time() - 86_400 * 30
            os.utime(f, (old_time, old_time))

        node = _audio_node("n-partial", str(log_dir), retention_days=7)
        registry.add_node(node)

        real_unlink = Path.unlink
        call_count = [0]

        def _selective_unlink(self, *args, **kwargs):
            call_count[0] += 1
            if self.name == "a.wav":
                raise OSError("locked")
            return real_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", _selective_unlink):
            deleted = registry.purge_audio_logs()

        # b.wav succeeded (1 deleted); a.wav failed (0 for that one).
        assert deleted == 1
        assert file_b.exists() is False
        assert file_a.exists() is True


# ---------------------------------------------------------------------------
# Additional integration tests to ensure full path coverage
# ---------------------------------------------------------------------------


class TestPurgeAudioLogsIntegration:
    """Integrated tests that exercise the full purge_audio_logs path."""

    def test_purge_deletes_old_files_and_keeps_recent(
        self, registry: DeviceRegistry, tmp_path: Path
    ):
        """Files older than retention are deleted; recent files are kept."""
        log_dir = tmp_path / "mixed"
        log_dir.mkdir()

        old = log_dir / "old.wav"
        old.write_bytes(b"old")
        os.utime(old, (time.time() - 86_400 * 10, time.time() - 86_400 * 10))

        recent = log_dir / "recent.wav"
        recent.write_bytes(b"new")

        node = _audio_node("n-mixed", str(log_dir), retention_days=7)
        registry.add_node(node)

        deleted = registry.purge_audio_logs()

        assert deleted == 1
        assert not old.exists()
        assert recent.exists()

    def test_purge_returns_zero_for_empty_log_dir(
        self, registry: DeviceRegistry, tmp_path: Path
    ):
        """Empty audio_log_dir causes purge to return 0."""
        log_dir = tmp_path / "empty_logs"
        log_dir.mkdir()

        node = _audio_node("n-empty", str(log_dir))
        registry.add_node(node)

        assert registry.purge_audio_logs() == 0

    def test_purge_skips_node_with_audio_logging_false(
        self, registry: DeviceRegistry, tmp_path: Path
    ):
        """Nodes with audio_logging=False are not examined during purge."""
        log_dir = tmp_path / "no_logging"
        log_dir.mkdir()

        old = log_dir / "old.wav"
        old.write_bytes(b"data")
        os.utime(old, (time.time() - 86_400 * 30, time.time() - 86_400 * 30))

        node = EdgeNode(
            node_id="n-no-log",
            friendly_name="No Log",
            room="R",
            ip_address="1.1.1.1",
            audio_logging=False,  # Disabled.
            audio_log_dir=str(log_dir),
            audio_log_retention_days=0,
        )
        registry.add_node(node)

        deleted = registry.purge_audio_logs()
        assert deleted == 0
        assert old.exists()

    def test_purge_missing_dir_returns_zero(self, registry: DeviceRegistry):
        """purge_audio_logs returns 0 when audio_log_dir does not exist."""
        node = _audio_node("n-missing", "/nonexistent/path/that/does/not/exist")
        registry.add_node(node)

        assert registry.purge_audio_logs() == 0

    def test_purge_multiple_nodes_accumulates_count(
        self, registry: DeviceRegistry, tmp_path: Path
    ):
        """Deleted file count is summed across all audio-logging nodes."""
        for i in range(3):
            log_dir = tmp_path / f"node{i}"
            log_dir.mkdir()
            old = log_dir / "old.wav"
            old.write_bytes(b"data")
            os.utime(old, (time.time() - 86_400 * 30, time.time() - 86_400 * 30))
            registry.add_node(_audio_node(f"n{i}", str(log_dir), retention_days=7))

        deleted = registry.purge_audio_logs()
        assert deleted == 3
