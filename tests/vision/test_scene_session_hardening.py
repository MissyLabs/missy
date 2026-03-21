"""Tests for session 9 hardening: thread safety, capture stats, path validation.

Covers:
- SceneSession thread safety (add_frame, add_observation, update_state, close, summarize)
- CameraHandle capture_count tracking on both success and failure
- PhotoSource directory traversal prevention
- TrustScorer thread safety
- Vault key validation (zero-filled, permissive permissions)
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SceneSession thread safety
# ---------------------------------------------------------------------------


class TestSceneSessionThreadSafety:
    """Verify SceneSession is safe under concurrent access."""

    def _make_frame(self, val: int = 128) -> np.ndarray:
        """Create a small test frame with a unique pixel value."""
        frame = np.full((8, 8, 3), val, dtype=np.uint8)
        return frame

    def test_concurrent_add_frame(self):
        """Multiple threads adding frames should not corrupt state."""
        from missy.vision.scene_memory import SceneSession, TaskType

        session = SceneSession("test-concurrent", TaskType.GENERAL, max_frames=200)
        errors: list[Exception] = []

        def add_frames(start_val: int, count: int):
            try:
                for i in range(count):
                    session.add_frame(
                        self._make_frame(start_val + i),
                        source=f"thread-{start_val}",
                        deduplicate=False,  # avoid dedup interfering
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_frames, args=(i * 50, 20)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent add_frame: {errors}"
        assert session.frame_count == 80  # 4 threads × 20 frames

    def test_concurrent_add_observation(self):
        """Multiple threads adding observations should not lose data."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("obs-test")
        n_per_thread = 50

        def add_obs(thread_id: int):
            for i in range(n_per_thread):
                session.add_observation(f"t{thread_id}-obs{i}")

        threads = [threading.Thread(target=add_obs, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(session.observations) == 4 * n_per_thread

    def test_concurrent_update_state(self):
        """Concurrent state updates should not crash."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("state-test")
        errors: list[Exception] = []

        def update(thread_id: int):
            try:
                for i in range(50):
                    session.update_state(**{f"key_{thread_id}_{i}": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        state = session.state
        assert len(state) == 200  # 4 threads × 50 keys

    def test_concurrent_close_and_read(self):
        """Closing while reading should not raise."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("close-test")
        session.add_frame(self._make_frame(), deduplicate=False)
        session.add_observation("note")
        errors: list[Exception] = []

        def read_loop():
            try:
                for _ in range(100):
                    session.get_latest_frame()
                    session.get_recent_frames()
                    session.summarize()
            except Exception as e:
                errors.append(e)

        def close_session():
            try:
                session.close()
            except Exception as e:
                errors.append(e)

        t_read = threading.Thread(target=read_loop)
        t_close = threading.Thread(target=close_session)
        t_read.start()
        t_close.start()
        t_read.join(timeout=10)
        t_close.join(timeout=10)

        assert not errors

    def test_get_frame_thread_safe(self):
        """get_frame under concurrent add_frame should not crash."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("getframe-test", max_frames=200)
        errors: list[Exception] = []

        def add_loop():
            try:
                for _ in range(50):
                    session.add_frame(self._make_frame(), deduplicate=False)
            except Exception as e:
                errors.append(e)

        def get_loop():
            try:
                for i in range(1, 51):
                    session.get_frame(i)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_loop)
        t2 = threading.Thread(target=get_loop)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors


# ---------------------------------------------------------------------------
# CameraHandle capture_count tracking
# ---------------------------------------------------------------------------


class TestCaptureCountTracking:
    """Verify _capture_count increments on both success and failure."""

    def test_capture_count_increments_on_success(self):
        """Successful capture should increment both capture_count and success_count."""
        from missy.vision.capture import CameraHandle, CaptureConfig

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=0))
        # Mock the internals
        mock_cap = MagicMock()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, frame)
        handle._cap = mock_cap
        handle._opened = True

        result = handle.capture()
        assert result.success
        assert handle._capture_count == 1
        assert handle._success_count == 1

    def test_capture_count_increments_on_failure(self):
        """Failed capture should increment capture_count but not success_count."""
        from missy.vision.capture import CameraHandle, CaptureConfig

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=0, max_retries=1))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        handle._cap = mock_cap
        handle._opened = True

        result = handle.capture()
        assert not result.success
        assert handle._capture_count == 1
        assert handle._success_count == 0

    def test_success_rate_after_mixed_captures(self):
        """Success rate should accurately reflect success/total ratio."""
        from missy.vision.capture import CameraHandle, CaptureConfig

        handle = CameraHandle("/dev/video0", CaptureConfig(warmup_frames=0, max_retries=1))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        handle._cap = mock_cap
        handle._opened = True

        # 2 successes
        mock_cap.read.return_value = (True, frame)
        handle.capture()
        handle.capture()

        # 1 failure
        mock_cap.read.return_value = (False, None)
        handle.capture()

        stats = handle.capture_stats
        assert stats["capture_count"] == 3
        assert stats["success_count"] == 2
        assert abs(stats["success_rate"] - 2 / 3) < 0.001


# ---------------------------------------------------------------------------
# PhotoSource path validation
# ---------------------------------------------------------------------------


class TestPhotoSourcePathValidation:
    """Verify PhotoSource prevents directory traversal."""

    def test_resolves_directory(self):
        """PhotoSource should resolve the directory to an absolute path."""
        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a relative symlink
            source = PhotoSource(tmpdir)
            assert source._directory == Path(tmpdir).resolve()

    def test_scan_filters_symlinks_outside_directory(self):
        """Files that resolve outside the scan directory should be excluded."""
        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as outside:
            # Create a real image file outside
            outside_img = Path(outside) / "secret.jpg"
            outside_img.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

            # Create a symlink inside tmpdir pointing outside
            link = Path(tmpdir) / "traversal.jpg"
            link.symlink_to(outside_img)

            # Also create a legitimate file
            legit = Path(tmpdir) / "legit.jpg"
            legit.write_bytes(b"\xff\xd8\xff\xe0")

            source = PhotoSource(tmpdir)
            files = source.scan()

            # The symlink pointing outside should be excluded
            file_names = [f.name for f in files]
            assert "legit.jpg" in file_names
            assert "traversal.jpg" not in file_names

    def test_scan_allows_files_inside_directory(self):
        """Legitimate files within the directory should be included."""
        from missy.vision.sources import PhotoSource

        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("a.jpg", "b.png", "c.bmp"):
                (Path(tmpdir) / name).write_bytes(b"\x00" * 10)

            source = PhotoSource(tmpdir)
            files = source.scan()
            assert len(files) == 3


# ---------------------------------------------------------------------------
# TrustScorer thread safety
# ---------------------------------------------------------------------------


class TestTrustScorerThreadSafety:
    """Verify TrustScorer operations are thread-safe."""

    def test_concurrent_record_success(self):
        """Many threads recording success should produce consistent scores."""
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        n_threads = 8
        n_per_thread = 100

        def record(thread_id: int):
            for _ in range(n_per_thread):
                scorer.record_success(f"entity-{thread_id}")

        threads = [threading.Thread(target=record, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for i in range(n_threads):
            score = scorer.score(f"entity-{i}")
            # 500 (default) + 100 * 10 = 1500, capped at 1000
            assert score == 1000

    def test_concurrent_mixed_operations(self):
        """Mixed success/failure/violation from multiple threads should not crash."""
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()
        errors: list[Exception] = []

        def success_loop():
            try:
                for _ in range(100):
                    scorer.record_success("shared")
            except Exception as e:
                errors.append(e)

        def failure_loop():
            try:
                for _ in range(100):
                    scorer.record_failure("shared")
            except Exception as e:
                errors.append(e)

        def violation_loop():
            try:
                for _ in range(50):
                    scorer.record_violation("shared")
            except Exception as e:
                errors.append(e)

        def read_loop():
            try:
                for _ in range(200):
                    scorer.score("shared")
                    scorer.is_trusted("shared")
                    scorer.get_scores()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=success_loop),
            threading.Thread(target=failure_loop),
            threading.Thread(target=violation_loop),
            threading.Thread(target=read_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_score_bounds_under_contention(self):
        """Score should never go below 0 or above 1000 even under contention."""
        from missy.security.trust import TrustScorer

        scorer = TrustScorer()

        def hammer_down():
            for _ in range(200):
                scorer.record_violation("bounded", weight=999)

        def hammer_up():
            for _ in range(200):
                scorer.record_success("bounded", weight=999)

        threads = [
            threading.Thread(target=hammer_down),
            threading.Thread(target=hammer_up),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        score = scorer.score("bounded")
        assert 0 <= score <= 1000


# ---------------------------------------------------------------------------
# Vault key validation
# ---------------------------------------------------------------------------


class TestVaultKeyValidation:
    """Verify vault key file validation."""

    def test_zero_filled_key_rejected(self):
        """A key file containing all zeros should be rejected."""
        from missy.security.vault import VaultError

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "vault.key"
            key_path.write_bytes(b"\x00" * 32)
            os.chmod(str(key_path), 0o600)

            try:
                from missy.security.vault import Vault

                with pytest.raises(VaultError, match="all zeros"):
                    Vault(vault_dir=tmpdir)
            except ImportError:
                pytest.skip("cryptography not installed")

    def test_wrong_length_key_rejected(self):
        """A key file with wrong length should be rejected."""
        from missy.security.vault import VaultError

        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "vault.key"
            key_path.write_bytes(b"\x01" * 16)  # 16 bytes instead of 32
            os.chmod(str(key_path), 0o600)

            try:
                from missy.security.vault import Vault

                with pytest.raises(VaultError, match="32 bytes"):
                    Vault(vault_dir=tmpdir)
            except ImportError:
                pytest.skip("cryptography not installed")

    def test_permissive_key_warns(self, caplog):
        """A key file with world-readable perms should log a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = Path(tmpdir) / "vault.key"
            key_path.write_bytes(os.urandom(32))
            os.chmod(str(key_path), 0o644)  # world-readable

            try:
                import logging

                from missy.security.vault import Vault

                with caplog.at_level(logging.WARNING):
                    vault = Vault(vault_dir=tmpdir)
                    assert vault is not None
                assert any("permissive" in r.message.lower() for r in caplog.records)
            except ImportError:
                pytest.skip("cryptography not installed")

    def test_symlink_key_rejected(self):
        """A symlink key file should be rejected."""
        from missy.security.vault import VaultError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create real key elsewhere
            real_key = Path(tmpdir) / "real.key"
            real_key.write_bytes(os.urandom(32))

            # Create vault dir with symlinked key
            vault_dir = Path(tmpdir) / "vault"
            vault_dir.mkdir()
            (vault_dir / "vault.key").symlink_to(real_key)

            try:
                from missy.security.vault import Vault

                with pytest.raises(VaultError, match="symlink"):
                    Vault(vault_dir=str(vault_dir))
            except ImportError:
                pytest.skip("cryptography not installed")


# ---------------------------------------------------------------------------
# SceneSession deduplication edge cases
# ---------------------------------------------------------------------------


class TestSceneSessionDedup:
    """Edge cases in deduplication under the new locking scheme."""

    def test_dedup_with_identical_frames(self):
        """Identical frames should be deduplicated."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("dedup-test")
        frame = np.ones((8, 8, 3), dtype=np.uint8) * 128

        result1 = session.add_frame(frame.copy(), deduplicate=True)
        assert result1 is not None

        result2 = session.add_frame(frame.copy(), deduplicate=True)
        assert result2 is None  # deduplicated

        assert session.frame_count == 1

    def test_dedup_disabled(self):
        """With deduplicate=False, identical frames should both be stored."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("nodedup-test")
        frame = np.ones((8, 8, 3), dtype=np.uint8) * 128

        session.add_frame(frame.copy(), deduplicate=False)
        session.add_frame(frame.copy(), deduplicate=False)

        assert session.frame_count == 2

    def test_frame_id_monotonic_despite_dedup(self):
        """Frame IDs should be monotonically increasing even after dedup."""
        from missy.vision.scene_memory import SceneSession

        session = SceneSession("mono-test")
        frame = np.ones((8, 8, 3), dtype=np.uint8) * 128

        f1 = session.add_frame(frame.copy(), deduplicate=True)
        session.add_frame(frame.copy(), deduplicate=True)  # deduplicated
        f3 = session.add_frame(np.ones((8, 8, 3), dtype=np.uint8) * 0, deduplicate=True)

        assert f1.frame_id < f3.frame_id


# ---------------------------------------------------------------------------
# CaptureConfig timeout edge cases
# ---------------------------------------------------------------------------


class TestCaptureTimeout:
    """Verify capture timeout behavior."""

    def test_capture_respects_timeout(self):
        """Capture should fail if all retries take longer than timeout."""

        from missy.vision.capture import CameraHandle, CaptureConfig

        config = CaptureConfig(
            warmup_frames=0,
            max_retries=10,
            timeout_seconds=0.01,  # very short timeout
            retry_delay=0.1,
        )
        handle = CameraHandle("/dev/video0", config)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        handle._cap = mock_cap
        handle._opened = True

        result = handle.capture()
        assert not result.success
        assert "timed out" in result.error.lower() or result.attempt_count < 10
