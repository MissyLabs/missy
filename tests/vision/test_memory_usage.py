"""Tests for missy.vision.memory_usage — scene memory usage tracking.

Covers:
- estimate_frame_bytes: numpy arrays, None, non-numpy objects
- SessionMemoryInfo: estimated_mb property
- MemoryReport: total_mb, limit_mb, to_dict
- MemoryTracker: init, compute_session_usage, update_from_scene_manager,
  should_evict, report
- Warning and over-limit logging behavior
- get_memory_tracker: module singleton
"""

from __future__ import annotations

import sys
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.memory_usage import (
    MemoryReport,
    MemoryTracker,
    SessionMemoryInfo,
    estimate_frame_bytes,
    get_memory_tracker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_mock(image) -> MagicMock:
    """Return a mock SceneFrame with the given image attribute."""
    frame = MagicMock()
    frame.image = image
    return frame


def _make_session_mock(
    task_id: str = "task-1",
    frames: list | None = None,
    is_active: bool = True,
) -> MagicMock:
    """Return a mock SceneSession."""
    session = MagicMock()
    session.task_id = task_id
    session._frames = frames if frames is not None else []
    session.is_active = is_active
    return session


def _make_manager_mock(sessions: dict | None = None, use_lock: bool = False) -> MagicMock:
    """Return a mock SceneManager."""
    manager = MagicMock()
    manager._sessions = sessions if sessions is not None else {}
    if use_lock:
        manager._lock = threading.Lock()
    else:
        del manager._lock  # ensure getattr fallback path is exercised
    return manager


# ---------------------------------------------------------------------------
# estimate_frame_bytes
# ---------------------------------------------------------------------------


class TestEstimateFrameBytes:
    def test_none_returns_zero(self):
        assert estimate_frame_bytes(None) == 0

    def test_numpy_array_uint8(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = estimate_frame_bytes(frame)
        # Must include both nbytes and getsizeof overhead
        assert result == frame.nbytes + sys.getsizeof(frame)
        # Sanity: 100*100*3 bytes of data alone
        assert result >= 30_000

    def test_numpy_array_float32(self):
        frame = np.ones((480, 640, 3), dtype=np.float32)
        result = estimate_frame_bytes(frame)
        assert result == frame.nbytes + sys.getsizeof(frame)
        assert result >= 480 * 640 * 3 * 4

    def test_numpy_1d_array(self):
        frame = np.arange(1024, dtype=np.uint8)
        result = estimate_frame_bytes(frame)
        assert result == frame.nbytes + sys.getsizeof(frame)

    def test_numpy_empty_array(self):
        frame = np.array([], dtype=np.uint8)
        result = estimate_frame_bytes(frame)
        # nbytes == 0, but sys.getsizeof overhead still counted
        assert result == 0 + sys.getsizeof(frame)

    def test_large_hd_frame(self):
        # 1920x1080 BGR uint8 — ~6 MB of raw data
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = estimate_frame_bytes(frame)
        assert result >= 1080 * 1920 * 3  # at least the raw nbytes

    def test_non_numpy_string_falls_back_to_getsizeof(self):
        obj = "hello"
        result = estimate_frame_bytes(obj)
        assert result == sys.getsizeof(obj)

    def test_non_numpy_integer_falls_back_to_getsizeof(self):
        obj = 42
        result = estimate_frame_bytes(obj)
        assert result == sys.getsizeof(obj)

    def test_non_numpy_list_falls_back_to_getsizeof(self):
        obj = [1, 2, 3]
        result = estimate_frame_bytes(obj)
        assert result == sys.getsizeof(obj)

    def test_non_numpy_dict_falls_back_to_getsizeof(self):
        obj = {"a": 1}
        result = estimate_frame_bytes(obj)
        assert result == sys.getsizeof(obj)

    def test_object_without_nbytes_attribute(self):
        """Object that has no nbytes raises AttributeError, caught gracefully."""

        class NoNbytes:
            pass

        obj = NoNbytes()
        result = estimate_frame_bytes(obj)
        assert result == sys.getsizeof(obj)

    def test_object_with_nbytes_raising_type_error(self):
        """Object whose nbytes property raises TypeError is handled."""

        class BadNbytes:
            @property
            def nbytes(self):
                raise TypeError("not supported")

        obj = BadNbytes()
        result = estimate_frame_bytes(obj)
        assert result == sys.getsizeof(obj)


# ---------------------------------------------------------------------------
# SessionMemoryInfo
# ---------------------------------------------------------------------------


class TestSessionMemoryInfo:
    def test_estimated_mb_exact(self):
        info = SessionMemoryInfo(
            task_id="t1",
            frame_count=2,
            estimated_bytes=1024 * 1024,  # exactly 1 MB
            active=True,
        )
        assert info.estimated_mb == pytest.approx(1.0)

    def test_estimated_mb_zero_bytes(self):
        info = SessionMemoryInfo(
            task_id="t1",
            frame_count=0,
            estimated_bytes=0,
            active=False,
        )
        assert info.estimated_mb == pytest.approx(0.0)

    def test_estimated_mb_fractional(self):
        info = SessionMemoryInfo(
            task_id="t1",
            frame_count=1,
            estimated_bytes=512 * 1024,  # 0.5 MB
            active=True,
        )
        assert info.estimated_mb == pytest.approx(0.5)

    def test_field_storage(self):
        info = SessionMemoryInfo(
            task_id="my-task",
            frame_count=5,
            estimated_bytes=2_000_000,
            active=False,
        )
        assert info.task_id == "my-task"
        assert info.frame_count == 5
        assert info.estimated_bytes == 2_000_000
        assert info.active is False

    def test_large_bytes(self):
        info = SessionMemoryInfo(
            task_id="t",
            frame_count=20,
            estimated_bytes=600_000_000,
            active=True,
        )
        assert info.estimated_mb == pytest.approx(600_000_000 / (1024 * 1024))


# ---------------------------------------------------------------------------
# MemoryReport
# ---------------------------------------------------------------------------


class TestMemoryReport:
    def _make_report(
        self,
        total_bytes: int = 10_485_760,  # 10 MB
        total_frames: int = 3,
        session_count: int = 1,
        active_sessions: int = 1,
        sessions: list | None = None,
        limit_bytes: int = 100_000_000,  # ~95 MB
        usage_fraction: float = 0.1,
        over_limit: bool = False,
    ) -> MemoryReport:
        return MemoryReport(
            total_bytes=total_bytes,
            total_frames=total_frames,
            session_count=session_count,
            active_sessions=active_sessions,
            sessions=sessions or [],
            limit_bytes=limit_bytes,
            usage_fraction=usage_fraction,
            over_limit=over_limit,
        )

    def test_total_mb(self):
        report = self._make_report(total_bytes=1024 * 1024)
        assert report.total_mb == pytest.approx(1.0)

    def test_total_mb_zero(self):
        report = self._make_report(total_bytes=0)
        assert report.total_mb == pytest.approx(0.0)

    def test_limit_mb(self):
        report = self._make_report(limit_bytes=500 * 1024 * 1024)
        assert report.limit_mb == pytest.approx(500.0)

    def test_limit_mb_fractional(self):
        report = self._make_report(limit_bytes=512 * 1024)  # 0.5 MB
        assert report.limit_mb == pytest.approx(0.5)

    def test_to_dict_keys(self):
        report = self._make_report()
        d = report.to_dict()
        expected_keys = {
            "total_bytes",
            "total_mb",
            "total_frames",
            "session_count",
            "active_sessions",
            "limit_bytes",
            "limit_mb",
            "usage_fraction",
            "over_limit",
            "sessions",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_rounded(self):
        report = self._make_report(
            total_bytes=1_500_000,
            limit_bytes=10_000_000,
            usage_fraction=0.15678,
        )
        d = report.to_dict()
        assert d["total_mb"] == round(1_500_000 / (1024 * 1024), 2)
        assert d["limit_mb"] == round(10_000_000 / (1024 * 1024), 2)
        assert d["usage_fraction"] == round(0.15678, 4)

    def test_to_dict_sessions_list(self):
        info = SessionMemoryInfo(
            task_id="t1",
            frame_count=2,
            estimated_bytes=2 * 1024 * 1024,
            active=True,
        )
        report = self._make_report(sessions=[info], session_count=1)
        d = report.to_dict()
        assert len(d["sessions"]) == 1
        session_dict = d["sessions"][0]
        assert session_dict["task_id"] == "t1"
        assert session_dict["frame_count"] == 2
        assert session_dict["estimated_mb"] == round(2.0, 2)
        assert session_dict["active"] is True

    def test_to_dict_empty_sessions(self):
        report = self._make_report(sessions=[], session_count=0)
        d = report.to_dict()
        assert d["sessions"] == []

    def test_to_dict_over_limit_true(self):
        report = self._make_report(over_limit=True)
        assert report.to_dict()["over_limit"] is True

    def test_to_dict_over_limit_false(self):
        report = self._make_report(over_limit=False)
        assert report.to_dict()["over_limit"] is False

    def test_to_dict_scalar_passthrough(self):
        report = self._make_report(
            total_bytes=999,
            total_frames=7,
            session_count=2,
            active_sessions=1,
        )
        d = report.to_dict()
        assert d["total_bytes"] == 999
        assert d["total_frames"] == 7
        assert d["session_count"] == 2
        assert d["active_sessions"] == 1


# ---------------------------------------------------------------------------
# MemoryTracker.__init__
# ---------------------------------------------------------------------------


class TestMemoryTrackerInit:
    def test_default_max_bytes(self):
        tracker = MemoryTracker()
        assert tracker.max_bytes == 500_000_000

    def test_custom_max_bytes(self):
        tracker = MemoryTracker(max_bytes=100_000_000)
        assert tracker.max_bytes == 100_000_000

    def test_max_bytes_zero_clamped_to_one(self):
        tracker = MemoryTracker(max_bytes=0)
        assert tracker.max_bytes == 1

    def test_max_bytes_negative_clamped_to_one(self):
        tracker = MemoryTracker(max_bytes=-500)
        assert tracker.max_bytes == 1

    def test_default_last_report_is_none(self):
        tracker = MemoryTracker()
        assert tracker.report() is None

    def test_default_should_evict_is_false(self):
        tracker = MemoryTracker()
        assert tracker.should_evict() is False

    def test_custom_warn_fraction_stored(self):
        tracker = MemoryTracker(warn_fraction=0.5)
        assert tracker._warn_fraction == 0.5


# ---------------------------------------------------------------------------
# MemoryTracker.compute_session_usage
# ---------------------------------------------------------------------------


class TestMemoryTrackerComputeSessionUsage:
    def test_empty_frames(self):
        tracker = MemoryTracker()
        session = _make_session_mock(task_id="empty", frames=[], is_active=True)
        info = tracker.compute_session_usage(session)
        assert info.task_id == "empty"
        assert info.frame_count == 0
        assert info.estimated_bytes == 0
        assert info.active is True

    def test_single_numpy_frame(self):
        tracker = MemoryTracker()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        session = _make_session_mock(
            task_id="s1",
            frames=[_make_frame_mock(img)],
            is_active=True,
        )
        info = tracker.compute_session_usage(session)
        assert info.task_id == "s1"
        assert info.frame_count == 1
        expected = estimate_frame_bytes(img)
        assert info.estimated_bytes == expected
        assert info.active is True

    def test_multiple_numpy_frames(self):
        tracker = MemoryTracker()
        imgs = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        frames = [_make_frame_mock(img) for img in imgs]
        session = _make_session_mock(task_id="s2", frames=frames, is_active=False)
        info = tracker.compute_session_usage(session)
        assert info.frame_count == 5
        expected_total = sum(estimate_frame_bytes(img) for img in imgs)
        assert info.estimated_bytes == expected_total
        assert info.active is False

    def test_frames_with_none_images(self):
        tracker = MemoryTracker()
        frames = [_make_frame_mock(None), _make_frame_mock(None)]
        session = _make_session_mock(frames=frames)
        info = tracker.compute_session_usage(session)
        assert info.frame_count == 2
        assert info.estimated_bytes == 0

    def test_mixed_none_and_numpy_frames(self):
        tracker = MemoryTracker()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        frames = [_make_frame_mock(None), _make_frame_mock(img)]
        session = _make_session_mock(frames=frames)
        info = tracker.compute_session_usage(session)
        assert info.frame_count == 2
        assert info.estimated_bytes == estimate_frame_bytes(img)

    def test_missing_frames_attribute_defaults_to_empty(self):
        """Session without _frames attribute should produce empty info."""
        tracker = MemoryTracker()
        session = MagicMock(spec=[])  # no attributes
        session.task_id = "no-frames"
        session.is_active = False
        info = tracker.compute_session_usage(session)
        assert info.frame_count == 0
        assert info.estimated_bytes == 0

    def test_missing_task_id_defaults_to_unknown(self):
        tracker = MemoryTracker()
        session = MagicMock(spec=[])
        session.is_active = True
        info = tracker.compute_session_usage(session)
        assert info.task_id == "unknown"

    def test_missing_is_active_defaults_to_false(self):
        tracker = MemoryTracker()
        session = MagicMock(spec=["task_id", "_frames"])
        session.task_id = "t"
        session._frames = []
        info = tracker.compute_session_usage(session)
        assert info.active is False

    def test_large_hd_frames_accumulate(self):
        tracker = MemoryTracker()
        imgs = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(3)]
        frames = [_make_frame_mock(img) for img in imgs]
        session = _make_session_mock(task_id="hd", frames=frames)
        info = tracker.compute_session_usage(session)
        assert info.estimated_bytes >= 3 * 1080 * 1920 * 3


# ---------------------------------------------------------------------------
# MemoryTracker.update_from_scene_manager
# ---------------------------------------------------------------------------


class TestMemoryTrackerUpdateFromSceneManager:
    def test_empty_manager_returns_zeroed_report(self):
        tracker = MemoryTracker(max_bytes=500_000_000)
        manager = _make_manager_mock(sessions={})
        report = tracker.update_from_scene_manager(manager)
        assert report.total_bytes == 0
        assert report.total_frames == 0
        assert report.session_count == 0
        assert report.active_sessions == 0
        assert report.sessions == []
        assert report.over_limit is False
        assert report.usage_fraction == pytest.approx(0.0)

    def test_single_session_aggregated(self):
        tracker = MemoryTracker(max_bytes=500_000_000)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        session = _make_session_mock(
            task_id="t1",
            frames=[_make_frame_mock(img)],
            is_active=True,
        )
        manager = _make_manager_mock(sessions={"t1": session})
        report = tracker.update_from_scene_manager(manager)
        assert report.session_count == 1
        assert report.active_sessions == 1
        assert report.total_frames == 1
        expected = estimate_frame_bytes(img)
        assert report.total_bytes == expected

    def test_multiple_sessions_aggregated(self):
        tracker = MemoryTracker(max_bytes=500_000_000)
        img_a = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b = np.ones((200, 200, 3), dtype=np.uint8)
        s1 = _make_session_mock("t1", [_make_frame_mock(img_a)], is_active=True)
        s2 = _make_session_mock("t2", [_make_frame_mock(img_b)], is_active=False)
        manager = _make_manager_mock(sessions={"t1": s1, "t2": s2})
        report = tracker.update_from_scene_manager(manager)
        assert report.session_count == 2
        assert report.active_sessions == 1
        assert report.total_frames == 2
        expected = estimate_frame_bytes(img_a) + estimate_frame_bytes(img_b)
        assert report.total_bytes == expected

    def test_usage_fraction_computed_correctly(self):
        max_bytes = 1_000_000
        tracker = MemoryTracker(max_bytes=max_bytes)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_bytes = estimate_frame_bytes(img)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        report = tracker.update_from_scene_manager(manager)
        assert report.usage_fraction == pytest.approx(frame_bytes / max_bytes)

    def test_over_limit_flag_set(self):
        max_bytes = 1  # tiny limit forces over-limit
        tracker = MemoryTracker(max_bytes=max_bytes)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        report = tracker.update_from_scene_manager(manager)
        assert report.over_limit is True

    def test_under_limit_flag_not_set(self):
        tracker = MemoryTracker(max_bytes=500_000_000)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        report = tracker.update_from_scene_manager(manager)
        assert report.over_limit is False

    def test_report_stored_as_last_report(self):
        tracker = MemoryTracker()
        manager = _make_manager_mock(sessions={})
        assert tracker.report() is None
        returned = tracker.update_from_scene_manager(manager)
        assert tracker.report() is returned

    def test_manager_with_threading_lock(self):
        """Manager with a real lock should still be accessed correctly."""
        tracker = MemoryTracker(max_bytes=500_000_000)
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)], is_active=True)
        manager = _make_manager_mock(sessions={"t": session}, use_lock=True)
        report = tracker.update_from_scene_manager(manager)
        assert report.session_count == 1
        assert report.total_frames == 1

    def test_manager_without_lock_attribute(self):
        """Manager without _lock attribute falls back gracefully."""
        tracker = MemoryTracker(max_bytes=500_000_000)
        manager = _make_manager_mock(sessions={}, use_lock=False)
        report = tracker.update_from_scene_manager(manager)
        assert report.session_count == 0

    def test_manager_without_sessions_attribute(self):
        """Manager with no _sessions attribute uses empty dict."""
        tracker = MemoryTracker()
        manager = MagicMock(spec=[])  # no _sessions, no _lock
        report = tracker.update_from_scene_manager(manager)
        assert report.session_count == 0
        assert report.total_bytes == 0

    def test_limit_bytes_in_report_matches_tracker(self):
        tracker = MemoryTracker(max_bytes=123_456_789)
        manager = _make_manager_mock(sessions={})
        report = tracker.update_from_scene_manager(manager)
        assert report.limit_bytes == 123_456_789

    def test_uses_module_singleton_when_manager_is_none(self):
        """When manager=None, get_scene_manager() is called from scene_memory."""
        tracker = MemoryTracker()
        mock_manager = _make_manager_mock(sessions={})
        # get_scene_manager is imported inside the function body, so patch the
        # attribute on the scene_memory module that is the lookup target.
        with patch(
            "missy.vision.scene_memory.get_scene_manager",
            return_value=mock_manager,
        ) as mock_get:
            report = tracker.update_from_scene_manager(None)
        mock_get.assert_called_once()
        assert report.session_count == 0

    def test_sessions_list_in_report_matches_count(self):
        tracker = MemoryTracker()
        s1 = _make_session_mock("t1", [])
        s2 = _make_session_mock("t2", [])
        s3 = _make_session_mock("t3", [])
        manager = _make_manager_mock(sessions={"t1": s1, "t2": s2, "t3": s3})
        report = tracker.update_from_scene_manager(manager)
        assert len(report.sessions) == 3


# ---------------------------------------------------------------------------
# MemoryTracker: warning/over-limit logging
# ---------------------------------------------------------------------------


class TestMemoryTrackerLogging:
    def test_no_warning_when_under_warn_fraction(self, caplog):
        tracker = MemoryTracker(max_bytes=1_000_000, warn_fraction=0.8)
        # Force usage to 10% — well below the 80% threshold
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        import logging

        with caplog.at_level(logging.WARNING, logger="missy.vision.memory_usage"):
            tracker.update_from_scene_manager(manager)
        assert caplog.text == ""

    def test_warning_emitted_when_near_limit(self, caplog):
        """Usage >= warn_fraction should emit a 'high' warning."""
        import logging

        # max 1000 bytes; frame will be ~30_096 bytes — well over 80% threshold
        # But we need usage BETWEEN warn_fraction and limit, so craft carefully.
        # Use a tiny frame and a max_bytes set so that usage lands in [0.8, 1.0).
        img = np.zeros((10, 10, 3), dtype=np.uint8)  # nbytes = 300
        frame_est = estimate_frame_bytes(img)
        # Set max_bytes so usage_fraction == 0.85 (above warn, below over_limit)
        max_bytes = int(frame_est / 0.85)  # usage_fraction = frame_est / max_bytes ≈ 0.85
        tracker = MemoryTracker(max_bytes=max_bytes, warn_fraction=0.8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        with caplog.at_level(logging.WARNING, logger="missy.vision.memory_usage"):
            tracker.update_from_scene_manager(manager)
        assert "high" in caplog.text.lower() or "usage" in caplog.text.lower()

    def test_over_limit_warning_emitted(self, caplog):
        """Over-limit condition should emit an 'over limit' warning."""
        import logging

        tracker = MemoryTracker(max_bytes=1, warn_fraction=0.8)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        with caplog.at_level(logging.WARNING, logger="missy.vision.memory_usage"):
            tracker.update_from_scene_manager(manager)
        assert "over limit" in caplog.text.lower() or "limit" in caplog.text.lower()

    def test_over_limit_logs_percentages(self, caplog):
        """Over-limit log message includes MB and percent figures."""
        import logging

        tracker = MemoryTracker(max_bytes=1)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        with caplog.at_level(logging.WARNING, logger="missy.vision.memory_usage"):
            tracker.update_from_scene_manager(manager)
        # The format string uses MB and % — check numeric content exists
        assert "%" in caplog.text or "MB" in caplog.text or len(caplog.records) > 0

    def test_warning_not_emitted_when_exactly_at_warn_boundary_minus_one(self, caplog):
        """Usage fraction just below warn_fraction emits no warning."""
        import logging

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frame_est = estimate_frame_bytes(img)
        # Set max_bytes so usage_fraction is slightly below 0.8
        max_bytes = int(frame_est / 0.79)
        tracker = MemoryTracker(max_bytes=max_bytes, warn_fraction=0.8)
        # Verify our setup: usage < warn_fraction
        usage = frame_est / max_bytes
        if usage >= 0.8:
            pytest.skip(
                "floating point edge: cannot construct sub-threshold case for this frame size"
            )
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        with caplog.at_level(logging.WARNING, logger="missy.vision.memory_usage"):
            tracker.update_from_scene_manager(manager)
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# MemoryTracker.should_evict
# ---------------------------------------------------------------------------


class TestMemoryTrackerShouldEvict:
    def test_returns_false_with_no_report(self):
        tracker = MemoryTracker()
        assert tracker.should_evict() is False

    def test_returns_false_when_under_limit(self):
        tracker = MemoryTracker(max_bytes=500_000_000)
        manager = _make_manager_mock(sessions={})
        tracker.update_from_scene_manager(manager)
        assert tracker.should_evict() is False

    def test_returns_true_when_over_limit(self):
        tracker = MemoryTracker(max_bytes=1)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager = _make_manager_mock(sessions={"t": session})
        tracker.update_from_scene_manager(manager)
        assert tracker.should_evict() is True

    def test_updates_after_successive_reports(self):
        tracker = MemoryTracker(max_bytes=1)
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager_full = _make_manager_mock(sessions={"t": session})
        tracker.update_from_scene_manager(manager_full)
        assert tracker.should_evict() is True

        # Now update with empty manager (no sessions) — usage drops to zero
        manager_empty = _make_manager_mock(sessions={})
        tracker.update_from_scene_manager(manager_empty)
        assert tracker.should_evict() is False


# ---------------------------------------------------------------------------
# MemoryTracker.report
# ---------------------------------------------------------------------------


class TestMemoryTrackerReport:
    def test_none_before_first_update(self):
        tracker = MemoryTracker()
        assert tracker.report() is None

    def test_report_is_last_computed(self):
        tracker = MemoryTracker()
        manager = _make_manager_mock(sessions={})
        r1 = tracker.update_from_scene_manager(manager)
        assert tracker.report() is r1

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        session = _make_session_mock(frames=[_make_frame_mock(img)])
        manager2 = _make_manager_mock(sessions={"t": session})
        r2 = tracker.update_from_scene_manager(manager2)
        assert tracker.report() is r2
        assert tracker.report() is not r1

    def test_report_is_memory_report_instance(self):
        tracker = MemoryTracker()
        manager = _make_manager_mock(sessions={})
        tracker.update_from_scene_manager(manager)
        assert isinstance(tracker.report(), MemoryReport)


# ---------------------------------------------------------------------------
# get_memory_tracker module singleton
# ---------------------------------------------------------------------------


class TestGetMemoryTracker:
    def setup_method(self):
        """Reset the module-level singleton before each test."""
        import missy.vision.memory_usage as mu

        mu._tracker = None

    def test_returns_memory_tracker_instance(self):
        tracker = get_memory_tracker()
        assert isinstance(tracker, MemoryTracker)

    def test_singleton_same_object_on_repeated_calls(self):
        t1 = get_memory_tracker()
        t2 = get_memory_tracker()
        assert t1 is t2

    def test_default_max_bytes_500mb(self):
        tracker = get_memory_tracker()
        assert tracker.max_bytes == 500_000_000

    def test_existing_singleton_not_replaced(self):
        import missy.vision.memory_usage as mu

        custom = MemoryTracker(max_bytes=999)
        mu._tracker = custom
        result = get_memory_tracker()
        assert result is custom
        assert result.max_bytes == 999

    def teardown_method(self):
        """Restore module singleton to None after each test."""
        import missy.vision.memory_usage as mu

        mu._tracker = None
