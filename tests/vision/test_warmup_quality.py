"""Tests for warmup quality assessment and capture_stats in CameraHandle.

Covers:
- _warmup(): return value, intensity tracking, stability logging, error paths
- _is_warmup_stable(): edge cases on intensity list length and spread
- capture_stats property: all keys, arithmetic, uptime sentinel
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(mean_value: int) -> np.ndarray:
    """Return a 10×10 BGR frame whose mean pixel value equals *mean_value*."""
    return np.full((10, 10, 3), mean_value, dtype=np.uint8)


def _make_handle(warmup_frames: int = 5) -> CameraHandle:
    """Construct a CameraHandle with a fake cap already injected.

    The handle is put into the *opened* state without calling open() so tests
    can exercise internal methods directly while staying independent of cv2.
    """
    config = CaptureConfig(warmup_frames=warmup_frames)
    handle = CameraHandle("/dev/video0", config)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    handle._cap = mock_cap
    handle._opened = True
    handle._open_time = time.monotonic()
    return handle


# ---------------------------------------------------------------------------
# _warmup() — return value and intensity tracking
# ---------------------------------------------------------------------------


class TestWarmupReturnValue:
    def test_returns_zero_when_warmup_frames_is_zero(self):
        handle = _make_handle(warmup_frames=0)
        result = handle._warmup()
        assert result == 0

    def test_returns_count_of_reads_issued(self):
        handle = _make_handle(warmup_frames=4)
        handle._cap.read.return_value = (True, _make_frame(100))
        discarded = handle._warmup()
        assert discarded == 4

    def test_returns_partial_count_when_exception_breaks_loop(self):
        handle = _make_handle(warmup_frames=5)
        # First two reads succeed, third raises
        handle._cap.read.side_effect = [
            (True, _make_frame(80)),
            (True, _make_frame(85)),
            RuntimeError("hardware error"),
        ]
        discarded = handle._warmup()
        # `discarded += 1` sits on the line *after* `self._cap.read()`.  When
        # the third read() raises, the increment is never reached for that
        # iteration, so only the two completed reads are counted.
        assert discarded == 2

    def test_ret_false_still_counts_as_discarded(self):
        handle = _make_handle(warmup_frames=3)
        handle._cap.read.return_value = (False, None)
        discarded = handle._warmup()
        assert discarded == 3

    def test_frame_none_still_counts_as_discarded(self):
        handle = _make_handle(warmup_frames=2)
        handle._cap.read.return_value = (True, None)
        discarded = handle._warmup()
        assert discarded == 2


# ---------------------------------------------------------------------------
# _warmup() — intensity recording
# ---------------------------------------------------------------------------


class TestWarmupIntensityTracking:
    def test_intensities_recorded_for_valid_frames(self):
        handle = _make_handle(warmup_frames=3)
        handle._cap.read.side_effect = [
            (True, _make_frame(50)),
            (True, _make_frame(60)),
            (True, _make_frame(70)),
        ]
        handle._warmup()
        assert len(handle._warmup_intensities) == 3

    def test_intensity_values_match_frame_means(self):
        handle = _make_handle(warmup_frames=2)
        frame_a = _make_frame(100)
        frame_b = _make_frame(200)
        handle._cap.read.side_effect = [
            (True, frame_a),
            (True, frame_b),
        ]
        handle._warmup()
        assert handle._warmup_intensities[0] == pytest.approx(float(np.mean(frame_a)))
        assert handle._warmup_intensities[1] == pytest.approx(float(np.mean(frame_b)))

    def test_failed_reads_not_appended_to_intensities(self):
        handle = _make_handle(warmup_frames=4)
        handle._cap.read.side_effect = [
            (False, None),
            (True, _make_frame(90)),
            (True, None),  # ret=True but frame=None
            (True, _make_frame(95)),
        ]
        handle._warmup()
        # Only the two frames with ret=True and frame is not None are recorded
        assert len(handle._warmup_intensities) == 2

    def test_intensities_replaced_on_each_warmup_call(self):
        handle = _make_handle(warmup_frames=2)
        handle._cap.read.return_value = (True, _make_frame(40))
        handle._warmup()
        first_intensities = list(handle._warmup_intensities)

        handle._cap.read.return_value = (True, _make_frame(200))
        handle._warmup()
        second_intensities = list(handle._warmup_intensities)

        assert first_intensities != second_intensities
        assert len(second_intensities) == 2

    def test_zero_size_frame_not_appended(self):
        """A frame with size == 0 should not contribute an intensity value."""
        handle = _make_handle(warmup_frames=2)
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        handle._cap.read.side_effect = [
            (True, empty_frame),
            (True, _make_frame(50)),
        ]
        handle._warmup()
        # Only the non-empty frame should be recorded
        assert len(handle._warmup_intensities) == 1

    def test_intensities_empty_after_exception_on_first_read(self):
        handle = _make_handle(warmup_frames=3)
        handle._cap.read.side_effect = RuntimeError("camera disconnected")
        handle._warmup()
        assert handle._warmup_intensities == []


# ---------------------------------------------------------------------------
# _warmup() — stability logging
# ---------------------------------------------------------------------------


class TestWarmupStabilityLogging:
    def test_stable_warmup_logs_debug(self):
        handle = _make_handle(warmup_frames=3)
        # Tight spread: 100, 101, 102 → spread < 5.0
        handle._cap.read.side_effect = [
            (True, _make_frame(100)),
            (True, _make_frame(101)),
            (True, _make_frame(102)),
        ]
        with patch("missy.vision.capture.logger") as mock_logger:
            handle._warmup()
        debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
        assert any("stabilized" in msg for msg in debug_calls)
        mock_logger.warning.assert_not_called()

    def test_unstable_warmup_logs_warning(self):
        handle = _make_handle(warmup_frames=3)
        # Wide spread: 50, 100, 150 → spread = 100 >= 5.0
        handle._cap.read.side_effect = [
            (True, _make_frame(50)),
            (True, _make_frame(100)),
            (True, _make_frame(150)),
        ]
        with patch("missy.vision.capture.logger") as mock_logger:
            handle._warmup()
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("stabilized" in msg or "spread" in msg for msg in warning_calls)

    def test_exactly_at_threshold_spread_is_stable(self):
        """Spread strictly less than 5.0 is stable; spread of exactly 5.0 is not."""
        handle = _make_handle(warmup_frames=3)
        # min=100, max=105 → spread = 5.0 (NOT < 5.0 → warning)
        handle._cap.read.side_effect = [
            (True, _make_frame(100)),
            (True, _make_frame(102)),
            (True, _make_frame(105)),
        ]
        with patch("missy.vision.capture.logger") as mock_logger:
            handle._warmup()
        warning_called = mock_logger.warning.called
        assert warning_called

    def test_spread_just_below_threshold_is_stable(self):
        handle = _make_handle(warmup_frames=3)
        # spread = 104 - 100 = 4 → < 5.0 → debug only
        handle._cap.read.side_effect = [
            (True, _make_frame(100)),
            (True, _make_frame(102)),
            (True, _make_frame(104)),
        ]
        with patch("missy.vision.capture.logger") as mock_logger:
            handle._warmup()
        mock_logger.warning.assert_not_called()

    def test_fewer_than_three_valid_frames_no_stability_log(self):
        handle = _make_handle(warmup_frames=2)
        handle._cap.read.return_value = (True, _make_frame(80))
        with patch("missy.vision.capture.logger") as mock_logger:
            handle._warmup()
        # Neither the stability debug message nor the spread warning should fire
        # (the stability assessment branch only runs when len(intensities) >= 3)
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert not any("spread" in msg for msg in warning_calls)

    def test_stability_assessment_uses_last_three_frames_only(self):
        """With 5 warmup frames, only the last 3 are used for spread assessment."""
        handle = _make_handle(warmup_frames=5)
        # First 2 frames are wildly different; last 3 are tight — expect stable
        handle._cap.read.side_effect = [
            (True, _make_frame(10)),
            (True, _make_frame(200)),
            (True, _make_frame(100)),
            (True, _make_frame(101)),
            (True, _make_frame(102)),
        ]
        with patch("missy.vision.capture.logger") as mock_logger:
            handle._warmup()
        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# _is_warmup_stable()
# ---------------------------------------------------------------------------


class TestIsWarmupStable:
    def test_returns_true_with_no_intensities(self):
        handle = _make_handle()
        handle._warmup_intensities = []
        assert handle._is_warmup_stable() is True

    def test_returns_true_with_one_intensity(self):
        handle = _make_handle()
        handle._warmup_intensities = [50.0]
        assert handle._is_warmup_stable() is True

    def test_returns_true_with_two_intensities(self):
        handle = _make_handle()
        handle._warmup_intensities = [50.0, 200.0]
        assert handle._is_warmup_stable() is True

    def test_returns_true_when_last_three_spread_below_threshold(self):
        handle = _make_handle()
        handle._warmup_intensities = [50.0, 100.0, 101.0, 102.0, 103.0]
        assert handle._is_warmup_stable() is True

    def test_returns_false_when_last_three_spread_at_or_above_threshold(self):
        handle = _make_handle()
        # Last three: 100, 103, 105 → spread = 5 → NOT < 5.0
        handle._warmup_intensities = [50.0, 100.0, 103.0, 105.0]
        assert handle._is_warmup_stable() is False

    def test_returns_false_with_large_spread(self):
        handle = _make_handle()
        handle._warmup_intensities = [10.0, 100.0, 200.0]
        assert handle._is_warmup_stable() is False

    def test_boundary_spread_just_below_five(self):
        handle = _make_handle()
        # spread = 4.99 → stable
        handle._warmup_intensities = [100.0, 102.0, 104.99]
        assert handle._is_warmup_stable() is True

    def test_boundary_spread_exactly_five(self):
        handle = _make_handle()
        # spread = 5.0 → NOT < 5.0 → unstable
        handle._warmup_intensities = [100.0, 102.0, 105.0]
        assert handle._is_warmup_stable() is False

    def test_only_last_three_entries_matter(self):
        """Early entries with extreme values should not affect stability result."""
        handle = _make_handle()
        # First entry is an outlier; last 3 are tight
        handle._warmup_intensities = [0.0, 100.0, 101.0, 102.0]
        assert handle._is_warmup_stable() is True

    def test_identical_intensities_are_stable(self):
        handle = _make_handle()
        handle._warmup_intensities = [80.0, 80.0, 80.0]
        assert handle._is_warmup_stable() is True


# ---------------------------------------------------------------------------
# capture_stats property — keys present
# ---------------------------------------------------------------------------


class TestCaptureStatsKeys:
    EXPECTED_KEYS = {
        "device_path",
        "is_open",
        "uptime_seconds",
        "capture_count",
        "success_count",
        "success_rate",
        "warmup_frames",
        "warmup_stable",
    }

    def test_all_expected_keys_present(self):
        handle = _make_handle()
        stats = handle.capture_stats
        assert set(stats.keys()) == self.EXPECTED_KEYS

    def test_no_extra_keys(self):
        handle = _make_handle()
        stats = handle.capture_stats
        extra = set(stats.keys()) - self.EXPECTED_KEYS
        assert extra == set()


# ---------------------------------------------------------------------------
# capture_stats property — device_path and is_open
# ---------------------------------------------------------------------------


class TestCaptureStatsBasicFields:
    def test_device_path_matches_constructor_argument(self):
        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video3", config)
        handle._cap = MagicMock()
        handle._cap.isOpened.return_value = True
        handle._opened = True
        handle._open_time = time.monotonic()
        assert handle.capture_stats["device_path"] == "/dev/video3"

    def test_is_open_true_when_cap_is_opened(self):
        handle = _make_handle()
        assert handle.capture_stats["is_open"] is True

    def test_is_open_false_when_not_opened(self):
        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        # _cap is None and _opened is False by default
        assert handle.capture_stats["is_open"] is False


# ---------------------------------------------------------------------------
# capture_stats property — uptime_seconds
# ---------------------------------------------------------------------------


class TestCaptureStatsUptime:
    def test_uptime_is_zero_when_camera_not_opened(self):
        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        assert handle.capture_stats["uptime_seconds"] == 0.0

    def test_uptime_positive_when_opened(self):
        handle = _make_handle()
        # _open_time was set to monotonic() just before this call, so uptime ≥ 0
        stats = handle.capture_stats
        assert stats["uptime_seconds"] >= 0.0

    def test_uptime_increases_over_time(self):
        handle = _make_handle()
        # Backdate the open_time by 2 seconds to guarantee measurable uptime
        handle._open_time = time.monotonic() - 2.0
        stats = handle.capture_stats
        assert stats["uptime_seconds"] >= 1.9

    def test_uptime_is_rounded_to_one_decimal(self):
        handle = _make_handle()
        handle._open_time = time.monotonic() - 3.456789
        uptime = handle.capture_stats["uptime_seconds"]
        # round(..., 1) means one decimal place
        assert uptime == round(uptime, 1)


# ---------------------------------------------------------------------------
# capture_stats property — capture_count and success_count
# ---------------------------------------------------------------------------


class TestCaptureStatsCounts:
    def test_initial_counts_are_zero(self):
        handle = _make_handle()
        stats = handle.capture_stats
        assert stats["capture_count"] == 0
        assert stats["success_count"] == 0

    def test_counts_reflect_injected_values(self):
        handle = _make_handle()
        handle._capture_count = 10
        handle._success_count = 7
        stats = handle.capture_stats
        assert stats["capture_count"] == 10
        assert stats["success_count"] == 7


# ---------------------------------------------------------------------------
# capture_stats property — success_rate
# ---------------------------------------------------------------------------


class TestCaptureStatsSuccessRate:
    def test_success_rate_zero_when_no_captures(self):
        handle = _make_handle()
        assert handle.capture_stats["success_rate"] == 0.0

    def test_success_rate_one_when_all_succeed(self):
        handle = _make_handle()
        handle._capture_count = 5
        handle._success_count = 5
        assert handle.capture_stats["success_rate"] == 1.0

    def test_success_rate_zero_when_none_succeed(self):
        handle = _make_handle()
        handle._capture_count = 3
        handle._success_count = 0
        assert handle.capture_stats["success_rate"] == 0.0

    def test_success_rate_computed_correctly(self):
        handle = _make_handle()
        handle._capture_count = 8
        handle._success_count = 6
        expected = round(6 / 8, 4)
        assert handle.capture_stats["success_rate"] == expected

    def test_success_rate_rounded_to_four_decimal_places(self):
        handle = _make_handle()
        handle._capture_count = 3
        handle._success_count = 1
        rate = handle.capture_stats["success_rate"]
        # round(1/3, 4) == 0.3333
        assert rate == pytest.approx(0.3333, abs=1e-4)

    def test_success_rate_is_float(self):
        handle = _make_handle()
        handle._capture_count = 4
        handle._success_count = 4
        assert isinstance(handle.capture_stats["success_rate"], float)


# ---------------------------------------------------------------------------
# capture_stats property — warmup_frames and warmup_stable
# ---------------------------------------------------------------------------


class TestCaptureStatsWarmupFields:
    def test_warmup_frames_reflects_recorded_intensities_count(self):
        handle = _make_handle()
        handle._warmup_intensities = [100.0, 101.0, 102.0]
        assert handle.capture_stats["warmup_frames"] == 3

    def test_warmup_frames_zero_with_no_intensities(self):
        handle = _make_handle()
        assert handle.capture_stats["warmup_frames"] == 0

    def test_warmup_stable_true_when_few_intensities(self):
        handle = _make_handle()
        handle._warmup_intensities = [80.0, 82.0]
        assert handle.capture_stats["warmup_stable"] is True

    def test_warmup_stable_true_with_tight_spread(self):
        handle = _make_handle()
        handle._warmup_intensities = [100.0, 101.0, 102.0]
        assert handle.capture_stats["warmup_stable"] is True

    def test_warmup_stable_false_with_wide_spread(self):
        handle = _make_handle()
        handle._warmup_intensities = [50.0, 100.0, 200.0]
        assert handle.capture_stats["warmup_stable"] is False

    def test_warmup_stable_is_bool(self):
        handle = _make_handle()
        assert isinstance(handle.capture_stats["warmup_stable"], bool)

    def test_warmup_frames_count_matches_len_of_valid_frames_only(self):
        """After a warmup with some failed reads, warmup_frames reflects only
        frames where intensities were actually recorded."""
        handle = _make_handle(warmup_frames=4)
        handle._cap.read.side_effect = [
            (False, None),
            (True, _make_frame(90)),
            (True, _make_frame(92)),
            (True, _make_frame(94)),
        ]
        handle._warmup()
        assert handle.capture_stats["warmup_frames"] == 3


# ---------------------------------------------------------------------------
# Interaction: warmup followed by capture_stats
# ---------------------------------------------------------------------------


class TestWarmupThenCaptureStats:
    def test_stats_consistent_after_stable_warmup(self):
        handle = _make_handle(warmup_frames=3)
        handle._cap.read.return_value = (True, _make_frame(100))
        handle._warmup()
        stats = handle.capture_stats
        assert stats["warmup_frames"] == 3
        assert stats["warmup_stable"] is True

    def test_stats_consistent_after_unstable_warmup(self):
        handle = _make_handle(warmup_frames=3)
        handle._cap.read.side_effect = [
            (True, _make_frame(50)),
            (True, _make_frame(128)),
            (True, _make_frame(200)),
        ]
        handle._warmup()
        stats = handle.capture_stats
        assert stats["warmup_frames"] == 3
        assert stats["warmup_stable"] is False

    def test_capture_count_zero_after_warmup_only(self):
        """_warmup() reads must not touch _capture_count or _success_count."""
        handle = _make_handle(warmup_frames=5)
        handle._cap.read.return_value = (True, _make_frame(128))
        handle._warmup()
        stats = handle.capture_stats
        assert stats["capture_count"] == 0
        assert stats["success_count"] == 0
        assert stats["success_rate"] == 0.0
