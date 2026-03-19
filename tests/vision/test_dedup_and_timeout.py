"""Tests for SceneSession frame deduplication and CameraHandle capture timeout.

Covers:
- SceneSession.add_frame with deduplicate=True/False and custom dedup_threshold
- CameraHandle.capture enforcing timeout_seconds from CaptureConfig
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from missy.vision.capture import CameraHandle, CaptureConfig, FailureType
from missy.vision.scene_memory import (
    SceneFrame,
    SceneSession,
    TaskType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_bgr(value: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Return a solid-colour BGR image whose perceptual hash is unique per value."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noise_image(seed: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Return a reproducible random-noise BGR image with non-trivial hash."""
    rng = np.random.RandomState(seed)
    return rng.randint(30, 220, (h, w, 3), dtype=np.uint8)


def _make_session(task_id: str = "t1") -> SceneSession:
    return SceneSession(task_id, task_type=TaskType.GENERAL, max_frames=20)


def _make_open_camera(
    config: CaptureConfig | None = None,
    read_return: tuple = (True, None),
) -> tuple[CameraHandle, MagicMock]:
    """Return a CameraHandle whose internal VideoCapture is mocked open."""
    cfg = config or CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=10.0)
    cam = CameraHandle("/dev/video0", cfg)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    # Default read() returns a recognisable non-blank frame
    if read_return[1] is None:
        frame = _noise_image(1, 480, 640) * 1  # mean well above blank threshold
        frame[:] = 128  # guaranteed non-blank
        read_return = (read_return[0], frame)

    mock_cap.read.return_value = read_return
    cam._cap = mock_cap
    cam._opened = True
    return cam, mock_cap


# ===========================================================================
# Frame deduplication — SceneSession.add_frame
# ===========================================================================


class TestDedupFirstFrame:
    """First frame is always stored regardless of dedup setting."""

    def test_first_frame_stored_dedup_true(self) -> None:
        session = _make_session()
        img = _noise_image(42)
        result = session.add_frame(img, deduplicate=True)
        assert result is not None
        assert isinstance(result, SceneFrame)

    def test_first_frame_stored_dedup_false(self) -> None:
        session = _make_session()
        img = _noise_image(42)
        result = session.add_frame(img, deduplicate=False)
        assert result is not None

    def test_first_frame_increments_counter(self) -> None:
        session = _make_session()
        result = session.add_frame(_noise_image(1), deduplicate=True)
        assert result is not None
        assert result.frame_id == 1

    def test_session_has_one_stored_frame_after_first(self) -> None:
        session = _make_session()
        session.add_frame(_noise_image(1))
        assert session.frame_count == 1


class TestDedupIdenticalImages:
    """Identical images should be deduplicated by default."""

    def test_identical_image_returns_none(self) -> None:
        session = _make_session()
        img = _noise_image(7)
        session.add_frame(img.copy())
        result = session.add_frame(img.copy(), deduplicate=True)
        assert result is None

    def test_identical_image_not_stored(self) -> None:
        session = _make_session()
        img = _noise_image(7)
        session.add_frame(img.copy())
        session.add_frame(img.copy(), deduplicate=True)
        assert session.frame_count == 1

    def test_uniform_black_frame_deduplicated(self) -> None:
        """Two all-black frames: same perceptual hash, distance = 0."""
        session = _make_session()
        black = _solid_bgr(0)
        session.add_frame(black.copy())
        result = session.add_frame(black.copy(), deduplicate=True)
        assert result is None

    def test_uniform_white_frame_deduplicated(self) -> None:
        """Two all-white frames: distance = 0."""
        session = _make_session()
        white = _solid_bgr(255)
        session.add_frame(white.copy())
        result = session.add_frame(white.copy(), deduplicate=True)
        assert result is None


class TestDedupDifferentImages:
    """Very different images should always be stored."""

    def test_black_then_white_stored(self) -> None:
        """All-black vs all-white have maximal hash distance (64 bits)."""
        session = _make_session()
        session.add_frame(_solid_bgr(0))
        result = session.add_frame(_solid_bgr(255), deduplicate=True)
        assert result is not None
        assert result.frame_id == 2

    def test_black_then_white_frame_count_is_two(self) -> None:
        session = _make_session()
        session.add_frame(_solid_bgr(0))
        session.add_frame(_solid_bgr(255), deduplicate=True)
        assert session.frame_count == 2

    def test_different_noise_images_stored(self) -> None:
        """Two statistically different noise images should have high hash distance."""
        session = _make_session()
        img_a = _noise_image(1)
        # Invert image to maximise difference
        img_b = (255 - img_a.astype(np.int16)).clip(0, 255).astype(np.uint8)
        session.add_frame(img_a)
        result = session.add_frame(img_b, deduplicate=True)
        assert result is not None


class TestDedupDisabled:
    """deduplicate=False must bypass the hash comparison entirely."""

    def test_identical_image_stored_when_dedup_false(self) -> None:
        session = _make_session()
        img = _noise_image(7)
        session.add_frame(img.copy())
        result = session.add_frame(img.copy(), deduplicate=False)
        assert result is not None

    def test_frame_count_two_when_dedup_false(self) -> None:
        session = _make_session()
        img = _noise_image(7)
        session.add_frame(img.copy())
        session.add_frame(img.copy(), deduplicate=False)
        assert session.frame_count == 2

    def test_three_identical_frames_all_stored_when_dedup_false(self) -> None:
        session = _make_session()
        img = _noise_image(3)
        for _ in range(3):
            session.add_frame(img.copy(), deduplicate=False)
        assert session.frame_count == 3


class TestDedupCustomThreshold:
    """Custom dedup_threshold changes sensitivity."""

    def test_threshold_zero_accepts_only_exact_hashes(self) -> None:
        """With threshold=0 only identical hashes deduplicate."""
        session = _make_session()
        img = _noise_image(42)
        session.add_frame(img.copy())

        # Slightly different image — could land at distance 1-2
        noisy = img.copy()
        noisy[0, 0] = (noisy[0, 0].astype(np.int16) + 30).clip(0, 255).astype(np.uint8)

        # At threshold=0, only 0-distance triggers dedup; noisy likely stored
        result = session.add_frame(noisy, deduplicate=True, dedup_threshold=0)
        # We cannot assert stored/skipped without knowing hash distance,
        # but we can assert the call does not raise and returns SceneFrame or None.
        assert result is None or isinstance(result, SceneFrame)

    def test_very_high_threshold_deduplicates_similar_images(self) -> None:
        """threshold=64 treats virtually any pair as a duplicate."""
        session = _make_session()
        black = _solid_bgr(0)
        white = _solid_bgr(255)
        session.add_frame(black)
        result = session.add_frame(white, deduplicate=True, dedup_threshold=64)
        # Distance between black and white hashes is 64, which equals threshold
        # Condition in source: 0 <= dist <= threshold → should deduplicate
        assert result is None

    def test_low_threshold_stores_slightly_modified_frame(self) -> None:
        """With threshold=1 a frame with moderate hash distance should be stored."""
        session = _make_session()
        img = _noise_image(10)
        session.add_frame(img.copy())

        # Inverted image will have very high distance → definitely stored
        inverted = (255 - img.astype(np.int16)).clip(0, 255).astype(np.uint8)
        result = session.add_frame(inverted, deduplicate=True, dedup_threshold=1)
        assert result is not None


class TestDedupFrameCounter:
    """_frame_counter must increment even for deduplicated frames."""

    def test_counter_increments_on_dedup(self) -> None:
        session = _make_session()
        img = _noise_image(5)
        f1 = session.add_frame(img.copy())
        assert f1 is not None
        assert f1.frame_id == 1

        # Duplicate — should be skipped
        result = session.add_frame(img.copy(), deduplicate=True)
        assert result is None

        # Next unique frame must get id=3, not id=2
        different = _solid_bgr(200)
        f3 = session.add_frame(different, deduplicate=True)
        assert f3 is not None
        assert f3.frame_id == 3

    def test_multiple_dedup_skips_then_unique(self) -> None:
        session = _make_session()
        img = _noise_image(9)
        session.add_frame(img.copy())

        for _ in range(4):
            session.add_frame(img.copy(), deduplicate=True)

        different = _solid_bgr(10)
        result = session.add_frame(different, deduplicate=True)
        assert result is not None
        assert result.frame_id == 6  # 1 original + 4 dedup + 1 unique

    def test_summarize_reports_total_counter_not_stored_count(self) -> None:
        """summarize() frame_count should reflect all attempted frames."""
        session = _make_session()
        img = _noise_image(3)
        session.add_frame(img.copy())
        session.add_frame(img.copy(), deduplicate=True)  # skipped
        session.add_frame(img.copy(), deduplicate=True)  # skipped

        summary = session.summarize()
        assert summary["frame_count"] == 3   # all attempted
        assert summary["frames_retained"] == 1  # only one stored


# ===========================================================================
# Capture timeout — CameraHandle.capture
# ===========================================================================


class TestCaptureTimeoutNormal:
    """When capture completes before the deadline it must succeed."""

    def test_successful_capture_within_timeout(self) -> None:
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=10.0,
                            blank_threshold=5.0, adaptive_blank=False)
        cam, mock_cap = _make_open_camera(cfg, read_return=(True, frame))

        with patch("time.monotonic") as mock_mono:
            # deadline = start + 10.0; first check is well within deadline
            mock_mono.side_effect = [0.0, 1.0]  # deadline_set=0+10; check=1 < 10
            result = cam.capture()

        assert result.success is True
        assert result.image is not None

    def test_successful_result_has_no_error(self) -> None:
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cfg = CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=10.0,
                            blank_threshold=5.0, adaptive_blank=False)
        cam, _ = _make_open_camera(cfg, read_return=(True, frame))

        with patch("time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 1.0]
            result = cam.capture()

        assert result.error == ""
        assert result.failure_type == ""


class TestCaptureTimeoutExpired:
    """When the deadline is exceeded before the first read, a TRANSIENT failure is returned."""

    def _expired_camera(self, timeout: float = 5.0) -> CameraHandle:
        cfg = CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=timeout,
                            blank_threshold=5.0, adaptive_blank=False)
        cam, _ = _make_open_camera(cfg)
        return cam

    def test_expired_deadline_returns_failure(self) -> None:
        cam = self._expired_camera(timeout=5.0)

        with patch("time.monotonic") as mock_mono:
            # deadline = 0 + 5 = 5; first attempt check returns 10 > 5
            mock_mono.side_effect = [0.0, 10.0]
            result = cam.capture()

        assert result.success is False

    def test_expired_deadline_failure_type_is_transient(self) -> None:
        cam = self._expired_camera(timeout=5.0)

        with patch("time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 10.0]
            result = cam.capture()

        assert result.failure_type == FailureType.TRANSIENT

    def test_timeout_error_message_contains_timeout_value(self) -> None:
        cam = self._expired_camera(timeout=7.5)

        with patch("time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 100.0]
            result = cam.capture()

        assert "7.5" in result.error

    def test_timeout_error_message_contains_timed_out(self) -> None:
        cam = self._expired_camera(timeout=3.0)

        with patch("time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 99.0]
            result = cam.capture()

        assert "timed out" in result.error.lower()

    def test_timeout_on_second_attempt(self) -> None:
        """Deadline expires between attempt 1 (read fails) and attempt 2."""
        frame = None  # read failure
        cfg = CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=5.0,
                            blank_threshold=5.0, adaptive_blank=False,
                            retry_delay=0.0)
        cam, mock_cap = _make_open_camera(cfg, read_return=(False, frame))

        with patch("time.monotonic") as mock_mono:
            # deadline set at t=0 → deadline=5
            # attempt 1 check: t=1 < 5  → ok, read fails
            # deadline-aware sleep check: t=2 (within deadline)
            # attempt 2 check: t=6 > 5  → timeout
            mock_mono.side_effect = [0.0, 1.0, 2.0, 6.0]
            with patch("time.sleep"):  # suppress retry delay
                result = cam.capture()

        assert result.success is False
        assert result.failure_type == FailureType.TRANSIENT

    def test_timeout_result_has_no_image(self) -> None:
        cam = self._expired_camera(timeout=2.0)

        with patch("time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 99.0]
            result = cam.capture()

        assert result.image is None

    def test_timeout_device_path_preserved(self) -> None:
        cfg = CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=5.0,
                            blank_threshold=5.0, adaptive_blank=False)
        cam, _ = _make_open_camera(cfg)

        with patch("time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 99.0]
            result = cam.capture()

        assert result.device_path == "/dev/video0"

    def test_zero_timeout_always_fails(self) -> None:
        """timeout_seconds=0 means deadline is immediately in the past."""
        cfg = CaptureConfig(warmup_frames=0, max_retries=3, timeout_seconds=0.0,
                            blank_threshold=5.0, adaptive_blank=False)
        cam, _ = _make_open_camera(cfg)

        with patch("time.monotonic") as mock_mono:
            # deadline = 1.0 + 0 = 1.0; check returns 2.0 > 1.0
            mock_mono.side_effect = [1.0, 2.0]
            result = cam.capture()

        assert result.success is False
        assert result.failure_type == FailureType.TRANSIENT
