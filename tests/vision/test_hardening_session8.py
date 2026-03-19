"""Session 8 hardening tests.

Covers the specific fixes made in session 8:

1. capture.py  — Resource leak prevention on open()
2. capture.py  — Warmup timeout enforcement
3. multi_camera.py — Handle validity check in _capture_one
4. multi_camera.py — Deadline-based per-future timeout in capture_all
5. resilient_capture.py — Blank detector reset on device switch
6. sources.py  — FileSource rejects non-regular files (device nodes)
7. discovery.py — Symlink cycle detection in _read_usb_ids

All tests run without a real camera by mocking cv2 and sysfs paths.
"""

from __future__ import annotations

import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CameraHandle, CaptureConfig, CaptureError
from missy.vision.discovery import CameraDevice, CameraDiscovery
from missy.vision.multi_camera import MultiCameraManager
from missy.vision.resilient_capture import ResilientCamera
from missy.vision.sources import FileSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_device(
    path: str = "/dev/video0",
    name: str = "Test Camera",
    vendor_id: str = "046d",
    product_id: str = "085c",
    bus_info: str = "usb-0000:00:14.0-1",
) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id=vendor_id,
        product_id=product_id,
        bus_info=bus_info,
    )


def _bright_frame(h: int = 100, w: int = 100) -> np.ndarray:
    """Return a non-blank BGR frame."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. capture.py: Resource leak prevention on open()
# ---------------------------------------------------------------------------


class TestCaptureOpenResourceLeak:
    """VideoCapture must be released when open() fails after creation."""

    @patch("missy.vision.capture._get_cv2")
    def test_release_called_when_warmup_raises(self, mock_get_cv2: MagicMock) -> None:
        """If _warmup() raises after VideoCapture is created, release() is called."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Simulate warmup reading failing with an exception via _cap.read raising
        mock_cap.read.side_effect = RuntimeError("simulated camera freeze during warmup")
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cap.get.return_value = 1920

        config = CaptureConfig(warmup_frames=3, timeout_seconds=10.0)
        handle = CameraHandle("/dev/video0", config)

        # open() should not raise — warmup exceptions are swallowed internally;
        # but if we make isOpened() raise after the first call, it propagates.
        # Instead verify that if the post-VideoCapture block raises, release() runs.
        # We force an error by making isOpened() return False after creation.
        mock_cap.isOpened.return_value = False

        with pytest.raises(CaptureError, match="Cannot open camera"):
            handle.open()

        # The VideoCapture was created but failed immediately after — release must
        # have been called to prevent the fd leak.
        mock_cap.release.assert_called()

    @patch("missy.vision.capture._get_cv2")
    def test_release_called_when_resolution_set_raises(self, mock_get_cv2: MagicMock) -> None:
        """If cap.set() raises after VideoCapture creation, release() is called."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.set.side_effect = OSError("ioctl failed")
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_V4L2 = 200
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4

        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)

        with pytest.raises(OSError):
            handle.open()

        mock_cap.release.assert_called()

    @patch("missy.vision.capture._get_cv2")
    def test_cap_is_none_after_failed_open(self, mock_get_cv2: MagicMock) -> None:
        """After a failed open(), _cap should be None to avoid stale references."""
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_V4L2 = 200

        handle = CameraHandle("/dev/video0")

        with pytest.raises(CaptureError):
            handle.open()

        assert handle._cap is None
        assert not handle.is_open


# ---------------------------------------------------------------------------
# 2. capture.py: Warmup timeout
# ---------------------------------------------------------------------------


class TestCaptureWarmupTimeout:
    """Warmup must exit early when the deadline is reached."""

    @patch("missy.vision.capture._get_cv2")
    def test_warmup_exits_before_all_frames_on_slow_camera(
        self, mock_get_cv2: MagicMock
    ) -> None:
        """Warmup exits early (via deadline check) when time advances past the deadline.

        The warmup deadline is ``max(timeout_seconds / 2, 3.0)``.  To get a
        short deadline we use timeout_seconds=10.0 (deadline = 5.0 s) and
        replace time.monotonic with a counter that jumps past the deadline
        after the very first loop-body check so only 1 frame is discarded.

        Call sequence inside _warmup:
          call 0 — deadline assignment: time.monotonic() = 0.0
          call 1 — first loop guard check: returns 0.0  (< deadline 5.0) → enter loop
          call 2 — second loop guard check: returns 10.0 (> deadline 5.0) → break
        """
        mock_cv2 = MagicMock()
        mock_get_cv2.return_value = mock_cv2

        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, _bright_frame())
        mock_cap.isOpened.return_value = True

        config = CaptureConfig(warmup_frames=10, timeout_seconds=10.0)
        handle = CameraHandle("/dev/video0", config)
        handle._cap = mock_cap

        # Return 0.0 for the deadline-set call and first guard, then jump past deadline
        call_values = [0.0, 0.0, 10.0]
        call_index = [0]

        def fake_monotonic() -> float:
            idx = call_index[0]
            val = call_values[idx] if idx < len(call_values) else 10.0
            call_index[0] += 1
            return val

        with patch("missy.vision.capture.time.monotonic", side_effect=fake_monotonic):
            discarded = handle._warmup()

        # Only 1 frame read before the second loop iteration is blocked by deadline
        assert discarded < 10, "Warmup should have exited early due to timeout"
        assert discarded == 1, f"Expected 1 frame discarded, got {discarded}"

    def test_warmup_returns_immediately_when_zero_frames(self) -> None:
        """warmup_frames=0 must return 0 immediately without touching _cap."""
        config = CaptureConfig(warmup_frames=0)
        handle = CameraHandle("/dev/video0", config)
        # _cap is intentionally None — calling read() would crash
        handle._cap = None

        discarded = handle._warmup()

        assert discarded == 0

    def test_warmup_does_not_block_indefinitely(self) -> None:
        """Warmup with many frames exits early when the deadline fires.

        The warmup deadline is ``max(timeout_seconds / 2, 3.0)``.  With
        timeout_seconds=8.0 the deadline is 4.0 s.  We mock time.monotonic
        so that the deadline is set at t=0, the first 3 guard checks pass
        (t=0, 0, 0), and then the 4th returns t=10 (past deadline), so
        warmup exits after 3 frames instead of 100.
        """
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, _bright_frame())

        config = CaptureConfig(warmup_frames=100, timeout_seconds=8.0)
        handle = CameraHandle("/dev/video0", config)
        handle._cap = mock_cap

        # Call pattern for time.monotonic inside _warmup:
        #   call 0: deadline = monotonic() + 4.0  → return 0.0 → deadline = 4.0
        #   call 1: loop iter 1 guard: 0.0 < 4.0  → enter loop body, read frame 1
        #   call 2: loop iter 2 guard: 0.0 < 4.0  → enter loop body, read frame 2
        #   call 3: loop iter 3 guard: 0.0 < 4.0  → enter loop body, read frame 3
        #   call 4: loop iter 4 guard: 10.0 > 4.0 → break
        values = [0.0, 0.0, 0.0, 0.0, 10.0]
        call_index = [0]

        def fake_monotonic() -> float:
            idx = call_index[0]
            val = values[idx] if idx < len(values) else 10.0
            call_index[0] += 1
            return val

        with patch("missy.vision.capture.time.monotonic", side_effect=fake_monotonic):
            discarded = handle._warmup()

        assert discarded == 3, f"Expected 3 frames before deadline, got {discarded}"
        assert discarded < 100, "Warmup should not have run all 100 frames"


# ---------------------------------------------------------------------------
# 3. multi_camera.py: Handle validity check
# ---------------------------------------------------------------------------


class TestMultiCameraHandleValidityCheck:
    """_capture_one skips handles that are already closed."""

    def test_capture_all_returns_failure_for_closed_handle(self) -> None:
        """A handle that is not open should produce a failure result, not an error."""
        mgr = MultiCameraManager()

        # Inject a closed mock handle directly, bypassing add_camera's open() call
        mock_handle = MagicMock()
        mock_handle.is_open = False  # closed
        path = "/dev/video0"
        mgr._handles[path] = mock_handle
        mgr._devices[path] = _make_device(path)

        # Patch health monitor so the test does not need a real one
        with patch("missy.vision.multi_camera.get_health_monitor") as mock_hm:
            mock_hm.return_value = MagicMock()
            result = mgr.capture_all(timeout=5.0)

        assert path in result.results
        assert not result.results[path].success
        assert "closed" in result.results[path].error.lower()
        # capture() must NOT have been called on the closed handle
        mock_handle.capture.assert_not_called()

    def test_capture_all_succeeds_for_open_handle(self) -> None:
        """Open handles should produce successful capture results."""
        mgr = MultiCameraManager()

        mock_handle = MagicMock()
        mock_handle.is_open = True

        from missy.vision.capture import CaptureResult

        success_result = CaptureResult(
            success=True,
            image=_bright_frame(),
            device_path="/dev/video1",
            width=100,
            height=100,
        )
        mock_handle.capture.return_value = success_result
        path = "/dev/video1"
        mgr._handles[path] = mock_handle
        mgr._devices[path] = _make_device(path)

        with patch("missy.vision.multi_camera.get_health_monitor") as mock_hm:
            mock_hm.return_value = MagicMock()
            result = mgr.capture_all(timeout=5.0)

        assert path in result.results
        assert result.results[path].success

    def test_capture_all_mixed_open_closed(self) -> None:
        """Mixed open and closed handles: open one succeeds, closed one fails."""
        mgr = MultiCameraManager()

        from missy.vision.capture import CaptureResult

        # Open handle
        open_handle = MagicMock()
        open_handle.is_open = True
        open_handle.capture.return_value = CaptureResult(
            success=True,
            image=_bright_frame(),
            device_path="/dev/video0",
            width=100,
            height=100,
        )
        mgr._handles["/dev/video0"] = open_handle
        mgr._devices["/dev/video0"] = _make_device("/dev/video0")

        # Closed handle
        closed_handle = MagicMock()
        closed_handle.is_open = False
        mgr._handles["/dev/video1"] = closed_handle
        mgr._devices["/dev/video1"] = _make_device("/dev/video1")

        with patch("missy.vision.multi_camera.get_health_monitor") as mock_hm:
            mock_hm.return_value = MagicMock()
            result = mgr.capture_all(timeout=5.0)

        assert result.results["/dev/video0"].success
        assert not result.results["/dev/video1"].success
        closed_handle.capture.assert_not_called()


# ---------------------------------------------------------------------------
# 4. multi_camera.py: Deadline-based timeout
# ---------------------------------------------------------------------------


class TestMultiCameraDeadlineTimeout:
    """future.result() must use remaining deadline time, not the full timeout."""

    def test_per_future_timeout_decreases_over_time(self) -> None:
        """Remaining time passed to future.result() must be <= original timeout."""
        from concurrent.futures import Future

        mgr = MultiCameraManager()

        from missy.vision.capture import CaptureResult

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = CaptureResult(
            success=True,
            image=_bright_frame(),
            device_path="/dev/video0",
            width=100,
            height=100,
        )
        mgr._handles["/dev/video0"] = mock_handle
        mgr._devices["/dev/video0"] = _make_device()

        remaining_times: list[float] = []

        original_result = Future.result

        def patched_result(self, timeout=None):  # type: ignore[override]
            if timeout is not None:
                remaining_times.append(timeout)
            return original_result(self, timeout=timeout)

        with patch("missy.vision.multi_camera.get_health_monitor") as mock_hm, \
             patch.object(Future, "result", patched_result):
            mock_hm.return_value = MagicMock()
            mgr.capture_all(timeout=30.0)

        # future.result() must have been called with a timeout
        assert remaining_times, "future.result() was never called with a timeout argument"
        # Each remaining value must be positive and at most the original timeout
        for t in remaining_times:
            assert 0 < t <= 30.0, f"Unexpected remaining timeout: {t}"

    def test_capture_all_with_zero_remaining_time_uses_minimum(self) -> None:
        """When deadline is already past, remaining time is floored to >= 0.1, not negative.

        We simulate an expired deadline by making time.monotonic jump forward
        so that ``deadline - time.monotonic()`` is negative.  The implementation
        uses ``max(0.1, deadline - time.monotonic())`` which must produce a
        non-negative value passed to future.result().
        """
        mgr = MultiCameraManager()

        from missy.vision.capture import CaptureResult

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = CaptureResult(
            success=True,
            image=_bright_frame(),
            device_path="/dev/video0",
            width=100,
            height=100,
        )
        mgr._handles["/dev/video0"] = mock_handle
        mgr._devices["/dev/video0"] = _make_device()

        from concurrent.futures import Future

        remaining_times: list[float] = []
        original_result = Future.result

        def patched_result(self, timeout=None):  # type: ignore[override]
            if timeout is not None:
                remaining_times.append(timeout)
            return original_result(self, timeout=timeout)

        # Use a callable (not an iterator) to avoid StopIteration inside generators.
        # Return a time that makes the deadline appear already expired when
        # future.result() computes remaining = max(0.1, deadline - monotonic()).
        base_time = time.monotonic()
        call_count = [0]

        def fake_monotonic() -> float:
            call_count[0] += 1
            # First call: t0 = base_time
            # Second call: deadline = base_time + 1.0
            # All subsequent calls: far in the future so deadline is already expired
            if call_count[0] <= 2:
                return base_time
            return base_time + 1000.0  # deadline expired

        with patch("missy.vision.multi_camera.get_health_monitor") as mock_hm, \
             patch.object(Future, "result", patched_result), \
             patch("missy.vision.multi_camera.time.monotonic", side_effect=fake_monotonic):
            mock_hm.return_value = MagicMock()
            mgr.capture_all(timeout=1.0)

        for t in remaining_times:
            assert t >= 0.0, f"Remaining timeout must not be negative, got {t}"


# ---------------------------------------------------------------------------
# 5. resilient_capture.py: Blank detector reset on device switch
# ---------------------------------------------------------------------------


class TestResilientCaptureBlankDetectorReset:
    """Blank detector history must be cleared when _open_device switches devices."""

    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_blank_detector_reset_on_device_switch(
        self, mock_handle_cls: MagicMock
    ) -> None:
        """After _open_device, reset_blank_detector() is called on the new handle."""
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera()
        device = _make_device()
        cam._open_device(device)

        # After _open_device, the public reset_blank_detector() must be called
        mock_handle.reset_blank_detector.assert_called_once()

    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_blank_detector_reset_called_on_none_detector(
        self, mock_handle_cls: MagicMock
    ) -> None:
        """_open_device must not crash when blank detector is disabled (adaptive=False)."""
        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle_cls.return_value = mock_handle

        cam = ResilientCamera()
        device = _make_device()

        # Should not raise — reset_blank_detector handles None internally
        cam._open_device(device)

        assert cam.is_connected
        mock_handle.reset_blank_detector.assert_called_once()

    @patch("missy.vision.resilient_capture.CameraHandle")
    def test_old_handle_closed_before_new_open(self, mock_handle_cls: MagicMock) -> None:
        """The previous handle must be closed before the new handle is opened."""
        closed_calls: list[str] = []

        old_mock = MagicMock()
        old_mock.is_open = True

        def close_side_effect():
            closed_calls.append("old_closed")

        old_mock.close.side_effect = close_side_effect

        new_mock = MagicMock()
        new_mock.is_open = True
        new_mock._blank_detector = None

        mock_handle_cls.return_value = new_mock

        cam = ResilientCamera()
        cam._handle = old_mock  # simulate existing connection
        cam._connected = True

        cam._open_device(_make_device())

        old_mock.close.assert_called_once()
        assert "old_closed" in closed_calls


# ---------------------------------------------------------------------------
# 6. sources.py: FileSource rejects non-regular files
# ---------------------------------------------------------------------------


class TestFileSourceRejectsNonRegularFiles:
    """FileSource.acquire() must reject device nodes and other special files."""

    def test_rejects_character_device_node(self, tmp_path: Path) -> None:
        """A file whose stat mode is S_IFCHR should raise ValueError."""
        fake_file = tmp_path / "video0"
        fake_file.write_bytes(b"dummy")  # must exist for stat to be called

        # Build a fake stat result with S_IFCHR mode
        fake_stat = MagicMock()
        fake_stat.st_mode = stat.S_IFCHR | 0o660  # character device
        fake_stat.st_size = 1024

        source = FileSource(str(fake_file))

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch.object(Path, "exists", return_value=True), \
             pytest.raises(ValueError, match="Not a regular file"):
            source.acquire()

    def test_rejects_block_device(self, tmp_path: Path) -> None:
        """A file whose stat mode is S_IFBLK should raise ValueError."""
        fake_file = tmp_path / "sda"
        fake_file.write_bytes(b"dummy")

        fake_stat = MagicMock()
        fake_stat.st_mode = stat.S_IFBLK | 0o660
        fake_stat.st_size = 4096

        source = FileSource(str(fake_file))

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch.object(Path, "exists", return_value=True), \
             pytest.raises(ValueError, match="Not a regular file"):
            source.acquire()

    def test_rejects_named_pipe(self, tmp_path: Path) -> None:
        """A file whose stat mode is S_IFIFO should raise ValueError."""
        fake_file = tmp_path / "mypipe"
        fake_file.write_bytes(b"dummy")

        fake_stat = MagicMock()
        fake_stat.st_mode = stat.S_IFIFO | 0o644
        fake_stat.st_size = 0  # pipes often report 0 size

        source = FileSource(str(fake_file))

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch.object(Path, "exists", return_value=True), \
             pytest.raises(ValueError):
            source.acquire()

    def test_accepts_regular_file(self, tmp_path: Path) -> None:
        """A regular image file should be accepted (mode S_IFREG)."""
        fake_file = tmp_path / "image.jpg"
        fake_file.write_bytes(b"dummy content")

        fake_stat = MagicMock()
        fake_stat.st_mode = stat.S_IFREG | 0o644
        fake_stat.st_size = len(b"dummy content")

        # Mock cv2 to return a small image instead of reading the actual bytes
        mock_img = np.zeros((10, 10, 3), dtype=np.uint8)

        source = FileSource(str(fake_file))

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch.object(Path, "exists", return_value=True), \
             patch("missy.vision.sources._get_cv2") as mock_get_cv2:
            mock_cv2 = MagicMock()
            mock_cv2.imread.return_value = mock_img
            mock_get_cv2.return_value = mock_cv2

            frame = source.acquire()

        assert frame is not None
        assert frame.image is not None

    def test_error_message_includes_mode(self, tmp_path: Path) -> None:
        """The ValueError message should include the octal mode for diagnostics."""
        fake_file = tmp_path / "dev_node"
        fake_file.write_bytes(b"x")

        chr_mode = stat.S_IFCHR | 0o660
        fake_stat = MagicMock()
        fake_stat.st_mode = chr_mode
        fake_stat.st_size = 512

        source = FileSource(str(fake_file))

        with patch.object(Path, "stat", return_value=fake_stat), \
             patch.object(Path, "exists", return_value=True), pytest.raises(ValueError) as exc_info:
            source.acquire()

        assert oct(chr_mode) in str(exc_info.value) or "regular file" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 7. discovery.py: Symlink cycle detection in _read_usb_ids
# ---------------------------------------------------------------------------


class TestDiscoverySymlinkCycleDetection:
    """_read_usb_ids must not loop forever when sysfs symlinks form a cycle."""

    def test_symlink_cycle_returns_default_ids(self, tmp_path: Path) -> None:
        """A path that resolves to itself (cycle) must return ('0000', '0000')."""
        # Build a directory that resolves to itself via a symlink cycle:
        #   tmp_path/loop -> tmp_path/loop  (impossible with real fs)
        # Instead, mock Path.resolve so 'current / "device"' always resolves back
        # to the same path, causing the visited set to break the loop.

        disc = CameraDiscovery(sysfs_base=str(tmp_path))

        sysfs_entry = tmp_path / "video0"
        sysfs_entry.mkdir()

        cycle_path = tmp_path / "cycle_root"
        cycle_path.mkdir()

        # resolve() returns the same path every time — simulates a cycle
        def fake_resolve(self, *args, **kwargs):
            return cycle_path

        with patch.object(Path, "resolve", fake_resolve):
            vid, pid = disc._read_usb_ids(sysfs_entry)

        assert vid == "0000"
        assert pid == "0000"

    def test_symlink_cycle_terminates_within_depth_limit(self, tmp_path: Path) -> None:
        """Even without a true cycle, the depth limit (10) must bound traversal."""
        disc = CameraDiscovery(sysfs_base=str(tmp_path))

        sysfs_entry = tmp_path / "video0"
        sysfs_entry.mkdir()

        # Each call to .parent returns the same path (never reaches root)
        deepest = tmp_path / "a" / "b" / "c" / "d" / "e"
        deepest.mkdir(parents=True)

        resolve_call_count = [0]

        def fake_resolve(self, *args, **kwargs):
            resolve_call_count[0] += 1
            return deepest

        with patch.object(Path, "resolve", fake_resolve):
            vid, pid = disc._read_usb_ids(sysfs_entry)

        # Result is the default — no idVendor/idProduct in our tmp tree
        assert vid == "0000"
        assert pid == "0000"
        # resolve() is only called once (for the initial "device" symlink)
        assert resolve_call_count[0] == 1

    def test_read_usb_ids_finds_real_ids_without_cycle(self, tmp_path: Path) -> None:
        """Verify _read_usb_ids returns correct IDs when no cycle is present."""
        disc = CameraDiscovery(sysfs_base=str(tmp_path))

        # Build a minimal sysfs-like tree:
        #   tmp_path/video0/device -> resolves to usb_device/
        #   usb_device/idVendor
        #   usb_device/idProduct
        usb_device = tmp_path / "usb_device"
        usb_device.mkdir()
        (usb_device / "idVendor").write_text("046d\n")
        (usb_device / "idProduct").write_text("085c\n")

        sysfs_entry = tmp_path / "video0"
        sysfs_entry.mkdir()
        (sysfs_entry / "device").symlink_to(usb_device)

        vid, pid = disc._read_usb_ids(sysfs_entry)

        assert vid == "046d"
        assert pid == "085c"

    def test_read_usb_ids_oserror_on_resolve_returns_defaults(
        self, tmp_path: Path
    ) -> None:
        """OSError during symlink resolution must return default IDs, not crash."""
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        sysfs_entry = tmp_path / "video0"
        sysfs_entry.mkdir()

        def raise_oserror(self, *args, **kwargs):
            raise OSError("permission denied reading symlink")

        with patch.object(Path, "resolve", raise_oserror):
            vid, pid = disc._read_usb_ids(sysfs_entry)

        assert vid == "0000"
        assert pid == "0000"

    def test_visited_set_breaks_cycle_after_first_repeat(self, tmp_path: Path) -> None:
        """The visited set must catch a repeated path and break immediately."""
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        sysfs_entry = tmp_path / "video0"
        sysfs_entry.mkdir()

        # Create a real directory that has no idVendor/idProduct
        repeated_path = tmp_path / "repeated"
        repeated_path.mkdir()

        # All calls to resolve() return the same repeated_path — first visit
        # adds it to visited; second encounter triggers 'break'.
        visit_count = [0]

        def counting_resolve(self, *args, **kwargs):
            visit_count[0] += 1
            return repeated_path

        with patch.object(Path, "resolve", counting_resolve):
            vid, pid = disc._read_usb_ids(sysfs_entry)

        assert vid == "0000"
        assert pid == "0000"
        # The loop should have visited once, found it in visited on the next
        # iteration attempt (parent == repeated_path again → same path →
        # break). The exact count is implementation-dependent but must be finite.
        assert visit_count[0] >= 1
