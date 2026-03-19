"""Session 13 comprehensive tests for missy.vision.resilient_capture.

Focus areas:
- Backoff jitter: verifying ±25% jitter range is applied to sleep delays
- Generic discovery fallback when no USB IDs are configured
- Concurrent disconnect during capture (thread safety)
- Cumulative failure counter behavior after long success sequences
- Unrecoverable failure types (PERMISSION, UNSUPPORTED) halt reconnection
- Device path change warning when camera moves to a different /dev/videoN path
- Context manager (__enter__/__exit__) error paths
- Reconnection with a fallback camera (different USB ID than preferred)
"""

from __future__ import annotations

import contextlib
import logging
import threading
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from missy.vision.capture import CaptureError, CaptureResult, FailureType
from missy.vision.discovery import CameraDevice
from missy.vision.resilient_capture import ResilientCamera

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_device(
    path: str = "/dev/video0",
    name: str = "Test Cam",
    vendor: str = "046d",
    product: str = "085c",
    bus: str = "usb-0000:00:14.0-1",
) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id=vendor,
        product_id=product,
        bus_info=bus,
    )


def _success_result(path: str = "/dev/video0") -> CaptureResult:
    return CaptureResult(
        success=True,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        device_path=path,
    )


def _fail_result(
    failure_type: str = FailureType.UNKNOWN,
    error: str = "capture error",
) -> CaptureResult:
    return CaptureResult(success=False, error=error, failure_type=failure_type)


def _mock_disc(device: CameraDevice | None = None, *, valid: bool = True) -> MagicMock:
    """Return a fully-configured mock CameraDiscovery."""
    m = MagicMock()
    m.validate_device.return_value = valid
    m.find_by_usb_id.return_value = device
    m.find_preferred.return_value = device
    m.discover.return_value = [device] if device else []
    m.rediscover_device.return_value = device
    return m


def _mock_handle(is_open: bool = True, result: CaptureResult | None = None) -> MagicMock:
    h = MagicMock()
    h.is_open = is_open
    h.capture.return_value = result or _success_result()
    return h


# ---------------------------------------------------------------------------
# 1. Backoff jitter — verify sleep delay is within ±25% of base delay
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.random.random", return_value=0.0)
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_jitter_lower_bound_when_random_zero(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_random: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """When random() == 0.0 the jitter factor is 0.75, giving 75% of base delay."""
    # random() == 0.0 → jitter = 0.75 + 0.0*0.5 = 0.75
    device = _make_device()
    mock_get_disc.return_value = _mock_disc(None)  # device never found → loop exhausts
    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=4.0,
        backoff_factor=1.0,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = device
    mock_get_disc.return_value.validate_device.return_value = True

    cam.capture()

    # Sleep must have been called; the jittered value is 4.0 * 0.75 = 3.0
    assert mock_sleep.called
    slept = mock_sleep.call_args[0][0]
    assert abs(slept - 3.0) < 1e-9, f"Expected 3.0, got {slept}"


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.random.random", return_value=1.0)
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_jitter_upper_bound_when_random_one(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_random: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """When random() == 1.0 the jitter factor is 1.25, giving 125% of base delay."""
    device = _make_device()
    mock_get_disc.return_value = _mock_disc(None)
    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=4.0,
        backoff_factor=1.0,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = device
    mock_get_disc.return_value.validate_device.return_value = True

    cam.capture()

    slept = mock_sleep.call_args[0][0]
    assert abs(slept - 5.0) < 1e-9, f"Expected 5.0, got {slept}"


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.random.random", return_value=0.5)
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_jitter_midpoint_when_random_half(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_random: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """When random() == 0.5 the jitter factor is exactly 1.0 (no change)."""
    device = _make_device()
    mock_get_disc.return_value = _mock_disc(None)
    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=6.0,
        backoff_factor=1.0,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = device
    mock_get_disc.return_value.validate_device.return_value = True

    cam.capture()

    slept = mock_sleep.call_args[0][0]
    assert abs(slept - 6.0) < 1e-9, f"Expected 6.0, got {slept}"


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.random.random")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_jitter_always_within_25_percent_range(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_random: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """All sleep calls must fall within [0.75*delay, 1.25*delay]."""
    import random as _real_random_module

    # Capture a reference to the unpatched function *before* the mock replaces it
    _real_random = _real_random_module.Random().random

    base_delay = 2.0
    device = _make_device()
    mock_get_disc.return_value = _mock_disc(None)

    # Use real random to get a variety of values, avoiding recursion
    mock_random.side_effect = lambda: _real_random()

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=5,
        reconnect_delay=base_delay,
        backoff_factor=1.0,  # fixed delay so we can check each call
        max_delay=base_delay,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = device
    mock_get_disc.return_value.validate_device.return_value = True

    cam.capture()

    for c in mock_sleep.call_args_list:
        slept = c[0][0]
        assert base_delay * 0.75 <= slept <= base_delay * 1.25, (
            f"Sleep {slept} is outside ±25% range of {base_delay}"
        )


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.random.random", return_value=0.0)
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_jitter_applied_to_device_not_found_path(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_random: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Jitter is applied on the device-not-found branch (no USB IDs configured)."""
    mock_get_disc.return_value = _mock_disc(None)
    cam = ResilientCamera(
        # No USB IDs → generic fallback path
        max_reconnect_attempts=1,
        reconnect_delay=8.0,
        backoff_factor=1.0,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = _make_device()
    mock_get_disc.return_value.validate_device.return_value = True

    cam.capture()

    slept = mock_sleep.call_args[0][0]
    # 8.0 * 0.75 = 6.0
    assert abs(slept - 6.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. Generic discovery fallback when no USB IDs configured
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_generic_fallback_calls_discover_force(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Without USB IDs, reconnect loop calls discovery.discover(force=True)."""
    disc = _mock_disc(None)
    mock_get_disc.return_value = disc

    cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0.0)
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = _make_device()
    disc.validate_device.return_value = True

    cam.capture()

    # discover(force=True) must have been called at least once
    calls = [c for c in disc.discover.call_args_list if c == call(force=True)]
    assert len(calls) >= 1


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_generic_fallback_logs_debug_message(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without USB IDs, a debug log 'No USB IDs configured' is emitted."""
    disc = _mock_disc(None)
    mock_get_disc.return_value = disc

    cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.0)
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = _make_device()
    disc.validate_device.return_value = True

    with caplog.at_level(logging.DEBUG, logger="missy.vision.resilient_capture"):
        cam.capture()

    assert any("No USB IDs configured" in r.message for r in caplog.records)


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_generic_fallback_does_not_call_rediscover_device(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Without USB IDs, rediscover_device() must not be called (it needs vendor/product)."""
    disc = _mock_disc(None)
    mock_get_disc.return_value = disc

    cam = ResilientCamera(max_reconnect_attempts=2, reconnect_delay=0.0)
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = _make_device()
    disc.validate_device.return_value = True

    cam.capture()

    disc.rediscover_device.assert_not_called()


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_generic_fallback_succeeds_when_find_preferred_returns_device(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Without USB IDs, if find_preferred() finds a device reconnect succeeds."""
    device = _make_device()
    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.find_preferred.return_value = device
    disc.find_by_usb_id.return_value = None
    disc.discover.return_value = [device]
    disc.rediscover_device.return_value = None
    mock_get_disc.return_value = disc

    success = _success_result()
    handle = MagicMock()
    handle.is_open = True
    handle.capture.return_value = success
    mock_handle_cls.return_value = handle

    cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.0)
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device
    disc.validate_device.return_value = True

    result = cam.capture()

    assert result.success


# ---------------------------------------------------------------------------
# 3. Concurrent disconnect during capture (thread safety)
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_concurrent_disconnect_does_not_raise(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_health: MagicMock,
) -> None:
    """disconnect() called from another thread during capture must not raise."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def slow_capture() -> CaptureResult:
        barrier.wait()  # sync with disconnect thread
        return _success_result()

    handle = MagicMock()
    handle.is_open = True
    handle.capture.side_effect = slow_capture
    mock_handle_cls.return_value = handle

    cam = ResilientCamera(max_reconnect_attempts=0, reconnect_delay=0.0)
    cam._connected = True
    cam._handle = handle
    cam._current_device = device
    disc.validate_device.return_value = True

    def disconnect_thread() -> None:
        try:
            barrier.wait()
            cam.disconnect()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=disconnect_thread, daemon=True)
    t.start()

    with contextlib.suppress(Exception):
        cam.capture()

    t.join(timeout=5)
    assert not errors, f"Thread raised: {errors}"


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_multiple_threads_capture_simultaneously(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Multiple threads calling capture() concurrently must not raise or corrupt state."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = MagicMock()
    handle.is_open = True
    handle.capture.return_value = _success_result()
    mock_handle_cls.return_value = handle

    cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.0)
    cam._connected = True
    cam._handle = handle
    cam._current_device = device
    disc.validate_device.return_value = True

    results: list[CaptureResult] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            results.append(cam.capture())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"Thread(s) raised: {errors}"
    assert len(results) == 5


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_disconnect_during_reconnect_loop_is_safe(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_health: MagicMock,
) -> None:
    """disconnect() while _reconnect_and_capture is looping must not raise."""
    device = _make_device()
    disc = _mock_disc(None)  # always return None to keep loop running
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    errors: list[Exception] = []
    disconnect_called = threading.Event()

    original_sleep = __import__("time").sleep

    def slow_sleep(secs: float) -> None:
        disconnect_called.set()
        original_sleep(min(secs, 0.01))

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=3,
        reconnect_delay=0.05,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = device

    def disconnector() -> None:
        disconnect_called.wait(timeout=2)
        try:
            cam.disconnect()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=disconnector, daemon=True)
    t.start()

    with patch("missy.vision.resilient_capture.time.sleep", side_effect=slow_sleep), contextlib.suppress(Exception):
        cam.capture()

    t.join(timeout=5)
    assert not errors, f"Disconnector raised: {errors}"


# ---------------------------------------------------------------------------
# 4. Cumulative failure counter after long success sequences
# ---------------------------------------------------------------------------


def test_cumulative_failures_start_at_zero() -> None:
    cam = ResilientCamera()
    assert cam.cumulative_failures == 0


def test_cumulative_failures_do_not_reset_after_success() -> None:
    """Successes do NOT reset the cumulative failure counter."""
    cam = ResilientCamera()
    cam._cumulative_failures = 7

    # Simulate a success by calling capture with a working handle and discovery
    device = _make_device()
    disc = _mock_disc(device)
    handle = _mock_handle(result=_success_result())

    with patch("missy.vision.resilient_capture.get_discovery", return_value=disc), \
         patch("missy.vision.resilient_capture.get_health_monitor"), \
         patch("missy.vision.resilient_capture.CameraHandle", return_value=handle):
        cam._connected = True
        cam._handle = handle
        cam._current_device = device
        result = cam.capture()

    assert result.success
    # Counter must still be 7 — successes don't reset it
    assert cam.cumulative_failures == 7


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_cumulative_failures_accumulate_across_reconnect_attempts(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Each failed reconnect attempt increments the cumulative counter."""
    disc = _mock_disc(None)  # device never found
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=4,
        reconnect_delay=0.0,
    )
    cam._connected = True
    handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    handle.is_open = True
    cam._handle = handle
    cam._current_device = _make_device()

    cam.capture()

    # 1 initial failure + 4 reconnect failures = 5 total
    assert cam.cumulative_failures >= 5


def test_threshold_warning_logged_at_exactly_ten(caplog: pytest.LogCaptureFixture) -> None:
    """Warning is logged when cumulative failures reach the threshold (10)."""
    cam = ResilientCamera()
    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        for _ in range(10):
            cam._record_failure()

    assert cam.cumulative_failures == 10
    assert any("cumulative failures" in r.message for r in caplog.records)


def test_threshold_warning_logged_once_per_call_above_threshold(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warning is emitted on every _record_failure call once threshold is exceeded."""
    cam = ResilientCamera()
    cam._cumulative_failures = 9  # one below threshold

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        cam._record_failure()  # hits 10 — first warning
        cam._record_failure()  # hits 11 — second warning

    warning_msgs = [r for r in caplog.records if "cumulative failures" in r.message]
    assert len(warning_msgs) == 2


def test_cumulative_failures_independent_per_instance() -> None:
    """Two separate ResilientCamera instances have independent counters."""
    cam_a = ResilientCamera()
    cam_b = ResilientCamera()

    for _ in range(5):
        cam_a._record_failure()

    assert cam_a.cumulative_failures == 5
    assert cam_b.cumulative_failures == 0


# ---------------------------------------------------------------------------
# 5. Unrecoverable failure types halt reconnection
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.get_discovery")
def test_permission_failure_returns_immediately_no_reconnect(
    mock_get_disc: MagicMock,
    mock_health: MagicMock,
) -> None:
    """PERMISSION failure on initial capture must not trigger _reconnect_and_capture."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    cam = ResilientCamera(max_reconnect_attempts=5, reconnect_delay=0.0)
    handle = _mock_handle(result=_fail_result(FailureType.PERMISSION, "Permission denied"))
    cam._connected = True
    cam._handle = handle
    cam._current_device = device

    with patch.object(cam, "_reconnect_and_capture") as mock_reconnect:
        result = cam.capture()

    assert not result.success
    assert result.failure_type == FailureType.PERMISSION
    mock_reconnect.assert_not_called()


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.get_discovery")
def test_unsupported_failure_returns_immediately_no_reconnect(
    mock_get_disc: MagicMock,
    mock_health: MagicMock,
) -> None:
    """UNSUPPORTED failure must not trigger reconnection attempts."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    cam = ResilientCamera(max_reconnect_attempts=5, reconnect_delay=0.0)
    handle = _mock_handle(result=_fail_result(FailureType.UNSUPPORTED, "Format not supported"))
    cam._connected = True
    cam._handle = handle
    cam._current_device = device

    with patch.object(cam, "_reconnect_and_capture") as mock_reconnect:
        result = cam.capture()

    assert not result.success
    assert result.failure_type == FailureType.UNSUPPORTED
    mock_reconnect.assert_not_called()


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_permission_failure_during_reconnect_aborts_loop(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """PERMISSION failure during reconnect capture must stop the retry loop."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    perm_result = _fail_result(FailureType.PERMISSION, "Permission denied")
    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = perm_result
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=3,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device

    result = cam.capture()

    assert not result.success
    assert result.failure_type == FailureType.PERMISSION
    # Only one reconnect attempt was made (abort on PERMISSION)
    assert reconnect_handle.capture.call_count == 1


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_unsupported_failure_during_reconnect_aborts_loop(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """UNSUPPORTED failure during reconnect must also abort the retry loop."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    unsup_result = _fail_result(FailureType.UNSUPPORTED, "Bad format")
    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = unsup_result
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=4,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device

    result = cam.capture()

    assert not result.success
    assert result.failure_type == FailureType.UNSUPPORTED
    assert reconnect_handle.capture.call_count == 1


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_transient_failure_does_trigger_reconnect(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """A TRANSIENT failure IS allowed to trigger the reconnect loop."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    # Reconnect succeeds
    success = _success_result()
    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = success
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=2,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.TRANSIENT))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device

    result = cam.capture()

    assert result.success
    assert cam.total_reconnects == 1


# ---------------------------------------------------------------------------
# 6. Device path change warning
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_device_path_change_emits_warning(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the camera reappears at a different path, a warning is logged."""
    old_device = _make_device(path="/dev/video0")
    new_device = _make_device(path="/dev/video2")

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.find_by_usb_id.return_value = None
    disc.find_preferred.return_value = None
    # rediscover returns the new device (different path)
    disc.rediscover_device.return_value = new_device
    mock_get_disc.return_value = disc

    success = _success_result(path="/dev/video2")
    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = success
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = old_device

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        cam.capture()

    warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("/dev/video0" in t and "/dev/video2" in t for t in warning_texts), (
        f"Expected path-change warning, got: {warning_texts}"
    )


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_no_path_change_warning_when_same_path(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No path-change warning when device reappears at the same path."""
    device = _make_device(path="/dev/video0")

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.rediscover_device.return_value = device
    mock_get_disc.return_value = disc

    success = _success_result()
    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = success
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        cam.capture()

    path_change_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "path changed" in r.message.lower()
    ]
    assert not path_change_warnings


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_path_change_warning_contains_both_old_and_new_paths(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Path-change warning message must include both the old and new device paths."""
    old_device = _make_device(path="/dev/video1")
    new_device = _make_device(path="/dev/video4")

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.rediscover_device.return_value = new_device
    mock_get_disc.return_value = disc

    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = _success_result(path="/dev/video4")
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = old_device

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        cam.capture()

    matching = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING
        and "/dev/video1" in r.message
        and "/dev/video4" in r.message
    ]
    assert matching, "Warning must mention both old and new device paths"


# ---------------------------------------------------------------------------
# 7. Context manager __enter__ / __exit__ error handling
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_discovery")
def test_context_manager_exit_calls_disconnect_after_exception(
    mock_get_disc: MagicMock,
) -> None:
    """__exit__ must call disconnect() even when the body raises."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = _mock_handle()
    with patch("missy.vision.resilient_capture.CameraHandle", return_value=handle):
        cam = ResilientCamera()

        with pytest.raises(ValueError), cam:
            assert cam.is_connected
            raise ValueError("body error")

        # disconnect() should have been called → handle set to None
        assert cam._handle is None
        assert not cam.is_connected


@patch("missy.vision.resilient_capture.get_discovery")
def test_context_manager_enter_propagates_capture_error_when_no_camera(
    mock_get_disc: MagicMock,
) -> None:
    """When no camera is available, __enter__ propagates CaptureError."""
    disc = _mock_disc(None)
    mock_get_disc.return_value = disc

    cam = ResilientCamera()
    with pytest.raises(CaptureError, match="No camera"):
        cam.__enter__()


@patch("missy.vision.resilient_capture.get_discovery")
def test_context_manager_exit_returns_none_not_suppresses(
    mock_get_disc: MagicMock,
) -> None:
    """__exit__ must return None (falsy) so exceptions propagate from with-block."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = _mock_handle()
    with patch("missy.vision.resilient_capture.CameraHandle", return_value=handle):
        cam = ResilientCamera()
        cam.connect()
        retval = cam.__exit__(None, None, None)

    assert not retval


@patch("missy.vision.resilient_capture.get_discovery")
def test_context_manager_happy_path_connects_and_disconnects(
    mock_get_disc: MagicMock,
) -> None:
    """Normal use of context manager: connected inside, disconnected after."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = _mock_handle()
    connected_inside: list[bool] = []

    with patch("missy.vision.resilient_capture.CameraHandle", return_value=handle):
        cam = ResilientCamera()
        with cam:
            connected_inside.append(cam.is_connected)

    assert connected_inside == [True]
    assert not cam.is_connected


# ---------------------------------------------------------------------------
# 8. Reconnection with a fallback camera (different USB ID than preferred)
# ---------------------------------------------------------------------------


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_fallback_camera_logs_warning(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When reconnect finds a different USB ID than preferred, a warning is logged."""
    preferred_device = _make_device(vendor="046d", product="085c", name="Logitech C922x")
    fallback_device = _make_device(
        path="/dev/video1", vendor="0c45", product="6366", name="Generic USB Camera"
    )

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.rediscover_device.return_value = fallback_device
    mock_get_disc.return_value = disc

    success = _success_result(path="/dev/video1")
    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = success
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = preferred_device

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        result = cam.capture()

    assert result.success
    fallback_warnings = [
        r.message for r in caplog.records
        if r.levelno == logging.WARNING and "fallback" in r.message.lower()
    ]
    assert fallback_warnings, f"Expected fallback warning, records: {[r.message for r in caplog.records]}"


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_fallback_camera_increments_reconnect_counter(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Successful reconnect via a fallback camera increments total_reconnects."""
    preferred = _make_device(vendor="046d", product="085c")
    fallback = _make_device(path="/dev/video1", vendor="0c45", product="6366", name="Fallback")

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.rediscover_device.return_value = fallback
    mock_get_disc.return_value = disc

    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = _success_result(path="/dev/video1")
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = preferred

    cam.capture()

    assert cam.total_reconnects == 1


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_fallback_camera_no_warning_when_ids_match(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No fallback warning when reconnected device has the same vendor/product IDs."""
    device = _make_device(vendor="046d", product="085c")

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.rediscover_device.return_value = device  # same IDs
    mock_get_disc.return_value = disc

    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = _success_result()
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        cam.capture()

    fallback_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "fallback" in r.message.lower()
    ]
    assert not fallback_warnings


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_fallback_camera_name_in_warning_message(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback warning message includes the fallback camera's name."""
    preferred = _make_device(vendor="046d", product="085c")
    fallback = _make_device(
        path="/dev/video1",
        vendor="0c45",
        product="6366",
        name="NoName BudgetCam 3000",
    )

    disc = MagicMock()
    disc.validate_device.return_value = True
    disc.rediscover_device.return_value = fallback
    mock_get_disc.return_value = disc

    reconnect_handle = MagicMock()
    reconnect_handle.is_open = True
    reconnect_handle.capture.return_value = _success_result(path="/dev/video1")
    mock_handle_cls.return_value = reconnect_handle

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=1,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = preferred

    with caplog.at_level(logging.WARNING, logger="missy.vision.resilient_capture"):
        cam.capture()

    matching = [
        r.message for r in caplog.records
        if "NoName BudgetCam 3000" in r.message
    ]
    assert matching, "Fallback warning must include the fallback camera name"


# ---------------------------------------------------------------------------
# 9. Additional edge cases for completeness
# ---------------------------------------------------------------------------


def test_is_connected_false_when_handle_closed() -> None:
    """is_connected is False when the handle reports is_open=False."""
    cam = ResilientCamera()
    cam._connected = True
    handle = MagicMock()
    handle.is_open = False
    cam._handle = handle

    assert not cam.is_connected


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_health_monitor_record_capture_called_on_success(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_health: MagicMock,
) -> None:
    """A successful capture records the event in the health monitor."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = _mock_handle(result=_success_result())
    cam = ResilientCamera()
    cam._connected = True
    cam._handle = handle
    cam._current_device = device

    cam.capture()

    mock_health.return_value.record_capture.assert_called_once()
    kwargs = mock_health.return_value.record_capture.call_args[1]
    assert kwargs["success"] is True


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.get_discovery")
def test_health_monitor_record_capture_called_on_failure(
    mock_get_disc: MagicMock,
    mock_health: MagicMock,
) -> None:
    """A failed capture also records an event in the health monitor with success=False."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = _mock_handle(result=_fail_result(FailureType.PERMISSION))
    cam = ResilientCamera(max_reconnect_attempts=0, reconnect_delay=0.0)
    cam._connected = True
    cam._handle = handle
    cam._current_device = device

    cam.capture()

    calls = mock_health.return_value.record_capture.call_args_list
    failure_calls = [c for c in calls if c[1].get("success") is False]
    assert failure_calls


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_max_reconnect_zero_returns_failure_immediately(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """With max_reconnect_attempts=0 the failure message says 0 attempts."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    cam = ResilientCamera(
        preferred_vendor_id="046d",
        preferred_product_id="085c",
        max_reconnect_attempts=0,
        reconnect_delay=0.0,
    )
    cam._connected = True
    fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
    fail_handle.is_open = True
    cam._handle = fail_handle
    cam._current_device = device

    result = cam.capture()

    assert not result.success
    assert "0" in result.error
    mock_sleep.assert_not_called()


@patch("missy.vision.resilient_capture.get_health_monitor")
@patch("missy.vision.resilient_capture.time.sleep")
@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_backoff_delay_capped_at_max_delay(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
    mock_sleep: MagicMock,
    mock_health: MagicMock,
) -> None:
    """Exponential backoff must never exceed max_delay regardless of attempt count."""
    disc = _mock_disc(None)
    mock_get_disc.return_value = disc
    disc.validate_device.return_value = True

    with patch("missy.vision.resilient_capture.random.random", return_value=0.5):
        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=6,
            reconnect_delay=2.0,
            backoff_factor=2.0,
            max_delay=5.0,
        )
        cam._connected = True
        fail_handle = _mock_handle(result=_fail_result(FailureType.UNKNOWN))
        fail_handle.is_open = True
        cam._handle = fail_handle
        cam._current_device = _make_device()

        cam.capture()

    # With random=0.5 jitter factor is exactly 1.0, so each slept value == jittered delay
    # The jittered delay is min(delay*1.0, max_delay) = min(delay, 5.0)
    for c in mock_sleep.call_args_list:
        slept = c[0][0]
        assert slept <= 5.0 * 1.25, f"Sleep {slept} exceeds max_delay * max_jitter"


@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_open_device_resets_blank_detector(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
) -> None:
    """_open_device() must call reset_blank_detector() on the new handle."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    handle = MagicMock()
    handle.is_open = True
    mock_handle_cls.return_value = handle

    cam = ResilientCamera()
    cam._open_device(device)

    handle.reset_blank_detector.assert_called_once()


@patch("missy.vision.resilient_capture.get_discovery")
@patch("missy.vision.resilient_capture.CameraHandle")
def test_open_device_closes_old_handle_before_opening_new(
    mock_handle_cls: MagicMock,
    mock_get_disc: MagicMock,
) -> None:
    """_open_device() must close any existing handle before opening a new one."""
    device = _make_device()
    disc = _mock_disc(device)
    mock_get_disc.return_value = disc

    old_handle = MagicMock()
    old_handle.is_open = True
    new_handle = MagicMock()
    new_handle.is_open = True
    mock_handle_cls.return_value = new_handle

    cam = ResilientCamera()
    cam._handle = old_handle

    cam._open_device(device)

    old_handle.close.assert_called_once()
    assert cam._handle is new_handle
