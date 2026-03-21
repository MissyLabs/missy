"""Targeted tests for missy.vision.multi_camera.


Covers scenarios NOT exercised by test_multi_camera.py or
test_multi_camera_stress.py:

1.  _capture_one closed-handle guard: is_open=False before capture
2.  closed-handle result has the expected error message
3.  errors dict is absent the _global key when cameras exist (only failures)
4.  capture_all health monitor called with empty error string on success
5.  capture_all health monitor receives correct error string on failure
6.  capture_best delegates to capture_all exactly once
7.  add_camera open-failure leaves zero handles and zero devices
8.  discover_and_connect max_cameras=0 opens nothing
9.  discover_and_connect uses manager default config for each handle
10. device_count vs connected_devices divergence with closed handles
11. status() when _devices has no entry for a registered path (defensive)
12. MultiCaptureResult.errors only contains failed devices
13. MultiCaptureResult.errors is absent the key for successful devices
14. elapsed_ms is a float (not int)
15. all_succeeded False when results is empty but errors is non-empty
16. any_succeeded iterates results not errors
17. max_workers=1 serialises captures (pool created with 1 worker)
18. remove_camera then connected_devices is empty
19. context manager __exit__ does not suppress exceptions
20. discover_and_connect max_cameras=1 limits to one even with many discovered
21. discover_and_connect with all-fail open uses warning log, not exception
22. closed_handle_is_open guard returns failure with specific message text
23. capture_all empty-results dict when all captures hit the closed-handle guard
24. capture_best no cameras returns CaptureResult not MultiCaptureResult
25. add_camera custom config takes precedence over manager default
26. connected_devices snapshot is independent (list not view)
27. close_all two cameras: both handles closed even when first raises
28. errors value equals capture result error text verbatim
29. best_result None when every success has image=None
30. MultiCaptureResult field defaults are independent across instances
31. capture_all with one open + one closed handle: open one succeeds
32. discover_and_connect force=True always passed to discovery.discover
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from missy.vision.capture import CaptureConfig, CaptureResult
from missy.vision.discovery import CameraDevice
from missy.vision.multi_camera import MultiCameraManager, MultiCaptureResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _device(
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


def _success(
    path: str = "/dev/video0",
    width: int = 1920,
    height: int = 1080,
    image: np.ndarray | None = None,
) -> CaptureResult:
    img = image if image is not None else np.full((height, width, 3), 128, dtype=np.uint8)
    return CaptureResult(
        success=True,
        image=img,
        device_path=path,
        width=width,
        height=height,
        error="",
    )


def _failure(path: str = "/dev/video0", error: str = "capture failed") -> CaptureResult:
    return CaptureResult(success=False, device_path=path, error=error, image=None)


def _mock_handle(
    is_open: bool = True,
    capture_result: CaptureResult | None = None,
) -> MagicMock:
    handle = MagicMock()
    handle.is_open = is_open
    handle.capture.return_value = capture_result or _success()
    return handle


# ---------------------------------------------------------------------------
# 1-2. _capture_one closed-handle guard
# ---------------------------------------------------------------------------


class TestClosedHandleGuardInCaptureOne:
    """The is_open guard inside _capture_one returns a descriptive failure."""

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_closed_handle_before_capture_produces_failed_result(self, MockHandle, mock_get_hm):
        """A handle with is_open=False at capture time gives success=False."""
        handle = _mock_handle(is_open=False)
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_all()

        assert result.any_succeeded is False
        assert "/dev/video0" in result.results
        assert result.results["/dev/video0"].success is False

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_closed_handle_error_message_mentions_closed(self, MockHandle, mock_get_hm):
        """The failure message for a pre-capture-closed handle mentions 'closed'."""
        handle = _mock_handle(is_open=False)
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_all()

        error_msg = result.results["/dev/video0"].error
        assert "closed" in error_msg.lower()

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_closed_handle_device_path_preserved_in_result(self, MockHandle, mock_get_hm):
        handle = _mock_handle(is_open=False)
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_all()

        assert result.results["/dev/video0"].device_path == "/dev/video0"


# ---------------------------------------------------------------------------
# 3. errors dict does not contain _global when cameras exist
# ---------------------------------------------------------------------------


class TestErrorsDictStructure:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_global_error_absent_when_cameras_present_but_fail(self, MockHandle, mock_get_hm):
        """_global error only appears when there are zero cameras at all."""
        handle = _mock_handle(capture_result=_failure("/dev/video0", error="boom"))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_all()

        assert "_global" not in result.errors
        assert "/dev/video0" in result.errors

    def test_global_error_present_only_when_no_cameras(self):
        mgr = MultiCameraManager()
        result = mgr.capture_all()

        assert "_global" in result.errors

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_errors_dict_excludes_successful_device_keys(self, MockHandle, mock_get_hm):
        """Successful captures must not appear as keys in result.errors."""
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_all()

        assert "/dev/video0" not in result.errors

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_errors_value_matches_capture_result_error_verbatim(self, MockHandle, mock_get_hm):
        """result.errors[device] contains the exact text from CaptureResult.error."""
        specific_error = "ENODEV: device disconnected during capture"
        handle = _mock_handle(capture_result=_failure("/dev/video0", error=specific_error))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_all()

        assert result.errors["/dev/video0"] == specific_error


# ---------------------------------------------------------------------------
# 4-5. Health monitor argument validation
# ---------------------------------------------------------------------------


class TestHealthMonitorArguments:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_health_monitor_error_is_empty_string_on_success(self, MockHandle, mock_get_hm):
        """On a successful capture the error kwarg to record_capture must be ''."""
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_health = MagicMock()
        mock_get_hm.return_value = mock_health

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.capture_all()

        call_kwargs = mock_health.record_capture.call_args.kwargs
        assert call_kwargs.get("error") == ""

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_health_monitor_success_kwarg_is_true_for_success(self, MockHandle, mock_get_hm):
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_health = MagicMock()
        mock_get_hm.return_value = mock_health

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.capture_all()

        call_kwargs = mock_health.record_capture.call_args.kwargs
        assert call_kwargs.get("success") is True

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_health_monitor_success_kwarg_is_false_for_failure(self, MockHandle, mock_get_hm):
        handle = _mock_handle(capture_result=_failure("/dev/video0", error="timeout"))
        MockHandle.return_value = handle
        mock_health = MagicMock()
        mock_get_hm.return_value = mock_health

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.capture_all()

        call_kwargs = mock_health.record_capture.call_args.kwargs
        assert call_kwargs.get("success") is False

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_health_monitor_not_called_for_closed_handle_guard_path(self, MockHandle, mock_get_hm):
        """When the closed-handle guard fires, health monitor is NOT called
        because the code returns before reaching the record_capture call."""
        handle = _mock_handle(is_open=False)
        MockHandle.return_value = handle
        mock_health = MagicMock()
        mock_get_hm.return_value = mock_health

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.capture_all()

        # The is_open guard returns early, before health monitor is invoked
        mock_health.record_capture.assert_not_called()


# ---------------------------------------------------------------------------
# 6. capture_best delegates to capture_all exactly once
# ---------------------------------------------------------------------------


class TestCaptureBestDelegation:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_best_calls_capture_all_once(self, MockHandle, mock_get_hm):
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        with patch.object(mgr, "capture_all", wraps=mgr.capture_all) as spy:
            mgr.capture_best()
            spy.assert_called_once()

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_capture_best_returns_capture_result_not_multi(self, MockHandle, mock_get_hm):
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        result = mgr.capture_best()

        assert isinstance(result, CaptureResult)
        assert not isinstance(result, MultiCaptureResult)

    def test_capture_best_no_cameras_returns_capture_result(self):
        mgr = MultiCameraManager()
        result = mgr.capture_best()

        assert isinstance(result, CaptureResult)
        assert result.success is False


# ---------------------------------------------------------------------------
# 7. add_camera open-failure leaves clean state
# ---------------------------------------------------------------------------


class TestAddCameraOpenFailureCleanState:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_handles_and_devices_both_empty_after_open_failure(self, MockHandle):
        handle = MagicMock()
        handle.open.side_effect = OSError("no such device")
        MockHandle.return_value = handle

        mgr = MultiCameraManager()
        with pytest.raises(OSError):
            mgr.add_camera(_device("/dev/video0"))

        assert mgr._handles == {}
        assert mgr._devices == {}

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_device_count_zero_after_open_failure(self, MockHandle):
        handle = MagicMock()
        handle.open.side_effect = RuntimeError("busy")
        MockHandle.return_value = handle

        mgr = MultiCameraManager()
        with pytest.raises(RuntimeError):
            mgr.add_camera(_device("/dev/video0"))

        assert mgr.device_count == 0


# ---------------------------------------------------------------------------
# 8. discover_and_connect max_cameras=0
# ---------------------------------------------------------------------------


class TestDiscoverAndConnectMaxCamerasZero:
    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_max_cameras_zero_opens_nothing(self, MockHandle, mock_get_discovery):
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _device("/dev/video0"),
            _device("/dev/video1"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect(max_cameras=0)

        assert connected == []
        assert mgr.device_count == 0
        MockHandle.assert_not_called()


# ---------------------------------------------------------------------------
# 9. discover_and_connect uses manager default config
# ---------------------------------------------------------------------------


class TestDiscoverAndConnectUsesManagerConfig:
    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_handle_created_with_manager_default_config(self, MockHandle, mock_get_discovery):
        """CameraHandle must be created with the manager's _config, not None."""
        handle = _mock_handle()
        MockHandle.return_value = handle

        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [_device("/dev/video0")]
        mock_get_discovery.return_value = mock_discovery

        custom_cfg = CaptureConfig(width=640, height=480)
        mgr = MultiCameraManager(config=custom_cfg)
        mgr.discover_and_connect()

        # add_camera is called with no explicit config, so it falls back to mgr._config
        MockHandle.assert_called_once_with("/dev/video0", custom_cfg)


# ---------------------------------------------------------------------------
# 10. device_count vs connected_devices divergence
# ---------------------------------------------------------------------------


class TestDeviceCountVsConnectedDevices:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_device_count_counts_all_handles_open_and_closed(self, MockHandle):
        """device_count includes even closed handles; connected_devices does not."""
        open_h = _mock_handle(is_open=True)
        closed_h = _mock_handle(is_open=False)
        MockHandle.side_effect = [open_h, closed_h]

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.add_camera(_device("/dev/video1"))

        assert mgr.device_count == 2
        assert len(mgr.connected_devices) == 1

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_connected_devices_empty_when_all_closed(self, MockHandle):
        closed_h = _mock_handle(is_open=False)
        MockHandle.return_value = closed_h

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        assert mgr.connected_devices == []
        assert mgr.device_count == 1


# ---------------------------------------------------------------------------
# 11. status() defensive path when _devices has no entry
# ---------------------------------------------------------------------------


class TestStatusDefensive:
    def test_status_when_handle_registered_but_no_device_entry(self):
        """Manually inject a handle without a corresponding _devices entry
        to exercise the defensive fallback in status()."""
        mgr = MultiCameraManager()
        handle = _mock_handle(is_open=True)

        with mgr._lock:
            mgr._handles["/dev/video99"] = handle
            # Intentionally leave mgr._devices empty

        s = mgr.status()

        cam_info = s["cameras"]["/dev/video99"]
        assert cam_info["name"] == ""
        assert cam_info["vendor_id"] == ""
        assert cam_info["product_id"] == ""
        assert cam_info["is_open"] is True


# ---------------------------------------------------------------------------
# 12-13. MultiCaptureResult property correctness
# ---------------------------------------------------------------------------


class TestMultiCaptureResultProperties:
    def test_all_succeeded_false_when_results_empty_and_errors_present(self):
        """all_succeeded must be False when results={} even if errors is populated."""
        mcr = MultiCaptureResult(
            results={},
            errors={"_global": "No cameras connected"},
        )
        assert mcr.all_succeeded is False

    def test_any_succeeded_uses_results_not_errors(self):
        """any_succeeded reads results.values(), not errors. Errors with no results
        still yields False."""
        mcr = MultiCaptureResult(
            results={},
            errors={"/dev/video0": "some error"},
        )
        assert mcr.any_succeeded is False

    def test_successful_devices_empty_when_results_empty(self):
        mcr = MultiCaptureResult()
        assert mcr.successful_devices == []

    def test_failed_devices_empty_when_results_empty(self):
        mcr = MultiCaptureResult()
        assert mcr.failed_devices == []

    def test_best_result_none_when_all_success_have_none_image(self):
        """success=True with image=None must not appear in best_result."""
        r = CaptureResult(
            success=True, image=None, device_path="/dev/video0", width=1920, height=1080
        )
        mcr = MultiCaptureResult(results={"/dev/video0": r})
        assert mcr.best_result is None

    def test_field_defaults_independent_across_instances(self):
        """Mutable default_factory must produce separate dicts per instance."""
        a = MultiCaptureResult()
        b = MultiCaptureResult()
        a.results["/dev/video0"] = _success("/dev/video0")
        assert "/dev/video0" not in b.results
        a.errors["key"] = "value"
        assert "key" not in b.errors


# ---------------------------------------------------------------------------
# 14. elapsed_ms is float
# ---------------------------------------------------------------------------


class TestElapsedMsType:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_elapsed_ms_is_float(self, MockHandle, mock_get_hm):
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        result = mgr.capture_all()

        assert isinstance(result.elapsed_ms, float)

    def test_elapsed_ms_default_is_float(self):
        mcr = MultiCaptureResult()
        assert isinstance(mcr.elapsed_ms, float)


# ---------------------------------------------------------------------------
# 15. max_workers=1 serialises captures
# ---------------------------------------------------------------------------


class TestMaxWorkersOne:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.ThreadPoolExecutor", wraps=ThreadPoolExecutor)
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_max_workers_one_creates_pool_with_one_worker(self, MockHandle, MockPool, mock_get_hm):
        """max_workers=1 passes max_workers=1 to the ThreadPoolExecutor."""
        handle = _mock_handle(capture_result=_success("/dev/video0"))
        MockHandle.return_value = handle
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager(max_workers=1)
        mgr.add_camera(_device("/dev/video0"))
        mgr.capture_all()

        MockPool.assert_called_once_with(max_workers=1)


# ---------------------------------------------------------------------------
# 16. remove_camera then connected_devices is empty
# ---------------------------------------------------------------------------


class TestConnectedDevicesAfterRemove:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_connected_devices_empty_after_single_remove(self, MockHandle):
        MockHandle.return_value = _mock_handle(is_open=True)
        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        assert "/dev/video0" in mgr.connected_devices

        mgr.remove_camera("/dev/video0")

        assert mgr.connected_devices == []

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_connected_devices_snapshot_is_a_new_list(self, MockHandle):
        """connected_devices returns a new list each call (snapshot semantics)."""
        MockHandle.return_value = _mock_handle(is_open=True)
        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))

        snap1 = mgr.connected_devices
        snap2 = mgr.connected_devices

        assert snap1 is not snap2
        assert snap1 == snap2


# ---------------------------------------------------------------------------
# 17. context manager does not suppress exceptions
# ---------------------------------------------------------------------------


class TestContextManagerExceptionPropagation:
    def test_exception_inside_context_propagates(self):
        """__exit__ must not swallow exceptions."""
        sentinel = RuntimeError("must propagate")
        with pytest.raises(RuntimeError, match="must propagate"), MultiCameraManager():
            raise sentinel

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_close_all_called_even_when_exception_propagates(self, MockHandle):
        handle = _mock_handle()
        MockHandle.return_value = handle

        with pytest.raises(ValueError), MultiCameraManager() as mgr:
            mgr.add_camera(_device("/dev/video0"))
            raise ValueError("propagated")

        handle.close.assert_called_once()


# ---------------------------------------------------------------------------
# 18. discover_and_connect max_cameras=1 limits to one
# ---------------------------------------------------------------------------


class TestDiscoverAndConnectMaxCamerasOne:
    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_max_cameras_one_opens_only_first_discovered(self, MockHandle, mock_get_discovery):
        MockHandle.return_value = _mock_handle()
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _device("/dev/video0"),
            _device("/dev/video1"),
            _device("/dev/video2"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect(max_cameras=1)

        assert len(connected) == 1
        assert connected == ["/dev/video0"]
        assert mgr.device_count == 1


# ---------------------------------------------------------------------------
# 19. discover_and_connect all-fail open: logs warning, returns empty
# ---------------------------------------------------------------------------


class TestDiscoverAndConnectAllFailWarning:
    @patch("missy.vision.multi_camera.logger")
    @patch("missy.vision.multi_camera.get_discovery")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_warning_logged_for_each_failed_open(self, MockHandle, mock_get_discovery, mock_logger):
        bad_handle = MagicMock()
        bad_handle.open.side_effect = RuntimeError("permission denied")
        MockHandle.return_value = bad_handle

        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = [
            _device("/dev/video0"),
            _device("/dev/video1"),
        ]
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        connected = mgr.discover_and_connect()

        assert connected == []
        assert mock_logger.warning.call_count == 2


# ---------------------------------------------------------------------------
# 20. capture_all: one open handle + one closed handle
# ---------------------------------------------------------------------------


class TestMixedOpenClosedHandlesInCaptureAll:
    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_open_handle_succeeds_closed_handle_fails(self, MockHandle, mock_get_hm):
        open_handle = _mock_handle(is_open=True, capture_result=_success("/dev/video0"))
        closed_handle = _mock_handle(is_open=False)
        MockHandle.side_effect = [open_handle, closed_handle]
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.add_camera(_device("/dev/video1"))

        result = mgr.capture_all()

        assert result.any_succeeded is True
        assert result.all_succeeded is False
        assert "/dev/video0" in result.successful_devices
        assert "/dev/video1" in result.failed_devices

    @patch("missy.vision.multi_camera.get_health_monitor")
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_all_closed_handles_produces_empty_successes(self, MockHandle, mock_get_hm):
        closed_h0 = _mock_handle(is_open=False)
        closed_h1 = _mock_handle(is_open=False)
        MockHandle.side_effect = [closed_h0, closed_h1]
        mock_get_hm.return_value = MagicMock()

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.add_camera(_device("/dev/video1"))

        result = mgr.capture_all()

        assert result.any_succeeded is False
        assert len(result.results) == 2
        assert len(result.successful_devices) == 0


# ---------------------------------------------------------------------------
# 21. discover_and_connect always passes force=True
# ---------------------------------------------------------------------------


class TestDiscoverAndConnectForceTrue:
    @patch("missy.vision.multi_camera.get_discovery")
    def test_force_true_always_passed(self, mock_get_discovery):
        mock_discovery = MagicMock()
        mock_discovery.discover.return_value = []
        mock_get_discovery.return_value = mock_discovery

        mgr = MultiCameraManager()
        mgr.discover_and_connect()
        mgr.discover_and_connect()  # call twice

        # Both calls must use force=True
        assert mock_discovery.discover.call_count == 2
        for c in mock_discovery.discover.call_args_list:
            assert c == call(force=True)


# ---------------------------------------------------------------------------
# 22. close_all: second camera closes even when first raises
# ---------------------------------------------------------------------------


class TestCloseAllPartialFailure:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_second_handle_closed_when_first_close_raises(self, MockHandle):
        bad_handle = MagicMock()
        bad_handle.is_open = True
        bad_handle.close.side_effect = RuntimeError("driver panic")

        good_handle = _mock_handle()
        MockHandle.side_effect = [bad_handle, good_handle]

        mgr = MultiCameraManager()
        mgr.add_camera(_device("/dev/video0"))
        mgr.add_camera(_device("/dev/video1"))

        mgr.close_all()  # must not raise

        good_handle.close.assert_called_once()
        assert mgr.device_count == 0

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_close_all_still_clears_state_when_all_close_raise(self, MockHandle):
        for_path = ["/dev/video0", "/dev/video1"]
        handles = []
        for _ in for_path:
            h = MagicMock()
            h.is_open = True
            h.close.side_effect = RuntimeError("crash")
            handles.append(h)
        MockHandle.side_effect = handles

        mgr = MultiCameraManager()
        for p in for_path:
            mgr.add_camera(_device(p))

        mgr.close_all()

        assert mgr._handles == {}
        assert mgr._devices == {}


# ---------------------------------------------------------------------------
# 23. add_camera custom config takes precedence over manager default
# ---------------------------------------------------------------------------


class TestAddCameraCustomConfigPrecedence:
    @patch("missy.vision.multi_camera.CameraHandle")
    def test_per_camera_config_overrides_manager_config(self, MockHandle):
        MockHandle.return_value = _mock_handle()
        manager_cfg = CaptureConfig(width=1920, height=1080)
        per_camera_cfg = CaptureConfig(width=320, height=240)

        mgr = MultiCameraManager(config=manager_cfg)
        mgr.add_camera(_device("/dev/video0"), config=per_camera_cfg)

        MockHandle.assert_called_once_with("/dev/video0", per_camera_cfg)

    @patch("missy.vision.multi_camera.CameraHandle")
    def test_no_per_camera_config_uses_manager_default(self, MockHandle):
        MockHandle.return_value = _mock_handle()
        manager_cfg = CaptureConfig(width=640, height=480)

        mgr = MultiCameraManager(config=manager_cfg)
        mgr.add_camera(_device("/dev/video0"))

        MockHandle.assert_called_once_with("/dev/video0", manager_cfg)
