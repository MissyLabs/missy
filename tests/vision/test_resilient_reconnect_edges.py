"""Resilient capture and security tests.


Covers under-tested paths in:
- ResilientCamera reconnection logic (device path change, USB ID mismatch)
- ResilientCamera cumulative failure warnings
- ResilientCamera unrecoverable failure types
- Pipeline CLAHE processing edge cases
- Provider format input validation
- Intent classifier edge cases
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import CaptureResult, FailureType
from missy.vision.discovery import CameraDevice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_device(path: str = "/dev/video0", name: str = "Test Cam",
                 vendor: str = "046d", product: str = "085c") -> CameraDevice:
    return CameraDevice(path, name, vendor, product, "usb-1")


def _mock_discovery(device: CameraDevice | None = None):
    mock = MagicMock()
    mock.validate_device.return_value = True
    mock.find_by_usb_id.return_value = device
    mock.find_preferred.return_value = device
    mock.discover.return_value = [device] if device else []
    mock.rediscover_device.return_value = device
    return mock


# ---------------------------------------------------------------------------
# ResilientCamera reconnection
# ---------------------------------------------------------------------------


class TestResilientCameraReconnection:
    """ResilientCamera reconnection warnings and device switching."""

    def test_device_path_change_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """When device path changes during reconnection, warning is logged."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=1,
            reconnect_delay=0.0,
        )

        old_device = _make_device("/dev/video0")
        new_device = _make_device("/dev/video2")

        # Set up initial device
        cam._current_device = old_device
        cam._connected = False

        mock_disc = _mock_discovery(new_device)
        mock_handle = MagicMock()
        mock_handle.capture.return_value = CaptureResult(success=True, image=np.zeros((10, 10, 3), dtype=np.uint8))
        mock_handle._blank_detector = None

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.CameraHandle", return_value=mock_handle),
            patch("missy.vision.resilient_capture.get_health_monitor"),
            caplog.at_level(logging.WARNING),
        ):
            result = cam._reconnect_and_capture()

        assert result.success
        assert any("path changed" in r.message for r in caplog.records)

    def test_usb_id_mismatch_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """When reconnection finds a different USB ID, warning is logged."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=1,
            reconnect_delay=0.0,
        )
        cam._current_device = _make_device("/dev/video0")
        cam._connected = False

        # Return a different device (different USB ID)
        different_device = _make_device("/dev/video0", "Other Cam", "1234", "5678")
        mock_disc = _mock_discovery(different_device)

        mock_handle = MagicMock()
        mock_handle.capture.return_value = CaptureResult(success=True, image=np.zeros((10, 10, 3), dtype=np.uint8))
        mock_handle._blank_detector = None

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.CameraHandle", return_value=mock_handle),
            patch("missy.vision.resilient_capture.get_health_monitor"),
            caplog.at_level(logging.WARNING),
        ):
            result = cam._reconnect_and_capture()

        assert result.success
        assert any("fallback camera" in r.message for r in caplog.records)

    def test_cumulative_failure_threshold_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """After 10 cumulative failures, a reliability warning is logged."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera()
        cam._cumulative_failures = 9

        with caplog.at_level(logging.WARNING):
            cam._record_failure()

        assert cam._cumulative_failures == 10
        assert any("unreliable" in r.message for r in caplog.records)

    def test_unrecoverable_failure_stops_reconnection(self) -> None:
        """PERMISSION failure should stop reconnection attempts."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=3,
            reconnect_delay=0.0,
        )
        cam._connected = False

        device = _make_device()
        mock_disc = _mock_discovery(device)

        mock_handle = MagicMock()
        mock_handle.capture.return_value = CaptureResult(
            success=False,
            error="Permission denied",
            failure_type=FailureType.PERMISSION,
        )
        mock_handle._blank_detector = None

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.CameraHandle", return_value=mock_handle),
            patch("missy.vision.resilient_capture.get_health_monitor"),
        ):
            result = cam._reconnect_and_capture()

        assert not result.success
        assert result.failure_type == FailureType.PERMISSION
        # Should have stopped after first attempt, not retried 3 times
        assert cam.cumulative_failures <= 2

    def test_all_reconnect_attempts_fail(self) -> None:
        """When all attempts fail, appropriate error message returned."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(
            max_reconnect_attempts=2,
            reconnect_delay=0.0,
        )
        cam._connected = False

        mock_disc = _mock_discovery(None)

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.get_health_monitor"),
        ):
            result = cam._reconnect_and_capture()

        assert not result.success
        assert "2 attempts" in result.error

    def test_capture_triggers_validation_check(self) -> None:
        """capture() validates device before attempting read."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera()
        device = _make_device()
        cam._current_device = device
        cam._connected = True

        mock_handle = MagicMock()
        mock_handle.is_open = True
        mock_handle.capture.return_value = CaptureResult(
            success=True, image=np.zeros((10, 10, 3), dtype=np.uint8),
            device_path="/dev/video0",
        )
        cam._handle = mock_handle

        mock_disc = _mock_discovery(device)
        mock_disc.validate_device.return_value = True

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.get_health_monitor"),
        ):
            result = cam.capture()

        assert result.success
        mock_disc.validate_device.assert_called_once_with(device)

    def test_invalid_device_triggers_reconnect(self) -> None:
        """When validate_device returns False, capture reconnects."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=1,
            reconnect_delay=0.0,
        )
        device = _make_device()
        cam._current_device = device
        cam._connected = True

        mock_handle = MagicMock()
        mock_handle.is_open = True
        cam._handle = mock_handle

        # First call validates current device as invalid
        # Then reconnection finds a new device
        mock_disc = MagicMock()
        mock_disc.validate_device.return_value = False
        mock_disc.rediscover_device.return_value = device

        new_handle = MagicMock()
        new_handle.capture.return_value = CaptureResult(
            success=True, image=np.zeros((10, 10, 3), dtype=np.uint8),
            device_path="/dev/video0",
        )
        new_handle._blank_detector = None

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.CameraHandle", return_value=new_handle),
            patch("missy.vision.resilient_capture.get_health_monitor"),
        ):
            result = cam.capture()

        assert result.success

    def test_context_manager(self) -> None:
        """ResilientCamera works as a context manager."""
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera()
        device = _make_device()

        mock_disc = _mock_discovery(device)
        mock_handle = MagicMock()
        mock_handle._blank_detector = None

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("missy.vision.resilient_capture.CameraHandle", return_value=mock_handle),
            patch("missy.vision.resilient_capture.get_health_monitor"),
            cam,
        ):
            assert cam.is_connected

        assert not cam.is_connected


# ---------------------------------------------------------------------------
# Pipeline edge cases
# ---------------------------------------------------------------------------


class TestPipelineEdgeCases:
    """ImagePipeline edge cases."""

    def test_process_returns_ndarray(self) -> None:
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = pipeline.process(img)
        assert isinstance(result, np.ndarray)

    def test_process_preserves_content(self) -> None:
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = pipeline.process(img)
        # Should still be a valid image
        assert result.ndim >= 2

    def test_quality_assessment_returns_dict(self) -> None:
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        quality = pipeline.assess_quality(img)
        assert isinstance(quality, dict)
        assert "overall" in quality or "brightness" in quality or len(quality) > 0

    def test_process_very_small_image(self) -> None:
        from missy.vision.pipeline import ImagePipeline

        pipeline = ImagePipeline()
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = pipeline.process(img)
        assert result is not None

    def test_process_large_image_resized(self) -> None:
        from missy.vision.pipeline import ImagePipeline, PipelineConfig

        config = PipelineConfig(max_dimension=1920)
        pipeline = ImagePipeline(config)
        img = np.full((4000, 6000, 3), 128, dtype=np.uint8)
        result = pipeline.process(img)
        assert max(result.shape[0], result.shape[1]) <= 1920


# ---------------------------------------------------------------------------
# Intent classifier edge cases
# ---------------------------------------------------------------------------


class TestIntentClassifierEdgeCases:
    """VisionIntentClassifier edge cases and boundary conditions."""

    def test_empty_input(self) -> None:
        from missy.vision.intent import ActivationDecision, classify_vision_intent

        result = classify_vision_intent("")
        assert result.decision != ActivationDecision.ACTIVATE

    def test_non_vision_request(self) -> None:
        from missy.vision.intent import ActivationDecision, classify_vision_intent

        result = classify_vision_intent("What is the weather today?")
        assert result.decision != ActivationDecision.ACTIVATE

    def test_clear_vision_request(self) -> None:
        from missy.vision.intent import ActivationDecision, classify_vision_intent

        result = classify_vision_intent("Missy, look at this painting and tell me what you think")
        assert result.decision == ActivationDecision.ACTIVATE
        assert result.confidence > 0.5

    def test_puzzle_intent_detected(self) -> None:
        from missy.vision.intent import classify_vision_intent

        result = classify_vision_intent("Can you help me figure out where this puzzle piece goes?")
        assert result.suggested_mode in ("puzzle", "general")

    def test_painting_intent_detected(self) -> None:
        from missy.vision.intent import classify_vision_intent

        result = classify_vision_intent("How can I improve this painting?")
        assert result.confidence > 0.3

    def test_whitespace_only_input(self) -> None:
        from missy.vision.intent import ActivationDecision, classify_vision_intent

        result = classify_vision_intent("   \t\n  ")
        assert result.decision != ActivationDecision.ACTIVATE

    def test_very_long_input(self) -> None:
        from missy.vision.intent import classify_vision_intent

        # Should not crash on very long input
        long_text = "look at this " * 1000
        result = classify_vision_intent(long_text)
        assert result is not None


# ---------------------------------------------------------------------------
# Provider format validation
# ---------------------------------------------------------------------------


class TestProviderFormatValidation:
    """provider_format.py input validation."""

    _DUMMY_B64 = "iVBORw0KGgo="  # minimal PNG stub

    def test_format_anthropic_with_valid_image(self) -> None:
        from missy.vision.provider_format import build_vision_message

        result = build_vision_message("anthropic", self._DUMMY_B64, "Describe this")
        assert result is not None
        assert result["role"] == "user"

    def test_format_openai_with_valid_image(self) -> None:
        from missy.vision.provider_format import build_vision_message

        result = build_vision_message("openai", self._DUMMY_B64, "Describe this")
        assert result is not None
        assert result["role"] == "user"

    def test_format_unknown_provider_fallback(self) -> None:
        from missy.vision.provider_format import format_image_for_provider

        # Unknown provider should fall back to anthropic format
        result = format_image_for_provider("unknown", self._DUMMY_B64)
        assert result is not None

    def test_format_with_empty_prompt_raises(self) -> None:
        from missy.vision.provider_format import build_vision_message

        with pytest.raises(ValueError, match="prompt"):
            build_vision_message("anthropic", self._DUMMY_B64, "")

    def test_format_anthropic_structure(self) -> None:
        from missy.vision.provider_format import format_image_for_anthropic

        result = format_image_for_anthropic(self._DUMMY_B64)
        assert result["type"] == "image"
        assert "source" in result

    def test_format_openai_structure(self) -> None:
        from missy.vision.provider_format import format_image_for_openai

        result = format_image_for_openai(self._DUMMY_B64)
        assert result["type"] == "image_url"
        assert "image_url" in result
