"""Tests for CLI vision commands: devices, capture, inspect, review, doctor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from missy.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


# ---------------------------------------------------------------------------
# missy vision devices
# ---------------------------------------------------------------------------


class TestVisionDevices:
    def test_no_cameras_detected(self, runner: CliRunner):
        with patch("missy.vision.discovery.CameraDiscovery") as MockDisc:
            inst = MagicMock()
            inst.discover.return_value = []
            MockDisc.return_value = inst

            result = runner.invoke(cli, ["vision", "devices"])
            assert result.exit_code == 0
            assert "No cameras detected" in result.output

    def test_cameras_listed(self, runner: CliRunner):
        mock_cam = MagicMock()
        mock_cam.device_path = "/dev/video0"
        mock_cam.name = "Logitech C922x"
        mock_cam.usb_id = "046d:085c"
        mock_cam.bus_info = "usb-0000:00:14.0-1"

        with patch("missy.vision.discovery.CameraDiscovery") as MockDisc, \
             patch("missy.vision.discovery.KNOWN_CAMERAS", {"046d:085c": "C922x Pro Stream"}):
            inst = MagicMock()
            inst.discover.return_value = [mock_cam]
            inst.find_preferred.return_value = mock_cam
            MockDisc.return_value = inst

            result = runner.invoke(cli, ["vision", "devices"])
            assert result.exit_code == 0
            assert "Logitech C922x" in result.output
            assert "Preferred camera" in result.output

    def test_troubleshooting_hints(self, runner: CliRunner):
        with patch("missy.vision.discovery.CameraDiscovery") as MockDisc:
            inst = MagicMock()
            inst.discover.return_value = []
            MockDisc.return_value = inst

            result = runner.invoke(cli, ["vision", "devices"])
            assert "lsusb" in result.output
            assert "video group" in result.output or "video" in result.output


# ---------------------------------------------------------------------------
# missy vision capture
# ---------------------------------------------------------------------------


class TestVisionCapture:
    def test_no_camera_exits_error(self, runner: CliRunner):
        with patch("missy.vision.discovery.find_preferred_camera", return_value=None):
            result = runner.invoke(cli, ["vision", "capture"])
            assert result.exit_code != 0

    def test_capture_with_device(self, runner: CliRunner, tmp_path: Path):
        out_file = str(tmp_path / "test.jpg")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.width = 1920
        mock_result.height = 1080
        mock_result.image = np.full((1080, 1920, 3), 128, dtype=np.uint8)

        with patch("missy.vision.capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            inst.capture_to_file.return_value = mock_result
            MockHandle.return_value = inst

            result = runner.invoke(cli, [
                "vision", "capture",
                "--device", "/dev/video0",
                "--output", out_file,
            ])
            assert result.exit_code == 0
            assert "Captured" in result.output

    def test_capture_best_mode(self, runner: CliRunner, tmp_path: Path):
        out_file = str(tmp_path / "best.jpg")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.width = 1920
        mock_result.height = 1080
        mock_result.image = np.full((1080, 1920, 3), 128, dtype=np.uint8)

        with patch("missy.vision.capture.CameraHandle") as MockHandle, \
             patch("cv2.imwrite", return_value=True), \
             patch("cv2.IMWRITE_JPEG_QUALITY", 1):
            inst = MagicMock()
            inst.capture_best.return_value = mock_result
            MockHandle.return_value = inst

            result = runner.invoke(cli, [
                "vision", "capture",
                "--device", "/dev/video0",
                "--output", out_file,
                "--best",
            ])
            assert result.exit_code == 0
            assert "Best frame" in result.output

    def test_capture_burst_mode(self, runner: CliRunner, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.width = 640
        mock_result.height = 480
        mock_result.image = np.full((480, 640, 3), 128, dtype=np.uint8)

        with patch("missy.vision.capture.CameraHandle") as MockHandle, \
             patch("cv2.imwrite", return_value=True), \
             patch("cv2.IMWRITE_JPEG_QUALITY", 1):
            inst = MagicMock()
            inst.capture_burst.return_value = [mock_result, mock_result, mock_result]
            MockHandle.return_value = inst

            result = runner.invoke(cli, [
                "vision", "capture",
                "--device", "/dev/video0",
                "--burst",
                "--count", "3",
            ])
            assert result.exit_code == 0
            assert "Burst complete" in result.output

    def test_capture_failure_shows_error(self, runner: CliRunner, tmp_path: Path):
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Device busy"

        with patch("missy.vision.capture.CameraHandle") as MockHandle:
            inst = MagicMock()
            inst.capture_to_file.return_value = mock_result
            MockHandle.return_value = inst

            result = runner.invoke(cli, [
                "vision", "capture",
                "--device", "/dev/video0",
                "--output", str(tmp_path / "test.jpg"),
            ])
            assert "Failed" in result.output


# ---------------------------------------------------------------------------
# missy vision doctor
# ---------------------------------------------------------------------------


class TestVisionDoctor:
    def _make_report(self, results, passed=1, warnings=0, errors=0, healthy=True):
        report = MagicMock()
        report.results = results
        report.passed = passed
        report.warnings = warnings
        report.errors = errors
        report.overall_healthy = healthy
        return report

    def test_doctor_runs(self, runner: CliRunner):
        mock_check = MagicMock()
        mock_check.name = "OpenCV import"
        mock_check.passed = True
        mock_check.message = "OK"
        mock_check.severity = "info"
        mock_check.details = {}

        with patch("missy.vision.doctor.VisionDoctor") as MockDoctor:
            inst = MagicMock()
            inst.run_all.return_value = self._make_report([mock_check])
            MockDoctor.return_value = inst

            result = runner.invoke(cli, ["vision", "doctor"])
            assert result.exit_code == 0
            assert "PASS" in result.output
            assert "healthy" in result.output

    def test_doctor_shows_failures(self, runner: CliRunner):
        pass_check = MagicMock()
        pass_check.name = "OpenCV import"
        pass_check.passed = True
        pass_check.message = "OK"
        pass_check.severity = "info"
        pass_check.details = {}

        fail_check = MagicMock()
        fail_check.name = "Camera access"
        fail_check.passed = False
        fail_check.message = "Permission denied"
        fail_check.severity = "error"
        fail_check.details = {"hint": "Add user to video group"}

        with patch("missy.vision.doctor.VisionDoctor") as MockDoctor:
            inst = MagicMock()
            inst.run_all.return_value = self._make_report(
                [pass_check, fail_check], passed=1, errors=1, healthy=False
            )
            MockDoctor.return_value = inst

            result = runner.invoke(cli, ["vision", "doctor"])
            assert result.exit_code == 0
            assert "FAIL" in result.output
            assert "issues" in result.output


# ---------------------------------------------------------------------------
# missy vision inspect
# ---------------------------------------------------------------------------


class TestVisionInspect:
    def test_inspect_file(self, runner: CliRunner, tmp_path: Path):
        # Create a dummy image file
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        mock_frame = MagicMock()
        mock_frame.image = np.full((100, 100, 3), 128, dtype=np.uint8)

        with patch("missy.vision.sources.FileSource") as MockSource, \
             patch("missy.vision.pipeline.ImagePipeline") as MockPipeline:
            src_inst = MagicMock()
            src_inst.acquire.return_value = mock_frame
            MockSource.return_value = src_inst

            pipe_inst = MagicMock()
            pipe_inst.assess_quality.return_value = {
                "width": 100,
                "height": 100,
                "brightness": 128.0,
                "contrast": 40.0,
                "sharpness": 80.0,
                "quality": "good",
                "issues": [],
            }
            MockPipeline.return_value = pipe_inst

            result = runner.invoke(cli, [
                "vision", "inspect",
                "--file", str(img_path),
            ])
            # Should not crash; specifics depend on CLI output format
            assert result.exit_code == 0


# ---------------------------------------------------------------------------
# missy vision review
# ---------------------------------------------------------------------------


class TestVisionReview:
    def test_review_mode_choices(self, runner: CliRunner):
        """Invalid mode should be rejected by Click."""
        result = runner.invoke(cli, [
            "vision", "review",
            "--mode", "invalid_mode",
        ])
        assert result.exit_code != 0

    def test_review_valid_modes(self):
        """All 4 modes should be accepted by Click."""
        runner = CliRunner()
        for mode in ("general", "puzzle", "painting", "inspection"):
            # Just test that Click doesn't reject the mode;
            # actual execution will fail without camera, which is fine
            result = runner.invoke(cli, [
                "vision", "review",
                "--mode", mode,
                "--device", "/dev/nonexistent",
            ])
            # It's OK if it exits with error (no camera), but it shouldn't be
            # a Click usage error (exit code 2)
            assert result.exit_code != 2 or mode not in result.output
