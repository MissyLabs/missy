"""Tests for missy.vision.doctor — vision diagnostics."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.vision.doctor import DiagnosticResult, DoctorReport, VisionDoctor

# ---------------------------------------------------------------------------
# DiagnosticResult tests
# ---------------------------------------------------------------------------


class TestDiagnosticResult:
    def test_passed(self):
        r = DiagnosticResult(name="test", passed=True, message="OK")
        assert r.passed is True

    def test_failed(self):
        r = DiagnosticResult(name="test", passed=False, message="Failed", severity="error")
        assert r.passed is False
        assert r.severity == "error"


# ---------------------------------------------------------------------------
# DoctorReport tests
# ---------------------------------------------------------------------------


class TestDoctorReport:
    def test_empty_report(self):
        report = DoctorReport()
        assert report.passed == 0
        assert report.failed == 0
        assert report.overall_healthy is True

    def test_add_passed(self):
        report = DoctorReport()
        report.add(DiagnosticResult(name="a", passed=True, message="OK"))
        assert report.passed == 1
        assert report.failed == 0
        assert report.overall_healthy is True

    def test_add_error(self):
        report = DoctorReport()
        report.add(DiagnosticResult(name="a", passed=False, message="Bad", severity="error"))
        assert report.failed == 1
        assert report.errors == 1
        assert report.overall_healthy is False

    def test_add_warning(self):
        report = DoctorReport()
        report.add(DiagnosticResult(name="a", passed=False, message="Warn", severity="warning"))
        assert report.warnings == 1
        assert report.overall_healthy is True  # warnings don't mark unhealthy


# ---------------------------------------------------------------------------
# VisionDoctor tests
# ---------------------------------------------------------------------------


class TestVisionDoctor:
    def setup_method(self):
        self.doctor = VisionDoctor()

    def test_check_numpy(self):
        result = self.doctor.check_numpy()
        assert result.passed is True
        assert "NumPy" in result.message

    @patch("missy.vision.doctor.grp")
    @patch("missy.vision.doctor.os")
    def test_check_video_group_member(self, mock_os, mock_grp):
        mock_os.getlogin.return_value = "testuser"
        # Simulate membership via supplementary groups (os.getgroups returns video gid).
        mock_os.getgroups.return_value = [44, 1000]
        mock_os.getegid.return_value = 1000
        mock_os.environ = {}

        mock_group = MagicMock()
        mock_group.gr_gid = 44
        mock_grp.getgrnam.return_value = mock_group

        result = self.doctor.check_video_group()
        assert result.passed is True

    @patch("missy.vision.doctor.grp")
    @patch("missy.vision.doctor.os")
    def test_check_video_group_not_member(self, mock_os, mock_grp):
        mock_os.getlogin.return_value = "testuser"
        # Simulate no membership: supplementary groups and egid don't include video gid.
        mock_os.getgroups.return_value = [1000, 1001]
        mock_os.getegid.return_value = 1000
        mock_os.environ = {}

        mock_group = MagicMock()
        mock_group.gr_gid = 44
        mock_grp.getgrnam.return_value = mock_group

        result = self.doctor.check_video_group()
        assert result.passed is False
        assert "usermod" in result.message

    @patch("missy.vision.doctor.grp")
    @patch("missy.vision.doctor.os")
    def test_check_video_group_no_group(self, mock_os, mock_grp):
        mock_os.getlogin.return_value = "testuser"
        mock_os.environ = {}
        mock_grp.getgrnam.side_effect = KeyError("video")

        result = self.doctor.check_video_group()
        assert result.passed is False

    def test_check_video_devices_no_dev(self):
        """Test with a path that has no video devices."""
        with patch("missy.vision.doctor.Path") as mock_path:
            mock_dev = MagicMock()
            mock_dev.glob.return_value = []
            mock_path.return_value = mock_dev

            result = self.doctor.check_video_devices()
            # On a real system without cameras, this would fail
            # We just verify it returns a DiagnosticResult
            assert isinstance(result, DiagnosticResult)

    def test_check_sysfs_no_dir(self):
        with patch("missy.vision.doctor.Path") as mock_path:
            mock_sysfs = MagicMock()
            mock_sysfs.exists.return_value = False
            mock_path.return_value = mock_sysfs

            result = self.doctor.check_sysfs()
            assert isinstance(result, DiagnosticResult)

    @patch("missy.vision.doctor.shutil")
    def test_check_screenshot_tools_found(self, mock_shutil):
        mock_shutil.which.side_effect = lambda t: "/usr/bin/scrot" if t == "scrot" else None

        result = self.doctor.check_screenshot_tools()
        assert result.passed is True
        assert "scrot" in result.message

    @patch("missy.vision.doctor.shutil")
    def test_check_screenshot_tools_none(self, mock_shutil):
        mock_shutil.which.return_value = None

        result = self.doctor.check_screenshot_tools()
        assert result.passed is False

    def test_check_opencv(self):
        """OpenCV check — passes if opencv is installed."""
        result = self.doctor.check_opencv()
        # In test environment, opencv may or may not be installed
        assert isinstance(result, DiagnosticResult)

    def test_run_all(self):
        """run_all should return a complete report."""
        report = self.doctor.run_all()
        assert isinstance(report, DoctorReport)
        assert len(report.results) >= 5  # at least the non-capture checks
