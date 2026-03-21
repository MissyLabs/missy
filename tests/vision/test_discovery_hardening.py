"""Tests for CameraDiscovery hardening: rediscover_device and validate_device."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.vision.discovery import CameraDevice, CameraDiscovery


def _make_device(
    path: str = "/dev/video0",
    name: str = "Test Camera",
    vendor_id: str = "046d",
    product_id: str = "085c",
) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id=vendor_id,
        product_id=product_id,
        bus_info="usb-0000:00:14.0-1",
    )


class TestRediscoverDevice:
    """Tests for CameraDiscovery.rediscover_device."""

    def test_finds_device_immediately(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        device = _make_device()
        with (
            patch.object(disc, "discover", return_value=[device]),
            patch.object(disc, "find_by_usb_id", return_value=device),
        ):
            result = disc.rediscover_device("046d", "085c")
            assert result is not None
            assert result.device_path == "/dev/video0"

    def test_retries_on_not_found(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        device = _make_device()
        call_count = 0

        def mock_find(vid, pid):
            nonlocal call_count
            call_count += 1
            return device if call_count >= 3 else None

        with (
            patch.object(disc, "discover", return_value=[]),
            patch.object(disc, "find_by_usb_id", side_effect=mock_find),
            patch("time.sleep"),
        ):
            result = disc.rediscover_device("046d", "085c", max_attempts=5, delay=0.01)
            assert result is not None
            assert call_count == 3

    def test_returns_none_after_max_attempts(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        with (
            patch.object(disc, "discover", return_value=[]),
            patch.object(disc, "find_by_usb_id", return_value=None),
            patch("time.sleep"),
        ):
            result = disc.rediscover_device("046d", "085c", max_attempts=3, delay=0.01)
            assert result is None

    def test_logs_path_change(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        new_device = _make_device(path="/dev/video2")
        with (
            patch.object(disc, "discover", return_value=[new_device]),
            patch.object(disc, "find_by_usb_id", return_value=new_device),
        ):
            result = disc.rediscover_device("046d", "085c", old_path="/dev/video0")
            assert result is not None
            assert result.device_path == "/dev/video2"

    def test_single_attempt_no_sleep(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        with (
            patch.object(disc, "discover", return_value=[]),
            patch.object(disc, "find_by_usb_id", return_value=None),
            patch("time.sleep") as mock_sleep,
        ):
            disc.rediscover_device("046d", "085c", max_attempts=1, delay=1.0)
            mock_sleep.assert_not_called()


class TestValidateDevice:
    """Tests for CameraDiscovery.validate_device."""

    def test_valid_device(self, tmp_path: Path) -> None:
        # Set up sysfs structure
        sysfs_base = tmp_path / "sysfs"
        sysfs_base.mkdir()
        video0 = sysfs_base / "video0"
        video0.mkdir()

        # Create device path
        dev_path = tmp_path / "dev" / "video0"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        dev_path.touch()

        # Create USB ID files
        usb_dir = video0 / "device"
        usb_dir.mkdir()
        real_usb = tmp_path / "usb_device"
        real_usb.mkdir()
        (real_usb / "idVendor").write_text("046d")
        (real_usb / "idProduct").write_text("085c")

        disc = CameraDiscovery(sysfs_base=str(sysfs_base))
        device = CameraDevice(
            device_path=str(dev_path),
            name="Test",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-test",
        )

        # Mock _read_usb_ids to return matching IDs
        with patch.object(disc, "_read_usb_ids", return_value=("046d", "085c")):
            assert disc.validate_device(device) is True

    def test_device_path_gone(self, tmp_path: Path) -> None:
        sysfs_base = tmp_path / "sysfs"
        sysfs_base.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs_base))
        device = _make_device(path=str(tmp_path / "nonexistent"))
        assert disc.validate_device(device) is False

    def test_sysfs_entry_gone(self, tmp_path: Path) -> None:
        sysfs_base = tmp_path / "sysfs"
        sysfs_base.mkdir()
        # Create device path but no sysfs entry
        dev_path = tmp_path / "dev" / "video0"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        dev_path.touch()

        disc = CameraDiscovery(sysfs_base=str(sysfs_base))
        device = CameraDevice(
            device_path=str(dev_path),
            name="Test",
            vendor_id="046d",
            product_id="085c",
            bus_info="",
        )
        assert disc.validate_device(device) is False

    def test_usb_id_mismatch(self, tmp_path: Path) -> None:
        sysfs_base = tmp_path / "sysfs"
        sysfs_base.mkdir()
        video0 = sysfs_base / "video0"
        video0.mkdir()

        dev_path = tmp_path / "dev" / "video0"
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        dev_path.touch()

        disc = CameraDiscovery(sysfs_base=str(sysfs_base))
        device = CameraDevice(
            device_path=str(dev_path),
            name="Test",
            vendor_id="046d",
            product_id="085c",
            bus_info="",
        )

        # Return different USB IDs (device number reused by different device)
        with patch.object(disc, "_read_usb_ids", return_value=("1234", "5678")):
            assert disc.validate_device(device) is False


class TestResilientCaptureValidation:
    """Tests for resilient capture using validate_device before capture."""

    def test_disconnects_when_device_invalid(self) -> None:
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(
            preferred_vendor_id="046d",
            preferred_product_id="085c",
            max_reconnect_attempts=1,
            reconnect_delay=0.01,
        )

        mock_device = _make_device()
        cam._current_device = mock_device
        cam._connected = True
        mock_handle = MagicMock()
        mock_handle.is_open = True
        cam._handle = mock_handle

        mock_disc = MagicMock()
        mock_disc.validate_device.return_value = False
        mock_disc.discover.return_value = []
        mock_disc.find_by_usb_id.return_value = None
        mock_disc.rediscover_device.return_value = None
        mock_disc.find_preferred.return_value = None

        with (
            patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc),
            patch("time.sleep"),
        ):
            result = cam.capture()
            assert not result.success
            mock_disc.validate_device.assert_called_once_with(mock_device)

    def test_skips_validation_when_not_connected(self) -> None:
        from missy.vision.resilient_capture import ResilientCamera

        cam = ResilientCamera(max_reconnect_attempts=1, reconnect_delay=0.01)
        cam._connected = False

        mock_disc = MagicMock()
        mock_disc.discover.return_value = []
        mock_disc.find_preferred.return_value = None

        with patch("missy.vision.resilient_capture.get_discovery", return_value=mock_disc):
            result = cam.capture()
            assert not result.success
            mock_disc.validate_device.assert_not_called()
