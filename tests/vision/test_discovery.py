"""Tests for missy.vision.discovery — USB camera discovery."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from missy.vision.discovery import CameraDevice, CameraDiscovery, KNOWN_CAMERAS


# ---------------------------------------------------------------------------
# CameraDevice tests
# ---------------------------------------------------------------------------


class TestCameraDevice:
    def test_usb_id_property(self):
        dev = CameraDevice(
            device_path="/dev/video0",
            name="Test Cam",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-0000:00:14.0-1",
        )
        assert dev.usb_id == "046d:085c"

    def test_is_logitech_c922(self):
        dev = CameraDevice(
            device_path="/dev/video0",
            name="Logitech C922x",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-0000:00:14.0-1",
        )
        assert dev.is_logitech_c922 is True

    def test_is_not_logitech_c922(self):
        dev = CameraDevice(
            device_path="/dev/video0",
            name="Generic Cam",
            vendor_id="1234",
            product_id="5678",
            bus_info="usb-0000:00:14.0-1",
        )
        assert dev.is_logitech_c922 is False

    def test_frozen(self):
        dev = CameraDevice(
            device_path="/dev/video0",
            name="Test",
            vendor_id="0000",
            product_id="0000",
            bus_info="",
        )
        with pytest.raises(AttributeError):
            dev.name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CameraDiscovery tests with fake sysfs
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_sysfs(tmp_path):
    """Create a fake /sys/class/video4linux/ structure."""
    sysfs = tmp_path / "video4linux"
    sysfs.mkdir()

    # Create video0 entry
    v0 = sysfs / "video0"
    v0.mkdir()
    (v0 / "name").write_text("Logitech C922x Pro Stream Webcam\n")
    (v0 / "index").write_text("0\n")

    # Create uevent with DEVNAME
    (v0 / "uevent").write_text("MAJOR=81\nMINOR=0\nDEVNAME=video0\n")

    # Create video1 (metadata node — index=1, should be skipped)
    v1 = sysfs / "video1"
    v1.mkdir()
    (v1 / "name").write_text("Logitech C922x Pro Stream Webcam\n")
    (v1 / "index").write_text("1\n")
    (v1 / "uevent").write_text("MAJOR=81\nMINOR=1\nDEVNAME=video1\n")

    return sysfs


class TestCameraDiscovery:
    def test_discover_no_sysfs(self, tmp_path):
        disc = CameraDiscovery(sysfs_base=str(tmp_path / "nonexistent"))
        cameras = disc.discover()
        assert cameras == []

    def test_discover_empty_sysfs(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        cameras = disc.discover()
        assert cameras == []

    def test_discover_skips_metadata_node(self, fake_sysfs, tmp_path):
        """video1 with index=1 should be skipped."""
        # Create fake /dev/video0 and /dev/video1
        disc = CameraDiscovery(sysfs_base=str(fake_sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)
            # video1 has index=1, so should be skipped
            # video0 has index=0, should be included
            device_paths = [c.device_path for c in cameras]
            assert "/dev/video1" not in device_paths

    def test_cache_ttl(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=100)

        # First call scans
        result1 = disc.discover()
        assert result1 == []

        # Second call uses cache (doesn't rescan)
        result2 = disc.discover()
        assert result2 == []

    def test_force_bypasses_cache(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=100)

        disc.discover()
        # Force scan should work even within cache TTL
        disc.discover(force=True)  # Should not raise

    def test_find_by_usb_id_no_match(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc.find_by_usb_id("ffff", "ffff") is None

    def test_find_by_name_no_match(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc.find_by_name("NonexistentCamera") == []

    def test_find_preferred_no_cameras(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc.find_preferred() is None

    def test_find_preferred_prefers_logitech_c922(self):
        disc = CameraDiscovery()
        # Manually populate cache
        generic = CameraDevice(
            device_path="/dev/video0",
            name="Generic Webcam",
            vendor_id="1234",
            product_id="5678",
            bus_info="usb-1",
        )
        logitech = CameraDevice(
            device_path="/dev/video2",
            name="Logitech C922x",
            vendor_id="046d",
            product_id="085c",
            bus_info="usb-2",
        )
        disc._cache = [generic, logitech]
        disc._cache_time = 1e18  # never expire

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.device_path == "/dev/video2"

    def test_find_preferred_falls_back_to_known(self):
        disc = CameraDiscovery()
        known = CameraDevice(
            device_path="/dev/video0",
            name="Logitech HD Webcam C270",
            vendor_id="046d",
            product_id="0825",
            bus_info="usb-1",
        )
        disc._cache = [known]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.usb_id == "046d:0825"

    def test_find_preferred_falls_back_to_first(self):
        disc = CameraDiscovery()
        unknown = CameraDevice(
            device_path="/dev/video0",
            name="Unknown Cam",
            vendor_id="ffff",
            product_id="ffff",
            bus_info="usb-1",
        )
        disc._cache = [unknown]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.device_path == "/dev/video0"

    def test_read_sysfs_attr_missing(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video99"
        entry.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc._read_sysfs_attr(entry, "nonexistent") == ""


# ---------------------------------------------------------------------------
# KNOWN_CAMERAS
# ---------------------------------------------------------------------------


class TestKnownCameras:
    def test_c922x_in_known(self):
        assert "046d:085c" in KNOWN_CAMERAS

    def test_c920_in_known(self):
        assert "046d:082d" in KNOWN_CAMERAS
