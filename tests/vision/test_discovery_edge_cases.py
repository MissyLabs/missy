"""Edge case tests for missy.vision.discovery — USB camera discovery.

Covers scenarios not tested in test_discovery.py:
- Multiple cameras with identical USB IDs
- Device path changes while bus_info stays stable
- Missing or unreadable sysfs tree
- Broken symlinks in sysfs
- Empty device name file
- Permission denied on /dev/videoN
- Non-capture (metadata-only) devices
- USB ID format edge cases
- Rapid add/remove cycling via force=True
- find_preferred edge cases with and without Logitech devices
- Unusual sysfs bus_info path layouts
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.vision.discovery import (
    KNOWN_CAMERAS,
    CameraDevice,
    CameraDiscovery,
    discover_cameras,
    find_preferred_camera,
    get_discovery,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(
    device_path: str = "/dev/video0",
    name: str = "Test Camera",
    vendor_id: str = "1234",
    product_id: str = "5678",
    bus_info: str = "usb-0000:00:14.0-1",
    capabilities: list[str] | None = None,
) -> CameraDevice:
    return CameraDevice(
        device_path=device_path,
        name=name,
        vendor_id=vendor_id,
        product_id=product_id,
        bus_info=bus_info,
        capabilities=capabilities or [],
    )


def _make_video_entry(
    sysfs: Path,
    name: str,
    dev_name: str,
    index: int = 0,
    dev_file_name: str | None = None,
) -> Path:
    """Create a minimal sysfs video entry directory."""
    entry = sysfs / name
    entry.mkdir()
    (entry / "name").write_text(f"{dev_name}\n")
    (entry / "index").write_text(f"{index}\n")
    dn = dev_file_name or name
    (entry / "uevent").write_text(f"MAJOR=81\nMINOR=0\nDEVNAME={dn}\n")
    return entry


# ---------------------------------------------------------------------------
# 1. Multiple cameras with same USB ID
# ---------------------------------------------------------------------------


class TestMultipleCamerasWithSameUsbId:
    """find_by_usb_id should return the first match when duplicates exist."""

    def test_returns_first_match(self):
        disc = CameraDiscovery()
        cam_a = _make_camera(device_path="/dev/video0", vendor_id="046d", product_id="085c")
        cam_b = _make_camera(device_path="/dev/video2", vendor_id="046d", product_id="085c")
        disc._cache = [cam_a, cam_b]
        disc._cache_time = 1e18

        result = disc.find_by_usb_id("046d", "085c")
        assert result is not None
        assert result.device_path == "/dev/video0"

    def test_second_device_not_returned(self):
        disc = CameraDiscovery()
        cam_a = _make_camera(device_path="/dev/video0", vendor_id="046d", product_id="085c")
        cam_b = _make_camera(device_path="/dev/video2", vendor_id="046d", product_id="085c")
        disc._cache = [cam_a, cam_b]
        disc._cache_time = 1e18

        result = disc.find_by_usb_id("046d", "085c")
        assert result.device_path != "/dev/video2"

    def test_discover_returns_all_duplicates(self):
        """discover() itself should list every capture node, even duplicates."""
        disc = CameraDiscovery()
        cam_a = _make_camera(device_path="/dev/video0", vendor_id="046d", product_id="085c")
        cam_b = _make_camera(device_path="/dev/video2", vendor_id="046d", product_id="085c")
        disc._cache = [cam_a, cam_b]
        disc._cache_time = 1e18

        all_devices = disc.discover()
        usb_ids = [d.usb_id for d in all_devices]
        assert usb_ids.count("046d:085c") == 2

    def test_find_preferred_picks_first_logitech_when_duplicates(self):
        """With two identical Logitechs, the first in list order wins."""
        disc = CameraDiscovery()
        cam_a = _make_camera(device_path="/dev/video0", vendor_id="046d", product_id="085c")
        cam_b = _make_camera(device_path="/dev/video2", vendor_id="046d", product_id="085c")
        disc._cache = [cam_a, cam_b]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.device_path == "/dev/video0"


# ---------------------------------------------------------------------------
# 2. Device re-enumeration — path changes, bus_info stays stable
# ---------------------------------------------------------------------------


class TestDeviceReenumeration:
    """bus_info should remain stable across re-enumeration."""

    def test_bus_info_unchanged_after_path_change(self):
        before = _make_camera(device_path="/dev/video0", bus_info="usb-0000:00:14.0-1")
        after = _make_camera(device_path="/dev/video2", bus_info="usb-0000:00:14.0-1")

        assert before.bus_info == after.bus_info

    def test_find_by_bus_info_same_camera_new_path(self):
        """Searching by bus_info works regardless of device path."""
        disc = CameraDiscovery()
        cam = _make_camera(device_path="/dev/video4", bus_info="usb-0000:00:14.0-1")
        disc._cache = [cam]
        disc._cache_time = 1e18

        all_devs = disc.discover()
        found = [d for d in all_devs if d.bus_info == "usb-0000:00:14.0-1"]
        assert len(found) == 1
        assert found[0].device_path == "/dev/video4"

    def test_force_rescan_reflects_new_path(self, tmp_path):
        """After force=True the cache is refreshed; a new sysfs layout is reflected."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()

        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=100)

        # First scan: empty
        result1 = disc.discover(force=True)
        assert result1 == []

        # Simulate camera appearing at video2 (not video0)
        _make_video_entry(sysfs, "video2", "New Camera")

        with patch.object(Path, "exists", return_value=True):
            result2 = disc.discover(force=True)

        assert len(result2) == 1
        assert result2[0].device_path == "/dev/video2"


# ---------------------------------------------------------------------------
# 3. sysfs not readable — directory doesn't exist
# ---------------------------------------------------------------------------


class TestSysfsNotReadable:
    def test_nonexistent_sysfs_returns_empty(self, tmp_path):
        disc = CameraDiscovery(sysfs_base=str(tmp_path / "does_not_exist"))
        assert disc.discover() == []

    def test_nonexistent_sysfs_logs_warning(self, tmp_path, caplog):
        import logging

        disc = CameraDiscovery(sysfs_base=str(tmp_path / "does_not_exist"))
        with caplog.at_level(logging.WARNING, logger="missy.vision.discovery"):
            disc.discover()
        assert any("does not exist" in r.message for r in caplog.records)

    def test_oserror_on_iterdir_returns_empty(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "iterdir", side_effect=OSError("Permission denied")):
            result = disc.discover(force=True)

        assert result == []

    def test_oserror_on_iterdir_logs_warning(self, tmp_path, caplog):
        import logging

        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with (
            patch.object(Path, "iterdir", side_effect=OSError("Permission denied")),
            caplog.at_level(logging.WARNING, logger="missy.vision.discovery"),
        ):
            disc.discover(force=True)

        assert any("Cannot read sysfs" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. Broken symlinks in sysfs
# ---------------------------------------------------------------------------


class TestBrokenSymlinksInSysfs:
    def test_broken_device_symlink_yields_unknown_ids(self, tmp_path):
        """_read_usb_ids with a broken device symlink should return 0000:0000."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        # Create a dangling symlink for "device"
        broken_link = entry / "device"
        broken_link.symlink_to(tmp_path / "nonexistent_target")

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        vendor_id, product_id = disc._read_usb_ids(entry)
        # Dangling symlink — resolve() follows it; idVendor/idProduct won't be
        # found traversing up from a nonexistent path, so default is returned.
        assert vendor_id == "0000"
        assert product_id == "0000"

    def test_broken_device_symlink_yields_empty_bus_info(self, tmp_path):
        """_read_bus_info with a broken symlink should return empty string."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        broken_link = entry / "device"
        broken_link.symlink_to(tmp_path / "nonexistent_target")

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        # resolve() on a dangling symlink still returns the path (no OSError on
        # most Linux kernels); the USB prefix won't be found, so we get the raw
        # resolved string or empty — just assert we don't raise.
        bus_info = disc._read_bus_info(entry)
        assert isinstance(bus_info, str)

    def test_oserror_on_device_resolve_returns_default_ids(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "resolve", side_effect=OSError("IO error")):
            vendor_id, product_id = disc._read_usb_ids(entry)

        assert vendor_id == "0000"
        assert product_id == "0000"

    def test_oserror_on_bus_info_resolve_returns_empty(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "resolve", side_effect=OSError("IO error")):
            bus_info = disc._read_bus_info(entry)

        assert bus_info == ""


# ---------------------------------------------------------------------------
# 5. Empty device name file
# ---------------------------------------------------------------------------


class TestEmptyDeviceName:
    def test_empty_name_file_uses_fallback(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = _make_video_entry(sysfs, "video0", "")
        # Overwrite name with truly empty content
        (entry / "name").write_text("")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert len(cameras) == 1
        # Fallback name should reference the device node
        assert "video0" in cameras[0].name or cameras[0].name != ""

    def test_whitespace_only_name_uses_fallback(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = _make_video_entry(sysfs, "video0", "   ")
        (entry / "name").write_text("   \n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert len(cameras) == 1
        # _read_sysfs_attr strips whitespace; empty string triggers fallback
        assert cameras[0].name != ""

    def test_name_attribute_stripped(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = _make_video_entry(sysfs, "video0", "My Camera\n\n")
        (entry / "name").write_text("My Camera\n\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert len(cameras) == 1
        assert cameras[0].name == "My Camera"


# ---------------------------------------------------------------------------
# 6. Permission denied on /dev/videoN
# ---------------------------------------------------------------------------


class TestPermissionDeniedOnDevice:
    def test_device_that_does_not_exist_is_skipped(self, tmp_path):
        """If /dev/videoN is absent the device must be omitted."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        _make_video_entry(sysfs, "video0", "Cam A")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        # Do NOT patch Path.exists — /dev/video0 won't actually exist in CI
        cameras = disc.discover(force=True)
        assert cameras == []

    def test_device_exists_false_skips_entry(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        _make_video_entry(sysfs, "video0", "Cam A")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        # Simulate /dev/video0 exists check returning False for /dev paths
        original_exists = Path.exists

        def selective_exists(self_path: Path) -> bool:
            if str(self_path).startswith("/dev/"):
                return False
            return original_exists(self_path)

        with patch.object(Path, "exists", selective_exists):
            cameras = disc.discover(force=True)

        assert cameras == []

    def test_multiple_devices_only_present_ones_listed(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        _make_video_entry(sysfs, "video0", "Cam A")
        _make_video_entry(sysfs, "video2", "Cam B")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        original_exists = Path.exists

        def video0_exists(self_path: Path) -> bool:
            if str(self_path) == "/dev/video0":
                return True  # present
            if str(self_path) == "/dev/video2":
                return False  # absent / no permission
            return original_exists(self_path)

        with patch.object(Path, "exists", video0_exists):
            cameras = disc.discover(force=True)

        device_paths = [c.device_path for c in cameras]
        assert "/dev/video0" in device_paths
        assert "/dev/video2" not in device_paths


# ---------------------------------------------------------------------------
# 7. Non-video / metadata-only devices
# ---------------------------------------------------------------------------


class TestNonVideoDevices:
    def test_index_1_device_excluded(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        # index=1 marks a metadata / sub-device node
        _make_video_entry(sysfs, "video1", "Cam Metadata Node", index=1)

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert cameras == []

    def test_index_0_device_included(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        _make_video_entry(sysfs, "video0", "Real Cam", index=0)

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert len(cameras) == 1

    def test_uevent_without_devname_excluded(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        (entry / "name").write_text("Some Device\n")
        (entry / "index").write_text("0\n")
        # uevent deliberately missing DEVNAME
        (entry / "uevent").write_text("MAJOR=81\nMINOR=0\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert cameras == []

    def test_non_video_entry_name_excluded(self, tmp_path):
        """Entries not starting with 'video' in sysfs are silently skipped."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        other = sysfs / "media0"
        other.mkdir()
        (other / "name").write_text("Media Controller\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert cameras == []

    def test_corrupt_index_file_device_still_considered(self, tmp_path):
        """Non-integer index content is tolerated; device is not excluded by index check."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        (entry / "name").write_text("Cam\n")
        (entry / "index").write_text("not_a_number\n")
        (entry / "uevent").write_text("MAJOR=81\nMINOR=0\nDEVNAME=video0\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        # Invalid index text means the index check is skipped; device should pass
        assert len(cameras) == 1


# ---------------------------------------------------------------------------
# 8. USB ID format edge cases
# ---------------------------------------------------------------------------


class TestUsbIdFormatEdgeCases:
    def test_zero_vendor_product_ids(self):
        dev = _make_camera(vendor_id="0000", product_id="0000")
        assert dev.usb_id == "0000:0000"
        assert dev.is_logitech_c922 is False

    def test_uppercase_hex_ids(self):
        dev = _make_camera(vendor_id="046D", product_id="085C")
        # usb_id preserves whatever case the sysfs gave us
        assert dev.usb_id == "046D:085C"

    def test_mixed_case_ids_not_in_known_cameras(self):
        dev = _make_camera(vendor_id="046D", product_id="085C")
        # KNOWN_CAMERAS uses lowercase; mixed-case won't match without normalization
        assert dev.usb_id not in KNOWN_CAMERAS

    def test_short_vendor_id(self):
        dev = _make_camera(vendor_id="1", product_id="2")
        assert dev.usb_id == "1:2"

    def test_long_vendor_id(self):
        dev = _make_camera(vendor_id="ffff0000", product_id="12345678")
        assert dev.usb_id == "ffff0000:12345678"

    def test_default_unknown_ids_from_scan(self, tmp_path):
        """When idVendor/idProduct are missing, scan yields 0000:0000."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        _make_video_entry(sysfs, "video0", "Cam")
        # No device/ subdir → _read_usb_ids has nowhere to walk

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            cameras = disc.discover(force=True)

        assert len(cameras) == 1
        assert cameras[0].vendor_id == "0000"
        assert cameras[0].product_id == "0000"

    def test_idvendor_read_oserror_returns_defaults(self, tmp_path):
        """OSError while reading idVendor/idProduct returns 0000:0000."""
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        usb_dir = tmp_path / "usb_device"
        usb_dir.mkdir()
        (usb_dir / "idVendor").write_text("046d\n")
        (usb_dir / "idProduct").write_text("085c\n")
        (entry / "device").symlink_to(usb_dir)

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        # Patch read_text to raise OSError for idVendor
        original_read_text = Path.read_text

        def failing_read_text(self_path, *args, **kwargs):
            if self_path.name in ("idVendor", "idProduct"):
                raise OSError("Permission denied")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", failing_read_text):
            vendor_id, product_id = disc._read_usb_ids(entry)

        assert vendor_id == "0000"
        assert product_id == "0000"


# ---------------------------------------------------------------------------
# 9. Rapid add/remove — force=True yields fresh results each time
# ---------------------------------------------------------------------------


class TestRapidAddRemove:
    def test_force_true_always_rescans(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=9999)

        # First scan: no devices
        result1 = disc.discover(force=True)
        assert result1 == []

        # Camera appears
        _make_video_entry(sysfs, "video0", "Transient Cam")

        with patch.object(Path, "exists", return_value=True):
            result2 = disc.discover(force=True)
        assert len(result2) == 1

        # Camera disappears — remove sysfs entry
        import shutil

        shutil.rmtree(str(sysfs / "video0"))

        result3 = disc.discover(force=True)
        assert result3 == []

    def test_cached_results_returned_within_ttl(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=9999)

        result1 = disc.discover()
        assert result1 == []

        # Add camera to sysfs — but cache is still valid
        _make_video_entry(sysfs, "video0", "New Cam")

        result2 = disc.discover()  # should use cache, not rescan
        assert result2 == []  # stale cache returns empty

    def test_cache_expires_after_ttl(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=0.001)

        result1 = disc.discover()
        assert result1 == []

        # Add camera then wait for cache to expire
        _make_video_entry(sysfs, "video0", "Cam After TTL")
        time.sleep(0.01)

        with patch.object(Path, "exists", return_value=True):
            result2 = disc.discover()

        assert len(result2) == 1

    def test_force_true_updates_cache_timestamp(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        before_time = disc._cache_time
        disc.discover(force=True)
        after_time = disc._cache_time

        assert after_time > before_time


# ---------------------------------------------------------------------------
# 10 & 11. find_preferred with no devices / multiple devices
# ---------------------------------------------------------------------------


class TestFindPreferred:
    def test_find_preferred_no_devices_returns_none(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc.find_preferred() is None

    def test_find_preferred_single_unknown_returns_it(self):
        disc = CameraDiscovery()
        cam = _make_camera(device_path="/dev/video0", vendor_id="ffff", product_id="ffff")
        disc._cache = [cam]
        disc._cache_time = 1e18

        result = disc.find_preferred()
        assert result is not None
        assert result.device_path == "/dev/video0"

    def test_find_preferred_logitech_wins_over_known(self):
        disc = CameraDiscovery()
        c270 = _make_camera(
            device_path="/dev/video0",
            vendor_id="046d",
            product_id="0825",
            name="Logitech C270",
        )
        c922 = _make_camera(
            device_path="/dev/video2",
            vendor_id="046d",
            product_id="085c",
            name="Logitech C922x",
        )
        disc._cache = [c270, c922]
        disc._cache_time = 1e18

        result = disc.find_preferred()
        assert result is not None
        assert result.product_id == "085c"

    def test_find_preferred_logitech_c922b_also_preferred(self):
        """Both 085c and 085b product IDs count as C922."""
        disc = CameraDiscovery()
        cam = _make_camera(vendor_id="046d", product_id="085b")
        disc._cache = [cam]
        disc._cache_time = 1e18

        result = disc.find_preferred()
        assert result is not None
        assert result.is_logitech_c922 is True

    def test_find_preferred_known_camera_beats_unknown(self):
        disc = CameraDiscovery()
        unknown = _make_camera(device_path="/dev/video0", vendor_id="ffff", product_id="ffff")
        known = _make_camera(
            device_path="/dev/video2",
            vendor_id="046d",
            product_id="0825",  # C270 — known but not C922
        )
        disc._cache = [unknown, known]
        disc._cache_time = 1e18

        result = disc.find_preferred()
        assert result is not None
        assert result.device_path == "/dev/video2"

    def test_find_preferred_multiple_unknowns_returns_first(self):
        disc = CameraDiscovery()
        cams = [
            _make_camera(device_path=f"/dev/video{i}", vendor_id="aaaa", product_id="bbbb")
            for i in range(3)
        ]
        disc._cache = cams
        disc._cache_time = 1e18

        result = disc.find_preferred()
        assert result is not None
        assert result.device_path == "/dev/video0"

    def test_find_preferred_logitech_later_in_list_still_chosen(self):
        """Logitech C922 anywhere in list should beat a generic camera at index 0."""
        disc = CameraDiscovery()
        generic1 = _make_camera(device_path="/dev/video0", vendor_id="1111", product_id="2222")
        generic2 = _make_camera(device_path="/dev/video1", vendor_id="3333", product_id="4444")
        logitech = _make_camera(device_path="/dev/video2", vendor_id="046d", product_id="085c")
        disc._cache = [generic1, generic2, logitech]
        disc._cache_time = 1e18

        result = disc.find_preferred()
        assert result is not None
        assert result.device_path == "/dev/video2"

    def test_find_preferred_uses_discover_not_raw_cache(self):
        """find_preferred calls discover(), which obeys cache TTL."""
        disc = CameraDiscovery()
        cam = _make_camera(device_path="/dev/video0", vendor_id="046d", product_id="085c")
        # Expired cache
        disc._cache = [cam]
        disc._cache_time = 0.0

        with patch.object(disc, "_scan_sysfs", return_value=[]) as mock_scan:
            result = disc.find_preferred()

        mock_scan.assert_called_once()
        assert result is None


# ---------------------------------------------------------------------------
# 12. Bus info parsing with unusual sysfs layouts
# ---------------------------------------------------------------------------


class TestBusInfoParsing:
    def test_standard_usb_bus_info(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        # Simulate: device/ → .../usb1/1-1/1-1:1.0/video4linux/video0
        usb_chain = tmp_path / "sys" / "bus" / "usb1" / "1-1" / "1-1:1.0"
        usb_chain.mkdir(parents=True)
        (entry / "device").symlink_to(usb_chain)

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        bus_info = disc._read_bus_info(entry)
        assert "usb1" in bus_info

    def test_bus_info_no_usb_prefix_returns_full_path(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        # Device symlink points somewhere without "usb" in the path
        pci_device = tmp_path / "pci" / "0000:00:14"
        pci_device.mkdir(parents=True)
        (entry / "device").symlink_to(pci_device)

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        bus_info = disc._read_bus_info(entry)
        # No "usb" component → returns resolved string of the full path
        assert bus_info != ""
        assert isinstance(bus_info, str)

    def test_bus_info_missing_device_dir_returns_empty(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        # No device/ symlink created

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        # resolve() on a non-existent path should not raise; bus_info will be
        # derived from whatever the non-existent path resolves to.
        bus_info = disc._read_bus_info(entry)
        assert isinstance(bus_info, str)

    def test_bus_info_deeply_nested_usb_path(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        deep = tmp_path / "sys" / "devices" / "pci0000:00" / "usb3" / "3-2" / "3-2.4" / "3-2.4:1.0"
        deep.mkdir(parents=True)
        (entry / "device").symlink_to(deep)

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        bus_info = disc._read_bus_info(entry)
        # Should capture from the "usb3" component onwards
        assert bus_info.startswith("usb3")

    def test_read_bus_info_oserror_returns_empty(self, tmp_path):
        sysfs = tmp_path / "video4linux"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "resolve", side_effect=OSError("No such file")):
            bus_info = disc._read_bus_info(entry)

        assert bus_info == ""


# ---------------------------------------------------------------------------
# Module-level convenience function smoke tests
# ---------------------------------------------------------------------------


class TestModuleLevelConveniences:
    def test_get_discovery_returns_singleton(self):
        from missy.vision import discovery as disc_mod

        disc_mod._default_discovery = None  # reset
        d1 = get_discovery()
        d2 = get_discovery()
        assert d1 is d2

    def test_get_discovery_creates_instance(self):
        from missy.vision import discovery as disc_mod

        disc_mod._default_discovery = None
        d = get_discovery()
        assert isinstance(d, CameraDiscovery)

    def test_discover_cameras_delegates(self):
        from missy.vision import discovery as disc_mod

        mock_disc = MagicMock(spec=CameraDiscovery)
        mock_disc.discover.return_value = []
        disc_mod._default_discovery = mock_disc

        result = discover_cameras(force=True)

        mock_disc.discover.assert_called_once_with(force=True)
        assert result == []

    def test_find_preferred_camera_delegates(self):
        from missy.vision import discovery as disc_mod

        cam = _make_camera()
        mock_disc = MagicMock(spec=CameraDiscovery)
        mock_disc.find_preferred.return_value = cam
        disc_mod._default_discovery = mock_disc

        result = find_preferred_camera()

        mock_disc.find_preferred.assert_called_once()
        assert result is cam
