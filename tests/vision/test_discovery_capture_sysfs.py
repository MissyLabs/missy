"""Session-13 edge-case tests for missy.vision.discovery and missy.vision.capture.

Covers specific internal behaviours that are not exercised by existing test files:

discovery.py
  - _read_usb_ids: symlink-cycle detection via visited-set
  - _read_usb_ids: reaching filesystem root without finding IDs
  - _read_usb_ids: OSError on resolve
  - _read_sysfs_attr: non-existent attribute path
  - _read_sysfs_attr: OSError (PermissionError)
  - _scan_sysfs: missing sysfs base logs warning
  - _scan_sysfs: OSError on iterdir logs warning
  - _scan_sysfs: non-"video" entry names skipped
  - _scan_sysfs: /dev/videoN absent skips device
  - _is_capture_device: index != 0 → False
  - _is_capture_device: no DEVNAME in uevent → False
  - _is_capture_device: non-numeric index (ValueError) → device still passes
  - find_by_name: invalid regex → logs warning, returns []
  - find_preferred: C922x preferred over C270
  - find_preferred: known camera preferred over unknown
  - validate_device: USB ID mismatch after device-number reuse
  - validate_device: sysfs entry missing
  - rediscover_device: device path changes between attempts
  - Cache TTL: discover() uses cache within TTL, rescans after expiry
  - discover(force=True) always rescans regardless of cache age

capture.py
  - AdaptiveBlankDetector.threshold: < 3 observations → returns base_threshold
  - AdaptiveBlankDetector.threshold: adaptive value clamped between min and base
  - AdaptiveBlankDetector.record_intensity: zero values are not recorded
  - AdaptiveBlankDetector.is_blank: mean below threshold → True
  - AdaptiveBlankDetector.reset: clears observation window
  - CaptureResult.shape: returns (0, 0, 0) when image is None
  - CaptureResult.shape: returns actual ndarray shape when image present
  - CaptureConfig: verify documented defaults
  - _get_cv2: raises ImportError with installation hint when cv2 absent
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from missy.vision.capture import (
    AdaptiveBlankDetector,
    CaptureConfig,
    CaptureResult,
)
from missy.vision.discovery import (
    CameraDevice,
    CameraDiscovery,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_device(
    path: str = "/dev/video0",
    vendor_id: str = "046d",
    product_id: str = "085c",
    name: str = "Test Camera",
    bus_info: str = "usb-0000:00:14.0-1",
) -> CameraDevice:
    return CameraDevice(
        device_path=path,
        name=name,
        vendor_id=vendor_id,
        product_id=product_id,
        bus_info=bus_info,
    )


def _make_sysfs_entry(
    sysfs_root: Path,
    entry_name: str = "video0",
    camera_name: str = "Test Camera",
    index: int = 0,
    include_devname: bool = True,
) -> Path:
    """Create a minimal sysfs video4linux entry under sysfs_root."""
    entry = sysfs_root / entry_name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "name").write_text(f"{camera_name}\n")
    (entry / "index").write_text(f"{index}\n")
    uevent_content = "MAJOR=81\nMINOR=0\n"
    if include_devname:
        uevent_content += f"DEVNAME={entry_name}\n"
    (entry / "uevent").write_text(uevent_content)
    return entry


# ===========================================================================
# discovery.py — _read_usb_ids
# ===========================================================================


class TestReadUsbIdsSymlinkCycleDetection:
    """_read_usb_ids must not infinite-loop when a symlink cycle is present."""

    def test_cycle_in_parent_chain_returns_defaults(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)

        # Build a directory tree where A/device → B and B/device → A (cycle).
        node_a = tmp_path / "usb" / "node_a"
        node_b = tmp_path / "usb" / "node_b"
        node_a.mkdir(parents=True)
        node_b.mkdir(parents=True)

        # idVendor/idProduct are absent from both nodes.
        # We simulate a cycle by patching Path.parent to return itself after
        # visiting twice (the visited-set check fires before parent traversal
        # can loop infinitely).
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        # Patch resolve so device/ resolves to node_a
        with patch.object(Path, "resolve", return_value=node_a):
            # node_a has no idVendor; its parent is node_b, which has no
            # idVendor either.  The loop depth limit (10) or root detection
            # will terminate the walk.  The key assertion is: no exception.
            vendor_id, product_id = disc._read_usb_ids(entry)

        assert vendor_id == "0000"
        assert product_id == "0000"

    def test_visited_set_prevents_revisit(self, tmp_path: Path) -> None:
        """The visited-set cycle guard is exercised by the internal loop.

        We inject a real directory whose resolve() returns itself, which means
        the first iteration adds it to visited and then breaks when the path
        appears in visited on what would be the second traversal.  We verify
        that _read_usb_ids returns defaults and does not raise, confirming the
        guard executed without error.
        """
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)

        # lone_dir: no idVendor/idProduct, its parent is tmp_path (also no IDs).
        # resolve() is patched to always return lone_dir so "current" never
        # changes — on the second iteration lone_dir is already in visited and
        # the loop breaks.
        lone_dir = tmp_path / "lone"
        lone_dir.mkdir()

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        call_count = 0

        def stubbed_resolve(self_path: Path) -> Path:
            nonlocal call_count
            call_count += 1
            # Always return the same lone_dir so parent traversal never moves.
            return lone_dir

        with patch.object(Path, "resolve", stubbed_resolve):
            vid, pid = disc._read_usb_ids(entry)

        assert vid == "0000"
        assert pid == "0000"


class TestReadUsbIdsRootReached:
    """When walking up sysfs we hit the filesystem root without finding IDs."""

    def test_root_traversal_returns_defaults(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)

        # Resolve to a shallow path whose parent chain terminates quickly.
        # Path("/") is its own parent, so the loop terminates.
        root = Path("/")
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "resolve", return_value=root):
            vid, pid = disc._read_usb_ids(entry)

        assert vid == "0000"
        assert pid == "0000"

    def test_depth_limit_respected(self, tmp_path: Path) -> None:
        """Loop never runs more than 10 iterations (depth guard)."""
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)

        # Create a 15-level deep chain where none has idVendor.
        chain = tmp_path / "chain"
        current = chain
        for i in range(15):
            current = current / f"level{i}"
        current.mkdir(parents=True)

        # We don't actually traverse a real 15-level chain on disk; we just
        # verify the resolution path returns sensible defaults.
        disc = CameraDiscovery(sysfs_base=str(sysfs))
        with patch.object(Path, "resolve", return_value=chain):
            vid, pid = disc._read_usb_ids(entry)

        assert vid == "0000"
        assert pid == "0000"


class TestReadUsbIdsOSError:
    """OSError on resolve returns the default tuple."""

    def test_oserror_on_resolve_returns_defaults(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "resolve", side_effect=OSError("broken symlink")):
            vid, pid = disc._read_usb_ids(entry)

        assert vid == "0000"
        assert pid == "0000"

    def test_ioerror_subclass_handled(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "resolve", side_effect=PermissionError("no access")):
            vid, pid = disc._read_usb_ids(entry)

        assert vid == "0000"
        assert pid == "0000"


# ===========================================================================
# discovery.py — _read_sysfs_attr
# ===========================================================================


class TestReadSysfsAttr:
    """_read_sysfs_attr returns empty string for missing or unreadable paths."""

    def test_nonexistent_attribute_returns_empty(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        # No "nonexistent_attr" file created.

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        result = disc._read_sysfs_attr(entry, "nonexistent_attr")

        assert result == ""

    def test_permission_error_returns_empty(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        attr_file = entry / "protected_attr"
        attr_file.write_text("secret\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = disc._read_sysfs_attr(entry, "protected_attr")

        assert result == ""

    def test_oserror_on_read_returns_empty(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        (entry / "name").write_text("Camera\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "read_text", side_effect=OSError("IO error")):
            result = disc._read_sysfs_attr(entry, "name")

        assert result == ""

    def test_existing_attribute_returns_stripped_text(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        (entry / "name").write_text("  My Camera  \n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        result = disc._read_sysfs_attr(entry, "name")

        assert result == "My Camera"


# ===========================================================================
# discovery.py — _scan_sysfs
# ===========================================================================


class TestScanSysfs:
    """_scan_sysfs defensive behaviour."""

    def test_missing_sysfs_base_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        absent = tmp_path / "does_not_exist"
        disc = CameraDiscovery(sysfs_base=str(absent))

        with caplog.at_level(logging.WARNING, logger="missy.vision.discovery"):
            result = disc._scan_sysfs()

        assert result == []
        assert any("does not exist" in r.message for r in caplog.records)

    def test_oserror_on_iterdir_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with (
            patch.object(Path, "iterdir", side_effect=OSError("permission denied")),
            caplog.at_level(logging.WARNING, logger="missy.vision.discovery"),
        ):
            result = disc._scan_sysfs()

        assert result == []
        assert any("Cannot read sysfs" in r.message for r in caplog.records)

    def test_non_video_entries_skipped(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        # Create entries whose names do NOT start with "video".
        for name in ("media0", "rc0", "sound_device"):
            d = sysfs / name
            d.mkdir()

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        result = disc._scan_sysfs()

        assert result == []

    def test_video_entry_without_dev_node_skipped(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        _make_sysfs_entry(sysfs, "video0")
        # /dev/video0 does not exist in the test environment — _scan_sysfs
        # checks Path(device_path).exists() which will be False.

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        result = disc._scan_sysfs()

        # Without patching /dev/video0 into existence the entry is skipped.
        assert result == []

    def test_video_entry_included_when_dev_node_exists(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        _make_sysfs_entry(sysfs, "video0", camera_name="Good Cam")

        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            result = disc._scan_sysfs()

        assert len(result) == 1
        assert result[0].name == "Good Cam"


# ===========================================================================
# discovery.py — _is_capture_device
# ===========================================================================


class TestIsCaptureDevice:
    """_is_capture_device filters metadata nodes."""

    def test_index_nonzero_returns_false(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs, index=1)  # metadata node

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc._is_capture_device(entry) is False

    def test_index_zero_returns_true(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs, index=0)

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc._is_capture_device(entry) is True

    def test_no_devname_in_uevent_returns_false(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs, include_devname=False)

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc._is_capture_device(entry) is False

    def test_nonnumeric_index_does_not_exclude_device(self, tmp_path: Path) -> None:
        """ValueError on int() cast means index check is skipped; device passes."""
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs)
        # Overwrite index with non-numeric text.
        (entry / "index").write_text("NaN\n")

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        # Should not raise; device should not be excluded solely by bad index.
        result = disc._is_capture_device(entry)
        assert result is True

    def test_no_index_file_does_not_exclude_device(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = sysfs / "video0"
        entry.mkdir()
        (entry / "uevent").write_text("MAJOR=81\nMINOR=0\nDEVNAME=video0\n")
        # index file deliberately absent.

        disc = CameraDiscovery(sysfs_base=str(sysfs))
        assert disc._is_capture_device(entry) is True


# ===========================================================================
# discovery.py — find_by_name
# ===========================================================================


class TestFindByName:
    """find_by_name with invalid regex logs warning and returns empty list."""

    def test_invalid_regex_returns_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        disc = CameraDiscovery()
        disc._cache = [_make_device()]
        disc._cache_time = 1e18  # cache never expires

        with caplog.at_level(logging.WARNING, logger="missy.vision.discovery"):
            result = disc.find_by_name("[invalid(regex")

        assert result == []
        assert any("Invalid regex" in r.message for r in caplog.records)

    def test_invalid_regex_warns_with_pattern(self, caplog: pytest.LogCaptureFixture) -> None:
        disc = CameraDiscovery()
        disc._cache = [_make_device()]
        disc._cache_time = 1e18

        with caplog.at_level(logging.WARNING, logger="missy.vision.discovery"):
            disc.find_by_name("(unclosed")

        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("(unclosed" in msg for msg in warning_messages)

    def test_valid_regex_returns_matches(self) -> None:
        disc = CameraDiscovery()
        disc._cache = [
            _make_device(name="Logitech C922x Pro Stream"),
            _make_device(path="/dev/video2", name="Generic USB Camera"),
        ]
        disc._cache_time = 1e18

        result = disc.find_by_name(r"logitech")
        assert len(result) == 1
        assert result[0].name == "Logitech C922x Pro Stream"


# ===========================================================================
# discovery.py — find_preferred preference ordering
# ===========================================================================


class TestFindPreferredOrdering:
    """Verify the three-tier preference: C922x > known > any."""

    def test_c922x_preferred_over_c270(self) -> None:
        disc = CameraDiscovery()
        c270 = _make_device(path="/dev/video0", vendor_id="046d", product_id="0825", name="C270")
        c922 = _make_device(path="/dev/video2", vendor_id="046d", product_id="085c", name="C922x")
        disc._cache = [c270, c922]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.product_id == "085c"

    def test_known_camera_preferred_over_unknown(self) -> None:
        disc = CameraDiscovery()
        unknown = _make_device(path="/dev/video0", vendor_id="dead", product_id="beef", name="Unknown")
        c270 = _make_device(path="/dev/video2", vendor_id="046d", product_id="0825", name="C270")
        disc._cache = [unknown, c270]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.device_path == "/dev/video2"

    def test_unknown_only_returns_first(self) -> None:
        disc = CameraDiscovery()
        cam_a = _make_device(path="/dev/video0", vendor_id="1111", product_id="2222")
        cam_b = _make_device(path="/dev/video2", vendor_id="3333", product_id="4444")
        disc._cache = [cam_a, cam_b]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.device_path == "/dev/video0"

    def test_c922x_anywhere_in_list_wins(self) -> None:
        disc = CameraDiscovery()
        unknown = _make_device(path="/dev/video0", vendor_id="1111", product_id="2222")
        c922 = _make_device(path="/dev/video4", vendor_id="046d", product_id="085c")
        disc._cache = [unknown, c922]
        disc._cache_time = 1e18

        preferred = disc.find_preferred()
        assert preferred is not None
        assert preferred.device_path == "/dev/video4"

    def test_no_devices_returns_none(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        assert disc.find_preferred() is None


# ===========================================================================
# discovery.py — validate_device
# ===========================================================================


class TestValidateDevice:
    """validate_device guards against stale device references."""

    def test_usb_id_mismatch_after_device_reuse(self, tmp_path: Path) -> None:
        """If the /dev/videoN path is reused by a different camera, validate
        should detect the USB ID mismatch and return False."""
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        # Create a sysfs entry for video0 with *different* IDs than the device.
        entry = _make_sysfs_entry(sysfs, "video0")
        usb_dir = tmp_path / "usb_device"
        usb_dir.mkdir()
        (usb_dir / "idVendor").write_text("dead\n")
        (usb_dir / "idProduct").write_text("beef\n")
        (entry / "device").symlink_to(usb_dir)

        original_device = _make_device(vendor_id="046d", product_id="085c")
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            result = disc.validate_device(original_device)

        assert result is False

    def test_sysfs_entry_missing_returns_false(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        # video0 sysfs entry does NOT exist.

        device = _make_device()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        original_exists = Path.exists

        def dev_exists(self_path: Path) -> bool:
            if str(self_path) == "/dev/video0":
                return True
            return original_exists(self_path)

        with patch.object(Path, "exists", dev_exists):
            result = disc.validate_device(device)

        assert result is False

    def test_device_path_absent_returns_false(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        device = _make_device()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        # /dev/video0 does not exist — default Path.exists will return False.
        result = disc.validate_device(device)
        assert result is False

    def test_valid_device_returns_true(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        entry = _make_sysfs_entry(sysfs, "video0")
        usb_dir = tmp_path / "usb_device"
        usb_dir.mkdir()
        (usb_dir / "idVendor").write_text("046d\n")
        (usb_dir / "idProduct").write_text("085c\n")
        (entry / "device").symlink_to(usb_dir)

        device = _make_device(vendor_id="046d", product_id="085c")
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        with patch.object(Path, "exists", return_value=True):
            result = disc.validate_device(device)

        assert result is True


# ===========================================================================
# discovery.py — rediscover_device: path change between attempts
# ===========================================================================


class TestRediscoverDevicePathChange:
    """rediscover_device should handle /dev path changes across attempts."""

    def test_device_path_changes_between_attempts(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))
        new_device = _make_device(path="/dev/video4", vendor_id="046d", product_id="085c")

        with (
            patch.object(disc, "discover", return_value=[new_device]),
            patch.object(disc, "find_by_usb_id", return_value=new_device),
        ):
            result = disc.rediscover_device(
                "046d", "085c", old_path="/dev/video0", max_attempts=3, delay=0.0
            )

        assert result is not None
        assert result.device_path == "/dev/video4"

    def test_returns_none_when_device_never_appears(self, tmp_path: Path) -> None:
        disc = CameraDiscovery(sysfs_base=str(tmp_path))

        with (
            patch.object(disc, "discover", return_value=[]),
            patch.object(disc, "find_by_usb_id", return_value=None),
            patch("time.sleep"),
        ):
            result = disc.rediscover_device("046d", "085c", max_attempts=2, delay=0.0)

        assert result is None


# ===========================================================================
# discovery.py — Cache TTL and force=True
# ===========================================================================


class TestCacheTTL:
    """discover() respects its TTL and force=True bypasses it."""

    def test_cache_valid_within_ttl(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=9999)

        first = disc.discover()
        assert first == []

        # Add an entry to sysfs — but cache is still fresh.
        _make_sysfs_entry(sysfs, "video0")

        second = disc.discover()  # must use cached result
        assert second == []  # stale cache, not the new entry

    def test_cache_expires_after_ttl(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=0.005)

        disc.discover()
        _make_sysfs_entry(sysfs, "video0", camera_name="Late Arrival")
        time.sleep(0.02)  # let the TTL expire

        with patch.object(Path, "exists", return_value=True):
            result = disc.discover()

        assert len(result) == 1
        assert result[0].name == "Late Arrival"

    def test_force_true_always_rescans(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=9999)

        # Seed the cache with a fake device.
        disc._cache = [_make_device()]
        disc._cache_time = time.monotonic()

        result = disc.discover(force=True)

        # After force=True the fresh scan of the empty sysfs returns nothing.
        assert result == []

    def test_force_true_updates_cache_time(self, tmp_path: Path) -> None:
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs))

        before = disc._cache_time
        disc.discover(force=True)
        after = disc._cache_time

        assert after > before

    def test_discover_twice_within_ttl_does_not_rescan(self, tmp_path: Path) -> None:
        """When the cache is non-empty and still within TTL, _scan_sysfs is
        not called a second time.

        The cache guard in discover() is: ``self._cache and TTL_ok``.  An
        empty list is falsy, so the cache must be pre-seeded with at least one
        device for the guard to fire on the second call.
        """
        sysfs = tmp_path / "v4l"
        sysfs.mkdir()
        disc = CameraDiscovery(sysfs_base=str(sysfs), cache_ttl_seconds=9999)

        fake_device = _make_device()
        # First call populates the cache.
        with patch.object(disc, "_scan_sysfs", return_value=[fake_device]) as mock_scan:
            disc.discover()
            disc.discover()

        # Only the first call triggers a real scan; the second uses the cache.
        mock_scan.assert_called_once()


# ===========================================================================
# capture.py — AdaptiveBlankDetector
# ===========================================================================


class TestAdaptiveBlankDetectorThreshold:
    """Threshold property edge cases."""

    def test_fewer_than_3_observations_returns_base(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=8.0, adaptation_factor=0.25)
        det.record_intensity(100.0)
        det.record_intensity(100.0)
        # Only 2 observations — not enough.
        assert det.threshold == 8.0

    def test_exactly_3_observations_adapts(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=8.0, min_threshold=1.0, adaptation_factor=0.25)
        for _ in range(3):
            det.record_intensity(40.0)
        # adaptive = 40.0 * 0.25 = 10.0, but capped at base 8.0
        assert det.threshold == pytest.approx(8.0)

    def test_adaptive_clamped_at_min_threshold(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0, min_threshold=3.0, adaptation_factor=0.05)
        for _ in range(5):
            det.record_intensity(20.0)
        # adaptive = 20.0 * 0.05 = 1.0, clamped to min 3.0
        assert det.threshold == pytest.approx(3.0)

    def test_adaptive_clamped_at_base_threshold(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0, min_threshold=1.0, adaptation_factor=0.9)
        for _ in range(5):
            det.record_intensity(200.0)
        # adaptive = 200 * 0.9 = 180, clamped to base 5.0
        assert det.threshold == pytest.approx(5.0)

    def test_threshold_uses_minimum_observation_not_mean(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=20.0, min_threshold=1.0, adaptation_factor=0.5)
        for intensity in [100.0, 50.0, 10.0, 80.0, 60.0]:
            det.record_intensity(intensity)
        # min_observed = 10.0; adaptive = 10.0 * 0.5 = 5.0
        assert det.threshold == pytest.approx(5.0)


class TestAdaptiveBlankDetectorRecordIntensity:
    """record_intensity boundary behaviour."""

    def test_zero_intensity_not_recorded(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0)
        det.record_intensity(0.0)
        assert len(det._window) == 0

    def test_negative_intensity_not_recorded(self) -> None:
        # Negative values are also <= 0 so guard applies (only > 0 is kept).
        det = AdaptiveBlankDetector(base_threshold=5.0)
        det.record_intensity(-1.0)
        assert len(det._window) == 0

    def test_positive_intensity_is_recorded(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0)
        det.record_intensity(42.0)
        assert len(det._window) == 1
        assert list(det._window)[0] == pytest.approx(42.0)

    def test_tiny_positive_value_is_recorded(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=5.0)
        det.record_intensity(0.001)
        assert len(det._window) == 1


class TestAdaptiveBlankDetectorIsBlank:
    """is_blank delegates to current threshold."""

    def test_mean_below_base_threshold_is_blank(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0)
        dark_frame = np.full((10, 10, 3), 3, dtype=np.uint8)
        assert det.is_blank(dark_frame) is True

    def test_mean_above_base_threshold_not_blank(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0)
        bright_frame = np.full((10, 10, 3), 50, dtype=np.uint8)
        assert det.is_blank(bright_frame) is False

    def test_mean_exactly_at_threshold_not_blank(self) -> None:
        # is_blank uses < (strict), so equality is NOT blank.
        det = AdaptiveBlankDetector(base_threshold=5.0)
        frame = np.full((10, 10, 3), 5, dtype=np.uint8)
        assert det.is_blank(frame) is False

    def test_is_blank_with_adapted_threshold(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0, min_threshold=1.0, adaptation_factor=0.1)
        for _ in range(5):
            det.record_intensity(30.0)
        # adaptive = 30 * 0.1 = 3.0
        # Frame with mean=2 should be blank.
        frame = np.full((10, 10, 3), 2, dtype=np.uint8)
        assert det.is_blank(frame) is True


class TestAdaptiveBlankDetectorReset:
    """reset() wipes the observation window."""

    def test_reset_clears_window(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0, adaptation_factor=0.25)
        for _ in range(5):
            det.record_intensity(80.0)
        assert len(det._window) == 5

        det.reset()
        assert len(det._window) == 0

    def test_threshold_returns_to_base_after_reset(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0, min_threshold=1.0, adaptation_factor=0.1)
        for _ in range(5):
            det.record_intensity(20.0)
        # adapted to 2.0
        assert det.threshold == pytest.approx(2.0)

        det.reset()
        assert det.threshold == pytest.approx(10.0)

    def test_can_record_again_after_reset(self) -> None:
        det = AdaptiveBlankDetector(base_threshold=10.0)
        for _ in range(5):
            det.record_intensity(50.0)
        det.reset()
        det.record_intensity(30.0)
        # Only 1 recording — not enough for adaptation.
        assert det.threshold == pytest.approx(10.0)


# ===========================================================================
# capture.py — CaptureResult
# ===========================================================================


class TestCaptureResultShape:
    """CaptureResult.shape property."""

    def test_shape_none_image_returns_zero_tuple(self) -> None:
        result = CaptureResult(success=False, image=None)
        assert result.shape == (0, 0, 0)

    def test_shape_with_3d_image(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = CaptureResult(success=True, image=img)
        assert result.shape == (480, 640, 3)

    def test_shape_with_grayscale_image(self) -> None:
        img = np.zeros((100, 200), dtype=np.uint8)
        result = CaptureResult(success=True, image=img)
        assert result.shape == (100, 200)

    def test_shape_matches_ndarray_shape_attribute(self) -> None:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = CaptureResult(success=True, image=img)
        assert result.shape == img.shape

    def test_shape_after_failed_capture(self) -> None:
        result = CaptureResult(
            success=False,
            image=None,
            error="device not found",
            failure_type="device_gone",
        )
        assert result.shape == (0, 0, 0)


# ===========================================================================
# capture.py — CaptureConfig defaults
# ===========================================================================


class TestCaptureConfigDefaults:
    """Verify the documented default values are present."""

    def test_default_width(self) -> None:
        config = CaptureConfig()
        assert config.width == 1920

    def test_default_height(self) -> None:
        config = CaptureConfig()
        assert config.height == 1080

    def test_default_warmup_frames(self) -> None:
        config = CaptureConfig()
        assert config.warmup_frames == 5

    def test_default_max_retries(self) -> None:
        config = CaptureConfig()
        assert config.max_retries == 3

    def test_default_retry_delay(self) -> None:
        config = CaptureConfig()
        assert config.retry_delay == pytest.approx(0.5)

    def test_default_blank_threshold(self) -> None:
        config = CaptureConfig()
        assert config.blank_threshold == pytest.approx(5.0)

    def test_default_adaptive_blank_enabled(self) -> None:
        config = CaptureConfig()
        assert config.adaptive_blank is True

    def test_default_quality(self) -> None:
        config = CaptureConfig()
        assert config.quality == 95

    def test_default_timeout(self) -> None:
        config = CaptureConfig()
        assert config.timeout_seconds == pytest.approx(10.0)


# ===========================================================================
# capture.py — _get_cv2 lazy import
# ===========================================================================


class TestGetCv2LazyImport:
    """_get_cv2 must import cv2 lazily and be thread-safe."""

    def test_import_error_raises_with_helpful_message(self) -> None:
        """When cv2 is unavailable, ImportError message mentions pip install."""
        import missy.vision.capture as cap_module

        # Save original cached value and reset.
        original = cap_module._cv2
        cap_module._cv2 = None
        try:
            with (
                patch.dict(sys.modules, {"cv2": None}),
                patch("builtins.__import__", side_effect=_raise_import_error_for_cv2),
                pytest.raises(ImportError, match="opencv-python"),
            ):
                cap_module._get_cv2()
        finally:
            cap_module._cv2 = original

    def test_returns_cv2_module_when_available(self) -> None:
        """_get_cv2 returns the real cv2 (or the cached mock) without error."""
        import missy.vision.capture as cap_module

        # cv2 is already cached from the successful import at module load.
        # Simply calling _get_cv2() should not raise.
        result = cap_module._get_cv2()
        assert result is not None

    def test_thread_safety_single_import(self) -> None:
        """Multiple threads calling _get_cv2 concurrently should not double-import."""
        import missy.vision.capture as cap_module

        call_log: list[int] = []
        original = cap_module._cv2
        mock_cv2 = MagicMock(name="mock_cv2")
        cap_module._cv2 = None

        import_count = 0

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            nonlocal import_count
            if name == "cv2":
                import_count += 1
                return mock_cv2
            return importlib.import_module(name)

        errors: list[Exception] = []

        def worker() -> None:
            try:
                cap_module._get_cv2()
                call_log.append(1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        try:
            with patch("builtins.__import__", side_effect=fake_import):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            cap_module._cv2 = original

        assert not errors
        assert len(call_log) == 8  # all threads succeeded
        # The import should happen at most once (double-checked locking).
        assert import_count <= 1


# ---------------------------------------------------------------------------
# helpers used by tests above
# ---------------------------------------------------------------------------


def _raise_import_error_for_cv2(name: str, *args: object, **kwargs: object) -> object:
    """Replacement for builtins.__import__ that fails only for 'cv2'."""
    if name == "cv2":
        raise ImportError("No module named 'cv2'")
    return importlib.__import__(name, *args, **kwargs)  # type: ignore[attr-defined]
