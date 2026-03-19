"""USB webcam discovery with stable device identification.

Discovers cameras by scanning ``/sys/class/video4linux/`` and matching USB
vendor/product IDs and device names.  Handles re-enumeration gracefully by
identifying cameras by their USB topology (bus/port) rather than volatile
``/dev/videoN`` paths.

Design decisions
----------------
- Uses sysfs directly for stable USB identification instead of relying on
  ``/dev/videoN`` ordering which changes across reboots/reconnects.
- Filters to ``VIDEO_CAPTURE`` capable devices only (skips metadata nodes).
- Caches discovery results with a configurable TTL to avoid repeated sysfs
  traversal.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraDevice:
    """Represents a discovered camera device."""

    device_path: str  # e.g. /dev/video0
    name: str  # e.g. "Logitech C922x Pro Stream Webcam"
    vendor_id: str  # e.g. "046d"
    product_id: str  # e.g. "085c"
    bus_info: str  # e.g. "usb-0000:00:14.0-1"
    capabilities: list[str] = field(default_factory=list)

    @property
    def usb_id(self) -> str:
        """Return ``vendor:product`` USB ID string."""
        return f"{self.vendor_id}:{self.product_id}"

    @property
    def is_logitech_c922(self) -> bool:
        return self.vendor_id == "046d" and self.product_id in ("085c", "085b")


# Known camera USB IDs for quick identification
KNOWN_CAMERAS: dict[str, str] = {
    "046d:085c": "Logitech C922x Pro Stream",
    "046d:085b": "Logitech C922 Pro Stream",
    "046d:0825": "Logitech HD Webcam C270",
    "046d:082d": "Logitech HD Pro Webcam C920",
    "046d:0843": "Logitech Webcam C930e",
}


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------


class CameraDiscovery:
    """Discovers USB cameras via sysfs with caching and re-enumeration safety.

    Parameters
    ----------
    cache_ttl_seconds:
        How long cached results remain valid.  Default 10 s.
    sysfs_base:
        Base path for video4linux sysfs entries.  Override for testing.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 10.0,
        sysfs_base: str = "/sys/class/video4linux",
    ) -> None:
        self._cache_ttl = cache_ttl_seconds
        self._sysfs_base = Path(sysfs_base)
        self._cache: list[CameraDevice] = []
        self._cache_time: float = 0.0

    # -- public API --

    def discover(self, *, force: bool = False) -> list[CameraDevice]:
        """Return list of available camera devices.

        Uses cached results if still valid unless *force* is True.
        """
        now = time.monotonic()
        if not force and self._cache and (now - self._cache_time) < self._cache_ttl:
            return list(self._cache)

        devices = self._scan_sysfs()
        self._cache = devices
        self._cache_time = now
        logger.info("Discovered %d camera device(s)", len(devices))
        for dev in devices:
            logger.debug("  %s — %s [%s]", dev.device_path, dev.name, dev.usb_id)
        return list(devices)

    def find_by_usb_id(self, vendor_id: str, product_id: str) -> Optional[CameraDevice]:
        """Find a specific camera by USB vendor/product ID."""
        for dev in self.discover():
            if dev.vendor_id == vendor_id and dev.product_id == product_id:
                return dev
        return None

    def find_by_name(self, name_pattern: str) -> list[CameraDevice]:
        """Find cameras whose name matches a regex pattern (case-insensitive)."""
        try:
            pattern = re.compile(name_pattern, re.IGNORECASE)
        except re.error as exc:
            logger.warning("Invalid regex pattern %r: %s", name_pattern, exc)
            return []
        return [dev for dev in self.discover() if pattern.search(dev.name)]

    def find_preferred(self) -> Optional[CameraDevice]:
        """Find the preferred camera (Logitech C922x first, then any known, then first)."""
        devices = self.discover()
        if not devices:
            return None

        # Prefer Logitech C922x
        for dev in devices:
            if dev.is_logitech_c922:
                return dev

        # Then any known camera
        for dev in devices:
            if dev.usb_id in KNOWN_CAMERAS:
                return dev

        # Fall back to first device
        return devices[0]

    # -- internal --

    def _scan_sysfs(self) -> list[CameraDevice]:
        """Walk /sys/class/video4linux/ and build CameraDevice entries."""
        devices: list[CameraDevice] = []

        if not self._sysfs_base.exists():
            logger.warning("sysfs path %s does not exist", self._sysfs_base)
            return devices

        try:
            entries = sorted(self._sysfs_base.iterdir())
        except OSError as exc:
            logger.warning("Cannot read sysfs directory %s: %s", self._sysfs_base, exc)
            return devices

        for entry in entries:
            if not entry.name.startswith("video"):
                continue

            device_path = f"/dev/{entry.name}"
            if not Path(device_path).exists():
                continue

            # Check this is a capture device (not metadata)
            if not self._is_capture_device(entry):
                continue

            name = self._read_sysfs_attr(entry, "name") or f"Unknown ({entry.name})"
            vendor_id, product_id = self._read_usb_ids(entry)
            bus_info = self._read_bus_info(entry)

            devices.append(
                CameraDevice(
                    device_path=device_path,
                    name=name,
                    vendor_id=vendor_id,
                    product_id=product_id,
                    bus_info=bus_info,
                )
            )

        return devices

    def _is_capture_device(self, sysfs_entry: Path) -> bool:
        """Check if this v4l2 device supports VIDEO_CAPTURE."""
        # v4l2 index file — device node 0 is typically the capture node.
        index_path = sysfs_entry / "index"
        if index_path.exists():
            try:
                idx = int(index_path.read_text().strip())
                if idx != 0:
                    return False
            except (ValueError, OSError):
                pass

        # Also check uevent for DEVNAME
        uevent = sysfs_entry / "uevent"
        if uevent.exists():
            try:
                text = uevent.read_text()
                if "DEVNAME" not in text:
                    return False
            except OSError:
                pass

        return True

    def _read_sysfs_attr(self, sysfs_entry: Path, attr: str) -> str:
        """Read a sysfs attribute, returning empty string on failure."""
        path = sysfs_entry / attr
        try:
            return path.read_text().strip() if path.exists() else ""
        except OSError:
            return ""

    def _read_usb_ids(self, sysfs_entry: Path) -> tuple[str, str]:
        """Walk up sysfs to find USB vendor/product IDs."""
        # Resolve symlink to find real device path
        try:
            real_path = (sysfs_entry / "device").resolve()
        except OSError:
            return ("0000", "0000")

        # Walk up looking for idVendor/idProduct files
        current = real_path
        for _ in range(10):  # limit depth
            vid_path = current / "idVendor"
            pid_path = current / "idProduct"
            if vid_path.exists() and pid_path.exists():
                try:
                    vid = vid_path.read_text().strip()
                    pid = pid_path.read_text().strip()
                    return (vid, pid)
                except OSError:
                    pass
            parent = current.parent
            if parent == current:
                break
            current = parent

        return ("0000", "0000")

    def _read_bus_info(self, sysfs_entry: Path) -> str:
        """Determine bus info from sysfs device symlink."""
        try:
            real_path = (sysfs_entry / "device").resolve()
            # The USB bus info is typically in the path
            parts = str(real_path).split("/")
            for i, part in enumerate(parts):
                if part.startswith("usb"):
                    return "/".join(parts[i:])
            return str(real_path)
        except OSError:
            return ""


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_discovery: CameraDiscovery | None = None


def get_discovery() -> CameraDiscovery:
    """Return (or create) the module-level CameraDiscovery instance."""
    global _default_discovery
    if _default_discovery is None:
        _default_discovery = CameraDiscovery()
    return _default_discovery


def discover_cameras(*, force: bool = False) -> list[CameraDevice]:
    """Convenience wrapper: discover all cameras."""
    return get_discovery().discover(force=force)


def find_preferred_camera() -> Optional[CameraDevice]:
    """Convenience wrapper: find preferred camera."""
    return get_discovery().find_preferred()
