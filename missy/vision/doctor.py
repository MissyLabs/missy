"""Vision subsystem diagnostics and health checks.

Provides the ``missy vision doctor`` functionality: tests camera pipeline,
permissions, discovery, capture, and reports actionable findings.
"""

from __future__ import annotations

import grp
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # "info", "warning", "error"


@dataclass
class DoctorReport:
    """Complete vision doctor report."""

    results: list[DiagnosticResult] = field(default_factory=list)
    overall_healthy: bool = True

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "warning")

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity == "error")

    def add(self, result: DiagnosticResult) -> None:
        self.results.append(result)
        if not result.passed and result.severity == "error":
            self.overall_healthy = False


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------


class VisionDoctor:
    """Runs diagnostic checks on the vision subsystem."""

    def run_all(self) -> DoctorReport:
        """Run all diagnostic checks and return a report."""
        report = DoctorReport()

        report.add(self.check_opencv())
        report.add(self.check_video_group())
        report.add(self.check_video_devices())
        report.add(self.check_sysfs())
        report.add(self.check_camera_discovery())
        report.add(self.check_screenshot_tools())
        report.add(self.check_numpy())
        report.add(self.check_captures_directory())

        # Only attempt capture if we found devices and opencv is available
        has_opencv = any(
            r.name == "opencv" and r.passed for r in report.results
        )
        has_devices = any(
            r.name == "video_devices" and r.passed for r in report.results
        )
        if has_opencv and has_devices:
            report.add(self.check_capture())

        return report

    def check_opencv(self) -> DiagnosticResult:
        """Check if OpenCV is installed, functional, and meets the minimum version."""
        try:
            import cv2

            version = cv2.__version__
            parts = version.split(".")
            try:
                major = int(parts[0])
            except (ValueError, IndexError):
                major = 0

            if major < 4:
                return DiagnosticResult(
                    name="opencv",
                    passed=False,
                    message=(
                        f"OpenCV {version} is below the minimum required version 4.0. "
                        "Upgrade with: pip install --upgrade opencv-python-headless"
                    ),
                    details={"version": version, "major": major, "minimum_major": 4},
                    severity="error",
                )

            return DiagnosticResult(
                name="opencv",
                passed=True,
                message=f"OpenCV {version} is available (>= 4.0)",
                details={"version": version, "major": major},
            )
        except ImportError:
            return DiagnosticResult(
                name="opencv",
                passed=False,
                message="OpenCV not installed. Install with: pip install opencv-python-headless",
                severity="error",
            )

    def check_numpy(self) -> DiagnosticResult:
        """Check numpy availability."""
        try:
            import numpy as np
            return DiagnosticResult(
                name="numpy",
                passed=True,
                message=f"NumPy {np.__version__} is available",
                details={"version": np.__version__},
            )
        except ImportError:
            return DiagnosticResult(
                name="numpy",
                passed=False,
                message="NumPy not installed",
                severity="error",
            )

    def check_video_group(self) -> DiagnosticResult:
        """Check if current user is in the 'video' group.

        Uses ``os.getgroups()`` to inspect the process's actual supplementary
        group list, which correctly reflects ``newgrp`` / ``sg`` activations and
        avoids false negatives from comparing usernames against ``gr_mem``.
        """
        try:
            username = os.getlogin()
        except OSError:
            username = os.environ.get("USER", "unknown")

        try:
            video_gid = grp.getgrnam("video").gr_gid
        except KeyError:
            return DiagnosticResult(
                name="video_group",
                passed=False,
                message="'video' group does not exist on this system",
                severity="warning",
            )

        try:
            current_gids = os.getgroups()
        except OSError:
            current_gids = []

        if video_gid in current_gids or os.getegid() == video_gid:
            return DiagnosticResult(
                name="video_group",
                passed=True,
                message=f"User '{username}' is in the 'video' group (gid={video_gid})",
                details={"video_gid": video_gid},
            )

        return DiagnosticResult(
            name="video_group",
            passed=False,
            message=(
                f"User '{username}' is NOT in the 'video' group. "
                f"Run: sudo usermod -aG video {username}"
            ),
            details={"video_gid": video_gid, "current_gids": current_gids},
            severity="warning",
        )

    def check_video_devices(self) -> DiagnosticResult:
        """Check for /dev/video* device nodes."""
        dev_path = Path("/dev")
        devices = sorted(dev_path.glob("video*"))

        if not devices:
            return DiagnosticResult(
                name="video_devices",
                passed=False,
                message="No /dev/video* devices found. Is a camera connected?",
                severity="error",
                details={"devices": []},
            )

        device_info = []
        for d in devices:
            try:
                d.stat()
                readable = os.access(str(d), os.R_OK)
                writable = os.access(str(d), os.W_OK)
                device_info.append({
                    "path": str(d),
                    "readable": readable,
                    "writable": writable,
                })
            except OSError:
                device_info.append({"path": str(d), "error": "stat failed"})

        accessible = [d for d in device_info if d.get("readable") and d.get("writable")]

        if accessible:
            return DiagnosticResult(
                name="video_devices",
                passed=True,
                message=f"Found {len(devices)} video device(s), {len(accessible)} accessible",
                details={"devices": device_info},
            )
        else:
            return DiagnosticResult(
                name="video_devices",
                passed=False,
                message=f"Found {len(devices)} video device(s) but none are accessible. Check permissions.",
                severity="error",
                details={"devices": device_info},
            )

    def check_sysfs(self) -> DiagnosticResult:
        """Check sysfs video4linux directory."""
        sysfs = Path("/sys/class/video4linux")
        if not sysfs.exists():
            return DiagnosticResult(
                name="sysfs",
                passed=False,
                message="/sys/class/video4linux not found — no V4L2 support",
                severity="warning",
            )

        entries = list(sysfs.iterdir())
        return DiagnosticResult(
            name="sysfs",
            passed=len(entries) > 0,
            message=f"Found {len(entries)} V4L2 device(s) in sysfs",
            details={"entries": [e.name for e in entries]},
        )

    def check_camera_discovery(self) -> DiagnosticResult:
        """Run camera discovery and report findings."""
        try:
            from missy.vision.discovery import CameraDiscovery

            disc = CameraDiscovery()
            cameras = disc.discover(force=True)

            if not cameras:
                return DiagnosticResult(
                    name="camera_discovery",
                    passed=False,
                    message="Discovery found no cameras",
                    severity="warning",
                )

            info = []
            unreadable: list[str] = []
            for c in cameras:
                readable = os.access(c.device_path, os.R_OK)
                if not readable:
                    unreadable.append(c.device_path)
                info.append(
                    {
                        "device": c.device_path,
                        "name": c.name,
                        "usb_id": c.usb_id,
                        "bus": c.bus_info,
                        "readable": readable,
                    }
                )

            if unreadable:
                return DiagnosticResult(
                    name="camera_discovery",
                    passed=False,
                    message=(
                        f"Discovered {len(cameras)} camera(s) but "
                        f"{len(unreadable)} are not readable by the current user: "
                        f"{', '.join(unreadable)}"
                    ),
                    details={"cameras": info, "unreadable": unreadable},
                    severity="warning",
                )

            return DiagnosticResult(
                name="camera_discovery",
                passed=True,
                message=f"Discovered {len(cameras)} camera(s): {cameras[0].name}",
                details={"cameras": info},
            )
        except Exception as exc:
            return DiagnosticResult(
                name="camera_discovery",
                passed=False,
                message=f"Discovery failed: {exc}",
                severity="error",
            )

    def check_screenshot_tools(self) -> DiagnosticResult:
        """Check if screenshot tools are available."""
        tools = ["scrot", "gnome-screenshot", "grim"]
        available = []
        for tool in tools:
            if shutil.which(tool):
                available.append(tool)

        if available:
            return DiagnosticResult(
                name="screenshot_tools",
                passed=True,
                message=f"Screenshot tools available: {', '.join(available)}",
                details={"tools": available},
            )
        else:
            return DiagnosticResult(
                name="screenshot_tools",
                passed=False,
                message="No screenshot tools found. Install scrot: sudo apt install scrot",
                severity="warning",
            )

    def check_capture(self) -> DiagnosticResult:
        """Attempt a test capture from the preferred camera."""
        try:
            from missy.vision.capture import CameraHandle, CaptureConfig
            from missy.vision.discovery import find_preferred_camera

            camera = find_preferred_camera()
            if not camera:
                return DiagnosticResult(
                    name="capture_test",
                    passed=False,
                    message="No camera available for capture test",
                    severity="warning",
                )

            config = CaptureConfig(warmup_frames=3, max_retries=2)
            handle = CameraHandle(camera.device_path, config)

            try:
                handle.open()
                result = handle.capture()
                if result.success:
                    return DiagnosticResult(
                        name="capture_test",
                        passed=True,
                        message=f"Capture test passed: {result.width}x{result.height} from {camera.name}",
                        details={
                            "width": result.width,
                            "height": result.height,
                            "device": camera.device_path,
                            "camera": camera.name,
                        },
                    )
                else:
                    return DiagnosticResult(
                        name="capture_test",
                        passed=False,
                        message=f"Capture test failed: {result.error}",
                        severity="error",
                    )
            finally:
                handle.close()

        except Exception as exc:
            return DiagnosticResult(
                name="capture_test",
                passed=False,
                message=f"Capture test error: {exc}",
                severity="error",
            )

    def check_captures_directory(self) -> DiagnosticResult:
        """Check if the captures directory exists, is writable, and has sufficient disk space."""
        _MIN_FREE_MB = 100
        captures_dir = Path.home() / ".missy" / "captures"
        try:
            if not captures_dir.exists():
                captures_dir.mkdir(parents=True, exist_ok=True)

            if not os.access(str(captures_dir), os.W_OK):
                return DiagnosticResult(
                    name="captures_dir",
                    passed=False,
                    message=f"Captures directory not writable: {captures_dir}",
                    severity="warning",
                )

            usage = shutil.disk_usage(captures_dir)
            free_mb = usage.free // (1024 * 1024)
            total_mb = usage.total // (1024 * 1024)

            if free_mb < _MIN_FREE_MB:
                return DiagnosticResult(
                    name="captures_dir",
                    passed=False,
                    message=(
                        f"Low disk space for captures: {free_mb} MB free "
                        f"(minimum {_MIN_FREE_MB} MB recommended). "
                        f"Free up space on the partition containing {captures_dir}."
                    ),
                    details={
                        "path": str(captures_dir),
                        "free_mb": free_mb,
                        "total_mb": total_mb,
                        "minimum_free_mb": _MIN_FREE_MB,
                    },
                    severity="warning",
                )

            return DiagnosticResult(
                name="captures_dir",
                passed=True,
                message=f"Captures directory ready: {captures_dir} ({free_mb} MB free)",
                details={
                    "path": str(captures_dir),
                    "free_mb": free_mb,
                    "total_mb": total_mb,
                },
            )
        except OSError as exc:
            return DiagnosticResult(
                name="captures_dir",
                passed=False,
                message=f"Cannot access captures directory: {exc}",
                severity="warning",
            )
