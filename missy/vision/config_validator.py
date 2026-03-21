"""Vision configuration validation.

Validates vision-related settings from ``config.yaml`` to catch
misconfigurations early (before camera open, capture, or analysis fails).

Checks:
- Resolution within supported ranges
- Warmup frame count sane
- Scene memory limits reasonable
- Auto-activation threshold bounded
- Device path format valid
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A single configuration validation finding."""

    field: str
    message: str
    severity: str  # "error", "warning", "info"
    current_value: Any = None
    suggested_value: Any = None


@dataclass
class ValidationResult:
    """Aggregated validation results."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "field": i.field,
                    "message": i.message,
                    "severity": i.severity,
                    "current_value": i.current_value,
                    "suggested_value": i.suggested_value,
                }
                for i in self.issues
            ],
        }


# ---------------------------------------------------------------------------
# Supported ranges
# ---------------------------------------------------------------------------

_MIN_WIDTH = 160
_MAX_WIDTH = 3840
_MIN_HEIGHT = 120
_MAX_HEIGHT = 2160
_COMMON_RESOLUTIONS = {
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
}
_MAX_WARMUP = 30
_MAX_RETRIES = 10
_MAX_SCENE_FRAMES = 100
_MAX_SCENE_SESSIONS = 20


def validate_vision_config(config: dict[str, Any]) -> ValidationResult:
    """Validate a vision configuration dictionary.

    Parameters
    ----------
    config:
        The ``vision:`` section of the Missy config, as a dict.

    Returns
    -------
    ValidationResult
        Contains ``valid`` flag and list of issues.
    """
    issues: list[ValidationIssue] = []

    # enabled
    enabled = config.get("enabled", True)
    if not isinstance(enabled, bool):
        issues.append(
            ValidationIssue(
                field="enabled",
                message="Must be a boolean",
                severity="error",
                current_value=enabled,
                suggested_value=True,
            )
        )

    # capture_width
    width = config.get("capture_width", 1920)
    if not isinstance(width, int):
        issues.append(
            ValidationIssue(
                field="capture_width",
                message="Must be an integer",
                severity="error",
                current_value=width,
            )
        )
    elif width < _MIN_WIDTH or width > _MAX_WIDTH:
        issues.append(
            ValidationIssue(
                field="capture_width",
                message=f"Must be between {_MIN_WIDTH} and {_MAX_WIDTH}",
                severity="error",
                current_value=width,
                suggested_value=1920,
            )
        )
    elif (width, config.get("capture_height", 1080)) not in _COMMON_RESOLUTIONS:
        issues.append(
            ValidationIssue(
                field="capture_width",
                message=f"Non-standard resolution {width}x{config.get('capture_height', 1080)} — camera may not support it",
                severity="warning",
                current_value=width,
            )
        )

    # capture_height
    height = config.get("capture_height", 1080)
    if not isinstance(height, int):
        issues.append(
            ValidationIssue(
                field="capture_height",
                message="Must be an integer",
                severity="error",
                current_value=height,
            )
        )
    elif height < _MIN_HEIGHT or height > _MAX_HEIGHT:
        issues.append(
            ValidationIssue(
                field="capture_height",
                message=f"Must be between {_MIN_HEIGHT} and {_MAX_HEIGHT}",
                severity="error",
                current_value=height,
                suggested_value=1080,
            )
        )

    # warmup_frames
    warmup = config.get("warmup_frames", 5)
    if not isinstance(warmup, int):
        issues.append(
            ValidationIssue(
                field="warmup_frames",
                message="Must be an integer",
                severity="error",
                current_value=warmup,
            )
        )
    elif warmup < 0:
        issues.append(
            ValidationIssue(
                field="warmup_frames",
                message="Cannot be negative",
                severity="error",
                current_value=warmup,
                suggested_value=5,
            )
        )
    elif warmup > _MAX_WARMUP:
        issues.append(
            ValidationIssue(
                field="warmup_frames",
                message=f"Excessive warmup ({warmup}) will slow capture startup",
                severity="warning",
                current_value=warmup,
                suggested_value=5,
            )
        )

    # max_retries
    retries = config.get("max_retries", 3)
    if isinstance(retries, int):
        if retries < 1:
            issues.append(
                ValidationIssue(
                    field="max_retries",
                    message="Must be at least 1",
                    severity="error",
                    current_value=retries,
                    suggested_value=3,
                )
            )
        elif retries > _MAX_RETRIES:
            issues.append(
                ValidationIssue(
                    field="max_retries",
                    message=f"Very high retry count ({retries}) may cause long delays",
                    severity="warning",
                    current_value=retries,
                    suggested_value=3,
                )
            )

    # auto_activate_threshold
    threshold = config.get("auto_activate_threshold", 0.80)
    if isinstance(threshold, (int, float)):
        if threshold < 0.0 or threshold > 1.0:
            issues.append(
                ValidationIssue(
                    field="auto_activate_threshold",
                    message="Must be between 0.0 and 1.0",
                    severity="error",
                    current_value=threshold,
                    suggested_value=0.80,
                )
            )
        elif threshold < 0.5:
            issues.append(
                ValidationIssue(
                    field="auto_activate_threshold",
                    message="Low threshold may cause excessive auto-activation",
                    severity="warning",
                    current_value=threshold,
                )
            )

    # scene_memory_max_frames
    max_frames = config.get("scene_memory_max_frames", 20)
    if isinstance(max_frames, int):
        if max_frames < 1:
            issues.append(
                ValidationIssue(
                    field="scene_memory_max_frames",
                    message="Must be at least 1",
                    severity="error",
                    current_value=max_frames,
                    suggested_value=20,
                )
            )
        elif max_frames > _MAX_SCENE_FRAMES:
            # Estimate memory: 1920x1080x3 = ~6MB per frame
            est_mb = max_frames * 6
            issues.append(
                ValidationIssue(
                    field="scene_memory_max_frames",
                    message=f"High frame count ({max_frames}) may use ~{est_mb} MB per session",
                    severity="warning",
                    current_value=max_frames,
                )
            )

    # scene_memory_max_sessions
    max_sessions = config.get("scene_memory_max_sessions", 5)
    if isinstance(max_sessions, int):
        if max_sessions < 1:
            issues.append(
                ValidationIssue(
                    field="scene_memory_max_sessions",
                    message="Must be at least 1",
                    severity="error",
                    current_value=max_sessions,
                    suggested_value=5,
                )
            )
        elif max_sessions > _MAX_SCENE_SESSIONS:
            issues.append(
                ValidationIssue(
                    field="scene_memory_max_sessions",
                    message=f"Very high session count ({max_sessions}) may use excessive memory",
                    severity="warning",
                    current_value=max_sessions,
                )
            )

    # preferred_device
    device = config.get("preferred_device", "")
    if device and not re.match(r"^/dev/video\d+$", device):
        issues.append(
            ValidationIssue(
                field="preferred_device",
                message=f"Device path should match /dev/videoN pattern, got: {device}",
                severity="warning",
                current_value=device,
            )
        )

    valid = not any(i.severity == "error" for i in issues)
    return ValidationResult(valid=valid, issues=issues)
