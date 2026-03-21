"""Tests for missy.vision.config_validator.

Covers:
- validate_vision_config with valid defaults
- Invalid types for each configurable field
- Out-of-range values for numeric fields
- Warning-level findings (non-standard resolution, excessive warmup,
  low threshold, high frame/session count, high retry count)
- Invalid preferred_device path format
- ValidationResult properties: errors, warnings, to_dict
- ValidationIssue dataclass field contract
"""

from __future__ import annotations

from missy.vision.config_validator import (
    ValidationIssue,
    ValidationResult,
    validate_vision_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_DEFAULTS: dict = {
    "enabled": True,
    "capture_width": 1920,
    "capture_height": 1080,
    "warmup_frames": 5,
    "max_retries": 3,
    "auto_activate_threshold": 0.80,
    "scene_memory_max_frames": 20,
    "scene_memory_max_sessions": 5,
    "preferred_device": "",
}


def _fields_with_errors(result: ValidationResult) -> list[str]:
    return [i.field for i in result.errors]


def _fields_with_warnings(result: ValidationResult) -> list[str]:
    return [i.field for i in result.warnings]


# ---------------------------------------------------------------------------
# ValidationIssue dataclass
# ---------------------------------------------------------------------------


class TestValidationIssue:
    def test_required_fields(self):
        issue = ValidationIssue(field="foo", message="bad", severity="error")
        assert issue.field == "foo"
        assert issue.message == "bad"
        assert issue.severity == "error"

    def test_optional_fields_default_to_none(self):
        issue = ValidationIssue(field="x", message="y", severity="warning")
        assert issue.current_value is None
        assert issue.suggested_value is None

    def test_optional_fields_set_correctly(self):
        issue = ValidationIssue(
            field="capture_width",
            message="too small",
            severity="error",
            current_value=10,
            suggested_value=1920,
        )
        assert issue.current_value == 10
        assert issue.suggested_value == 1920

    def test_info_severity_accepted(self):
        issue = ValidationIssue(field="x", message="y", severity="info")
        assert issue.severity == "info"


# ---------------------------------------------------------------------------
# ValidationResult dataclass and properties
# ---------------------------------------------------------------------------


class TestValidationResult:
    def _make_result(self) -> ValidationResult:
        return ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(field="a", message="err", severity="error"),
                ValidationIssue(field="b", message="warn", severity="warning"),
                ValidationIssue(field="c", message="info", severity="info"),
                ValidationIssue(field="d", message="warn2", severity="warning"),
            ],
        )

    def test_errors_property_filters_by_severity(self):
        result = self._make_result()
        errors = result.errors
        assert len(errors) == 1
        assert all(i.severity == "error" for i in errors)

    def test_warnings_property_filters_by_severity(self):
        result = self._make_result()
        warnings = result.warnings
        assert len(warnings) == 2
        assert all(i.severity == "warning" for i in warnings)

    def test_errors_excludes_warnings(self):
        result = self._make_result()
        assert not any(i.severity == "warning" for i in result.errors)

    def test_warnings_excludes_errors(self):
        result = self._make_result()
        assert not any(i.severity == "error" for i in result.warnings)

    def test_to_dict_structure(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["valid"] is False
        assert d["error_count"] == 1
        assert d["warning_count"] == 2
        assert len(d["issues"]) == 4

    def test_to_dict_issue_keys(self):
        result = self._make_result()
        issue_dicts = result.to_dict()["issues"]
        required_keys = {"field", "message", "severity", "current_value", "suggested_value"}
        for issue_dict in issue_dicts:
            assert required_keys == set(issue_dict.keys())

    def test_to_dict_valid_true(self):
        result = ValidationResult(valid=True, issues=[])
        d = result.to_dict()
        assert d["valid"] is True
        assert d["error_count"] == 0
        assert d["warning_count"] == 0
        assert d["issues"] == []

    def test_to_dict_preserves_current_and_suggested_values(self):
        result = ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(
                    field="capture_width",
                    message="bad",
                    severity="error",
                    current_value=50,
                    suggested_value=1920,
                )
            ],
        )
        issue_dict = result.to_dict()["issues"][0]
        assert issue_dict["current_value"] == 50
        assert issue_dict["suggested_value"] == 1920

    def test_empty_result_is_valid(self):
        result = ValidationResult(valid=True)
        assert result.errors == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# validate_vision_config — valid configurations
# ---------------------------------------------------------------------------


class TestValidDefaults:
    def test_all_defaults_valid(self):
        result = validate_vision_config(VALID_DEFAULTS)
        assert result.valid is True
        assert result.errors == []

    def test_empty_dict_uses_defaults_and_is_valid(self):
        result = validate_vision_config({})
        assert result.valid is True
        assert result.errors == []

    def test_enabled_true(self):
        result = validate_vision_config({"enabled": True})
        assert result.valid is True

    def test_enabled_false(self):
        result = validate_vision_config({"enabled": False})
        assert result.valid is True

    def test_known_standard_resolutions(self):
        standard = [
            (640, 480),
            (800, 600),
            (1280, 720),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
        ]
        for width, height in standard:
            result = validate_vision_config(
                {**VALID_DEFAULTS, "capture_width": width, "capture_height": height}
            )
            assert result.valid is True, f"{width}x{height} should be valid"
            # No warning for standard resolutions
            width_warnings = [w for w in result.warnings if w.field == "capture_width"]
            assert width_warnings == [], f"{width}x{height} should not warn"

    def test_preferred_device_empty_string_no_warning(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": ""})
        device_issues = [i for i in result.issues if i.field == "preferred_device"]
        assert device_issues == []

    def test_preferred_device_valid_path(self):
        for path in ["/dev/video0", "/dev/video1", "/dev/video99"]:
            result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": path})
            device_issues = [i for i in result.issues if i.field == "preferred_device"]
            assert device_issues == [], f"{path} should be accepted without issues"

    def test_threshold_at_boundary_zero(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 0.0})
        # 0.0 is in-range but below 0.5 — warning, not error
        assert result.valid is True
        assert "auto_activate_threshold" in _fields_with_warnings(result)

    def test_threshold_at_boundary_one(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 1.0})
        assert result.valid is True
        assert "auto_activate_threshold" not in _fields_with_errors(result)

    def test_warmup_zero_is_valid(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 0})
        assert result.valid is True
        assert "warmup_frames" not in _fields_with_errors(result)

    def test_max_retries_one_is_valid(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 1})
        assert result.valid is True
        assert "max_retries" not in _fields_with_errors(result)

    def test_scene_frames_one_is_valid(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 1})
        assert result.valid is True
        assert "scene_memory_max_frames" not in _fields_with_errors(result)

    def test_scene_sessions_one_is_valid(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 1})
        assert result.valid is True
        assert "scene_memory_max_sessions" not in _fields_with_errors(result)


# ---------------------------------------------------------------------------
# validate_vision_config — invalid types
# ---------------------------------------------------------------------------


class TestInvalidTypes:
    def test_enabled_string(self):
        result = validate_vision_config({**VALID_DEFAULTS, "enabled": "yes"})
        assert result.valid is False
        assert "enabled" in _fields_with_errors(result)

    def test_enabled_integer(self):
        result = validate_vision_config({**VALID_DEFAULTS, "enabled": 1})
        assert result.valid is False
        assert "enabled" in _fields_with_errors(result)

    def test_enabled_none(self):
        result = validate_vision_config({**VALID_DEFAULTS, "enabled": None})
        assert result.valid is False
        assert "enabled" in _fields_with_errors(result)

    def test_capture_width_string(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": "1920"})
        assert result.valid is False
        assert "capture_width" in _fields_with_errors(result)

    def test_capture_width_float(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": 1920.0})
        assert result.valid is False
        assert "capture_width" in _fields_with_errors(result)

    def test_capture_width_none(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": None})
        assert result.valid is False
        assert "capture_width" in _fields_with_errors(result)

    def test_capture_height_string(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": "1080"})
        assert result.valid is False
        assert "capture_height" in _fields_with_errors(result)

    def test_capture_height_float(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 1080.5})
        assert result.valid is False
        assert "capture_height" in _fields_with_errors(result)

    def test_capture_height_none(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": None})
        assert result.valid is False
        assert "capture_height" in _fields_with_errors(result)

    def test_warmup_frames_string(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": "5"})
        assert result.valid is False
        assert "warmup_frames" in _fields_with_errors(result)

    def test_warmup_frames_float(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 5.0})
        assert result.valid is False
        assert "warmup_frames" in _fields_with_errors(result)

    def test_warmup_frames_none(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": None})
        assert result.valid is False
        assert "warmup_frames" in _fields_with_errors(result)

    def test_max_retries_string_no_error(self):
        # max_retries only validates when isinstance(retries, int); string silently skips
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": "3"})
        assert "max_retries" not in _fields_with_errors(result)

    def test_auto_activate_threshold_string_no_error(self):
        # threshold only validates when isinstance(threshold, (int, float)); string skips
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": "0.8"})
        assert "auto_activate_threshold" not in _fields_with_errors(result)

    def test_scene_memory_max_frames_string_no_error(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": "20"})
        assert "scene_memory_max_frames" not in _fields_with_errors(result)

    def test_scene_memory_max_sessions_string_no_error(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": "5"})
        assert "scene_memory_max_sessions" not in _fields_with_errors(result)


# ---------------------------------------------------------------------------
# validate_vision_config — out-of-range values (errors)
# ---------------------------------------------------------------------------


class TestOutOfRangeErrors:
    # capture_width
    def test_capture_width_below_minimum(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": 159})
        assert result.valid is False
        assert "capture_width" in _fields_with_errors(result)

    def test_capture_width_at_minimum(self):
        # 160 is the boundary; (160, 1080) is non-standard so only a warning
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": 160})
        assert result.valid is True
        assert "capture_width" not in _fields_with_errors(result)

    def test_capture_width_above_maximum(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": 3841})
        assert result.valid is False
        assert "capture_width" in _fields_with_errors(result)

    def test_capture_width_at_maximum_with_standard_height(self):
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 3840, "capture_height": 2160}
        )
        assert result.valid is True
        assert "capture_width" not in _fields_with_errors(result)

    def test_capture_width_error_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_width": 50})
        error = next(i for i in result.errors if i.field == "capture_width")
        assert error.suggested_value == 1920
        assert error.current_value == 50

    # capture_height
    def test_capture_height_below_minimum(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 119})
        assert result.valid is False
        assert "capture_height" in _fields_with_errors(result)

    def test_capture_height_at_minimum(self):
        # 120 is the boundary; (1920, 120) is non-standard so only a width warning
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 120})
        assert result.valid is True
        assert "capture_height" not in _fields_with_errors(result)

    def test_capture_height_above_maximum(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 2161})
        assert result.valid is False
        assert "capture_height" in _fields_with_errors(result)

    def test_capture_height_error_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 10})
        error = next(i for i in result.errors if i.field == "capture_height")
        assert error.suggested_value == 1080
        assert error.current_value == 10

    # warmup_frames
    def test_warmup_frames_negative(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": -1})
        assert result.valid is False
        assert "warmup_frames" in _fields_with_errors(result)

    def test_warmup_frames_negative_error_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": -5})
        error = next(i for i in result.errors if i.field == "warmup_frames")
        assert error.suggested_value == 5
        assert error.current_value == -5

    # max_retries
    def test_max_retries_zero(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 0})
        assert result.valid is False
        assert "max_retries" in _fields_with_errors(result)

    def test_max_retries_negative(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": -1})
        assert result.valid is False
        assert "max_retries" in _fields_with_errors(result)

    def test_max_retries_zero_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 0})
        error = next(i for i in result.errors if i.field == "max_retries")
        assert error.suggested_value == 3

    # auto_activate_threshold
    def test_threshold_negative(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": -0.1})
        assert result.valid is False
        assert "auto_activate_threshold" in _fields_with_errors(result)

    def test_threshold_above_one(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 1.1})
        assert result.valid is False
        assert "auto_activate_threshold" in _fields_with_errors(result)

    def test_threshold_exactly_negative_zero_point_zero_one(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": -0.01})
        assert result.valid is False
        assert "auto_activate_threshold" in _fields_with_errors(result)

    def test_threshold_error_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 2.0})
        error = next(i for i in result.errors if i.field == "auto_activate_threshold")
        assert error.suggested_value == 0.80

    # scene_memory_max_frames
    def test_scene_frames_zero(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 0})
        assert result.valid is False
        assert "scene_memory_max_frames" in _fields_with_errors(result)

    def test_scene_frames_negative(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": -10})
        assert result.valid is False
        assert "scene_memory_max_frames" in _fields_with_errors(result)

    def test_scene_frames_zero_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 0})
        error = next(i for i in result.errors if i.field == "scene_memory_max_frames")
        assert error.suggested_value == 20

    # scene_memory_max_sessions
    def test_scene_sessions_zero(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 0})
        assert result.valid is False
        assert "scene_memory_max_sessions" in _fields_with_errors(result)

    def test_scene_sessions_negative(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": -1})
        assert result.valid is False
        assert "scene_memory_max_sessions" in _fields_with_errors(result)

    def test_scene_sessions_zero_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 0})
        error = next(i for i in result.errors if i.field == "scene_memory_max_sessions")
        assert error.suggested_value == 5


# ---------------------------------------------------------------------------
# validate_vision_config — warning-level issues
# ---------------------------------------------------------------------------


class TestWarnings:
    def test_non_standard_resolution_raises_width_warning(self):
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 1024, "capture_height": 768}
        )
        assert result.valid is True
        assert "capture_width" in _fields_with_warnings(result)

    def test_non_standard_width_warning_contains_resolution_string(self):
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 1024, "capture_height": 768}
        )
        warning = next(i for i in result.warnings if i.field == "capture_width")
        assert "1024" in warning.message
        assert "768" in warning.message

    def test_non_standard_resolution_still_valid(self):
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 1024, "capture_height": 768}
        )
        assert result.valid is True

    def test_non_standard_resolution_no_height_error(self):
        # Height is in-range so no error for it
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 1024, "capture_height": 768}
        )
        assert "capture_height" not in _fields_with_errors(result)

    def test_excessive_warmup_raises_warning(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 31})
        assert result.valid is True
        assert "warmup_frames" in _fields_with_warnings(result)

    def test_excessive_warmup_at_boundary(self):
        # Exactly at the limit (30) should NOT warn; 31 should
        result_ok = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 30})
        assert "warmup_frames" not in _fields_with_warnings(result_ok)

        result_warn = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 31})
        assert "warmup_frames" in _fields_with_warnings(result_warn)

    def test_excessive_warmup_warning_has_suggested_value(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 50})
        warning = next(i for i in result.warnings if i.field == "warmup_frames")
        assert warning.suggested_value == 5

    def test_excessive_warmup_warning_mentions_frame_count(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 50})
        warning = next(i for i in result.warnings if i.field == "warmup_frames")
        assert "50" in warning.message

    def test_low_threshold_below_0_5_warns(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 0.3})
        assert result.valid is True
        assert "auto_activate_threshold" in _fields_with_warnings(result)

    def test_low_threshold_at_0_5_no_warning(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 0.5})
        assert "auto_activate_threshold" not in _fields_with_warnings(result)

    def test_low_threshold_just_below_0_5_warns(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 0.49})
        assert "auto_activate_threshold" in _fields_with_warnings(result)

    def test_low_threshold_integer_zero_warns(self):
        # Integer 0 is valid (isinstance check passes for int) and below 0.5
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 0})
        assert result.valid is True
        assert "auto_activate_threshold" in _fields_with_warnings(result)

    def test_high_max_retries_warns(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 11})
        assert result.valid is True
        assert "max_retries" in _fields_with_warnings(result)

    def test_max_retries_at_limit_no_warning(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 10})
        assert "max_retries" not in _fields_with_warnings(result)

    def test_max_retries_one_over_limit_warns(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 11})
        warning = next(i for i in result.warnings if i.field == "max_retries")
        assert "11" in warning.message

    def test_high_scene_frames_warns(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 101})
        assert result.valid is True
        assert "scene_memory_max_frames" in _fields_with_warnings(result)

    def test_scene_frames_at_limit_no_warning(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 100})
        assert "scene_memory_max_frames" not in _fields_with_warnings(result)

    def test_high_scene_frames_warning_mentions_memory_estimate(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 200})
        warning = next(i for i in result.warnings if i.field == "scene_memory_max_frames")
        # 200 * 6 = 1200 MB — message should reference MB
        assert "MB" in warning.message or "1200" in warning.message

    def test_high_scene_sessions_warns(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 21})
        assert result.valid is True
        assert "scene_memory_max_sessions" in _fields_with_warnings(result)

    def test_scene_sessions_at_limit_no_warning(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 20})
        assert "scene_memory_max_sessions" not in _fields_with_warnings(result)

    def test_high_scene_sessions_warning_mentions_session_count(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 25})
        warning = next(i for i in result.warnings if i.field == "scene_memory_max_sessions")
        assert "25" in warning.message


# ---------------------------------------------------------------------------
# validate_vision_config — preferred_device format
# ---------------------------------------------------------------------------


class TestPreferredDevice:
    def test_valid_video0(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/video0"})
        assert "preferred_device" not in [i.field for i in result.issues]

    def test_valid_video9(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/video9"})
        assert "preferred_device" not in [i.field for i in result.issues]

    def test_valid_video_multi_digit(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/video12"})
        assert "preferred_device" not in [i.field for i in result.issues]

    def test_invalid_no_leading_slash(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "dev/video0"})
        assert result.valid is True  # only a warning, not an error
        assert "preferred_device" in _fields_with_warnings(result)

    def test_invalid_wrong_subsystem(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/sda"})
        assert "preferred_device" in _fields_with_warnings(result)

    def test_invalid_no_number(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/video"})
        assert "preferred_device" in _fields_with_warnings(result)

    def test_invalid_extra_path_component(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/video0/extra"})
        assert "preferred_device" in _fields_with_warnings(result)

    def test_invalid_device_warning_has_current_value(self):
        bad_path = "/dev/cam0"
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": bad_path})
        warning = next(i for i in result.warnings if i.field == "preferred_device")
        assert warning.current_value == bad_path

    def test_invalid_device_warning_mentions_path_in_message(self):
        bad_path = "/dev/cam0"
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": bad_path})
        warning = next(i for i in result.warnings if i.field == "preferred_device")
        assert bad_path in warning.message

    def test_invalid_device_is_warning_not_error(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": "/dev/cam0"})
        assert "preferred_device" not in _fields_with_errors(result)
        assert "preferred_device" in _fields_with_warnings(result)

    def test_empty_string_bypasses_check(self):
        result = validate_vision_config({**VALID_DEFAULTS, "preferred_device": ""})
        device_issues = [i for i in result.issues if i.field == "preferred_device"]
        assert device_issues == []


# ---------------------------------------------------------------------------
# validate_vision_config — multiple simultaneous errors
# ---------------------------------------------------------------------------


class TestMultipleErrors:
    def test_multiple_errors_all_reported(self):
        config = {
            "enabled": "yes",
            "capture_width": 50,
            "capture_height": 10,
            "warmup_frames": -1,
        }
        result = validate_vision_config(config)
        assert result.valid is False
        error_fields = _fields_with_errors(result)
        assert "enabled" in error_fields
        assert "capture_width" in error_fields
        assert "capture_height" in error_fields
        assert "warmup_frames" in error_fields

    def test_error_and_warning_both_reported(self):
        config = {
            **VALID_DEFAULTS,
            "capture_width": 50,  # error: out of range
            "warmup_frames": 31,  # warning: excessive
        }
        result = validate_vision_config(config)
        assert result.valid is False
        assert "capture_width" in _fields_with_errors(result)
        assert "warmup_frames" in _fields_with_warnings(result)

    def test_warnings_only_result_is_valid(self):
        config = {
            **VALID_DEFAULTS,
            "warmup_frames": 31,  # warning
            "auto_activate_threshold": 0.3,  # warning
            "scene_memory_max_frames": 150,  # warning
            "preferred_device": "/dev/cam0",  # warning
        }
        result = validate_vision_config(config)
        assert result.valid is True
        assert result.errors == []
        assert len(result.warnings) == 4

    def test_valid_flag_false_only_for_errors(self):
        # All warnings, no errors
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 50})
        assert result.valid is True

        # One error
        result_err = validate_vision_config({**VALID_DEFAULTS, "capture_width": 1})
        assert result_err.valid is False

    def test_to_dict_counts_match_issues(self):
        config = {
            **VALID_DEFAULTS,
            "capture_width": 50,  # error
            "warmup_frames": 31,  # warning
            "auto_activate_threshold": 0.2,  # warning
        }
        result = validate_vision_config(config)
        d = result.to_dict()
        assert d["error_count"] == len(result.errors)
        assert d["warning_count"] == len(result.warnings)
        assert len(d["issues"]) == len(result.issues)


# ---------------------------------------------------------------------------
# validate_vision_config — edge cases for boundary values
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    def test_width_exactly_160(self):
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 160, "capture_height": 120}
        )
        assert "capture_width" not in _fields_with_errors(result)

    def test_width_exactly_3840_height_2160(self):
        result = validate_vision_config(
            {**VALID_DEFAULTS, "capture_width": 3840, "capture_height": 2160}
        )
        assert "capture_width" not in _fields_with_errors(result)

    def test_height_exactly_120(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 120})
        assert "capture_height" not in _fields_with_errors(result)

    def test_height_exactly_2160(self):
        result = validate_vision_config({**VALID_DEFAULTS, "capture_height": 2160})
        assert "capture_height" not in _fields_with_errors(result)

    def test_warmup_exactly_0(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 0})
        assert "warmup_frames" not in _fields_with_errors(result)
        assert "warmup_frames" not in _fields_with_warnings(result)

    def test_warmup_exactly_30(self):
        result = validate_vision_config({**VALID_DEFAULTS, "warmup_frames": 30})
        assert "warmup_frames" not in _fields_with_errors(result)
        assert "warmup_frames" not in _fields_with_warnings(result)

    def test_retries_exactly_10(self):
        result = validate_vision_config({**VALID_DEFAULTS, "max_retries": 10})
        assert "max_retries" not in _fields_with_errors(result)
        assert "max_retries" not in _fields_with_warnings(result)

    def test_threshold_exactly_0_5(self):
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 0.5})
        assert "auto_activate_threshold" not in _fields_with_errors(result)
        assert "auto_activate_threshold" not in _fields_with_warnings(result)

    def test_scene_frames_exactly_100(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_frames": 100})
        assert "scene_memory_max_frames" not in _fields_with_errors(result)
        assert "scene_memory_max_frames" not in _fields_with_warnings(result)

    def test_scene_sessions_exactly_20(self):
        result = validate_vision_config({**VALID_DEFAULTS, "scene_memory_max_sessions": 20})
        assert "scene_memory_max_sessions" not in _fields_with_errors(result)
        assert "scene_memory_max_sessions" not in _fields_with_warnings(result)

    def test_threshold_integer_one_is_valid_no_warning(self):
        # Integer 1 is isinstance (int, float) and equals 1.0 — valid
        result = validate_vision_config({**VALID_DEFAULTS, "auto_activate_threshold": 1})
        assert "auto_activate_threshold" not in _fields_with_errors(result)
        assert "auto_activate_threshold" not in _fields_with_warnings(result)
