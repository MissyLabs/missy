"""Tests for vision config validator edge cases.

Covers:
- Valid configurations
- Invalid types
- Boundary values
- Resolution validation
- Device path format checking
"""

from __future__ import annotations

from missy.vision.config_validator import validate_vision_config


class TestConfigValidatorValid:
    """Test that valid configs pass validation."""

    def test_default_config_valid(self):
        result = validate_vision_config({})
        assert result.valid

    def test_standard_1080p(self):
        result = validate_vision_config({
            "capture_width": 1920,
            "capture_height": 1080,
        })
        assert result.valid

    def test_standard_720p(self):
        result = validate_vision_config({
            "capture_width": 1280,
            "capture_height": 720,
        })
        assert result.valid

    def test_4k_valid(self):
        result = validate_vision_config({
            "capture_width": 3840,
            "capture_height": 2160,
        })
        assert result.valid

    def test_all_fields_specified(self):
        result = validate_vision_config({
            "enabled": True,
            "capture_width": 1920,
            "capture_height": 1080,
            "warmup_frames": 5,
            "max_retries": 3,
            "auto_activate_threshold": 0.80,
            "scene_memory_max_frames": 20,
            "scene_memory_max_sessions": 5,
            "preferred_device": "",
        })
        assert result.valid


class TestConfigValidatorInvalid:
    """Test that invalid configs are caught."""

    def test_enabled_not_bool(self):
        result = validate_vision_config({"enabled": "yes"})
        assert not result.valid
        assert any(i.field == "enabled" for i in result.errors)

    def test_width_too_small(self):
        result = validate_vision_config({"capture_width": 50})
        assert not result.valid

    def test_width_too_large(self):
        result = validate_vision_config({"capture_width": 10000})
        assert not result.valid

    def test_height_too_small(self):
        result = validate_vision_config({"capture_height": 10})
        assert not result.valid

    def test_height_too_large(self):
        result = validate_vision_config({"capture_height": 5000})
        assert not result.valid

    def test_width_not_int(self):
        result = validate_vision_config({"capture_width": "wide"})
        assert not result.valid

    def test_height_not_int(self):
        result = validate_vision_config({"capture_height": 10.5})
        assert not result.valid

    def test_warmup_negative(self):
        result = validate_vision_config({"warmup_frames": -1})
        assert not result.valid

    def test_warmup_not_int(self):
        result = validate_vision_config({"warmup_frames": "five"})
        assert not result.valid

    def test_retries_zero(self):
        result = validate_vision_config({"max_retries": 0})
        assert not result.valid

    def test_threshold_too_high(self):
        result = validate_vision_config({"auto_activate_threshold": 1.5})
        assert not result.valid

    def test_threshold_negative(self):
        result = validate_vision_config({"auto_activate_threshold": -0.1})
        assert not result.valid

    def test_scene_frames_zero(self):
        result = validate_vision_config({"scene_memory_max_frames": 0})
        assert not result.valid

    def test_scene_sessions_zero(self):
        result = validate_vision_config({"scene_memory_max_sessions": 0})
        assert not result.valid


class TestConfigValidatorWarnings:
    """Test warning-level issues."""

    def test_non_standard_resolution_warns(self):
        result = validate_vision_config({
            "capture_width": 1024,
            "capture_height": 768,
        })
        # Non-standard but within range = warning, not error
        assert result.valid
        assert result.warnings

    def test_excessive_warmup_warns(self):
        result = validate_vision_config({"warmup_frames": 50})
        assert result.valid
        assert any(i.field == "warmup_frames" for i in result.warnings)

    def test_excessive_retries_warns(self):
        result = validate_vision_config({"max_retries": 50})
        assert result.valid
        assert result.warnings

    def test_low_threshold_warns(self):
        result = validate_vision_config({"auto_activate_threshold": 0.3})
        assert result.valid
        assert result.warnings

    def test_high_scene_frames_warns(self):
        result = validate_vision_config({"scene_memory_max_frames": 200})
        assert result.valid
        assert result.warnings

    def test_high_scene_sessions_warns(self):
        result = validate_vision_config({"scene_memory_max_sessions": 50})
        assert result.valid
        assert result.warnings

    def test_invalid_device_path_warns(self):
        result = validate_vision_config({"preferred_device": "/dev/sda1"})
        assert result.valid  # warning, not error
        assert result.warnings


class TestValidationResult:
    """Test ValidationResult methods."""

    def test_to_dict(self):
        result = validate_vision_config({})
        d = result.to_dict()
        assert "valid" in d
        assert "error_count" in d
        assert "warning_count" in d
        assert "issues" in d

    def test_error_count(self):
        result = validate_vision_config({"capture_width": "bad", "capture_height": "bad"})
        assert result.to_dict()["error_count"] >= 2

    def test_empty_config_no_errors(self):
        result = validate_vision_config({})
        assert result.to_dict()["error_count"] == 0
