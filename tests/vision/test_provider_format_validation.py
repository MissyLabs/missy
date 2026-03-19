"""Tests for provider_format.py input validation."""

import pytest

from missy.vision.provider_format import (
    build_vision_message,
    format_image_for_provider,
)


class TestFormatImageForProviderValidation:
    """Tests for input validation in format_image_for_provider."""

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider("", "abc123", "image/jpeg")

    def test_whitespace_provider_raises(self):
        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider("   ", "abc123", "image/jpeg")

    def test_none_provider_raises(self):
        with pytest.raises(ValueError, match="provider_name"):
            format_image_for_provider(None, "abc123", "image/jpeg")

    def test_empty_image_base64_raises(self):
        with pytest.raises(ValueError, match="image_base64"):
            format_image_for_provider("anthropic", "", "image/jpeg")

    def test_none_image_base64_raises(self):
        with pytest.raises(ValueError, match="image_base64"):
            format_image_for_provider("anthropic", None, "image/jpeg")

    def test_empty_media_type_raises(self):
        with pytest.raises(ValueError, match="media_type"):
            format_image_for_provider("anthropic", "abc123", "")

    def test_whitespace_media_type_raises(self):
        with pytest.raises(ValueError, match="media_type"):
            format_image_for_provider("anthropic", "abc123", "   ")

    def test_valid_inputs_succeed(self):
        result = format_image_for_provider("anthropic", "abc123", "image/jpeg")
        assert result["type"] == "image"

    def test_provider_name_stripped(self):
        """Leading/trailing whitespace in provider_name should be stripped."""
        result = format_image_for_provider("  anthropic  ", "abc123", "image/jpeg")
        assert result["type"] == "image"


class TestBuildVisionMessageValidation:
    """Tests for input validation in build_vision_message."""

    def test_empty_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt"):
            build_vision_message("anthropic", "abc123", "", "image/jpeg")

    def test_none_prompt_raises(self):
        with pytest.raises(ValueError, match="prompt"):
            build_vision_message("anthropic", "abc123", None, "image/jpeg")

    def test_valid_inputs_succeed(self):
        msg = build_vision_message("anthropic", "abc123", "describe this", "image/jpeg")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
