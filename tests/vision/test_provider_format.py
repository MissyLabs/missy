"""Tests for missy.vision.provider_format — provider-specific image formatting."""

from __future__ import annotations

from missy.vision.provider_format import (
    build_vision_message,
    format_image_for_anthropic,
    format_image_for_openai,
    format_image_for_provider,
)


class TestFormatImageForAnthropic:
    def test_structure(self):
        result = format_image_for_anthropic("abc123")
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/jpeg"
        assert result["source"]["data"] == "abc123"

    def test_custom_media_type(self):
        result = format_image_for_anthropic("abc", media_type="image/png")
        assert result["source"]["media_type"] == "image/png"


class TestFormatImageForOpenai:
    def test_structure(self):
        result = format_image_for_openai("abc123")
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert "abc123" in result["image_url"]["url"]
        assert result["image_url"]["detail"] == "auto"

    def test_custom_detail(self):
        result = format_image_for_openai("abc", detail="high")
        assert result["image_url"]["detail"] == "high"


class TestFormatImageForProvider:
    def test_anthropic(self):
        result = format_image_for_provider("anthropic", "abc")
        assert result["type"] == "image"

    def test_openai(self):
        result = format_image_for_provider("openai", "abc")
        assert result["type"] == "image_url"

    def test_gpt(self):
        result = format_image_for_provider("gpt", "abc")
        assert result["type"] == "image_url"

    def test_ollama(self):
        result = format_image_for_provider("ollama", "abc")
        assert result["type"] == "image_url"

    def test_unknown_defaults_to_anthropic(self):
        result = format_image_for_provider("unknown_provider", "abc")
        assert result["type"] == "image"

    def test_case_insensitive(self):
        result = format_image_for_provider("ANTHROPIC", "abc")
        assert result["type"] == "image"


class TestBuildVisionMessage:
    def test_anthropic_message(self):
        msg = build_vision_message("anthropic", "abc", "Describe this image")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "Describe this image"

    def test_openai_message(self):
        msg = build_vision_message("openai", "abc", "What do you see?")
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "image_url"
        assert msg["content"][1]["text"] == "What do you see?"
