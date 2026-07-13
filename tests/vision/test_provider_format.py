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

    def test_openai_codex(self):
        """Regression: "openai-codex" is a real, known Missy provider
        name (missy/providers/codex_provider.py), not an unknown one --
        it was previously falling into the generic unknown-provider
        fallback below (Anthropic-shaped block), which
        CodexProvider._messages_to_input()'s block parser doesn't
        recognise at all, silently dropping the image from the request
        entirely for any deployment using openai-codex as its provider."""
        result = format_image_for_provider("openai-codex", "abc")
        assert result["type"] == "image_url"

    def test_codex_alias(self):
        result = format_image_for_provider("codex", "abc")
        assert result["type"] == "image_url"

    def test_openai_codex_case_insensitive(self):
        result = format_image_for_provider("OPENAI-CODEX", "abc")
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

    def test_openai_codex_message_is_parseable_by_codex_provider(self):
        """Cross-module regression: the block build_vision_message()
        produces for "openai-codex" must be something
        CodexProvider._messages_to_input() actually recognises as an
        image, not just structurally valid JSON -- this is the exact
        format/parser mismatch that silently dropped images from every
        openai-codex vision_analyze call."""
        from missy.providers.base import Message
        from missy.providers.codex_provider import _messages_to_input

        msg_dict = build_vision_message("openai-codex", "ZmFrZQ==", "Read the text in this image")
        message = Message(role=msg_dict["role"], content=msg_dict["content"])

        result = _messages_to_input([message])
        blocks = result[0]["content"]
        image_blocks = [b for b in blocks if b["type"] == "input_image"]
        text_blocks = [b for b in blocks if b["type"] == "input_text"]

        assert len(image_blocks) == 1, "image block must survive the round-trip, not be dropped"
        assert "ZmFrZQ==" in image_blocks[0]["image_url"]
        assert text_blocks[0]["text"] == "Read the text in this image"

    def test_ollama_message_is_parseable_by_ollama_provider(self):
        """Same cross-module regression, for Ollama's distinct
        content+images shape."""
        from missy.providers.base import Message
        from missy.providers.ollama_provider import _message_to_ollama_payload

        msg_dict = build_vision_message("ollama", "ZmFrZQ==", "Read the text in this image")
        message = Message(role=msg_dict["role"], content=msg_dict["content"])

        payload = _message_to_ollama_payload(message)
        assert payload["images"] == ["ZmFrZQ=="]
        assert payload["content"] == "Read the text in this image"
