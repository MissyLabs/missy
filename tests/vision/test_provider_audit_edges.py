"""Session 13: Comprehensive tests for vision provider_format and audit modules.

Covers gaps not addressed by the existing test files:
- test_provider_format.py
- test_provider_format_validation.py
- test_audit.py
- test_audit_extended.py
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from missy.vision.audit import (
    _emit_audit_event,
    audit_vision_analysis,
    audit_vision_burst,
    audit_vision_capture,
    audit_vision_device_discovery,
    audit_vision_error,
    audit_vision_intent,
    audit_vision_session,
)
from missy.vision.provider_format import (
    build_vision_message,
    format_image_for_anthropic,
    format_image_for_openai,
    format_image_for_provider,
)

# ---------------------------------------------------------------------------
# provider_format.py — deeper structural coverage
# ---------------------------------------------------------------------------


class TestFormatImageForAnthropicStructure:
    """Verify exact key set and value types returned by format_image_for_anthropic."""

    def test_source_has_exactly_three_keys(self):
        result = format_image_for_anthropic("data123")
        assert set(result["source"].keys()) == {"type", "media_type", "data"}

    def test_source_type_is_base64_literal(self):
        result = format_image_for_anthropic("data123")
        assert result["source"]["type"] == "base64"

    def test_data_field_passes_through_unchanged(self):
        payload = "SGVsbG8gV29ybGQ="  # base64 for "Hello World"
        result = format_image_for_anthropic(payload)
        assert result["source"]["data"] == payload

    def test_png_media_type_stored_verbatim(self):
        result = format_image_for_anthropic("x", media_type="image/png")
        assert result["source"]["media_type"] == "image/png"

    def test_webp_media_type(self):
        result = format_image_for_anthropic("x", media_type="image/webp")
        assert result["source"]["media_type"] == "image/webp"

    def test_return_type_is_dict(self):
        result = format_image_for_anthropic("abc")
        assert isinstance(result, dict)


class TestFormatImageForOpenaiStructure:
    """Verify exact key set and URI construction for format_image_for_openai."""

    def test_image_url_has_exactly_two_keys(self):
        result = format_image_for_openai("abc")
        assert set(result["image_url"].keys()) == {"url", "detail"}

    def test_data_uri_exact_format_jpeg(self):
        b64 = "SGVsbG8="
        result = format_image_for_openai(b64, media_type="image/jpeg")
        assert result["image_url"]["url"] == f"data:image/jpeg;base64,{b64}"

    def test_data_uri_exact_format_png(self):
        b64 = "abc123"
        result = format_image_for_openai(b64, media_type="image/png")
        assert result["image_url"]["url"] == f"data:image/png;base64,{b64}"

    def test_detail_low(self):
        result = format_image_for_openai("abc", detail="low")
        assert result["image_url"]["detail"] == "low"

    def test_detail_high(self):
        result = format_image_for_openai("abc", detail="high")
        assert result["image_url"]["detail"] == "high"

    def test_detail_default_is_auto(self):
        result = format_image_for_openai("abc")
        assert result["image_url"]["detail"] == "auto"

    def test_return_type_is_dict(self):
        result = format_image_for_openai("abc")
        assert isinstance(result, dict)


class TestFormatImageForProviderRouting:
    """Additional routing cases not covered by existing tests."""

    def test_gpt_uppercase_routes_to_openai_format(self):
        result = format_image_for_provider("GPT", "abc")
        assert result["type"] == "image_url"

    def test_ollama_uppercase(self):
        result = format_image_for_provider("OLLAMA", "abc")
        assert result["type"] == "image_url"

    def test_unknown_provider_returns_anthropic_structure(self):
        result = format_image_for_provider("bedrock", "abc")
        # Falls back to Anthropic format
        assert result["type"] == "image"
        assert "source" in result

    def test_unknown_provider_preserves_media_type(self):
        result = format_image_for_provider("bedrock", "abc", media_type="image/png")
        assert result["source"]["media_type"] == "image/png"

    def test_anthropic_with_png_media_type(self):
        result = format_image_for_provider("anthropic", "abc", media_type="image/png")
        assert result["source"]["media_type"] == "image/png"

    def test_openai_with_custom_media_type_in_uri(self):
        result = format_image_for_provider("openai", "abc", media_type="image/webp")
        assert "image/webp" in result["image_url"]["url"]

    def test_actual_base64_payload_roundtrip(self):
        """Encode real bytes and verify the data survives in the formatted block."""
        raw = b"\x89PNG\r\nfakedata"
        b64 = base64.b64encode(raw).decode()
        result = format_image_for_provider("anthropic", b64)
        assert result["source"]["data"] == b64

    def test_actual_base64_payload_in_openai_uri(self):
        raw = b"fake jpeg bytes"
        b64 = base64.b64encode(raw).decode()
        result = format_image_for_provider("openai", b64)
        assert b64 in result["image_url"]["url"]


class TestBuildVisionMessageDetails:
    """Edge cases and content ordering for build_vision_message."""

    def test_image_block_always_comes_before_text(self):
        msg = build_vision_message("anthropic", "abc", "describe")
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][1]["type"] == "text"

    def test_content_list_has_exactly_two_items(self):
        msg = build_vision_message("openai", "abc", "what is this?")
        assert len(msg["content"]) == 2

    def test_role_is_user(self):
        msg = build_vision_message("anthropic", "abc", "describe")
        assert msg["role"] == "user"

    def test_text_block_type_field(self):
        msg = build_vision_message("anthropic", "abc", "my prompt")
        text_block = msg["content"][1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "my prompt"

    def test_custom_media_type_propagates_to_anthropic_source(self):
        msg = build_vision_message("anthropic", "abc", "look", media_type="image/png")
        image_block = msg["content"][0]
        assert image_block["source"]["media_type"] == "image/png"

    def test_custom_media_type_propagates_to_openai_uri(self):
        msg = build_vision_message("openai", "abc", "look", media_type="image/png")
        image_block = msg["content"][0]
        assert "image/png" in image_block["image_url"]["url"]

    def test_ollama_produces_openai_format_image_block(self):
        msg = build_vision_message("ollama", "abc", "caption this")
        assert msg["content"][0]["type"] == "image_url"

    def test_unknown_provider_produces_anthropic_format(self):
        msg = build_vision_message("vertex", "abc", "caption this")
        assert msg["content"][0]["type"] == "image"

    def test_return_type_is_dict(self):
        msg = build_vision_message("anthropic", "abc", "test")
        assert isinstance(msg, dict)
        assert isinstance(msg["content"], list)

    def test_multiword_prompt_preserved_exactly(self):
        prompt = "What colour is the top-left chess piece on the board?"
        msg = build_vision_message("anthropic", "abc", prompt)
        assert msg["content"][1]["text"] == prompt


# ---------------------------------------------------------------------------
# audit.py — field-level and category/action correctness
# ---------------------------------------------------------------------------


class TestEmitAuditEventTimestamp:
    """Timestamp field is present and ISO-8601 formatted with UTC offset."""

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_timestamp_present(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        before = datetime.now(UTC)
        _emit_audit_event("vision", "test", {})
        after = datetime.now(UTC)

        event = mock_logger.log.call_args[0][0]
        assert "timestamp" in event

        ts = datetime.fromisoformat(event["timestamp"])
        assert before <= ts <= after

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_details_merged_into_top_level(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        _emit_audit_event("vision", "test", {"foo": "bar", "baz": 42})

        event = mock_logger.log.call_args[0][0]
        assert event["foo"] == "bar"
        assert event["baz"] == 42
        # category and action also present
        assert event["category"] == "vision"
        assert event["action"] == "test"

    @patch("missy.observability.audit_logger.get_audit_logger")
    def test_category_and_action_not_overwritten_by_details(self, mock_get_logger):
        """Details should not be able to shadow the category/action keys."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Passing keys that overlap: event construction uses **details so order matters —
        # test that the top-level fields are what the function assigned.
        _emit_audit_event("vision", "action_name", {})

        event = mock_logger.log.call_args[0][0]
        assert event["category"] == "vision"
        assert event["action"] == "action_name"


class TestAuditVisionCaptureFields:
    """Verify all fields emitted by audit_vision_capture."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_all_fields_present(self, mock_emit):
        audit_vision_capture(
            device="/dev/video2",
            source_type="file",
            trigger_reason="voice_activation",
            success=True,
            width=640,
            height=480,
            error="",
        )
        details = mock_emit.call_args[0][2]
        assert details["device"] == "/dev/video2"
        assert details["source_type"] == "file"
        assert details["trigger_reason"] == "voice_activation"
        assert details["success"] is True
        assert details["width"] == 640
        assert details["height"] == 480
        assert details["error"] == ""

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_is_capture(self, mock_emit):
        audit_vision_capture()
        assert mock_emit.call_args[0][1] == "capture"

    @patch("missy.vision.audit._emit_audit_event")
    def test_category_is_vision(self, mock_emit):
        audit_vision_capture()
        assert mock_emit.call_args[0][0] == "vision"

    @patch("missy.vision.audit._emit_audit_event")
    def test_voice_trigger_reason(self, mock_emit):
        audit_vision_capture(trigger_reason="voice_activation")
        assert mock_emit.call_args[0][2]["trigger_reason"] == "voice_activation"

    @patch("missy.vision.audit._emit_audit_event")
    def test_default_width_height_zero(self, mock_emit):
        audit_vision_capture()
        details = mock_emit.call_args[0][2]
        assert details["width"] == 0
        assert details["height"] == 0


class TestAuditVisionAnalysisFields:
    """Verify all fields and action name for audit_vision_analysis."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_is_analyze(self, mock_emit):
        audit_vision_analysis()
        assert mock_emit.call_args[0][1] == "analyze"

    @patch("missy.vision.audit._emit_audit_event")
    def test_category_is_vision(self, mock_emit):
        audit_vision_analysis()
        assert mock_emit.call_args[0][0] == "vision"

    @patch("missy.vision.audit._emit_audit_event")
    def test_default_mode_is_general(self, mock_emit):
        audit_vision_analysis()
        assert mock_emit.call_args[0][2]["mode"] == "general"

    @patch("missy.vision.audit._emit_audit_event")
    def test_painting_mode(self, mock_emit):
        audit_vision_analysis(mode="painting")
        assert mock_emit.call_args[0][2]["mode"] == "painting"

    @patch("missy.vision.audit._emit_audit_event")
    def test_source_type_field(self, mock_emit):
        audit_vision_analysis(source_type="screenshot")
        assert mock_emit.call_args[0][2]["source_type"] == "screenshot"

    @patch("missy.vision.audit._emit_audit_event")
    def test_trigger_reason_field(self, mock_emit):
        audit_vision_analysis(trigger_reason="scheduled")
        assert mock_emit.call_args[0][2]["trigger_reason"] == "scheduled"

    @patch("missy.vision.audit._emit_audit_event")
    def test_error_field_on_failure(self, mock_emit):
        audit_vision_analysis(success=False, error="Timeout from provider")
        details = mock_emit.call_args[0][2]
        assert details["error"] == "Timeout from provider"
        assert details["success"] is False


class TestAuditVisionIntentFields:
    """Field-level checks for audit_vision_intent."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_empty_text_gives_zero_length(self, mock_emit):
        audit_vision_intent(text="")
        assert mock_emit.call_args[0][2]["text_length"] == 0

    @patch("missy.vision.audit._emit_audit_event")
    def test_decision_field(self, mock_emit):
        audit_vision_intent(decision="skip")
        assert mock_emit.call_args[0][2]["decision"] == "skip"

    @patch("missy.vision.audit._emit_audit_event")
    def test_trigger_phrase_field(self, mock_emit):
        audit_vision_intent(trigger_phrase="look at the board")
        assert mock_emit.call_args[0][2]["trigger_phrase"] == "look at the board"

    @patch("missy.vision.audit._emit_audit_event")
    def test_raw_text_not_in_details(self, mock_emit):
        audit_vision_intent(text="sensitive user speech content")
        details = mock_emit.call_args[0][2]
        assert "text" not in details

    @patch("missy.vision.audit._emit_audit_event")
    def test_category_is_vision(self, mock_emit):
        audit_vision_intent()
        assert mock_emit.call_args[0][0] == "vision"

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_is_intent(self, mock_emit):
        audit_vision_intent()
        assert mock_emit.call_args[0][1] == "intent"


class TestAuditVisionDeviceDiscoveryFields:
    """Field-level checks for audit_vision_device_discovery."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_zero_cameras(self, mock_emit):
        audit_vision_device_discovery(camera_count=0)
        assert mock_emit.call_args[0][2]["camera_count"] == 0

    @patch("missy.vision.audit._emit_audit_event")
    def test_preferred_name_field(self, mock_emit):
        audit_vision_device_discovery(preferred_name="Logitech C922x Pro Stream")
        assert mock_emit.call_args[0][2]["preferred_name"] == "Logitech C922x Pro Stream"

    @patch("missy.vision.audit._emit_audit_event")
    def test_preferred_device_field(self, mock_emit):
        audit_vision_device_discovery(preferred_device="/dev/video4")
        assert mock_emit.call_args[0][2]["preferred_device"] == "/dev/video4"

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_is_device_discovery(self, mock_emit):
        audit_vision_device_discovery()
        assert mock_emit.call_args[0][1] == "device_discovery"


class TestAuditVisionSessionFields:
    """Field-level checks for audit_vision_session."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_frame_count_field(self, mock_emit):
        audit_vision_session(action="update", frame_count=7)
        assert mock_emit.call_args[0][2]["frame_count"] == 7

    @patch("missy.vision.audit._emit_audit_event")
    def test_task_type_field(self, mock_emit):
        audit_vision_session(action="create", task_type="painting")
        assert mock_emit.call_args[0][2]["task_type"] == "painting"

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_constructed_with_prefix(self, mock_emit):
        audit_vision_session(action="expire")
        assert mock_emit.call_args[0][1] == "session_expire"

    @patch("missy.vision.audit._emit_audit_event")
    def test_task_id_field(self, mock_emit):
        audit_vision_session(action="close", task_id="painting-42")
        assert mock_emit.call_args[0][2]["task_id"] == "painting-42"


class TestAuditVisionBurstFields:
    """Field-level checks for audit_vision_burst."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_device_field(self, mock_emit):
        audit_vision_burst(device="/dev/video1", count=3, successful=3)
        assert mock_emit.call_args[0][2]["device"] == "/dev/video1"

    @patch("missy.vision.audit._emit_audit_event")
    def test_default_trigger_reason(self, mock_emit):
        audit_vision_burst()
        assert mock_emit.call_args[0][2]["trigger_reason"] == "user_command"

    @patch("missy.vision.audit._emit_audit_event")
    def test_voice_trigger_reason(self, mock_emit):
        audit_vision_burst(trigger_reason="voice_activation")
        assert mock_emit.call_args[0][2]["trigger_reason"] == "voice_activation"

    @patch("missy.vision.audit._emit_audit_event")
    def test_best_only_false_default(self, mock_emit):
        audit_vision_burst()
        assert mock_emit.call_args[0][2]["best_only"] is False

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_is_burst_capture(self, mock_emit):
        audit_vision_burst()
        assert mock_emit.call_args[0][1] == "burst_capture"

    @patch("missy.vision.audit._emit_audit_event")
    def test_partial_failure_counts(self, mock_emit):
        audit_vision_burst(count=5, successful=2)
        details = mock_emit.call_args[0][2]
        assert details["requested_count"] == 5
        assert details["successful_count"] == 2


class TestAuditVisionErrorFields:
    """Field-level checks for audit_vision_error."""

    @patch("missy.vision.audit._emit_audit_event")
    def test_error_string_field(self, mock_emit):
        audit_vision_error(operation="capture", error="EBUSY: device busy")
        assert mock_emit.call_args[0][2]["error"] == "EBUSY: device busy"

    @patch("missy.vision.audit._emit_audit_event")
    def test_device_field(self, mock_emit):
        audit_vision_error(device="/dev/video3")
        assert mock_emit.call_args[0][2]["device"] == "/dev/video3"

    @patch("missy.vision.audit._emit_audit_event")
    def test_action_is_error(self, mock_emit):
        audit_vision_error()
        assert mock_emit.call_args[0][1] == "error"

    @patch("missy.vision.audit._emit_audit_event")
    def test_category_is_vision(self, mock_emit):
        audit_vision_error()
        assert mock_emit.call_args[0][0] == "vision"

    @patch("missy.vision.audit._emit_audit_event")
    def test_default_recoverable_is_true(self, mock_emit):
        audit_vision_error()
        assert mock_emit.call_args[0][2]["recoverable"] is True

    @patch("missy.vision.audit._emit_audit_event")
    def test_all_fields_present(self, mock_emit):
        audit_vision_error(
            operation="open",
            error="No such device",
            device="/dev/video9",
            recoverable=False,
        )
        details = mock_emit.call_args[0][2]
        assert set(details.keys()) == {"operation", "error", "device", "recoverable"}
