"""Tests for Discord image command handlers and image analysis utilities.

Tests :func:`maybe_handle_image_command` and helpers in
:mod:`missy.channels.discord.image_analyze`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.channels.discord.image_analyze import (
    find_latest_image,
    is_image_attachment,
)
from missy.channels.discord.image_commands import (
    ImageCommandResult,
    maybe_handle_image_command,
)

# ---------------------------------------------------------------------------
# is_image_attachment
# ---------------------------------------------------------------------------


class TestIsImageAttachment:
    def test_png_content_type(self):
        assert is_image_attachment({"content_type": "image/png"}) is True

    def test_jpeg_content_type(self):
        assert is_image_attachment({"content_type": "image/jpeg"}) is True

    def test_gif_content_type(self):
        assert is_image_attachment({"content_type": "image/gif"}) is True

    def test_webp_content_type(self):
        assert is_image_attachment({"content_type": "image/webp"}) is True

    def test_non_image_content_type(self):
        assert is_image_attachment({"content_type": "application/pdf"}) is False

    def test_image_by_extension(self):
        assert is_image_attachment({"filename": "photo.jpg"}) is True

    def test_image_by_png_extension(self):
        assert is_image_attachment({"filename": "screenshot.png"}) is True

    def test_image_by_bmp_extension(self):
        assert is_image_attachment({"filename": "image.bmp"}) is True

    def test_non_image_extension(self):
        assert is_image_attachment({"filename": "doc.txt"}) is False

    def test_no_content_type_or_filename(self):
        assert is_image_attachment({}) is False

    def test_none_content_type(self):
        assert is_image_attachment({"content_type": None}) is False

    def test_case_insensitive_content_type(self):
        assert is_image_attachment({"content_type": "IMAGE/PNG"}) is True

    def test_case_insensitive_extension(self):
        assert is_image_attachment({"filename": "PHOTO.JPEG"}) is True


# ---------------------------------------------------------------------------
# find_latest_image
# ---------------------------------------------------------------------------


class TestFindLatestImage:
    def test_finds_first_image(self):
        messages = [
            {"attachments": [{"content_type": "image/png", "url": "http://a.png"}]},
            {"attachments": [{"content_type": "image/jpeg", "url": "http://b.jpg"}]},
        ]
        result = find_latest_image(messages)
        assert result["url"] == "http://a.png"

    def test_skips_non_images(self):
        messages = [
            {"attachments": [{"content_type": "application/pdf", "filename": "doc.pdf"}]},
            {"attachments": [{"content_type": "image/png", "url": "http://img.png"}]},
        ]
        result = find_latest_image(messages)
        assert result["url"] == "http://img.png"

    def test_no_images_returns_none(self):
        messages = [
            {"attachments": [{"content_type": "application/pdf"}]},
        ]
        assert find_latest_image(messages) is None

    def test_empty_messages(self):
        assert find_latest_image([]) is None

    def test_messages_without_attachments(self):
        messages = [{"content": "hello"}, {"attachments": None}]
        assert find_latest_image(messages) is None

    def test_empty_attachments_list(self):
        messages = [{"attachments": []}]
        assert find_latest_image(messages) is None


# ---------------------------------------------------------------------------
# maybe_handle_image_command - non-commands
# ---------------------------------------------------------------------------


class TestImageCommandNonCommands:
    @pytest.mark.asyncio
    async def test_non_command_not_handled(self):
        result = await maybe_handle_image_command(
            content="hello", channel_id="123", rest_client=MagicMock()
        )
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_empty_content(self):
        result = await maybe_handle_image_command(
            content="", channel_id="123", rest_client=MagicMock()
        )
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_none_content(self):
        result = await maybe_handle_image_command(
            content=None, channel_id="123", rest_client=MagicMock()
        )
        assert result.handled is False

    @pytest.mark.asyncio
    async def test_unknown_bang_command(self):
        result = await maybe_handle_image_command(
            content="!unknown", channel_id="123", rest_client=MagicMock()
        )
        assert result.handled is False


# ---------------------------------------------------------------------------
# !analyze command
# ---------------------------------------------------------------------------


class TestAnalyzeCommand:
    @pytest.mark.asyncio
    async def test_no_rest_client(self):
        result = await maybe_handle_image_command(
            content="!analyze", channel_id="123", rest_client=None
        )
        assert result.handled is True
        assert "not available" in result.reply.lower()

    @pytest.mark.asyncio
    async def test_analyze_no_image_found(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(return_value=[
            {"attachments": [{"content_type": "text/plain"}]}
        ])
        result = await maybe_handle_image_command(
            content="!analyze", channel_id="123", rest_client=rest
        )
        assert result.handled is True
        assert "No image found" in result.reply

    @pytest.mark.asyncio
    async def test_analyze_fetch_messages_error(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(side_effect=RuntimeError("HTTP 403"))
        result = await maybe_handle_image_command(
            content="!analyze", channel_id="123", rest_client=rest
        )
        assert result.handled is True
        assert "Failed to fetch" in result.reply

    @pytest.mark.asyncio
    async def test_analyze_success(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(return_value=[
            {"attachments": [{"content_type": "image/png", "url": "http://img.png", "filename": "test.png"}]}
        ])
        rest.download_attachment = MagicMock(return_value=b"\x89PNG\r\n")

        with patch("missy.channels.discord.image_commands._handle_analyze") as mock_analyze:
            mock_analyze.return_value = ImageCommandResult(True, "**Analysis of `test.png`:**\nA screenshot")
            result = await maybe_handle_image_command(
                content="!analyze", channel_id="123", rest_client=rest
            )
            assert result.handled is True

    @pytest.mark.asyncio
    async def test_analyze_with_question(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(return_value=[
            {"attachments": [{"content_type": "image/png", "url": "http://img.png", "filename": "test.png"}]}
        ])

        with patch("missy.channels.discord.image_commands._handle_analyze") as mock_analyze:
            mock_analyze.return_value = ImageCommandResult(True, "Result")
            result = await maybe_handle_image_command(
                content="!analyze What is in this?", channel_id="123", rest_client=rest
            )
            assert result.handled is True


# ---------------------------------------------------------------------------
# !screenshot command
# ---------------------------------------------------------------------------


class TestScreenshotCommand:
    @pytest.mark.asyncio
    async def test_screenshot_no_subcommand(self):
        rest = MagicMock()
        result = await maybe_handle_image_command(
            content="!screenshot", channel_id="123", rest_client=rest
        )
        assert result.handled is True
        assert "Usage" in result.reply

    @pytest.mark.asyncio
    async def test_screenshot_wrong_subcommand(self):
        rest = MagicMock()
        result = await maybe_handle_image_command(
            content="!screenshot delete", channel_id="123", rest_client=rest
        )
        assert result.handled is True
        assert "Usage" in result.reply

    @pytest.mark.asyncio
    async def test_screenshot_no_rest_client(self):
        result = await maybe_handle_image_command(
            content="!screenshot save", channel_id="123", rest_client=None
        )
        assert result.handled is True
        assert "not available" in result.reply.lower()

    @pytest.mark.asyncio
    async def test_screenshot_save_no_image(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(return_value=[])
        result = await maybe_handle_image_command(
            content="!screenshot save", channel_id="123", rest_client=rest
        )
        assert result.handled is True
        assert "No image found" in result.reply

    @pytest.mark.asyncio
    async def test_screenshot_save_fetch_error(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(side_effect=RuntimeError("fail"))
        result = await maybe_handle_image_command(
            content="!screenshot save", channel_id="123", rest_client=rest
        )
        assert result.handled is True
        assert "Failed to fetch" in result.reply

    @pytest.mark.asyncio
    async def test_screenshot_save_success(self):
        rest = MagicMock()
        rest.get_channel_messages = MagicMock(return_value=[
            {"attachments": [{"content_type": "image/png", "url": "http://img.png", "filename": "test.png"}]}
        ])

        with patch("missy.channels.discord.image_commands._handle_screenshot") as mock_ss:
            mock_ss.return_value = ImageCommandResult(True, "Saved to `/tmp/test.png`")
            result = await maybe_handle_image_command(
                content="!screenshot save", channel_id="123", rest_client=rest
            )
            assert result.handled is True

    @pytest.mark.asyncio
    async def test_screenshot_case_insensitive(self):
        rest = MagicMock()
        result = await maybe_handle_image_command(
            content="!SCREENSHOT save", channel_id="123", rest_client=rest
        )
        # The command lowering should handle this
        assert result.handled is True


# ---------------------------------------------------------------------------
# VoiceCommandResult / ImageCommandResult dataclass
# ---------------------------------------------------------------------------


class TestResultDataclass:
    def test_image_result_handled(self):
        r = ImageCommandResult(True, "reply")
        assert r.handled is True
        assert r.reply == "reply"

    def test_image_result_not_handled(self):
        r = ImageCommandResult(False)
        assert r.handled is False
        assert r.reply is None

    def test_image_result_frozen(self):
        r = ImageCommandResult(True, "reply")
        with pytest.raises(AttributeError):
            r.handled = False
