"""Tests for Discord image analysis features.

Covers:
- image_analyze.py: is_image_attachment, find_latest_image, analyze_image_bytes
- image_commands.py: !analyze, !screenshot command parsing
- rest.py: get_channel_messages, download_attachment
- channel.py: attachment policy allowing images through
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# image_analyze helpers
# ---------------------------------------------------------------------------


class TestIsImageAttachment:
    def test_png_content_type(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert is_image_attachment({"content_type": "image/png", "filename": "shot.png"})

    def test_jpeg_content_type(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert is_image_attachment({"content_type": "image/jpeg", "filename": "photo.jpg"})

    def test_gif_content_type(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert is_image_attachment({"content_type": "image/gif", "filename": "anim.gif"})

    def test_webp_content_type(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert is_image_attachment({"content_type": "image/webp", "filename": "img.webp"})

    def test_text_file_rejected(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert not is_image_attachment({"content_type": "text/plain", "filename": "notes.txt"})

    def test_pdf_rejected(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert not is_image_attachment({"content_type": "application/pdf", "filename": "doc.pdf"})

    def test_fallback_to_extension_when_no_content_type(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert is_image_attachment({"filename": "screenshot.png"})
        assert is_image_attachment({"filename": "photo.JPEG"})  # case insensitive

    def test_unknown_extension_rejected(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert not is_image_attachment({"filename": "data.csv"})

    def test_empty_attachment(self):
        from missy.channels.discord.image_analyze import is_image_attachment

        assert not is_image_attachment({})


class TestValidateImageAttachment:
    def test_accepts_valid_discord_cdn_image_metadata(self):
        from missy.channels.discord.image_analyze import validate_image_attachment

        result = validate_image_attachment(
            {
                "content_type": "image/png",
                "filename": "shot.png",
                "url": "https://cdn.discordapp.com/attachments/1/2/shot.png",
                "size": 1024,
                "width": 800,
                "height": 600,
            }
        )

        assert result.allowed is True
        assert result.reasons == []
        assert result.details["filename"] == "shot.png"
        assert result.details["url_host"] == "cdn.discordapp.com"

    def test_rejects_oversize_image_metadata(self):
        from missy.channels.discord.image_analyze import (
            MAX_IMAGE_ATTACHMENT_BYTES,
            validate_image_attachment,
        )

        result = validate_image_attachment(
            {
                "content_type": "image/png",
                "filename": "shot.png",
                "url": "https://cdn.discordapp.com/attachments/1/2/shot.png",
                "size": MAX_IMAGE_ATTACHMENT_BYTES + 1,
            }
        )

        assert result.allowed is False
        assert "image_too_large" in result.reasons

    def test_rejects_mime_extension_mismatch(self):
        from missy.channels.discord.image_analyze import validate_image_attachment

        result = validate_image_attachment(
            {
                "content_type": "image/png",
                "filename": "photo.jpg",
                "url": "https://cdn.discordapp.com/attachments/1/2/photo.jpg",
            }
        )

        assert result.allowed is False
        assert "mime_extension_mismatch" in result.reasons

    def test_rejects_bad_dimensions(self):
        from missy.channels.discord.image_analyze import validate_image_attachment

        result = validate_image_attachment(
            {
                "content_type": "image/jpeg",
                "filename": "photo.jpg",
                "url": "https://cdn.discordapp.com/attachments/1/2/photo.jpg",
                "width": 10000,
                "height": 500,
            }
        )

        assert result.allowed is False
        assert "invalid_width" in result.reasons

    def test_rejects_non_discord_cdn_url(self):
        from missy.channels.discord.image_analyze import validate_image_attachment

        result = validate_image_attachment(
            {
                "content_type": "image/png",
                "filename": "shot.png",
                "url": "https://example.com/shot.png",
            }
        )

        assert result.allowed is False
        assert "invalid_discord_cdn_url" in result.reasons


class TestFindLatestImage:
    def test_finds_image_in_first_message(self):
        from missy.channels.discord.image_analyze import find_latest_image

        messages = [
            {
                "attachments": [
                    {"content_type": "image/png", "filename": "a.png", "url": "http://x"}
                ]
            },
            {"attachments": []},
        ]
        att = find_latest_image(messages)
        assert att is not None
        assert att["filename"] == "a.png"

    def test_finds_image_in_later_message(self):
        from missy.channels.discord.image_analyze import find_latest_image

        messages = [
            {"attachments": []},
            {"attachments": [{"content_type": "text/plain", "filename": "x.txt"}]},
            {
                "attachments": [
                    {"content_type": "image/jpeg", "filename": "b.jpg", "url": "http://y"}
                ]
            },
        ]
        att = find_latest_image(messages)
        assert att is not None
        assert att["filename"] == "b.jpg"

    def test_returns_none_when_no_images(self):
        from missy.channels.discord.image_analyze import find_latest_image

        messages = [
            {"attachments": [{"content_type": "text/plain", "filename": "x.txt"}]},
            {"attachments": []},
        ]
        assert find_latest_image(messages) is None

    def test_returns_none_for_empty_list(self):
        from missy.channels.discord.image_analyze import find_latest_image

        assert find_latest_image([]) is None


class TestAnalyzeImageBytes:
    @patch("missy.gateway.client.PolicyHTTPClient")
    def test_calls_ollama_vision(self, mock_client_cls):
        from missy.channels.discord.image_analyze import analyze_image_bytes

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "A screenshot of a terminal"}}
        mock_client_cls.return_value.post.return_value = mock_resp

        result = analyze_image_bytes(b"\x89PNG\r\n", "What is this?")
        assert result == "A screenshot of a terminal"
        mock_client_cls.return_value.post.assert_called_once()

    @patch("missy.gateway.client.PolicyHTTPClient")
    def test_raises_on_http_error(self, mock_client_cls):
        from missy.channels.discord.image_analyze import analyze_image_bytes

        mock_client_cls.return_value.post.side_effect = Exception("Connection refused")

        with pytest.raises(RuntimeError, match="Vision model request failed"):
            analyze_image_bytes(b"\x89PNG\r\n", "What is this?")


class TestAnalyzeDiscordAttachment:
    @patch("missy.channels.discord.image_analyze.analyze_image_bytes")
    def test_downloads_and_analyzes(self, mock_analyze):
        from missy.channels.discord.image_analyze import analyze_discord_attachment

        mock_analyze.return_value = "A desktop screenshot"
        mock_rest = MagicMock()
        mock_rest.download_attachment.return_value = b"\x89PNG"

        result = analyze_discord_attachment(
            mock_rest,
            {
                "url": "https://cdn.discordapp.com/a.png",
                "filename": "a.png",
                "content_type": "image/png",
            },
            "Describe it",
        )
        assert result["analysis"] == "A desktop screenshot"
        assert result["filename"] == "a.png"
        mock_rest.download_attachment.assert_called_once()

    @patch("missy.channels.discord.image_analyze.analyze_image_bytes")
    def test_saves_to_disk(self, mock_analyze, tmp_path):
        from missy.channels.discord.image_analyze import analyze_discord_attachment

        mock_analyze.return_value = "saved"
        mock_rest = MagicMock()
        mock_rest.download_attachment.return_value = b"\x89PNG"

        save_path = str(tmp_path / "saved.png")
        result = analyze_discord_attachment(
            mock_rest,
            {
                "url": "https://cdn.discordapp.com/a.png",
                "filename": "a.png",
                "content_type": "image/png",
            },
            "Describe it",
            save_path=save_path,
        )
        assert result["saved_to"] == save_path
        assert (tmp_path / "saved.png").exists()

    def test_raises_on_no_url(self):
        from missy.channels.discord.image_analyze import analyze_discord_attachment

        with pytest.raises(ValueError, match="no download URL"):
            analyze_discord_attachment(MagicMock(), {}, "q")


class TestSaveDiscordAttachment:
    def test_saves_with_timestamp(self, tmp_path):
        from missy.channels.discord.image_analyze import save_discord_attachment

        mock_rest = MagicMock()
        mock_rest.download_attachment.return_value = b"\x89PNG"

        path = save_discord_attachment(
            mock_rest,
            {
                "url": "https://cdn.discordapp.com/a.png",
                "filename": "shot.png",
                "content_type": "image/png",
            },
            save_dir=str(tmp_path),
        )
        assert path.endswith("_shot.png")
        assert (tmp_path / path.split("/")[-1]).exists()


# ---------------------------------------------------------------------------
# REST client methods
# ---------------------------------------------------------------------------


class TestGetChannelMessages:
    def test_fetches_messages(self):
        from missy.channels.discord.rest import DiscordRestClient

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"id": "1"}, {"id": "2"}]
        mock_http.get.return_value = mock_resp

        client = DiscordRestClient(bot_token="Bot test", http_client=mock_http)
        result = client.get_channel_messages("123456789012345678", limit=5)
        assert len(result) == 2
        mock_http.get.assert_called_once()
        # Verify limit param was passed.
        call_kwargs = mock_http.get.call_args
        assert call_kwargs.kwargs.get("params", {}).get("limit") == 5

    def test_clamps_limit(self):
        from missy.channels.discord.rest import DiscordRestClient

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_http.get.return_value = mock_resp

        client = DiscordRestClient(bot_token="Bot test", http_client=mock_http)
        client.get_channel_messages("123456789012345678", limit=999)
        call_kwargs = mock_http.get.call_args
        assert call_kwargs.kwargs.get("params", {}).get("limit") == 100

    def test_invalid_channel_id_raises(self):
        from missy.channels.discord.rest import DiscordRestClient

        client = DiscordRestClient(bot_token="Bot test", http_client=MagicMock())
        with pytest.raises(ValueError, match="Invalid Discord"):
            client.get_channel_messages("not-a-snowflake")


class TestDownloadAttachment:
    def test_downloads_cdn_url(self):
        from missy.channels.discord.rest import DiscordRestClient

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG"
        mock_http.get.return_value = mock_resp

        client = DiscordRestClient(bot_token="Bot test", http_client=mock_http)
        data = client.download_attachment("https://cdn.discordapp.com/attachments/1/2/img.png")
        assert data == b"\x89PNG"

    def test_downloads_media_url(self):
        from missy.channels.discord.rest import DiscordRestClient

        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = b"\xff\xd8"
        mock_http.get.return_value = mock_resp

        client = DiscordRestClient(bot_token="Bot test", http_client=mock_http)
        data = client.download_attachment("https://media.discordapp.net/attachments/1/2/img.jpg")
        assert data == b"\xff\xd8"

    def test_rejects_non_cdn_url(self):
        from missy.channels.discord.rest import DiscordRestClient

        client = DiscordRestClient(bot_token="Bot test", http_client=MagicMock())
        with pytest.raises(ValueError, match="Not a Discord CDN URL"):
            client.download_attachment("https://evil.com/payload.exe")


# ---------------------------------------------------------------------------
# Image commands
# ---------------------------------------------------------------------------


class TestImageCommands:
    @pytest.mark.asyncio
    async def test_analyze_not_handled_for_non_command(self):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        result = await maybe_handle_image_command(
            content="hello world",
            channel_id="123456789012345678",
            rest_client=MagicMock(),
        )
        assert not result.handled

    @pytest.mark.asyncio
    async def test_analyze_not_handled_for_other_command(self):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        result = await maybe_handle_image_command(
            content="join my voice channel",
            channel_id="123456789012345678",
            rest_client=MagicMock(),
        )
        assert not result.handled

    @pytest.mark.asyncio
    async def test_analyze_no_rest_client(self):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        result = await maybe_handle_image_command(
            content="!analyze",
            channel_id="123456789012345678",
            rest_client=None,
        )
        assert result.handled
        assert "not available" in result.reply

    @pytest.mark.asyncio
    @patch("missy.channels.discord.image_analyze.find_latest_image")
    @patch("missy.channels.discord.image_analyze.analyze_discord_attachment")
    async def test_analyze_finds_and_analyzes(self, mock_analyze, mock_find):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        mock_rest = MagicMock()
        mock_rest.get_channel_messages.return_value = [{"attachments": []}]
        mock_find.return_value = {"url": "http://x", "filename": "shot.png"}
        mock_analyze.return_value = {"analysis": "A terminal window", "filename": "shot.png"}

        result = await maybe_handle_image_command(
            content="!analyze What errors are shown?",
            channel_id="123456789012345678",
            rest_client=mock_rest,
        )
        assert result.handled
        assert "A terminal window" in result.reply
        assert "shot.png" in result.reply

    @pytest.mark.asyncio
    @patch("missy.channels.discord.image_analyze.find_latest_image")
    async def test_analyze_no_image_found(self, mock_find):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        mock_rest = MagicMock()
        mock_rest.get_channel_messages.return_value = []
        mock_find.return_value = None

        result = await maybe_handle_image_command(
            content="!analyze",
            channel_id="123456789012345678",
            rest_client=mock_rest,
        )
        assert result.handled
        assert "No image found" in result.reply

    @pytest.mark.asyncio
    async def test_screenshot_usage(self):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        result = await maybe_handle_image_command(
            content="!screenshot",
            channel_id="123456789012345678",
            rest_client=MagicMock(),
        )
        assert result.handled
        assert "Usage" in result.reply

    @pytest.mark.asyncio
    @patch("missy.channels.discord.image_analyze.find_latest_image")
    @patch("missy.channels.discord.image_analyze.save_discord_attachment")
    async def test_screenshot_save(self, mock_save, mock_find):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        mock_rest = MagicMock()
        mock_rest.get_channel_messages.return_value = []
        mock_find.return_value = {"url": "http://x", "filename": "shot.png"}
        mock_save.return_value = "/home/user/workspace/screenshots/20260316_shot.png"

        result = await maybe_handle_image_command(
            content="!screenshot save",
            channel_id="123456789012345678",
            rest_client=mock_rest,
        )
        assert result.handled
        assert "Saved to" in result.reply

    @pytest.mark.asyncio
    @patch("missy.channels.discord.image_analyze.find_latest_image")
    @patch("missy.channels.discord.image_analyze.save_discord_attachment")
    async def test_screenshot_save_custom_dir(self, mock_save, mock_find):
        from missy.channels.discord.image_commands import maybe_handle_image_command

        mock_rest = MagicMock()
        mock_rest.get_channel_messages.return_value = []
        mock_find.return_value = {"url": "http://x", "filename": "shot.png"}
        mock_save.return_value = "/tmp/docs/20260316_shot.png"

        result = await maybe_handle_image_command(
            content="!screenshot save /tmp/docs",
            channel_id="123456789012345678",
            rest_client=mock_rest,
        )
        assert result.handled
        # Verify custom dir was passed through.
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][2] == "/tmp/docs" or call_args.kwargs.get("save_dir") == "/tmp/docs"


# ---------------------------------------------------------------------------
# Attachment policy in channel.py
# ---------------------------------------------------------------------------


class TestAttachmentPolicy:
    def _make_channel(self):
        """Build a minimal DiscordChannel for testing."""
        from missy.channels.discord.channel import DiscordChannel
        from missy.channels.discord.config import (
            DiscordAccountConfig,
            DiscordDMPolicy,
            DiscordGuildPolicy,
        )

        cfg = DiscordAccountConfig(
            token_env_var="DISCORD_BOT_TOKEN",
            dm_policy=DiscordDMPolicy.OPEN,
            guild_policies={
                "333": DiscordGuildPolicy(enabled=True, require_mention=False),
            },
        )
        ch = DiscordChannel(account_config=cfg)
        ch._bot_user_id = "999"
        ch._rest = MagicMock()
        ch._gateway = MagicMock()
        ch._gateway.bot_user_id = "999"
        return ch

    @pytest.mark.asyncio
    async def test_image_attachment_allowed(self):
        ch = self._make_channel()
        data = {
            "author": {"id": "111", "bot": False},
            "channel_id": "222",
            "guild_id": "333",
            "content": "<@999> what is this?",
            "id": "444",
            "attachments": [
                {
                    "content_type": "image/png",
                    "filename": "shot.png",
                    "url": "https://cdn.discordapp.com/attachments/1/2/shot.png",
                    "size": 1024,
                    "width": 800,
                    "height": 600,
                }
            ],
        }
        await ch._handle_message(data)
        # Image allowed through — message should be enqueued.
        assert not ch._queue.empty()
        msg = await ch._queue.get()
        assert len(msg.metadata["discord_image_attachments"]) == 1
        assert msg.metadata["discord_image_attachments"][0]["filename"] == "shot.png"
        assert msg.metadata["discord_image_attachments"][0]["width"] == 800

    @pytest.mark.asyncio
    async def test_non_image_attachment_denied(self):
        ch = self._make_channel()
        data = {
            "author": {"id": "111", "bot": False},
            "channel_id": "222",
            "guild_id": "333",
            "content": "<@999> here is a file",
            "id": "444",
            "attachments": [
                {"content_type": "application/zip", "filename": "archive.zip", "url": "http://x"}
            ],
        }
        await ch._handle_message(data)
        # Non-image should be denied — queue stays empty.
        assert ch._queue.empty()

    @pytest.mark.asyncio
    async def test_mixed_attachments_denied(self):
        ch = self._make_channel()
        data = {
            "author": {"id": "111", "bot": False},
            "channel_id": "222",
            "guild_id": "333",
            "content": "<@999> files",
            "id": "444",
            "attachments": [
                {
                    "content_type": "image/png",
                    "filename": "shot.png",
                    "url": "https://cdn.discordapp.com/attachments/1/2/shot.png",
                },
                {"content_type": "application/pdf", "filename": "doc.pdf", "url": "http://y"},
            ],
        }
        await ch._handle_message(data)
        # Mixed = denied (non-image present).
        assert ch._queue.empty()

    @pytest.mark.asyncio
    async def test_invalid_image_metadata_denied_with_audit_details(self):
        ch = self._make_channel()
        ch._emit_audit = MagicMock()
        data = {
            "author": {"id": "111", "bot": False},
            "channel_id": "222",
            "guild_id": "333",
            "content": "<@999> look",
            "id": "444",
            "attachments": [
                {
                    "content_type": "image/png",
                    "filename": "huge.png",
                    "url": "https://cdn.discordapp.com/attachments/1/2/huge.png",
                    "size": 9 * 1024 * 1024,
                }
            ],
        }

        await ch._handle_message(data)

        assert ch._queue.empty()
        ch._emit_audit.assert_any_call(
            "discord.channel.attachment_denied",
            "deny",
            {
                "author_id": "111",
                "channel_id": "222",
                "attachment_count": 1,
                "attachments": [
                    {
                        "filename": "huge.png",
                        "content_type": "image/png",
                        "size": 9 * 1024 * 1024,
                        "width": None,
                        "height": None,
                        "url_host": "cdn.discordapp.com",
                        "max_size": 8 * 1024 * 1024,
                        "max_dimension": 8192,
                        "max_pixels": 40_000_000,
                        "reasons": ["image_too_large"],
                    }
                ],
                "reason": "attachment_metadata_not_permitted",
            },
        )

    @pytest.mark.asyncio
    async def test_no_attachment_passes_through(self):
        ch = self._make_channel()
        data = {
            "author": {"id": "111", "bot": False},
            "channel_id": "222",
            "guild_id": "333",
            "content": "<@999> hello",
            "id": "444",
            "attachments": [],
        }
        await ch._handle_message(data)
        assert not ch._queue.empty()
        msg = await ch._queue.get()
        assert msg.metadata["discord_image_attachments"] == []
