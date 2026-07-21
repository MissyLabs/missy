"""Tests for missy.channels.discord.zip_attachment — inbound .zip
attachment metadata validation, mirroring test_discord_text_attachment.py.
"""

from __future__ import annotations


class TestIsZipAttachment:
    def test_zip_content_type(self):
        from missy.channels.discord.zip_attachment import is_zip_attachment

        assert is_zip_attachment({"content_type": "application/zip", "filename": "project.zip"})

    def test_alternate_zip_content_type(self):
        from missy.channels.discord.zip_attachment import is_zip_attachment

        assert is_zip_attachment(
            {"content_type": "application/x-zip-compressed", "filename": "project.zip"}
        )

    def test_extension_fallback_when_no_content_type(self):
        from missy.channels.discord.zip_attachment import is_zip_attachment

        assert is_zip_attachment({"filename": "archive.zip"})

    def test_image_is_not_zip(self):
        from missy.channels.discord.zip_attachment import is_zip_attachment

        assert not is_zip_attachment({"content_type": "image/png", "filename": "shot.png"})

    def test_text_is_not_zip(self):
        from missy.channels.discord.zip_attachment import is_zip_attachment

        assert not is_zip_attachment({"content_type": "text/plain", "filename": "notes.txt"})


class TestValidateZipAttachment:
    def _valid(self, **overrides):
        base = {
            "filename": "project.zip",
            "content_type": "application/zip",
            "url": "https://cdn.discordapp.com/attachments/1/2/project.zip",
            "size": 4096,
        }
        base.update(overrides)
        return base

    def test_valid_zip_attachment_allowed(self):
        from missy.channels.discord.zip_attachment import validate_zip_attachment

        v = validate_zip_attachment(self._valid())
        assert v.allowed is True
        assert v.reasons == []

    def test_non_discord_cdn_url_denied(self):
        from missy.channels.discord.zip_attachment import validate_zip_attachment

        v = validate_zip_attachment(self._valid(url="https://evil.example.com/project.zip"))
        assert v.allowed is False
        assert "invalid_discord_cdn_url" in v.reasons

    def test_missing_url_denied(self):
        from missy.channels.discord.zip_attachment import validate_zip_attachment

        v = validate_zip_attachment(self._valid(url=""))
        assert v.allowed is False
        assert "missing_url" in v.reasons

    def test_oversized_attachment_denied(self):
        from missy.channels.discord.zip_attachment import (
            MAX_ZIP_ATTACHMENT_BYTES,
            validate_zip_attachment,
        )

        v = validate_zip_attachment(self._valid(size=MAX_ZIP_ATTACHMENT_BYTES + 1))
        assert v.allowed is False
        assert "zip_attachment_too_large" in v.reasons

    def test_unsupported_content_type_and_extension_denied(self):
        from missy.channels.discord.zip_attachment import validate_zip_attachment

        v = validate_zip_attachment(
            self._valid(filename="a.bin", content_type="application/octet-stream")
        )
        assert v.allowed is False
        assert "unsupported_content_type" in v.reasons

    def test_extension_only_recognised_without_content_type(self):
        from missy.channels.discord.zip_attachment import validate_zip_attachment

        v = validate_zip_attachment(self._valid(filename="archive.zip", content_type=""))
        assert v.allowed is True

    def test_negative_size_denied(self):
        from missy.channels.discord.zip_attachment import validate_zip_attachment

        v = validate_zip_attachment(self._valid(size=-5))
        assert v.allowed is False
        assert "invalid_size" in v.reasons
