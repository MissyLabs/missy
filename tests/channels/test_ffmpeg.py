"""Tests for missy/channels/discord/ffmpeg.py — 100% coverage."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from missy.channels.discord.ffmpeg import ensure_ffmpeg_available


class TestEnsureFfmpegAvailable:
    def test_returns_path_when_found(self) -> None:
        with patch("missy.channels.discord.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"):
            result = ensure_ffmpeg_available()
        assert result == "/usr/bin/ffmpeg"

    def test_raises_runtime_error_when_not_found(self) -> None:
        with (
            patch("missy.channels.discord.ffmpeg.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="ffmpeg not found"),
        ):
            ensure_ffmpeg_available()

    def test_returned_path_is_string(self) -> None:
        with patch(
            "missy.channels.discord.ffmpeg.shutil.which", return_value="/usr/local/bin/ffmpeg"
        ):
            result = ensure_ffmpeg_available()
        assert isinstance(result, str)

    def test_error_message_contains_install_hint(self) -> None:
        with (
            patch("missy.channels.discord.ffmpeg.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="apt install ffmpeg"),
        ):
            ensure_ffmpeg_available()
