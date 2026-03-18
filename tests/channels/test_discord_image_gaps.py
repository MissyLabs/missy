"""Gap coverage tests for Discord image modules.

Targets uncovered lines in:
- missy/channels/discord/image_commands.py: 68, 110-112, 116, 122, 173-175
- missy/channels/discord/image_analyze.py: 63-65, 78-83, 203, 219

Patch-target notes
------------------
Both _get_ollama_base_url and _get_vision_model import load_config *inside*
the function body:  ``from missy.config.settings import load_config``.
Because load_config is never bound at module level, the correct patch target
is ``missy.config.settings.load_config``.

Similarly, PolicyHTTPClient is imported inside analyze_image_bytes:
``from missy.gateway.client import PolicyHTTPClient``.
Patch target: ``missy.gateway.client.PolicyHTTPClient``.

_handle_analyze and _handle_screenshot import from image_analyze inside the
function body.  Patching the names on ``missy.channels.discord.image_analyze``
intercepts those lookups correctly.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# image_commands.py gap coverage
# ---------------------------------------------------------------------------


class TestImageCommandsFallthrough:
    """Line 68 — ImageCommandResult(False) fallthrough after both cmd branches.

    The line at 68 is structurally dead code: both !analyze and !screenshot
    are handled by the preceding `if cmd ==` guards, so execution never
    reaches it.  The tests below confirm the boundary conditions that
    exercise the surrounding paths and demonstrate that the two live branches
    are always taken before line 68.
    """

    @pytest.mark.asyncio
    async def test_unknown_bang_command_returns_false_before_line_68(self):
        """!unknown hits line 52-53 and returns False — does not reach line 68."""
        from missy.channels.discord.image_commands import maybe_handle_image_command

        result = await maybe_handle_image_command(
            content="!resize",
            channel_id="123456789012345678",
            rest_client=MagicMock(),
        )
        assert result.handled is False
        assert result.reply is None

    @pytest.mark.asyncio
    async def test_analyze_branch_taken_not_fallthrough(self):
        """!analyze is routed to _handle_analyze — line 68 is never hit."""
        from missy.channels.discord.image_commands import maybe_handle_image_command

        rest = MagicMock()
        rest.get_channel_messages.side_effect = RuntimeError("forced")

        result = await maybe_handle_image_command(
            content="!analyze",
            channel_id="123456789012345678",
            rest_client=rest,
        )
        # Handled is True because _handle_analyze ran (even though it errored).
        assert result.handled is True

    @pytest.mark.asyncio
    async def test_screenshot_branch_taken_not_fallthrough(self):
        """!screenshot is routed to _handle_screenshot — line 68 is never hit."""
        from missy.channels.discord.image_commands import maybe_handle_image_command

        rest = MagicMock()

        result = await maybe_handle_image_command(
            content="!screenshot",
            channel_id="123456789012345678",
            rest_client=rest,
        )
        # Returns usage string — _handle_screenshot ran.
        assert result.handled is True
        assert "Usage" in result.reply


class TestHandleAnalyzeExceptionPath:
    """Lines 110-112 — analyze_discord_attachment raises an exception."""

    @pytest.mark.asyncio
    async def test_analyze_discord_attachment_exception_returns_error_reply(self):
        """When analyze_discord_attachment raises, handled=True with error message."""
        from missy.channels.discord.image_commands import _handle_analyze

        rest = MagicMock()
        rest.get_channel_messages.return_value = [
            {
                "attachments": [
                    {
                        "content_type": "image/png",
                        "url": "https://cdn.discordapp.com/a.png",
                        "filename": "error.png",
                    }
                ]
            }
        ]

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            side_effect=RuntimeError("Ollama unreachable"),
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/a.png",
                "filename": "error.png",
            },
        ):
            result = await _handle_analyze("123456789012345678", rest, "")

        assert result.handled is True
        assert "Analysis failed" in result.reply
        assert "Ollama unreachable" in result.reply

    @pytest.mark.asyncio
    async def test_analyze_exception_with_question_passes_through(self):
        """Exception path is identical regardless of whether a question was provided."""
        from missy.channels.discord.image_commands import _handle_analyze

        rest = MagicMock()
        rest.get_channel_messages.return_value = [
            {
                "attachments": [
                    {
                        "content_type": "image/jpeg",
                        "url": "https://cdn.discordapp.com/b.jpg",
                        "filename": "b.jpg",
                    }
                ]
            }
        ]

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            side_effect=ConnectionError("timeout"),
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/b.jpg",
                "filename": "b.jpg",
            },
        ):
            result = await _handle_analyze(
                "123456789012345678", rest, "Is there an error?"
            )

        assert result.handled is True
        assert "Analysis failed" in result.reply


class TestHandleAnalyzeEmptyAnalysis:
    """Line 116 — vision model returns empty analysis string."""

    @pytest.mark.asyncio
    async def test_empty_analysis_returns_specific_message(self):
        """When analyze_discord_attachment returns analysis='', reply says so."""
        from missy.channels.discord.image_commands import _handle_analyze

        rest = MagicMock()
        rest.get_channel_messages.return_value = [
            {
                "attachments": [
                    {
                        "content_type": "image/png",
                        "url": "https://cdn.discordapp.com/x.png",
                        "filename": "x.png",
                    }
                ]
            }
        ]

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            return_value={"analysis": "", "filename": "x.png"},
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/x.png",
                "filename": "x.png",
            },
        ):
            result = await _handle_analyze("123456789012345678", rest, "")

        assert result.handled is True
        assert result.reply == "Vision model returned empty analysis."

    @pytest.mark.asyncio
    async def test_whitespace_only_analysis_treated_as_non_empty(self):
        """Whitespace-only analysis is truthy — it is NOT the empty-analysis path."""
        from missy.channels.discord.image_commands import _handle_analyze

        rest = MagicMock()
        rest.get_channel_messages.return_value = []

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            return_value={"analysis": "   ", "filename": "ws.png"},
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/ws.png",
                "filename": "ws.png",
            },
        ):
            result = await _handle_analyze("123456789012345678", rest, "")

        # "   " is truthy so we get a formatted reply, not the empty message.
        assert result.handled is True
        assert result.reply != "Vision model returned empty analysis."


class TestHandleAnalysisTruncation:
    """Line 122 — long analysis is truncated to fit Discord's 2000-char limit."""

    @pytest.mark.asyncio
    async def test_long_analysis_is_truncated_with_ellipsis(self):
        """Analysis longer than ~1975 chars is truncated and ends with '...'."""
        from missy.channels.discord.image_commands import _handle_analyze

        long_text = "A" * 3000
        rest = MagicMock()
        rest.get_channel_messages.return_value = []

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            return_value={"analysis": long_text, "filename": "big.png"},
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/big.png",
                "filename": "big.png",
            },
        ):
            result = await _handle_analyze("123456789012345678", rest, "")

        assert result.handled is True
        assert result.reply.endswith("...")
        assert len(result.reply) <= 2000

    @pytest.mark.asyncio
    async def test_short_analysis_is_not_truncated(self):
        """Analysis short enough to fit is returned verbatim (no truncation)."""
        from missy.channels.discord.image_commands import _handle_analyze

        short_text = "A small screenshot of a terminal."
        rest = MagicMock()
        rest.get_channel_messages.return_value = []

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            return_value={"analysis": short_text, "filename": "small.png"},
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/small.png",
                "filename": "small.png",
            },
        ):
            result = await _handle_analyze("123456789012345678", rest, "")

        assert result.handled is True
        assert short_text in result.reply
        assert not result.reply.endswith("...")

    @pytest.mark.asyncio
    async def test_truncation_boundary_exactly_at_limit(self):
        """Analysis exactly at max_body length is not truncated."""
        from missy.channels.discord.image_commands import _handle_analyze

        filename = "boundary.png"
        header = f"**Analysis of `{filename}`:**\n"
        max_body = 2000 - len(header) - 10
        exact_text = "B" * max_body

        rest = MagicMock()
        rest.get_channel_messages.return_value = []

        with patch(
            "missy.channels.discord.image_analyze.analyze_discord_attachment",
            return_value={"analysis": exact_text, "filename": filename},
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/boundary.png",
                "filename": filename,
            },
        ):
            result = await _handle_analyze("123456789012345678", rest, "")

        assert result.handled is True
        assert not result.reply.endswith("...")


class TestHandleScreenshotErrorPath:
    """Lines 173-175 — save_discord_attachment raises inside _handle_screenshot."""

    @pytest.mark.asyncio
    async def test_save_error_returns_handled_with_error_reply(self):
        """When save_discord_attachment raises, handled=True with 'Save failed' reply."""
        from missy.channels.discord.image_commands import _handle_screenshot

        rest = MagicMock()
        rest.get_channel_messages.return_value = [
            {
                "attachments": [
                    {
                        "content_type": "image/png",
                        "url": "https://cdn.discordapp.com/shot.png",
                        "filename": "shot.png",
                    }
                ]
            }
        ]

        with patch(
            "missy.channels.discord.image_analyze.save_discord_attachment",
            side_effect=OSError("Permission denied"),
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/shot.png",
                "filename": "shot.png",
            },
        ):
            result = await _handle_screenshot(
                "123456789012345678", rest, "save"
            )

        assert result.handled is True
        assert "Save failed" in result.reply
        assert "Permission denied" in result.reply

    @pytest.mark.asyncio
    async def test_save_error_with_custom_dir_in_reply(self):
        """Save error is reported even when a custom directory is specified."""
        from missy.channels.discord.image_commands import _handle_screenshot

        rest = MagicMock()
        rest.get_channel_messages.return_value = [
            {
                "attachments": [
                    {
                        "content_type": "image/jpeg",
                        "url": "https://cdn.discordapp.com/img.jpg",
                        "filename": "img.jpg",
                    }
                ]
            }
        ]

        with patch(
            "missy.channels.discord.image_analyze.save_discord_attachment",
            side_effect=ValueError("path traversal"),
        ), patch(
            "missy.channels.discord.image_analyze.find_latest_image",
            return_value={
                "url": "https://cdn.discordapp.com/img.jpg",
                "filename": "img.jpg",
            },
        ):
            result = await _handle_screenshot(
                "123456789012345678", rest, "save /tmp/docs"
            )

        assert result.handled is True
        assert "Save failed" in result.reply


# ---------------------------------------------------------------------------
# image_analyze.py gap coverage
# ---------------------------------------------------------------------------


class TestGetOllamaBaseUrlConfigPath:
    """Lines 63-65 — _get_ollama_base_url returns URL from loaded config.

    load_config is imported inside the function body so the correct patch
    target is missy.config.settings.load_config.
    """

    def test_returns_base_url_from_config_when_available(self):
        """When config provides ollama.base_url, that value is returned (stripped)."""
        from missy.channels.discord.image_analyze import _get_ollama_base_url

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.base_url = "http://my-ollama-host:11434/"

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            url = _get_ollama_base_url()

        assert url == "http://my-ollama-host:11434"

    def test_returns_default_when_provider_cfg_missing(self):
        """When config has no ollama provider entry, default URL is returned."""
        from missy.channels.discord.image_analyze import _get_ollama_base_url

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = None

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            url = _get_ollama_base_url()

        assert url == "http://localhost:11434"

    def test_returns_default_when_base_url_is_none(self):
        """When provider cfg exists but base_url is None/falsy, default is returned."""
        from missy.channels.discord.image_analyze import _get_ollama_base_url

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.base_url = None

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            url = _get_ollama_base_url()

        assert url == "http://localhost:11434"

    def test_returns_default_when_load_config_raises(self):
        """If load_config throws, the exception is swallowed and default returned."""
        from missy.channels.discord.image_analyze import _get_ollama_base_url

        with patch(
            "missy.config.settings.load_config",
            side_effect=ImportError("no config module"),
        ):
            url = _get_ollama_base_url()

        assert url == "http://localhost:11434"

    def test_trailing_slash_is_stripped(self):
        """Trailing slash is stripped from base_url via rstrip('/')."""
        from missy.channels.discord.image_analyze import _get_ollama_base_url

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.base_url = "http://ollama:11434///"

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            url = _get_ollama_base_url()

        assert url == "http://ollama:11434"


class TestGetVisionModelConfigPath:
    """Lines 78-83 — _get_vision_model returns vision_model from config extra.

    load_config is imported inside the function body so the correct patch
    target is missy.config.settings.load_config.
    """

    def test_returns_vision_model_from_extra_dict(self):
        """When extra dict has vision_model key, that value is returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.extra = {"vision_model": "llava:13b"}

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            model = _get_vision_model()

        assert model == "llava:13b"

    def test_returns_default_when_extra_lacks_vision_model(self):
        """When extra dict exists but has no vision_model key, default returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.extra = {"other_key": "value"}

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            model = _get_vision_model()

        assert model == "minicpm-v"

    def test_returns_default_when_extra_is_none(self):
        """When provider_cfg.extra is None (getattr returns None), default returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        mock_provider_cfg = MagicMock(spec=[])  # no 'extra' attribute at all
        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            model = _get_vision_model()

        assert model == "minicpm-v"

    def test_returns_default_when_extra_is_non_dict(self):
        """When extra is not a dict (isinstance check fails), default returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.extra = "not-a-dict"

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            model = _get_vision_model()

        assert model == "minicpm-v"

    def test_returns_default_when_provider_not_in_config(self):
        """When config has no ollama entry, default model is returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = None

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            model = _get_vision_model()

        assert model == "minicpm-v"

    def test_returns_default_when_load_config_raises(self):
        """If load_config throws, the exception is swallowed and default returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        with patch(
            "missy.config.settings.load_config",
            side_effect=RuntimeError("db locked"),
        ):
            model = _get_vision_model()

        assert model == "minicpm-v"

    def test_returns_default_when_vision_model_is_empty_string(self):
        """Empty string vision_model is falsy — default is returned."""
        from missy.channels.discord.image_analyze import _get_vision_model

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.extra = {"vision_model": ""}

        mock_cfg = MagicMock()
        mock_cfg.providers.get.return_value = mock_provider_cfg

        with patch(
            "missy.config.settings.load_config",
            return_value=mock_cfg,
        ):
            model = _get_vision_model()

        assert model == "minicpm-v"


class TestAnalyzeImageBytesRaiseForStatusPath:
    """Line 203 — raise_for_status raises after a successful post call.

    The existing test covers ``client.post(...)`` raising directly.  These
    tests cover the distinct branch where ``post()`` succeeds but
    ``raise_for_status()`` raises an HTTP error (e.g. 422 from Ollama).

    PolicyHTTPClient is imported inside analyze_image_bytes so the correct
    patch target is missy.gateway.client.PolicyHTTPClient.
    """

    def test_raise_for_status_wraps_in_runtime_error(self):
        """raise_for_status() raising is caught and re-raised as RuntimeError."""
        from missy.channels.discord.image_analyze import analyze_image_bytes

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("422 Unprocessable Entity")

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch(
            "missy.gateway.client.PolicyHTTPClient",
            return_value=mock_client,
        ), pytest.raises(RuntimeError, match="Vision model request failed"):
            analyze_image_bytes(b"\x89PNG\r\n", "What is this?")

    def test_raise_for_status_message_preserved_in_error(self):
        """The original HTTP error message is included in the RuntimeError."""
        from missy.channels.discord.image_analyze import analyze_image_bytes

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch(
            "missy.gateway.client.PolicyHTTPClient",
            return_value=mock_client,
        ), pytest.raises(RuntimeError, match="503 Service Unavailable"):
            analyze_image_bytes(b"\xff\xd8\xff", "Describe this")

    def test_successful_post_and_raise_for_status_returns_content(self):
        """When neither post nor raise_for_status raise, content is returned."""
        from missy.channels.discord.image_analyze import analyze_image_bytes

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None  # does not raise
        mock_resp.json.return_value = {"message": {"content": "a cat"}}

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch(
            "missy.gateway.client.PolicyHTTPClient",
            return_value=mock_client,
        ):
            result = analyze_image_bytes(b"\x89PNG\r\n", "What animal?")

        assert result == "a cat"


class TestSaveDiscordAttachmentPathTraversal:
    """Line 219 — realpath resolves outside save_dir, raises ValueError."""

    def test_path_traversal_filename_raises_value_error(self, tmp_path):
        """A filename that resolves outside save_dir is rejected with ValueError.

        os.path.realpath is patched so that the destination path resolves to a
        location outside the save directory, simulating a symlink escape without
        needing real filesystem symlinks.
        """
        from missy.channels.discord.image_analyze import save_discord_attachment

        save_dir = str(tmp_path / "screenshots")
        outside_dir = str(tmp_path / "outside")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(outside_dir, exist_ok=True)

        mock_rest = MagicMock()
        mock_rest.download_attachment.return_value = b"\x89PNG"

        os.path.realpath(save_dir)
        outside_resolved = os.path.realpath(outside_dir) + "/payload.png"

        original_realpath = os.path.realpath

        def fake_realpath(p):
            # For the constructed destination file path return a path outside
            # save_dir; for the directory itself return the true path so that
            # the startswith guard fires correctly.
            if os.path.basename(p).endswith(".png") and p != save_dir:
                return outside_resolved
            return original_realpath(p)

        with patch("os.path.realpath", side_effect=fake_realpath):
            with pytest.raises(ValueError, match="resolves outside save directory"):
                save_discord_attachment(
                    mock_rest,
                    {"url": "https://cdn.discordapp.com/a.png", "filename": "escape.png"},
                    save_dir=save_dir,
                )

    def test_safe_filename_does_not_raise(self, tmp_path):
        """A normal filename that resolves inside save_dir completes successfully."""
        from missy.channels.discord.image_analyze import save_discord_attachment

        save_dir = str(tmp_path)
        mock_rest = MagicMock()
        mock_rest.download_attachment.return_value = b"\x89PNG"

        result = save_discord_attachment(
            mock_rest,
            {"url": "https://cdn.discordapp.com/a.png", "filename": "normal.png"},
            save_dir=save_dir,
        )

        assert result.startswith(save_dir)
        assert result.endswith("_normal.png")
        assert os.path.isfile(result)

    def test_no_url_raises_value_error(self):
        """Attachment without url or proxy_url raises ValueError (line 203)."""
        from missy.channels.discord.image_analyze import save_discord_attachment

        with pytest.raises(ValueError, match="no download URL"):
            save_discord_attachment(
                MagicMock(),
                {},
                save_dir="/tmp",
            )
