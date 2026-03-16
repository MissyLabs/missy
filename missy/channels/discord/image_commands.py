"""Image analysis command handlers for Discord.

Commands:
- !analyze [question]       — analyze the last image posted in this channel
- !screenshot save          — save the last image to disk for documentation
- !screenshot save [dir]    — save to a specific directory

Parsed from MESSAGE_CREATE events. Requires a
:class:`~missy.channels.discord.rest.DiscordRestClient` for fetching
channel messages and downloading attachments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageCommandResult:
    handled: bool
    reply: str | None = None


async def maybe_handle_image_command(
    *,
    content: str,
    channel_id: str,
    rest_client: Any,
) -> ImageCommandResult:
    """Parse and execute an image command if applicable.

    Args:
        content: The message text (with bot mention already stripped).
        channel_id: The Discord channel snowflake ID.
        rest_client: A :class:`DiscordRestClient` instance.

    Returns:
        An :class:`ImageCommandResult` indicating whether the message
        was handled and an optional reply.
    """
    text = (content or "").strip()
    if not text.startswith("!"):
        return ImageCommandResult(False)

    cmd_part, _, rest = text.partition(" ")
    cmd = cmd_part.lower()

    if cmd not in ("!analyze", "!screenshot"):
        return ImageCommandResult(False)

    if rest_client is None:
        return ImageCommandResult(True, "REST client is not available.")

    rest_arg = rest.strip()

    # ------- !analyze [question] -------
    if cmd == "!analyze":
        return await _handle_analyze(channel_id, rest_client, rest_arg)

    # ------- !screenshot save [dir] -------
    if cmd == "!screenshot":
        return await _handle_screenshot(channel_id, rest_client, rest_arg)

    return ImageCommandResult(False)


async def _handle_analyze(
    channel_id: str,
    rest_client: Any,
    question: str,
) -> ImageCommandResult:
    """Fetch the last image from the channel and run vision analysis."""
    import asyncio

    from missy.channels.discord.image_analyze import (
        analyze_discord_attachment,
        find_latest_image,
    )

    if not question:
        question = "Describe what you see in this image in detail. If it's a screenshot, describe the UI elements, any error messages, and the overall state of the application."

    try:
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(
            None, rest_client.get_channel_messages, channel_id, 25
        )
    except Exception as exc:
        logger.error("Failed to fetch channel messages: %s", exc)
        return ImageCommandResult(True, f"Failed to fetch messages: {exc}")

    attachment = find_latest_image(messages)
    if attachment is None:
        return ImageCommandResult(
            True,
            "No image found in the last 25 messages. Upload a screenshot and try again.",
        )

    filename = attachment.get("filename", "image")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: analyze_discord_attachment(rest_client, attachment, question),
        )
    except Exception as exc:
        logger.error("Image analysis failed: %s", exc)
        return ImageCommandResult(True, f"Analysis failed: {exc}")

    analysis = result.get("analysis", "")
    if not analysis:
        return ImageCommandResult(True, "Vision model returned empty analysis.")

    # Truncate for Discord's 2000 char limit, leaving room for header.
    header = f"**Analysis of `{filename}`:**\n"
    max_body = 2000 - len(header) - 10
    if len(analysis) > max_body:
        analysis = analysis[:max_body].rstrip() + "..."

    return ImageCommandResult(True, header + analysis)


async def _handle_screenshot(
    channel_id: str,
    rest_client: Any,
    args: str,
) -> ImageCommandResult:
    """Save the last image from the channel to disk."""
    import asyncio

    from missy.channels.discord.image_analyze import (
        find_latest_image,
        save_discord_attachment,
    )

    parts = args.split(None, 1)
    subcmd = parts[0].lower() if parts else ""

    if subcmd != "save":
        return ImageCommandResult(
            True,
            "Usage: `!screenshot save` or `!screenshot save /path/to/dir`",
        )

    save_dir = parts[1].strip() if len(parts) > 1 else "~/workspace/screenshots"

    try:
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(
            None, rest_client.get_channel_messages, channel_id, 25
        )
    except Exception as exc:
        logger.error("Failed to fetch channel messages: %s", exc)
        return ImageCommandResult(True, f"Failed to fetch messages: {exc}")

    attachment = find_latest_image(messages)
    if attachment is None:
        return ImageCommandResult(
            True,
            "No image found in the last 25 messages. Upload a screenshot first.",
        )

    try:
        loop = asyncio.get_running_loop()
        saved_path = await loop.run_in_executor(
            None,
            lambda: save_discord_attachment(rest_client, attachment, save_dir),
        )
    except Exception as exc:
        logger.error("Failed to save screenshot: %s", exc)
        return ImageCommandResult(True, f"Save failed: {exc}")

    return ImageCommandResult(True, f"Saved to `{saved_path}`")
