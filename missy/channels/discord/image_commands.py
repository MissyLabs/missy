"""Image analysis command handlers for Discord.

Commands:
- !analyze [question]       — analyze the last image posted in this channel
- !screenshot save          — save the last image to disk for documentation
- !screenshot save [dir]    — save to a specific directory

Natural-language equivalents (no `!` prefix required) are also recognised,
for example "analyze the image", "what's in this screenshot?", "save the
screenshot", or "save the image to /tmp/shots". See :func:`infer_image_intent`.

Parsed from MESSAGE_CREATE events. Requires a
:class:`~missy.channels.discord.rest.DiscordRestClient` for fetching
channel messages and downloading attachments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageCommandResult:
    handled: bool
    reply: str | None = None


@dataclass(frozen=True)
class ImageIntent:
    action: str  # "analyze" | "screenshot_save"
    question: str | None = None
    save_dir: str | None = None


_BOT_MENTION_RE = re.compile(r"^(<@!?\d+>\s*)+")
_SPACE_RE = re.compile(r"\s+")


def _clean_text(content: str | None) -> str:
    text = _SPACE_RE.sub(" ", (content or "").strip())
    return _BOT_MENTION_RE.sub("", text).strip()


def _strip_leading_politeness(text: str) -> str:
    previous = text
    while True:
        current = re.sub(
            r"^(please|can you|could you|would you)\s+",
            "",
            previous,
            flags=re.I,
        ).strip()
        if current == previous:
            return current
        previous = current


_IMAGE_NOUN = r"(?:image|picture|photo|screenshot)s?"
_ARTICLE = r"(?:the|this|that|my)\s+"
_RECENCY = r"(?:last|latest|recent)\s+"
_ANALYZE_VERBS = r"(?:analyze|analyse|describe|examine|inspect|look at|check out|check)"


def infer_image_intent(content: str | None) -> ImageIntent | None:
    """Parse supported natural-language Discord image requests.

    Recognises flexible phrasings for analyzing or saving the most recent
    image posted in the channel, without requiring a `!` bang prefix.
    Returns ``None`` for anything that is not clearly an image-related
    request, to avoid hijacking ordinary conversation.
    """
    text = _strip_leading_politeness(_clean_text(content))
    if not text or text.startswith("!"):
        return None

    # ------- "save the screenshot to <path>" -------
    save_with_target = re.fullmatch(
        rf"save\s+(?:{_ARTICLE})?(?:{_RECENCY})?{_IMAGE_NOUN}\s+to\s+(?P<target>.+?)\.?",
        text,
        flags=re.I,
    )
    if save_with_target:
        target = save_with_target.group("target").strip()
        if target.lower() in {"disk", "file", "the disk"}:
            return ImageIntent(action="screenshot_save")
        return ImageIntent(action="screenshot_save", save_dir=target)

    # ------- "save the screenshot" / "save that image" -------
    if re.fullmatch(
        rf"save\s+(?:{_ARTICLE})?(?:{_RECENCY})?{_IMAGE_NOUN}\.?",
        text,
        flags=re.I,
    ):
        return ImageIntent(action="screenshot_save")

    # ------- "analyze/describe/look at the image [and tell me X]" -------
    analyze_match = re.fullmatch(
        rf"{_ANALYZE_VERBS}\s+(?:{_ARTICLE})?(?:{_RECENCY})?{_IMAGE_NOUN}"
        rf"(?:\s+and\s+(?:tell me|describe|explain)\s+(?P<question>.+)"
        rf"|\s*[:,]\s*(?P<question2>.+))?\??\.?",
        text,
        flags=re.I,
    )
    if analyze_match:
        question = analyze_match.group("question") or analyze_match.group("question2")
        return ImageIntent(action="analyze", question=question.strip() if question else None)

    # ------- "what's in this picture?" -------
    what_match = re.fullmatch(
        rf"what(?:'s|\s+is)\s+(?:in|on)\s+(?:{_ARTICLE}|a\s+){_IMAGE_NOUN}\??",
        text,
        flags=re.I,
    )
    if what_match:
        return ImageIntent(action="analyze")

    return None


async def maybe_handle_image_command(
    *,
    content: str,
    channel_id: str,
    rest_client: Any,
) -> ImageCommandResult:
    """Parse and execute an image command if applicable.

    Tries the legacy `!analyze` / `!screenshot` bang-command syntax first,
    then falls back to natural-language intent inference (see
    :func:`infer_image_intent`) for messages without a `!` prefix.

    Args:
        content: The message text (with bot mention already stripped).
        channel_id: The Discord channel snowflake ID.
        rest_client: A :class:`DiscordRestClient` instance.

    Returns:
        An :class:`ImageCommandResult` indicating whether the message
        was handled and an optional reply.
    """
    text = (content or "").strip()

    if text.startswith("!"):
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
        return await _handle_screenshot(channel_id, rest_client, rest_arg)

    intent = infer_image_intent(content)
    if intent is None:
        return ImageCommandResult(False)

    if rest_client is None:
        return ImageCommandResult(True, "REST client is not available.")

    if intent.action == "analyze":
        return await _handle_analyze(channel_id, rest_client, intent.question or "")

    # intent.action == "screenshot_save"
    save_args = f"save {intent.save_dir}" if intent.save_dir else "save"
    return await _handle_screenshot(channel_id, rest_client, save_args)


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
