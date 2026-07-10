"""Natural-language voice action handlers for Discord.

Recognised examples:
- join my voice channel
- join the General voice channel
- talk to me in the General voice channel
- leave the voice channel
- say hello world in voice

Parsed from MESSAGE_CREATE events and routed to
:class:`~missy.channels.discord.voice.DiscordVoiceManager`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from missy.channels.discord.voice import DiscordVoiceError, DiscordVoiceManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceCommandResult:
    handled: bool
    reply: str | None = None


@dataclass(frozen=True)
class VoiceIntent:
    action: str
    channel_name: str | None = None
    channel_id: int | None = None
    speech: str | None = None


_BOT_MENTION_RE = re.compile(r"^(<@!?\d+>\s*)+")
_SPACE_RE = re.compile(r"\s+")


_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")


def _clean_text(content: str | None) -> str:
    text = _SPACE_RE.sub(" ", (content or "").strip())
    text = _BOT_MENTION_RE.sub("", text).strip()
    # All intent patterns below use re.fullmatch, so a trailing full stop from
    # an ordinary sentence (e.g. "join the General voice channel.") would
    # otherwise silently fail to match anything.
    return _TRAILING_PUNCT_RE.sub("", text).strip()


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


_TRAILING_CLAUSE_RE = re.compile(
    r"\s*,\s*.*$|\s+(?:and|then|so|please)\b.*$",
    re.I,
)


def _normalise_channel_target(raw_target: str) -> tuple[str | None, int | None] | None:
    target = raw_target.strip(" .,")
    # Drop a trailing conversational clause (e.g. "the General voice channel
    # and report your voice status" or "the General voice channel, then say
    # hi" -> "the General voice channel") so a verbose, natural request
    # doesn't get treated as a literal channel name. The comma branch has no
    # leading \s+ requirement because commas are usually attached directly
    # to the preceding word (e.g. "channel, then ...").
    target = _TRAILING_CLAUSE_RE.sub("", target).strip(" .,")
    target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.I)
    target = re.sub(r"^(?:my|current|my current|this)\s+", "", target, flags=re.I)
    target = re.sub(r"\s+(?:voice\s+channel|voice\s+room|voice)$", "", target, flags=re.I)
    target = re.sub(r"^(?:voice\s+channel|voice\s+room|voice)\s+", "", target, flags=re.I)
    target = target.strip(" .")

    if target.startswith("#"):
        target = target[1:].strip()

    if target.lower() in {
        "",
        "me",
        "mine",
        "my",
        "current",
        "channel",
        "room",
        "voice",
        "voice channel",
        "voice room",
    }:
        return (None, None)

    if target.isdigit():
        return (None, int(target))

    return (target, None)


def parse_voice_intent(content: str | None) -> VoiceIntent | None:
    """Parse supported natural-language Discord voice requests."""
    text = _strip_leading_politeness(_clean_text(content))
    if not text or text.startswith("!"):
        return None

    lower = text.lower()
    if re.fullmatch(
        r"(?:leave|exit)\s+(?:the\s+)?(?:current\s+)?voice(?:\s+channel|\s+room)?",
        lower,
    ) or re.fullmatch(
        r"(?:disconnect|drop out)\s+(?:from\s+)?(?:the\s+)?voice(?:\s+channel|\s+room)?",
        lower,
    ):
        return VoiceIntent(action="leave")

    say_match = re.fullmatch(
        r"(?:say|speak|read)\s+(?P<speech>.+?)\s+(?:in|to|over|through)\s+"
        r"(?:the\s+)?voice(?:\s+channel|\s+room)?",
        text,
        flags=re.I,
    )
    if say_match:
        return VoiceIntent(action="say", speech=say_match.group("speech").strip())

    tell_match = re.fullmatch(
        r"tell\s+(?:the\s+)?voice(?:\s+channel|\s+room)?\s+(?P<speech>.+)",
        text,
        flags=re.I,
    )
    if tell_match:
        return VoiceIntent(action="say", speech=tell_match.group("speech").strip())

    no_speech_match = re.fullmatch(
        r"(?:say|speak|read)\s+(?:in|to|over|through)\s+(?:the\s+)?voice"
        r"(?:\s+channel|\s+room)?",
        text,
        flags=re.I,
    )
    if no_speech_match:
        return VoiceIntent(action="say")

    join_patterns = (
        r"(?:join me|talk to me|talk with me|listen to me|listen)\s+"
        r"(?:in|on|from)\s+(?P<target>.+)",
        r"(?:connect|come)\s+(?:to|into)\s+(?P<target>.+)",
        r"(?:join|enter)\s+(?P<target>.+)",
    )
    for pattern in join_patterns:
        match = re.fullmatch(pattern, text, flags=re.I)
        if not match:
            continue
        target_text = match.group("target")
        if "voice" not in target_text.lower() and not target_text.strip().startswith("#"):
            continue
        target = _normalise_channel_target(target_text)
        if target is None:
            continue
        channel_name, channel_id = target
        return VoiceIntent(action="join", channel_name=channel_name, channel_id=channel_id)

    return None


async def maybe_handle_voice_command(
    *,
    content: str,
    channel_id: str,
    guild_id: str | None,
    author_id: str,
    voice: DiscordVoiceManager | None,
) -> VoiceCommandResult:
    """Parse and execute a voice request if applicable.

    Returns a result indicating whether the message was handled and an
    optional reply to send back.
    """
    intent = parse_voice_intent(content)
    if intent is None:
        return VoiceCommandResult(False)

    if not guild_id:
        return VoiceCommandResult(True, "Voice requests only work in servers.")

    if voice is None:
        return VoiceCommandResult(True, "Voice is not enabled on this bot.")

    if not voice.is_ready:
        return VoiceCommandResult(True, "Voice is still starting up, try again in a moment.")

    gid = int(guild_id)

    if intent.action == "join":
        try:
            if intent.channel_id is not None:
                channel_name = await voice.join(
                    gid,
                    channel_id=intent.channel_id,
                )
            elif intent.channel_name:
                channel_name = await voice.join(
                    gid,
                    channel_name=intent.channel_name,
                )
            else:
                # No argument: join the user's current voice channel.
                channel_name = await voice.join(
                    gid,
                    user_id=int(author_id),
                )
            logger.info(
                "Discord voice: joined %r guild=%s by=%s",
                channel_name,
                guild_id,
                author_id,
            )
            # Build status message.
            capabilities = []
            if voice.can_listen:
                capabilities.append("listening")
            if voice.can_speak:
                capabilities.append("speaking")
            status = f" ({', '.join(capabilities)})" if capabilities else ""
            return VoiceCommandResult(True, f"Joined **{channel_name}**{status}")
        except DiscordVoiceError as exc:
            return VoiceCommandResult(True, str(exc))

    if intent.action == "leave":
        try:
            channel_name = await voice.leave(gid)
            if channel_name:
                logger.info("Discord voice: left %r guild=%s", channel_name, guild_id)
                return VoiceCommandResult(True, f"Left **{channel_name}**")
            return VoiceCommandResult(True, "I'm not in a voice channel.")
        except DiscordVoiceError as exc:
            return VoiceCommandResult(True, str(exc))

    if intent.action == "say":
        if not intent.speech:
            return VoiceCommandResult(
                True,
                "Tell me what to say in voice, for example: `say hello in voice`",
            )
        try:
            await voice.say(gid, intent.speech)
            logger.info(
                "Discord voice: spoke %d chars guild=%s by=%s",
                len(intent.speech),
                guild_id,
                author_id,
            )
            return VoiceCommandResult(True)
        except DiscordVoiceError as exc:
            return VoiceCommandResult(True, str(exc))

    return VoiceCommandResult(False)
