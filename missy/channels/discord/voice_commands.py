"""Voice command handlers for Discord.

Commands:
- !join              — join the user's current voice channel
- !join <name>       — join a voice channel by name
- !join <id>         — join a voice channel by snowflake ID
- !leave             — leave the current voice channel
- !say <text>        — speak text via TTS

Parsed from MESSAGE_CREATE events and routed to
:class:`~missy.channels.discord.voice.DiscordVoiceManager`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from missy.channels.discord.voice import DiscordVoiceError, DiscordVoiceManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceCommandResult:
    handled: bool
    reply: str | None = None


async def maybe_handle_voice_command(
    *,
    content: str,
    channel_id: str,
    guild_id: str | None,
    author_id: str,
    voice: DiscordVoiceManager | None,
) -> VoiceCommandResult:
    """Parse and execute a voice command if applicable.

    Returns a result indicating whether the message was handled and an
    optional reply to send back.
    """
    text = (content or "").strip()
    if not text.startswith("!"):
        return VoiceCommandResult(False)

    cmd_part, _, rest = text.partition(" ")
    cmd = cmd_part.lower()

    if cmd not in ("!join", "!leave", "!say"):
        return VoiceCommandResult(False)

    if not guild_id:
        return VoiceCommandResult(True, "Voice commands only work in servers.")

    if voice is None:
        return VoiceCommandResult(True, "Voice is not enabled on this bot.")

    if not voice.is_ready:
        return VoiceCommandResult(True, "Voice is still starting up, try again in a moment.")

    gid = int(guild_id)
    rest = rest.strip()

    # ------- !join -------
    if cmd == "!join":
        try:
            if not rest:
                # No argument: join the user's current voice channel.
                channel_name = await voice.join(
                    gid,
                    user_id=int(author_id),
                )
            elif rest.isdigit():
                # Numeric argument: join by channel ID.
                channel_name = await voice.join(
                    gid,
                    channel_id=int(rest),
                )
            else:
                # Text argument: join by channel name.
                channel_name = await voice.join(
                    gid,
                    channel_name=rest,
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

    # ------- !leave -------
    if cmd == "!leave":
        try:
            channel_name = await voice.leave(gid)
            if channel_name:
                logger.info("Discord voice: left %r guild=%s", channel_name, guild_id)
                return VoiceCommandResult(True, f"Left **{channel_name}**")
            return VoiceCommandResult(True, "I'm not in a voice channel.")
        except DiscordVoiceError as exc:
            return VoiceCommandResult(True, str(exc))

    # ------- !say -------
    if cmd == "!say":
        if not rest:
            return VoiceCommandResult(True, "Usage: `!say <text>`")
        try:
            await voice.say(gid, rest)
            logger.info(
                "Discord voice: spoke %d chars guild=%s by=%s",
                len(rest),
                guild_id,
                author_id,
            )
            return VoiceCommandResult(True)
        except DiscordVoiceError as exc:
            return VoiceCommandResult(True, str(exc))

    return VoiceCommandResult(False)
