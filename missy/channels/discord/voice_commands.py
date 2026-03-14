"""Message-command helpers for Discord voice MVP.

Commands:
- !join <channel_id>
- !leave
- !say <text>

These are parsed from MESSAGE_CREATE events (content string) and routed to
:class:`~missy.channels.discord.voice.DiscordVoiceManager`.

We keep this logic separate so `channel.py` changes stay surgical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from missy.channels.discord.voice import DiscordVoiceError, DiscordVoiceManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceCommandResult:
    handled: bool
    reply: Optional[str] = None


async def maybe_handle_voice_command(
    *,
    content: str,
    channel_id: str,
    guild_id: Optional[str],
    voice: Optional[DiscordVoiceManager],
) -> VoiceCommandResult:
    text = (content or "").strip()
    if not text.startswith("!"):
        return VoiceCommandResult(False)

    if text.lower().startswith("!join"):
        if not guild_id:
            return VoiceCommandResult(True, "Voice commands only work in servers.")
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip().isdigit():
            return VoiceCommandResult(True, "Usage: `!join <channel_id>`")
        if voice is None:
            return VoiceCommandResult(True, "Voice is not enabled on this bot.")
        try:
            await voice.join(int(guild_id), int(parts[1].strip()))
            logger.info("Discord voice: joined guild=%s channel=%s", guild_id, parts[1].strip())
            return VoiceCommandResult(True, "joined")
        except DiscordVoiceError as exc:
            logger.warning("Discord voice: join failed guild=%s channel=%s err=%s", guild_id, parts[1].strip(), exc)
            return VoiceCommandResult(True, str(exc))

    if text.lower().startswith("!leave"):
        if not guild_id:
            return VoiceCommandResult(True, "Voice commands only work in servers.")
        if voice is None:
            return VoiceCommandResult(True, "Voice is not enabled on this bot.")
        try:
            await voice.leave(int(guild_id))
            logger.info("Discord voice: left guild=%s", guild_id)
            return VoiceCommandResult(True, "left")
        except DiscordVoiceError as exc:
            logger.warning("Discord voice: leave failed guild=%s err=%s", guild_id, exc)
            return VoiceCommandResult(True, str(exc))

    if text.lower().startswith("!say"):
        if not guild_id:
            return VoiceCommandResult(True, "Voice commands only work in servers.")
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            return VoiceCommandResult(True, "Usage: `!say <text>`")
        if voice is None:
            return VoiceCommandResult(True, "Voice is not enabled on this bot.")
        try:
            await voice.say(int(guild_id), parts[1].strip())
            logger.info("Discord voice: spoke guild=%s chars=%d", guild_id, len(parts[1].strip()))
            return VoiceCommandResult(True, "speaking")
        except DiscordVoiceError as exc:
            logger.warning("Discord voice: say failed guild=%s err=%s", guild_id, exc)
            return VoiceCommandResult(True, str(exc))

    return VoiceCommandResult(False)
