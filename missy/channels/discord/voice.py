"""Discord voice MVP manager.

This module is intentionally minimal and *only* responsible for voice transport.
Text/event handling remains on Missy's existing websocket gateway.

API:
- join(guild_id, channel_id)
- leave(guild_id)
- say(guild_id, text)

Notes:
- Requires optional dependency: py-cord[voice]
- Requires system ffmpeg installed (see missy.channels.discord.ffmpeg)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from missy.channels.discord.ffmpeg import ensure_ffmpeg_available

logger = logging.getLogger(__name__)


class DiscordVoiceError(RuntimeError):
    """Actionable voice-layer error."""


@dataclass
class _GuildVoiceState:
    voice_client: object  # discord.VoiceClient
    lock: asyncio.Lock


class DiscordVoiceManager:
    """Manages Discord voice connections per guild.

    Uses Pycord's voice implementation (discord module) but does not own the full bot.

    You must provide:
      - a discord.Client (or subclass) instance
      - a TTS renderer that can render text to a WAV file path

    The TTS renderer signature is: (text: str, out_path: str) -> None
    """

    def __init__(
        self,
        *,
        discord_client: object,
        tts_render: Callable[[str, str], None],
    ) -> None:
        self._client = discord_client
        self._tts_render = tts_render
        self._guild_states: Dict[int, _GuildVoiceState] = {}

        try:
            ensure_ffmpeg_available()
        except Exception as e:  # pragma: no cover
            raise DiscordVoiceError(str(e)) from e

        # Lazy import so base install remains REST/text-only.
        try:
            import discord  # type: ignore

            self._discord = discord
        except Exception as e:  # pragma: no cover
            raise DiscordVoiceError(
                "py-cord[voice] is not installed. Install with: pip install .[discord_voice]"
            ) from e

    async def join(self, guild_id: int, channel_id: int) -> None:
        guild = getattr(self._client, "get_guild", lambda _gid: None)(guild_id)
        if guild is None:
            raise DiscordVoiceError(
                "Guild not found in voice client cache. Ensure the voice client is logged in."
            )

        channel = getattr(guild, "get_channel", lambda _cid: None)(channel_id)
        if channel is None:
            raise DiscordVoiceError("Voice channel not found (check the channel_id).")

        if not hasattr(channel, "connect"):
            raise DiscordVoiceError("Target channel is not a connectable voice channel.")

        state = self._guild_states.get(guild_id)
        if state and getattr(state.voice_client, "is_connected", lambda: False)():
            current_channel = getattr(state.voice_client, "channel", None)
            if current_channel and getattr(current_channel, "id", None) == channel_id:
                return
            if hasattr(state.voice_client, "move_to"):
                await state.voice_client.move_to(channel)
                return

        voice_client = await channel.connect()
        self._guild_states[guild_id] = _GuildVoiceState(
            voice_client=voice_client, lock=asyncio.Lock()
        )

    async def leave(self, guild_id: int) -> None:
        state = self._guild_states.get(guild_id)
        if not state:
            return

        async with state.lock:
            try:
                if hasattr(state.voice_client, "disconnect"):
                    await state.voice_client.disconnect(force=True)
            finally:
                self._guild_states.pop(guild_id, None)

    async def say(self, guild_id: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            raise DiscordVoiceError("No text provided.")

        state = self._guild_states.get(guild_id)
        if not state or not getattr(state.voice_client, "is_connected", lambda: False)():
            raise DiscordVoiceError("Not connected to voice. Use `!join <channel_id>` first.")

        async with state.lock:
            if hasattr(state.voice_client, "is_playing") and state.voice_client.is_playing():
                if hasattr(state.voice_client, "stop"):
                    state.voice_client.stop()

            path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(prefix="missy-tts-", suffix=".wav", delete=False) as f:
                    path = f.name

                await asyncio.to_thread(self._tts_render, text, path)

                source = self._discord.FFmpegPCMAudio(path)
                done = asyncio.Event()

                def _after(err: Optional[BaseException]) -> None:
                    if err:
                        logger.exception("Voice playback error", exc_info=err)
                    done.set()

                state.voice_client.play(source, after=_after)

                try:
                    await asyncio.wait_for(done.wait(), timeout=60)
                except asyncio.TimeoutError:
                    if hasattr(state.voice_client, "stop"):
                        state.voice_client.stop()
                    raise DiscordVoiceError("TTS playback timed out.")
            finally:
                if path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass


_voice_manager: Optional[DiscordVoiceManager] = None


def init_voice_manager(*, discord_client: object, tts_render: Callable[[str, str], None]) -> DiscordVoiceManager:
    global _voice_manager
    _voice_manager = DiscordVoiceManager(discord_client=discord_client, tts_render=tts_render)
    return _voice_manager


def get_voice_manager() -> DiscordVoiceManager:
    if _voice_manager is None:
        raise DiscordVoiceError("Voice manager not initialized.")
    return _voice_manager
