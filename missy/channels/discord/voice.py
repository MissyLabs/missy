"""Discord voice manager using discord.py.

Runs a lightweight discord.py Client dedicated to voice connections.
The text gateway remains Missy's own raw WebSocket implementation; this
module only handles voice join/leave/say.

API:
- start(bot_token)       — connect the voice-only client
- stop()                 — disconnect and shut down
- join(guild_id)         — join a voice channel (by user location, name, or ID)
- leave(guild_id)        — leave the current voice channel
- say(guild_id, text)    — speak text via TTS in the connected channel
- get_user_voice_channel — look up which voice channel a user is in

Requires: pip install -e ".[discord_voice]"   (discord.py[voice] + ffmpeg)
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from missy.channels.discord.ffmpeg import ensure_ffmpeg_available

logger = logging.getLogger(__name__)


class DiscordVoiceError(RuntimeError):
    """Actionable voice-layer error."""


@dataclass
class _GuildVoiceState:
    voice_client: Any  # discord.VoiceClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class DiscordVoiceManager:
    """Manages Discord voice connections per guild using discord.py.

    Runs its own discord.py Client for voice transport while the rest of
    Missy continues to use the custom raw WebSocket gateway for text.

    Args:
        tts_render: Callable that renders text to a WAV file.
            Signature: (text: str, out_path: str) -> None
    """

    def __init__(
        self,
        *,
        tts_render: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._tts_render = tts_render
        self._guild_states: Dict[int, _GuildVoiceState] = {}
        self._client: Any = None  # discord.Client
        self._discord: Any = None  # discord module
        self._ready = asyncio.Event()
        self._client_task: Optional[asyncio.Task] = None

        try:
            ensure_ffmpeg_available()
        except Exception as e:
            raise DiscordVoiceError(str(e)) from e

    async def start(self, bot_token: str) -> None:
        """Start the discord.py voice client in the background."""
        try:
            import discord  # type: ignore[import-untyped]
        except ImportError as e:
            raise DiscordVoiceError(
                "discord.py[voice] is not installed. "
                "Install with: pip install -e '.[discord_voice]'"
            ) from e

        self._discord = discord

        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.message_content = False  # not needed for voice-only

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            logger.info(
                "Discord voice client ready as %s (id=%s)",
                self._client.user, self._client.user.id,
            )
            self._ready.set()

        # Strip "Bot " prefix if present — discord.py adds it automatically.
        raw_token = bot_token
        if raw_token.startswith("Bot "):
            raw_token = raw_token[4:]

        self._client_task = asyncio.create_task(
            self._client.start(raw_token),
        )
        # Wait for the client to be ready (with timeout).
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise DiscordVoiceError(
                "Discord voice client did not become ready within 30 seconds."
            )

    async def stop(self) -> None:
        """Disconnect all voice connections and shut down the client."""
        for guild_id in list(self._guild_states):
            try:
                await self.leave(guild_id)
            except Exception:
                pass
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
        if self._client_task is not None:
            self._client_task.cancel()
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    # ------------------------------------------------------------------
    # Guild / channel lookup helpers
    # ------------------------------------------------------------------

    def _get_guild(self, guild_id: int) -> Any:
        """Get a guild from the discord.py client cache."""
        if self._client is None:
            raise DiscordVoiceError("Voice client is not started.")
        guild = self._client.get_guild(guild_id)
        if guild is None:
            raise DiscordVoiceError(
                "Guild not found. Make sure the bot is a member of the server."
            )
        return guild

    def get_user_voice_channel(
        self, guild_id: int, user_id: int
    ) -> Optional[Any]:
        """Return the voice channel a user is currently in, or None."""
        guild = self._get_guild(guild_id)
        member = guild.get_member(user_id)
        if member is None:
            return None
        voice_state = member.voice
        if voice_state is None or voice_state.channel is None:
            return None
        return voice_state.channel

    def find_voice_channel(
        self, guild_id: int, query: str
    ) -> Optional[Any]:
        """Find a voice channel by name (case-insensitive) or ID."""
        guild = self._get_guild(guild_id)
        discord = self._discord

        # Try as numeric ID first.
        if query.isdigit():
            ch = guild.get_channel(int(query))
            if ch is not None and isinstance(ch, discord.VoiceChannel):
                return ch

        # Case-insensitive name match.
        query_lower = query.lower()
        for ch in guild.voice_channels:
            if ch.name.lower() == query_lower:
                return ch

        # Partial match as fallback.
        for ch in guild.voice_channels:
            if query_lower in ch.name.lower():
                return ch

        return None

    def list_voice_channels(self, guild_id: int) -> list[str]:
        """Return names of all voice channels in the guild."""
        guild = self._get_guild(guild_id)
        return [ch.name for ch in guild.voice_channels]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def join(
        self,
        guild_id: int,
        *,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> str:
        """Join a voice channel. Returns the channel name joined.

        Resolution order:
        1. If *user_id* is given and no channel specified, join the user's
           current voice channel.
        2. If *channel_name* is given, find the channel by name.
        3. If *channel_id* is given, join by ID.
        4. If nothing is given but *user_id* is set, follow the user.

        Raises:
            DiscordVoiceError: If the channel cannot be resolved or joined.
        """
        guild = self._get_guild(guild_id)
        target: Any = None

        # Resolve target channel.
        if channel_id is not None:
            target = guild.get_channel(channel_id)
            if target is None:
                raise DiscordVoiceError(
                    f"Voice channel with ID {channel_id} not found."
                )
        elif channel_name is not None:
            target = self.find_voice_channel(guild_id, channel_name)
            if target is None:
                names = ", ".join(self.list_voice_channels(guild_id)) or "(none)"
                raise DiscordVoiceError(
                    f"No voice channel matching \"{channel_name}\". "
                    f"Available: {names}"
                )
        elif user_id is not None:
            target = self.get_user_voice_channel(guild_id, user_id)
            if target is None:
                raise DiscordVoiceError(
                    "You're not in a voice channel. Join one first, "
                    "or specify a channel name: `!join General`"
                )
        else:
            raise DiscordVoiceError(
                "Specify a voice channel name or join a voice channel first. "
                "Usage: `!join` or `!join <channel name>`"
            )

        # Check if already connected to this channel.
        state = self._guild_states.get(guild_id)
        if state and getattr(state.voice_client, "is_connected", lambda: False)():
            current = getattr(state.voice_client, "channel", None)
            if current and current.id == target.id:
                return target.name
            # Move to the new channel.
            await state.voice_client.move_to(target)
            return target.name

        # Connect.
        voice_client = await target.connect()
        self._guild_states[guild_id] = _GuildVoiceState(voice_client=voice_client)
        return target.name

    async def leave(self, guild_id: int) -> Optional[str]:
        """Leave the voice channel in the given guild.

        Returns the channel name that was left, or None if not connected.
        """
        state = self._guild_states.pop(guild_id, None)
        if state is None:
            return None

        channel_name = None
        async with state.lock:
            try:
                ch = getattr(state.voice_client, "channel", None)
                channel_name = getattr(ch, "name", None)
                if hasattr(state.voice_client, "disconnect"):
                    await state.voice_client.disconnect(force=True)
            except Exception as exc:
                logger.warning("Voice disconnect error: %s", exc)
        return channel_name

    async def say(self, guild_id: int, text: str) -> None:
        """Speak text via TTS in the connected voice channel.

        Raises:
            DiscordVoiceError: If not connected or TTS is unavailable.
        """
        text = (text or "").strip()
        if not text:
            raise DiscordVoiceError("No text provided.")

        if self._tts_render is None:
            raise DiscordVoiceError("TTS is not configured.")

        state = self._guild_states.get(guild_id)
        if not state or not getattr(state.voice_client, "is_connected", lambda: False)():
            raise DiscordVoiceError(
                "I'm not in a voice channel. Use `!join` to bring me in first."
            )

        async with state.lock:
            # Stop any currently playing audio.
            if hasattr(state.voice_client, "is_playing") and state.voice_client.is_playing():
                if hasattr(state.voice_client, "stop"):
                    state.voice_client.stop()

            path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="missy-tts-", suffix=".wav", delete=False
                ) as f:
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

    def is_connected(self, guild_id: int) -> bool:
        """Return True if the bot is in a voice channel in this guild."""
        state = self._guild_states.get(guild_id)
        if state is None:
            return False
        return bool(
            getattr(state.voice_client, "is_connected", lambda: False)()
        )

    def current_channel_name(self, guild_id: int) -> Optional[str]:
        """Return the name of the voice channel the bot is in, or None."""
        state = self._guild_states.get(guild_id)
        if state is None:
            return None
        ch = getattr(state.voice_client, "channel", None)
        return getattr(ch, "name", None) if ch else None
