"""Discord voice manager — full conversational voice via discord.py.

Runs a lightweight discord.py Client dedicated to voice connections.
The text gateway remains Missy's own raw WebSocket implementation; this
module handles voice join/leave/say and continuous listen→STT→agent→TTS.

Requires: pip install -e ".[discord_voice,voice]"

Dependencies:
- discord.py[voice]       — voice connection and audio playback
- discord-ext-voice-recv  — audio receiving (discord.py 2.x lacks it)
- faster-whisper           — speech-to-text
- piper (binary)           — text-to-speech
- ffmpeg (binary)          — audio codec support
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import struct
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Optional

from missy.channels.discord.ffmpeg import ensure_ffmpeg_available

logger = logging.getLogger(__name__)

# Silence threshold: if no audio packets arrive for this many seconds,
# consider the user done speaking and trigger transcription.
_SILENCE_TIMEOUT_S = 1.5

# Minimum speech duration in seconds to bother transcribing.
_MIN_SPEECH_S = 0.3

# Discord sends audio at 48kHz stereo; Whisper expects 16kHz mono.
_DISCORD_SAMPLE_RATE = 48000
_WHISPER_SAMPLE_RATE = 16000


def _clean_for_speech(text: str) -> str:
    """Strip markdown, code blocks, and tool artifacts for TTS delivery."""
    import re

    s = text.strip()

    # Remove code blocks (``` ... ```)
    s = re.sub(r"```[\s\S]*?```", "", s)

    # Remove inline code (`...`)
    s = re.sub(r"`([^`]+)`", r"\1", s)

    # Remove markdown bold/italic markers
    s = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", s)
    s = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", s)

    # Remove markdown headers
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.MULTILINE)

    # Remove markdown links [text](url) → text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)

    # Remove bare URLs
    s = re.sub(r"https?://\S+", "", s)

    # Collapse multiple newlines
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Truncate very long responses for spoken delivery (keep first ~500 chars).
    if len(s) > 600:
        # Try to cut at a sentence boundary.
        cut = s[:600]
        last_period = cut.rfind(".")
        if last_period > 200:
            s = cut[:last_period + 1]
        else:
            s = cut.rstrip() + "..."

    return s.strip()


class DiscordVoiceError(RuntimeError):
    """Actionable voice-layer error."""


@dataclass
class _GuildVoiceState:
    voice_client: Any  # discord.VoiceClient or VoiceRecvClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    listening: bool = False
    listen_task: Optional[asyncio.Task] = None


class DiscordVoiceManager:
    """Manages Discord voice connections per guild using discord.py.

    Supports full conversational voice: the bot listens to users speaking,
    transcribes with STT, runs the agent, synthesises a response with TTS,
    and plays it back in the voice channel.
    """

    def __init__(
        self,
        *,
        agent_callback: Optional[Callable[..., Coroutine]] = None,
        text_channel_callback: Optional[Callable[[str, str], Coroutine]] = None,
    ) -> None:
        """
        Args:
            agent_callback: async (prompt, session_id) -> response_text
            text_channel_callback: async (channel_id, message) -> None
                Sends a text message to a Discord text channel (for transcripts).
        """
        self._agent_callback = agent_callback
        self._text_channel_callback = text_channel_callback
        self._guild_states: Dict[int, _GuildVoiceState] = {}
        self._client: Any = None
        self._discord: Any = None
        self._voice_recv: Any = None
        self._stt_engine: Any = None
        self._tts_engine: Any = None
        self._ready = asyncio.Event()
        self._client_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        try:
            ensure_ffmpeg_available()
        except Exception as e:
            raise DiscordVoiceError(str(e)) from e

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, bot_token: str) -> None:
        """Start the discord.py voice client and load STT/TTS engines."""
        try:
            import discord
        except ImportError as e:
            raise DiscordVoiceError(
                "discord.py[voice] is not installed. "
                "Install with: pip install -e '.[discord_voice]'"
            ) from e

        try:
            from discord.ext import voice_recv  # type: ignore[import-untyped]
            self._voice_recv = voice_recv
        except ImportError as e:
            raise DiscordVoiceError(
                "discord-ext-voice-recv is not installed. "
                "Install with: pip install discord-ext-voice-recv"
            ) from e

        self._discord = discord
        self._loop = asyncio.get_event_loop()

        # Load STT engine.
        try:
            from missy.channels.voice.stt.whisper import FasterWhisperSTT
            self._stt_engine = FasterWhisperSTT(model_size="base.en")
            self._stt_engine.load()
            logger.info("Discord voice: STT engine loaded (faster-whisper base.en)")
        except Exception as exc:
            logger.warning("Discord voice: STT unavailable: %s", exc)
            self._stt_engine = None

        # Load TTS engine.
        try:
            from missy.channels.voice.tts.piper import PiperTTS
            self._tts_engine = PiperTTS()
            self._tts_engine.load()
            logger.info("Discord voice: TTS engine loaded (piper)")
        except Exception as exc:
            logger.warning("Discord voice: TTS unavailable: %s", exc)
            self._tts_engine = None

        # Start discord.py client.
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.guilds = True
        intents.members = True
        intents.message_content = False

        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():
            logger.info(
                "Discord voice client ready as %s (id=%s)",
                self._client.user, self._client.user.id,
            )
            self._ready.set()

        raw_token = bot_token
        if raw_token.startswith("Bot "):
            raw_token = raw_token[4:]

        self._client_task = asyncio.create_task(self._client.start(raw_token))
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise DiscordVoiceError(
                "Discord voice client did not become ready within 30 seconds."
            )

    async def stop(self) -> None:
        """Disconnect all voice connections, unload engines, shut down."""
        for guild_id in list(self._guild_states):
            try:
                await self.leave(guild_id)
            except Exception:
                pass
        if self._stt_engine is not None:
            try:
                self._stt_engine.unload()
            except Exception:
                pass
        if self._tts_engine is not None:
            try:
                self._tts_engine.unload()
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

    @property
    def can_listen(self) -> bool:
        return self._stt_engine is not None and self._voice_recv is not None

    @property
    def can_speak(self) -> bool:
        return self._tts_engine is not None

    # ------------------------------------------------------------------
    # Guild / channel lookup helpers
    # ------------------------------------------------------------------

    def _get_guild(self, guild_id: int) -> Any:
        if self._client is None:
            raise DiscordVoiceError("Voice client is not started.")
        guild = self._client.get_guild(guild_id)
        if guild is None:
            raise DiscordVoiceError(
                "Guild not found. Make sure the bot is a member of the server."
            )
        return guild

    def get_user_voice_channel(self, guild_id: int, user_id: int) -> Optional[Any]:
        guild = self._get_guild(guild_id)
        member = guild.get_member(user_id)
        if member is None:
            return None
        voice_state = member.voice
        if voice_state is None or voice_state.channel is None:
            return None
        return voice_state.channel

    def find_voice_channel(self, guild_id: int, query: str) -> Optional[Any]:
        guild = self._get_guild(guild_id)

        if query.isdigit():
            ch = guild.get_channel(int(query))
            if ch is not None and isinstance(ch, self._discord.VoiceChannel):
                return ch

        query_lower = query.lower()
        for ch in guild.voice_channels:
            if ch.name.lower() == query_lower:
                return ch
        for ch in guild.voice_channels:
            if query_lower in ch.name.lower():
                return ch
        return None

    def list_voice_channels(self, guild_id: int) -> list[str]:
        guild = self._get_guild(guild_id)
        return [ch.name for ch in guild.voice_channels]

    # ------------------------------------------------------------------
    # Join / Leave
    # ------------------------------------------------------------------

    async def join(
        self,
        guild_id: int,
        *,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> str:
        """Join a voice channel and start listening. Returns channel name."""
        guild = self._get_guild(guild_id)
        target: Any = None

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
                    f'No voice channel matching "{channel_name}". '
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

        # Already connected to this channel?
        state = self._guild_states.get(guild_id)
        if state and getattr(state.voice_client, "is_connected", lambda: False)():
            current = getattr(state.voice_client, "channel", None)
            if current and current.id == target.id:
                if not state.listening:
                    self._start_listening(guild_id, state)
                return target.name
            await state.voice_client.move_to(target)
            return target.name

        # Connect using VoiceRecvClient for audio receive support.
        voice_client = await target.connect(cls=self._voice_recv.VoiceRecvClient)
        state = _GuildVoiceState(voice_client=voice_client)
        self._guild_states[guild_id] = state

        # Start listening automatically.
        self._start_listening(guild_id, state)
        return target.name

    async def leave(self, guild_id: int) -> Optional[str]:
        """Leave voice and stop listening. Returns channel name left."""
        state = self._guild_states.pop(guild_id, None)
        if state is None:
            return None

        channel_name = None
        # Stop listening.
        if state.listen_task is not None:
            state.listening = False
            state.listen_task.cancel()
            try:
                await state.listen_task
            except (asyncio.CancelledError, Exception):
                pass

        async with state.lock:
            try:
                ch = getattr(state.voice_client, "channel", None)
                channel_name = getattr(ch, "name", None)
                if hasattr(state.voice_client, "disconnect"):
                    await state.voice_client.disconnect(force=True)
            except Exception as exc:
                logger.warning("Voice disconnect error: %s", exc)
        return channel_name

    # ------------------------------------------------------------------
    # TTS playback
    # ------------------------------------------------------------------

    async def say(self, guild_id: int, text: str) -> None:
        """Speak text via TTS in the connected voice channel."""
        text = (text or "").strip()
        if not text:
            raise DiscordVoiceError("No text provided.")

        if self._tts_engine is None:
            raise DiscordVoiceError("TTS is not configured.")

        state = self._guild_states.get(guild_id)
        if not state or not getattr(state.voice_client, "is_connected", lambda: False)():
            raise DiscordVoiceError(
                "I'm not in a voice channel. Use `!join` to bring me in first."
            )

        await self._play_tts(state, text)

    async def _play_tts(self, state: _GuildVoiceState, text: str) -> None:
        """Synthesize text and play it through the voice connection."""
        async with state.lock:
            # Stop any currently playing audio.
            vc = state.voice_client
            if hasattr(vc, "is_playing") and vc.is_playing():
                if hasattr(vc, "stop"):
                    vc.stop()

            # Synthesize to WAV via piper.
            audio_buf = await self._tts_engine.synthesize(text)

            # Write to temp file for FFmpegPCMAudio.
            path: Optional[str] = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="missy-tts-", suffix=".wav", delete=False
                ) as f:
                    f.write(audio_buf.data)
                    path = f.name

                source = self._discord.FFmpegPCMAudio(path)
                done = asyncio.Event()

                def _after(err: Optional[BaseException]) -> None:
                    if err:
                        logger.exception("Voice playback error", exc_info=err)
                    done.set()

                vc.play(source, after=_after)

                try:
                    await asyncio.wait_for(done.wait(), timeout=120)
                except asyncio.TimeoutError:
                    if hasattr(vc, "stop"):
                        vc.stop()
                    logger.warning("TTS playback timed out")
            finally:
                if path:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

    # ------------------------------------------------------------------
    # Listening / STT / conversation loop
    # ------------------------------------------------------------------

    def _start_listening(self, guild_id: int, state: _GuildVoiceState) -> None:
        """Begin listening for user speech in the voice channel."""
        if not self.can_listen:
            logger.info("Discord voice: listening disabled (STT or voice_recv unavailable)")
            return

        state.listening = True
        SinkClass = _make_sink_class(self._voice_recv)
        sink = SinkClass(
            loop=self._loop,
            on_speech_done=lambda user_id, pcm, sr: asyncio.run_coroutine_threadsafe(
                self._handle_speech(guild_id, user_id, pcm, sr),
                self._loop,
            ),
            bot_user_id=self._client.user.id if self._client.user else 0,
        )
        state.voice_client.listen(sink)
        logger.info("Discord voice: now listening in guild %d", guild_id)

    async def _handle_speech(
        self,
        guild_id: int,
        user_id: int,
        pcm_48k: bytes,
        sample_rate: int,
    ) -> None:
        """Process captured speech: resample → STT → agent → TTS → playback."""
        state = self._guild_states.get(guild_id)
        if state is None or not state.listening:
            return

        # Don't process while bot is speaking.
        vc = state.voice_client
        if hasattr(vc, "is_playing") and vc.is_playing():
            return

        try:
            # Resample 48kHz → 16kHz mono PCM for Whisper.
            pcm_16k = _resample_pcm(pcm_48k, sample_rate, _WHISPER_SAMPLE_RATE)

            # Minimum speech check.
            duration_s = len(pcm_16k) / (2 * _WHISPER_SAMPLE_RATE)
            if duration_s < _MIN_SPEECH_S:
                return

            # Transcribe.
            result = await self._stt_engine.transcribe(
                pcm_16k, sample_rate=_WHISPER_SAMPLE_RATE, channels=1,
            )
            transcript = result.text.strip()
            if not transcript:
                return

            # Look up user name for logging.
            guild = self._get_guild(guild_id)
            member = guild.get_member(user_id)
            user_name = member.display_name if member else str(user_id)
            logger.info(
                "Discord voice STT [%s]: %r (confidence=%.2f, %dms)",
                user_name, transcript, result.confidence, result.processing_ms,
            )

            # Run agent.
            if self._agent_callback is None:
                logger.warning("Discord voice: no agent callback configured")
                return

            session_id = f"voice-{user_id}"
            voice_ctx = (
                "[CONTEXT] You are responding via Discord voice chat. "
                "Your text response will be spoken aloud automatically — "
                "do NOT use the tts_speak tool or any audio tools. "
                "Keep your response concise, conversational, and natural "
                "for spoken delivery. Avoid markdown, code blocks, and long lists."
            )
            response = await self._agent_callback(
                f"{voice_ctx}\n\n[Voice from {user_name}]: {transcript}",
                session_id,
            )

            if not response or not response.strip():
                return

            # Clean response for spoken delivery.
            response = _clean_for_speech(response)
            if not response:
                return

            logger.info(
                "Discord voice response to %s: %r",
                user_name, response[:100],
            )

            # Speak response via TTS.
            if self.can_speak:
                try:
                    await self._play_tts(state, response)
                except Exception as exc:
                    logger.error("Discord voice TTS playback failed: %s", exc)

        except Exception as exc:
            logger.exception("Discord voice _handle_speech error: %s", exc)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_connected(self, guild_id: int) -> bool:
        state = self._guild_states.get(guild_id)
        if state is None:
            return False
        return bool(getattr(state.voice_client, "is_connected", lambda: False)())

    def is_listening(self, guild_id: int) -> bool:
        state = self._guild_states.get(guild_id)
        if state is None:
            return False
        return state.listening

    def current_channel_name(self, guild_id: int) -> Optional[str]:
        state = self._guild_states.get(guild_id)
        if state is None:
            return None
        ch = getattr(state.voice_client, "channel", None)
        return getattr(ch, "name", None) if ch else None


# ======================================================================
# Audio sink — collects per-user speech with silence detection
# ======================================================================


def _make_sink_class(voice_recv_module: Any) -> type:
    """Dynamically create a sink class inheriting from voice_recv.AudioSink.

    This is necessary because discord-ext-voice-recv is a lazy import and
    AudioSink must be the base class for the sink to be accepted.
    """

    class _SpeechCollectorSink(voice_recv_module.AudioSink):
        """AudioSink that buffers PCM per user and fires a callback on silence.

        Runs on discord.py's voice receive thread. The callback is
        dispatched to the asyncio event loop via run_coroutine_threadsafe.
        """

        def __init__(
            self,
            *,
            loop: asyncio.AbstractEventLoop,
            on_speech_done: Callable[[int, bytes, int], Any],
            bot_user_id: int = 0,
        ) -> None:
            super().__init__()
            self._loop = loop
            self._on_speech_done = on_speech_done
            self._bot_user_id = bot_user_id

            # Per-user audio buffers: user_id -> list of PCM chunks.
            self._buffers: Dict[int, list[bytes]] = defaultdict(list)
            # Per-user last-packet timestamp.
            self._last_packet_time: Dict[int, float] = {}
            # Per-user silence timer handles.
            self._timers: Dict[int, asyncio.TimerHandle] = {}
            self._lock = threading.Lock()

        def wants_opus(self) -> bool:
            """We want decoded PCM, not raw Opus."""
            return False

        def write(self, user: Any, data: Any) -> None:
            """Called by discord-ext-voice-recv for each audio packet."""
            if user is None:
                return
            user_id = user.id if hasattr(user, "id") else int(user)

            # Ignore our own audio (echo prevention).
            if user_id == self._bot_user_id:
                return

            pcm = data.pcm if hasattr(data, "pcm") else data
            if not pcm:
                return

            now = time.monotonic()
            with self._lock:
                self._buffers[user_id].append(pcm)
                self._last_packet_time[user_id] = now

                # Reset the silence timer for this user.
                old_timer = self._timers.pop(user_id, None)
                if old_timer is not None:
                    old_timer.cancel()

                self._timers[user_id] = self._loop.call_later(
                    _SILENCE_TIMEOUT_S,
                    self._on_silence,
                    user_id,
                )

        def _on_silence(self, user_id: int) -> None:
            """Called when a user has been silent for _SILENCE_TIMEOUT_S."""
            with self._lock:
                chunks = self._buffers.pop(user_id, [])
                self._last_packet_time.pop(user_id, None)
                self._timers.pop(user_id, None)

            if not chunks:
                return

            pcm = b"".join(chunks)
            # Fire the callback (which schedules an async coroutine).
            try:
                self._on_speech_done(user_id, pcm, _DISCORD_SAMPLE_RATE)
            except Exception as exc:
                logger.error("_SpeechCollectorSink callback error: %s", exc)

        def cleanup(self) -> None:
            """Called when the sink is removed."""
            with self._lock:
                for timer in self._timers.values():
                    timer.cancel()
                self._timers.clear()
                self._buffers.clear()
                self._last_packet_time.clear()

    return _SpeechCollectorSink


# ======================================================================
# Audio resampling helper
# ======================================================================


def _resample_pcm(
    pcm: bytes,
    from_rate: int,
    to_rate: int,
) -> bytes:
    """Resample 16-bit signed LE PCM from one sample rate to another.

    Uses linear interpolation. Converts stereo to mono if the data
    appears to be stereo (Discord sends stereo 48kHz).
    """
    if from_rate == to_rate:
        return pcm

    # Parse 16-bit samples.
    sample_count = len(pcm) // 2
    samples = struct.unpack(f"<{sample_count}h", pcm[:sample_count * 2])

    # Convert stereo to mono (Discord sends 2-channel audio).
    # Heuristic: if sample count is even and sounds like stereo, mix down.
    if sample_count >= 2:
        mono = []
        for i in range(0, len(samples) - 1, 2):
            mono.append((samples[i] + samples[i + 1]) // 2)
        samples = mono

    # Linear interpolation resample.
    ratio = from_rate / to_rate / 2  # /2 because we already halved via stereo→mono
    out_count = int(len(samples) / ratio)
    if out_count == 0:
        return b""

    resampled = []
    for i in range(out_count):
        src_idx = i * ratio
        idx = int(src_idx)
        frac = src_idx - idx
        if idx + 1 < len(samples):
            val = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        elif idx < len(samples):
            val = float(samples[idx])
        else:
            break
        # Clamp to int16 range.
        val = max(-32768, min(32767, int(val)))
        resampled.append(val)

    return struct.pack(f"<{len(resampled)}h", *resampled)
