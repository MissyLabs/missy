"""VoiceChannel — BaseChannel wrapper around VoiceServer.

:class:`VoiceChannel` integrates the WebSocket voice server into the Missy
channel framework.  It starts :class:`~missy.channels.voice.server.VoiceServer`
in a dedicated daemon thread with its own asyncio event loop, wires up the
STT/TTS engines and device registry, and exposes the server via
:meth:`get_server`.

Voice is inherently event-driven: audio arrives asynchronously from edge
nodes, so :meth:`receive` and :meth:`send` are not applicable and both raise
:exc:`NotImplementedError` with an explanatory message.

Example::

    from missy.channels.voice.channel import VoiceChannel

    channel = VoiceChannel(
        host="127.0.0.1",
        port=8765,
        stt_model="base.en",
        tts_voice="en_US-lessac-medium",
    )
    channel.start(agent_runtime)
    # ... voice server is running in background thread ...
    channel.stop()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Callable
from typing import Any

from missy.channels.base import BaseChannel, ChannelMessage
from missy.channels.voice.pairing import PairingManager
from missy.channels.voice.presence import PresenceStore
from missy.channels.voice.registry import DeviceRegistry
from missy.channels.voice.server import VoiceServer
from missy.channels.voice.stt.whisper import FasterWhisperSTT
from missy.channels.voice.tts.piper import PiperTTS

logger = logging.getLogger(__name__)


def _build_agent_callback(agent_runtime: Any) -> Callable[..., Any]:
    """Construct an async agent callback from *agent_runtime*.

    If *agent_runtime* exposes ``run_async`` it is used directly.  Otherwise,
    the synchronous ``run`` method is wrapped to execute in the default thread
    pool executor, keeping the event loop unblocked.

    Args:
        agent_runtime: An agent runtime object that has either an async
            ``run_async(prompt, session_id, metadata) -> str`` method or a
            synchronous ``run(prompt, session_id) -> str`` method.

    Returns:
        An async callable with signature
        ``async (prompt: str, session_id: str, metadata: dict) -> str``.
    """
    if hasattr(agent_runtime, "run_async") and asyncio.iscoroutinefunction(agent_runtime.run_async):

        async def _async_cb(prompt: str, session_id: str, metadata: dict) -> str:
            return await agent_runtime.run_async(prompt, session_id, metadata)

        return _async_cb

    # Fall back: wrap synchronous run() in an executor.
    async def _sync_cb(prompt: str, session_id: str, metadata: dict) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, agent_runtime.run, prompt, session_id)

    return _sync_cb


class VoiceChannel(BaseChannel):
    """Event-driven voice channel backed by a WebSocket server.

    Manages the full lifecycle of the voice subsystem: device registry,
    STT/TTS engines, pairing manager, presence store, and the WebSocket server
    itself.  The server runs in a daemon thread with a dedicated asyncio event
    loop, so it does not block the caller's thread.

    Args:
        host: Interface to bind the WebSocket server on.  Defaults to
            ``"127.0.0.1"``.  Pass ``"0.0.0.0"`` to expose on all interfaces
            (a ``voice.bind.warning`` audit event will be emitted).
        port: TCP port for the WebSocket server.  Defaults to ``8765``.
        registry_path: Path to the JSON device registry file.  Defaults to
            ``"~/.missy/devices.json"``.
        stt_model: Whisper model size passed to
            :class:`~missy.channels.voice.stt.whisper.FasterWhisperSTT`.
            Defaults to ``"base.en"``.
        tts_voice: Piper voice name passed to
            :class:`~missy.channels.voice.tts.piper.PiperTTS`.
            Defaults to ``"en_US-lessac-medium"``.
        audio_chunk_size: Number of bytes per binary WebSocket frame when
            streaming TTS audio.  Defaults to ``4096``.
        debug_transcripts: When ``True``, STT transcripts are forwarded to the
            client as ``transcript`` frames.  Defaults to ``False``.
        bind_requires_policy: Reserved flag for future network-policy
            integration.  Has no effect in the current implementation.
    """

    name = "voice"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        registry_path: str = "~/.missy/devices.json",
        stt_model: str = "base.en",
        tts_voice: str = "en_US-lessac-medium",
        audio_chunk_size: int = 4096,
        debug_transcripts: bool = False,
        bind_requires_policy: bool = True,
    ) -> None:
        self._host = host
        self._port = port
        self._registry_path = registry_path
        self._stt_model = stt_model
        self._tts_voice = tts_voice
        self._audio_chunk_size = audio_chunk_size
        self._debug_transcripts = debug_transcripts
        self._bind_requires_policy = bind_requires_policy

        # Populated during start().
        self._registry: DeviceRegistry | None = None
        self._pairing_manager: PairingManager | None = None
        self._presence_store: PresenceStore | None = None
        self._server: VoiceServer | None = None

        # Thread / loop management.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    def receive(self) -> ChannelMessage | None:
        """Not applicable to the voice channel.

        The voice channel is fully event-driven: audio arrives from edge nodes
        via the WebSocket server and is processed asynchronously.  There is no
        poll-based receive path.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("VoiceChannel is event-driven; use start(agent_runtime) instead")

    def send(self, message: str) -> None:
        """Not applicable to the voice channel.

        TTS responses are sent directly to the originating edge node by the
        WebSocket server during audio processing.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("VoiceChannel is event-driven; use start(agent_runtime) instead")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, agent_runtime: Any) -> None:
        """Initialise all subsystems and start the WebSocket server.

        Loads the device registry, constructs STT/TTS engines, wiring manager,
        and presence store, creates a :class:`~missy.channels.voice.server.VoiceServer`,
        then starts it in a daemon thread.

        Args:
            agent_runtime: An object exposing either
                ``async run_async(prompt, session_id, metadata) -> str``
                or synchronous ``run(prompt, session_id) -> str``.  The
                appropriate wrapper is selected automatically.

        Raises:
            RuntimeError: If :meth:`start` is called while the channel is
                already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("VoiceChannel is already running.")

        # Device registry.
        registry = DeviceRegistry(registry_path=self._registry_path)
        registry.load()
        self._registry = registry

        pairing_manager = PairingManager(registry)
        self._pairing_manager = pairing_manager

        presence_store = PresenceStore(registry)
        self._presence_store = presence_store

        # STT / TTS engines.
        stt_engine = FasterWhisperSTT(model_size=self._stt_model)
        tts_engine = PiperTTS(voice=self._tts_voice)

        # Build the async agent callback.
        agent_callback = _build_agent_callback(agent_runtime)

        # Construct the server.
        server = VoiceServer(
            registry=registry,
            pairing_manager=pairing_manager,
            presence_store=presence_store,
            stt_engine=stt_engine,
            tts_engine=tts_engine,
            agent_callback=agent_callback,
            host=self._host,
            port=self._port,
            audio_chunk_size=self._audio_chunk_size,
            debug_transcripts=self._debug_transcripts,
        )
        self._server = server

        # Start the server in a dedicated daemon thread.
        started_event = threading.Event()
        error_holder: list[BaseException] = []

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _runner() -> None:
                try:
                    await server.start()
                except Exception as exc:
                    error_holder.append(exc)
                    started_event.set()
                    return

                started_event.set()

                # Keep the loop alive until stop() is called.
                while server._running:  # noqa: SLF001
                    await asyncio.sleep(0.25)

            try:
                loop.run_until_complete(_runner())
            except Exception:
                logger.error("VoiceChannel: event loop terminated with error.", exc_info=True)
            finally:
                loop.close()
                self._loop = None

        self._thread = threading.Thread(
            target=_run_loop,
            name="missy-voice-server",
            daemon=True,
        )
        self._thread.start()

        # Wait until the server is ready (or has failed).
        started_event.wait(timeout=30)

        if error_holder:
            self._thread = None
            self._server = None
            raise RuntimeError(
                f"VoiceChannel failed to start: {error_holder[0]}"
            ) from error_holder[0]

        logger.info("VoiceChannel: started on ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Stop the WebSocket server and join the background thread.

        Idempotent: calling :meth:`stop` when not running is a no-op.
        """
        server = self._server
        loop = self._loop
        thread = self._thread

        if server is None or loop is None or thread is None:
            logger.debug("VoiceChannel.stop(): not running — no-op.")
            return

        # Schedule server.stop() on the server's event loop from this thread.
        future = concurrent.futures.Future()  # type: concurrent.futures.Future[None]

        async def _stop_coro() -> None:
            try:
                await server.stop()
            finally:
                future.set_result(None)

        asyncio.run_coroutine_threadsafe(_stop_coro(), loop)

        try:
            future.result(timeout=15)
        except Exception:
            logger.warning("VoiceChannel.stop(): server stop timed out or raised.", exc_info=True)

        thread.join(timeout=10)
        if thread.is_alive():
            logger.warning("VoiceChannel.stop(): background thread did not exit cleanly.")

        self._thread = None
        self._server = None
        self._loop = None
        logger.info("VoiceChannel: stopped.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_server(self) -> VoiceServer | None:
        """Return the underlying :class:`~missy.channels.voice.server.VoiceServer` instance.

        Returns:
            The :class:`VoiceServer` if the channel has been started, or
            ``None`` if :meth:`start` has not been called yet.
        """
        return self._server

    def get_presence_context(self) -> str:
        """Return a human-readable presence summary for agent context injection.

        Delegates to :meth:`~missy.channels.voice.presence.PresenceStore.get_context_summary`.

        Returns:
            A string such as ``"Living Room: occupied | Bedroom: unknown"``,
            or ``"(no nodes registered)"`` if no presence store exists yet.
        """
        if self._presence_store is None:
            return "(no nodes registered)"
        return self._presence_store.get_context_summary()
