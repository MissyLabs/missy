"""WebSocket voice channel server for Missy.

Implements the :class:`VoiceServer`, which accepts WebSocket connections from
edge nodes (e.g. ReSpeaker devices, Raspberry Pi units), authenticates them
against the :class:`~missy.channels.voice.registry.DeviceRegistry`, handles
the audio-in / audio-out exchange, and streams TTS responses back over the
same connection.

Protocol overview (all JSON unless otherwise noted):

Client → Server::

    {"type": "auth",        "node_id": "...", "token": "..."}
    {"type": "audio_start", "sample_rate": 16000, "channels": 1, "format": "pcm_s16le"}
    <binary frames: raw PCM audio bytes>
    {"type": "audio_end"}
    {"type": "heartbeat",   "node_id": "...", "occupancy": null,
                             "noise_level": null, "wake_word_fp": false}
    {"type": "pair_request","friendly_name": "...", "room": "...",
                             "hardware_profile": {}}

Server → Client::

    {"type": "auth_ok",       "node_id": "...", "room": "..."}
    {"type": "auth_fail",     "reason": "..."}
    {"type": "pair_pending",  "node_id": "..."}
    {"type": "transcript",    "text": "...", "confidence": 0.95}
    {"type": "response_text", "text": "..."}
    {"type": "audio_start",   "sample_rate": 22050, "format": "wav"}
    <binary frames: WAV audio bytes>
    {"type": "audio_end"}
    {"type": "error",  "message": "..."}
    {"type": "muted"}

Example::

    from missy.channels.voice.server import VoiceServer

    server = VoiceServer(
        registry=registry,
        pairing_manager=pairing_manager,
        presence_store=presence_store,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        agent_callback=my_async_callable,
    )
    await server.start()
    # ...
    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import websockets
import websockets.exceptions
from websockets.server import WebSocketServerProtocol

from missy.channels.voice.pairing import PairingManager
from missy.channels.voice.presence import PresenceStore
from missy.channels.voice.registry import DeviceRegistry, EdgeNode
from missy.channels.voice.stt.base import STTEngine
from missy.channels.voice.tts.base import TTSEngine
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

# Maximum audio payload held in memory per connection (10 MB).
_MAX_AUDIO_BYTES = 10 * 1024 * 1024

# Audit task-id used for all server-level events.
_TASK_ID = "voice-server"


def _emit(
    session_id: str,
    event_type: str,
    result: str,
    detail: dict[str, Any] | None = None,
) -> None:
    """Publish a voice-server audit event on the shared :data:`~missy.core.events.event_bus`.

    Args:
        session_id: Identifier of the session (usually the ``node_id``).
        event_type: Dotted event-type string, e.g. ``"voice.connection.auth_ok"``.
        result: One of ``"allow"``, ``"deny"``, or ``"error"``.
        detail: Optional structured metadata for the event.
    """
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id=session_id,
                task_id=_TASK_ID,
                event_type=event_type,
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail=detail or {},
            )
        )
    except Exception:
        logger.debug("voice server: audit emit failed for %r", event_type, exc_info=True)


class VoiceServer:
    """WebSocket server that bridges edge-node audio to the Missy agent.

    Each incoming WebSocket connection is handled in its own coroutine.  The
    protocol is stateful: the first frame from a client *must* be either an
    ``auth`` or a ``pair_request`` message; any other first frame causes an
    immediate close.

    Args:
        registry: The device registry used for token verification and node
            metadata look-ups.
        pairing_manager: Handles first-contact pair requests from unregistered
            nodes.
        presence_store: Tracks room occupancy and sensor readings reported via
            heartbeat frames.
        stt_engine: Speech-to-text engine.  :meth:`~STTEngine.load` is called
            during :meth:`start`; :meth:`~STTEngine.unload` during
            :meth:`stop`.
        tts_engine: Text-to-speech engine.  Same lifecycle as *stt_engine*.
        agent_callback: Async callable with signature
            ``async (prompt: str, session_id: str, metadata: dict) -> str``
            that routes the transcribed text to the agent and returns its
            reply.
        host: Interface to bind on.  Defaults to ``"127.0.0.1"``.  Binding to
            ``"0.0.0.0"`` emits a ``voice.bind.warning`` audit event.
        port: TCP port to listen on.  Defaults to ``8765``.
        audio_chunk_size: Number of audio bytes sent in each binary WebSocket
            frame when streaming TTS audio back to the client.  Defaults to
            ``4096``.
        debug_transcripts: When ``True``, a ``transcript`` JSON frame is sent
            to the client after each STT pass.  Defaults to ``False``.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        pairing_manager: PairingManager,
        presence_store: PresenceStore,
        stt_engine: STTEngine,
        tts_engine: TTSEngine,
        agent_callback: Callable[..., Any],
        host: str = "127.0.0.1",
        port: int = 8765,
        audio_chunk_size: int = 4096,
        debug_transcripts: bool = False,
    ) -> None:
        self._registry = registry
        self._pairing_manager = pairing_manager
        self._presence_store = presence_store
        self._stt = stt_engine
        self._tts = tts_engine
        self._agent_callback = agent_callback
        self._host = host
        self._port = port
        self._audio_chunk_size = audio_chunk_size
        self._debug_transcripts = debug_transcripts

        self._running: bool = False
        self._ws_server: Any | None = None  # websockets.WebSocketServer
        # Tracks node_ids of currently connected, authenticated connections.
        self._connected_nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load STT/TTS engines and start accepting WebSocket connections.

        Emits ``voice.bind.warning`` if binding to ``"0.0.0.0"``.

        Raises:
            OSError: If the underlying TCP socket cannot be bound (e.g. port
                already in use).
        """
        if self._running:
            logger.debug("VoiceServer.start() called but server is already running.")
            return

        if self._host == "0.0.0.0":
            logger.warning(
                "VoiceServer: binding to 0.0.0.0 exposes the voice channel on all interfaces."
            )
            _emit(
                session_id="system",
                event_type="voice.bind.warning",
                result="allow",
                detail={"host": self._host, "port": self._port},
            )

        logger.info("VoiceServer: loading STT engine (%s)…", self._stt.name)
        self._stt.load()

        logger.info("VoiceServer: loading TTS engine (%s)…", self._tts.name)
        self._tts.load()

        self._ws_server = await websockets.serve(
            self._handle_connection,
            self._host,
            self._port,
        )
        self._running = True
        logger.info(
            "VoiceServer: listening on ws://%s:%d", self._host, self._port
        )

    async def stop(self) -> None:
        """Close all connections and unload STT/TTS engines.

        Idempotent: calling :meth:`stop` on a server that is not running is a
        no-op.
        """
        if not self._running:
            return

        self._running = False

        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()
            self._ws_server = None

        self._stt.unload()
        self._tts.unload()
        self._connected_nodes.clear()
        logger.info("VoiceServer: stopped.")

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self, websocket: WebSocketServerProtocol
    ) -> None:
        """Coroutine invoked by the websockets library for each new connection.

        The first message must be ``auth`` or ``pair_request``.  Any other
        message type results in an immediate close with a ``denied`` audit
        event.

        Args:
            websocket: The newly accepted WebSocket connection.
        """
        remote_addr = websocket.remote_address
        logger.debug("VoiceServer: new connection from %s", remote_addr)

        node: EdgeNode | None = None

        try:
            # ----------------------------------------------------------
            # First frame: must be auth or pair_request.
            # ----------------------------------------------------------
            try:
                raw_first = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                logger.debug("VoiceServer: connection from %s closed before first frame.", remote_addr)
                return

            if isinstance(raw_first, bytes):
                logger.debug("VoiceServer: first frame is binary — rejecting.")
                _emit(
                    session_id="unknown",
                    event_type="voice.connection.rejected_unauthenticated",
                    result="deny",
                    detail={"remote": str(remote_addr), "reason": "binary frame before auth"},
                )
                await websocket.close(1008, "First frame must be JSON auth or pair_request")
                return

            try:
                first_msg: dict[str, Any] = json.loads(raw_first)
            except json.JSONDecodeError:
                _emit(
                    session_id="unknown",
                    event_type="voice.connection.rejected_unauthenticated",
                    result="deny",
                    detail={"remote": str(remote_addr), "reason": "malformed JSON"},
                )
                await websocket.close(1008, "Malformed JSON")
                return

            msg_type = first_msg.get("type", "")

            if msg_type == "auth":
                node = await self._handle_auth(
                    websocket,
                    node_id=first_msg.get("node_id", ""),
                    token=first_msg.get("token", ""),
                )
                if node is None:
                    # _handle_auth already sent auth_fail and closed.
                    return

            elif msg_type == "pair_request":
                await self._handle_pair_request(websocket, first_msg)
                # pair_request ends the connection after sending pair_pending.
                return

            else:
                logger.debug(
                    "VoiceServer: unexpected first message type %r from %s — closing.",
                    msg_type,
                    remote_addr,
                )
                _emit(
                    session_id="unknown",
                    event_type="voice.connection.rejected_unauthenticated",
                    result="deny",
                    detail={
                        "remote": str(remote_addr),
                        "reason": f"unexpected first message type: {msg_type!r}",
                    },
                )
                await self._send_json(websocket, {"type": "error", "message": "First message must be auth or pair_request"})
                await websocket.close(1008, "Protocol violation")
                return

            # ----------------------------------------------------------
            # At this point node is set and authenticated.
            # Track the connection.
            # ----------------------------------------------------------
            self._connected_nodes.add(node.node_id)

            try:
                await self._message_loop(websocket, node)
            finally:
                self._connected_nodes.discard(node.node_id)
                _emit(
                    session_id=node.node_id,
                    event_type="voice.connection.closed",
                    result="allow",
                    detail={"node_id": node.node_id},
                )
                try:
                    self._registry.mark_offline(node.node_id)
                except Exception:
                    logger.debug("VoiceServer: mark_offline failed for %s", node.node_id, exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            if node is not None:
                logger.debug("VoiceServer: connection closed for node %s", node.node_id)
            else:
                logger.debug("VoiceServer: connection closed before auth from %s", remote_addr)
        except Exception:
            logger.error(
                "VoiceServer: unexpected error in connection handler for %s",
                remote_addr,
                exc_info=True,
            )
            try:
                await self._send_json(websocket, {"type": "error", "message": "Internal server error"})
                await websocket.close(1011, "Internal error")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Per-connection message loop
    # ------------------------------------------------------------------

    async def _message_loop(
        self,
        websocket: WebSocketServerProtocol,
        node: EdgeNode,
    ) -> None:
        """Dispatch incoming messages for an authenticated *node*.

        Handles ``heartbeat``, ``audio_start``/binary frames/``audio_end``
        sequences.  Unrecognised message types are logged and skipped.

        Args:
            websocket: The authenticated WebSocket connection.
            node: The authenticated :class:`EdgeNode`.
        """
        # State for accumulating audio across binary frames.
        in_audio_session: bool = False
        audio_buffer: bytes = b""
        audio_sample_rate: int = 16000
        audio_channels: int = 1

        async for raw in websocket:
            if isinstance(raw, bytes):
                # Binary frame — audio payload accumulation.
                if not in_audio_session:
                    logger.debug(
                        "VoiceServer: received binary frame outside audio session from %s — ignoring.",
                        node.node_id,
                    )
                    continue

                audio_buffer += raw

                if len(audio_buffer) > _MAX_AUDIO_BYTES:
                    logger.warning(
                        "VoiceServer: audio buffer exceeded %d bytes for node %s — closing connection.",
                        _MAX_AUDIO_BYTES,
                        node.node_id,
                    )
                    await self._send_json(
                        websocket,
                        {"type": "error", "message": "Audio buffer size limit exceeded (10 MB)"},
                    )
                    await websocket.close(1009, "Audio buffer too large")
                    return

                continue

            # JSON frame.
            try:
                msg: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("VoiceServer: malformed JSON from %s — skipping.", node.node_id)
                await self._send_json(websocket, {"type": "error", "message": "Malformed JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "audio_start":
                in_audio_session = True
                audio_buffer = b""
                audio_sample_rate = int(msg.get("sample_rate", 16000))
                audio_channels = int(msg.get("channels", 1))
                logger.debug(
                    "VoiceServer: audio_start from %s — rate=%d ch=%d",
                    node.node_id,
                    audio_sample_rate,
                    audio_channels,
                )

            elif msg_type == "audio_end":
                if not in_audio_session:
                    logger.debug(
                        "VoiceServer: audio_end outside audio session from %s — ignoring.",
                        node.node_id,
                    )
                    continue

                in_audio_session = False
                captured = audio_buffer
                audio_buffer = b""

                await self._handle_audio(
                    websocket,
                    node=node,
                    audio_buffer=captured,
                    sample_rate=audio_sample_rate,
                    channels=audio_channels,
                )

            elif msg_type == "heartbeat":
                await self._handle_heartbeat(websocket, node=node, data=msg)

            elif msg_type == "auth":
                # Re-auth on an already-authenticated connection is ignored.
                logger.debug("VoiceServer: spurious auth frame from %s — ignoring.", node.node_id)

            else:
                logger.debug(
                    "VoiceServer: unknown message type %r from %s — skipping.",
                    msg_type,
                    node.node_id,
                )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _handle_auth(
        self,
        websocket: WebSocketServerProtocol,
        node_id: str,
        token: str,
    ) -> EdgeNode | None:
        """Authenticate a node by verifying its token against the registry.

        On success sends ``auth_ok`` and returns the :class:`EdgeNode`.
        On failure sends ``auth_fail``, closes the connection, and returns
        ``None``.

        Security guarantees:

        * Nodes with ``paired=False`` are always rejected (must go through the
          pair_request flow first).
        * Nodes with ``policy_mode="muted"`` receive a ``muted`` frame and are
          disconnected immediately.

        Args:
            websocket: The connection being authenticated.
            node_id: The ``node_id`` claimed by the client.
            token: The plaintext auth token sent by the client.

        Returns:
            The :class:`EdgeNode` on success, or ``None`` on failure.
        """
        # Verify token.
        if not self._registry.verify_token(node_id, token):
            reason = "invalid credentials"
            logger.info("VoiceServer: auth failure for node %r — %s", node_id, reason)
            _emit(
                session_id=node_id,
                event_type="voice.connection.auth_fail",
                result="deny",
                detail={"node_id": node_id, "reason": reason},
            )
            await self._send_json(websocket, {"type": "auth_fail", "reason": reason})
            await websocket.close(1008, "Authentication failed")
            return None

        node = self._registry.get_node(node_id)

        # Should not happen if verify_token passed, but guard defensively.
        if node is None:
            reason = "node not found after token verification"
            _emit(
                session_id=node_id,
                event_type="voice.connection.auth_fail",
                result="deny",
                detail={"node_id": node_id, "reason": reason},
            )
            await self._send_json(websocket, {"type": "auth_fail", "reason": reason})
            await websocket.close(1008, "Authentication failed")
            return None

        if not node.paired:
            reason = "node is not yet approved"
            logger.info("VoiceServer: auth failure for node %r — %s", node_id, reason)
            _emit(
                session_id=node_id,
                event_type="voice.connection.auth_fail",
                result="deny",
                detail={"node_id": node_id, "reason": reason},
            )
            await self._send_json(websocket, {"type": "auth_fail", "reason": reason})
            await websocket.close(1008, "Authentication failed")
            return None

        if node.policy_mode == "muted":
            logger.info("VoiceServer: node %r is muted — rejecting.", node_id)
            _emit(
                session_id=node_id,
                event_type="voice.connection.rejected_muted",
                result="deny",
                detail={"node_id": node_id},
            )
            await self._send_json(websocket, {"type": "muted"})
            await websocket.close(1008, "Node is muted")
            return None

        # Auth succeeded.
        _emit(
            session_id=node_id,
            event_type="voice.connection.auth_ok",
            result="allow",
            detail={"node_id": node_id, "room": node.room},
        )
        await self._send_json(
            websocket,
            {"type": "auth_ok", "node_id": node.node_id, "room": node.room},
        )
        logger.info(
            "VoiceServer: node %r authenticated (room=%r, policy=%r).",
            node_id,
            node.room,
            node.policy_mode,
        )
        return node

    # ------------------------------------------------------------------
    # Pair request
    # ------------------------------------------------------------------

    async def _handle_pair_request(
        self,
        websocket: WebSocketServerProtocol,
        data: dict[str, Any],
    ) -> None:
        """Register a new edge node and send ``pair_pending``.

        The node is created with ``paired=False``.  An operator must approve
        it via the CLI before it can authenticate.

        Args:
            websocket: The requesting WebSocket connection.
            data: The decoded JSON frame containing ``friendly_name``, ``room``,
                and ``hardware_profile`` keys.
        """
        friendly_name = data.get("friendly_name", "unknown")
        room = data.get("room", "unknown")
        hardware_profile = data.get("hardware_profile", {})
        remote_addr = websocket.remote_address
        ip_address = str(remote_addr[0]) if remote_addr else "unknown"

        node_id = self._pairing_manager.initiate_pairing(
            node_id="",
            friendly_name=friendly_name,
            room=room,
            ip_address=ip_address,
            hardware_profile=hardware_profile,
        )

        _emit(
            session_id=node_id,
            event_type="voice.pair_request",
            result="allow",
            detail={
                "node_id": node_id,
                "friendly_name": friendly_name,
                "room": room,
                "ip_address": ip_address,
            },
        )

        await self._send_json(websocket, {"type": "pair_pending", "node_id": node_id})
        logger.info(
            "VoiceServer: pair_request from %s — assigned node_id=%r (pending approval).",
            ip_address,
            node_id,
        )
        # Close after informing the client; they must reconnect with auth once approved.
        await websocket.close(1000, "Pairing pending")

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    async def _handle_audio(
        self,
        websocket: WebSocketServerProtocol,
        node: EdgeNode,
        audio_buffer: bytes,
        sample_rate: int,
        channels: int = 1,
    ) -> None:
        """Transcribe audio, invoke the agent, synthesise a reply, and stream it back.

        Processing steps:

        1. Emit ``voice.audio.received`` audit event.
        2. Optionally persist audio to disk (if ``node.audio_logging`` is set).
        3. Call the STT engine.  On STT failure send ``error`` frame and return.
        4. If ``debug_transcripts`` is enabled, send a ``transcript`` frame.
        5. Call ``agent_callback`` with the transcribed text.
        6. Call the TTS engine.  On TTS failure send ``response_text`` only.
        7. Stream WAV audio back in chunks between ``audio_start``/``audio_end``.

        Args:
            websocket: The authenticated connection.
            node: The sending :class:`EdgeNode`.
            audio_buffer: Accumulated raw PCM audio bytes.
            sample_rate: Sample rate reported by the client (Hz).
            channels: Channel count reported by the client.
        """
        _emit(
            session_id=node.node_id,
            event_type="voice.audio.received",
            result="allow",
            detail={
                "node_id": node.node_id,
                "bytes": len(audio_buffer),
                "sample_rate": sample_rate,
                "channels": channels,
            },
        )

        # ------------------------------------------------------------------
        # Optional audio logging.
        # ------------------------------------------------------------------
        if node.audio_logging and node.audio_log_dir:
            await self._log_audio_to_disk(node, audio_buffer, sample_rate, channels)

        # ------------------------------------------------------------------
        # Speech-to-text.
        # ------------------------------------------------------------------
        try:
            transcript = await self._stt.transcribe(
                audio_buffer,
                sample_rate=sample_rate,
                channels=channels,
            )
        except Exception:
            logger.error(
                "VoiceServer: STT failed for node %s", node.node_id, exc_info=True
            )
            await self._send_json(
                websocket,
                {"type": "error", "message": "Speech recognition failed"},
            )
            return

        if self._debug_transcripts:
            await self._send_json(
                websocket,
                {
                    "type": "transcript",
                    "text": transcript.text,
                    "confidence": transcript.confidence,
                },
            )

        if not transcript.text.strip():
            logger.debug(
                "VoiceServer: empty transcript from node %s — skipping agent call.",
                node.node_id,
            )
            return

        # ------------------------------------------------------------------
        # Agent callback.
        # ------------------------------------------------------------------
        metadata: dict[str, Any] = {
            "room": node.room,
            "node_id": node.node_id,
            "hardware_profile": node.hardware_profile,
            "confidence": transcript.confidence,
            "language": transcript.language,
        }
        try:
            response_text: str = await self._agent_callback(
                transcript.text, node.node_id, metadata
            )
        except Exception:
            logger.error(
                "VoiceServer: agent_callback failed for node %s", node.node_id, exc_info=True
            )
            await self._send_json(
                websocket,
                {"type": "error", "message": "Agent processing failed"},
            )
            return

        # Always send the text response.
        await self._send_json(websocket, {"type": "response_text", "text": response_text})

        # ------------------------------------------------------------------
        # Text-to-speech + audio streaming.
        # ------------------------------------------------------------------
        try:
            audio_buf = await self._tts.synthesize(response_text)
        except Exception:
            logger.error(
                "VoiceServer: TTS failed for node %s — sending text only.", node.node_id, exc_info=True
            )
            # response_text already sent above; skip audio streaming gracefully.
            return

        await self._send_json(
            websocket,
            {
                "type": "audio_start",
                "sample_rate": audio_buf.sample_rate,
                "format": audio_buf.format,
            },
        )

        # Stream audio in chunks.
        data = audio_buf.data
        chunk_size = self._audio_chunk_size
        offset = 0
        while offset < len(data):
            chunk = data[offset : offset + chunk_size]
            await websocket.send(chunk)
            offset += chunk_size

        await self._send_json(websocket, {"type": "audio_end"})

    # ------------------------------------------------------------------
    # Audio logging helper
    # ------------------------------------------------------------------

    async def _log_audio_to_disk(
        self,
        node: EdgeNode,
        audio_buffer: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        """Write a received audio buffer to the node's audio log directory.

        The file is named ``<unix_timestamp_ns>.pcm`` and written in a thread
        pool executor to avoid blocking the event loop.

        Args:
            node: The originating :class:`EdgeNode`.
            audio_buffer: Raw PCM bytes to persist.
            sample_rate: Sample rate for metadata purposes.
            channels: Channel count for metadata purposes.
        """
        log_dir = Path(node.audio_log_dir).expanduser()
        timestamp_ns = time.time_ns()
        filename = log_dir / f"{timestamp_ns}.pcm"

        loop = asyncio.get_event_loop()
        try:
            def _write() -> None:
                log_dir.mkdir(parents=True, exist_ok=True)
                filename.write_bytes(audio_buffer)

            await loop.run_in_executor(None, _write)
            _emit(
                session_id=node.node_id,
                event_type="voice.audio.logged",
                result="allow",
                detail={
                    "node_id": node.node_id,
                    "path": str(filename),
                    "bytes": len(audio_buffer),
                    "sample_rate": sample_rate,
                    "channels": channels,
                },
            )
            logger.debug("VoiceServer: audio logged to %s", filename)
        except Exception:
            logger.warning(
                "VoiceServer: failed to log audio for node %s", node.node_id, exc_info=True
            )

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _handle_heartbeat(
        self,
        websocket: WebSocketServerProtocol,
        node: EdgeNode,
        data: dict[str, Any],
    ) -> None:
        """Process a heartbeat frame from an authenticated node.

        Updates presence data in the :class:`~missy.channels.voice.presence.PresenceStore`
        and marks the node as online in the registry.

        Args:
            websocket: The authenticated connection (unused but present for
                symmetry with other handlers).
            node: The sending :class:`EdgeNode`.
            data: The decoded heartbeat JSON frame.
        """
        occupancy: bool | None = data.get("occupancy")
        noise_level: float | None = data.get("noise_level")
        wake_word_fp: bool = bool(data.get("wake_word_fp", False))

        try:
            self._presence_store.update(
                node.node_id,
                occupancy=occupancy,
                noise_level=noise_level,
                wake_word_fp=wake_word_fp,
            )
        except Exception:
            logger.debug(
                "VoiceServer: presence_store.update failed for node %s", node.node_id, exc_info=True
            )

        try:
            remote_addr = websocket.remote_address
            ip = str(remote_addr[0]) if remote_addr else node.ip_address
            self._registry.mark_online(node.node_id, ip_address=ip)
        except Exception:
            logger.debug(
                "VoiceServer: mark_online failed for node %s", node.node_id, exc_info=True
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _send_json(websocket: WebSocketServerProtocol, frame: dict[str, Any]) -> None:
        """Serialise *frame* to JSON and send it over *websocket*.

        Args:
            websocket: The target WebSocket connection.
            frame: The data to serialise and transmit.
        """
        await websocket.send(json.dumps(frame))

    # ------------------------------------------------------------------
    # Public status / introspection
    # ------------------------------------------------------------------

    def get_connected_nodes(self) -> list[str]:
        """Return the list of currently connected (authenticated) node IDs.

        Returns:
            A snapshot list of ``node_id`` strings for all nodes that have
            passed authentication and have not yet disconnected.
        """
        return list(self._connected_nodes)

    def get_status(self) -> dict[str, Any]:
        """Return a summary of the server's current operational state.

        Returns:
            A dict with keys:
            ``running``, ``connected_nodes``, ``host``, ``port``,
            ``stt_engine``, ``tts_engine``.
        """
        return {
            "running": self._running,
            "connected_nodes": len(self._connected_nodes),
            "host": self._host,
            "port": self._port,
            "stt_engine": self._stt.name,
            "tts_engine": self._tts.name,
        }
