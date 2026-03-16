"""ScreencastChannel — BaseChannel wrapper around ScreencastServer.

Follows the VoiceChannel pattern: starts the server in a dedicated daemon
thread with its own asyncio event loop, manages lifecycle of all
sub-components (token registry, session manager, analyzer, server).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import socket
import threading
from typing import Any

from missy.channels.base import BaseChannel, ChannelMessage
from missy.channels.screencast.analyzer import FrameAnalyzer
from missy.channels.screencast.auth import ScreencastTokenRegistry
from missy.channels.screencast.server import ScreencastServer
from missy.channels.screencast.session_manager import AnalysisResult, SessionManager

logger = logging.getLogger(__name__)


def _get_lan_ip() -> str:
    """Return the machine's LAN IP address, or ``127.0.0.1`` as fallback."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


class ScreencastChannel(BaseChannel):
    """Event-driven screencast channel backed by an HTTP+WebSocket server.

    Manages the full lifecycle: token registry, session manager, frame
    analyzer, and the combined HTTP/WS server.  The server runs in a daemon
    thread with a dedicated asyncio event loop.

    Args:
        host: Interface to bind on.  Defaults to ``"127.0.0.1"``.
        port: TCP port.  Defaults to ``8780``.
        max_sessions: Maximum concurrent screencast sessions.
        frame_save_dir: If set, save captured frames to this directory.
        vision_model: Override the default Ollama vision model.
        analysis_prompt: Custom prompt for the vision model.
        capture_url_base: Override the base URL shown to users (e.g. for
            reverse proxies).  If empty, computed from host/port.
    """

    name = "screencast"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8780,
        max_sessions: int = 20,
        frame_save_dir: str = "",
        vision_model: str = "",
        analysis_prompt: str = "",
        capture_url_base: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._max_sessions = max_sessions
        self._frame_save_dir = frame_save_dir
        self._vision_model = vision_model
        self._analysis_prompt = analysis_prompt
        self._capture_url_base = capture_url_base

        # Populated during start().
        self._token_registry: ScreencastTokenRegistry | None = None
        self._session_manager: SessionManager | None = None
        self._analyzer: FrameAnalyzer | None = None
        self._server: ScreencastServer | None = None

        # Thread / loop management.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Discord REST client reference for posting analysis results.
        self._discord_rest: Any = None

    # ------------------------------------------------------------------
    # BaseChannel interface
    # ------------------------------------------------------------------

    def receive(self) -> ChannelMessage | None:
        raise NotImplementedError(
            "ScreencastChannel is event-driven; use start() instead"
        )

    def send(self, message: str) -> None:
        raise NotImplementedError(
            "ScreencastChannel is event-driven; use start() instead"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise all subsystems and start the server."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("ScreencastChannel is already running.")

        # Token registry.
        token_registry = ScreencastTokenRegistry()
        self._token_registry = token_registry

        # Session manager.
        session_manager = SessionManager(max_sessions=self._max_sessions)
        self._session_manager = session_manager

        # Frame analyzer.
        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=token_registry,
            analysis_prompt=self._analysis_prompt,
            vision_model=self._vision_model,
            frame_save_dir=self._frame_save_dir,
            discord_callback=self._post_to_discord,
        )
        self._analyzer = analyzer

        # Server.
        server = ScreencastServer(
            token_registry=token_registry,
            session_manager=session_manager,
            host=self._host,
            port=self._port,
        )
        self._server = server

        # Start in a dedicated daemon thread.
        started_event = threading.Event()
        error_holder: list[BaseException] = []

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _runner() -> None:
                try:
                    await server.start()
                    await analyzer.start()
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
                logger.error(
                    "ScreencastChannel: event loop terminated with error.",
                    exc_info=True,
                )
            finally:
                loop.close()
                self._loop = None

        self._thread = threading.Thread(
            target=_run_loop,
            name="missy-screencast-server",
            daemon=True,
        )
        self._thread.start()

        # Wait until the server is ready (or has failed).
        started_event.wait(timeout=30)

        if error_holder:
            self._thread = None
            self._server = None
            raise RuntimeError(
                f"ScreencastChannel failed to start: {error_holder[0]}"
            ) from error_holder[0]

        logger.info(
            "ScreencastChannel: started on http://%s:%d",
            self._host,
            self._port,
        )

    def stop(self) -> None:
        """Stop the server and join the background thread.  Idempotent."""
        server = self._server
        analyzer = self._analyzer
        loop = self._loop
        thread = self._thread

        if server is None or loop is None or thread is None:
            logger.debug("ScreencastChannel.stop(): not running — no-op.")
            return

        future: concurrent.futures.Future[None] = concurrent.futures.Future()

        async def _stop_coro() -> None:
            try:
                if analyzer is not None:
                    await analyzer.stop()
                await server.stop()
            finally:
                future.set_result(None)

        asyncio.run_coroutine_threadsafe(_stop_coro(), loop)

        try:
            future.result(timeout=15)
        except Exception:
            logger.warning(
                "ScreencastChannel.stop(): stop timed out or raised.",
                exc_info=True,
            )

        thread.join(timeout=10)
        if thread.is_alive():
            logger.warning(
                "ScreencastChannel.stop(): background thread did not exit cleanly."
            )

        self._thread = None
        self._server = None
        self._loop = None
        logger.info("ScreencastChannel: stopped.")

    # ------------------------------------------------------------------
    # Public API (called from Discord commands)
    # ------------------------------------------------------------------

    def set_discord_rest(self, rest_client: Any) -> None:
        """Attach a Discord REST client for posting analysis results."""
        self._discord_rest = rest_client

    def create_session(
        self,
        *,
        created_by: str = "",
        discord_channel_id: str = "",
        label: str = "",
    ) -> tuple[str, str, str]:
        """Create a screencast session and return ``(session_id, token, share_url)``.

        The share URL includes the session_id and token as query parameters.
        """
        if self._token_registry is None:
            raise RuntimeError("ScreencastChannel is not running.")

        session_id, token = self._token_registry.create_session(
            created_by=created_by,
            discord_channel_id=discord_channel_id,
            label=label,
        )

        base = self._capture_url_base
        if not base:
            host = self._host
            if host in ("0.0.0.0", "::"):
                host = _get_lan_ip()
            tls = self._server is not None and getattr(self._server, "_tls_enabled", False)
            scheme = "https" if tls else "http"
            base = f"{scheme}://{host}:{self._port}"
        share_url = f"{base}/?session_id={session_id}&token={token}"

        return session_id, token, share_url

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a screencast session."""
        if self._token_registry is None:
            return False
        return self._token_registry.revoke_session(session_id)

    def get_active_sessions(self) -> list[dict[str, Any]]:
        """Return info about all active sessions."""
        if self._token_registry is None:
            return []
        sessions = self._token_registry.list_active()
        return [
            {
                "session_id": s.session_id,
                "label": s.label,
                "created_by": s.created_by,
                "created_at": s.created_at,
                "frame_count": s.frame_count,
                "analysis_count": s.analysis_count,
                "last_frame_at": s.last_frame_at,
            }
            for s in sessions
        ]

    def get_latest_analysis(self, session_id: str) -> AnalysisResult | None:
        """Return the latest analysis result for a session."""
        if self._session_manager is None:
            return None
        return self._session_manager.get_latest_result(session_id)

    def get_results(self, session_id: str, limit: int = 10) -> list[AnalysisResult]:
        """Return recent analysis results for a session."""
        if self._session_manager is None:
            return []
        return self._session_manager.get_results(session_id, limit)

    def get_status(self) -> dict[str, Any]:
        """Return server operational status."""
        if self._server is None:
            return {"running": False}
        return self._server.get_status()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post_to_discord(
        self,
        session_id: str,
        discord_channel_id: str,
        analysis_text: str,
    ) -> None:
        """Post an analysis result to the originating Discord channel."""
        rest = self._discord_rest
        if rest is None or not discord_channel_id:
            return

        label = ""
        if self._token_registry:
            s = self._token_registry.get_session(session_id)
            if s:
                label = s.label

        header = f"**Screen analysis{f' ({label})' if label else ''}:**\n"
        max_body = 2000 - len(header) - 10
        if len(analysis_text) > max_body:
            analysis_text = analysis_text[:max_body].rstrip() + "..."
        text = header + analysis_text

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, rest.send_message, discord_channel_id, text)
        except Exception:
            logger.debug(
                "ScreencastChannel: failed to post analysis to Discord channel %s",
                discord_channel_id,
                exc_info=True,
            )
