"""Background frame analysis pipeline for the screencast channel.

Runs as an asyncio task, dequeues frames from the session manager's queue,
sends them to the Ollama vision model via ``analyze_image_bytes()``, and
stores results back in the session manager.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from missy.channels.screencast.auth import ScreencastTokenRegistry
from missy.channels.screencast.session_manager import (
    AnalysisResult,
    FrameMetadata,
    SessionManager,
)
from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_TASK_ID = "screencast-analyzer"

_DEFAULT_PROMPT = (
    "Describe what you see on this screen. Note any important UI state, "
    "error messages, active applications, and content being displayed."
)


def _emit(
    session_id: str,
    event_type: str,
    result: str,
    detail: dict[str, Any] | None = None,
) -> None:
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
        logger.debug("screencast analyzer: audit emit failed for %r", event_type, exc_info=True)


class FrameAnalyzer:
    """Background vision model analysis pipeline.

    Args:
        session_manager: The session manager that provides the frame queue.
        token_registry: The token registry for updating frame stats.
        analysis_prompt: Custom prompt for the vision model.
        vision_model: Override the default vision model name.
        frame_save_dir: If set, save frames to this directory.
        discord_callback: Optional async callback to post results to Discord.
            Signature: ``async (session_id, channel_id, text) -> None``.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        token_registry: ScreencastTokenRegistry,
        *,
        analysis_prompt: str = "",
        vision_model: str = "",
        frame_save_dir: str = "",
        discord_callback: Callable[..., Any] | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._token_registry = token_registry
        self._prompt = analysis_prompt or _DEFAULT_PROMPT
        self._vision_model = vision_model
        self._frame_save_dir = frame_save_dir
        self._discord_callback = discord_callback
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background analysis task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="screencast-analyzer")
        logger.info("FrameAnalyzer: started.")

    async def stop(self) -> None:
        """Cancel and await the background task."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("FrameAnalyzer: stopped.")

    async def _run(self) -> None:
        """Main loop: dequeue frames and analyze them."""
        while self._running:
            try:
                metadata, data = await asyncio.wait_for(
                    self._session_manager.dequeue_frame(),
                    timeout=2.0,
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("FrameAnalyzer: dequeue error", exc_info=True)
                await asyncio.sleep(1.0)
                continue

            try:
                await self._process_frame(metadata, data)
            except Exception:
                logger.error(
                    "FrameAnalyzer: error processing frame %d for session %s",
                    metadata.frame_number,
                    metadata.session_id,
                    exc_info=True,
                )
                _emit(
                    metadata.session_id,
                    "screencast.analysis.failed",
                    "error",
                    {
                        "frame_number": metadata.frame_number,
                    },
                )

    async def _process_frame(self, metadata: FrameMetadata, data: bytes) -> None:
        """Analyze a single frame via the vision model."""
        start_ms = time.monotonic_ns() // 1_000_000

        # Optionally save the frame to disk.
        if self._frame_save_dir:
            await self._save_frame(metadata, data)

        # Run the synchronous vision model call in a thread executor.
        loop = asyncio.get_running_loop()
        analysis_text = await loop.run_in_executor(
            None,
            self._call_vision_model,
            data,
        )

        elapsed_ms = int((time.monotonic_ns() // 1_000_000) - start_ms)

        result = AnalysisResult(
            session_id=metadata.session_id,
            frame_number=metadata.frame_number,
            analysis_text=analysis_text,
            model=self._vision_model or "minicpm-v",
            processing_ms=elapsed_ms,
        )
        self._session_manager.store_result(result)

        # Update stats in the token registry.
        session = self._token_registry.get_session(metadata.session_id)
        if session is not None:
            self._token_registry.update_frame_stats(
                metadata.session_id,
                analysis_count=session.analysis_count + 1,
            )

        _emit(
            metadata.session_id,
            "screencast.analysis.completed",
            "allow",
            {
                "frame_number": metadata.frame_number,
                "processing_ms": elapsed_ms,
                "text_length": len(analysis_text),
            },
        )

        logger.info(
            "FrameAnalyzer: session=%s frame=%d analyzed in %dms (%d chars)",
            metadata.session_id,
            metadata.frame_number,
            elapsed_ms,
            len(analysis_text),
        )

        # Post to Discord if callback is set.
        if self._discord_callback is not None and session is not None:
            try:
                await self._discord_callback(
                    metadata.session_id,
                    session.discord_channel_id,
                    analysis_text,
                )
            except Exception:
                logger.debug("FrameAnalyzer: discord callback failed", exc_info=True)

    def _call_vision_model(self, image_data: bytes) -> str:
        """Synchronous call to the Ollama vision model."""
        from missy.channels.discord.image_analyze import analyze_image_bytes

        return analyze_image_bytes(image_data, self._prompt, timeout=120)

    async def _save_frame(self, metadata: FrameMetadata, data: bytes) -> None:
        """Save a frame to disk atomically."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save_frame_sync, metadata, data)

    def _save_frame_sync(self, metadata: FrameMetadata, data: bytes) -> None:
        """Write frame bytes to ``{frame_save_dir}/{session_id}/{ts}_{seq}.jpg``."""
        session_dir = os.path.join(self._frame_save_dir, metadata.session_id)
        os.makedirs(session_dir, exist_ok=True, mode=0o700)

        ts = time.strftime("%Y%m%d_%H%M%S")
        ext = "png" if metadata.format == "png" else "jpg"
        filename = f"{ts}_{metadata.frame_number:06d}.{ext}"
        path = os.path.join(session_dir, filename)

        # Atomic write via temp file + rename.
        tmp_path = path + ".tmp"
        try:
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            os.rename(tmp_path, path)
        except Exception:
            # Clean up partial write.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        _emit(
            metadata.session_id,
            "screencast.frame.saved",
            "allow",
            {"path": path, "size_bytes": len(data)},
        )
        logger.debug("FrameAnalyzer: saved frame to %s", path)
