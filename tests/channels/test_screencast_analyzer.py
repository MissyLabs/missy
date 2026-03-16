"""Tests for the screencast frame analyzer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.channels.screencast.analyzer import FrameAnalyzer
from missy.channels.screencast.auth import ScreencastTokenRegistry
from missy.channels.screencast.session_manager import (
    AnalysisResult,
    FrameMetadata,
    SessionManager,
)


@pytest.fixture
def registry() -> ScreencastTokenRegistry:
    return ScreencastTokenRegistry()


@pytest.fixture
def session_manager() -> SessionManager:
    sm = SessionManager()
    return sm


class TestFrameAnalyzer:
    """Tests for FrameAnalyzer."""

    @pytest.mark.asyncio
    async def test_start_stop(self, session_manager: SessionManager, registry: ScreencastTokenRegistry) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
        )
        await analyzer.start()
        assert analyzer._running is True
        assert analyzer._task is not None

        await analyzer.stop()
        assert analyzer._running is False
        assert analyzer._task is None

    @pytest.mark.asyncio
    async def test_processes_frame(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        """Test that the analyzer processes a queued frame."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        # Create a session in the registry.
        session_id, _token = registry.create_session(label="test")

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
        )

        # Mock the vision model call.
        with patch.object(analyzer, "_call_vision_model", return_value="Screen shows a terminal."):
            await analyzer.start()

            # Enqueue a frame.
            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            session_manager.enqueue_frame(meta, b"\xff\xd8\xff" + b"\x00" * 100)

            # Wait for processing.
            await asyncio.sleep(0.5)

            await analyzer.stop()

        # Check result was stored.
        result = session_manager.get_latest_result(session_id)
        assert result is not None
        assert result.analysis_text == "Screen shows a terminal."
        assert result.frame_number == 1

    @pytest.mark.asyncio
    async def test_error_does_not_crash(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        """Test that an error in vision model doesn't crash the analyzer."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        session_id, _ = registry.create_session()

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
        )

        with patch.object(analyzer, "_call_vision_model", side_effect=RuntimeError("model down")):
            await analyzer.start()

            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            session_manager.enqueue_frame(meta, b"\xff\xd8\xff" + b"\x00" * 100)

            await asyncio.sleep(0.5)

            # Analyzer should still be running.
            assert analyzer._running is True

            await analyzer.stop()

    @pytest.mark.asyncio
    async def test_discord_callback(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        """Test that analysis posts to Discord via callback."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        session_id, _ = registry.create_session(discord_channel_id="chan789")
        callback = AsyncMock()

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
            discord_callback=callback,
        )

        with patch.object(analyzer, "_call_vision_model", return_value="Desktop visible"):
            await analyzer.start()

            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            session_manager.enqueue_frame(meta, b"\xff\xd8\xff" + b"\x00" * 100)

            await asyncio.sleep(0.5)
            await analyzer.stop()

        callback.assert_called_once_with(session_id, "chan789", "Desktop visible")

    def test_call_vision_model(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        """Test that _call_vision_model delegates to analyze_image_bytes."""
        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
        )
        with patch(
            "missy.channels.discord.image_analyze.analyze_image_bytes",
            return_value="mocked result",
        ) as mock_fn:
            result = analyzer._call_vision_model(b"fake image data")
            assert result == "mocked result"
            mock_fn.assert_called_once()
