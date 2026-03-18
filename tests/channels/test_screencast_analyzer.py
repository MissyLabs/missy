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


# ---------------------------------------------------------------------------
# _emit exception handling (lines 53-54)
# ---------------------------------------------------------------------------

class TestAnalyzerEmitException:
    """Test that analyzer _emit swallows exceptions from event_bus.publish."""

    def test_emit_swallows_exception(self) -> None:
        from missy.channels.screencast import analyzer as az_mod

        with patch.object(az_mod.event_bus, "publish", side_effect=RuntimeError("bus down")):
            # Should not raise.
            az_mod._emit("sess-x", "screencast.test", "allow", {"k": "v"})


# ---------------------------------------------------------------------------
# Dequeue timeout loop (line 116) and error in dequeue (lines 119-122)
# ---------------------------------------------------------------------------

class TestAnalyzerDequeueEdgeCases:
    """Test the _run loop's dequeue error and timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_in_dequeue_continues(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        """TimeoutError from dequeue_frame is silently re-looped."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
        )

        # Raise TimeoutError once then CancelledError to exit.
        call_count = 0

        async def _fake_dequeue():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError()
            raise asyncio.CancelledError()

        with patch.object(session_manager, "dequeue_frame", side_effect=_fake_dequeue):
            await analyzer.start()
            await asyncio.sleep(0.2)
            await analyzer.stop()

        # The loop handled TimeoutError without crashing.
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_generic_exception_in_dequeue_continues(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        """An unexpected exception during dequeue_frame is logged and the loop continues."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
        )

        call_count = 0

        async def _fake_dequeue():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("dequeue failure")
            # Let the task end naturally after the sleep in the error path.
            raise asyncio.CancelledError()

        with patch.object(session_manager, "dequeue_frame", side_effect=_fake_dequeue):
            await analyzer.start()
            # Give the background task time to call dequeue_frame at least once.
            for _ in range(20):
                await asyncio.sleep(0.05)
                if call_count >= 1:
                    break
            await analyzer.stop()

        assert call_count >= 1


# ---------------------------------------------------------------------------
# Frame save when frame_save_dir is set (line 148)
# ---------------------------------------------------------------------------

class TestAnalyzerFrameSave:
    """Test that frames are saved to disk when frame_save_dir is configured."""

    @pytest.mark.asyncio
    async def test_frame_saved_to_directory(
        self,
        session_manager: SessionManager,
        registry: ScreencastTokenRegistry,
        tmp_path,
    ) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        session_id, _ = registry.create_session(label="save-test")

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
            frame_save_dir=str(tmp_path),
        )

        with patch.object(analyzer, "_call_vision_model", return_value="saved frame analysis"):
            await analyzer.start()

            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            session_manager.enqueue_frame(meta, b"\xff\xd8\xff" + b"\x00" * 50)

            await asyncio.sleep(0.5)
            await analyzer.stop()

        session_dir = tmp_path / session_id
        assert session_dir.exists()
        saved_files = list(session_dir.glob("*.jpg"))
        assert len(saved_files) == 1

    @pytest.mark.asyncio
    async def test_png_frame_saved_with_png_extension(
        self,
        session_manager: SessionManager,
        registry: ScreencastTokenRegistry,
        tmp_path,
    ) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        session_id, _ = registry.create_session(label="png-save-test")

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
            frame_save_dir=str(tmp_path),
        )

        with patch.object(analyzer, "_call_vision_model", return_value="png analysis"):
            await analyzer.start()

            meta = FrameMetadata(session_id=session_id, frame_number=1, format="png")
            session_manager.enqueue_frame(meta, b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

            await asyncio.sleep(0.5)
            await analyzer.stop()

        session_dir = tmp_path / session_id
        saved_files = list(session_dir.glob("*.png"))
        assert len(saved_files) == 1


# ---------------------------------------------------------------------------
# Discord callback failure (lines 204-205)
# ---------------------------------------------------------------------------

class TestAnalyzerDiscordCallbackFailure:
    """Test that a failing discord_callback is caught and does not crash the analyzer."""

    @pytest.mark.asyncio
    async def test_discord_callback_failure_is_swallowed(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry
    ) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        session_manager.set_queue(queue)

        session_id, _ = registry.create_session(discord_channel_id="chan-fail")

        async def _failing_callback(sid, channel_id, text):
            raise RuntimeError("discord exploded")

        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
            discord_callback=_failing_callback,
        )

        with patch.object(analyzer, "_call_vision_model", return_value="result text"):
            await analyzer.start()

            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            session_manager.enqueue_frame(meta, b"\xff\xd8\xff" + b"\x00" * 50)

            await asyncio.sleep(0.5)

            # Analyzer must still be running after the callback failure.
            assert analyzer._running is True
            await analyzer.stop()


# ---------------------------------------------------------------------------
# _save_frame_sync failure cleanup (lines 237-241)
# ---------------------------------------------------------------------------

class TestSaveFrameSyncFailure:
    """Test that _save_frame_sync cleans up tmp file on write failure."""

    def test_cleanup_on_write_failure(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry, tmp_path
    ) -> None:
        """If os.write fails, the .tmp file is removed and the exception re-raised."""
        import os

        session_id, _ = registry.create_session()
        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
            frame_save_dir=str(tmp_path),
        )

        meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
        data = b"\xff\xd8\xff" + b"\x00" * 50

        original_write = os.write

        def _failing_write(fd, buf):
            original_write(fd, b"")  # write nothing, then raise
            raise OSError("disk full")

        with patch("os.write", side_effect=_failing_write):
            with pytest.raises(OSError, match="disk full"):
                analyzer._save_frame_sync(meta, data)

        # No .tmp file should remain.
        session_dir = tmp_path / session_id
        tmp_files = list(session_dir.glob("*.tmp"))
        assert tmp_files == []

    def test_cleanup_on_open_failure(
        self, session_manager: SessionManager, registry: ScreencastTokenRegistry, tmp_path
    ) -> None:
        """If os.open fails, the exception propagates (no tmp file created)."""
        import os

        session_id, _ = registry.create_session()
        analyzer = FrameAnalyzer(
            session_manager=session_manager,
            token_registry=registry,
            frame_save_dir=str(tmp_path),
        )

        meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
        data = b"\xff\xd8\xff" + b"\x00" * 50

        with patch("os.open", side_effect=PermissionError("no permission")):
            with pytest.raises(PermissionError):
                analyzer._save_frame_sync(meta, data)
