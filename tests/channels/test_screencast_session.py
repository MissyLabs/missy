"""Tests for the screencast session manager."""

from __future__ import annotations

import asyncio

import pytest

from missy.channels.screencast.session_manager import (
    AnalysisResult,
    FrameMetadata,
    SessionManager,
    SessionState,
)


class TestSessionManager:
    """Tests for SessionManager."""

    def test_register_and_unregister(self) -> None:
        sm = SessionManager(max_sessions=5)
        state = sm.register_connection("s1", "1.2.3.4:5678")
        assert isinstance(state, SessionState)
        assert state.session_id == "s1"
        assert sm.connection_count == 1

        sm.unregister_connection("s1")
        assert sm.connection_count == 0

    def test_at_capacity(self) -> None:
        sm = SessionManager(max_sessions=2)
        sm.register_connection("s1")
        assert sm.at_capacity is False
        sm.register_connection("s2")
        assert sm.at_capacity is True

    def test_get_connection(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1", "addr")
        state = sm.get_connection("s1")
        assert state is not None
        assert state.remote_address == "addr"

    def test_get_connection_missing(self) -> None:
        sm = SessionManager()
        assert sm.get_connection("nope") is None

    def test_store_and_get_results(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1")

        for i in range(5):
            sm.store_result(AnalysisResult(
                session_id="s1",
                frame_number=i,
                analysis_text=f"analysis {i}",
            ))

        results = sm.get_results("s1", limit=3)
        assert len(results) == 3
        assert results[-1].frame_number == 4

    def test_get_latest_result(self) -> None:
        sm = SessionManager()
        assert sm.get_latest_result("s1") is None

        sm.store_result(AnalysisResult(session_id="s1", frame_number=1, analysis_text="a"))
        sm.store_result(AnalysisResult(session_id="s1", frame_number=2, analysis_text="b"))

        latest = sm.get_latest_result("s1")
        assert latest is not None
        assert latest.frame_number == 2
        assert latest.analysis_text == "b"

    def test_results_bounded(self) -> None:
        sm = SessionManager()
        for i in range(60):
            sm.store_result(AnalysisResult(session_id="s1", frame_number=i))

        results = sm.get_results("s1", limit=100)
        assert len(results) == 50  # _MAX_RESULTS_PER_SESSION

    @pytest.mark.asyncio
    async def test_enqueue_dequeue_frame(self) -> None:
        sm = SessionManager()
        queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        sm.set_queue(queue)

        meta = FrameMetadata(session_id="s1", frame_number=1, format="jpeg")
        data = b"\xff\xd8\xff" + b"\x00" * 100

        ok = sm.enqueue_frame(meta, data)
        assert ok is True

        got_meta, got_data = await sm.dequeue_frame()
        assert got_meta.session_id == "s1"
        assert got_data == data

    def test_enqueue_no_queue(self) -> None:
        sm = SessionManager()
        meta = FrameMetadata(session_id="s1", frame_number=1, format="jpeg")
        assert sm.enqueue_frame(meta, b"data") is False

    @pytest.mark.asyncio
    async def test_enqueue_full_returns_false(self) -> None:
        sm = SessionManager()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        sm.set_queue(queue)

        meta = FrameMetadata(session_id="s1", frame_number=1, format="jpeg")
        assert sm.enqueue_frame(meta, b"a") is True
        assert sm.enqueue_frame(meta, b"b") is True
        assert sm.enqueue_frame(meta, b"c") is False

    def test_get_status(self) -> None:
        sm = SessionManager(max_sessions=10)
        sm.register_connection("s1", "1.2.3.4:1234")
        status = sm.get_status()
        assert status["connected_sessions"] == 1
        assert status["max_sessions"] == 10
        assert "s1" in status["sessions"]

    def test_unregister_preserves_results(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1")
        sm.store_result(AnalysisResult(session_id="s1", frame_number=1, analysis_text="x"))
        sm.unregister_connection("s1")

        # Results are still accessible after disconnect.
        assert sm.get_latest_result("s1") is not None
