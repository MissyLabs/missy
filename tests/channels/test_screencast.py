"""Comprehensive tests for the screencast channel modules.

Covers:
- missy/channels/screencast/auth.py          (ScreencastTokenRegistry, ScreencastSession)
- missy/channels/screencast/session_manager.py (SessionManager, FrameMetadata,
                                                AnalysisResult, SessionState)
- missy/channels/screencast/server.py         (ScreencastServer)
- missy/channels/screencast/channel.py        (ScreencastChannel)
- missy/channels/screencast/analyzer.py       (FrameAnalyzer)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
import threading
import time
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from missy.channels.screencast.auth import ScreencastSession, ScreencastTokenRegistry
from missy.channels.screencast.analyzer import FrameAnalyzer, _DEFAULT_PROMPT
from missy.channels.screencast.channel import ScreencastChannel, _get_lan_ip
from missy.channels.screencast.server import (
    ScreencastServer,
    _JPEG_MAGIC,
    _MAX_CONCURRENT_CONNECTIONS,
    _MAX_FRAME_BYTES,
    _MIN_DIMENSION,
    _MAX_DIMENSION,
    _MIN_FRAME_INTERVAL,
    _PNG_MAGIC,
)
from missy.channels.screencast.session_manager import (
    AnalysisResult,
    FrameMetadata,
    SessionManager,
    SessionState,
    _MAX_RESULTS_PER_SESSION,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ScreencastTokenRegistry:
    return ScreencastTokenRegistry()


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(max_sessions=5)


@pytest.fixture
def server(registry: ScreencastTokenRegistry, session_manager: SessionManager) -> ScreencastServer:
    return ScreencastServer(
        token_registry=registry,
        session_manager=session_manager,
        host="127.0.0.1",
        port=0,
    )


# ---------------------------------------------------------------------------
# 1. ScreencastTokenRegistry (auth.py)
# ---------------------------------------------------------------------------


class TestScreencastSessionDataclass:
    """ScreencastSession dataclass construction and field defaults."""

    def test_required_fields(self) -> None:
        s = ScreencastSession(session_id="sid", token_hash="hash")
        assert s.session_id == "sid"
        assert s.token_hash == "hash"

    def test_defaults(self) -> None:
        s = ScreencastSession(session_id="s", token_hash="h")
        assert s.created_by == ""
        assert s.discord_channel_id == ""
        assert s.label == ""
        assert s.active is True
        assert s.last_frame_at == 0.0
        assert s.frame_count == 0
        assert s.analysis_count == 0

    def test_created_at_auto_populated(self) -> None:
        before = time.time()
        s = ScreencastSession(session_id="s", token_hash="h")
        after = time.time()
        assert before <= s.created_at <= after

    def test_explicit_overrides(self) -> None:
        s = ScreencastSession(
            session_id="s",
            token_hash="h",
            created_by="u1",
            discord_channel_id="c1",
            label="lbl",
            active=False,
            last_frame_at=1.0,
            frame_count=5,
            analysis_count=3,
        )
        assert s.created_by == "u1"
        assert s.discord_channel_id == "c1"
        assert s.label == "lbl"
        assert s.active is False
        assert s.last_frame_at == 1.0
        assert s.frame_count == 5
        assert s.analysis_count == 3


class TestScreencastTokenRegistryCreate:
    """create_session return values, uniqueness, and metadata storage."""

    def test_returns_string_session_id_and_token(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        assert isinstance(session_id, str)
        assert isinstance(token, str)

    def test_session_id_is_urlsafe(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        # URL-safe base64 chars only (letters, digits, -, _)
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_=" for c in session_id)

    def test_token_length_sufficient(self) -> None:
        reg = ScreencastTokenRegistry()
        _, token = reg.create_session()
        assert len(token) >= 20

    def test_session_id_length_sufficient(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        assert len(session_id) >= 10

    def test_successive_sessions_are_unique(self) -> None:
        reg = ScreencastTokenRegistry()
        ids = set()
        tokens = set()
        for _ in range(10):
            sid, tok = reg.create_session()
            ids.add(sid)
            tokens.add(tok)
        assert len(ids) == 10
        assert len(tokens) == 10

    def test_metadata_stored(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session(
            created_by="user42",
            discord_channel_id="chan99",
            label="my-session",
        )
        s = reg.get_session(session_id)
        assert s is not None
        assert s.created_by == "user42"
        assert s.discord_channel_id == "chan99"
        assert s.label == "my-session"
        assert s.active is True

    def test_token_not_stored_in_plaintext(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        s = reg.get_session(session_id)
        assert s is not None
        # The plaintext token must never appear in the stored hash field.
        assert token not in s.token_hash


class TestScreencastTokenRegistryVerify:
    """verify_token — correct, incorrect, revoked, missing."""

    def test_correct_token_returns_true(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        assert reg.verify_token(session_id, token) is True

    def test_wrong_token_returns_false(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        assert reg.verify_token(session_id, "wrong-token-value") is False

    def test_wrong_session_id_returns_false(self) -> None:
        reg = ScreencastTokenRegistry()
        _, token = reg.create_session()
        assert reg.verify_token("no-such-session", token) is False

    def test_empty_token_returns_false(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        assert reg.verify_token(session_id, "") is False

    def test_revoked_session_returns_false(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        reg.revoke_session(session_id)
        assert reg.verify_token(session_id, token) is False

    def test_cross_session_token_fails(self) -> None:
        """Token from session A must not verify against session B."""
        reg = ScreencastTokenRegistry()
        s1, t1 = reg.create_session()
        s2, _t2 = reg.create_session()
        assert reg.verify_token(s2, t1) is False


class TestScreencastTokenRegistryGetSession:
    """get_session — found, missing."""

    def test_get_existing_session(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session(label="lbl")
        s = reg.get_session(session_id)
        assert s is not None
        assert s.session_id == session_id

    def test_get_missing_session_returns_none(self) -> None:
        reg = ScreencastTokenRegistry()
        assert reg.get_session("does-not-exist") is None


class TestScreencastTokenRegistryListActive:
    """list_active — filters inactive sessions."""

    def test_all_active_returned(self) -> None:
        reg = ScreencastTokenRegistry()
        s1, _ = reg.create_session()
        s2, _ = reg.create_session()
        active_ids = {s.session_id for s in reg.list_active()}
        assert s1 in active_ids
        assert s2 in active_ids

    def test_revoked_excluded(self) -> None:
        reg = ScreencastTokenRegistry()
        s1, _ = reg.create_session()
        s2, _ = reg.create_session()
        reg.revoke_session(s1)
        active_ids = {s.session_id for s in reg.list_active()}
        assert s1 not in active_ids
        assert s2 in active_ids

    def test_empty_registry_returns_empty_list(self) -> None:
        reg = ScreencastTokenRegistry()
        assert reg.list_active() == []

    def test_all_revoked_returns_empty_list(self) -> None:
        reg = ScreencastTokenRegistry()
        s1, _ = reg.create_session()
        reg.revoke_session(s1)
        assert reg.list_active() == []


class TestScreencastTokenRegistryRevoke:
    """revoke_session — return value and state mutation."""

    def test_revoke_existing_returns_true(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        assert reg.revoke_session(session_id) is True

    def test_revoke_marks_inactive(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.revoke_session(session_id)
        s = reg.get_session(session_id)
        assert s is not None
        assert s.active is False

    def test_revoke_nonexistent_returns_false(self) -> None:
        reg = ScreencastTokenRegistry()
        assert reg.revoke_session("ghost") is False

    def test_double_revoke_returns_true_then_true(self) -> None:
        """Revoking an already-revoked session still returns True (it exists)."""
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        assert reg.revoke_session(session_id) is True
        assert reg.revoke_session(session_id) is True

    def test_revoke_does_not_delete_session(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.revoke_session(session_id)
        # Session still retrievable, just inactive.
        assert reg.get_session(session_id) is not None


class TestScreencastTokenRegistryUpdateFrameStats:
    """update_frame_stats — counter updates and last_frame_at refresh."""

    def test_updates_frame_count(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.update_frame_stats(session_id, frame_count=7)
        s = reg.get_session(session_id)
        assert s is not None
        assert s.frame_count == 7

    def test_updates_analysis_count(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.update_frame_stats(session_id, analysis_count=3)
        s = reg.get_session(session_id)
        assert s is not None
        assert s.analysis_count == 3

    def test_updates_both_counters(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.update_frame_stats(session_id, frame_count=10, analysis_count=5)
        s = reg.get_session(session_id)
        assert s is not None
        assert s.frame_count == 10
        assert s.analysis_count == 5

    def test_last_frame_at_updated(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        before = time.time()
        reg.update_frame_stats(session_id, frame_count=1)
        s = reg.get_session(session_id)
        assert s is not None
        assert s.last_frame_at >= before

    def test_partial_update_leaves_other_field_unchanged(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.update_frame_stats(session_id, frame_count=4)
        reg.update_frame_stats(session_id, analysis_count=2)
        s = reg.get_session(session_id)
        assert s is not None
        assert s.frame_count == 4
        assert s.analysis_count == 2

    def test_nonexistent_session_noop(self) -> None:
        reg = ScreencastTokenRegistry()
        # Must not raise.
        reg.update_frame_stats("ghost", frame_count=1, analysis_count=1)


class TestScreencastTokenRegistryHashToken:
    """_hash_token — determinism and salt sensitivity."""

    def test_deterministic(self) -> None:
        h1 = ScreencastTokenRegistry._hash_token("sid", "tok")
        h2 = ScreencastTokenRegistry._hash_token("sid", "tok")
        assert h1 == h2

    def test_returns_hex_string(self) -> None:
        h = ScreencastTokenRegistry._hash_token("sid", "tok")
        # A valid hex string: only 0-9 a-f chars.
        int(h, 16)  # Raises ValueError if not hex.

    def test_different_tokens_differ(self) -> None:
        h1 = ScreencastTokenRegistry._hash_token("sid", "token-a")
        h2 = ScreencastTokenRegistry._hash_token("sid", "token-b")
        assert h1 != h2

    def test_different_salts_differ(self) -> None:
        h1 = ScreencastTokenRegistry._hash_token("salt-1", "tok")
        h2 = ScreencastTokenRegistry._hash_token("salt-2", "tok")
        assert h1 != h2

    def test_empty_token_produces_hash(self) -> None:
        # Should not raise.
        h = ScreencastTokenRegistry._hash_token("sid", "")
        assert len(h) > 0


class TestScreencastTokenRegistryConcurrency:
    """Thread-safety of create_session."""

    def test_concurrent_create_session_no_duplicates(self) -> None:
        reg = ScreencastTokenRegistry()
        results: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _create() -> None:
            try:
                session_id, _ = reg.create_session()
                with lock:
                    results.append(session_id)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_create) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Unexpected errors: {errors}"
        assert len(results) == 50
        # All session IDs must be unique.
        assert len(set(results)) == 50

    def test_concurrent_verify_and_revoke_safe(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        errors: list[Exception] = []
        lock = threading.Lock()

        def _verify() -> None:
            try:
                reg.verify_token(session_id, token)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        def _revoke() -> None:
            try:
                reg.revoke_session(session_id)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_verify) for _ in range(20)]
        threads += [threading.Thread(target=_revoke) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# 2. SessionManager (session_manager.py)
# ---------------------------------------------------------------------------


class TestFrameMetadataDataclass:
    """FrameMetadata construction and defaults."""

    def test_required_fields(self) -> None:
        m = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
        assert m.session_id == "s"
        assert m.frame_number == 1
        assert m.format == "jpeg"

    def test_defaults(self) -> None:
        m = FrameMetadata(session_id="s", frame_number=0, format="png")
        assert m.width == 0
        assert m.height == 0
        assert m.size_bytes == 0

    def test_timestamp_auto_set(self) -> None:
        before = time.time()
        m = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
        after = time.time()
        assert before <= m.timestamp <= after

    def test_explicit_dimensions(self) -> None:
        m = FrameMetadata(session_id="s", frame_number=1, format="jpeg", width=1920, height=1080)
        assert m.width == 1920
        assert m.height == 1080


class TestAnalysisResultDataclass:
    """AnalysisResult construction and defaults."""

    def test_required_fields(self) -> None:
        r = AnalysisResult(session_id="s", frame_number=2)
        assert r.session_id == "s"
        assert r.frame_number == 2

    def test_defaults(self) -> None:
        r = AnalysisResult(session_id="s", frame_number=0)
        assert r.analysis_text == ""
        assert r.model == ""
        assert r.processing_ms == 0

    def test_timestamp_auto_set(self) -> None:
        before = time.time()
        r = AnalysisResult(session_id="s", frame_number=1)
        after = time.time()
        assert before <= r.timestamp <= after

    def test_explicit_fields(self) -> None:
        r = AnalysisResult(
            session_id="s",
            frame_number=5,
            analysis_text="hello",
            model="minicpm-v",
            processing_ms=150,
        )
        assert r.analysis_text == "hello"
        assert r.model == "minicpm-v"
        assert r.processing_ms == 150


class TestSessionStateDataclass:
    """SessionState construction and defaults."""

    def test_required_fields(self) -> None:
        st = SessionState(session_id="s")
        assert st.session_id == "s"

    def test_defaults(self) -> None:
        st = SessionState(session_id="s")
        assert st.frame_count == 0
        assert st.capture_interval_ms == 10000
        assert st.remote_address == ""

    def test_connected_at_auto_set(self) -> None:
        before = time.time()
        st = SessionState(session_id="s")
        after = time.time()
        assert before <= st.connected_at <= after


class TestSessionManagerRegisterUnregister:
    """register_connection / unregister_connection basics."""

    def test_register_returns_session_state(self) -> None:
        sm = SessionManager()
        state = sm.register_connection("s1", "1.2.3.4:5678")
        assert isinstance(state, SessionState)
        assert state.session_id == "s1"
        assert state.remote_address == "1.2.3.4:5678"

    def test_connection_count_increments(self) -> None:
        sm = SessionManager()
        assert sm.connection_count == 0
        sm.register_connection("s1")
        assert sm.connection_count == 1
        sm.register_connection("s2")
        assert sm.connection_count == 2

    def test_unregister_decrements_count(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1")
        sm.unregister_connection("s1")
        assert sm.connection_count == 0

    def test_unregister_nonexistent_is_noop(self) -> None:
        sm = SessionManager()
        sm.unregister_connection("ghost")  # Must not raise.
        assert sm.connection_count == 0

    def test_register_initialises_results_deque(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1")
        # After registration the deque should exist even before any results.
        assert sm.get_results("s1") == []

    def test_register_same_session_twice_overwrites(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1", "addr-a")
        sm.register_connection("s1", "addr-b")
        state = sm.get_connection("s1")
        assert state is not None
        assert state.remote_address == "addr-b"
        assert sm.connection_count == 1


class TestSessionManagerGetConnection:
    """get_connection — present and missing."""

    def test_get_existing(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1", "host:1234")
        state = sm.get_connection("s1")
        assert state is not None
        assert state.remote_address == "host:1234"

    def test_get_missing_returns_none(self) -> None:
        sm = SessionManager()
        assert sm.get_connection("none") is None


class TestSessionManagerCapacity:
    """at_capacity and connection_count properties."""

    def test_not_at_capacity_initially(self) -> None:
        sm = SessionManager(max_sessions=3)
        assert sm.at_capacity is False

    def test_at_capacity_when_full(self) -> None:
        sm = SessionManager(max_sessions=2)
        sm.register_connection("s1")
        assert sm.at_capacity is False
        sm.register_connection("s2")
        assert sm.at_capacity is True

    def test_capacity_restores_after_unregister(self) -> None:
        sm = SessionManager(max_sessions=1)
        sm.register_connection("s1")
        assert sm.at_capacity is True
        sm.unregister_connection("s1")
        assert sm.at_capacity is False

    def test_connection_count_matches_registered(self) -> None:
        sm = SessionManager(max_sessions=10)
        for i in range(7):
            sm.register_connection(f"s{i}")
        assert sm.connection_count == 7


class TestSessionManagerQueue:
    """set_queue / enqueue_frame / dequeue_frame."""

    def test_set_queue_attaches_queue(self) -> None:
        sm = SessionManager()
        q: asyncio.Queue = asyncio.Queue()
        sm.set_queue(q)
        assert sm.queue is q

    def test_enqueue_without_queue_returns_false(self) -> None:
        sm = SessionManager()
        meta = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
        assert sm.enqueue_frame(meta, b"data") is False

    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue(self) -> None:
        sm = SessionManager()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        sm.set_queue(q)

        meta = FrameMetadata(session_id="s1", frame_number=1, format="jpeg")
        data = _JPEG_MAGIC + b"\x00" * 50

        assert sm.enqueue_frame(meta, data) is True
        got_meta, got_data = await sm.dequeue_frame()
        assert got_meta.session_id == "s1"
        assert got_data == data

    @pytest.mark.asyncio
    async def test_dequeue_blocks_until_item_available(self) -> None:
        sm = SessionManager()
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        sm.set_queue(q)

        meta = FrameMetadata(session_id="s1", frame_number=1, format="jpeg")
        data = b"payload"

        # Schedule enqueue after a short delay.
        async def _enqueue_later() -> None:
            await asyncio.sleep(0.05)
            sm.enqueue_frame(meta, data)

        asyncio.create_task(_enqueue_later())
        got_meta, got_data = await asyncio.wait_for(sm.dequeue_frame(), timeout=2.0)
        assert got_meta.frame_number == 1
        assert got_data == data

    @pytest.mark.asyncio
    async def test_enqueue_when_full_returns_false(self) -> None:
        sm = SessionManager()
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        sm.set_queue(q)

        meta = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
        assert sm.enqueue_frame(meta, b"a") is True
        assert sm.enqueue_frame(meta, b"b") is True
        # Queue is now full — must return False without raising.
        assert sm.enqueue_frame(meta, b"c") is False

    @pytest.mark.asyncio
    async def test_dequeue_without_queue_raises_runtime_error(self) -> None:
        sm = SessionManager()
        with pytest.raises(RuntimeError, match="Queue not initialized"):
            await sm.dequeue_frame()


class TestSessionManagerResults:
    """store_result / get_results / get_latest_result / bounded deque."""

    def test_store_and_retrieve_result(self) -> None:
        sm = SessionManager()
        r = AnalysisResult(session_id="s1", frame_number=1, analysis_text="text")
        sm.store_result(r)
        results = sm.get_results("s1")
        assert len(results) == 1
        assert results[0].analysis_text == "text"

    def test_get_results_respects_limit(self) -> None:
        sm = SessionManager()
        for i in range(20):
            sm.store_result(AnalysisResult(session_id="s1", frame_number=i))
        results = sm.get_results("s1", limit=5)
        assert len(results) == 5

    def test_get_results_returns_most_recent(self) -> None:
        sm = SessionManager()
        for i in range(10):
            sm.store_result(AnalysisResult(session_id="s1", frame_number=i))
        results = sm.get_results("s1", limit=3)
        frame_numbers = [r.frame_number for r in results]
        assert frame_numbers == [7, 8, 9]

    def test_get_results_empty_session_returns_empty_list(self) -> None:
        sm = SessionManager()
        assert sm.get_results("no-such-session") == []

    def test_get_latest_result_returns_last(self) -> None:
        sm = SessionManager()
        sm.store_result(AnalysisResult(session_id="s1", frame_number=1, analysis_text="first"))
        sm.store_result(AnalysisResult(session_id="s1", frame_number=2, analysis_text="second"))
        latest = sm.get_latest_result("s1")
        assert latest is not None
        assert latest.analysis_text == "second"

    def test_get_latest_result_no_results_returns_none(self) -> None:
        sm = SessionManager()
        assert sm.get_latest_result("ghost") is None

    def test_results_bounded_by_max(self) -> None:
        sm = SessionManager()
        # Store more than the maximum.
        for i in range(_MAX_RESULTS_PER_SESSION + 10):
            sm.store_result(AnalysisResult(session_id="s1", frame_number=i))
        results = sm.get_results("s1", limit=1000)
        assert len(results) == _MAX_RESULTS_PER_SESSION

    def test_store_result_without_prior_registration(self) -> None:
        """store_result should create the deque on-demand if needed."""
        sm = SessionManager()
        sm.store_result(AnalysisResult(session_id="s-new", frame_number=1, analysis_text="x"))
        assert sm.get_latest_result("s-new") is not None

    def test_unregister_preserves_results(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1")
        sm.store_result(AnalysisResult(session_id="s1", frame_number=1, analysis_text="kept"))
        sm.unregister_connection("s1")
        assert sm.get_latest_result("s1") is not None


class TestSessionManagerGetStatus:
    """get_status summary dict."""

    def test_status_keys_present(self) -> None:
        sm = SessionManager(max_sessions=10)
        status = sm.get_status()
        assert "connected_sessions" in status
        assert "max_sessions" in status
        assert "queue_size" in status
        assert "sessions" in status

    def test_status_reflects_max_sessions(self) -> None:
        sm = SessionManager(max_sessions=7)
        assert sm.get_status()["max_sessions"] == 7

    def test_status_counts_connections(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1")
        sm.register_connection("s2")
        assert sm.get_status()["connected_sessions"] == 2

    def test_status_queue_size_zero_when_no_queue(self) -> None:
        sm = SessionManager()
        assert sm.get_status()["queue_size"] == 0

    @pytest.mark.asyncio
    async def test_status_queue_size_reflects_queued_frames(self) -> None:
        sm = SessionManager()
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        sm.set_queue(q)

        meta = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
        sm.enqueue_frame(meta, b"data1")
        sm.enqueue_frame(meta, b"data2")

        assert sm.get_status()["queue_size"] == 2

    def test_status_sessions_contains_connection_info(self) -> None:
        sm = SessionManager()
        sm.register_connection("s1", "192.168.1.1:5000")
        status = sm.get_status()
        assert "s1" in status["sessions"]
        assert status["sessions"]["s1"]["remote_address"] == "192.168.1.1:5000"


# ---------------------------------------------------------------------------
# 3. ScreencastServer (server.py) — _validate_image_magic and get_status
# ---------------------------------------------------------------------------


class TestValidateImageMagic:
    """_validate_image_magic static method."""

    def test_jpeg_magic_valid(self) -> None:
        data = _JPEG_MAGIC + b"\x00" * 100
        assert ScreencastServer._validate_image_magic(data) is True

    def test_png_magic_valid(self) -> None:
        data = _PNG_MAGIC + b"\x00" * 100
        assert ScreencastServer._validate_image_magic(data) is True

    def test_gif_invalid(self) -> None:
        assert ScreencastServer._validate_image_magic(b"GIF89a\x00" * 5) is False

    def test_bmp_invalid(self) -> None:
        assert ScreencastServer._validate_image_magic(b"BM\x00" * 10) is False

    def test_all_zeros_invalid(self) -> None:
        assert ScreencastServer._validate_image_magic(b"\x00" * 20) is False

    def test_empty_bytes_invalid(self) -> None:
        assert ScreencastServer._validate_image_magic(b"") is False

    def test_too_short_jpeg_start_invalid(self) -> None:
        # Only first two bytes of JPEG magic — less than 8 bytes total.
        assert ScreencastServer._validate_image_magic(b"\xff\xd8") is False

    def test_exactly_8_bytes_png_valid(self) -> None:
        assert ScreencastServer._validate_image_magic(_PNG_MAGIC) is True

    def test_exactly_3_bytes_jpeg_valid(self) -> None:
        # 3 bytes == _JPEG_MAGIC; but total len == 3 < 8 — must still be False
        # per the implementation guard (len < 8 → False).
        assert ScreencastServer._validate_image_magic(_JPEG_MAGIC) is False

    def test_jpeg_with_trailing_null_bytes(self) -> None:
        data = _JPEG_MAGIC + b"\x00" * 1000
        assert ScreencastServer._validate_image_magic(data) is True

    def test_png_exactly_8_bytes(self) -> None:
        assert ScreencastServer._validate_image_magic(_PNG_MAGIC) is True

    def test_random_binary_invalid(self) -> None:
        assert ScreencastServer._validate_image_magic(b"\xde\xad\xbe\xef\x00\x00\x00\x00") is False


class TestScreencastServerConstruction:
    """ScreencastServer construction with defaults and custom values."""

    def test_default_host_and_port(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
        )
        assert srv._host == "127.0.0.1"
        assert srv._port == 8780

    def test_custom_host_and_port(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="0.0.0.0",
            port=9000,
        )
        assert srv._host == "0.0.0.0"
        assert srv._port == 9000

    def test_not_running_initially(self, server: ScreencastServer) -> None:
        assert server._running is False

    def test_no_active_connections_initially(self, server: ScreencastServer) -> None:
        assert server._active_connections == 0

    def test_registry_and_session_manager_stored(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
        )
        assert srv._registry is registry
        assert srv._sessions is session_manager


class TestScreencastServerGetStatus:
    """get_status method."""

    def test_not_running_status(self, server: ScreencastServer) -> None:
        status = server.get_status()
        assert status["running"] is False

    def test_status_contains_host_and_port(self, server: ScreencastServer) -> None:
        status = server.get_status()
        assert status["host"] == "127.0.0.1"
        assert "port" in status

    def test_status_contains_active_connections(self, server: ScreencastServer) -> None:
        status = server.get_status()
        assert status["active_connections"] == 0

    def test_status_contains_sessions_subdict(self, server: ScreencastServer) -> None:
        status = server.get_status()
        assert "sessions" in status

    @pytest.mark.asyncio
    async def test_status_running_after_start(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        try:
            assert srv.get_status()["running"] is True
        finally:
            await srv.stop()


# ---------------------------------------------------------------------------
# 4. ScreencastChannel (channel.py)
# ---------------------------------------------------------------------------


def _make_channel(**kwargs: Any) -> ScreencastChannel:
    defaults = {"host": "127.0.0.1", "port": 8780, "max_sessions": 5}
    defaults.update(kwargs)
    return ScreencastChannel(**defaults)


class TestScreencastChannelConstruction:
    """Default and custom construction."""

    def test_channel_name(self) -> None:
        assert ScreencastChannel.name == "screencast"

    def test_default_host_and_port(self) -> None:
        ch = ScreencastChannel()
        assert ch._host == "127.0.0.1"
        assert ch._port == 8780

    def test_custom_parameters(self) -> None:
        ch = ScreencastChannel(
            host="0.0.0.0",
            port=9000,
            max_sessions=10,
            frame_save_dir="/tmp/frames",
            vision_model="llava",
            analysis_prompt="describe it",
            capture_url_base="https://my.host",
        )
        assert ch._host == "0.0.0.0"
        assert ch._port == 9000
        assert ch._max_sessions == 10
        assert ch._frame_save_dir == "/tmp/frames"
        assert ch._vision_model == "llava"
        assert ch._analysis_prompt == "describe it"
        assert ch._capture_url_base == "https://my.host"

    def test_initially_not_started(self) -> None:
        ch = _make_channel()
        assert ch._token_registry is None
        assert ch._session_manager is None
        assert ch._analyzer is None
        assert ch._server is None
        assert ch._loop is None
        assert ch._thread is None


class TestScreencastChannelNotImplemented:
    """receive() and send() raise NotImplementedError."""

    def test_receive_raises(self) -> None:
        ch = _make_channel()
        with pytest.raises(NotImplementedError):
            ch.receive()

    def test_send_raises(self) -> None:
        ch = _make_channel()
        with pytest.raises(NotImplementedError):
            ch.send("anything")

    def test_receive_error_message(self) -> None:
        ch = _make_channel()
        with pytest.raises(NotImplementedError, match="event-driven"):
            ch.receive()

    def test_send_error_message(self) -> None:
        ch = _make_channel()
        with pytest.raises(NotImplementedError, match="event-driven"):
            ch.send("x")


class TestScreencastChannelNotRunningFallbacks:
    """All public methods degrade gracefully when the channel is not running."""

    def test_create_session_raises_runtime_error(self) -> None:
        ch = _make_channel()
        with pytest.raises(RuntimeError, match="not running"):
            ch.create_session()

    def test_revoke_session_returns_false(self) -> None:
        ch = _make_channel()
        assert ch.revoke_session("fake-id") is False

    def test_get_active_sessions_returns_empty_list(self) -> None:
        ch = _make_channel()
        assert ch.get_active_sessions() == []

    def test_get_latest_analysis_returns_none(self) -> None:
        ch = _make_channel()
        assert ch.get_latest_analysis("fake") is None

    def test_get_results_returns_empty_list(self) -> None:
        ch = _make_channel()
        assert ch.get_results("fake") == []

    def test_get_status_returns_not_running(self) -> None:
        ch = _make_channel()
        assert ch.get_status() == {"running": False}

    def test_stop_is_noop(self) -> None:
        ch = _make_channel()
        ch.stop()  # Must not raise.


class TestScreencastChannelAlreadyRunning:
    """start() raises RuntimeError when already running."""

    def test_raises_if_thread_alive(self) -> None:
        ch = _make_channel()
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = True
        ch._thread = mock_thread
        with pytest.raises(RuntimeError, match="already running"):
            ch.start()


class TestScreencastChannelCreateSessionMocked:
    """create_session with a mocked token registry."""

    def test_returns_session_id_token_and_url(self) -> None:
        ch = _make_channel(host="127.0.0.1", port=8780)
        mock_reg = MagicMock()
        mock_reg.create_session.return_value = ("sid-001", "tok-001")
        ch._token_registry = mock_reg

        sid, tok, url = ch.create_session(created_by="u1", discord_channel_id="c1", label="l1")
        assert sid == "sid-001"
        assert tok == "tok-001"
        assert "sid-001" in url
        assert "tok-001" in url

    def test_share_url_uses_http_scheme_for_localhost(self) -> None:
        ch = _make_channel(host="127.0.0.1", port=8780)
        mock_reg = MagicMock()
        mock_reg.create_session.return_value = ("s", "t")
        ch._token_registry = mock_reg
        _, _, url = ch.create_session()
        assert url.startswith("http://")

    def test_share_url_includes_port(self) -> None:
        ch = _make_channel(host="127.0.0.1", port=9123)
        mock_reg = MagicMock()
        mock_reg.create_session.return_value = ("s", "t")
        ch._token_registry = mock_reg
        _, _, url = ch.create_session()
        assert ":9123" in url

    def test_custom_capture_url_base_used(self) -> None:
        ch = _make_channel(capture_url_base="https://proxy.example.com")
        mock_reg = MagicMock()
        mock_reg.create_session.return_value = ("s2", "t2")
        ch._token_registry = mock_reg
        _, _, url = ch.create_session()
        assert url.startswith("https://proxy.example.com")

    def test_0000_host_uses_lan_ip(self) -> None:
        ch = _make_channel(host="0.0.0.0", port=8780)
        mock_reg = MagicMock()
        mock_reg.create_session.return_value = ("s3", "t3")
        ch._token_registry = mock_reg
        with patch(
            "missy.channels.screencast.channel._get_lan_ip",
            return_value="10.0.0.42",
        ):
            _, _, url = ch.create_session()
        assert "0.0.0.0" not in url
        assert "10.0.0.42" in url


class TestScreencastChannelRevokeSessionMocked:
    """revoke_session with a mocked registry."""

    def test_revoke_delegates_to_registry(self) -> None:
        ch = _make_channel()
        mock_reg = MagicMock()
        mock_reg.revoke_session.return_value = True
        ch._token_registry = mock_reg
        assert ch.revoke_session("sid") is True
        mock_reg.revoke_session.assert_called_once_with("sid")

    def test_revoke_returns_false_when_missing(self) -> None:
        ch = _make_channel()
        mock_reg = MagicMock()
        mock_reg.revoke_session.return_value = False
        ch._token_registry = mock_reg
        assert ch.revoke_session("missing") is False


class TestScreencastChannelGetActiveSessions:
    """get_active_sessions delegates to token registry."""

    def test_empty_when_no_sessions(self) -> None:
        ch = _make_channel()
        mock_reg = MagicMock()
        mock_reg.list_active.return_value = []
        ch._token_registry = mock_reg
        assert ch.get_active_sessions() == []

    def test_maps_session_fields_correctly(self) -> None:
        ch = _make_channel()
        now = time.time()
        fake_s = MagicMock()
        fake_s.session_id = "s-x"
        fake_s.label = "lbl-x"
        fake_s.created_by = "owner"
        fake_s.created_at = now
        fake_s.frame_count = 42
        fake_s.analysis_count = 7
        fake_s.last_frame_at = now

        mock_reg = MagicMock()
        mock_reg.list_active.return_value = [fake_s]
        ch._token_registry = mock_reg

        sessions = ch.get_active_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s["session_id"] == "s-x"
        assert s["label"] == "lbl-x"
        assert s["frame_count"] == 42
        assert s["analysis_count"] == 7


class TestGetLanIp:
    """_get_lan_ip helper."""

    def test_returns_string(self) -> None:
        ip = _get_lan_ip()
        assert isinstance(ip, str)
        assert len(ip) > 0

    def test_fallback_on_oserror(self) -> None:
        with patch("missy.channels.screencast.channel.socket.socket") as mock_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = MagicMock(return_value=mock_sock)
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect.side_effect = OSError("unreachable")
            mock_cls.return_value = mock_sock
            assert _get_lan_ip() == "127.0.0.1"


class TestScreencastChannelDiscordRest:
    """set_discord_rest stores the reference."""

    def test_set_stores_reference(self) -> None:
        ch = _make_channel()
        mock_rest = MagicMock()
        ch.set_discord_rest(mock_rest)
        assert ch._discord_rest is mock_rest

    def test_set_none_clears_reference(self) -> None:
        ch = _make_channel()
        ch.set_discord_rest(MagicMock())
        ch.set_discord_rest(None)
        assert ch._discord_rest is None


class TestScreencastChannelGetStatusMocked:
    """get_status delegates to _server.get_status()."""

    def test_delegates_to_server(self) -> None:
        ch = _make_channel()
        mock_server = MagicMock()
        mock_server.get_status.return_value = {"running": True, "active_connections": 3}
        ch._server = mock_server
        status = ch.get_status()
        assert status["running"] is True
        assert status["active_connections"] == 3

    def test_returns_not_running_when_no_server(self) -> None:
        ch = _make_channel()
        assert ch.get_status() == {"running": False}


# ---------------------------------------------------------------------------
# 5. FrameAnalyzer (analyzer.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=50)


@pytest.fixture
def sm_with_queue(session_manager: SessionManager, analyzer_queue: asyncio.Queue) -> SessionManager:
    session_manager.set_queue(analyzer_queue)
    return session_manager


class TestFrameAnalyzerConstruction:
    """Default prompt and attribute initialisation."""

    def test_default_prompt_used_when_empty(self) -> None:
        sm = SessionManager()
        reg = ScreencastTokenRegistry()
        fa = FrameAnalyzer(session_manager=sm, token_registry=reg)
        assert fa._prompt == _DEFAULT_PROMPT

    def test_custom_prompt_used(self) -> None:
        sm = SessionManager()
        reg = ScreencastTokenRegistry()
        fa = FrameAnalyzer(session_manager=sm, token_registry=reg, analysis_prompt="custom")
        assert fa._prompt == "custom"

    def test_not_running_initially(self) -> None:
        sm = SessionManager()
        reg = ScreencastTokenRegistry()
        fa = FrameAnalyzer(session_manager=sm, token_registry=reg)
        assert fa._running is False
        assert fa._task is None

    def test_discord_callback_stored(self) -> None:
        sm = SessionManager()
        reg = ScreencastTokenRegistry()
        cb = AsyncMock()
        fa = FrameAnalyzer(session_manager=sm, token_registry=reg, discord_callback=cb)
        assert fa._discord_callback is cb

    def test_frame_save_dir_stored(self) -> None:
        sm = SessionManager()
        reg = ScreencastTokenRegistry()
        fa = FrameAnalyzer(session_manager=sm, token_registry=reg, frame_save_dir="/tmp/x")
        assert fa._frame_save_dir == "/tmp/x"


class TestFrameAnalyzerStartStop:
    """start() / stop() lifecycle."""

    @pytest.mark.asyncio
    async def test_start_sets_running(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)
        await fa.start()
        assert fa._running is True
        assert fa._task is not None
        await fa.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)
        await fa.start()
        await fa.stop()
        assert fa._running is False
        assert fa._task is None

    @pytest.mark.asyncio
    async def test_double_start_is_noop(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)
        await fa.start()
        task_before = fa._task
        await fa.start()  # Second call — should not create a new task.
        assert fa._task is task_before
        await fa.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)
        await fa.stop()  # Must not raise.


class TestFrameAnalyzerProcessFrame:
    """Frame processing — result storage, stat updates, callbacks."""

    @pytest.mark.asyncio
    async def test_processes_frame_and_stores_result(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        session_id, _ = registry.create_session(label="t")
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)

        with patch.object(fa, "_call_vision_model", return_value="A terminal window"):
            await fa.start()
            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            sm_with_queue.enqueue_frame(meta, _JPEG_MAGIC + b"\x00" * 50)
            await asyncio.sleep(0.4)
            await fa.stop()

        result = sm_with_queue.get_latest_result(session_id)
        assert result is not None
        assert result.analysis_text == "A terminal window"
        assert result.frame_number == 1

    @pytest.mark.asyncio
    async def test_analysis_count_increments_in_registry(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        session_id, _ = registry.create_session()
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)

        with patch.object(fa, "_call_vision_model", return_value="result"):
            await fa.start()
            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            sm_with_queue.enqueue_frame(meta, _JPEG_MAGIC + b"\x00" * 50)
            await asyncio.sleep(0.4)
            await fa.stop()

        s = registry.get_session(session_id)
        assert s is not None
        assert s.analysis_count == 1

    @pytest.mark.asyncio
    async def test_discord_callback_invoked_with_correct_args(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        session_id, _ = registry.create_session(discord_channel_id="disc-ch")
        callback = AsyncMock()
        fa = FrameAnalyzer(
            session_manager=sm_with_queue,
            token_registry=registry,
            discord_callback=callback,
        )

        with patch.object(fa, "_call_vision_model", return_value="chat visible"):
            await fa.start()
            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            sm_with_queue.enqueue_frame(meta, _JPEG_MAGIC + b"\x00" * 50)
            await asyncio.sleep(0.4)
            await fa.stop()

        callback.assert_called_once_with(session_id, "disc-ch", "chat visible")

    @pytest.mark.asyncio
    async def test_vision_model_error_does_not_stop_analyzer(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        session_id, _ = registry.create_session()
        fa = FrameAnalyzer(session_manager=sm_with_queue, token_registry=registry)

        with patch.object(fa, "_call_vision_model", side_effect=RuntimeError("model gone")):
            await fa.start()
            meta = FrameMetadata(session_id=session_id, frame_number=1, format="jpeg")
            sm_with_queue.enqueue_frame(meta, _JPEG_MAGIC + b"\x00" * 50)
            await asyncio.sleep(0.4)
            # Analyzer must still be running after an error.
            assert fa._running is True
            await fa.stop()

    @pytest.mark.asyncio
    async def test_no_discord_callback_when_session_missing_from_registry(
        self,
        sm_with_queue: SessionManager,
        registry: ScreencastTokenRegistry,
    ) -> None:
        """When a session_id is not in the registry, Discord callback must not fire."""
        callback = AsyncMock()
        fa = FrameAnalyzer(
            session_manager=sm_with_queue,
            token_registry=registry,  # Registry has no session for "orphan"
            discord_callback=callback,
        )

        with patch.object(fa, "_call_vision_model", return_value="text"):
            await fa.start()
            meta = FrameMetadata(session_id="orphan-sid", frame_number=1, format="jpeg")
            sm_with_queue.enqueue_frame(meta, _JPEG_MAGIC + b"\x00" * 50)
            await asyncio.sleep(0.4)
            await fa.stop()

        callback.assert_not_called()


class TestFrameAnalyzerSaveFrame:
    """_save_frame_sync writes a file with the correct name and permissions."""

    def test_save_frame_sync_creates_file(self, registry: ScreencastTokenRegistry) -> None:
        sm = SessionManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            fa = FrameAnalyzer(
                session_manager=sm,
                token_registry=registry,
                frame_save_dir=tmpdir,
            )
            meta = FrameMetadata(
                session_id="test-sid",
                frame_number=42,
                format="jpeg",
            )
            data = _JPEG_MAGIC + b"\x00" * 100
            fa._save_frame_sync(meta, data)

            session_dir = os.path.join(tmpdir, "test-sid")
            files = os.listdir(session_dir)
            assert len(files) == 1
            assert files[0].endswith(".jpg")

    def test_save_frame_sync_png_extension(self, registry: ScreencastTokenRegistry) -> None:
        sm = SessionManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            fa = FrameAnalyzer(
                session_manager=sm,
                token_registry=registry,
                frame_save_dir=tmpdir,
            )
            meta = FrameMetadata(
                session_id="test-sid",
                frame_number=1,
                format="png",
            )
            fa._save_frame_sync(meta, _PNG_MAGIC + b"\x00" * 50)
            session_dir = os.path.join(tmpdir, "test-sid")
            files = os.listdir(session_dir)
            assert files[0].endswith(".png")

    def test_save_frame_sync_file_permissions(self, registry: ScreencastTokenRegistry) -> None:
        sm = SessionManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            fa = FrameAnalyzer(
                session_manager=sm,
                token_registry=registry,
                frame_save_dir=tmpdir,
            )
            meta = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
            fa._save_frame_sync(meta, _JPEG_MAGIC + b"\x00" * 50)
            session_dir = os.path.join(tmpdir, "s")
            files = os.listdir(session_dir)
            path = os.path.join(session_dir, files[0])
            mode = oct(os.stat(path).st_mode)[-3:]
            assert mode == "600"

    def test_save_frame_sync_frame_number_in_filename(
        self, registry: ScreencastTokenRegistry
    ) -> None:
        sm = SessionManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            fa = FrameAnalyzer(
                session_manager=sm,
                token_registry=registry,
                frame_save_dir=tmpdir,
            )
            meta = FrameMetadata(session_id="s", frame_number=7, format="jpeg")
            fa._save_frame_sync(meta, _JPEG_MAGIC + b"\x00" * 20)
            session_dir = os.path.join(tmpdir, "s")
            files = os.listdir(session_dir)
            # Frame number 7 zero-padded to 6 digits.
            assert "000007" in files[0]

    @pytest.mark.asyncio
    async def test_save_frame_async_wrapper(self, registry: ScreencastTokenRegistry) -> None:
        sm = SessionManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            fa = FrameAnalyzer(
                session_manager=sm,
                token_registry=registry,
                frame_save_dir=tmpdir,
            )
            meta = FrameMetadata(session_id="s", frame_number=1, format="jpeg")
            await fa._save_frame(meta, _JPEG_MAGIC + b"\x00" * 20)
            session_dir = os.path.join(tmpdir, "s")
            assert len(os.listdir(session_dir)) == 1


class TestFrameAnalyzerCallVisionModel:
    """_call_vision_model delegates to analyze_image_bytes."""

    def test_delegates_with_prompt(self, registry: ScreencastTokenRegistry) -> None:
        sm = SessionManager()
        fa = FrameAnalyzer(
            session_manager=sm,
            token_registry=registry,
            analysis_prompt="my prompt",
        )
        with patch(
            "missy.channels.discord.image_analyze.analyze_image_bytes",
            return_value="mocked",
        ) as mock_fn:
            result = fa._call_vision_model(b"fake")
            assert result == "mocked"
            mock_fn.assert_called_once()
            _args, kwargs = mock_fn.call_args
            # prompt is the second positional arg.
            assert _args[1] == "my prompt" or kwargs.get("prompt") == "my prompt"


# ---------------------------------------------------------------------------
# 6. WebSocket protocol integration tests (server.py) — real server on port 0
# ---------------------------------------------------------------------------


class TestScreencastServerProtocol:
    """Lightweight integration tests against a real in-process WebSocket server."""

    @pytest.mark.asyncio
    async def test_successful_auth_returns_auth_ok(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]
        session_id, token = registry.create_session()

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.send(json.dumps({"type": "auth", "session_id": session_id, "token": token}))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "auth_ok"
                assert resp["session_id"] == session_id
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_wrong_token_returns_auth_fail(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]
        session_id, _ = registry.create_session()

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.send(json.dumps({"type": "auth", "session_id": session_id, "token": "bad"}))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "auth_fail"
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_non_auth_first_message_rejected(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.send(json.dumps({"type": "heartbeat"}))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "error"
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_frame_enqueued_after_auth(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]
        session_id, token = registry.create_session()

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.send(json.dumps({"type": "auth", "session_id": session_id, "token": token}))
                await ws.recv()  # auth_ok

                await ws.send(json.dumps({"type": "frame", "format": "jpeg", "width": 640, "height": 480, "seq": 1}))
                await ws.send(_JPEG_MAGIC + b"\x00" * 500)
                await asyncio.sleep(0.2)

                assert session_manager.queue is not None
                assert session_manager.queue.qsize() >= 1
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_invalid_image_magic_returns_error(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        import websockets

        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        port = srv._ws_server.sockets[0].getsockname()[1]
        session_id, token = registry.create_session()

        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/") as ws:
                await ws.send(json.dumps({"type": "auth", "session_id": session_id, "token": token}))
                await ws.recv()  # auth_ok

                await ws.send(json.dumps({"type": "frame", "format": "jpeg", "width": 640, "height": 480, "seq": 1}))
                # Garbage binary — no valid magic bytes.
                await ws.send(b"GIF89a" + b"\x00" * 100)
                await asyncio.sleep(0.1)

                resp = json.loads(await ws.recv())
                assert resp["type"] == "error"
                assert "magic" in resp["message"].lower() or "invalid" in resp["message"].lower()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_start_stop_idempotent(
        self,
        registry: ScreencastTokenRegistry,
        session_manager: SessionManager,
    ) -> None:
        srv = ScreencastServer(
            token_registry=registry,
            session_manager=session_manager,
            host="127.0.0.1",
            port=0,
        )
        await srv.start()
        # Second start must be a no-op (no exception).
        await srv.start()
        assert srv._running is True
        await srv.stop()
        # Second stop must also be a no-op.
        await srv.stop()
        assert srv._running is False
