"""Tests for missy.core.session."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from uuid import UUID

import pytest

from missy.core.session import Session, SessionManager

# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------


class TestSession:
    def test_create_with_timezone_aware_datetime(self):
        sid = SessionManager.generate_session_id()
        now = datetime.now(tz=UTC)
        session = Session(id=sid, created_at=now)
        assert session.id == sid
        assert session.created_at == now

    def test_create_with_naive_datetime_raises(self):
        sid = SessionManager.generate_session_id()
        naive = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            Session(id=sid, created_at=naive)

    def test_metadata_defaults_to_empty_dict(self):
        session = Session(
            id=SessionManager.generate_session_id(),
            created_at=datetime.now(tz=UTC),
        )
        assert session.metadata == {}

    def test_metadata_is_stored(self):
        meta = {"user": "alice", "source": "cli"}
        session = Session(
            id=SessionManager.generate_session_id(),
            created_at=datetime.now(tz=UTC),
            metadata=meta,
        )
        assert session.metadata == meta


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------


class TestIDGenerators:
    def test_generate_session_id_returns_uuid(self):
        uid = SessionManager.generate_session_id()
        assert isinstance(uid, UUID)

    def test_generate_session_id_unique_each_call(self):
        ids = {SessionManager.generate_session_id() for _ in range(20)}
        assert len(ids) == 20

    def test_generate_task_id_returns_uuid(self):
        uid = SessionManager.generate_task_id()
        assert isinstance(uid, UUID)

    def test_generate_task_id_unique_each_call(self):
        ids = {SessionManager.generate_task_id() for _ in range(20)}
        assert len(ids) == 20

    def test_session_id_and_task_id_are_independent(self):
        sid = SessionManager.generate_session_id()
        tid = SessionManager.generate_task_id()
        assert sid != tid


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class TestSessionManager:
    def test_get_current_session_returns_none_initially(self):
        manager = SessionManager()
        assert manager.get_current_session() is None

    def test_create_session_returns_session(self):
        manager = SessionManager()
        session = manager.create_session()
        assert isinstance(session, Session)

    def test_create_session_binds_to_current_thread(self):
        manager = SessionManager()
        session = manager.create_session()
        assert manager.get_current_session() is session

    def test_create_session_has_utc_timestamp(self):
        manager = SessionManager()
        session = manager.create_session()
        assert session.created_at.tzinfo is not None

    def test_create_session_with_metadata(self):
        manager = SessionManager()
        meta = {"env": "test"}
        session = manager.create_session(metadata=meta)
        assert session.metadata == meta

    def test_create_session_without_metadata_uses_empty_dict(self):
        manager = SessionManager()
        session = manager.create_session()
        assert session.metadata == {}

    def test_create_session_replaces_existing_session(self):
        manager = SessionManager()
        first = manager.create_session()
        second = manager.create_session()
        assert manager.get_current_session() is second
        assert first is not second

    def test_clear_session_removes_binding(self):
        manager = SessionManager()
        manager.create_session()
        manager.clear_session()
        assert manager.get_current_session() is None

    def test_clear_session_when_no_session_is_safe(self):
        manager = SessionManager()
        manager.clear_session()  # should not raise
        assert manager.get_current_session() is None

    def test_thread_isolation(self):
        """Sessions created in different threads are independent."""
        manager = SessionManager()
        sessions: list[Session | None] = [None, None]
        errors: list[Exception] = []

        def thread_fn(index: int) -> None:
            try:
                sessions[index] = manager.create_session(
                    metadata={"thread": index}
                )
            except Exception as exc:
                errors.append(exc)

        t0 = threading.Thread(target=thread_fn, args=(0,))
        t1 = threading.Thread(target=thread_fn, args=(1,))
        t0.start()
        t1.start()
        t0.join()
        t1.join()

        assert not errors
        assert sessions[0] is not None
        assert sessions[1] is not None
        assert sessions[0].id != sessions[1].id

    def test_main_thread_session_unaffected_by_child_thread(self):
        """The main thread's session is not overwritten by a child thread."""
        manager = SessionManager()
        main_session = manager.create_session(metadata={"owner": "main"})

        def child_fn() -> None:
            manager.create_session(metadata={"owner": "child"})

        t = threading.Thread(target=child_fn)
        t.start()
        t.join()

        # The main thread session is unchanged.
        assert manager.get_current_session() is main_session

    def test_session_id_is_uuid(self):
        manager = SessionManager()
        session = manager.create_session()
        assert isinstance(session.id, UUID)
