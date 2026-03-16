"""Tests for the screencast token registry (auth.py)."""

from __future__ import annotations

import time

import pytest

from missy.channels.screencast.auth import ScreencastSession, ScreencastTokenRegistry


class TestScreencastTokenRegistry:
    """Tests for ScreencastTokenRegistry."""

    def test_create_session_returns_id_and_token(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session(
            created_by="user123",
            discord_channel_id="chan456",
            label="debugging",
        )
        assert isinstance(session_id, str)
        assert len(session_id) > 10
        assert isinstance(token, str)
        assert len(token) > 20

    def test_verify_token_correct(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        assert reg.verify_token(session_id, token) is True

    def test_verify_token_wrong_token(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _token = reg.create_session()
        assert reg.verify_token(session_id, "wrong-token") is False

    def test_verify_token_wrong_session(self) -> None:
        reg = ScreencastTokenRegistry()
        _session_id, token = reg.create_session()
        assert reg.verify_token("nonexistent-session", token) is False

    def test_verify_token_revoked_session(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, token = reg.create_session()
        reg.revoke_session(session_id)
        assert reg.verify_token(session_id, token) is False

    def test_get_session(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _token = reg.create_session(
            created_by="user1",
            label="test",
        )
        session = reg.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert session.created_by == "user1"
        assert session.label == "test"
        assert session.active is True

    def test_get_session_nonexistent(self) -> None:
        reg = ScreencastTokenRegistry()
        assert reg.get_session("nope") is None

    def test_list_active(self) -> None:
        reg = ScreencastTokenRegistry()
        s1, _ = reg.create_session(label="one")
        s2, _ = reg.create_session(label="two")
        s3, _ = reg.create_session(label="three")
        reg.revoke_session(s2)

        active = reg.list_active()
        active_ids = {s.session_id for s in active}
        assert s1 in active_ids
        assert s2 not in active_ids
        assert s3 in active_ids

    def test_revoke_session_returns_true(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        assert reg.revoke_session(session_id) is True

    def test_revoke_session_nonexistent_returns_false(self) -> None:
        reg = ScreencastTokenRegistry()
        assert reg.revoke_session("nope") is False

    def test_update_frame_stats(self) -> None:
        reg = ScreencastTokenRegistry()
        session_id, _ = reg.create_session()
        reg.update_frame_stats(session_id, frame_count=5, analysis_count=2)

        session = reg.get_session(session_id)
        assert session is not None
        assert session.frame_count == 5
        assert session.analysis_count == 2
        assert session.last_frame_at > 0

    def test_update_frame_stats_nonexistent_noop(self) -> None:
        reg = ScreencastTokenRegistry()
        # Should not raise.
        reg.update_frame_stats("nope", frame_count=1)

    def test_hash_token_deterministic(self) -> None:
        h1 = ScreencastTokenRegistry._hash_token("sid", "tok")
        h2 = ScreencastTokenRegistry._hash_token("sid", "tok")
        assert h1 == h2

    def test_hash_token_different_salts(self) -> None:
        h1 = ScreencastTokenRegistry._hash_token("sid1", "tok")
        h2 = ScreencastTokenRegistry._hash_token("sid2", "tok")
        assert h1 != h2

    def test_multiple_sessions_independent_tokens(self) -> None:
        reg = ScreencastTokenRegistry()
        s1, t1 = reg.create_session()
        s2, t2 = reg.create_session()
        assert s1 != s2
        assert t1 != t2
        assert reg.verify_token(s1, t1) is True
        assert reg.verify_token(s2, t2) is True
        assert reg.verify_token(s1, t2) is False
        assert reg.verify_token(s2, t1) is False
