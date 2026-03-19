"""Tests for vision subsystem graceful shutdown."""

from unittest.mock import MagicMock, patch

import pytest

from missy.vision.shutdown import (
    register_shutdown_hook,
    reset_shutdown_state,
    vision_shutdown,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset shutdown state before each test."""
    reset_shutdown_state()
    yield
    reset_shutdown_state()


class TestVisionShutdown:
    """Tests for vision_shutdown()."""

    def test_returns_shutdown_status(self):
        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor") as mock_mon,
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            mock_mon.return_value._persist_path = None
            result = vision_shutdown()

        assert result["status"] == "shutdown"
        assert isinstance(result["steps"], list)

    def test_idempotent_second_call_returns_already_shutdown(self):
        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor") as mock_mon,
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            mock_mon.return_value._persist_path = None
            vision_shutdown()
            result = vision_shutdown()

        assert result["status"] == "already_shutdown"

    def test_closes_active_scene_sessions(self):
        mock_mgr = MagicMock()
        mock_mgr.list_sessions.return_value = [
            {"task_id": "a", "active": True},
            {"task_id": "b", "active": False},
        ]

        with (
            patch("missy.vision.scene_memory.get_scene_manager", return_value=mock_mgr),
            patch("missy.vision.health_monitor.get_health_monitor") as mock_mon,
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mon.return_value._persist_path = None
            result = vision_shutdown()

        mock_mgr.close_all.assert_called_once()
        assert any("1 active" in s for s in result["steps"])

    def test_saves_health_monitor_when_persist_path_set(self):
        mock_mon = MagicMock()
        mock_mon._persist_path = "/tmp/health.db"

        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor", return_value=mock_mon),
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            result = vision_shutdown()

        mock_mon.save.assert_called_once()
        assert any("saved" in s for s in result["steps"])

    def test_skips_health_save_without_persist_path(self):
        mock_mon = MagicMock()
        mock_mon._persist_path = None

        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor", return_value=mock_mon),
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            result = vision_shutdown()

        mock_mon.save.assert_not_called()
        assert any("no persist" in s for s in result["steps"])

    def test_continues_on_scene_cleanup_failure(self):
        with (
            patch(
                "missy.vision.scene_memory.get_scene_manager",
                side_effect=RuntimeError("boom"),
            ),
            patch("missy.vision.health_monitor.get_health_monitor") as mock_mon,
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mon.return_value._persist_path = None
            result = vision_shutdown()

        assert result["status"] == "shutdown"
        assert any("failed" in s for s in result["steps"])

    def test_continues_on_health_save_failure(self):
        mock_mon = MagicMock()
        mock_mon._persist_path = "/tmp/health.db"
        mock_mon.save.side_effect = OSError("disk full")

        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor", return_value=mock_mon),
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            result = vision_shutdown()

        assert result["status"] == "shutdown"
        assert any("save failed" in s for s in result["steps"])

    def test_audit_event_logged(self):
        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor") as mock_mon,
            patch("missy.vision.audit.audit_vision_session") as mock_audit,
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            mock_mon.return_value._persist_path = None
            vision_shutdown()

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args[1]
        assert call_kwargs["task_id"] == "shutdown"
        assert call_kwargs["action"] == "close"


class TestRegisterShutdownHook:
    """Tests for register_shutdown_hook()."""

    def test_registers_atexit(self):
        with patch("missy.vision.shutdown.atexit.register") as mock_reg:
            register_shutdown_hook()
        mock_reg.assert_called_once_with(vision_shutdown)


class TestResetShutdownState:
    """Tests for reset_shutdown_state()."""

    def test_allows_shutdown_after_reset(self):
        with (
            patch("missy.vision.scene_memory.get_scene_manager") as mock_mgr,
            patch("missy.vision.health_monitor.get_health_monitor") as mock_mon,
            patch("missy.vision.audit.audit_vision_session"),
        ):
            mock_mgr.return_value.list_sessions.return_value = []
            mock_mon.return_value._persist_path = None

            result1 = vision_shutdown()
            assert result1["status"] == "shutdown"

            reset_shutdown_state()

            result2 = vision_shutdown()
            assert result2["status"] == "shutdown"
