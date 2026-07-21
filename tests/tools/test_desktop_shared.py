"""Tests for missy.tools.builtin._desktop_shared -- the shared config-load
and fail-closed confirmation helper used by obs_tools.py, vtube_tools.py,
and desktop_tools.py. Centralized here rather than duplicated per-file so
this exact behavior can't drift between the three tool modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from missy.tools.builtin._desktop_shared import load_missy_config, require_approval


class TestLoadMissyConfig:
    def test_returns_none_on_missing_file(self, monkeypatch):
        monkeypatch.setenv("MISSY_CONFIG", "/nonexistent/path/config.yaml")
        assert load_missy_config() is None

    def test_returns_config_on_success(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("providers: {}\n", encoding="utf-8")
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))

        result = load_missy_config()
        assert result is not None
        assert result.obs.enabled is False

    def test_returns_none_on_invalid_yaml(self, monkeypatch, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("not: valid: yaml: [", encoding="utf-8")
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
        assert load_missy_config() is None


class TestRequireApproval:
    def test_fails_closed_when_no_gate_configured(self):
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=None):
            result = require_approval("do a thing", "because")
        assert result is not None
        assert "approval" in result.lower()

    def test_returns_none_when_approved(self):
        mock_gate = MagicMock()
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate):
            result = require_approval("do a thing", "because")
        assert result is None
        mock_gate.request.assert_called_once_with(
            action="do a thing", reason="because", risk="high"
        )

    def test_returns_error_when_denied(self):
        from missy.agent.approval import ApprovalDenied

        mock_gate = MagicMock()
        mock_gate.request.side_effect = ApprovalDenied("nope")
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate):
            result = require_approval("do a thing", "because")
        assert result is not None
        assert "not granted" in result.lower()

    def test_returns_error_when_timed_out(self):
        from missy.agent.approval import ApprovalTimeout

        mock_gate = MagicMock()
        mock_gate.request.side_effect = ApprovalTimeout("too slow")
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate):
            result = require_approval("do a thing", "because")
        assert result is not None

    def test_custom_risk_level_is_forwarded(self):
        mock_gate = MagicMock()
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate):
            require_approval("do a thing", "because", risk="medium")
        assert mock_gate.request.call_args.kwargs["risk"] == "medium"
