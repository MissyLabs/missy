"""Tests for missy.tools.builtin._desktop_shared -- the shared config-load
and fail-closed confirmation helper used by obs_tools.py, vtube_tools.py,
and desktop_tools.py. Centralized here rather than duplicated per-file so
this exact behavior can't drift between the three tool modules.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from missy.tools.builtin._desktop_shared import (
    check_rate_limit,
    check_window_allowed,
    load_missy_config,
    require_approval,
    reset_rate_limits,
)


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


class TestCheckRateLimit:
    def test_allows_calls_within_budget(self):
        for _ in range(5):
            assert check_rate_limit("test_tool_a", max_calls=5) is None

    def test_denies_call_over_budget(self):
        for _ in range(3):
            assert check_rate_limit("test_tool_b", max_calls=3) is None
        result = check_rate_limit("test_tool_b", max_calls=3)
        assert result is not None
        assert "test_tool_b" in result
        assert "Rate limit exceeded" in result

    def test_denied_call_is_not_recorded(self):
        """A rejected call must not itself count against the budget --
        otherwise the caller could never recover once at the limit."""
        for _ in range(2):
            check_rate_limit("test_tool_c", max_calls=2)
        check_rate_limit("test_tool_c", max_calls=2)  # denied
        check_rate_limit("test_tool_c", max_calls=2)  # denied again
        # Still exactly 2 recorded calls -- denials didn't inflate the count.
        from missy.tools.builtin._desktop_shared import _rate_limit_calls

        assert len(_rate_limit_calls["test_tool_c"]) == 2

    def test_zero_or_negative_max_calls_means_unlimited(self):
        for _ in range(100):
            assert check_rate_limit("test_tool_d", max_calls=0) is None
        for _ in range(100):
            assert check_rate_limit("test_tool_e", max_calls=-1) is None

    def test_independent_buckets_per_key(self):
        for _ in range(3):
            assert check_rate_limit("test_tool_f", max_calls=3) is None
        # A different key has its own, unaffected budget.
        assert check_rate_limit("test_tool_g", max_calls=3) is None

    def test_old_calls_roll_off_the_window(self):
        assert check_rate_limit("test_tool_h", max_calls=1, window_seconds=0.05) is None
        assert check_rate_limit("test_tool_h", max_calls=1, window_seconds=0.05) is not None
        time.sleep(0.1)
        assert check_rate_limit("test_tool_h", max_calls=1, window_seconds=0.05) is None

    def test_reset_rate_limits_clears_all_buckets(self):
        for _ in range(5):
            check_rate_limit("test_tool_i", max_calls=5)
        assert check_rate_limit("test_tool_i", max_calls=5) is not None
        reset_rate_limits()
        assert check_rate_limit("test_tool_i", max_calls=5) is None

    def test_concurrent_calls_never_exceed_budget(self):
        import threading

        results = []
        lock = threading.Lock()

        def worker():
            r = check_rate_limit("test_tool_j", max_calls=10)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed = sum(1 for r in results if r is None)
        assert allowed == 10


class TestCheckWindowAllowed:
    def _config(self, tmp_path, **desktop_overrides):
        import yaml

        cfg_file = tmp_path / "config.yaml"
        desktop = {"enabled": True, "window_allowlist": [], "unrestricted": False}
        desktop.update(desktop_overrides)
        cfg_file.write_text(yaml.safe_dump({"providers": {}, "desktop": desktop}), encoding="utf-8")
        return cfg_file

    def test_allowed_when_desktop_not_enabled(self, monkeypatch):
        """Backward compatibility: x11_click/type/key predate this guardrail
        and must stay unrestricted for anyone who never opted into desktop:."""
        monkeypatch.setenv("MISSY_CONFIG", "/nonexistent/config.yaml")
        assert check_window_allowed("AnyWindow") is None

    def test_allowed_when_unrestricted(self, monkeypatch, tmp_path):
        cfg_file = self._config(tmp_path, unrestricted=True)
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
        assert check_window_allowed("AnyWindow") is None

    def test_allowed_when_matches_allowlist_substring(self, monkeypatch, tmp_path):
        cfg_file = self._config(tmp_path, window_allowlist=["firefox"])
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
        assert check_window_allowed("Mozilla Firefox - GitHub") is None

    def test_allowlist_match_is_case_insensitive(self, monkeypatch, tmp_path):
        cfg_file = self._config(tmp_path, window_allowlist=["FIREFOX"])
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
        assert check_window_allowed("mozilla firefox") is None

    def test_requires_approval_when_not_on_allowlist(self, monkeypatch, tmp_path):
        cfg_file = self._config(tmp_path, window_allowlist=["firefox"])
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=None):
            result = check_window_allowed("Unknown App")
        assert result is not None
        assert "approval" in result.lower()

    def test_approved_non_allowlisted_window_proceeds(self, monkeypatch, tmp_path):
        cfg_file = self._config(tmp_path, window_allowlist=["firefox"])
        monkeypatch.setenv("MISSY_CONFIG", str(cfg_file))
        mock_gate = MagicMock()
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate):
            result = check_window_allowed("Unknown App")
        assert result is None
        mock_gate.request.assert_called_once()
