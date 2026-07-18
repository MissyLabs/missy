"""Tests for the cross-session GlobalBudget ceiling (F19)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from missy.agent.cost_tracker import BudgetExceededError
from missy.agent.global_budget import GlobalBudget, _period_key


def _budget(tmp_path: Path, **kw) -> GlobalBudget:
    kw.setdefault("path", str(tmp_path / "global_budget.json"))
    return GlobalBudget(**kw)


class TestDisabled:
    def test_zero_limit_is_noop(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=0.0)
        assert b.enabled is False
        b.record(5.0)  # no-op
        assert b.total_spent() == 0.0
        assert b.remaining() == 0.0
        assert b.would_exceed(100.0) is False
        b.check()  # never raises
        assert not (tmp_path / "global_budget.json").exists()


class TestRecording:
    def test_accumulates_across_instances(self, tmp_path: Path) -> None:
        # Separate instances = separate sessions/processes sharing the file.
        p = str(tmp_path / "gb.json")
        GlobalBudget(max_spend_usd=1.0, path=p).record(0.3)
        GlobalBudget(max_spend_usd=1.0, path=p).record(0.4)
        assert GlobalBudget(max_spend_usd=1.0, path=p).total_spent() == pytest.approx(0.7)

    def test_remaining_and_fraction(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=2.0)
        b.record(0.5)
        assert b.remaining() == pytest.approx(1.5)
        assert b.usage_fraction() == pytest.approx(0.25)

    def test_negative_and_zero_cost_ignored(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=1.0)
        b.record(-1.0)
        b.record(0.0)
        assert b.total_spent() == 0.0


class TestCeiling:
    def test_check_raises_when_exceeded(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=1.0)
        b.record(1.0)
        with pytest.raises(BudgetExceededError):
            b.check()

    def test_check_ok_below_ceiling(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=1.0)
        b.record(0.99)
        b.check()  # no raise

    def test_would_exceed(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=1.0)
        b.record(0.8)
        assert b.would_exceed(0.3) is True
        assert b.would_exceed(0.1) is False


class TestAlert:
    def test_alert_fires_once_on_threshold(self, tmp_path: Path) -> None:
        fired: list[str] = []
        b = _budget(tmp_path, max_spend_usd=1.0, alert_threshold=0.8, alert_fn=fired.append)
        b.record(0.5)  # 50% — no alert
        assert fired == []
        b.record(0.35)  # 85% — alert
        assert len(fired) == 1
        assert "80%" in fired[0] or "85%" in fired[0]
        b.record(0.05)  # 90% — no second alert
        assert len(fired) == 1

    def test_alert_disabled_when_threshold_zero(self, tmp_path: Path) -> None:
        fired: list[str] = []
        b = _budget(tmp_path, max_spend_usd=1.0, alert_threshold=0.0, alert_fn=fired.append)
        b.record(1.0)
        assert fired == []


class TestPeriodReset:
    def test_period_key(self) -> None:
        now = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
        assert _period_key("total", now) == "total"
        assert _period_key("daily", now) == "2026-07-18"
        assert _period_key("monthly", now) == "2026-07"

    def test_daily_rollover_resets(self, tmp_path: Path) -> None:
        p = str(tmp_path / "gb.json")
        # Seed a record for a *past* day directly.
        Path(p).write_text(
            json.dumps({"period": "daily", "key": "2000-01-01", "spent": 5.0, "alerted": True})
        )
        b = GlobalBudget(max_spend_usd=10.0, period="daily", path=p)
        # Today's bucket is fresh — the stale day is not counted.
        assert b.total_spent() == 0.0
        b.record(0.25)
        assert b.total_spent() == pytest.approx(0.25)

    def test_invalid_period_defaults_to_total(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=1.0, period="weekly")
        assert b.period == "total"


class TestResetAndSummary:
    def test_reset_clears(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=1.0)
        b.record(0.5)
        b.reset()
        assert b.total_spent() == 0.0

    def test_summary_shape(self, tmp_path: Path) -> None:
        b = _budget(tmp_path, max_spend_usd=2.0)
        b.record(0.5)
        s = b.summary()
        assert s["enabled"] is True
        assert s["max_spend_usd"] == 2.0
        assert s["total_spent_usd"] == pytest.approx(0.5)
        assert s["remaining_usd"] == pytest.approx(1.5)

    def test_corrupt_file_recovers(self, tmp_path: Path) -> None:
        p = str(tmp_path / "gb.json")
        Path(p).write_text("not json{{{")
        b = GlobalBudget(max_spend_usd=1.0, path=p)
        assert b.total_spent() == 0.0  # starts fresh, no crash
        b.record(0.1)
        assert b.total_spent() == pytest.approx(0.1)
