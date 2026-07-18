"""Global, cross-session spend ceiling (F19).

``CostTracker`` enforces a budget *per session*. Nothing bounded the *aggregate*
spend across every session, scheduled job, and proactive run, so a fleet of
individually-under-cap background runs could collectively overspend.

``GlobalBudget`` closes that: a persistent (flock-serialized, cross-process)
running total with a configurable period (``total`` / ``daily`` / ``monthly``)
that auto-resets at period rollover, a hard ceiling that raises
:class:`~missy.agent.cost_tracker.BudgetExceededError`, and a one-shot threshold
alert (default 80%) so an operator hears about it before the hard stop.

Off by default: with ``max_spend_usd <= 0`` every operation is a no-op, so the
existing unlimited behaviour is unchanged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from missy.agent.cost_tracker import BudgetExceededError

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_BUDGET_PATH = "~/.missy/global_budget.json"
_VALID_PERIODS = frozenset({"total", "daily", "monthly"})


def _period_key(period: str, now: datetime) -> str:
    """Return the bucket key for *period* at time *now*."""
    if period == "daily":
        return now.strftime("%Y-%m-%d")
    if period == "monthly":
        return now.strftime("%Y-%m")
    return "total"


class GlobalBudget:
    """Persistent cross-session spend ceiling with period reset + alerts.

    Args:
        max_spend_usd: Hard ceiling. ``<= 0`` disables all enforcement
            (every method becomes a no-op) — the default, unlimited behaviour.
        period: ``"total"`` (never resets), ``"daily"``, or ``"monthly"``.
        path: Persistence file. Defaults to ``~/.missy/global_budget.json``.
        alert_threshold: Fraction (0..1) at which ``alert_fn`` fires once per
            period. ``0`` disables the alert.
        alert_fn: Optional callback invoked with a human-readable message the
            first time spend crosses ``alert_threshold`` in the current period.
    """

    def __init__(
        self,
        max_spend_usd: float = 0.0,
        *,
        period: str = "total",
        path: str | os.PathLike[str] = DEFAULT_GLOBAL_BUDGET_PATH,
        alert_threshold: float = 0.8,
        alert_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.max_spend_usd = max_spend_usd or 0.0
        self.period = period if period in _VALID_PERIODS else "total"
        self._path = Path(path).expanduser()
        self._lock_path = str(self._path) + ".lock"
        self.alert_threshold = alert_threshold
        self._alert_fn = alert_fn
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_spend_usd > 0

    # ------------------------------------------------------------------
    # Persistence (flock-serialized so concurrent processes don't clobber)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _cross_process_locked(self):
        import fcntl

        dir_path = os.path.dirname(self._lock_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True, mode=0o700)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load(self) -> dict:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError, TypeError):
            pass
        return {}

    def _save(self, data: dict) -> None:
        dir_path = os.path.dirname(self._path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=dir_path or ".", prefix=".gbudget-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self._path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            logger.debug("GlobalBudget: could not persist to %s", self._path, exc_info=True)

    def _current(self, data: dict, now: datetime) -> dict:
        """Return the record for the current period, resetting on rollover."""
        key = _period_key(self.period, now)
        if data.get("period") != self.period or data.get("key") != key:
            return {"period": self.period, "key": key, "spent": 0.0, "alerted": False}
        data.setdefault("spent", 0.0)
        data.setdefault("alerted", False)
        return data

    # ------------------------------------------------------------------
    # Public API (all no-op when disabled)
    # ------------------------------------------------------------------

    def record(self, cost_usd: float) -> None:
        """Add *cost_usd* to the current-period total; fire the alert on cross."""
        if not self.enabled or cost_usd <= 0:
            return
        with self._lock, self._cross_process_locked():
            now = datetime.now(UTC)
            rec = self._current(self._load(), now)
            before = float(rec["spent"])
            rec["spent"] = before + cost_usd
            fire_alert = False
            if (
                self.alert_threshold > 0
                and not rec["alerted"]
                and rec["spent"] >= self.max_spend_usd * self.alert_threshold
            ):
                rec["alerted"] = True
                fire_alert = True
            self._save(rec)
        if fire_alert and self._alert_fn is not None:
            with contextlib.suppress(Exception):
                self._alert_fn(
                    f"Global budget at {rec['spent'] / self.max_spend_usd:.0%} "
                    f"(${rec['spent']:.4f} of ${self.max_spend_usd:.2f} this {self.period})."
                )

    def total_spent(self) -> float:
        """Return spend in the current period (0.0 when disabled)."""
        if not self.enabled:
            return 0.0
        with self._lock, self._cross_process_locked():
            rec = self._current(self._load(), datetime.now(UTC))
            return float(rec["spent"])

    def remaining(self) -> float:
        """Return remaining budget this period (0.0 floor)."""
        if not self.enabled:
            return 0.0
        return max(0.0, self.max_spend_usd - self.total_spent())

    def usage_fraction(self) -> float:
        """Return spent/limit (0.0 when disabled)."""
        if not self.enabled:
            return 0.0
        return self.total_spent() / self.max_spend_usd

    def would_exceed(self, cost_usd: float) -> bool:
        """Return True if adding *cost_usd* would breach the ceiling."""
        if not self.enabled:
            return False
        return self.total_spent() + max(0.0, cost_usd) > self.max_spend_usd

    def check(self) -> None:
        """Raise :class:`BudgetExceededError` when the ceiling is reached."""
        if not self.enabled:
            return
        spent = self.total_spent()
        if spent >= self.max_spend_usd:
            raise BudgetExceededError(spent, self.max_spend_usd)

    def reset(self) -> None:
        """Clear the persisted total (operator override)."""
        with self._lock, self._cross_process_locked():
            self._save(
                {
                    "period": self.period,
                    "key": _period_key(self.period, datetime.now(UTC)),
                    "spent": 0.0,
                    "alerted": False,
                }
            )

    def summary(self) -> dict:
        """Return a status dict for CLI/API display."""
        return {
            "enabled": self.enabled,
            "period": self.period,
            "max_spend_usd": self.max_spend_usd,
            "total_spent_usd": round(self.total_spent(), 6),
            "remaining_usd": round(self.remaining(), 6),
            "usage_fraction": round(self.usage_fraction(), 4),
        }


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_ACTIVE: GlobalBudget | None = None
_ACTIVE_LOCK = threading.Lock()


def init_global_budget(
    max_spend_usd: float = 0.0,
    *,
    period: str = "total",
    path: str | os.PathLike[str] = DEFAULT_GLOBAL_BUDGET_PATH,
    alert_threshold: float = 0.8,
    alert_fn: Callable[[str], None] | None = None,
) -> GlobalBudget:
    """Install the process-level GlobalBudget and return it.

    Safe to call more than once (e.g. on config hot-reload); the newest call
    wins. Idempotent for the common ``max_spend_usd <= 0`` (disabled) case.
    """
    global _ACTIVE
    budget = GlobalBudget(
        max_spend_usd,
        period=period,
        path=path,
        alert_threshold=alert_threshold,
        alert_fn=alert_fn,
    )
    with _ACTIVE_LOCK:
        _ACTIVE = budget
    return budget


def get_global_budget() -> GlobalBudget:
    """Return the active GlobalBudget, or a disabled (no-op) one if unset.

    Callers on the cost hot path (``AgentRuntime._record_cost``/
    ``_check_budget``) use this; when nothing initialized a budget the returned
    instance is disabled, so global enforcement is simply off.
    """
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            return _ACTIVE
    return GlobalBudget(0.0)
