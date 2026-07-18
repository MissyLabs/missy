"""F07 — Landlock bootstrap wiring (apply_landlock_if_enabled + config + CLI).

The real Landlock apply is process-wide and irreversible, so every test mocks it
out — we assert the *wiring/gating*, never actually sandbox the test process.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from missy.cli.main import cli
from missy.security import landlock as ll


def _cfg(enabled: bool):
    return SimpleNamespace(landlock_enabled=enabled)


class TestConfigFlag:
    def test_default_false(self) -> None:
        from missy.config.settings import MissyConfig

        assert MissyConfig.__dataclass_fields__["landlock_enabled"].default is False


class TestApplyIfEnabled:
    def test_disabled_is_noop(self) -> None:
        with patch.object(ll, "apply_landlock_from_config") as m:
            result = ll.apply_landlock_if_enabled(_cfg(False))
        assert result == {"applied": False, "reason": "disabled"}
        m.assert_not_called()

    def test_enabled_unsupported_kernel(self) -> None:
        with (
            patch.object(ll.LandlockPolicy, "is_available", return_value=False),
            patch.object(ll, "apply_landlock_from_config") as m,
        ):
            result = ll.apply_landlock_if_enabled(_cfg(True))
        assert result == {"applied": False, "reason": "unsupported"}
        m.assert_not_called()

    def test_enabled_supported_applies(self) -> None:
        with (
            patch.object(ll.LandlockPolicy, "is_available", return_value=True),
            patch.object(ll, "apply_landlock_from_config", return_value=True) as m,
        ):
            result = ll.apply_landlock_if_enabled(_cfg(True))
        assert result == {"applied": True}
        m.assert_called_once()

    def test_apply_returns_false(self) -> None:
        with (
            patch.object(ll.LandlockPolicy, "is_available", return_value=True),
            patch.object(ll, "apply_landlock_from_config", return_value=False),
        ):
            result = ll.apply_landlock_if_enabled(_cfg(True))
        assert result["applied"] is False
        assert result["reason"] == "not_applied"

    def test_magicmock_config_does_not_apply(self) -> None:
        # Critical regression guard: a MagicMock config (as gateway_start tests
        # use) must NOT trigger a real, irreversible Landlock apply — a truthy
        # Mock attribute would otherwise sandbox the whole test runner.
        from unittest.mock import MagicMock

        with patch.object(ll, "apply_landlock_from_config") as m:
            result = ll.apply_landlock_if_enabled(MagicMock())
        assert result == {"applied": False, "reason": "disabled"}
        m.assert_not_called()

    def test_apply_exception_never_fatal(self) -> None:
        with (
            patch.object(ll.LandlockPolicy, "is_available", return_value=True),
            patch.object(ll, "apply_landlock_from_config", side_effect=RuntimeError("boom")),
        ):
            result = ll.apply_landlock_if_enabled(_cfg(True))
        assert result["applied"] is False
        assert result["reason"] == "error"
        assert "boom" in result["error"]


def _runner() -> CliRunner:
    sig = inspect.signature(CliRunner.__init__)
    kwargs = {"mix_stderr": False} if "mix_stderr" in sig.parameters else {}
    return CliRunner(**kwargs)


class TestSecurityLandlockCLI:
    def _invoke(self, *, enabled: bool, supported: bool):
        cfg = SimpleNamespace(landlock_enabled=enabled)
        status = {
            "available": supported,
            "applied": False,
            "kernel_version": "6.17.0",
            "platform": "linux",
        }
        with (
            patch("missy.cli.main._load_subsystems", return_value=cfg),
            patch("missy.security.landlock.landlock_status", return_value=status),
        ):
            return _runner().invoke(cli, ["security", "landlock"])

    def test_supported_disabled_hint(self) -> None:
        r = self._invoke(enabled=False, supported=True)
        assert r.exit_code == 0, r.output
        assert "landlock_enabled: true" in r.output

    def test_enabled_unsupported_warns(self) -> None:
        r = self._invoke(enabled=True, supported=False)
        assert r.exit_code == 0, r.output
        assert "doesn't support" in r.output

    def test_enabled_supported(self) -> None:
        r = self._invoke(enabled=True, supported=True)
        assert r.exit_code == 0, r.output
        assert "yes" in r.output
