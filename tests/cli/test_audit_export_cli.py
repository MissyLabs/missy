"""Tests for `missy audit export` and `missy audit verify-bundle` (F22)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from missy.cli.main import cli


def _runner() -> CliRunner:
    sig = inspect.signature(CliRunner.__init__)
    kwargs = {"mix_stderr": False} if "mix_stderr" in sig.parameters else {}
    return CliRunner(**kwargs)


def _combined(result) -> str:
    out = result.output
    try:
        err = result.stderr
    except (ValueError, AttributeError):
        err = ""
    if err and err not in out:
        out += err
    return out


def _write_log(path: Path) -> None:
    lines = [
        {
            "timestamp": "2026-07-18T00:00:00+00:00",
            "category": "security",
            "event_type": "e1",
            "detail": {},
        },
        {
            "timestamp": "2026-07-18T01:00:00+00:00",
            "category": "network",
            "event_type": "e2",
            "detail": {},
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def _run(args, audit_log_path: str):
    cfg = SimpleNamespace(audit_log_path=audit_log_path)
    with patch("missy.cli.main._load_subsystems", return_value=cfg):
        return _runner().invoke(cli, args)


class TestAuditExportVerifyRoundTrip:
    def test_export_then_verify_bundle(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        _write_log(log)
        bundle = tmp_path / "bundle.json"

        r1 = _run(["audit", "export", "--out", str(bundle)], str(log))
        assert r1.exit_code == 0, _combined(r1)
        assert bundle.exists()
        assert "Exported 2 event" in _combined(r1)

        # verify-bundle takes no config; invoke directly.
        r2 = _runner().invoke(cli, ["audit", "verify-bundle", str(bundle)])
        assert r2.exit_code == 0, _combined(r2)
        assert "authentic" in _combined(r2).lower()

    def test_verify_detects_tampering(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        _write_log(log)
        bundle = tmp_path / "bundle.json"
        _run(["audit", "export", "--out", str(bundle)], str(log))

        # Tamper with an event after export.
        data = json.loads(bundle.read_text())
        data["events"][0]["detail"] = {"injected": "evil"}
        bundle.write_text(json.dumps(data))

        r = _runner().invoke(cli, ["audit", "verify-bundle", str(bundle)])
        assert r.exit_code == 1
        assert "FAILED" in _combined(r)

    def test_category_filter_flag(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        _write_log(log)
        bundle = tmp_path / "bundle.json"
        r = _run(["audit", "export", "--out", str(bundle), "--category", "security"], str(log))
        assert r.exit_code == 0, _combined(r)
        assert "Exported 1 event" in _combined(r)

    def test_verify_missing_bundle_file(self) -> None:
        r = _runner().invoke(cli, ["audit", "verify-bundle", "/nonexistent/bundle.json"])
        assert r.exit_code == 1
        assert "Could not read bundle" in _combined(r)
