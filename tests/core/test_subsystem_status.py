"""Gateway subsystem startup outcomes must be visible (premortem item 7)."""

from __future__ import annotations

import os

from missy.core.subsystem_status import (
    FAILED,
    RUNNING,
    SubsystemStatusRegistry,
    pid_is_alive,
    read_status_file,
)


def test_status_file_round_trips_with_owner_only_mode(tmp_path) -> None:
    registry = SubsystemStatusRegistry()
    registry.record("voice", FAILED, "refuses a plaintext non-loopback bind")
    registry.record("scheduler", RUNNING, "3 job(s)")
    path = str(tmp_path / "run" / "gateway_status.json")
    registry.write(path)

    status = read_status_file(path)
    assert status["pid"] == os.getpid()
    assert status["subsystems"]["voice"]["state"] == FAILED
    assert status["subsystems"]["scheduler"]["detail"] == "3 job(s)"
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert list(registry.failed()) == ["voice"]


def test_missing_or_corrupt_status_file_reads_as_none(tmp_path) -> None:
    assert read_status_file(str(tmp_path / "absent.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_status_file(str(bad)) is None


def test_pid_liveness() -> None:
    assert pid_is_alive(os.getpid())
    assert not pid_is_alive(0)
    assert not pid_is_alive("123")
    assert not pid_is_alive(2**22 + 12345)


def test_diagnostics_reports_failed_subsystem_as_error(monkeypatch) -> None:
    from missy.api import diagnostics

    registry = SubsystemStatusRegistry()
    registry.record("voice", FAILED, "bind refused")
    registry.record("web_console", RUNNING, "http://127.0.0.1:8080")
    monkeypatch.setattr("missy.core.subsystem_status._registry", registry)

    section = diagnostics._subsystems_section()
    assert section["status"] == "error"
    by_name = {check["name"]: check for check in section["checks"]}
    assert by_name["voice"]["status"] == "error"
    assert by_name["web_console"]["status"] == "ok"


def test_gateway_status_shows_live_subsystems(monkeypatch, capsys) -> None:
    from missy.cli import main as cli_main

    registry = SubsystemStatusRegistry()
    registry.record("voice", FAILED, "bind refused")
    registry.write()  # default path under the isolated test HOME

    cli_main._print_live_gateway_status()
    out = capsys.readouterr().out
    assert "running" in out and "voice" in out and "failed" in out
