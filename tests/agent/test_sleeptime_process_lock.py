"""RATE-04: at most one SleeptimeWorker per memory store, across processes."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import patch

from missy.agent.sleeptime import SleeptimeConfig, SleeptimeWorker
from missy.memory.sqlite_store import SQLiteMemoryStore


def _worker(db):
    return SleeptimeWorker(
        config=SleeptimeConfig(enabled=True, check_interval_seconds=3600),
        memory_store=SQLiteMemoryStore(str(db)),
    )


def _start_in_subprocess(db) -> str:
    script = textwrap.dedent(
        f"""
        from missy.agent.sleeptime import SleeptimeConfig, SleeptimeWorker
        from missy.memory.sqlite_store import SQLiteMemoryStore
        w = SleeptimeWorker(
            config=SleeptimeConfig(enabled=True, check_interval_seconds=3600),
            memory_store=SQLiteMemoryStore({str(db)!r}),
        )
        print(w.start())
        w.stop()
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr


def test_second_process_is_refused(tmp_path):
    db = tmp_path / "memory.db"
    assert _start_in_subprocess(db) == "True"  # control: free store starts
    holder = _worker(db)
    assert holder.start() is True
    try:
        assert _start_in_subprocess(db) == "False"
    finally:
        holder.stop()


def test_lock_released_on_stop(tmp_path):
    db = tmp_path / "memory.db"
    first = _worker(db)
    assert first.start() is True
    first.stop()
    second = _worker(db)
    try:
        assert second.start() is True
    finally:
        second.stop()


def test_refusal_is_audited(tmp_path):
    db = tmp_path / "memory.db"
    worker = _worker(db)
    with (
        patch.object(SleeptimeWorker, "_acquire_process_lock", return_value=False),
        patch("missy.core.events.event_bus.publish") as publish,
    ):
        assert worker.start() is False
    assert publish.call_args.args[0].event_type == "sleeptime.start_refused"
