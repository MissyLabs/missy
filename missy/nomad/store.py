"""Private, atomic operation journal for Nomad plans and owned jobs."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from missy.nomad.errors import NomadConfigurationError, NomadValidationError

_MAX_STATE_BYTES = 16 * 1024 * 1024


class NomadStateStore:
    """Persist exact plans and non-secret workload provenance with mode 0600."""

    def __init__(self, state_dir: str) -> None:
        self.directory = Path(state_dir).expanduser().resolve()
        self.path = self.directory / "operations.json"
        self.lock_path = self.directory / ".operations.lock"
        self._thread_lock = threading.RLock()

    def _prepare(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        if not self.lock_path.exists():
            fd = os.open(self.lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        os.chmod(self.lock_path, 0o600)

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": 1,
            "plans": {},
            "jobs": {},
            "schedules": {},
            "tasks": {},
            "benchmarks": {},
        }

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            size = self.path.stat().st_size
            if size > _MAX_STATE_BYTES:
                raise NomadConfigurationError("Nomad operation journal exceeds its size limit.")
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except NomadConfigurationError:
            raise
        except Exception as exc:
            raise NomadConfigurationError(
                "Nomad operation journal is unreadable or invalid."
            ) from exc
        if not isinstance(data, dict) or data.get("version") != 1:
            raise NomadConfigurationError("Nomad operation journal has an unsupported format.")
        # Version 1 predates task and benchmark indexes.  Adding empty maps is
        # a backward-compatible in-memory migration; the next mutation writes
        # the expanded shape atomically.
        data.setdefault("tasks", {})
        data.setdefault("benchmarks", {})
        for key in ("plans", "jobs", "schedules", "tasks", "benchmarks"):
            if not isinstance(data.get(key), dict):
                raise NomadConfigurationError(f"Nomad operation journal field {key!r} is invalid.")
        return data

    def read(self) -> dict[str, Any]:
        with self._thread_lock:
            self._prepare()
            with self.lock_path.open("r+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
                try:
                    return copy.deepcopy(self._read_unlocked())
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        encoded = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")
        if len(encoded) > _MAX_STATE_BYTES:
            raise NomadValidationError("Nomad operation journal would exceed its size limit.")
        fd, tmp_name = tempfile.mkstemp(prefix=".operations-", suffix=".tmp", dir=self.directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o600)
            with contextlib.suppress(OSError):
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)

    def mutate(self, callback: Callable[[dict[str, Any]], Any]) -> Any:
        with self._thread_lock:
            self._prepare()
            with self.lock_path.open("r+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    data = self._read_unlocked()
                    result = callback(data)
                    self._write_unlocked(data)
                    return result
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def put_plan(self, plan_id: str, record: dict[str, Any]) -> None:
        self.mutate(lambda data: data["plans"].__setitem__(plan_id, copy.deepcopy(record)))

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        record = self.read()["plans"].get(plan_id)
        if not isinstance(record, dict):
            raise KeyError(f"No Nomad plan found with id {plan_id!r}.")
        return record

    def mark_plan_submitted(self, plan_id: str, *, evaluation_id: str) -> None:
        def update(data: dict[str, Any]) -> None:
            record = data["plans"].get(plan_id)
            if not isinstance(record, dict):
                raise KeyError(f"No Nomad plan found with id {plan_id!r}.")
            record["submitted_at"] = datetime.now(tz=UTC).isoformat()
            record["evaluation_id"] = evaluation_id

        self.mutate(update)

    def put_job(self, namespace: str, job_id: str, record: dict[str, Any]) -> None:
        key = f"{namespace}/{job_id}"
        self.mutate(lambda data: data["jobs"].__setitem__(key, copy.deepcopy(record)))

    def get_job(self, namespace: str, job_id: str) -> dict[str, Any]:
        key = f"{namespace}/{job_id}"
        record = self.read()["jobs"].get(key)
        if not isinstance(record, dict):
            raise KeyError(f"No owned Nomad job record found for {key!r}.")
        return record

    def list_jobs(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(item) for item in self.read()["jobs"].values()]

    def put_schedule(self, schedule_id: str, record: dict[str, Any]) -> None:
        self.mutate(lambda data: data["schedules"].__setitem__(schedule_id, copy.deepcopy(record)))

    def get_schedule(self, schedule_id: str) -> dict[str, Any]:
        record = self.read()["schedules"].get(schedule_id)
        if not isinstance(record, dict):
            raise KeyError(f"No Nomad schedule found with id {schedule_id!r}.")
        return record

    def list_schedules(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(item) for item in self.read()["schedules"].values()]

    def put_task(self, task_id: str, record: dict[str, Any]) -> None:
        self.mutate(lambda data: data["tasks"].__setitem__(task_id, copy.deepcopy(record)))

    def get_task(self, task_id: str) -> dict[str, Any]:
        record = self.read()["tasks"].get(task_id)
        if not isinstance(record, dict):
            raise KeyError(f"No Nomad offload task found with id {task_id!r}.")
        return record

    def put_benchmark(self, benchmark_id: str, record: dict[str, Any]) -> None:
        self.mutate(
            lambda data: data["benchmarks"].__setitem__(benchmark_id, copy.deepcopy(record))
        )

    def get_benchmark(self, benchmark_id: str) -> dict[str, Any]:
        record = self.read()["benchmarks"].get(benchmark_id)
        if not isinstance(record, dict):
            raise KeyError(f"No Nomad benchmark found with id {benchmark_id!r}.")
        return record
