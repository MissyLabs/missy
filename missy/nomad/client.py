"""Structured wrapper around the Nomad CLI.

Every command is constructed from fixed argv elements.  The ACL token is
provided only through a minimal subprocess environment and is never included in
arguments, stdout, exceptions, or audit data.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from missy.config.settings import NomadConfig
from missy.nomad.credentials import (
    NomadCredentials,
    build_cli_environment,
    load_credentials,
    resolve_binary,
)
from missy.nomad.errors import (
    NomadAuthorizationError,
    NomadCommandError,
    NomadMutationUnknown,
)
from missy.security.censor import censor_response

_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class NomadCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class NomadClient:
    """Execute bounded Nomad CLI operations with one validated identity."""

    def __init__(
        self,
        config: NomadConfig,
        *,
        credentials: NomadCredentials | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.config = config
        self.credentials = credentials or load_credentials(config)
        self.binary = resolve_binary(config)
        self.environment = build_cli_environment(config, self.credentials)
        self._runner = runner or subprocess.run

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not _ID_RE.fullmatch(str(value or "")):
            raise NomadCommandError(f"Invalid Nomad {label}.")
        return str(value)

    def _run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        allowed_returncodes: set[int] | None = None,
        mutation: bool = False,
        timeout: int | None = None,
    ) -> NomadCommandResult:
        allowed = allowed_returncodes or {0}
        argv = [self.binary, *args]
        try:
            completed = self._runner(
                argv,
                input=input_text,
                stdin=subprocess.DEVNULL if input_text is None else None,
                text=True,
                capture_output=True,
                timeout=timeout or self.config.request_timeout_seconds,
                env=dict(self.environment),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if mutation:
                raise NomadMutationUnknown(
                    "Nomad mutation timed out; its server-side effect is unknown and must be reconciled."
                ) from exc
            raise NomadCommandError("Nomad read-only command timed out.") from exc
        except OSError as exc:
            raise NomadCommandError(f"Nomad CLI could not be executed: {exc}") from exc

        stdout = str(completed.stdout or "")
        stderr = str(completed.stderr or "")
        if len(stdout.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
            raise NomadCommandError("Nomad CLI stdout exceeded the safe response limit.")
        if len(stderr.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
            stderr = stderr[:_MAX_OUTPUT_BYTES] + "\n[stderr truncated]"
        result = NomadCommandResult(tuple(argv), completed.returncode, stdout, stderr)
        if completed.returncode not in allowed:
            message = censor_response(stderr.strip() or stdout.strip() or "Nomad command failed")
            message = message[:1000]
            if "403" in message or "permission denied" in message.lower():
                capability = " ".join(str(item) for item in args[:2])
                namespace = next(
                    (
                        str(item).split("=", 1)[1]
                        for item in args
                        if str(item).startswith("-namespace=")
                    ),
                    "cluster-visible scope",
                )
                raise NomadAuthorizationError(
                    f"Nomad denied capability {capability!r} in {namespace!r}: {message}"
                )
            raise NomadCommandError(f"Nomad command failed ({completed.returncode}): {message}")
        return result

    @staticmethod
    def _json(result: NomadCommandResult) -> Any:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise NomadCommandError("Nomad CLI returned invalid JSON.") from exc

    @staticmethod
    def _namespace_args(namespace: str) -> list[str]:
        return [f"-namespace={NomadClient._identifier(namespace, 'namespace')}"]

    def namespaces(self) -> list[dict[str, Any]]:
        value = self._json(self._run(["namespace", "list", "-json"]))
        return value if isinstance(value, list) else []

    def node_pools(self) -> list[dict[str, Any]]:
        value = self._json(self._run(["node", "pool", "list", "-json"]))
        return value if isinstance(value, list) else []

    def nodes(self) -> list[dict[str, Any]]:
        value = self._json(self._run(["node", "status", "-json"]))
        return value if isinstance(value, list) else []

    def node(self, node_id: str, *, stats: bool = False) -> dict[str, Any]:
        args = ["node", "status", "-json"]
        if stats:
            args.append("-stats")
        args.append(self._identifier(node_id, "node id"))
        value = self._json(self._run(args))
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad node status returned an unexpected value.")
        return value

    def node_allocations(self, node_id: str) -> list[dict[str, Any]]:
        safe_id = self._identifier(node_id, "node id")
        value = self._json(self._run(["operator", "api", f"/v1/node/{safe_id}/allocations"]))
        return value if isinstance(value, list) else []

    def node_stats(self, node_id: str) -> dict[str, Any]:
        safe_id = self._identifier(node_id, "node id")
        value = self._json(self._run(["operator", "api", f"/v1/client/stats?node_id={safe_id}"]))
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad node statistics returned an unexpected value.")
        return value

    def jobs(self, namespace: str) -> list[dict[str, Any]]:
        value = self._json(
            self._run(["operator", "api", *self._namespace_args(namespace), "/v1/jobs"])
        )
        return value if isinstance(value, list) else []

    def inspect_job(self, namespace: str, job_id: str) -> dict[str, Any]:
        value = self._json(
            self._run(
                [
                    "job",
                    "inspect",
                    "-json",
                    *self._namespace_args(namespace),
                    self._identifier(job_id, "job id"),
                ]
            )
        )
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad job inspect returned an unexpected value.")
        return value.get("Job", value) if isinstance(value.get("Job", value), dict) else value

    def job_status(self, namespace: str, job_id: str) -> dict[str, Any]:
        value = self._json(
            self._run(
                [
                    "job",
                    "status",
                    "-json",
                    *self._namespace_args(namespace),
                    self._identifier(job_id, "job id"),
                ]
            )
        )
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad job status returned an unexpected value.")
        return value

    def allocations(self, namespace: str, job_id: str) -> list[dict[str, Any]]:
        value = self._json(
            self._run(
                [
                    "job",
                    "allocs",
                    "-json",
                    *self._namespace_args(namespace),
                    self._identifier(job_id, "job id"),
                ]
            )
        )
        return value if isinstance(value, list) else []

    def deployments(self, namespace: str, job_id: str) -> list[dict[str, Any]]:
        value = self._json(
            self._run(
                [
                    "job",
                    "deployments",
                    "-json",
                    *self._namespace_args(namespace),
                    self._identifier(job_id, "job id"),
                ]
            )
        )
        return value if isinstance(value, list) else []

    def evaluation(self, namespace: str, evaluation_id: str) -> dict[str, Any]:
        value = self._json(
            self._run(
                [
                    "eval",
                    "status",
                    "-json",
                    *self._namespace_args(namespace),
                    self._identifier(evaluation_id, "evaluation id"),
                ]
            )
        )
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad evaluation status returned an unexpected value.")
        return value

    def allocation(self, namespace: str, allocation_id: str) -> dict[str, Any]:
        value = self._json(
            self._run(
                [
                    "alloc",
                    "status",
                    "-json",
                    *self._namespace_args(namespace),
                    self._identifier(allocation_id, "allocation id"),
                ]
            )
        )
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad allocation status returned an unexpected value.")
        return value

    def allocation_logs(
        self,
        namespace: str,
        allocation_id: str,
        task: str,
        *,
        stderr: bool = False,
        lines: int = 100,
    ) -> str:
        bounded_lines = min(max(int(lines), 1), 500)
        args = [
            "alloc",
            "logs",
            *self._namespace_args(namespace),
            "-stderr" if stderr else "-stdout",
            "-tail",
            "-n",
            str(bounded_lines),
            self._identifier(allocation_id, "allocation id"),
            self._identifier(task, "task name"),
        ]
        result = self._run(args, timeout=min(self.config.request_timeout_seconds, 30))
        return censor_response(result.stdout[:262_144])

    def validate_job(self, namespace: str, spec: dict[str, Any]) -> str:
        payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        result = self._run(
            ["job", "validate", "-json", *self._namespace_args(namespace), "-"],
            input_text=payload,
        )
        return censor_response(result.stdout.strip())

    def plan_job(self, namespace: str, spec: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        result = self._run(
            [
                "job",
                "plan",
                "-json",
                "-json-output",
                *self._namespace_args(namespace),
                "-",
            ],
            input_text=payload,
            allowed_returncodes={0, 1},
        )
        value = self._json(result)
        if not isinstance(value, dict):
            raise NomadCommandError("Nomad job plan returned an unexpected value.")
        value["_exit_code"] = result.returncode
        return value

    def run_job(self, namespace: str, spec: dict[str, Any], *, check_index: int) -> str:
        payload = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        result = self._run(
            [
                "job",
                "run",
                "-json",
                "-detach",
                f"-check-index={int(check_index)}",
                *self._namespace_args(namespace),
                "-",
            ],
            input_text=payload,
            mutation=True,
        )
        evaluation_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
        if not evaluation_id or not _ID_RE.fullmatch(evaluation_id):
            raise NomadMutationUnknown(
                "Nomad accepted a run command but returned no usable evaluation id; reconcile the job."
            )
        return evaluation_id

    def restart_job(self, namespace: str, job_id: str) -> str:
        result = self._run(
            [
                "job",
                "restart",
                "-yes",
                *self._namespace_args(namespace),
                self._identifier(job_id, "job id"),
            ],
            mutation=True,
        )
        return censor_response(result.stdout.strip())

    def scale_job(self, namespace: str, job_id: str, group: str, count: int) -> str:
        result = self._run(
            [
                "job",
                "scale",
                *self._namespace_args(namespace),
                self._identifier(job_id, "job id"),
                self._identifier(group, "group name"),
                str(int(count)),
            ],
            mutation=True,
        )
        return censor_response(result.stdout.strip())

    def stop_job(self, namespace: str, job_id: str, *, purge: bool = False) -> str:
        args = ["job", "stop", "-detach", *self._namespace_args(namespace)]
        if purge:
            args.append("-purge")
        args.append(self._identifier(job_id, "job id"))
        result = self._run(args, mutation=True)
        return censor_response(result.stdout.strip())
