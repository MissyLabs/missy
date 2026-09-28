"""Framework-neutral authenticated HTTP API facade. No server is launched."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, Any]


class APIError(Exception):
    def __init__(self, status: int, category: str, message: str, retriable: bool = False):
        self.status, self.category, self.message, self.retriable = (
            status,
            category,
            message,
            retriable,
        )
        super().__init__(message)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


class FoundryAPI:
    """An authenticated route table over an injected FoundryService."""

    def __init__(self, service: Any, authenticate: Callable[[Mapping[str, str]], Any]):
        self.service, self.authenticate = service, authenticate

    def handle(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: Mapping[str, Any] | None = None,
    ) -> Response:
        try:
            principal = self.authenticate(headers)
            if principal is None or not _get(principal, "subject"):
                raise APIError(401, "unauthenticated", "Authentication required")
            method = method.upper()
            parts = [p for p in path.split("?")[0].split("/") if p]
            if parts and parts[0] in ("api", "v1"):
                parts = parts[1:]
            if len(parts) < 2 or parts[0] != "projects":
                raise APIError(404, "not_found", "Unknown API route")
            project, tail = parts[1], parts[2:]
            if project != _get(principal, "project_id"):
                raise APIError(403, "forbidden", "Project is outside caller scope")
            data = dict(body or {})
            permissions = _get(principal, "permissions", ())
            if not isinstance(permissions, (set, frozenset, list, tuple)):
                permissions = ()
            op = self._route(method, tail)
            if op is None:
                raise APIError(404, "not_found", "Unknown API route")
            if op in ("snapshot_start", "benchmark_start", "cancel"):
                permission = {
                    "snapshot_start": "snapshot:start",
                    "benchmark_start": "execute",
                    "cancel": "execute",
                }[op]
                # The core service uses `execute`; allow explicit compatible
                # spelling for cancel/snapshot without ever deriving permission.
                if permission not in permissions:
                    raise APIError(403, "forbidden", "Operation is not permitted")
                idem = self._header(headers, "Idempotency-Key")
                if not idem or len(idem) > 128:
                    raise APIError(400, "invalid_request", "A valid Idempotency-Key is required")
            result = self._dispatch(op, principal, tail, data, headers)
            status = 202 if op in ("snapshot_start", "benchmark_start", "cancel") else 200
            return Response(status, {"ok": True, "data": result})
        except APIError as e:
            return Response(
                e.status,
                {
                    "ok": False,
                    "error": {
                        "category": e.category,
                        "message": e.message,
                        "retriable": e.retriable,
                    },
                },
            )
        except KeyError:
            return Response(
                400,
                {
                    "ok": False,
                    "error": {
                        "category": "invalid_request",
                        "message": "A required field is missing",
                        "retriable": False,
                    },
                },
            )
        except Exception as e:
            category = getattr(e, "category", None)
            if category in ("authorization", "forbidden"):
                return Response(
                    403,
                    {
                        "ok": False,
                        "error": {
                            "category": "forbidden",
                            "message": "Operation is not permitted",
                            "retriable": False,
                        },
                    },
                )
            if category in ("missing", "not_found"):
                return Response(
                    404,
                    {
                        "ok": False,
                        "error": {
                            "category": "not_found",
                            "message": "Resource not found",
                            "retriable": False,
                        },
                    },
                )
            if category in (
                "invalid_input",
                "quota",
                "policy",
                "conflict",
                "unavailable",
                "evidence",
            ):
                if category == "unavailable":
                    return Response(
                        503,
                        {
                            "ok": False,
                            "error": {
                                "category": "unavailable",
                                "message": "Required service capability is unavailable",
                                "retriable": True,
                            },
                        },
                    )
                messages = {
                    "invalid_input": "Request input is invalid",
                    "quota": "Request exceeds an allowed limit",
                    "policy": "Request is not allowed by policy",
                    "conflict": "Request conflicts with the current resource state",
                    "evidence": "Required evidence is unavailable or invalid",
                }
                return Response(
                    400 if category != "conflict" else 409,
                    {
                        "ok": False,
                        "error": {
                            "category": category,
                            "message": messages[category],
                            "retriable": False,
                        },
                    },
                )
            return Response(
                503,
                {
                    "ok": False,
                    "error": {
                        "category": "service_failure",
                        "message": "Application service failed",
                        "retriable": True,
                    },
                },
            )

    @staticmethod
    def _header(headers: Mapping[str, str], key: str) -> str | None:
        return next((v for k, v in headers.items() if k.lower() == key.lower()), None)

    @staticmethod
    def _route(method: str, tail: list[str]) -> str | None:
        if tail in ([], ["repositories"]) and method == "GET":
            return "repositories"
        if tail == ["snapshots"] and method == "POST":
            return "snapshot_start"
        if len(tail) == 2 and tail[0] == "snapshots" and method == "GET":
            return "snapshot_status"
        if tail == ["benchmark", "plan"] and method == "POST":
            return "benchmark_plan"
        if tail == ["benchmark", "start"] and method == "POST":
            return "benchmark_start"
        if len(tail) == 2 and tail[0] == "runs" and method == "GET":
            return "run_status"
        if len(tail) == 3 and tail[0] == "runs" and tail[2] == "cancel" and method == "POST":
            return "cancel"
        if tail == ["compare"] and method == "POST":
            return "compare"
        if len(tail) == 3 and tail[0] == "runs" and tail[2] == "artifacts" and method == "GET":
            return "artifacts"
        if tail == ["report"] and method == "POST":
            return "report"
        return None

    def _dispatch(
        self,
        op: str,
        principal: Any,
        tail: list[str],
        data: dict[str, Any],
        headers: Mapping[str, str],
    ) -> Any:
        idem = self._header(headers, "Idempotency-Key")
        if op == "repositories":
            return self.service.list_repositories(principal)
        if op == "benchmark_plan":
            return self.service.benchmark_plan(principal, data.get("workload", data))
        if op == "benchmark_start":
            return self.service.benchmark_start(principal, data["plan_id"], idem)
        if op == "run_status":
            return self.service.run_status(principal, tail[1])
        # Cancellation is state-idempotent (canceling an already-canceled run
        # has the same outcome), not keyed: the required HTTP idempotency
        # header is validated at the boundary but is not persisted or replayed.
        if op == "cancel":
            return self.service.run_cancel(principal, tail[1])
        if op == "snapshot_start":
            return self.service.snapshot_start(
                principal, data["repository_id"], data["commit_sha"], idem
            )
        if op == "snapshot_status":
            return self.service.snapshot_status(principal, tail[1])
        if op == "compare":
            return self.service.compare_runs(principal, data["run_ids"])
        if op == "artifacts":
            return self.service.artifacts(principal, tail[1])
        if op == "report":
            return self.service.report_draft(principal, data["run_ids"])
        raise APIError(404, "not_found", "Unknown operation")
