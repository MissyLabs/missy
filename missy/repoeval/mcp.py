"""Constrained MCP-style tool facade; transport/server integration is external."""

from __future__ import annotations

from typing import Any


class MCPError(ValueError):
    def __init__(self, category: str, message: str):
        self.category = category
        super().__init__(message)


class FoundryMCP:
    """Expose only bounded read, planning, execution, and draft tools."""

    _BACKEND_ERRORS = {
        "authorization": ("forbidden", "Operation is not permitted"),
        "forbidden": ("forbidden", "Operation is not permitted"),
        "missing": ("not_found", "Resource not found"),
        "not_found": ("not_found", "Resource not found"),
        "invalid_input": ("invalid_request", "Request is invalid"),
        "invalid_request": ("invalid_request", "Request is invalid"),
        "policy": ("policy", "Operation is not permitted by policy"),
        "quota": ("quota", "Project quota exceeded"),
        "conflict": ("conflict", "Resource state conflicts with request"),
        "unavailable": ("unavailable", "Required service capability is unavailable"),
        "evidence": ("evidence", "Required verification evidence is unavailable"),
    }
    TOOL_NAMES = frozenset(
        {
            "repoeval_repositories",
            "repoeval_snapshot_start",
            "repoeval_snapshot_status",
            "repoeval_benchmark_plan",
            "repoeval_benchmark_start",
            "repoeval_run_status",
            "repoeval_cancel",
            "repoeval_artifacts",
            "repoeval_compare",
            "repoeval_report",
        }
    )

    def __init__(self, service: Any, principal: Any):
        self.service, self.principal = service, principal

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self.TOOL_NAMES:
            raise MCPError("unsupported_operation", "Tool is not exposed; mutation refused")
        if not isinstance(arguments, dict):
            raise MCPError("invalid_request", "Arguments must be an object")
        expected = {tool["name"]: set(tool["input_schema"]["properties"]) for tool in self.tools()}
        if set(arguments) != expected[name]:
            raise MCPError("invalid_request", "Arguments do not match tool schema")
        p = self.principal
        try:
            if name == "repoeval_repositories":
                return self.service.list_repositories(p)
            if name == "repoeval_benchmark_plan":
                return self.service.benchmark_plan(p, arguments["workload"])
            if name == "repoeval_benchmark_start":
                return self.service.benchmark_start(
                    p, arguments["plan_id"], arguments["idempotency_key"]
                )
            if name == "repoeval_run_status":
                return self.service.run_status(p, arguments["run_id"])
            if name == "repoeval_cancel":
                if not arguments.get("idempotency_key"):
                    raise MCPError("invalid_request", "idempotency_key is required")
                return self.service.run_cancel(p, arguments["run_id"])
            if name == "repoeval_snapshot_start":
                return self.service.snapshot_start(
                    p,
                    arguments["repository_id"],
                    arguments["commit_sha"],
                    arguments["idempotency_key"],
                )
            if name == "repoeval_snapshot_status":
                return self.service.snapshot_status(p, arguments["snapshot_id"])
            if name == "repoeval_compare":
                return self.service.compare_runs(p, arguments["run_ids"])
            if name == "repoeval_artifacts":
                return self.service.artifacts(p, arguments["resource_id"])
            if name == "repoeval_report":
                return self.service.report_draft(p, arguments["run_ids"])
        except Exception as exc:
            # Backend messages and arbitrary backend category values are not
            # safe to expose, including when the backend raised MCPError.
            try:
                backend_category = getattr(exc, "category", None)
            except Exception:
                backend_category = None
            category, message = self._BACKEND_ERRORS.get(
                backend_category if isinstance(backend_category, str) else None,
                ("service_failure", "Application service failed"),
            )
            raise MCPError(category, message) from None
        raise MCPError("unsupported_operation", "Tool is not exposed")

    def tools(self) -> list[dict[str, Any]]:
        """Small explicit schemas; unknown parameters are not forwarded."""
        fields = {
            "repoeval_repositories": {},
            "repoeval_snapshot_start": {
                "repository_id": "string",
                "commit_sha": "string",
                "idempotency_key": "string",
            },
            "repoeval_snapshot_status": {"snapshot_id": "string"},
            "repoeval_benchmark_plan": {"workload": "object"},
            "repoeval_benchmark_start": {"plan_id": "string", "idempotency_key": "string"},
            "repoeval_run_status": {"run_id": "string"},
            "repoeval_cancel": {"run_id": "string", "idempotency_key": "string"},
            "repoeval_artifacts": {"resource_id": "string"},
            "repoeval_compare": {"run_ids": "array"},
            "repoeval_report": {"run_ids": "array"},
        }
        return [
            {
                "name": n,
                "input_schema": {
                    "type": "object",
                    "properties": f,
                    "required": list(f),
                    "additionalProperties": False,
                },
            }
            for n, f in fields.items()
        ]
