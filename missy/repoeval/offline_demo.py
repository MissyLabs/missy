"""Local-only wiring exercise. A passing fixture score is never a verified run.

Usage: python -m missy.repoeval.offline_demo --checkout /path/to/clean/git/tree \
    --commit FULL_HEAD_SHA

This module does not launch a server, submit jobs, execute repository content,
write artifacts, or contact a provider. Only the scanner runs fixed Git reads.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .api import FoundryAPI
from .contracts import sha256_json
from .control import FoundryService, Principal
from .evaluation import evaluate_tool_call
from .mcp import FoundryMCP, MCPError
from .placement import PoolCapacity, ResourceEnvelope
from .planning import CapacitySnapshot, PlanningAuthority, ProjectPolicySnapshot
from .provider import AdapterRequest, AdapterResponse, ProviderBroker, ProviderRequest, ProviderSpec
from .scanner import ScanRefused, scan_repository

CASE = Path(__file__).resolve().parent / "workloads/missy/tool-call-correctness"
OFFLINE_IMAGE = "registry.invalid/repoeval/offline-fixture@sha256:" + "1" * 64
PROJECT = "offline-fixture"
PROVIDER = "offline-fake"
MODEL = "fixture-v1"


class FakeAdapter:
    """Only a constant checked-in fixture response; no SDK or network access."""

    def __init__(self, content: str, *, fail_timeout: bool = False):
        self.content = content
        self.fail_timeout = fail_timeout
        self.calls = 0

    def complete(self, request: AdapterRequest, credential: str) -> AdapterResponse:
        self.calls += 1
        if (
            request.registry_key != PROVIDER
            or request.model != MODEL
            or credential != "fixture-only"
        ):
            raise ValueError("Offline fixture request identity mismatch")
        if self.fail_timeout:
            raise TimeoutError("Synthetic fixture timeout")
        return AdapterResponse(
            content=self.content, actual_model=MODEL, request_id="fixture-response"
        )


def _fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Load trusted catalog inputs only after checking their declared digests."""
    workload = json.loads((CASE / "workload.json").read_text(encoding="utf-8"))
    task = workload["task"]
    prompt = (CASE / "prompt.md").read_bytes()
    if hashlib.sha256(prompt).hexdigest() != task["prompt_sha256"]:
        raise ValueError("Catalog prompt digest mismatch")
    for name, expected in task["fixture_digests"].items():
        path = CASE / "fixtures" / name
        if path.name != name or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Catalog fixture digest mismatch")
    tool = json.loads((CASE / "fixtures/calculator-tool.json").read_text(encoding="utf-8"))
    case = json.loads((CASE / "fixtures/case.json").read_text(encoding="utf-8"))
    oracle = json.loads((CASE / "fixtures/oracle.json").read_text(encoding="utf-8"))
    return workload, tool, case, oracle["version"]


def run_offline_demo(
    checkout: str | Path, commit_sha: str, *, adapter: FakeAdapter | None = None
) -> dict[str, Any]:
    """Scan an explicitly selected clean checkout, then exercise isolated facades.

    The catalog's Missy SHA is NOT transplanted onto the checkout. A temporary,
    trusted *fixture-only* snapshot registration is built for this exact scan.
    This illustrates plan policy plumbing, not an independently verified
    production snapshot or approval for the placeholder image.
    """
    if adapter is not None and type(adapter) is not FakeAdapter:
        raise TypeError("Only exact offline FakeAdapter instances are accepted")
    result: dict[str, Any] = {
        "mode": "offline-fixture",
        "benchmark_state": "incomplete",
        "scheduler_called": False,
        "provider_network_called": False,
        "repository_content_executed": False,
        "manifest_persisted": False,
    }
    try:
        scan = scan_repository(checkout, commit_sha)
    except ScanRefused:
        result["scan"] = {"status": "refused", "reason": "checkout_not_verified"}
        result["plan"] = {"status": "blocked", "reason": "scan_required"}
        return result
    result["scan"] = {
        "status": "scanned",
        "commit_sha": scan["repository"]["commit_sha"],
        "file_count": scan["inventory"]["file_count"],
        "evidence_sha256": scan["provenance"]["evidence_sha256"],
        "offline": scan["provenance"]["offline"],
        "executed_repository_content": scan["provenance"]["executed_repository_content"],
    }

    workload, tool, case, _oracle_version = _fixture()
    result["catalog"] = {
        "id": workload["id"],
        "source_commit": workload["repository"]["commit_sha"],
        "prompt_and_fixtures_sha256_checked": True,
        "catalog_provider_targets": len(workload["providers"]),
    }
    # Catalog drafts cannot be planned as-is: missing provider and an
    # unapproved image. This derived, isolated workload is NOT the Missy run.
    fixture_workload = copy.deepcopy(workload)
    fixture_workload["id"] = "offline.fixture-tool-call.v1"
    fixture_workload["description"] = (
        "Synthetic local scan and fixture-only adapter; not a Missy benchmark"
    )
    fixture_workload["repository"] = {
        "repository_id": "local/offline-fixture",
        "commit_sha": commit_sha,
        "snapshot_id": "snapshot-local-" + commit_sha[:24],
    }
    fixture_workload["providers"] = [{"registry_key": PROVIDER, "model": MODEL, "settings": {}}]
    fixture_workload["sandbox"]["image_digest"] = OFFLINE_IMAGE
    principal = Principal("offline-operator", PROJECT, frozenset({"read", "execute"}))
    validator = fixture_workload["validation"]["validators"][0]
    observed_at = time.time()
    # Explicit synthetic fixture facts, never a live placement attestation.
    planning_authority = PlanningAuthority(
        lambda: CapacitySnapshot(
            "synthetic-offline-capacity-v1",
            observed_at,
            (PoolCapacity("staging", 4000, 8192, 16384),),
        ),
        lambda project: ProjectPolicySnapshot(
            PROJECT,
            "synthetic-offline-policy-v1",
            observed_at,
            frozenset({"local/offline-fixture"}),
            frozenset({OFFLINE_IMAGE}),
            {PROVIDER: frozenset({MODEL})},
            ResourceEnvelope(500, 512, 1024),
            10_000_000,
            frozenset({"offline"}),
            "synthetic-audit-fixture",
        ),
        # Fixed fixture clock keeps the API and MCP plans identical; not a
        # production clock, and this service cannot dispatch by construction.
        clock=lambda: observed_at,
    )
    service = FoundryService(
        repositories={PROJECT: {"local/offline-fixture"}},
        providers={PROVIDER},
        limits={
            "cpu_mhz": 500,
            "memory_mb": 512,
            "disk_mb": 1024,
            "repetitions": 1,
            "parallelism": 1,
            "timeout_seconds": 120,
            "network_policies": ("offline",),
        },
        verified_snapshots={
            fixture_workload["repository"]["snapshot_id"]: {
                "id": fixture_workload["repository"]["snapshot_id"],
                "project_id": PROJECT,
                "repository_id": "local/offline-fixture",
                "commit_sha": commit_sha,
                "state": "verified",
            }
        },
        approved_images={OFFLINE_IMAGE},
        approved_models={PROVIDER: {MODEL}},
        approved_prompts={workload["task"]["prompt_sha256"]},
        approved_fixtures=workload["task"]["fixture_digests"],
        approved_validators={
            fixture_workload["validation"]["evaluator_version"]: {
                (validator["id"], validator["version"], sha256_json(validator["parameters"]))
            }
        },
        dispatcher=None,
        allow_demo_dispatch=False,
        planning_authority=planning_authority,
    )
    # Fixture authentication exercises API refusal without impersonating a
    # production identity provider. This header is not a real credential.
    api = FoundryAPI(
        service, lambda headers: principal if headers.get("X-Offline-Fixture") == "yes" else None
    )
    mcp = FoundryMCP(service, principal)
    path = f"/api/projects/{PROJECT}"
    denied = api.handle("POST", path + "/benchmark/plan", {}, {"workload": fixture_workload})
    result["authentication"] = {
        "missing_fixture_header_status": denied.status,
        "production_authentication": False,
    }
    if denied.status != 401:
        raise RuntimeError("Offline authentication gate did not fail closed")
    headers = {"X-Offline-Fixture": "yes"}
    catalog_attempt = api.handle("POST", path + "/benchmark/plan", headers, {"workload": workload})
    result["catalog"]["as_is_plan_refused"] = catalog_attempt.status != 200
    if not result["catalog"]["as_is_plan_refused"]:
        raise RuntimeError("Catalog draft was unexpectedly accepted")
    planned = api.handle("POST", path + "/benchmark/plan", headers, {"workload": fixture_workload})
    if planned.status != 200:
        result["plan"] = {"status": "refused", "category": planned.body["error"]["category"]}
        return result
    plan_id = planned.body["data"]["id"]
    mcp_plan = mcp.call("repoeval_benchmark_plan", {"workload": fixture_workload})
    result["plan"] = {
        "status": "planned",
        "id": plan_id,
        "api_mcp_agree": plan_id == mcp_plan["id"],
        "scope": "synthetic-fixture-only",
        "independent_snapshot_attestation": False,
        "image_available_for_execution": False,
    }

    # An execute-capable principal cannot start this plan: there is no
    # dispatcher, outbox, scheduler, or actual approved executable image.
    blocked = api.handle(
        "POST",
        path + "/benchmark/start",
        {**headers, "Idempotency-Key": "offline-demo-01"},
        {"plan_id": plan_id},
    )
    try:
        mcp.call(
            "repoeval_benchmark_start", {"plan_id": plan_id, "idempotency_key": "offline-demo-01"}
        )
        mcp_category = "unexpected_success"
    except MCPError as exc:
        mcp_category = exc.category
    result["execution"] = {
        "status": "blocked"
        if blocked.status == 503 and mcp_category == "unavailable"
        else "unexpected",
        "api_status": blocked.status,
        "api_category": blocked.body.get("error", {}).get("category"),
        "mcp_category": mcp_category,
        "run_created": False,
    }
    if result["execution"]["status"] != "blocked":
        raise RuntimeError("Offline execution gate did not fail closed")

    expected_call = case["expected_call"]
    fake = (
        adapter if adapter is not None else FakeAdapter(json.dumps(expected_call, sort_keys=True))
    )
    broker = ProviderBroker(
        {PROVIDER: ProviderSpec(PROVIDER, "fixture-ref", frozenset({MODEL}))},
        {PROVIDER: fake},
        lambda _ref: "fixture-only",
        lambda project, key: project == PROJECT and key == PROVIDER,
    )
    request = ProviderRequest(
        messages=({"role": "user", "content": case["input"]},),
        settings={"seed": 1002},
        tools=({"name": tool["name"], "parameters": tool["parameters"]},),
    )
    response = broker.execute(
        project_id=PROJECT,
        registry_key=PROVIDER,
        model=MODEL,
        request=request,
        budget_microusd=0,
        timeout_seconds=1,
        token_cap=128,
    )
    if (
        response.error_category is not None
        or response.content is None
        or response.content_truncated
    ):
        result["provider"] = {
            "status": "failed",
            "category": response.error_category or "incomplete_response",
            "request_digest": response.request_digest,
            "adapter_kind": "injected-offline-only",
        }
        result["evaluation"] = {"status": "not_scored"}
    else:
        evaluation = evaluate_tool_call(response.content, tool)
        try:
            exact = json.loads(response.content) == expected_call
        except (ValueError, TypeError):
            exact = False
        result["provider"] = {
            "status": "fake_response",
            "request_digest": response.request_digest,
            "adapter_kind": "injected-offline-only",
        }
        result["evaluation"] = {
            "status": "fixture_scored",
            "validator_status": evaluation["status"],
            "exact_fixture_call": exact,
            "tool_executed": evaluation["facts"].get("executed", False),
            "benchmark_verified": False,
        }
    # Artifact manifests describe stored bytes. A fake response and a scanner
    # inventory are neither stored run artifacts nor scheduler evidence.
    result["artifacts"] = {
        "status": "unavailable",
        "manifest_persisted": False,
        "reason": "no_run_or_verified_artifact_store",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout", required=True, help="Explicit local clean Git checkout to scan"
    )
    parser.add_argument("--commit", required=True, help="Exact lowercase full HEAD commit SHA")
    args = parser.parse_args()
    print(json.dumps(run_offline_demo(args.checkout, args.commit), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
