from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from missy.config.settings import NomadConfig
from missy.nomad.credentials import NomadCredentials
from missy.nomad.errors import (
    NomadAuthorizationError,
    NomadCommandError,
    NomadOwnershipError,
    NomadValidationError,
)
from missy.nomad.manager import NomadManager
from missy.nomad.store import NomadStateStore

from .conftest import healthy_node


class FakeClient:
    def __init__(self, tmp_path: Path) -> None:
        expires = datetime.now(tz=UTC) + timedelta(days=30)
        placeholder = tmp_path / "placeholder"
        self.credentials = NomadCredentials(
            placeholder, placeholder, placeholder, placeholder, "secret", expires, "missy"
        )
        self.remote: dict[tuple[str, str], dict[str, Any]] = {}
        self.last_run: tuple[str, dict[str, Any], int] | None = None
        self.alloc_state = "running"
        self.logs = "benchmark complete\n"

    def nodes(self) -> list[dict[str, Any]]:
        return [{"ID": "node-1"}]

    def node(self, node_id: str) -> dict[str, Any]:
        assert node_id == "node-1"
        return healthy_node()

    def node_allocations(self, node_id: str) -> list[dict[str, Any]]:
        return []

    def node_stats(self, node_id: str) -> dict[str, Any]:
        return {"Memory": {"Available": 4_000_000_000}, "DiskStats": []}

    def namespaces(self) -> list[dict[str, Any]]:
        return [{"Name": "testing", "Description": "tests"}]

    def node_pools(self) -> list[dict[str, Any]]:
        return [{"Name": "staging", "Description": "tests"}]

    def jobs(self, namespace: str) -> list[dict[str, Any]]:
        return [
            {"ID": job_id, "Namespace": ns, **job}
            for (ns, job_id), job in self.remote.items()
            if ns == namespace
        ]

    def validate_job(self, namespace: str, spec: dict[str, Any]) -> str:
        assert namespace == "testing"
        assert spec["Job"]["Namespace"] == namespace
        return "Job validation successful"

    def plan_job(self, namespace: str, spec: dict[str, Any]) -> dict[str, Any]:
        key = (namespace, spec["Job"]["ID"])
        index = int(self.remote.get(key, {}).get("JobModifyIndex") or 0)
        return {
            "JobModifyIndex": index,
            "Diff": {"Type": "Added" if not index else "Edited"},
            "Annotations": {"DesiredTGUpdates": {"workload": {"Place": 1}}},
            "FailedTGAllocs": {},
            "_exit_code": 1,
        }

    def run_job(self, namespace: str, spec: dict[str, Any], *, check_index: int) -> str:
        self.last_run = (namespace, spec, check_index)
        job = dict(spec["Job"])
        job["Status"] = "running"
        job["JobModifyIndex"] = check_index + 1
        self.remote[(namespace, job["ID"])] = job
        return "eval-1"

    def inspect_job(self, namespace: str, job_id: str) -> dict[str, Any]:
        return self.remote[(namespace, job_id)]

    def job_status(self, namespace: str, job_id: str) -> dict[str, Any]:
        return self.remote[(namespace, job_id)]

    def allocations(self, namespace: str, job_id: str) -> list[dict[str, Any]]:
        return [
            {
                "ID": "alloc-1",
                "NodeID": "node-1",
                "TaskGroup": "workload",
                "DesiredStatus": "run",
                "ClientStatus": self.alloc_state,
                "TaskStates": {
                    "workload": {
                        "State": "dead" if self.alloc_state == "complete" else "running",
                        "Failed": self.alloc_state == "failed",
                        "Events": [{"Type": "Started", "DisplayMessage": "started"}],
                    }
                },
            }
        ]

    def deployments(self, namespace: str, job_id: str) -> list[dict[str, Any]]:
        return []

    def evaluation(self, namespace: str, evaluation_id: str) -> dict[str, Any]:
        return {"ID": evaluation_id, "Status": "complete", "FailedTGAllocs": {}}

    def allocation_logs(
        self,
        namespace: str,
        allocation_id: str,
        task: str,
        *,
        stderr: bool = False,
        lines: int = 100,
    ) -> str:
        return "" if stderr else self.logs

    def restart_job(self, namespace: str, job_id: str) -> str:
        return "restart requested"

    def stop_job(self, namespace: str, job_id: str, *, purge: bool = False) -> str:
        self.remote[(namespace, job_id)]["Status"] = "dead"
        return "purged" if purge else "stopped"


@pytest.fixture
def manager(nomad_config: NomadConfig, tmp_path: Path) -> tuple[NomadManager, FakeClient]:
    fake = FakeClient(tmp_path)
    store = NomadStateStore(nomad_config.state_dir)
    return NomadManager(nomad_config, client=fake, store=store), fake


def workload(pinned_image: str, **updates: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "purpose": "test benchmark",
        "image": pinned_image,
        "job_type": "batch",
        "command": "/bin/sh",
        "args": ["-c", "printf 'benchmark complete\\n'"],
        "expected_output": "benchmark complete",
        "architecture": "amd64",
        "idempotency_key": "bench-1",
    }
    data.update(updates)
    return data


def test_discovery_is_timestamped_and_structured(manager: tuple[NomadManager, FakeClient]) -> None:
    service, _ = manager
    result = service.discover(namespace="testing")
    assert result["identity"]["identity_cn"] == "missy"
    assert result["nodes"][0]["Drivers"]["docker"]["Healthy"] is True
    assert result["namespaces"][0]["name"] == "testing"


def test_plan_and_submit_exact_artifact(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, client = manager
    plan = service.plan(workload(pinned_image))
    assert plan["state"] == "planned_not_submitted"
    assert plan["plan"]["job_modify_index"] == 0
    submitted = service.submit(plan["plan_id"])
    assert submitted["state"] == "submitted_not_yet_verified"
    assert client.last_run is not None
    assert client.last_run[2] == 0
    assert client.last_run[1]["Job"]["ID"] == plan["job_id"]


def test_submit_rejects_tampered_stored_artifact(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image))
    record = service.store.get_plan(plan["plan_id"])
    record["spec"]["Job"]["Name"] = "tampered"
    service.store.put_plan(plan["plan_id"], record)
    with pytest.raises(NomadValidationError, match="integrity"):
        service.submit(plan["plan_id"])


def test_submit_requires_replan_after_concurrent_index_change(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, client = manager
    plan = service.plan(workload(pinned_image))
    record = service.store.get_plan(plan["plan_id"])
    client.remote[("testing", plan["job_id"])] = dict(record["spec"]["Job"])
    client.run_job = MagicMock(side_effect=NomadCommandError("check-index failed"))
    client.inspect_job = MagicMock(wraps=client.inspect_job)
    with pytest.raises(NomadValidationError, match="fresh plan is required"):
        service.submit(plan["plan_id"])
    client.inspect_job.assert_called_once_with("testing", plan["job_id"])


def test_plan_expires(manager: tuple[NomadManager, FakeClient], pinned_image: str) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image))
    record = service.store.get_plan(plan["plan_id"])
    record["expires_at"] = (datetime.now(tz=UTC) - timedelta(seconds=1)).isoformat()
    service.store.put_plan(plan["plan_id"], record)
    with pytest.raises(NomadValidationError, match="expired"):
        service.submit(plan["plan_id"])


def test_duplicate_idempotency_key_is_rejected(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image))
    service.submit(plan["plan_id"])
    with pytest.raises(NomadValidationError, match="idempotency"):
        service.plan(workload(pinned_image))


def test_result_requires_terminal_exit_and_expected_output(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, client = manager
    plan = service.plan(workload(pinned_image))
    submitted = service.submit(plan["plan_id"])
    client.alloc_state = "complete"
    result = service.result("testing", submitted["job_id"])
    assert result["complete"] is True
    assert result["expected_output_verified"] is True
    client.logs = "wrong output"
    assert service.result("testing", submitted["job_id"])["complete"] is False


def test_mutation_requires_local_and_remote_ownership(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, client = manager
    plan = service.plan(workload(pinned_image))
    submitted = service.submit(plan["plan_id"])
    client.remote[("testing", submitted["job_id"])]["Meta"]["tracking-id"] = "other"
    with pytest.raises(NomadOwnershipError, match="does not match"):
        service.action("testing", submitted["job_id"], "restart")


def test_purge_requires_double_opt_in(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image, disposable=True))
    submitted = service.submit(plan["plan_id"])
    with pytest.raises(NomadValidationError, match="allow_purge"):
        service.action("testing", submitted["job_id"], "purge", confirm_purge=True)
    service.config.allow_purge = True
    with pytest.raises(NomadValidationError, match="confirm_purge"):
        service.action("testing", submitted["job_id"], "purge", confirm_purge=False)
    result = service.action("testing", submitted["job_id"], "purge", confirm_purge=True)
    assert result["persistent_data_deleted"] is False


def test_purge_rejects_owned_but_non_disposable_job(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image))
    submitted = service.submit(plan["plan_id"])
    service.config.allow_purge = True
    with pytest.raises(NomadValidationError, match="explicitly declared disposable"):
        service.action("testing", submitted["job_id"], "purge", confirm_purge=True)


def test_reconcile_surfaces_ownership_mismatch(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, client = manager
    plan = service.plan(workload(pinned_image))
    submitted = service.submit(plan["plan_id"])
    client.remote[("testing", submitted["job_id"])]["Meta"]["owner"] = "someone-else"
    result = service.reconcile()
    assert result["jobs"][0]["state"] == "ownership_mismatch"


def test_offload_requires_approved_template_and_survives_manager_restart(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, client = manager
    with pytest.raises(NomadValidationError, match="not explicitly enabled"):
        service.plan_offload("arbitrary-shell", parameters={})
    planned = service.plan_offload(
        "repository-check",
        parameters={"repository": "repo@abc123"},
        input_artifacts={"source": "artifact://missy/inputs/repo.tar.zst"},
        output_artifacts={"result": "artifact://missy/results/result.json"},
        idempotency_key="offload-1",
    )
    restarted = NomadManager(service.config, client=client, store=service.store)
    submitted = restarted.submit_offload(planned["task_id"])
    assert submitted["task_id"] == planned["task_id"]
    assert submitted["state"] == "submitted_not_yet_verified"


def test_structured_result_requires_declared_artifact_and_schema(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, client = manager
    planned = service.plan_offload(
        "repository-check",
        parameters={"repository": "repo@abc123"},
        output_artifacts={"result": "artifact://missy/results/result.json"},
    )
    service.submit_offload(planned["task_id"])
    client.alloc_state = "complete"
    client.logs = (
        "benchmark complete\n"
        'MISSY_RESULT_JSON={"outputs":{"result":"artifact://missy/results/result.json"}}\n'
    )
    result = service.offload_result(planned["task_id"])
    assert result["outcome"] == "completed"
    assert result["output_artifacts_verified"] is True


def test_plan_scale_reuses_full_safe_update_path(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image, job_type="service", count=1))
    submitted = service.submit(plan["plan_id"])
    scaled = service.plan_scale("testing", submitted["job_id"], 2)
    assert scaled["operation"] == "scale"
    assert scaled["previous_count"] == 1
    assert scaled["plan"]["job_modify_index"] == 1
    assert scaled["placement"]["demand"]["temporary_rollout_allocations"] == 1


def test_benchmark_fanout_respects_parallelism_and_preserves_runs(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, client = manager
    benchmark = service.plan_benchmark(
        {
            "name": "repository comparison",
            "workload_template": "repository-check",
            "workload_version": "git:abc123",
            "input_artifacts": {"source": "artifact://missy/inputs/repo.tar.zst"},
            "parameters": {"repository": "repo@abc123"},
            "output_prefix": "artifact://missy/benchmarks",
            "warmup_runs": 1,
            "repetitions": 2,
            "parallelism": 2,
        }
    )
    started = service.start_benchmark(benchmark["benchmark_id"])
    assert len(started["runs"]) == 3
    assert sum(run["state"] == "submitted_not_yet_verified" for run in started["runs"]) == 2
    client.alloc_state = "failed"
    advanced = service.benchmark_status(benchmark["benchmark_id"])
    assert any(run["state"] == "failed" for run in advanced["runs"])
    assert sum(run["state"] == "submitted_not_yet_verified" for run in advanced["runs"]) == 1


def test_benchmark_parallelism_is_bounded_by_live_capacity(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, client = manager
    client.node = lambda _node_id: healthy_node(cpu=1_000, allocated_cpu=100)  # type: ignore[method-assign]
    with pytest.raises(NomadValidationError, match="observed authorized capacity"):
        service.plan_benchmark(
            {
                "name": "capacity test",
                "workload_template": "repository-check",
                "workload_version": "git:abc123",
                "parameters": {"repository": "repo@abc123"},
                "output_prefix": "artifact://missy/benchmarks",
                "repetitions": 2,
                "parallelism": 2,
            }
        )


def test_discovery_labels_denied_and_unavailable_data(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, client = manager
    client.namespaces = MagicMock(side_effect=NomadAuthorizationError("denied"))
    client.node_pools = MagicMock(side_effect=RuntimeError("offline"))
    result = service.discover()
    assert any("namespaces denied" in gap for gap in result["data_gaps"])
    assert any("node pools unavailable" in gap for gap in result["data_gaps"])


def test_result_distinguishes_cancelled_failed_and_timed_out(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, client = manager
    plan = service.plan(workload(pinned_image, idempotency_key="outcome-1"))
    submitted = service.submit(plan["plan_id"])
    service.action("testing", submitted["job_id"], "cancel")
    assert service.result("testing", submitted["job_id"])["outcome"] == "cancelled"

    plan = service.plan(workload(pinned_image, idempotency_key="outcome-2"))
    submitted = service.submit(plan["plan_id"])
    client.alloc_state = "failed"
    assert service.result("testing", submitted["job_id"])["outcome"] == "failed"

    plan = service.plan(workload(pinned_image, idempotency_key="outcome-3", max_run_seconds=1))
    submitted = service.submit(plan["plan_id"])
    client.alloc_state = "running"
    local = service.store.get_job("testing", submitted["job_id"])
    local["submitted_at"] = (datetime.now(tz=UTC) - timedelta(minutes=1)).isoformat()
    service.store.put_job("testing", submitted["job_id"], local)
    assert service.result("testing", submitted["job_id"])["outcome"] == "timed_out"


def test_owned_restart_stop_cancel_and_invalid_action(
    manager: tuple[NomadManager, FakeClient], pinned_image: str
) -> None:
    service, _ = manager
    plan = service.plan(workload(pinned_image))
    submitted = service.submit(plan["plan_id"])
    assert service.action("testing", submitted["job_id"], "restart")["state"] == "restart_requested"
    assert service.action("testing", submitted["job_id"], "stop")["state"] == "stopped"
    assert service.action("testing", submitted["job_id"], "cancel")["state"] == "cancelled"
    with pytest.raises(NomadValidationError, match="action must be"):
        service.action("testing", submitted["job_id"], "exec")


def test_offload_status_and_cancel_use_durable_task_record(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, _ = manager
    planned = service.plan_offload("repository-check", parameters={"repository": "repo@abc123"})
    service.submit_offload(planned["task_id"])
    assert service.offload_status(planned["task_id"])["task"]["state"] == "running"
    assert service.cancel_offload(planned["task_id"])["state"] == "cancelled"


def test_benchmark_results_keep_failures_and_cancel_pending_runs(
    manager: tuple[NomadManager, FakeClient],
) -> None:
    service, client = manager
    benchmark = service.plan_benchmark(
        {
            "name": "result test",
            "workload_template": "repository-check",
            "workload_version": "git:abc123",
            "parameters": {"repository": "repo@abc123"},
            "output_prefix": "artifact://missy/benchmarks",
            "repetitions": 2,
            "parallelism": 1,
        }
    )
    service.start_benchmark(benchmark["benchmark_id"])
    client.alloc_state = "failed"
    results = service.benchmark_results(benchmark["benchmark_id"])
    assert results["comparison"]["valid"] is False
    assert any(run["state"] == "failed" for run in results["runs"])
    cancelled = service.cancel_benchmark(benchmark["benchmark_id"])
    assert cancelled["state"] == "cancelled"
    assert any(run["state"] == "cancelled_before_submission" for run in cancelled["runs"])


@pytest.mark.parametrize(
    "event,message,expected",
    [
        ("Downloading Artifacts", "pulling image", "image_pull"),
        ("Driver Failure", "driver unhealthy", "driver"),
        ("Check Unhealthy", "health check", "health_check"),
        ("Setup Failure", "network address missing", "network_reachability"),
        ("Placement Failure", "failed to place", "scheduling"),
        ("Terminated", "exit code 1", "process"),
        ("Started", "ready", None),
    ],
)
def test_failure_category(event: str, message: str, expected: str | None) -> None:
    assert NomadManager._failure_category(event, message) == expected


def test_structured_result_schema_reports_missing_and_wrong_types() -> None:
    schema = {
        "required": ["score", "count"],
        "properties": {
            "score": {"type": "number"},
            "count": {"type": "integer"},
        },
    }
    valid, errors = NomadManager._validate_structured_result({"score": True}, schema)
    assert valid is False
    assert "missing required field 'count'" in errors
    assert "field 'score' is not number" in errors
