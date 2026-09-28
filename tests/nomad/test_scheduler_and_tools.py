from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from missy.config.settings import NomadConfig
from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager
from missy.tools.builtin import register_builtin_tools
from missy.tools.builtin.nomad_tools import (
    NomadBenchmarkTool,
    NomadDiscoverTool,
    NomadJobActionTool,
    NomadJobResultTool,
    NomadJobStatusTool,
    NomadOffloadTool,
    NomadPlanJobTool,
    NomadRecommendPlacementTool,
    NomadReconcileTool,
    NomadScaleTool,
    NomadScheduleTool,
    NomadSubmitJobTool,
)
from missy.tools.registry import ToolRegistry


def nomad_request(image: str) -> dict:
    return {
        "purpose": "scheduled benchmark",
        "image": image,
        "job_type": "batch",
        "namespace": "testing",
        "node_pool": "staging",
        "datacenter": "dc1",
    }


def test_scheduled_job_round_trips_nomad_fields(pinned_image: str) -> None:
    job = ScheduledJob(
        name="benchmark",
        schedule="every 2 hours",
        task="structured Nomad task",
        execution_target="nomad",
        nomad_request=nomad_request(pinned_image),
        nomad_overlap_policy="replace",
        last_nomad_job_id="missy-old",
        last_nomad_evaluation_id="eval-old",
        nomad_queued_runs=2,
        nomad_runs=[{"nomad_job_id": "missy-old", "state": "running"}],
    )
    restored = ScheduledJob.from_dict(job.to_dict())
    assert restored.execution_target == "nomad"
    assert restored.nomad_request == job.nomad_request
    assert restored.nomad_overlap_policy == "replace"
    assert restored.last_nomad_job_id == "missy-old"
    assert restored.nomad_queued_runs == 2
    assert restored.nomad_runs[0]["state"] == "running"


def test_scheduler_submits_fresh_nomad_batch(
    tmp_path: Path, nomad_config: NomadConfig, pinned_image: str
) -> None:
    manager = SchedulerManager(
        jobs_file=str(tmp_path / "jobs.json"),
        nomad_config=nomad_config,
        reconcile_interval_seconds=0,
    )
    job = manager.add_job(
        name="benchmark",
        schedule="every 2 hours",
        task="structured Nomad task",
        provider="nomad",
        capability_mode="no-tools",
        execution_target="nomad",
        nomad_request=nomad_request(pinned_image),
        nomad_overlap_policy="skip",
    )
    fake = MagicMock()
    fake.plan.return_value = {"plan_id": "plan-1"}
    fake.submit.return_value = {
        "state": "submitted_not_yet_verified",
        "job_id": "missy-scheduled-job",
        "evaluation_id": "eval-1",
        "plan_id": "plan-1",
    }
    with patch("missy.nomad.manager.NomadManager", return_value=fake):
        manager._run_job(job.id)
    assert job.run_count == 1
    assert job.last_nomad_job_id == "missy-scheduled-job"
    assert job.last_nomad_evaluation_id == "eval-1"
    payload = fake.plan.call_args.args[0]
    assert payload["job_type"] == "batch"
    assert payload["job_id"].startswith("missy-scheduled-")
    assert payload["idempotency_key"].startswith(f"schedule:{job.id}:run:1")
    assert json.loads(job.last_result or "{}")["state"] == "submitted_not_yet_verified"


def test_scheduler_skip_overlap_does_not_submit_new_job(
    tmp_path: Path, nomad_config: NomadConfig, pinned_image: str
) -> None:
    manager = SchedulerManager(jobs_file=str(tmp_path / "jobs.json"), nomad_config=nomad_config)
    job = manager.add_job(
        name="benchmark",
        schedule="every 2 hours",
        task="structured Nomad task",
        execution_target="nomad",
        nomad_request=nomad_request(pinned_image),
        nomad_overlap_policy="skip",
    )
    job.last_nomad_job_id = "missy-still-running"
    fake = MagicMock()
    fake.status.return_value = {"allocations": [{"client_status": "running"}]}
    with patch("missy.nomad.manager.NomadManager", return_value=fake):
        result = manager._run_nomad_job(job)
    assert json.loads(result)["state"] == "skipped_overlap"
    fake.plan.assert_not_called()


def test_scheduler_queue_overlap_is_explicit_and_durable(
    tmp_path: Path, nomad_config: NomadConfig, pinned_image: str
) -> None:
    manager = SchedulerManager(jobs_file=str(tmp_path / "jobs.json"), nomad_config=nomad_config)
    job = manager.add_job(
        name="benchmark",
        schedule="every 2 hours",
        task="structured Nomad task",
        execution_target="nomad",
        nomad_request=nomad_request(pinned_image),
        nomad_overlap_policy="queue",
    )
    job.last_nomad_job_id = "missy-still-running"
    fake = MagicMock()
    fake.status.return_value = {"allocations": [{"client_status": "running"}]}
    fake._state_from_status.return_value = "running"
    with patch("missy.nomad.manager.NomadManager", return_value=fake):
        result = manager._run_nomad_job(job)
    assert json.loads(result)["state"] == "queued_overlap"
    assert job.nomad_queued_runs == 1
    fake.plan.assert_not_called()


def test_all_nomad_tools_are_registered() -> None:
    registry = ToolRegistry()
    register_builtin_tools(registry)
    expected = {
        "nomad_discover",
        "nomad_recommend_placement",
        "nomad_plan_job",
        "nomad_submit_job",
        "nomad_job_status",
        "nomad_job_result",
        "nomad_job_action",
        "nomad_plan_scale",
        "nomad_offload_task",
        "nomad_benchmark",
        "nomad_reconcile",
        "nomad_schedule",
    }
    assert expected <= set(registry.list_tools())


def test_all_nomad_tool_schemas_are_well_formed() -> None:
    classes = [
        NomadDiscoverTool,
        NomadRecommendPlacementTool,
        NomadPlanJobTool,
        NomadSubmitJobTool,
        NomadJobStatusTool,
        NomadJobResultTool,
        NomadJobActionTool,
        NomadScaleTool,
        NomadOffloadTool,
        NomadBenchmarkTool,
        NomadReconcileTool,
        NomadScheduleTool,
    ]
    for cls in classes:
        tool = cls()
        schema = tool.get_schema()
        assert schema["name"] == tool.name
        assert schema["parameters"]["type"] == "object"


def test_nomad_tool_execution_routes_to_constrained_manager() -> None:
    manager = MagicMock()
    manager.discover.return_value = {"observed_at": "now"}
    manager.recommend.return_value = {"recommended_node": {"name": "worker"}}
    manager.plan.return_value = {"plan_id": "plan-1"}
    manager.submit.return_value = {"job_id": "missy-job"}
    manager.status.return_value = {"job_status": "running"}
    manager.observe.return_value = {"observation": "healthy_deployment"}
    manager.result.return_value = {"outcome": "completed"}
    manager.action.return_value = {"state": "stopped"}
    manager.plan_scale.return_value = {"operation": "scale"}
    manager.reconcile.return_value = {"jobs": []}
    manager.plan_offload.return_value = {"task_id": "task-1"}
    manager.submit_offload.return_value = {"state": "submitted_not_yet_verified"}
    manager.offload_status.return_value = {"task_id": "task-1"}
    manager.offload_result.return_value = {"outcome": "completed"}
    manager.cancel_offload.return_value = {"state": "cancelled"}
    manager.plan_benchmark.return_value = {"benchmark_id": "benchmark-1"}
    manager.start_benchmark.return_value = {"state": "running"}
    manager.benchmark_status.return_value = {"state": "running"}
    manager.benchmark_results.return_value = {"state": "complete"}
    manager.cancel_benchmark.return_value = {"state": "cancelled"}

    cases = [
        (NomadDiscoverTool(), {"namespace": "testing"}),
        (NomadRecommendPlacementTool(), {"request": {}}),
        (NomadPlanJobTool(), {"request": {}}),
        (NomadSubmitJobTool(), {"plan_id": "plan-1"}),
        (NomadJobStatusTool(), {"namespace": "testing", "job_id": "missy-job"}),
        (
            NomadJobStatusTool(),
            {"namespace": "testing", "job_id": "missy-job", "observe": True},
        ),
        (NomadJobResultTool(), {"namespace": "testing", "job_id": "missy-job"}),
        (
            NomadJobActionTool(),
            {"namespace": "testing", "job_id": "missy-job", "action": "stop"},
        ),
        (
            NomadScaleTool(),
            {"namespace": "testing", "job_id": "missy-job", "count": 2},
        ),
        (NomadReconcileTool(), {}),
    ]
    for tool, arguments in cases:
        tool._manager = MagicMock(return_value=manager)
        assert tool.execute(**arguments).success is True

    offload = NomadOffloadTool()
    offload._manager = MagicMock(return_value=manager)
    for action in ("plan", "submit", "status", "result", "cancel"):
        assert offload.execute(action=action, task_id="task-1").success is True

    benchmark = NomadBenchmarkTool()
    benchmark._manager = MagicMock(return_value=manager)
    for action in ("plan", "start", "status", "results", "cancel"):
        assert benchmark.execute(action=action, benchmark_id="benchmark-1").success is True


def test_nomad_tool_reports_invalid_action_without_raising() -> None:
    tool = NomadOffloadTool()
    tool._manager = MagicMock(return_value=MagicMock())
    result = tool.execute(action="arbitrary")
    assert result.success is False
    assert "action must be" in (result.error or "")


def test_nomad_schedule_tool_create_list_and_controls(
    tmp_path: Path, nomad_config: NomadConfig, pinned_image: str
) -> None:
    scheduler = SchedulerManager(jobs_file=str(tmp_path / "jobs.json"), nomad_config=nomad_config)
    config = MagicMock()
    config.nomad = nomad_config
    config.scheduling.max_jobs = 10
    config.scheduling.active_hours = ""
    config.scheduling.misfire_grace_seconds = 60
    tool = NomadScheduleTool()
    tool._scheduler = MagicMock(return_value=scheduler)
    tool._config = MagicMock(return_value=config)

    created = tool.execute(
        action="create",
        name="repository check",
        schedule="every 2 hours",
        request={
            "purpose": "scheduled repository check",
            "image": pinned_image,
            "job_type": "batch",
        },
        overlap_policy="queue",
    )
    assert created.success is True
    schedule_id = created.output["schedule_id"]

    scheduled_job = scheduler.list_jobs()[0]
    scheduled_job.last_nomad_job_id = "missy-scheduled-job"
    scheduled_job.nomad_runs = [
        {"nomad_job_id": "missy-scheduled-job", "state": "submitted_not_yet_verified"}
    ]
    scheduler._save_jobs()
    cluster = MagicMock()
    cluster.status.return_value = {"allocations": [{"client_status": "complete"}]}
    cluster._state_from_status.return_value = "completed_unverified"
    cluster.result.return_value = {"outcome": "completed"}
    with patch("missy.tools.builtin.nomad_tools.NomadManager", return_value=cluster):
        listed = tool.execute(action="list")
    assert listed.success is True
    assert listed.output[0]["schedule_id"] == schedule_id
    assert listed.output[0]["runs"][0]["state"] == "completed"
    assert "completed_at" in listed.output[0]["runs"][0]

    assert tool.execute(action="pause", schedule_id=schedule_id).output["state"] == "paused"
    assert tool.execute(action="resume", schedule_id=schedule_id).output["state"] == "scheduled"
    assert tool.execute(action="remove", schedule_id=schedule_id).output["state"] == "removed"
    assert tool.execute(action="pause", schedule_id="missing").success is False
