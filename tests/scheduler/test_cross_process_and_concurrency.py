"""SCHED-01/02/03, DATA-01/06, DGAP-01: cross-process merge, locking, misfire,
global active hours, and run traceability for :class:`SchedulerManager`."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from missy.scheduler.jobs import ScheduledJob
from missy.scheduler.manager import SchedulerManager


def _mgr(path, **kw) -> SchedulerManager:
    return SchedulerManager(jobs_file=str(path), reconcile_interval_seconds=0, **kw)


def _ids_on_disk(path) -> set[str]:
    return {r["id"] for r in json.loads(path.read_text())}


class TestCrossProcessMerge:
    def test_cli_add_is_seen_by_gateway_and_not_clobbered(self, tmp_path):
        path = tmp_path / "jobs.json"
        gateway = _mgr(path)
        gateway.open_offline()
        existing = gateway.add_job("existing", "every 5 minutes", "a")

        cli = _mgr(path)
        cli.open_offline()
        added = cli.add_job("from-cli", "every 10 minutes", "b")

        # Gateway saves its (stale) view, e.g. after a run of another job.
        gateway._jobs[existing.id].run_count += 1
        gateway._save_jobs()

        assert _ids_on_disk(path) == {existing.id, added.id}
        assert added.id in {j.id for j in gateway.list_jobs()}

    def test_cli_pause_reaches_gateway_before_next_run(self, tmp_path):
        path = tmp_path / "jobs.json"
        gateway = _mgr(path)
        gateway.open_offline()
        job = gateway.add_job("j", "every 5 minutes", "task")

        cli = _mgr(path)
        cli.open_offline()
        cli.pause_job(job.id)

        with patch("missy.agent.runtime.AgentRuntime") as runtime_cls:
            gateway._run_job(job.id)
        runtime_cls.assert_not_called()
        assert gateway._jobs[job.id].enabled is False

    def test_cli_remove_is_not_resurrected(self, tmp_path):
        path = tmp_path / "jobs.json"
        gateway = _mgr(path)
        gateway.open_offline()
        a = gateway.add_job("a", "every 5 minutes", "x")
        b = gateway.add_job("b", "every 5 minutes", "y")

        cli = _mgr(path)
        cli.open_offline()
        cli.remove_job(a.id)

        gateway._jobs[b.id].run_count += 1
        gateway._save_jobs()
        assert _ids_on_disk(path) == {b.id}
        assert a.id not in gateway._jobs

    def test_run_state_from_runner_wins_over_older_disk(self, tmp_path):
        path = tmp_path / "jobs.json"
        gateway = _mgr(path)
        gateway.open_offline()
        job = gateway.add_job("j", "every 5 minutes", "t")
        cli = _mgr(path)
        cli.open_offline()
        cli.pause_job(job.id)  # config change on disk

        live = gateway._jobs[job.id]
        live.run_count = 7
        live.last_run = datetime.now(tz=UTC)
        gateway._save_jobs()

        record = json.loads(path.read_text())[0]
        assert record["run_count"] == 7
        assert record["enabled"] is False  # newer config from CLI kept

    def test_offline_mode_never_starts_background_thread(self, tmp_path):
        mgr = _mgr(tmp_path / "jobs.json")
        mgr.open_offline()
        mgr.add_job("j", "every 5 minutes", "t")
        assert mgr._scheduler.running is False

    def test_running_scheduler_reconciles_added_job(self, tmp_path):
        path = tmp_path / "jobs.json"
        gateway = _mgr(path)
        gateway.start()
        try:
            cli = _mgr(path)
            cli.open_offline()
            added = cli.add_job("late", "every 10 minutes", "t")
            assert gateway.reconcile() is True
            assert gateway._scheduler.get_job(added.id) is not None
            assert gateway.reconcile() is False  # unchanged -> no-op
        finally:
            gateway.stop()


class TestConcurrency:
    def test_parallel_add_and_save_do_not_raise_or_lose_jobs(self, tmp_path):
        mgr = _mgr(tmp_path / "jobs.json")
        mgr.open_offline()
        errors: list[Exception] = []

        def adder(n: int) -> None:
            try:
                for i in range(10):
                    mgr.add_job(f"j{n}-{i}", "every 5 minutes", "t")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def saver() -> None:
            try:
                for _ in range(30):
                    mgr._save_jobs()
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=adder, args=(n,)) for n in range(4)]
        threads += [threading.Thread(target=saver) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(mgr.list_jobs()) == 40
        assert len(_ids_on_disk(tmp_path / "jobs.json")) == 40


class TestMisfireAndOneShots:
    def test_job_defaults_configured(self, tmp_path):
        mgr = _mgr(tmp_path / "jobs.json", misfire_grace_seconds=120)
        defaults = mgr._scheduler._job_defaults
        assert defaults["misfire_grace_time"] == 120
        assert defaults["coalesce"] is True
        assert defaults["max_instances"] == 1

    def test_missed_event_is_audited(self, tmp_path):
        from apscheduler.events import EVENT_JOB_MISSED

        mgr = _mgr(tmp_path / "jobs.json")
        mgr.open_offline()
        job = mgr.add_job("j", "every 5 minutes", "t")
        with patch.object(mgr, "_emit_event") as emit:
            mgr._on_scheduler_event(
                SimpleNamespace(code=EVENT_JOB_MISSED, job_id=job.id, scheduled_run_time=None)
            )
        assert emit.call_args.kwargs["event_type"] == "scheduler.job.missed"

    def test_missed_one_shot_is_marked_and_disabled(self, tmp_path):
        from apscheduler.events import EVENT_JOB_MISSED

        mgr = _mgr(tmp_path / "jobs.json")
        mgr.open_offline()
        future = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
        job = mgr.add_job("once", f"at {future}", "t")
        mgr._on_scheduler_event(
            SimpleNamespace(code=EVENT_JOB_MISSED, job_id=job.id, scheduled_run_time=None)
        )
        saved = json.loads((tmp_path / "jobs.json").read_text())[0]
        assert saved["enabled"] is False
        assert saved["last_error"].startswith("missed")

    def test_overlap_event_is_audited(self, tmp_path):
        from apscheduler.events import EVENT_JOB_MAX_INSTANCES

        mgr = _mgr(tmp_path / "jobs.json")
        with patch.object(mgr, "_emit_event") as emit:
            mgr._on_scheduler_event(
                SimpleNamespace(code=EVENT_JOB_MAX_INSTANCES, job_id="x", scheduled_run_time=None)
            )
        assert emit.call_args.kwargs["event_type"] == "scheduler.job.skipped_overlap"

    def test_spent_one_shot_not_reregistered_on_start(self, tmp_path):
        path = tmp_path / "jobs.json"
        past = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
        job = ScheduledJob(name="done", schedule=f"at {past}", task="t", run_count=1)
        path.write_text(json.dumps([job.to_dict()]))
        path.chmod(0o600)
        mgr = _mgr(path)
        mgr.start()
        try:
            assert mgr._scheduler.get_job(job.id) is None
        finally:
            mgr.stop()


class TestGlobalActiveHours:
    def test_default_window_applies_to_job_without_its_own(self):
        job = ScheduledJob(name="j", schedule="every 5 minutes", task="t")
        now = datetime.now()
        start = (now + timedelta(hours=2)).strftime("%H:%M")
        end = (now + timedelta(hours=3)).strftime("%H:%M")
        assert job.should_run_now() is True
        assert job.should_run_now(f"{start}-{end}") is False

    def test_job_window_wins_over_default(self):
        job = ScheduledJob(name="j", schedule="x", task="t", active_hours="00:00-23:59")
        assert job.should_run_now("03:00-03:01") is True

    def test_run_job_skips_outside_global_window(self, tmp_path):
        now = datetime.now()
        window = f"{(now + timedelta(hours=2)).strftime('%H:%M')}-{(now + timedelta(hours=3)).strftime('%H:%M')}"
        mgr = _mgr(tmp_path / "jobs.json", default_active_hours=window)
        mgr.open_offline()
        job = mgr.add_job("j", "every 5 minutes", "t")
        with patch("missy.agent.runtime.AgentRuntime") as runtime_cls:
            mgr._run_job(job.id)
        runtime_cls.assert_not_called()


class TestRunTraceability:
    def test_run_records_session_cost_and_aware_next_run(self, tmp_path):
        mgr = _mgr(tmp_path / "jobs.json")
        mgr.start()
        try:
            job = mgr.add_job("j", "every 5 minutes", "t")
            tracker = SimpleNamespace(total_cost_usd=0.25)
            agent = MagicMock()
            agent.run.return_value = "done"
            agent._peek_cost_tracker.return_value = tracker
            with patch("missy.agent.runtime.AgentRuntime", return_value=agent):
                mgr._run_job(job.id)
                mgr._run_job(job.id)
        finally:
            mgr.stop()
        live = mgr._jobs[job.id]
        assert live.last_session_id
        assert live.last_cost_usd == pytest.approx(0.25)
        assert live.total_cost_usd == pytest.approx(0.5)
        assert live.next_run is not None and live.next_run.tzinfo is not None
        agent._memory_store.register_session.assert_called()
        name = agent._memory_store.register_session.call_args.kwargs["name"]
        assert name.startswith("job:j:")

    def test_feature_kwargs_forwarded_to_agent_config(self, tmp_path):
        mgr = _mgr(tmp_path / "jobs.json", default_feature_kwargs={"model_routing_enabled": True})
        mgr.open_offline()
        job = mgr.add_job("j", "every 5 minutes", "t")
        with (
            patch("missy.agent.runtime.AgentRuntime") as runtime_cls,
            patch("missy.agent.runtime.AgentConfig") as config_cls,
        ):
            runtime_cls.return_value.run.return_value = "ok"
            mgr._run_job(job.id)
        assert config_cls.call_args.kwargs["model_routing_enabled"] is True
