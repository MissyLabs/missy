"""All scheduler calls in these tests use an isolated in-memory fake."""

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event, Lock

import pytest

from missy.repoeval.dispatch import (
    DispatchConflict,
    DispatchError,
    DurableDispatcher,
    JobObservation,
    SQLiteOutbox,
)

RUN = "run-abcdefghij"


def job(run_id=RUN):
    job_id = "foundry-" + run_id
    return {
        "ID": job_id,
        "Name": job_id,
        "Namespace": "sandbox",
        "Type": "batch",
        "Meta": {"foundry_run_id": run_id},
        "TaskGroups": [{"Name": "worker", "Count": 1}],
    }


class FakeScheduler:
    def __init__(self):
        self.jobs = {}
        self.submissions = 0
        self.stops = 0
        self.submit_error = False
        self.stop_error = False
        self.lookup_error = False
        self.lock = Lock()

    def submit(self, namespace, payload):
        with self.lock:
            self.submissions += 1
            self.jobs[(namespace, payload["ID"])] = (deepcopy(payload), "pending")
        if self.submit_error:
            raise TimeoutError("the scheduler may have accepted the job")

    def lookup(self, namespace, job_id):
        if self.lookup_error:
            raise TimeoutError("scheduler read unavailable")
        record = self.jobs.get((namespace, job_id))
        return JobObservation(deepcopy(record[0]), record[1]) if record else None

    def stop(self, namespace, job_id):
        with self.lock:
            self.stops += 1
            record = self.jobs[(namespace, job_id)]
            self.jobs[(namespace, job_id)] = (record[0], "stopped")
        if self.stop_error:
            raise TimeoutError("stop effect unknown")


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / "outbox.db"
    store = SQLiteOutbox.initialize(path)
    scheduler = FakeScheduler()
    dispatcher = DurableDispatcher(store, scheduler, lambda *_: True, enabled=True)
    return store, scheduler, dispatcher


def _reserve_other_process(path, queue):
    dispatcher = DurableDispatcher(
        SQLiteOutbox(path), FakeScheduler(), lambda *_: True, enabled=True
    )
    try:
        queue.put(dispatcher.reserve("project", RUN, "some-idem-key", job())["state"])
    except DispatchConflict:
        queue.put("conflict")


def _dispatch_other_process(path, queue):
    class SchedulerWithSharedCounter:
        def submit(self, namespace, payload):
            queue.put((namespace, payload["ID"]))

        def lookup(self, namespace, job_id):
            return None

        def stop(self, namespace, job_id):
            raise AssertionError("unexpected stop")

    dispatcher = DurableDispatcher(
        SQLiteOutbox(path), SchedulerWithSharedCounter(), lambda *_: True, enabled=True
    )
    dispatcher.dispatch("project", RUN)


def test_disabled_and_explicit_initialization(tmp_path):
    path = tmp_path / "state.db"
    with pytest.raises(DispatchError):
        SQLiteOutbox(path)
    store = SQLiteOutbox.initialize(path)
    disabled = DurableDispatcher(store, FakeScheduler(), lambda *_: True)
    with pytest.raises(DispatchError, match="disabled"):
        disabled.reserve("project", RUN, "some-idem-key", job())
    assert store.get("project", RUN) is None


def test_concurrent_idempotent_submission_and_restart(setup):
    store, scheduler, dispatcher = setup
    with ThreadPoolExecutor(max_workers=12) as pool:
        reserved = list(
            pool.map(
                lambda _: dispatcher.reserve("project", RUN, "some-idem-key", job()), range(12)
            )
        )
        states = list(pool.map(lambda _: dispatcher.dispatch("project", RUN), range(12)))
    assert {r["state"] for r in reserved} == {"reserved"}
    # Other callers may see the durable attempt marker while submit is in flight.
    assert {s["state"] for s in states} <= {"dispatching", "submitted"}
    deadline = time.monotonic() + 5
    while dispatcher.reconcile("project", RUN)["state"] != "submitted":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert scheduler.submissions == 1
    restarted = DurableDispatcher(
        SQLiteOutbox(store.path), scheduler, lambda *_: True, enabled=True
    )
    assert restarted.dispatch("project", RUN)["state"] == "submitted"
    assert scheduler.submissions == 1
    with pytest.raises(DispatchConflict):
        restarted.reserve("project", "run-different123", "some-idem-key", job("run-different123"))
    with pytest.raises(DispatchConflict):
        restarted.reserve("project", RUN, "different-key", job())


def test_process_safe_reservation(tmp_path):
    store = SQLiteOutbox.initialize(tmp_path / "state.db")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    procs = [
        context.Process(target=_reserve_other_process, args=(store.path, queue)) for _ in range(4)
    ]
    for process in procs:
        process.start()
    result = [queue.get(timeout=20) for _ in procs]
    for process in procs:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert result == ["reserved"] * 4
    assert store.get("project", RUN)["state"] == "reserved"


def test_multiple_processes_only_one_submission_attempt(tmp_path):
    store = SQLiteOutbox.initialize(tmp_path / "state.db")
    DurableDispatcher(store, FakeScheduler(), lambda *_: True, enabled=True).reserve(
        "project", RUN, "some-idem-key", job()
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    procs = [
        context.Process(target=_dispatch_other_process, args=(store.path, queue)) for _ in range(5)
    ]
    for process in procs:
        process.start()
    for process in procs:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert queue.get(timeout=10) == ("sandbox", job()["ID"])
    assert queue.empty()
    assert store.get("project", RUN)["state"] == "uncertain"


def test_restart_from_committed_attempt_marker_never_replays(setup):
    store, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    # Simulates process death immediately after committing attempt marker.
    with store.transaction() as db:
        row = dispatcher._row(db, "project", RUN)
        dispatcher._update(db, row, "dispatching")
    restarted = DurableDispatcher(
        SQLiteOutbox(store.path), scheduler, lambda *_: True, enabled=True
    )
    assert restarted.dispatch("project", RUN)["state"] == "dispatching"
    assert restarted.reconcile("project", RUN)["state"] == "uncertain"
    assert scheduler.submissions == 0


def test_unknown_submit_never_replayed_and_requires_lookup(setup):
    store, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    scheduler.submit_error = True
    assert dispatcher.dispatch("project", RUN)["state"] == "uncertain"
    assert scheduler.submissions == 1
    scheduler.lookup_error = True
    assert dispatcher.reconcile("project", RUN)["state"] == "uncertain"
    scheduler.lookup_error = False
    assert dispatcher.reconcile("project", RUN)["state"] == "submitted"
    assert dispatcher.dispatch("project", RUN)["state"] == "submitted"
    assert scheduler.submissions == 1


def test_missing_lookup_not_evidence_for_replay(setup):
    _, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    scheduler.submit = lambda *_: (_ for _ in ()).throw(TimeoutError("unknown"))
    assert dispatcher.dispatch("project", RUN)["state"] == "uncertain"
    assert dispatcher.reconcile("project", RUN)["state"] == "uncertain"
    assert dispatcher.cancel("project", RUN)["state"] == "uncertain"
    assert scheduler.stops == 0


def test_mismatched_scheduler_job_is_conflict(setup):
    _, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    dispatcher.dispatch("project", RUN)
    corrupted = job()
    corrupted["TaskGroups"] = []
    scheduler.jobs[("sandbox", job()["ID"])] = (corrupted, "running")
    with pytest.raises(DispatchConflict):
        dispatcher.reconcile("project", RUN)
    assert dispatcher.status("project", RUN)["state"] == "conflict"
    assert dispatcher.cancel("project", RUN)["state"] == "conflict"
    assert scheduler.stops == 0


def test_cancel_and_stop_uncertainty_not_retried(setup):
    store, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    dispatcher.dispatch("project", RUN)
    scheduler.stop_error = True
    assert dispatcher.cancel("project", RUN)["state"] == "stop_uncertain"
    assert scheduler.stops == 1
    restarted = DurableDispatcher(
        SQLiteOutbox(store.path), scheduler, lambda *_: True, enabled=True
    )
    assert restarted.cancel("project", RUN)["state"] == "cancelled"
    assert scheduler.stops == 1


def test_cancel_during_submit_stops_once_without_second_cancel(setup):
    store, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    entered, release = Event(), Event()
    original_submit = scheduler.submit

    def blocked_submit(namespace, payload):
        entered.set()
        assert release.wait(5)
        original_submit(namespace, payload)

    scheduler.submit = blocked_submit
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(dispatcher.dispatch, "project", RUN)
        assert entered.wait(5)
        assert dispatcher.cancel("project", RUN)["state"] == "uncertain"
        assert dispatcher.status("project", RUN)["cancel_requested"] == 1
        assert scheduler.stops == 0
        release.set()
        assert submission.result(timeout=5)["state"] == "cancelled"
    assert scheduler.submissions == scheduler.stops == 1
    assert dispatcher.status("project", RUN)["stop_attempted"] == 1
    restarted = DurableDispatcher(
        SQLiteOutbox(store.path), scheduler, lambda *_: True, enabled=True
    )
    assert restarted.cancel("project", RUN)["state"] == "cancelled"
    assert scheduler.stops == 1


def test_cancel_during_uncertain_submit_uses_only_exact_lookup(setup):
    _, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    entered, release = Event(), Event()

    def timed_out_submit(namespace, payload):
        entered.set()
        assert release.wait(5)
        scheduler.jobs[(namespace, payload["ID"])] = (deepcopy(payload), "pending")
        scheduler.submissions += 1
        raise TimeoutError("accepted, but result lost")

    scheduler.submit = timed_out_submit
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(dispatcher.dispatch, "project", RUN)
        assert entered.wait(5)
        assert dispatcher.cancel("project", RUN)["state"] == "uncertain"
        release.set()
        assert submission.result(timeout=5)["state"] == "cancelled"
    assert scheduler.submissions == scheduler.stops == 1


def test_cancel_during_submit_stop_timeout_not_replayed(setup):
    store, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    entered, release = Event(), Event()
    original_submit = scheduler.submit
    scheduler.stop_error = True

    def blocked_submit(namespace, payload):
        entered.set()
        assert release.wait(5)
        original_submit(namespace, payload)

    scheduler.submit = blocked_submit
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(dispatcher.dispatch, "project", RUN)
        assert entered.wait(5)
        assert dispatcher.cancel("project", RUN)["state"] == "uncertain"
        release.set()
        assert submission.result(timeout=5)["state"] == "stop_uncertain"
    assert scheduler.stops == 1
    scheduler.lookup_error = True
    restarted = DurableDispatcher(
        SQLiteOutbox(store.path), scheduler, lambda *_: True, enabled=True
    )
    assert restarted.cancel("project", RUN)["state"] == "stop_uncertain"
    assert scheduler.stops == 1
    scheduler.lookup_error = False
    assert restarted.reconcile("project", RUN)["state"] == "cancelled"
    assert scheduler.stops == 1


def test_cancel_during_submit_with_mismatched_job_never_stops(setup):
    _, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    entered, release = Event(), Event()

    def colliding_submit(namespace, payload):
        entered.set()
        assert release.wait(5)
        corrupted = deepcopy(payload)
        corrupted["TaskGroups"] = []
        scheduler.jobs[(namespace, payload["ID"])] = (corrupted, "pending")

    scheduler.submit = colliding_submit
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(dispatcher.dispatch, "project", RUN)
        assert entered.wait(5)
        assert dispatcher.cancel("project", RUN)["state"] == "uncertain"
        release.set()
        with pytest.raises(DispatchConflict):
            submission.result(timeout=5)
    assert dispatcher.status("project", RUN)["state"] == "conflict"
    assert scheduler.stops == 0


def test_competing_reconcile_and_cancel_claim_one_stop(setup):
    _, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    dispatcher.dispatch("project", RUN)
    entered, release = Event(), Event()
    original_stop = scheduler.stop

    def blocked_stop(namespace, job_id):
        entered.set()
        assert release.wait(5)
        original_stop(namespace, job_id)

    scheduler.stop = blocked_stop
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(dispatcher.cancel, "project", RUN)
        assert entered.wait(5)
        assert dispatcher.reconcile("project", RUN)["state"] == "stop_uncertain"
        assert dispatcher.cancel("project", RUN)["state"] == "stop_uncertain"
        assert dispatcher.status("project", RUN)["stop_attempted"] == 1
        release.set()
        assert first.result(timeout=5)["state"] == "cancelled"
    assert scheduler.stops == 1


def test_cancel_before_submit_and_completion_not_verified(setup):
    _, scheduler, dispatcher = setup
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    assert dispatcher.cancel("project", RUN)["state"] == "cancelled"
    dispatcher.dispatch("project", RUN)
    assert scheduler.submissions == 0
    second = "run-abcdefghik"
    dispatcher.reserve("project", second, "second-idem-key", job(second))
    dispatcher.dispatch("project", second)
    scheduler.jobs[("sandbox", job(second)["ID"])] = (job(second), "complete")
    assert dispatcher.reconcile("project", second)["state"] == "collecting"
    assert dispatcher.status("project", second)["state"] != "verified"


def test_revocation_and_identity_validation(setup):
    store, scheduler, dispatcher = setup
    denied = DurableDispatcher(store, scheduler, lambda *_: False, enabled=True)
    with pytest.raises(DispatchError, match="authorized"):
        denied.reserve("project", RUN, "some-idem-key", job())
    incorrect = job()
    incorrect["ID"] = "not-the-run"
    with pytest.raises(ValueError, match="identity"):
        dispatcher.reserve("project", RUN, "some-idem-key", incorrect)
    dispatcher.reserve("project", RUN, "some-idem-key", job())
    with pytest.raises(DispatchError, match="revoked"):
        denied.dispatch("project", RUN)
    assert dispatcher.status("project", RUN)["state"] == "reserved"
    assert scheduler.submissions == 0
