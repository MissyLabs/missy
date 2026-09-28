"""All Nomad calls terminate at a fake; no CLI, network, or cluster access."""

from copy import deepcopy

import pytest

from missy.config.settings import NomadConfig
from missy.nomad.client import NomadClient
from missy.repoeval.nomad import JobRequest, plan_job
from missy.repoeval.nomad_adapter import NomadAdapterError, NomadSchedulerAdapter
from missy.repoeval.placement import PoolCapacity

IMAGE = "registry.example/worker@sha256:" + "a" * 64
RUN = "run-abcdefghij"


def job():
    request = JobRequest(
        run_id=RUN,
        repository_commit_sha="b" * 40,
        snapshot_id="snapshot-123",
        image_digest=IMAGE,
        workload_class="tool-call",
        cpu_mhz=1000,
        memory_mb=1024,
        disk_mb=512,
        timeout_seconds=60,
        namespace="sandbox",
        datacenter="dc1",
    )
    return deepcopy(plan_job(request, [PoolCapacity("staging", 2000, 4096, 4096)]).job)


class FakeNomadClient(NomadClient):
    def __init__(self):  # Never invoke NomadClient.__init__ or subprocess.
        self.config = NomadConfig(
            enabled=True,
            allowed_namespaces=["sandbox"],
            allowed_datacenters=["dc1"],
            allowed_node_pools=["staging"],
        )
        self.calls = []
        self.inspected = None
        self.status = None
        self.allocs = []

    def run_job(self, namespace, spec, *, check_index):
        self.calls.append(("run", namespace, deepcopy(spec), check_index))
        return "eval-123"

    def inspect_job(self, namespace, job_id):
        self.calls.append(("inspect", namespace, job_id))
        if self.inspected is None:
            raise TimeoutError("read unavailable")
        return deepcopy(self.inspected)

    def job_status(self, namespace, job_id):
        self.calls.append(("status", namespace, job_id))
        if self.status is None:
            raise TimeoutError("read unavailable")
        return deepcopy(self.status)

    def allocations(self, namespace, job_id):
        self.calls.append(("allocs", namespace, job_id))
        return deepcopy(self.allocs)

    def stop_job(self, namespace, job_id, *, purge=False):
        self.calls.append(("stop", namespace, job_id, purge))


@pytest.fixture
def fixture():
    client = FakeNomadClient()
    expected = job()
    reservations = {("sandbox", expected["ID"]): expected}
    adapter = NomadSchedulerAdapter(
        client,
        lambda ns, identifier: deepcopy(reservations.get((ns, identifier))),
        approved_image_digest=IMAGE,
        expected_create_index=lambda ns, identifier: (
            17 if (ns, identifier) in reservations else None
        ),
        enabled=True,
    )
    client.inspected = {**deepcopy(expected), "CreateIndex": 17}
    client.status = {
        "ID": expected["ID"],
        "Namespace": "sandbox",
        "CreateIndex": 17,
        "Status": "running",
    }
    return client, adapter, reservations


def test_disabled_by_default_and_no_client_constructor_side_effects(fixture):
    client, adapter, reservations = fixture
    adapter.enabled = False
    with pytest.raises(NomadAdapterError, match="disabled"):
        adapter.submit("sandbox", job())
    assert not client.calls
    assert reservations


def test_submit_rejects_unfinished_custody_even_for_exact_reviewed_job(fixture):
    client, adapter, _ = fixture
    with pytest.raises(NomadAdapterError, match="custody"):
        adapter.submit("sandbox", job())
    assert not client.calls


@pytest.mark.parametrize(
    "change",
    [
        lambda j: j.update(NodePool="production"),
        lambda j: j.update(Namespace="production"),
        lambda j: j.update(Datacenters=["dc2"]),
        lambda j: j.update(ID="foundry-run-different123"),
        lambda j: j["TaskGroups"][0]["Tasks"][0]["Config"].update(image="worker:latest"),
        lambda j: j["TaskGroups"][0]["Tasks"][0]["Config"].update(command="sh"),
        lambda j: j["TaskGroups"][0]["Tasks"][0].update(Env={"TOKEN": "secret"}),
        lambda j: j["TaskGroups"][0]["Tasks"][0]["Config"].update(volumes=["/tmp:/mnt"]),
        lambda j: j.update(Template=[{"data": "secret"}]),
        lambda j: j["TaskGroups"][0].update(Count=2),
        lambda j: j["Meta"].update(snapshot_id="../path"),
        lambda j: j["TaskGroups"][0].update(MaxRunDuration=0),
        lambda j: j["TaskGroups"][0]["Tasks"][0]["Resources"].update(CPU=99999),
    ],
)
def test_rejects_altered_template_without_nomad_call(fixture, change):
    client, adapter, _ = fixture
    changed = job()
    change(changed)
    with pytest.raises(NomadAdapterError):
        adapter.submit("sandbox", changed)
    assert client.calls == []


def test_rejects_wrong_scope_missing_config_and_missing_reservation(fixture):
    client, adapter, reservations = fixture
    with pytest.raises(NomadAdapterError):
        adapter.lookup("production", job()["ID"])
    client.config.allowed_node_pools.clear()
    with pytest.raises(NomadAdapterError):
        adapter.submit("sandbox", job())
    client.config.allowed_node_pools.append("staging")
    reservations.clear()
    with pytest.raises(NomadAdapterError, match="evidence missing"):
        adapter.lookup("sandbox", job()["ID"])
    assert not client.calls


def test_disabled_submission_never_creates_unknown_outcome(fixture):
    client, adapter, _ = fixture
    with pytest.raises(NomadAdapterError, match="custody"):
        adapter.submit("sandbox", job())
    assert not client.calls


@pytest.mark.parametrize(
    "field, value",
    [
        ("ID", "foundry-run-another123"),
        ("Namespace", "production"),
        ("NodePool", "production"),
        ("CreateIndex", 18),
    ],
)
def test_lookup_rejects_wrong_scheduler_identity(fixture, field, value):
    client, adapter, _ = fixture
    client.inspected[field] = value
    with pytest.raises(NomadAdapterError):
        adapter.lookup("sandbox", job()["ID"])
    assert not any(c[0] == "stop" for c in client.calls)


def test_lookup_requires_status_identity_and_incarnation(fixture):
    client, adapter, _ = fixture
    client.status["CreateIndex"] = 18
    with pytest.raises(NomadAdapterError, match="identity"):
        adapter.lookup("sandbox", job()["ID"])
    client.status["CreateIndex"] = 17
    observed = adapter.lookup("sandbox", job()["ID"])
    assert observed.state == "running" and observed.job == job()


def test_missing_or_unknown_lookup_never_implies_absent(fixture):
    client, adapter, _ = fixture
    client.inspected = None
    with pytest.raises(TimeoutError):
        adapter.lookup("sandbox", job()["ID"])
    client.inspected = {**job(), "CreateIndex": 17}
    client.status["Status"] = "unknown"
    with pytest.raises(NomadAdapterError, match="unresolved"):
        adapter.lookup("sandbox", job()["ID"])


@pytest.mark.parametrize(
    "allocation_states, result",
    [
        (["complete"], "complete"),
        (["complete", "failed"], "failed"),
    ],
)
def test_terminal_state_requires_exact_allocation_evidence(fixture, allocation_states, result):
    client, adapter, _ = fixture
    client.status["Status"] = "dead"
    client.allocs = [
        {"JobID": job()["ID"], "Namespace": "sandbox", "ClientStatus": state}
        for state in allocation_states
    ]
    assert adapter.lookup("sandbox", job()["ID"]).state == result
    client.allocs[0]["Namespace"] = "production"
    with pytest.raises(NomadAdapterError):
        adapter.lookup("sandbox", job()["ID"])


def test_stop_rechecks_exact_job_but_refuses_nonatomic_mutation(fixture):
    client, adapter, _ = fixture
    with pytest.raises(NomadAdapterError, match="atomic"):
        adapter.stop("sandbox", job()["ID"])
    client.inspected["NodePool"] = "production"
    with pytest.raises(NomadAdapterError):
        adapter.stop("sandbox", job()["ID"])
    assert not any(call[0] == "stop" for call in client.calls)


def test_stop_requires_previously_pinned_create_index(fixture):
    client, adapter, reservations = fixture
    adapter.expected_create_index = None
    with pytest.raises(NomadAdapterError, match="incarnation"):
        adapter.stop("sandbox", job()["ID"])
    assert not client.calls
    adapter.expected_create_index = lambda *_: 19
    with pytest.raises(NomadAdapterError, match="incarnation"):
        adapter.stop("sandbox", job()["ID"])
    assert not any(call[0] == "stop" for call in client.calls)
    assert reservations


def test_secret_bytes_never_echoed_from_rejected_payload(fixture):
    _, adapter, _ = fixture
    altered = job()
    altered["TaskGroups"][0]["Tasks"][0]["Env"]["API_TOKEN"] = "topsecret123"
    with pytest.raises(NomadAdapterError) as error:
        adapter.submit("sandbox", altered)
    assert "topsecret123" not in str(error.value)
    assert "API_TOKEN" not in str(error.value)
