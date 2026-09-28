from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from missy.config.settings import NomadConfig
from missy.nomad.client import NomadClient
from missy.nomad.credentials import load_credentials
from missy.nomad.errors import (
    NomadAuthorizationError,
    NomadCommandError,
    NomadMutationUnknown,
)
from missy.nomad.store import NomadStateStore


class Runner:
    def __init__(self, responses: list[subprocess.CompletedProcess[str] | BaseException]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def proc(stdout: str = "", stderr: str = "", code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["nomad"], code, stdout, stderr)


def client(config: NomadConfig, runner: Runner) -> NomadClient:
    return NomadClient(config, credentials=load_credentials(config), runner=runner)


def test_token_is_environment_only(nomad_config: NomadConfig) -> None:
    runner = Runner([proc("[]")])
    assert client(nomad_config, runner).nodes() == []
    argv, kwargs = runner.calls[0]
    assert "test-token-value" not in " ".join(argv)
    assert kwargs["env"]["NOMAD_TOKEN"] == "test-token-value"
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert "-tls-skip-verify" not in argv


def test_plan_exit_one_is_success(nomad_config: NomadConfig) -> None:
    runner = Runner([proc('{"JobModifyIndex":0,"Diff":{}}', code=1)])
    result = client(nomad_config, runner).plan_job("testing", {"Job": {}})
    assert result["_exit_code"] == 1
    assert runner.calls[0][1]["input"] is not None
    assert runner.calls[0][1]["stdin"] is None


def test_invalid_json_fails_closed(nomad_config: NomadConfig) -> None:
    runner = Runner([proc("not json")])
    with pytest.raises(NomadCommandError, match="invalid JSON"):
        client(nomad_config, runner).nodes()


def test_403_is_classified(nomad_config: NomadConfig) -> None:
    runner = Runner([proc(stderr="Unexpected response code: 403 permission denied", code=1)])
    with pytest.raises(NomadAuthorizationError, match="namespace list.*cluster-visible scope"):
        client(nomad_config, runner).namespaces()


def test_mutation_timeout_has_unknown_effect(nomad_config: NomadConfig) -> None:
    runner = Runner([subprocess.TimeoutExpired(cmd="nomad", timeout=60)])
    with pytest.raises(NomadMutationUnknown, match="effect is unknown"):
        client(nomad_config, runner).run_job(
            "testing", {"Job": {"ID": "missy-task"}}, check_index=0
        )


def test_read_timeout_is_not_unknown_mutation(nomad_config: NomadConfig) -> None:
    runner = Runner([subprocess.TimeoutExpired(cmd="nomad", timeout=60)])
    with pytest.raises(NomadCommandError, match="read-only"):
        client(nomad_config, runner).nodes()


def test_identifier_prevents_argument_injection(nomad_config: NomadConfig) -> None:
    runner = Runner([])
    with pytest.raises(NomadCommandError, match="Invalid Nomad job id"):
        client(nomad_config, runner).job_status("testing", "--tls-skip-verify")
    assert runner.calls == []


def test_all_fixed_read_and_lifecycle_commands(nomad_config: NomadConfig) -> None:
    responses = [
        proc('[{"Name":"testing"}]'),
        proc('[{"Name":"staging"}]'),
        proc('{"ID":"node-1"}'),
        proc("[]"),
        proc('{"Memory":{"Available":1}}'),
        proc('[{"ID":"missy-job"}]'),
        proc('{"Job":{"ID":"missy-job"}}'),
        proc('{"ID":"missy-job","Status":"running"}'),
        proc('[{"ID":"alloc-1"}]'),
        proc('[{"ID":"deploy-1"}]'),
        proc('{"ID":"eval-1"}'),
        proc('{"ID":"alloc-1"}'),
        proc("valid"),
        proc("eval-2\n"),
        proc("restarted"),
        proc("scaled"),
        proc("stopped"),
        proc("purged"),
    ]
    runner = Runner(responses)
    api = client(nomad_config, runner)
    assert api.namespaces()[0]["Name"] == "testing"
    assert api.node_pools()[0]["Name"] == "staging"
    assert api.node("node-1", stats=True)["ID"] == "node-1"
    assert api.node_allocations("node-1") == []
    assert api.node_stats("node-1")["Memory"]["Available"] == 1
    assert api.jobs("testing")[0]["ID"] == "missy-job"
    assert api.inspect_job("testing", "missy-job")["ID"] == "missy-job"
    assert api.job_status("testing", "missy-job")["Status"] == "running"
    assert api.allocations("testing", "missy-job")[0]["ID"] == "alloc-1"
    assert api.deployments("testing", "missy-job")[0]["ID"] == "deploy-1"
    assert api.evaluation("testing", "eval-1")["ID"] == "eval-1"
    assert api.allocation("testing", "alloc-1")["ID"] == "alloc-1"
    assert api.validate_job("testing", {"Job": {}}) == "valid"
    assert api.run_job("testing", {"Job": {"ID": "missy-job"}}, check_index=0) == "eval-2"
    assert api.restart_job("testing", "missy-job") == "restarted"
    assert api.scale_job("testing", "missy-job", "workload", 2) == "scaled"
    assert api.stop_job("testing", "missy-job") == "stopped"
    assert api.stop_job("testing", "missy-job", purge=True) == "purged"
    assert any("-purge" in argv for argv, _ in runner.calls)


def test_logs_are_bounded_and_censored(nomad_config: NomadConfig) -> None:
    runner = Runner([proc("token=abc123456789012345678901234567890\n")])
    output = client(nomad_config, runner).allocation_logs(
        "testing", "alloc-1", "workload", lines=99_999
    )
    assert "abc123" not in output
    assert "500" in runner.calls[0][0]


def test_store_atomic_private_round_trip(tmp_path: Path) -> None:
    store = NomadStateStore(str(tmp_path / "state"))
    store.put_plan("p1", {"spec": {"Job": {"ID": "missy-one"}}})
    store.put_job("testing", "missy-one", {"job_id": "missy-one"})
    assert store.get_plan("p1")["spec"]["Job"]["ID"] == "missy-one"
    assert store.get_job("testing", "missy-one")["job_id"] == "missy-one"
    assert os.stat(store.directory).st_mode & 0o777 == 0o700
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    text = store.path.read_text()
    assert json.loads(text)["version"] == 1


def test_store_does_not_return_mutable_internal_state(tmp_path: Path) -> None:
    store = NomadStateStore(str(tmp_path / "state"))
    store.put_job("testing", "missy-one", {"nested": {"value": 1}})
    first = store.get_job("testing", "missy-one")
    first["nested"]["value"] = 2
    assert store.get_job("testing", "missy-one")["nested"]["value"] == 1
