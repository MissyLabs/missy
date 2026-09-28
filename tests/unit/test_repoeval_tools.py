"""Foundry client safety tests. All responses are in-memory, never networked."""

from __future__ import annotations

import os

import httpx
import pytest

from missy.gateway.client import PolicyHTTPClient
from missy.tools.builtin.repoeval_tools import (
    RepoevalFoundryTool,
    _no_secrets,
    _safe,
    _token_file_valid,
)

SHA = "a" * 40
IMAGE = "sandbox@sha256:" + "b" * 64
TEST_TOKEN = "test-only-foundry-credential-123456"
TOKEN_PATH = ""


@pytest.fixture(autouse=True)
def protected_token(tmp_path, monkeypatch):
    path = tmp_path / "foundry.token"
    path.write_text(TEST_TOKEN + "\n", encoding="ascii")
    path.chmod(0o600)
    monkeypatch.setitem(globals(), "TOKEN_PATH", str(path))
    return path


class FakeResponse:
    status_code = 200

    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status

    def json(self):
        return self._data


class FakeClient:
    def __init__(self):
        self.category = "tool"
        self.calls = []
        self.responses = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def response(data, *, status=200, project="alpha"):
    return FakeResponse({"ok": True, "data": {"project_id": project, **data}}, status=status)


def tool(*, host="foundry.example", available=True, base_path="/api", authenticated=True):
    client = FakeClient()
    instance = RepoevalFoundryTool(
        base_url=f"https://{host}{base_path}",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        http_client=client,
        api_available=available,
        token_file=TOKEN_PATH if authenticated else "",
    )
    return instance, client


def list_repos(instance, client):
    client.responses.append(FakeResponse({"ok": True, "data": ["repo-a"]}))
    result = instance.execute(action="list")
    assert result.success
    assert result.output == {
        "repositories": ["repo-a"],
        "route_project_id": "alpha",
        "scope_source": "authenticated_project_route",
    }


def workload():
    return {
        "schema_version": "1.0",
        "id": "workload-a",
        "version": "1",
        "repository": {"repository_id": "repo-a", "commit_sha": SHA, "snapshot_id": "snap-1"},
        "sandbox": {"image_digest": IMAGE},
        "execution": {"repetitions": 1, "parallelism": 1, "timeout_seconds": 60, "max_attempts": 1},
        "providers": [{"registry_key": "approved"}],
    }


def plan(instance, client, **changes):
    list_repos(instance, client)
    data = {
        "id": "plan-a",
        "state": "planned",
        "workload": workload(),
        "placement": {"pool": "staging"},
        "policy_checks": dict.fromkeys(
            ("repository", "image", "providers", "budget", "quota", "egress", "audit"), True
        ),
    }
    data.update(changes)
    client.responses.append(response(data))
    assert instance.execute(action="plan", workload=workload()).success


def test_default_and_host_allowlist_fail_closed():
    for instance in (
        RepoevalFoundryTool(
            base_url="https://foundry.example",
            project_id="alpha",
            allowed_hosts=["foundry.example"],
        ),
        tool(host="attacker.example")[0],
        tool(base_path="/api/v1")[0],
        tool(authenticated=False)[0],
        RepoevalFoundryTool(
            base_url="https://user:pass@foundry.example",
            project_id="alpha",
            allowed_hosts=["foundry.example"],
            api_available=True,
        ),
        RepoevalFoundryTool(
            base_url="http://foundry.example",
            project_id="alpha",
            allowed_hosts=["foundry.example"],
            api_available=True,
        ),
        RepoevalFoundryTool(
            base_url="https://foundry.example?url=https://attacker.example",
            project_id="alpha",
            allowed_hosts=["foundry.example"],
            api_available=True,
        ),
    ):
        assert not instance.execute(action="list").success


def test_no_unauthenticated_transport_or_non_tool_network_category():
    for client in (None, FakeClient()):
        instance = RepoevalFoundryTool(
            base_url="https://foundry.example/api",
            project_id="alpha",
            allowed_hosts=["foundry.example"],
            http_client=client,
            api_available=True,
        )
        assert not instance.execute(action="list").success
        if client is not None:
            assert not client.calls
    client = FakeClient()
    client.category = "repoeval_foundry"  # not a network policy category
    instance = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        http_client=client,
        api_available=True,
        token_file=TOKEN_PATH,
    )
    assert not instance.execute(action="list").success
    assert not client.calls


def test_lazy_default_policy_client_is_tool_category():
    instance = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        api_available=True,
        token_file=TOKEN_PATH,
    )
    assert instance.registration_ready
    assert isinstance(instance._client, PolicyHTTPClient)
    assert instance._client.category == "tool"
    assert instance._client._sync_client is None


def test_root_and_api_routes_both_match_foundry_contract():
    for base_path in ("", "/api"):
        instance, client = tool(base_path=base_path)
        list_repos(instance, client)
        assert client.calls[0][1] == (
            f"https://foundry.example{base_path}/projects/alpha/repositories"
        )


@pytest.mark.parametrize("project_id", [".", "..", "owner/repo", "owner%2frepo"])
def test_project_route_identity_rejects_traversal_or_repository_path(project_id):
    client = FakeClient()
    instance = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id=project_id,
        allowed_hosts=["foundry.example"],
        http_client=client,
        api_available=True,
        token_file=TOKEN_PATH,
    )
    assert not instance.registration_ready
    assert not instance.execute(action="list").success
    assert not client.calls


def test_short_secret_shaped_metadata_is_redacted_without_rejecting_settings():
    assert _safe({"provider": {"description": "sk-short", "max_tokens": 512}}) == {
        "provider": {"description": "[redacted]", "max_tokens": 512}
    }
    assert not _no_secrets({"providers": [{"notes": "sk-short"}]})
    assert _no_secrets({"providers": [{"max_tokens": 512, "token_budget": 1024}]})


def test_project_scope_and_no_remote_error_or_secret_echo():
    instance, client = tool()
    client.responses.append(FakeResponse({"ok": True, "data": ["repo-a"]}))
    assert instance.execute(action="list").success
    assert client.calls[0][1] == "https://foundry.example/api/projects/alpha/repositories"
    assert client.calls[0][2]["follow_redirects"] is False
    client.responses.append(
        FakeResponse({"ok": False, "error": {"message": "Bearer super-secret"}}, 403)
    )
    denied = instance.execute(action="list")
    assert not denied.success and "super-secret" not in str(denied)
    client.responses.append(
        response(
            {"repositories": [{"id": "repo-a"}], "token": "super-secret", "notes": "Bearer hidden"}
        )
    )
    output = instance.execute(action="list").output
    assert output is None


def test_snapshot_requires_registered_repo_immutable_sha_approval_and_idempotency():
    instance, client = tool()
    request = {
        "action": "snapshot",
        "repository_id": "repo-a",
        "commit_sha": SHA,
        "self_approve": True,
        "idempotency_key": "scan-0001",
    }
    assert not instance.execute(**request).success
    list_repos(instance, client)
    for changes in (
        {"repository_id": "repo-b"},
        {"commit_sha": "main"},
        {"self_approve": False},
        {"idempotency_key": ""},
        {"idempotency_key": "short"},
    ):
        assert not instance.execute(**(request | changes)).success
    assert len(client.calls) == 1
    client.responses.append(
        response(
            {"id": "snap-1", "state": "requested", "repository_id": "repo-a", "commit_sha": SHA},
            status=202,
        )
    )
    result = instance.execute(**request)
    assert result.success and result.output["acknowledged"]
    assert result.output["execution_complete"] is False
    assert client.calls[-1][2]["headers"]["Idempotency-Key"] == "scan-0001"
    assert client.calls[-1][2]["json"] == {"repository_id": "repo-a", "commit_sha": SHA}


def test_slash_repository_identity_only_in_json_bodies_and_not_other_resource_ids():
    instance, client = tool()
    client.responses.append(FakeResponse({"ok": True, "data": ["MissyLabs/missy"]}))
    assert instance.execute(action="list").output["repositories"] == ["MissyLabs/missy"]
    request = {
        "action": "snapshot",
        "repository_id": "MissyLabs/missy",
        "commit_sha": SHA,
        "self_approve": True,
        "idempotency_key": "scan-slash-repo-01",
    }
    client.responses.append(
        response(
            {
                "id": "snap-1",
                "repository_id": "MissyLabs/missy",
                "commit_sha": SHA,
                "state": "requested",
            },
            status=202,
        )
    )
    assert instance.execute(**request).success
    assert client.calls[-1][1] == "https://foundry.example/api/projects/alpha/snapshots"
    assert client.calls[-1][2]["json"]["repository_id"] == "MissyLabs/missy"
    candidate = workload()
    candidate["repository"]["repository_id"] = "MissyLabs/missy"
    client.responses.append(response({"id": "plan-a", "state": "planned", "workload": candidate}))
    assert "lacks reviewed" in instance.execute(action="plan", workload=candidate).error
    assert client.calls[-1][1] == "https://foundry.example/api/projects/alpha/benchmark/plan"
    assert (
        client.calls[-1][2]["json"]["workload"]["repository"]["repository_id"] == "MissyLabs/missy"
    )
    count = len(client.calls)
    for action, args in (
        ("status", {"resource_type": "snapshot", "resource_id": "MissyLabs/missy"}),
        ("status", {"resource_type": "snapshot", "resource_id": ".."}),
        ("status", {"resource_type": "snapshot", "resource_id": "%2e%2e"}),
        ("cancel", {"run_id": "MissyLabs/missy", "idempotency_key": "cancel-test-01"}),
        ("cancel", {"run_id": "..", "idempotency_key": "cancel-test-01"}),
        (
            "start",
            {
                "plan_id": "MissyLabs/missy",
                "self_approve": True,
                "idempotency_key": "start-test-01",
            },
        ),
    ):
        assert not instance.execute(action=action, **args).success
    assert len(client.calls) == count


@pytest.mark.parametrize(
    "malicious",
    [
        "../repo",
        "owner/../repo",
        "owner//repo",
        "owner/repo/extra",
        "owner/%2e%2e",
        "owner%2frepo",
        "https://host/owner/repo",
        "owner\\repo",
        "owner/repo?x=y",
    ],
)
def test_invalid_repository_ids_fail_before_snapshot_or_plan_request(malicious):
    instance, client = tool()
    list_repos(instance, client)
    count = len(client.calls)
    assert not instance.execute(
        action="snapshot",
        repository_id=malicious,
        commit_sha=SHA,
        self_approve=True,
        idempotency_key="invalid-repo-01",
    ).success
    candidate = workload()
    candidate["repository"]["repository_id"] = malicious
    assert not instance.execute(action="plan", workload=candidate).success
    assert len(client.calls) == count


def test_plan_requires_pinned_project_repository_and_bounded_inputs():
    instance, client = tool()
    assert not instance.execute(action="plan", workload=workload()).success
    list_repos(instance, client)
    for key, value in (
        ("repository", {"repository_id": "different", "commit_sha": SHA, "snapshot_id": "snap-1"}),
        ("sandbox", {"image_digest": "latest"}),
        (
            "execution",
            {"repetitions": 33, "parallelism": 1, "timeout_seconds": 60, "max_attempts": 1},
        ),
    ):
        candidate = workload() | {key: value}
        assert not instance.execute(action="plan", workload=candidate).success
    assert len(client.calls) == 1


def test_plan_refuses_secret_fields_and_response_with_changed_workload():
    instance, client = tool()
    list_repos(instance, client)
    unsafe = workload()
    unsafe["providers"][0]["api_key"] = "private"
    assert not instance.execute(action="plan", workload=unsafe).success
    list_repos(instance, client)
    client.responses.append(
        response(
            {
                "id": "plan-a",
                "state": "planned",
                "workload": {"repository": {"repository_id": "other"}},
                "placement": {"pool": "staging"},
                "policy_checks": dict.fromkeys(
                    ("repository", "image", "providers", "budget", "quota", "egress", "audit"), True
                ),
            }
        )
    )
    assert not instance.execute(action="plan", workload=workload()).success
    assert not instance.execute(
        action="start", plan_id="plan-a", idempotency_key="benchmark-0001", self_approve=True
    ).success


@pytest.mark.parametrize(
    "replacement",
    [
        {"placement": {"pool": "production"}},
        {"policy_checks": {"budget": False}},
        {"placement": {"pool": "staging"}, "policy_checks": {}},
    ],
)
def test_start_refuses_non_staging_or_unverified_server_plan(replacement):
    instance, client = tool()
    list_repos(instance, client)
    data = {
        "id": "plan-a",
        "state": "planned",
        "workload": workload(),
        "placement": {"pool": "staging"},
        "policy_checks": dict.fromkeys(
            ("repository", "image", "providers", "budget", "quota", "egress", "audit"), True
        ),
    }
    data.update(replacement)
    client.responses.append(response(data))
    assert not instance.execute(action="plan", workload=workload()).success
    assert not instance.execute(
        action="start", plan_id="plan-a", idempotency_key="benchmark-0001", self_approve=True
    ).success
    assert len(client.calls) == 2


def test_start_requires_explicit_approval_and_reuses_idempotency_key():
    instance, client = tool()
    plan(instance, client)
    assert not instance.execute(
        action="start", plan_id="plan-a", idempotency_key="benchmark-0001", self_approve=False
    ).success
    client.responses.append(
        response(
            {"id": "run-1", "plan_id": "plan-a", "job_id": "job-1", "state": "submitted"},
            status=202,
        )
    )
    started = instance.execute(
        action="start", plan_id="plan-a", idempotency_key="benchmark-0001", self_approve=True
    )
    assert started.success and started.output["acknowledged"]
    assert started.output["execution_complete"] is False
    assert client.calls[-1][2]["headers"]["Idempotency-Key"] == "benchmark-0001"


def test_unknown_actions_arguments_and_capabilities_cannot_make_requests():
    instance, client = tool()
    assert not instance.execute(action="deploy", command="nomad run").success
    assert not instance.execute(action="list", arbitrary="widen-scope").success
    assert not instance.execute(action="capabilities").success  # route not implemented upstream
    assert not client.calls


def test_status_compare_cancel_artifacts_and_report_are_fixed_routes_only():
    instance, client = tool()
    assert not instance.execute(action="status", resource_type="provider", resource_id="x").success
    assert not instance.execute(
        action="cancel", run_id="../foreign", idempotency_key="cancel-0001"
    ).success
    assert not instance.execute(action="compare", run_ids=["run-a", "run-a"]).success
    client.responses += [
        response({"id": "run-a", "state": "running"}),
        response({"id": "run-a", "state": "cancelled"}, status=202),
    ]
    assert instance.execute(action="status", resource_type="run", resource_id="run-a").success
    assert not instance.execute(action="compare", run_ids=["run-a", "run-b"]).success
    assert not instance.execute(action="artifacts", run_id="run-a").success
    assert not instance.execute(action="report", run_ids=["run-a"]).success
    cancelled = instance.execute(action="cancel", run_id="run-a", idempotency_key="cancel-0001")
    assert cancelled.success and cancelled.output["execution_complete"] is False
    assert [call[1].rsplit("/projects/alpha", 1)[-1] for call in client.calls] == [
        "/runs/run-a",
        "/runs/run-a/cancel",
    ]


def test_pr1_plan_without_policy_and_placement_is_rejected_without_start_call():
    instance, client = tool()
    list_repos(instance, client)
    client.responses.append(response({"id": "plan-a", "state": "planned", "workload": workload()}))
    assert "lacks reviewed" in instance.execute(action="plan", workload=workload()).error
    assert not instance.execute(
        action="start", plan_id="plan-a", idempotency_key="benchmark-0001", self_approve=True
    ).success
    assert len(client.calls) == 2


def test_failed_list_refresh_revokes_cached_repository_scope():
    instance, client = tool()
    list_repos(instance, client)
    client.responses.append(FakeResponse({"ok": True, "data": ["../foreign"]}))
    assert not instance.execute(action="list").success
    assert not instance.execute(action="plan", workload=workload()).success
    assert len(client.calls) == 2


def test_pr1_unscoped_compare_report_artifacts_are_explicitly_unavailable():
    instance, client = tool()
    for action, arguments in (
        ("compare", {"run_ids": ["run-a", "run-b"]}),
        ("report", {"run_ids": ["run-a"]}),
        ("artifacts", {"run_id": "run-a"}),
    ):
        result = instance.execute(action=action, **arguments)
        assert not result.success and "unavailable" in result.error
    assert not client.calls


@pytest.mark.parametrize("action", ["snapshot", "start", "cancel"])
def test_mutation_rejects_unbound_or_incomplete_acknowledgement(action):
    instance, client = tool()
    if action == "snapshot":
        list_repos(instance, client)
        kwargs = {
            "action": "snapshot",
            "repository_id": "repo-a",
            "commit_sha": SHA,
            "self_approve": True,
            "idempotency_key": "scan-0001",
        }
        data = {"id": "snap-1", "repository_id": "repo-a", "commit_sha": SHA, "state": "requested"}
    elif action == "start":
        plan(instance, client)
        kwargs = {
            "action": "start",
            "plan_id": "plan-a",
            "self_approve": True,
            "idempotency_key": "benchmark-0001",
        }
        data = {"id": "run-1", "plan_id": "plan-a", "state": "failed", "job_id": None}
    else:
        kwargs = {"action": "cancel", "run_id": "run-a", "idempotency_key": "cancel-0001"}
        data = {"id": "run-a", "state": "cancelled"}
    client.responses.append(response(data, project="other", status=202))
    assert not instance.execute(**kwargs).success
    client.responses.append(response(data, status=202))
    assert instance.execute(**kwargs).success is (action != "start")


@pytest.mark.parametrize("mode", [0o644, 0o400, 0o660, 0o1600])
def test_unsafe_modes_fail_at_request_time(protected_token, mode):
    instance, client = tool()
    protected_token.chmod(mode)
    assert not instance.execute(action="list").success
    assert not client.calls
    assert not _token_file_valid(str(protected_token))


@pytest.mark.parametrize(
    "content", [b"", b"x" * 4097, b"a\nb", b"bad\r\n", b"has spaces", b"\xff", b"a\n\n"]
)
def test_bad_token_bytes_never_send(protected_token, content):
    instance, client = tool()
    protected_token.write_bytes(content)
    assert not instance.execute(action="list").success
    assert not client.calls


def test_symlinks_hardlinks_owner_fifo_and_bad_paths_refused(
    protected_token, tmp_path, monkeypatch
):
    link = tmp_path / "symlink"
    link.symlink_to(protected_token)
    assert not _token_file_valid(str(link))
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    assert not _token_file_valid(str(parent_link / protected_token.name))
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo, 0o600)
    assert not _token_file_valid(str(fifo))
    for path in (
        "relative",
        str(tmp_path / "missing"),
        str(tmp_path),
        str(tmp_path) + "/../" + tmp_path.name + "/foundry.token",
        str(tmp_path) + "//foundry.token",
        str(protected_token) + "\n",
    ):
        assert not _token_file_valid(path)
    real_uid = os.getuid()
    with monkeypatch.context() as patch:
        patch.setattr(os, "getuid", lambda: real_uid + 1)
        assert not _token_file_valid(str(protected_token))
    os.link(protected_token, tmp_path / "hardlink")
    assert not _token_file_valid(str(protected_token))


def test_registration_does_not_read_and_each_request_rereads(protected_token, monkeypatch):
    original_read = os.read
    with monkeypatch.context() as patch:
        patch.setattr(os, "read", lambda *a: pytest.fail("registration read credential"))
        instance, client = tool()
        assert instance.registration_ready
    list_repos(instance, client)
    assert client.calls[-1][2]["headers"]["Authorization"] == f"Bearer {TEST_TOKEN}"
    replacement = "test-only-rotated-token-98765"
    protected_token.write_text(replacement, encoding="ascii")
    list_repos(instance, client)
    assert client.calls[-1][2]["headers"]["Authorization"] == f"Bearer {replacement}"
    assert os.read is original_read
    assert TEST_TOKEN not in repr(instance.__dict__)
    protected_token.unlink()
    assert not instance.execute(action="list").success
    assert len(client.calls) == 2


def test_token_echo_and_transport_errors_are_never_returned(caplog):
    instance, client = tool()
    for data in ([TEST_TOKEN], {"id": "run-a", "notes": TEST_TOKEN}, {TEST_TOKEN: "echo"}):
        client.responses.append(FakeResponse({"ok": True, "data": data}))
        result = instance.execute(action="list")
        assert not result.success
        assert TEST_TOKEN not in str(result)

    def fail(*args, **kwargs):
        raise RuntimeError(f"Authorization: Bearer {TEST_TOKEN}")

    client.get = fail
    result = instance.execute(action="list")
    assert not result.success and TEST_TOKEN not in str(result)
    assert TEST_TOKEN not in caplog.text


def test_response_uris_are_not_exposed_or_fetched():
    instance, client = tool()
    client.responses.append(
        response(
            {
                "id": "run-a",
                "uri": "https://elsewhere.example/?secret=private",
                "artifact_uri": "file:///etc/passwd",
                "notes": "see https://example.test/private",
            }
        )
    )
    result = instance.execute(action="status", resource_type="run", resource_id="run-a")
    assert result.success
    assert (
        result.output["uri"]
        == result.output["artifact_uri"]
        == result.output["notes"]
        == "[redacted]"
    )
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "base",
    [
        "https://foundry.example:0/api",
        "https://foundry.example:/api",
        "https://foundry.example:bad/api",
        "https://foundry.example:65536/api",
        "https://foundry.example\n/api",
        " https://foundry.example/api",
        "https://foundry.example\\evil/api",
        "https://@foundry.example/api",
        "https://foundry.example/%61pi",
        "https://foundry.example/api/../api",
    ],
)
def test_malformed_endpoint_never_registers(base):
    instance = RepoevalFoundryTool(
        base_url=base,
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        api_available=True,
        token_file=TOKEN_PATH,
    )
    assert not instance.registration_ready


def test_real_policy_gateway_with_fake_http_transport_keeps_auth_private(monkeypatch, caplog):
    import missy.gateway.client as gateway
    from missy.core.events import event_bus

    checks, events, requests = [], [], []

    class Policy:
        def check_network_resolved(self, host, session, task, *, category):
            checks.append((host, category))
            return True, "192.0.2.1"

    monkeypatch.setattr(gateway, "get_policy_engine", lambda: Policy())
    # Exercise real network-policy routing and pinning, no DNS or real transport.
    monkeypatch.setattr(event_bus, "publish", events.append)

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "data": ["repo-a"]})

    client = PolicyHTTPClient(category="tool")
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(handle), follow_redirects=False
    )
    instance = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        api_available=True,
        token_file=TOKEN_PATH,
        http_client=client,
    )
    try:
        result = instance.execute(action="list")
        assert result.success
        assert checks == [("foundry.example", "tool")]
        assert requests[0].headers["Authorization"] == f"Bearer {TEST_TOKEN}"
        assert TEST_TOKEN not in str(result) + repr(events) + caplog.text
        # A redirect is refused, not followed even with a credential present.
        client._sync_client.close()
        requests.clear()

        def redirect(request):
            requests.append(request)
            return httpx.Response(302, headers={"location": "https://attacker.example"})

        client._sync_client = httpx.Client(
            transport=httpx.MockTransport(redirect), follow_redirects=False
        )
        assert not instance.execute(action="list").success
        assert len(requests) == 1
    finally:
        client.close()


def test_network_policy_denial_never_reaches_transport(monkeypatch):
    import missy.gateway.client as gateway
    from missy.core.exceptions import PolicyViolationError

    checks = []

    class Policy:
        def check_network_resolved(self, host, session, task, *, category):
            checks.append((host, category))
            raise PolicyViolationError("Denied", category="network")

    monkeypatch.setattr(gateway, "get_policy_engine", lambda: Policy())
    monkeypatch.setattr(gateway, "_interactive_approval", None)
    client = PolicyHTTPClient(category="tool")
    client._sync_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail("Policy-denied request reached transport")
        )
    )
    instance = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        api_available=True,
        token_file=TOKEN_PATH,
        http_client=client,
    )
    try:
        assert not instance.execute(action="list").success
        assert checks == [("foundry.example", "tool")]
    finally:
        client.close()
