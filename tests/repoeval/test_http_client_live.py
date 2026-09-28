"""Actual HTTP wire between the policy gateway, Foundry tool, and scoped server.

Only a temporary 127.0.0.1 listener and a test credential exist here. No
production service, scheduler, repository checkout or external host is used.
"""

from __future__ import annotations

import http.client
import threading
from contextlib import contextmanager

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.gateway.client import PolicyHTTPClient
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine
from missy.repoeval.control import Principal
from missy.repoeval.http_server import CredentialVerifier, FoundryHTTPServer, _FoundryRequestHandler
from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool
from tests.repoeval.test_client_core_contract import (
    COMMIT,
    PROJECT,
    REPO,
    TOKEN,
    _verified_test_run,
    _wired,
)


@pytest.fixture
def loopback_policy():
    """Explicitly allow only loopback for the gateway; restore global policy."""
    previous = engine_module._engine
    init_policy_engine(
        MissyConfig(
            network=NetworkPolicy(default_deny=True, allowed_cidrs=["127.0.0.0/8"]),
            filesystem=FilesystemPolicy(),
            shell=ShellPolicy(),
            plugins=PluginPolicy(),
            providers={},
            workspace_path="/tmp",
            audit_log_path="/tmp/missy-foundry-test-audit.log",
        )
    )
    try:
        yield
    finally:
        engine_module._engine = previous


@contextmanager
def live_client(tmp_path, *, prefix="/api", redirected=False):
    # Reuse the contract fixture's real, dispatcher-free FoundryService and
    # test-owned 0600 token. Replace its in-memory wire with the actual gateway.
    _, _, service = _wired(tmp_path, planning=True)
    principal = Principal("fixture-user", PROJECT, frozenset({"read", "snapshot:start", "execute"}))
    verifier = CredentialVerifier({TOKEN: principal})
    server = FoundryHTTPServer(service, PROJECT, verifier, port=0)
    requests = []

    class RecordingHandler(_FoundryRequestHandler):
        def do_GET(self):
            requests.append(("GET", self.path))
            super().do_GET()

        def do_POST(self):
            requests.append(("POST", self.path))
            super().do_POST()

    if redirected:

        class RedirectHandler(RecordingHandler):
            def do_GET(self):
                if self.path.endswith("/repositories") and not requests:
                    requests.append(("GET", self.path))
                    self.send_response(302)
                    self.send_header(
                        "Location",
                        f"http://127.0.0.1:{self.server.server_port}/api/projects/foreign/repositories",
                    )
                    self.send_header("Content-Length", "0")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    return
                super().do_GET()

        server.RequestHandlerClass = RedirectHandler
    else:
        server.RequestHandlerClass = RecordingHandler
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway = PolicyHTTPClient(category="tool", timeout=3, max_response_bytes=1024 * 1024)
    tool = RepoevalFoundryTool(
        base_url=f"http://127.0.0.1:{server.server_port}{prefix}",
        project_id=PROJECT,
        allowed_hosts=["127.0.0.1"],
        http_client=gateway,
        api_available=True,
        token_file=str(tmp_path / "fixture.token"),
    )
    try:
        assert tool.registration_ready
        yield tool, gateway, service, server, requests
    finally:
        gateway.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert server.socket.fileno() == -1


@pytest.mark.parametrize("prefix", ("/api", ""))
def test_real_wire_list_snapshot_status_compare_and_no_start(tmp_path, loopback_policy, prefix):
    with live_client(tmp_path, prefix=prefix) as (tool, _gateway, service, server, requests):
        assert server.server_address[0] == "127.0.0.1"
        listed = tool.execute(action="list")
        assert listed.success and listed.output["repositories"] == [REPO]
        assert listed.output["route_project_id"] == PROJECT

        requested = tool.execute(
            action="snapshot",
            repository_id=REPO,
            commit_sha=COMMIT,
            idempotency_key="wire-snapshot-01",
            self_approve=True,
        )
        assert requested.success and requested.output["acknowledged"] is True
        snapshot_id = requested.output["resource"]["id"]
        status = tool.execute(action="status", resource_type="snapshot", resource_id=snapshot_id)
        assert status.success and status.output["state"] == "requested"
        assert status.output["project_id"] == PROJECT

        first, second = "run-wire0001", "run-wire0002"
        _verified_test_run(service, first)
        _verified_test_run(service, second)
        compared = tool.execute(action="compare", run_ids=[first, second])
        assert compared.success and compared.output["comparable"] is True
        assert compared.output["project_id"] == PROJECT

        assert service.dispatcher is None and service.allow_demo_dispatch is False
        assert not tool.execute(
            action="start",
            plan_id="plan-unsupported",
            idempotency_key="wire-start-01",
            self_approve=True,
        ).success
        assert requests == [
            ("GET", f"{prefix}/projects/{PROJECT}/repositories"),
            ("POST", f"{prefix}/projects/{PROJECT}/snapshots"),
            ("GET", f"{prefix}/projects/{PROJECT}/snapshots/{snapshot_id}"),
            ("POST", f"{prefix}/projects/{PROJECT}/compare"),
        ]
        assert not service.store.get("plan-unsupported")


def test_project_boundary_and_authentication_on_real_wire(tmp_path, loopback_policy):
    with live_client(tmp_path) as (tool, gateway, service, server, requests):
        assert tool.execute(action="list").success
        foreign_client = RepoevalFoundryTool(
            base_url=f"http://127.0.0.1:{server.server_port}/api",
            project_id="foreign",
            allowed_hosts=["127.0.0.1"],
            http_client=gateway,
            api_available=True,
            token_file=str(tmp_path / "fixture.token"),
        )
        assert foreign_client.registration_ready
        assert not foreign_client.execute(action="list").success
        assert requests[-1] == ("GET", "/api/projects/foreign/repositories")
        for path, expected in (
            (f"/api/projects/{PROJECT}/repositories", 401),
            ("/api/projects/foreign/repositories", 404),
            (f"/projects/{PROJECT}-foreign/repositories", 404),
            (f"/v1/projects/{PROJECT}/repositories", 404),
            (f"/api/v1/projects/{PROJECT}/repositories", 404),
        ):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            headers = {} if expected == 401 else {"Authorization": "Bearer " + TOKEN}
            try:
                conn.request("GET", path, headers=headers)
                response = conn.getresponse()
                assert response.status == expected
                response.read()
            finally:
                conn.close()
        assert service.repositories == {PROJECT: frozenset({REPO})}


def test_real_gateway_does_not_follow_redirect_or_request_other_project(tmp_path, loopback_policy):
    with live_client(tmp_path, redirected=True) as (tool, _gateway, _service, _server, requests):
        assert not tool.execute(action="list").success
        assert requests == [("GET", f"/api/projects/{PROJECT}/repositories")]
