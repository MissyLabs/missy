"""Client-only loopback wire checks; no Foundry server or core is imported."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool

PROJECT = "missy-evaluation"
TOKEN = "fixture-wire-token"


@contextmanager
def wire_client(tmp_path, *, redirect=False):
    previous = engine_module._engine
    init_policy_engine(
        MissyConfig(
            network=NetworkPolicy(default_deny=True, allowed_cidrs=["127.0.0.0/8"]),
            filesystem=FilesystemPolicy(),
            shell=ShellPolicy(),
            plugins=PluginPolicy(),
            providers={},
            workspace_path=str(tmp_path),
            audit_log_path=str(tmp_path / "audit.log"),
        )
    )
    requests = []

    class FixtureHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append(("GET", self.path, self.headers.get("Authorization")))
            if redirect:
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/foreign")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                self._reply(401, {"ok": False})
            elif self.path == f"/api/projects/{PROJECT}/repositories":
                self._reply(200, {"ok": True, "data": ["MissyLabs/missy"]})
            else:
                self._reply(404, {"ok": False})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            requests.append(("POST", self.path, body))
            if (
                self.path == f"/api/projects/{PROJECT}/snapshots"
                and self.headers.get("Authorization") == f"Bearer {TOKEN}"
                and self.headers.get("Idempotency-Key") == "snapshot-01"
            ):
                self._reply(
                    202,
                    {
                        "ok": True,
                        "data": {
                            "id": "snapshot-1",
                            "project_id": PROJECT,
                            "repository_id": body["repository_id"],
                            "commit_sha": body["commit_sha"],
                            "state": "requested",
                        },
                    },
                )
            else:
                self._reply(404, {"ok": False})

    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "foundry.token"
    path.write_text(TOKEN + "\n")
    path.chmod(0o600)
    gateway = PolicyHTTPClient(category="tool", timeout=3, max_response_bytes=1024 * 1024)
    tool = RepoevalFoundryTool(
        base_url=f"http://127.0.0.1:{server.server_port}/api",
        project_id=PROJECT,
        allowed_hosts=["127.0.0.1"],
        http_client=gateway,
        api_available=True,
        token_file=str(path),
    )
    try:
        yield tool, requests
    finally:
        gateway.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        engine_module._engine = previous


def test_project_pinned_wire_and_snapshot_request(tmp_path):
    with wire_client(tmp_path) as (tool, requests):
        assert tool.registration_ready
        assert tool.execute(action="list").output["repositories"] == ["MissyLabs/missy"]
        assert not tool.execute(
            action="snapshot",
            repository_id="MissyLabs/other",
            commit_sha="a" * 40,
            acknowledge_project_scope=True,
            idempotency_key="snapshot-01",
        ).success
        result = tool.execute(
            action="snapshot",
            repository_id="MissyLabs/missy",
            commit_sha="a" * 40,
            acknowledge_project_scope=True,
            idempotency_key="snapshot-01",
        )
        assert result.success and result.output["execution_complete"] is False
        assert requests[0][:2] == ("GET", f"/api/projects/{PROJECT}/repositories")
        assert requests[1][:2] == ("POST", f"/api/projects/{PROJECT}/snapshots")
        assert requests[1][2] == {"repository_id": "MissyLabs/missy", "commit_sha": "a" * 40}
        assert all(TOKEN not in str(entry[2]) for entry in requests[1:])


def test_redirect_refused_and_no_other_endpoint_called(tmp_path):
    with wire_client(tmp_path, redirect=True) as (tool, requests):
        assert not tool.execute(action="list").success
        assert len(requests) == 1


@pytest.mark.parametrize("port", [1, 65534])
def test_absent_api_fails_closed(tmp_path, port):
    path = tmp_path / "foundry.token"
    path.write_text(TOKEN)
    path.chmod(0o600)
    tool = RepoevalFoundryTool(
        base_url=f"http://127.0.0.1:{port}/api",
        project_id=PROJECT,
        allowed_hosts=["127.0.0.1"],
        http_client=PolicyHTTPClient(category="tool", timeout=0.5),
        api_available=True,
        token_file=str(path),
    )
    assert not tool.execute(action="list").success
