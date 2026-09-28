"""Actual isolated loopback HTTP transport tests, no installed services touched."""

import http.client
import json
import socket
import threading
from contextlib import contextmanager

import pytest

from missy.repoeval.control import Principal
from missy.repoeval.http_server import (
    MAX_REQUEST_BYTES,
    CredentialVerifier,
    ExternalTLSProxyContract,
    FoundryHTTPServer,
)

PROJECT = "test-project"
PATH = f"/api/projects/{PROJECT}/repositories"
TOKEN = "test-secret-never-log"


class Service:
    def __init__(self):
        self.calls = []

    def list_repositories(self, principal):
        self.calls.append(("read", principal.subject, principal.project_id))
        return [{"id": "repo-one"}]

    def snapshot_start(self, principal, repository_id, commit_sha, idem):
        self.calls.append(("snapshot", principal.subject, repository_id, idem))
        return {"id": "snapshot-one"}


@contextmanager
def listener(service=None, permissions=("read", "snapshot:start")):
    service = service or Service()
    verifier = CredentialVerifier(
        {TOKEN: Principal("fixture-user", PROJECT, frozenset(permissions))}
    )
    server = FoundryHTTPServer(service, PROJECT, verifier)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(server, method="GET", path=PATH, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=7)
    conn.request(method, path, body=body, headers=headers or {})
    response = conn.getresponse()
    data = response.read()
    result = response.status, json.loads(data), dict(response.getheaders())
    conn.close()
    return result


def auth(**headers):
    return {"Authorization": f"Bearer {TOKEN}", **headers}


def raw_request(server, payload):
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=7) as sock:
        sock.settimeout(7)
        sock.sendall(payload)
        file = sock.makefile("rb")
        status = int(file.readline().split()[1])
        headers = {}
        while line := file.readline():
            if line == b"\r\n":
                break
            k, v = line.decode("ascii").strip().split(":", 1)
            headers[k.lower()] = v.strip()
        body = file.read(int(headers["content-length"]))
        return status, json.loads(body)


@pytest.mark.parametrize("prefix", ("/api", ""))
def test_live_loopback_read_and_project_scope(prefix):
    path = f"{prefix}/projects/{PROJECT}/repositories"
    with listener() as (server, service):
        assert server.server_address[0] == "127.0.0.1"
        status, data, headers = request(server, path=path, headers=auth())
        assert (status, data["data"]) == (200, [{"id": "repo-one"}])
        assert headers["Cache-Control"] == "no-store"
        assert service.calls == [("read", "fixture-user", PROJECT)]
        for denied_path in (
            f"{prefix}/projects/other/repositories",
            path + "/another",
            path + "?query=true",
            f"{prefix}/projects/test-projected/repositories",
            "/api/v1/projects/test-project/repositories",
            "/v1/projects/test-project/repositories",
            f"{prefix}/projects/test-project/%72epositories",
        ):
            assert request(server, path=denied_path, headers=auth())[0] == 404
        assert len(service.calls) == 1


def test_auth_failure_and_invalid_methods_do_not_dispatch(capsys):
    with listener() as (server, service):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}):
            assert request(server, headers=headers)[0] == 401
        for method in ("PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"):
            assert request(server, method=method, headers=auth())[0] == 405
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=7)
        conn.request("HEAD", PATH, headers=auth())
        assert conn.getresponse().status == 405
        conn.close()
        assert not service.calls
    assert TOKEN not in capsys.readouterr().err


def test_post_json_and_operation_authorization():
    body = json.dumps({"repository_id": "repo-one", "commit_sha": "a" * 40})
    with listener() as (server, service):
        headers = auth(**{"Content-Type": "application/json", "Idempotency-Key": "once"})
        status, data, _ = request(
            server, "POST", "/api/projects/test-project/snapshots", body, headers
        )
        assert status == 202
        assert data["data"]["id"] == "snapshot-one"
        assert service.calls[-1] == ("snapshot", "fixture-user", "repo-one", "once")
        assert (
            request(
                server,
                "POST",
                "/api/projects/test-project/snapshots",
                body,
                auth(**{"Content-Type": "application/json"}),
            )[0]
            == 400
        )
    with listener(permissions=("read",)) as (server, service):
        assert (
            request(server, "POST", "/api/projects/test-project/snapshots", body, headers)[0] == 403
        )
        assert service.calls == []


def test_reject_framing_and_bad_json_before_service():
    with listener() as (server, service):
        base = f"GET {PATH} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\n"
        for rest in (
            "Authorization: Bearer wrong\r\n",
            "Transfer-Encoding: chunked\r\n",
            "Content-Length: 1\r\nContent-Length: 1\r\n",
            "Content-Length: -1\r\n",
            "Content-Length: 01\r\n",
            "Content-Length: 1\r\n",
            f"Content-Length: {MAX_REQUEST_BYTES + 1}\r\n",
            "X-Long: " + "x" * 17000 + "\r\n",
            "X-Padding: benign\r\n Authorization: Bearer anything\r\n",
            "Header without colon\r\n",
        ):
            assert raw_request(server, (base + rest + "\r\n").encode())[0] in (400, 413, 405)
        post = (
            f"POST /api/projects/{PROJECT}/snapshots HTTP/1.1\r\n"
            f"Host: localhost\r\nAuthorization: Bearer {TOKEN}\r\n"
        )
        assert (
            raw_request(server, (post + "Content-Type: application/json\r\n\r\n").encode())[0]
            == 411
        )
        for content in (b"[]", b"{", b"\xff"):
            headers = (
                post
                + "Content-Type: application/json\r\n"
                + f"Content-Length: {len(content)}\r\n\r\n"
            ).encode()
            assert raw_request(server, headers + content)[0] == 400
        assert service.calls == []


def test_post_requires_json_and_strict_idempotency_header():
    path = f"/api/projects/{PROJECT}/snapshots"
    with listener() as (server, service):
        base = (
            f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
            f"Authorization: Bearer {TOKEN}\r\nContent-Type: application/json\r\n"
            "Content-Length: 2\r\n"
        )
        for extra in (
            "Idempotency-Key: valid\r\nIdempotency-Key: another\r\n",
            "Idempotency-Key: bad value\r\n",
            "Transfer-Encoding: chunked\r\n",
        ):
            assert raw_request(server, (base + extra + "\r\n{}").encode())[0] == 400
        assert request(server, "POST", path, b"{}", auth())[0] == 415
        assert service.calls == []


def test_no_secrets_or_unbounded_response(capsys):
    class BrokenService(Service):
        def list_repositories(self, principal):
            raise RuntimeError("secret: " + TOKEN)

    with listener(BrokenService()) as (server, _):
        status, data, _ = request(server, path=PATH + "?password=secret", headers=auth())
        assert status == 404
        status, data, _ = request(server, headers=auth())
        assert (status, data["error"]["category"]) == (503, "service_failure")
        assert TOKEN not in repr(data)

    class LargeService(Service):
        def list_repositories(self, principal):
            return ["x" * (2 * 1024 * 1024)]

    with listener(LargeService()) as (server, _):
        status, data, _ = request(server, headers=auth())
        assert (status, data["error"]["category"]) == (503, "service_failure")
    assert TOKEN not in capsys.readouterr().err


def test_credentials_and_bind_require_explicit_operator_decisions():
    with pytest.raises(ValueError):
        CredentialVerifier({})
    with pytest.raises(ValueError):
        CredentialVerifier({"bad token": Principal("u", PROJECT, frozenset({"read"}))})
    verifier = CredentialVerifier({TOKEN: Principal("u", PROJECT, frozenset({"read"}))})
    assert verifier.verify(TOKEN).project_id == PROJECT
    assert verifier.verify("other") is None
    assert TOKEN not in repr(verifier._entries)
    with pytest.raises(ValueError):
        FoundryHTTPServer(Service(), "other", verifier)
    with pytest.raises(ValueError):
        FoundryHTTPServer(Service(), PROJECT, verifier, bind_host="0.0.0.0")
    with pytest.raises(ValueError):
        FoundryHTTPServer(
            Service(),
            PROJECT,
            verifier,
            bind_host="0.0.0.0",
            external_tls_proxy_contract=ExternalTLSProxyContract(True, True, False),
        )
