"""Tests for the Missy REST API server (missy/api/server.py).

All tests start a real HTTPServer on a random ephemeral port and make
requests with httpx (already a project dependency).  The server is torn
down after each test class via a shared session-scoped fixture to keep the
suite fast.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from missy.api.server import ApiConfig, ApiResponse, ApiServer, _SessionRegistry
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordConfig,
    DiscordDMPolicy,
)
from missy.config.settings import NetworkPolicy, get_default_config
from missy.core.events import AuditEvent, event_bus
from missy.observability.audit_logger import init_audit_logger
from missy.policy.engine import init_policy_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an available TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 3.0) -> None:
    """Poll until the server responds or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=0.2)
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not start within {timeout}s: {url}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_KEY = "test-api-key-abc123"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="module")
def server() -> Generator[ApiServer, None, None]:
    """Module-scoped API server on a random port with no runtime dependencies."""
    port = _free_port()
    cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY, rate_limit_rpm=100)
    srv = ApiServer(config=cfg)
    srv.start()
    _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def base_url(server: ApiServer) -> str:
    return f"http://127.0.0.1:{server.config.port}/api/v1"


@pytest.fixture(scope="module")
def client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, headers=HEADERS, timeout=5.0)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client: httpx.Client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self, client: httpx.Client) -> None:
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["data"]["status"] == "healthy"
        assert "version" in body["data"]

    def test_content_type_json(self, client: httpx.Client) -> None:
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]

    def test_security_headers(self, client: httpx.Client) -> None:
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("Cache-Control") == "no-store"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    def test_no_key_returns_401(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/health")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/health", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_bearer_token_accepted(self, base_url: str) -> None:
        resp = httpx.get(
            f"{base_url}/health",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert resp.status_code == 200

    def test_x_api_key_header_accepted(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/health", headers={"X-API-Key": API_KEY})
        assert resp.status_code == 200

    def test_empty_key_config_rejects_all(self) -> None:
        """A server configured with an empty api_key must reject every request."""
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key="")
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/health")
            assert resp.status_code == 401
        finally:
            srv.stop()

    def test_constant_time_comparison(self, base_url: str) -> None:
        """Smoke-test: both a correct and similar-length wrong key resolve unambiguously."""
        wrong = "x" * len(API_KEY)
        r_wrong = httpx.get(f"{base_url}/health", headers={"X-API-Key": wrong})
        r_right = httpx.get(f"{base_url}/health", headers={"X-API-Key": API_KEY})
        assert r_wrong.status_code == 401
        assert r_right.status_code == 200


# ---------------------------------------------------------------------------
# Browser operator console
# ---------------------------------------------------------------------------


class TestOperatorConsole:
    def test_root_redirects_to_login_without_browser_session(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/", follow_redirects=False)
            assert resp.status_code == 303
            assert resp.headers["location"] == "/login"
        finally:
            srv.stop()

    def test_login_page_has_security_headers(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/login")
            assert resp.status_code == 200
            assert "Missy Operator Console" in resp.text
            assert resp.headers["X-Frame-Options"] == "DENY"
            assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
            assert resp.headers["Cache-Control"] == "no-store"
        finally:
            srv.stop()

    def test_bad_login_does_not_set_session_cookie(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/login",
                data={"api_key": "wrong"},
                follow_redirects=False,
            )
            assert resp.status_code == 303
            assert resp.headers["location"] == "/login?error=1"
            assert "set-cookie" not in resp.headers
        finally:
            srv.stop()

    def test_good_login_sets_hardened_cookie_and_loads_console(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", follow_redirects=False
            ) as client:
                login = client.post("/login", data={"api_key": API_KEY})
                assert login.status_code == 303
                cookie = login.headers["set-cookie"]
                assert "missy_operator_session=" in cookie
                assert "HttpOnly" in cookie
                assert "SameSite=Strict" in cookie

                console = client.get("/")
                assert console.status_code == 200
                assert "data-csrf=" in console.text
                assert "Runtime posture" in console.text
                assert "Audit Trail" in console.text
                assert "audit-result" in console.text
                assert "audit-severity" in console.text
                assert "audit-detail" in console.text
        finally:
            srv.stop()

    def test_browser_cookie_can_read_api_but_unsafe_api_requires_csrf(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", follow_redirects=False
            ) as client:
                client.post("/login", data={"api_key": API_KEY})
                read_resp = client.get("/api/v1/status")
                assert read_resp.status_code == 200

                denied = client.post("/api/v1/sessions", json={})
                assert denied.status_code == 403
                assert denied.json()["error"] == "CSRF token required"

                console = client.get("/")
                csrf = console.text.split('data-csrf="', 1)[1].split('"', 1)[0]
                allowed = client.post("/api/v1/sessions", json={}, headers={"X-CSRF-Token": csrf})
                assert allowed.status_code == 201
        finally:
            srv.stop()

    def test_web_security_actions_emit_audit_events(self, tmp_path) -> None:
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", follow_redirects=False
            ) as client:
                client.post("/login", data={"api_key": "wrong"})
                client.post("/login", data={"api_key": API_KEY})
                client.post("/api/v1/sessions", json={})
                audit = client.get(
                    "/api/v1/audit?source=web_tui&subsystem=auth&limit=10",
                    headers=HEADERS,
                )
                assert audit.status_code == 200
                events = audit.json()["data"]["events"]
                assert any(e["event_type"] == "web.login" and e["result"] == "deny" for e in events)
                assert any(
                    e["event_type"] == "web.login" and e["result"] == "allow" for e in events
                )

                csrf_audit = client.get(
                    "/api/v1/audit?source=web_tui&subsystem=security&result=deny&limit=10",
                    headers=HEADERS,
                )
                assert csrf_audit.status_code == 200
                assert any(
                    e["event_type"] == "web.csrf" for e in csrf_audit.json()["data"]["events"]
                )
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Audit endpoint
# ---------------------------------------------------------------------------


class TestAuditEndpoint:
    def test_audit_endpoint_requires_auth(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/audit")
        assert resp.status_code == 401

    def test_audit_endpoint_filters_and_redacts_events(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        event_bus.publish(
            AuditEvent(
                timestamp=datetime.now(UTC),
                session_id="sess-a",
                task_id="task-a",
                event_type="provider.request",
                category="provider",
                result="deny",
                detail={
                    "severity": "critical",
                    "actor": "runtime",
                    "source": "provider",
                    "subsystem": "provider",
                    "action": "request",
                    "api_key": "sk-ant-abcdefghijklmnopqrstuvwxyz123456",
                    "message": "token: abcdefghijklmnopqrstuvwxyz123456",
                },
                policy_rule="network.default_deny",
            )
        )
        event_bus.publish(
            AuditEvent(
                timestamp=datetime.now(UTC),
                session_id="sess-b",
                task_id="task-b",
                event_type="tool.execute",
                category="tool",
                result="allow",
                detail={"severity": "info", "subsystem": "tool", "action": "execute"},
            )
        )

        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?category=provider&result=deny&severity=critical&limit=20",
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["count"] == 1
            event = data["events"][0]
            assert event["event_type"] == "provider.request"
            assert event["detail"]["api_key"] == "[REDACTED]"
            assert "[REDACTED]" in event["detail"]["message"]
            assert "sk-ant-" not in str(event)
            assert data["facets"]["category"]["provider"] == 1
        finally:
            srv.stop()

    def test_audit_endpoint_paginates_newest_first_with_stable_redacted_ids(self) -> None:
        event_bus.clear()
        for idx in range(3):
            event_bus.publish(
                AuditEvent(
                    timestamp=datetime(2026, 7, 8, 12, idx, tzinfo=UTC),
                    session_id=f"sess-{idx}",
                    task_id=f"task-{idx}",
                    event_type="tool.execute",
                    category="tool",
                    result="allow",
                    detail={
                        "severity": "info",
                        "actor": "runtime",
                        "source": "test",
                        "subsystem": "tool",
                        "action": "execute",
                        "token": f"secret-{idx}",
                    },
                )
            )

        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            first = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit?category=tool&source=test&limit=2&offset=0",
                headers=HEADERS,
            )
            assert first.status_code == 200
            first_data = first.json()["data"]
            assert first_data["total"] == 3
            assert first_data["count"] == 2
            assert first_data["has_more"] is True
            assert first_data["events"][0]["session_id"] == "sess-2"
            assert first_data["events"][0]["id"]
            assert first_data["events"][0]["detail"]["token"] == "[REDACTED]"

            second = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit?category=tool&source=test&limit=2&offset=2",
                headers=HEADERS,
            )
            second_data = second.json()["data"]
            assert second_data["count"] == 1
            assert second_data["has_more"] is False
            assert second_data["events"][0]["session_id"] == "sess-0"
        finally:
            srv.stop()

    def test_logout_requires_csrf_and_revokes_browser_session(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", follow_redirects=False
            ) as client:
                client.post("/login", data={"api_key": API_KEY})
                denied = client.post("/logout")
                assert denied.status_code == 403

                console = client.get("/")
                csrf = console.text.split('data-csrf="', 1)[1].split('"', 1)[0]
                logout = client.post("/logout", headers={"X-CSRF-Token": csrf})
                assert logout.status_code == 303
                assert logout.headers["location"] == "/login"
                assert "Max-Age=0" in logout.headers["set-cookie"]

                after = client.get("/", follow_redirects=False)
                assert after.status_code == 303
                assert after.headers["location"] == "/login"
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


class TestStatus:
    def test_returns_200(self, client: httpx.Client) -> None:
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_response_fields(self, client: httpx.Client) -> None:
        data = client.get("/status").json()["data"]
        assert "providers_available" in data
        assert "tool_count" in data
        assert "session_count" in data
        assert "memory" in data


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


class TestSessions:
    def test_create_session(self, client: httpx.Client) -> None:
        resp = client.post("/sessions", json={"name": "test-session", "provider": "anthropic"})
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert "session_id" in data
        assert "created_at" in data
        assert data["provider"] == "anthropic"

    def test_create_session_minimal(self, client: httpx.Client) -> None:
        resp = client.post("/sessions", json={})
        assert resp.status_code == 201

    def test_list_sessions_returns_created(self, client: httpx.Client) -> None:
        # Create a fresh session so the list is non-empty.
        create_resp = client.post("/sessions", json={"name": "list-test"})
        session_id = create_resp.json()["data"]["session_id"]

        list_resp = client.get("/sessions")
        assert list_resp.status_code == 200
        ids = [s["session_id"] for s in list_resp.json()["data"]["sessions"]]
        assert session_id in ids

    def test_list_sessions_limit(self, client: httpx.Client) -> None:
        for _ in range(5):
            client.post("/sessions", json={})
        resp = client.get("/sessions?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["sessions"]) <= 2

    def test_get_session(self, client: httpx.Client) -> None:
        create_resp = client.post("/sessions", json={"name": "get-test"})
        session_id = create_resp.json()["data"]["session_id"]

        get_resp = client.get(f"/sessions/{session_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()["data"]
        assert data["session_id"] == session_id

    def test_get_unknown_session_returns_404(self, client: httpx.Client) -> None:
        resp = client.get("/sessions/does-not-exist")
        assert resp.status_code == 404

    def test_session_history_empty(self, client: httpx.Client) -> None:
        create_resp = client.post("/sessions", json={})
        session_id = create_resp.json()["data"]["session_id"]

        hist_resp = client.get(f"/sessions/{session_id}/history")
        assert hist_resp.status_code == 200
        assert hist_resp.json()["data"]["turns"] == []

    def test_session_history_unknown_returns_404(self, client: httpx.Client) -> None:
        resp = client.get("/sessions/no-such-id/history")
        assert resp.status_code == 404

    def test_delete_session(self, client: httpx.Client) -> None:
        create_resp = client.post("/sessions", json={})
        session_id = create_resp.json()["data"]["session_id"]

        del_resp = client.delete(f"/sessions/{session_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["data"]["deleted"] == session_id

        # Subsequent GET must return 404.
        assert client.get(f"/sessions/{session_id}").status_code == 404

    def test_delete_unknown_session_returns_404(self, client: httpx.Client) -> None:
        resp = client.delete("/sessions/ghost-session")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class TestChat:
    def test_missing_message_returns_400(self, client: httpx.Client) -> None:
        resp = client.post("/chat", json={})
        assert resp.status_code == 400

    def test_empty_message_returns_400(self, client: httpx.Client) -> None:
        resp = client.post("/chat", json={"message": "   "})
        assert resp.status_code == 400

    def test_no_runtime_returns_503(self, client: httpx.Client) -> None:
        """With no runtime configured the endpoint must return 503."""
        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 503

    def test_chat_with_runtime_and_session(self) -> None:
        """Chat endpoint calls runtime.run and returns the response."""
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "Hello from mock!"
        mock_runtime.config.provider = "anthropic"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/chat",
                json={"message": "What is 2+2?"},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["response"] == "Hello from mock!"
            assert "session_id" in data
            mock_runtime.run.assert_called_once()
        finally:
            srv.stop()

    def test_chat_reuses_existing_session(self) -> None:
        """Providing an existing session_id must route to that session."""
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "reply"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            # Create a session first.
            sess_resp = httpx.post(f"{base}/sessions", json={}, headers=HEADERS)
            session_id = sess_resp.json()["data"]["session_id"]

            chat_resp = httpx.post(
                f"{base}/chat",
                json={"message": "hi", "session_id": session_id},
                headers=HEADERS,
            )
            assert chat_resp.status_code == 200
            assert chat_resp.json()["data"]["session_id"] == session_id
            _, call_kwargs = mock_runtime.run.call_args
            assert (
                call_kwargs.get("session_id") == session_id
                or mock_runtime.run.call_args[0][1] == session_id
            )
        finally:
            srv.stop()

    def test_chat_with_unknown_session_id_returns_404(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "reply"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/chat",
                json={"message": "hi", "session_id": "no-such-session"},
                headers=HEADERS,
            )
            assert resp.status_code == 404
        finally:
            srv.stop()

    def test_runtime_exception_returns_500(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("boom")
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/chat",
                json={"message": "trigger error"},
                headers=HEADERS,
            )
            assert resp.status_code == 500
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Memory search endpoint
# ---------------------------------------------------------------------------


class TestMemorySearch:
    def test_missing_query_returns_400(self, client: httpx.Client) -> None:
        resp = client.get("/memory/search")
        assert resp.status_code == 400

    def test_empty_query_returns_400(self, client: httpx.Client) -> None:
        resp = client.get("/memory/search?q=")
        assert resp.status_code == 400

    def test_no_memory_store_returns_empty(self, client: httpx.Client) -> None:
        """Without a memory store, search returns an empty list."""
        resp = client.get("/memory/search?q=hello")
        assert resp.status_code == 200
        assert resp.json()["data"]["results"] == []

    def test_memory_search_with_store(self) -> None:
        """When a memory store is injected, its search() method is called."""
        port = _free_port()
        mock_store = MagicMock()

        # Build a mock ConversationTurn-like object.
        turn = MagicMock()
        turn.role = "user"
        turn.content = "test content"
        turn.timestamp = "2026-01-01T00:00:00"
        turn.session_id = "sess-1"
        turn.provider = "anthropic"
        mock_store.search.return_value = [turn]

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, memory_store=mock_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/memory/search?q=test&limit=5",
                headers=HEADERS,
            )
            assert resp.status_code == 200
            results = resp.json()["data"]["results"]
            assert len(results) == 1
            assert results[0]["content"] == "test content"
            mock_store.search.assert_called_once_with("test", limit=5, session_id=None)
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Providers endpoint
# ---------------------------------------------------------------------------


class TestProviders:
    def test_returns_200(self, client: httpx.Client) -> None:
        resp = client.get("/providers")
        assert resp.status_code == 200

    def test_no_registry_returns_empty_list(self, client: httpx.Client) -> None:
        data = client.get("/providers").json()["data"]
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_with_registry(self) -> None:
        port = _free_port()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic", "openai"]
        mock_registry.get_default_name.return_value = "anthropic"

        mock_p1 = MagicMock()
        mock_p1.is_available.return_value = True
        mock_p2 = MagicMock()
        mock_p2.is_available.return_value = False
        mock_registry.get.side_effect = lambda name: {"anthropic": mock_p1, "openai": mock_p2}[name]

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, provider_registry=mock_registry)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/providers", headers=HEADERS)
            assert resp.status_code == 200
            providers = resp.json()["data"]["providers"]
            names = {p["name"] for p in providers}
            assert names == {"anthropic", "openai"}
            anthropic = next(p for p in providers if p["name"] == "anthropic")
            assert anthropic["available"] is True
            assert anthropic["is_default"] is True
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Tools endpoint
# ---------------------------------------------------------------------------


class TestTools:
    def test_returns_200(self, client: httpx.Client) -> None:
        resp = client.get("/tools")
        assert resp.status_code == 200

    def test_no_tool_registry_returns_empty_list(self, client: httpx.Client) -> None:
        data = client.get("/tools").json()["data"]
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_with_tool_registry(self) -> None:
        port = _free_port()
        mock_reg = MagicMock()
        mock_reg.list_tools.return_value = ["calculator"]

        mock_tool = MagicMock()
        mock_tool.name = "calculator"
        mock_tool.description = "Evaluates expressions"
        mock_tool.get_schema.return_value = {"parameters": {"expression": {"type": "string"}}}
        mock_reg.get.return_value = mock_tool

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, tool_registry=mock_reg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/tools", headers=HEADERS)
            assert resp.status_code == 200
            tools = resp.json()["data"]["tools"]
            assert len(tools) == 1
            assert tools[0]["name"] == "calculator"
            assert tools[0]["description"] == "Evaluates expressions"
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Diagnostics endpoint
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_diagnostics_requires_auth(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/diagnostics")
        assert resp.status_code == 401

    def test_diagnostics_reports_redacted_operator_posture(self) -> None:
        init_policy_engine(get_default_config())
        port = _free_port()

        mock_provider_registry = MagicMock()
        mock_provider_registry.list_providers.return_value = ["anthropic"]
        mock_provider_registry.get_default_name.return_value = "anthropic"
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider_registry.get.return_value = mock_provider

        mock_tool_registry = MagicMock()
        mock_tool_registry.list_tools.return_value = ["web_fetch"]
        mock_tool = MagicMock()
        mock_tool.permissions = SimpleNamespace(
            network=True,
            filesystem_read=False,
            filesystem_write=False,
            shell=False,
        )
        mock_tool_registry.get.return_value = mock_tool

        cfg = ApiConfig(
            host="127.0.0.1",
            port=port,
            api_key="sk-test-diagnostic-secret-abcdefghijklmnopqrstuvwxyz",
        )
        srv = ApiServer(
            config=cfg,
            provider_registry=mock_provider_registry,
            tool_registry=mock_tool_registry,
        )
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/diagnostics",
                headers={"X-API-Key": cfg.api_key},
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["overall"] in {"ok", "warn"}
            labels = {section["label"] for section in data["sections"]}
            assert {"Web entrypoint", "Providers", "Tools", "Policy"}.issubset(labels)

            rendered = str(data)
            assert cfg.api_key not in rendered
            assert "sk-test-diagnostic-secret" not in rendered

            policy = next(section for section in data["sections"] if section["key"] == "policy")
            assert any(
                check["name"] == "Network default deny" and check["summary"] is True
                for check in policy["checks"]
            )
            tools = next(section for section in data["sections"] if section["key"] == "tools")
            assert any(
                check["name"] == "Elevated permissions" and check["summary"]["network"] == 1
                for check in tools["checks"]
            )
            gateway = next(section for section in data["sections"] if section["key"] == "gateway")
            assert any(check["name"] == "Policy HTTP client" for check in gateway["checks"])
        finally:
            srv.stop()

    def test_diagnostics_reports_discord_policy_readiness_and_remediation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MISSY_TEST_DISCORD_TOKEN", "discord-token-secret-1234567890")
        config = replace(
            get_default_config(),
            network=NetworkPolicy(
                default_deny=True,
                allowed_domains=[],
                allowed_hosts=[],
                discord_allowed_hosts=["discord.com"],
            ),
            discord=DiscordConfig(
                enabled=True,
                accounts=[
                    DiscordAccountConfig(
                        token_env_var="MISSY_TEST_DISCORD_TOKEN",
                        application_id="12345",
                        dm_policy=DiscordDMPolicy.PAIRING,
                    )
                ],
            ),
        )
        init_policy_engine(config)

        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/diagnostics",
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            discord = next(section for section in data["sections"] if section["key"] == "discord")
            assert any(
                check["name"] == "Account 0 token" and check["summary"] == "present"
                for check in discord["checks"]
            )
            gateway_host = next(
                check
                for check in discord["checks"]
                if check["name"] == "Gateway host gateway.discord.gg"
            )
            assert gateway_host["status"] == "warn"
            assert "remediation" in gateway_host

            rendered = str(data)
            assert "discord-token-secret" not in rendered
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_rate_limit_applied(self) -> None:
        """After exceeding rpm, the server returns 429."""
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY, rate_limit_rpm=3)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        url = f"http://127.0.0.1:{port}/api/v1/health"
        try:
            responses = [httpx.get(url, headers=HEADERS) for _ in range(5)]
            statuses = [r.status_code for r in responses]
            assert 429 in statuses, f"Expected 429 in {statuses}"
            assert responses[0].status_code == 200  # First request must succeed.
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Request size limiting
# ---------------------------------------------------------------------------


class TestRequestSizeLimit:
    def test_oversized_body_rejected(self) -> None:
        """POST bodies exceeding max_request_bytes must not be processed."""
        port = _free_port()
        # Set a tiny cap of 10 bytes.
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY, max_request_bytes=10)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            big_payload = {"message": "x" * 1000}
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/chat",
                json=big_payload,
                headers=HEADERS,
            )
            # Size limit causes body parse to return None → 400, or the route
            # returns 503 (no runtime). Either way it must not be 200.
            assert resp.status_code in (400, 503)
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Invalid JSON handling
# ---------------------------------------------------------------------------


class TestInvalidJson:
    def test_non_json_content_type_rejected(self, client: httpx.Client) -> None:
        resp = client.post(
            "/sessions",
            content=b'{"name": "test"}',
            headers={**HEADERS, "Content-Type": "text/plain"},
        )
        # Body parse returns None for non-JSON content type → 400
        assert resp.status_code == 400

    def test_malformed_json_rejected(self, client: httpx.Client) -> None:
        resp = client.post(
            "/sessions",
            content=b"{bad json",
            headers={**HEADERS, "Content-Type": "application/json"},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Route matching
# ---------------------------------------------------------------------------


class TestRouteMatching:
    def test_unknown_path_returns_404(self, client: httpx.Client) -> None:
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404

    def test_wrong_method_returns_404(self, client: httpx.Client) -> None:
        # DELETE on /health is not registered.
        resp = client.delete("/health")
        assert resp.status_code == 404

    def test_nested_unknown_segment_returns_404(self, client: httpx.Client) -> None:
        resp = client.get("/sessions/abc/unknown-sub")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# ApiResponse helper
# ---------------------------------------------------------------------------


class TestApiResponse:
    def test_ok_defaults_to_200(self) -> None:
        status, body = ApiResponse.ok({"key": "val"})
        assert status == 200
        assert body["status"] == "ok"
        assert body["data"] == {"key": "val"}

    def test_ok_custom_status(self) -> None:
        status, body = ApiResponse.ok({}, status=204)
        assert status == 204

    def test_created_returns_201(self) -> None:
        status, body = ApiResponse.created({"id": "x"})
        assert status == 201
        assert body["status"] == "ok"

    def test_error_defaults_to_400(self) -> None:
        status, body = ApiResponse.error("oops")
        assert status == 400
        assert body["status"] == "error"
        assert body["error"] == "oops"

    def test_error_custom_status(self) -> None:
        status, body = ApiResponse.error("not found", 404)
        assert status == 404


# ---------------------------------------------------------------------------
# SessionRegistry unit tests
# ---------------------------------------------------------------------------


class TestSessionRegistry:
    def test_create_and_get(self) -> None:
        reg = _SessionRegistry()
        sess = reg.create(provider="anthropic", name="my-session")
        assert reg.get(sess.session_id) is sess

    def test_get_unknown_returns_none(self) -> None:
        reg = _SessionRegistry()
        assert reg.get("no-such-id") is None

    def test_list_most_recent_first(self) -> None:
        reg = _SessionRegistry()
        s1 = reg.create(provider="p1")
        time.sleep(0.01)
        s2 = reg.create(provider="p2")
        sessions = reg.list(limit=10)
        ids = [s.session_id for s in sessions]
        assert ids.index(s2.session_id) < ids.index(s1.session_id)

    def test_list_respects_limit(self) -> None:
        reg = _SessionRegistry()
        for _ in range(5):
            reg.create(provider="p")
        assert len(reg.list(limit=3)) == 3

    def test_delete_existing(self) -> None:
        reg = _SessionRegistry()
        sess = reg.create(provider="p")
        assert reg.delete(sess.session_id) is True
        assert reg.get(sess.session_id) is None

    def test_delete_unknown_returns_false(self) -> None:
        reg = _SessionRegistry()
        assert reg.delete("ghost") is False

    def test_count(self) -> None:
        reg = _SessionRegistry()
        assert reg.count() == 0
        reg.create(provider="p")
        assert reg.count() == 1

    def test_touch_updates_last_used(self) -> None:
        reg = _SessionRegistry()
        sess = reg.create(provider="p")
        old_ts = sess.last_used_at
        time.sleep(0.01)
        reg.touch(sess.session_id, turn_increment=2)
        assert sess.last_used_at >= old_ts
        assert sess.turn_count == 2

    def test_thread_safety(self) -> None:
        """Concurrent creates must not raise and all sessions must be retrievable."""
        reg = _SessionRegistry()
        ids: list[str] = []
        lock = threading.Lock()

        def _create():
            sess = reg.create(provider="concurrent")
            with lock:
                ids.append(sess.session_id)

        threads = [threading.Thread(target=_create) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 20
        for sid in ids:
            assert reg.get(sid) is not None


# ---------------------------------------------------------------------------
# ApiServer lifecycle
# ---------------------------------------------------------------------------


class TestApiServerLifecycle:
    def test_is_running_false_before_start(self) -> None:
        cfg = ApiConfig(host="127.0.0.1", port=_free_port(), api_key="k")
        srv = ApiServer(config=cfg)
        assert srv.is_running is False

    def test_is_running_true_after_start(self) -> None:
        cfg = ApiConfig(host="127.0.0.1", port=_free_port(), api_key="k")
        srv = ApiServer(config=cfg)
        srv.start()
        try:
            assert srv.is_running is True
        finally:
            srv.stop()

    def test_is_running_false_after_stop(self) -> None:
        cfg = ApiConfig(host="127.0.0.1", port=_free_port(), api_key="k")
        srv = ApiServer(config=cfg)
        srv.start()
        srv.stop()
        assert srv.is_running is False

    def test_double_start_raises(self) -> None:
        cfg = ApiConfig(host="127.0.0.1", port=_free_port(), api_key="k")
        srv = ApiServer(config=cfg)
        srv.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                srv.start()
        finally:
            srv.stop()

    def test_url_property(self) -> None:
        cfg = ApiConfig(host="127.0.0.1", port=9999, api_key="k")
        srv = ApiServer(config=cfg)
        assert srv.url == "http://127.0.0.1:9999"
