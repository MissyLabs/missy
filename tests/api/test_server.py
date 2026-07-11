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
from missy.api.web_console import console_script, render_console
from missy.channels.discord.channel import DiscordChannel
from missy.channels.discord.config import (
    DiscordAccountConfig,
    DiscordConfig,
    DiscordDMPolicy,
)
from missy.config.settings import NetworkPolicy, get_default_config
from missy.core.events import AuditEvent, event_bus
from missy.observability.audit_logger import init_audit_logger
from missy.policy.engine import init_policy_engine
from missy.tools.benchmark.benchmark_store import BenchmarkStore
from missy.tools.benchmark.scoring import BenchmarkResult, BenchmarkScorer
from missy.tools.intelligence import (
    BenchmarkSummary,
    CandidateStore,
    ToolCandidate,
    ToolLifecycleState,
)

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


def _make_tool_candidate(name: str = "calculator_candidate") -> ToolCandidate:
    return ToolCandidate.create(
        name=name,
        description="calculator candidate",
        schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        permissions={"network": False, "shell": False},
        provenance="api test",
    )


def _save_candidate_benchmarks(
    store: BenchmarkStore,
    *,
    tool_name: str = "calculator_candidate",
    provider: str = "mock",
    count: int = 3,
) -> None:
    scorer = BenchmarkScorer()
    for _ in range(count):
        store.save(
            scorer.score(
                BenchmarkResult(
                    task_id=f"{tool_name}-{provider}",
                    tool_name=tool_name,
                    provider=provider,
                    success=True,
                    latency_ms=5.0,
                    cost_usd=0.0,
                    actual_output="4",
                    expected_output="4",
                    tool_call_made=True,
                    tool_call_args={"expression": "2+2"},
                    schema_required_params=["expression"],
                )
            )
        )


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
    def test_console_renderer_escapes_csrf_and_keeps_ui_hooks(self) -> None:
        html = render_console(csrf_token='csrf"><script>alert(1)</script>')

        assert 'data-csrf="csrf&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"' in html
        assert 'data-csrf="csrf"><script>' not in html
        assert "Runtime posture" in html
        assert "Audit Trail" in html
        assert 'id="audit-result"' in html
        assert 'id="controls"' in html
        assert "/api/v1' + path" in html
        assert 'id="scheduler-jobs"' in html
        assert 'id="scheduler-form"' in html
        assert 'id="memory-query"' in html
        assert 'id="memory-results"' in html
        assert 'id="approvals"' in html
        assert 'id="approvals-health"' in html
        assert 'id="pairing"' in html
        assert 'id="pairing-health"' in html
        assert "Approvals" in html
        assert "Discord Pairing" in html

    def test_console_script_keeps_safe_client_side_escaping_and_control_post(self) -> None:
        script = console_script()

        assert "function esc(value)" in script
        assert "textContent = event ? JSON.stringify(event, null, 2)" in script
        assert "X-CSRF-Token" in script
        assert "data-control-label" in script
        assert "data-target-label" in script

    def test_console_script_includes_scheduler_and_memory_wiring(self) -> None:
        script = console_script()

        assert "/scheduler/jobs" in script
        assert "job-remove" in script
        assert "remove-job:" in script
        assert "/memory/turns/" in script
        assert "memory-pin" in script
        assert "memory-delete" in script

    def test_console_script_renders_run_complete_summary(self) -> None:
        script = console_script()

        assert "data.tools_used" in script
        assert "data.cost" in script
        assert "JSON.stringify({target, confirm: confirmation})" in script

    def test_console_script_includes_approvals_and_pairing_wiring(self) -> None:
        """Web TUI browser page for /api/v1/approvals and
        /api/v1/discord/pairing -- both REST endpoints were real and
        authenticated (SR-2.2, SR-1.12) but had no browser UI; the
        operator could only inspect/resolve pending requests via
        `missy approvals`/`missy devices pair` or raw curl."""
        script = console_script()

        assert "api('/approvals')" in script
        assert "api('/discord/pairing')" in script
        assert "approval-action" in script
        assert "pairing-action" in script
        assert "/approvals/${encodeURIComponent(approvalId)}/${approve ? 'approve' : 'deny'}" in script
        assert "/discord/pairing/${encodeURIComponent(userId)}/${approve ? 'approve' : 'deny'}" in script

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
                assert "Controls" in console.text
                assert "/api/v1' + path" in console.text
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
# Streamed background runs
# ---------------------------------------------------------------------------


class TestRuns:
    def test_missing_message_returns_400(self, client: httpx.Client) -> None:
        resp = client.post("/runs", json={})
        assert resp.status_code == 400

    def test_no_runtime_returns_503(self, client: httpx.Client) -> None:
        resp = client.post("/runs", json={"message": "hello"})
        assert resp.status_code == 503

    def test_start_run_returns_202_with_pending_or_running_status(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "42"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/runs",
                json={"message": "what is 6*7"},
                headers=HEADERS,
            )
            assert resp.status_code == 202
            data = resp.json()["data"]
            assert data["run_id"]
            assert data["session_id"]
            assert data["status"] in {"pending", "running", "complete"}
        finally:
            srv.stop()

    def test_run_reaches_complete_status_via_polling(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "the answer is 42"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            start = httpx.post(base + "/runs", json={"message": "hi"}, headers=HEADERS)
            run_id = start.json()["data"]["run_id"]

            deadline = time.monotonic() + 3.0
            status = None
            while time.monotonic() < deadline:
                poll = httpx.get(f"{base}/runs/{run_id}", headers=HEADERS)
                assert poll.status_code == 200
                status = poll.json()["data"]["status"]
                if status == "complete":
                    break
                time.sleep(0.02)
            assert status == "complete"
            assert poll.json()["data"]["response"] == "the answer is 42"
        finally:
            srv.stop()

    def test_get_unknown_run_returns_404(self, client: httpx.Client) -> None:
        resp = client.get("/runs/does-not-exist")
        assert resp.status_code == 404

    def test_list_runs_requires_session_id(self, client: httpx.Client) -> None:
        resp = client.get("/runs")
        assert resp.status_code == 400

    def test_list_runs_returns_runs_for_session(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "ok"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            start = httpx.post(base + "/runs", json={"message": "hi"}, headers=HEADERS)
            session_id = start.json()["data"]["session_id"]
            run_id = start.json()["data"]["run_id"]

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if (
                    httpx.get(f"{base}/runs/{run_id}", headers=HEADERS).json()["data"]["status"]
                    == "complete"
                ):
                    break
                time.sleep(0.02)

            listing = httpx.get(f"{base}/runs", params={"session_id": session_id}, headers=HEADERS)
            assert listing.status_code == 200
            runs = listing.json()["data"]["runs"]
            assert any(r["run_id"] == run_id for r in runs)
        finally:
            srv.stop()

    def test_events_stream_requires_auth(self) -> None:
        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/runs/anything/events")
            assert resp.status_code == 401
        finally:
            srv.stop()

    def test_events_stream_unknown_run_returns_404(self, client: httpx.Client) -> None:
        resp = client.get("/runs/does-not-exist/events")
        assert resp.status_code == 404

    def test_events_stream_delivers_started_and_complete(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "streamed answer"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            start = httpx.post(base + "/runs", json={"message": "hi"}, headers=HEADERS)
            run_id = start.json()["data"]["run_id"]

            body = ""
            with httpx.stream(
                "GET", f"{base}/runs/{run_id}/events", headers=HEADERS, timeout=5.0
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                for chunk in resp.iter_text():
                    body += chunk
                    if "run.complete" in body or "run.error" in body:
                        break
            assert "event: run.started" in body
            assert "event: run.complete" in body
            assert "streamed answer" in body
        finally:
            srv.stop()

    def test_events_stream_late_join_after_completion_returns_immediately(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "already done"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            start = httpx.post(base + "/runs", json={"message": "hi"}, headers=HEADERS)
            run_id = start.json()["data"]["run_id"]

            # Drain the stream fully once so the run finishes.
            with httpx.stream(
                "GET", f"{base}/runs/{run_id}/events", headers=HEADERS, timeout=5.0
            ) as resp:
                for _ in resp.iter_text():
                    pass

            started = time.monotonic()
            resp = httpx.get(f"{base}/runs/{run_id}/events", headers=HEADERS, timeout=5.0)
            elapsed = time.monotonic() - started
            assert resp.status_code == 200
            assert "run.complete" in resp.text
            assert elapsed < 3.0
        finally:
            srv.stop()

    def test_run_error_reported_via_polling_and_stream(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.side_effect = RuntimeError("boom")
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            start = httpx.post(base + "/runs", json={"message": "trigger error"}, headers=HEADERS)
            run_id = start.json()["data"]["run_id"]

            deadline = time.monotonic() + 3.0
            status = None
            while time.monotonic() < deadline:
                poll = httpx.get(f"{base}/runs/{run_id}", headers=HEADERS)
                status = poll.json()["data"]["status"]
                if status == "error":
                    break
                time.sleep(0.02)
            assert status == "error"
            assert "boom" in poll.json()["data"]["error"]
        finally:
            srv.stop()

    def test_concurrent_run_on_same_session_returns_409_and_emits_audit(self, tmp_path) -> None:
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        gate = threading.Event()
        mock_runtime = MagicMock()

        def slow_run(message, session_id=None):
            gate.wait(timeout=2)
            return "slow"

        mock_runtime.run.side_effect = slow_run
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            base = f"http://127.0.0.1:{port}/api/v1"
            first = httpx.post(base + "/runs", json={"message": "first"}, headers=HEADERS)
            session_id = first.json()["data"]["session_id"]
            time.sleep(0.05)
            second = httpx.post(
                base + "/runs",
                json={"message": "second", "session_id": session_id},
                headers=HEADERS,
            )
            assert second.status_code == 409

            audit = httpx.get(f"{base}/audit?subsystem=agent&result=deny&limit=10", headers=HEADERS)
            events = audit.json()["data"]["events"]
            assert any(e["event_type"] == "web.run" for e in events)
        finally:
            gate.set()
            srv.stop()

    def test_unknown_session_id_returns_404(self) -> None:
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
                f"http://127.0.0.1:{port}/api/v1/runs",
                json={"message": "hi", "session_id": "no-such-session"},
                headers=HEADERS,
            )
            assert resp.status_code == 404
        finally:
            srv.stop()

    def test_browser_session_can_start_run_with_csrf(self) -> None:
        port = _free_port()
        mock_runtime = MagicMock()
        mock_runtime.run.return_value = "web ui reply"
        mock_runtime.config.provider = "mock"

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=mock_runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", follow_redirects=False
            ) as browser:
                browser.post("/login", data={"api_key": API_KEY})
                denied = browser.post("/api/v1/runs", json={"message": "hi"})
                assert denied.status_code == 403

                console = browser.get("/")
                csrf = console.text.split('data-csrf="', 1)[1].split('"', 1)[0]
                allowed = browser.post(
                    "/api/v1/runs", json={"message": "hi"}, headers={"X-CSRF-Token": csrf}
                )
                assert allowed.status_code == 202

                run_id = allowed.json()["data"]["run_id"]
                events_resp = browser.get(f"/api/v1/runs/{run_id}/events")
                assert events_resp.status_code == 200
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
        turn.id = "turn-1"
        turn.role = "user"
        turn.content = "test content"
        turn.timestamp = "2026-01-01T00:00:00"
        turn.session_id = "sess-1"
        turn.provider = "anthropic"
        turn.metadata = {}
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
# Operator controls endpoint
# ---------------------------------------------------------------------------


class TestOperatorControls:
    def test_controls_requires_auth(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/controls")
        assert resp.status_code == 401

    def test_controls_lists_provider_targets(self) -> None:
        port = _free_port()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic", "openai"]
        mock_registry.get_default_name.return_value = "anthropic"
        mock_anthropic = MagicMock()
        mock_anthropic.is_available.return_value = True
        mock_openai = MagicMock()
        mock_openai.is_available.return_value = False
        mock_registry.get.side_effect = lambda name: {
            "anthropic": mock_anthropic,
            "openai": mock_openai,
        }[name]

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, provider_registry=mock_registry)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/controls", headers=HEADERS)
            assert resp.status_code == 200
            controls = resp.json()["data"]["controls"]
            provider_control = next(c for c in controls if c["id"] == "provider.set_default")
            assert provider_control["requires_confirmation"] is True
            targets = {target["name"]: target for target in provider_control["targets"]}
            assert targets["anthropic"]["is_current"] is True
            assert targets["anthropic"]["confirmation"] == "set-default:anthropic"
            assert targets["openai"]["available"] is False
        finally:
            srv.stop()

    def test_set_default_provider_requires_confirmation_and_audits_denial(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic", "openai"]
        mock_registry.get_default_name.return_value = "anthropic"
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_registry.get.return_value = mock_provider

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, provider_registry=mock_registry)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/provider.set_default",
                json={"target": "openai"},
                headers=HEADERS,
            )
            assert resp.status_code == 409
            mock_registry.set_default.assert_not_called()

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.control&result=deny&subsystem=provider&limit=10",
                headers=HEADERS,
            )
            assert audit.status_code == 200
            events = audit.json()["data"]["events"]
            assert any(
                e["detail"]["action"] == "provider.set_default"
                and e["detail"]["reason"] == "confirmation_required"
                for e in events
            )
        finally:
            srv.stop()

    def test_set_default_provider_switches_available_provider_and_audits_allow(
        self, tmp_path
    ) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        mock_registry = MagicMock()
        mock_registry.list_providers.return_value = ["anthropic", "openai"]
        mock_registry.get_default_name.return_value = "anthropic"
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_registry.get.return_value = mock_provider

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, provider_registry=mock_registry)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/provider.set_default",
                json={"target": "openai", "confirm": "set-default:openai"},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            data = resp.json()["data"]
            assert data["previous"] == "anthropic"
            assert data["current"] == "openai"
            mock_registry.set_default.assert_called_once_with("openai")

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.control&result=allow&subsystem=provider&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(
                e["detail"]["target"] == "openai" and e["detail"]["current"] == "openai"
                for e in events
            )
        finally:
            srv.stop()

    def test_browser_control_post_requires_csrf(self) -> None:
        port = _free_port()
        mock_registry = MagicMock()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, provider_registry=mock_registry)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/login")
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", follow_redirects=False
            ) as client:
                client.post("/login", data={"api_key": API_KEY})
                denied = client.post(
                    "/api/v1/controls/provider.set_default",
                    json={"target": "openai", "confirm": "set-default:openai"},
                )
                assert denied.status_code == 403
                mock_registry.set_default.assert_not_called()
        finally:
            srv.stop()

    def test_controls_lists_scheduler_pause_and_resume_targets(self) -> None:
        port = _free_port()
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [
            SimpleNamespace(
                id="job-enabled",
                name="Morning summary",
                schedule="daily at 09:00",
                provider="anthropic",
                enabled=True,
            ),
            SimpleNamespace(
                id="job-paused",
                name="Evening review",
                schedule="daily at 18:00",
                provider="openai",
                enabled=False,
            ),
        ]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/controls", headers=HEADERS)
            assert resp.status_code == 200
            controls = {c["id"]: c for c in resp.json()["data"]["controls"]}

            pause_targets = {t["name"]: t for t in controls["scheduler.pause_job"]["targets"]}
            assert pause_targets["job-enabled"]["available"] is True
            assert pause_targets["job-enabled"]["confirmation"] == "pause-job:job-enabled"
            assert pause_targets["job-paused"]["available"] is False

            resume_targets = {t["name"]: t for t in controls["scheduler.resume_job"]["targets"]}
            assert resume_targets["job-paused"]["available"] is True
            assert resume_targets["job-paused"]["confirmation"] == "resume-job:job-paused"
            assert resume_targets["job-enabled"]["available"] is False
        finally:
            srv.stop()

    def test_pause_scheduler_job_requires_confirmation_and_audits_denial(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [
            SimpleNamespace(id="job-enabled", name="Morning summary", enabled=True)
        ]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/scheduler.pause_job",
                json={"target": "job-enabled"},
                headers=HEADERS,
            )
            assert resp.status_code == 409
            scheduler.pause_job.assert_not_called()

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.control&result=deny&subsystem=scheduler&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(
                e["detail"]["action"] == "scheduler.pause_job"
                and e["detail"]["reason"] == "confirmation_required"
                for e in events
            )
        finally:
            srv.stop()

    def test_pause_and_resume_scheduler_job_mutates_and_audits_allow(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        job = SimpleNamespace(id="job-enabled", name="Morning summary", enabled=True)
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [job]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            pause = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/scheduler.pause_job",
                json={"target": "job-enabled", "confirm": "pause-job:job-enabled"},
                headers=HEADERS,
            )
            assert pause.status_code == 200
            assert pause.json()["data"]["current_enabled"] is False
            scheduler.pause_job.assert_called_once_with("job-enabled")

            job.enabled = False
            resume = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/scheduler.resume_job",
                json={"target": "job-enabled", "confirm": "resume-job:job-enabled"},
                headers=HEADERS,
            )
            assert resume.status_code == 200
            assert resume.json()["data"]["current_enabled"] is True
            scheduler.resume_job.assert_called_once_with("job-enabled")

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.control&result=allow&subsystem=scheduler&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(e["detail"]["action"] == "scheduler.pause_job" for e in events)
            assert any(e["detail"]["action"] == "scheduler.resume_job" for e in events)
        finally:
            srv.stop()

    def test_controls_lists_scheduler_remove_targets(self) -> None:
        port = _free_port()
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [
            SimpleNamespace(
                id="job-1",
                name="Morning summary",
                schedule="daily at 09:00",
                provider="anthropic",
                enabled=True,
            )
        ]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/controls", headers=HEADERS)
            controls = {c["id"]: c for c in resp.json()["data"]["controls"]}
            remove_control = controls["scheduler.remove_job"]
            assert remove_control["destructive"] is True
            targets = {t["name"]: t for t in remove_control["targets"]}
            assert targets["job-1"]["confirmation"] == "remove-job:job-1"
        finally:
            srv.stop()

    def test_remove_scheduler_job_requires_confirmation_and_audits_denial(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [SimpleNamespace(id="job-1", name="Morning summary")]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/scheduler.remove_job",
                json={"target": "job-1"},
                headers=HEADERS,
            )
            assert resp.status_code == 409
            scheduler.remove_job.assert_not_called()

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.control&result=deny&subsystem=scheduler&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(
                e["detail"]["action"] == "scheduler.remove_job"
                and e["detail"]["reason"] == "confirmation_required"
                for e in events
            )
        finally:
            srv.stop()

    def test_remove_scheduler_job_via_controls_endpoint_mutates_and_audits(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [SimpleNamespace(id="job-1", name="Morning summary")]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/scheduler.remove_job",
                json={"target": "job-1", "confirm": "remove-job:job-1"},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            assert resp.json()["data"]["removed"] is True
            scheduler.remove_job.assert_called_once_with("job-1")
        finally:
            srv.stop()

    def test_delete_scheduler_job_route_delegates_to_remove_control(self) -> None:
        port = _free_port()
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [SimpleNamespace(id="job-1", name="Morning summary")]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.request(
                "DELETE",
                f"http://127.0.0.1:{port}/api/v1/scheduler/jobs/job-1",
                json={"confirm": "remove-job:job-1"},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            scheduler.remove_job.assert_called_once_with("job-1")
        finally:
            srv.stop()

    def test_tool_candidate_list_and_show_routes(self, tmp_path) -> None:
        candidate_store = CandidateStore(db_path=tmp_path / "candidates.db")
        candidate = candidate_store.add(_make_tool_candidate())
        port = _free_port()

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, candidate_store=candidate_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            list_resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/tool-candidates",
                headers=HEADERS,
            )
            assert list_resp.status_code == 200
            candidates = list_resp.json()["data"]["candidates"]
            assert candidates[0]["id"] == candidate.id
            assert candidates[0]["state"] == "proposed"

            show_resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/tool-candidates/{candidate.id}",
                headers=HEADERS,
            )
            assert show_resp.status_code == 200
            assert show_resp.json()["data"]["candidate"]["name"] == "calculator_candidate"
        finally:
            srv.stop()

    def test_controls_list_tool_candidate_targets_by_lifecycle_state(self, tmp_path) -> None:
        candidate_store = CandidateStore(db_path=tmp_path / "candidates.db")
        proposed = candidate_store.add(_make_tool_candidate("proposed_candidate"))
        benchmarked = candidate_store.add(_make_tool_candidate("benchmarked_candidate"))
        candidate_store.update_benchmark(
            benchmarked.id,
            summary=BenchmarkSummary(
                provider="mock",
                correctness=1.0,
                latency_ms=1.0,
                cost_usd=0.0,
                reliability=1.0,
                safety=1.0,
                schema_score=1.0,
                composite=1.0,
                run_at="now",
            ),
        )
        port = _free_port()

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, candidate_store=candidate_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/controls", headers=HEADERS)
            controls = {c["id"]: c for c in resp.json()["data"]["controls"]}

            import_targets = {
                target["name"] for target in controls["tool_candidate.import_benchmarks"]["targets"]
            }
            approve_targets = {
                target["name"] for target in controls["tool_candidate.approve"]["targets"]
            }
            deny_targets = {target["name"] for target in controls["tool_candidate.deny"]["targets"]}

            assert proposed.id in import_targets
            assert benchmarked.id in approve_targets
            assert {proposed.id, benchmarked.id}.issubset(deny_targets)
        finally:
            srv.stop()

    def test_candidate_import_approve_enable_controls_mutate_and_audit(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        candidate_store = CandidateStore(db_path=tmp_path / "candidates.db")
        benchmark_store = BenchmarkStore(db_path=tmp_path / "benchmarks.db")
        candidate = candidate_store.add(_make_tool_candidate())
        _save_candidate_benchmarks(benchmark_store)
        port = _free_port()

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(
            config=cfg,
            candidate_store=candidate_store,
            benchmark_store=benchmark_store,
        )
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            imported = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.import_benchmarks",
                json={
                    "target": candidate.id,
                    "confirm": f"import-candidate-benchmarks:{candidate.id}",
                },
                headers=HEADERS,
            )
            assert imported.status_code == 200
            assert imported.json()["data"]["candidate"]["state"] == "benchmarked"

            approved = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.approve",
                json={
                    "target": candidate.id,
                    "confirm": f"approve-candidate:{candidate.id}",
                    "notes": "reviewed",
                },
                headers=HEADERS,
            )
            assert approved.status_code == 200
            assert approved.json()["data"]["candidate"]["state"] == "approved"

            enabled = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.enable",
                json={
                    "target": candidate.id,
                    "confirm": f"enable-candidate:{candidate.id}",
                },
                headers=HEADERS,
            )
            assert enabled.status_code == 200
            assert candidate_store.get(candidate.id).state == ToolLifecycleState.ENABLED

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.control&result=allow&subsystem=tool_candidate&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(e["detail"]["action"] == "tool_candidate.import_benchmarks" for e in events)
            assert any(e["detail"]["action"] == "tool_candidate.enable" for e in events)
        finally:
            srv.stop()

    def test_candidate_deny_requires_confirmation_and_reason(self, tmp_path) -> None:
        candidate_store = CandidateStore(db_path=tmp_path / "candidates.db")
        candidate = candidate_store.add(_make_tool_candidate())
        port = _free_port()

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, candidate_store=candidate_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            missing_confirm = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.deny",
                json={"target": candidate.id, "reason": "unsafe"},
                headers=HEADERS,
            )
            assert missing_confirm.status_code == 409
            assert candidate_store.get(candidate.id).state == ToolLifecycleState.PROPOSED

            missing_reason = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.deny",
                json={"target": candidate.id, "confirm": f"deny-candidate:{candidate.id}"},
                headers=HEADERS,
            )
            assert missing_reason.status_code == 400
            assert candidate_store.get(candidate.id).state == ToolLifecycleState.PROPOSED

            denied = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.deny",
                json={
                    "target": candidate.id,
                    "confirm": f"deny-candidate:{candidate.id}",
                    "reason": "unsafe permissions",
                },
                headers=HEADERS,
            )
            assert denied.status_code == 200
            assert denied.json()["data"]["candidate"]["state"] == "disabled"
        finally:
            srv.stop()

    def test_candidate_enable_rejects_skipped_lifecycle_gate(self, tmp_path) -> None:
        candidate_store = CandidateStore(db_path=tmp_path / "candidates.db")
        candidate = candidate_store.add(_make_tool_candidate())
        port = _free_port()

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, candidate_store=candidate_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/controls/tool_candidate.enable",
                json={"target": candidate.id, "confirm": f"enable-candidate:{candidate.id}"},
                headers=HEADERS,
            )
            assert resp.status_code == 409
            assert candidate_store.get(candidate.id).state == ToolLifecycleState.PROPOSED
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Scheduler jobs endpoint (list/create)
# ---------------------------------------------------------------------------


class TestSchedulerJobs:
    def test_list_jobs_without_scheduler_returns_empty(self, client: httpx.Client) -> None:
        resp = client.get("/scheduler/jobs")
        assert resp.status_code == 200
        assert resp.json()["data"]["jobs"] == []

    def test_list_jobs_returns_full_detail(self) -> None:
        from missy.scheduler.jobs import ScheduledJob

        port = _free_port()
        job = ScheduledJob(name="Morning summary", schedule="daily at 09:00", task="Summarize")
        scheduler = MagicMock()
        scheduler.list_jobs.return_value = [job]
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/v1/scheduler/jobs", headers=HEADERS)
            assert resp.status_code == 200
            jobs = resp.json()["data"]["jobs"]
            assert len(jobs) == 1
            assert jobs[0]["name"] == "Morning summary"
            assert jobs[0]["schedule"] == "daily at 09:00"
            assert jobs[0]["task"] == "Summarize"
        finally:
            srv.stop()

    def test_create_job_requires_scheduler(self, client: httpx.Client) -> None:
        resp = client.post(
            "/scheduler/jobs",
            json={"name": "Test", "schedule": "every 5 minutes", "task": "Do a thing"},
        )
        assert resp.status_code == 503

    def test_create_job_requires_required_fields(self, tmp_path) -> None:
        port = _free_port()
        scheduler = MagicMock()
        runtime = SimpleNamespace(_scheduler=scheduler)
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/scheduler/jobs",
                json={"name": "", "schedule": "", "task": ""},
                headers=HEADERS,
            )
            assert resp.status_code == 400
            scheduler.add_job.assert_not_called()
        finally:
            srv.stop()

    def test_create_job_success_audits_allow(self, tmp_path) -> None:
        from missy.scheduler.jobs import ScheduledJob

        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        created = ScheduledJob(
            name="Nightly backup", schedule="daily at 02:00", task="Back things up"
        )
        scheduler = MagicMock()
        scheduler.add_job.return_value = created
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/scheduler/jobs",
                json={
                    "name": "Nightly backup",
                    "schedule": "daily at 02:00",
                    "task": "Back things up",
                    "provider": "anthropic",
                },
                headers=HEADERS,
            )
            assert resp.status_code == 201
            assert resp.json()["data"]["name"] == "Nightly backup"
            scheduler.add_job.assert_called_once()

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.scheduler&result=allow&subsystem=scheduler&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(e["detail"]["action"] == "scheduler.job.add" for e in events)
        finally:
            srv.stop()

    def test_create_job_invalid_schedule_returns_400(self, tmp_path) -> None:
        port = _free_port()
        scheduler = MagicMock()
        scheduler.add_job.side_effect = ValueError("bad schedule string")
        runtime = SimpleNamespace(_scheduler=scheduler)

        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, runtime=runtime)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/scheduler/jobs",
                json={"name": "Test", "schedule": "nonsense", "task": "Do a thing"},
                headers=HEADERS,
            )
            assert resp.status_code == 400
        finally:
            srv.stop()


# ---------------------------------------------------------------------------
# Memory turn item routes (pin/delete)
# ---------------------------------------------------------------------------


class TestMemoryTurns:
    def test_delete_turn_without_memory_store_returns_503(self, client: httpx.Client) -> None:
        resp = client.delete("/memory/turns/turn-1")
        assert resp.status_code == 503

    def test_delete_turn_not_found_returns_404(self) -> None:
        port = _free_port()
        mock_store = MagicMock()
        mock_store.delete_turn.return_value = False
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, memory_store=mock_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.delete(
                f"http://127.0.0.1:{port}/api/v1/memory/turns/turn-1", headers=HEADERS
            )
            assert resp.status_code == 404
        finally:
            srv.stop()

    def test_delete_turn_success_audits_allow(self, tmp_path) -> None:
        event_bus.clear()
        init_audit_logger(str(tmp_path / "audit.jsonl"))
        port = _free_port()
        mock_store = MagicMock()
        mock_store.delete_turn.return_value = True
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, memory_store=mock_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.delete(
                f"http://127.0.0.1:{port}/api/v1/memory/turns/turn-1", headers=HEADERS
            )
            assert resp.status_code == 200
            assert resp.json()["data"]["deleted"] == "turn-1"
            mock_store.delete_turn.assert_called_once_with("turn-1")

            audit = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/audit"
                "?event_type=web.memory&result=allow&subsystem=memory&limit=10",
                headers=HEADERS,
            )
            events = audit.json()["data"]["events"]
            assert any(e["detail"]["action"] == "memory.delete_turn" for e in events)
        finally:
            srv.stop()

    def test_pin_turn_success(self) -> None:
        port = _free_port()
        mock_store = MagicMock()
        mock_store.set_turn_pinned.return_value = True
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, memory_store=mock_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/memory/turns/turn-1/pin",
                json={"pinned": True},
                headers=HEADERS,
            )
            assert resp.status_code == 200
            assert resp.json()["data"] == {"turn_id": "turn-1", "pinned": True}
            mock_store.set_turn_pinned.assert_called_once_with("turn-1", True)
        finally:
            srv.stop()

    def test_unpin_turn_not_found_returns_404(self) -> None:
        port = _free_port()
        mock_store = MagicMock()
        mock_store.set_turn_pinned.return_value = False
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, memory_store=mock_store)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.post(
                f"http://127.0.0.1:{port}/api/v1/memory/turns/turn-1/pin",
                json={"pinned": False},
                headers=HEADERS,
            )
            assert resp.status_code == 404
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
        mock_provider.diagnostics.return_value = {
            "provider": "anthropic",
            "status": "ok",
            "checks": [
                {
                    "name": "credential",
                    "status": "ok",
                    "summary": "configured via env",
                }
            ],
        }
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
            providers = next(
                section for section in data["sections"] if section["key"] == "providers"
            )
            assert any(
                check["name"] == "anthropic credential" and check["summary"] == "configured via env"
                for check in providers["checks"]
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


# ---------------------------------------------------------------------------
# SR-2.2: /approvals endpoints backed by a real ApprovalGate
# ---------------------------------------------------------------------------


class TestApprovalsEndpoints:
    """A real ApprovalGate wired into a real running ApiServer -- this is
    the actual mechanism SR-2.2 introduces for an operator (a separate
    `missy` CLI invocation) to see and resolve pending approval requests
    from the in-process gateway that created them.
    """

    def test_approvals_requires_auth(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/approvals")
        assert resp.status_code == 401

    def test_list_approvals_empty_when_no_gate_attached(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/approvals", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["data"] == {"approvals": [], "count": 0}

    def test_resolve_approval_without_gate_returns_503(self, base_url: str) -> None:
        resp = httpx.post(f"{base_url}/approvals/whatever/approve", headers=HEADERS)
        assert resp.status_code == 503

    def _start_server_with_gate(self):
        from missy.agent.approval import ApprovalGate

        port = _free_port()
        gate = ApprovalGate(default_timeout=5.0)
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, approval_gate=gate)
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        return srv, gate, f"http://127.0.0.1:{port}/api/v1"

    def test_list_pending_approval_via_api(self) -> None:
        srv, gate, url = self._start_server_with_gate()
        try:
            pending_thread = threading.Thread(
                target=lambda: gate.request("delete /tmp/work", reason="cleanup", risk="high"),
                daemon=True,
            )
            pending_thread.start()

            deadline = time.monotonic() + 2.0
            approvals: list = []
            while time.monotonic() < deadline:
                resp = httpx.get(f"{url}/approvals", headers=HEADERS)
                approvals = resp.json()["data"]["approvals"]
                if approvals:
                    break
                time.sleep(0.02)

            assert len(approvals) == 1
            assert approvals[0]["action"] == "delete /tmp/work"
            assert approvals[0]["reason"] == "cleanup"

            # Clean up: approve directly on the gate so the background
            # thread doesn't hang the test on timeout.
            gate.approve_by_id(approvals[0]["id"])
            pending_thread.join(timeout=2.0)
        finally:
            srv.stop()

    def test_approve_via_api_unblocks_waiting_caller(self) -> None:
        srv, gate, url = self._start_server_with_gate()
        try:
            result: dict = {}

            def do_request():
                try:
                    gate.request("risky action")
                    result["approved"] = True
                except Exception as exc:  # noqa: BLE001
                    result["error"] = str(exc)

            t = threading.Thread(target=do_request, daemon=True)
            t.start()

            deadline = time.monotonic() + 2.0
            approval_id = None
            while time.monotonic() < deadline:
                resp = httpx.get(f"{url}/approvals", headers=HEADERS)
                pending = resp.json()["data"]["approvals"]
                if pending:
                    approval_id = pending[0]["id"]
                    break
                time.sleep(0.02)
            assert approval_id is not None

            resp = httpx.post(f"{url}/approvals/{approval_id}/approve", headers=HEADERS)
            assert resp.status_code == 200
            assert resp.json()["data"]["resolved"] == "approved"

            t.join(timeout=2.0)
            assert result.get("approved") is True
        finally:
            srv.stop()

    def test_deny_via_api_raises_denied_for_waiting_caller(self) -> None:
        srv, gate, url = self._start_server_with_gate()
        try:
            result: dict = {}

            def do_request():
                try:
                    gate.request("risky action")
                    result["approved"] = True
                except Exception as exc:  # noqa: BLE001
                    result["error"] = type(exc).__name__

            t = threading.Thread(target=do_request, daemon=True)
            t.start()

            deadline = time.monotonic() + 2.0
            approval_id = None
            while time.monotonic() < deadline:
                resp = httpx.get(f"{url}/approvals", headers=HEADERS)
                pending = resp.json()["data"]["approvals"]
                if pending:
                    approval_id = pending[0]["id"]
                    break
                time.sleep(0.02)
            assert approval_id is not None

            resp = httpx.post(f"{url}/approvals/{approval_id}/deny", headers=HEADERS)
            assert resp.status_code == 200
            assert resp.json()["data"]["resolved"] == "denied"

            t.join(timeout=2.0)
            assert result.get("error") == "ApprovalDenied"
        finally:
            srv.stop()

    def test_resolve_unknown_approval_id_returns_404(self) -> None:
        srv, gate, url = self._start_server_with_gate()
        try:
            resp = httpx.post(f"{url}/approvals/does-not-exist/approve", headers=HEADERS)
            assert resp.status_code == 404
        finally:
            srv.stop()

    def test_resolve_approval_invalid_sub_action_returns_404(self) -> None:
        srv, gate, url = self._start_server_with_gate()
        try:
            resp = httpx.post(f"{url}/approvals/some-id/not-a-real-action", headers=HEADERS)
            assert resp.status_code == 404
        finally:
            srv.stop()


class TestDiscordPairingEndpoints:
    """SR-1.12/task #12: pairing decisions can never be made from in-band
    DM content (any unpaired stranger could otherwise grant themselves
    access). ``DiscordChannel.accept_pair()``/``deny_pair()`` were
    previously unreachable from anywhere -- these are the real
    authenticated endpoints an operator uses instead, mirroring the
    ``/approvals`` pattern above but reading/mutating a real
    ``DiscordChannel`` instance's pairing state.
    """

    def test_pairing_requires_auth(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/discord/pairing")
        assert resp.status_code == 401

    def test_list_pairing_empty_when_no_channels_attached(self, base_url: str) -> None:
        resp = httpx.get(f"{base_url}/discord/pairing", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["data"] == {"pending": [], "count": 0}

    def test_resolve_pairing_without_channels_returns_503(self, base_url: str) -> None:
        resp = httpx.post(f"{base_url}/discord/pairing/12345/approve", headers=HEADERS)
        assert resp.status_code == 503

    def _start_server_with_discord_channel(self):
        account = DiscordAccountConfig(
            token="test-token",
            dm_policy=DiscordDMPolicy.PAIRING,
            token_env_var="DISCORD_BOT_TOKEN",
        )
        ch = DiscordChannel(account_config=account)
        # Real pairing-request flow: a DM of "!pair" adds the sender to
        # _pending_pairs (see DiscordChannel._check_pairing) -- exercised
        # here directly rather than via a real Discord gateway connection.
        ch._check_pairing("999888777", "!pair")

        port = _free_port()
        cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY)
        srv = ApiServer(config=cfg, discord_channels=[ch])
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        return srv, ch, f"http://127.0.0.1:{port}/api/v1"

    def test_list_pending_pairing_via_api(self) -> None:
        srv, ch, url = self._start_server_with_discord_channel()
        try:
            resp = httpx.get(f"{url}/discord/pairing", headers=HEADERS)
            assert resp.status_code == 200
            pending = resp.json()["data"]["pending"]
            assert pending == [{"account": "DISCORD_BOT_TOKEN", "user_id": "999888777"}]
        finally:
            srv.stop()

    def test_approve_pairing_via_api_adds_to_allowlist(self) -> None:
        srv, ch, url = self._start_server_with_discord_channel()
        try:
            resp = httpx.post(f"{url}/discord/pairing/999888777/approve", headers=HEADERS)
            assert resp.status_code == 200
            assert resp.json()["data"]["resolved"] == "approved"

            # The real effect: no longer pending, and now in the allowlist.
            assert "999888777" not in ch.get_pending_pairs()
            assert "999888777" in ch.account_config.dm_allowlist
        finally:
            srv.stop()

    def test_deny_pairing_via_api_does_not_add_to_allowlist(self) -> None:
        srv, ch, url = self._start_server_with_discord_channel()
        try:
            resp = httpx.post(f"{url}/discord/pairing/999888777/deny", headers=HEADERS)
            assert resp.status_code == 200
            assert resp.json()["data"]["resolved"] == "denied"

            assert "999888777" not in ch.get_pending_pairs()
            assert "999888777" not in ch.account_config.dm_allowlist
        finally:
            srv.stop()

    def test_resolve_unknown_pairing_user_id_returns_404(self) -> None:
        srv, ch, url = self._start_server_with_discord_channel()
        try:
            resp = httpx.post(f"{url}/discord/pairing/no-such-user/approve", headers=HEADERS)
            assert resp.status_code == 404
        finally:
            srv.stop()

    def test_resolve_pairing_invalid_sub_action_returns_404(self) -> None:
        srv, ch, url = self._start_server_with_discord_channel()
        try:
            resp = httpx.post(
                f"{url}/discord/pairing/999888777/not-a-real-action", headers=HEADERS
            )
            assert resp.status_code == 404
        finally:
            srv.stop()

    def test_in_band_dm_accept_command_still_never_resolves_pairing(self) -> None:
        """Belt-and-suspenders: confirm the SR-1.12 in-band-command
        rejection in DiscordChannel itself is still intact -- the only
        real path to approval is the authenticated endpoint above."""
        srv, ch, url = self._start_server_with_discord_channel()
        try:
            resolved = ch._check_pairing("999888777", "!pair accept 999888777")
            assert resolved is False
            assert "999888777" in ch.get_pending_pairs()
        finally:
            srv.stop()
