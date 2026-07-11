"""Agent-as-a-Service REST API server for Missy.

Provides a minimal HTTP API for programmatic agent management: sessions,
chat, memory search, provider/tool introspection, and status.

The server binds to ``127.0.0.1`` by default (loopback only) and requires
an API key for every request.  All security decisions match the posture of
the existing :class:`~missy.channels.webhook.WebhookChannel`:

- Constant-time API key comparison to prevent timing attacks.
- Per-IP rate limiting (sliding window).
- Hard request body size cap.
- Security headers on every response.
- Secrets censored from agent output.

Example::

    from missy.api.server import ApiConfig, ApiServer
    from missy.agent.runtime import AgentRuntime, AgentConfig

    runtime = AgentRuntime(AgentConfig())
    server = ApiServer(ApiConfig(api_key="my-secret"), runtime=runtime)
    server.start()
    # server is now serving on http://127.0.0.1:8080
    server.stop()
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from missy.api.audit_browser import query_audit_events, redact_audit_value
from missy.api.diagnostics import build_diagnostics
from missy.api.operator_controls import execute_operator_control, list_operator_controls
from missy.api.run_stream import RunConflictError, RunRegistry, format_sse
from missy.api.web_console import render_console, render_login, render_message
from missy.api.web_sessions import WebSession, WebSessionStore
from missy.core.events import AuditEvent, event_bus

if TYPE_CHECKING:
    from missy.agent.runtime import AgentRuntime
    from missy.memory.sqlite_store import SQLiteMemoryStore
    from missy.providers.registry import ProviderRegistry
    from missy.tools.benchmark.benchmark_store import BenchmarkStore
    from missy.tools.intelligence.candidate_store import CandidateStore
    from missy.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_VERSION = "v1"
_API_PREFIX = f"/api/{_API_VERSION}"
_SERVER_BANNER = "missy-api"

# Maximum body bytes accepted per request (1 MiB).
_MAX_REQUEST_BYTES = 1_048_576
# Rate-limit sliding window in seconds.
_RATE_WINDOW_SECONDS = 60
# Maximum IPs tracked before evicting stale entries.
_MAX_TRACKED_IPS = 10_000

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ApiConfig:
    """Configuration for the :class:`ApiServer`.

    Attributes:
        host: Bind address.  Defaults to ``"127.0.0.1"`` (loopback only).
        port: TCP port to listen on.  Defaults to ``8080``.
        api_key: Required authentication key.  An empty value causes every
            request to be rejected with ``401``.
        max_request_bytes: Hard cap on request body size in bytes.
        rate_limit_rpm: Maximum requests per IP per 60-second window.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = ""
    max_request_bytes: int = _MAX_REQUEST_BYTES
    rate_limit_rpm: int = 60
    web_ui_enabled: bool = True
    web_session_ttl_seconds: int = 8 * 60 * 60
    web_cookie_name: str = "missy_operator_session"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


class ApiResponse:
    """Factory for canonical JSON response envelopes."""

    @staticmethod
    def ok(data: Any, status: int = 200) -> tuple[int, dict]:
        """Return a ``200 OK`` (or custom status) success envelope."""
        return status, {"status": "ok", "data": data}

    @staticmethod
    def created(data: Any) -> tuple[int, dict]:
        """Return a ``201 Created`` success envelope."""
        return 201, {"status": "ok", "data": data}

    @staticmethod
    def error(message: str, status: int = 400) -> tuple[int, dict]:
        """Return an error envelope with the given HTTP status code."""
        return status, {"status": "error", "error": message}


# ---------------------------------------------------------------------------
# Session registry (in-memory; survives per-server lifetime)
# ---------------------------------------------------------------------------


@dataclass
class _ApiSession:
    """Lightweight session record managed by the API server."""

    session_id: str
    created_at: str
    provider: str
    name: str = ""
    turn_count: int = 0
    last_used_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "provider": self.provider,
            "turn_count": self.turn_count,
        }


class _SessionRegistry:
    """Thread-safe in-process registry of API sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, _ApiSession] = {}
        self._lock = threading.Lock()

    def create(self, provider: str, name: str = "") -> _ApiSession:
        import uuid

        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        sess = _ApiSession(session_id=session_id, created_at=now, provider=provider, name=name)
        with self._lock:
            self._sessions[session_id] = sess
        return sess

    def get(self, session_id: str) -> _ApiSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list(self, limit: int = 20) -> list[_ApiSession]:
        with self._lock:
            sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.last_used_at, reverse=True)
        return sessions[:limit]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def touch(self, session_id: str, turn_increment: int = 0) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.last_used_at = datetime.now(UTC).isoformat()
                sess.turn_count += turn_increment

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------


def _make_handler(
    api_config: ApiConfig,
    session_registry: _SessionRegistry,
    web_sessions: WebSessionStore,
    rate_tracker: dict,
    rate_lock: threading.Lock,
    runtime: AgentRuntime | None,
    memory_store: SQLiteMemoryStore | None,
    provider_registry: ProviderRegistry | None,
    tool_registry: ToolRegistry | None,
    candidate_store: CandidateStore | None,
    benchmark_store: BenchmarkStore | None,
    run_registry: RunRegistry,
    approval_gate: Any | None = None,
    discord_channels: list | None = None,
):
    """Return a configured :class:`BaseHTTPRequestHandler` subclass.

    All dependencies are captured by closure so the handler class can be
    instantiated by :class:`~http.server.ThreadingHTTPServer` without arguments.
    """

    class ApiHandler(BaseHTTPRequestHandler):
        server_version = _SERVER_BANNER
        sys_version = ""

        # ----------------------------------------------------------------
        # HTTP verb dispatch
        # ----------------------------------------------------------------

        def version_string(self) -> str:  # type: ignore[override]
            return _SERVER_BANNER

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def do_DELETE(self) -> None:
            self._handle("DELETE")

        # ----------------------------------------------------------------
        # Core pipeline
        # ----------------------------------------------------------------

        def _handle(self, method: str) -> None:
            """Rate-limit, authenticate, route, and respond."""
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}

            client_ip = self.client_address[0]
            if not self._check_rate_limit(client_ip):
                if path.startswith(_API_PREFIX):
                    self._send_json(*ApiResponse.error("Rate limit exceeded", 429))
                else:
                    self._send_html(
                        429,
                        self._render_message("Rate limit exceeded", "Too many requests."),
                    )
                return

            if api_config.web_ui_enabled and not path.startswith(_API_PREFIX):
                self._route_web(method, path, params)
                return

            events_prefix = f"{_API_PREFIX}/runs/"
            if method == "GET" and path.startswith(events_prefix) and path.endswith("/events"):
                if self._authenticate() is None:
                    self._send_json(*ApiResponse.error("Unauthorized", 401))
                    return
                run_id = path[len(events_prefix) : -len("/events")]
                self._handle_run_events_stream(run_id)
                return

            auth_kind = self._authenticate()
            if auth_kind is None:
                self._send_json(*ApiResponse.error("Unauthorized", 401))
                return
            if auth_kind == "web_session" and method in {"POST", "PUT", "PATCH", "DELETE"}:
                web_session = self._current_web_session()
                provided = self.headers.get("X-CSRF-Token", "")
                expected = web_session.csrf_token if web_session is not None else ""
                if not provided or not hmac.compare_digest(provided, expected):
                    self._emit_web_audit(
                        event_type="web.csrf",
                        result="deny",
                        action="api_csrf_check",
                        subsystem="security",
                        severity="warning",
                        path=path,
                        method=method,
                    )
                    self._send_json(*ApiResponse.error("CSRF token required", 403))
                    return

            status, body = self._route(method, path, params)
            self._send_json(status, body)

        def _authenticate(self) -> str | None:
            """Return the authenticated credential kind, if any.

            Accepts either ``Authorization: Bearer <key>`` or
            ``X-API-Key: <key>``. Browser sessions are accepted only when
            the Web UI is enabled. Unsafe browser-authenticated API requests
            are CSRF-checked by the caller.

            An empty configured key never authenticates API-key requests.
            """
            expected = api_config.api_key
            if expected:
                auth_header = self.headers.get("Authorization", "")
                if auth_header.lower().startswith("bearer "):
                    provided = auth_header[7:]
                else:
                    provided = self.headers.get("X-API-Key", "")

                if provided and hmac.compare_digest(provided.encode(), expected.encode()):
                    return "api_key"

            if api_config.web_ui_enabled and self._current_web_session() is not None:
                return "web_session"

            return None

        def _current_web_session(self) -> WebSession | None:
            return web_sessions.get(self._cookie_value(api_config.web_cookie_name))

        def _cookie_value(self, name: str) -> str | None:
            raw = self.headers.get("Cookie")
            if not raw:
                return None
            jar = cookies.SimpleCookie()
            try:
                jar.load(raw)
            except cookies.CookieError:
                return None
            morsel = jar.get(name)
            return morsel.value if morsel is not None else None

        def _check_rate_limit(self, client_ip: str) -> bool:
            """Sliding-window rate limit: returns False when limit exceeded."""
            now = time.monotonic()
            cutoff = now - _RATE_WINDOW_SECONDS
            with rate_lock:
                timestamps = rate_tracker.get(client_ip, [])
                timestamps = [t for t in timestamps if t > cutoff]
                if len(timestamps) >= api_config.rate_limit_rpm:
                    rate_tracker[client_ip] = timestamps
                    return False
                timestamps.append(now)
                rate_tracker[client_ip] = timestamps
                if len(rate_tracker) > _MAX_TRACKED_IPS:
                    _evict_stale(rate_tracker, cutoff)
            return True

        # ----------------------------------------------------------------
        # Router
        # ----------------------------------------------------------------

        def _route(self, method: str, path: str, params: dict) -> tuple[int, dict]:
            """Map (method, path) to a handler method."""
            # Static routes
            if method == "GET" and path == f"{_API_PREFIX}/health":
                return self._handle_health()
            if method == "GET" and path == f"{_API_PREFIX}/status":
                return self._handle_status()
            if method == "GET" and path == f"{_API_PREFIX}/providers":
                return self._handle_list_providers()
            if method == "GET" and path == f"{_API_PREFIX}/tools":
                return self._handle_list_tools()
            if method == "GET" and path == f"{_API_PREFIX}/diagnostics":
                return self._handle_diagnostics()
            if method == "GET" and path == f"{_API_PREFIX}/costs/summary":
                return self._handle_cost_summary(params)
            if method == "GET" and path == f"{_API_PREFIX}/controls":
                return self._handle_list_controls()
            if method == "GET" and path == f"{_API_PREFIX}/approvals":
                return self._handle_list_approvals()
            if method == "GET" and path == f"{_API_PREFIX}/discord/pairing":
                return self._handle_list_discord_pairing()
            if method == "GET" and path == f"{_API_PREFIX}/tool-candidates":
                return self._handle_list_tool_candidates(params)
            if method == "GET" and path == f"{_API_PREFIX}/audit":
                return self._handle_audit_events(params)
            if method == "GET" and path == f"{_API_PREFIX}/memory/search":
                return self._handle_memory_search(params)
            if method == "GET" and path == f"{_API_PREFIX}/scheduler/jobs":
                return self._handle_list_scheduled_jobs()
            if method == "POST" and path == f"{_API_PREFIX}/scheduler/jobs":
                body = self._read_body()
                if body is None:
                    return ApiResponse.error("Invalid JSON body", 400)
                return self._handle_create_scheduled_job(body)

            # Session collection
            if method == "POST" and path == f"{_API_PREFIX}/sessions":
                body = self._read_body()
                if body is None:
                    return ApiResponse.error("Invalid JSON body", 400)
                return self._handle_create_session(body)
            if method == "GET" and path == f"{_API_PREFIX}/sessions":
                return self._handle_list_sessions(params)

            # Chat
            if method == "POST" and path == f"{_API_PREFIX}/chat":
                body = self._read_body()
                if body is None:
                    return ApiResponse.error("Invalid JSON body", 400)
                return self._handle_chat(body)

            # Streamed background runs
            if method == "POST" and path == f"{_API_PREFIX}/runs":
                body = self._read_body()
                if body is None:
                    return ApiResponse.error("Invalid JSON body", 400)
                return self._handle_start_run(body)
            if method == "GET" and path == f"{_API_PREFIX}/runs":
                return self._handle_list_runs(params)

            runs_prefix = f"{_API_PREFIX}/runs/"
            if path.startswith(runs_prefix):
                run_id = path[len(runs_prefix) :]
                if not run_id or "/" in run_id:
                    return ApiResponse.error("Not found", 404)
                if method == "GET":
                    return self._handle_get_run(run_id)

            controls_prefix = f"{_API_PREFIX}/controls/"
            if method == "POST" and path.startswith(controls_prefix):
                body = self._read_body()
                if body is None:
                    return ApiResponse.error("Invalid JSON body", 400)
                control_id = path[len(controls_prefix) :]
                if not control_id or "/" in control_id:
                    return ApiResponse.error("Not found", 404)
                return self._handle_execute_control(control_id, body)

            # Approval item routes — extract {id}/approve or {id}/deny
            approvals_prefix = f"{_API_PREFIX}/approvals/"
            if method == "POST" and path.startswith(approvals_prefix):
                rest = path[len(approvals_prefix) :]
                segments = rest.split("/", 1)
                approval_id = segments[0]
                sub = segments[1] if len(segments) > 1 else ""
                if not approval_id:
                    return ApiResponse.error("Not found", 404)
                if sub == "approve":
                    return self._handle_resolve_approval(approval_id, approve=True)
                if sub == "deny":
                    return self._handle_resolve_approval(approval_id, approve=False)
                return ApiResponse.error("Not found", 404)

            # Discord pairing item routes — extract {user_id}/approve or /deny
            pairing_prefix = f"{_API_PREFIX}/discord/pairing/"
            if method == "POST" and path.startswith(pairing_prefix):
                rest = path[len(pairing_prefix) :]
                segments = rest.split("/", 1)
                pairing_user_id = segments[0]
                sub = segments[1] if len(segments) > 1 else ""
                if not pairing_user_id:
                    return ApiResponse.error("Not found", 404)
                if sub == "approve":
                    return self._handle_resolve_discord_pairing(pairing_user_id, approve=True)
                if sub == "deny":
                    return self._handle_resolve_discord_pairing(pairing_user_id, approve=False)
                return ApiResponse.error("Not found", 404)

            candidates_prefix = f"{_API_PREFIX}/tool-candidates/"
            if path.startswith(candidates_prefix):
                candidate_id = path[len(candidates_prefix) :]
                if not candidate_id or "/" in candidate_id:
                    return ApiResponse.error("Not found", 404)
                if method == "GET":
                    return self._handle_get_tool_candidate(candidate_id)

            # Memory turn item routes — extract {id} segment
            turns_prefix = f"{_API_PREFIX}/memory/turns/"
            if path.startswith(turns_prefix):
                rest = path[len(turns_prefix) :]
                segments = rest.split("/", 1)
                turn_id = segments[0]
                sub = segments[1] if len(segments) > 1 else ""
                if not turn_id:
                    return ApiResponse.error("Not found", 404)
                if method == "DELETE" and sub == "":
                    return self._handle_delete_memory_turn(turn_id)
                if method == "POST" and sub == "pin":
                    body = self._read_body()
                    if body is None:
                        return ApiResponse.error("Invalid JSON body", 400)
                    return self._handle_pin_memory_turn(turn_id, body)

            # Scheduler job item routes — extract {id} segment
            jobs_prefix = f"{_API_PREFIX}/scheduler/jobs/"
            if path.startswith(jobs_prefix):
                job_id = path[len(jobs_prefix) :]
                if not job_id or "/" in job_id:
                    return ApiResponse.error("Not found", 404)
                if method == "DELETE":
                    body = self._read_body() or {}
                    return self._handle_execute_control(
                        "scheduler.remove_job", {**body, "target": job_id}
                    )

            # Session item routes — extract {id} segment
            prefix = f"{_API_PREFIX}/sessions/"
            if path.startswith(prefix):
                rest = path[len(prefix) :]
                segments = rest.split("/", 1)
                session_id = segments[0]
                sub = segments[1] if len(segments) > 1 else ""

                if not session_id:
                    return ApiResponse.error("Not found", 404)

                if method == "GET" and sub == "history":
                    return self._handle_session_history(session_id, params)
                if method == "GET" and sub == "":
                    return self._handle_get_session(session_id)
                if method == "DELETE" and sub == "":
                    return self._handle_delete_session(session_id)

            return ApiResponse.error("Not found", 404)

        # ----------------------------------------------------------------
        # Request body
        # ----------------------------------------------------------------

        def _read_body(self) -> dict | None:
            """Read, size-check, and parse the JSON request body.

            Returns ``None`` on any parse or size error.
            """
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if content_type != "application/json":
                return None

            try:
                length = int(self.headers.get("Content-Length", 0))
                if length < 0:
                    return None
            except (ValueError, TypeError):
                return None

            if length > api_config.max_request_bytes:
                return None

            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None

        # ----------------------------------------------------------------
        # Response sender
        # ----------------------------------------------------------------

        def _send_json(self, status: int, data: dict) -> None:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(
            self,
            status: int,
            body_text: str,
            *,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = body_text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            )
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str, *, extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(303)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _route_web(self, method: str, path: str, params: dict) -> None:
            if method == "GET" and path == "/login":
                if self._current_web_session() is not None:
                    self._redirect("/")
                    return
                message = str(params.get("error", ""))
                self._send_html(200, self._render_login(error=message == "1"))
                return

            if method == "POST" and path == "/login":
                self._handle_web_login()
                return

            if method == "POST" and path == "/logout":
                self._handle_web_logout()
                return

            if method == "GET" and path == "/":
                session = self._current_web_session()
                if session is None:
                    self._redirect("/login")
                    return
                self._send_html(200, self._render_console(session))
                return

            self._send_html(404, self._render_message("Not found", "No operator page exists here."))

        def _handle_web_login(self) -> None:
            body = self._read_form()
            provided = body.get("api_key", "")
            expected = api_config.api_key
            if not expected or not provided or not hmac.compare_digest(provided, expected):
                self._emit_web_audit(
                    event_type="web.login",
                    result="deny",
                    action="login",
                    subsystem="auth",
                    severity="warning",
                    reason="invalid_key",
                )
                self._redirect("/login?error=1")
                return

            session = web_sessions.create()
            self._emit_web_audit(
                event_type="web.login",
                result="allow",
                action="login",
                subsystem="auth",
                severity="info",
            )
            cookie = self._make_cookie(api_config.web_cookie_name, session.token)
            self._redirect("/", extra_headers={"Set-Cookie": cookie})

        def _handle_web_logout(self) -> None:
            token = self._cookie_value(api_config.web_cookie_name)
            session = web_sessions.get(token)
            provided = self.headers.get("X-CSRF-Token", "")
            if (
                session is None
                or not provided
                or not hmac.compare_digest(provided, session.csrf_token)
            ):
                self._emit_web_audit(
                    event_type="web.logout",
                    result="deny",
                    action="logout",
                    subsystem="auth",
                    severity="warning",
                    reason="csrf_required",
                )
                self._send_html(
                    403,
                    self._render_message(
                        "CSRF token required", "Refresh the console and try again."
                    ),
                )
                return
            web_sessions.revoke(token)
            self._emit_web_audit(
                event_type="web.logout",
                result="allow",
                action="logout",
                subsystem="auth",
                severity="info",
            )
            self._redirect(
                "/login",
                extra_headers={
                    "Set-Cookie": self._make_cookie(api_config.web_cookie_name, "", max_age=0)
                },
            )

        def _read_form(self) -> dict[str, str]:
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if content_type != "application/x-www-form-urlencoded":
                return {}
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                return {}
            if length < 0 or length > min(api_config.max_request_bytes, 64_000):
                return {}
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            return {k: v[0] if v else "" for k, v in parse_qs(raw, keep_blank_values=True).items()}

        def _make_cookie(self, name: str, value: str, *, max_age: int | None = None) -> str:
            morsel = cookies.SimpleCookie()
            morsel[name] = value
            morsel[name]["path"] = "/"
            morsel[name]["httponly"] = True
            morsel[name]["samesite"] = "Strict"
            if max_age is not None:
                morsel[name]["max-age"] = str(max_age)
            return morsel.output(header="").strip()

        def _emit_web_audit(
            self,
            *,
            event_type: str,
            result: str,
            action: str,
            subsystem: str,
            severity: str,
            **detail: Any,
        ) -> None:
            safe_detail = {
                "actor": "operator",
                "source": "web_tui",
                "subsystem": subsystem,
                "action": action,
                "severity": severity,
                "remote_addr": self.client_address[0],
                **detail,
            }
            with contextlib.suppress(Exception):
                event_bus.publish(
                    AuditEvent.now(
                        session_id="web",
                        task_id="-",
                        event_type=event_type,
                        category="channel",
                        result=result,  # type: ignore[arg-type]
                        detail=redact_audit_value(safe_detail),
                    )
                )

        def _render_login(self, *, error: bool = False) -> str:
            return render_login(error=error)

        def _render_message(self, title: str, message: str) -> str:
            return render_message(title, message)

        def _render_console(self, session: WebSession) -> str:
            return render_console(csrf_token=session.csrf_token)

        # ----------------------------------------------------------------
        # Route handlers
        # ----------------------------------------------------------------

        def _handle_health(self) -> tuple[int, dict]:
            """GET /api/v1/health — liveness probe."""
            return ApiResponse.ok({"status": "healthy", "version": "0.1.0"})

        def _handle_status(self) -> tuple[int, dict]:
            """GET /api/v1/status — agent and subsystem status."""
            reg = provider_registry
            providers_available: list[str] = []
            default_provider: str | None = None
            if reg is not None:
                try:
                    providers_available = [p.name for p in reg.get_available()]
                    default_provider = reg.get_default_name()
                except Exception:
                    pass

            tool_count = 0
            tr = tool_registry
            if tr is not None:
                with contextlib.suppress(Exception):
                    tool_count = len(tr.list_tools())

            mem_stats: dict = {}
            ms = memory_store
            if ms is not None:
                try:
                    recent = ms.get_recent_turns(limit=1)
                    mem_stats["has_memory"] = True
                    mem_stats["last_turn_at"] = recent[0].timestamp if recent else None
                except Exception:
                    mem_stats["has_memory"] = False

            return ApiResponse.ok(
                {
                    "providers_available": providers_available,
                    "default_provider": default_provider,
                    "tool_count": tool_count,
                    "session_count": session_registry.count(),
                    "memory": mem_stats,
                }
            )

        def _handle_list_providers(self) -> tuple[int, dict]:
            """GET /api/v1/providers — list registered providers."""
            reg = provider_registry
            if reg is None:
                return ApiResponse.ok({"providers": []})

            result = []
            try:
                for name in reg.list_providers():
                    provider = reg.get(name)
                    available = False
                    with contextlib.suppress(Exception):
                        available = provider.is_available() if provider else False
                    result.append(
                        {
                            "name": name,
                            "available": available,
                            "is_default": name == reg.get_default_name(),
                        }
                    )
            except Exception as exc:
                logger.warning("Error listing providers: %s", exc)
            return ApiResponse.ok({"providers": result})

        def _handle_list_tools(self) -> tuple[int, dict]:
            """GET /api/v1/tools — list registered tools with schemas."""
            tr = tool_registry
            if tr is None:
                return ApiResponse.ok({"tools": []})

            result = []
            try:
                for name in tr.list_tools():
                    tool = tr.get(name)
                    if tool is None:
                        continue
                    schema: dict = {}
                    with contextlib.suppress(Exception):
                        schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
                    result.append(
                        {
                            "name": name,
                            "description": getattr(tool, "description", ""),
                            "schema": schema,
                        }
                    )
            except Exception as exc:
                logger.warning("Error listing tools: %s", exc)
            return ApiResponse.ok({"tools": result})

        def _handle_audit_events(self, params: dict) -> tuple[int, dict]:
            """GET /api/v1/audit — browse redacted audit events with filters."""
            return ApiResponse.ok(query_audit_events(params))

        def _handle_diagnostics(self) -> tuple[int, dict]:
            """GET /api/v1/diagnostics — local operator doctor summary."""
            return ApiResponse.ok(
                build_diagnostics(
                    api_config=api_config,
                    session_count=session_registry.count(),
                    runtime=runtime,
                    memory_store=memory_store,
                    provider_registry=provider_registry,
                    tool_registry=tool_registry,
                )
            )

        def _handle_cost_summary(self, params: dict) -> tuple[int, dict]:
            """GET /api/v1/costs/summary — dashboard-wide spend and usage.

            Query params:
                limit (int, optional, default 10): Number of top sessions
                    (by most recent activity) to include in the breakdown.

            Response data fields:
                totals (dict): Aggregate call_count/session_count/token/cost
                    totals across every recorded session.
                sessions (list): Per-session cost breakdown, most recent first.
                budget (dict): The configured ``max_spend_usd`` cap (0 means
                    unlimited) plus whether it applies per-session.
            """
            ms = memory_store
            if ms is None:
                return ApiResponse.ok(
                    {
                        "totals": {
                            "call_count": 0,
                            "session_count": 0,
                            "total_prompt_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_cost_usd": 0.0,
                        },
                        "sessions": [],
                        "budget": {"max_spend_usd": 0.0, "scope": "per_session"},
                    }
                )
            try:
                limit = max(1, min(int(params.get("limit", 10)), 100))
            except (TypeError, ValueError):
                limit = 10
            totals = {
                "call_count": 0,
                "session_count": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_usd": 0.0,
            }
            sessions: list = []
            try:
                totals = ms.get_cost_totals()
                sessions = ms.get_total_costs(limit=limit)
            except Exception as exc:
                logger.warning("Cost summary error: %s", exc)
            budget_cap = (
                float(getattr(runtime.config, "max_spend_usd", 0.0) or 0.0) if runtime else 0.0
            )
            return ApiResponse.ok(
                {
                    "totals": totals,
                    "sessions": sessions,
                    "budget": {"max_spend_usd": budget_cap, "scope": "per_session"},
                }
            )

        def _handle_list_controls(self) -> tuple[int, dict]:
            """GET /api/v1/controls — list safe operator controls."""
            return ApiResponse.ok(
                list_operator_controls(
                    provider_registry=provider_registry,
                    scheduler=getattr(runtime, "_scheduler", None) if runtime is not None else None,
                    candidate_store=candidate_store,
                )
            )

        def _handle_execute_control(self, control_id: str, body: dict) -> tuple[int, dict]:
            """POST /api/v1/controls/{id} — execute a confirmed operator control."""
            status, data, audit_detail = execute_operator_control(
                control_id,
                body,
                provider_registry=provider_registry,
                scheduler=getattr(runtime, "_scheduler", None) if runtime is not None else None,
                candidate_store=candidate_store,
                benchmark_store=benchmark_store,
            )
            result = "allow" if status < 400 else "deny"
            severity = "info" if result == "allow" else "warning"
            self._emit_web_audit(
                event_type="web.control",
                result=result,
                action=str(audit_detail.get("action") or control_id),
                subsystem=str(audit_detail.get("subsystem") or "control"),
                severity=severity,
                status_code=status,
                **{
                    key: value
                    for key, value in audit_detail.items()
                    if key not in {"action", "subsystem", "severity"}
                },
            )
            if status >= 400:
                return ApiResponse.error(str(data.get("message") or "Control denied"), status)
            return ApiResponse.ok(data)

        def _handle_list_approvals(self) -> tuple[int, dict]:
            """GET /api/v1/approvals — list pending approval requests.

            SR-2.2: this is the real, in-process view of whatever
            ApprovalGate the gateway was started with (e.g. proactive
            triggers with requires_confirmation=True gate through it).
            A separate `missy` CLI invocation cannot see this state
            directly since it runs as its own process; this endpoint is
            the actual mechanism for an operator to inspect and resolve
            pending requests while the gateway is running.
            """
            if approval_gate is None:
                return ApiResponse.ok({"approvals": [], "count": 0})
            pending = approval_gate.list_pending()
            return ApiResponse.ok({"approvals": pending, "count": len(pending)})

        def _handle_resolve_approval(self, approval_id: str, *, approve: bool) -> tuple[int, dict]:
            """POST /api/v1/approvals/{id}/approve or /deny."""
            if approval_gate is None:
                return ApiResponse.error("No approval gate attached to this server", 503)
            resolver = approval_gate.approve_by_id if approve else approval_gate.deny_by_id
            resolved = resolver(approval_id)
            if not resolved:
                return ApiResponse.error(
                    f"No pending approval with id {approval_id!r} (already resolved, "
                    "timed out, or never existed).",
                    404,
                )
            self._emit_web_audit(
                event_type="web.approval",
                result="allow" if approve else "deny",
                action="approve" if approve else "deny",
                subsystem="approval_gate",
                severity="info",
                approval_id=approval_id,
            )
            return ApiResponse.ok(
                {"id": approval_id, "resolved": "approved" if approve else "denied"}
            )

        def _handle_list_discord_pairing(self) -> tuple[int, dict]:
            """GET /api/v1/discord/pairing — list pending Discord DM pairing requests.

            SR-1.12: pairing decisions can never be made from in-band DM
            content (any unpaired stranger could otherwise grant
            themselves access). This authenticated endpoint is the real
            approval surface an operator uses instead --
            ``DiscordChannel.accept_pair()``/``deny_pair()`` were
            previously unreachable from any process outside the running
            gateway's own in-memory state.
            """
            channels = discord_channels or []
            pending = []
            for ch in channels:
                account_label = getattr(getattr(ch, "account_config", None), "token_env_var", "")
                for user_id in ch.get_pending_pairs():
                    pending.append({"account": account_label, "user_id": user_id})
            return ApiResponse.ok({"pending": pending, "count": len(pending)})

        def _handle_resolve_discord_pairing(
            self, user_id: str, *, approve: bool
        ) -> tuple[int, dict]:
            """POST /api/v1/discord/pairing/{user_id}/approve or /deny."""
            channels = discord_channels or []
            if not channels:
                return ApiResponse.error("No Discord channels attached to this server", 503)

            resolved_any = False
            for ch in channels:
                if user_id in ch.get_pending_pairs():
                    if approve:
                        ch.accept_pair(user_id)
                    else:
                        ch.deny_pair(user_id)
                    resolved_any = True

            if not resolved_any:
                return ApiResponse.error(
                    f"No pending pairing request for user {user_id!r} (already "
                    "resolved or never requested).",
                    404,
                )

            self._emit_web_audit(
                event_type="web.discord_pairing",
                result="allow" if approve else "deny",
                action="approve" if approve else "deny",
                subsystem="discord",
                severity="info",
                user_id=user_id,
            )
            return ApiResponse.ok(
                {"user_id": user_id, "resolved": "approved" if approve else "denied"}
            )

        def _handle_list_tool_candidates(self, params: dict) -> tuple[int, dict]:
            """GET /api/v1/tool-candidates — list reviewable tool candidates."""
            if candidate_store is None:
                return ApiResponse.error("Candidate store is not attached", 503)
            try:
                from missy.tools.intelligence import ToolLifecycleState

                state_param = str(params.get("state") or "").strip()
                state = ToolLifecycleState(state_param) if state_param else None
                owner = str(params.get("owner") or "").strip() or None
                limit = max(1, min(int(params.get("limit", 100)), 500))
                candidates = candidate_store.list_all(state=state, owner=owner, limit=limit)
            except ValueError:
                return ApiResponse.error("Invalid candidate state", 400)
            except Exception as exc:
                logger.warning("Tool candidate list error: %s", exc)
                return ApiResponse.error("Candidate store is unavailable", 503)
            return ApiResponse.ok(
                {
                    "candidates": [candidate.to_dict() for candidate in candidates],
                    "count": len(candidates),
                }
            )

        def _handle_get_tool_candidate(self, candidate_id: str) -> tuple[int, dict]:
            """GET /api/v1/tool-candidates/{id} — show one candidate."""
            if candidate_store is None:
                return ApiResponse.error("Candidate store is not attached", 503)
            try:
                candidate = candidate_store.get(candidate_id)
            except Exception as exc:
                logger.warning("Tool candidate lookup error: %s", exc)
                return ApiResponse.error("Candidate store is unavailable", 503)
            if candidate is None:
                return ApiResponse.error("Candidate not found", 404)
            return ApiResponse.ok({"candidate": candidate.to_dict()})

        def _handle_chat(self, body: dict) -> tuple[int, dict]:
            """POST /api/v1/chat — send a message to the agent and get a reply.

            Request body fields:
                message (str, required): The user message text.
                session_id (str, optional): Existing session ID to continue.
                provider (str, optional): Override the provider for this turn.

            Response data fields:
                response (str): The agent's reply.
                session_id (str): The session used/created.
                usage (dict): Token usage if available (may be empty).
            """
            message = (body.get("message") or "").strip()
            if not message:
                return ApiResponse.error("'message' field is required and must be non-empty", 400)

            session_id: str | None = body.get("session_id") or None
            provider_override: str | None = body.get("provider") or None

            if runtime is None:
                return ApiResponse.error("Agent runtime is not configured", 503)

            # Create a session record in the registry when none is provided.
            if session_id is None:
                default_provider = (
                    (provider_registry.get_default_name() if provider_registry is not None else "")
                    or provider_override
                    or runtime.config.provider
                )
                api_sess = session_registry.create(provider=default_provider)
                session_id = api_sess.session_id
            else:
                if session_registry.get(session_id) is None:
                    return ApiResponse.error(f"Session '{session_id}' not found", 404)

            # Optionally switch provider before this run.
            if provider_override and runtime is not None:
                try:
                    runtime.switch_provider(provider_override)
                except Exception as exc:
                    logger.debug("Could not switch provider to %r: %s", provider_override, exc)

            try:
                response_text = runtime.run(message, session_id=session_id)
            except Exception as exc:
                logger.error("Agent runtime error: %s", exc, exc_info=True)
                return ApiResponse.error(f"Agent error: {exc}", 500)

            # Censor secrets from outbound response.
            try:
                from missy.security.censor import censor_response

                response_text = censor_response(response_text)
            except Exception:
                pass

            session_registry.touch(session_id, turn_increment=1)

            return ApiResponse.ok(
                {
                    "response": response_text,
                    "session_id": session_id,
                    "usage": {},
                }
            )

        def _handle_start_run(self, body: dict) -> tuple[int, dict]:
            """POST /api/v1/runs — start a background run streamed via SSE.

            Request body fields:
                message (str, required): The user message text.
                session_id (str, optional): Existing session ID to continue.
                provider (str, optional): Override the provider for this turn.

            Response data fields mirror :meth:`RunHandle.to_dict`: ``run_id``,
            ``session_id``, ``status``, and (once finished) ``response`` or
            ``error``. Connect to ``GET /api/v1/runs/{run_id}/events`` to
            watch progress live.
            """
            message = (body.get("message") or "").strip()
            if not message:
                return ApiResponse.error("'message' field is required and must be non-empty", 400)

            if runtime is None:
                return ApiResponse.error("Agent runtime is not configured", 503)

            session_id: str | None = body.get("session_id") or None
            provider_override: str | None = body.get("provider") or None

            if session_id is None:
                default_provider = (
                    (provider_registry.get_default_name() if provider_registry is not None else "")
                    or provider_override
                    or runtime.config.provider
                )
                api_sess = session_registry.create(provider=default_provider)
                session_id = api_sess.session_id
                provider_name = default_provider
            else:
                api_sess = session_registry.get(session_id)
                if api_sess is None:
                    return ApiResponse.error(f"Session '{session_id}' not found", 404)
                provider_name = provider_override or api_sess.provider or runtime.config.provider

            if provider_override:
                try:
                    runtime.switch_provider(provider_override)
                except Exception as exc:
                    logger.debug("Could not switch provider to %r: %s", provider_override, exc)

            try:
                handle = run_registry.start(
                    runtime=runtime,
                    message=message,
                    session_id=session_id,
                    provider=provider_name,
                )
            except RunConflictError:
                self._emit_web_audit(
                    event_type="web.run",
                    result="deny",
                    action="run.start",
                    subsystem="agent",
                    severity="warning",
                    session_id=session_id,
                    reason="run_in_progress",
                )
                return ApiResponse.error(
                    f"Session '{session_id}' already has a run in progress", 409
                )

            session_registry.touch(session_id, turn_increment=1)

            return ApiResponse.ok(handle.to_dict(), status=202)

        def _handle_list_runs(self, params: dict) -> tuple[int, dict]:
            """GET /api/v1/runs?session_id=... — list recent runs for a session."""
            session_id = str(params.get("session_id") or "").strip()
            if not session_id:
                return ApiResponse.error("Query parameter 'session_id' is required", 400)
            try:
                limit = max(1, min(int(params.get("limit", 20)), 100))
            except (ValueError, TypeError):
                limit = 20
            return ApiResponse.ok({"runs": run_registry.list_for_session(session_id, limit=limit)})

        def _handle_get_run(self, run_id: str) -> tuple[int, dict]:
            """GET /api/v1/runs/{run_id} — poll a run's current status/result."""
            handle = run_registry.get(run_id)
            if handle is None:
                return ApiResponse.error(f"Run '{run_id}' not found", 404)
            return ApiResponse.ok(handle.to_dict())

        def _handle_run_events_stream(self, run_id: str) -> None:
            """GET /api/v1/runs/{run_id}/events — SSE stream of run progress.

            Emits ``run.started``, ``run.start`` (bus-sourced), ``tool.request``,
            ``tool.result``, and a terminal ``run.complete``/``run.error``
            event. Late joiners (including reconnects after the run already
            finished) receive a synthesized terminal event immediately rather
            than hanging.
            """
            if not run_id or run_registry.get(run_id) is None:
                self._send_json(*ApiResponse.error(f"Run '{run_id}' not found", 404))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                for event in run_registry.stream(run_id):
                    self.wfile.write(format_sse(event))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _handle_create_session(self, body: dict) -> tuple[int, dict]:
            """POST /api/v1/sessions — create a new API session.

            Request body fields:
                name (str, optional): Friendly label for the session.
                provider (str, optional): Provider name for this session.

            Response data fields:
                session_id (str): The new session's ID.
                created_at (str): ISO-8601 creation timestamp.
                provider (str): Provider name associated with this session.
            """
            name = str(body.get("name") or "")[:128]
            provider = str(body.get("provider") or "")[:64]

            if not provider:
                if provider_registry is not None:
                    provider = provider_registry.get_default_name() or ""
                if not provider and runtime is not None:
                    provider = runtime.config.provider

            sess = session_registry.create(provider=provider, name=name)
            return ApiResponse.created(sess.to_dict())

        def _handle_list_sessions(self, params: dict) -> tuple[int, dict]:
            """GET /api/v1/sessions — list active API sessions.

            Query parameters:
                limit (int, default 20): Maximum number of sessions to return.

            Response data fields:
                sessions (list[dict]): Session records, most-recent first.
            """
            try:
                limit = max(1, min(int(params.get("limit", 20)), 200))
            except (ValueError, TypeError):
                limit = 20

            sessions = session_registry.list(limit=limit)

            # Augment with memory store turn counts when available. The
            # counts lookup is session-independent, so it must be built once
            # before the loop -- building it per-session re-ran the same
            # 1000-row query once per returned session (e.g. 200x for a
            # limit=200 request).
            if memory_store is not None:
                try:
                    db_sessions = memory_store.list_sessions(limit=1000)
                    counts = {s["session_id"]: s["turn_count"] for s in db_sessions}
                except Exception:
                    counts = {}
                for sess in sessions:
                    if sess.session_id in counts:
                        sess.turn_count = counts[sess.session_id]

            return ApiResponse.ok({"sessions": [s.to_dict() for s in sessions]})

        def _handle_get_session(self, session_id: str) -> tuple[int, dict]:
            """GET /api/v1/sessions/{id} — get details of a single session."""
            sess = session_registry.get(session_id)
            if sess is None:
                return ApiResponse.error(f"Session '{session_id}' not found", 404)
            return ApiResponse.ok(sess.to_dict())

        def _handle_session_history(self, session_id: str, params: dict) -> tuple[int, dict]:
            """GET /api/v1/sessions/{id}/history — retrieve conversation turns.

            Query parameters:
                limit (int, default 50): Maximum number of turns to return.

            Response data fields:
                turns (list[dict]): Turns in chronological order.
            """
            sess = session_registry.get(session_id)
            if sess is None:
                return ApiResponse.error(f"Session '{session_id}' not found", 404)

            try:
                limit = max(1, min(int(params.get("limit", 50)), 500))
            except (ValueError, TypeError):
                limit = 50

            if memory_store is None:
                return ApiResponse.ok({"turns": []})

            try:
                turns = memory_store.get_session_turns(session_id, limit=limit)
            except Exception as exc:
                logger.warning("Memory store error: %s", exc)
                return ApiResponse.error("Memory store unavailable", 503)

            return ApiResponse.ok(
                {
                    "turns": [
                        {
                            "id": t.id,
                            "role": t.role,
                            "content": t.content,
                            "timestamp": t.timestamp,
                            "provider": t.provider,
                            "pinned": bool(t.metadata.get("pinned")),
                        }
                        for t in turns
                    ]
                }
            )

        def _handle_delete_session(self, session_id: str) -> tuple[int, dict]:
            """DELETE /api/v1/sessions/{id} — end a session."""
            if not session_registry.delete(session_id):
                return ApiResponse.error(f"Session '{session_id}' not found", 404)
            return ApiResponse.ok({"deleted": session_id})

        def _handle_memory_search(self, params: dict) -> tuple[int, dict]:
            """GET /api/v1/memory/search?q=query — full-text search memory.

            Query parameters:
                q (str, required): FTS5 query string.
                limit (int, default 10): Maximum results.
                session_id (str, optional): Scope search to a session.

            Response data fields:
                results (list[dict]): Matching conversation turns.
            """
            query = (params.get("q") or "").strip()
            if not query:
                return ApiResponse.error("Query parameter 'q' is required", 400)

            try:
                limit = max(1, min(int(params.get("limit", 10)), 100))
            except (ValueError, TypeError):
                limit = 10

            search_session_id: str | None = params.get("session_id") or None

            if memory_store is None:
                return ApiResponse.ok({"results": []})

            try:
                turns = memory_store.search(query, limit=limit, session_id=search_session_id)
            except Exception as exc:
                logger.warning("Memory search error: %s", exc)
                return ApiResponse.error("Memory search failed", 503)

            return ApiResponse.ok(
                {
                    "results": [
                        {
                            "id": t.id,
                            "role": t.role,
                            "content": t.content,
                            "timestamp": t.timestamp,
                            "session_id": t.session_id,
                            "provider": t.provider,
                            "pinned": bool(t.metadata.get("pinned")),
                        }
                        for t in turns
                    ]
                }
            )

        def _handle_delete_memory_turn(self, turn_id: str) -> tuple[int, dict]:
            """DELETE /api/v1/memory/turns/{id} — permanently delete a turn."""
            if memory_store is None:
                return ApiResponse.error("Memory store unavailable", 503)
            try:
                deleted = memory_store.delete_turn(turn_id)
            except Exception as exc:
                logger.warning("Memory delete error: %s", exc)
                return ApiResponse.error("Memory store unavailable", 503)
            result = "allow" if deleted else "deny"
            self._emit_web_audit(
                event_type="web.memory",
                result=result,
                action="memory.delete_turn",
                subsystem="memory",
                severity="info" if deleted else "warning",
                turn_id=turn_id,
            )
            if not deleted:
                return ApiResponse.error(f"Turn '{turn_id}' not found", 404)
            return ApiResponse.ok({"deleted": turn_id})

        def _handle_pin_memory_turn(self, turn_id: str, body: dict) -> tuple[int, dict]:
            """POST /api/v1/memory/turns/{id}/pin — pin or unpin a turn.

            Request body fields:
                pinned (bool, default True): Desired pinned state.
            """
            if memory_store is None:
                return ApiResponse.error("Memory store unavailable", 503)
            pinned = bool(body.get("pinned", True))
            try:
                updated = memory_store.set_turn_pinned(turn_id, pinned)
            except Exception as exc:
                logger.warning("Memory pin error: %s", exc)
                return ApiResponse.error("Memory store unavailable", 503)
            result = "allow" if updated else "deny"
            self._emit_web_audit(
                event_type="web.memory",
                result=result,
                action="memory.set_pinned",
                subsystem="memory",
                severity="info",
                turn_id=turn_id,
                pinned=pinned,
            )
            if not updated:
                return ApiResponse.error(f"Turn '{turn_id}' not found", 404)
            return ApiResponse.ok({"turn_id": turn_id, "pinned": pinned})

        def _handle_list_scheduled_jobs(self) -> tuple[int, dict]:
            """GET /api/v1/scheduler/jobs — list scheduled jobs in full detail."""
            scheduler = getattr(runtime, "_scheduler", None) if runtime is not None else None
            if scheduler is None:
                return ApiResponse.ok({"jobs": []})
            try:
                jobs = list(scheduler.list_jobs())
            except Exception as exc:
                logger.warning("Scheduler list error: %s", exc)
                return ApiResponse.error("Scheduler unavailable", 503)
            return ApiResponse.ok(
                {
                    "jobs": [
                        job.to_dict() if hasattr(job, "to_dict") else _job_fallback_dict(job)
                        for job in jobs
                    ]
                }
            )

        def _handle_create_scheduled_job(self, body: dict) -> tuple[int, dict]:
            """POST /api/v1/scheduler/jobs — create a new scheduled job.

            Request body fields:
                name (str, required): Human-readable job name.
                schedule (str, required): Human-readable schedule string.
                task (str, required): Prompt/task text run when the job fires.
                provider (str, optional): AI provider name.
                description (str, optional): Job description.
                active_hours (str, optional): ``"HH:MM-HH:MM"`` active window.
                timezone (str, optional): IANA timezone string.
            """
            scheduler = getattr(runtime, "_scheduler", None) if runtime is not None else None
            if scheduler is None:
                return ApiResponse.error("Scheduler unavailable", 503)

            name = str(body.get("name") or "").strip()[:128]
            schedule = str(body.get("schedule") or "").strip()[:256]
            task = str(body.get("task") or "").strip()
            provider = str(body.get("provider") or "anthropic").strip()[:64]
            description = str(body.get("description") or "")[:512]
            active_hours = str(body.get("active_hours") or "")[:32]
            timezone = str(body.get("timezone") or "")[:64]

            if not name or not schedule or not task:
                self._emit_web_audit(
                    event_type="web.scheduler",
                    result="deny",
                    action="scheduler.job.add",
                    subsystem="scheduler",
                    severity="warning",
                    reason="missing_required_field",
                )
                return ApiResponse.error("name, schedule, and task are required", 400)

            try:
                job = scheduler.add_job(
                    name,
                    schedule,
                    task,
                    provider=provider,
                    description=description,
                    active_hours=active_hours,
                    timezone=timezone,
                )
            except ValueError as exc:
                self._emit_web_audit(
                    event_type="web.scheduler",
                    result="deny",
                    action="scheduler.job.add",
                    subsystem="scheduler",
                    severity="warning",
                    reason="invalid_schedule",
                    error=str(exc)[:240],
                )
                return ApiResponse.error(str(exc), 400)
            except Exception as exc:
                logger.warning("Scheduler add_job error: %s", exc)
                self._emit_web_audit(
                    event_type="web.scheduler",
                    result="error",
                    action="scheduler.job.add",
                    subsystem="scheduler",
                    severity="warning",
                    reason="scheduler_error",
                )
                return ApiResponse.error("Failed to create scheduled job", 503)

            self._emit_web_audit(
                event_type="web.scheduler",
                result="allow",
                action="scheduler.job.add",
                subsystem="scheduler",
                severity="info",
                job_id=job.id,
                name=name,
                schedule=schedule,
            )
            return ApiResponse.created(job.to_dict() if hasattr(job, "to_dict") else {"id": job.id})

        def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
            logger.debug("api: " + format, *args)

    return ApiHandler


# ---------------------------------------------------------------------------
# Stale-IP eviction helper (called under rate_lock)
# ---------------------------------------------------------------------------


def _job_fallback_dict(job: Any) -> dict[str, Any]:
    """Best-effort serialization for job-like objects without ``to_dict``."""
    return {
        "id": str(getattr(job, "id", "")),
        "name": str(getattr(job, "name", "")),
        "schedule": str(getattr(job, "schedule", "")),
        "provider": str(getattr(job, "provider", "")),
        "enabled": bool(getattr(job, "enabled", False)),
    }


def _evict_stale(tracker: dict, cutoff: float) -> None:
    stale = [ip for ip, ts in tracker.items() if not ts or all(t <= cutoff for t in ts)]
    for ip in stale:
        del tracker[ip]


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ApiServer:
    """Lifecycle manager for the Missy REST API HTTP server.

    The server runs in a background daemon thread so it does not block the
    calling process.  All handler dependencies are injected via constructor
    arguments and captured in a closure; the :class:`ThreadingHTTPServer` never needs
    to be subclassed.

    Args:
        config: :class:`ApiConfig` instance controlling bind address, port,
            API key, and rate limits.
        runtime: Optional :class:`~missy.agent.runtime.AgentRuntime` to
            power the ``/chat`` endpoint.  When ``None``, chat returns
            ``503``.
        memory_store: Optional :class:`~missy.memory.sqlite_store.SQLiteMemoryStore`
            for history and search endpoints.
        provider_registry: Optional :class:`~missy.providers.registry.ProviderRegistry`
            for the ``/providers`` and ``/status`` endpoints.
        tool_registry: Optional :class:`~missy.tools.registry.ToolRegistry` for
            the ``/tools`` endpoint.
        candidate_store: Optional tool candidate lifecycle store for
            ``/tool-candidates`` and candidate operator controls.
        benchmark_store: Optional benchmark result store used by the candidate
            benchmark-import operator control.
    """

    def __init__(
        self,
        config: ApiConfig,
        runtime: AgentRuntime | None = None,
        memory_store: SQLiteMemoryStore | None = None,
        provider_registry: ProviderRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        candidate_store: CandidateStore | None = None,
        benchmark_store: BenchmarkStore | None = None,
        approval_gate: Any | None = None,
        discord_channels: list | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.memory_store = memory_store
        self.provider_registry = provider_registry
        self.tool_registry = tool_registry
        self.candidate_store = candidate_store
        self.benchmark_store = benchmark_store
        self.approval_gate = approval_gate
        # SR-1.12/task #12: shared reference to the live DiscordChannel
        # list. Channels are constructed later (async, after the API
        # server starts) so this is the *same* list object the caller
        # keeps appending to -- read lazily at request time rather than
        # snapshotted here, since it may still be empty at __init__ time.
        self.discord_channels = discord_channels

        self._session_registry = _SessionRegistry()
        self._web_sessions = WebSessionStore(config.web_session_ttl_seconds)
        self._run_registry = RunRegistry()
        self._rate_tracker: dict[str, list[float]] = {}
        self._rate_lock = threading.Lock()

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the API server in a background daemon thread.

        The server is ready to accept connections by the time this method
        returns.

        Raises:
            RuntimeError: If the server is already running.
            OSError: If the port is not available.
        """
        if self._server is not None:
            raise RuntimeError("ApiServer is already running.")

        if not self.config.api_key:
            logger.warning(
                "ApiServer: api_key is empty — every request will be rejected with 401. "
                "Set ApiConfig.api_key to enable the API."
            )

        if self.config.host not in ("127.0.0.1", "::1", "localhost"):
            logger.warning(
                "ApiServer: binding to %s (non-loopback) — ensure firewall rules "
                "restrict access appropriately.",
                self.config.host,
            )

        handler_class = _make_handler(
            api_config=self.config,
            session_registry=self._session_registry,
            web_sessions=self._web_sessions,
            rate_tracker=self._rate_tracker,
            rate_lock=self._rate_lock,
            runtime=self.runtime,
            memory_store=self.memory_store,
            provider_registry=self.provider_registry,
            tool_registry=self.tool_registry,
            candidate_store=self.candidate_store,
            benchmark_store=self.benchmark_store,
            run_registry=self._run_registry,
            approval_gate=self.approval_gate,
            discord_channels=self.discord_channels,
        )

        self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler_class)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="missy-api",
        )
        self._thread.start()
        logger.info("ApiServer listening on %s:%d", self.config.host, self.config.port)

    def stop(self, timeout: float = 5.0) -> None:
        """Shut down the API server and wait for the background thread.

        Args:
            timeout: Maximum seconds to wait for the thread to exit.
        """
        server = self._server
        thread = self._thread
        if server is not None:
            server.shutdown()
            self._server = None
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None
        logger.info("ApiServer stopped.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return ``True`` while the server thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def url(self) -> str:
        """Return the base URL of the API server."""
        return f"http://{self.config.host}:{self.config.port}"
