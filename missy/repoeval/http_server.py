"""Opt-in stdlib HTTP transport for a single, explicitly scoped Foundry project.

Constructing a server does not start a thread. The caller owns the server lifetime,
credential source, network isolation, and (if necessary) TLS-terminating proxy.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from missy.repoeval.api import FoundryAPI, Response

MAX_HEADER_BYTES = 16 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
_PROJECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RESOURCE_ID = r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}"
_TOKEN = re.compile(r"[\x21-\x7e]{1,256}\Z")
_IDEMPOTENCY = re.compile(r"[\x21-\x7e]{1,128}\Z")
_PATHS = (
    ("GET", r""),
    ("GET", r"/repositories"),
    ("POST", r"/snapshots"),
    ("GET", rf"/snapshots/{_RESOURCE_ID}"),
    ("POST", r"/benchmark/plan"),
    ("POST", r"/benchmark/start"),
    ("GET", rf"/runs/{_RESOURCE_ID}"),
    ("POST", rf"/runs/{_RESOURCE_ID}/cancel"),
    ("GET", rf"/runs/{_RESOURCE_ID}/artifacts"),
    ("POST", r"/compare"),
    ("POST", r"/report"),
)


def _parse_content_length(value: str) -> int | None:
    """Parse canonical bounded decimal framing, without converting attacker-sized input."""
    # Any valid value fits in this many decimal digits. Reject longer strings
    # before scanning them or calling int(), including strings of many 9s.
    if not value or len(value) > len(str(MAX_REQUEST_BYTES)):
        return None
    if not value.isascii() or not value.isdigit():
        return None
    if len(value) > 1 and value[0] == "0":
        return None
    size = int(value)
    return size if size <= MAX_REQUEST_BYTES else None


@dataclass(frozen=True)
class ExternalTLSProxyContract:
    """Operator assertion, not proof, of externally enforced network controls.

    A non-loopback listener is reachable without TLS or proxy authentication if
    the asserted backend isolation is not actually enforced. Never use this as
    a substitute for a firewall/network namespace and an authenticated proxy.
    """

    tls_terminated: bool
    proxy_authentication_enforced: bool
    backend_access_restricted_to_proxy: bool


class CredentialVerifier:
    """Explicitly injected credentials mapped to project-scoped principals.

    Keep only hashes after construction. Compare every stored fixed-size digest,
    including after a match, so token equality checks do not short-circuit.
    This is not a claim of whole-request constant-time behavior.
    """

    def __init__(self, credentials: Mapping[str, Any]):
        if not credentials:
            raise ValueError("At least one explicit credential is required")
        entries = []
        for token, principal in credentials.items():
            if not isinstance(token, str) or not _TOKEN.fullmatch(token) or " " in token:
                raise ValueError("Invalid credential format")
            if not getattr(principal, "subject", None) or not getattr(
                principal, "project_id", None
            ):
                raise ValueError("A credential must map to a scoped principal")
            entries.append((hashlib.sha256(token.encode("ascii")).digest(), principal))
        self._entries = tuple(entries)

    def verify(self, token: str) -> Any | None:
        if not _TOKEN.fullmatch(token) or " " in token:
            return None
        digest = hashlib.sha256(token.encode("ascii")).digest()
        principal = None
        for stored_digest, candidate in self._entries:
            if hmac.compare_digest(digest, stored_digest):
                principal = candidate
        return principal


def _error(status: int, category: str, message: str) -> Response:
    return Response(
        status,
        {"ok": False, "error": {"category": category, "message": message, "retriable": False}},
    )


class FoundryHTTPServer(ThreadingHTTPServer):
    """One project, one injected service and verifier; no implicit serve_forever."""

    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        service: Any,
        project_id: str,
        credential_verifier: CredentialVerifier,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        *,
        external_tls_proxy_contract: ExternalTLSProxyContract | None = None,
    ) -> None:
        if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
            raise ValueError("Invalid project ID")
        if not isinstance(credential_verifier, CredentialVerifier):
            raise TypeError("An explicitly configured CredentialVerifier is required")
        if any(p.project_id != project_id for _, p in credential_verifier._entries):
            raise ValueError("Every credential must belong to this listener's project")
        try:
            ip = ipaddress.ip_address(bind_host)
        except ValueError as exc:
            raise ValueError("Bind host must be an explicit IP address") from exc
        if ip.version != 4:
            raise ValueError("This listener supports IPv4 only")
        if not ip.is_loopback and external_tls_proxy_contract != ExternalTLSProxyContract(
            tls_terminated=True,
            proxy_authentication_enforced=True,
            backend_access_restricted_to_proxy=True,
        ):
            raise ValueError("Non-loopback bind requires an authenticated TLS proxy contract")
        self.project_id = project_id
        self.credential_verifier = credential_verifier
        self.service = service
        # These are the only two prefixes generated by RepoevalFoundryTool.
        # Validate the complete tail below rather than trusting startswith as
        # a project boundary (e.g. project-foreign must never match project).
        self._path_prefixes = (f"/api/projects/{project_id}", f"/projects/{project_id}")
        super().__init__((bind_host, port), _FoundryRequestHandler)

    def _authenticate(self, headers: Mapping[str, str]) -> Any | None:
        value = headers.get("Authorization", "")
        if not value.startswith("Bearer "):
            return None
        return self.credential_verifier.verify(value[7:])


class _FoundryRequestHandler(BaseHTTPRequestHandler):
    server: FoundryHTTPServer
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        self.request.settimeout(5)
        super().setup()

    def log_message(self, format: str, *args: object) -> None:
        # The default handler logs the raw URL. Never log credentials or request data.
        pass

    def handle_expect_100(self) -> bool:
        self.close_connection = True
        self._send(_error(417, "invalid_request", "Expect is not supported"))
        return False

    def parse_request(self) -> bool:
        # stdlib's default accepts RFC 7230 obs-fold and silently discards
        # malformed lines. The authorization/framing parser must not disagree
        # with a proxy over which bytes constitute a header.
        parsed = super().parse_request()
        if parsed and self._raw_headers_invalid():
            self.close_connection = True
            self._send(_error(400, "invalid_request", "Invalid request headers"))
            return False
        return parsed

    def _raw_headers_invalid(self) -> bool:
        raw = self.headers.as_bytes()
        return (
            len(self.raw_requestline) + len(raw) > MAX_HEADER_BYTES
            or b"\n " in raw
            or b"\n\t" in raw
            or any(b":" not in line for line in raw.split(b"\n") if line.strip())
        )

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # BaseHTTPRequestHandler otherwise includes a potentially secret-bearing
        # request method/path in its generated error page.
        self.close_connection = True
        self._send(_error(405, "method_not_allowed", "HTTP method is not supported"))

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_HEAD(self) -> None:
        self._reject_method()

    def do_OPTIONS(self) -> None:
        self._reject_method()

    def do_PUT(self) -> None:
        self._reject_method()

    def do_PATCH(self) -> None:
        self._reject_method()

    def do_DELETE(self) -> None:
        self._reject_method()

    def do_TRACE(self) -> None:
        self._reject_method()

    def do_CONNECT(self) -> None:
        self._reject_method()

    def _reject_method(self) -> None:
        self.close_connection = True
        self._send(_error(405, "method_not_allowed", "HTTP method is not supported"))

    def _send(self, response: Response) -> None:
        try:
            payload = json.dumps(response.body, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError("Oversize response")
        except (ValueError, TypeError, UnicodeError, RecursionError):
            response = _error(503, "service_failure", "Application service failed")
            payload = json.dumps(response.body, separators=(",", ":")).encode("utf-8")
        self.send_response(response.status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def _handle(self, method: str) -> None:
        self.close_connection = True
        # Refuse malformed/duplicated framing and auth headers before dispatch.
        if (
            self._raw_headers_invalid()
            or len(self.headers.get_all("Host", [])) != 1
            or len(self.headers.get_all("Authorization", [])) > 1
            or len(self.headers.get_all("Content-Length", [])) > 1
            or any(
                not v.isascii() or any(ord(c) < 32 or ord(c) > 126 for c in v)
                for v in self.headers.values()
            )
            or self.headers.get("Transfer-Encoding") is not None
            or self.headers.get("Expect") is not None
            or self.headers.get("Upgrade") is not None
        ):
            self._send(_error(400, "invalid_request", "Invalid request headers"))
            return
        # Authenticate first, but never pass the raw header or token to the service.
        principal = self.server._authenticate(
            {"Authorization": self.headers.get("Authorization", "")}
        )
        if principal is None:
            self._send(_error(401, "unauthenticated", "Authentication required"))
            return
        prefix = next((p for p in self.server._path_prefixes if self.path.startswith(p)), None)
        if prefix is None:
            self._send(_error(404, "not_found", "Unknown API route"))
            return
        tail = self.path[len(prefix) :]
        if not any(op == method and re.fullmatch(pattern, tail) for op, pattern in _PATHS):
            self._send(_error(404, "not_found", "Unknown API route"))
            return
        lengths = self.headers.get_all("Content-Length", [])
        if method == "POST" and not lengths:
            self._send(_error(411, "invalid_request", "Content-Length is required"))
            return
        length = lengths[0] if lengths else "0"
        size = _parse_content_length(length)
        if size is None:
            self._send(_error(413, "invalid_request", "Invalid or oversized Content-Length"))
            return
        if method == "GET" and size:
            self._send(_error(400, "invalid_request", "GET body is not supported"))
            return
        if method == "POST" and (
            not size
            or len(self.headers.get_all("Content-Type", [])) != 1
            or self.headers["Content-Type"].lower() != "application/json"
        ):
            self._send(_error(415, "invalid_request", "JSON request body is required"))
            return
        body: dict[str, Any] = {}
        if size:
            try:
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete request")
                data = json.loads(raw.decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("JSON object required")
                body = data
            except (ValueError, UnicodeError, TimeoutError, RecursionError):
                self._send(_error(400, "invalid_request", "Invalid JSON request body"))
                return
        # Forward only the validated project route, principal, and required
        # idempotency header. The facade remains the operation authorization gate.
        headers: dict[str, str] = {}
        if len(self.headers.get_all("Idempotency-Key", [])) > 1 or (
            "Idempotency-Key" in self.headers
            and not _IDEMPOTENCY.fullmatch(self.headers["Idempotency-Key"])
        ):
            self._send(_error(400, "invalid_request", "Invalid idempotency header"))
            return
        if "Idempotency-Key" in self.headers:
            headers["Idempotency-Key"] = self.headers["Idempotency-Key"]
        # The wire credential must not reach the service-facing route facade.
        request_api = FoundryAPI(self.server.service, lambda _: principal)
        self._send(request_api.handle(method, self.path, headers, body))
