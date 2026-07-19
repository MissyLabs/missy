"""MCP client: stdio subprocess and HTTP transports."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import subprocess
import threading
import time
import uuid
from typing import Any

from missy.mcp.annotations import ToolAnnotation

logger = logging.getLogger(__name__)


class McpClient:
    """JSON-RPC client for a single MCP server.

    Supports stdio (subprocess) and HTTP (JSON-RPC over HTTP POST, incl. the
    ``text/event-stream`` response form) transports. HTTP servers may require
    authentication (F17): an ``Authorization: Bearer <token>`` header (or
    arbitrary custom headers) is sent with every request.

    Args:
        name: Human-readable server name.
        command: Shell command to launch the server (for stdio transport).
        url: HTTP endpoint (for HTTP transport).
        headers: Extra HTTP headers (e.g. ``{"Authorization": "Bearer …"}``)
            sent on every HTTP request. Ignored for stdio transport.
    """

    def __init__(
        self,
        name: str,
        command: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self._command = command
        self._url = url
        self._headers = dict(headers or {})
        self._proc: subprocess.Popen | None = None
        self._http: Any = None  # lazily-built httpx.Client for HTTP transport
        self._session_id: str | None = None  # MCP-Session-Id echoed by server
        self._lock = threading.Lock()
        self._tools: list[dict] = []
        self._tool_annotations: dict[str, ToolAnnotation] = {}

    def connect(self) -> None:
        """Start the MCP server process and perform the initialize handshake."""
        if self._command:
            import os
            import shlex

            # Pass only safe environment variables to prevent secret leakage
            _SAFE_VARS = (
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "TERM",
                "XDG_RUNTIME_DIR",
                "TMPDIR",
                "TMP",
                "TEMP",
                "NODE_PATH",
                "NPM_CONFIG_PREFIX",
            )
            env = {k: os.environ[k] for k in _SAFE_VARS if k in os.environ}

            self._proc = subprocess.Popen(
                shlex.split(self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            # Verify the process is alive before attempting handshake
            if self._proc.poll() is not None:
                stderr = (
                    self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                )
                raise RuntimeError(
                    f"MCP server process exited immediately with code {self._proc.returncode}: {stderr[:500]}"
                )
            self._initialize()
        elif self._url:
            import httpx

            self._http = httpx.Client(
                timeout=30.0,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    **self._headers,
                },
            )
            self._initialize()
        else:
            raise RuntimeError("MCP client requires either a command or a url")

    def _initialize(self) -> None:
        resp = self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "missy", "version": "0.1.0"},
            },
        )
        if resp.get("error"):
            raise RuntimeError(f"MCP init failed: {resp['error']}")
        self._notify("notifications/initialized")
        self._tools = self._list_tools()

    # Maximum size for a single MCP response line (1 MB).
    _MAX_RESPONSE_BYTES = 1024 * 1024

    def _rpc(self, method: str, params: Any = None, *, timeout: float = 30.0) -> dict:
        req_id = str(uuid.uuid4())
        request = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            request["params"] = params
        if self._url:
            return self._http_rpc(request, req_id, timeout=timeout)
        line = json.dumps(request) + "\n"
        with self._lock:
            if not self._proc or self._proc.stdin is None:
                raise RuntimeError("MCP server not connected")
            self._proc.stdin.write(line.encode())
            self._proc.stdin.flush()
            # Use a deadline-bounded read when a real file descriptor is
            # available; fall back to direct read for mock/pipe objects
            # without fileno().
            try:
                # Availability hardening: a bare select()-then-readline()
                # here previously only proved *some* bytes were available
                # before handing off to an un-timed readline() -- a stalled
                # or malicious server that writes a partial response and
                # never sends the trailing newline caused readline() to
                # block indefinitely, still holding self._lock, with no way
                # back (the process is still alive per is_alive()/poll(),
                # so McpManager's health_check() auto-recovery never kicks
                # in). Live reproduced: a real subprocess writing a valid
                # JSON prefix with no trailing newline, then stalling, left
                # a call with timeout=1.0 still blocked 5+ seconds later.
                # _read_line_with_deadline() bounds the ENTIRE wait (both
                # the "nothing arrived yet" and "a partial line arrived,
                # then stalled" cases) within one deadline computed from
                # *timeout*, raising TimeoutError and tearing the
                # connection down (same response-stream-desync rationale
                # as before) if it's ever exceeded.
                response_line = self._read_line_with_deadline(timeout)
            except (TypeError, ValueError, AttributeError):
                # fileno not available (e.g. mock/pipe in tests) — the
                # deadline-bounded read requires a real fileno for
                # select(); fall back to the legacy un-timed readline()
                # for compatibility with such streams.
                response_line = self._proc.stdout.readline(self._MAX_RESPONSE_BYTES)
        if not response_line:
            raise RuntimeError("MCP server closed connection")
        resp = json.loads(response_line)
        # Validate response ID matches request to prevent response confusion
        resp_id = resp.get("id")
        if resp_id is not None and resp_id != req_id:
            raise RuntimeError(f"MCP response ID mismatch: expected {req_id}, got {resp_id}")
        return resp

    def _http_rpc(self, request: dict, req_id: str, *, timeout: float = 30.0) -> dict:
        """Send one JSON-RPC request over HTTP POST and return the response.

        Handles both response forms the MCP Streamable HTTP transport permits:
        a plain ``application/json`` body, or a ``text/event-stream`` (SSE)
        body whose ``data:`` line carries the JSON-RPC response. Auth headers
        (F17) were installed on the client at connect time. An ``MCP-Session-Id``
        returned by the server is captured and echoed on subsequent requests.
        """
        with self._lock:
            if self._http is None:
                raise RuntimeError("MCP HTTP client not connected")
            headers = {}
            if self._session_id:
                headers["MCP-Session-Id"] = self._session_id
            resp = self._http.post(self._url, json=request, headers=headers, timeout=timeout)
            if resp.status_code == 401 or resp.status_code == 403:
                raise RuntimeError(
                    f"MCP HTTP auth failed ({resp.status_code}) for {self.name!r} — "
                    "check the server's bearer token / credentials."
                )
            resp.raise_for_status()
            session = resp.headers.get("MCP-Session-Id")
            if session:
                self._session_id = session
            body = self._parse_http_body(resp)
        resp_id = body.get("id")
        if resp_id is not None and resp_id != req_id:
            raise RuntimeError(f"MCP response ID mismatch: expected {req_id}, got {resp_id}")
        return body

    @staticmethod
    def _parse_http_body(resp: Any) -> dict:
        """Extract a JSON-RPC object from a JSON or SSE HTTP response."""
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            # Parse SSE: the JSON-RPC message is on a `data:` line.
            for raw in resp.text.splitlines():
                line = raw.strip()
                if line.startswith("data:"):
                    payload = line[len("data:") :].strip()
                    if payload:
                        try:
                            return json.loads(payload)
                        except (ValueError, TypeError):
                            continue
            raise RuntimeError("MCP HTTP SSE response contained no data payload")
        return resp.json()

    def _read_line_with_deadline(self, timeout: float) -> bytes:
        """Read a single newline-terminated line from stdout without ever
        blocking past *timeout* total, counted from when this method is
        called.

        Raises:
            TimeoutError: If the deadline passes before a full line (or
                EOF) is observed. Tears the connection down first, for the
                same response-stream-desynchronization reason
                :meth:`_rpc`'s own timeout path does: once we stop reading
                partway through, this pipe can never be trusted again for
                a future call.
        """
        import select

        deadline = time.monotonic() + timeout
        buf = bytearray()
        while True:
            idx = buf.find(b"\n")
            if idx != -1:
                return bytes(buf[: idx + 1])
            if len(buf) >= self._MAX_RESPONSE_BYTES:
                return bytes(buf)
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self._proc.stdout], [], [], max(remaining, 0))[0]
            ):
                self._teardown_after_timeout()
                if buf:
                    raise TimeoutError(
                        f"MCP server {self.name!r} sent a partial response and then "
                        f"stalled past its {timeout}s timeout (connection has been torn "
                        "down to prevent response-stream desynchronization; it will be "
                        "restarted on next health check)"
                    )
                raise TimeoutError(
                    f"MCP server {self.name!r} did not respond within {timeout}s "
                    "(connection has been torn down to prevent response-stream "
                    "desynchronization; it will be restarted on next health check)"
                )
            # select() already confirmed data is available, so read1() (a
            # single, at-most-one-syscall read) returns immediately without
            # blocking further -- unlike readline(), which would keep
            # waiting for a newline that may never come.
            chunk = self._proc.stdout.read1(self._MAX_RESPONSE_BYTES - len(buf))
            if not chunk:
                return bytes(buf)  # EOF
            buf.extend(chunk)

    def _teardown_after_timeout(self) -> None:
        """Forcibly kill the subprocess after an RPC timeout.

        Called while already holding ``self._lock``. Uses ``kill()``
        directly rather than ``disconnect()``'s graceful
        ``terminate()``-then-``wait()`` sequence: the server already
        failed to respond to a request within its timeout budget, so
        waiting further for a graceful exit just delays the inevitable
        and risks reading yet another stale/unexpected byte from the
        same corrupted pipe before it closes.
        """
        if not self._proc:
            return
        with contextlib.suppress(Exception):
            self._proc.kill()
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=5)
        for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if pipe:
                with contextlib.suppress(OSError):
                    pipe.close()
        self._proc = None

    def _notify(self, method: str, params: Any = None) -> None:
        note = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            note["params"] = params
        if self._url and self._http is not None:
            headers = {"MCP-Session-Id": self._session_id} if self._session_id else {}
            with self._lock, contextlib.suppress(Exception):
                self._http.post(self._url, json=note, headers=headers)
            return
        if self._proc and self._proc.stdin:
            self._proc.stdin.write((json.dumps(note) + "\n").encode())
            self._proc.stdin.flush()

    #: Tool names from MCP servers must match this pattern.
    _SAFE_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

    def _list_tools(self) -> list[dict]:
        resp = self._rpc("tools/list")
        raw_tools = resp.get("result", {}).get("tools", [])
        # Validate tool names at import time to prevent namespace injection
        # and reject tools with potentially dangerous names.
        validated: list[dict] = []
        for tool in raw_tools:
            name = tool.get("name", "")
            if not name or not self._SAFE_TOOL_NAME_RE.match(name):
                logger.warning(
                    "MCP server %r: rejecting tool with invalid name %r",
                    self.name,
                    name,
                )
                continue
            if "__" in name:
                logger.warning(
                    "MCP server %r: rejecting tool name %r (contains '__')",
                    self.name,
                    name,
                )
                continue
            validated.append(tool)
            # Parse annotations if present in the tool manifest.
            ann_data = tool.get("annotations")
            if isinstance(ann_data, dict):
                try:
                    self._tool_annotations[name] = ToolAnnotation.from_mcp_dict(ann_data)
                    logger.debug("MCP server %r: parsed annotation for tool %r", self.name, name)
                except Exception:
                    logger.debug(
                        "MCP server %r: failed to parse annotation for tool %r",
                        self.name,
                        name,
                        exc_info=True,
                    )
        return validated

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on this MCP server and return the result as a string."""
        resp = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if resp.get("error"):
            return f"[MCP error] {resp['error']}"
        result = resp.get("result", {})
        content = result.get("content", [])
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else str(result)

    @property
    def tools(self) -> list[dict]:
        """Raw MCP tool definitions."""
        return self._tools

    @property
    def tool_annotations(self) -> dict[str, ToolAnnotation]:
        """Mapping of tool name to its parsed :class:`~missy.mcp.annotations.ToolAnnotation`.

        Only tools that carried an ``annotations`` key in the MCP manifest
        will have an entry here.  Use
        :class:`~missy.mcp.annotations.AnnotationRegistry.get_or_default` for
        a safe fallback.

        Returns:
            A shallow copy of the internal annotations dict.
        """
        return dict(self._tool_annotations)

    def is_alive(self) -> bool:
        # An HTTP transport has no local process; it's "alive" once its client
        # is built (health is re-verified per-call by the manager).
        if getattr(self, "_url", None):
            return getattr(self, "_http", None) is not None
        proc = getattr(self, "_proc", None)
        return proc is not None and proc.poll() is None

    def disconnect(self) -> None:
        # Defensive getattr: some callers/tests build the client via
        # ``McpClient.__new__`` and set only ``_proc`` without running __init__.
        http = getattr(self, "_http", None)
        if http is not None:
            with contextlib.suppress(Exception):
                http.close()
            self._http = None
            self._session_id = None
        if getattr(self, "_proc", None):
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            finally:
                # Explicitly close subprocess pipes to prevent fd leaks.
                for pipe in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                    if pipe:
                        with contextlib.suppress(OSError):
                            pipe.close()
            self._proc = None
