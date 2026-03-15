"""MCP client: stdio subprocess and HTTP transports."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class McpClient:
    """JSON-RPC client for a single MCP server.

    Supports stdio (subprocess) transport.

    Args:
        name: Human-readable server name.
        command: Shell command to launch the server (for stdio transport).
        url: HTTP endpoint (for HTTP transport; not yet implemented).
    """

    def __init__(self, name: str, command: str | None = None, url: str | None = None):
        self.name = name
        self._command = command
        self._url = url
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._tools: list[dict] = []

    def connect(self) -> None:
        """Start the MCP server process and perform the initialize handshake."""
        if self._command:
            import os
            import shlex

            # Pass only safe environment variables to prevent secret leakage
            _SAFE_VARS = (
                "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM",
                "XDG_RUNTIME_DIR", "TMPDIR", "TMP", "TEMP",
                "NODE_PATH", "NPM_CONFIG_PREFIX",
            )
            env = {k: os.environ[k] for k in _SAFE_VARS if k in os.environ}

            self._proc = subprocess.Popen(
                shlex.split(self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            self._initialize()
        else:
            raise NotImplementedError("HTTP MCP transport not yet implemented")

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

    def _rpc(self, method: str, params: Any = None) -> dict:
        request = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method}
        if params is not None:
            request["params"] = params
        line = json.dumps(request) + "\n"
        with self._lock:
            if not self._proc or self._proc.stdin is None:
                raise RuntimeError("MCP server not connected")
            self._proc.stdin.write(line.encode())
            self._proc.stdin.flush()
            response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed connection")
        return json.loads(response_line)

    def _notify(self, method: str, params: Any = None) -> None:
        note = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            note["params"] = params
        if self._proc and self._proc.stdin:
            self._proc.stdin.write((json.dumps(note) + "\n").encode())
            self._proc.stdin.flush()

    def _list_tools(self) -> list[dict]:
        resp = self._rpc("tools/list")
        return resp.get("result", {}).get("tools", [])

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

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def disconnect(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
