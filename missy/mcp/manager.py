"""MCP server lifecycle manager."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import stat
import threading
from pathlib import Path

from missy.mcp.client import McpClient

logger = logging.getLogger(__name__)

MCP_CONFIG_PATH = "~/.missy/mcp.json"

#: Tool names may only contain alphanumeric characters, hyphens, and underscores.
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


class McpManager:
    """Manages MCP server connections and exposes their tools to the agent.

    Config format (~/.missy/mcp.json)::

        [
            {"name": "filesystem", "command": "npx @modelcontextprotocol/server-filesystem /tmp"},
            {"name": "postgres", "command": "npx @modelcontextprotocol/server-postgres postgresql://..."}
        ]
    """

    def __init__(
        self,
        config_path: str = MCP_CONFIG_PATH,
        block_injection: bool = True,
    ):
        self._config_path = Path(config_path).expanduser()
        self._clients: dict[str, McpClient] = {}
        self._lock = threading.Lock()
        self._block_injection = block_injection

    def connect_all(self) -> None:
        """Load config and connect to all configured MCP servers."""
        if not self._config_path.exists():
            logger.debug("No MCP config at %s; skipping", self._config_path)
            return
        # Security: verify file permissions before loading
        try:
            st = self._config_path.stat()
            if st.st_uid != os.getuid():
                logger.warning(
                    "MCP config %s owned by uid %d, expected %d; refusing to load",
                    self._config_path,
                    st.st_uid,
                    os.getuid(),
                )
                return
            if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                logger.warning(
                    "MCP config %s is group/world-writable (mode %o); refusing to load",
                    self._config_path,
                    st.st_mode,
                )
                return
        except OSError as exc:
            logger.warning("MCP: cannot stat config %s: %s", self._config_path, exc)
            return
        try:
            servers = json.loads(self._config_path.read_text())
        except Exception as exc:
            logger.warning("MCP config parse error: %s", exc)
            return
        for entry in servers:
            name = entry.get("name", "unknown")
            try:
                self.add_server(name, command=entry.get("command"), url=entry.get("url"))
            except Exception as exc:
                logger.warning("MCP: failed to connect %r: %s", name, exc)

    def add_server(
        self, name: str, command: str | None = None, url: str | None = None
    ) -> McpClient:
        """Connect to a new MCP server and persist the config.

        If the config entry for this server has a ``"digest"`` key, the
        tool manifest digest is verified after connection.  A mismatch
        causes the server to be disconnected and an error to be raised.
        """
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid MCP server name: {name!r} "
                "(must contain only alphanumeric, hyphens, underscores)"
            )
        if "__" in name:
            raise ValueError(f"Invalid MCP server name: {name!r} (must not contain '__')")
        client = McpClient(name=name, command=command, url=url)
        client.connect()

        # Digest verification (Feature 3)
        expected_digest = self._get_server_digest(name)
        if expected_digest is not None:
            from missy.mcp.digest import compute_tool_manifest_digest, verify_digest

            actual_digest = compute_tool_manifest_digest(client.tools)
            if not verify_digest(expected_digest, actual_digest):
                client.disconnect()
                logger.warning(
                    "MCP: digest mismatch for %r: expected=%s actual=%s",
                    name,
                    expected_digest,
                    actual_digest,
                )
                try:
                    from missy.core.events import AuditEvent, event_bus

                    event_bus.publish(
                        AuditEvent.now(
                            session_id="",
                            task_id="",
                            event_type="mcp.digest_mismatch",
                            category="security",
                            result="deny",
                            detail={
                                "server": name,
                                "expected": expected_digest,
                                "actual": actual_digest,
                            },
                        )
                    )
                except Exception:
                    pass
                raise ValueError(
                    f"MCP server {name!r} tool manifest digest mismatch: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            logger.info("MCP: digest verified for %r", name)
        else:
            logger.debug(
                "MCP: no digest pinned for %r — consider running 'missy mcp pin %s'",
                name,
                name,
            )

        with self._lock:
            self._clients[name] = client
        self._save_config()
        logger.info("MCP: connected to %r (%d tools)", name, len(client.tools))
        return client

    def _get_server_digest(self, name: str) -> str | None:
        """Return the pinned digest for server *name* from the config file, or None."""
        if not self._config_path.exists():
            return None
        try:
            servers = json.loads(self._config_path.read_text())
            for entry in servers:
                if entry.get("name") == name:
                    return entry.get("digest")
        except Exception:
            pass
        return None

    def pin_server_digest(self, name: str) -> str:
        """Compute and persist the digest for a connected server.

        Args:
            name: Name of the MCP server (must be connected).

        Returns:
            The computed digest string.

        Raises:
            KeyError: If the server is not connected.
        """
        with self._lock:
            client = self._clients.get(name)
        if client is None:
            raise KeyError(f"MCP server {name!r} is not connected.")

        from missy.mcp.digest import compute_tool_manifest_digest

        digest = compute_tool_manifest_digest(client.tools)

        # Update digest in config file
        if self._config_path.exists():
            try:
                servers = json.loads(self._config_path.read_text())
                for entry in servers:
                    if entry.get("name") == name:
                        entry["digest"] = digest
                        break
                self._config_path.write_text(json.dumps(servers, indent=2))
            except Exception as exc:
                logger.warning("MCP: failed to persist digest for %r: %s", name, exc)

        return digest

    def remove_server(self, name: str) -> None:
        with self._lock:
            client = self._clients.pop(name, None)
        if client:
            client.disconnect()
            self._save_config()

    def restart_server(self, name: str) -> None:
        with self._lock:
            client = self._clients.get(name)
        if client:
            cmd = client._command
            url = client._url
            client.disconnect()
            new_client = McpClient(name=name, command=cmd, url=url)
            new_client.connect()
            with self._lock:
                self._clients[name] = new_client

    def health_check(self) -> None:
        """Restart any dead MCP servers."""
        with self._lock:
            dead = [n for n, c in self._clients.items() if not c.is_alive()]
        for name in dead:
            logger.warning("MCP: %r is dead; restarting", name)
            try:
                self.restart_server(name)
            except Exception as exc:
                logger.error("MCP: failed to restart %r: %s", name, exc)

    def all_tools(self) -> list[dict]:
        """Return all tool definitions from all connected servers, namespaced."""
        tools = []
        with self._lock:
            for server_name, client in self._clients.items():
                for tool in client.tools:
                    namespaced = dict(tool)
                    namespaced["name"] = f"{server_name}__{tool['name']}"
                    namespaced["_mcp_server"] = server_name
                    namespaced["_mcp_tool"] = tool["name"]
                    tools.append(namespaced)
        return tools

    def call_tool(self, namespaced_name: str, arguments: dict) -> str:
        """Call an MCP tool by its namespaced name (server__tool)."""
        if "__" not in namespaced_name:
            return f"[MCP error] invalid tool name: {namespaced_name}"
        server_name, tool_name = namespaced_name.split("__", 1)
        # Validate tool name characters to prevent injection via crafted names.
        if not _SAFE_NAME_RE.match(tool_name):
            return f"[MCP error] unsafe tool name: {tool_name!r}"
        with self._lock:
            client = self._clients.get(server_name)
        if not client:
            return f"[MCP error] server {server_name!r} not connected"
        result = client.call_tool(tool_name, arguments)
        # Defense-in-depth: scan MCP tool results for prompt injection.
        try:
            from missy.security.sanitizer import InputSanitizer

            warnings = InputSanitizer().check_for_injection(result)
            if warnings:
                logger.warning(
                    "MCP tool %r returned content with injection patterns: %s",
                    namespaced_name,
                    warnings,
                )
                if getattr(self, "_block_injection", False):
                    return (
                        f"[MCP BLOCKED] Tool {namespaced_name!r} output contained "
                        f"injection patterns and was blocked: {warnings}"
                    )
                result = f"[SECURITY WARNING: MCP tool output may contain injection] {result}"
        except Exception:
            logger.debug("MCP injection scan failed; tool output passed through", exc_info=True)
        return result

    def list_servers(self) -> list[dict]:
        with self._lock:
            return [
                {"name": n, "alive": c.is_alive(), "tools": len(c.tools)}
                for n, c in self._clients.items()
            ]

    def shutdown(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
        for c in clients:
            with contextlib.suppress(Exception):
                c.disconnect()

    def _save_config(self) -> None:
        with self._lock:
            entries = [
                {"name": name, "command": c._command, "url": c._url}
                for name, c in self._clients.items()
            ]
        self._config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Write with restrictive permissions (owner read/write only) to
        # prevent other users from reading server commands or URLs.
        import os
        import tempfile

        data = json.dumps(entries, indent=2)
        dir_path = str(self._config_path.parent)
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        closed = False
        try:
            os.write(fd, data.encode())
            os.fchmod(fd, 0o600)
            os.close(fd)
            closed = True
            os.replace(tmp_path, str(self._config_path))
        except Exception:
            if not closed:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
