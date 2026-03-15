"""MCP server lifecycle manager."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import stat
import threading
from pathlib import Path

from missy.mcp.client import McpClient

logger = logging.getLogger(__name__)

MCP_CONFIG_PATH = "~/.missy/mcp.json"


class McpManager:
    """Manages MCP server connections and exposes their tools to the agent.

    Config format (~/.missy/mcp.json)::

        [
            {"name": "filesystem", "command": "npx @modelcontextprotocol/server-filesystem /tmp"},
            {"name": "postgres", "command": "npx @modelcontextprotocol/server-postgres postgresql://..."}
        ]
    """

    def __init__(self, config_path: str = MCP_CONFIG_PATH):
        self._config_path = Path(config_path).expanduser()
        self._clients: dict[str, McpClient] = {}
        self._lock = threading.Lock()

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
                    self._config_path, st.st_uid, os.getuid(),
                )
                return
            if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                logger.warning(
                    "MCP config %s is group/world-writable (mode %o); refusing to load",
                    self._config_path, st.st_mode,
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
        """Connect to a new MCP server and persist the config."""
        if "__" in name:
            raise ValueError(
                f"Invalid MCP server name: {name!r} (must not contain '__')"
            )
        client = McpClient(name=name, command=command, url=url)
        client.connect()
        with self._lock:
            self._clients[name] = client
        self._save_config()
        logger.info("MCP: connected to %r (%d tools)", name, len(client.tools))
        return client

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
        with self._lock:
            client = self._clients.get(server_name)
        if not client:
            return f"[MCP error] server {server_name!r} not connected"
        return client.call_tool(tool_name, arguments)

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
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(json.dumps(entries, indent=2))
