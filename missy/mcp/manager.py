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
from typing import Any

from missy.mcp.annotations import BUILTIN_ANNOTATIONS, AnnotationRegistry
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
        approval_gate: Any | None = None,
    ):
        self._config_path = Path(config_path).expanduser()
        self._clients: dict[str, McpClient] = {}
        self._lock = threading.Lock()
        self._block_injection = block_injection
        # SR-4.7: an ApprovalGate to block on for tools whose annotation
        # sets requires_approval (destructive/mutating MCP tools). None
        # means "no confirmation infrastructure available" -- calls to
        # such tools then fail closed rather than running unconfirmed.
        self._approval_gate = approval_gate
        self._annotation_registry = AnnotationRegistry()
        # Seed registry with known built-in tool annotations.
        for tool_name, annotation in BUILTIN_ANNOTATIONS.items():
            self._annotation_registry.register(tool_name, annotation)

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
        # Register per-tool annotations from the client into the shared registry.
        # Namespaced names follow the same server__tool convention used by all_tools().
        for tool_name, annotation in client.tool_annotations.items():
            namespaced = f"{name}__{tool_name}"
            self._annotation_registry.register(namespaced, annotation)
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

    def call_tool(
        self,
        namespaced_name: str,
        arguments: dict,
        session_id: str = "",
        task_id: str = "",
    ) -> str:
        """Call an MCP tool by its namespaced name (server__tool).

        SR-4.7: this is the single dispatch chokepoint for every MCP tool
        call, so it is where the manifest-pinning and approval-annotation
        requirements are enforced -- immediately before execution, not
        only at connect time.
        """
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

        # Re-verify the pinned manifest digest immediately before dispatch.
        # Connect-time verification (add_server()) alone is not enough: a
        # malicious or compromised server could mutate its tool manifest
        # (e.g. widen a tool's effective behavior) after the initial
        # connection without ever triggering a reconnect.
        expected_digest = self._get_server_digest(server_name)
        if expected_digest is not None:
            from missy.mcp.digest import compute_tool_manifest_digest, verify_digest

            actual_digest = compute_tool_manifest_digest(client.tools)
            if not verify_digest(expected_digest, actual_digest):
                logger.warning(
                    "MCP: digest drift detected for %r at call time "
                    "(expected=%s actual=%s); denying call to %r",
                    server_name,
                    expected_digest,
                    actual_digest,
                    namespaced_name,
                )
                self._emit_call_audit(
                    namespaced_name, session_id, task_id, "deny", "digest_mismatch_at_call_time"
                )
                return (
                    f"[MCP BLOCKED] Server {server_name!r}'s tool manifest no longer "
                    "matches its pinned digest; call denied. Run 'missy mcp pin "
                    f"{server_name}' after verifying the change is expected."
                )

        # Annotation-driven approval gate: destructive/mutating MCP tools
        # must be confirmed by a human before running, same as SR-2.2's
        # proactive-trigger gating -- absence of a configured ApprovalGate
        # means absence of confirmation infrastructure, which must fail
        # closed (deny), not silently run unconfirmed.
        annotation = self.get_annotation(namespaced_name)
        if annotation is not None and annotation.to_policy_hints()["requires_approval"]:
            if self._approval_gate is None:
                self._emit_call_audit(
                    namespaced_name, session_id, task_id, "deny", "no_approval_gate"
                )
                return (
                    f"[MCP DENIED] Tool {namespaced_name!r} requires human approval "
                    "(destructive/mutating), but no approval gate is configured for "
                    "this session."
                )
            try:
                self._approval_gate.request(
                    action=f"MCP tool call: {namespaced_name}",
                    reason=f"arguments={arguments!r}",
                    risk="high" if annotation.mutating else "medium",
                )
            except Exception as exc:
                self._emit_call_audit(
                    namespaced_name, session_id, task_id, "deny", f"approval_failed: {exc}"
                )
                return f"[MCP DENIED] Approval for {namespaced_name!r} was not granted: {exc}"

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
                    self._emit_call_audit(
                        namespaced_name, session_id, task_id, "deny", "injection_detected"
                    )
                    return (
                        f"[MCP BLOCKED] Tool {namespaced_name!r} output contained "
                        f"injection patterns and was blocked: {warnings}"
                    )
                result = f"[SECURITY WARNING: MCP tool output may contain injection] {result}"
        except Exception:
            logger.debug("MCP injection scan failed; tool output passed through", exc_info=True)

        self._emit_call_audit(namespaced_name, session_id, task_id, "allow", "")
        return result

    @staticmethod
    def _emit_call_audit(
        namespaced_name: str, session_id: str, task_id: str, result: str, detail: str
    ) -> None:
        """Emit an ``mcp.tool_execute`` audit event for a call() outcome.

        Distinct from the generic ``tool_execute`` event the ToolRegistry
        already emits when an MCP tool is dispatched as a registered
        BaseTool -- this one captures MCP-specific decisions (digest
        drift, approval outcome) the registry has no visibility into.
        """
        try:
            from missy.core.events import AuditEvent, event_bus

            event_bus.publish(
                AuditEvent.now(
                    session_id=session_id,
                    task_id=task_id,
                    event_type="mcp.tool_execute",
                    category="security" if result == "deny" else "plugin",
                    result=result,  # type: ignore[arg-type]
                    detail={"tool": namespaced_name, "reason": detail},
                )
            )
        except Exception:
            logger.debug("MCP: failed to emit call audit event", exc_info=True)

    def list_servers(self) -> list[dict]:
        with self._lock:
            return [
                {"name": n, "alive": c.is_alive(), "tools": len(c.tools)}
                for n, c in self._clients.items()
            ]

    def get_annotation(self, tool_name: str):
        """Return the :class:`~missy.mcp.annotations.ToolAnnotation` for *tool_name*.

        Accepts both namespaced names (``"server__tool"``) and bare built-in
        names (``"file_read"``).

        Args:
            tool_name: Fully-qualified namespaced tool name or built-in name.

        Returns:
            The stored :class:`~missy.mcp.annotations.ToolAnnotation`, or
            ``None`` if no annotation has been registered for this tool.
        """
        return self._annotation_registry.get(tool_name)

    def get_all_annotations(self) -> dict:
        """Return a snapshot of all registered annotations.

        Returns:
            A dict mapping tool name (str) to
            :class:`~missy.mcp.annotations.ToolAnnotation`.
        """
        return self._annotation_registry.get_all_annotations()

    @property
    def annotation_registry(self) -> AnnotationRegistry:
        """The shared :class:`~missy.mcp.annotations.AnnotationRegistry` for this manager.

        Provides full filtering and summarisation capabilities.

        Returns:
            The :class:`~missy.mcp.annotations.AnnotationRegistry` instance.
        """
        return self._annotation_registry

    def shutdown(self) -> None:
        with self._lock:
            clients = list(self._clients.values())
        for c in clients:
            with contextlib.suppress(Exception):
                c.disconnect()

    def _save_config(self) -> None:
        # SR-1.11: rebuilding entries from self._clients alone drops any
        # digest pinned via `missy mcp pin` — this method is called
        # unconditionally after every successful add_server(), including on
        # reconnect, so without this the very next restart after a
        # successful pin+verify silently erases the pin and every
        # subsequent connection skips digest verification with no operator
        # signal that protection was lost. Read whatever digest currently
        # exists on disk for each server name and carry it forward.
        existing_digests: dict[str, str] = {}
        if self._config_path.exists():
            try:
                existing = json.loads(self._config_path.read_text())
                for entry in existing:
                    digest = entry.get("digest")
                    entry_name = entry.get("name")
                    if digest and entry_name:
                        existing_digests[entry_name] = digest
            except Exception:
                logger.warning(
                    "MCP: could not read existing config at %s to preserve pinned "
                    "digests; any existing pins will not be carried forward",
                    self._config_path,
                )

        with self._lock:
            entries = [
                {"name": name, "command": c._command, "url": c._url}
                for name, c in self._clients.items()
            ]
        for entry in entries:
            digest = existing_digests.get(entry["name"])
            if digest:
                entry["digest"] = digest
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
