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
from missy.mcp.client import McpCallResult, McpClient

logger = logging.getLogger(__name__)

MCP_CONFIG_PATH = "~/.missy/mcp.json"

#: Tool names may only contain alphanumeric characters, hyphens, and underscores.
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _resolve_secret(value: str) -> str:
    """Resolve a ``vault://KEY`` or ``$ENV`` reference to its secret value.

    A plain string is returned unchanged. Used so an MCP server's bearer token
    / header value can be stored as a ``vault://`` reference rather than in
    plaintext in ``mcp.json``. Falls back to the literal on any resolution error.
    """
    if not isinstance(value, str):
        return value
    try:
        from missy.security.vault import Vault

        return Vault().resolve(value)
    except Exception:
        return value


def _resolve_mcp_auth_headers(entry: dict) -> dict[str, str] | None:
    """Build HTTP auth headers for an MCP server entry (F17).

    Supported config keys (HTTP servers only):

    * ``bearer_token``: value (``vault://…``-resolvable) sent as
      ``Authorization: Bearer <token>``.
    * ``headers``: a dict of extra headers; each value is ``vault://``/``$ENV``
      resolved so credentials can live in the vault, not ``mcp.json``.

    Returns ``None`` when no auth is configured (so a plain HTTP/stdio server is
    unaffected).
    """
    if not isinstance(entry, dict) or not entry.get("url"):
        return None
    headers: dict[str, str] = {}
    token = entry.get("bearer_token")
    if token:
        headers["Authorization"] = f"Bearer {_resolve_secret(str(token))}"
    raw_headers = entry.get("headers")
    if isinstance(raw_headers, dict):
        for key, val in raw_headers.items():
            headers[str(key)] = _resolve_secret(str(val))
    return headers or None


def _insecure_auth_kwargs(entry: dict) -> dict[str, bool]:
    """Return ``{"allow_insecure_auth": True}`` only when the entry opts in (SEC-05)."""
    if isinstance(entry, dict) and entry.get("allow_insecure_auth") is True:
        return {"allow_insecure_auth": True}
    return {}


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
        self._desired_servers: dict[str, dict] = {}
        self._live_server_fingerprints: dict[str, str] = {}
        self._annotation_states: dict[str, str] = {}
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
        self._refresh_desired_state()

    def _read_config_servers(self) -> list[dict] | None:
        """Read and validate mcp.json, returning its server entries.

        Returns ``None`` (logging the reason) if the file is missing,
        fails the ownership/permission check, or fails to parse.
        Factored out of :meth:`connect_all` so
        :meth:`_connect_new_servers_from_config` can reuse the exact
        same security checks rather than risk a second, divergent
        implementation that forgets one of them. Uses ``getattr`` (not
        a direct attribute access) so a minimal test double built via
        ``McpManager.__new__()`` -- which bypasses ``__init__`` and
        never sets ``_config_path`` -- doesn't crash ``health_check()``.
        """
        config_path = getattr(self, "_config_path", None)
        if config_path is None:
            logger.debug("No MCP config at %s; skipping", config_path)
            return None
        if not config_path.exists():
            logger.debug("No MCP config at %s; desired state is empty", config_path)
            return []
        # Security: verify file permissions before loading
        try:
            st = config_path.stat()
            if st.st_uid != os.getuid():
                logger.warning(
                    "MCP config %s owned by uid %d, expected %d; refusing to load",
                    config_path,
                    st.st_uid,
                    os.getuid(),
                )
                return None
            if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                logger.warning(
                    "MCP config %s is group/world-writable (mode %o); refusing to load",
                    config_path,
                    st.st_mode,
                )
                return None
        except OSError as exc:
            logger.warning("MCP: cannot stat config %s: %s", config_path, exc)
            return None
        try:
            parsed = json.loads(config_path.read_text())
            if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
                logger.warning("MCP config must be a JSON array of server objects")
                return None
            return parsed
        except Exception as exc:
            logger.warning("MCP config parse error: %s", exc)
            return None

    def _refresh_desired_state(self) -> list[dict] | None:
        """Refresh operator-authored desired state without mutating the file."""
        servers = self._read_config_servers()
        if servers is None:
            return None
        desired: dict[str, dict] = {}
        for entry in servers:
            name = entry.get("name")
            if isinstance(name, str) and name:
                desired[name] = dict(entry)
        self._desired_servers = desired
        return list(desired.values())

    @staticmethod
    def _connection_fingerprint(entry: dict) -> str:
        connection = {
            key: entry.get(key)
            for key in (
                "command",
                "url",
                "bearer_token",
                "headers",
                "allow_insecure_auth",
                "trusted_read_only_tools",
            )
        }
        return json.dumps(connection, sort_keys=True, separators=(",", ":"), default=str)

    def connect_all(self) -> None:
        """Load config and connect to all configured MCP servers."""
        servers = self._refresh_desired_state()
        if servers is None:
            return
        for entry in servers:
            name = entry.get("name", "unknown")
            try:
                self.add_server(
                    name,
                    command=entry.get("command"),
                    url=entry.get("url"),
                    headers=_resolve_mcp_auth_headers(entry),
                    persist=False,
                    **_insecure_auth_kwargs(entry),
                )
            except Exception as exc:
                logger.warning("MCP: failed to connect %r: %s", name, exc)

    def _connect_new_servers_from_config(self) -> None:
        """Connect any server present in mcp.json but not yet tracked.

        ``connect_all()`` only ever runs once, at construction
        (``AgentRuntime._make_mcp_manager()``). For a long-running
        process (``missy chat``/``missy api start``/the Discord bot),
        a separate ``missy mcp add`` CLI invocation edits mcp.json and
        exits without ever touching this daemon's in-memory
        ``self._clients`` -- so a brand-new server was silently never
        connected until the daemon was restarted, contradicting
        ``_sync_mcp_tools()``'s own documented claim that servers
        "connected... after startup (via `missy mcp add`... or
        health_check()) are reflected on the very next turn." Called
        from :meth:`health_check` (already the periodic call site) so
        the fix takes effect through the same existing polling loop.
        Removed entries are disconnected and changed connection definitions
        are replaced, while this reconciliation never writes mcp.json.
        """
        # Only disconnect names that were part of the prior desired state.  A
        # live client may also be installed programmatically (or by a test /
        # embedding application); absence from mcp.json is not evidence that
        # such an unmanaged client was removed by the operator.
        previously_desired = set(getattr(self, "_desired_servers", {}))
        servers = self._refresh_desired_state()
        if servers is None:
            return
        desired_by_name = {str(entry.get("name")): entry for entry in servers if entry.get("name")}
        with self._lock:
            known = set(self._clients.keys())
            removed = (previously_desired - set(desired_by_name)) & known
            removed_clients = [self._clients.pop(name) for name in removed]
        for client in removed_clients:
            with contextlib.suppress(Exception):
                client.disconnect()
        for name in removed:
            self._drop_server_annotations(name)
            self._live_server_fingerprints.pop(name, None)
        for entry in servers:
            name = entry.get("name", "unknown")
            if name in known:
                with self._lock:
                    current = self._clients.get(name)
                live_fingerprint = self._live_server_fingerprints.get(name)
                desired_fingerprint = self._connection_fingerprint(entry)
                connection_changed = (
                    live_fingerprint != desired_fingerprint
                    if live_fingerprint is not None
                    else current is not None
                    and (
                        getattr(current, "_command", None) != entry.get("command")
                        or getattr(current, "_url", None) != entry.get("url")
                    )
                )
                if current is not None and connection_changed:
                    with contextlib.suppress(Exception):
                        current.disconnect()
                    with self._lock:
                        self._clients.pop(name, None)
                    self._drop_server_annotations(name)
                else:
                    continue
            try:
                self.add_server(
                    name,
                    command=entry.get("command"),
                    url=entry.get("url"),
                    headers=_resolve_mcp_auth_headers(entry),
                    persist=False,
                    **_insecure_auth_kwargs(entry),
                )
                logger.info("MCP: connected newly-configured server %r via health_check", name)
            except Exception as exc:
                logger.warning("MCP: failed to connect newly-configured server %r: %s", name, exc)

    def add_server(
        self,
        name: str,
        command: str | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        allow_insecure_auth: bool = False,
        persist: bool = True,
    ) -> McpClient:
        """Connect to a new MCP server and persist the config.

        If the config entry for this server has a ``"digest"`` key, the
        tool manifest digest is verified after connection.  A mismatch
        causes the server to be disconnected and an error to be raised.

        Args:
            headers: Extra HTTP headers for an authenticated HTTP MCP server
                (F17), e.g. ``{"Authorization": "Bearer …"}``. Ignored for
                stdio (command) servers.
        """
        if not _SAFE_NAME_RE.match(name):
            raise ValueError(
                f"Invalid MCP server name: {name!r} "
                "(must contain only alphanumeric, hyphens, underscores)"
            )
        if "__" in name:
            raise ValueError(f"Invalid MCP server name: {name!r} (must not contain '__')")
        client = McpClient(
            name=name,
            command=command,
            url=url,
            headers=headers,
            **({"allow_insecure_auth": True} if allow_insecure_auth else {}),
        )
        client._command = command
        client._url = url
        client.connect()

        # Digest verification (Feature 3)
        expected_digest = self._get_server_digest(name)
        if expected_digest is not None:
            from missy.mcp.digest import compute_tool_manifest_digest, verify_digest

            if not expected_digest.startswith("sha256:v2:"):
                client.disconnect()
                raise ValueError(
                    f"MCP server {name!r} manifest digest mismatch: the configured "
                    "pin uses a legacy/incomplete format. "
                    "Review the full advertised schema and run 'missy mcp pin "
                    f"{name}' to migrate to sha256:v2."
                )
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
            # A server's readOnlyHint is self-reported, not an operator
            # authorization to bypass approval. Only an explicit per-tool
            # override in the owner-controlled mcp.json may grant that trust.
            trusted = (
                getattr(self, "_desired_servers", {})
                .get(name, {})
                .get("trusted_read_only_tools", [])
            )
            if annotation.read_only and (not isinstance(trusted, list) or tool_name not in trusted):
                from missy.mcp.annotations import ToolAnnotation

                annotation = ToolAnnotation.from_mcp_dict({})
            self._annotation_registry.register(namespaced, annotation)
        annotation_states = getattr(client, "tool_annotation_states", {})
        if isinstance(annotation_states, dict):
            if not hasattr(self, "_annotation_states"):
                self._annotation_states = {}
            for tool_name, state in annotation_states.items():
                self._annotation_states[f"{name}__{tool_name}"] = str(state)
        if persist:
            desired_servers = getattr(self, "_desired_servers", {})
            existing = dict(desired_servers.get(name, {"name": name}))
            existing.update({"name": name, "command": command, "url": url})
            desired_servers[name] = existing
            self._desired_servers = desired_servers
            self._save_config()
        if not hasattr(self, "_live_server_fingerprints"):
            self._live_server_fingerprints = {}
        live_entry = getattr(self, "_desired_servers", {}).get(
            name, {"command": command, "url": url}
        )
        self._live_server_fingerprints[name] = self._connection_fingerprint(live_entry)
        logger.info("MCP: connected to %r (%d tools)", name, len(client.tools))
        return client

    def _get_server_digest(self, name: str) -> str | None:
        """Return the pinned digest for server *name* from the config file, or None."""
        desired = getattr(self, "_desired_servers", {}).get(name)
        if desired and desired.get("digest"):
            return str(desired["digest"])
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

        entry = dict(self._desired_servers.get(name, {"name": name}))
        entry["digest"] = digest
        self._desired_servers[name] = entry
        self._save_config(upsert_entries={name: entry})

        return digest

    def remove_server(self, name: str) -> None:
        with self._lock:
            client = self._clients.pop(name, None)
        desired_servers = getattr(self, "_desired_servers", {})
        if client is None and name not in desired_servers:
            return
        if client:
            client.disconnect()
        desired_servers.pop(name, None)
        self._live_server_fingerprints.pop(name, None)
        self._drop_server_annotations(name)
        self._save_config(remove_names={name})

    def _drop_server_annotations(self, name: str) -> None:
        prefix = f"{name}__"
        names = set(getattr(self, "_annotation_states", {}))
        names.update(
            tool_name
            for tool_name in self._annotation_registry.get_all_annotations()
            if tool_name.startswith(prefix)
        )
        for tool_name in names:
            if tool_name.startswith(prefix):
                getattr(self, "_annotation_states", {}).pop(tool_name, None)
                self._annotation_registry.unregister(tool_name)

    def restart_server(self, name: str) -> None:
        """Reconnect a dead MCP server, going through the same full
        connection path as an initial `add_server()` call.

        A prior version built a bare `McpClient` directly and swapped it
        into `self._clients` without going through `add_server()`'s digest
        verification or `tool_annotations` re-registration into
        `self._annotation_registry`. `call_tool()`'s SR-4.7 approval gate
        (`self.get_annotation(namespaced_name)`) is a silent no-op for any
        tool that was never registered -- so a server that died and came
        back with a widened or destructive tool (`requires_approval=True`)
        had that tool exposed via `all_tools()` and freely dispatchable via
        `call_tool()` with the approval gate never consulted, and with no
        digest re-check even if one was pinned. Reusing `add_server()`
        directly (rather than re-deriving a partial subset of its logic
        here) keeps both paths in sync by construction.
        """
        with self._lock:
            client = self._clients.get(name)
        if client:
            cmd = client._command
            url = client._url
            headers = getattr(client, "_headers", None)
            insecure = getattr(client, "_allow_insecure_auth", False) is True
            client.disconnect()
            with self._lock:
                self._clients.pop(name, None)
            self.add_server(
                name,
                command=cmd,
                url=url,
                headers=headers,
                persist=False,
                **({"allow_insecure_auth": True} if insecure else {}),
            )

    def health_check(self) -> None:
        """Restart any dead MCP servers, and connect any newly-configured ones.

        See :meth:`_connect_new_servers_from_config` for why the
        latter is necessary for a long-running process to ever pick
        up a separate ``missy mcp add`` invocation.
        """
        with self._lock:
            dead = [n for n, c in self._clients.items() if not c.is_alive()]
        for name in dead:
            logger.warning("MCP: %r is dead; restarting", name)
            try:
                self.restart_server(name)
            except Exception as exc:
                logger.error("MCP: failed to restart %r: %s", name, exc)
        self._connect_new_servers_from_config()

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

    def _check_digest_drift(
        self, server_name: str, client: Any, namespaced_name: str
    ) -> str | None:
        """Return a denial message if *server_name*'s live manifest no
        longer matches its pinned digest, or ``None`` if it's unpinned or
        still matches.

        Factored out of :meth:`call_tool` so it can be called both before
        AND after an :class:`~missy.agent.approval.ApprovalGate` wait --
        the gate blocks synchronously for up to its configured timeout, a
        window in which a compromised/updated server could mutate its
        manifest after the pre-approval check but before actual dispatch.
        """
        expected_digest = self._get_server_digest(server_name)
        if expected_digest is None:
            return None
        if not expected_digest.startswith("sha256:v2:"):
            return (
                f"[MCP BLOCKED] Server {server_name!r} manifest digest uses a "
                "legacy/incomplete pin; call denied until an operator reviews and repins it."
            )

        from missy.mcp.digest import compute_tool_manifest_digest, verify_digest

        actual_digest = compute_tool_manifest_digest(client.tools)
        if verify_digest(expected_digest, actual_digest):
            return None

        logger.warning(
            "MCP: digest drift detected for %r at call time "
            "(expected=%s actual=%s); denying call to %r",
            server_name,
            expected_digest,
            actual_digest,
            namespaced_name,
        )
        return (
            f"[MCP BLOCKED] Server {server_name!r}'s tool manifest no longer "
            "matches its pinned digest; call denied. Run 'missy mcp pin "
            f"{server_name}' after verifying the change is expected."
        )

    def call_tool(
        self,
        namespaced_name: str,
        arguments: dict,
        session_id: str = "",
        task_id: str = "",
    ) -> McpCallResult:
        """Call an MCP tool by its namespaced name (server__tool).

        SR-4.7: this is the single dispatch chokepoint for every MCP tool
        call, so it is where the manifest-pinning and approval-annotation
        requirements are enforced -- immediately before execution, not
        only at connect time.
        """
        if "__" not in namespaced_name:
            return McpCallResult(
                f"[MCP error] invalid tool name: {namespaced_name}",
                is_error=True,
                error_kind="validation",
            )
        server_name, tool_name = namespaced_name.split("__", 1)
        # Validate tool name characters to prevent injection via crafted names.
        if not _SAFE_NAME_RE.match(tool_name):
            return McpCallResult(
                f"[MCP error] unsafe tool name: {tool_name!r}",
                is_error=True,
                error_kind="validation",
            )
        with self._lock:
            client = self._clients.get(server_name)
        if not client:
            return McpCallResult(
                f"[MCP error] server {server_name!r} not connected",
                is_error=True,
                error_kind="transport",
                transport_certainty="not_sent",
            )

        # Re-verify the pinned manifest digest immediately before dispatch.
        # Connect-time verification (add_server()) alone is not enough: a
        # malicious or compromised server could mutate its tool manifest
        # (e.g. widen a tool's effective behavior) after the initial
        # connection without ever triggering a reconnect.
        digest_error = self._check_digest_drift(server_name, client, namespaced_name)
        if digest_error is not None:
            self._emit_call_audit(
                namespaced_name, session_id, task_id, "deny", "digest_mismatch_at_call_time"
            )
            return McpCallResult(digest_error, is_error=True, error_kind="policy")

        # Annotation-driven approval gate: destructive/mutating MCP tools
        # must be confirmed by a human before running, same as SR-2.2's
        # proactive-trigger gating -- absence of a configured ApprovalGate
        # means absence of confirmation infrastructure, which must fail
        # closed (deny), not silently run unconfirmed.
        annotation = self.get_annotation(namespaced_name)
        if annotation is None:
            from missy.mcp.annotations import ToolAnnotation

            annotation = ToolAnnotation.from_mcp_dict({})
        if annotation.to_policy_hints()["requires_approval"]:
            if self._approval_gate is None:
                self._emit_call_audit(
                    namespaced_name, session_id, task_id, "deny", "no_approval_gate"
                )
                return McpCallResult(
                    f"[MCP DENIED] Tool {namespaced_name!r} requires human approval "
                    "(destructive/mutating), but no approval gate is configured for "
                    "this session.",
                    is_error=True,
                    error_kind="policy",
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
                return McpCallResult(
                    f"[MCP DENIED] Approval for {namespaced_name!r} was not granted: {exc}",
                    is_error=True,
                    error_kind="policy",
                )

            # ApprovalGate.request() blocks synchronously waiting for a
            # human response (up to its configured timeout, 60s by
            # default in production) -- a compromised/updated server could
            # mutate its advertised manifest during that window, after the
            # digest check above ran but before dispatch actually happens.
            # Re-verifying here closes that gap: the operator's approval
            # is only honored against the manifest state that's still
            # current right before the call, not whatever was current
            # when the approval prompt was first shown.
            digest_error = self._check_digest_drift(server_name, client, namespaced_name)
            if digest_error is not None:
                self._emit_call_audit(
                    namespaced_name,
                    session_id,
                    task_id,
                    "deny",
                    "digest_mismatch_after_approval_wait",
                )
                return McpCallResult(digest_error, is_error=True, error_kind="policy")

        try:
            result = client.call_tool(tool_name, arguments)
        except Exception as exc:
            result = McpCallResult(
                f"[MCP error] uncertain transport outcome: {exc}",
                is_error=True,
                error_kind="transport",
                transport_certainty="uncertain",
            )
        # Defense-in-depth: scan MCP tool results for prompt injection.
        try:
            from missy.security.sanitizer import InputSanitizer

            result_text = result if isinstance(result, str) else json.dumps(result, default=str)
            warnings = InputSanitizer().check_for_injection(result_text)
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
                    return McpCallResult(
                        f"[MCP BLOCKED] Tool {namespaced_name!r} output contained "
                        f"injection patterns and was blocked: {warnings}",
                        is_error=True,
                        error_kind="policy",
                    )
                result = McpCallResult(
                    f"[SECURITY WARNING: MCP tool output may contain injection] {result_text}",
                    is_error=bool(getattr(result, "is_error", False)),
                    error_kind=getattr(result, "error_kind", None),
                    protocol_error=getattr(result, "protocol_error", None),
                    transport_certainty=getattr(result, "transport_certainty", "completed"),
                )
        except Exception as exc:
            # Do not ask logging to format the traceback here.  Traceback
            # formatting can itself import modules (for example via
            # ``linecache``), so an unavailable or compromised import path
            # could otherwise raise again before the fail-closed result is
            # returned.  The exception type is enough diagnostic context and
            # keeps this security boundary independent of traceback machinery.
            logger.error(
                "MCP injection scan failed; blocking tool output (%s)",
                type(exc).__name__,
            )
            self._emit_call_audit(
                namespaced_name, session_id, task_id, "deny", "injection_scan_failed"
            )
            return McpCallResult(
                f"[MCP BLOCKED] Tool {namespaced_name!r} output could not be "
                "security-scanned for prompt injection.",
                is_error=True,
                error_kind="policy",
            )

        is_error = bool(getattr(result, "is_error", False))
        self._emit_call_audit(
            namespaced_name,
            session_id,
            task_id,
            "error" if is_error else "allow",
            str(getattr(result, "error_kind", "") or ""),
        )
        if isinstance(result, McpCallResult):
            return result
        return McpCallResult(str(result))

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
            clients = dict(self._clients)
        desired_servers = getattr(self, "_desired_servers", {})
        annotation_states = getattr(self, "_annotation_states", {})
        names = set(desired_servers) | set(clients)
        result = []
        for name in sorted(names):
            client = clients.get(name)
            states = {
                tool_name.removeprefix(f"{name}__"): state
                for tool_name, state in annotation_states.items()
                if tool_name.startswith(f"{name}__")
            }
            result.append(
                {
                    "name": name,
                    "alive": bool(client and client.is_alive()),
                    "tools": len(client.tools) if client else 0,
                    "desired": name in desired_servers,
                    "annotation_states": states,
                }
            )
        return result

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

    def _save_config(
        self,
        *,
        remove_names: set[str] | None = None,
        upsert_entries: dict[str, dict] | None = None,
    ) -> None:
        # SR-1.11: rebuilding entries from self._clients alone drops any
        # digest pinned via `missy mcp pin` — this method is called
        # unconditionally after every successful add_server(), including on
        # reconnect, so without this the very next restart after a
        # successful pin+verify silently erases the pin and every
        # subsequent connection skips digest verification with no operator
        # signal that protection was lost. Read whatever digest currently
        # exists on disk for each server name and carry it forward.
        # The same applies to operator-authored per-server keys
        # (``bearer_token``/``headers``/``allow_insecure_auth``): they only
        # ever live on disk (secrets stay as vault:// references), so every
        # non-connection key of an existing entry is carried forward too.
        existing_entries: dict[str, dict] = {
            str(name): dict(entry)
            for name, entry in getattr(self, "_desired_servers", {}).items()
            if isinstance(entry, dict)
        }
        if self._config_path.exists():
            try:
                existing = json.loads(self._config_path.read_text())
                if not isinstance(existing, list):
                    raise ValueError("MCP config must be a JSON array")
                for entry in existing:
                    if not isinstance(entry, dict):
                        raise ValueError("MCP config entries must be objects")
                    entry_name = entry.get("name")
                    if entry_name:
                        existing_entries[entry_name] = dict(entry)
            except Exception:
                logger.warning(
                    "MCP: could not read existing config at %s to preserve pinned "
                    "digests; any existing pins will not be carried forward",
                    self._config_path,
                )

        for name in remove_names or set():
            existing_entries.pop(name, None)
        for name, entry in (upsert_entries or {}).items():
            existing_entries[name] = dict(entry)
        with self._lock:
            clients = list(self._clients.items())
        for name, client in clients:
            entry = existing_entries.setdefault(name, {"name": name})
            command = getattr(client, "_command", None)
            url = getattr(client, "_url", None)
            entry["command"] = command if isinstance(command, str) else None
            entry["url"] = url if isinstance(url, str) else None
        entries = list(existing_entries.values())
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
            self._desired_servers = {
                str(entry["name"]): dict(entry)
                for entry in entries
                if isinstance(entry.get("name"), str) and entry.get("name")
            }
        except Exception:
            if not closed:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
