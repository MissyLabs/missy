"""Integration tests for McpManager, SkillDiscovery, and ToolRegistry interaction.

Covers:
1.  McpManager lifecycle: add server → list tools → remove server
2.  McpManager health check: healthy server passes, dead server restarts
3.  McpManager tool namespacing: tools are exposed as server__tool
4.  McpManager digest pinning: matching digest loads, mismatch refuses
5.  McpManager concurrent tool calls: multiple tools from the same server
6.  SkillDiscovery scan: finds SKILL.md files recursively
7.  SkillDiscovery YAML frontmatter: parses name, description, version, tools
8.  SkillDiscovery fuzzy search: finds skills by partial name and description
9.  SkillDiscovery duplicate handling: same skill name in multiple directories
10. ToolRegistry + MCP tools: MCP-derived tool descriptors alongside built-in tools
11. ToolRegistry permission check: tools with required permissions are checked
12. Tool execution audit trail: every execution emits a structured audit event
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.events import event_bus
from missy.mcp.digest import compute_tool_manifest_digest
from missy.mcp.manager import McpManager
from missy.policy.engine import init_policy_engine
from missy.skills.discovery import SkillDiscovery, SkillManifest
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SKILL_TEMPLATE = """\
---
name: {name}
description: {description}
version: {version}
author: {author}
tools: [{tools}]
---

# Instructions
{instructions}
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_skill_md(
    path: Path,
    *,
    name: str,
    description: str = "A test skill",
    version: str = "1.0.0",
    author: str = "Tester",
    tools: str = "web_fetch",
    instructions: str = "Use the tools wisely.",
) -> Path:
    """Write a SKILL.md file under *path* and return the file path."""
    skill_file = path / "SKILL.md"
    skill_file.write_text(
        _SKILL_TEMPLATE.format(
            name=name,
            description=description,
            version=version,
            author=author,
            tools=tools,
            instructions=instructions,
        )
    )
    return skill_file


def _make_mock_client(
    *,
    tools: list[dict] | None = None,
    command: str = "fake-cmd",
    url: str | None = None,
    alive: bool = True,
) -> MagicMock:
    """Build a fully-stubbed McpClient mock."""
    client = MagicMock()
    client._command = command
    client._url = url
    client.tools = tools if tools is not None else []
    client.is_alive.return_value = alive
    return client


def _make_config(
    *,
    allowed_domains: list[str] | None = None,
    allowed_hosts: list[str] | None = None,
    shell_enabled: bool = False,
    shell_commands: list[str] | None = None,
    read_paths: list[str] | None = None,
    write_paths: list[str] | None = None,
) -> MissyConfig:
    """Return a minimal MissyConfig without touching the real filesystem."""
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=allowed_domains or [],
            allowed_hosts=allowed_hosts or [],
        ),
        filesystem=FilesystemPolicy(
            allowed_read_paths=read_paths or ["/tmp"],
            allowed_write_paths=write_paths or ["/tmp"],
        ),
        shell=ShellPolicy(enabled=shell_enabled, allowed_commands=shell_commands or []),
        plugins=PluginPolicy(enabled=False, allowed_plugins=[]),
        providers={},
        workspace_path="/tmp",
        audit_log_path="~/.missy/audit.log",
    )


# ---------------------------------------------------------------------------
# Concrete tool helpers for ToolRegistry tests
# ---------------------------------------------------------------------------


class _EchoTool(BaseTool):
    name = "echo"
    description = "Return input unchanged."
    permissions = ToolPermissions()

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("text", ""))


class _ShellTool(BaseTool):
    """Declares shell permission to exercise policy checks."""

    name = "shell_exec"
    description = "Execute a shell command."
    permissions = ToolPermissions(shell=True)

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="executed")


class _NetworkTool(BaseTool):
    """Declares network permission against a specific host."""

    name = "net_fetch"
    description = "Fetch a URL."
    permissions = ToolPermissions(network=True, allowed_hosts=["allowed.example.com"])

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="fetched")


class _FsReadTool(BaseTool):
    """Declares filesystem-read permission."""

    name = "fs_read"
    description = "Read a file."
    permissions = ToolPermissions(filesystem_read=True, allowed_paths=["/tmp"])

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="file contents")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_event_bus() -> None:
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture()
def tmp_mcp_config(tmp_path: Path) -> str:
    """Path to a temp mcp.json that does not yet exist."""
    return str(tmp_path / "mcp.json")


@pytest.fixture()
def manager(tmp_mcp_config: str) -> McpManager:
    return McpManager(config_path=tmp_mcp_config)


@pytest.fixture()
def discovery() -> SkillDiscovery:
    return SkillDiscovery()


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


# ---------------------------------------------------------------------------
# 1. McpManager lifecycle
# ---------------------------------------------------------------------------


class TestMcpManagerLifecycle:
    """add_server → list tools → remove_server round-trip."""

    def test_add_server_populates_client_list(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        tools = [{"name": "read_file", "description": "Reads a file"}]
        mock_client = _make_mock_client(tools=tools)

        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("storage", command="mcp-storage")

        servers = manager.list_servers()
        assert len(servers) == 1
        assert servers[0]["name"] == "storage"
        assert servers[0]["alive"] is True
        assert servers[0]["tools"] == 1

    def test_add_server_persists_config(self, manager: McpManager, tmp_mcp_config: str) -> None:
        # _save_config reads client._command directly, so the mock must carry the
        # same command string that was passed to add_server.
        mock_client = _make_mock_client(
            tools=[{"name": "ping", "description": "Ping"}],
            command="mcp-pinger",
        )

        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("pinger", command="mcp-pinger")

        saved = json.loads(Path(tmp_mcp_config).read_text())
        assert len(saved) == 1
        assert saved[0]["name"] == "pinger"
        assert saved[0]["command"] == "mcp-pinger"

    def test_remove_server_disconnects_and_clears(self, manager: McpManager) -> None:
        mock_client = _make_mock_client()
        manager._clients["bye"] = mock_client

        manager.remove_server("bye")

        mock_client.disconnect.assert_called_once()
        assert manager.list_servers() == []

    def test_remove_nonexistent_server_is_idempotent(self, manager: McpManager) -> None:
        # Should not raise; just a no-op.
        manager.remove_server("ghost")

    def test_full_lifecycle(self, manager: McpManager) -> None:
        """add → list → remove → confirm empty."""
        tools = [{"name": "do_thing", "description": "Does a thing"}]
        mock_client = _make_mock_client(tools=tools)

        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("svc", command="mcp-svc")

        assert len(manager.list_servers()) == 1

        manager.remove_server("svc")

        assert manager.list_servers() == []

    def test_add_multiple_servers(self, manager: McpManager) -> None:
        for server_name in ("alpha", "beta", "gamma"):
            client = _make_mock_client(tools=[{"name": "t", "description": "t"}])
            with patch("missy.mcp.manager.McpClient", return_value=client):
                manager.add_server(server_name, command=f"mcp-{server_name}")

        servers = manager.list_servers()
        names = {s["name"] for s in servers}
        assert names == {"alpha", "beta", "gamma"}

    def test_shutdown_disconnects_all_clients(self, manager: McpManager) -> None:
        clients = [_make_mock_client() for _ in range(3)]
        for i, c in enumerate(clients):
            manager._clients[f"srv{i}"] = c

        manager.shutdown()

        for c in clients:
            c.disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# 2. McpManager health check
# ---------------------------------------------------------------------------


class TestMcpManagerHealthCheck:
    """health_check() restarts dead servers and leaves healthy ones alone."""

    def test_healthy_server_not_restarted(self, manager: McpManager) -> None:
        alive = _make_mock_client(alive=True)
        manager._clients["healthy"] = alive

        with patch.object(manager, "restart_server") as mock_restart:
            manager.health_check()

        mock_restart.assert_not_called()

    def test_dead_server_is_restarted(self, manager: McpManager) -> None:
        dead = _make_mock_client(alive=False, command="mcp-dead")
        manager._clients["dead"] = dead

        new_client = _make_mock_client(tools=[{"name": "revived", "description": "back"}])
        with patch("missy.mcp.manager.McpClient", return_value=new_client):
            manager.health_check()

        dead.disconnect.assert_called_once()
        new_client.connect.assert_called_once()
        assert manager._clients["dead"] is new_client

    def test_mixed_health_only_dead_restarted(self, manager: McpManager) -> None:
        alive = _make_mock_client(alive=True)
        dead = _make_mock_client(alive=False, command="mcp-dead")
        manager._clients["alive_srv"] = alive
        manager._clients["dead_srv"] = dead

        new_client = _make_mock_client()
        with patch("missy.mcp.manager.McpClient", return_value=new_client):
            manager.health_check()

        alive.disconnect.assert_not_called()
        dead.disconnect.assert_called_once()

    def test_health_check_does_not_propagate_restart_errors(self, manager: McpManager) -> None:
        dead = _make_mock_client(alive=False, command="mcp-bad")
        manager._clients["fragile"] = dead

        with patch.object(manager, "restart_server", side_effect=RuntimeError("no restart")):
            # Must not raise; errors are logged and swallowed.
            manager.health_check()


# ---------------------------------------------------------------------------
# 3. McpManager tool namespacing
# ---------------------------------------------------------------------------


class TestMcpToolNamespacing:
    """all_tools() must prefix each tool name with the server name."""

    def test_single_server_tools_are_namespaced(self, manager: McpManager) -> None:
        client = _make_mock_client(
            tools=[
                {"name": "read", "description": "Read a resource"},
                {"name": "write", "description": "Write a resource"},
            ]
        )
        manager._clients["store"] = client

        tools = manager.all_tools()

        names = {t["name"] for t in tools}
        assert names == {"store__read", "store__write"}

    def test_namespaced_tool_retains_original_tool_name(self, manager: McpManager) -> None:
        client = _make_mock_client(tools=[{"name": "search", "description": "Search"}])
        manager._clients["idx"] = client

        tools = manager.all_tools()

        assert tools[0]["_mcp_tool"] == "search"
        assert tools[0]["_mcp_server"] == "idx"

    def test_multiple_servers_all_tools_namespaced(self, manager: McpManager) -> None:
        for srv, tool_name in (("db", "query"), ("fs", "list_dir"), ("net", "fetch")):
            client = _make_mock_client(tools=[{"name": tool_name, "description": "desc"}])
            manager._clients[srv] = client

        tools = manager.all_tools()
        names = {t["name"] for t in tools}

        assert "db__query" in names
        assert "fs__list_dir" in names
        assert "net__fetch" in names

    def test_call_tool_routes_to_correct_server(self, manager: McpManager) -> None:
        client_a = _make_mock_client(tools=[])
        client_b = _make_mock_client(tools=[])
        client_a.call_tool.return_value = "result-from-a"
        client_b.call_tool.return_value = "result-from-b"
        manager._clients["svc_a"] = client_a
        manager._clients["svc_b"] = client_b

        result = manager.call_tool("svc_b__do_work", {"arg": "val"})

        assert result == "result-from-b"
        client_a.call_tool.assert_not_called()
        client_b.call_tool.assert_called_once_with("do_work", {"arg": "val"})

    def test_call_tool_missing_namespace_separator(self, manager: McpManager) -> None:
        result = manager.call_tool("notnamespaced", {})
        assert "[MCP error]" in result

    def test_call_tool_unknown_server_name(self, manager: McpManager) -> None:
        result = manager.call_tool("unknown__any_tool", {})
        assert "[MCP error]" in result
        assert "not connected" in result

    def test_empty_servers_returns_no_tools(self, manager: McpManager) -> None:
        assert manager.all_tools() == []


# ---------------------------------------------------------------------------
# 4. McpManager digest pinning
# ---------------------------------------------------------------------------


class TestMcpDigestPinning:
    """Digest pinning: matching digest → load; mismatch → refuse and disconnect."""

    def _write_config(self, tmp_mcp_config: str, entries: list[dict]) -> None:
        p = Path(tmp_mcp_config)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(entries))

    def test_no_pinned_digest_loads_without_verification(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        self._write_config(tmp_mcp_config, [{"name": "srv", "command": "mcp-srv"}])
        tools = [{"name": "op", "description": "Op"}]
        mock_client = _make_mock_client(tools=tools)

        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("srv", command="mcp-srv")

        # Server should be present with no exception raised.
        assert manager.list_servers()[0]["name"] == "srv"

    def test_matching_digest_allows_connection(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        tools = [{"name": "ping", "description": "Ping"}]
        correct_digest = compute_tool_manifest_digest(tools)
        self._write_config(
            tmp_mcp_config,
            [{"name": "srv", "command": "mcp-srv", "digest": correct_digest}],
        )
        mock_client = _make_mock_client(tools=tools)

        with patch("missy.mcp.manager.McpClient", return_value=mock_client):
            manager.add_server("srv", command="mcp-srv")

        assert len(manager.list_servers()) == 1

    def test_mismatched_digest_raises_and_disconnects(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        tools = [{"name": "ping", "description": "Ping"}]
        self._write_config(
            tmp_mcp_config,
            [{"name": "srv", "command": "mcp-srv", "digest": "sha256:deadbeef00"}],
        )
        mock_client = _make_mock_client(tools=tools)

        with (
            patch("missy.mcp.manager.McpClient", return_value=mock_client),
            pytest.raises(ValueError, match="digest mismatch"),
        ):
            manager.add_server("srv", command="mcp-srv")

        mock_client.disconnect.assert_called_once()
        assert manager.list_servers() == []

    def test_digest_mismatch_emits_security_audit_event(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        tools = [{"name": "t", "description": "T"}]
        self._write_config(
            tmp_mcp_config,
            [{"name": "srv", "command": "mcp-srv", "digest": "sha256:wrongdigest"}],
        )
        mock_client = _make_mock_client(tools=tools)

        with (
            patch("missy.mcp.manager.McpClient", return_value=mock_client),
            pytest.raises(ValueError),
        ):
            manager.add_server("srv", command="mcp-srv")

        events = event_bus.get_events(event_type="mcp.digest_mismatch")
        assert len(events) == 1
        assert events[0].result == "deny"
        assert events[0].detail["server"] == "srv"

    def test_pin_server_digest_computes_and_persists(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        # Pre-populate the config file and inject a client directly.
        self._write_config(tmp_mcp_config, [{"name": "srv", "command": "mcp-srv"}])
        tools = [{"name": "ping", "description": "Ping"}]
        client = _make_mock_client(tools=tools)
        manager._clients["srv"] = client

        digest = manager.pin_server_digest("srv")

        expected = compute_tool_manifest_digest(tools)
        assert digest == expected
        saved = json.loads(Path(tmp_mcp_config).read_text())
        assert saved[0]["digest"] == expected

    def test_pin_nonexistent_server_raises_key_error(self, manager: McpManager) -> None:
        with pytest.raises(KeyError, match="not connected"):
            manager.pin_server_digest("ghost")


# ---------------------------------------------------------------------------
# 5. McpManager concurrent tool calls
# ---------------------------------------------------------------------------


class TestMcpConcurrentToolCalls:
    """Multiple simultaneous tool calls to the same server must not race."""

    def test_concurrent_calls_from_single_server(self, manager: McpManager) -> None:
        call_count = 0
        lock = threading.Lock()

        def _call_tool(name: str, args: dict) -> str:
            nonlocal call_count
            with lock:
                call_count += 1
            return f"result-{name}"

        client = _make_mock_client(tools=[])
        client.call_tool.side_effect = _call_tool
        manager._clients["worker"] = client

        n_threads = 20
        results: list[str] = []
        results_lock = threading.Lock()

        def _invoke(idx: int) -> None:
            r = manager.call_tool(f"worker__tool_{idx}", {"n": idx})
            with results_lock:
                results.append(r)

        threads = [threading.Thread(target=_invoke, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == n_threads
        assert call_count == n_threads

    def test_concurrent_calls_to_different_servers(self, manager: McpManager) -> None:
        for name in ("srv_a", "srv_b", "srv_c"):
            c = _make_mock_client(tools=[])
            c.call_tool.return_value = f"ok-from-{name}"
            manager._clients[name] = c

        errors: list[Exception] = []
        results: list[str] = []
        lock = threading.Lock()

        def _invoke(srv: str, tool: str) -> None:
            try:
                r = manager.call_tool(f"{srv}__{tool}", {})
                with lock:
                    results.append(r)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_invoke, args=(srv, "op"))
            for srv in ("srv_a", "srv_b", "srv_c")
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert len(results) == 15


# ---------------------------------------------------------------------------
# 6. SkillDiscovery scan
# ---------------------------------------------------------------------------


class TestSkillDiscoveryScan:
    """scan_directory() finds SKILL.md files recursively."""

    def test_finds_single_skill(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        subdir = tmp_path / "web-search"
        subdir.mkdir()
        _make_skill_md(subdir, name="web-search")

        manifests = discovery.scan_directory(str(tmp_path))

        assert len(manifests) == 1
        assert manifests[0].name == "web-search"

    def test_finds_deeply_nested_skill(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        nested = tmp_path / "category" / "subcategory" / "deep-skill"
        nested.mkdir(parents=True)
        _make_skill_md(nested, name="deep-skill")

        manifests = discovery.scan_directory(str(tmp_path))

        assert len(manifests) == 1
        assert manifests[0].name == "deep-skill"

    def test_finds_multiple_skills_in_sibling_dirs(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        names = ["alpha", "beta", "gamma", "delta"]
        for n in names:
            d = tmp_path / n
            d.mkdir()
            _make_skill_md(d, name=n)

        manifests = discovery.scan_directory(str(tmp_path))

        found_names = {m.name for m in manifests}
        assert found_names == set(names)

    def test_empty_directory_returns_empty_list(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        assert discovery.scan_directory(str(tmp_path)) == []

    def test_nonexistent_directory_returns_empty_list(self, discovery: SkillDiscovery) -> None:
        assert discovery.scan_directory("/tmp/__nonexistent_skills_dir_xyz__") == []

    def test_invalid_skills_are_skipped(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        good = tmp_path / "good"
        good.mkdir()
        _make_skill_md(good, name="good-skill")

        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("No frontmatter at all — should be skipped.")

        manifests = discovery.scan_directory(str(tmp_path))

        assert len(manifests) == 1
        assert manifests[0].name == "good-skill"

    def test_skills_without_name_are_skipped(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        nameless = tmp_path / "nameless"
        nameless.mkdir()
        (nameless / "SKILL.md").write_text("---\ndescription: no name here\n---\n\nBody.\n")

        manifests = discovery.scan_directory(str(tmp_path))

        assert manifests == []

    def test_scan_result_includes_absolute_path(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        subdir = tmp_path / "myskill"
        subdir.mkdir()
        _make_skill_md(subdir, name="myskill")

        manifests = discovery.scan_directory(str(tmp_path))

        assert manifests[0].path == str((subdir / "SKILL.md").resolve())


# ---------------------------------------------------------------------------
# 7. SkillDiscovery YAML frontmatter parsing
# ---------------------------------------------------------------------------


class TestSkillDiscoveryFrontmatter:
    """parse_skill_md() correctly extracts all frontmatter fields."""

    def test_parses_all_standard_fields(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        content = (
            "---\n"
            "name: data-processor\n"
            "description: Processes structured data\n"
            "version: 2.1.3\n"
            "author: DataLabs\n"
            "tools: [json_parse, csv_read, db_write]\n"
            "---\n"
            "\n"
            "# How to process data\n"
            "Parse the input, transform, and write to db.\n"
        )
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(content)

        manifest = discovery.parse_skill_md(str(skill_file))

        assert manifest.name == "data-processor"
        assert manifest.description == "Processes structured data"
        assert manifest.version == "2.1.3"
        assert manifest.author == "DataLabs"
        assert manifest.tools == ["json_parse", "csv_read", "db_write"]
        assert "Parse the input" in manifest.instructions

    def test_minimal_frontmatter_uses_defaults(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        (tmp_path / "SKILL.md").write_text("---\nname: bare\n---\n\nMinimal.\n")

        manifest = discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

        assert manifest.name == "bare"
        assert manifest.description == ""
        assert manifest.version == ""
        assert manifest.author == ""
        assert manifest.tools == []
        assert manifest.instructions == "Minimal."

    def test_tools_as_comma_string_are_parsed(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: multi\ndescription: Many tools\ntools: web_fetch, shell_exec\n---\n\nInstructions.\n"
        (tmp_path / "SKILL.md").write_text(content)

        manifest = discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

        assert "web_fetch" in manifest.tools
        assert "shell_exec" in manifest.tools

    def test_tools_as_inline_list_are_parsed(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: listed\ndescription: listed tools\ntools: [tool_a, tool_b, tool_c]\n---\n\nBody.\n"
        (tmp_path / "SKILL.md").write_text(content)

        manifest = discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

        assert manifest.tools == ["tool_a", "tool_b", "tool_c"]

    def test_file_not_found_raises(self, discovery: SkillDiscovery) -> None:
        with pytest.raises(FileNotFoundError):
            discovery.parse_skill_md("/tmp/__no_such_skill__.md")

    def test_missing_frontmatter_raises_value_error(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        (tmp_path / "SKILL.md").write_text("# Just a markdown file\nNo frontmatter.\n")

        with pytest.raises(ValueError, match="No YAML frontmatter"):
            discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

    def test_missing_name_field_raises_value_error(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        (tmp_path / "SKILL.md").write_text("---\ndescription: has no name\n---\n\nBody.\n")

        with pytest.raises(ValueError, match="Missing required 'name'"):
            discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

    def test_quoted_string_values_are_unquoted(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = '---\nname: "quoted-skill"\ndescription: \'single quoted\'\nversion: "3.0.0"\n---\n\nBody.\n'
        (tmp_path / "SKILL.md").write_text(content)

        manifest = discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

        assert manifest.name == "quoted-skill"
        assert manifest.description == "single quoted"
        assert manifest.version == "3.0.0"

    def test_instructions_body_preserved_verbatim(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        body = "## Step 1\nDo thing A.\n\n## Step 2\nDo thing B.\n"
        content = f"---\nname: steps\n---\n\n{body}"
        (tmp_path / "SKILL.md").write_text(content)

        manifest = discovery.parse_skill_md(str(tmp_path / "SKILL.md"))

        assert "Step 1" in manifest.instructions
        assert "Step 2" in manifest.instructions


# ---------------------------------------------------------------------------
# 8. SkillDiscovery fuzzy search
# ---------------------------------------------------------------------------


class TestSkillDiscoverySearch:
    """search() does a case-insensitive substring match on name and description."""

    @pytest.fixture()
    def skill_pool(self) -> list[SkillManifest]:
        return [
            SkillManifest(
                name="web-search",
                description="Search the web using DuckDuckGo",
                version="1.0.0",
                author="A",
            ),
            SkillManifest(
                name="file-manager",
                description="Manage files and directories on disk",
                version="1.0.0",
                author="B",
            ),
            SkillManifest(
                name="calculator",
                description="Perform arithmetic and math calculations",
                version="1.0.0",
                author="C",
            ),
            SkillManifest(
                name="data-exporter",
                description="Export data to CSV or JSON files",
                version="1.0.0",
                author="D",
            ),
        ]

    def test_name_match_ranked_first(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("web", skill_pool)
        assert results[0].name == "web-search"

    def test_description_only_match_returned(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("arithmetic", skill_pool)
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_case_insensitive_name_match(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("CALCULATOR", skill_pool)
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_case_insensitive_description_match(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("DUCKDUCKGO", skill_pool)
        assert len(results) == 1
        assert results[0].name == "web-search"

    def test_no_match_returns_empty(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("totally-nonexistent-xyz", skill_pool)
        assert results == []

    def test_empty_query_returns_all(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("", skill_pool)
        assert len(results) == len(skill_pool)

    def test_partial_name_match(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        results = discovery.search("file", skill_pool)
        # "file-manager" matches by name; "data-exporter" description contains "files"
        name_match = next(r for r in results if r.name == "file-manager")
        assert name_match is not None
        # name match must come before description match
        assert results.index(name_match) == 0

    def test_multiple_description_matches(
        self, discovery: SkillDiscovery, skill_pool: list[SkillManifest]
    ) -> None:
        # "files" appears in "file-manager" (name match) and "data-exporter" (description)
        results = discovery.search("files", skill_pool)
        result_names = [r.name for r in results]
        assert "file-manager" in result_names
        assert "data-exporter" in result_names

    def test_search_on_empty_skill_list(self, discovery: SkillDiscovery) -> None:
        assert discovery.search("anything", []) == []


# ---------------------------------------------------------------------------
# 9. SkillDiscovery duplicate handling
# ---------------------------------------------------------------------------


class TestSkillDiscoveryDuplicates:
    """Same skill name discovered in multiple directories."""

    def test_duplicate_name_both_loaded(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        """Two directories with the same skill name produce two distinct manifests."""
        for suffix in ("v1", "v2"):
            d = tmp_path / suffix
            d.mkdir()
            _make_skill_md(d, name="shared-skill", version=f"1.0.{suffix[-1]}")

        manifests = discovery.scan_directory(str(tmp_path))

        assert len(manifests) == 2
        paths = {m.path for m in manifests}
        assert len(paths) == 2

    def test_duplicate_skills_have_different_paths(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        for idx in range(3):
            d = tmp_path / f"dir{idx}"
            d.mkdir()
            _make_skill_md(d, name="my-skill")

        manifests = discovery.scan_directory(str(tmp_path))

        assert len(manifests) == 3
        assert len({m.path for m in manifests}) == 3

    def test_search_returns_all_duplicates(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        for idx in range(3):
            d = tmp_path / f"slot{idx}"
            d.mkdir()
            _make_skill_md(d, name="duplicate", description="Duplicated skill")

        manifests = discovery.scan_directory(str(tmp_path))
        results = discovery.search("duplicate", manifests)

        assert len(results) == 3


# ---------------------------------------------------------------------------
# 10. ToolRegistry + MCP tools together
# ---------------------------------------------------------------------------


class TestToolRegistryWithMcpTools:
    """Built-in tools and MCP-derived tool descriptors coexist in the registry."""

    def test_mcp_tool_descriptors_registered_alongside_builtins(
        self, registry: ToolRegistry
    ) -> None:
        # Register a real built-in tool.
        registry.register(_EchoTool())

        # MCP returns raw tool dicts; the registry stores BaseTool instances.
        # Simulate an MCP-backed adapter tool.
        mcp_adapter = MagicMock(spec=BaseTool)
        mcp_adapter.name = "fs__read_file"
        mcp_adapter.description = "Read a file via MCP"
        mcp_adapter.permissions = ToolPermissions()
        mcp_adapter.execute.return_value = ToolResult(success=True, output="file-contents")
        registry.register(mcp_adapter)

        names = registry.list_tools()
        assert "echo" in names
        assert "fs__read_file" in names

    def test_mcp_tool_is_callable_via_registry(self, registry: ToolRegistry) -> None:
        mcp_adapter = MagicMock(spec=BaseTool)
        mcp_adapter.name = "db__query"
        mcp_adapter.description = "Run a DB query via MCP"
        mcp_adapter.permissions = ToolPermissions()
        mcp_adapter.execute.return_value = ToolResult(success=True, output="rows")
        registry.register(mcp_adapter)

        result = registry.execute("db__query", sql="SELECT 1")

        assert result.success is True
        assert result.output == "rows"
        mcp_adapter.execute.assert_called_once_with(sql="SELECT 1")

    def test_registry_replaces_duplicate_name(self, registry: ToolRegistry) -> None:
        first = MagicMock(spec=BaseTool)
        first.name = "worker"
        first.permissions = ToolPermissions()
        first.execute.return_value = ToolResult(success=True, output="first")

        second = MagicMock(spec=BaseTool)
        second.name = "worker"
        second.permissions = ToolPermissions()
        second.execute.return_value = ToolResult(success=True, output="second")

        registry.register(first)
        registry.register(second)

        result = registry.execute("worker")
        assert result.output == "second"

    def test_execute_unknown_tool_raises_key_error(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError, match="no-such-tool"):
            registry.execute("no-such-tool")

    def test_many_tools_listed_sorted(self, registry: ToolRegistry) -> None:
        for name in ("zebra", "alpha", "mango", "fs__read"):
            t = MagicMock(spec=BaseTool)
            t.name = name
            t.permissions = ToolPermissions()
            t.execute.return_value = ToolResult(success=True, output=name)
            registry.register(t)

        names = registry.list_tools()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# 11. ToolRegistry permission check
# ---------------------------------------------------------------------------


class TestToolRegistryPermissions:
    """Policy checks are enforced before tool execution."""

    def test_tool_without_permissions_executes_without_policy(self, registry: ToolRegistry) -> None:
        registry.register(_EchoTool())
        result = registry.execute("echo", text="hello")
        assert result.success is True

    def test_shell_tool_denied_when_shell_disabled(self, registry: ToolRegistry) -> None:
        config = _make_config(shell_enabled=False)
        init_policy_engine(config)
        registry.register(_ShellTool())

        result = registry.execute("shell_exec", command="ls")

        assert result.success is False
        assert result.error is not None

    def test_shell_tool_allowed_when_command_whitelisted(self, registry: ToolRegistry) -> None:
        config = _make_config(shell_enabled=True, shell_commands=["ls"])
        init_policy_engine(config)
        registry.register(_ShellTool())

        result = registry.execute("shell_exec", command="ls")

        assert result.success is True

    def test_network_tool_denied_when_host_blocked(self, registry: ToolRegistry) -> None:
        # default_deny=True and allowed.example.com is not in allowed_hosts
        config = _make_config()
        init_policy_engine(config)
        registry.register(_NetworkTool())

        result = registry.execute("net_fetch")

        assert result.success is False
        assert result.error is not None

    def test_network_tool_allowed_when_host_whitelisted(self, registry: ToolRegistry) -> None:
        config = _make_config(allowed_hosts=["allowed.example.com"])
        init_policy_engine(config)
        registry.register(_NetworkTool())

        result = registry.execute("net_fetch")

        assert result.success is True

    def test_filesystem_read_tool_allowed_when_path_permitted(self, registry: ToolRegistry) -> None:
        config = _make_config(read_paths=["/tmp"], write_paths=["/tmp"])
        init_policy_engine(config)
        registry.register(_FsReadTool())

        result = registry.execute("fs_read", path="/tmp/some_file.txt")

        assert result.success is True

    def test_policy_engine_not_initialised_denies_privileged_tool(
        self, registry: ToolRegistry
    ) -> None:
        """Fail-closed: when no policy engine is installed, privileged tools are denied."""
        import missy.policy.engine as engine_mod

        original = engine_mod._engine
        try:
            engine_mod._engine = None
            registry.register(_ShellTool())
            result = registry.execute("shell_exec", command="ls")
            assert result.success is False
        finally:
            engine_mod._engine = original


# ---------------------------------------------------------------------------
# 12. Tool execution audit trail
# ---------------------------------------------------------------------------


class TestToolExecutionAuditTrail:
    """Every execute() call emits a structured audit event on the event bus."""

    def test_successful_execution_emits_allow_event(self, registry: ToolRegistry) -> None:
        registry.register(_EchoTool())
        registry.execute("echo", session_id="sess-1", task_id="task-1", text="hi")

        events = event_bus.get_events(event_type="tool_execute")
        assert len(events) >= 1
        allow_events = [e for e in events if e.result == "allow"]
        assert len(allow_events) == 1
        evt = allow_events[0]
        assert evt.session_id == "sess-1"
        assert evt.task_id == "task-1"
        assert evt.detail["tool"] == "echo"

    def test_failed_tool_emits_error_event(self, registry: ToolRegistry) -> None:
        failing = MagicMock(spec=BaseTool)
        failing.name = "broken"
        failing.permissions = ToolPermissions()
        failing.execute.return_value = ToolResult(
            success=False, output=None, error="deliberate failure"
        )
        registry.register(failing)

        registry.execute("broken", session_id="s2", task_id="t2")

        events = event_bus.get_events(event_type="tool_execute")
        assert any(e.result == "error" for e in events)

    def test_tool_exception_emits_error_event(self, registry: ToolRegistry) -> None:
        crashing = MagicMock(spec=BaseTool)
        crashing.name = "crasher"
        crashing.permissions = ToolPermissions()
        crashing.execute.side_effect = RuntimeError("unexpected boom")
        registry.register(crashing)

        result = registry.execute("crasher")

        assert result.success is False
        events = event_bus.get_events(event_type="tool_execute")
        assert any(e.result == "error" for e in events)

    def test_policy_denied_tool_emits_deny_event(self, registry: ToolRegistry) -> None:
        config = _make_config(shell_enabled=False)
        init_policy_engine(config)
        registry.register(_ShellTool())

        registry.execute("shell_exec", session_id="s3", task_id="t3", command="rm -rf /")

        events = event_bus.get_events(event_type="tool_execute")
        deny_events = [e for e in events if e.result == "deny"]
        assert len(deny_events) >= 1
        assert deny_events[0].detail["tool"] == "shell_exec"

    def test_multiple_executions_each_emit_audit_event(self, registry: ToolRegistry) -> None:
        registry.register(_EchoTool())

        for i in range(5):
            registry.execute("echo", session_id=f"s{i}", task_id=f"t{i}", text=f"msg-{i}")

        events = event_bus.get_events(event_type="tool_execute")
        assert len(events) == 5

    def test_audit_event_category_is_plugin(self, registry: ToolRegistry) -> None:
        registry.register(_EchoTool())
        registry.execute("echo", text="audit-check")

        events = event_bus.get_events(event_type="tool_execute")
        assert all(e.category == "plugin" for e in events)

    def test_audit_event_detail_contains_tool_name(self, registry: ToolRegistry) -> None:
        registry.register(_EchoTool())
        registry.execute("echo", text="detail-check")

        events = event_bus.get_events(event_type="tool_execute")
        assert events[0].detail["tool"] == "echo"

    def test_mcp_digest_mismatch_emits_security_event(
        self, manager: McpManager, tmp_mcp_config: str
    ) -> None:
        """Reconfirms the security event emitted by digest mismatch (cross-subsystem check)."""
        tools = [{"name": "op", "description": "Op"}]
        config_path = Path(tmp_mcp_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps([{"name": "srv", "command": "c", "digest": "sha256:badhash"}])
        )
        mock_client = _make_mock_client(tools=tools)

        with (
            patch("missy.mcp.manager.McpClient", return_value=mock_client),
            pytest.raises(ValueError),
        ):
            manager.add_server("srv", command="c")

        events = event_bus.get_events(event_type="mcp.digest_mismatch")
        assert len(events) == 1
        assert events[0].category == "security"
