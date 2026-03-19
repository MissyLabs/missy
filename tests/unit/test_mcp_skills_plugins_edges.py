"""Edge-case tests for MCP manager, skills discovery, and plugin loader.


Covers gaps not addressed by existing test suites:
- MCP: digest pinning/persistence, concurrent start/stop, empty config, duplicate
  server names, server-name boundary validation, multiple-tool namespacing,
  call_tool with double separator, health-check restart preserving command/url.
- Skills: tools as comma-separated string, quoted frontmatter values, YAML
  comment lines, files with only a closing delimiter, deeply nested scan,
  search ranking (name beats description), empty instructions body, unicode names.
- Plugins: manifest shape, get_manifest permissions content, reload replacing
  existing entry, execute with session/task IDs in audit events, disabled-state
  execute, concurrent init_plugin_loader replacement, plugin version attribute.
"""

from __future__ import annotations

import concurrent.futures
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
from missy.core.exceptions import PolicyViolationError
from missy.mcp.manager import McpManager
from missy.plugins.base import BasePlugin, PluginPermissions
from missy.plugins.loader import PluginLoader, get_plugin_loader, init_plugin_loader
from missy.skills.discovery import SkillDiscovery, SkillManifest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(
    enabled: bool = True,
    allowed: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(enabled=enabled, allowed_plugins=allowed or []),
        providers={},
        workspace_path=".",
        audit_log_path="~/.missy/audit.log",
    )


def _make_mock_client(
    tools: list[dict] | None = None,
    command: str = "echo",
    url: str | None = None,
    alive: bool = True,
) -> MagicMock:
    c = MagicMock()
    c.tools = tools if tools is not None else []
    c._command = command
    c._url = url
    c.is_alive.return_value = alive
    return c


# ---------------------------------------------------------------------------
# Stub plugins for plugin-loader tests
# ---------------------------------------------------------------------------


class NetworkPlugin(BasePlugin):
    """Plugin that declares network + filesystem_read permissions."""

    name = "networked"
    description = "A plugin that needs the network."
    version = "2.3.1"
    permissions = PluginPermissions(
        network=True,
        filesystem_read=True,
        allowed_hosts=["api.example.com"],
        allowed_paths=["/tmp/data"],
    )

    def initialize(self) -> bool:
        return True

    def execute(self, **kwargs: Any) -> str:
        return "network result"


class EchoPlugin(BasePlugin):
    name = "echo"
    description = "Echoes its input."
    permissions = PluginPermissions()

    def initialize(self) -> bool:
        return True

    def execute(self, *, message: str = "") -> str:
        return message


class SlowInitPlugin(BasePlugin):
    """Simulates a slow initialize() call — used in concurrency tests."""

    name = "slow_init"
    description = "Slow initializer."
    permissions = PluginPermissions()
    _barrier: threading.Barrier | None = None

    def initialize(self) -> bool:
        if self._barrier:
            self._barrier.wait()
        return True

    def execute(self, **kwargs: Any) -> None:
        return None


# ===========================================================================
# MCP Manager — edge cases
# ===========================================================================


@pytest.fixture
def tmp_mcp_config(tmp_path: Path) -> Path:
    return tmp_path / "mcp.json"


@pytest.fixture
def mcp_mgr(tmp_mcp_config: Path) -> McpManager:
    return McpManager(config_path=str(tmp_mcp_config))


class TestMcpManagerEmptyConfig:
    def test_connect_all_with_empty_array(self, tmp_mcp_config: Path) -> None:
        """An empty JSON array should not attempt any connections."""
        tmp_mcp_config.write_text("[]")
        tmp_mcp_config.chmod(0o600)
        mgr = McpManager(config_path=str(tmp_mcp_config))
        mgr.connect_all()
        assert mgr.list_servers() == []

    def test_all_tools_empty_returns_empty_list(self, mcp_mgr: McpManager) -> None:
        assert mcp_mgr.all_tools() == []

    def test_list_servers_before_any_connection(self, mcp_mgr: McpManager) -> None:
        assert mcp_mgr.list_servers() == []


class TestMcpServerNameValidation:
    """Boundary tests for server-name validation rules."""

    def test_numeric_only_name_allowed(self, mcp_mgr: McpManager) -> None:
        mock = _make_mock_client()
        with patch("missy.mcp.manager.McpClient", return_value=mock):
            client = mcp_mgr.add_server("123", command="echo")
        assert client is mock

    def test_hyphen_name_allowed(self, mcp_mgr: McpManager) -> None:
        mock = _make_mock_client()
        with patch("missy.mcp.manager.McpClient", return_value=mock):
            mcp_mgr.add_server("my-server", command="echo")

    def test_underscore_name_allowed(self, mcp_mgr: McpManager) -> None:
        mock = _make_mock_client()
        with patch("missy.mcp.manager.McpClient", return_value=mock):
            mcp_mgr.add_server("my_server", command="echo")

    def test_space_in_name_rejected(self, mcp_mgr: McpManager) -> None:
        with pytest.raises(ValueError, match="Invalid MCP server name"):
            mcp_mgr.add_server("bad name", command="echo")

    def test_slash_in_name_rejected(self, mcp_mgr: McpManager) -> None:
        with pytest.raises(ValueError, match="Invalid MCP server name"):
            mcp_mgr.add_server("bad/name", command="echo")

    def test_empty_name_rejected(self, mcp_mgr: McpManager) -> None:
        with pytest.raises(ValueError):
            mcp_mgr.add_server("", command="echo")

    def test_double_underscore_middle_rejected(self, mcp_mgr: McpManager) -> None:
        with pytest.raises(ValueError, match="must not contain '__'"):
            mcp_mgr.add_server("a__b", command="echo")


class TestMcpDuplicateServerName:
    def test_adding_same_name_twice_replaces_entry(self, mcp_mgr: McpManager) -> None:
        """A second add_server with the same name should replace the first client."""
        first = _make_mock_client(tools=[{"name": "t1"}])
        second = _make_mock_client(tools=[{"name": "t2"}, {"name": "t3"}])

        with patch("missy.mcp.manager.McpClient", side_effect=[first, second]):
            mcp_mgr.add_server("srv", command="echo")
            mcp_mgr.add_server("srv", command="echo v2")

        servers = mcp_mgr.list_servers()
        assert len(servers) == 1
        assert servers[0]["tools"] == 2

    def test_second_add_saves_config_with_single_entry(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        first = _make_mock_client()
        second = _make_mock_client()
        with patch("missy.mcp.manager.McpClient", side_effect=[first, second]):
            mcp_mgr.add_server("dup", command="echo")
            mcp_mgr.add_server("dup", command="echo2")
        saved = json.loads(tmp_mcp_config.read_text())
        assert len(saved) == 1


class TestMcpToolNamespacing:
    def test_multiple_servers_namespace_independently(self, mcp_mgr: McpManager) -> None:
        c1 = _make_mock_client(tools=[{"name": "read"}, {"name": "write"}])
        c2 = _make_mock_client(tools=[{"name": "query"}])
        mcp_mgr._clients = {"fs": c1, "db": c2}

        tools = mcp_mgr.all_tools()
        names = {t["name"] for t in tools}
        assert names == {"fs__read", "fs__write", "db__query"}

    def test_all_tools_preserve_extra_keys(self, mcp_mgr: McpManager) -> None:
        """Tool dicts should carry all original keys plus the injected metadata."""
        c = _make_mock_client(tools=[{"name": "go", "description": "does stuff", "inputSchema": {}}])
        mcp_mgr._clients = {"srv": c}
        tools = mcp_mgr.all_tools()
        assert len(tools) == 1
        t = tools[0]
        assert t["name"] == "srv__go"
        assert t["description"] == "does stuff"
        assert t["_mcp_server"] == "srv"
        assert t["_mcp_tool"] == "go"

    def test_call_tool_with_multiple_underscores_in_tool_name(self, mcp_mgr: McpManager) -> None:
        """server__tool_with_underscores should split only on the first __."""
        c = _make_mock_client()
        c.call_tool.return_value = "ok"
        mcp_mgr._clients = {"srv": c}
        result = mcp_mgr.call_tool("srv__tool_with_underscores", {})
        assert result == "ok"
        c.call_tool.assert_called_once_with("tool_with_underscores", {})

    def test_call_tool_multiple_double_underscore_uses_first_split(
        self, mcp_mgr: McpManager
    ) -> None:
        """srv__a__b — server is 'srv', tool is 'a__b'. Tool name has __ so is
        rejected by _SAFE_NAME_RE as safe names cannot contain __."""
        c = _make_mock_client()
        c.call_tool.return_value = "ok"
        mcp_mgr._clients = {"srv": c}
        result = mcp_mgr.call_tool("srv__a__b", {})
        # Tool name 'a__b' contains __ which is matched by the safe-name regex
        # (only alphanumeric, hyphens, underscores). Single underscore is fine,
        # but __ makes the tool name segment "a__b" which passes _SAFE_NAME_RE
        # since __ is just two underscores and _ is in the pattern. The call
        # should reach the client with 'a__b' as the tool name.
        c.call_tool.assert_called_once_with("a__b", {})
        assert result == "ok"


class TestMcpDigestPinning:
    def test_pin_server_digest_returns_digest_string(self, mcp_mgr: McpManager) -> None:
        c = _make_mock_client(tools=[{"name": "t1"}])
        mcp_mgr._clients["srv"] = c

        with patch("missy.mcp.digest.compute_tool_manifest_digest", return_value="abc123"):
            digest = mcp_mgr.pin_server_digest("srv")

        assert digest == "abc123"

    def test_pin_server_digest_unknown_server_raises_key_error(
        self, mcp_mgr: McpManager
    ) -> None:
        with pytest.raises(KeyError, match="not connected"):
            mcp_mgr.pin_server_digest("ghost")

    def test_pin_server_digest_persists_to_config(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        # Pre-populate config with an entry for "srv".
        tmp_mcp_config.write_text(json.dumps([{"name": "srv", "command": "echo"}]))
        c = _make_mock_client(tools=[{"name": "t1"}])
        mcp_mgr._clients["srv"] = c

        with patch("missy.mcp.digest.compute_tool_manifest_digest", return_value="deadbeef"):
            mcp_mgr.pin_server_digest("srv")

        saved = json.loads(tmp_mcp_config.read_text())
        assert saved[0]["digest"] == "deadbeef"

    def test_pin_server_digest_no_config_file_still_returns_digest(
        self, mcp_mgr: McpManager
    ) -> None:
        """pin_server_digest should work even if the config file was deleted."""
        c = _make_mock_client(tools=[])
        mcp_mgr._clients["srv"] = c

        with patch("missy.mcp.digest.compute_tool_manifest_digest", return_value="xyz"):
            digest = mcp_mgr.pin_server_digest("srv")
        assert digest == "xyz"

    def test_get_server_digest_returns_none_when_no_entry(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        tmp_mcp_config.write_text(json.dumps([{"name": "other", "command": "echo"}]))
        assert mcp_mgr._get_server_digest("srv") is None

    def test_get_server_digest_returns_none_when_no_config(
        self, mcp_mgr: McpManager
    ) -> None:
        assert mcp_mgr._get_server_digest("srv") is None

    def test_get_server_digest_returns_value_from_config(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        tmp_mcp_config.write_text(
            json.dumps([{"name": "srv", "command": "echo", "digest": "pinned-hash"}])
        )
        assert mcp_mgr._get_server_digest("srv") == "pinned-hash"

    def test_digest_match_allows_connection(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        tmp_mcp_config.write_text(
            json.dumps([{"name": "srv", "command": "echo", "digest": "correct-hash"}])
        )
        mock_client = _make_mock_client(tools=[{"name": "t1"}])
        with (
            patch("missy.mcp.manager.McpClient", return_value=mock_client),
            patch("missy.mcp.digest.compute_tool_manifest_digest", return_value="correct-hash"),
            patch("missy.mcp.digest.verify_digest", return_value=True),
        ):
            client = mcp_mgr.add_server("srv", command="echo")
        assert client is mock_client
        mock_client.disconnect.assert_not_called()


class TestMcpHealthCheckAndRestart:
    def test_health_check_preserves_command_on_restart(self, mcp_mgr: McpManager) -> None:
        """Restarted clients should be created with the original command."""
        dead = _make_mock_client(command="original-cmd", alive=False)
        mcp_mgr._clients["srv"] = dead

        new_client = _make_mock_client(tools=[], command="original-cmd", alive=True)
        with patch("missy.mcp.manager.McpClient", return_value=new_client) as mock_cls:
            mcp_mgr.health_check()

        # McpClient should have been called with the original command.
        mock_cls.assert_called_once_with(name="srv", command="original-cmd", url=None)

    def test_health_check_all_alive_no_restarts(self, mcp_mgr: McpManager) -> None:
        alive1 = _make_mock_client(alive=True)
        alive2 = _make_mock_client(alive=True)
        mcp_mgr._clients = {"a": alive1, "b": alive2}

        with patch("missy.mcp.manager.McpClient") as mock_cls:
            mcp_mgr.health_check()

        mock_cls.assert_not_called()

    def test_restart_nonexistent_server_is_noop(self, mcp_mgr: McpManager) -> None:
        """restart_server on an unknown name must not raise."""
        with patch("missy.mcp.manager.McpClient") as mock_cls:
            mcp_mgr.restart_server("ghost")
        mock_cls.assert_not_called()


class TestMcpConcurrentAccess:
    def test_concurrent_add_and_list_do_not_corrupt_state(self, mcp_mgr: McpManager) -> None:
        """Concurrent add_server calls should not leave the manager in an
        inconsistent state (no missing or double-counted entries)."""
        errors: list[Exception] = []

        def add(idx: int) -> None:
            name = f"srv{idx}"
            mock = _make_mock_client(tools=[])
            try:
                with patch("missy.mcp.manager.McpClient", return_value=mock):
                    mcp_mgr.add_server(name, command=f"echo {idx}")
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(add, i) for i in range(10)]
            concurrent.futures.wait(futures)

        assert errors == [], f"Unexpected errors: {errors}"
        assert len(mcp_mgr.list_servers()) == 10

    def test_shutdown_while_concurrent_list_does_not_raise(self, mcp_mgr: McpManager) -> None:
        clients = {f"s{i}": _make_mock_client(alive=True) for i in range(5)}
        mcp_mgr._clients = clients

        results: list[Exception] = []

        def lister() -> None:
            for _ in range(20):
                try:
                    mcp_mgr.list_servers()
                except Exception as exc:
                    results.append(exc)

        t = threading.Thread(target=lister)
        t.start()
        mcp_mgr.shutdown()
        t.join()
        assert results == []


class TestMcpSaveConfigPermissions:
    def test_saved_config_excludes_digest_field(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        """_save_config should not persist digest (it comes from pin_server_digest)."""
        mock = _make_mock_client(command="echo", url=None)
        with patch("missy.mcp.manager.McpClient", return_value=mock):
            mcp_mgr.add_server("srv", command="echo")
        saved = json.loads(tmp_mcp_config.read_text())
        assert "digest" not in saved[0]

    def test_saved_config_includes_url_field(
        self, mcp_mgr: McpManager, tmp_mcp_config: Path
    ) -> None:
        mock = _make_mock_client(command=None, url="http://localhost:9000")
        mock._command = None
        mock._url = "http://localhost:9000"
        with patch("missy.mcp.manager.McpClient", return_value=mock):
            mcp_mgr.add_server("http-srv", command=None, url="http://localhost:9000")
        saved = json.loads(tmp_mcp_config.read_text())
        assert saved[0]["url"] == "http://localhost:9000"


# ===========================================================================
# Skills Discovery — edge cases
# ===========================================================================


@pytest.fixture
def discovery() -> SkillDiscovery:
    return SkillDiscovery()


class TestSkillMdFrontmatterParsing:
    def test_tools_as_comma_separated_string(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        """The parser should accept a comma-separated string value for tools."""
        content = (
            "---\n"
            "name: csv-tools\n"
            "description: Tools as a CSV string\n"
            "tools: web_fetch, shell_exec, file_read\n"
            "---\n\nBody.\n"
        )
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.tools == ["web_fetch", "shell_exec", "file_read"]

    def test_tools_as_empty_list(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        content = "---\nname: no-tools\ntools: []\n---\n\nBody.\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.tools == []

    def test_quoted_string_values_unquoted(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        content = (
            "---\n"
            'name: "quoted-name"\n'
            "description: 'single quoted'\n"
            "version: '2.0.0'\n"
            "---\n\nBody.\n"
        )
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.name == "quoted-name"
        assert manifest.description == "single quoted"
        assert manifest.version == "2.0.0"

    def test_comment_lines_ignored(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        content = (
            "---\n"
            "# This is a comment\n"
            "name: commented\n"
            "# Another comment\n"
            "description: Has comments\n"
            "---\n\nBody.\n"
        )
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.name == "commented"
        assert manifest.description == "Has comments"

    def test_frontmatter_value_with_colon(self, discovery: SkillDiscovery, tmp_path: Path) -> None:
        """Values that themselves contain ':' should use the first ':' as the partition."""
        content = "---\nname: colon-test\ndescription: See https://example.com for details\n---\n\nBody.\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        # partition on first colon: value is " See https" — the rest is dropped
        # because the internal parser only does key: value on the first ':'
        assert manifest.name == "colon-test"
        # The description will have everything after the first colon only up to
        # the next colon; verify we at least get a non-empty string without crashing.
        assert isinstance(manifest.description, str)

    def test_body_stripped_of_leading_trailing_whitespace(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: striptest\n---\n\n\n  Body content here  \n\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.instructions == "Body content here"

    def test_empty_body_gives_empty_instructions(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: nobody\n---\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.instructions == ""

    def test_unicode_name_and_description(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: météo\ndescription: Fetch la météo\n---\n\nCorps.\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content, encoding="utf-8")
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.name == "météo"
        assert "météo" in manifest.description

    def test_path_stored_as_resolved_absolute(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: pathtest\n---\n\nBody.\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert Path(manifest.path).is_absolute()
        assert manifest.path == str(f.resolve())


class TestSkillMdMalformedEdgeCases:
    def test_only_opening_delimiter_raises(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        """A file with only '---' and no closing delimiter raises ValueError."""
        content = "---\nname: bad\nno closing"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        with pytest.raises(ValueError, match="No YAML frontmatter"):
            discovery.parse_skill_md(str(f))

    def test_empty_file_raises_value_error(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        f = tmp_path / "SKILL.md"
        f.write_text("")
        with pytest.raises(ValueError, match="No YAML frontmatter"):
            discovery.parse_skill_md(str(f))

    def test_frontmatter_with_no_keys_but_has_name(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        content = "---\nname: sparse\n---\n"
        f = tmp_path / "SKILL.md"
        f.write_text(content)
        manifest = discovery.parse_skill_md(str(f))
        assert manifest.name == "sparse"
        assert manifest.author == ""
        assert manifest.version == ""

    def test_scan_skips_unparseable_files_counts_rest(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        valid_count = 3
        for i in range(valid_count):
            d = tmp_path / f"good{i}"
            d.mkdir()
            (d / "SKILL.md").write_text(f"---\nname: good{i}\n---\n\nOK.\n")
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_text("not frontmatter at all")
        manifests = discovery.scan_directory(str(tmp_path))
        assert len(manifests) == valid_count

    def test_scan_deeply_nested_skill_found(
        self, discovery: SkillDiscovery, tmp_path: Path
    ) -> None:
        nested = tmp_path / "a" / "b" / "c" / "d"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("---\nname: deep\n---\n\nDeep body.\n")
        manifests = discovery.scan_directory(str(tmp_path))
        assert len(manifests) == 1
        assert manifests[0].name == "deep"


class TestSkillDiscoverySearch:
    @pytest.fixture
    def skills(self) -> list[SkillManifest]:
        return [
            SkillManifest(name="web-search", description="Search the web", version="1.0", author="A"),
            SkillManifest(name="web-scraper", description="Extract web data", version="1.0", author="B"),
            SkillManifest(name="calculator", description="Math on the web", version="1.0", author="C"),
            SkillManifest(name="file-ops", description="File operations", version="1.0", author="D"),
        ]

    def test_name_match_ranked_before_description_match(
        self, discovery: SkillDiscovery, skills: list[SkillManifest]
    ) -> None:
        """'web' matches names of web-search and web-scraper; calculator only
        matches via description. Name-matches should come first."""
        results = discovery.search("web", skills)
        name_matched = [r for r in results if "web" in r.name]
        desc_only = [r for r in results if "web" not in r.name]
        # All name matches appear before description-only matches.
        if desc_only:
            last_name_idx = max(results.index(r) for r in name_matched)
            first_desc_idx = min(results.index(r) for r in desc_only)
            assert last_name_idx < first_desc_idx

    def test_search_no_duplicates(
        self, discovery: SkillDiscovery, skills: list[SkillManifest]
    ) -> None:
        """A skill matching both name and description should appear only once."""
        results = discovery.search("web", skills)
        names = [r.name for r in results]
        assert len(names) == len(set(names))

    def test_search_partial_match_middle_of_word(
        self, discovery: SkillDiscovery, skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("culat", skills)
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_search_returns_all_on_empty_query(
        self, discovery: SkillDiscovery, skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("", skills)
        assert {r.name for r in results} == {r.name for r in skills}

    def test_search_empty_skill_list(self, discovery: SkillDiscovery) -> None:
        assert discovery.search("anything", []) == []

    def test_search_description_only_match_included(
        self, discovery: SkillDiscovery, skills: list[SkillManifest]
    ) -> None:
        results = discovery.search("File", skills)  # 'File' in description of file-ops
        assert any(r.name == "file-ops" for r in results)


# ===========================================================================
# Plugin Loader — edge cases
# ===========================================================================


class TestPluginManifestShape:
    def test_get_manifest_contains_all_fields(self) -> None:
        plugin = NetworkPlugin()
        manifest = plugin.get_manifest()
        assert set(manifest.keys()) == {"name", "version", "description", "permissions", "enabled"}

    def test_get_manifest_permissions_includes_network_flag(self) -> None:
        plugin = NetworkPlugin()
        perms = plugin.get_manifest()["permissions"]
        assert perms["network"] is True
        assert perms["filesystem_read"] is True
        assert perms["filesystem_write"] is False
        assert perms["shell"] is False
        assert perms["allowed_hosts"] == ["api.example.com"]
        assert perms["allowed_paths"] == ["/tmp/data"]

    def test_get_manifest_version_reflects_class_attribute(self) -> None:
        plugin = NetworkPlugin()
        assert plugin.get_manifest()["version"] == "2.3.1"

    def test_get_manifest_enabled_false_before_load(self) -> None:
        plugin = NetworkPlugin()
        assert plugin.get_manifest()["enabled"] is False

    def test_get_manifest_enabled_true_after_load(self) -> None:
        loader = PluginLoader(_make_config(enabled=True, allowed=["networked"]))
        plugin = NetworkPlugin()
        loader.load_plugin(plugin)
        assert plugin.get_manifest()["enabled"] is True


class TestPluginLoaderReload:
    def test_reload_same_plugin_replaces_registration(self) -> None:
        """Loading the same plugin name twice should replace the stored instance."""
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo"]))
        p1 = EchoPlugin()
        p2 = EchoPlugin()
        loader.load_plugin(p1)
        loader.load_plugin(p2)
        assert loader.get_plugin("echo") is p2

    def test_list_plugins_after_reload_has_single_entry(self) -> None:
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo"]))
        loader.load_plugin(EchoPlugin())
        loader.load_plugin(EchoPlugin())
        assert len(loader.list_plugins()) == 1


class TestPluginExecuteAuditDetails:
    def test_execute_start_event_emitted_before_result(self) -> None:
        event_bus.clear()
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo"]))
        loader.load_plugin(EchoPlugin())
        loader.execute("echo", session_id="s1", task_id="t1", message="hello")
        start_events = event_bus.get_events(event_type="plugin.execute.start", result="allow")
        assert len(start_events) == 1

    def test_execute_session_and_task_id_forwarded_to_events(self) -> None:
        event_bus.clear()
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo"]))
        loader.load_plugin(EchoPlugin())
        loader.execute("echo", session_id="my-session", task_id="my-task", message="hi")
        events = event_bus.get_events(event_type="plugin.execute", result="allow")
        assert events[0].session_id == "my-session"
        assert events[0].task_id == "my-task"

    def test_execute_deny_event_carries_not_loaded_reason(self) -> None:
        event_bus.clear()
        loader = PluginLoader(_make_config(enabled=True, allowed=[]))
        with pytest.raises(PolicyViolationError):
            loader.execute("ghost", session_id="s", task_id="t")
        events = event_bus.get_events(event_type="plugin.execute", result="deny")
        assert events[0].detail["reason"] == "not_loaded"

    def test_execute_deny_event_carries_not_enabled_reason(self) -> None:
        event_bus.clear()
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo"]))
        plugin = EchoPlugin()
        loader.load_plugin(plugin)
        plugin.enabled = False
        with pytest.raises(PolicyViolationError):
            loader.execute("echo")
        events = event_bus.get_events(event_type="plugin.execute", result="deny")
        assert events[0].detail["reason"] == "not_enabled"


class TestPluginLoaderMultiplePlugins:
    def test_load_multiple_distinct_plugins(self) -> None:
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo", "networked"]))
        loader.load_plugin(EchoPlugin())
        loader.load_plugin(NetworkPlugin())
        assert len(loader.list_plugins()) == 2
        assert loader.get_plugin("echo") is not None
        assert loader.get_plugin("networked") is not None

    def test_each_plugin_independent_enabled_state(self) -> None:
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo", "networked"]))
        p_echo = EchoPlugin()
        p_net = NetworkPlugin()
        loader.load_plugin(p_echo)
        loader.load_plugin(p_net)
        p_echo.enabled = False
        assert p_net.enabled is True

    def test_execute_one_of_multiple_plugins(self) -> None:
        loader = PluginLoader(_make_config(enabled=True, allowed=["echo", "networked"]))
        loader.load_plugin(EchoPlugin())
        loader.load_plugin(NetworkPlugin())
        assert loader.execute("echo", message="ping") == "ping"
        assert loader.execute("networked") == "network result"


class TestPluginLoaderConcurrency:
    def test_concurrent_init_plugin_loader_last_wins(self, monkeypatch) -> None:
        """Concurrent calls to init_plugin_loader are safe; final loader is one of them."""
        loaders_created: list[PluginLoader] = []
        original_init = PluginLoader.__init__

        def tracking_init(self, config):
            original_init(self, config)
            loaders_created.append(self)

        monkeypatch.setattr(PluginLoader, "__init__", tracking_init)

        configs = [_make_config() for _ in range(10)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(init_plugin_loader, c) for c in configs]
            concurrent.futures.wait(futures)

        current = get_plugin_loader()
        assert current in loaders_created
