"""Focused regressions for the issue-145 security/reliability audit."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.agent.checkpoint import CheckpointManager
from missy.channels.discord.config import guild_mode_to_capability
from missy.channels.discord.voice import DiscordVoiceManager, _GuildVoiceState, _make_sink_class
from missy.channels.voice.server import VoiceServer
from missy.config.hotreload import ConfigWatcher, _is_security_widening, candidate_config_digest
from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
    ToolPolicyConfig,
    load_config,
)
from missy.mcp.client import McpClient
from missy.mcp.digest import compute_tool_manifest_digest
from missy.mcp.manager import McpManager
from missy.mcp.tool_wrapper import McpToolWrapper
from missy.memory.resilient import ResilientMemoryStore
from missy.memory.sqlite_store import ConversationTurn as SQLiteConversationTurn
from missy.memory.sqlite_store import SQLiteMemoryStore
from missy.memory.store import MemoryStore
from missy.policy.filesystem import FilesystemPolicyEngine
from missy.policy.shell import ShellPolicyEngine, ShellRedirectParseError
from missy.security.sandbox import FallbackSandbox, SandboxConfig
from missy.tools.builtin.shell_exec import ShellExecTool
from missy.tools.registry import ToolRegistry


def test_discord_guild_modes_map_explicitly() -> None:
    assert guild_mode_to_capability("no_tools") == "no-tools"
    assert guild_mode_to_capability("safe_chat_only") == "safe-chat"
    assert guild_mode_to_capability("full") == "discord"
    with pytest.raises(ValueError):
        guild_mode_to_capability("typo")


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("no_tools", "no-tools"), ("safe_chat_only", "safe-chat"), ("full", "discord")],
)
def test_slash_ask_forwards_guild_capability(mode: str, expected: str) -> None:
    import asyncio

    from missy.channels.discord import commands

    agent = SimpleNamespace(run=MagicMock(return_value="ok"))
    channel = SimpleNamespace(
        _agent_runtime=agent,
        _emit_audit=MagicMock(),
        account_config=SimpleNamespace(guild_policies={"g": SimpleNamespace(mode=mode)}),
    )
    interaction = {
        "data": {"name": "ask", "options": [{"name": "prompt", "value": "hello"}]},
        "member": {"user": {"id": "123456789012345"}},
        "guild_id": "g",
        "channel_id": "c",
    }
    with patch("missy.security.secrets.SecretsDetector.has_secrets", return_value=False):
        result = asyncio.run(
            commands.handle_slash_command(interaction, channel, capability_mode=expected)
        )
    assert result == "ok"
    assert agent.run.call_args.kwargs["_capability_mode"] == expected


def test_mcp_protocol_is_error_survives_string_rendering() -> None:
    client = McpClient(name="test", command="echo")
    client._rpc = MagicMock(
        return_value={"result": {"isError": True, "content": [{"type": "text", "text": "failed"}]}}
    )
    result = client.call_tool("bad", {})
    assert str(result) == "failed"
    assert result.is_error is True

    manager = MagicMock()
    manager.call_tool.return_value = result
    wrapped = McpToolWrapper(manager, "srv__bad", "", {}, MagicMock())
    assert wrapped.execute().success is False


def test_mcp_uncertain_transport_failure_is_typed_and_not_raised() -> None:
    client = McpClient(name="test", command="echo")
    client._rpc = MagicMock(side_effect=TimeoutError("lost after send"))
    result = client.call_tool("mutate", {})
    assert result.is_error is True
    assert result.error_kind == "transport"
    assert result.transport_certainty == "uncertain"


def test_mcp_missing_and_invalid_annotations_are_explicitly_diagnostic() -> None:
    client = McpClient(name="test", command="echo")
    client._rpc = MagicMock(
        return_value={
            "result": {
                "tools": [
                    {"name": "missing", "inputSchema": {}},
                    {"name": "invalid", "annotations": "unsafe", "inputSchema": {}},
                    {"name": "declared", "annotations": {}, "inputSchema": {}},
                ]
            }
        }
    )
    client._tools = client._list_tools()
    assert client.tool_annotation_states == {
        "missing": "missing",
        "invalid": "invalid",
        "declared": "declared",
    }
    assert all(annotation.requires_approval for annotation in client.tool_annotations.values())


def test_mcp_hostile_read_only_hint_does_not_remove_approval(tmp_path: Path) -> None:
    from missy.mcp.annotations import ToolAnnotation

    manager = McpManager(config_path=str(tmp_path / "mcp.json"))
    client = MagicMock()
    client._command = "echo"
    client._url = None
    client._headers = None
    client._allow_insecure_auth = False
    client.tools = [{"name": "exfiltrate", "inputSchema": {}}]
    client.tool_annotations = {
        "exfiltrate": ToolAnnotation.from_mcp_dict({"readOnlyHint": True})
    }
    client.tool_annotation_states = {"exfiltrate": "declared"}
    client.connect.return_value = None
    with patch("missy.mcp.manager.McpClient", return_value=client):
        manager.add_server("hostile", command="echo", persist=False)
    annotation = manager.get_annotation("hostile__exfiltrate")
    assert annotation is not None
    assert annotation.requires_approval is True


def test_mcp_digest_covers_schema_and_annotations() -> None:
    base = [{"name": "x", "description": "x", "inputSchema": {"type": "object"}}]
    changed = [
        {
            "name": "x",
            "description": "x",
            "inputSchema": {"type": "object", "required": ["dangerous"]},
        }
    ]
    assert compute_tool_manifest_digest(base) != compute_tool_manifest_digest(changed)
    assert compute_tool_manifest_digest(base).startswith("sha256:v2:")
    assert compute_tool_manifest_digest(base * 2) == compute_tool_manifest_digest(
        list(reversed(base * 2))
    )
    complete = [
        {
            "name": "x",
            "description": "x",
            "inputSchema": {"properties": {"a": {"type": "string"}}, "type": "object"},
            "outputSchema": {"type": "string"},
            "annotations": {"readOnlyHint": True},
        },
        {"name": "a", "description": "a", "inputSchema": {}},
    ]
    reordered_keys = [
        {"inputSchema": {}, "description": "a", "name": "a"},
        {
            "annotations": {"readOnlyHint": True},
            "outputSchema": {"type": "string"},
            "inputSchema": {"type": "object", "properties": {"a": {"type": "string"}}},
            "description": "x",
            "name": "x",
        },
    ]
    assert compute_tool_manifest_digest(complete) == compute_tool_manifest_digest(reordered_keys)
    changed_output = json.loads(json.dumps(complete))
    changed_output[0]["outputSchema"] = {"type": "number"}
    assert compute_tool_manifest_digest(complete) != compute_tool_manifest_digest(changed_output)
    changed_annotation = json.loads(json.dumps(complete))
    changed_annotation[0]["annotations"]["readOnlyHint"] = False
    assert compute_tool_manifest_digest(complete) != compute_tool_manifest_digest(
        changed_annotation
    )


def test_mcp_save_preserves_offline_desired_server(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            [
                {"name": "online", "command": "old"},
                {"name": "offline", "command": "missing", "digest": "sha256:v2:pinned"},
            ]
        )
    )
    manager = McpManager(config_path=str(path))
    client = MagicMock(_command="new", _url=None)
    manager._clients["online"] = client
    manager._save_config()
    saved = {entry["name"]: entry for entry in json.loads(path.read_text())}
    assert set(saved) == {"online", "offline"}
    assert saved["offline"]["digest"] == "sha256:v2:pinned"


def test_mcp_startup_connections_never_rewrite_desired_config(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    original = json.dumps(
        [
            {"name": "offline", "command": "missing", "digest": "sha256:v2:keep"},
            {"name": "online", "command": "ok"},
        ],
        indent=2,
    )
    path.write_text(original)
    manager = McpManager(config_path=str(path))
    with patch.object(
        manager, "add_server", side_effect=[RuntimeError("offline"), MagicMock()]
    ) as add:
        manager.connect_all()
    assert path.read_text() == original
    assert all(call.kwargs["persist"] is False for call in add.call_args_list)


def test_mcp_wrapper_reconciliation_adds_and_removes_tools() -> None:
    from missy.agent.runtime import AgentRuntime

    runtime = AgentRuntime.__new__(AgentRuntime)
    manager = MagicMock()
    manager.get_annotation.return_value = None
    manager.all_tools.return_value = [{"name": "srv__one", "description": "one", "inputSchema": {}}]
    runtime._mcp_manager = manager
    registry = ToolRegistry()
    runtime._sync_mcp_tools(registry)
    assert registry.get("srv__one") is not None

    manager.all_tools.return_value = [{"name": "srv__two", "description": "two", "inputSchema": {}}]
    runtime._sync_mcp_tools(registry)
    assert registry.get("srv__one") is None
    assert registry.get("srv__two") is not None


def test_mcp_broken_wrapper_does_not_block_unrelated_reconciliation() -> None:
    from missy.agent.runtime import AgentRuntime

    class FaultyRegistry(ToolRegistry):
        def register(self, tool) -> None:
            if tool.name == "a__broken":
                raise ValueError("broken wrapper")
            super().register(tool)

    runtime = AgentRuntime.__new__(AgentRuntime)
    manager = MagicMock()
    manager.get_annotation.return_value = None
    manager.all_tools.return_value = [
        {"name": "a__broken", "description": "bad", "inputSchema": {}},
        {"name": "b__good", "description": "good", "inputSchema": {}},
    ]
    runtime._mcp_manager = manager
    registry = FaultyRegistry()
    runtime._sync_mcp_tools(registry)
    assert registry.get("a__broken") is None
    assert registry.get("b__good") is not None


def test_mcp_reconciliation_preserves_operator_disabled_state() -> None:
    from missy.agent.runtime import AgentRuntime

    runtime = AgentRuntime.__new__(AgentRuntime)
    manager = MagicMock()
    manager.get_annotation.return_value = None
    manager.all_tools.return_value = [{"name": "srv__tool", "description": "v1", "inputSchema": {}}]
    runtime._mcp_manager = manager
    registry = ToolRegistry()
    runtime._sync_mcp_tools(registry)
    registry.disable("srv__tool")

    runtime._sync_mcp_tools(registry)
    assert registry.is_enabled("srv__tool") is False

    manager.all_tools.return_value = [
        {
            "name": "srv__tool",
            "description": "v2",
            "inputSchema": {"type": "object"},
        }
    ]
    runtime._sync_mcp_tools(registry)
    assert registry.get("srv__tool").description == "v2"
    assert registry.is_enabled("srv__tool") is False


def test_mcp_health_reconciles_removed_and_changed_desired_servers(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps([{"name": "srv", "command": "old"}]))
    path.chmod(0o600)
    manager = McpManager(config_path=str(path))
    old = MagicMock(_command="old", _url=None)
    old.is_alive.return_value = True
    manager._clients["srv"] = old
    manager._live_server_fingerprints["srv"] = manager._connection_fingerprint(
        manager._desired_servers["srv"]
    )

    path.write_text(json.dumps([{"name": "srv", "command": "new"}]))
    with patch.object(manager, "add_server", return_value=MagicMock()) as add:
        manager.health_check()
    old.disconnect.assert_called_once()
    assert add.call_args.kwargs["command"] == "new"
    assert add.call_args.kwargs["persist"] is False

    replacement = MagicMock(_command="new", _url=None)
    replacement.is_alive.return_value = True
    manager._clients["srv"] = replacement
    manager._desired_servers = {"srv": {"name": "srv", "command": "new"}}
    path.write_text("[]")
    manager.health_check()
    replacement.disconnect.assert_called_once()
    assert "srv" not in manager._clients


def test_protected_path_overrides_parent_write_allow(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    engine = FilesystemPolicyEngine(
        FilesystemPolicy(
            allowed_write_paths=[str(tmp_path)],
            protected_write_paths=[str(config)],
        )
    )
    with pytest.raises(Exception, match="operator-protected"):
        engine.check_write(config)


def test_shell_subprocess_cannot_overwrite_operator_protected_file(tmp_path: Path) -> None:
    protected = tmp_path / "config.yaml"
    protected.write_text("safe")
    fake_engine = SimpleNamespace(
        filesystem=SimpleNamespace(protected_write_paths=(str(protected),))
    )
    with patch("missy.policy.engine.get_policy_engine", return_value=fake_engine):
        result = ShellExecTool().execute(
            command=(
                'python3 -c "from pathlib import Path; '
                f"Path({str(protected)!r}).write_text('owned')\""
            )
        )
    assert result.success is False
    assert protected.read_text() == "safe"


@pytest.mark.parametrize(
    "command",
    [
        "bash -o pipefail -c 'echo owned > /protected'",
        "bash -O extglob -c 'echo owned > /protected'",
        "bash --noprofile -c 'echo owned > /protected'",
        "sh -c 'echo owned > /protected'",
    ],
)
def test_shell_nested_launchers_fail_closed(command: str) -> None:
    engine = ShellPolicyEngine(ShellPolicy(enabled=True, allowed_commands=[]))
    with pytest.raises(ShellRedirectParseError, match="nested shell launchers"):
        engine.extract_redirect_targets(command)


def test_shell_fallback_sandbox_keeps_operator_protected_boundary(tmp_path: Path) -> None:
    protected = tmp_path / "config.yaml"
    protected.write_text("safe")
    fake_engine = SimpleNamespace(
        filesystem=SimpleNamespace(protected_write_paths=(str(protected),))
    )
    tool = ShellExecTool()
    tool._sandbox = FallbackSandbox(SandboxConfig(require_isolation=False))
    with patch("missy.policy.engine.get_policy_engine", return_value=fake_engine):
        result = tool.execute(
            command=(
                'python3 -c "from pathlib import Path; '
                f"Path({str(protected)!r}).write_text('owned')\""
            )
        )
    assert result.success is False
    assert protected.read_text() == "safe"


def _security_config(*, shell_enabled: bool, commands: list[str]) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(enabled=shell_enabled, allowed_commands=commands),
        plugins=PluginPolicy(),
        providers={},
        workspace_path=".",
        audit_log_path="audit.log",
        tools=ToolPolicyConfig(profile="minimal"),
    )


def test_hot_reload_distinguishes_narrowing_and_widening() -> None:
    old = _security_config(shell_enabled=True, commands=["git", "ls"])
    narrow = _security_config(shell_enabled=True, commands=["git"])
    wide = _security_config(shell_enabled=True, commands=["git", "ls", "rm"])
    assert _is_security_widening(old, narrow) is False
    assert _is_security_widening(old, wide) is True
    assert candidate_config_digest(wide) != candidate_config_digest(narrow)


def test_hot_reload_detects_removed_protection_and_landlock(tmp_path: Path) -> None:
    protected = str(tmp_path / "config.yaml")
    old = _security_config(shell_enabled=True, commands=["git"])
    new = _security_config(shell_enabled=True, commands=["git"])
    old.filesystem.protected_write_paths = [protected]
    new.filesystem.protected_write_paths = []
    assert _is_security_widening(old, new) is True

    old.filesystem.protected_write_paths = []
    old.landlock_enabled = True
    new.landlock_enabled = False
    assert _is_security_widening(old, new) is True


def test_hot_reload_approval_is_exact_and_one_time(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    watcher = ConfigWatcher(str(config_path), lambda _config: None)
    approval = tmp_path / "config.yaml.reload-approval"
    approval.write_text("sha256:exact")
    approval.chmod(0o600)
    assert watcher._consume_widening_approval("sha256:different") is False
    assert approval.exists()
    assert watcher._consume_widening_approval("sha256:exact") is True
    assert not approval.exists()


def test_hot_reload_rejects_and_audits_widening_at_reload_boundary(tmp_path: Path) -> None:
    from missy.core.events import event_bus

    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    config_path.chmod(0o600)
    callback = MagicMock()
    watcher = ConfigWatcher(str(config_path), callback)
    old = _security_config(shell_enabled=True, commands=["git"])
    wide = _security_config(shell_enabled=True, commands=["git", "rm"])
    watcher._active_config = old
    events = []
    event_bus.subscribe("config.reload_widening", events.append)
    try:
        with patch("missy.config.settings.load_config", return_value=wide):
            watcher._do_reload()
    finally:
        event_bus.unsubscribe("config.reload_widening", events.append)
    callback.assert_not_called()
    assert watcher._active_config is old
    assert len(events) == 1
    assert events[0].result == "deny"


def test_hot_reload_exact_approval_allows_candidate_once(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}")
    config_path.chmod(0o600)
    callback = MagicMock()
    watcher = ConfigWatcher(str(config_path), callback)
    watcher._active_config = _security_config(shell_enabled=True, commands=["git"])
    wide = _security_config(shell_enabled=True, commands=["git", "rm"])
    approval = tmp_path / "config.yaml.reload-approval"
    approval.write_text(candidate_config_digest(wide))
    approval.chmod(0o600)
    with patch("missy.config.settings.load_config", return_value=wide):
        watcher._do_reload()
    callback.assert_called_once_with(wide)
    assert watcher._active_config is wide
    assert not approval.exists()


def test_quoted_landlock_false_and_agent_policy_inheritance(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "landlock_enabled: 'false'\ntools:\n  profile: minimal\nagents:\n  default: {}\n"
    )
    config = load_config(str(path))
    assert config.landlock_enabled is False
    assert config.agents["default"].tools is None

    path.write_text("agents:\n  default:\n    profil: minimal\n")
    with pytest.raises(Exception, match="profil"):
        load_config(str(path))


def test_json_compaction_rejects_zero(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memory.json"))
    store.add_turn("s", "user", "hello")
    with pytest.raises(ValueError, match="at least 1"):
        store.compact_session("s", keep_recent=0)


def test_sqlite_files_are_private(tmp_path: Path) -> None:
    parent = tmp_path / "memory"
    parent.mkdir(mode=0o755)
    path = parent / "memory.db"
    store = SQLiteMemoryStore(str(path))
    store.add_turn(SQLiteConversationTurn.new("s", "user", "hello"))
    assert parent.stat().st_mode & 0o077 == 0
    assert path.stat().st_mode & 0o077 == 0
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            assert sidecar.stat().st_mode & 0o077 == 0


def test_retention_compares_timestamp_instants_not_lexical_offsets(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(str(path))
    old = datetime.now(UTC) - timedelta(days=90)
    equivalent = old.astimezone().isoformat()
    first = SQLiteConversationTurn.new("s", "user", "one")
    second = SQLiteConversationTurn.new("s", "user", "two")
    first.timestamp = old.isoformat().replace("+00:00", "Z")
    second.timestamp = equivalent
    store.add_turn(first)
    store.add_turn(second)
    assert store.cleanup(older_than_days=30, dry_run=True) == 2
    assert store.cleanup(older_than_days=30) == 2


def test_checkpoint_censors_secrets_and_uses_private_files(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.db"
    manager = CheckpointManager(str(path))
    secret = "AKIA" + "Z" * 16
    checkpoint_id = manager.create("s", "t", f"prompt {secret}")
    manager.update(
        checkpoint_id,
        [{"role": "assistant", "content": f"provider said {secret}"}],
        [],
        1,
    )
    conn = sqlite3.connect(path)
    prompt, messages = conn.execute(
        "SELECT prompt, loop_messages FROM checkpoints WHERE id = ?", (checkpoint_id,)
    ).fetchone()
    conn.close()
    assert secret not in prompt
    assert secret not in messages
    assert path.stat().st_mode & 0o077 == 0


def test_resilient_memory_quarantines_poison_and_replays_later_operation() -> None:
    from missy.core.events import event_bus

    primary = MagicMock()

    def add_turn(turn) -> None:
        if turn.id == "poison":
            raise ValueError("permanent")

    primary.add_turn.side_effect = add_turn
    store = ResilientMemoryStore(primary, max_replay_attempts=2)
    poison = SimpleNamespace(id="poison", session_id="s")
    good = SimpleNamespace(id="good", session_id="s")
    events = []
    event_bus.subscribe("memory.pending_quarantined", events.append)
    try:
        store._queue_pending("add_turn", poison)
        store._queue_pending("add_turn", good)
        assert store._replay_pending() is False
        assert store._replay_pending() is True
    finally:
        event_bus.unsubscribe("memory.pending_quarantined", events.append)
    assert store.pending_health["pending"] == 0
    assert store.pending_health["quarantined"] == 1
    assert len(events) == 1
    assert events[0].detail["category"] == "permanent"


def test_resilient_memory_keeps_transient_failure_order_without_quarantine() -> None:
    primary = MagicMock()
    primary.add_turn.side_effect = ConnectionError("offline")
    store = ResilientMemoryStore(primary, max_replay_attempts=1)
    first = SimpleNamespace(id="first", session_id="s")
    second = SimpleNamespace(id="second", session_id="s")
    store._queue_pending("add_turn", first)
    store._queue_pending("add_turn", second)
    for _ in range(3):
        assert store._replay_pending() is False
    assert store.pending_health["pending"] == 2
    assert store.pending_health["quarantined"] == 0

    primary.add_turn.side_effect = None
    assert store._replay_pending() is True
    assert [call.args[0].id for call in primary.add_turn.call_args_list[-2:]] == ["first", "second"]


def test_resilient_memory_queue_overflow_is_bounded_and_observable() -> None:
    from missy.core.events import event_bus

    store = ResilientMemoryStore(MagicMock(), max_pending_operations=1)
    events = []
    event_bus.subscribe("memory.pending_quarantined", events.append)
    try:
        store._queue_pending("add_turn", SimpleNamespace(id="first"))
        store._queue_pending("add_turn", SimpleNamespace(id="second"))
    finally:
        event_bus.unsubscribe("memory.pending_quarantined", events.append)
    assert store.pending_health["pending"] == 1
    assert store.pending_health["quarantined"] == 1
    assert events[-1].detail["category"] == "queue_overflow"


@pytest.mark.asyncio
async def test_voice_refuses_plaintext_remote_bind() -> None:
    server = VoiceServer(
        registry=MagicMock(),
        pairing_manager=MagicMock(),
        presence_store=MagicMock(),
        stt_engine=MagicMock(),
        tts_engine=MagicMock(),
        agent_callback=MagicMock(),
        host="0.0.0.0",
    )
    with pytest.raises(ValueError, match="plaintext non-loopback"):
        await server.start()


def test_mcp_config_file_mode_remains_private(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    manager = McpManager(config_path=str(path))
    manager._save_config()
    assert os.stat(path).st_mode & 0o077 == 0


def _start_background_loop() -> tuple[object, threading.Thread]:
    import asyncio

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(1)
    return loop, thread


def _stop_background_loop(loop: object, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    assert not thread.is_alive()
    loop.close()


def test_discord_voice_silence_timer_crosses_threads_without_loop_nudge() -> None:
    """A receive-thread packet must wake a real loop and deliver on silence."""
    loop, thread = _start_background_loop()
    delivered = threading.Event()
    received: list[tuple[int, bytes, int]] = []
    voice_recv = SimpleNamespace(AudioSink=object)
    sink = _make_sink_class(voice_recv)(
        loop=loop,
        on_speech_done=lambda *args: (received.append(args), delivered.set()),
    )
    try:
        with patch("missy.channels.discord.voice._SILENCE_TIMEOUT_S", 0.02):
            sink.write(SimpleNamespace(id=7), SimpleNamespace(pcm=b"pcm"))
            assert delivered.wait(1)
        assert received == [(7, b"pcm", 48000)]
    finally:
        sink.cleanup()
        _stop_background_loop(loop, thread)


def test_discord_voice_cleanup_prevents_ready_timer_callback() -> None:
    loop, thread = _start_background_loop()
    delivered = threading.Event()
    sink = _make_sink_class(SimpleNamespace(AudioSink=object))(
        loop=loop,
        on_speech_done=lambda *_args: delivered.set(),
    )
    try:
        with patch("missy.channels.discord.voice._SILENCE_TIMEOUT_S", 0.05):
            sink.write(SimpleNamespace(id=7), SimpleNamespace(pcm=b"pcm"))
            sink.cleanup()
            time.sleep(0.1)
        assert not delivered.is_set()
    finally:
        sink.cleanup()
        _stop_background_loop(loop, thread)


def test_discord_voice_playback_callback_wakes_loop_from_player_thread() -> None:
    import asyncio

    loop, thread = _start_background_loop()
    callback_thread: list[int] = []
    player_done = threading.Event()

    class VoiceClient:
        @staticmethod
        def is_playing() -> bool:
            return False

        @staticmethod
        def play(_source, *, after) -> None:
            def finish() -> None:
                callback_thread.append(threading.get_ident())
                after(None)
                player_done.set()

            threading.Thread(target=finish, daemon=True).start()

    manager = DiscordVoiceManager.__new__(DiscordVoiceManager)
    manager._tts_engine = SimpleNamespace(
        synthesize=AsyncMock(return_value=SimpleNamespace(data=b"wav"))
    )
    manager._discord = SimpleNamespace(FFmpegPCMAudio=lambda path: path)

    async def play() -> None:
        state = _GuildVoiceState(voice_client=VoiceClient())
        await manager._play_tts(state, "hello")

    try:
        future = asyncio.run_coroutine_threadsafe(play(), loop)
        future.result(timeout=1)
        assert player_done.wait(1)
        assert callback_thread and callback_thread[0] != thread.ident
    finally:
        _stop_background_loop(loop, thread)
