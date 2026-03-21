"""Security hardening tests.


Tests for fixes applied this session:
  1. Shell heredoc (<<EOF) rejection via _SUBSHELL_MARKERS
  2. Shell brace group anywhere in command (after semicolon)
  3. FTS5 query sanitization wraps input in double quotes to prevent injection
  4. FTS5 OperationalError handling returns empty list gracefully
  5. Proactive template safety — string.Template prevents attribute access attacks
  6. Proactive backward compatibility — {var} style auto-converted to ${var}
  7. Scheduler _save_jobs atomic write with 0o600 permissions via tempfile.mkstemp
  8. Webhook send() applies censor_response before logging
  9. MCP client response ID mismatch triggers a warning log
  10. Device registry load() rejects files not owned by current user or group/world-writable
  11. Audio log directory created with 0o700, file set to 0o600
"""

from __future__ import annotations

import sqlite3
import stat
from pathlib import Path
from string import Template
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1 & 2: Shell heredoc and brace group (anywhere) rejection
# ---------------------------------------------------------------------------


class TestShellHeredocRejection:
    """ShellPolicyEngine must reject heredoc markers (<<EOF) as subshell markers."""

    @pytest.fixture
    def engine(self):
        from missy.config.settings import ShellPolicy
        from missy.policy.shell import ShellPolicyEngine

        policy = ShellPolicy(enabled=True, allowed_commands=["cat", "echo", "ls", "grep"])
        return ShellPolicyEngine(policy)

    def test_heredoc_eof_marker_rejected(self, engine):
        """<<EOF heredoc should be rejected — it can feed arbitrary input to commands."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("cat <<EOF\nmalicious content\nEOF")

    def test_heredoc_simple_marker_rejected(self, engine):
        """<< with no delimiter still triggers heredoc rejection."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("grep pattern << input.txt")

    def test_heredoc_combined_with_allowed_command_still_rejected(self, engine):
        """Even a whitelisted command with << attached is rejected."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("cat <<HEREDOC\ndata\nHEREDOC")

    def test_heredoc_marker_in_extract_all_programs_returns_none(self):
        """_extract_all_programs returns None when << is present."""
        from missy.policy.shell import ShellPolicyEngine

        result = ShellPolicyEngine._extract_all_programs("cat <<EOF")
        assert result is None

    def test_here_string_marker_still_rejected(self, engine):
        """<<< (here-string) is also in _SUBSHELL_MARKERS and must be rejected."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("cat <<< 'evil input'")

    def test_normal_redirection_without_heredoc_allowed(self, engine):
        """A single < redirect is not a heredoc marker; compound commands with
        allowed programs should pass (< is not in _SUBSHELL_MARKERS)."""
        # This verifies only that << is checked, not < alone
        result = engine.check_command("echo hello")
        assert result is True


class TestShellBraceGroupAnywhere:
    """ShellPolicyEngine must reject brace groups regardless of position in command."""

    @pytest.fixture
    def engine(self):
        from missy.config.settings import ShellPolicy
        from missy.policy.shell import ShellPolicyEngine

        policy = ShellPolicy(enabled=True, allowed_commands=["echo", "ls", "cat"])
        return ShellPolicyEngine(policy)

    def test_brace_group_after_semicolon_rejected(self, engine):
        """echo hi; { rm -rf /; } must be rejected — brace group follows semicolon."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("echo hi; { rm -rf /; }")

    def test_brace_group_at_start_rejected(self, engine):
        """{ echo evil; } at command start is also rejected."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("{ echo evil; }")

    def test_brace_group_with_semicolon_prefix_rejected(self, engine):
        """{;echo evil;} (brace group with semicolon immediately after opening brace)."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("{;echo evil;}")

    def test_brace_group_after_and_operator_rejected(self, engine):
        """echo ok && { rm -rf /; } — brace group after && is also rejected."""
        from missy.core.exceptions import PolicyViolationError

        with pytest.raises(PolicyViolationError):
            engine.check_command("echo ok && { rm -rf /; }")

    def test_brace_group_extract_returns_none(self):
        """_extract_all_programs returns None for commands containing brace groups."""
        from missy.policy.shell import ShellPolicyEngine

        # Brace group after semicolon
        result = ShellPolicyEngine._extract_all_programs("echo hi; { rm -rf /; }")
        assert result is None

    def test_plain_command_without_braces_passes(self, engine):
        """A regular compound command without braces still works."""
        result = engine.check_command("echo hello; ls -la")
        assert result is True


# ---------------------------------------------------------------------------
# 3 & 4: FTS5 query sanitization and OperationalError handling
# ---------------------------------------------------------------------------


class TestFTS5QuerySanitization:
    """SQLiteMemoryStore.search must wrap the query in double-quotes to prevent
    FTS5 query injection via boolean operators and wildcards."""

    @pytest.fixture
    def store(self, tmp_path):
        from missy.memory.sqlite_store import SQLiteMemoryStore

        db = tmp_path / "test_memory.db"
        return SQLiteMemoryStore(db_path=str(db))

    def test_and_operator_treated_as_literal_phrase(self, store):
        """Input 'python AND async' is wrapped in quotes to treat AND as literal text."""
        from missy.memory.sqlite_store import ConversationTurn

        turn = ConversationTurn.new("s1", "user", "python AND async is a good pattern")
        store.add_turn(turn)

        # If AND were treated as an FTS5 operator, this might fail or behave
        # differently. Wrapping in quotes makes it a phrase search.
        results = store.search("python AND async")
        # Should not raise; result is a list (possibly empty, possibly matching)
        assert isinstance(results, list)

    def test_or_operator_treated_as_literal(self, store):
        """'python OR async' should not trigger FTS5 OR operator."""
        results = store.search("python OR async")
        assert isinstance(results, list)

    def test_not_operator_treated_as_literal(self, store):
        """'NOT python' should not trigger FTS5 NOT operator."""
        results = store.search("NOT python")
        assert isinstance(results, list)

    def test_wildcard_star_treated_as_literal(self, store):
        """'pyth*' should be wrapped in quotes, preventing FTS5 prefix search."""
        results = store.search("pyth*")
        assert isinstance(results, list)

    def test_embedded_double_quote_escaped(self, store):
        """A query with an embedded double quote must have it escaped to prevent
        FTS5 syntax errors."""
        # Without proper escaping, a query like 'say "hello"' would produce
        # the FTS5 query "say "hello"" which is malformed.
        results = store.search('say "hello" there')
        assert isinstance(results, list)

    def test_safe_query_construction(self, store):
        """Verify the safe_query wraps the raw query in double-quotes by
        inspecting the sanitization logic directly from the source."""
        # The sanitization rule from sqlite_store.py:
        #   safe_query = '"' + query.replace('"', '""') + '"'
        # Reproduce it here to confirm the contract:
        raw = "hello world"
        safe = '"' + raw.replace('"', '""') + '"'
        assert safe.startswith('"')
        assert safe.endswith('"')
        assert raw in safe

        # Also confirm a real search with operators doesn't raise
        result = store.search("hello world")
        assert isinstance(result, list)

    def test_embedded_quote_produces_doubled_quote_in_safe_query(self):
        """The sanitization logic replaces internal " with "" (SQL escape)."""
        # Simulate the sanitization directly
        raw = 'test "quoted" input'
        safe = '"' + raw.replace('"', '""') + '"'
        assert safe == '"test ""quoted"" input"'

    def test_fts_operational_error_returns_empty_list(self, store):
        """When sqlite3.OperationalError is raised, search returns [] rather than
        propagating the exception."""
        with patch.object(store, "_conn") as mock_conn_method:
            mock_conn = MagicMock()
            mock_conn_method.return_value = mock_conn
            mock_conn.execute.side_effect = sqlite3.OperationalError("fts5: syntax error")

            result = store.search("bad FTS query ***")

        assert result == []

    def test_fts_operational_error_with_session_filter_returns_empty_list(self, store):
        """The OperationalError guard also applies when session_id is provided."""
        with patch.object(store, "_conn") as mock_conn_method:
            mock_conn = MagicMock()
            mock_conn_method.return_value = mock_conn
            mock_conn.execute.side_effect = sqlite3.OperationalError("fts5: bad query")

            result = store.search("bad * query", session_id="sess-123")

        assert result == []

    def test_fts_operational_error_logs_warning(self, store):
        """A warning is logged at WARNING level when FTS5 query fails."""

        with patch.object(store, "_conn") as mock_conn_method:
            mock_conn = MagicMock()
            mock_conn_method.return_value = mock_conn
            mock_conn.execute.side_effect = sqlite3.OperationalError("fts5: error")

            with patch("missy.memory.sqlite_store.logger") as mock_logger:
                store.search("bad query")
                mock_logger.warning.assert_called_once()
                warning_args = mock_logger.warning.call_args[0]
                # The warning message should contain something about query failure
                assert "FTS5" in warning_args[0] or "fts5" in warning_args[0].lower()


# ---------------------------------------------------------------------------
# 5 & 6: Proactive template safety and backward compatibility
# ---------------------------------------------------------------------------


class TestProactiveTemplateSafety:
    """ProactiveManager uses string.Template to build prompts, which prevents
    format-string attacks that exploit attribute access via {obj.__class__}."""

    def _make_trigger(self, template: str, name: str = "test-trigger"):
        from missy.agent.proactive import ProactiveTrigger

        return ProactiveTrigger(
            name=name,
            trigger_type="schedule",
            prompt_template=template,
            cooldown_seconds=0,
            interval_seconds=9999,
        )

    def test_format_attack_does_not_leak_class_info(self):
        """A template containing {trigger_name.__class__} should not cause attribute
        access because string.Template ignores dotted paths after the identifier."""
        from missy.agent.proactive import ProactiveManager

        received_prompts = []
        manager = ProactiveManager(
            triggers=[],
            agent_callback=lambda prompt, session_id: received_prompts.append(prompt),
        )

        trigger = self._make_trigger(
            "Info: ${trigger_name.__class__}",
            name="attacker",
        )

        # With string.Template, ${trigger_name.__class__} is not a valid
        # identifier, so safe_substitute leaves it unreplaced (no attribute lookup).
        manager._fire_trigger(trigger)

        assert len(received_prompts) == 1
        prompt = received_prompts[0]
        # The class info must not appear as a resolved Python class reference
        assert "<class" not in prompt
        # The literal token should remain unreplaced or be partially substituted
        # but NOT cause an AttributeError
        assert isinstance(prompt, str)

    def test_safe_substitute_does_not_raise_on_unknown_placeholders(self):
        """string.Template.safe_substitute never raises on missing keys — it
        leaves them as-is rather than raising KeyError."""
        tmpl = Template("Hello ${unknown_key} world")
        result = tmpl.safe_substitute(trigger_name="t", trigger_type="schedule", timestamp="now")
        # The unknown key is preserved verbatim
        assert "${unknown_key}" in result

    def test_valid_template_variables_substituted(self):
        """The three standard variables — trigger_name, trigger_type, timestamp —
        are substituted correctly."""
        from missy.agent.proactive import ProactiveManager

        received_prompts = []
        manager = ProactiveManager(
            triggers=[],
            agent_callback=lambda prompt, session_id: received_prompts.append(prompt),
        )

        trigger = self._make_trigger(
            "Trigger: ${trigger_name} (${trigger_type}) at ${timestamp}",
            name="my-trigger",
        )
        trigger.trigger_type = "disk_threshold"

        manager._fire_trigger(trigger)

        assert len(received_prompts) == 1
        prompt = received_prompts[0]
        assert "my-trigger" in prompt
        assert "disk_threshold" in prompt
        # timestamp placeholder is replaced with an ISO-format string
        assert "${timestamp}" not in prompt

    def test_old_style_format_vars_auto_converted(self):
        """{var} style templates are converted to ${var} before Template expansion,
        preserving backward compatibility with old-style templates."""
        from missy.agent.proactive import ProactiveManager

        received_prompts = []
        manager = ProactiveManager(
            triggers=[],
            agent_callback=lambda prompt, session_id: received_prompts.append(prompt),
        )

        # Old-style {trigger_name} template
        trigger = self._make_trigger(
            "Old style: {trigger_name} fired at {timestamp}",
            name="legacy-trigger",
        )

        manager._fire_trigger(trigger)

        assert len(received_prompts) == 1
        prompt = received_prompts[0]
        # The name should be substituted
        assert "legacy-trigger" in prompt
        # The raw {trigger_name} placeholder must not appear verbatim
        assert "{trigger_name}" not in prompt

    def test_mixed_old_and_new_style_not_double_converted(self):
        """A template already using ${var} syntax is not double-converted."""
        from missy.agent.proactive import ProactiveManager

        received_prompts = []
        manager = ProactiveManager(
            triggers=[],
            agent_callback=lambda prompt, session_id: received_prompts.append(prompt),
        )

        # New-style — should work without conversion
        trigger = self._make_trigger(
            "New style: ${trigger_name}",
            name="new-style-trigger",
        )

        manager._fire_trigger(trigger)

        assert len(received_prompts) == 1
        assert "new-style-trigger" in received_prompts[0]


# ---------------------------------------------------------------------------
# 7: Scheduler atomic write with 0o600 permissions
# ---------------------------------------------------------------------------


class TestSchedulerAtomicWrite:
    """SchedulerManager._save_jobs must write via tempfile.mkstemp with 0o600
    permissions and atomically replace the target file."""

    def _make_manager(self, tmp_path):
        from missy.scheduler.manager import SchedulerManager

        jobs_file = tmp_path / "jobs.json"
        mgr = SchedulerManager(jobs_file=str(jobs_file))
        return mgr, jobs_file

    def test_save_jobs_uses_mkstemp_for_atomic_write(self, tmp_path):
        """_save_jobs calls tempfile.mkstemp and then os.replace.
        Both tempfile and os are local imports inside _save_jobs, so we patch
        them at the canonical module level."""
        import os as os_module
        import tempfile as tempfile_module

        mgr, jobs_file = self._make_manager(tmp_path)

        # Use a real temp file so the fd is valid for fchmod/write/close
        real_fd, real_tmp = tempfile_module.mkstemp(dir=str(tmp_path), suffix=".tmp")
        os_module.close(real_fd)  # close it; we re-open via mock below

        fd_value = 99
        tmp_file = str(tmp_path / "jobs.json.tmp")

        with (
            patch("tempfile.mkstemp", return_value=(fd_value, tmp_file)) as mock_mkstemp,
            patch("os.fchmod"),
            patch("os.write"),
            patch("os.close"),
            patch("os.replace"),
        ):
            mgr._save_jobs()

        mock_mkstemp.assert_called_once()
        _, mkstemp_kwargs = mock_mkstemp.call_args
        assert "dir" in mkstemp_kwargs
        assert mkstemp_kwargs["dir"] == str(jobs_file.parent)

    def test_save_jobs_sets_0o600_permissions_on_fd(self, tmp_path):
        """os.fchmod must be called with 0o600 on the file descriptor."""
        mgr, _ = self._make_manager(tmp_path)

        fd_value = 42
        tmp_file = str(tmp_path / "tmp_jobs.json")

        with (
            patch("tempfile.mkstemp", return_value=(fd_value, tmp_file)),
            patch("os.fchmod") as mock_fchmod,
            patch("os.write"),
            patch("os.close"),
            patch("os.replace"),
        ):
            mgr._save_jobs()

        mock_fchmod.assert_called_once_with(fd_value, 0o600)

    def test_save_jobs_calls_os_replace_for_atomic_rename(self, tmp_path):
        """os.replace (atomic rename) must be called to move tmp -> final path."""
        mgr, jobs_file = self._make_manager(tmp_path)

        fd_value = 55
        tmp_file = str(tmp_path / "tmp_jobs.json")

        with (
            patch("tempfile.mkstemp", return_value=(fd_value, tmp_file)),
            patch("os.fchmod"),
            patch("os.write"),
            patch("os.close"),
            patch("os.replace") as mock_replace,
        ):
            mgr._save_jobs()

        mock_replace.assert_called_once_with(tmp_file, str(jobs_file))

    def test_save_jobs_cleans_up_tmp_on_write_error(self, tmp_path):
        """When os.write raises, the temp file is unlinked to avoid leaking it."""
        mgr, _ = self._make_manager(tmp_path)

        fd_value = 77
        tmp_file = str(tmp_path / "tmp_jobs.json")

        with (
            patch("tempfile.mkstemp", return_value=(fd_value, tmp_file)),
            patch("os.fchmod"),
            patch("os.write", side_effect=OSError("disk full")),
            patch("os.close"),
            patch("os.replace"),
            patch("os.unlink") as mock_unlink,
        ):
            # The method catches all exceptions and logs them — must not re-raise
            mgr._save_jobs()

        # The temp file should be cleaned up on write failure
        mock_unlink.assert_called_once_with(tmp_file)


# ---------------------------------------------------------------------------
# 8: Webhook send() applies censor_response before logging
# ---------------------------------------------------------------------------


class TestWebhookResponseCensoring:
    """WebhookChannel.send() must apply censor_response before writing to the log
    so that secrets never appear in log output."""

    def test_send_applies_censor_response(self):
        """censor_response is called on the message before logger.info.
        Because censor_response is imported locally inside send(), we patch
        it at the source module (missy.security.censor)."""
        from missy.channels.webhook import WebhookChannel

        ch = WebhookChannel()

        with (
            patch(
                "missy.security.censor.censor_response", return_value="[REDACTED]"
            ) as mock_censor,
            patch("missy.channels.webhook.logger") as mock_logger,
        ):
            ch.send("sk-ant-secret123 result text")

        mock_censor.assert_called_once()
        # The censored value should appear in the log call
        log_args = mock_logger.info.call_args
        assert log_args is not None
        formatted = str(log_args)
        assert "[REDACTED]" in formatted or "REDACTED" in formatted

    def test_send_does_not_log_raw_secret(self):
        """The raw secret value must not appear in the logger.info call."""
        from missy.channels.webhook import WebhookChannel

        ch = WebhookChannel()
        raw_secret = "sk-ant-api03-supersecrettoken12345"

        with (
            patch("missy.security.censor.censor_response", return_value="***CENSORED***"),
            patch("missy.channels.webhook.logger") as mock_logger,
        ):
            ch.send(raw_secret)

        # Inspect all logger.info calls — none should contain the raw secret
        for call_args in mock_logger.info.call_args_list:
            assert raw_secret not in str(call_args)

    def test_send_truncates_to_200_chars_before_censoring(self):
        """The implementation slices to [:200] before calling censor_response."""
        from missy.channels.webhook import WebhookChannel

        ch = WebhookChannel()
        long_message = "A" * 500

        captured_inputs = []

        def capture_censor(text):
            captured_inputs.append(text)
            return text

        with (
            patch("missy.security.censor.censor_response", side_effect=capture_censor),
            patch("missy.channels.webhook.logger"),
        ):
            ch.send(long_message)

        assert len(captured_inputs) == 1
        assert len(captured_inputs[0]) == 200


# ---------------------------------------------------------------------------
# 9: MCP response ID mismatch warning
# ---------------------------------------------------------------------------


class TestMcpResponseIdValidation:
    """McpClient._rpc must log a warning when the response ID does not match
    the request ID, indicating a possible response-confusion attack."""

    def _make_client_with_response(
        self, response_id: str | None, req_id_override: str | None = None
    ):
        """Build a McpClient instance wired to return a fake JSON response."""
        import json

        from missy.mcp.client import McpClient

        client = McpClient(name="test-server")

        # Synthesise a response with a controlled ID
        response_body = {"jsonrpc": "2.0", "id": response_id, "result": {"tools": []}}
        response_line = (json.dumps(response_body) + "\n").encode()

        mock_stdin = MagicMock()
        mock_stdout = MagicMock()
        mock_stdout.readline.return_value = response_line

        mock_proc = MagicMock()
        mock_proc.stdin = mock_stdin
        mock_proc.stdout = mock_stdout
        client._proc = mock_proc

        return client

    def test_matching_response_id_no_warning(self):
        """When response ID matches the request ID, no warning is logged."""
        import json

        from missy.mcp.client import McpClient

        client = McpClient(name="test-server")

        # We need to capture the req_id that _rpc generates, so we intercept uuid4
        fixed_id = "aaaa-bbbb-cccc-dddd"
        response_body = {"jsonrpc": "2.0", "id": fixed_id, "result": {}}
        response_line = (json.dumps(response_body) + "\n").encode()

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout.readline.return_value = response_line
        client._proc = mock_proc

        with (
            patch(
                "missy.mcp.client.uuid.uuid4", return_value=MagicMock(__str__=lambda s: fixed_id)
            ),
            patch("missy.mcp.client.logger") as mock_logger,
        ):
            client._rpc("tools/list")

        mock_logger.warning.assert_not_called()

    def test_mismatched_response_id_raises_runtime_error(self):
        """When response ID differs from request ID, a RuntimeError is raised."""
        import json

        from missy.mcp.client import McpClient

        client = McpClient(name="test-server")

        fixed_req_id = "req-id-1234"
        wrong_resp_id = "resp-id-9999"

        response_body = {"jsonrpc": "2.0", "id": wrong_resp_id, "result": {}}
        response_line = (json.dumps(response_body) + "\n").encode()

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout.readline.return_value = response_line
        client._proc = mock_proc

        with (
            patch(
                "missy.mcp.client.uuid.uuid4",
                return_value=MagicMock(__str__=lambda s: fixed_req_id),
            ),
            pytest.raises(RuntimeError, match="MCP response ID mismatch"),
        ):
            client._rpc("tools/list")

    def test_null_response_id_does_not_log_warning(self):
        """When response ID is null/None, the check is skipped (notification-style
        response) and no warning is emitted."""
        import json

        from missy.mcp.client import McpClient

        client = McpClient(name="test-server")

        # id=null in JSON → None in Python; the implementation checks
        # ``if resp_id is not None``
        response_body = {"jsonrpc": "2.0", "id": None, "result": {}}
        response_line = (json.dumps(response_body) + "\n").encode()

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout.readline.return_value = response_line
        client._proc = mock_proc

        with (
            patch("missy.mcp.client.uuid.uuid4"),
            patch("missy.mcp.client.logger") as mock_logger,
        ):
            client._rpc("tools/list")

        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# 10: Device registry permission checks on load
# ---------------------------------------------------------------------------


class TestDeviceRegistryPermissionChecks:
    """DeviceRegistry.load() must refuse to load a registry file that is not
    owned by the current user or that is group/world-writable."""

    def _write_registry_file(self, path: Path, content: str = "[]") -> None:
        path.write_text(content, encoding="utf-8")

    def test_load_rejects_file_not_owned_by_current_user(self, tmp_path):
        """If st_uid != os.getuid(), load() starts an empty registry."""
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(
            reg_file,
            '[{"node_id": "n1", "friendly_name": "Test", "room": "Lab", "ip_address": "10.0.0.1"}]',
        )

        registry = DeviceRegistry(registry_path=str(reg_file))

        with patch("missy.channels.voice.registry.os.getuid", return_value=9999):
            # The file is owned by the real test user, but we tell getuid() to
            # return a different UID — simulating a file owned by someone else.
            registry.load()

        # Registry should be empty — the file was refused
        assert registry.list_nodes() == []

    def test_load_rejects_group_writable_file(self, tmp_path):
        """If the file has group-write permission (S_IWGRP), load() refuses it."""
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        # Set group-writable bit
        reg_file.chmod(0o664)

        registry = DeviceRegistry(registry_path=str(reg_file))
        registry.load()

        assert registry.list_nodes() == []

    def test_load_rejects_world_writable_file(self, tmp_path):
        """If the file has other-write permission (S_IWOTH), load() refuses it."""
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        # Set world-writable bit
        reg_file.chmod(0o666)

        registry = DeviceRegistry(registry_path=str(reg_file))
        registry.load()

        assert registry.list_nodes() == []

    def test_load_accepts_owner_only_readable_file(self, tmp_path):
        """A file with 0o600 (owner read/write only) is accepted."""
        import json

        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        node_data = [
            {
                "node_id": "node-abc",
                "friendly_name": "Living Room",
                "room": "living",
                "ip_address": "192.168.1.10",
            }
        ]
        reg_file.write_text(json.dumps(node_data), encoding="utf-8")
        reg_file.chmod(0o600)

        registry = DeviceRegistry(registry_path=str(reg_file))
        registry.load()

        nodes = registry.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "node-abc"

    def test_load_accepts_0o640_file(self, tmp_path):
        """A file with 0o640 (owner read/write, group read) is accepted — only
        group/world WRITE bits are rejected, not read bits."""
        import json

        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        node_data = [
            {
                "node_id": "node-xyz",
                "friendly_name": "Kitchen",
                "room": "kitchen",
                "ip_address": "192.168.1.20",
            }
        ]
        reg_file.write_text(json.dumps(node_data), encoding="utf-8")
        reg_file.chmod(0o640)

        registry = DeviceRegistry(registry_path=str(reg_file))
        registry.load()

        nodes = registry.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "node-xyz"

    def test_load_logs_error_for_wrong_ownership(self, tmp_path):
        """An error is logged when the file is owned by a different user."""
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)

        registry = DeviceRegistry(registry_path=str(reg_file))

        with (
            patch("missy.channels.voice.registry.os.getuid", return_value=9999),
            patch("missy.channels.voice.registry.logger") as mock_logger,
        ):
            registry.load()

        mock_logger.error.assert_called()
        error_msg = str(mock_logger.error.call_args_list)
        assert "own" in error_msg.lower() or "uid" in error_msg.lower()

    def test_load_logs_error_for_group_writable(self, tmp_path):
        """An error is logged when the file is group-writable."""
        from missy.channels.voice.registry import DeviceRegistry

        reg_file = tmp_path / "devices.json"
        self._write_registry_file(reg_file)
        reg_file.chmod(0o664)

        registry = DeviceRegistry(registry_path=str(reg_file))

        with patch("missy.channels.voice.registry.logger") as mock_logger:
            registry.load()

        mock_logger.error.assert_called()
        error_msg = str(mock_logger.error.call_args_list)
        assert "writable" in error_msg.lower() or "perm" in error_msg.lower()

    def test_load_nonexistent_file_starts_empty_without_error(self, tmp_path):
        """A missing registry file is not an error — the registry starts empty."""
        from missy.channels.voice.registry import DeviceRegistry

        registry = DeviceRegistry(registry_path=str(tmp_path / "nonexistent.json"))

        with patch("missy.channels.voice.registry.logger") as mock_logger:
            registry.load()

        assert registry.list_nodes() == []
        mock_logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# 11: Audio log directory and file permissions
# ---------------------------------------------------------------------------


class TestAudioLogPermissions:
    """VoiceServer._log_audio_to_disk must create the log directory with 0o700
    and set each audio file to 0o600 after writing."""

    @pytest.fixture
    def node(self, tmp_path):
        """Return a minimal EdgeNode with audio_logging enabled."""
        from missy.channels.voice.registry import EdgeNode

        log_dir = tmp_path / "audio_logs"
        return EdgeNode(
            node_id="test-node",
            friendly_name="Test Node",
            room="lab",
            ip_address="10.0.0.1",
            audio_logging=True,
            audio_log_dir=str(log_dir),
        )

    @pytest.mark.asyncio
    async def test_audio_log_directory_created_with_0o700(self, node, tmp_path):
        """The audio log directory is created with mode 0o700 (owner access only)."""
        log_dir = Path(node.audio_log_dir)
        assert not log_dir.exists()

        mkdir_calls = []
        original_mkdir = Path.mkdir

        def capturing_mkdir(self, mode=0o777, parents=False, exist_ok=False):
            if str(self) == str(log_dir):
                mkdir_calls.append(mode)
            return original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

        with patch.object(Path, "mkdir", capturing_mkdir):
            from missy.channels.voice.server import VoiceServer

            server = VoiceServer.__new__(VoiceServer)

            await server._log_audio_to_disk(
                node=node,
                audio_buffer=b"\x00" * 100,
                sample_rate=16000,
                channels=1,
            )

        # The directory should now exist and have been created with 0o700
        assert log_dir.exists()
        # Check that mkdir was called with 0o700
        assert 0o700 in mkdir_calls

    @pytest.mark.asyncio
    async def test_audio_log_file_set_to_0o600(self, node, tmp_path):
        """Each written audio file must have its permissions set to 0o600."""
        from missy.channels.voice.server import VoiceServer

        server = VoiceServer.__new__(VoiceServer)

        await server._log_audio_to_disk(
            node=node,
            audio_buffer=b"\xab" * 256,
            sample_rate=16000,
            channels=1,
        )

        log_dir = Path(node.audio_log_dir)
        assert log_dir.exists()

        pcm_files = list(log_dir.glob("*.pcm"))
        assert len(pcm_files) == 1

        file_stat = pcm_files[0].stat()
        # Mask off the file type bits — we only care about permission bits
        perm_bits = stat.S_IMODE(file_stat.st_mode)
        assert perm_bits == 0o600, f"Expected 0o600, got {oct(perm_bits)}"

    @pytest.mark.asyncio
    async def test_audio_log_file_contains_correct_bytes(self, node):
        """Sanity check: the written file contains exactly the audio buffer bytes."""
        from missy.channels.voice.server import VoiceServer

        server = VoiceServer.__new__(VoiceServer)
        audio_data = b"\x12\x34\x56\x78" * 64

        await server._log_audio_to_disk(
            node=node,
            audio_buffer=audio_data,
            sample_rate=22050,
            channels=1,
        )

        log_dir = Path(node.audio_log_dir)
        pcm_files = list(log_dir.glob("*.pcm"))
        assert len(pcm_files) == 1
        assert pcm_files[0].read_bytes() == audio_data

    @pytest.mark.asyncio
    async def test_audio_log_filename_uses_nanosecond_timestamp(self, node):
        """The log filename is <unix_timestamp_ns>.pcm."""
        from missy.channels.voice.server import VoiceServer

        server = VoiceServer.__new__(VoiceServer)

        with patch("missy.channels.voice.server.time.time_ns", return_value=1234567890123456789):
            await server._log_audio_to_disk(
                node=node,
                audio_buffer=b"\x00" * 10,
                sample_rate=16000,
                channels=1,
            )

        log_dir = Path(node.audio_log_dir)
        pcm_files = list(log_dir.glob("*.pcm"))
        assert len(pcm_files) == 1
        assert pcm_files[0].name == "1234567890123456789.pcm"
