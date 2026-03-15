"""Session 17 security hardening tests.

Tests for fixes applied this session:
  1. Scheduler jobs file permission checks in _load_jobs():
     - Rejects files not owned by current user (os.getuid())
     - Rejects files with group-writable or world-writable permissions (mode & 0o022)
     - Handles OSError from stat() gracefully
  2. Scheduler max_attempts clamping in ScheduledJob.from_dict() to min(value, 10)
  3. Self-create tool expanded blocklist with indirect execution patterns:
     __import__(, getattr(, importlib., compile(, code.interact(,
     child_process, require('fs'), require("fs"), $(, backtick
  4. Webhook security headers on all responses (X-Content-Type-Options, X-Frame-Options,
     Cache-Control) and 405 for non-POST methods with server version masking
  5. MCP response ID mismatch now raises RuntimeError instead of logging a warning
  6. Runtime error message sanitization: unexpected tool errors return
     "Tool execution failed due to an internal error." and args logged keys-only
"""

from __future__ import annotations

import json
import os
import stat
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1: Scheduler _load_jobs() permission checks
# ---------------------------------------------------------------------------


class TestSchedulerLoadJobsPermissionChecks:
    """SchedulerManager._load_jobs() must reject jobs files with unsafe ownership
    or permissions, and handle stat() failures gracefully."""

    def _make_manager(self, tmp_path: Path):
        from missy.scheduler.manager import SchedulerManager

        jobs_file = tmp_path / "jobs.json"
        return SchedulerManager(jobs_file=str(jobs_file)), jobs_file

    def _write_valid_jobs_file(self, jobs_file: Path) -> None:
        payload = [
            {
                "id": "job-001",
                "name": "Test Job",
                "schedule": "every 5 minutes",
                "task": "say hello",
                "provider": "anthropic",
            }
        ]
        jobs_file.write_text(json.dumps(payload), encoding="utf-8")

    def test_load_rejects_file_not_owned_by_current_user(self, tmp_path):
        """_load_jobs must skip loading when st_uid != os.getuid()."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)

        with patch("missy.scheduler.manager.os.getuid", return_value=9999):
            # The real file is owned by the test runner, but getuid returns 9999,
            # so the ownership check fails.
            mgr._load_jobs()

        assert mgr._jobs == {}

    def test_load_rejects_group_writable_file(self, tmp_path):
        """_load_jobs must skip loading when the file has the group-write bit set."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)
        jobs_file.chmod(0o664)

        mgr._load_jobs()

        assert mgr._jobs == {}

    def test_load_rejects_world_writable_file(self, tmp_path):
        """_load_jobs must skip loading when the file has the world-write bit set."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)
        jobs_file.chmod(0o646)

        mgr._load_jobs()

        assert mgr._jobs == {}

    def test_load_rejects_group_and_world_writable_file(self, tmp_path):
        """_load_jobs must skip loading when both group and world write bits are set."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)
        jobs_file.chmod(0o666)

        mgr._load_jobs()

        assert mgr._jobs == {}

    def test_load_accepts_owner_only_permissions(self, tmp_path):
        """_load_jobs must accept files with 0o600 (owner r/w only)."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)
        jobs_file.chmod(0o600)

        mgr._load_jobs()

        assert len(mgr._jobs) == 1

    def test_load_accepts_0o640_permissions(self, tmp_path):
        """_load_jobs must accept files with 0o640 (group read, not write)."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)
        jobs_file.chmod(0o640)

        mgr._load_jobs()

        assert len(mgr._jobs) == 1

    def test_load_handles_oserror_from_stat_gracefully(self, tmp_path):
        """When stat() raises OSError, _load_jobs returns without raising."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)

        # Patch the jobs_file attribute on the manager so its stat() raises
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.stat.side_effect = OSError("permission denied")
        mgr.jobs_file = mock_path

        # Must not raise — OSError is caught and logged
        mgr._load_jobs()

        # No jobs loaded
        assert mgr._jobs == {}

    def test_load_logs_error_for_wrong_ownership(self, tmp_path):
        """An error is logged when the file owner uid does not match current user."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)

        with (
            patch("missy.scheduler.manager.os.getuid", return_value=9999),
            patch("missy.scheduler.manager.logger") as mock_logger,
        ):
            mgr._load_jobs()

        mock_logger.error.assert_called()
        error_text = str(mock_logger.error.call_args_list)
        assert "uid" in error_text.lower() or "own" in error_text.lower() or "user" in error_text.lower()

    def test_load_logs_error_for_unsafe_permissions(self, tmp_path):
        """An error is logged when the file has group/world write bits."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)
        jobs_file.chmod(0o664)

        with patch("missy.scheduler.manager.logger") as mock_logger:
            mgr._load_jobs()

        mock_logger.error.assert_called()
        error_text = str(mock_logger.error.call_args_list)
        assert "perm" in error_text.lower() or "writable" in error_text.lower() or "unsafe" in error_text.lower()

    def test_load_logs_error_on_stat_oserror(self, tmp_path):
        """An error is logged when stat() raises OSError."""
        mgr, jobs_file = self._make_manager(tmp_path)
        self._write_valid_jobs_file(jobs_file)

        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.stat.side_effect = OSError("permission denied")
        mgr.jobs_file = mock_path

        with patch("missy.scheduler.manager.logger") as mock_logger:
            mgr._load_jobs()

        mock_logger.error.assert_called()

    def test_load_nonexistent_file_returns_silently(self, tmp_path):
        """_load_jobs does not raise or log when the file does not exist."""
        mgr, _ = self._make_manager(tmp_path)
        # jobs_file does not exist

        with patch("missy.scheduler.manager.logger") as mock_logger:
            mgr._load_jobs()

        assert mgr._jobs == {}
        mock_logger.error.assert_not_called()


# ---------------------------------------------------------------------------
# 2: Scheduler max_attempts clamping
# ---------------------------------------------------------------------------


class TestScheduledJobMaxAttemptsClamping:
    """ScheduledJob.from_dict() must clamp max_attempts to at most 10."""

    def test_max_attempts_100_clamped_to_10(self):
        """from_dict with max_attempts=100 must produce max_attempts=10."""
        from missy.scheduler.jobs import ScheduledJob

        data = {
            "id": "job-clamp-test",
            "name": "Clamp Test",
            "schedule": "every hour",
            "task": "run task",
            "max_attempts": 100,
        }
        job = ScheduledJob.from_dict(data)

        assert job.max_attempts == 10

    def test_max_attempts_3_stays_3(self):
        """from_dict with max_attempts=3 must produce max_attempts=3."""
        from missy.scheduler.jobs import ScheduledJob

        data = {
            "id": "job-no-clamp",
            "name": "No Clamp Test",
            "schedule": "every hour",
            "task": "run task",
            "max_attempts": 3,
        }
        job = ScheduledJob.from_dict(data)

        assert job.max_attempts == 3

    def test_max_attempts_10_stays_10(self):
        """from_dict with max_attempts=10 stays exactly at 10 (boundary value)."""
        from missy.scheduler.jobs import ScheduledJob

        data = {
            "id": "job-boundary",
            "name": "Boundary Test",
            "schedule": "every hour",
            "task": "run task",
            "max_attempts": 10,
        }
        job = ScheduledJob.from_dict(data)

        assert job.max_attempts == 10

    def test_max_attempts_11_clamped_to_10(self):
        """from_dict with max_attempts=11 is clamped to 10."""
        from missy.scheduler.jobs import ScheduledJob

        data = {
            "id": "job-11",
            "name": "Eleven Test",
            "schedule": "every hour",
            "task": "run task",
            "max_attempts": 11,
        }
        job = ScheduledJob.from_dict(data)

        assert job.max_attempts == 10

    def test_max_attempts_missing_defaults_to_3(self):
        """from_dict with no max_attempts key defaults to 3 (not clamped)."""
        from missy.scheduler.jobs import ScheduledJob

        data = {
            "id": "job-default",
            "name": "Default Test",
            "schedule": "every hour",
            "task": "run task",
        }
        job = ScheduledJob.from_dict(data)

        assert job.max_attempts == 3

    def test_max_attempts_1_stays_1(self):
        """from_dict with max_attempts=1 is preserved (minimum positive value)."""
        from missy.scheduler.jobs import ScheduledJob

        data = {
            "id": "job-one",
            "name": "One Attempt",
            "schedule": "every hour",
            "task": "run task",
            "max_attempts": 1,
        }
        job = ScheduledJob.from_dict(data)

        assert job.max_attempts == 1


# ---------------------------------------------------------------------------
# 3: Self-create tool expanded blocklist
# ---------------------------------------------------------------------------


class TestSelfCreateToolExpandedBlocklist:
    """SelfCreateTool must reject scripts containing any pattern in _DANGEROUS_PATTERNS,
    including the newly added indirect execution and shell expansion patterns."""

    @pytest.fixture
    def tool(self):
        from missy.tools.builtin.self_create_tool import SelfCreateTool

        return SelfCreateTool()

    def _create(self, tool, script: str, language: str = "python"):
        return tool.execute(
            action="create",
            tool_name="test_tool",
            language=language,
            script=script,
        )

    def test_dunder_import_rejected(self, tool):
        """__import__( is blocked as an indirect import mechanism."""
        result = self._create(tool, "x = __import__('os')\nx.system('ls')")
        assert result.success is False
        assert "__import__(" in result.error

    def test_getattr_rejected(self, tool):
        """getattr( is blocked as it can bypass attribute restrictions."""
        result = self._create(tool, "f = getattr(os, 'system')\nf('ls')")
        assert result.success is False
        assert "getattr(" in result.error

    def test_importlib_rejected(self, tool):
        """importlib. is blocked as an indirect import mechanism."""
        result = self._create(tool, "import importlib\nmod = importlib.import_module('os')")
        assert result.success is False
        assert "importlib." in result.error

    def test_compile_rejected(self, tool):
        """compile( is blocked as it enables dynamic code execution."""
        # Script contains only compile( — no other blocked patterns
        result = self._create(tool, "ast_tree = compile('1 + 1', '<string>', 'eval')")
        assert result.success is False
        assert "compile(" in result.error

    def test_code_interact_rejected(self, tool):
        """code.interact( is blocked as it opens an interactive interpreter."""
        result = self._create(tool, "import code\ncode.interact(local=locals())")
        assert result.success is False
        assert "code.interact(" in result.error

    def test_child_process_rejected(self, tool):
        """child_process is blocked as it is the Node.js subprocess module."""
        # Use spawn instead of exec to avoid triggering the 'exec(' pattern first
        result = self._create(
            tool,
            "const cp = require('child_process');\ncp.spawn('ls', []);",
            language="node",
        )
        assert result.success is False
        assert "child_process" in result.error

    def test_require_fs_single_quote_rejected(self, tool):
        """require('fs') is blocked as a Node.js filesystem access pattern."""
        result = self._create(
            tool,
            "const fs = require('fs');\nfs.readFileSync('/etc/passwd');",
            language="node",
        )
        assert result.success is False
        assert "require('fs')" in result.error

    def test_require_fs_double_quote_rejected(self, tool):
        """require("fs") is blocked as a Node.js filesystem access pattern."""
        result = self._create(
            tool,
            'const fs = require("fs");\nfs.writeFileSync("/tmp/x", "evil");',
            language="node",
        )
        assert result.success is False
        assert 'require("fs")' in result.error

    def test_dollar_paren_shell_expansion_rejected(self, tool):
        """$( is blocked as it triggers shell command substitution."""
        result = self._create(tool, "result=$(cat /etc/passwd)\necho $result", language="bash")
        assert result.success is False
        assert "$(" in result.error

    def test_backtick_shell_expansion_rejected(self, tool):
        """A backtick is blocked as it triggers shell command substitution."""
        result = self._create(tool, "result=`cat /etc/passwd`\necho $result", language="bash")
        assert result.success is False
        # The error message should indicate the backtick pattern
        assert result.error is not None and len(result.error) > 0

    def test_clean_python_script_accepted(self, tool, tmp_path):
        """A benign Python script with no dangerous patterns is created successfully."""
        with patch("missy.tools.builtin.self_create_tool.CUSTOM_TOOLS_DIR", tmp_path):
            result = tool.execute(
                action="create",
                tool_name="safe_tool",
                language="python",
                script="print('hello world')\n",
            )
        assert result.success is True

    def test_case_insensitive_pattern_matching(self, tool):
        """Pattern detection is case-insensitive; EVAL( must be caught too."""
        result = self._create(tool, "EVAL('import os')")
        assert result.success is False


# ---------------------------------------------------------------------------
# 4: Webhook security headers and method restrictions
# ---------------------------------------------------------------------------


class TestWebhookSecurityHeaders:
    """WebhookChannel must attach X-Content-Type-Options, X-Frame-Options, and
    Cache-Control security headers to every response, return 405 for non-POST
    methods, and not expose server version information."""

    def _make_handler_instance(self):
        """Return a Handler instance with a mock socket for response capture."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)

        # Access the inner Handler class by starting the server briefly and
        # inspecting the request handler class.  Instead, we reconstruct the
        # handler directly via channel.start() with a real (ephemeral) socket,
        # then shut it down.  For speed, we monkey-patch the handler class by
        # calling start() and reading the server's RequestHandlerClass.
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()
        return handler_class

    def _make_mock_request(self, method: str = "GET") -> MagicMock:
        """Build a minimal mock that simulates an incoming HTTP request."""
        mock = MagicMock()
        mock.command = method
        mock.path = "/"
        mock.headers = {}
        mock.client_address = ("127.0.0.1", 12345)

        # Capture written bytes
        output = BytesIO()
        mock.wfile = output
        mock.rfile = BytesIO(b"")
        return mock

    def test_get_returns_405(self, tmp_path):
        """GET /  must receive a 405 Method Not Allowed response."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        responses_sent = []

        # Instantiate handler with mocked socket internals
        handler = handler_class.__new__(handler_class)
        handler.client_address = ("127.0.0.1", 0)
        handler.server = MagicMock()
        headers_sent = []
        handler.send_response = lambda code: responses_sent.append(code)
        handler.send_header = lambda k, v: headers_sent.append((k, v))
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.do_GET()

        assert 405 in responses_sent

    def test_security_headers_present_on_405(self):
        """do_GET (405) must include all three security headers."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        responses_sent = []
        headers_sent = []

        handler = handler_class.__new__(handler_class)
        handler.client_address = ("127.0.0.1", 0)
        handler.server = MagicMock()
        handler.send_response = lambda code: responses_sent.append(code)
        handler.send_header = lambda k, v: headers_sent.append((k, v))
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.do_GET()

        header_names = [k for k, _ in headers_sent]
        assert "X-Content-Type-Options" in header_names
        assert "X-Frame-Options" in header_names
        assert "Cache-Control" in header_names

    def test_security_header_values_correct(self):
        """Security header values must match the prescribed constants."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        headers_sent = []

        handler = handler_class.__new__(handler_class)
        handler.client_address = ("127.0.0.1", 0)
        handler.server = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = lambda k, v: headers_sent.append((k, v))
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.do_GET()

        header_map = dict(headers_sent)
        assert header_map.get("X-Content-Type-Options") == "nosniff"
        assert header_map.get("X-Frame-Options") == "DENY"
        assert header_map.get("Cache-Control") == "no-store"

    def test_put_returns_405(self):
        """do_PUT must also return 405."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        responses_sent = []

        handler = handler_class.__new__(handler_class)
        handler.client_address = ("127.0.0.1", 0)
        handler.server = MagicMock()
        handler.send_response = lambda code: responses_sent.append(code)
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.do_PUT()

        assert 405 in responses_sent

    def test_delete_returns_405(self):
        """do_DELETE must also return 405."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        responses_sent = []

        handler = handler_class.__new__(handler_class)
        handler.client_address = ("127.0.0.1", 0)
        handler.server = MagicMock()
        handler.send_response = lambda code: responses_sent.append(code)
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.do_DELETE()

        assert 405 in responses_sent

    def test_patch_returns_405(self):
        """do_PATCH must also return 405."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        responses_sent = []

        handler = handler_class.__new__(handler_class)
        handler.client_address = ("127.0.0.1", 0)
        handler.server = MagicMock()
        handler.send_response = lambda code: responses_sent.append(code)
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = BytesIO()
        handler.rfile = BytesIO()

        handler.do_PATCH()

        assert 405 in responses_sent

    def test_server_version_string_masked(self):
        """Handler.version_string() must not reveal the real server/Python version."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        channel.start()
        handler_class = channel._server.RequestHandlerClass
        channel.stop()

        handler = handler_class.__new__(handler_class)
        version = handler.version_string()

        assert "Python" not in version
        assert "BaseHTTP" not in version
        # The masked string should simply be "missy"
        assert version == "missy"


# ---------------------------------------------------------------------------
# 5: MCP response ID mismatch raises RuntimeError
# ---------------------------------------------------------------------------


class TestMcpResponseIdMismatchRaisesError:
    """McpClient._rpc must raise RuntimeError (not just log a warning) when the
    JSON-RPC response ID does not match the request ID."""

    def _make_client_with_response(self, response_id):
        from missy.mcp.client import McpClient

        client = McpClient(name="test-server")

        body = {"jsonrpc": "2.0", "id": response_id, "result": {}}
        response_line = (json.dumps(body) + "\n").encode()

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout.readline.return_value = response_line
        client._proc = mock_proc
        return client

    def test_mismatched_id_raises_runtime_error(self):
        """_rpc raises RuntimeError when response ID != request ID."""
        from missy.mcp.client import McpClient

        fixed_req_id = "req-abc-123"
        wrong_resp_id = "resp-xyz-999"

        client = self._make_client_with_response(wrong_resp_id)

        with (
            patch(
                "missy.mcp.client.uuid.uuid4",
                return_value=MagicMock(__str__=lambda s: fixed_req_id),
            ),
            pytest.raises(RuntimeError, match="MCP response ID mismatch"),
        ):
            client._rpc("tools/list")

    def test_mismatched_id_error_message_contains_both_ids(self):
        """The RuntimeError message must include both expected and received IDs."""
        from missy.mcp.client import McpClient

        fixed_req_id = "req-known"
        wrong_resp_id = "resp-unknown"

        client = self._make_client_with_response(wrong_resp_id)

        with (
            patch(
                "missy.mcp.client.uuid.uuid4",
                return_value=MagicMock(__str__=lambda s: fixed_req_id),
            ),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                client._rpc("tools/list")

        msg = str(exc_info.value)
        assert fixed_req_id in msg
        assert wrong_resp_id in msg

    def test_matching_id_does_not_raise(self):
        """_rpc returns the response dict normally when IDs match."""
        from missy.mcp.client import McpClient

        fixed_id = "match-id-42"
        client = self._make_client_with_response(fixed_id)

        with patch(
            "missy.mcp.client.uuid.uuid4",
            return_value=MagicMock(__str__=lambda s: fixed_id),
        ):
            result = client._rpc("tools/list")

        assert isinstance(result, dict)
        assert result.get("id") == fixed_id

    def test_null_response_id_does_not_raise(self):
        """_rpc skips the ID check (and does not raise) when response id is None."""
        from missy.mcp.client import McpClient

        # id=None in the response signals a notification-style message; skip check
        client = self._make_client_with_response(None)

        # Should not raise regardless of what uuid4 returns
        with patch("missy.mcp.client.uuid.uuid4"):
            result = client._rpc("tools/list")

        assert isinstance(result, dict)

    def test_mismatched_id_does_not_merely_log_warning(self):
        """The mismatch must raise, not silently log.  Verify no warning-only path."""
        from missy.mcp.client import McpClient

        fixed_req_id = "req-log-check"
        wrong_resp_id = "resp-different"

        client = self._make_client_with_response(wrong_resp_id)

        raised = False
        with (
            patch(
                "missy.mcp.client.uuid.uuid4",
                return_value=MagicMock(__str__=lambda s: fixed_req_id),
            ),
            patch("missy.mcp.client.logger") as mock_logger,
        ):
            try:
                client._rpc("tools/list")
            except RuntimeError:
                raised = True

        assert raised, "Expected RuntimeError to be raised for ID mismatch"


# ---------------------------------------------------------------------------
# 6: Runtime tool error message sanitization
# ---------------------------------------------------------------------------


class TestRuntimeToolErrorSanitization:
    """AgentRuntime._execute_tool must not leak internal exception messages
    to the model when an unexpected error occurs, and must log only argument
    keys (not values) to prevent secret leakage in logs."""

    def _make_runtime(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        with patch("missy.agent.runtime.get_registry"), patch(
            "missy.agent.runtime.SessionManager"
        ):
            runtime = AgentRuntime.__new__(AgentRuntime)
            runtime._config = AgentConfig(provider="anthropic")
            return runtime

    def _make_tool_call(self, name: str = "test_tool", args: dict | None = None):
        from missy.providers.base import ToolCall

        return ToolCall(
            id="tc-001",
            name=name,
            arguments=args or {"secret_key": "super-secret-value", "param": "normal"},
        )

    def _call_execute_tool(self, runtime, tool_call, mock_registry):
        """Helper that calls _execute_tool with the registry properly patched."""
        from missy.agent.runtime import AgentRuntime

        # Ensure _TRANSIENT_ERRORS is initialised so the method body can run
        if not AgentRuntime._TRANSIENT_ERRORS:
            AgentRuntime._TRANSIENT_ERRORS = AgentRuntime._init_transient_errors()

        with patch("missy.agent.runtime.get_tool_registry", return_value=mock_registry):
            return runtime._execute_tool(
                tool_call=tool_call,
                session_id="sess-1",
                task_id="task-1",
            )

    def test_unexpected_exception_returns_sanitized_message(self):
        """An unexpected Exception during tool execution must yield the safe
        message 'Tool execution failed due to an internal error.' to the caller."""
        from missy.agent.runtime import AgentRuntime

        runtime = AgentRuntime.__new__(AgentRuntime)
        tool_call = self._make_tool_call()

        mock_registry = MagicMock()
        mock_registry.execute.side_effect = ValueError("database connection string: user:pass@host")

        result = self._call_execute_tool(runtime, tool_call, mock_registry)

        assert result.is_error is True
        assert result.content == "Tool execution failed due to an internal error."

    def test_unexpected_exception_does_not_leak_exception_message(self):
        """The original exception message must not appear in the ToolResult content."""
        from missy.agent.runtime import AgentRuntime

        runtime = AgentRuntime.__new__(AgentRuntime)
        tool_call = self._make_tool_call()

        secret_in_exception = "secret_api_key=sk-proj-abcdef"
        mock_registry = MagicMock()
        mock_registry.execute.side_effect = AttributeError(secret_in_exception)

        result = self._call_execute_tool(runtime, tool_call, mock_registry)

        assert secret_in_exception not in result.content

    def test_tool_args_logged_with_keys_only(self):
        """logger.info for tool execution must include argument keys but not values."""
        from missy.agent.runtime import AgentRuntime

        runtime = AgentRuntime.__new__(AgentRuntime)
        sensitive_value = "supersecretpassword"
        tool_call = self._make_tool_call(
            args={"password": sensitive_value, "username": "admin"}
        )

        mock_registry = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "done"
        mock_result.success = True
        mock_registry.execute.return_value = mock_result

        # Ensure _TRANSIENT_ERRORS is initialised
        if not AgentRuntime._TRANSIENT_ERRORS:
            AgentRuntime._TRANSIENT_ERRORS = AgentRuntime._init_transient_errors()

        logged_messages = []

        def capture_info(fmt, *args, **kwargs):
            try:
                logged_messages.append(fmt % args if args else fmt)
            except Exception:
                logged_messages.append(str(fmt))

        with (
            patch("missy.agent.runtime.get_tool_registry", return_value=mock_registry),
            patch("missy.agent.runtime.logger") as mock_logger,
        ):
            mock_logger.info.side_effect = capture_info
            runtime._execute_tool(
                tool_call=tool_call,
                session_id="sess-1",
                task_id="task-1",
            )

        # The sensitive value must not appear in any logged message
        combined_log = " ".join(logged_messages)
        assert sensitive_value not in combined_log

    def test_tool_args_keys_appear_in_log(self):
        """The argument key names (not values) should appear in the info log."""
        from missy.agent.runtime import AgentRuntime

        runtime = AgentRuntime.__new__(AgentRuntime)
        tool_call = self._make_tool_call(
            args={"file_path": "/etc/passwd", "mode": "read"}
        )

        mock_registry = MagicMock()
        mock_result = MagicMock()
        mock_result.output = "content"
        mock_result.success = True
        mock_registry.execute.return_value = mock_result

        if not AgentRuntime._TRANSIENT_ERRORS:
            AgentRuntime._TRANSIENT_ERRORS = AgentRuntime._init_transient_errors()

        # Capture every argument passed to logger.info across all calls
        all_logged_args: list[str] = []

        def capture_info(fmt, *args, **kwargs):
            # Stringify the full call — format string plus positional args
            all_logged_args.append(str(fmt))
            all_logged_args.extend(str(a) for a in args)

        with (
            patch("missy.agent.runtime.get_tool_registry", return_value=mock_registry),
            patch("missy.agent.runtime.logger") as mock_logger,
        ):
            mock_logger.info.side_effect = capture_info
            runtime._execute_tool(
                tool_call=tool_call,
                session_id="sess-1",
                task_id="task-1",
            )

        combined_log = " ".join(all_logged_args)
        # Key names are passed as a list via list(tool_call.arguments.keys())
        assert "file_path" in combined_log or "mode" in combined_log
        # The actual path value must not appear
        assert "/etc/passwd" not in combined_log
