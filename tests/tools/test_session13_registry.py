"""Session 13: Comprehensive edge-case tests for ToolRegistry.

Covers gaps not addressed by the existing test files:

- Registration: duplicate names, overwrite identity, large batches
- Execution: stripping internal kwargs, no-parameter tools, output types
- Discovery: list_tools returns a *copy*, filtering, ordering stability
- Schema / metadata: get_schema structure, repr, ToolPermissions combinations
- Security: fail-closed for every elevated permission combination, path kwarg
  inference for all four kwarg names, shell list→string coercion edge cases
- Audit events: censor is applied to error detail, event carries tool name
  and correct session/task IDs, session/task IDs default to empty strings
- Singleton: thread-safe concurrent init, registry isolation between threads
- Concurrent execution: multiple tools run concurrently without state bleed
- Edge cases: zero-tool registry, tool output of None/False/0/"", exception
  message propagation, tool replacing itself
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.tools import registry as registry_module
from missy.tools.base import BaseTool, ToolPermissions, ToolResult
from missy.tools.registry import ToolRegistry, get_tool_registry, init_tool_registry

# ---------------------------------------------------------------------------
# Reusable concrete tool implementations
# ---------------------------------------------------------------------------


class EchoTool(BaseTool):
    name = "echo"
    description = "Return the input text unchanged"
    permissions = ToolPermissions()

    def execute(self, *, text: str = "") -> ToolResult:
        return ToolResult(success=True, output=text)


class NoParamTool(BaseTool):
    """A tool that takes zero parameters."""

    name = "noop"
    description = "Does nothing, takes nothing"
    permissions = ToolPermissions()

    def execute(self) -> ToolResult:  # type: ignore[override]
        return ToolResult(success=True, output=None)


class NoneOutputTool(BaseTool):
    """Tool whose output is explicitly None on success."""

    name = "none_output"
    description = "Always returns None output"
    permissions = ToolPermissions()

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=None)


class FalseyOutputTool(BaseTool):
    """Tool whose output is a falsey-but-not-None value."""

    name = "falsey"
    description = "Returns 0, False, or empty string"
    permissions = ToolPermissions()

    def __init__(self, output: Any = 0) -> None:
        self._output = output

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=self._output)


class NetworkTool(BaseTool):
    name = "net"
    description = "Makes network requests"
    permissions = ToolPermissions(network=True, allowed_hosts=["api.example.com"])

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="network result")


class FileReadTool(BaseTool):
    name = "fs_read"
    description = "Reads a file"
    permissions = ToolPermissions(filesystem_read=True, allowed_paths=["/tmp"])

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="file content")


class FileWriteTool(BaseTool):
    name = "fs_write"
    description = "Writes a file"
    permissions = ToolPermissions(filesystem_write=True, allowed_paths=["/tmp"])

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="written")


class ShellTool(BaseTool):
    name = "shell"
    description = "Runs shell commands"
    permissions = ToolPermissions(shell=True)

    def execute(self, *, command: str | list = "echo hi", **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="done")


class CombinedPermTool(BaseTool):
    """Tool that requires all four permission types simultaneously."""

    name = "everything"
    description = "Uses all permissions"
    permissions = ToolPermissions(
        network=True,
        filesystem_read=True,
        filesystem_write=True,
        shell=True,
        allowed_hosts=["internal.corp"],
        allowed_paths=["/var/data"],
    )

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="everything ran")


class ToolWithSchema(BaseTool):
    """Tool with explicit 'parameters' class attribute for get_schema."""

    name = "schema_tool"
    description = "A tool with a declared JSON schema"
    permissions = ToolPermissions()
    parameters = {
        "query": {"type": "string", "description": "Search query", "required": True},
        "limit": {"type": "integer", "description": "Max results"},
    }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="results")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_event_bus():
    event_bus.clear()
    yield
    event_bus.clear()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Restore the module-level registry singleton after each test."""
    original = registry_module._registry
    yield
    registry_module._registry = original


# ---------------------------------------------------------------------------
# 1. Registration edge cases
# ---------------------------------------------------------------------------


class TestRegistrationEdgeCases:
    def test_register_returns_none(self):
        """register() has no return value (returns None implicitly)."""
        reg = ToolRegistry()
        result = reg.register(EchoTool())
        assert result is None

    def test_overwrite_preserves_new_instance_identity(self):
        """After overwriting, get() returns the exact new object, not a copy."""
        reg = ToolRegistry()
        first = EchoTool()
        second = EchoTool()
        reg.register(first)
        reg.register(second)
        assert reg.get("echo") is second
        assert reg.get("echo") is not first

    def test_re_register_same_object_is_idempotent(self):
        """Registering the same object twice is safe; the same object is stored."""
        reg = ToolRegistry()
        tool = EchoTool()
        reg.register(tool)
        reg.register(tool)
        assert reg.get("echo") is tool
        assert reg.list_tools() == ["echo"]

    def test_register_many_tools_all_retrievable(self):
        """Registering 20 tools retains all of them."""
        reg = ToolRegistry()
        tools = []
        for i in range(20):

            class _T(BaseTool):
                name = f"tool_{i:02d}"
                description = f"Tool number {i}"
                permissions = ToolPermissions()

                def execute(self, **kwargs: Any) -> ToolResult:
                    return ToolResult(success=True, output=self.name)

            instance = _T()
            tools.append(instance)
            reg.register(instance)

        assert len(reg.list_tools()) == 20
        for t in tools:
            assert reg.get(t.name) is t

    def test_register_tool_with_empty_allowed_paths(self):
        """ToolPermissions with filesystem_read=True but empty allowed_paths."""

        class NoPathTool(BaseTool):
            name = "no_path"
            description = "FS read but no allowed_paths"
            permissions = ToolPermissions(filesystem_read=True, allowed_paths=[])

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(NoPathTool())
        assert reg.get("no_path") is not None

    def test_register_tool_with_empty_allowed_hosts(self):
        """ToolPermissions with network=True but no allowed_hosts."""

        class HostlessTool(BaseTool):
            name = "no_host"
            description = "Network but no hosts"
            permissions = ToolPermissions(network=True, allowed_hosts=[])

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output="ok")

        mock_engine = MagicMock()
        reg = ToolRegistry()
        reg.register(HostlessTool())
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine):
            result = reg.execute("no_host")
        # No hosts to check means check_network is never called; tool should succeed.
        mock_engine.check_network.assert_not_called()
        assert result.success is True


# ---------------------------------------------------------------------------
# 2. list_tools isolation
# ---------------------------------------------------------------------------


class TestListToolsIsolation:
    def test_list_tools_returns_new_list_each_call(self):
        """Mutating the returned list must not affect the registry."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        names = reg.list_tools()
        names.clear()
        # Registry still has the tool.
        assert reg.list_tools() == ["echo"]

    def test_list_tools_sorted_after_multiple_insertions(self):
        """Insertion order must not bleed into the sorted output."""
        reg = ToolRegistry()
        for name in ["zebra", "alpha", "mango", "beta"]:

            class _T(BaseTool):
                description = "x"
                permissions = ToolPermissions()

                def execute(self, **kw: Any) -> ToolResult:
                    return ToolResult(success=True, output="x")

            _T.name = name
            reg.register(_T())

        assert reg.list_tools() == sorted(["zebra", "alpha", "mango", "beta"])

    def test_list_tools_reflects_overwrite(self):
        """After overwriting a tool the name appears exactly once in list."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(EchoTool())  # overwrite
        names = reg.list_tools()
        assert names.count("echo") == 1

    def test_empty_registry_list_is_empty_list_type(self):
        """An empty registry returns [] not some other falsey object."""
        reg = ToolRegistry()
        result = reg.list_tools()
        assert result == []
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 3. Execution with falsey / None outputs
# ---------------------------------------------------------------------------


class TestFalseyOutputs:
    def test_none_output_is_preserved(self):
        """success=True with output=None is a valid, distinct result."""
        reg = ToolRegistry()
        reg.register(NoneOutputTool())
        result = reg.execute("none_output")
        assert result.success is True
        assert result.output is None
        assert result.error is None

    def test_zero_output_is_preserved(self):
        reg = ToolRegistry()
        reg.register(FalseyOutputTool(output=0))
        result = reg.execute("falsey")
        assert result.success is True
        assert result.output == 0

    def test_false_output_is_preserved(self):
        reg = ToolRegistry()
        reg.register(FalseyOutputTool(output=False))
        result = reg.execute("falsey")
        assert result.success is True
        assert result.output is False

    def test_empty_string_output_is_preserved(self):
        reg = ToolRegistry()
        reg.register(FalseyOutputTool(output=""))
        result = reg.execute("falsey")
        assert result.success is True
        assert result.output == ""

    def test_no_param_tool_succeeds(self):
        """A tool whose execute() accepts no keyword arguments."""
        reg = ToolRegistry()
        reg.register(NoParamTool())
        result = reg.execute("noop")
        assert result.success is True
        assert result.output is None


# ---------------------------------------------------------------------------
# 4. Internal-key stripping is thorough
# ---------------------------------------------------------------------------


class TestKwargStripping:
    def test_session_id_not_forwarded(self):
        """session_id must be stripped before calling tool.execute()."""
        tool = MagicMock(spec=BaseTool)
        tool.name = "spy"
        tool.permissions = ToolPermissions()
        tool.execute.return_value = ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(tool)
        reg.execute("spy", session_id="s99")
        _, kwargs = tool.execute.call_args
        assert "session_id" not in kwargs

    def test_task_id_not_forwarded(self):
        """task_id must be stripped before calling tool.execute()."""
        tool = MagicMock(spec=BaseTool)
        tool.name = "spy"
        tool.permissions = ToolPermissions()
        tool.execute.return_value = ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(tool)
        reg.execute("spy", task_id="t42")
        _, kwargs = tool.execute.call_args
        assert "task_id" not in kwargs

    def test_custom_kwargs_are_forwarded(self):
        """Domain kwargs (not session_id / task_id) must reach the tool."""
        tool = MagicMock(spec=BaseTool)
        tool.name = "spy"
        tool.permissions = ToolPermissions()
        tool.execute.return_value = ToolResult(success=True, output="ok")

        reg = ToolRegistry()
        reg.register(tool)
        reg.execute("spy", session_id="s1", task_id="t1", x=1, y="hello")
        tool.execute.assert_called_once_with(x=1, y="hello")


# ---------------------------------------------------------------------------
# 5. Exception message propagation
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    def test_exception_message_in_error_field(self):
        """The original exception message must appear verbatim in ToolResult.error."""
        msg = "Unexpected internal state: counter overflowed at 2**32"
        tool = MagicMock(spec=BaseTool)
        tool.name = "overflower"
        tool.permissions = ToolPermissions()
        tool.execute.side_effect = OverflowError(msg)

        reg = ToolRegistry()
        reg.register(tool)
        result = reg.execute("overflower")
        assert result.success is False
        assert msg in result.error

    def test_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt is NOT caught by the broad except clause."""
        tool = MagicMock(spec=BaseTool)
        tool.name = "interrupter"
        tool.permissions = ToolPermissions()
        tool.execute.side_effect = KeyboardInterrupt

        reg = ToolRegistry()
        reg.register(tool)
        with pytest.raises(KeyboardInterrupt):
            reg.execute("interrupter")

    def test_system_exit_propagates(self):
        """SystemExit is NOT caught by the broad except clause."""
        tool = MagicMock(spec=BaseTool)
        tool.name = "quitter"
        tool.permissions = ToolPermissions()
        tool.execute.side_effect = SystemExit(1)

        reg = ToolRegistry()
        reg.register(tool)
        with pytest.raises(SystemExit):
            reg.execute("quitter")

    def test_exception_with_no_message(self):
        """An exception whose str() is empty still produces a non-crashing result."""
        tool = MagicMock(spec=BaseTool)
        tool.name = "silent_crash"
        tool.permissions = ToolPermissions()
        tool.execute.side_effect = RuntimeError()

        reg = ToolRegistry()
        reg.register(tool)
        result = reg.execute("silent_crash")
        assert result.success is False
        # error field exists (even if it is empty string)
        assert result.error is not None


# ---------------------------------------------------------------------------
# 6. Audit event contents
# ---------------------------------------------------------------------------


class TestAuditEventContents:
    def test_audit_event_carries_tool_name(self):
        """The published audit event detail must include the tool name."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("echo", text="hi")
            event = mock_bus.publish.call_args[0][0]
        assert event.detail["tool"] == "echo"

    def test_audit_event_carries_session_and_task_id(self):
        """session_id and task_id must be forwarded to the AuditEvent."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("echo", session_id="ses-abc", task_id="task-xyz", text="x")
            event = mock_bus.publish.call_args[0][0]
        assert event.session_id == "ses-abc"
        assert event.task_id == "task-xyz"

    def test_audit_event_session_task_defaults_to_empty_string(self):
        """When session_id / task_id are omitted they default to empty string."""
        reg = ToolRegistry()
        reg.register(EchoTool())
        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("echo")
            event = mock_bus.publish.call_args[0][0]
        assert event.session_id == ""
        assert event.task_id == ""

    def test_audit_event_type_is_tool_execute(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("echo")
            event = mock_bus.publish.call_args[0][0]
        assert event.event_type == "tool_execute"

    def test_audit_event_detail_secret_censored(self):
        """Secrets in tool error messages must be redacted in audit events."""
        api_key = "sk-1234567890abcdef1234567890abcdef12345678"
        tool = MagicMock(spec=BaseTool)
        tool.name = "leaker"
        tool.permissions = ToolPermissions()
        tool.execute.side_effect = ValueError(f"Auth failed: {api_key}")

        reg = ToolRegistry()
        reg.register(tool)
        with patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("leaker")
            event = mock_bus.publish.call_args[0][0]
        # The raw API key must not appear in the audit message.
        assert api_key not in event.detail.get("message", "")

    def test_policy_deny_audit_event_result_is_deny(self):
        """A permission denial must emit result='deny', not 'error'."""
        mock_engine = MagicMock()
        mock_engine.check_network.side_effect = PolicyViolationError(
            "host blocked", category="network", detail="blocked"
        )
        reg = ToolRegistry()
        reg.register(NetworkTool())
        with patch("missy.tools.registry.get_policy_engine", return_value=mock_engine), patch("missy.tools.registry.event_bus") as mock_bus:
            reg.execute("net")
            event = mock_bus.publish.call_args[0][0]
        assert event.result == "deny"


# ---------------------------------------------------------------------------
# 7. Policy – combined permissions
# ---------------------------------------------------------------------------


class TestCombinedPermissions:
    @patch("missy.tools.registry.get_policy_engine")
    def test_all_four_checks_called_for_combined_tool(self, mock_get_engine):
        """A tool with all four permissions must trigger all four engine checks."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(CombinedPermTool())
        reg.execute("everything", command="ls")

        engine.check_network.assert_called()
        engine.check_read.assert_called()
        engine.check_write.assert_called()
        engine.check_shell.assert_called()

    @patch("missy.tools.registry.get_policy_engine")
    def test_only_read_check_for_read_only_tool(self, mock_get_engine):
        """A read-only tool must NOT call check_write or check_shell."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(FileReadTool())
        reg.execute("fs_read")

        engine.check_read.assert_called()
        engine.check_write.assert_not_called()
        engine.check_shell.assert_not_called()
        engine.check_network.assert_not_called()

    @patch("missy.tools.registry.get_policy_engine")
    def test_only_write_check_for_write_only_tool(self, mock_get_engine):
        """A write-only tool must NOT call check_read or check_network."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(FileWriteTool())
        reg.execute("fs_write")

        engine.check_write.assert_called()
        engine.check_read.assert_not_called()
        engine.check_network.assert_not_called()
        engine.check_shell.assert_not_called()

    @patch("missy.tools.registry.get_policy_engine")
    def test_file_path_kwarg_checked_for_read(self, mock_get_engine):
        """The 'file_path' kwarg variant must be checked against the read policy."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(FileReadTool())
        reg.execute("fs_read", file_path="/home/user/secret.txt")

        paths = [c[0][0] for c in engine.check_read.call_args_list]
        assert "/home/user/secret.txt" in paths

    @patch("missy.tools.registry.get_policy_engine")
    def test_target_kwarg_checked_for_write(self, mock_get_engine):
        """The 'target' kwarg variant must be checked against the write policy."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(FileWriteTool())
        reg.execute("fs_write", target="/etc/crontab")

        paths = [c[0][0] for c in engine.check_write.call_args_list]
        assert "/etc/crontab" in paths

    @patch("missy.tools.registry.get_policy_engine")
    def test_destination_kwarg_checked_for_write(self, mock_get_engine):
        """The 'destination' kwarg variant must be checked against the write policy."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(FileWriteTool())
        reg.execute("fs_write", destination="/var/log/audit.log")

        paths = [c[0][0] for c in engine.check_write.call_args_list]
        assert "/var/log/audit.log" in paths

    @patch("missy.tools.registry.get_policy_engine")
    def test_shell_command_list_coerced_to_string(self, mock_get_engine):
        """A list-valued 'command' kwarg must be joined to a string for check_shell."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.execute("shell", command=["git", "commit", "-m", "fix"])

        shell_arg = engine.check_shell.call_args[0][0]
        assert shell_arg == "git commit -m fix"

    @patch("missy.tools.registry.get_policy_engine")
    def test_shell_missing_command_kwarg_uses_default(self, mock_get_engine):
        """When no 'command' kwarg is provided, check_shell is called with 'shell'."""
        engine = MagicMock()
        mock_get_engine.return_value = engine

        reg = ToolRegistry()
        reg.register(ShellTool())
        reg.execute("shell")

        shell_arg = engine.check_shell.call_args[0][0]
        assert shell_arg == "shell"

    def test_fail_closed_for_combined_permissions(self):
        """All-permissions tool is denied when engine is absent."""
        import missy.policy.engine as pe

        old = pe._engine
        pe._engine = None
        try:
            reg = ToolRegistry()
            reg.register(CombinedPermTool())
            result = reg.execute("everything")
            assert result.success is False
            assert "policy engine not initialised" in result.error.lower()
        finally:
            pe._engine = old


# ---------------------------------------------------------------------------
# 8. Schema validation (BaseTool.get_schema)
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_get_schema_required_fields_present(self):
        """Every schema must contain name, description, and parameters."""
        schema = ToolWithSchema().get_schema()
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema

    def test_get_schema_parameters_type_is_object(self):
        schema = ToolWithSchema().get_schema()
        assert schema["parameters"]["type"] == "object"

    def test_get_schema_required_list_only_required_params(self):
        """Only params with required=True must appear in the required list."""
        schema = ToolWithSchema().get_schema()
        required = schema["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_get_schema_required_key_stripped_from_properties(self):
        """The 'required' sentinel inside each property dict must be removed."""
        schema = ToolWithSchema().get_schema()
        props = schema["parameters"]["properties"]
        assert "required" not in props.get("query", {})
        assert "required" not in props.get("limit", {})

    def test_get_schema_no_parameters_attribute_gives_empty_props(self):
        """A tool with no 'parameters' class attr returns empty props and required."""
        schema = EchoTool().get_schema()
        assert schema["parameters"]["properties"] == {}
        assert schema["parameters"]["required"] == []

    def test_tool_repr_format(self):
        """__repr__ must include both the class name and the tool name."""
        tool = EchoTool()
        r = repr(tool)
        assert "EchoTool" in r
        assert "echo" in r


# ---------------------------------------------------------------------------
# 9. Singleton thread safety
# ---------------------------------------------------------------------------


class TestSingletonThreadSafety:
    def test_concurrent_init_does_not_raise(self):
        """Multiple threads calling init_tool_registry() concurrently must not crash."""
        errors: list[Exception] = []

        def _init():
            try:
                init_tool_registry()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_init) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert isinstance(get_tool_registry(), ToolRegistry)

    def test_concurrent_init_last_write_wins(self):
        """After concurrent inits the singleton is a valid ToolRegistry."""
        registries: list[ToolRegistry] = []
        lock = threading.Lock()

        def _init():
            r = init_tool_registry()
            with lock:
                registries.append(r)

        threads = [threading.Thread(target=_init) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = get_tool_registry()
        assert final in registries


# ---------------------------------------------------------------------------
# 10. Concurrent execution does not bleed state
# ---------------------------------------------------------------------------


class TestConcurrentExecution:
    def test_concurrent_executions_isolated_results(self):
        """Simultaneous executions of distinct tools must return independent results."""
        reg = ToolRegistry()

        class SlowTool(BaseTool):
            description = "Echoes its own name"
            permissions = ToolPermissions()

            def __init__(self, n: str) -> None:
                self._n = n

            @property
            def name(self) -> str:  # type: ignore[override]
                return self._n

            def execute(self, **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output=self._n)

        for i in range(5):
            reg.register(SlowTool(f"worker_{i}"))

        results: dict[str, ToolResult] = {}
        lock = threading.Lock()

        def _run(tool_name: str) -> None:
            r = reg.execute(tool_name)
            with lock:
                results[tool_name] = r

        threads = [threading.Thread(target=_run, args=(f"worker_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        for name, result in results.items():
            assert result.success is True
            assert result.output == name

    def test_concurrent_registration_then_list(self):
        """Tools registered from multiple threads must all appear in list_tools."""
        reg = ToolRegistry()
        errors: list[Exception] = []

        def _register(idx: int) -> None:
            try:

                class _T(BaseTool):
                    description = "concurrent"
                    permissions = ToolPermissions()

                    def execute(self, **kw: Any) -> ToolResult:
                        return ToolResult(success=True, output="ok")

                _T.name = f"concurrent_{idx:03d}"
                reg.register(_T())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_register, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        names = reg.list_tools()
        # All 15 tools must be present (dict is the underlying store; no dupes).
        assert len(names) == 15


# ---------------------------------------------------------------------------
# 11. KeyError message is informative
# ---------------------------------------------------------------------------


class TestKeyErrorMessage:
    def test_key_error_includes_missing_name(self):
        """The KeyError message must contain the attempted tool name."""
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="definitely_not_here"):
            reg.execute("definitely_not_here")

    def test_key_error_on_empty_string_name(self):
        """An empty-string tool name raises KeyError (not a different exception)."""
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.execute("")

    def test_get_does_not_raise_for_unknown_name(self):
        """get() must return None for an unknown name, not raise KeyError."""
        reg = ToolRegistry()
        assert reg.get("definitely_not_here") is None


# ---------------------------------------------------------------------------
# 12. ToolResult contract
# ---------------------------------------------------------------------------


class TestToolResultContract:
    def test_success_result_error_is_none(self):
        r = ToolResult(success=True, output="x")
        assert r.error is None

    def test_failure_result_with_error_message(self):
        r = ToolResult(success=False, output=None, error="something broke")
        assert r.success is False
        assert r.error == "something broke"

    def test_success_with_complex_output(self):
        data = {"key": [1, 2, 3], "nested": {"a": True}}
        r = ToolResult(success=True, output=data)
        assert r.output is data

    def test_failure_result_output_can_be_non_none(self):
        """output need not be None even when success=False."""
        r = ToolResult(success=False, output="partial data", error="partial failure")
        assert r.output == "partial data"
        assert r.success is False
