"""Tests for structured progress reporting (Feature 5)."""

from __future__ import annotations

from unittest.mock import MagicMock

from missy.agent.progress import (
    AuditReporter,
    CLIReporter,
    NullReporter,
    ProgressReporter,
)


class TestNullReporter:
    """Tests for NullReporter."""

    def test_null_reporter_no_errors(self):
        r = NullReporter()
        # All methods should be callable without error
        r.on_start("test task")
        r.on_progress(50.0, "halfway")
        r.on_tool_start("file_read")
        r.on_tool_done("file_read", "ok")
        r.on_iteration(0, 10)
        r.on_complete("done")
        r.on_error("something went wrong")

    def test_null_reporter_is_progress_reporter(self):
        r = NullReporter()
        assert isinstance(r, ProgressReporter)


class TestAuditReporter:
    """Tests for AuditReporter."""

    def test_audit_reporter_emits_events(self):
        from missy.core.events import event_bus

        received = []
        event_types = [
            "agent.progress.start",
            "agent.progress.tool_start",
            "agent.progress.tool_done",
            "agent.progress.complete",
        ]
        for et in event_types:
            event_bus.subscribe(et, lambda e, _r=received: _r.append(e))

        r = AuditReporter(session_id="test-session", task_id="test-task")
        r.on_start("test task")
        r.on_tool_start("file_read")
        r.on_tool_done("file_read", "ok")
        r.on_complete("finished")

        assert len(received) == 4

    def test_audit_reporter_is_progress_reporter(self):
        r = AuditReporter()
        assert isinstance(r, ProgressReporter)


class TestCLIReporter:
    """Tests for CLIReporter."""

    def test_cli_reporter_no_errors(self, capsys):
        r = CLIReporter()
        r.on_start("test task")
        r.on_tool_start("file_read")
        r.on_complete("done")

        captured = capsys.readouterr()
        assert "Starting: test task" in captured.err
        assert "Tool: file_read" in captured.err

    def test_cli_reporter_is_progress_reporter(self):
        r = CLIReporter()
        assert isinstance(r, ProgressReporter)


class TestRuntimeWithReporter:
    """Tests for AgentRuntime integration with ProgressReporter."""

    def test_runtime_default_null(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(provider="anthropic")
        agent = AgentRuntime(config)
        assert isinstance(agent._progress, NullReporter)

    def test_runtime_accepts_custom_reporter(self):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        mock_reporter = MagicMock(spec=ProgressReporter)
        config = AgentConfig(provider="anthropic")
        agent = AgentRuntime(config, progress_reporter=mock_reporter)
        assert agent._progress is mock_reporter
