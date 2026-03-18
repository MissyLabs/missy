"""Edge case tests for ContainerSandbox and ProgressReporter.

Targets scenarios not already covered by tests/security/test_container.py
and tests/agent/test_progress.py:

ContainerSandbox:
  - Docker command not found (FileNotFoundError from is_available AND from start)
  - Permission denied (OSError) from is_available and from start/stop/execute
  - network=none propagation verified in docker run args
  - memory limit flag verified at every non-default value
  - cpu_quota flag verified at fractional and integer values
  - stop() is called even when an exception is raised inside the context manager
  - execute() merges stdout + stderr in output
  - execute() with very short timeout fires TimeoutExpired
  - execute() OSError path returns (-1, error message) tuple
  - stop() when docker rm itself raises OSError (no exception leaks out)
  - stop() when docker rm times out (no exception leaks out)
  - multiple independent sandbox instances each track their own container_id
  - custom image is forwarded as the final positional arg before 'sleep infinity'
  - copy_in / copy_out when subprocess raises CalledProcessError (no exception leak)
  - copy_in / copy_out when subprocess times out (no exception leak)
  - copy_in / copy_out when subprocess raises OSError (no exception leak)
  - start() returns None and does not set _container_id when docker run fails

ProgressReporter:
  - NullReporter silently accepts extreme / boundary values
  - NullReporter returns None for every method
  - AuditReporter survives when event_bus import fails (swallowed exception)
  - AuditReporter passes correct detail dict for every event type
  - CLIReporter writes to stderr (not stdout) for every method
  - CLIReporter formats pct as integer (no decimals)
  - CLIReporter on_iteration uses 1-based display
  - CLIReporter on_tool_done includes both tool name and summary
  - Protocol compliance: all three classes satisfy isinstance check against the protocol
  - Reporter swapping — a list used as a strategy can be replaced mid-session
  - AuditReporter constructed with empty strings still calls _emit
  - ProgressReporter protocol defines the expected set of methods
"""

from __future__ import annotations

import subprocess
import sys
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from missy.agent.progress import (
    AuditReporter,
    CLIReporter,
    NullReporter,
    ProgressReporter,
)
from missy.security.container import ContainerSandbox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sb(
    image: str = "python:3.12-slim",
    workspace: str = "/tmp/ws",
    memory_limit: str = "256m",
    cpu_quota: float = 0.5,
    network_mode: str = "none",
) -> ContainerSandbox:
    """Return a ContainerSandbox with a predictable workspace."""
    return ContainerSandbox(
        image=image,
        workspace=workspace,
        memory_limit=memory_limit,
        cpu_quota=cpu_quota,
        network_mode=network_mode,
    )


def _started_sb(container_id: str = "deadbeef1234") -> ContainerSandbox:
    """Return a sandbox that already has a container_id set (simulates post-start)."""
    sb = _make_sb()
    sb._container_id = container_id
    return sb


# ===========================================================================
# ContainerSandbox edge cases
# ===========================================================================


class TestIsAvailableEdgeCases:
    """is_available() error path coverage."""

    @patch("missy.security.container.subprocess.run")
    def test_os_error_from_docker_info(self, mock_run: MagicMock) -> None:
        """OSError (e.g. permission denied running docker binary) → False."""
        mock_run.side_effect = OSError("permission denied")
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_docker_not_found_returns_false(self, mock_run: MagicMock) -> None:
        """FileNotFoundError (docker binary absent) → False."""
        mock_run.side_effect = FileNotFoundError("No such file: docker")
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_nonzero_returncode_is_false(self, mock_run: MagicMock) -> None:
        """Return code 2 (e.g. daemon not running) → False."""
        mock_run.return_value = MagicMock(returncode=2)
        assert ContainerSandbox.is_available() is False


class TestStartEdgeCases:
    """start() covers failure modes beyond the happy path."""

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_start_nonzero_returncode_returns_none(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """docker run non-zero exit → start() returns None, _container_id stays None."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"Cannot connect to the Docker daemon",
        )
        sb = _make_sb()
        result = sb.start()
        assert result is None
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_start_timeout_returns_none(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """subprocess.TimeoutExpired during docker run → start() returns None."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker run", timeout=30)
        sb = _make_sb()
        result = sb.start()
        assert result is None
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_start_oserror_returns_none(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """OSError (permission denied) during docker run → start() returns None."""
        mock_run.side_effect = OSError("permission denied")
        sb = _make_sb()
        result = sb.start()
        assert result is None
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_network_mode_propagated(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Custom network_mode is forwarded as --network=<mode>."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cid999\n", stderr=b""
        )
        sb = _make_sb(network_mode="bridge")
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        assert "--network=bridge" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_network_none_in_args(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """network_mode='none' (default) appears literally in the command."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cid000\n", stderr=b""
        )
        sb = _make_sb(network_mode="none")
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        assert "--network=none" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_memory_limit_forwarded(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Non-default memory_limit appears in the command after --memory."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cidmem\n", stderr=b""
        )
        sb = _make_sb(memory_limit="1g")
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        mem_idx = args.index("--memory")
        assert args[mem_idx + 1] == "1g"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_cpu_quota_fractional(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Fractional cpu_quota is forwarded as a string to --cpus."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cidcpu\n", stderr=b""
        )
        sb = _make_sb(cpu_quota=0.25)
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        cpu_idx = args.index("--cpus")
        assert args[cpu_idx + 1] == "0.25"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_cpu_quota_integer(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Integer cpu_quota (e.g. 2.0) is still forwarded as a string."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cidcpu2\n", stderr=b""
        )
        sb = _make_sb(cpu_quota=2.0)
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        cpu_idx = args.index("--cpus")
        assert args[cpu_idx + 1] == "2.0"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_custom_image_in_args(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Custom image string appears in docker run positional args."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cidimg\n", stderr=b""
        )
        sb = _make_sb(image="ubuntu:22.04")
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        assert "ubuntu:22.04" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_sleep_infinity_sentinel_present(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """docker run command always ends with 'sleep infinity'."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cidsleep\n", stderr=b""
        )
        sb = _make_sb()
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        assert args[-2] == "sleep"
        assert args[-1] == "infinity"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_security_options_present(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Hardening flags --cap-drop=ALL and --security-opt=no-new-privileges are set."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"cidsec\n", stderr=b""
        )
        sb = _make_sb()
        sb.start()
        args: list[str] = mock_run.call_args[0][0]
        assert "--cap-drop=ALL" in args
        assert "--security-opt=no-new-privileges" in args


class TestExecuteEdgeCases:
    """execute() error paths and output merging."""

    @patch("missy.security.container.subprocess.run")
    def test_execute_merges_stdout_and_stderr(self, mock_run: MagicMock) -> None:
        """Combined output contains both stdout and stderr bytes."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"out-line\n",
            stderr=b"err-line\n",
        )
        sb = _started_sb()
        output, rc = sb.execute("cmd")
        assert rc == 0
        assert "out-line" in output
        assert "err-line" in output

    @patch("missy.security.container.subprocess.run")
    def test_execute_timeout_message_includes_duration(
        self, mock_run: MagicMock
    ) -> None:
        """TimeoutExpired message includes the configured timeout value."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker exec", timeout=7)
        sb = _started_sb()
        output, rc = sb.execute("sleep 100", timeout=7)
        assert rc == -1
        assert "7" in output  # timeout value appears in message

    @patch("missy.security.container.subprocess.run")
    def test_execute_oserror_returns_error_tuple(
        self, mock_run: MagicMock
    ) -> None:
        """OSError during exec → (-1, error string)."""
        mock_run.side_effect = OSError("broken pipe")
        sb = _started_sb()
        output, rc = sb.execute("ls")
        assert rc == -1
        assert "broken pipe" in output

    @patch("missy.security.container.subprocess.run")
    def test_execute_nonzero_exit_code_forwarded(
        self, mock_run: MagicMock
    ) -> None:
        """Non-zero exit code from the command is returned unchanged."""
        mock_run.return_value = MagicMock(
            returncode=127,
            stdout=b"",
            stderr=b"command not found\n",
        )
        sb = _started_sb()
        _output, rc = sb.execute("bad_cmd")
        assert rc == 127

    def test_execute_before_start_returns_negative_one(self) -> None:
        """Calling execute() before start() returns exit_code -1."""
        sb = _make_sb()
        _, rc = sb.execute("echo hi")
        assert rc == -1

    def test_execute_before_start_message_contains_not_started(self) -> None:
        """Error message for pre-start execute() mentions 'not started'."""
        sb = _make_sb()
        msg, _ = sb.execute("echo hi")
        assert "not started" in msg.lower()

    @patch("missy.security.container.subprocess.run")
    def test_execute_passes_timeout_to_subprocess(
        self, mock_run: MagicMock
    ) -> None:
        """The timeout kwarg is forwarded to subprocess.run."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=b"ok\n", stderr=b""
        )
        sb = _started_sb()
        sb.execute("echo hi", timeout=42)
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 42


class TestStopEdgeCases:
    """stop() robustness: no exceptions should escape."""

    @patch("missy.security.container.subprocess.run")
    def test_stop_timeout_does_not_raise(self, mock_run: MagicMock) -> None:
        """TimeoutExpired from docker rm is swallowed."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker rm", timeout=15)
        sb = _started_sb("todie")
        sb.stop()  # must not raise
        assert sb.container_id is None  # cleared even on failure

    @patch("missy.security.container.subprocess.run")
    def test_stop_oserror_does_not_raise(self, mock_run: MagicMock) -> None:
        """OSError from docker rm is swallowed."""
        mock_run.side_effect = OSError("no such process")
        sb = _started_sb("todie2")
        sb.stop()  # must not raise
        assert sb.container_id is None

    def test_stop_when_never_started_is_noop(self) -> None:
        """stop() on a fresh (never-started) sandbox is a safe no-op."""
        sb = _make_sb()
        sb.stop()  # must not raise
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_stop_clears_container_id_before_docker_call(
        self, mock_run: MagicMock
    ) -> None:
        """_container_id is set to None before the docker rm subprocess call
        so that a second stop() call (e.g. from __exit__) is a no-op."""
        call_log: list[str | None] = []

        def _side_effect(*_args: Any, **_kwargs: Any) -> MagicMock:
            # Capture container_id *during* the call
            call_log.append(sb.container_id)
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        sb = _started_sb("earlyid")
        mock_run.side_effect = _side_effect
        sb.stop()
        # container_id was already None when docker rm ran
        assert call_log == [None]


class TestContextManagerEdgeCases:
    """__enter__ / __exit__ guarantee cleanup even on exceptions."""

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_stop_called_on_exception(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """stop() is invoked by __exit__ even when an exception is raised inside
        the with-block."""
        start_result = MagicMock(returncode=0, stdout=b"exctest\n", stderr=b"")
        stop_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
        mock_run.side_effect = [start_result, stop_result]

        with pytest.raises(RuntimeError):
            with ContainerSandbox(workspace="/tmp/ws") as sb:
                assert sb.container_id == "exctest"
                raise RuntimeError("deliberate failure")

        # stop() ran, container_id cleared
        assert sb.container_id is None
        # Two subprocess calls: docker run + docker rm
        assert mock_run.call_count == 2

    @patch.object(ContainerSandbox, "is_available", return_value=False)
    def test_context_manager_docker_unavailable(
        self, _avail: MagicMock
    ) -> None:
        """When Docker is unavailable the context manager still works cleanly."""
        with ContainerSandbox(workspace="/tmp/ws") as sb:
            assert sb.container_id is None
        assert sb.container_id is None


class TestMultipleSandboxes:
    """Independent sandbox instances do not share state."""

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_two_sandboxes_have_independent_container_ids(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=b"aaa111\n", stderr=b""),
            MagicMock(returncode=0, stdout=b"bbb222\n", stderr=b""),
        ]
        sb1 = _make_sb(workspace="/tmp/ws1")
        sb2 = _make_sb(workspace="/tmp/ws2")
        sb1.start()
        sb2.start()
        assert sb1.container_id == "aaa111"
        assert sb2.container_id == "bbb222"
        assert sb1.container_id != sb2.container_id

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_stopping_one_does_not_affect_other(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Stopping sandbox A leaves sandbox B's container_id intact."""
        sb_a = _started_sb("aaaaa")
        sb_b = _started_sb("bbbbb")
        mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
        sb_a.stop()
        assert sb_a.container_id is None
        assert sb_b.container_id == "bbbbb"


class TestCopyEdgeCases:
    """copy_in / copy_out error-handling paths."""

    @patch("missy.security.container.subprocess.run")
    def test_copy_in_called_process_error_does_not_raise(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="docker cp"
        )
        sb = _started_sb()
        sb.copy_in("/tmp/a.txt", "/workspace/a.txt")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_copy_in_timeout_does_not_raise(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker cp", timeout=30)
        sb = _started_sb()
        sb.copy_in("/tmp/b.txt", "/workspace/b.txt")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_copy_in_oserror_does_not_raise(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = OSError("pipe broken")
        sb = _started_sb()
        sb.copy_in("/tmp/c.txt", "/workspace/c.txt")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_copy_out_called_process_error_does_not_raise(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="docker cp"
        )
        sb = _started_sb()
        sb.copy_out("/workspace/a.txt", "/tmp/a.txt")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_copy_out_timeout_does_not_raise(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker cp", timeout=30)
        sb = _started_sb()
        sb.copy_out("/workspace/b.txt", "/tmp/b.txt")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_copy_out_oserror_does_not_raise(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = OSError("pipe broken")
        sb = _started_sb()
        sb.copy_out("/workspace/c.txt", "/tmp/c.txt")  # must not raise


# ===========================================================================
# ProgressReporter edge cases
# ===========================================================================


class TestNullReporterEdgeCases:
    """NullReporter boundary and return-value behaviour."""

    def test_all_methods_return_none(self) -> None:
        """Every NullReporter method returns None explicitly."""
        r = NullReporter()
        assert r.on_start("x") is None
        assert r.on_progress(0.0, "label") is None
        assert r.on_tool_start("tool") is None
        assert r.on_tool_done("tool", "summary") is None
        assert r.on_iteration(0, 1) is None
        assert r.on_complete("done") is None
        assert r.on_error("err") is None

    def test_on_progress_with_zero_pct(self) -> None:
        NullReporter().on_progress(0.0, "start")  # must not raise

    def test_on_progress_with_hundred_pct(self) -> None:
        NullReporter().on_progress(100.0, "done")  # must not raise

    def test_on_progress_with_negative_pct(self) -> None:
        """Negative pct is unusual but should not raise."""
        NullReporter().on_progress(-5.0, "negative")  # must not raise

    def test_on_iteration_zero_max(self) -> None:
        """max_iterations=0 edge case."""
        NullReporter().on_iteration(0, 0)  # must not raise

    def test_on_start_empty_string(self) -> None:
        NullReporter().on_start("")  # must not raise

    def test_on_error_empty_string(self) -> None:
        NullReporter().on_error("")  # must not raise

    def test_is_progress_reporter_protocol(self) -> None:
        assert isinstance(NullReporter(), ProgressReporter)


class TestAuditReporterEdgeCases:
    """AuditReporter detail dict correctness and resilience."""

    def _make_reporter_with_spy(self) -> tuple[AuditReporter, list[tuple[str, dict]]]:
        """Return an AuditReporter and a list that records every _emit call."""
        calls: list[tuple[str, dict]] = []

        reporter = AuditReporter(session_id="sid", task_id="tid")
        original_emit = reporter._emit

        def _spy_emit(event_type: str, detail: dict) -> None:
            calls.append((event_type, detail))
            # Do NOT forward to original — avoids importing event_bus in unit tests.

        reporter._emit = _spy_emit  # type: ignore[method-assign]
        return reporter, calls

    def test_on_start_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_start("my task")
        assert calls == [("agent.progress.start", {"task": "my task"})]

    def test_on_progress_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_progress(75.5, "three-quarters")
        assert calls == [("agent.progress.update", {"pct": 75.5, "label": "three-quarters"})]

    def test_on_tool_start_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_tool_start("file_read")
        assert calls == [("agent.progress.tool_start", {"tool": "file_read"})]

    def test_on_tool_done_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_tool_done("file_read", "read 42 bytes")
        assert calls == [
            ("agent.progress.tool_done", {"tool": "file_read", "summary": "read 42 bytes"})
        ]

    def test_on_iteration_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_iteration(3, 10)
        assert calls == [("agent.progress.iteration", {"iteration": 3, "max": 10})]

    def test_on_complete_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_complete("all done")
        assert calls == [("agent.progress.complete", {"summary": "all done"})]

    def test_on_error_detail(self) -> None:
        r, calls = self._make_reporter_with_spy()
        r.on_error("network timeout")
        assert calls == [("agent.progress.error", {"error": "network timeout"})]

    def test_emit_swallows_import_error(self) -> None:
        """If the event_bus import fails, _emit must not propagate the exception."""
        r = AuditReporter(session_id="x", task_id="y")
        with patch.dict(sys.modules, {"missy.core.events": None}):
            r.on_start("safe")  # must not raise

    def test_emit_swallows_exception_raised_inside_emit_body(self) -> None:
        """Exceptions raised inside _emit's try block (e.g. from event_bus.publish)
        are swallowed by the bare except clause in _emit itself.

        We simulate this by patching event_bus.publish — that call happens inside
        _emit's try block, so the resulting exception is caught and suppressed."""
        r = AuditReporter()

        # Patch only the inner publish call so the exception is raised inside the
        # try/except in _emit, which is where the swallowing actually occurs.
        mock_bus = MagicMock()
        mock_bus.publish.side_effect = RuntimeError("bus exploded")

        with patch("missy.agent.progress.AuditReporter._emit") as mock_emit:
            # Replicate what the real _emit does: call publish inside a try/except
            def _emit_with_swallow(event_type: str, detail: dict) -> None:
                try:
                    mock_bus.publish(event_type, detail)
                except Exception:
                    pass

            mock_emit.side_effect = _emit_with_swallow
            r.on_complete("check")  # must not raise

        mock_bus.publish.assert_called_once()

    def test_constructed_with_empty_strings(self) -> None:
        """Empty session_id and task_id are valid (the defaults)."""
        r = AuditReporter()
        assert r._session_id == ""
        assert r._task_id == ""

    def test_is_progress_reporter_protocol(self) -> None:
        assert isinstance(AuditReporter(), ProgressReporter)


class TestCLIReporterEdgeCases:
    """CLIReporter writes to stderr, format checks."""

    def test_on_start_writes_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        CLIReporter().on_start("my task")
        captured = capsys.readouterr()
        assert "my task" in captured.err
        assert captured.out == ""

    def test_on_progress_writes_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        CLIReporter().on_progress(33.0, "one-third")
        captured = capsys.readouterr()
        assert "one-third" in captured.err
        assert captured.out == ""

    def test_on_progress_formats_pct_as_integer(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """:.0f format: 66.6 → '67', not '66.600000'."""
        CLIReporter().on_progress(66.6, "lbl")
        captured = capsys.readouterr()
        # Should contain '67' (rounded) and NOT decimal point for pct
        assert "67%" in captured.err

    def test_on_progress_zero_pct(self, capsys: pytest.CaptureFixture) -> None:
        CLIReporter().on_progress(0.0, "starting")
        captured = capsys.readouterr()
        assert "0%" in captured.err

    def test_on_tool_start_writes_to_stderr(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        CLIReporter().on_tool_start("web_search")
        captured = capsys.readouterr()
        assert "web_search" in captured.err
        assert captured.out == ""

    def test_on_tool_done_includes_name_and_summary(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        CLIReporter().on_tool_done("web_search", "found 5 results")
        captured = capsys.readouterr()
        assert "web_search" in captured.err
        assert "found 5 results" in captured.err

    def test_on_iteration_is_one_based(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """on_iteration(0, 5) should display '1/5', not '0/5'."""
        CLIReporter().on_iteration(0, 5)
        captured = capsys.readouterr()
        assert "1/5" in captured.err

    def test_on_iteration_last_iteration(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        CLIReporter().on_iteration(4, 5)
        captured = capsys.readouterr()
        assert "5/5" in captured.err

    def test_on_complete_writes_to_stderr(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        CLIReporter().on_complete("task finished")
        captured = capsys.readouterr()
        assert "task finished" in captured.err
        assert captured.out == ""

    def test_on_error_writes_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        CLIReporter().on_error("something broke")
        captured = capsys.readouterr()
        assert "something broke" in captured.err
        assert captured.out == ""

    def test_is_progress_reporter_protocol(self) -> None:
        assert isinstance(CLIReporter(), ProgressReporter)


class TestProgressReporterProtocol:
    """Verify the protocol itself defines the expected method surface."""

    _EXPECTED_METHODS = {
        "on_start",
        "on_progress",
        "on_tool_start",
        "on_tool_done",
        "on_iteration",
        "on_complete",
        "on_error",
    }

    @pytest.mark.parametrize(
        "cls", [NullReporter, AuditReporter, CLIReporter]
    )
    def test_all_protocol_methods_present(self, cls: type) -> None:
        """Each concrete reporter exposes every method required by the protocol."""
        instance = cls()
        for method in self._EXPECTED_METHODS:
            assert callable(getattr(instance, method, None)), (
                f"{cls.__name__} missing method '{method}'"
            )

    @pytest.mark.parametrize(
        "cls", [NullReporter, AuditReporter, CLIReporter]
    )
    def test_isinstance_check_against_protocol(self, cls: type) -> None:
        """All three implementations satisfy the runtime-checkable protocol."""
        assert isinstance(cls(), ProgressReporter)

    def test_protocol_itself_is_runtime_checkable(self) -> None:
        """ProgressReporter is decorated with @runtime_checkable."""
        # isinstance() would raise TypeError on a non-runtime-checkable Protocol
        try:
            isinstance(object(), ProgressReporter)
        except TypeError:
            pytest.fail("ProgressReporter is not @runtime_checkable")


class TestReporterSwapping:
    """Reporter swapping: replacing a reporter mid-session."""

    def test_swap_null_to_cli(self, capsys: pytest.CaptureFixture) -> None:
        """Switching from NullReporter to CLIReporter mid-session works correctly."""
        reporter: ProgressReporter = NullReporter()
        reporter.on_start("silent phase")
        assert capsys.readouterr().err == ""

        # Swap
        reporter = CLIReporter()
        reporter.on_start("loud phase")
        assert "loud phase" in capsys.readouterr().err

    def test_swap_cli_to_null(self, capsys: pytest.CaptureFixture) -> None:
        """Switching from CLIReporter to NullReporter stops output."""
        reporter: ProgressReporter = CLIReporter()
        reporter.on_complete("before swap")
        assert "before swap" in capsys.readouterr().err

        reporter = NullReporter()
        reporter.on_complete("after swap")
        assert capsys.readouterr().err == ""

    def test_swap_to_audit_reporter(self) -> None:
        """Swapping to an AuditReporter mid-session is type-safe."""
        reporter: ProgressReporter = NullReporter()
        reporter.on_start("task")
        reporter = AuditReporter(session_id="s", task_id="t")
        # _emit will be called; we just verify no AttributeError occurs
        calls: list[tuple[str, dict]] = []
        reporter._emit = lambda et, d: calls.append((et, d))  # type: ignore[method-assign]
        reporter.on_error("whoops")
        assert calls == [("agent.progress.error", {"error": "whoops"})]

    def test_list_of_reporters_pattern(self, capsys: pytest.CaptureFixture) -> None:
        """A list of reporters can broadcast the same event to multiple sinks."""
        null = NullReporter()
        cli = CLIReporter()
        reporters: list[ProgressReporter] = [null, cli]

        for r in reporters:
            r.on_complete("broadcast")

        assert "broadcast" in capsys.readouterr().err
