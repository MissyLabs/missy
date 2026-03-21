"""Comprehensive tests for missy/security/container.py.

Covers ContainerConfig, parse_container_config, and ContainerSandbox across
all public methods and failure modes.  No real Docker daemon is required —
every subprocess call is mocked.

Coverage targets (numbered to match requirements):
 1.  ContainerConfig defaults
 2.  parse_container_config with empty dict, partial dict, full dict, non-dict
 3.  ContainerSandbox.__init__ stores parameters correctly
 4.  is_available() when docker exists
 5.  is_available() when docker not found (FileNotFoundError)
 6.  is_available() when docker times out
 7.  start() when Docker not available returns None
 8.  start() when Docker available returns container ID
 9.  start() when docker run fails returns None
10.  start() when docker run times out returns None
11.  execute() when container not started returns error
12.  execute() with successful command
13.  execute() when command times out
14.  execute() with OSError
15.  copy_in() when container not started (no-op with warning)
16.  copy_in() successful
17.  copy_in() failure handling
18.  copy_out() when container not started
19.  copy_out() successful
20.  copy_out() failure handling
21.  stop() when no container (no-op)
22.  stop() successful removal
23.  stop() when removal fails
24.  stop() when removal times out
25.  Context manager integration
26.  container_id property before/after start
27.  Security flags in start command
28.  Workspace bind mount is read-only
"""

from __future__ import annotations

import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.security.container import (
    ContainerConfig,
    ContainerSandbox,
    parse_container_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKSPACE = "/tmp/test_workspace"


def _make_sandbox(**kwargs) -> ContainerSandbox:
    """Return a ContainerSandbox with a deterministic workspace."""
    kwargs.setdefault("workspace", _WORKSPACE)
    return ContainerSandbox(**kwargs)


def _started_sandbox(container_id: str = "deadbeef0000") -> ContainerSandbox:
    """Return a sandbox whose container is already running (simulates post-start state)."""
    sb = _make_sandbox()
    sb._container_id = container_id
    return sb


def _make_proc(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """Return a mock subprocess.CompletedProcess-like object."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


# ---------------------------------------------------------------------------
# 1. ContainerConfig defaults
# ---------------------------------------------------------------------------


class TestContainerConfigDefaults:
    def test_enabled_is_false_by_default(self) -> None:
        assert ContainerConfig().enabled is False

    def test_image_default(self) -> None:
        assert ContainerConfig().image == "python:3.12-slim"

    def test_memory_limit_default(self) -> None:
        assert ContainerConfig().memory_limit == "256m"

    def test_cpu_quota_default(self) -> None:
        assert ContainerConfig().cpu_quota == 0.5

    def test_network_mode_default(self) -> None:
        assert ContainerConfig().network_mode == "none"

    def test_all_fields_override(self) -> None:
        cfg = ContainerConfig(
            enabled=True,
            image="alpine:latest",
            memory_limit="1g",
            cpu_quota=2.0,
            network_mode="host",
        )
        assert cfg.enabled is True
        assert cfg.image == "alpine:latest"
        assert cfg.memory_limit == "1g"
        assert cfg.cpu_quota == 2.0
        assert cfg.network_mode == "host"


# ---------------------------------------------------------------------------
# 2. parse_container_config
# ---------------------------------------------------------------------------


class TestParseContainerConfig:
    def test_empty_dict_returns_defaults(self) -> None:
        cfg = parse_container_config({})
        assert cfg.enabled is False
        assert cfg.image == "python:3.12-slim"
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_quota == 0.5
        assert cfg.network_mode == "none"

    def test_partial_dict_overrides_only_given_fields(self) -> None:
        cfg = parse_container_config({"enabled": True, "image": "ubuntu:22.04"})
        assert cfg.enabled is True
        assert cfg.image == "ubuntu:22.04"
        # Unchanged defaults:
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_quota == 0.5
        assert cfg.network_mode == "none"

    def test_full_dict_sets_all_fields(self) -> None:
        data = {
            "enabled": True,
            "image": "debian:bullseye",
            "memory_limit": "512m",
            "cpu_quota": 1.0,
            "network_mode": "bridge",
        }
        cfg = parse_container_config(data)
        assert cfg.enabled is True
        assert cfg.image == "debian:bullseye"
        assert cfg.memory_limit == "512m"
        assert cfg.cpu_quota == 1.0
        assert cfg.network_mode == "bridge"

    def test_non_dict_input_returns_defaults(self) -> None:
        cfg = parse_container_config("invalid")  # type: ignore[arg-type]
        assert cfg.enabled is False
        assert cfg.image == "python:3.12-slim"

    def test_none_input_returns_defaults(self) -> None:
        cfg = parse_container_config(None)  # type: ignore[arg-type]
        assert cfg.enabled is False

    def test_list_input_returns_defaults(self) -> None:
        cfg = parse_container_config(["enabled", True])  # type: ignore[arg-type]
        assert cfg.enabled is False

    def test_enabled_falsy_value_coerced_to_bool(self) -> None:
        cfg = parse_container_config({"enabled": 0})
        assert cfg.enabled is False

    def test_enabled_truthy_value_coerced_to_bool(self) -> None:
        cfg = parse_container_config({"enabled": 1})
        assert cfg.enabled is True

    def test_cpu_quota_coerced_to_float(self) -> None:
        cfg = parse_container_config({"cpu_quota": "0.75"})
        assert cfg.cpu_quota == 0.75
        assert isinstance(cfg.cpu_quota, float)

    def test_image_coerced_to_str(self) -> None:
        # If someone passes a non-string, it should still become a str
        cfg = parse_container_config({"image": 42})
        assert cfg.image == "42"


# ---------------------------------------------------------------------------
# 3. ContainerSandbox.__init__ stores parameters correctly
# ---------------------------------------------------------------------------


class TestContainerSandboxInit:
    def test_image_stored(self) -> None:
        sb = _make_sandbox(image="alpine:3.18")
        assert sb.image == "alpine:3.18"

    def test_memory_limit_stored(self) -> None:
        sb = _make_sandbox(memory_limit="128m")
        assert sb.memory_limit == "128m"

    def test_cpu_quota_stored(self) -> None:
        sb = _make_sandbox(cpu_quota=0.25)
        assert sb.cpu_quota == 0.25

    def test_network_mode_stored(self) -> None:
        sb = _make_sandbox(network_mode="bridge")
        assert sb.network_mode == "bridge"

    def test_workspace_expanded(self) -> None:
        sb = ContainerSandbox(workspace="~/workspace")
        # tilde must be expanded — result must not contain ~
        assert "~" not in sb.workspace

    def test_workspace_absolute_path_unchanged(self) -> None:
        sb = _make_sandbox(workspace="/srv/data")
        assert sb.workspace == "/srv/data"

    def test_container_id_initially_none(self) -> None:
        sb = _make_sandbox()
        assert sb._container_id is None

    def test_default_image(self) -> None:
        sb = ContainerSandbox(workspace=_WORKSPACE)
        assert sb.image == "python:3.12-slim"


# ---------------------------------------------------------------------------
# 4–6. ContainerSandbox.is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @patch("missy.security.container.subprocess.run")
    def test_docker_info_success_returns_true(self, mock_run: MagicMock) -> None:
        """Requirement 4: is_available() returns True when docker info exits 0."""
        mock_run.return_value = _make_proc(returncode=0)
        assert ContainerSandbox.is_available() is True

    @patch("missy.security.container.subprocess.run")
    def test_docker_info_nonzero_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=1)
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_file_not_found_returns_false(self, mock_run: MagicMock) -> None:
        """Requirement 5: FileNotFoundError (docker binary absent) → False."""
        mock_run.side_effect = FileNotFoundError("No such file: docker")
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_timeout_returns_false(self, mock_run: MagicMock) -> None:
        """Requirement 6: TimeoutExpired → False."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker info", timeout=5)
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_oserror_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("permission denied")
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_docker_info_called_with_correct_args(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0)
        ContainerSandbox.is_available()
        mock_run.assert_called_once_with(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )


# ---------------------------------------------------------------------------
# 7. start() when Docker not available
# ---------------------------------------------------------------------------


class TestStartDockerUnavailable:
    @patch.object(ContainerSandbox, "is_available", return_value=False)
    def test_returns_none(self, _avail: MagicMock) -> None:
        """Requirement 7: start() returns None when Docker is not available."""
        sb = _make_sandbox()
        assert sb.start() is None

    @patch.object(ContainerSandbox, "is_available", return_value=False)
    def test_container_id_remains_none(self, _avail: MagicMock) -> None:
        sb = _make_sandbox()
        sb.start()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=False)
    def test_docker_run_never_called(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        _make_sandbox().start()
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# 8. start() happy path
# ---------------------------------------------------------------------------


class TestStartSuccess:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_returns_container_id(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 8: start() returns the container ID string."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"sha256abc\n")
        sb = _make_sandbox()
        cid = sb.start()
        assert cid == "sha256abc"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_sets_container_id_property(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0, stdout=b"cid12345\n")
        sb = _make_sandbox()
        sb.start()
        assert sb.container_id == "cid12345"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_docker_run_called(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0, stdout=b"cid\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        assert args[0] == "docker"
        assert args[1] == "run"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_container_id_stripped_of_whitespace(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _make_proc(returncode=0, stdout=b"  abc123  \n")
        sb = _make_sandbox()
        cid = sb.start()
        assert cid == "abc123"


# ---------------------------------------------------------------------------
# 9. start() when docker run fails
# ---------------------------------------------------------------------------


class TestStartFailure:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_nonzero_returncode_returns_none(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 9: docker run exit code != 0 → start() returns None."""
        mock_run.return_value = _make_proc(returncode=125, stderr=b"docker: Error")
        sb = _make_sandbox()
        assert sb.start() is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_nonzero_does_not_set_container_id(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _make_proc(returncode=1, stderr=b"error")
        sb = _make_sandbox()
        sb.start()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_nonzero_logged_as_error(
        self, _avail: MagicMock, mock_run: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_run.return_value = _make_proc(returncode=1, stderr=b"container error")
        with caplog.at_level(logging.ERROR, logger="missy.security.container"):
            _make_sandbox().start()
        assert any("Failed to start" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 10. start() when docker run times out
# ---------------------------------------------------------------------------


class TestStartTimeout:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_timeout_returns_none(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 10: TimeoutExpired during docker run → start() returns None."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker run", timeout=30)
        sb = _make_sandbox()
        assert sb.start() is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_timeout_does_not_set_container_id(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker run", timeout=30)
        sb = _make_sandbox()
        sb.start()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_timeout_does_not_raise(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker run", timeout=30)
        _make_sandbox().start()  # must not raise


# ---------------------------------------------------------------------------
# 11. execute() when container not started
# ---------------------------------------------------------------------------


class TestExecuteNoContainer:
    def test_returns_error_string(self) -> None:
        """Requirement 11: execute() returns an error message when not started."""
        sb = _make_sandbox()
        msg, _ = sb.execute("ls")
        assert len(msg) > 0

    def test_returns_exit_code_negative_one(self) -> None:
        sb = _make_sandbox()
        _, rc = sb.execute("ls")
        assert rc == -1

    def test_message_mentions_not_started(self) -> None:
        sb = _make_sandbox()
        msg, _ = sb.execute("ls")
        assert "not started" in msg.lower()

    def test_subprocess_not_called(self) -> None:
        with patch("missy.security.container.subprocess.run") as mock_run:
            _make_sandbox().execute("ls")
            mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# 12. execute() with successful command
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    @patch("missy.security.container.subprocess.run")
    def test_returns_stdout_and_zero_exit(self, mock_run: MagicMock) -> None:
        """Requirement 12: execute() returns combined output and exit code."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"hello\n", stderr=b"")
        sb = _started_sandbox()
        output, rc = sb.execute("echo hello")
        assert rc == 0
        assert "hello" in output

    @patch("missy.security.container.subprocess.run")
    def test_uses_docker_exec_with_sh(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0, stdout=b"ok\n")
        sb = _started_sandbox("mycontainer")
        sb.execute("echo ok")
        args = mock_run.call_args[0][0]
        assert args[0:3] == ["docker", "exec", "mycontainer"]
        assert args[3:5] == ["/bin/sh", "-c"]

    @patch("missy.security.container.subprocess.run")
    def test_command_passed_as_shell_arg(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0, stdout=b"")
        sb = _started_sandbox()
        sb.execute("cat /etc/hostname")
        args = mock_run.call_args[0][0]
        assert args[-1] == "cat /etc/hostname"

    @patch("missy.security.container.subprocess.run")
    def test_stdout_and_stderr_merged_in_output(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(
            returncode=0, stdout=b"stdout text\n", stderr=b"stderr text\n"
        )
        sb = _started_sandbox()
        output, _ = sb.execute("mixed")
        assert "stdout text" in output
        assert "stderr text" in output

    @patch("missy.security.container.subprocess.run")
    def test_nonzero_exit_code_forwarded(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=2)
        sb = _started_sandbox()
        _, rc = sb.execute("false")
        assert rc == 2

    @patch("missy.security.container.subprocess.run")
    def test_custom_timeout_forwarded_to_subprocess(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0)
        sb = _started_sandbox()
        sb.execute("sleep 1", timeout=99)
        _, kwargs = mock_run.call_args
        assert kwargs.get("timeout") == 99


# ---------------------------------------------------------------------------
# 13. execute() when command times out
# ---------------------------------------------------------------------------


class TestExecuteTimeout:
    @patch("missy.security.container.subprocess.run")
    def test_timeout_returns_negative_one(self, mock_run: MagicMock) -> None:
        """Requirement 13: TimeoutExpired during execute → exit code -1."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker exec", timeout=5)
        sb = _started_sandbox()
        _, rc = sb.execute("sleep 999", timeout=5)
        assert rc == -1

    @patch("missy.security.container.subprocess.run")
    def test_timeout_message_contains_duration(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker exec", timeout=5)
        sb = _started_sandbox()
        msg, _ = sb.execute("sleep 999", timeout=5)
        assert "5" in msg

    @patch("missy.security.container.subprocess.run")
    def test_timeout_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker exec", timeout=1)
        _started_sandbox().execute("x", timeout=1)  # must not raise


# ---------------------------------------------------------------------------
# 14. execute() with OSError
# ---------------------------------------------------------------------------


class TestExecuteOsError:
    @patch("missy.security.container.subprocess.run")
    def test_oserror_returns_negative_one(self, mock_run: MagicMock) -> None:
        """Requirement 14: OSError during execute → exit code -1."""
        mock_run.side_effect = OSError("broken pipe")
        sb = _started_sandbox()
        _, rc = sb.execute("ls")
        assert rc == -1

    @patch("missy.security.container.subprocess.run")
    def test_oserror_message_returned(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("broken pipe")
        sb = _started_sandbox()
        msg, _ = sb.execute("ls")
        assert "broken pipe" in msg

    @patch("missy.security.container.subprocess.run")
    def test_oserror_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("no device")
        _started_sandbox().execute("ls")  # must not raise


# ---------------------------------------------------------------------------
# 15–17. copy_in
# ---------------------------------------------------------------------------


class TestCopyIn:
    def test_no_container_is_noop_does_not_raise(self) -> None:
        """Requirement 15: copy_in() with no container is a silent no-op."""
        _make_sandbox().copy_in("/tmp/src.txt", "/work/dst.txt")

    def test_no_container_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="missy.security.container"):
            _make_sandbox().copy_in("/tmp/src.txt", "/work/dst.txt")
        assert any("copy_in" in r.message for r in caplog.records)

    @patch("missy.security.container.subprocess.run")
    def test_success_calls_docker_cp(self, mock_run: MagicMock) -> None:
        """Requirement 16: copy_in() calls docker cp with correct args."""
        mock_run.return_value = _make_proc(returncode=0)
        sb = _started_sandbox("cid001")
        sb.copy_in("/tmp/file.txt", "/app/file.txt")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "cp", "/tmp/file.txt", "cid001:/app/file.txt"]

    @patch("missy.security.container.subprocess.run")
    def test_called_process_error_does_not_raise(self, mock_run: MagicMock) -> None:
        """Requirement 17: CalledProcessError during copy_in is swallowed."""
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="docker cp")
        _started_sandbox().copy_in("/a", "/b")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_timeout_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker cp", timeout=30)
        _started_sandbox().copy_in("/a", "/b")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_oserror_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("pipe error")
        _started_sandbox().copy_in("/a", "/b")  # must not raise


# ---------------------------------------------------------------------------
# 18–20. copy_out
# ---------------------------------------------------------------------------


class TestCopyOut:
    def test_no_container_is_noop_does_not_raise(self) -> None:
        """Requirement 18: copy_out() with no container is a silent no-op."""
        _make_sandbox().copy_out("/work/dst.txt", "/tmp/local.txt")

    def test_no_container_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="missy.security.container"):
            _make_sandbox().copy_out("/work/dst.txt", "/tmp/local.txt")
        assert any("copy_out" in r.message for r in caplog.records)

    @patch("missy.security.container.subprocess.run")
    def test_success_calls_docker_cp(self, mock_run: MagicMock) -> None:
        """Requirement 19: copy_out() calls docker cp with correct args."""
        mock_run.return_value = _make_proc(returncode=0)
        sb = _started_sandbox("cid002")
        sb.copy_out("/app/result.txt", "/tmp/result.txt")
        args = mock_run.call_args[0][0]
        assert args == ["docker", "cp", "cid002:/app/result.txt", "/tmp/result.txt"]

    @patch("missy.security.container.subprocess.run")
    def test_called_process_error_does_not_raise(self, mock_run: MagicMock) -> None:
        """Requirement 20: CalledProcessError during copy_out is swallowed."""
        mock_run.side_effect = subprocess.CalledProcessError(returncode=1, cmd="docker cp")
        _started_sandbox().copy_out("/a", "/b")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_timeout_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker cp", timeout=30)
        _started_sandbox().copy_out("/a", "/b")  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_oserror_does_not_raise(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("io error")
        _started_sandbox().copy_out("/a", "/b")  # must not raise


# ---------------------------------------------------------------------------
# 21–24. stop
# ---------------------------------------------------------------------------


class TestStop:
    def test_no_container_is_noop(self) -> None:
        """Requirement 21: stop() with no container does nothing and does not raise."""
        sb = _make_sandbox()
        sb.stop()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_no_container_subprocess_not_called(self, mock_run: MagicMock) -> None:
        _make_sandbox().stop()
        mock_run.assert_not_called()

    @patch("missy.security.container.subprocess.run")
    def test_success_calls_docker_rm(self, mock_run: MagicMock) -> None:
        """Requirement 22: stop() calls docker rm -f <cid>."""
        mock_run.return_value = _make_proc(returncode=0)
        sb = _started_sandbox("rmme1234")
        sb.stop()
        args = mock_run.call_args[0][0]
        assert args == ["docker", "rm", "-f", "rmme1234"]

    @patch("missy.security.container.subprocess.run")
    def test_success_clears_container_id(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0)
        sb = _started_sandbox()
        sb.stop()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_nonzero_rm_does_not_raise(self, mock_run: MagicMock) -> None:
        """Requirement 23: docker rm failure is swallowed."""
        mock_run.return_value = _make_proc(returncode=1, stderr=b"no such container")
        _started_sandbox().stop()  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_rm_failure_clears_container_id(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=1)
        sb = _started_sandbox()
        sb.stop()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_timeout_does_not_raise(self, mock_run: MagicMock) -> None:
        """Requirement 24: TimeoutExpired during stop is swallowed."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker rm", timeout=15)
        _started_sandbox().stop()  # must not raise

    @patch("missy.security.container.subprocess.run")
    def test_timeout_clears_container_id(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker rm", timeout=15)
        sb = _started_sandbox()
        sb.stop()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_container_id_cleared_before_docker_call(self, mock_run: MagicMock) -> None:
        """container_id is None during the docker rm call so a second stop() is a no-op."""
        id_during_call: list[str | None] = []

        def capture(*_args, **_kwargs):
            id_during_call.append(sb.container_id)
            return _make_proc(returncode=0)

        sb = _started_sandbox("earlyid")
        mock_run.side_effect = capture
        sb.stop()
        assert id_during_call == [None]


# ---------------------------------------------------------------------------
# 25. Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_enter_starts_container(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 25: __enter__ calls start(), __exit__ calls stop()."""
        mock_run.side_effect = [
            _make_proc(returncode=0, stdout=b"ctx001\n"),  # docker run
            _make_proc(returncode=0),  # docker rm
        ]
        with ContainerSandbox(workspace=_WORKSPACE) as sb:
            assert sb.container_id == "ctx001"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_exit_stops_container(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _make_proc(returncode=0, stdout=b"ctx002\n"),
            _make_proc(returncode=0),
        ]
        with ContainerSandbox(workspace=_WORKSPACE) as sb:
            pass
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_stop_called_on_exception(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """stop() is guaranteed even when the body raises."""
        mock_run.side_effect = [
            _make_proc(returncode=0, stdout=b"ctx003\n"),
            _make_proc(returncode=0),
        ]
        with pytest.raises(ValueError), ContainerSandbox(workspace=_WORKSPACE) as sb:
            raise ValueError("deliberate")

        assert sb.container_id is None
        assert mock_run.call_count == 2

    @patch.object(ContainerSandbox, "is_available", return_value=False)
    def test_docker_unavailable_context_manager_ok(self, _avail: MagicMock) -> None:
        """Context manager works cleanly when Docker is not available."""
        with ContainerSandbox(workspace=_WORKSPACE) as sb:
            assert sb.container_id is None
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_enter_returns_sandbox_instance(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _make_proc(returncode=0, stdout=b"ctx004\n"),
            _make_proc(returncode=0),
        ]
        sb_outer = ContainerSandbox(workspace=_WORKSPACE)
        with sb_outer as sb_inner:
            assert sb_outer is sb_inner


# ---------------------------------------------------------------------------
# 26. container_id property
# ---------------------------------------------------------------------------


class TestContainerIdProperty:
    def test_property_is_none_before_start(self) -> None:
        """Requirement 26: container_id is None before start() is called."""
        sb = _make_sandbox()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_property_set_after_start(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0, stdout=b"prop_cid\n")
        sb = _make_sandbox()
        sb.start()
        assert sb.container_id == "prop_cid"

    @patch("missy.security.container.subprocess.run")
    def test_property_cleared_after_stop(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _make_proc(returncode=0)
        sb = _started_sandbox("will_be_cleared")
        sb.stop()
        assert sb.container_id is None

    def test_property_is_read_only_internally(self) -> None:
        """container_id is backed by _container_id and exposed read-only via property."""
        sb = _make_sandbox()
        sb._container_id = "injected"
        assert sb.container_id == "injected"


# ---------------------------------------------------------------------------
# 27. Security flags in start command
# ---------------------------------------------------------------------------


class TestSecurityFlags:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_cap_drop_all_present(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 27: --cap-drop=ALL is in the docker run command."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"sectest\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        assert "--cap-drop=ALL" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_no_new_privileges_present(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 27: --security-opt=no-new-privileges is in the docker run command."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"sectest\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        assert "--security-opt=no-new-privileges" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_detach_flag_present(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Container runs detached (-d)."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"detach\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        assert "-d" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_sleep_infinity_sentinel(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Container uses 'sleep infinity' as its entrypoint."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"s\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        assert args[-2] == "sleep"
        assert args[-1] == "infinity"


# ---------------------------------------------------------------------------
# 28. Workspace bind mount is read-only
# ---------------------------------------------------------------------------


class TestWorkspaceBindMount:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_workspace_mounted_readonly(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """Requirement 28: workspace bind mount ends with ':ro'."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"romount\n")
        sb = _make_sandbox(workspace="/srv/mywork")
        sb.start()
        args = mock_run.call_args[0][0]
        # Find the -v flag and its value
        v_idx = args.index("-v")
        mount_spec = args[v_idx + 1]
        assert mount_spec.endswith(":ro"), f"Expected mount spec ending ':ro', got: {mount_spec!r}"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_workspace_host_path_in_mount(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """The host-side path in the -v spec matches the workspace parameter."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"ptest\n")
        sb = _make_sandbox(workspace="/data/project")
        sb.start()
        args = mock_run.call_args[0][0]
        v_idx = args.index("-v")
        mount_spec = args[v_idx + 1]
        host_path = mount_spec.split(":")[0]
        assert host_path == "/data/project"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_workspace_container_path_is_workspace(
        self, _avail: MagicMock, mock_run: MagicMock
    ) -> None:
        """Container-side mount target is /workspace."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"cpath\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        v_idx = args.index("-v")
        mount_spec = args[v_idx + 1]
        container_path = mount_spec.split(":")[1]
        assert container_path == "/workspace"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_workdir_set_to_workspace(self, _avail: MagicMock, mock_run: MagicMock) -> None:
        """--workdir is set to /workspace."""
        mock_run.return_value = _make_proc(returncode=0, stdout=b"wd\n")
        _make_sandbox().start()
        args = mock_run.call_args[0][0]
        wd_idx = args.index("--workdir")
        assert args[wd_idx + 1] == "/workspace"
