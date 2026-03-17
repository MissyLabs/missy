"""Tests for the container-per-session sandbox.

No actual Docker daemon is required — Docker interactions are mocked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from missy.security.container import (
    ContainerConfig,
    ContainerSandbox,
    parse_container_config,
)

# ---------------------------------------------------------------------------
# ContainerConfig
# ---------------------------------------------------------------------------


class TestContainerConfig:
    def test_defaults(self) -> None:
        cfg = ContainerConfig()
        assert not cfg.enabled
        assert cfg.image == "python:3.12-slim"
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_quota == 0.5
        assert cfg.network_mode == "none"

    def test_parse_container_config(self) -> None:
        data = {
            "enabled": True,
            "image": "ubuntu:22.04",
            "memory_limit": "512m",
            "cpu_quota": 2.0,
            "network_mode": "bridge",
        }
        cfg = parse_container_config(data)
        assert cfg.enabled
        assert cfg.image == "ubuntu:22.04"
        assert cfg.memory_limit == "512m"
        assert cfg.cpu_quota == 2.0
        assert cfg.network_mode == "bridge"

    def test_parse_empty_dict(self) -> None:
        cfg = parse_container_config({})
        assert not cfg.enabled
        assert cfg.image == "python:3.12-slim"

    def test_parse_non_dict(self) -> None:
        cfg = parse_container_config("not a dict")  # type: ignore[arg-type]
        assert not cfg.enabled


# ---------------------------------------------------------------------------
# ContainerSandbox.is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    @patch("missy.security.container.subprocess.run")
    def test_is_available_with_docker(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert ContainerSandbox.is_available() is True
        mock_run.assert_called_once_with(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )

    @patch("missy.security.container.subprocess.run")
    def test_is_available_without_docker(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError("docker not found")
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_is_available_docker_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker info", timeout=5)
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_is_available_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1)
        assert ContainerSandbox.is_available() is False


# ---------------------------------------------------------------------------
# ContainerSandbox.start
# ---------------------------------------------------------------------------


class TestStart:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_start_creates_container(self, _mock_avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"abc123def456\n",
            stderr=b"",
        )
        sb = ContainerSandbox(image="python:3.12-slim", workspace="/tmp/ws")
        cid = sb.start()

        assert cid == "abc123def456"
        assert sb.container_id == "abc123def456"

        # Verify docker run args
        args = mock_run.call_args[0][0]
        assert args[0:3] == ["docker", "run", "-d"]
        assert "--memory" in args
        assert "python:3.12-slim" in args

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_resource_limits(self, _mock_avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"container123\n",
            stderr=b"",
        )
        sb = ContainerSandbox(memory_limit="512m", cpu_quota=1.5, workspace="/tmp/ws")
        sb.start()

        args = mock_run.call_args[0][0]
        # Check memory flag
        mem_idx = args.index("--memory")
        assert args[mem_idx + 1] == "512m"
        # Check CPU flag
        cpu_idx = args.index("--cpus")
        assert args[cpu_idx + 1] == "1.5"

    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_network_none_default(self, _mock_avail: MagicMock, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"container456\n",
            stderr=b"",
        )
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb.start()

        args = mock_run.call_args[0][0]
        assert "--network=none" in args

    @patch.object(ContainerSandbox, "is_available", return_value=False)
    def test_start_without_docker(self, _mock_avail: MagicMock) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        cid = sb.start()
        assert cid is None
        assert sb.container_id is None


# ---------------------------------------------------------------------------
# ContainerSandbox.execute
# ---------------------------------------------------------------------------


class TestExecute:
    @patch("missy.security.container.subprocess.run")
    def test_execute_runs_command(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"hello world\n",
            stderr=b"",
        )
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb._container_id = "abc123"

        output, exit_code = sb.execute("echo hello world")
        assert exit_code == 0
        assert "hello world" in output

        args = mock_run.call_args[0][0]
        assert args[:3] == ["docker", "exec", "abc123"]
        assert args[3:5] == ["/bin/sh", "-c"]
        assert args[5] == "echo hello world"

    def test_execute_without_start(self) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        output, exit_code = sb.execute("echo test")
        assert exit_code == -1
        assert "not started" in output.lower()

    @patch("missy.security.container.subprocess.run")
    def test_execute_timeout(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker exec", timeout=10)
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb._container_id = "abc123"

        output, exit_code = sb.execute("sleep 60", timeout=10)
        assert exit_code == -1
        assert "timed out" in output.lower()


# ---------------------------------------------------------------------------
# ContainerSandbox.stop
# ---------------------------------------------------------------------------


class TestStop:
    @patch("missy.security.container.subprocess.run")
    def test_stop_removes_container(self, mock_run: MagicMock) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb._container_id = "abc123"

        sb.stop()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["docker", "rm", "-f", "abc123"]
        assert sb.container_id is None

    def test_stop_no_container(self) -> None:
        """stop() is a no-op when no container is running."""
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb.stop()  # Should not raise
        assert sb.container_id is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    @patch("missy.security.container.subprocess.run")
    @patch.object(ContainerSandbox, "is_available", return_value=True)
    def test_context_manager(self, _mock_avail: MagicMock, mock_run: MagicMock) -> None:
        # start returns container ID
        start_result = MagicMock(returncode=0, stdout=b"ctx123\n", stderr=b"")
        # stop returns success
        stop_result = MagicMock(returncode=0, stdout=b"", stderr=b"")
        mock_run.side_effect = [start_result, stop_result]

        with ContainerSandbox(workspace="/tmp/ws") as sb:
            assert sb.container_id == "ctx123"

        # After exit, container should be removed
        assert sb.container_id is None
        assert mock_run.call_count == 2
        # Second call should be docker rm
        rm_args = mock_run.call_args_list[1][0][0]
        assert rm_args[0:3] == ["docker", "rm", "-f"]


# ---------------------------------------------------------------------------
# copy_in / copy_out
# ---------------------------------------------------------------------------


class TestCopyOperations:
    @patch("missy.security.container.subprocess.run")
    def test_copy_in(self, mock_run: MagicMock) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb._container_id = "abc123"
        sb.copy_in("/tmp/local.txt", "/workspace/remote.txt")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["docker", "cp", "/tmp/local.txt", "abc123:/workspace/remote.txt"]

    @patch("missy.security.container.subprocess.run")
    def test_copy_out(self, mock_run: MagicMock) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb._container_id = "abc123"
        sb.copy_out("/workspace/remote.txt", "/tmp/local.txt")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["docker", "cp", "abc123:/workspace/remote.txt", "/tmp/local.txt"]

    def test_copy_in_no_container(self) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb.copy_in("/tmp/local.txt", "/workspace/remote.txt")  # Should not raise

    def test_copy_out_no_container(self) -> None:
        sb = ContainerSandbox(workspace="/tmp/ws")
        sb.copy_out("/workspace/remote.txt", "/tmp/local.txt")  # Should not raise
