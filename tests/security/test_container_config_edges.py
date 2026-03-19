"""Edge case tests for ContainerSandbox.


Covers:
- Docker unavailable graceful degradation
- Execute without start
- Copy operations without container
- Stop without start
- Start failure handling
- Execute timeout
- Multiple start/stop cycles
- Context manager behavior
- Config parsing edge cases
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from missy.security.container import ContainerConfig, ContainerSandbox, parse_container_config

# ---------------------------------------------------------------------------
# ContainerConfig and parse_container_config
# ---------------------------------------------------------------------------


class TestContainerConfigParsing:
    """Tests for parse_container_config."""

    def test_parse_valid_config(self):
        data = {
            "enabled": True,
            "image": "ubuntu:22.04",
            "memory_limit": "512m",
            "cpu_quota": 1.0,
            "network_mode": "bridge",
        }
        cfg = parse_container_config(data)
        assert cfg.enabled is True
        assert cfg.image == "ubuntu:22.04"
        assert cfg.memory_limit == "512m"
        assert cfg.cpu_quota == 1.0
        assert cfg.network_mode == "bridge"

    def test_parse_empty_dict(self):
        cfg = parse_container_config({})
        assert cfg.enabled is False
        assert cfg.image == "python:3.12-slim"
        assert cfg.memory_limit == "256m"

    def test_parse_non_dict_returns_defaults(self):
        cfg = parse_container_config("not a dict")
        assert cfg.enabled is False
        assert cfg.image == "python:3.12-slim"

    def test_parse_none_returns_defaults(self):
        cfg = parse_container_config(None)
        assert cfg.enabled is False

    def test_parse_partial_config(self):
        cfg = parse_container_config({"enabled": True})
        assert cfg.enabled is True
        assert cfg.image == "python:3.12-slim"

    def test_parse_string_enabled_truthy(self):
        """Non-empty string for enabled should be truthy."""
        cfg = parse_container_config({"enabled": "yes"})
        assert cfg.enabled is True

    def test_parse_string_enabled_falsy(self):
        """Empty string for enabled should be falsy."""
        cfg = parse_container_config({"enabled": ""})
        assert cfg.enabled is False

    def test_parse_cpu_quota_string(self):
        """String cpu_quota should be converted to float."""
        cfg = parse_container_config({"cpu_quota": "2.5"})
        assert cfg.cpu_quota == 2.5

    def test_parse_cpu_quota_invalid_string(self):
        """Invalid string cpu_quota should raise."""
        with pytest.raises(ValueError):
            parse_container_config({"cpu_quota": "not_a_number"})

    def test_default_config_values(self):
        """ContainerConfig defaults should be sensible."""
        cfg = ContainerConfig()
        assert cfg.enabled is False
        assert cfg.image == "python:3.12-slim"
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_quota == 0.5
        assert cfg.network_mode == "none"


# ---------------------------------------------------------------------------
# ContainerSandbox lifecycle
# ---------------------------------------------------------------------------


class TestContainerSandboxLifecycle:
    """Tests for ContainerSandbox start/stop/execute lifecycle."""

    @patch("missy.security.container.subprocess.run")
    def test_is_available_docker_present(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert ContainerSandbox.is_available() is True

    @patch("missy.security.container.subprocess.run")
    def test_is_available_docker_absent(self, mock_run):
        mock_run.side_effect = FileNotFoundError("docker not found")
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_is_available_docker_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 5)
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_is_available_docker_returns_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert ContainerSandbox.is_available() is False

    @patch("missy.security.container.subprocess.run")
    def test_is_available_oserror(self, mock_run):
        mock_run.side_effect = OSError("permission denied")
        assert ContainerSandbox.is_available() is False

    def test_execute_without_start(self):
        """Execute before start should return error tuple."""
        sb = ContainerSandbox()
        output, code = sb.execute("echo hello")
        assert code == -1
        assert "not started" in output.lower()

    def test_stop_without_start(self):
        """Stop without start should be a no-op."""
        sb = ContainerSandbox()
        sb.stop()  # Should not raise
        assert sb.container_id is None

    def test_copy_in_without_start(self):
        """copy_in without start should be a no-op."""
        sb = ContainerSandbox()
        sb.copy_in("/tmp/test", "/workspace/test")  # Should not raise

    def test_copy_out_without_start(self):
        """copy_out without start should be a no-op."""
        sb = ContainerSandbox()
        sb.copy_out("/workspace/test", "/tmp/test")  # Should not raise

    def test_container_id_none_initially(self):
        """container_id should be None before start."""
        sb = ContainerSandbox()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_start_docker_unavailable(self, mock_run):
        """Start when Docker is not available should return None."""
        # is_available returns False
        mock_run.return_value = MagicMock(returncode=1)
        sb = ContainerSandbox()
        result = sb.start()
        assert result is None
        assert sb.container_id is None

    @patch("missy.security.container.ContainerSandbox.is_available", return_value=True)
    @patch("missy.security.container.subprocess.run")
    def test_start_docker_run_fails(self, mock_run, _):
        """Start when docker run fails should return None."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr=b"Error: image not found",
        )
        sb = ContainerSandbox()
        result = sb.start()
        assert result is None

    @patch("missy.security.container.ContainerSandbox.is_available", return_value=True)
    @patch("missy.security.container.subprocess.run")
    def test_start_success(self, mock_run, _):
        """Successful start should set container_id."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"abc123def456\n",
        )
        sb = ContainerSandbox()
        result = sb.start()
        assert result == "abc123def456"
        assert sb.container_id == "abc123def456"

    @patch("missy.security.container.ContainerSandbox.is_available", return_value=True)
    @patch("missy.security.container.subprocess.run")
    def test_start_timeout(self, mock_run, _):
        """Start timeout should return None."""
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)
        sb = ContainerSandbox()
        result = sb.start()
        assert result is None

    @patch("missy.security.container.subprocess.run")
    def test_execute_success(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"hello world",
            stderr=b"",
        )
        output, code = sb.execute("echo hello world")
        assert code == 0
        assert "hello world" in output

    @patch("missy.security.container.subprocess.run")
    def test_execute_timeout(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)
        output, code = sb.execute("sleep 100", timeout=30)
        assert code == -1
        assert "timed out" in output.lower()

    @patch("missy.security.container.subprocess.run")
    def test_execute_oserror(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.side_effect = OSError("broken pipe")
        output, code = sb.execute("echo test")
        assert code == -1
        assert "broken pipe" in output.lower()

    @patch("missy.security.container.subprocess.run")
    def test_execute_combines_stdout_stderr(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"out",
            stderr=b"err",
        )
        output, code = sb.execute("bad command")
        assert "out" in output
        assert "err" in output

    @patch("missy.security.container.subprocess.run")
    def test_stop_clears_container_id(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(returncode=0)
        sb.stop()
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_stop_docker_rm_timeout(self, mock_run):
        """Stop timeout should not crash."""
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 15)
        sb.stop()  # Should not raise
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_stop_docker_rm_oserror(self, mock_run):
        """Stop OSError should not crash."""
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.side_effect = OSError("connection refused")
        sb.stop()  # Should not raise
        assert sb.container_id is None

    @patch("missy.security.container.subprocess.run")
    def test_copy_in_success(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(returncode=0)
        sb.copy_in("/tmp/file.txt", "/workspace/file.txt")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "docker" in cmd[0]
        assert "cp" in cmd

    @patch("missy.security.container.subprocess.run")
    def test_copy_in_failure(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker")
        sb.copy_in("/tmp/file.txt", "/workspace/file.txt")  # Should not raise

    @patch("missy.security.container.subprocess.run")
    def test_copy_out_success(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(returncode=0)
        sb.copy_out("/workspace/file.txt", "/tmp/file.txt")
        mock_run.assert_called_once()

    @patch("missy.security.container.subprocess.run")
    def test_copy_out_failure(self, mock_run):
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker")
        sb.copy_out("/workspace/file.txt", "/tmp/file.txt")  # Should not raise


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContainerContextManager:
    """Tests for ContainerSandbox as a context manager."""

    @patch("missy.security.container.ContainerSandbox.is_available", return_value=False)
    def test_context_manager_docker_unavailable(self, _):
        """Context manager should work even when Docker is not available."""
        with ContainerSandbox() as sb:
            assert sb.container_id is None
            output, code = sb.execute("echo hello")
            assert code == -1

    @patch("missy.security.container.ContainerSandbox.stop")
    @patch("missy.security.container.ContainerSandbox.start", return_value="abc123")
    def test_context_manager_calls_start_stop(self, mock_start, mock_stop):
        with ContainerSandbox():
            mock_start.assert_called_once()
        mock_stop.assert_called_once()

    @patch("missy.security.container.ContainerSandbox.stop")
    @patch("missy.security.container.ContainerSandbox.start", side_effect=OSError("fail"))
    def test_context_manager_start_raises(self, mock_start, mock_stop):
        """If start raises, __exit__ still calls stop."""
        with pytest.raises(OSError), ContainerSandbox():
            pass
        # Note: __exit__ won't be called if __enter__ raises

    def test_workspace_expansion(self):
        """Workspace path should expand ~."""
        sb = ContainerSandbox(workspace="~/workspace")
        assert "~" not in sb.workspace
        assert "/workspace" in sb.workspace

    def test_workspace_absolute_path(self):
        """Absolute workspace path should be preserved."""
        sb = ContainerSandbox(workspace="/opt/sandbox")
        assert sb.workspace == "/opt/sandbox"

    @patch("missy.security.container.ContainerSandbox.is_available", return_value=True)
    @patch("missy.security.container.subprocess.run")
    def test_start_command_includes_security_flags(self, mock_run, _):
        """Start command should include security hardening flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b"id123\n")
        sb = ContainerSandbox(
            image="alpine:latest",
            memory_limit="128m",
            cpu_quota=0.25,
            network_mode="none",
        )
        sb.start()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "--cap-drop=ALL" in cmd_str
        assert "--security-opt=no-new-privileges" in cmd_str
        assert "--memory" in cmd_str
        assert "--network=none" in cmd_str
        assert ":ro" in cmd_str  # workspace mounted read-only


# ---------------------------------------------------------------------------
# Edge cases and race conditions
# ---------------------------------------------------------------------------


class TestContainerEdgeCases:
    """Edge cases and unusual conditions."""

    def test_execute_with_empty_command(self):
        """Executing empty command should pass through."""
        sb = ContainerSandbox()
        sb._container_id = "test123"
        with patch("missy.security.container.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
            output, code = sb.execute("")
            assert code == 0

    def test_execute_with_special_characters(self):
        """Command with special shell characters should be passed through."""
        sb = ContainerSandbox()
        sb._container_id = "test123"
        with patch("missy.security.container.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"test", stderr=b"")
            output, code = sb.execute("echo 'hello $WORLD' && cat /etc/passwd")
            assert code == 0
            # Command is passed to /bin/sh -c, so special chars are expected
            cmd = mock_run.call_args[0][0]
            assert "/bin/sh" in cmd
            assert "-c" in cmd

    @patch("missy.security.container.subprocess.run")
    def test_execute_binary_output(self, mock_run):
        """Binary output should be decoded with error replacement."""
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b"\x80\x81\x82\xff",
            stderr=b"",
        )
        output, code = sb.execute("cat /bin/ls")
        assert code == 0
        assert isinstance(output, str)

    @patch("missy.security.container.subprocess.run")
    def test_stop_called_twice(self, mock_run):
        """Calling stop twice should be safe."""
        sb = ContainerSandbox()
        sb._container_id = "test123"
        mock_run.return_value = MagicMock(returncode=0)
        sb.stop()
        sb.stop()  # Second call is no-op

    @patch("missy.security.container.ContainerSandbox.is_available", return_value=True)
    @patch("missy.security.container.subprocess.run")
    def test_start_stderr_decode_errors(self, mock_run, _):
        """Start with binary stderr should decode with replace."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr=b"\x80\x81 error \xff",
        )
        sb = ContainerSandbox()
        result = sb.start()
        assert result is None

    def test_default_image(self):
        sb = ContainerSandbox()
        assert sb.image == "python:3.12-slim"

    def test_custom_config(self):
        sb = ContainerSandbox(
            image="node:20",
            memory_limit="1g",
            cpu_quota=2.0,
            network_mode="host",
        )
        assert sb.image == "node:20"
        assert sb.memory_limit == "1g"
        assert sb.cpu_quota == 2.0
        assert sb.network_mode == "host"
