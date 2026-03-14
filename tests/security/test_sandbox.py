"""Tests for the Docker sandbox and fallback sandbox.

No actual Docker daemon is required — Docker interactions are mocked.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from missy.security.sandbox import (
    DockerSandbox,
    FallbackSandbox,
    SandboxConfig,
    get_sandbox,
    parse_sandbox_config,
)

# ---------------------------------------------------------------------------
# SandboxConfig
# ---------------------------------------------------------------------------


class TestSandboxConfig:
    def test_defaults(self) -> None:
        cfg = SandboxConfig()
        assert not cfg.enabled
        assert cfg.image == "python:3.11-slim"
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_limit == 1.0
        assert cfg.network_disabled
        assert cfg.read_only_root
        assert cfg.allowed_bind_mounts == []
        assert cfg.timeout == 30

    def test_parse_sandbox_config(self) -> None:
        data = {
            "enabled": True,
            "image": "ubuntu:22.04",
            "memory_limit": "512m",
            "cpu_limit": 2.0,
            "network_disabled": False,
            "read_only_root": False,
            "allowed_bind_mounts": ["/tmp/work"],
            "timeout": 60,
        }
        cfg = parse_sandbox_config(data)
        assert cfg.enabled
        assert cfg.image == "ubuntu:22.04"
        assert cfg.memory_limit == "512m"
        assert cfg.cpu_limit == 2.0
        assert not cfg.network_disabled
        assert not cfg.read_only_root
        assert cfg.allowed_bind_mounts == ["/tmp/work"]
        assert cfg.timeout == 60

    def test_parse_sandbox_config_empty(self) -> None:
        cfg = parse_sandbox_config({})
        assert not cfg.enabled

    def test_parse_sandbox_config_not_dict(self) -> None:
        cfg = parse_sandbox_config("invalid")  # type: ignore
        assert not cfg.enabled


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------


class TestDockerSandbox:
    def test_is_available_true(self) -> None:
        sandbox = DockerSandbox()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert sandbox.is_available()

    def test_is_available_false(self) -> None:
        sandbox = DockerSandbox()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            assert not sandbox.is_available()

    def test_is_available_cached(self) -> None:
        sandbox = DockerSandbox()
        sandbox._docker_available = True
        assert sandbox.is_available()

    def test_execute_empty_command(self) -> None:
        sandbox = DockerSandbox()
        result = sandbox.execute("")
        assert not result.success
        assert result.error == "command must not be empty"
        assert result.sandboxed

    def test_execute_success(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b"hello\n",
                stderr=b"",
            )
            result = sandbox.execute("echo hello")
            assert result.success
            assert result.output == "hello\n"
            assert result.sandboxed

            # Verify Docker command structure
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "docker"
            assert call_args[1] == "run"
            assert "--rm" in call_args
            assert "--network=none" in call_args
            assert "--read-only" in call_args
            assert "--cap-drop=ALL" in call_args

    def test_execute_failure(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout=b"",
                stderr=b"error\n",
            )
            result = sandbox.execute("bad command")
            assert not result.success
            assert "Exit code: 1" in result.error

    def test_execute_timeout(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)
            result = sandbox.execute("sleep 999")
            assert not result.success
            assert "timed out" in result.error

    def test_execute_network_enabled(self) -> None:
        cfg = SandboxConfig(enabled=True, network_disabled=False)
        sandbox = DockerSandbox(cfg)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("curl example.com", network=True)
            call_args = mock_run.call_args[0][0]
            assert "--network=none" not in call_args

    def test_execute_network_forced_disabled(self) -> None:
        cfg = SandboxConfig(enabled=True, network_disabled=True)
        sandbox = DockerSandbox(cfg)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("curl example.com", network=True)
            call_args = mock_run.call_args[0][0]
            assert "--network=none" in call_args

    def test_execute_read_only_disabled(self) -> None:
        cfg = SandboxConfig(enabled=True, read_only_root=False)
        sandbox = DockerSandbox(cfg)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("echo hello")
            call_args = mock_run.call_args[0][0]
            assert "--read-only" not in call_args

    def test_execute_with_env(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("env", env={"FOO": "bar"})
            call_args = mock_run.call_args[0][0]
            assert "-e" in call_args
            idx = call_args.index("-e")
            assert call_args[idx + 1] == "FOO=bar"

    def test_execute_with_cwd(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("ls", cwd="/opt/app")
            call_args = mock_run.call_args[0][0]
            idx = call_args.index("--workdir")
            assert call_args[idx + 1] == "/opt/app"

    def test_bind_mount_allowed(self) -> None:
        cfg = SandboxConfig(
            enabled=True,
            allowed_bind_mounts=["/tmp/work"],
        )
        sandbox = DockerSandbox(cfg)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("ls", bind_mounts=["/tmp/work:/data"])
            call_args = mock_run.call_args[0][0]
            assert "-v" in call_args

    def test_bind_mount_denied(self) -> None:
        cfg = SandboxConfig(
            enabled=True,
            allowed_bind_mounts=["/tmp/safe"],
        )
        sandbox = DockerSandbox(cfg)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("ls", bind_mounts=["/etc/passwd:/data"])
            call_args = mock_run.call_args[0][0]
            assert "-v" not in call_args

    def test_output_truncation(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=b"x" * 50_000,
                stderr=b"",
            )
            result = sandbox.execute("big output")
            assert result.success
            assert "[Output truncated]" in result.output

    def test_timeout_capped_at_300(self) -> None:
        sandbox = DockerSandbox(SandboxConfig(enabled=True))
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"ok", stderr=b"")
            sandbox.execute("echo test", timeout=999)
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# FallbackSandbox
# ---------------------------------------------------------------------------


class TestFallbackSandbox:
    def test_is_available(self) -> None:
        sandbox = FallbackSandbox()
        assert sandbox.is_available()

    def test_execute_empty(self) -> None:
        sandbox = FallbackSandbox()
        result = sandbox.execute("")
        assert not result.success
        assert not result.sandboxed

    def test_execute_success(self) -> None:
        sandbox = FallbackSandbox()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"hello\n", stderr=b"")
            result = sandbox.execute("echo hello")
            assert result.success
            assert result.output == "hello\n"
            assert not result.sandboxed

    def test_execute_timeout(self) -> None:
        sandbox = FallbackSandbox()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 30)
            result = sandbox.execute("sleep 999")
            assert not result.success
            assert "timed out" in result.error


# ---------------------------------------------------------------------------
# get_sandbox
# ---------------------------------------------------------------------------


class TestGetSandbox:
    def test_returns_docker_when_available(self) -> None:
        cfg = SandboxConfig(enabled=True)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            sandbox = get_sandbox(cfg)
            assert isinstance(sandbox, DockerSandbox)

    def test_returns_fallback_when_docker_unavailable(self) -> None:
        cfg = SandboxConfig(enabled=True)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError
            sandbox = get_sandbox(cfg)
            assert isinstance(sandbox, FallbackSandbox)

    def test_returns_fallback_when_disabled(self) -> None:
        cfg = SandboxConfig(enabled=False)
        sandbox = get_sandbox(cfg)
        assert isinstance(sandbox, FallbackSandbox)

    def test_returns_fallback_default(self) -> None:
        sandbox = get_sandbox()
        assert isinstance(sandbox, FallbackSandbox)
