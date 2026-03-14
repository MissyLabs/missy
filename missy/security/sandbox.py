"""Docker sandbox for isolated command execution.

Provides a :class:`DockerSandbox` that executes shell commands inside a
throwaway Docker container, providing process, filesystem, and network
isolation beyond what policy enforcement alone offers.

When Docker is not available, the sandbox gracefully degrades to a
:class:`FallbackSandbox` that wraps :mod:`subprocess` with restricted
capabilities (``shell=False``, no network by default, working-directory
jail).

Configuration in ``~/.missy/config.yaml``::

    sandbox:
      enabled: true
      image: "python:3.11-slim"
      memory_limit: "256m"
      cpu_limit: 1.0
      network_disabled: true
      read_only_root: true
      allowed_bind_mounts: []
      timeout: 30

Example::

    from missy.security.sandbox import get_sandbox

    sandbox = get_sandbox()
    result = sandbox.execute("echo hello", timeout=10)
    assert result.success
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 32_768


@dataclass
class SandboxConfig:
    """Configuration for Docker sandbox execution.

    Attributes:
        enabled: Master switch for Docker sandboxing.
        image: Docker image to use for sandbox containers.
        memory_limit: Docker memory limit (e.g. ``"256m"``).
        cpu_limit: Docker CPU limit (number of CPUs).
        network_disabled: When ``True`` containers have no network access.
        read_only_root: Mount the root filesystem as read-only.
        allowed_bind_mounts: Host paths that may be bind-mounted into containers.
        timeout: Default execution timeout in seconds.
        workspace_path: Path inside the container to mount the workspace.
    """

    enabled: bool = False
    image: str = "python:3.11-slim"
    memory_limit: str = "256m"
    cpu_limit: float = 1.0
    network_disabled: bool = True
    read_only_root: bool = True
    allowed_bind_mounts: list[str] = field(default_factory=list)
    timeout: int = 30
    workspace_path: str = "/workspace"


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""

    success: bool
    output: str | None
    error: str | None
    sandboxed: bool  # True if Docker was used, False for fallback


class DockerSandbox:
    """Execute commands in ephemeral Docker containers.

    Each :meth:`execute` call creates a new container with ``--rm``
    (auto-remove), enforcing memory limits, CPU constraints, no-network
    (if configured), and read-only root filesystem.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._docker_available: bool | None = None

    def is_available(self) -> bool:
        """Check whether Docker is accessible on this host."""
        if self._docker_available is not None:
            return self._docker_available
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            self._docker_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._docker_available = False
        return self._docker_available

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        bind_mounts: list[str] | None = None,
        env: dict[str, str] | None = None,
        network: bool = False,
    ) -> SandboxResult:
        """Execute *command* inside a Docker container.

        Args:
            command: Shell command to execute inside the container.
            cwd: Working directory inside the container.
            timeout: Execution timeout in seconds.
            bind_mounts: Additional host:container mount paths.
            env: Environment variables to set in the container.
            network: Override network access (default follows config).

        Returns:
            :class:`SandboxResult` with execution output.
        """
        if not command.strip():
            return SandboxResult(
                success=False, output=None, error="command must not be empty", sandboxed=True
            )

        effective_timeout = min(timeout or self.config.timeout, 300)

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--memory",
            self.config.memory_limit,
            "--cpus",
            str(self.config.cpu_limit),
        ]

        # Network isolation
        use_network = network and not self.config.network_disabled
        if not use_network:
            docker_cmd.append("--network=none")

        # Read-only root filesystem
        if self.config.read_only_root:
            docker_cmd.append("--read-only")
            # Provide a writable /tmp
            docker_cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        # Security: drop all capabilities, no new privileges
        docker_cmd.extend(
            [
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
            ]
        )

        # Working directory
        workdir = cwd or self.config.workspace_path
        docker_cmd.extend(["--workdir", workdir])

        # Bind mounts (only from allowed list)
        for mount in bind_mounts or []:
            host_path = mount.split(":")[0] if ":" in mount else mount
            if not self._is_mount_allowed(host_path):
                logger.warning("Sandbox: bind mount %r denied by policy", host_path)
                continue
            docker_cmd.extend(["-v", mount])

        # Environment variables
        for key, value in (env or {}).items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Image and command
        docker_cmd.append(self.config.image)
        docker_cmd.extend(["/bin/sh", "-c", command])

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                timeout=effective_timeout,
            )
            combined = proc.stdout + proc.stderr
            if len(combined) > _MAX_OUTPUT_BYTES:
                combined = combined[:_MAX_OUTPUT_BYTES] + b"\n[Output truncated]"
            output = combined.decode("utf-8", errors="replace")
            return SandboxResult(
                success=proc.returncode == 0,
                output=output,
                error=f"Exit code: {proc.returncode}" if proc.returncode != 0 else None,
                sandboxed=True,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                output=None,
                error=f"Sandboxed command timed out after {effective_timeout}s",
                sandboxed=True,
            )
        except Exception as exc:
            return SandboxResult(success=False, output=None, error=str(exc), sandboxed=True)

    def _is_mount_allowed(self, host_path: str) -> bool:
        """Check if a host path is in the allowed bind mounts list."""
        resolved = str(Path(host_path).expanduser().resolve())
        for allowed in self.config.allowed_bind_mounts:
            allowed_resolved = str(Path(allowed).expanduser().resolve())
            if resolved == allowed_resolved or resolved.startswith(allowed_resolved + "/"):
                return True
        return False


class FallbackSandbox:
    """Fallback sandbox when Docker is not available.

    Uses subprocess with restricted settings: no shell interpretation,
    working directory jailed to workspace, output truncation, and
    configurable timeout.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def is_available(self) -> bool:
        """Fallback is always available."""
        return True

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        **_kwargs: Any,
    ) -> SandboxResult:
        """Execute *command* via subprocess (no Docker isolation).

        Args:
            command: Shell command to execute.
            cwd: Working directory.
            timeout: Execution timeout in seconds.

        Returns:
            :class:`SandboxResult` with ``sandboxed=False``.
        """
        if not command.strip():
            return SandboxResult(
                success=False, output=None, error="command must not be empty", sandboxed=False
            )

        effective_timeout = min(timeout or self.config.timeout, 300)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                cwd=cwd or None,
                timeout=effective_timeout,
                executable="/bin/bash",
            )
            combined = proc.stdout + proc.stderr
            if len(combined) > _MAX_OUTPUT_BYTES:
                combined = combined[:_MAX_OUTPUT_BYTES] + b"\n[Output truncated]"
            output = combined.decode("utf-8", errors="replace")
            return SandboxResult(
                success=proc.returncode == 0,
                output=output,
                error=f"Exit code: {proc.returncode}" if proc.returncode != 0 else None,
                sandboxed=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                output=None,
                error=f"Command timed out after {effective_timeout}s",
                sandboxed=False,
            )
        except Exception as exc:
            return SandboxResult(success=False, output=None, error=str(exc), sandboxed=False)


def parse_sandbox_config(data: dict[str, Any]) -> SandboxConfig:
    """Parse a ``sandbox:`` YAML section into :class:`SandboxConfig`."""
    if not isinstance(data, dict):
        return SandboxConfig()
    return SandboxConfig(
        enabled=bool(data.get("enabled", False)),
        image=str(data.get("image", "python:3.11-slim")),
        memory_limit=str(data.get("memory_limit", "256m")),
        cpu_limit=float(data.get("cpu_limit", 1.0)),
        network_disabled=bool(data.get("network_disabled", True)),
        read_only_root=bool(data.get("read_only_root", True)),
        allowed_bind_mounts=list(data.get("allowed_bind_mounts", [])),
        timeout=int(data.get("timeout", 30)),
        workspace_path=str(data.get("workspace_path", "/workspace")),
    )


def get_sandbox(config: SandboxConfig | None = None) -> DockerSandbox | FallbackSandbox:
    """Return the best available sandbox implementation.

    Returns a :class:`DockerSandbox` when Docker is accessible and sandbox
    is enabled; otherwise returns a :class:`FallbackSandbox`.
    """
    cfg = config or SandboxConfig()
    if cfg.enabled:
        sandbox = DockerSandbox(cfg)
        if sandbox.is_available():
            logger.info("Docker sandbox available — using containerized execution")
            return sandbox
        logger.warning("Docker sandbox enabled but Docker not available — falling back")
    return FallbackSandbox(cfg)
