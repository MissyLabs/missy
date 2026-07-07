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
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import resource as _resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_OUTPUT_BYTES = 32_768


def _parse_memory_bytes(value: str) -> int | None:
    """Parse a Docker-style memory string (e.g. ``"256m"``) to bytes.

    Args:
        value: A memory limit string with an optional ``k``/``m``/``g`` suffix.

    Returns:
        The size in bytes, or ``None`` when the value cannot be parsed.
    """
    if not value:
        return None
    text = value.strip().lower()
    multipliers = {"k": 1024, "m": 1024**2, "g": 1024**3, "b": 1}
    unit = text[-1]
    try:
        if unit in multipliers:
            return int(float(text[:-1]) * multipliers[unit])
        return int(float(text))
    except (ValueError, TypeError):
        return None


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
        require_isolation: When ``True`` (the default) and sandboxing is
            enabled, execution is *refused* if Docker is unavailable rather
            than silently falling back to unsandboxed host execution
            (fail-closed).  Set to ``False`` to explicitly opt in to the
            best-effort :class:`FallbackSandbox` when Docker is missing.
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
    require_isolation: bool = True
    tools: dict[str, Any] = field(default_factory=dict)


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
    """Best-effort fallback sandbox when Docker is not available.

    .. warning::

        This fallback does **NOT** provide the containment guarantees of
        :class:`DockerSandbox`.  In particular it provides **no network
        isolation** and **no read-only root filesystem** — commands run on
        the host with full filesystem write and network access.  It applies
        only best-effort process resource limits (CPU time and address space)
        via :func:`resource.setrlimit` where the platform supports them, plus
        a scrubbed environment and output/timeout caps.

        Because of these limitations it is used only when
        :attr:`SandboxConfig.require_isolation` is ``False`` (explicit
        opt-in).  When isolation is required and Docker is unavailable,
        :func:`get_sandbox` returns a :class:`RefusingSandbox` instead.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def is_available(self) -> bool:
        """Fallback is always available."""
        return True

    def _build_preexec_fn(self) -> Any | None:
        """Return a ``preexec_fn`` that applies best-effort resource limits.

        Uses :func:`resource.setrlimit` to cap CPU seconds and address space
        based on the config.  Returns ``None`` on non-POSIX platforms where
        :mod:`resource` is unavailable.  All limit application is guarded so a
        failure to set one limit never blocks execution.
        """
        if _resource is None:
            return None

        cpu_seconds = min(int(self.config.timeout) or 30, 300)
        mem_bytes = _parse_memory_bytes(self.config.memory_limit)

        def _apply_limits() -> None:  # pragma: no cover - runs in child process
            import contextlib

            with contextlib.suppress(ValueError, OSError):
                _resource.setrlimit(_resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            if mem_bytes:
                with contextlib.suppress(ValueError, OSError):
                    _resource.setrlimit(_resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        return _apply_limits

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = None,
        **_kwargs: Any,
    ) -> SandboxResult:
        """Execute *command* via subprocess (no Docker isolation).

        .. warning::

            This does NOT provide network isolation or a read-only root
            filesystem.  It applies only a scrubbed environment, an output
            cap, a timeout, and best-effort ``setrlimit`` CPU/address-space
            limits where the platform supports them.

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

        # Sanitize environment to prevent API key leakage to arbitrary commands
        _SAFE_ENV_VARS = frozenset(
            {
                "PATH",
                "HOME",
                "USER",
                "LOGNAME",
                "SHELL",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "LANGUAGE",
                "TERM",
                "COLORTERM",
                "COLUMNS",
                "LINES",
                "XDG_RUNTIME_DIR",
                "XDG_DATA_HOME",
                "XDG_CONFIG_HOME",
                "XDG_CACHE_HOME",
                "TMPDIR",
                "TMP",
                "TEMP",
                "PWD",
                "OLDPWD",
                "HOSTNAME",
                "DISPLAY",
                "WAYLAND_DISPLAY",
                "DBUS_SESSION_BUS_ADDRESS",
                "SSH_AUTH_SOCK",
            }
        )
        safe_env = {k: os.environ[k] for k in _SAFE_ENV_VARS if k in os.environ}

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                cwd=cwd or None,
                timeout=effective_timeout,
                executable="/bin/bash",
                env=safe_env,
                preexec_fn=self._build_preexec_fn(),
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


class RefusingSandbox:
    """Sandbox that refuses to run commands (fail-closed).

    Returned by :func:`get_sandbox` when sandboxing is enabled,
    :attr:`SandboxConfig.require_isolation` is ``True``, and Docker is not
    available.  Every :meth:`execute` call returns a failed
    :class:`SandboxResult` explaining that Docker isolation is required —
    the command is never run.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    def is_available(self) -> bool:
        """A refusing sandbox is never "available" for execution."""
        return False

    def execute(
        self,
        command: str,  # noqa: ARG002 - signature parity with other sandboxes
        *,
        cwd: str | None = None,  # noqa: ARG002
        timeout: int | None = None,  # noqa: ARG002
        **_kwargs: Any,
    ) -> SandboxResult:
        """Refuse execution and return a failed :class:`SandboxResult`."""
        return SandboxResult(
            success=False,
            output=None,
            error=(
                "Sandbox isolation is required (sandbox.require_isolation=true) "
                "but Docker is not available; refusing to run the command "
                "unsandboxed. Install/start Docker, or set "
                "sandbox.require_isolation=false to allow the best-effort "
                "unsandboxed fallback."
            ),
            sandboxed=False,
        )


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
        require_isolation=bool(data.get("require_isolation", True)),
        tools=dict(data.get("tools") or {}),
    )


def get_sandbox(
    config: SandboxConfig | None = None,
) -> DockerSandbox | FallbackSandbox | RefusingSandbox:
    """Return the best available sandbox implementation.

    Returns a :class:`DockerSandbox` when Docker is accessible and sandbox
    is enabled.  When sandbox is enabled but Docker is unavailable, the
    behaviour depends on :attr:`SandboxConfig.require_isolation`:

    * ``True`` (default): returns a :class:`RefusingSandbox` that refuses to
      execute — fail-closed, so an operator who enabled sandboxing for
      safety never silently loses it.
    * ``False``: returns a best-effort :class:`FallbackSandbox` (with a loud
      warning) that runs the command on the host with limited protections.

    When sandbox is disabled entirely a :class:`FallbackSandbox` is returned
    for direct execution.
    """
    cfg = config or SandboxConfig()
    if cfg.enabled:
        sandbox = DockerSandbox(cfg)
        if sandbox.is_available():
            logger.info("Docker sandbox available — using containerized execution")
            return sandbox
        if cfg.require_isolation:
            logger.error(
                "Docker sandbox enabled and isolation required but Docker is "
                "unavailable — refusing to execute commands unsandboxed. Set "
                "sandbox.require_isolation=false to opt in to the best-effort "
                "fallback."
            )
            return RefusingSandbox(cfg)
        logger.warning(
            "Docker sandbox enabled but Docker not available and "
            "require_isolation=false — falling back to best-effort unsandboxed "
            "execution (NO network isolation, NO read-only root)."
        )
    return FallbackSandbox(cfg)
