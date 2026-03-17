"""Container-per-session sandbox for isolated tool execution.

Provides a :class:`ContainerSandbox` that manages a long-lived Docker
container for the duration of a session, executing commands inside it.

Unlike :mod:`missy.security.sandbox` which creates a new container per
command (``docker run --rm``), this module keeps a single container alive
across multiple :meth:`execute` calls, supporting ``copy_in`` / ``copy_out``
for file transfers.

When Docker is not available all methods degrade gracefully — ``start()``
returns ``None``, ``execute()`` returns an error tuple, and ``stop()``
is a no-op.

Configuration in ``~/.missy/config.yaml``::

    container:
      enabled: true
      image: "python:3.12-slim"
      memory_limit: "256m"
      cpu_quota: 0.5
      network_mode: "none"

Example::

    from missy.security.container import ContainerSandbox

    with ContainerSandbox() as sb:
        stdout, rc = sb.execute("echo hello")
        assert rc == 0
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for container-per-session sandbox.

    Attributes:
        enabled: Master switch for container sandboxing.
        image: Docker image to use for session containers.
        memory_limit: Docker memory limit (e.g. ``"256m"``).
        cpu_quota: CPU quota as a fraction of one CPU (e.g. ``0.5``).
        network_mode: Docker network mode (``"none"`` disables networking).
    """

    enabled: bool = False
    image: str = "python:3.12-slim"
    memory_limit: str = "256m"
    cpu_quota: float = 0.5
    network_mode: str = "none"


def parse_container_config(data: dict) -> ContainerConfig:
    """Parse a ``container:`` YAML section into :class:`ContainerConfig`."""
    if not isinstance(data, dict):
        return ContainerConfig()
    return ContainerConfig(
        enabled=bool(data.get("enabled", False)),
        image=str(data.get("image", "python:3.12-slim")),
        memory_limit=str(data.get("memory_limit", "256m")),
        cpu_quota=float(data.get("cpu_quota", 0.5)),
        network_mode=str(data.get("network_mode", "none")),
    )


class ContainerSandbox:
    """Manage a long-lived Docker container for session-scoped isolation.

    The container is created on :meth:`start` and destroyed on :meth:`stop`.
    Commands are executed inside it via ``docker exec``.  The workspace
    directory is bind-mounted read-only by default.

    Parameters:
        image: Docker image to use.
        workspace: Host path to mount into the container.
        memory_limit: Memory limit (Docker ``--memory`` flag).
        cpu_quota: CPU quota as a fraction (Docker ``--cpus`` flag).
        network_mode: Docker ``--network`` flag value.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        workspace: str = "~/workspace",
        memory_limit: str = "256m",
        cpu_quota: float = 0.5,
        network_mode: str = "none",
    ) -> None:
        self.image = image
        self.workspace = str(Path(workspace).expanduser())
        self.memory_limit = memory_limit
        self.cpu_quota = cpu_quota
        self.network_mode = network_mode
        self._container_id: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check whether Docker is accessible on this host."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @property
    def container_id(self) -> str | None:
        """Return the running container ID, or ``None`` if not started."""
        return self._container_id

    def start(self) -> str | None:
        """Create and start a container, returning its ID.

        Returns:
            The container ID string, or ``None`` if Docker is unavailable.
        """
        if not self.is_available():
            logger.debug("Docker not available — container sandbox disabled")
            return None

        cmd = [
            "docker",
            "run",
            "-d",
            "--memory",
            self.memory_limit,
            "--cpus",
            str(self.cpu_quota),
            f"--network={self.network_mode}",
            "-v",
            f"{self.workspace}:/workspace:ro",
            "--workdir",
            "/workspace",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            self.image,
            "sleep",
            "infinity",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="replace").strip()
                logger.error("Failed to start container: %s", err)
                return None
            self._container_id = result.stdout.decode("utf-8", errors="replace").strip()
            logger.info("Container started: %s", self._container_id[:12])
            return self._container_id
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("Failed to start container: %s", exc)
            return None

    def execute(self, command: str, timeout: int = 30) -> tuple[str, int]:
        """Run a command inside the container.

        Args:
            command: Shell command to execute.
            timeout: Execution timeout in seconds.

        Returns:
            A ``(output, exit_code)`` tuple.  If the container is not
            running, returns an error message with exit code ``-1``.
        """
        if not self._container_id:
            return ("Container not started", -1)

        cmd = [
            "docker",
            "exec",
            self._container_id,
            "/bin/sh",
            "-c",
            command,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout)
            combined = result.stdout + result.stderr
            output = combined.decode("utf-8", errors="replace")
            return (output, result.returncode)
        except subprocess.TimeoutExpired:
            return (f"Command timed out after {timeout}s", -1)
        except OSError as exc:
            return (str(exc), -1)

    def copy_in(self, local_path: str, container_path: str) -> None:
        """Copy a file from the host into the container.

        Args:
            local_path: Path on the host.
            container_path: Destination path inside the container.
        """
        if not self._container_id:
            logger.warning("copy_in called but container not started")
            return

        cmd = ["docker", "cp", local_path, f"{self._container_id}:{container_path}"]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.error("copy_in failed: %s", exc)

    def copy_out(self, container_path: str, local_path: str) -> None:
        """Copy a file from the container to the host.

        Args:
            container_path: Path inside the container.
            local_path: Destination path on the host.
        """
        if not self._container_id:
            logger.warning("copy_out called but container not started")
            return

        cmd = ["docker", "cp", f"{self._container_id}:{container_path}", local_path]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.error("copy_out failed: %s", exc)

    def stop(self) -> None:
        """Stop and remove the container."""
        if not self._container_id:
            return

        cid = self._container_id
        self._container_id = None

        try:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                capture_output=True,
                timeout=15,
            )
            logger.info("Container removed: %s", cid[:12])
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.error("Failed to remove container %s: %s", cid[:12], exc)

    def __enter__(self) -> ContainerSandbox:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.stop()
