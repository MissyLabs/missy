"""Central policy engine for the Missy framework.

:class:`PolicyEngine` composes the three domain-specific engines
(network, filesystem, shell) behind a single facade.  A module-level
singleton is managed by :func:`init_policy_engine` and retrieved by
:func:`get_policy_engine`.

Typical usage::

    from missy.config.settings import load_config
    from missy.policy.engine import init_policy_engine, get_policy_engine

    config = load_config("missy.yaml")
    init_policy_engine(config)

    engine = get_policy_engine()
    engine.check_network("api.github.com", session_id="s1", task_id="t1")
"""

from __future__ import annotations

import threading
from pathlib import Path

from missy.config.settings import MissyConfig

from .filesystem import FilesystemPolicyEngine
from .network import NetworkPolicyEngine
from .rest_policy import RestPolicy
from .shell import ShellPolicyEngine


class PolicyEngine:
    """Facade that composes network, filesystem, shell, and REST policy engines.

    Args:
        config: The fully-populated runtime configuration.
    """

    def __init__(self, config: MissyConfig) -> None:
        self.config: MissyConfig = config
        self.network: NetworkPolicyEngine = NetworkPolicyEngine(config.network)
        self.filesystem: FilesystemPolicyEngine = FilesystemPolicyEngine(config.filesystem)
        self.shell: ShellPolicyEngine = ShellPolicyEngine(config.shell)
        self.rest_policy: RestPolicy = RestPolicy.from_config(
            getattr(config.network, "rest_policies", []) or []
        )

    # ------------------------------------------------------------------
    # Delegate methods
    # ------------------------------------------------------------------

    def check_network(
        self,
        host: str,
        session_id: str = "",
        task_id: str = "",
        category: str = "",
    ) -> bool:
        """Evaluate a network access request.

        Delegates to :meth:`NetworkPolicyEngine.check_host`.

        Args:
            host: Hostname or IP address (without port or scheme).
            session_id: Optional calling session identifier.
            task_id: Optional calling task identifier.
            category: Request category (``"provider"``, ``"tool"``,
                ``"discord"``).

        Returns:
            ``True`` when the host is allowed.

        Raises:
            PolicyViolationError: When the host is denied.
            ValueError: When *host* is empty.
        """
        return self.network.check_host(
            host,
            session_id=session_id,
            task_id=task_id,
            category=category,
        )

    def check_network_resolved(
        self,
        host: str,
        session_id: str = "",
        task_id: str = "",
        category: str = "",
    ) -> tuple[bool, str | None]:
        """Like :meth:`check_network`, but also returns the validated IP.

        SR-1.9b: delegates to :meth:`NetworkPolicyEngine.check_host_resolved`
        so callers that actually open the connection
        (:class:`~missy.gateway.client.PolicyHTTPClient`) can pin it to
        the exact address this check validated, closing the
        check-time/connect-time DNS-rebinding TOCTOU window.
        """
        return self.network.check_host_resolved(
            host,
            session_id=session_id,
            task_id=task_id,
            category=category,
        )

    def check_write(
        self,
        path: str | Path,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Evaluate a filesystem write request.

        Delegates to :meth:`FilesystemPolicyEngine.check_write`.

        Args:
            path: The filesystem path the caller intends to write.
            session_id: Optional calling session identifier.
            task_id: Optional calling task identifier.

        Returns:
            ``True`` when the path is within an allowed write directory.

        Raises:
            PolicyViolationError: When the path is denied.
        """
        return self.filesystem.check_write(path, session_id=session_id, task_id=task_id)

    def check_read(
        self,
        path: str | Path,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Evaluate a filesystem read request.

        Delegates to :meth:`FilesystemPolicyEngine.check_read`.

        Args:
            path: The filesystem path the caller intends to read.
            session_id: Optional calling session identifier.
            task_id: Optional calling task identifier.

        Returns:
            ``True`` when the path is within an allowed read directory.

        Raises:
            PolicyViolationError: When the path is denied.
        """
        return self.filesystem.check_read(path, session_id=session_id, task_id=task_id)

    def check_shell(
        self,
        command: str,
        session_id: str = "",
        task_id: str = "",
    ) -> bool:
        """Evaluate a shell command execution request.

        Delegates to :meth:`ShellPolicyEngine.check_command` for program-name
        allow-listing, then — SR-1.7 — routes every redirection target in
        *command* through the filesystem policy engine too. An allowed
        program name says nothing about what files a shell redirection lets
        it touch: ``echo x > /etc/cron.d/pwn`` with only ``echo`` allowlisted
        would otherwise write an arbitrary host file with no filesystem
        check at all, since ``ShellPolicyEngine`` only ever inspected
        program names.

        Args:
            command: The shell command string to evaluate.
            session_id: Optional calling session identifier.
            task_id: Optional calling task identifier.

        Returns:
            ``True`` when the command — and every redirection target it
            contains — is allowed.

        Raises:
            PolicyViolationError: When the command is denied, or any
                redirection target falls outside the filesystem policy's
                allowed read/write paths.
        """
        self.shell.check_command(command, session_id=session_id, task_id=task_id)

        write_targets, read_targets = self.shell.extract_redirect_targets(command)
        for target in write_targets:
            self.filesystem.check_write(target, session_id=session_id, task_id=task_id)
        for target in read_targets:
            self.filesystem.check_read(target, session_id=session_id, task_id=task_id)

        return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_engine: PolicyEngine | None = None
_lock: threading.Lock = threading.Lock()


def init_policy_engine(config: MissyConfig) -> PolicyEngine:
    """Construct and install the process-level :class:`PolicyEngine`.

    Calling this function a second time replaces the existing engine with a
    new one built from *config*.  The replacement is performed atomically
    under a lock so that concurrent calls from multiple threads are safe.

    Args:
        config: The runtime configuration to build the engine from.

    Returns:
        The newly installed :class:`PolicyEngine` instance.
    """
    global _engine
    engine = PolicyEngine(config)
    with _lock:
        _engine = engine
    return engine


def get_policy_engine() -> PolicyEngine:
    """Return the process-level :class:`PolicyEngine`.

    Args:
        None

    Returns:
        The currently installed :class:`PolicyEngine`.

    Raises:
        RuntimeError: When :func:`init_policy_engine` has not yet been called.
    """
    with _lock:
        engine = _engine
    if engine is None:
        raise RuntimeError(
            "PolicyEngine has not been initialised. "
            "Call missy.policy.engine.init_policy_engine(config) first."
        )
    return engine
