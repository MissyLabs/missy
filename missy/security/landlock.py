"""Linux Landlock LSM filesystem policy enforcement for Missy.

Landlock is a Linux security module available since kernel 5.13 that allows
an unprivileged process to restrict its own filesystem access using the kernel
itself as the enforcement point.  Unlike the Python-level
:class:`~missy.policy.filesystem.FilesystemPolicyEngine`, a Landlock ruleset
cannot be bypassed by code that circumvents Python-layer checks.  Once applied
via :meth:`LandlockPolicy.apply`, the restrictions are permanent for the
process and all of its descendants — they cannot be removed or weakened.

The three Landlock syscalls (``landlock_create_ruleset``,
``landlock_add_rule``, ``landlock_restrict_self``) are invoked directly via
:mod:`ctypes` so that no external C extension is required.

Typical usage::

    from missy.config.settings import load_config
    from missy.security.landlock import apply_landlock_from_config

    config = load_config()
    applied = apply_landlock_from_config(config)
    if applied:
        print("Landlock filesystem restrictions are active")

Direct usage::

    from missy.security.landlock import LandlockPolicy

    policy = (
        LandlockPolicy()
        .add_read_path("/usr")
        .add_read_path("/etc")
        .add_write_path("/tmp")
    )
    policy.apply()

Checking status in diagnostics::

    from missy.security.landlock import landlock_status

    info = landlock_status()
    print(info["available"], info["applied"])
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import logging
import os
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.config.settings import MissyConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Landlock ABI constants (kernel uapi/linux/landlock.h)
# ---------------------------------------------------------------------------

#: Flag for landlock_create_ruleset() to query the ABI version instead of
#: creating a real ruleset.
LANDLOCK_CREATE_RULESET_VERSION: int = 1 << 0

# --- Individual file-access rights (Landlock ABI v1) ---
LANDLOCK_ACCESS_FS_EXECUTE: int = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE: int = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE: int = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR: int = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR: int = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE: int = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR: int = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR: int = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG: int = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK: int = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO: int = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK: int = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM: int = 1 << 12

# --- Composite masks ---

#: All read access rights (files and directories).
LANDLOCK_ACCESS_FS_READ: int = (
    LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
)

#: All write / mutation access rights.
LANDLOCK_ACCESS_FS_WRITE: int = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_SYM
)

# --- Syscall numbers (x86_64 Linux) ---
_SYS_landlock_create_ruleset: int = 444
_SYS_landlock_add_rule: int = 445
_SYS_landlock_restrict_self: int = 446

#: Rule type constant: restrict by path beneath a directory tree.
LANDLOCK_RULE_PATH_BENEATH: int = 1

# --- PR_SET_NO_NEW_PRIVS (required before landlock_restrict_self) ---
_PR_SET_NO_NEW_PRIVS: int = 38

# ---------------------------------------------------------------------------
# ctypes structure layouts
# ---------------------------------------------------------------------------


class _RulesetAttr(ctypes.Structure):
    """``struct landlock_ruleset_attr`` from ``<linux/landlock.h>``.

    Fields:
        handled_access_fs: Bitmask of all FS access rights the ruleset
            handles.  Any right *not* listed here will be allowed
            unconditionally.
    """

    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    """``struct landlock_path_beneath_attr`` from ``<linux/landlock.h>``.

    Fields:
        allowed_access: Bitmask of FS access rights allowed for this path.
        parent_fd: Open file descriptor (``O_PATH``) for the parent directory.
    """

    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

#: True once :func:`apply_landlock_from_config` or :meth:`LandlockPolicy.apply`
#: has successfully applied a ruleset to this process.
_applied_globally: bool = False


# ---------------------------------------------------------------------------
# LandlockPolicy
# ---------------------------------------------------------------------------


class LandlockPolicy:
    """Build and apply a Landlock filesystem ruleset to the current process.

    Rules are accumulated with :meth:`add_read_path`, :meth:`add_write_path`,
    and :meth:`add_execute_path`.  Call :meth:`apply` to activate the policy.

    Once applied, the restrictions are permanent — they cannot be removed or
    weakened for this process or any child process.

    Example::

        policy = (
            LandlockPolicy()
            .add_read_path("/usr")
            .add_read_path("/etc")
            .add_write_path("/tmp")
            .add_write_path(os.path.expanduser("~/.missy"))
        )
        policy.apply()
    """

    def __init__(self) -> None:
        self._applied: bool = False
        self._rules: list[dict[str, str]] = []
        self._libc: ctypes.CDLL | None = None

    # ------------------------------------------------------------------
    # Class-level availability check
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if Landlock is supported on this kernel.

        The check works by calling ``landlock_create_ruleset`` with the
        ``LANDLOCK_CREATE_RULESET_VERSION`` flag, which queries the ABI
        version without creating any persistent state.  If the syscall
        returns a non-negative integer the kernel supports Landlock.

        Returns:
            ``True`` on Linux kernel >= 5.13 with Landlock enabled,
            ``False`` on older kernels, non-Linux platforms, or when the
            ``libc`` shared library cannot be located.
        """
        if not sys.platform.startswith("linux"):
            return False
        try:
            libc = cls._load_libc()
            result = libc.syscall(
                _SYS_landlock_create_ruleset,
                None,
                ctypes.c_size_t(0),
                ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
            )
            # A positive result is the ABI version number (>= 1 means supported).
            return int(result) >= 1
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Rule builders (fluent API)
    # ------------------------------------------------------------------

    def add_read_path(self, path: str) -> LandlockPolicy:
        """Allow read access to *path* and everything beneath it.

        Args:
            path: Absolute path to a file or directory.

        Returns:
            ``self`` for method chaining.
        """
        self._rules.append({"path": path, "access": "read"})
        return self

    def add_write_path(self, path: str) -> LandlockPolicy:
        """Allow read and write access to *path* and everything beneath it.

        This grants both :data:`LANDLOCK_ACCESS_FS_READ` and
        :data:`LANDLOCK_ACCESS_FS_WRITE`.

        Args:
            path: Absolute path to a file or directory.

        Returns:
            ``self`` for method chaining.
        """
        self._rules.append({"path": path, "access": "read_write"})
        return self

    def add_execute_path(self, path: str) -> LandlockPolicy:
        """Allow execute access to *path* and everything beneath it.

        Args:
            path: Absolute path to a file or directory.

        Returns:
            ``self`` for method chaining.
        """
        self._rules.append({"path": path, "access": "execute"})
        return self

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self) -> bool:
        """Apply the accumulated ruleset to this process.

        Steps:

        1. Check Landlock availability — return ``False`` if unsupported.
        2. Create a Landlock ruleset via ``landlock_create_ruleset``.
        3. For each rule, open the target path with ``O_PATH`` and call
           ``landlock_add_rule``.
        4. Set ``PR_SET_NO_NEW_PRIVS`` (required by the kernel before
           ``landlock_restrict_self``).
        5. Call ``landlock_restrict_self`` to activate the policy.
        6. Close the ruleset file descriptor.

        Returns:
            ``True`` if the ruleset was successfully applied.
            ``False`` if Landlock is not available on this kernel.

        Raises:
            RuntimeError: If :meth:`apply` has already been called on this
                instance, or if a syscall fails unexpectedly.
        """
        global _applied_globally

        if self._applied:
            raise RuntimeError(
                "LandlockPolicy.apply() has already been called on this instance. "
                "A Landlock ruleset cannot be applied twice."
            )

        if not self.is_available():
            logger.info("Landlock is not available on this kernel — skipping")
            return False

        libc = self._get_libc()
        ruleset_fd: int = -1
        path_fds: list[int] = []

        try:
            ruleset_fd = self._create_ruleset(libc)
            for rule in self._rules:
                fd = self._open_path(rule["path"])
                if fd < 0:
                    # Path does not exist; skip silently (already logged).
                    continue
                path_fds.append(fd)
                access_mask = self._access_mask_for(rule["access"])
                self._add_rule(libc, ruleset_fd, fd, access_mask)

            self._set_no_new_privs(libc)
            self._restrict_self(libc, ruleset_fd)

        except OSError as exc:
            raise RuntimeError(
                f"Failed to apply Landlock ruleset: {exc}"
            ) from exc
        finally:
            for fd in path_fds:
                with contextlib.suppress(OSError):
                    os.close(fd)
            if ruleset_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(ruleset_fd)

        self._applied = True
        _applied_globally = True
        rule_count = len(self._rules)
        logger.info(
            "Landlock filesystem policy applied (%d rule%s)",
            rule_count,
            "" if rule_count == 1 else "s",
        )
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def applied(self) -> bool:
        """``True`` if :meth:`apply` has been called successfully."""
        return self._applied

    @property
    def rules(self) -> list[dict[str, str]]:
        """A snapshot of the accumulated rules.

        Returns:
            A copy of the internal rule list so callers cannot mutate it.
        """
        return list(self._rules)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_libc() -> ctypes.CDLL:
        """Load and return the C standard library.

        Raises:
            OSError: If the library cannot be found or loaded.
        """
        name = ctypes.util.find_library("c")
        if name is None:
            raise OSError("Cannot locate libc shared library")
        lib = ctypes.CDLL(name, use_errno=True)
        return lib

    def _get_libc(self) -> ctypes.CDLL:
        """Return a cached ``libc`` handle, loading it on first call."""
        if self._libc is None:
            self._libc = self._load_libc()
        return self._libc

    def _syscall(self, libc: ctypes.CDLL, number: int, *args: object) -> int:
        """Invoke a raw syscall and raise :exc:`OSError` on failure.

        Args:
            libc: Handle to the C standard library.
            number: Linux syscall number.
            *args: Positional arguments forwarded to ``libc.syscall``.

        Returns:
            The non-negative return value of the syscall.

        Raises:
            OSError: If the syscall returns a negative value.
        """
        result: int = libc.syscall(number, *args)
        if result < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return result

    def _create_ruleset(self, libc: ctypes.CDLL) -> int:
        """Call ``landlock_create_ruleset`` and return the ruleset fd.

        The ruleset is created to handle all known FS access rights so that
        Landlock will deny any right not explicitly granted by an
        :meth:`add_rule` call.

        Args:
            libc: Handle to the C standard library.

        Returns:
            A non-negative file descriptor for the new ruleset.

        Raises:
            OSError: If the syscall fails.
        """
        handled = (
            LANDLOCK_ACCESS_FS_READ
            | LANDLOCK_ACCESS_FS_WRITE
            | LANDLOCK_ACCESS_FS_EXECUTE
        )
        attr = _RulesetAttr(handled_access_fs=handled)
        return self._syscall(
            libc,
            _SYS_landlock_create_ruleset,
            ctypes.byref(attr),
            ctypes.c_size_t(ctypes.sizeof(attr)),
            ctypes.c_uint32(0),
        )

    @staticmethod
    def _open_path(path: str) -> int:
        """Open *path* with ``O_PATH`` for use in a Landlock rule.

        ``O_PATH`` does not require read or write permission on the target;
        it merely obtains a file descriptor that can be used in
        ``landlock_add_rule``.

        Args:
            path: Absolute filesystem path to open.

        Returns:
            A non-negative file descriptor, or ``-1`` if the path does not
            exist or cannot be opened.
        """
        O_PATH = 0o10000000  # noqa: N806 — Linux O_PATH flag (0x200000)
        try:
            fd = os.open(path, O_PATH)
            return fd
        except OSError as exc:
            logger.debug("Skipping Landlock rule for %r: %s", path, exc)
            return -1

    def _add_rule(
        self,
        libc: ctypes.CDLL,
        ruleset_fd: int,
        path_fd: int,
        access_mask: int,
    ) -> None:
        """Call ``landlock_add_rule`` for a single path.

        Args:
            libc: Handle to the C standard library.
            ruleset_fd: File descriptor of the ruleset to add the rule to.
            path_fd: ``O_PATH`` file descriptor for the allowed path.
            access_mask: Bitmask of allowed access rights.

        Raises:
            OSError: If the syscall fails.
        """
        attr = _PathBeneathAttr(
            allowed_access=access_mask,
            parent_fd=path_fd,
        )
        self._syscall(
            libc,
            _SYS_landlock_add_rule,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(attr),
            ctypes.c_uint32(0),
        )

    @staticmethod
    def _set_no_new_privs(libc: ctypes.CDLL) -> None:
        """Set the ``PR_SET_NO_NEW_PRIVS`` prctl flag.

        This is a kernel prerequisite for ``landlock_restrict_self``.
        The flag prevents the process from gaining new privileges via
        setuid/setgid executables.

        Args:
            libc: Handle to the C standard library.

        Raises:
            OSError: If the prctl call fails.
        """
        result: int = libc.prctl(
            ctypes.c_int(_PR_SET_NO_NEW_PRIVS),
            ctypes.c_ulong(1),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
            ctypes.c_ulong(0),
        )
        if result < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))

    def _restrict_self(self, libc: ctypes.CDLL, ruleset_fd: int) -> None:
        """Call ``landlock_restrict_self`` to activate the ruleset.

        After this call returns successfully the restrictions are permanent
        for this process and all future children.

        Args:
            libc: Handle to the C standard library.
            ruleset_fd: File descriptor of the finalised ruleset.

        Raises:
            OSError: If the syscall fails.
        """
        self._syscall(
            libc,
            _SYS_landlock_restrict_self,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(0),
        )

    @staticmethod
    def _access_mask_for(access_type: str) -> int:
        """Translate a rule access type string to a Landlock bitmask.

        Args:
            access_type: One of ``"read"``, ``"read_write"``, or
                ``"execute"``.

        Returns:
            The corresponding Landlock access bitmask.

        Raises:
            ValueError: For unrecognised *access_type* strings.
        """
        if access_type == "read":
            return LANDLOCK_ACCESS_FS_READ
        if access_type == "read_write":
            return LANDLOCK_ACCESS_FS_READ | LANDLOCK_ACCESS_FS_WRITE
        if access_type == "execute":
            return (
                LANDLOCK_ACCESS_FS_READ
                | LANDLOCK_ACCESS_FS_EXECUTE
            )
        raise ValueError(f"Unknown Landlock access type: {access_type!r}")


# ---------------------------------------------------------------------------
# Config-driven helper
# ---------------------------------------------------------------------------


def apply_landlock_from_config(config: MissyConfig) -> bool:  # type: ignore[name-defined]
    """Apply Landlock restrictions derived from Missy's filesystem policy.

    The following paths are always included regardless of config:

    * Read-only: ``/usr``, ``/lib``, ``/lib64``, ``/etc``, ``/bin``,
      ``/sbin``, ``/proc/self`` (needed by Python internals), and all
      entries on :data:`sys.path`.
    * Read-write: ``~/.missy`` (Missy's own data directory) and ``/tmp``.

    User-configured paths from ``config.filesystem.allowed_read_paths`` and
    ``config.filesystem.allowed_write_paths`` are appended on top.

    Args:
        config: The fully-populated :class:`~missy.config.settings.MissyConfig`
            instance.

    Returns:
        ``True`` if a Landlock ruleset was successfully applied, ``False``
        if Landlock is not available on this kernel.
    """
    if not LandlockPolicy.is_available():
        logger.info("Landlock not available on this kernel — skipping")
        return False

    policy = LandlockPolicy()

    # System read-only paths needed by Python and Missy itself.
    for system_path in ("/usr", "/lib", "/lib64", "/etc", "/bin", "/sbin"):
        if os.path.exists(system_path):
            policy.add_read_path(system_path)

    # /proc/self is accessed by Python's import machinery and ctypes.
    if os.path.exists("/proc/self"):
        policy.add_read_path("/proc/self")

    # All entries on sys.path (site-packages, stdlib, source tree, etc.).
    for sys_path_entry in sys.path:
        if sys_path_entry and os.path.exists(sys_path_entry):
            policy.add_read_path(sys_path_entry)

    # Missy data directory — always read+write.
    missy_dir = os.path.expanduser("~/.missy")
    policy.add_write_path(missy_dir)

    # /tmp — always read+write for temporary file operations.
    if os.path.exists("/tmp"):
        policy.add_write_path("/tmp")

    # User-configured read paths.
    for read_path in config.filesystem.allowed_read_paths:
        expanded = os.path.expanduser(read_path)
        if os.path.exists(expanded):
            policy.add_read_path(expanded)
        else:
            logger.debug("Landlock: skipping non-existent read path %r", expanded)

    # User-configured write paths.
    for write_path in config.filesystem.allowed_write_paths:
        expanded = os.path.expanduser(write_path)
        if os.path.exists(expanded):
            policy.add_write_path(expanded)
        else:
            logger.debug("Landlock: skipping non-existent write path %r", expanded)

    return policy.apply()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def landlock_status() -> dict[str, object]:
    """Return a diagnostic snapshot of Landlock availability and state.

    This function is safe to call at any time; it never modifies process
    state.

    Returns:
        A dictionary with the following keys:

        ``available``
            ``True`` if the kernel supports Landlock.
        ``applied``
            ``True`` if a ruleset has been applied to this process via
            :func:`apply_landlock_from_config` or
            :meth:`LandlockPolicy.apply`.
        ``kernel_version``
            The kernel release string from :func:`platform.release`.
        ``platform``
            The OS platform identifier from :data:`sys.platform`.
    """
    return {
        "available": LandlockPolicy.is_available(),
        "applied": _applied_globally,
        "kernel_version": platform.release(),
        "platform": sys.platform,
    }
