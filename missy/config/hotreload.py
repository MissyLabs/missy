"""Config hot-reload: watch config.yaml for changes and re-apply policy."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigWatcher:
    """Watches a config file and triggers reload when it changes.

    Args:
        config_path: Path to the YAML config file.
        reload_fn: Callable invoked with the new config on change.
        debounce_seconds: Wait this long after last change before reloading (default 2).
        poll_interval: File stat check interval in seconds (default 1).
    """

    def __init__(
        self,
        config_path: str,
        reload_fn: Callable,
        debounce_seconds: float = 2.0,
        poll_interval: float = 1.0,
    ):
        self._path = Path(config_path).expanduser()
        self._reload_fn = reload_fn
        self._debounce = debounce_seconds
        self._poll = poll_interval
        self._last_mtime: float = 0.0
        self._last_change_time: float = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_config = None

    def start(self) -> None:
        """Start the background file watcher."""
        try:
            from missy.config.settings import load_config

            self._active_config = load_config(str(self._path))
        except Exception:
            logger.warning("ConfigWatcher: could not snapshot initial policy", exc_info=True)
        try:
            self._last_mtime = self._path.stat().st_mtime
        except OSError:
            self._last_mtime = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True, name="missy-hotreload")
        self._thread.start()
        logger.info("ConfigWatcher: watching %s", self._path)

    def stop(self) -> None:
        """Stop the background file watcher."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _watch(self) -> None:
        pending_reload = False
        while not self._stop.wait(self._poll):
            try:
                mtime = self._path.stat().st_mtime
            except OSError:
                continue

            if mtime != self._last_mtime:
                self._last_mtime = mtime
                self._last_change_time = time.monotonic()
                pending_reload = True
                logger.debug("ConfigWatcher: change detected in %s", self._path)

            if pending_reload and (time.monotonic() - self._last_change_time) >= self._debounce:
                pending_reload = False
                self._do_reload()

    def _check_file_safety(self) -> bool:
        """Verify config file ownership and permissions before reload.

        Returns True if the file is safe to load, False otherwise.
        Rejects symlinks, files not owned by the current user, and
        files that are group- or world-writable.
        """
        try:
            if self._path.is_symlink():
                logger.warning("ConfigWatcher: %s is a symlink; refusing to reload", self._path)
                return False
            st = self._path.stat()
            if st.st_uid != os.getuid():
                logger.warning(
                    "ConfigWatcher: %s is owned by uid %d, expected %d; refusing to reload",
                    self._path,
                    st.st_uid,
                    os.getuid(),
                )
                return False
            if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                logger.warning(
                    "ConfigWatcher: %s is group- or world-writable (mode %o); refusing to reload",
                    self._path,
                    st.st_mode,
                )
                return False
        except OSError as exc:
            logger.warning("ConfigWatcher: cannot stat %s: %s", self._path, exc)
            return False
        return True

    def _do_reload(self) -> None:
        logger.info("ConfigWatcher: reloading %s", self._path)
        if not self._check_file_safety():
            return
        try:
            from missy.config.settings import load_config

            new_config = load_config(str(self._path))
            if (
                self._active_config is not None
                and self._active_config is not new_config
                and _is_security_widening(self._active_config, new_config)
            ):
                digest = candidate_config_digest(new_config)
                if not self._consume_widening_approval(digest):
                    logger.error(
                        "ConfigWatcher: rejected unapproved security-policy widening (%s)",
                        digest,
                    )
                    _emit_reload_denial(digest)
                    return
            self._reload_fn(new_config)
            self._active_config = new_config
            logger.info("ConfigWatcher: reload complete")
        except Exception as exc:
            logger.error("ConfigWatcher: reload failed: %s", exc)

    def _consume_widening_approval(self, digest: str) -> bool:
        """Consume a one-time approval bound to an exact candidate digest."""
        approval = self._path.parent / f"{self._path.name}.reload-approval"
        fd = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(approval, flags)
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return False
            if st.st_uid != os.getuid() or st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                return False
            content = os.read(fd, 4096).decode("utf-8").strip()
            if content != digest:
                return False
            current = approval.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (st.st_dev, st.st_ino):
                return False
            approval.unlink()
            return True
        except (OSError, UnicodeDecodeError):
            return False
        finally:
            if fd >= 0:
                os.close(fd)


def candidate_config_digest(config) -> str:
    """Return the canonical digest used for one-time widening approvals."""
    payload = asdict(config) if is_dataclass(config) else config
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _paths_widened(old_paths: list[str], new_paths: list[str]) -> bool:
    old = [Path(path).expanduser().resolve() for path in old_paths]
    for raw in new_paths:
        candidate = Path(raw).expanduser().resolve()
        if not any(candidate == prior or candidate.is_relative_to(prior) for prior in old):
            return True
    return False


def _is_security_widening(old, new) -> bool:
    """Conservatively identify access expansions while allowing narrowing."""
    if (not old.shell.enabled and new.shell.enabled) or (
        not old.shell.unrestricted and new.shell.unrestricted
    ):
        return True
    if not set(new.shell.allowed_commands).issubset(old.shell.allowed_commands):
        return True
    if not set(new.shell.allowed_env_vars).issubset(old.shell.allowed_env_vars):
        return True
    if _paths_widened(old.filesystem.allowed_read_paths, new.filesystem.allowed_read_paths):
        return True
    if _paths_widened(old.filesystem.allowed_write_paths, new.filesystem.allowed_write_paths):
        return True
    # Removing a protected path (or replacing it with a narrower child) is a
    # widening even when the ordinary allowlists are unchanged.
    if _paths_widened(new.filesystem.protected_write_paths, old.filesystem.protected_write_paths):
        return True
    if old.network.default_deny and not new.network.default_deny:
        return True
    for field in (
        "allowed_cidrs",
        "allowed_domains",
        "allowed_hosts",
        "provider_allowed_hosts",
        "tool_allowed_hosts",
        "discord_allowed_hosts",
        "presets",
    ):
        if not set(getattr(new.network, field)).issubset(getattr(old.network, field)):
            return True
    if new.network.rest_policies != old.network.rest_policies:
        return True
    if not old.plugins.enabled and new.plugins.enabled:
        return True
    if not set(new.plugins.allowed_plugins).issubset(old.plugins.allowed_plugins):
        return True
    if not set(new.providers).issubset(old.providers):
        return True
    if any(new.providers[name] != old.providers[name] for name in new.providers):
        return True
    if old.landlock_enabled and not new.landlock_enabled:
        return True

    old_sandbox = old.sandbox
    new_sandbox = new.sandbox
    if old_sandbox is not None and new_sandbox is None:
        if old_sandbox.enabled:
            return True
    elif old_sandbox is not None and new_sandbox is not None:
        if old_sandbox.enabled and not new_sandbox.enabled:
            return True
        if old_sandbox.network_disabled and not new_sandbox.network_disabled:
            return True
        if old_sandbox.read_only_root and not new_sandbox.read_only_root:
            return True
        if old_sandbox.require_isolation and not new_sandbox.require_isolation:
            return True
        # Bind-mount strings contain host/container/mode components and are
        # not ordinary paths. Any newly introduced or changed mapping is a
        # potential host-filesystem expansion; removals are narrowing.
        if not set(new_sandbox.allowed_bind_mounts).issubset(old_sandbox.allowed_bind_mounts):
            return True
        if new_sandbox.tools != old_sandbox.tools:
            return True

    profile_rank = {"minimal": 0, "coding": 2, "messaging": 2, "full": 3}
    old_tools = old.tools
    new_tools = new.tools
    if profile_rank.get(new_tools.profile, 3) > profile_rank.get(old_tools.profile, 3):
        return True
    if (
        profile_rank.get(new_tools.profile, 3) == profile_rank.get(old_tools.profile, 3)
        and new_tools.profile != old_tools.profile
    ):
        return True
    if not set(new_tools.allow).issubset(old_tools.allow):
        return True
    if not set(new_tools.also_allow).issubset(old_tools.also_allow):
        return True
    if not set(old_tools.deny).issubset(new_tools.deny):
        return True
    if not set(old_tools.disabled_tools).issubset(new_tools.disabled_tools):
        return True
    if (
        new_tools.by_provider != old_tools.by_provider
        or new_tools.by_model != old_tools.by_model
        or new_tools.groups != old_tools.groups
    ):
        return True
    # Per-agent policy changes are uncommon and difficult to order safely;
    # require an exact approval unless the mapping is unchanged.
    return new.agents != old.agents


def _emit_reload_denial(digest: str) -> None:
    try:
        from missy.core.events import AuditEvent, event_bus

        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type="config.reload_widening",
                category="security",
                result="deny",
                detail={"candidate_digest": digest},
            )
        )
    except Exception:
        logger.debug("ConfigWatcher: failed to audit widening denial", exc_info=True)


def _apply_config(new_config) -> None:
    """Re-initialise subsystems with updated config."""
    from missy.observability.audit_logger import init_audit_logger
    from missy.observability.otel import init_otel
    from missy.policy.engine import PolicyEngine, init_policy_engine
    from missy.providers.registry import ProviderRegistry, init_registry

    # Construct both new subsystem instances before installing either one
    # globally. init_policy_engine()/init_registry() each construct their
    # replacement before atomically swapping it in, but _apply_config()
    # previously called them sequentially with no such guarantee across
    # the pair: if init_policy_engine() succeeded but init_registry()
    # then raised (e.g. a config that passes load_config()'s own
    # validation but still fails ProviderRegistry.from_config(), such as
    # a malformed provider block), the process ended up with a policy
    # engine on the NEW config and a provider registry still on the OLD
    # config -- a genuinely inconsistent runtime state masked by a
    # generic "reload failed" log line that reads as "nothing changed".
    # Building both here first (discarded; PolicyEngine's __init__ and
    # ProviderRegistry.from_config() are pure config-driven construction
    # with no side effect that isn't idempotent on a second call with the
    # same config) surfaces either construction failure before either
    # singleton is touched.
    PolicyEngine(new_config)
    ProviderRegistry.from_config(new_config)

    # Built-ins are registered only at startup. A stale Foundry instance must
    # never retain its project/endpoint/credential authority after a reload,
    # even when a caller kept a direct reference outside the tool registry.
    # Do not change the registry or the behavior of unrelated tools here.
    from missy.tools.builtin.repoeval_tools import (
        RepoevalFoundryMutateTool,
        RepoevalFoundryReadTool,
    )
    from missy.tools.registry import get_tool_registry

    try:
        registry = get_tool_registry()
    except RuntimeError:  # No registry during early initialization.
        registry = None
    if registry is not None:
        for name, cls in (
            ("repoeval_foundry_read", RepoevalFoundryReadTool),
            ("repoeval_foundry_mutate", RepoevalFoundryMutateTool),
        ):
            foundry = registry.get(name)
            if isinstance(foundry, cls) and not foundry.matches_config(new_config.repoeval_foundry):
                foundry.revoke()

    try:
        from missy.providers.registry import get_registry as _get_registry

        previous_config_default = _get_registry()._config_default_provider
    except Exception:
        previous_config_default = None

    init_policy_engine(new_config)
    init_registry(new_config)
    logger.info("ConfigWatcher: policy engine and provider registry updated")

    # Provider-preference hierarchy: init_registry() just installed a
    # brand-new ProviderRegistry whose own is_default/set_default
    # bookkeeping starts back at None regardless of what it was before
    # this reload -- seed it from the persisted default_provider so an
    # operator-driven config.yaml edit (or any other reload) doesn't
    # silently blank out the Web TUI's "default" indicator until someone
    # manually re-runs `missy providers switch`.
    default_provider = str(getattr(new_config, "default_provider", "") or "").strip()
    # DATA-05: init_registry() carried the live default over; only re-seed it
    # when the operator actually changed default_provider in config.
    if default_provider and default_provider != (previous_config_default or "").strip():
        try:
            from missy.providers.registry import get_registry

            get_registry().set_default(default_provider)
        except Exception as exc:
            logger.warning(
                "ConfigWatcher: could not seed registry default provider %r: %s",
                default_provider,
                exc,
            )

    # SR-4.6/observability: init_otel() was only ever called once, at
    # process bootstrap (missy/cli/main.py's _load_subsystems()) -- toggling
    # observability.otel_enabled (or changing otel_endpoint/otel_protocol)
    # on a running `missy gateway start` daemon had no effect whatsoever,
    # despite ConfigWatcher/_apply_config existing specifically to make
    # config changes take effect without a restart. init_otel() itself
    # unwinds any previously active exporter's publish() wrapper before
    # installing a new one, so this is safe to call on every reload.
    try:
        init_otel(new_config)
        logger.info("ConfigWatcher: OpenTelemetry exporter updated")
    except Exception as exc:
        logger.warning("ConfigWatcher: OpenTelemetry re-init failed: %s", exc)

    # init_audit_logger() was only ever called once, at process bootstrap
    # (_load_subsystems()) -- editing audit_log_path on a running `missy
    # gateway start` daemon (e.g. moving to a different volume, or because
    # the old location became unwritable/full) had no effect whatsoever:
    # every subsequent event kept being written to the stale path forever,
    # with no error surfaced anywhere. init_audit_logger() reuses and
    # reconfigures the same, already-subscribed AuditLogger instance in
    # place rather than constructing a new one (see AuditLogger.reconfigure()'s
    # docstring for why a fresh instance would silently fail to actually
    # replace it), so this is safe to call on every reload.
    try:
        init_audit_logger(new_config.audit_log_path)
        logger.info("ConfigWatcher: audit logger updated")
    except Exception as exc:
        logger.warning("ConfigWatcher: audit logger re-init failed: %s", exc)
