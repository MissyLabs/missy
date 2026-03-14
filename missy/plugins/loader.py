"""Plugin loader with security checks.

:class:`PluginLoader` is the gatekeeper for all plugin activity.  Plugins
are disabled by default; they can only be loaded when:

1. ``config.plugins.enabled`` is ``True``.
2. The plugin's :attr:`~missy.plugins.base.BasePlugin.name` appears in
   ``config.plugins.allowed_plugins``.

Every load and execute attempt emits an :class:`~missy.core.events.AuditEvent`
through :data:`~missy.core.events.event_bus`.  Denied operations raise
:class:`~missy.core.exceptions.PolicyViolationError`.

Example::

    from missy.plugins.loader import init_plugin_loader, get_plugin_loader
    from missy.config.settings import load_config

    config = load_config("missy.yaml")
    loader = init_plugin_loader(config)
    loader.load_plugin(my_plugin)
    result = loader.execute("my_plugin", session_id="s1", task_id="t1")
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from missy.config.settings import MissyConfig
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import PolicyViolationError
from missy.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class PluginLoader:
    """Loads and executes plugins subject to policy enforcement.

    Args:
        config: The runtime :class:`~missy.config.settings.MissyConfig`.
            The ``plugins`` sub-config controls which plugins are allowed.
    """

    def __init__(self, config: MissyConfig) -> None:
        self.config = config
        self._plugins: dict[str, BasePlugin] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_plugin(self, plugin: BasePlugin) -> bool:
        """Load *plugin* if policy permits, then call its :meth:`initialize` method.

        Policy checks performed in order:

        1. ``config.plugins.enabled`` must be ``True``.
        2. ``plugin.name`` must appear in ``config.plugins.allowed_plugins``.

        If both checks pass, :meth:`~missy.plugins.base.BasePlugin.initialize`
        is called.  A failed initialisation (returns ``False`` or raises) is
        treated as an error but does **not** raise
        :class:`~missy.core.exceptions.PolicyViolationError`.

        Args:
            plugin: The plugin instance to load.

        Returns:
            ``True`` when the plugin was loaded and initialised successfully.

        Raises:
            PolicyViolationError: When the plugin is denied by policy.
        """
        if not self.config.plugins.enabled:
            detail_msg = f"Plugin {plugin.name!r} denied: plugins are disabled in configuration."
            self._emit_event(
                event_type="plugin.load",
                result="deny",
                detail={
                    "plugin": plugin.name,
                    "reason": "plugins_disabled",
                },
            )
            raise PolicyViolationError(
                detail_msg,
                category="plugin",
                detail=detail_msg,
            )

        if plugin.name not in self.config.plugins.allowed_plugins:
            detail_msg = f"Plugin {plugin.name!r} denied: not in allowed_plugins list."
            self._emit_event(
                event_type="plugin.load",
                result="deny",
                detail={
                    "plugin": plugin.name,
                    "reason": "not_in_allowed_list",
                    "allowed": list(self.config.plugins.allowed_plugins),
                },
            )
            raise PolicyViolationError(
                detail_msg,
                category="plugin",
                detail=detail_msg,
            )

        # Policy passed — attempt initialisation.
        try:
            success = plugin.initialize()
        except Exception as exc:
            logger.exception("Plugin %r raised during initialize().", plugin.name)
            self._emit_event(
                event_type="plugin.load",
                result="error",
                detail={"plugin": plugin.name, "error": str(exc)},
            )
            return False

        if not success:
            logger.warning("Plugin %r initialize() returned False.", plugin.name)
            self._emit_event(
                event_type="plugin.load",
                result="error",
                detail={"plugin": plugin.name, "error": "initialize() returned False"},
            )
            return False

        plugin.enabled = True
        self._plugins[plugin.name] = plugin
        logger.info("Loaded plugin %r (%s).", plugin.name, type(plugin).__name__)
        self._emit_event(
            event_type="plugin.load",
            result="allow",
            detail={"plugin": plugin.name, "manifest": plugin.get_manifest()},
        )
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_plugins(self) -> list[dict[str, Any]]:
        """Return the manifest for each currently loaded plugin.

        Returns:
            A list of manifest dicts produced by
            :meth:`~missy.plugins.base.BasePlugin.get_manifest`.
        """
        return [plugin.get_manifest() for plugin in self._plugins.values()]

    def get_plugin(self, name: str) -> BasePlugin | None:
        """Return the plugin registered under *name*, or ``None``.

        Args:
            name: Plugin name to look up.

        Returns:
            The :class:`~missy.plugins.base.BasePlugin` instance, or ``None``
            if no plugin with that name has been loaded.
        """
        return self._plugins.get(name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        name: str,
        session_id: str = "",
        task_id: str = "",
        **kwargs: Any,
    ) -> Any:
        """Execute a loaded plugin with policy checks and audit events.

        Execution is denied if:

        * No plugin with *name* has been loaded.
        * The plugin's :attr:`~missy.plugins.base.BasePlugin.enabled` flag is
          ``False`` (e.g. after a failed initialisation).

        Audit events are emitted before and after execution.  Exceptions from
        the plugin's :meth:`~missy.plugins.base.BasePlugin.execute` method
        propagate to the caller after an ``"error"`` event is emitted.

        Args:
            name: Name of the plugin to execute.
            session_id: Forwarded to audit events.
            task_id: Forwarded to audit events.
            **kwargs: Keyword arguments forwarded to the plugin's
                :meth:`~missy.plugins.base.BasePlugin.execute` method.

        Returns:
            The return value of the plugin's :meth:`execute` method.

        Raises:
            PolicyViolationError: When the plugin is not loaded or not enabled.
            Exception: Any exception raised by the plugin's :meth:`execute`.
        """
        plugin = self._plugins.get(name)

        if plugin is None:
            detail_msg = f"Plugin {name!r} is not loaded."
            self._emit_event(
                event_type="plugin.execute",
                result="deny",
                detail={"plugin": name, "reason": "not_loaded"},
                session_id=session_id,
                task_id=task_id,
            )
            raise PolicyViolationError(detail_msg, category="plugin", detail=detail_msg)

        if not plugin.enabled:
            detail_msg = f"Plugin {name!r} is loaded but not enabled."
            self._emit_event(
                event_type="plugin.execute",
                result="deny",
                detail={"plugin": name, "reason": "not_enabled"},
                session_id=session_id,
                task_id=task_id,
            )
            raise PolicyViolationError(detail_msg, category="plugin", detail=detail_msg)

        self._emit_event(
            event_type="plugin.execute.start",
            result="allow",
            detail={"plugin": name},
            session_id=session_id,
            task_id=task_id,
        )

        try:
            result = plugin.execute(**kwargs)
        except Exception as exc:
            logger.exception("Plugin %r raised during execute().", name)
            self._emit_event(
                event_type="plugin.execute",
                result="error",
                detail={"plugin": name, "error": str(exc)},
                session_id=session_id,
                task_id=task_id,
            )
            raise

        self._emit_event(
            event_type="plugin.execute",
            result="allow",
            detail={"plugin": name},
            session_id=session_id,
            task_id=task_id,
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        event_type: str,
        result: str,
        detail: dict[str, Any],
        session_id: str = "",
        task_id: str = "",
    ) -> None:
        """Publish a plugin audit event to the global event bus.

        Args:
            event_type: Dotted event type string.
            result: One of ``"allow"``, ``"deny"``, or ``"error"``.
            detail: Structured event data.
            session_id: Optional session identifier.
            task_id: Optional task identifier.
        """
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type=event_type,
                category="plugin",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit plugin audit event %r", event_type)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_loader: PluginLoader | None = None
_lock: threading.Lock = threading.Lock()


def init_plugin_loader(config: MissyConfig) -> PluginLoader:
    """Create and install a process-level :class:`PluginLoader`.

    Subsequent calls replace the existing loader atomically under a lock.

    Args:
        config: Runtime configuration; its ``plugins`` sub-config controls
            which plugins are permitted.

    Returns:
        The newly installed :class:`PluginLoader`.
    """
    global _loader
    loader = PluginLoader(config)
    with _lock:
        _loader = loader
    return loader


def get_plugin_loader() -> PluginLoader:
    """Return the process-level :class:`PluginLoader`.

    Returns:
        The currently installed :class:`PluginLoader`.

    Raises:
        RuntimeError: When :func:`init_plugin_loader` has not yet been called.
    """
    with _lock:
        loader = _loader
    if loader is None:
        raise RuntimeError(
            "PluginLoader has not been initialised. "
            "Call missy.plugins.loader.init_plugin_loader(config) first."
        )
    return loader
