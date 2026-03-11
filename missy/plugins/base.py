"""Base classes for Missy plugins.

Plugins are externally-loaded components that extend the agent's capabilities.
Unlike skills, plugins are **disabled by default** and must be explicitly
enabled via configuration.  Every plugin must declare a complete
:class:`PluginPermissions` manifest before it can be loaded.

Example::

    from missy.plugins.base import BasePlugin, PluginPermissions

    class WeatherPlugin(BasePlugin):
        name = "weather"
        description = "Fetches weather data from an external API."
        permissions = PluginPermissions(
            network=True,
            allowed_hosts=["api.openweathermap.org"],
        )

        def initialize(self) -> bool:
            # Validate API key, set up client, etc.
            return True

        def execute(self, *, location: str = "") -> dict:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PluginPermissions:
    """Declares every resource a plugin requires.

    Plugins must list all permissions explicitly.  The security review process
    reads the manifest produced by :meth:`BasePlugin.get_manifest` to confirm
    that declared permissions match expected behaviour.

    Attributes:
        network: Plugin may make outbound network requests.
        filesystem_read: Plugin may read from the filesystem.
        filesystem_write: Plugin may write to the filesystem.
        shell: Plugin may execute shell commands.
        allowed_hosts: Explicit hostnames the plugin is permitted to contact.
            Meaningful only when ``network`` is ``True``.
        allowed_paths: Filesystem paths the plugin is permitted to access.
            Meaningful only when ``filesystem_read`` or ``filesystem_write``
            is ``True``.
    """

    network: bool = False
    filesystem_read: bool = False
    filesystem_write: bool = False
    shell: bool = False
    allowed_hosts: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(default_factory=list)


class BasePlugin(ABC):
    """Abstract base for all Missy plugins.

    Concrete subclasses must declare :attr:`name`, :attr:`description`, and
    :attr:`permissions` as class attributes, and implement both
    :meth:`initialize` and :meth:`execute`.

    Class attributes:
        name: Unique registry key for the plugin (e.g. ``"weather"``).
        description: One-line description shown in manifests and help text.
        version: Semantic version string.  Defaults to ``"0.1.0"``.
        permissions: :class:`PluginPermissions` declaring all required
            resources.
        enabled: Whether the plugin is currently active.  Plugins start
            disabled; the :class:`~missy.plugins.loader.PluginLoader` sets
            this to ``True`` after a successful :meth:`initialize` call.
    """

    name: str
    description: str
    version: str = "0.1.0"
    permissions: PluginPermissions
    enabled: bool = False  # Plugins are disabled by default.

    @abstractmethod
    def initialize(self) -> bool:
        """Perform any setup required before the plugin can be used.

        Implementations should acquire external resources (API connections,
        file handles, etc.) here and return ``True`` on success.

        Returns:
            ``True`` when initialisation succeeded, ``False`` otherwise.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Run the plugin's primary action.

        Args:
            **kwargs: Plugin-specific keyword arguments.

        Returns:
            Plugin-specific result value.
        """
        ...

    def get_manifest(self) -> dict[str, Any]:
        """Return a serialisable manifest describing this plugin.

        The manifest is used during security review to confirm that declared
        permissions match the plugin's intended behaviour.

        Returns:
            A dictionary containing ``name``, ``version``, ``description``,
            ``permissions``, and ``enabled``.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "permissions": vars(self.permissions),
            "enabled": self.enabled,
        }
