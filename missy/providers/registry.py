"""Provider registry for the Missy framework.

The :class:`ProviderRegistry` is the single point of truth for which AI
provider instances are active in a given process.  The module-level
singleton is initialised by :func:`init_registry` and retrieved by
:func:`get_registry`.

Example::

    from missy.config.settings import load_config
    from missy.providers.registry import init_registry, get_registry

    config = load_config("missy.yaml")
    init_registry(config)

    registry = get_registry()
    provider = registry.get("anthropic")
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from missy.config.settings import MissyConfig, ProviderConfig

from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# Maps the canonical provider *name* field in ProviderConfig to the
# corresponding concrete provider class.
_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "ollama": OllamaProvider,
}


class ProviderRegistry:
    """Registry that maps string names to :class:`~.base.BaseProvider` instances.

    Providers are registered by name and can be retrieved individually or
    filtered to only those that report themselves as available.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, name: str, provider: BaseProvider) -> None:
        """Add *provider* under the given *name*.

        A previous registration under the same name is silently replaced.

        Args:
            name: Registry key for the provider.
            provider: The provider instance to register.
        """
        self._providers[name] = provider

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseProvider]:
        """Return the provider registered under *name*, or ``None``.

        Args:
            name: Registry key to look up.

        Returns:
            The :class:`~.base.BaseProvider` instance, or ``None`` if the
            name is not registered.
        """
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """Return a sorted list of all registered provider names.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        return sorted(self._providers)

    def get_available(self) -> list[BaseProvider]:
        """Return providers that report themselves as available.

        Iterates over all registered providers and returns those for which
        :meth:`~.base.BaseProvider.is_available` returns ``True``.

        Returns:
            A list of available :class:`~.base.BaseProvider` instances in
            registration order (dict insertion order, Python 3.7+).
        """
        available: list[BaseProvider] = []
        for provider in self._providers.values():
            try:
                if provider.is_available():
                    available.append(provider)
            except Exception:
                logger.exception(
                    "is_available() raised for provider %r; treating as unavailable",
                    provider.name,
                )
        return available

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: MissyConfig) -> "ProviderRegistry":
        """Build a registry from *config*, skipping providers that cannot be instantiated.

        Iterates over ``config.providers`` and attempts to construct a
        concrete provider for each entry.  Providers whose ``name`` field
        does not map to a known implementation are skipped with a warning.

        Args:
            config: The fully-populated runtime configuration.

        Returns:
            A new :class:`ProviderRegistry` containing successfully
            constructed provider instances.
        """
        registry = cls()
        for key, provider_config in config.providers.items():
            provider_name = provider_config.name or key
            provider_cls = _PROVIDER_CLASSES.get(provider_name)
            if provider_cls is None:
                logger.warning(
                    "Unknown provider name %r (key=%r); skipping.", provider_name, key
                )
                continue
            try:
                instance = provider_cls(provider_config)
                registry.register(key, instance)
                logger.debug("Registered provider %r (%s).", key, provider_cls.__name__)
            except Exception:
                logger.exception(
                    "Failed to construct provider %r; skipping.", key
                )
        return registry


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ProviderRegistry] = None
_lock: threading.Lock = threading.Lock()


def init_registry(config: MissyConfig) -> ProviderRegistry:
    """Construct and install the process-level :class:`ProviderRegistry`.

    A subsequent call replaces the existing registry atomically.

    Args:
        config: Runtime configuration used to populate the registry.

    Returns:
        The newly installed :class:`ProviderRegistry`.
    """
    global _registry
    registry = ProviderRegistry.from_config(config)
    with _lock:
        _registry = registry
    return registry


def get_registry() -> ProviderRegistry:
    """Return the process-level :class:`ProviderRegistry`.

    Returns:
        The currently installed :class:`ProviderRegistry`.

    Raises:
        RuntimeError: When :func:`init_registry` has not yet been called.
    """
    with _lock:
        registry = _registry
    if registry is None:
        raise RuntimeError(
            "ProviderRegistry has not been initialised. "
            "Call missy.providers.registry.init_registry(config) first."
        )
    return registry
