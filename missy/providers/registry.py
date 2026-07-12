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

import contextlib
import logging
import threading
from urllib.parse import urlparse

from missy.config.settings import MissyConfig, ProviderConfig

from .acpx_provider import AcpxProvider
from .anthropic_provider import AnthropicProvider
from .base import BaseProvider
from .codex_provider import CodexProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Maps the canonical provider *name* field in ProviderConfig to the
# corresponding concrete provider class.
_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "acpx": AcpxProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "openai-codex": CodexProvider,
    "ollama": OllamaProvider,
}


class ProviderRegistry:
    """Registry that maps string names to :class:`~.base.BaseProvider` instances.

    Providers are registered by name and can be retrieved individually or
    filtered to only those that report themselves as available.
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._key_indices: dict[str, int] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}
        self._default_name: str | None = None
        # Guards every mutation of the dicts above, and every read that
        # iterates them (rather than a single dict.get()/[] lookup,
        # which CPython already makes atomic). Found live via a
        # concurrency stress test: concurrent register() (a dict
        # mutation) racing with get_available()/list_providers()/
        # key_for() (each iterating self._providers directly) could
        # raise "RuntimeError: dictionary changed size during
        # iteration" -- a real, pre-existing thread-safety gap, not
        # merely a theoretical one.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(
        self, name: str, provider: BaseProvider, config: ProviderConfig | None = None
    ) -> None:
        """Add *provider* under the given *name*.

        A previous registration under the same name is silently replaced.

        Args:
            name: Registry key for the provider.
            provider: The provider instance to register.
            config: Optional provider configuration for key rotation support.
        """
        with self._lock:
            self._providers[name] = provider
            if config is not None:
                self._provider_configs[name] = config
                self._key_indices.setdefault(name, 0)

    def get_config(self, provider_name: str) -> ProviderConfig | None:
        """Return the :class:`ProviderConfig` registered for *provider_name*.

        Used by callers that need per-provider tunables (e.g. the
        runtime's per-provider :class:`~missy.agent.circuit_breaker.CircuitBreaker`
        threshold/cooldown, SR-4.8 residual) without duplicating the
        registry's own config bookkeeping.

        Args:
            provider_name: Registry key of the provider.

        Returns:
            The registered :class:`ProviderConfig`, or ``None`` if the
            provider was registered without one (or isn't registered at
            all).
        """
        return self._provider_configs.get(provider_name)

    def rotate_key(self, provider_name: str) -> None:
        """Rotate to the next API key for the named provider (round-robin).

        If the provider has multiple ``api_keys`` configured, advances the
        internal index and updates the provider's active API key.  Has no
        effect when fewer than two keys are configured.

        Args:
            provider_name: Registry key of the provider to rotate.
        """
        with self._lock:
            config = self._provider_configs.get(provider_name)
            provider = self._providers.get(provider_name)
            if config is None or provider is None:
                logger.warning("rotate_key: provider %r not found.", provider_name)
                return
            keys = getattr(config, "api_keys", [])
            if len(keys) < 2:
                logger.debug(
                    "rotate_key: provider %r has fewer than 2 api_keys; skipping rotation.",
                    provider_name,
                )
                return
            current_idx = self._key_indices.get(provider_name, 0)
            next_idx = (current_idx + 1) % len(keys)
            self._key_indices[provider_name] = next_idx
            next_key = keys[next_idx]
        # Update the provider's api_key attribute if accessible.
        if hasattr(provider, "api_key"):
            provider.api_key = next_key  # type: ignore[attr-defined]
        elif hasattr(provider, "_api_key"):
            provider._api_key = next_key  # type: ignore[attr-defined]
        logger.info("rotate_key: provider %r rotated to key index %d.", provider_name, next_idx)

    def set_default(self, name: str) -> None:
        """Set the default provider by name.

        Args:
            name: Registry key of the provider to make default.

        Raises:
            ValueError: If the name is not registered or the provider is
                not available.
        """
        provider = self._providers.get(name)
        if provider is None:
            raise ValueError(f"Provider {name!r} is not registered.")
        # is_available() may perform real I/O (e.g. an HTTP health check);
        # deliberately not held under self._lock so a slow/blocking check
        # for one provider can't stall unrelated register()/rotate_key()
        # calls on other providers. Only the final assignment needs the
        # lock (a single attribute write is already atomic in CPython,
        # but taking it anyway keeps this consistent with every other
        # mutation of registry state going through the same guard).
        try:
            if not provider.is_available():
                raise ValueError(f"Provider {name!r} is not available.")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Provider {name!r} availability check failed: {exc}") from exc
        with self._lock:
            self._default_name = name
        logger.info("Default provider set to %r.", name)

    def get_default_name(self) -> str | None:
        """Return the name of the current default provider, or ``None``."""
        return self._default_name

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseProvider | None:
        """Return the provider registered under *name*, or ``None``.

        Args:
            name: Registry key to look up.

        Returns:
            The :class:`~.base.BaseProvider` instance, or ``None`` if the
            name is not registered.
        """
        return self._providers.get(name)

    def key_for(self, provider: BaseProvider) -> str | None:
        """Return the registry key *provider* was registered under, or ``None``.

        Used by callers that only hold a provider instance (e.g. selected
        via :meth:`get_available`) but need the registry key to call
        :meth:`rotate_key`, since a provider's registry key need not match
        its class-level ``name`` attribute.

        Args:
            provider: A provider instance previously passed to :meth:`register`.

        Returns:
            The registry key, or ``None`` if *provider* is not registered
            (identity comparison, not equality).
        """
        with self._lock:
            snapshot = list(self._providers.items())
        for key, registered in snapshot:
            if registered is provider:
                return key
        return None

    def list_providers(self) -> list[str]:
        """Return a sorted list of all registered provider names.

        Returns:
            A new list of string keys in ascending alphabetical order.
        """
        with self._lock:
            return sorted(self._providers)

    def get_available(self) -> list[BaseProvider]:
        """Return providers that report themselves as available.

        Iterates over all registered providers and returns those for which
        :meth:`~.base.BaseProvider.is_available` returns ``True``.

        Returns:
            A list of available :class:`~.base.BaseProvider` instances in
            registration order (dict insertion order, Python 3.7+).
        """
        # Snapshot under the lock, then iterate (and call each provider's
        # potentially I/O-bound is_available()) outside it -- otherwise
        # a slow health check would hold self._lock for its full
        # duration, blocking unrelated register()/rotate_key() calls
        # from other threads for no good reason.
        with self._lock:
            snapshot = list(self._providers.values())
        available: list[BaseProvider] = []
        for provider in snapshot:
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
    def from_config(cls, config: MissyConfig) -> ProviderRegistry:
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
        # Auto-populate provider_allowed_hosts from provider base_url entries
        # so users don't have to duplicate hosts in network policy manually.
        #
        # Availability/transparency hardening: this mutates the network
        # policy's "provider" category egress allowlist -- a security-
        # relevant action -- but previously did so completely silently
        # (logger.debug() only, invisible at default log levels; no audit
        # trail at all). An operator setting base_url for one provider
        # (e.g. pointing to a self-hosted OpenAI-compatible endpoint)
        # would never see that their policy's effective allowlist grew,
        # and nothing covered this path under the "structured audit events
        # for privileged actions" guarantee. SSRF/DNS-
        # rebinding via a maliciously-crafted base_url is already closed
        # independently by SR-1.9a's rebinding check (every hostname
        # match, including ones added here, still re-verifies the
        # resolved IP isn't private/loopback/link-local before any
        # request is allowed) and by NetworkPolicyEngine's own bare-IP
        # path never consulting allowed_hosts at all -- this fix is about
        # making the widening itself visible and auditable, not about an
        # exploitable bypass.
        existing = {h.lower() for h in config.network.provider_allowed_hosts}
        for provider_config in config.providers.values():
            if provider_config.enabled and provider_config.base_url:
                parsed = urlparse(provider_config.base_url)
                host = parsed.hostname
                if host and host.lower() not in existing:
                    config.network.provider_allowed_hosts.append(host)
                    existing.add(host.lower())
                    logger.warning(
                        "Provider %r's base_url expanded the network policy's "
                        "'provider' category egress allowlist to include %r. "
                        "If this host is unexpected, check config.providers "
                        "for a stray or malicious base_url.",
                        provider_config.name,
                        host,
                    )
                    with contextlib.suppress(Exception):
                        from missy.core.events import AuditEvent, event_bus

                        event_bus.publish(
                            AuditEvent.now(
                                session_id="",
                                task_id="",
                                event_type="provider.base_url_egress_widened",
                                category="network",
                                result="allow",
                                detail={
                                    "provider": provider_config.name,
                                    "host": host,
                                    "base_url": provider_config.base_url,
                                },
                            )
                        )

        for key, provider_config in config.providers.items():
            if not provider_config.enabled:
                logger.info("Provider %r is disabled; skipping.", key)
                continue
            provider_name = provider_config.name or key
            provider_cls = _PROVIDER_CLASSES.get(provider_name)
            if provider_cls is None:
                logger.warning("Unknown provider name %r (key=%r); skipping.", provider_name, key)
                continue
            try:
                instance = provider_cls(provider_config)
                # Attach a rate limiter sized from the provider's configured
                # RPM/TPM/wait limits so operators can tune to their plan tier.
                instance.rate_limiter = RateLimiter(
                    requests_per_minute=getattr(provider_config, "requests_per_minute", 60),
                    tokens_per_minute=getattr(provider_config, "tokens_per_minute", 100_000),
                    max_wait_seconds=getattr(provider_config, "max_wait_seconds", 30.0),
                )
                registry.register(key, instance, config=provider_config)
                logger.debug("Registered provider %r (%s).", key, provider_cls.__name__)
            except Exception:
                logger.exception("Failed to construct provider %r; skipping.", key)
        return registry


# ---------------------------------------------------------------------------
# Model router
# ---------------------------------------------------------------------------


class ModelRouter:
    """Routes tasks to fast/primary/premium provider tiers based on complexity."""

    PREMIUM_KEYWORDS = frozenset(
        ["debug", "architect", "refactor", "analyze", "optimize", "complex"]
    )
    FAST_INDICATORS = frozenset(["what", "how", "when", "where", "who", "list", "show"])

    def score_complexity(self, prompt: str, history_length: int = 0, tool_count: int = 0) -> str:
        """Return 'fast' | 'primary' | 'premium' for the given inputs.

        Args:
            prompt: The user prompt text.
            history_length: Number of prior turns in the conversation.
            tool_count: Number of tools available or requested.

        Returns:
            One of ``"fast"``, ``"primary"``, or ``"premium"``.
        """
        prompt_lower = prompt.lower()
        words = set(prompt_lower.split())

        # Force fast if message is very short and contains only fast indicators.
        if len(prompt) < 80 and tool_count == 0 and words & self.FAST_INDICATORS:
            return "fast"

        # Premium signals.
        if (
            words & self.PREMIUM_KEYWORDS
            or history_length > 10
            or tool_count > 3
            or len(prompt) > 500
        ):
            return "premium"

        return "primary"

    def select_model(self, provider_config: ProviderConfig, tier: str) -> str:
        """Return the model name for the given tier, falling back to primary model.

        Args:
            provider_config: The provider configuration to inspect.
            tier: One of ``"fast"``, ``"primary"``, or ``"premium"``.

        Returns:
            The model identifier string appropriate for the tier.
        """
        if tier == "fast" and getattr(provider_config, "fast_model", ""):
            return provider_config.fast_model
        if tier == "premium" and getattr(provider_config, "premium_model", ""):
            return provider_config.premium_model
        return provider_config.model


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: ProviderRegistry | None = None
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
