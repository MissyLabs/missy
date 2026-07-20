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
import copy
import logging
import threading
import time
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

    AVAILABILITY_CACHE_SECONDS = 5.0
    AVAILABILITY_BULK_DEADLINE_SECONDS = 0.5

    def __init__(self) -> None:
        self._providers: dict[str, BaseProvider] = {}
        self._key_indices: dict[str, int] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}
        self._default_name: str | None = None
        # Runtime operator toggle (Web TUI provider enable/disable).
        # Distinct from ProviderConfig.enabled, which gates construction
        # in from_config(): a name in this set stays registered (so it
        # can be re-enabled without a restart) but is excluded from
        # get_available() and refused by set_default().
        self._runtime_disabled: set[str] = set()
        self._effective_provider_hosts: tuple[str, ...] = ()
        self._availability_cache: dict[str, tuple[bool, float]] = {}
        self._availability_inflight: dict[str, threading.Event] = {}
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

        Registration is idempotent for the same instance.  A different
        implementation cannot silently shadow live/default/disabled state.

        Args:
            name: Registry key for the provider.
            provider: The provider instance to register.
            config: Optional provider configuration for key rotation support.
        """
        with self._lock:
            existing = self._providers.get(name)
            if existing is not None and existing is not provider:
                raise ValueError(f"Provider name {name!r} is already registered.")
            if existing is provider:
                existing_config = self._provider_configs.get(name)
                if config is not None and existing_config is not None and config != existing_config:
                    raise ValueError(
                        f"Provider name {name!r} is already registered with different config."
                    )
                return
            self._providers[name] = provider
            if config is not None:
                self._provider_configs[name] = copy.deepcopy(config)
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
        with self._lock:
            config = self._provider_configs.get(provider_name)
            return copy.deepcopy(config) if config is not None else None

    @property
    def effective_provider_hosts(self) -> tuple[str, ...]:
        """Return the immutable derived provider-egress host snapshot."""
        return self._effective_provider_hosts

    def rotate_key(self, provider_name: str) -> None:
        """Rotate to the next API key for the named provider (round-robin).

        If the provider has multiple ``api_keys`` configured, advances the
        internal index and updates the provider's active API key.  Has no
        effect when fewer than two keys are configured.

        No-ops (beyond a debug log) when the provider reports
        ``is_multi_account = True`` (currently only
        :class:`~missy.providers.openai_provider.OpenAIProvider` with
        ``key_rotation_strategy: "round_robin"``): that provider already
        balances every call across all configured accounts internally via
        its own per-call round-robin selection, each with its own
        independent rate limiter, so mutating this registry's separate
        legacy ``_key_indices``/``provider.api_key`` bookkeeping here would
        be dead state that nothing reads -- confusing at best, since the
        "rotated to key index N" log would imply an effect this call never
        actually has for such a provider.

        Args:
            provider_name: Registry key of the provider to rotate.
        """
        with self._lock:
            config = self._provider_configs.get(provider_name)
            provider = self._providers.get(provider_name)
            if config is None or provider is None:
                logger.warning("rotate_key: provider %r not found.", provider_name)
                return
            if getattr(provider, "is_multi_account", False):
                logger.debug(
                    "rotate_key: provider %r already balances calls across its "
                    "configured accounts internally; skipping legacy rotation.",
                    provider_name,
                )
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

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a registered provider at runtime.

        Disabling keeps the provider registered (so it can be re-enabled
        without a restart) but excludes it from :meth:`get_available`
        and makes :meth:`set_default` refuse it.

        Args:
            name: Registry key of the provider to toggle.
            enabled: Desired enabled state.

        Raises:
            ValueError: If the name is not registered, or when disabling
                the current default provider (switch the default first).
        """
        with self._lock:
            if name not in self._providers:
                raise ValueError(f"Provider {name!r} is not registered.")
            if not enabled and name == self._default_name:
                raise ValueError(
                    f"Provider {name!r} is the current default; "
                    "set a different default before disabling it."
                )
            if enabled:
                self._runtime_disabled.discard(name)
            else:
                self._runtime_disabled.add(name)
            self._availability_cache.pop(name, None)
        logger.info("Provider %r %s at runtime.", name, "enabled" if enabled else "disabled")

    def is_enabled(self, name: str) -> bool:
        """Return whether a registered provider is runtime-enabled."""
        with self._lock:
            return name in self._providers and name not in self._runtime_disabled

    def set_default(self, name: str) -> None:
        """Set the default provider by name.

        Args:
            name: Registry key of the provider to make default.

        Raises:
            ValueError: If the name is not registered, runtime-disabled,
                or the provider is not available.
        """
        with self._lock:
            provider = self._providers.get(name)
            enabled = name not in self._runtime_disabled
        if provider is None:
            raise ValueError(f"Provider {name!r} is not registered.")
        if not enabled:
            raise ValueError(f"Provider {name!r} is disabled; enable it first.")
        if not self._availability_for(name, provider):
            raise ValueError(f"Provider {name!r} is not available or its probe timed out.")
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
        with self._lock:
            snapshot = [
                (name, provider)
                for name, provider in self._providers.items()
                if name not in self._runtime_disabled
            ]
        deadline = time.monotonic() + self.AVAILABILITY_BULK_DEADLINE_SECONDS
        events = [self._ensure_availability_probe(name, provider) for name, provider in snapshot]
        for event in events:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            event.wait(remaining)
        now = time.monotonic()
        with self._lock:
            return [
                provider
                for name, provider in snapshot
                if (cached := self._availability_cache.get(name)) is not None
                and now - cached[1] <= self.AVAILABILITY_CACHE_SECONDS
                and cached[0]
            ]

    def availability_status(self) -> dict[str, dict[str, object]]:
        """Return credential-free probe freshness and in-flight state."""
        now = time.monotonic()
        with self._lock:
            names = list(self._providers)
            return {
                name: {
                    "available": self._availability_cache.get(name, (False, 0.0))[0],
                    "age_seconds": (
                        max(0.0, now - self._availability_cache[name][1])
                        if name in self._availability_cache
                        else None
                    ),
                    "stale": (
                        name not in self._availability_cache
                        or now - self._availability_cache[name][1] > self.AVAILABILITY_CACHE_SECONDS
                    ),
                    "in_flight": name in self._availability_inflight,
                    "enabled": name not in self._runtime_disabled,
                }
                for name in names
            }

    def _availability_for(self, name: str, provider: BaseProvider) -> bool:
        event = self._ensure_availability_probe(name, provider)
        event.wait(self.AVAILABILITY_BULK_DEADLINE_SECONDS)
        now = time.monotonic()
        with self._lock:
            cached = self._availability_cache.get(name)
            return bool(cached and now - cached[1] <= self.AVAILABILITY_CACHE_SECONDS and cached[0])

    def _ensure_availability_probe(self, name: str, provider: BaseProvider) -> threading.Event:
        now = time.monotonic()
        with self._lock:
            cached = self._availability_cache.get(name)
            if cached is not None and now - cached[1] <= self.AVAILABILITY_CACHE_SECONDS:
                ready = threading.Event()
                ready.set()
                return ready
            existing = self._availability_inflight.get(name)
            if existing is not None:
                return existing
            event = threading.Event()
            self._availability_inflight[name] = event

        def probe() -> None:
            try:
                available = bool(provider.is_available())
            except Exception:
                available = False
                logger.warning("Availability probe failed for provider %r; details withheld.", name)
            checked_at = time.monotonic()
            with self._lock:
                if self._providers.get(name) is provider and name not in self._runtime_disabled:
                    self._availability_cache[name] = (available, checked_at)
                self._availability_inflight.pop(name, None)
                event.set()

        threading.Thread(
            target=probe,
            daemon=True,
            name=f"missy-provider-probe-{name[:32]}",
        ).start()
        return event

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
        # Runtime construction must not rewrite the operator's parsed config:
        # doing so makes config diff/rollback lie and lets a failed provider
        # partially widen caller-owned policy state.  Providers and the
        # registry receive an isolated snapshot instead.
        config = copy.deepcopy(config)
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
                                detail={"provider": provider_config.name, "host": host},
                            )
                        )

        registry._effective_provider_hosts = tuple(config.network.provider_allowed_hosts)

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
