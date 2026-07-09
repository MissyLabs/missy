"""Provider-aware tool enablement based on benchmark results.

:class:`ToolProviderGate` decides whether a given (tool, provider) pair
should be exposed to the agent loop for a turn.  Two signals feed the
decision, checked in order:

1. **Operator overrides** — explicit ``enable``/``disable`` calls persisted
   by :class:`ProviderGateStore`.  These always win, in either direction:
   an operator can force-enable a tool the benchmarks call weak, or
   force-disable one the benchmarks call strong.
2. **Benchmark data** — when a (tool, provider) pair has at least
   ``min_samples`` benchmark runs (from either the direct
   :class:`~missy.tools.benchmark.runner.BenchmarkRunner` or the
   :class:`~missy.tools.benchmark.llm_runner.LLMBenchmarkRunner`) and its
   mean composite score is below ``min_composite``, the tool is treated as
   disabled for that provider.

With no override and no (or insufficient) benchmark data, a tool is enabled
by default — gating only ever *removes* access based on evidence, it never
requires benchmarking before first use.

This module only decides *availability*; it does not touch tool execution
policy (still enforced by :class:`~missy.tools.registry.ToolRegistry`) or the
static config-driven :mod:`~missy.policy.tool_policy_pipeline` layers. Both
gates are applied independently by :class:`~missy.agent.runtime.AgentRuntime`.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from missy.core.events import AuditEvent, event_bus

logger = logging.getLogger(__name__)

_DEFAULT_OVERRIDES_PATH = Path("~/.missy/tool_provider_overrides.json")


@dataclass(frozen=True)
class GateDecision:
    """Outcome of a gating check for one (tool, provider) pair.

    Attributes:
        enabled: Whether the tool should be exposed to this provider.
        reason: Human-readable explanation ("operator override", a
            benchmark summary, or "no data").
        source: One of ``"override"``, ``"benchmark"``, or ``"default"``.
    """

    enabled: bool
    reason: str
    source: str


class ProviderGateStore:
    """Persists explicit operator enable/disable overrides.

    Backed by a small JSON file at *path* (default
    ``~/.missy/tool_provider_overrides.json``) since the data volume is tiny
    and human-readable overrides are useful for `missy config diff`-style
    inspection. Every mutation emits an audit event.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path or _DEFAULT_OVERRIDES_PATH).expanduser()
        self._lock = threading.Lock()
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def get(self, tool_name: str, provider_name: str) -> bool | None:
        """Return the explicit override for *(tool_name, provider_name)*, or ``None``."""
        data = self._read()
        entry = data.get(tool_name, {})
        if provider_name not in entry:
            return None
        return bool(entry[provider_name])

    def set(
        self,
        tool_name: str,
        provider_name: str,
        enabled: bool,
        actor: str = "operator",
    ) -> None:
        """Persist an explicit override and emit an audit event."""
        with self._lock:
            data = self._read()
            data.setdefault(tool_name, {})[provider_name] = bool(enabled)
            self._write(data)
        _emit_audit(
            "tool.provider_gate.override_set",
            {
                "tool": tool_name,
                "provider": provider_name,
                "enabled": enabled,
                "actor": actor,
            },
        )
        logger.info(
            "ProviderGateStore: %s override for tool=%r provider=%r by %r",
            "enabled" if enabled else "disabled",
            tool_name,
            provider_name,
            actor,
        )

    def clear(self, tool_name: str, provider_name: str, actor: str = "operator") -> bool:
        """Remove an override, reverting to benchmark-driven/default behavior.

        Returns:
            ``True`` if an override existed and was removed.
        """
        with self._lock:
            data = self._read()
            entry = data.get(tool_name, {})
            if provider_name not in entry:
                return False
            del entry[provider_name]
            if not entry:
                data.pop(tool_name, None)
            self._write(data)
        _emit_audit(
            "tool.provider_gate.override_cleared",
            {"tool": tool_name, "provider": provider_name, "actor": actor},
        )
        return True

    def list_overrides(self) -> dict[str, dict[str, bool]]:
        """Return all stored overrides as ``{tool_name: {provider_name: enabled}}``."""
        return self._read()

    def _read(self) -> dict[str, dict[str, bool]]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("ProviderGateStore: failed to read %s; treating as empty.", self._path)
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(tool): {str(p): bool(v) for p, v in providers.items()}
            for tool, providers in raw.items()
            if isinstance(providers, dict)
        }

    def _write(self, data: dict[str, dict[str, bool]]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)
        self._path.chmod(0o600)


class ToolProviderGate:
    """Combine operator overrides and benchmark data into gating decisions.

    Args:
        overrides: :class:`ProviderGateStore` for explicit operator
            enable/disable calls. Defaults to the module-level singleton.
        benchmark_store: :class:`~missy.tools.benchmark.benchmark_store.BenchmarkStore`
            supplying aggregate scores. Defaults to the module-level singleton.
        min_samples: Minimum benchmark runs before a score is trusted.
        min_composite: Composite score threshold below which a provider is
            considered weak for a tool.
    """

    def __init__(
        self,
        overrides: ProviderGateStore | None = None,
        benchmark_store: Any = None,
        min_samples: int = 3,
        min_composite: float = 0.4,
    ) -> None:
        self._overrides = overrides or get_provider_gate_store()
        if benchmark_store is None:
            from missy.tools.benchmark import get_benchmark_store

            benchmark_store = get_benchmark_store()
        self._benchmark_store = benchmark_store
        self._min_samples = max(1, min_samples)
        self._min_composite = min_composite

    def decide(self, tool_name: str, provider_name: str) -> GateDecision:
        """Return the :class:`GateDecision` for *(tool_name, provider_name)*."""
        override = self._overrides.get(tool_name, provider_name)
        if override is not None:
            return GateDecision(
                enabled=override,
                reason=f"operator override: {'enabled' if override else 'disabled'}",
                source="override",
            )

        summaries = self._benchmark_store.provider_summary(tool_name)
        for summary in summaries:
            if summary.provider != provider_name:
                continue
            if summary.run_count < self._min_samples:
                return GateDecision(
                    enabled=True,
                    reason=(
                        f"only {summary.run_count} benchmark run(s), "
                        f"below min_samples={self._min_samples}"
                    ),
                    source="default",
                )
            if summary.mean_composite < self._min_composite:
                return GateDecision(
                    enabled=False,
                    reason=(
                        f"benchmark composite {summary.mean_composite:.3f} < "
                        f"threshold {self._min_composite:.3f} over "
                        f"{summary.run_count} run(s)"
                    ),
                    source="benchmark",
                )
            return GateDecision(
                enabled=True,
                reason=(
                    f"benchmark composite {summary.mean_composite:.3f} >= "
                    f"threshold {self._min_composite:.3f} over "
                    f"{summary.run_count} run(s)"
                ),
                source="benchmark",
            )

        return GateDecision(enabled=True, reason="no benchmark data", source="default")

    def filter_tools(
        self, tool_names: list[str], provider_name: str
    ) -> tuple[list[str], dict[str, str]]:
        """Split *tool_names* into (allowed, {denied_name: reason}).

        Denials are audited individually so operators can trace why a tool
        disappeared from a provider's turn.
        """
        allowed: list[str] = []
        denied: dict[str, str] = {}
        for name in tool_names:
            decision = self.decide(name, provider_name)
            if decision.enabled:
                allowed.append(name)
            else:
                denied[name] = decision.reason
                _emit_audit(
                    "tool.provider_gate.denied",
                    {"tool": name, "provider": provider_name, "reason": decision.reason},
                )
        return allowed, denied

    def recommend_provider(self, tool_name: str, candidate_providers: list[str]) -> str | None:
        """Suggest the best-performing available provider for *tool_name*.

        Used as the "fallback behavior when a provider is weak" hook: when a
        provider is gated off a tool, callers can offer this recommendation
        (e.g. in CLI diagnostics or a future auto-routing decision) instead of
        silently failing.

        Args:
            tool_name: Tool to evaluate.
            candidate_providers: Providers to rank (only ones with benchmark
                data and no disabling override are considered).

        Returns:
            The provider name with the highest mean composite score among
            *candidate_providers* that this gate would currently enable, or
            ``None`` if none qualify.
        """
        best: tuple[str, float] | None = None
        summaries = {s.provider: s for s in self._benchmark_store.provider_summary(tool_name)}
        for provider_name in candidate_providers:
            decision = self.decide(tool_name, provider_name)
            if not decision.enabled:
                continue
            summary = summaries.get(provider_name)
            score = summary.mean_composite if summary else 0.0
            if best is None or score > best[1]:
                best = (provider_name, score)
        return best[0] if best else None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: ProviderGateStore | None = None
_store_lock = threading.Lock()


def get_provider_gate_store(path: Path | str | None = None) -> ProviderGateStore:
    """Return (or lazily create) the module-level :class:`ProviderGateStore`."""
    global _store
    with _store_lock:
        if _store is None:
            _store = ProviderGateStore(path=path)
        return _store


def _emit_audit(event_type: str, detail: dict[str, Any]) -> None:
    try:
        event_bus.publish(
            AuditEvent.now(
                session_id="",
                task_id="",
                event_type=event_type,
                category="tool",
                result="allow",
                detail=detail,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("ToolProviderGate: audit emit failed: %s", exc)
