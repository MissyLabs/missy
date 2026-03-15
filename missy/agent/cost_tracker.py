"""Per-session cost tracking and budget enforcement.

Accumulates token usage from every provider call and computes the running
USD cost using a built-in pricing table.  When a ``max_spend_usd`` limit
is set, :meth:`CostTracker.check_budget` raises :class:`BudgetExceededError`
to halt the agent loop before the limit is exceeded.

Example::

    from missy.agent.cost_tracker import CostTracker

    tracker = CostTracker(max_spend_usd=0.50)
    tracker.record(model="claude-sonnet-4-20250514", prompt_tokens=500, completion_tokens=200)
    print(tracker.total_cost_usd)
    tracker.check_budget()  # raises BudgetExceededError if over limit
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when the accumulated cost exceeds the configured budget."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exceeded: ${spent:.4f} spent against ${limit:.4f} limit.")


# ---------------------------------------------------------------------------
# Pricing table — USD per 1 000 tokens (input / output)
# ---------------------------------------------------------------------------

# Models are matched by prefix so that dated variants (e.g.
# "claude-sonnet-4-20250514") automatically match the base entry.
# More specific prefixes are checked first.

_PRICING: list[tuple[str, float, float]] = [
    # Anthropic
    ("claude-opus-4", 0.015, 0.075),
    ("claude-sonnet-4", 0.003, 0.015),
    ("claude-haiku-4", 0.0008, 0.004),
    ("claude-3-5-sonnet", 0.003, 0.015),
    ("claude-3-5-haiku", 0.0008, 0.004),
    ("claude-3-opus", 0.015, 0.075),
    ("claude-3-sonnet", 0.003, 0.015),
    ("claude-3-haiku", 0.00025, 0.00125),
    # OpenAI
    ("gpt-4.1", 0.002, 0.008),
    ("gpt-4.1-mini", 0.0004, 0.0016),
    ("gpt-4.1-nano", 0.0001, 0.0004),
    ("gpt-4o-mini", 0.00015, 0.0006),
    ("gpt-4o", 0.0025, 0.01),
    ("gpt-4-turbo", 0.01, 0.03),
    ("gpt-4", 0.03, 0.06),
    ("gpt-3.5-turbo", 0.0005, 0.0015),
    ("o3-mini", 0.0011, 0.0044),
    ("o3", 0.01, 0.04),
    ("o4-mini", 0.0011, 0.0044),
    # Ollama / local — zero cost
    ("llama", 0.0, 0.0),
    ("mistral", 0.0, 0.0),
    ("codellama", 0.0, 0.0),
    ("deepseek", 0.0, 0.0),
    ("phi", 0.0, 0.0),
    ("qwen", 0.0, 0.0),
    ("gemma", 0.0, 0.0),
]


def _lookup_pricing(model: str) -> tuple[float, float]:
    """Return (input_cost_per_1k, output_cost_per_1k) for *model*.

    Falls back to zero if no prefix matches (e.g. unknown local models).
    """
    model_lower = model.lower()
    for prefix, inp, out in _PRICING:
        if model_lower.startswith(prefix):
            return inp, out
    logger.debug("No pricing entry for model %r; assuming zero cost.", model)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Usage record
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """A single provider call's token usage and cost.

    Attributes:
        model: The model identifier used for the call.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.
        cost_usd: Computed cost in USD.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class CostTracker:
    """Accumulate token usage and enforce an optional spending budget.

    Thread-safe: all mutations are guarded by an internal lock.

    Args:
        max_spend_usd: Optional maximum spend in USD.  When ``0`` or
            ``None``, no budget enforcement is performed.
    """

    #: Maximum individual records to retain in memory.  Totals
    #: remain accurate after eviction; only per-call detail is lost.
    _MAX_RECORDS: int = 10_000

    def __init__(self, max_spend_usd: float = 0.0) -> None:
        self.max_spend_usd = max_spend_usd or 0.0
        self._records: list[UsageRecord] = []
        self._total_prompt: int = 0
        self._total_completion: int = 0
        self._total_cost: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Record usage
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> UsageRecord:
        """Record token usage from a single provider call.

        Args:
            model: The model identifier (e.g. ``"claude-sonnet-4-20250514"``).
            prompt_tokens: Number of input tokens consumed.
            completion_tokens: Number of output tokens generated.

        Returns:
            The :class:`UsageRecord` that was appended.
        """
        inp_rate, out_rate = _lookup_pricing(model)
        cost = (prompt_tokens / 1000.0) * inp_rate + (completion_tokens / 1000.0) * out_rate

        rec = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )

        with self._lock:
            self._records.append(rec)
            self._total_prompt += prompt_tokens
            self._total_completion += completion_tokens
            self._total_cost += cost
            # Evict oldest records to prevent unbounded memory growth
            # in long-running sessions.  Totals remain accurate.
            if len(self._records) > self._MAX_RECORDS:
                self._records = self._records[-self._MAX_RECORDS:]

        return rec

    def record_from_response(self, response) -> UsageRecord | None:
        """Extract usage from a :class:`CompletionResponse` and record it.

        Args:
            response: A :class:`~missy.providers.base.CompletionResponse` or
                any object with ``.model`` and ``.usage`` attributes.

        Returns:
            The recorded :class:`UsageRecord`, or ``None`` if usage data
            is missing.
        """
        usage = getattr(response, "usage", None) or {}
        model = getattr(response, "model", "") or ""
        prompt = usage.get("prompt_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0) or 0

        if not model and not prompt and not completion:
            return None

        return self.record(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
        )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def check_budget(self) -> None:
        """Raise :class:`BudgetExceededError` if accumulated cost exceeds limit.

        Does nothing when ``max_spend_usd`` is ``0`` (unlimited).

        Raises:
            BudgetExceededError: When ``total_cost_usd >= max_spend_usd``.
        """
        if self.max_spend_usd <= 0:
            return
        with self._lock:
            if self._total_cost >= self.max_spend_usd:
                raise BudgetExceededError(self._total_cost, self.max_spend_usd)

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        """Accumulated cost in USD across all recorded calls."""
        with self._lock:
            return self._total_cost

    @property
    def total_prompt_tokens(self) -> int:
        """Accumulated input tokens across all recorded calls."""
        with self._lock:
            return self._total_prompt

    @property
    def total_completion_tokens(self) -> int:
        """Accumulated output tokens across all recorded calls."""
        with self._lock:
            return self._total_completion

    @property
    def total_tokens(self) -> int:
        """Accumulated total tokens (input + output)."""
        with self._lock:
            return self._total_prompt + self._total_completion

    @property
    def call_count(self) -> int:
        """Number of provider calls recorded."""
        with self._lock:
            return len(self._records)

    def get_summary(self) -> dict:
        """Return a summary dict suitable for audit logging or CLI display.

        Returns:
            A dict with keys ``total_cost_usd``, ``total_prompt_tokens``,
            ``total_completion_tokens``, ``total_tokens``, ``call_count``,
            ``max_spend_usd``, and ``budget_remaining_usd``.
        """
        with self._lock:
            remaining = (
                max(0.0, self.max_spend_usd - self._total_cost) if self.max_spend_usd > 0 else None
            )
            return {
                "total_cost_usd": round(self._total_cost, 6),
                "total_prompt_tokens": self._total_prompt,
                "total_completion_tokens": self._total_completion,
                "total_tokens": self._total_prompt + self._total_completion,
                "call_count": len(self._records),
                "max_spend_usd": self.max_spend_usd,
                "budget_remaining_usd": round(remaining, 6) if remaining is not None else None,
            }

    def reset(self) -> None:
        """Clear all recorded usage data."""
        with self._lock:
            self._records.clear()
            self._total_prompt = 0
            self._total_completion = 0
            self._total_cost = 0.0
