"""Sleep Mode — memory consolidation when context window fills up.

When the context window reaches a configurable threshold (default 80%),
:class:`MemoryConsolidator` summarises older conversation history, extracts
key facts, and compresses the context to free up token budget.

Internally this class now delegates to a :class:`~missy.agent.condensers.PipelineCondenser`
produced by :func:`~missy.agent.condensers.create_default_pipeline`.  The public
API (``should_consolidate``, ``consolidate``, ``extract_key_facts``,
``estimate_tokens``) is fully preserved for backward compatibility.

Example::

    from missy.agent.consolidation import MemoryConsolidator

    mc = MemoryConsolidator(threshold_pct=0.8, max_tokens=30000)
    if mc.should_consolidate(current_tokens=25000):
        messages, summary = mc.consolidate(messages, system_prompt)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.agent.condensers import PipelineCondenser
    from missy.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Number of recent messages to always keep intact.
_RECENT_KEEP = 4

# Keywords that indicate a message contains a meaningful fact or decision.
_FACT_KEYWORDS = (
    "result:",
    "decided:",
    "found:",
    "error:",
    "success:",
    "created:",
    "updated:",
    "deleted:",
    "confirmed:",
    "output:",
)


class MemoryConsolidator:
    """Consolidates conversation history when the context window is nearly full.

    Compression is performed by a :class:`~missy.agent.condensers.PipelineCondenser`
    accessible as :attr:`pipeline`.  The pipeline is built lazily on first use
    so that importing this module does not create unnecessary objects.

    Args:
        threshold_pct: Fraction of ``max_tokens`` at which consolidation
            triggers (default ``0.8``).
        max_tokens: Total token budget for the context window (default
            ``30000``).
        provider: Optional :class:`~missy.providers.base.BaseProvider` passed
            to the pipeline's :class:`~missy.agent.condensers.SummarizingCondenser`
            for LLM-powered summarisation.
        pipeline: Override the default pipeline with a custom
            :class:`~missy.agent.condensers.PipelineCondenser` instance.
    """

    def __init__(
        self,
        threshold_pct: float = 0.8,
        max_tokens: int = 30_000,
        provider: BaseProvider | None = None,
        pipeline: PipelineCondenser | None = None,
    ) -> None:
        self._threshold_pct = threshold_pct
        self._max_tokens = max_tokens
        self._provider = provider
        self._pipeline = pipeline

    def should_consolidate(self, current_tokens: int) -> bool:
        """Check whether consolidation should trigger.

        Args:
            current_tokens: Current estimated token usage.

        Returns:
            ``True`` if ``current_tokens / max_tokens >= threshold_pct``.
        """
        if self._max_tokens <= 0:
            return False
        return current_tokens / self._max_tokens >= self._threshold_pct

    @property
    def pipeline(self) -> PipelineCondenser:
        """Lazily-built default pipeline (or the custom one supplied at init)."""
        if self._pipeline is None:
            from missy.agent.condensers import create_default_pipeline

            self._pipeline = create_default_pipeline(
                provider=self._provider,
                max_tokens=self._max_tokens,
            )
        return self._pipeline

    def consolidate(
        self,
        messages: list[dict],
        system_prompt: str,
    ) -> tuple[list[dict], str]:
        """Compress older messages into a summary while preserving recent context.

        Delegates to the condenser :attr:`pipeline`, then constructs a
        human-readable summary string from the surviving summary messages for
        backward compatibility with callers that use the second return value.

        Args:
            messages: Full message history (list of dicts with ``role``
                and ``content`` keys).
            system_prompt: The current system prompt forwarded to the pipeline.

        Returns:
            A 2-tuple of ``(consolidated_messages, consolidation_summary)``
            where *consolidated_messages* is the compressed list and
            *consolidation_summary* describes the condensation that occurred.
        """
        if not messages:
            return [], ""

        if len(messages) <= _RECENT_KEEP:
            # Nothing old to consolidate.
            return list(messages), ""

        result = self.pipeline.condense(messages, system_prompt)
        condensed = result.messages

        # Build a human-readable summary from any summary blocks the pipeline
        # produced, so callers that inspect the second return value still work.
        summary_lines: list[str] = []
        for msg in condensed:
            content = str(msg.get("content", ""))
            if content.startswith(
                ("[Conversation Summary]", "[Session context consolidated]")
            ):
                summary_lines.append(content)

        if summary_lines:
            consolidation_summary = "\n---\n".join(summary_lines)
        else:
            # Pipeline dropped old messages without producing summaries —
            # generate a compact fact list from the dropped content for the log.
            old_messages = messages[: max(0, len(messages) - _RECENT_KEEP)]
            key_facts = self.extract_key_facts(old_messages)
            if key_facts:
                consolidation_summary = "\n".join(f"- {f}" for f in key_facts)
            else:
                consolidation_summary = "- (previous conversation context condensed)"

        return condensed, consolidation_summary

    def extract_key_facts(self, messages: list[dict]) -> list[str]:
        """Extract key facts and decisions from a list of messages.

        Heuristics:
        - Keep lines containing fact keywords (``result:``, ``decided:``, etc.)
        - Keep full content of ``tool`` role messages (truncated to 200 chars).
        - Keep short user messages (under 120 chars) as they often contain
          instructions or decisions.
        - Skip verbose assistant prose and long tool outputs.

        Args:
            messages: Message dicts to scan.

        Returns:
            A deduplicated list of fact strings.
        """
        facts: list[str] = []
        seen: set[str] = set()

        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            if role == "tool":
                # Tool results often contain key information.
                name = msg.get("name", "tool")
                snippet = content[:200].strip()
                if snippet:
                    fact = f"[{name}] {snippet}"
                    if fact not in seen:
                        facts.append(fact)
                        seen.add(fact)
                continue

            # Scan for keyword-bearing lines.
            for line in content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if any(kw in lower for kw in _FACT_KEYWORDS) and stripped not in seen:
                    facts.append(stripped)
                    seen.add(stripped)

            # Short user messages are often instructions.
            if role == "user" and 0 < len(content) <= 120 and content not in seen:
                facts.append(content)
                seen.add(content)

        return facts

    @staticmethod
    def estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate: total characters / 4.

        Args:
            messages: Message dicts to estimate.

        Returns:
            Estimated token count (minimum 0).
        """
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4
