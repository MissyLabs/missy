"""DAG-based conversation summarization engine.

Compresses conversation history into hierarchical summaries using the
configured LLM provider.  Supports three escalation tiers to guarantee
progress even when the model misbehaves.

Usage::

    from missy.agent.summarizer import Summarizer
    s = Summarizer(provider=my_provider)
    text, tier = s.summarize_turns(turns)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from missy.memory.sqlite_store import ConversationTurn, SummaryRecord

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_LEAF_PROMPT = """\
Summarize the following conversation excerpt. Preserve:
- Key decisions and conclusions
- Action items and next steps
- Code references, file paths, and commands
- Error messages and their resolutions
- Important facts, names, and numbers

Be concise but retain all actionable detail. Do not add commentary.

{prior_context}

--- CONVERSATION ---
{transcript}
--- END ---

Summary:"""

_CONDENSED_PROMPT = """\
Condense the following summaries into a single, higher-level summary.
Preserve cross-summary themes, key decisions, and actionable details.
Remove redundancy while keeping all unique information.

{summaries_text}

Condensed summary:"""

_AGGRESSIVE_PROMPT = """\
Compress the following text into fewer than {target_tokens} tokens.
Keep only the most critical facts, decisions, and references.

{text}

Compressed:"""


class Summarizer:
    """Summarizes conversation chunks using an LLM provider.

    Args:
        provider: A configured provider instance with a ``chat()`` or
            ``complete()`` method.
        timeout: HTTP timeout in seconds for LLM calls.
    """

    def __init__(self, provider: Any, *, timeout: int = 120) -> None:
        self._provider = provider
        self._timeout = timeout
        self.tier_counts = {"normal": 0, "aggressive": 0, "fallback": 0}

    def summarize_turns(
        self,
        turns: list[ConversationTurn],
        prior_summary: str = "",
        *,
        target_tokens: int = 1_200,
    ) -> tuple[str, str]:
        """Summarize a list of conversation turns.

        Returns:
            A ``(summary_text, tier_used)`` tuple.
        """
        transcript = self._format_turns(turns)
        input_tokens = _approx_tokens(transcript)

        prior_context = ""
        if prior_summary:
            prior_context = f"For continuity, here is the previous summary:\n{prior_summary}\n"

        prompt = _LEAF_PROMPT.format(prior_context=prior_context, transcript=transcript)
        return self._escalate(prompt, input_tokens, target_tokens)

    def summarize_summaries(
        self,
        summaries: list[SummaryRecord],
        *,
        target_tokens: int = 2_000,
    ) -> tuple[str, str]:
        """Condense multiple summaries into a higher-level summary.

        Returns:
            A ``(summary_text, tier_used)`` tuple.
        """
        parts = []
        for i, s in enumerate(summaries, 1):
            time_info = ""
            if s.time_range_start and s.time_range_end:
                time_info = f" ({s.time_range_start} to {s.time_range_end})"
            parts.append(f"### Summary {i} [depth={s.depth}]{time_info}\n{s.content}")
        summaries_text = "\n\n".join(parts)
        input_tokens = _approx_tokens(summaries_text)

        prompt = _CONDENSED_PROMPT.format(summaries_text=summaries_text)
        return self._escalate(prompt, input_tokens, target_tokens)

    # ------------------------------------------------------------------
    # Three-tier escalation
    # ------------------------------------------------------------------

    def _escalate(self, prompt: str, input_tokens: int, target_tokens: int) -> tuple[str, str]:
        """Try normal → aggressive → fallback summarization."""
        # Tier 1: Normal
        try:
            result = self._call_llm(prompt, temperature=0.2)
            result_tokens = _approx_tokens(result)
            if result and result_tokens <= input_tokens:
                self.tier_counts["normal"] += 1
                return result.strip(), "normal"
            logger.debug(
                "Tier 1 output (%d tokens) >= input (%d tokens), escalating",
                result_tokens,
                input_tokens,
            )
        except Exception:
            logger.warning("Tier 1 summarization failed", exc_info=True)

        # Tier 2: Aggressive
        try:
            aggressive_prompt = _AGGRESSIVE_PROMPT.format(
                target_tokens=target_tokens,
                text=prompt,
            )
            result = self._call_llm(aggressive_prompt, temperature=0.1)
            if result:
                self.tier_counts["aggressive"] += 1
                return result.strip(), "aggressive"
        except Exception:
            logger.warning("Tier 2 summarization failed", exc_info=True)

        # Tier 3: Deterministic fallback
        logger.warning("Falling back to deterministic truncation")
        self.tier_counts["fallback"] += 1
        truncated = prompt[: target_tokens * 4]  # ~target_tokens tokens
        return truncated.rstrip() + "\n[TRUNCATED — summarization failed]", "fallback"

    def _call_llm(self, prompt: str, temperature: float = 0.2) -> str:
        """Send a prompt to the provider and return the text response."""
        from missy.providers.base import Message

        messages = [Message(role="user", content=prompt)]
        start = time.monotonic()
        response = self._provider.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.debug("LLM summarization took %d ms", elapsed_ms)
        return response.content or ""

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_turns(turns: list[ConversationTurn]) -> str:
        """Format turns into a timestamped transcript."""
        lines = []
        for t in turns:
            ts = t.timestamp[:19] if t.timestamp else "?"
            lines.append(f"[{ts}] {t.role}: {t.content}")
        return "\n".join(lines)
