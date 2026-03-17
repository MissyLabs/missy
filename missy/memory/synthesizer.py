"""Unified memory synthesizer for Missy.

Merges all memory subsystems (conversation history, learnings, summaries,
playbook entries) into a single deduplicated, relevance-ranked context block.
Inspired by CoWork-OS unified context approach.

Example::

    from missy.memory.synthesizer import MemorySynthesizer

    synth = MemorySynthesizer(max_tokens=4500)
    synth.add_fragments("conversation", ["discussed Docker setup"])
    synth.add_fragments("learnings", ["always check ports first"], base_relevance=0.7)
    block = synth.synthesize("How do I fix Docker networking?")
"""

from __future__ import annotations

from dataclasses import dataclass


def _approx_tokens(text: str) -> int:
    """Approximate token count using the 4-chars-per-token heuristic."""
    return max(1, len(text) // 4)


def _word_set(text: str) -> set[str]:
    """Return a lowercase word set from *text*."""
    return set(text.lower().split())


@dataclass
class MemoryFragment:
    """A single piece of memory from any source.

    Attributes:
        source: Origin subsystem (e.g. "conversation", "learnings").
        content: The actual text content.
        relevance: Relevance score in [0, 1].
        timestamp: ISO-format timestamp (informational, not used for ranking).
    """

    source: str
    content: str
    relevance: float = 0.5
    timestamp: str = ""


class MemorySynthesizer:
    """Merges memory fragments into a single relevance-ranked context block.

    Args:
        max_tokens: Maximum token budget for the synthesized output.
    """

    def __init__(self, max_tokens: int = 4500) -> None:
        self._max_tokens = max_tokens
        self._fragments: list[MemoryFragment] = []

    # ------------------------------------------------------------------
    # Fragment ingestion
    # ------------------------------------------------------------------

    def add_fragments(
        self,
        source: str,
        items: list[str],
        base_relevance: float = 0.5,
    ) -> None:
        """Add memory items from a named source.

        Args:
            source: Label for the originating subsystem (e.g. "conversation",
                "learnings", "playbook", "summaries").
            items: List of text strings to add as fragments.
            base_relevance: Default relevance score assigned to each item
                before query-based boosting.
        """
        for item in items:
            self._fragments.append(
                MemoryFragment(
                    source=source,
                    content=item,
                    relevance=base_relevance,
                )
            )

    # ------------------------------------------------------------------
    # Relevance scoring
    # ------------------------------------------------------------------

    def score_relevance(self, fragment: MemoryFragment, query: str) -> float:
        """Score *fragment* relevance against *query*.

        Combines base relevance with keyword overlap: the final score is
        ``base_relevance * 0.5 + overlap_ratio * 0.5`` where *overlap_ratio*
        is the number of shared words divided by the query word count.

        Args:
            fragment: The memory fragment to score.
            query: The current user query.

        Returns:
            A float in [0, 1].
        """
        query_words = _word_set(query)
        if not query_words:
            return fragment.relevance

        content_words = _word_set(fragment.content)
        shared = len(query_words & content_words)
        overlap = shared / max(len(query_words), 1)
        return fragment.relevance * 0.5 + overlap * 0.5

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        fragments: list[MemoryFragment],
        threshold: float = 0.8,
    ) -> list[MemoryFragment]:
        """Remove near-duplicate fragments.

        Two fragments are considered duplicates when their word-level overlap
        exceeds *threshold*.  When a duplicate pair is found, the fragment
        with the higher relevance score is kept.

        Args:
            fragments: Input fragment list (not mutated).
            threshold: Overlap ratio above which two fragments are duplicates.

        Returns:
            A deduplicated list of fragments.
        """
        kept: list[MemoryFragment] = []
        for frag in fragments:
            frag_words = _word_set(frag.content)
            is_dup = False
            for existing in kept:
                existing_words = _word_set(existing.content)
                total = len(frag_words | existing_words)
                if total == 0:
                    is_dup = True
                    break
                overlap = len(frag_words & existing_words) / total
                if overlap >= threshold:
                    # Keep the higher-relevance fragment.
                    if frag.relevance > existing.relevance:
                        kept.remove(existing)
                        kept.append(frag)
                    is_dup = True
                    break
            if not is_dup:
                kept.append(frag)
        return kept

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, query: str) -> str:
        """Produce a single context block from all stored fragments.

        Steps:
        1. Score each fragment against *query*.
        2. Deduplicate near-identical fragments.
        3. Sort by relevance (descending).
        4. Truncate to fit within ``max_tokens``.
        5. Format with source labels.

        Args:
            query: The current user query for relevance scoring.

        Returns:
            A formatted string ready for injection into the system prompt,
            or an empty string when no fragments are available.
        """
        if not self._fragments:
            return ""

        # Score relevance
        scored: list[MemoryFragment] = []
        for frag in self._fragments:
            score = self.score_relevance(frag, query)
            scored.append(
                MemoryFragment(
                    source=frag.source,
                    content=frag.content,
                    relevance=score,
                    timestamp=frag.timestamp,
                )
            )

        # Deduplicate
        unique = self.deduplicate(scored)

        # Sort by relevance descending
        unique.sort(key=lambda f: f.relevance, reverse=True)

        # Build output within token budget
        lines: list[str] = []
        used_tokens = 0
        for frag in unique:
            line = f"[{frag.source}] {frag.content}"
            line_tokens = _approx_tokens(line)
            if used_tokens + line_tokens > self._max_tokens:
                break
            lines.append(line)
            used_tokens += line_tokens

        return "\n".join(lines)
