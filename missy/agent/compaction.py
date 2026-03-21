"""Compaction engine — manages summarization of conversation history.

Runs leaf passes (turns → depth-0 summaries) and condensation passes
(depth-N summaries → depth-N+1 summaries) to keep active context within
token budget limits while preserving all original messages in the DB.

Usage::

    from missy.agent.compaction import compact_session, compact_if_needed

    compact_session("sess-1", memory_store, summarizer, fresh_tail_count=16)
    compact_if_needed("sess-1", memory_store, summarizer, budget)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.agent.context import TokenBudget
    from missy.agent.summarizer import Summarizer
    from missy.memory.sqlite_store import SQLiteMemoryStore

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_LEAF_CHUNK_TOKENS = 20_000
_DEFAULT_CONDENSED_MIN_FANOUT = 4
_DEFAULT_CONTEXT_THRESHOLD = 0.75
_DEFAULT_FRESH_TAIL = 16


def compact_session(
    session_id: str,
    memory_store: SQLiteMemoryStore,
    summarizer: Summarizer,
    *,
    fresh_tail_count: int = _DEFAULT_FRESH_TAIL,
    leaf_chunk_tokens: int = _DEFAULT_LEAF_CHUNK_TOKENS,
    condensed_min_fanout: int = _DEFAULT_CONDENSED_MIN_FANOUT,
    max_condense_depth: int = -1,
) -> dict:
    """Run a full compaction pass on a session.

    1. Leaf pass: summarize raw turns outside the fresh tail.
    2. Condensation pass: merge same-depth summaries when fanout is met.

    Args:
        session_id: Session to compact.
        memory_store: The SQLite memory store.
        summarizer: A configured Summarizer instance.
        fresh_tail_count: Number of recent turns protected from compaction.
        leaf_chunk_tokens: Max source tokens per leaf summary chunk.
        condensed_min_fanout: Minimum summaries before condensation triggers.
        max_condense_depth: Maximum condensation depth (-1 = unlimited).

    Returns:
        Dict with stats: ``leaf_summaries_created``, ``condensed_summaries_created``,
        ``turns_compacted``, ``tiers_used``.
    """
    from missy.memory.sqlite_store import SummaryRecord

    stats: dict = {
        "leaf_summaries_created": 0,
        "condensed_summaries_created": 0,
        "turns_compacted": 0,
        "tiers_used": [],
    }

    # --- Leaf pass ---
    turns = memory_store.get_session_turns(session_id, limit=10_000)
    if len(turns) <= fresh_tail_count:
        logger.debug("Session %s has %d turns (≤ fresh tail), skipping", session_id, len(turns))
        return stats

    # Identify turns already covered by a leaf summary.
    existing_leaf_turn_ids: set[str] = set()
    for s in memory_store.get_summaries(session_id, depth=0, limit=10_000):
        existing_leaf_turn_ids.update(s.source_turn_ids)

    # Evictable turns = all except fresh tail, not yet summarized.
    evictable = turns[:-fresh_tail_count]
    unsummarized = [t for t in evictable if t.id not in existing_leaf_turn_ids]

    if unsummarized:
        # Get most recent existing summary for continuity.
        prior_summaries = memory_store.get_summaries(session_id, depth=0, limit=1)
        prior_summary = prior_summaries[-1].content if prior_summaries else ""

        # Chunk into groups of ~leaf_chunk_tokens.
        chunks = _chunk_turns(unsummarized, leaf_chunk_tokens)

        for chunk in chunks:
            summary_text, tier = summarizer.summarize_turns(
                chunk,
                prior_summary=prior_summary,
            )
            time_start = chunk[0].timestamp if chunk else None
            time_end = chunk[-1].timestamp if chunk else None

            record = SummaryRecord.new(
                session_id=session_id,
                depth=0,
                content=summary_text,
                source_turn_ids=[t.id for t in chunk],
                time_range_start=time_start,
                time_range_end=time_end,
                descendant_count=len(chunk),
            )
            memory_store.add_summary(record)

            stats["leaf_summaries_created"] += 1
            stats["turns_compacted"] += len(chunk)
            stats["tiers_used"].append(tier)
            prior_summary = summary_text

    # --- Condensation pass ---
    depth = 0
    max_depth = max_condense_depth if max_condense_depth >= 0 else 100
    while depth <= max_depth:
        uncompacted = memory_store.get_uncompacted_summaries(session_id, depth)
        if len(uncompacted) < condensed_min_fanout:
            break

        summary_text, tier = summarizer.summarize_summaries(uncompacted)
        time_start = uncompacted[0].time_range_start
        time_end = uncompacted[-1].time_range_end
        total_descendants = sum(s.descendant_count for s in uncompacted)

        parent = SummaryRecord.new(
            session_id=session_id,
            depth=depth + 1,
            content=summary_text,
            source_summary_ids=[s.id for s in uncompacted],
            time_range_start=time_start,
            time_range_end=time_end,
            descendant_count=total_descendants,
        )
        memory_store.add_summary(parent)
        memory_store.mark_summary_compacted([s.id for s in uncompacted], parent.id)

        stats["condensed_summaries_created"] += 1
        stats["tiers_used"].append(tier)
        depth += 1

    logger.info(
        "Compaction for session %s: %d leaf, %d condensed, %d turns compacted",
        session_id,
        stats["leaf_summaries_created"],
        stats["condensed_summaries_created"],
        stats["turns_compacted"],
    )
    return stats


def should_compact(
    session_id: str,
    memory_store: SQLiteMemoryStore,
    token_budget: int,
    threshold: float = _DEFAULT_CONTEXT_THRESHOLD,
) -> bool:
    """Return True if the session's token count exceeds the threshold."""
    total_tokens = memory_store.get_session_token_count(session_id)
    limit = int(token_budget * threshold)
    return total_tokens > limit


def compact_if_needed(
    session_id: str,
    memory_store: SQLiteMemoryStore,
    summarizer: Summarizer,
    budget: TokenBudget,
) -> dict | None:
    """Compact a session only if it exceeds the context threshold.

    Returns:
        Compaction stats dict if compaction ran, None otherwise.
    """
    threshold = getattr(budget, "context_threshold", _DEFAULT_CONTEXT_THRESHOLD)
    if not should_compact(session_id, memory_store, budget.total, threshold):
        return None

    fresh_tail = getattr(budget, "fresh_tail_count", _DEFAULT_FRESH_TAIL)
    leaf_chunk = getattr(budget, "leaf_chunk_tokens", _DEFAULT_LEAF_CHUNK_TOKENS)
    fanout = getattr(budget, "condensed_min_fanout", _DEFAULT_CONDENSED_MIN_FANOUT)

    return compact_session(
        session_id,
        memory_store,
        summarizer,
        fresh_tail_count=fresh_tail,
        leaf_chunk_tokens=leaf_chunk,
        condensed_min_fanout=fanout,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _chunk_turns(turns: list, max_tokens: int) -> list[list]:
    """Split turns into chunks of approximately max_tokens each."""
    chunks: list[list] = []
    current: list = []
    current_tokens = 0

    for turn in turns:
        turn_tokens = max(1, len(turn.content) // 4)
        if current and current_tokens + turn_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(turn)
        current_tokens += turn_tokens

    if current:
        chunks.append(current)
    return chunks
