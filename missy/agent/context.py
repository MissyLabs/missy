"""Context window management with token budget.

Assembles conversation context within configurable token budget limits,
dropping the oldest history entries first when over budget and enriching
the system prompt with retrieved memory and learnings.

Example::

    from missy.agent.context import ContextManager, TokenBudget

    mgr = ContextManager(TokenBudget(total=20_000))
    system, messages = mgr.build_messages(
        system="You are Missy.",
        new_message="Hello",
        history=[],
    )
"""

from __future__ import annotations

from dataclasses import dataclass


def _approx_tokens(text: str) -> int:
    """Approximate token count using the 4-chars-per-token heuristic.

    Args:
        text: Input string.

    Returns:
        Estimated token count (minimum 1).
    """
    return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    """Token allocation constraints for context assembly.

    Attributes:
        total: Total token budget for the context window.
        system_reserve: Tokens reserved for the system prompt itself.
        tool_definitions_reserve: Tokens reserved for tool schema definitions.
        memory_fraction: Fraction of remaining budget allocated to injected
            memory results.
        learnings_fraction: Fraction of remaining budget allocated to past
            learnings.
    """

    total: int = 30_000
    system_reserve: int = 2_000
    tool_definitions_reserve: int = 2_000
    memory_fraction: float = 0.15
    learnings_fraction: float = 0.05


class ContextManager:
    """Assembles conversation context within token budget limits.

    Args:
        budget: Token allocation configuration.  Uses defaults when
            not provided.
    """

    def __init__(self, budget: TokenBudget | None = None) -> None:
        self._budget = budget or TokenBudget()

    def build_messages(
        self,
        system: str,
        new_message: str,
        history: list[dict],
        memory_results: list[str] | None = None,
        learnings: list[str] | None = None,
        tool_definitions: list | None = None,
    ) -> tuple[str, list[dict]]:
        """Return ``(enriched_system, messages_list)`` within token budget.

        The system prompt is enriched with retrieved memory snippets and past
        learnings.  Conversation history is pruned from the oldest end when
        the combined content would exceed the available token budget.

        Args:
            system: Base system prompt text.
            new_message: The new user message for this turn.
            history: List of past message dicts (``{"role": ..., "content":
                ...}``), ordered chronologically.
            memory_results: Optional list of relevant memory snippet strings
                to inject into the system prompt.
            learnings: Optional list of learning strings (up to 5 used) to
                append to the system prompt.
            tool_definitions: Ignored (reserved for future use; accounted for
                via ``TokenBudget.tool_definitions_reserve``).

        Returns:
            A 2-tuple of ``(enriched_system_prompt, messages_list)`` where
            *messages_list* contains only the history entries that fit within
            the budget plus the new user message.
        """
        budget = self._budget
        available = budget.total - budget.system_reserve - budget.tool_definitions_reserve

        memory_budget = int(available * budget.memory_fraction)
        learnings_budget = int(available * budget.learnings_fraction)

        enriched_system = system

        if memory_results:
            memory_text = "\n".join(memory_results)
            if _approx_tokens(memory_text) > memory_budget:
                # Truncate to character limit equivalent of memory_budget tokens
                memory_text = memory_text[: memory_budget * 4]
            enriched_system += f"\n\n## Relevant Memory\n{memory_text}"

        if learnings:
            learnings_text = "\n".join(f"- {l}" for l in learnings[:5])
            if _approx_tokens(learnings_text) <= learnings_budget:
                enriched_system += f"\n\n## Past Learnings\n{learnings_text}"

        # Prune history: keep newest messages that fit within remaining budget
        history_budget = available - memory_budget - learnings_budget
        kept: list[dict] = []
        used = _approx_tokens(new_message)

        for turn in reversed(history):
            turn_tokens = _approx_tokens(str(turn.get("content", "")))
            if used + turn_tokens > history_budget:
                break
            kept.insert(0, turn)
            used += turn_tokens

        kept.append({"role": "user", "content": new_message})
        return enriched_system, kept
