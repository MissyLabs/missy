"""Memory Condenser Pipeline for Missy.

Provides a composable pipeline of chained condensers that progressively compress
conversation history to free up context window space.  Inspired by the OpenHands
memory condensation architecture.

The default pipeline runs four stages in order:

1. :class:`ObservationMaskingCondenser` — shrinks large tool outputs in-place.
2. :class:`AmortizedForgettingCondenser` — drops low-importance older messages.
3. :class:`SummarizingCondenser` — LLM-summarises remaining old messages (when a
   provider is available; falls back to keyword extraction otherwise).
4. :class:`WindowCondenser` — hard cap on total message count as a safety net.

Example::

    from missy.agent.condensers import create_default_pipeline

    pipeline = create_default_pipeline(provider=my_provider, max_tokens=30_000)
    result = pipeline.condense(messages, system_prompt="You are Missy.")
    compressed = result.messages
    print(result.metadata)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Keywords that indicate a message contains a meaningful fact or outcome.
_IMPORTANCE_KEYWORDS = frozenset(
    [
        "error",
        "success",
        "result",
        "failed",
        "created",
        "updated",
        "deleted",
        "confirmed",
        "found",
        "output",
    ]
)

# Characters shown in masked tool output previews.
_MASK_PREVIEW_CHARS = 200


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CondenserResult:
    """Result of a single condensation step.

    Attributes:
        messages: The condensed message list after this step.
        metadata: Statistics and diagnostic information about what was done.
    """

    messages: list[dict]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseCondenser(ABC):
    """Abstract base for all message condensers.

    All subclasses must implement :meth:`condense`.  Condensers are stateless
    by design so that they can be composed freely in a :class:`PipelineCondenser`.
    """

    @abstractmethod
    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Condense *messages*, returning a :class:`CondenserResult`.

        Args:
            messages: The current message list (not mutated).
            system_prompt: The active system prompt (available for context but
                not required by all condensers).

        Returns:
            A :class:`CondenserResult` with the compressed message list and
            metadata about what was done.
        """
        ...


# ---------------------------------------------------------------------------
# WindowCondenser
# ---------------------------------------------------------------------------


class WindowCondenser(BaseCondenser):
    """Keep only the last *max_messages* messages, discarding earlier ones.

    This is the simplest and fastest condenser and works well as a hard cap at
    the end of a pipeline.

    Args:
        max_messages: Maximum number of messages to retain (default ``20``).
    """

    def __init__(self, max_messages: int = 20) -> None:
        self.max_messages = max_messages

    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Drop messages beyond the window, keeping the most recent.

        Args:
            messages: Input message list.
            system_prompt: Ignored.

        Returns:
            :class:`CondenserResult` with at most *max_messages* messages.
            Metadata key ``dropped`` holds the number of messages removed.
        """
        if len(messages) <= self.max_messages:
            return CondenserResult(messages=list(messages), metadata={"dropped": 0})

        dropped = len(messages) - self.max_messages
        kept = messages[-self.max_messages :]
        return CondenserResult(
            messages=list(kept),
            metadata={"dropped": dropped, "kept": self.max_messages},
        )


# ---------------------------------------------------------------------------
# ObservationMaskingCondenser
# ---------------------------------------------------------------------------


class ObservationMaskingCondenser(BaseCondenser):
    """Replace oversized tool outputs with concise placeholder summaries.

    Tool role messages that exceed *max_output_chars* have their content
    replaced with a one-line mask that includes the character count and a short
    preview.  All other messages are passed through unchanged.

    Args:
        max_output_chars: Character threshold above which content is masked
            (default ``2000``).
    """

    def __init__(self, max_output_chars: int = 2000) -> None:
        self.max_output_chars = max_output_chars

    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Mask tool outputs that exceed the character limit.

        Args:
            messages: Input message list.
            system_prompt: Ignored.

        Returns:
            :class:`CondenserResult` where oversized tool messages have their
            content replaced.  Metadata key ``masked`` counts the number of
            messages that were masked.
        """
        result: list[dict] = []
        masked_count = 0

        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            if role == "tool" and len(content) > self.max_output_chars:
                preview = content[:_MASK_PREVIEW_CHARS].strip()
                replacement = f"[Tool output masked: {len(content)} chars, preview: {preview}...]"
                # Shallow-copy the message dict, replacing only content.
                result.append({**msg, "content": replacement})
                masked_count += 1
            else:
                result.append(dict(msg))

        return CondenserResult(
            messages=result,
            metadata={"masked": masked_count},
        )


# ---------------------------------------------------------------------------
# AmortizedForgettingCondenser
# ---------------------------------------------------------------------------


class AmortizedForgettingCondenser(BaseCondenser):
    """Gradually forget low-importance messages using exponential score decay.

    Each message receives an importance score starting at ``1.0`` for the most
    recent and decreasing by *decay_rate* per position toward the oldest.
    Messages with importance below *forget_threshold* are dropped.

    Bonuses applied before threshold check:

    - Tool result messages: ``+0.2``
    - Messages containing importance keywords (``error``, ``success``, etc.): ``+0.1``

    The first message and the last *always_keep* messages are always preserved
    regardless of score.

    Args:
        forget_threshold: Messages with score below this are dropped (default
            ``0.3``).
        decay_rate: Score reduction per position from the end (default ``0.05``).
        always_keep: Number of tail messages always preserved (default ``4``).
    """

    def __init__(
        self,
        forget_threshold: float = 0.3,
        decay_rate: float = 0.05,
        always_keep: int = 4,
    ) -> None:
        self.forget_threshold = forget_threshold
        self.decay_rate = decay_rate
        self.always_keep = always_keep

    def _score(self, msg: dict, position_from_end: int) -> float:
        """Compute the importance score for a single message.

        Args:
            msg: The message dict.
            position_from_end: 0 = most recent, increasing toward oldest.

        Returns:
            Importance score as a float in roughly ``[0, 1.3]``.
        """
        score = max(0.0, 1.0 - self.decay_rate * position_from_end)

        role = msg.get("role", "")
        content = str(msg.get("content", "")).lower()

        if role == "tool":
            score += 0.2

        if any(kw in content for kw in _IMPORTANCE_KEYWORDS):
            score += 0.1

        return score

    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Drop messages with low importance scores.

        Args:
            messages: Input message list.
            system_prompt: Ignored.

        Returns:
            :class:`CondenserResult` with low-scoring old messages removed.
            Metadata keys: ``dropped``, ``kept``.
        """
        if not messages:
            return CondenserResult(messages=[], metadata={"dropped": 0, "kept": 0})

        n = len(messages)
        tail_n = min(self.always_keep, n)

        # Always-keep indices: first message and last always_keep messages.
        always_keep_indices: set[int] = {0}
        for i in range(n - tail_n, n):
            always_keep_indices.add(i)

        kept: list[dict] = []
        dropped = 0

        for idx, msg in enumerate(messages):
            if idx in always_keep_indices:
                kept.append(dict(msg))
                continue

            # position_from_end: 0 means last message.
            pos = (n - 1) - idx
            score = self._score(msg, pos)

            if score >= self.forget_threshold:
                kept.append(dict(msg))
            else:
                dropped += 1

        return CondenserResult(
            messages=kept,
            metadata={"dropped": dropped, "kept": len(kept)},
        )


# ---------------------------------------------------------------------------
# SummarizingCondenser
# ---------------------------------------------------------------------------


class SummarizingCondenser(BaseCondenser):
    """Summarise older messages using an LLM provider.

    The most recent *preserve_recent* messages are kept verbatim.  Older
    messages are chunked into groups of *chunk_size* and each chunk is
    summarised by the provider into a single ``"user"`` role message prefixed
    with ``[Conversation Summary]``.

    If no provider is given (or the provider call fails), the condenser falls
    back to simple keyword extraction from the old messages, matching the
    behaviour of the original :class:`~missy.agent.consolidation.MemoryConsolidator`.

    Args:
        provider: A :class:`~missy.providers.base.BaseProvider` instance used
            for summarisation.  ``None`` activates keyword-extraction fallback.
        model: Optional model name override passed as ``model=`` kwarg to the
            provider's ``complete`` call.
        preserve_recent: Number of most-recent messages to keep intact (default
            ``6``).
        chunk_size: Number of old messages per summarisation chunk (default
            ``10``).
    """

    _SUMMARISE_SYSTEM = (
        "You are a concise summariser.  Given a block of conversation history, "
        "produce a compact factual summary in 3-6 bullet points.  Focus on "
        "decisions made, facts discovered, errors encountered, and tasks "
        "completed.  Omit pleasantries and verbose prose."
    )

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

    def __init__(
        self,
        provider: BaseProvider | None = None,
        model: str | None = None,
        preserve_recent: int = 6,
        chunk_size: int = 10,
    ) -> None:
        self.provider = provider
        self.model = model
        self.preserve_recent = preserve_recent
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_chunk(self, messages: list[dict]) -> str:
        """Render a message chunk as a plain text block for the LLM."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "?")
            content = str(msg.get("content", ""))
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _summarise_via_llm(self, chunk: list[dict]) -> str | None:
        """Call the provider to summarise *chunk*; return None on failure."""
        if self.provider is None:
            return None
        try:
            from missy.providers.base import Message

            text = self._format_chunk(chunk)
            prompt = f"Summarise this conversation segment:\n\n{text}"
            kwargs: dict = {}
            if self.model:
                kwargs["model"] = self.model
            response = self.provider.complete(
                [Message(role="user", content=prompt)],
                system=self._SUMMARISE_SYSTEM,
                **kwargs,
            )
            return response.content.strip()
        except Exception:
            logger.warning("SummarizingCondenser: LLM summarisation failed, using fallback")
            return None

    def _keyword_fallback(self, messages: list[dict]) -> str:
        """Extract key facts from *messages* when the LLM is unavailable."""
        facts: list[str] = []
        seen: set[str] = set()

        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            if role == "tool":
                name = msg.get("name", "tool")
                snippet = content[:200].strip()
                if snippet:
                    fact = f"[{name}] {snippet}"
                    if fact not in seen:
                        facts.append(fact)
                        seen.add(fact)
                continue

            for line in content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if any(kw in lower for kw in self._FACT_KEYWORDS) and stripped not in seen:
                    facts.append(stripped)
                    seen.add(stripped)

            if role == "user" and 0 < len(content) <= 120 and content not in seen:
                facts.append(content)
                seen.add(content)

        if facts:
            return "\n".join(f"- {f}" for f in facts)
        return "- (previous conversation context — no key facts extracted)"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Summarise older messages, keeping the most recent intact.

        Args:
            messages: Input message list.
            system_prompt: Ignored.

        Returns:
            :class:`CondenserResult` where old message chunks are replaced by
            summary messages.  Metadata keys: ``chunks_summarised``,
            ``messages_summarised``, ``used_llm``.
        """
        if not messages:
            return CondenserResult(messages=[], metadata={"chunks_summarised": 0})

        tail_n = min(self.preserve_recent, len(messages))
        old = messages[:-tail_n] if tail_n > 0 else list(messages)
        recent = list(messages[-tail_n:]) if tail_n > 0 else []

        if not old:
            return CondenserResult(
                messages=list(messages),
                metadata={"chunks_summarised": 0, "messages_summarised": 0},
            )

        # Break old messages into chunks and summarise each.
        summary_messages: list[dict] = []
        chunks_summarised = 0
        used_llm = False

        for start in range(0, len(old), self.chunk_size):
            chunk = old[start : start + self.chunk_size]
            llm_summary = self._summarise_via_llm(chunk)
            if llm_summary is not None:
                used_llm = True
                summary_text = llm_summary
            else:
                summary_text = self._keyword_fallback(chunk)

            summary_messages.append(
                {
                    "role": "user",
                    "content": f"[Conversation Summary]\n{summary_text}",
                }
            )
            chunks_summarised += 1

        return CondenserResult(
            messages=summary_messages + recent,
            metadata={
                "chunks_summarised": chunks_summarised,
                "messages_summarised": len(old),
                "used_llm": used_llm,
            },
        )


# ---------------------------------------------------------------------------
# LLMAttentionCondenser
# ---------------------------------------------------------------------------


class LLMAttentionCondenser(BaseCondenser):
    """Ask an LLM to select the most important messages to retain.

    The condenser serialises the message list and asks the provider to return a
    JSON array of integer indices corresponding to the messages that matter
    most.  The last *preserve_recent* messages are always kept regardless of
    the LLM's selection.

    If the LLM call fails or its response cannot be parsed, the condenser falls
    back to keeping every other message plus the last *preserve_recent*.

    Args:
        provider: :class:`~missy.providers.base.BaseProvider` used to rank
            messages.  ``None`` activates the fallback immediately.
        model: Optional model name override.
        preserve_recent: Number of tail messages always preserved (default
            ``4``).
    """

    _ATTENTION_SYSTEM = (
        "You are a memory manager for an AI assistant.  Given a numbered list "
        "of conversation messages, return ONLY a JSON array of integers — the "
        "0-based indices of the messages that are most important to keep for "
        "future context.  Choose at most half of the messages.  Example: [0,3,5]"
    )

    def __init__(
        self,
        provider: BaseProvider | None = None,
        model: str | None = None,
        preserve_recent: int = 4,
    ) -> None:
        self.provider = provider
        self.model = model
        self.preserve_recent = preserve_recent

    def _parse_indices(self, text: str, max_index: int) -> list[int] | None:
        """Extract a list of valid indices from the LLM's JSON response."""
        import json
        import re

        # Find the first JSON array in the response.
        match = re.search(r"\[[\d,\s]*\]", text)
        if not match:
            return None
        try:
            indices = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if not isinstance(indices, list):
            return None
        valid = [i for i in indices if isinstance(i, int) and 0 <= i <= max_index]
        return valid if valid else None

    def _ask_llm(self, messages: list[dict]) -> list[int] | None:
        """Return LLM-selected indices, or None on failure."""
        if self.provider is None:
            return None
        try:
            from missy.providers.base import Message

            numbered = "\n".join(
                f"{i}: [{msg.get('role', '?')}] {str(msg.get('content', ''))[:120]}"
                for i, msg in enumerate(messages)
            )
            prompt = f"Messages:\n{numbered}\n\nWhich indices should be kept?"
            kwargs: dict = {}
            if self.model:
                kwargs["model"] = self.model
            response = self.provider.complete(
                [Message(role="user", content=prompt)],
                system=self._ATTENTION_SYSTEM,
                **kwargs,
            )
            return self._parse_indices(response.content, len(messages) - 1)
        except Exception:
            logger.warning("LLMAttentionCondenser: provider call failed, using fallback")
            return None

    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Keep LLM-selected messages plus the most recent tail.

        Args:
            messages: Input message list.
            system_prompt: Ignored.

        Returns:
            :class:`CondenserResult` with selected messages retained.  Metadata
            keys: ``used_llm``, ``kept``, ``dropped``.
        """
        if not messages:
            return CondenserResult(
                messages=[], metadata={"used_llm": False, "kept": 0, "dropped": 0}
            )

        n = len(messages)
        tail_n = min(self.preserve_recent, n)
        tail_indices: set[int] = set(range(n - tail_n, n))

        body = messages[: n - tail_n]
        llm_indices = self._ask_llm(body) if body else []
        used_llm = llm_indices is not None and bool(body)

        if llm_indices is None:
            # Fallback: keep every other message from the body.
            fallback_indices = set(range(0, len(body), 2))
            selected_body = [body[i] for i in sorted(fallback_indices)]
        else:
            selected_body = [body[i] for i in sorted(set(llm_indices))]

        tail = [messages[i] for i in sorted(tail_indices)]
        kept = selected_body + tail
        dropped = n - len(kept)

        return CondenserResult(
            messages=[dict(m) for m in kept],
            metadata={"used_llm": used_llm, "kept": len(kept), "dropped": dropped},
        )


# ---------------------------------------------------------------------------
# PipelineCondenser
# ---------------------------------------------------------------------------


class PipelineCondenser(BaseCondenser):
    """Chain multiple condensers in sequence.

    Each step receives the output messages of the previous step.  Metadata from
    all steps is collected under the step's class name.

    Args:
        steps: Ordered list of :class:`BaseCondenser` instances to apply.
    """

    def __init__(self, steps: list[BaseCondenser]) -> None:
        self.steps = steps

    def condense(self, messages: list[dict], system_prompt: str = "") -> CondenserResult:
        """Run all pipeline steps in order.

        Args:
            messages: Initial message list.
            system_prompt: Forwarded to every step unchanged.

        Returns:
            :class:`CondenserResult` with the final compressed messages and
            per-step metadata keyed by class name.
        """
        result = CondenserResult(messages=list(messages), metadata={})
        for step in self.steps:
            step_result = step.condense(result.messages, system_prompt)
            result.messages = step_result.messages
            result.metadata[step.__class__.__name__] = step_result.metadata
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_default_pipeline(
    provider: BaseProvider | None = None,
    model: str | None = None,
    max_tokens: int = 30_000,
) -> PipelineCondenser:
    """Create the default four-stage condenser pipeline.

    Pipeline stages:

    1. :class:`ObservationMaskingCondenser` — replace large tool outputs with
       one-line placeholders.
    2. :class:`AmortizedForgettingCondenser` — drop low-importance old messages
       using exponential score decay.
    3. :class:`SummarizingCondenser` — LLM-summarise remaining old messages
       (keyword fallback when no provider is supplied).
    4. :class:`WindowCondenser` — hard cap to ensure the message list never
       exceeds a safe size regardless of earlier stages.

    The window size in stage 4 is derived from *max_tokens*: one message is
    assumed to cost roughly 150 tokens on average, giving a cap of
    ``max_tokens // 150`` (minimum 10).

    Args:
        provider: Optional provider used by :class:`SummarizingCondenser`.
        model: Optional model name override for LLM steps.
        max_tokens: Total token budget; influences the window cap.

    Returns:
        A :class:`PipelineCondenser` ready for use.
    """
    window_cap = max(10, max_tokens // 150)

    steps: list[BaseCondenser] = [
        ObservationMaskingCondenser(),
        AmortizedForgettingCondenser(),
        SummarizingCondenser(provider=provider, model=model),
        WindowCondenser(max_messages=window_cap),
    ]
    return PipelineCondenser(steps=steps)
