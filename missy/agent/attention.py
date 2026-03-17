"""Brain-inspired attention system for the Missy agent.

Five subsystems that track focus, detect urgency, and guide how the agent
prioritises tool calls and context retrieval.  Inspired by Zuckerman's
attention framework.

Example::

    from missy.agent.attention import AttentionSystem

    attn = AttentionSystem()
    state = attn.process("The server is down! Fix it immediately!")
    print(state.urgency)       # high
    print(state.topics)        # ["server"]
    print(state.priority_tools)  # ["shell_exec", "file_read"]
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttentionState:
    """Result of the attention pipeline for a single input.

    Attributes:
        urgency: Urgency score in [0.0, 1.0].
        topics: Extracted focus topics / entities.
        focus_duration: How many consecutive turns the same topic has
            persisted (starts at 1).
        priority_tools: Tools that should be prioritised given the
            current attention state.
        context_filter: Topic words that can be used to filter memory
            fragments for selective retrieval.
    """

    urgency: float = 0.0
    topics: list[str] = field(default_factory=list)
    focus_duration: int = 1
    priority_tools: list[str] = field(default_factory=list)
    context_filter: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------
# Individual attention subsystems
# -----------------------------------------------------------------------

_URGENCY_KEYWORDS: set[str] = {
    "error",
    "critical",
    "urgent",
    "broken",
    "down",
    "failed",
    "security",
    "immediately",
    "asap",
    "emergency",
}

_TOPIC_PREPOSITIONS: set[str] = {"about", "with", "for", "the"}

_FILE_TOPIC_WORDS: set[str] = {
    "file",
    "files",
    "directory",
    "folder",
    "path",
    "read",
    "write",
    "edit",
    "config",
    "log",
    "logs",
}


class AlertingAttention:
    """Detect urgent signals in input text."""

    def score(self, text: str) -> float:
        """Return an urgency score in [0.0, 1.0].

        Score is the fraction of input words matching urgency keywords,
        capped at 1.0.
        """
        words = text.lower().split()
        if not words:
            return 0.0
        matched = sum(1 for w in words if w.strip("!.,?;:") in _URGENCY_KEYWORDS)
        return min(matched / len(words), 1.0)


class OrientingAttention:
    """Identify what the user is focused on."""

    def extract_topics(self, text: str) -> list[str]:
        """Extract topic words from *text*.

        Heuristic: capitalised words (not at sentence start) and words
        that follow common topic prepositions.

        Returns:
            A deduplicated list of topic strings.
        """
        words = text.split()
        topics: list[str] = []

        for i, word in enumerate(words):
            clean = word.strip("!.,?;:\"'()")
            if not clean:
                continue
            # Capitalised word not at sentence start
            if clean[0].isupper() and i > 0 and not words[i - 1].endswith((".", "!", "?")):
                topics.append(clean)
            # Word after a topic preposition
            if i > 0:
                prev = words[i - 1].lower().strip("!.,?;:\"'()")
                if prev in _TOPIC_PREPOSITIONS:
                    topics.append(clean)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in topics:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique


class SustainedAttention:
    """Track topic continuity across turns."""

    def __init__(self) -> None:
        self._previous_topics: list[str] = []
        self._duration: int = 0

    def update(self, topics: list[str]) -> int:
        """Update with new topics and return the current focus duration.

        If more than 50 % of previous topics overlap with the current set,
        the duration is incremented; otherwise it resets to 1.
        """
        if not topics:
            self._previous_topics = []
            self._duration = 1
            return self._duration

        if self._previous_topics:
            prev_set = {t.lower() for t in self._previous_topics}
            curr_set = {t.lower() for t in topics}
            overlap = len(prev_set & curr_set) / max(len(prev_set), 1)
            if overlap > 0.5:
                self._duration += 1
            else:
                self._duration = 1
        else:
            self._duration = 1

        self._previous_topics = list(topics)
        return self._duration


class SelectiveAttention:
    """Filter memory fragments to only the relevant subset."""

    @staticmethod
    def filter(fragments: list[str], topics: list[str]) -> list[str]:
        """Return fragments where any topic word appears in the content.

        Args:
            fragments: List of text fragments (e.g. memory snippets).
            topics: Current focus topics.

        Returns:
            Filtered list of relevant fragments.
        """
        if not topics:
            return list(fragments)
        topic_lower = {t.lower() for t in topics}
        return [f for f in fragments if any(t in f.lower() for t in topic_lower)]


class ExecutiveAttention:
    """Decide tool priority based on attention signals."""

    @staticmethod
    def prioritise(urgency: float, topics: list[str]) -> list[str]:
        """Suggest priority tools.

        - urgency > 0.5 → prioritise ``shell_exec``, ``file_read``.
        - topics contain file-related words → prioritise file tools.
        - Otherwise return an empty list (no priority override).
        """
        priority: list[str] = []
        if urgency > 0.5:
            priority = ["shell_exec", "file_read"]

        topic_lower = {t.lower() for t in topics}
        if topic_lower & _FILE_TOPIC_WORDS and "file_read" not in priority:
            priority.append("file_read")
            if "file_write" not in priority:
                priority.append("file_write")

        return priority


# -----------------------------------------------------------------------
# Unified attention system
# -----------------------------------------------------------------------


class AttentionSystem:
    """Five-subsystem attention pipeline.

    Call :meth:`process` with the current user input (and optional history)
    to receive an :class:`AttentionState` that downstream subsystems can
    use for prioritisation and filtering.
    """

    def __init__(self) -> None:
        self._alerting = AlertingAttention()
        self._orienting = OrientingAttention()
        self._sustained = SustainedAttention()
        self._selective = SelectiveAttention()
        self._executive = ExecutiveAttention()

    def process(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> AttentionState:
        """Run the full attention pipeline.

        Args:
            user_input: The current user message.
            history: Optional conversation history (unused currently;
                reserved for future context-based orienting).

        Returns:
            An :class:`AttentionState` summarising urgency, topics,
            continuity, and suggested tool priorities.
        """
        urgency = self._alerting.score(user_input)
        topics = self._orienting.extract_topics(user_input)
        focus_duration = self._sustained.update(topics)
        priority_tools = self._executive.prioritise(urgency, topics)
        context_filter = [t.lower() for t in topics]

        return AttentionState(
            urgency=urgency,
            topics=topics,
            focus_duration=focus_duration,
            priority_tools=priority_tools,
            context_filter=context_filter,
        )
