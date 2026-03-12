"""Cross-task learning extraction and storage.

After each completed agent run, :func:`extract_learnings` analyses which
tools were used and whether the run succeeded, producing a
:class:`TaskLearning` record that can be persisted and fed back into
future runs via :class:`~missy.agent.context.ContextManager`.

Example::

    from missy.agent.learnings import extract_learnings

    learning = extract_learnings(
        tool_names_used=["shell_exec", "file_write"],
        final_response="Successfully created the file.",
        prompt="Create a config file",
    )
    print(learning.lesson)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TaskLearning:
    """A learning record extracted from a completed agent task.

    Attributes:
        task_type: Coarse category derived from the tools used
            (e.g. ``"shell"``, ``"web"``, ``"file"``).
        approach: Ordered list of tool names actually invoked (capped at 5).
        outcome: One of ``"success"``, ``"failure"``, or ``"partial"``.
        lesson: Human-readable lesson sentence summarising what worked or
            failed.
        timestamp: ISO-8601 UTC timestamp of when the learning was created.
    """

    task_type: str
    approach: list[str]
    outcome: str  # "success" | "failure" | "partial"
    lesson: str
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def extract_task_type(tool_names_used: list[str]) -> str:
    """Derive a coarse task-type label from the set of tools invoked.

    Priority order: shell+web > shell+file > shell > web > file > chat.

    Args:
        tool_names_used: List of tool name strings used during the run.

    Returns:
        A short label string such as ``"shell"``, ``"web"``, ``"file"``,
        or ``"chat"``.
    """
    s = set(tool_names_used)
    if "shell_exec" in s and "web_fetch" in s:
        return "shell+web"
    if "shell_exec" in s and "file_write" in s:
        return "shell+file"
    if "shell_exec" in s:
        return "shell"
    if "web_fetch" in s:
        return "web"
    if "file_read" in s or "file_write" in s:
        return "file"
    return "chat"


def extract_outcome(final_response: str) -> str:
    """Infer task outcome from the model's final response text.

    Checks for success/failure keywords via simple case-insensitive substring
    matching.

    Args:
        final_response: The last assistant response string.

    Returns:
        One of ``"success"``, ``"failure"``, or ``"partial"``.
    """
    low = final_response.lower()
    if any(w in low for w in ("successfully", "completed", "done", "finished", "worked")):
        return "success"
    if any(w in low for w in ("failed", "error", "unable", "couldn't", "cannot")):
        return "failure"
    return "partial"


def extract_learnings(
    tool_names_used: list[str],
    final_response: str,
    prompt: str,
) -> TaskLearning:
    """Build a :class:`TaskLearning` from a completed agent run.

    Args:
        tool_names_used: Names of all tools invoked during the run.
        final_response: The last text response from the model.
        prompt: The original user prompt (currently unused but included for
            future richer extraction).

    Returns:
        A :class:`TaskLearning` instance capturing the approach and lesson.
    """
    task_type = extract_task_type(tool_names_used)
    outcome = extract_outcome(final_response)
    approach = tool_names_used[:5] if tool_names_used else ["direct_response"]

    if outcome == "success":
        lesson = f"{task_type}: {' → '.join(approach)} succeeded"
    else:
        lesson = f"{task_type}: {' → '.join(approach)} — outcome {outcome}"

    return TaskLearning(
        task_type=task_type,
        approach=approach,
        outcome=outcome,
        lesson=lesson,
    )
