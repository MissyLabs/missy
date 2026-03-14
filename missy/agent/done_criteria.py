"""DONE criteria engine for compound task detection and verification.

Provides utilities for detecting multi-step compound prompts, defining
verifiable completion conditions, and generating verification prompts that
guide the model to confirm task outcomes.

Example::

    from missy.agent.done_criteria import is_compound_task, DoneCriteria

    if is_compound_task("Search for the file then delete it"):
        criteria = DoneCriteria(conditions=["file found", "file deleted"])
        criteria.verified = [True, False]
        print(criteria.pending)  # ["file deleted"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_COMPOUND_PATTERNS = [
    re.compile(r"\b(and then|then|after that|followed by|subsequently)\b", re.I),
    re.compile(r"^\s*\d+[\.\)]\s", re.M),  # numbered list
    re.compile(r"^\s*[-*]\s", re.M),  # bullet list
    re.compile(r"\b(first|second|third|finally|lastly)\b", re.I),
]


@dataclass
class DoneCriteria:
    """Tracks verifiable completion conditions for a task.

    Attributes:
        conditions: Human-readable list of conditions that must be true for
            the task to be considered complete.
        verified: Parallel boolean list indicating which conditions have been
            confirmed.  Must align with *conditions* by index.
    """

    conditions: list[str] = field(default_factory=list)
    verified: list[bool] = field(default_factory=list)

    @property
    def all_met(self) -> bool:
        """Return ``True`` when every condition has been verified.

        Returns:
            ``True`` iff *conditions* is non-empty and all entries in
            *verified* are ``True``.
        """
        return bool(self.conditions) and all(self.verified)

    @property
    def pending(self) -> list[str]:
        """Return the subset of conditions not yet verified.

        Returns:
            A list of condition strings whose corresponding *verified* entry
            is ``False``.
        """
        return [c for c, v in zip(self.conditions, self.verified) if not v]


def is_compound_task(prompt: str) -> bool:
    """Return ``True`` if *prompt* appears to be a multi-step compound task.

    Detects numbered/bulleted lists, sequential connectives (``"then"``,
    ``"after that"``, etc.), and ordinal words (``"first"``, ``"finally"``).

    Args:
        prompt: The raw user prompt string.

    Returns:
        ``True`` when any compound pattern matches.
    """
    return any(p.search(prompt) for p in _COMPOUND_PATTERNS)


def make_done_prompt() -> str:
    """Return a prompt fragment that instructs the model to define DONE conditions.

    Returns:
        A string to prepend to a compound task prompt.
    """
    return (
        "Before proceeding, define your DONE conditions. "
        "List the specific, verifiable outcomes that must be true for this task to be complete."
    )


def make_verification_prompt() -> str:
    """Return a verification prompt to inject after tool call results.

    Returns:
        A string instructing the model to review tool output and assess
        whether DONE conditions are satisfied.
    """
    return (
        "Review the tool output above. "
        "If the task is fully complete, reply with a concise summary for the user. "
        "If it failed or is incomplete, call the appropriate tool again to fix it — "
        "do not describe what went wrong, just retry with corrected parameters."
    )
