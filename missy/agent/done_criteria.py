"""DONE criteria engine for compound task detection and verification.

Provides utilities for detecting multi-step compound prompts, defining
verifiable completion conditions, and generating verification prompts that
guide the model to confirm task outcomes.

Wiring status (SR-4.4): :func:`make_verification_prompt` is used by
:class:`~missy.agent.runtime.AgentRuntime` today -- it's injected as a
static prompt fragment after every round of tool results. The real,
code-level completion gate lives in ``AgentRuntime._tool_loop()``: it
rejects a model's "done" claim (deterministically, from the actual
``ToolResult.is_error`` flags of the immediately preceding round, not
from anything in this module) when that round contained an unresolved
error, up to a bounded number of retries. :func:`is_compound_task`,
:func:`make_done_prompt`, and the :class:`DoneCriteria` dataclass are
not currently wired into any production code path -- they remain
available for a future feature that surfaces the model's own declared
per-condition completion state, but no such feature exists yet.

FX-round2-F4: :func:`is_observation_task` and
:func:`make_no_fabricated_observation_prompt` ARE wired into
``AgentRuntime._tool_loop()``. The existing ``make_verification_prompt``
mechanism only fires *after* a round of tool calls, so it structurally
cannot catch the harness's actual failure mode: the model answering a
vision/memory-observation-implying request with a confident, specific,
fabricated observation on its *very first* response, having made zero
tool calls at all (e.g. reporting fabricated frame-comparison detail
with ``tools_used: []``). ``_tool_loop()`` detects this case directly --
a "stop" response, no tool call ever made this task, and a user request
:func:`is_observation_task` classifies as observation-implying -- and
retries once with :func:`make_no_fabricated_observation_prompt` rather
than accepting the response as final.

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
        return [c for c, v in zip(self.conditions, self.verified, strict=False) if not v]


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
        "Review the tool output above, especially any errors or non-zero exit codes. "
        "Do NOT report the task as complete unless every tool call relevant to it "
        "actually succeeded — if a command errored, timed out, or a resource (file, "
        "repository, URL) was not directly confirmed to exist via a tool call, the "
        "task is NOT done, even if you already produced content for it. "
        "If something failed or is incomplete, call the appropriate tool again to "
        "fix it — do not describe what went wrong, just retry with corrected "
        "parameters. Only once every relevant step has genuinely succeeded should "
        "you reply with a concise summary for the user. "
        "When you report specific names, values, or counts (filenames, directory "
        "entries, sizes, IDs) copy them exactly from the tool output above — never "
        "substitute a plausible-sounding placeholder or example value (e.g. "
        "'file-1.txt', 'dir-a') that isn't literally present in that output. A real "
        "tool call happening this turn does not license inventing what it returned; "
        "if you are unsure what a result actually contained, look at it again above "
        "rather than guessing."
    )


# FX-round2-F4: keyword markers for requests that imply the model must
# observe something real -- a captured image/frame, or a stored memory
# record -- rather than reason from general knowledge. Deliberately
# keyword-based (not an LLM classification call) so this check is cheap,
# deterministic, and has no failure mode of its own to fabricate around.
# False positives (a request that merely mentions "picture" in passing)
# only cost one extra corrective nudge on a genuine zero-tool-call
# response -- never a false rejection of a response that actually used a
# tool, since the runtime-side check below is also gated on
# `not tool_names_used`.
_VISION_OBSERVATION_MARKERS = (
    "capture",
    "photo",
    "picture",
    "camera",
    "webcam",
    "screenshot",
    "frame",
    "look at",
    "what do you see",
    "what does it look like",
)

_MEMORY_OBSERVATION_MARKERS = (
    "memory",
    "recall",
    "remember",
    "what did i",
    "what did you",
    "search for",
)

_OBSERVATION_MARKER_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(m) for m in _VISION_OBSERVATION_MARKERS + _MEMORY_OBSERVATION_MARKERS)
    + r")\b",
    re.I,
)


def is_observation_task(prompt: str) -> bool:
    """Return ``True`` if *prompt* implies a vision or memory observation.

    Used to gate :func:`make_no_fabricated_observation_prompt`: a request
    matching this is expected to be answered from a real ``vision_*`` or
    ``memory_*`` tool call, not from the model's general knowledge or an
    assumption about what such a call would likely return.

    Args:
        prompt: The raw user prompt string.

    Returns:
        ``True`` when any observation-implying marker matches.
    """
    return bool(_OBSERVATION_MARKER_RE.search(prompt))


def make_no_fabricated_observation_prompt() -> str:
    """Return a corrective prompt for a fabricated observation claim.

    Injected when a request classified by :func:`is_observation_task` was
    answered without a single tool call this task -- the response cannot
    be grounded in a real capture or memory lookup, so any specific
    observational claim in it (what a photo showed, what a memory record
    said) is necessarily invented rather than observed.

    Returns:
        A string instructing the model to make the actual tool call
        before reporting any observation.
    """
    return (
        "You answered a request that requires observing something real "
        "(a captured image/frame, or a stored memory record) without "
        "making a single tool call. Any specific observation in your "
        "answer — what something looked like, what a record said, a "
        "detail you described — was necessarily invented, not observed. "
        "Call the actual vision_*/memory_* tool this request needs before "
        "answering. If the tool genuinely isn't available or the call "
        "fails, say so plainly instead of describing a result you never "
        "obtained."
    )
