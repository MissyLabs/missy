"""General-purpose guards against fabricated or unfulfilled tool-free responses.

Complements the narrower checks already in :mod:`missy.agent.done_criteria`
and :class:`~missy.agent.runtime.AgentRuntime`'s own placeholder-artifact
and observation-fabrication retries. Those catch specific shapes (an
internal placeholder string leaking out; a vision/memory-observation
request answered with zero tool calls). This module catches two broader,
request-agnostic patterns in any text-only response (``tools_used`` empty
for the whole task):

1. **Fabrication** -- the response describes having already run a
   command, checked a log, or completed an action, phrased as if it
   happened, when no tool call backs it up.
2. **Promise without action** -- the response says it is *about to* do
   something ("I'll...", "Working on it...") and then simply stops,
   producing no tool call to actually do it.

Both are a likely root cause of a session "misresponding" to a later
prompt: the model's own prior turn asserted something false or
unfinished, and a subsequent turn either builds on that false premise or
never receives the thing that was promised.

Design mirrors :mod:`missy.agent.done_criteria`'s existing
:func:`~missy.agent.done_criteria.is_observation_task`: cheap, regex-based,
deterministic classification rather than an LLM call, so the guard itself
has no failure mode to fabricate around. A false positive only costs one
extra corrective retry on a genuine zero-tool-call response -- it can
never fire on a response that actually used a tool, since both detectors
require ``tools_used`` to be empty.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Fabrication: response describes an action or observation as already done
# ---------------------------------------------------------------------------

_FABRICATION_PATTERNS: list[re.Pattern[str]] = [
    # First-person claims of having already run/checked/investigated something.
    re.compile(
        r"(?i)\bI(?:'ve| have)?\s+(?:ran|executed|checked|performed|ran\s+a|"
        r"looked\s+at|reviewed|inspected|examined|verified|confirmed|"
        r"tested|scanned|monitored|queried|found\s+that|found\s+the|"
        r"can\s+see\s+that|can\s+see\s+the)\b"
    ),
    # A fenced code block shaped like real command output (shell prompts,
    # `ls -l`/`ps`/`docker ps`-style column headers) with no tool call
    # behind it.
    re.compile(
        r"```(?:bash|shell|console|sh|text|output)?\s*\n"
        r"(?:[$#>].*\n|(?:total \d|drwx|Filesystem|CONTAINER|NAME\s|PID\s|USER\s))"
    ),
    # Claims a concrete artifact was produced/delivered.
    re.compile(
        r"(?i)\b(?:I(?:'ve| have)?\s+)?(?:generated|posted|created|saved|"
        r"uploaded|deployed|installed|deleted|removed|wrote|written|"
        r"downloaded)\b[^.!?\n]{0,40}\b(?:image|file|script|server|"
        r"container|process|document|report|screenshot)\b"
    ),
    # References an external data source as if it had genuinely been read.
    re.compile(
        r"(?i)\b(?:according to (?:the )?(?:logs?|output|results?|data|"
        r"metrics|dashboard)|based on (?:the )?(?:output|logs?|results?|"
        r"metrics))\b"
    ),
]

#: Minimum response length to consider for fabrication detection -- a very
#: short response ("ok", "sure") can't meaningfully match any of the
#: patterns above, so skip the (cheap, but non-zero) regex work.
_MIN_FABRICATION_LENGTH = 20


def detect_fabrication(text: str, tools_used: list[str]) -> bool:
    """Return ``True`` if a tool-free response reads like fabricated tool output.

    Args:
        text: The candidate final response text.
        tools_used: Every tool name invoked so far this task. Any non-empty
            list short-circuits to ``False`` -- this check only applies to
            responses with nothing real behind them.

    Returns:
        ``True`` when *text* matches a fabrication pattern and no tool was
        ever actually called this task.
    """
    if tools_used:
        return False
    if not text or len(text) < _MIN_FABRICATION_LENGTH:
        return False
    return any(p.search(text) for p in _FABRICATION_PATTERNS)


def make_fabrication_retry_prompt() -> str:
    """Return the corrective prompt for a detected fabrication."""
    return (
        "Your previous response described running a command, checking "
        "something, or producing a result, but you made no tool call this "
        "task -- none of that actually happened. Call the real tool needed "
        "for this request now, or say plainly that you haven't done it yet."
    )


# ---------------------------------------------------------------------------
# Promise without action: response announces intent, calls no tool, stops
# ---------------------------------------------------------------------------

_PROMISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bI'(?:ll|m going to)\s+\w+"),
    re.compile(r"(?i)\bI'm\s+\w+ing\b"),
    re.compile(r"(?i)\bI will\s+\w+"),
    re.compile(
        r"(?i)^(?:On it|Working on it|Spinning up|Starting|Kicking off|"
        r"Setting up|Pulling|Generating|Building|Deploying|Running|"
        r"Creating|Launching|Firing up|Booting|Preparing|Fetching)\b",
        re.MULTILINE,
    ),
]

#: Phrases that match _PROMISE_PATTERNS syntactically but are ordinary
#: conversational future tense, not an unfulfilled action promise -- e.g.
#: "I'll be around if you need anything" or "I'll keep that in mind" never
#: imply a pending tool call, so retrying on them would just loop on a
#: perfectly fine chat reply.
_PROMISE_EXEMPTIONS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bI'll\s+(?:be|keep|remember|stay|wait|let you know)\b"),
    re.compile(r"(?i)\bI will\s+(?:be|keep|remember|stay|wait|let you know)\b"),
]

_MIN_PROMISE_LENGTH = 15


def detect_promise_without_action(text: str, tools_used: list[str]) -> bool:
    """Return ``True`` if a tool-free response promises action it never takes.

    Args:
        text: The candidate final response text.
        tools_used: Every tool name invoked so far this task. Any non-empty
            list short-circuits to ``False``.

    Returns:
        ``True`` when *text* matches a promise-of-action pattern, isn't
        covered by :data:`_PROMISE_EXEMPTIONS`, and no tool was ever
        actually called this task.
    """
    if tools_used:
        return False
    if not text or len(text) < _MIN_PROMISE_LENGTH:
        return False
    if any(p.search(text) for p in _PROMISE_EXEMPTIONS):
        return False
    return any(p.search(text) for p in _PROMISE_PATTERNS)


def make_promise_retry_prompt() -> str:
    """Return the corrective prompt for a detected unfulfilled promise."""
    return (
        "Your previous response said you were about to do something but "
        "made no tool call -- that action never happened. Call the "
        "appropriate tool now to actually do it, rather than describing "
        "an intention."
    )
