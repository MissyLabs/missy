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


def make_fabrication_retry_prompt(user_input: str = "") -> str:
    """Return the corrective prompt for a detected fabrication.

    Args:
        user_input: The current task's original request, if available.
            Restated verbatim for the same cross-task-anchoring reason
            documented on :func:`make_promise_retry_prompt`.
    """
    anchor = (
        f"\n\nThe request you must actually fulfill right now is:\n{user_input}"
        if user_input
        else ""
    )
    return (
        "Your previous response described running a command, checking "
        "something, or producing a result, but you made no tool call this "
        "task -- none of that actually happened. Call the real tool needed "
        f"for this request now, or say plainly that you haven't done it yet.{anchor}"
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


def make_promise_retry_prompt(user_input: str = "") -> str:
    """Return the corrective prompt for a detected unfulfilled promise.

    Args:
        user_input: The current task's original request, if available.
            Restated verbatim so the retry re-anchors the model to *this*
            turn's obligation instead of drifting to a different, still
            -incomplete prior turn visible in the flattened history above
            it (see the SH-004 harness finding: a promise-retry with no
            task anchor let the model complete an unrelated earlier
            turn's obligation and report on that instead).
    """
    anchor = (
        f"\n\nThe request you must actually fulfill right now is:\n{user_input}"
        if user_input
        else ""
    )
    return (
        "Your previous response said you were about to do something but "
        "made no tool call -- that action never happened. Call the "
        "appropriate tool now to actually do it, rather than describing "
        f"an intention.{anchor}"
    )


# ---------------------------------------------------------------------------
# Identity confusion: delegate answers as the underlying coding-assistant
# harness ("Claude Code") instead of as Missy, and falsely disclaims
# Missy's own dispatched tools as unavailable
# ---------------------------------------------------------------------------

# 3rd tool-specific validation run (2026-07-14) found this as the dominant,
# most pervasive failure mode: the acpx-wrapped delegate intermittently
# self-identifies as "Claude Code, a coding assistant running in a
# terminal" -- a distinct entity from "the Missy agent" -- and refuses
# Missy's own dispatched tools (vision_*, audio_*, incus_*, memory_*,
# x11_*, atspi_*, browser_*, self_create_tool, code_evolve) as "belonging
# to the Missy platform" and "not callable from this context." It also
# manifests as refusing to engage with the current message at all
# ("this Discord message is directed at the Missy agent, not at me").
# Unlike detect_fabrication/detect_promise_without_action, this is NOT
# gated on tools_used being empty: the harness observed the confusion
# appearing even in the same turn as a genuine, successful tool call for a
# sibling tool in the identical namespace (e.g. calling vision_capture
# successfully, then denying vision_analyze exists) -- the point here is
# correcting a false statement about identity/capability, not verifying an
# action was taken.
_IDENTITY_CONFUSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bI(?:'m| am)\s+Claude\s+Code\b"),
    re.compile(r"(?i)\bnot\s+operating\s+as\s+Missy\b"),
    re.compile(r"(?i)\bbelong(?:s)?\s+to\s+the\s+Missy\s+(?:agent\s+)?platform\b"),
    re.compile(
        r"(?i)\b(?:not|isn'?t)\s+(?:callable|available)\s+(?:to|from|in)\s+"
        r"(?:me|this\s+context)\b"
    ),
    re.compile(
        r"(?i)\bdirected\s+at\s+the\s+Missy\s+agent\b.{0,40}\bnot\s+at\s+me\b",
        re.DOTALL,
    ),
    re.compile(r"(?i)\bI\s+(?:don'?t|do\s+not)\s+have\s+access\s+to\s+the\s+Missy\b"),
    re.compile(r"(?i)\bI\s+should\s+not\s+(?:be\s+)?(?:respond|act|execut)\w*\b"),
    re.compile(r"(?i)\bcoding\s+assistant\s+running\s+in\s+a\s+terminal\b"),
    re.compile(r"(?i)\bmy\s+available\s+tools\s+are\s+(?:file|coding)[/,]"),
]

_MIN_IDENTITY_CONFUSION_LENGTH = 15


def detect_identity_confusion(text: str) -> bool:
    """Return ``True`` if a response denies being Missy or denies her tools.

    Unlike :func:`detect_fabrication`/:func:`detect_promise_without_action`,
    this check runs regardless of whether tools were used this task: the
    harness observed the delegate make a genuine, successful call to one
    Missy tool and then, in the same reply, falsely deny access to a
    sibling tool in the identical namespace.

    Args:
        text: The candidate final response text.

    Returns:
        ``True`` when *text* matches an identity-confusion pattern.
    """
    if not text or len(text) < _MIN_IDENTITY_CONFUSION_LENGTH:
        return False
    return any(p.search(text) for p in _IDENTITY_CONFUSION_PATTERNS)


def make_identity_confusion_retry_prompt(
    user_input: str = "", available_tool_names: set[str] | frozenset[str] | None = None
) -> str:
    """Return the corrective prompt for a detected identity-confusion reply.

    4th tool-specific validation run (2026-07-14/15) found that the prior
    wording here -- an assertive "[System reminder]: You ARE Missy, not a
    separate assistant..." framing -- itself resembles the shape of a
    jailbreak/identity-override attempt (a bracketed pseudo-system tag,
    injected via a user-role turn, telling the model to disregard its own
    stated reasoning). Some models correctly treat that shape with
    suspicion regardless of who sends it, refuse to comply, and label the
    correction itself "prompt injection" -- a self-reinforcing over-refusal
    spiral distinct from (and more severe than) the original bug. This
    wording instead cites a concrete, checkable fact (the literal tool
    names present in the tool list this turn) rather than asserting a
    claim about identity, and drops the fake-system-authority framing.

    Args:
        user_input: The current task's original request, if available,
            restated verbatim for the same reason documented on
            :func:`make_promise_retry_prompt`.
        available_tool_names: The tool names actually offered this turn
            (e.g. ``AgentRuntime._tool_loop``'s ``allowed_tool_names``).
            When given, a sample is cited by name as the grounding fact.
    """
    anchor = f"\n\nThe request you must actually answer is:\n{user_input}" if user_input else ""
    if available_tool_names:
        sample = ", ".join(sorted(available_tool_names)[:8])
        tools_fact = f" For reference, these are present right now: {sample}."
    else:
        tools_fact = ""
    return (
        "Your previous reply described a different assistant, or treated "
        "this message as meant for someone else. The tool list attached to "
        "this conversation didn't change between that reply and now, and "
        f"every tool in it is directly callable via <tool_call>.{tools_fact} "
        "Please re-read the tool list above and answer the request below "
        f"using it.{anchor}"
    )


# ---------------------------------------------------------------------------
# False capability denial: claims a whole category of tool doesn't exist
# ("no X11", "headless", "no browser", "no camera") without ever naming
# Missy/Claude Code explicitly -- the same root bug as identity confusion
# above, but phrased as a plausible-sounding hardware/environment fact
# rather than an explicit "I am a different assistant" statement. Regexes
# alone can't tell a genuine capability gap from a false one, so this
# check additionally requires the corresponding tool category to actually
# be present in *this turn's* available tool set before flagging -- e.g.
# "no X11" only counts as a false denial when an x11_* tool is actually
# offered, never when X11 tools genuinely aren't configured/available.
# ---------------------------------------------------------------------------

_CAPABILITY_DENIAL_MARKERS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile(r"(?i)\bno\s+(?:x11|display|desktop\s+environment|screen)\b|\bheadless\b"),
        ("x11_", "atspi_"),
    ),
    (
        re.compile(r"(?i)\bno\s+browser\b|\bno\s+(?:js|javascript)\s+runtime\b"),
        ("browser_",),
    ),
    (
        re.compile(r"(?i)\bno\s+(?:gui|mouse|keyboard|window\s+system|input\s+device)\b"),
        ("x11_", "atspi_"),
    ),
    (
        re.compile(r"(?i)\bno\s+camera\b|\bno\s+webcam\b"),
        ("vision_",),
    ),
    (
        re.compile(r"(?i)\bno\s+accessibility\s+(?:tools?|tree)\b"),
        ("atspi_",),
    ),
    (
        re.compile(r"(?i)\bno\s+(?:audio|sound)\s+(?:device|hardware|system)\b"),
        ("audio_", "tts_"),
    ),
]

_MIN_CAPABILITY_DENIAL_LENGTH = 10


def detect_false_capability_denial(
    text: str, tools_used: list[str], available_tool_names: set[str] | frozenset[str]
) -> bool:
    """Return ``True`` for a confident capability denial contradicted by the tool list.

    Args:
        text: The candidate final response text.
        tools_used: Every tool name invoked so far this task. Any non-empty
            list short-circuits to ``False`` -- if a tool from the denied
            category was actually just called, this isn't the bug being
            guarded against here.
        available_tool_names: The tool names actually offered to the model
            this turn (e.g. ``AgentRuntime._tool_loop``'s
            ``allowed_tool_names``).

    Returns:
        ``True`` when *text* denies a capability category and a tool from
        that exact category is present in *available_tool_names*.
    """
    if tools_used:
        return False
    if not text or len(text) < _MIN_CAPABILITY_DENIAL_LENGTH:
        return False
    for pattern, tool_prefixes in _CAPABILITY_DENIAL_MARKERS:
        if pattern.search(text) and any(
            name.startswith(prefix) for name in available_tool_names for prefix in tool_prefixes
        ):
            return True
    return False


def make_capability_denial_retry_prompt(
    user_input: str = "", available_tool_names: set[str] | frozenset[str] | None = None
) -> str:
    """Return the corrective prompt for a detected false capability denial.

    Reworded for the same reason documented on
    :func:`make_identity_confusion_retry_prompt`: cites the literal tool
    names present this turn as a checkable fact rather than a bracketed
    "[System reminder]" assertion, which some models pattern-match as an
    injected identity-override attempt and refuse on principle.

    Args:
        user_input: The current task's original request, if available,
            restated verbatim for the same reason documented on
            :func:`make_promise_retry_prompt`.
        available_tool_names: The tool names actually offered this turn.
            When given, a sample is cited by name as the grounding fact.
    """
    anchor = f"\n\nThe request you must actually answer is:\n{user_input}" if user_input else ""
    if available_tool_names:
        sample = ", ".join(sorted(available_tool_names)[:8])
        tools_fact = f" These are present in your tool list this turn: {sample}."
    else:
        tools_fact = ""
    return (
        "Your previous reply said a capability (display, browser, camera, "
        "accessibility tree, audio device, etc.) isn't available. A tool "
        f"for exactly that capability is present this turn.{tools_fact} "
        "Please re-read the tool list above, call the matching tool, and "
        f"report its real result.{anchor}"
    )
