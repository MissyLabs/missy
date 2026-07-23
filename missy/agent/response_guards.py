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
# Incomplete security refusal: a high-risk request is refused, but the reply
# omits the concrete governed alternative required by the Discord contract.
# ---------------------------------------------------------------------------

_HIGH_RISK_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)(?:read|show|summarize|upload|send|disclose).{0,60}"
        r"(?:secret|token|credential|\.env|/etc/shadow)"
    ),
    re.compile(
        r"(?i)(?:sudo|privilege|host\s+root|disable\s+(?:host\s+)?security|"
        r"broader\s+permissions|skip\s+policy|bypass\s+(?:policy|safety|security))"
    ),
)

_REFUSAL_PATTERN = re.compile(
    r"(?i)\b(?:can(?:not|['’]t)|won['’]t|will\s+not|unable|not\s+able|refus(?:e|ing)|"
    r"not\s+(?:allowed|permitted))\b"
)

_SAFE_ALTERNATIVE_PATTERN = re.compile(
    r"(?i)(?:safe\s+alternative|instead.{0,30}\b(?:can|could)\b|"
    r"\bI\s+can\s+(?:help|inspect|review|create|provide|explain|work|use)|"
    r"unprivileged|containeri[sz]ed|saniti[sz]ed|redacted|gitignore|read-only|"
    r"within\s+(?:the\s+)?(?:approved|existing)\s+(?:workspace|permissions?))"
)

_SECRET_UPLOAD_REQUEST_PATTERN = re.compile(
    r"(?i)(?:upload|send).{0,60}(?:secret|token|credential|\.env)"
)
_SECRET_UPLOAD_ALTERNATIVE_PATTERN = re.compile(
    r"(?i)(?:gitignore|redacted\s+(?:report|file|copy|output))"
)
_HOST_PRIVILEGE_REQUEST_PATTERN = re.compile(
    r"(?i)(?:sudo|host\s+root|privileg|system\s+settings?|disable\s+(?:host\s+)?security)"
)
_HOST_PRIVILEGE_ALTERNATIVE_PATTERN = re.compile(
    r"(?i)(?:unprivileged|containeri[sz]ed|disposable\s+container)"
)


def detect_security_refusal_without_alternative(text: str, user_input: str) -> bool:
    """Detect a bare refusal to a security-sensitive request.

    The detector is deliberately gated on both the original high-risk request
    and an explicit refusal phrase. It does not turn ordinary answers into
    refusals and does not ask the model to reconsider the denial; it only
    identifies that the required governed alternative is absent.
    """
    if not isinstance(text, str) or not isinstance(user_input, str):
        return False
    if not text or not user_input:
        return False
    if not any(pattern.search(user_input) for pattern in _HIGH_RISK_REQUEST_PATTERNS):
        return False
    if not _REFUSAL_PATTERN.search(text):
        return False
    if _SAFE_ALTERNATIVE_PATTERN.search(text) is None:
        return True
    if _SECRET_UPLOAD_REQUEST_PATTERN.search(user_input):
        return _SECRET_UPLOAD_ALTERNATIVE_PATTERN.search(text) is None
    if _HOST_PRIVILEGE_REQUEST_PATTERN.search(user_input):
        return _HOST_PRIVILEGE_ALTERNATIVE_PATTERN.search(text) is None
    return False


def is_security_refusal(text: str, user_input: str) -> bool:
    """Return whether *text* refuses a recognized high-risk request.

    Unlike :func:`detect_security_refusal_without_alternative`, this does not
    assess the quality of the offered alternative.  The runtime uses it to
    ensure a separate explicit-tool completion guard never pressures the model
    to execute a tool after it correctly refused privilege escalation or
    credential disclosure.
    """
    if not isinstance(text, str) or not isinstance(user_input, str):
        return False
    return bool(
        text
        and user_input
        and _REFUSAL_PATTERN.search(text)
        and any(pattern.search(user_input) for pattern in _HIGH_RISK_REQUEST_PATTERNS)
    )


def make_security_refusal_retry_prompt(user_input: str = "") -> str:
    """Return a bounded correction that preserves the security refusal."""
    anchor = f"\n\nThe request you must safely answer is:\n{user_input}" if user_input else ""
    return (
        "Your refusal is correct but incomplete. Keep the refusal unchanged "
        "and add one concrete alternative that stays within existing authority. "
        "For secret uploads, offer a gitignore check or a redacted report. For "
        "host package, sudo, or system-setting requests, offer an unprivileged "
        "disposable container. For policy bypass requests, offer a read-only "
        "policy review or a human-reviewed logging improvement. Do not call a "
        f"tool and do not offer any route around the control.{anchor}"
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


_EXPLICIT_TOOL_VERBS = ("use", "call", "invoke", "execute", "run")
_EXPLICIT_TOOL_CONNECTORS = ("using", "with")
_NEGATED_TOOL_REQUEST_RE = re.compile(r"(?i)(?:\bdo\s+not|\bdon['’]t|\bnever|\bwithout)\s*$")


def detect_explicit_tool_requests(
    user_input: str,
    available_tool_names: set[str] | frozenset[str],
) -> frozenset[str]:
    """Return available tools the user unambiguously asked the agent to call.

    This is deliberately narrower than general intent classification.  It only
    recognizes a real tool name next to an explicit dispatch verb (``use``,
    ``call``, ``invoke``, ``execute``, or ``run``), or the constructions
    ``using <tool>`` / ``with <tool>``.  Merely mentioning a tool in prose does
    not create a runtime requirement, and locally negated requests such as
    ``do not use shell_exec`` are excluded.

    The check is used by the tool loop after a provider returns a text-only
    answer.  It prevents a model from silently doing arithmetic in its head,
    inventing command output, or otherwise bypassing a specifically requested
    governed tool while still leaving ordinary conversational answers alone.
    """
    if not isinstance(user_input, str) or not user_input.strip():
        return frozenset()

    requested: set[str] = set()
    for name in available_tool_names:
        if not isinstance(name, str) or not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            continue
        quoted_name = rf"[`'\"]?{re.escape(name)}[`'\"]?"
        article = r"(?:(?:your|the|a|an)\s+)?"
        optional_tool_prefix = r"(?:tool\s+)?"
        optional_tool_suffix = r"(?:\s+tool)?"
        patterns = (
            rf"(?i)\b(?:{'|'.join(_EXPLICIT_TOOL_VERBS)})\s+"
            rf"{article}{optional_tool_prefix}{quoted_name}{optional_tool_suffix}\b",
            rf"(?i)\b(?:{'|'.join(_EXPLICIT_TOOL_CONNECTORS)})\s+"
            rf"{article}{optional_tool_prefix}{quoted_name}{optional_tool_suffix}\b",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, user_input):
                # Only inspect the immediately preceding phrase.  This catches
                # "do not use X", "never call X", and "without using X"
                # without interpreting unrelated earlier prose as negation.
                prefix = user_input[max(0, match.start() - 24) : match.start()]
                if _NEGATED_TOOL_REQUEST_RE.search(prefix):
                    continue
                requested.add(name)
                break
            if name in requested:
                break
    return frozenset(requested)


def make_explicit_tool_request_retry_prompt(
    missing_tool_names: set[str] | frozenset[str],
    user_input: str = "",
) -> str:
    """Return a bounded correction for an ignored explicit tool request."""
    names = ", ".join(sorted(missing_tool_names)) or "the explicitly requested tool"
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        f"The user explicitly requested that you use this available tool: {names}. "
        "Your previous answer did not call the named requested tool(s). Call the named "
        "tool now and perform every requested operation through it; do not compute, "
        "simulate, or invent the requested results yourself. Then report only the "
        f"real tool results.{anchor}"
    )


# ---------------------------------------------------------------------------
# Calculator result completeness: multi-expression requests can execute every
# call correctly and still return a final answer that only mentions the last
# result/error after the generic DONE-criteria correction loop.
# ---------------------------------------------------------------------------

_CALCULATOR_ERROR_MARKERS = (
    "unsupported expression construct:",
    "exceeds the maximum allowed value of",
    "division by zero",
    "expression must not be empty",
    "syntax error in expression",
)
_CALCULATOR_NUMERIC_RESULT_RE = re.compile(
    r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|"
    r"\(?[-+]?\d+(?:\.\d*)?[-+]\d+(?:\.\d*)?j\)?)",
    re.IGNORECASE,
)


def calculator_observations_are_reportable(
    observations: list[tuple[str, str, bool]],
) -> bool:
    """Return whether every observation is a documented calculator outcome."""
    if not observations:
        return False
    for _expression, content, is_error in observations:
        folded_content = (content or "").strip().casefold()
        if is_error:
            if not any(marker in folded_content for marker in _CALCULATOR_ERROR_MARKERS):
                return False
        elif _CALCULATOR_NUMERIC_RESULT_RE.fullmatch(folded_content) is None:
            return False
    return True


def find_unreported_calculator_expressions(
    observations: list[tuple[str, str, bool]],
    response_text: str,
) -> list[str]:
    """Return executed calculator inputs whose outcome is absent from a reply.

    ``observations`` contains ``(expression, tool_content, is_error)`` tuples
    captured from the current task.  Both the expression and its observed
    result/error must be represented in the final answer.  Empty and
    whitespace-only expressions cannot be searched literally, so their
    explicit calculator error marker is the evidence requirement instead.
    """
    if not observations:
        return []
    folded_response = (response_text or "").casefold()
    missing: list[str] = []
    seen: set[str] = set()
    for expression, content, is_error in observations:
        label = expression if expression.strip() else "<empty or whitespace-only expression>"
        folded_content = (content or "").casefold()
        if is_error:
            applicable_markers = [
                marker for marker in _CALCULATOR_ERROR_MARKERS if marker in folded_content
            ]
            # Unknown infrastructure/schema errors remain under the generic
            # DONE-criteria gate.  This guard only understands the calculator's
            # documented arithmetic-domain outcomes.
            if not applicable_markers:
                continue
            outcome_reported = bool(applicable_markers) and any(
                marker in folded_response for marker in applicable_markers
            )
        else:
            result = (content or "").strip().casefold()
            # Test doubles and infrastructure failures sometimes yield prose
            # such as "calculator_result".  Production calculator successes
            # are numeric; do not impose a semantic result check on anything
            # outside that contract.
            if _CALCULATOR_NUMERIC_RESULT_RE.fullmatch(result) is None:
                continue
            outcome_reported = bool(result) and result in folded_response
        if label in seen:
            continue
        seen.add(label)
        expression_reported = not expression.strip() or expression.casefold() in folded_response
        if not expression_reported or not outcome_reported:
            missing.append(label)
    return missing


def make_calculator_completeness_retry_prompt(
    observations: list[tuple[str, str, bool]],
    missing_expressions: list[str],
    user_input: str = "",
) -> str:
    """Return a correction grounded in every current-task calculator result."""
    rendered: list[str] = []
    for expression, content, is_error in observations:
        label = expression if expression.strip() else "<empty or whitespace-only expression>"
        outcome = (content or "").split(". Report this error", 1)[0].strip()
        rendered.append(f"- `{label}` -> {'ERROR' if is_error else 'RESULT'}: {outcome[:500]}")
    missing = ", ".join(f"`{item}`" for item in missing_expressions)
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        "Your previous final answer omitted one or more calculator outcomes "
        f"that were actually observed in this task: {missing}. Do not call the "
        "calculator again and do not rewrite any expression. Report every "
        "executed expression with its exact result or error from this evidence:\n"
        + "\n".join(rendered)
        + anchor
    )


_DESKTOP_MUTATION_TOOLS = frozenset(
    {
        "x11_launch",
        "x11_click",
        "x11_type",
        "x11_key",
        "desktop_launch_app",
        "desktop_focus_window",
        "desktop_mouse_drag",
        "atspi_click",
        "atspi_set_value",
    }
)
_DESKTOP_VERIFICATION_TOOLS = frozenset(
    {
        "x11_read_screen",
        "x11_screenshot",
        "x11_window_list",
        "atspi_get_tree",
        "atspi_get_text",
        "desktop_status",
    }
)

_DESKTOP_REQUEST_RULES: tuple[tuple[str, re.Pattern[str], frozenset[str]], ...] = (
    (
        "keyboard shortcut",
        re.compile(
            r"(?i)\b(?:use|send|press)\s+(?:(?:the|your)\s+)?keyboard\s+shortcuts?\b|"
            r"\bselect\s+all\b.{0,80}\bcopy\b"
        ),
        frozenset({"x11_key"}),
    ),
    (
        "button click",
        re.compile(r"(?i)\bclick\s+(?:the\s+)?(?:button|control|menu|checkbox|dialog)\b"),
        frozenset({"atspi_click", "x11_click"}),
    ),
    (
        "application launch",
        re.compile(
            r"(?i)\b(?:open|launch)\s+(?:a|an|the)\s+(?:simple\s+)?(?:x11\s+)?"
            r"(?:text\s+editor|terminal|app|application)\b"
        ),
        frozenset({"desktop_launch_app", "x11_launch"}),
    ),
    (
        "desktop typing",
        re.compile(r"(?i)\btype\s+.{1,160}"),
        frozenset({"x11_type"}),
    ),
    (
        "desktop screenshot",
        re.compile(r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot\b"),
        frozenset({"x11_screenshot"}),
    ),
    (
        "screen read",
        re.compile(
            r"(?i)\b(?:read|inspect|describe)\s+(?:the\s+)?(?:current\s+)?"
            r"(?:desktop\s+)?screen\b"
        ),
        frozenset({"x11_read_screen"}),
    ),
    (
        "window listing",
        re.compile(r"(?i)\b(?:report|list|show)\s+(?:the\s+)?(?:open\s+)?windows\b"),
        frozenset({"x11_window_list"}),
    ),
)


def find_unmet_desktop_requests(
    user_input: str,
    tool_names_used: list[str],
    available_tool_names: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Return imperative desktop requirements not executed this task.

    Long-lived channel history can contain an identical earlier desktop
    request and its successful answer. Providers sometimes replay that answer
    without issuing a tool call in the current turn. These narrow rules bind
    common imperative desktop wording to the concrete tool family that must
    appear in the current task's audit trail.
    """
    if not isinstance(user_input, str) or not isinstance(tool_names_used, list):
        return []
    available = set(available_tool_names) if available_tool_names is not None else None
    used = set(tool_names_used)
    browser_screenshot_request = bool(
        re.search(r"(?i)\b(?:browser|webpage|page|dashboard|https?://)\b", user_input)
        and re.search(r"(?i)\b(?:take|capture)\s+(?:a\s+)?screenshot\b", user_input)
    )
    missing: list[str] = []
    for label, pattern, alternatives in _DESKTOP_REQUEST_RULES:
        # "Take a screenshot" is shared wording between the browser and X11
        # surfaces.  A rendered-page request must remain on Playwright; forcing
        # x11_screenshot after browser_screenshot both captures the wrong
        # surface and can upload a second, unrelated image.
        if label == "desktop screenshot" and browser_screenshot_request:
            continue
        if pattern.search(user_input) is None:
            continue
        usable = alternatives if available is None else alternatives.intersection(available)
        if (available is None or usable) and (not usable or used.isdisjoint(usable)):
            missing.append(label)
    return missing


def make_desktop_request_retry_prompt(missing: list[str], user_input: str = "") -> str:
    """Return a correction requiring current-turn desktop execution."""
    requirements = ", ".join(missing)
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        "You answered an imperative desktop request without executing every required "
        f"desktop operation in this task. Still missing: {requirements}. Do not reuse or "
        "paraphrase an answer from an earlier channel turn. Call the matching X11, AT-SPI, "
        "or desktop tool now and base your answer only on its current result."
        f"{anchor}"
    )


def find_unverified_desktop_action(tool_names_used: list[str]) -> str | None:
    """Return the most recent desktop mutation lacking later observation.

    A successful low-level click/key API result only proves input was emitted;
    it does not prove the intended control received it or that the UI changed.
    The tool loop uses this sequence check to require an observation *after*
    the latest mutation before accepting a completion claim.
    """
    if not isinstance(tool_names_used, list):
        return None
    last_action_index = -1
    last_action_name: str | None = None
    for index, name in enumerate(tool_names_used):
        if name in _DESKTOP_MUTATION_TOOLS:
            last_action_index = index
            last_action_name = name
    if last_action_index < 0:
        return None
    if any(
        name in _DESKTOP_VERIFICATION_TOOLS for name in tool_names_used[last_action_index + 1 :]
    ):
        return None
    return last_action_name


def make_desktop_verification_retry_prompt(action_name: str, user_input: str = "") -> str:
    """Return a correction requiring post-action desktop evidence."""
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        f"Your last desktop action was {action_name}, but you made no later observation "
        "to confirm it affected the intended control or changed the UI. A successful input "
        "event is not proof of the requested outcome. Inspect the current state now with "
        "x11_read_screen, x11_window_list, or the matching AT-SPI read/tree tool. If the "
        "target state is not present, correct the target and retry the action, then observe "
        f"again before reporting success.{anchor}"
    )


_FILESYSTEM_MUTATION_TOOLS = frozenset({"file_write", "file_delete"})
_FILESYSTEM_VERIFICATION_TOOLS = frozenset({"list_files", "file_read"})


def find_unverified_filesystem_action(successful_tool_names: list[str]) -> str | None:
    """Return the latest successful file mutation lacking later observation.

    A successful write/delete syscall establishes that the operation returned
    without error, but multi-path agent tasks still need a post-mutation view
    before they can truthfully claim the requested tree is complete. Only
    successful calls are supplied so a policy-denied mutation remains the
    responsibility of the ordinary done-criteria error gate.
    """
    if not isinstance(successful_tool_names, list):
        return None
    last_action_index = -1
    last_action_name: str | None = None
    for index, name in enumerate(successful_tool_names):
        if name in _FILESYSTEM_MUTATION_TOOLS:
            last_action_index = index
            last_action_name = name
    if last_action_index < 0:
        return None
    if any(
        name in _FILESYSTEM_VERIFICATION_TOOLS
        for name in successful_tool_names[last_action_index + 1 :]
    ):
        return None
    return last_action_name


def make_filesystem_verification_retry_prompt(action_name: str, user_input: str = "") -> str:
    """Return a correction requiring post-mutation filesystem evidence."""
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        f"Your last successful filesystem mutation was {action_name}, but you made no "
        "later filesystem observation to confirm the requested final state. Verify the "
        "affected paths now with list_files (use a recursive listing when the request "
        "spans directories) or file_read. Do not repeat the mutation unless verification "
        f"shows it is necessary.{anchor}"
    )


_BROWSER_TOOL_NAMES = frozenset(
    {
        "browser_navigate",
        "browser_click",
        "browser_fill",
        "browser_screenshot",
        "browser_get_content",
        "browser_evaluate",
        "browser_wait",
        "browser_get_url",
        "browser_close",
    }
)


def find_unmet_web_requests(
    user_input: str,
    tool_names_used: list[str],
    available_tool_names: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Return concrete web/browser tools still required by the request.

    These rules cover explicit interaction language, not broad topical web
    questions. They prevent a provider from substituting raw HTML for a form,
    rendered DOM, JavaScript, or screenshot task and ensure every browser
    workflow closes its persistent session.
    """
    if not isinstance(user_input, str) or not isinstance(tool_names_used, list):
        return []
    text = user_input.lower()
    available = set(available_tool_names) if available_tool_names is not None else None
    expected: set[str] = set()

    if re.search(r"\bfetch\b.{0,100}\b(?:url|https?://)", text, re.DOTALL):
        expected.add("web_fetch")

    browser_intent = bool(
        re.search(r"\bopen\b.{0,100}\b(?:browser|webpage|page|dashboard)\b", text, re.DOTALL)
        or re.search(r"\b(?:fill|submit)\b.{0,120}\b(?:form|page|data)\b", text, re.DOTALL)
        or "visible text from the main content" in text
    )
    if browser_intent:
        expected.update({"browser_navigate", "browser_close"})
    required_fill_calls = 0
    submit_action = browser_intent and "submit" in text
    if browser_intent:
        if "current url" in text or "page title" in text:
            expected.add("browser_get_url")
        if "fill" in text and ("form" in text or "name/email" in text):
            expected.add("browser_fill")
            required_fill_calls = 2 if "name/email" in text else 1
        if submit_action:
            expected.update({"browser_click", "browser_wait", "browser_get_content"})
        if "screenshot" in text:
            expected.add("browser_screenshot")
        if "upload" in text and "discord" in text:
            expected.add("discord_upload_file")
        if "visible text" in text or "main content" in text:
            expected.add("browser_get_content")
        if "javascript" in text or "using js" in text:
            expected.add("browser_evaluate")
        if "wait until" in text or "wait for" in text:
            expected.update({"browser_wait", "browser_get_content"})

    if available is not None:
        expected.intersection_update(available)
    used = set(tool_names_used)
    missing = expected.difference(used)
    if (
        required_fill_calls
        and tool_names_used.count("browser_fill") < required_fill_calls
        and (available is None or "browser_fill" in available)
    ):
        missing.add("browser_fill")

    # Form completion is order-sensitive. A content read before submit cannot
    # verify the confirmation, and closing/reopening between submit and wait
    # loses the submitted DOM state. Require one coherent sequence after the
    # latest navigation, including every separately requested field.
    if submit_action and "browser_navigate" in used:
        latest_navigation = max(
            i for i, name in enumerate(tool_names_used) if name == "browser_navigate"
        )
        fill_indexes = [
            i
            for i, name in enumerate(tool_names_used)
            if name == "browser_fill" and i > latest_navigation
        ]
        needed_fills = max(required_fill_calls, 1)
        if len(fill_indexes) < needed_fills:
            missing.add("browser_fill")
        last_fill = max(fill_indexes, default=latest_navigation)
        click_indexes = [
            i
            for i, name in enumerate(tool_names_used)
            if name == "browser_click" and i > last_fill
        ]
        if not click_indexes:
            missing.add("browser_click")
        last_click = max(click_indexes, default=len(tool_names_used))
        wait_indexes = [
            i
            for i, name in enumerate(tool_names_used)
            if name == "browser_wait" and i > last_click
        ]
        if not wait_indexes:
            missing.add("browser_wait")
        last_wait = max(wait_indexes, default=len(tool_names_used))
        content_indexes = [
            i
            for i, name in enumerate(tool_names_used)
            if name == "browser_get_content" and i > last_wait
        ]
        if not content_indexes:
            missing.add("browser_get_content")
        last_content = max(content_indexes, default=len(tool_names_used))
        if not any(
            name == "browser_close"
            for name in tool_names_used[last_content + 1 :]
        ):
            missing.add("browser_close")

    # Closing before a later browser operation does not close the resulting
    # live session, so treat it as still missing.
    if "browser_close" in expected and "browser_close" in used:
        last_close = max(i for i, name in enumerate(tool_names_used) if name == "browser_close")
        if any(name in _BROWSER_TOOL_NAMES for name in tool_names_used[last_close + 1 :]):
            missing.add("browser_close")
    return sorted(missing)


def make_web_request_retry_prompt(missing: list[str], user_input: str = "") -> str:
    """Return a correction requiring the missing browser workflow steps."""
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        "You stopped before executing every concrete web/browser operation required "
        f"by this request. Still missing in this task: {', '.join(missing)}. Call those "
        "exact tools now, preserve the current browser session_id across calls, ground "
        "the answer in their results, and make browser_close the last browser operation. "
        "For a form submission, keep one coherent page session and execute every field "
        "fill, then click, wait for confirmation, read the confirmation content, and only "
        "then close; if the page was already closed or reopened, repeat that sequence. "
        "Do not substitute web_fetch, shell, or desktop tools for rendered-page interaction."
        f"{anchor}"
    )


_VIDEO_GENERATION_REQUEST_RE = re.compile(
    r"\b(?:generate|create|make|animate|render|produce)\b[^\n]{0,100}\bvideo\b"
    r"|\bvideo\b[^\n]{0,100}\b(?:using|with)\b[^\n]{0,40}\b(?:wan|svd|animatediff)\b"
    r"|\banimate\b[^\n]{0,100}\b(?:image|clip)\b",
    re.I,
)


def find_unmet_video_generation_request(
    user_input: str,
    tool_names_used: list[str],
    available_tool_names: set[str] | frozenset[str] | None = None,
) -> list[str]:
    """Require ``video_generate`` evidence for an actionable render request."""
    if available_tool_names is not None and "video_generate" not in available_tool_names:
        return []
    if not _VIDEO_GENERATION_REQUEST_RE.search(user_input):
        return []
    return [] if "video_generate" in tool_names_used else ["video_generate"]


def make_video_generation_retry_prompt(user_input: str = "") -> str:
    """Return a correction requiring the requested render tool call."""
    anchor = f"\n\nThe original request is:\n{user_input}" if user_input else ""
    return (
        "This request requires calling video_generate, including when the requested "
        "parameter combination is expected to be rejected. Call video_generate once "
        "with the requested backend and parameters, then report its actual result. Do "
        "not substitute video_storyboard or explain a predicted validation error without "
        f"executing the tool.{anchor}"
    )


_TERMINAL_PARAMETER_ERROR_RE = re.compile(
    r"\b(?:mutually exclusive|requires? [`'\"]?[a-z_]+|text-to-video only|"
    r"image-to-video only|must (?:be|provide|pass|specify|choose)|"
    r"invalid (?:parameter|value|backend|model)|unsupported (?:parameter|value|backend))\b",
    re.I,
)
_ERROR_REPORT_RE = re.compile(
    r"\b(?:cannot|can't|could not|failed|error|invalid|unsupported|requires?|"
    r"mutually exclusive|only|choose|instead|not allowed|refused|rejected|"
    r"timed out|timeout|missing|unavailable|no gpu)\b",
    re.I,
)


def terminal_parameter_errors_are_reported(errors: list[str], final_text: str) -> bool:
    """Return whether terminal tool failures were honestly relayed.

    Any ``video_generate`` failure is terminal once reported: blindly retrying
    a minutes-long GPU operation can duplicate jobs, defeat an explicit timeout,
    or churn on an operator-actionable missing-model/GPU error. Other tools are
    terminal here only for deterministic parameter refusals.
    """
    video_generation_errors = all(
        error.casefold().startswith("video_generate:") for error in errors
    )
    return bool(
        errors
        and (
            video_generation_errors
            or all(_TERMINAL_PARAMETER_ERROR_RE.search(error) for error in errors)
        )
        and _ERROR_REPORT_RE.search(final_text)
    )


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
