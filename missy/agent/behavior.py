"""Humanistic behavior layer for the Missy agent.

Shapes agent responses to feel natural, context-aware, and non-robotic.
Three cooperating classes handle distinct concerns:

- :class:`BehaviorLayer`: Augments system prompts and produces per-turn
  behavioral guidelines by reading the current context and persona.
- :class:`IntentInterpreter`: Classifies user intent and urgency with
  keyword/pattern heuristics (no ML required).
- :class:`ResponseShaper`: Post-processes raw LLM output to strip robotic
  artifacts without touching code blocks or technical content.

The shared *context dict* passed through all public APIs has this shape::

    {
        "user_tone":       str,   # from BehaviorLayer.analyze_user_tone()
        "topic":           str,   # current conversation topic
        "turn_count":      int,   # turns elapsed in this session
        "has_tool_results": bool, # whether tool results are present
        "intent":          str,   # from IntentInterpreter.classify_intent()
        "urgency":         str,   # from IntentInterpreter.extract_urgency()
    }

Example::

    from missy.agent.behavior import BehaviorLayer, IntentInterpreter, ResponseShaper
    from missy.agent.persona import PersonaConfig

    persona = PersonaConfig(name="Missy", tone="warm")
    layer = BehaviorLayer(persona)
    interpreter = IntentInterpreter()
    shaper = ResponseShaper()

    messages = [{"role": "user", "content": "hey, quick q — how do i restart nginx?"}]
    tone = layer.analyze_user_tone(messages)
    intent = interpreter.classify_intent(messages[-1]["content"])
    urgency = interpreter.extract_urgency(messages[-1]["content"])

    ctx = {
        "user_tone": tone, "topic": "nginx", "turn_count": 1,
        "has_tool_results": False, "intent": intent, "urgency": urgency,
    }

    system = layer.shape_system_prompt("You are a helpful assistant.", ctx)
    guidelines = layer.get_response_guidelines(ctx)
    raw_response = "As an AI language model, I can help you restart nginx by..."
    clean = shaper.shape_response(raw_response, persona, ctx)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from missy.agent.persona import PersonaConfig


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Tone detection keyword sets
_CASUAL_SIGNALS: frozenset[str] = frozenset(
    {
        "hey",
        "hi",
        "yo",
        "sup",
        "lol",
        "haha",
        "cool",
        "awesome",
        "thanks",
        "thx",
        "ty",
        "np",
        "btw",
        "fyi",
        "gonna",
        "wanna",
        "kinda",
        "sorta",
        "ngl",
        "tbh",
        "imo",
        "imho",
        "asap",
        "dunno",
        "ya",
        "yep",
        "nope",
    }
)

_FORMAL_SIGNALS: frozenset[str] = frozenset(
    {
        "please",
        "kindly",
        "would",
        "could",
        "appreciate",
        "regarding",
        "furthermore",
        "however",
        "therefore",
        "nevertheless",
        "accordingly",
        "sincerely",
        "respectfully",
        "pursuant",
        "henceforth",
    }
)

_FRUSTRATED_SIGNALS: frozenset[str] = frozenset(
    {
        "wrong",
        "broken",
        "doesn't work",
        "not working",
        "still",
        "again",
        "useless",
        "failed",
        "why",
        "terrible",
        "ridiculous",
        "ugh",
        "argh",
        "wtf",
        "seriously",
        "frustrated",
        "annoying",
        "waste",
    }
)

_TECHNICAL_SIGNALS: frozenset[str] = frozenset(
    {
        "function",
        "class",
        "method",
        "api",
        "endpoint",
        "request",
        "response",
        "query",
        "schema",
        "config",
        "yaml",
        "json",
        "xml",
        "bash",
        "shell",
        "script",
        "module",
        "import",
        "package",
        "library",
        "framework",
        "database",
        "sql",
        "index",
        "cache",
        "token",
        "auth",
        "oauth",
        "async",
        "await",
        "thread",
        "process",
        "port",
        "socket",
        "ssl",
        "tls",
        "docker",
        "kubernetes",
        "container",
        "deployment",
        "pipeline",
        "ci",
    }
)

# Intent classification patterns
_GREETING_PATTERNS: re.Pattern[str] = re.compile(
    r"^\s*(hey|hi|hello|good\s+(morning|afternoon|evening)|howdy|greetings|sup|yo)\b",
    re.IGNORECASE,
)

_FAREWELL_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(bye|goodbye|see\s+you|later|farewell|take\s+care|cya|ttyl|signing\s+off)\b",
    re.IGNORECASE,
)

_QUESTION_PATTERNS: re.Pattern[str] = re.compile(
    r"(\?|^(what|who|where|when|why|how|which|is|are|do|does|can|could|would|should|will)\b)",
    re.IGNORECASE,
)

_COMMAND_PATTERNS: re.Pattern[str] = re.compile(
    r"^(please\s+)?(do|run|execute|start|stop|restart|create|delete|remove|add|update"
    r"|show|list|find|check|fix|open|close|make|set|get|install|deploy|write|read)\b",
    re.IGNORECASE,
)

_CLARIFICATION_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(what\s+do\s+you\s+mean|can\s+you\s+clarify|could\s+you\s+elaborate"
    r"|i\s+don'?t\s+understand|not\s+sure\s+what|please\s+explain"
    r"|what\s+exactly|more\s+detail)\b",
    re.IGNORECASE,
)

_FEEDBACK_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(that'?s?\s+(great|perfect|good|wrong|bad|off|incorrect|right|helpful"
    r"|not\s+what)|you\s+(got\s+it|missed|didn'?t)|not\s+quite|almost|close"
    r"|actually|correction|mistake)\b",
    re.IGNORECASE,
)

_EXPLORATION_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(tell\s+me\s+(more|about)|i'?m?\s+(curious|interested|wondering)"
    r"|what\s+(else|other)|could\s+you\s+(also|additionally)|explore"
    r"|ideas?\s+for|suggest|recommend|options?)\b",
    re.IGNORECASE,
)

_TROUBLESHOOT_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(error|exception|traceback|stack\s*trace|log|debug|diagnose|troubleshoot"
    r"|segfault|core\s*dump|panic|errno|exit\s+code|return\s+code|status\s+code"
    r"|failing|failed\s+with|throws?|raised?|caught|unhandled"
    r"|timeout|connection\s+refused|permission\s+denied)\b",
    re.IGNORECASE,
)

_CONFIRMATION_PATTERNS: re.Pattern[str] = re.compile(
    r"^\s*(ok|okay|k|yes|yeah|yep|yup|sure|got\s+it|sounds?\s+good"
    r"|makes?\s+sense|understood|perfect|great|go\s+ahead|proceed"
    r"|do\s+it|confirmed?|approved?|agreed|right)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_FRUSTRATION_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(still\s+not|doesn'?t\s+work|not\s+working|tried\s+(that|this|again)"
    r"|you\s+already|i\s+already|same\s+(error|issue|problem)|again|why\s+(is|isn'?t"
    r"|won'?t|can'?t)|nothing\s+works?|useless)\b",
    re.IGNORECASE,
)

# Urgency patterns
_HIGH_URGENCY_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(asap|immediately|right\s+now|urgent|emergency|critical|production\s+down"
    r"|outage|broken|crash|down|failing|blocking|stop(ped)?|not\s+working)\b",
    re.IGNORECASE,
)

_MEDIUM_URGENCY_PATTERNS: re.Pattern[str] = re.compile(
    r"\b(soon|today|quickly|fast|hurry|need\s+to|have\s+to|must|important"
    r"|deadline|before\s+(end|close|morning|tonight))\b",
    re.IGNORECASE,
)

# Robotic phrases to strip from responses
_ROBOTIC_PHRASES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"As an AI(?: language model)?[,.]?\s*",
        r"As a large language model[,.]?\s*",
        r"I(?:'m| am) an AI(?: assistant)?[,.]?\s*",
        r"I don'?t have (?:feelings?|emotions?|personal opinions?)[,.]?\s*",
        r"I(?:'m| am) not capable of (?:feeling|experiencing)[^.]*\.\s*",
        r"I cannot (?:provide|give|offer) (?:personal|actual) advice[^.]*\.\s*",
        r"(?:Please note|Note) that I(?:'m| am) an AI[^.]*\.\s*",
        r"As (?:your|an?) (?:AI |virtual |digital )?assistant[,.]?\s*",
        r"I(?:'m| am) here to (?:help|assist)(?: you)?[,.]?\s*",
        r"Certainly[,!]\s*(?:I(?:'ll| will) (?:help|assist)(?: you)?[!.]?\s*)?",
        r"Of course[,!]\s*(?:I(?:'ll| will) (?:help|assist)(?: you)?[!.]?\s*)?",
        r"Absolutely[,!]\s*(?:I(?:'ll| will) (?:help|assist)(?: you)?[!.]?\s*)?",
        r"Great question[,!]\s*",
        r"That(?:'s| is) a great question[,!]\s*",
        r"I(?:'d| would) be happy to (?:help|assist)(?: you)?[,.]?\s*",
    ]
]

# Code block detector — used to skip modification inside fenced blocks
_CODE_BLOCK_RE: re.Pattern[str] = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`",
    re.DOTALL,
)

# Tone → response tone mapping
_TONE_ADAPTATION_MAP: dict[str, str] = {
    "casual": (
        "Match the user's relaxed register. Keep language conversational and "
        "skip unnecessary formality. Short sentences work well."
    ),
    "formal": (
        "Maintain a professional, precise tone. Use complete sentences and avoid colloquialisms."
    ),
    "frustrated": (
        "Acknowledge the difficulty directly before diving into solutions. "
        "Be empathetic, get to the point quickly, and avoid lengthy preamble."
    ),
    "technical": (
        "Use accurate technical vocabulary freely. Omit over-explaining basics "
        "unless asked. Code examples are welcome."
    ),
    "brief": (
        "Be as concise as possible. Bullet points or a single direct answer "
        "are preferred over long paragraphs."
    ),
    "verbose": (
        "The user communicates in detail; thorough explanations with examples are appropriate."
    ),
}


# ---------------------------------------------------------------------------
# BehaviorLayer
# ---------------------------------------------------------------------------


class BehaviorLayer:
    """Shapes system prompts and per-turn guidelines to reflect the active persona.

    Args:
        persona: Optional :class:`~missy.agent.persona.PersonaConfig`. When
            ``None``, sensible defaults apply.

    Example::

        layer = BehaviorLayer()
        ctx = {"user_tone": "casual", "turn_count": 3, "has_tool_results": False,
               "topic": "python", "intent": "question", "urgency": "low"}
        prompt = layer.shape_system_prompt("You are helpful.", ctx)
    """

    def __init__(self, persona: PersonaConfig | None = None) -> None:
        self._persona = persona

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shape_system_prompt(
        self,
        base_prompt: str,
        context: dict | None = None,
    ) -> str:
        """Enhance *base_prompt* with persona and contextual behavioral instructions.

        Args:
            base_prompt: The raw system prompt string produced by the runtime.
            context: Optional context dict describing the current conversation
                state (see module docstring for keys).

        Returns:
            The enriched system prompt. The base text is always preserved; all
            additions are appended as clearly delimited sections.
        """
        ctx = context or {}
        sections: list[str] = [base_prompt.rstrip()]

        # Persona identity block
        persona_block = self._build_persona_block()
        if persona_block:
            sections.append(persona_block)

        # Behavioral guidelines for this turn
        guidelines = self.get_response_guidelines(ctx)
        if guidelines:
            sections.append(f"## Response guidelines\n{guidelines}")

        return "\n\n".join(sections)

    def analyze_user_tone(self, messages: list[dict]) -> str:
        """Infer the user's tone from recent messages.

        Inspects the most recent five user messages. Returns one of:
        ``"casual"``, ``"formal"``, ``"frustrated"``, ``"technical"``,
        ``"brief"``, or ``"verbose"``.

        Args:
            messages: Conversation history as a list of dicts with ``"role"``
                and ``"content"`` keys.

        Returns:
            A tone label string.
        """
        user_texts = [
            m.get("content", "")
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ][-5:]

        if not user_texts:
            return "casual"

        combined = " ".join(user_texts).lower()
        words = combined.split()
        word_set = frozenset(words)

        # Frustration takes priority — it overrides tone when clearly present
        if _FRUSTRATED_SIGNALS & word_set or _FRUSTRATION_PATTERNS.search(combined):
            return "frustrated"

        # Score the remaining tones
        scores: dict[str, float] = {
            "casual": 0.0,
            "formal": 0.0,
            "technical": 0.0,
        }

        casual_hits = len(_CASUAL_SIGNALS & word_set)
        formal_hits = len(_FORMAL_SIGNALS & word_set)
        technical_hits = len(_TECHNICAL_SIGNALS & word_set)

        scores["casual"] = casual_hits
        scores["formal"] = formal_hits
        scores["technical"] = technical_hits

        # Brief vs verbose based on average message length
        avg_len = sum(len(t.split()) for t in user_texts) / len(user_texts)
        if avg_len < 8:
            return "brief"
        if avg_len > 40:
            return "verbose"

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0.0:
            return "casual"
        return best

    def get_response_guidelines(self, context: dict) -> str:
        """Produce a plain-text block of behavioral guidelines for this turn.

        Guidelines adapt to user tone, conversation length, tool usage, and
        urgency signals so the agent can calibrate depth and style dynamically.

        Args:
            context: Context dict (see module docstring).

        Returns:
            A newline-separated string of concise directive sentences.
        """
        lines: list[str] = []
        ctx = context or {}

        user_tone: str = ctx.get("user_tone", "casual")
        has_tool_results: bool = bool(ctx.get("has_tool_results", False))
        topic: str = ctx.get("topic", "")
        urgency: str = ctx.get("urgency", "low")
        intent: str = ctx.get("intent", "")

        # Tone adaptation
        tone_guidance = self.get_tone_adaptation(user_tone)
        if tone_guidance:
            lines.append(tone_guidance)

        # Length guidance based on conversation depth
        if self.should_be_concise(ctx):
            lines.append(
                "The conversation is long — keep your answer concise and avoid "
                "restating context the user already knows."
            )

        # Tool result acknowledgement
        if has_tool_results:
            lines.append(
                "You have tool results available. Weave them into your reply "
                "naturally rather than dumping raw output."
            )

        # Urgency-driven tone
        if urgency == "high":
            lines.append(
                "The user seems to be under time pressure. Lead with the answer, "
                "then provide detail. Skip lengthy preamble."
            )
        elif urgency == "medium":
            lines.append("Keep the response focused and actionable.")

        # Intent-specific guidance
        if intent == "frustration":
            lines.append(
                "Acknowledge the difficulty with empathy before offering "
                "solutions. Avoid defensiveness."
            )
        elif intent == "exploration":
            lines.append(
                "The user is open to ideas — you may offer related suggestions "
                "and proactively surface useful context."
            )
        elif intent == "greeting":
            lines.append("Respond warmly and briefly. Match the energy of the greeting.")
        elif intent == "farewell":
            lines.append("Offer a friendly, brief farewell. No need to recap the session.")
        elif intent == "troubleshooting":
            lines.append(
                "The user is debugging an issue. Structure your response as: "
                "likely cause → diagnostic steps → fix. Include exact commands "
                "or code where applicable."
            )
        elif intent == "confirmation":
            lines.append(
                "The user is confirming or acknowledging. Proceed with the "
                "next action rather than restating what was already agreed upon."
            )
        elif intent == "clarification":
            lines.append(
                "The user needs more detail. Re-explain using a different "
                "approach — analogies, examples, or step-by-step breakdowns."
            )
        elif intent == "command":
            lines.append(
                "Direct instruction detected. Execute the requested action "
                "and report the result concisely."
            )

        # Vision-specific guidance
        vision_mode = ctx.get("vision_mode", "")
        if vision_mode == "painting":
            lines.append(
                "You are reviewing artwork. Be warm, encouraging, and supportive. "
                "Lead with genuine appreciation. Frame suggestions as exciting "
                "possibilities, not corrections. Never use harsh or dismissive "
                "language. End with encouragement about their artistic journey."
            )
        elif vision_mode == "puzzle":
            lines.append(
                "You are helping with a puzzle. Be patient and observant. Give "
                "specific, actionable placement suggestions. Reference actual "
                "colors and patterns you see. Celebrate progress made."
            )
        elif vision_mode:
            lines.append(
                "Visual input is available. Reference specific visual details "
                "in your response. Be observant and specific."
            )

        # Topic-based technical depth
        if topic:
            topic_lower = topic.lower()
            if any(w in topic_lower for w in ("code", "script", "function", "class", "api")):
                lines.append(
                    "Technical topic detected — include concrete examples or "
                    "code snippets where they add clarity."
                )

        # Persona-specific behavioral tendencies
        if self._persona is not None:
            for tendency in self._persona.behavioral_tendencies or []:
                lines.append(tendency)

        # Persona-specific response style rules
        if self._persona is not None:
            for rule in self._persona.response_style_rules or []:
                lines.append(rule)

        return "\n".join(f"- {line}" for line in lines if line.strip())

    def should_be_concise(self, context: dict | None) -> bool:
        """Return ``True`` when the response should be kept brief.

        Conciseness is recommended after 10 or more turns, when the detected
        user tone is ``"brief"``, or when urgency is ``"high"``.

        Args:
            context: Context dict (see module docstring).  May be ``None``.
        """
        ctx = context or {}
        turn_count: int = ctx.get("turn_count", 0)
        user_tone: str = ctx.get("user_tone", "")
        urgency: str = ctx.get("urgency", "low")

        return turn_count >= 10 or user_tone == "brief" or urgency == "high"

    def get_tone_adaptation(self, user_tone: str) -> str:
        """Map a user tone label to a recommended response tone directive.

        Args:
            user_tone: One of the labels returned by :meth:`analyze_user_tone`.

        Returns:
            A directive string, or an empty string for unknown tone values.
        """
        return _TONE_ADAPTATION_MAP.get(user_tone, "")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_persona_block(self) -> str:
        """Render the persona definition as a system prompt section."""
        p = self._persona
        if p is None:
            return ""

        parts: list[str] = ["## Persona"]

        if p.name:
            parts.append(f"Your name is {p.name}.")

        if p.identity_description:
            parts.append(p.identity_description)

        if p.tone:
            parts.append(f"Your overall tone is {p.tone}.")

        traits = getattr(p, "personality_traits", None) or []
        if traits:
            trait_list = ", ".join(traits)
            parts.append(f"Personality traits: {trait_list}.")

        boundaries = getattr(p, "boundaries", None) or []
        if boundaries:
            parts.append("Hard boundaries you must always respect:")
            parts.extend(f"- {b}" for b in boundaries)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# IntentInterpreter
# ---------------------------------------------------------------------------


class IntentInterpreter:
    """Classify user intent and urgency with keyword/pattern heuristics.

    No external dependencies or ML models are required.

    Example::

        interp = IntentInterpreter()
        assert interp.classify_intent("hey there!") == "greeting"
        assert interp.extract_urgency("production is down!") == "high"
    """

    def classify_intent(self, user_input: str) -> str:
        """Return the most likely intent category for *user_input*.

        Categories, in evaluation order:
        ``"greeting"``, ``"farewell"``, ``"confirmation"``, ``"frustration"``,
        ``"troubleshooting"``, ``"clarification"``, ``"feedback"``,
        ``"exploration"``, ``"command"``, ``"question"``.
        Falls back to ``"question"`` when no pattern matches.

        Args:
            user_input: Raw user message text.

        Returns:
            One of the ten intent category strings.
        """
        text = user_input.strip()

        if _GREETING_PATTERNS.match(text):
            return "greeting"

        if _FAREWELL_PATTERNS.search(text):
            return "farewell"

        if _CONFIRMATION_PATTERNS.match(text):
            return "confirmation"

        if _FRUSTRATION_PATTERNS.search(text):
            return "frustration"

        if _TROUBLESHOOT_PATTERNS.search(text):
            return "troubleshooting"

        if _CLARIFICATION_PATTERNS.search(text):
            return "clarification"

        if _FEEDBACK_PATTERNS.search(text):
            return "feedback"

        if _EXPLORATION_PATTERNS.search(text):
            return "exploration"

        if _COMMAND_PATTERNS.match(text):
            return "command"

        if _QUESTION_PATTERNS.search(text):
            return "question"

        # Default: treat unrecognised input as a question
        return "question"

    def extract_urgency(self, user_input: str) -> str:
        """Return an urgency level for *user_input*.

        Args:
            user_input: Raw user message text.

        Returns:
            ``"high"``, ``"medium"``, or ``"low"``.
        """
        if _HIGH_URGENCY_PATTERNS.search(user_input):
            return "high"
        if _MEDIUM_URGENCY_PATTERNS.search(user_input):
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# ResponseShaper
# ---------------------------------------------------------------------------


class ResponseShaper:
    """Post-process raw LLM output to remove robotic artifacts.

    The shaper never modifies content inside fenced or inline code blocks, so
    technical responses are always left intact.

    Example::

        shaper = ResponseShaper()
        raw = "Certainly! As an AI, I can help you with that. Here is the answer."
        clean = shaper.shape_response(raw, persona=None, context={})
        # "Here is the answer."
    """

    def shape_response(
        self,
        response: str,
        persona: PersonaConfig | None,
        context: dict,
    ) -> str:
        """Apply humanising post-processing to *response*.

        Steps performed (in order):

        1. Extract and stash code blocks so they are never touched.
        2. Remove robotic preamble phrases.
        3. Trim leading/trailing blank lines.
        4. Restore code blocks.

        Args:
            response: Raw LLM response text.
            persona: Optional persona (reserved for future style enforcement).
            context: Current context dict (unused currently; reserved for
                adaptive rules).

        Returns:
            The cleaned response string.
        """
        if not response:
            return response

        # Stash code blocks by replacing them with unique placeholders
        stash: list[str] = []

        def _stash_block(m: re.Match[str]) -> str:
            stash.append(m.group(0))
            return f"\x00CODE_BLOCK_{len(stash) - 1}\x00"

        working = _CODE_BLOCK_RE.sub(_stash_block, response)

        # Strip robotic phrases
        for pattern in _ROBOTIC_PHRASES:
            working = pattern.sub("", working)

        # Collapse multiple blank lines to at most one
        working = re.sub(r"\n{3,}", "\n\n", working)

        # Restore code blocks
        for i, block in enumerate(stash):
            working = working.replace(f"\x00CODE_BLOCK_{i}\x00", block)

        return working.strip()

    def detect_robotic_patterns(self, text: str) -> list[str]:
        """Return a list of robotic phrases detected in *text*.

        Useful for auditing or testing. Does not modify *text*.

        Args:
            text: Arbitrary text to inspect.

        Returns:
            A list of matched substrings (may be empty).
        """
        found: list[str] = []
        for pattern in _ROBOTIC_PHRASES:
            for match in pattern.finditer(text):
                matched = match.group(0).strip()
                if matched:
                    found.append(matched)
        return found
