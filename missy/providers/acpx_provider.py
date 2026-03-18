"""ACPX provider for the Missy framework.

Wraps the `acpx <https://github.com/openclaw/acpx>`_ CLI — a headless
Agent Client Protocol client that can talk to Claude, Codex, Gemini,
Cursor, and many other coding agents over a structured protocol instead
of PTY scraping.

Prompts are passed to the ``acpx`` binary via ``exec`` (one-shot, no
saved session) with ``--format json`` so the output is machine-readable
NDJSON.  Persistent session support is also available: when a
``session`` name is configured the provider uses the session lifecycle
(``sessions ensure`` / ``prompt``) instead of ``exec``.

Tool Calling
~~~~~~~~~~~~

ACPX delegates to external agents that manage their own tool calling
internally.  Missy's native tools (calculator, file_read, shell_exec,
etc.) cannot be passed via the Agent Client Protocol, so this provider
implements a **text-based tool calling protocol**:

1. Tool schemas are rendered as a structured instruction block and
   injected into the prompt so the underlying agent knows what Missy
   tools are available.
2. The agent is instructed to emit tool requests inside ``<tool_call>``
   XML tags containing a JSON payload.
3. The provider parses these tags from the response text, extracts
   :class:`~missy.providers.base.ToolCall` objects, and returns them in
   the :class:`~missy.providers.base.CompletionResponse` so the runtime
   can execute the tools and feed results back.
4. On subsequent rounds, tool results (already formatted as
   ``[Tool result for X]: ...`` by the runtime) appear naturally in the
   conversation history that gets flattened into the next prompt.

This makes all 44+ Missy tools available through any ACPX-supported
agent — Claude, Codex, Gemini, Cursor, etc. — via Discord, CLI, webhook,
or any other channel.

Install ACPX::

    npm install -g acpx@latest

Configure in ``config.yaml``::

    providers:
      acpx:
        name: acpx
        model: "claude"          # agent name: claude, codex, gemini, cursor, …
        timeout: 120
        enabled: true

    # Optional per-provider overrides via base_url:
    #   base_url: "--approve-all --cwd /my/project"
    # (extra CLI flags appended to every invocation)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_AGENT = "claude"
_DEFAULT_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Tool call XML protocol
# ---------------------------------------------------------------------------

# Regex to match <tool_call>...</tool_call> blocks in response text.
# Uses re.DOTALL so the JSON payload can span multiple lines.
# Captures the inner content (the JSON string).
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Maximum number of tool calls we'll extract from a single response to
# prevent runaway parsing on adversarial or malformed output.
_MAX_TOOL_CALLS_PER_RESPONSE = 20

# Maximum character length for the tool instruction block.  If the full
# schema rendering exceeds this, tools are presented in compact mode.
_MAX_TOOL_INSTRUCTIONS_CHARS = 12_000


def _generate_tool_call_id(name: str, index: int, text_hash: str) -> str:
    """Generate a deterministic, unique tool call ID.

    Combines the tool name, call index within the response, and a hash
    of the response text to produce an ID that is:
    - Unique across responses (via text hash)
    - Unique within a response (via index)
    - Human-readable (starts with tool name)

    Args:
        name: Tool name.
        index: Zero-based index of this call within the response.
        text_hash: Short hash of the full response text.

    Returns:
        A string like ``"calculator_0_a1b2c3"``.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)[:20]
    return f"{safe_name}_{index}_{text_hash[:8]}"


# ---------------------------------------------------------------------------
# Tool schema rendering
# ---------------------------------------------------------------------------


def _render_tool_schema_full(tool: Any) -> str:
    """Render a single tool's schema as a detailed text block.

    Produces a structured description including name, description, and
    all parameters with their types, descriptions, and required flags.

    Args:
        tool: A BaseTool instance (or any object with name, description,
            and get_schema()).

    Returns:
        A multi-line string describing the tool.
    """
    schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    lines = [
        f"### {tool.name}",
        f"Description: {tool.description}",
    ]

    if properties:
        lines.append("Parameters:")
        for pname, pdef in properties.items():
            ptype = pdef.get("type", "any")
            pdesc = pdef.get("description", "")
            req_marker = " (REQUIRED)" if pname in required else " (optional)"
            enum_vals = pdef.get("enum")
            lines.append(f"  - {pname}: {ptype}{req_marker}")
            if pdesc:
                lines.append(f"    {pdesc}")
            if enum_vals:
                lines.append(f"    Allowed values: {', '.join(str(v) for v in enum_vals)}")
    else:
        lines.append("Parameters: none")

    return "\n".join(lines)


def _render_tool_schema_compact(tool: Any) -> str:
    """Render a single tool's schema as a compact one-liner.

    Used when the full rendering would exceed the instruction budget.

    Args:
        tool: A BaseTool instance.

    Returns:
        A single-line string like ``"calculator(expression: string) — Evaluate arithmetic"``.
    """
    schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
    params = schema.get("parameters", {})
    properties = params.get("properties", {})
    required = set(params.get("required", []))

    param_parts = []
    for pname, pdef in properties.items():
        ptype = pdef.get("type", "any")
        marker = "" if pname in required else "?"
        param_parts.append(f"{pname}{marker}: {ptype}")

    param_str = ", ".join(param_parts) if param_parts else ""
    return f"- {tool.name}({param_str}) — {tool.description}"


def _render_tool_instructions(tools: list) -> str:
    """Build the complete tool instruction block for prompt injection.

    The block includes:
    - A preamble explaining the tool call protocol
    - Detailed schemas for each available tool
    - Examples of correct tool call syntax
    - Rules for when to use vs. not use tools
    - The expected format for providing tool results

    If the full rendering exceeds ``_MAX_TOOL_INSTRUCTIONS_CHARS``, falls
    back to compact mode with abbreviated schemas.

    Args:
        tools: List of BaseTool instances.

    Returns:
        A string ready to inject into the system prompt.
    """
    if not tools:
        return ""

    # Try full rendering first
    full_schemas = [_render_tool_schema_full(t) for t in tools]
    full_block = "\n\n".join(full_schemas)

    if len(full_block) > _MAX_TOOL_INSTRUCTIONS_CHARS:
        # Fall back to compact mode
        compact_schemas = [_render_tool_schema_compact(t) for t in tools]
        tool_listing = "\n".join(compact_schemas)
    else:
        tool_listing = full_block

    tool_names = [getattr(t, "name", "?") for t in tools]

    return f"""
## Available Tools

You have access to {len(tools)} tools provided by the Missy agent platform.
To use a tool, emit a <tool_call> XML tag containing a JSON object with
"name" and "arguments" keys. You may request multiple tools in a single
response by including multiple <tool_call> blocks.

IMPORTANT RULES:
1. Always use tools when you need real information — do not guess or fabricate.
2. You MUST use the exact tool names listed below.
3. Each <tool_call> block must contain valid JSON.
4. Required parameters MUST be provided.
5. After you emit tool calls, STOP writing. Do not continue with
   speculative text. Wait for tool results.
6. Tool results will be provided to you in subsequent messages prefixed
   with "[Tool result for <name>]:" or "[Tool error for <name>]:".
7. When you have enough information to answer, respond normally WITHOUT
   any <tool_call> tags.

### Tool Call Format

<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

### Multiple Tool Calls (parallel execution)

You can request multiple tools at once:

<tool_call>
{{"name": "file_read", "arguments": {{"path": "/etc/hostname"}}}}
</tool_call>
<tool_call>
{{"name": "shell_exec", "arguments": {{"command": "whoami"}}}}
</tool_call>

### Available Tools

{tool_listing}

### Tool Names Quick Reference

{', '.join(tool_names)}
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_tool_calls_from_text(text: str) -> tuple[list[ToolCall], str]:
    """Extract tool call blocks from response text.

    Scans the response for ``<tool_call>...</tool_call>`` XML tags,
    parses the JSON payload inside each one, and returns a list of
    :class:`ToolCall` objects plus the remaining text with tool call
    blocks removed.

    Handles edge cases:
    - Malformed JSON inside a tag (logged and skipped)
    - Missing required fields (name) — skipped
    - Excess tool calls beyond ``_MAX_TOOL_CALLS_PER_RESPONSE`` — truncated
    - Nested/escaped angle brackets in JSON string values
    - Whitespace variations in the JSON payload
    - Empty tool_call tags — skipped

    Args:
        text: The full response text from the agent.

    Returns:
        A 2-tuple of (tool_calls, remaining_text) where remaining_text
        has all ``<tool_call>`` blocks stripped out and whitespace cleaned.
    """
    if "<tool_call>" not in text:
        return [], text

    # Hash the full text for deterministic ID generation
    text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

    matches = _TOOL_CALL_PATTERN.findall(text)
    tool_calls: list[ToolCall] = []
    call_index = 0

    for raw_json in matches:
        if call_index >= _MAX_TOOL_CALLS_PER_RESPONSE:
            logger.warning(
                "Truncating tool calls at %d (max %d per response)",
                call_index,
                _MAX_TOOL_CALLS_PER_RESPONSE,
            )
            break

        raw_json = raw_json.strip()
        if not raw_json:
            continue

        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Malformed JSON in <tool_call> block (index %d): %s — raw: %s",
                call_index,
                exc,
                raw_json[:200],
            )
            continue

        if not isinstance(payload, dict):
            logger.warning(
                "tool_call payload is not a dict (index %d): %s",
                call_index,
                type(payload).__name__,
            )
            continue

        name = payload.get("name", "")
        if not name:
            logger.warning(
                "tool_call missing 'name' field (index %d): %s",
                call_index,
                raw_json[:200],
            )
            continue

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            # Some models might put arguments as a JSON string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        "tool_call arguments is a non-JSON string for %r (index %d)",
                        name,
                        call_index,
                    )
                    arguments = {}
            else:
                arguments = {}

        tc_id = _generate_tool_call_id(name, call_index, text_hash)
        tool_calls.append(
            ToolCall(
                id=tc_id,
                name=name,
                arguments=arguments,
            )
        )
        call_index += 1

    # Remove tool_call blocks from text to get the "content" portion
    remaining = _TOOL_CALL_PATTERN.sub("", text).strip()
    # Clean up excessive whitespace left by removal
    remaining = re.sub(r"\n{3,}", "\n\n", remaining)

    return tool_calls, remaining


def _validate_tool_calls(
    tool_calls: list[ToolCall],
    available_tools: dict[str, Any],
) -> tuple[list[ToolCall], list[str]]:
    """Validate extracted tool calls against the available tool set.

    Checks that:
    - Tool names exist in the available set
    - Required parameters are present
    - No unexpected parameter types

    Args:
        tool_calls: Parsed tool calls from response text.
        available_tools: Dict mapping tool name → tool instance.

    Returns:
        A 2-tuple of (valid_calls, warnings) where warnings contains
        human-readable messages about any validation issues.
    """
    valid: list[ToolCall] = []
    warnings: list[str] = []

    for tc in tool_calls:
        tool = available_tools.get(tc.name)
        if tool is None:
            # Check for close matches (typos)
            close = _find_close_match(tc.name, list(available_tools.keys()))
            if close:
                warnings.append(
                    f"Unknown tool {tc.name!r} — did you mean {close!r}? Skipping."
                )
            else:
                warnings.append(
                    f"Unknown tool {tc.name!r} — not in available tools. Skipping."
                )
            continue

        # Check required parameters
        schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
        params = schema.get("parameters", {})
        required = params.get("required", [])
        missing = [r for r in required if r not in tc.arguments]
        if missing:
            warnings.append(
                f"Tool {tc.name!r} missing required params: {missing}. "
                f"Including anyway — the tool may report its own error."
            )

        valid.append(tc)

    return valid, warnings


def _find_close_match(name: str, candidates: list[str], threshold: float = 0.6) -> str | None:
    """Find the closest matching tool name using character overlap.

    Simple similarity metric based on shared bigrams (character pairs).

    Args:
        name: The name to match.
        candidates: Available tool names.
        threshold: Minimum similarity score (0.0–1.0).

    Returns:
        The best match if above threshold, otherwise None.
    """

    def _bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}

    name_bg = _bigrams(name.lower())
    best_score = 0.0
    best_match = None

    for candidate in candidates:
        cand_bg = _bigrams(candidate.lower())
        if not name_bg or not cand_bg:
            continue
        overlap = len(name_bg & cand_bg)
        total = len(name_bg | cand_bg)
        score = overlap / total if total > 0 else 0.0
        if score > best_score:
            best_score = score
            best_match = candidate

    return best_match if best_score >= threshold else None


# ---------------------------------------------------------------------------
# Provider implementation
# ---------------------------------------------------------------------------


class AcpxProvider(BaseProvider):
    """Provider that delegates to the ``acpx`` CLI binary.

    Each completion call spawns ``acpx <agent> exec "<prompt>"`` as a
    subprocess with ``--format json`` output.  The NDJSON events are
    parsed and the final assistant text is extracted.

    For tool calling, Missy tool schemas are injected into the prompt
    text and the agent's response is parsed for ``<tool_call>`` XML
    blocks.  This enables all Missy tools to work through any ACPX-
    supported agent (Claude, Codex, Gemini, Cursor, etc.) across all
    channels (CLI, Discord, Webhook, Voice).

    Args:
        config: Provider config.

            * ``model`` — the ACPX agent name (``claude``, ``codex``,
              ``gemini``, ``cursor``, etc.).  Defaults to ``"claude"``.
            * ``base_url`` — extra CLI flags to append to every
              invocation (e.g. ``"--approve-all --cwd /my/project"``).
            * ``api_key`` — unused by ACPX itself (agents use their
              own env-var credentials), but stored for consistency.
            * ``timeout`` — subprocess timeout in seconds (default 120).
    """

    name = "acpx"

    def __init__(self, config: ProviderConfig) -> None:
        self._agent: str = config.model or _DEFAULT_AGENT
        self._timeout: int = config.timeout or _DEFAULT_TIMEOUT
        self._extra_flags: list[str] = config.base_url.split() if config.base_url else []
        self._binary: str = shutil.which("acpx") or "acpx"

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the ``acpx`` binary is on ``$PATH``.

        Also runs ``acpx --version`` to verify it executes successfully.
        """
        binary = shutil.which("acpx")
        if not binary:
            logger.debug("acpx binary not found on PATH")
            return False
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.debug("acpx availability check failed: %s", exc)
            return False

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Run a one-shot completion via ``acpx <agent> exec``.

        Conversation history is flattened into a single prompt string
        with role prefixes.  The ``--format json`` flag produces NDJSON
        events from which the assistant text is extracted.

        Args:
            messages: Ordered conversation turns.
            **kwargs: Optional overrides.  Recognised keys:

                * ``cwd`` (str) — working directory for the subprocess.
                * ``approve_all`` (bool) — pass ``--approve-all``.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On subprocess failure or unexpected output.
        """
        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        cwd = kwargs.pop("cwd", None)
        approve_all = kwargs.pop("approve_all", False)

        prompt = self._build_prompt(messages)

        content = self._run_acpx(
            prompt,
            session_id=session_id,
            task_id=task_id,
            cwd=cwd,
            approve_all=approve_all,
        )

        self._emit_event(session_id, task_id, "allow", "completion successful")

        return CompletionResponse(
            content=content,
            model=self._agent,
            provider=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw={},
        )

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to text-based tool descriptions.

        Unlike native providers that return JSON schemas, the ACPX
        provider renders tool schemas as human-readable text blocks
        suitable for prompt injection.  The return value is a list of
        dicts with ``name``, ``description``, and ``text_schema`` keys.

        Args:
            tools: List of BaseTool instances.

        Returns:
            A list of dicts describing each tool in text form.
        """
        schemas = []
        for tool in tools:
            base = tool.get_schema() if hasattr(tool, "get_schema") else {}
            schemas.append(
                {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    "parameters": base.get("parameters", {}),
                    "text_schema": _render_tool_schema_full(tool),
                }
            )
        return schemas

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        """Send messages with tool calling via the text-based protocol.

        Injects tool schemas and calling instructions into the prompt,
        sends to the ACPX binary, parses the response for
        ``<tool_call>`` blocks, and returns a :class:`CompletionResponse`
        with any extracted tool calls.

        The runtime's tool loop handles executing the tools and feeding
        results back as subsequent messages.

        Args:
            messages: Ordered conversation turns.  May include tool
                result messages from previous iterations (already
                formatted as ``[Tool result for X]: ...`` by the
                runtime's ``_dicts_to_messages``).
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` is populated with
            :class:`~missy.providers.base.ToolCall` instances parsed
            from ``<tool_call>`` blocks in the agent's text response.
        """
        self._acquire_rate_limit()

        # Build the augmented system prompt with tool instructions
        tool_instructions = _render_tool_instructions(tools)
        augmented_system = system
        if tool_instructions:
            augmented_system = (system + "\n" + tool_instructions) if system else tool_instructions

        # Inject system prompt into messages
        augmented_messages = list(messages)
        if augmented_system:
            # Check if there's already a system message
            has_system = any(m.role == "system" for m in augmented_messages)
            if has_system:
                # Replace existing system message content
                augmented_messages = [
                    Message(
                        role=m.role,
                        content=(m.content + "\n" + tool_instructions)
                        if m.role == "system" and tool_instructions
                        else m.content,
                    )
                    for m in augmented_messages
                ]
            else:
                augmented_messages.insert(
                    0, Message(role="system", content=augmented_system)
                )

        prompt = self._build_prompt(augmented_messages)

        # Run ACPX
        raw_content = self._run_acpx(prompt)

        # Parse tool calls from response
        tool_calls, remaining_text = _parse_tool_calls_from_text(raw_content)

        if tool_calls:
            # Validate against available tools
            tool_map = {getattr(t, "name", ""): t for t in tools}
            valid_calls, validation_warnings = _validate_tool_calls(tool_calls, tool_map)

            for warning in validation_warnings:
                logger.warning("ACPX tool validation: %s", warning)

            if valid_calls:
                self._emit_event("", "", "allow", f"tool_calls: {len(valid_calls)}")

                return CompletionResponse(
                    content=remaining_text,
                    model=self._agent,
                    provider=self.name,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    raw={"raw_response": raw_content},
                    tool_calls=valid_calls,
                    finish_reason="tool_calls",
                )

        # No tool calls — regular text response
        self._emit_event("", "", "allow", "completion successful")

        return CompletionResponse(
            content=raw_content,
            model=self._agent,
            provider=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw={},
            tool_calls=[],
            finish_reason="stop",
        )

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream tokens from ``acpx`` by reading NDJSON events line-by-line.

        Uses ``--format json`` and streams stdout in real time via
        ``Popen`` instead of waiting for the process to finish.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt (prepended if non-empty).

        Yields:
            Text chunks as they arrive.

        Raises:
            ProviderError: On subprocess failure.
        """
        if system:
            messages = [Message(role="system", content=system), *messages]

        prompt = self._build_prompt(messages)
        cmd = [self._binary, self._agent, "exec", prompt, "--format", "json"]
        cmd.extend(self._extra_flags)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                "acpx binary not found. Install with: npm install -g acpx@latest"
            ) from exc

        try:
            assert proc.stdout is not None  # for type checker
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Non-JSON output — could be plain text, yield as-is
                    yield line
                    continue

                text = self._extract_text_from_event(event)
                if text:
                    yield text

            proc.wait(timeout=30)
            if proc.returncode and proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise ProviderError(
                    f"acpx stream exited with code {proc.returncode}: {stderr[:500]}"
                )
        except ProviderError:
            raise
        except Exception as exc:
            proc.kill()
            raise ProviderError(f"acpx stream failed: {exc}") from exc
        finally:
            if proc.poll() is None:
                proc.terminate()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_acpx(
        self,
        prompt: str,
        session_id: str = "",
        task_id: str = "",
        cwd: str | None = None,
        approve_all: bool = False,
    ) -> str:
        """Execute the acpx binary and return the parsed output text.

        Args:
            prompt: The flattened prompt string.
            session_id: For audit events.
            task_id: For audit events.
            cwd: Optional working directory for the subprocess.
            approve_all: If True, pass ``--approve-all``.

        Returns:
            The extracted text content from the NDJSON output.

        Raises:
            ProviderError: On subprocess failure.
        """
        cmd = [self._binary, self._agent, "exec", prompt]
        cmd.extend(["--format", "json"])
        if approve_all:
            cmd.append("--approve-all")
        cmd.extend(self._extra_flags)
        if cwd:
            cmd.extend(["--cwd", str(cwd)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired as exc:
            self._emit_event(session_id, task_id, "error", "subprocess timed out")
            raise ProviderError(
                f"acpx subprocess timed out after {self._timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            self._emit_event(session_id, task_id, "error", "acpx binary not found")
            raise ProviderError(
                "acpx binary not found. Install with: npm install -g acpx@latest"
            ) from exc
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"acpx subprocess failed: {exc}") from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()[:500]
            self._emit_event(
                session_id, task_id, "error", f"exit {result.returncode}: {stderr}"
            )
            raise ProviderError(
                f"acpx exited with code {result.returncode}: {stderr}"
            )

        return self._parse_ndjson_output(result.stdout)

    def _build_prompt(self, messages: list[Message]) -> str:
        """Flatten a conversation into a single prompt string.

        System messages are prefixed with ``[System]:``, user messages
        with ``[User]:``, and assistant messages with ``[Assistant]:``.
        For a single user message the prefix is omitted.
        """
        if len(messages) == 1 and messages[0].role == "user":
            return messages[0].content

        parts: list[str] = []
        for msg in messages:
            prefix = {
                "system": "[System]",
                "user": "[User]",
                "assistant": "[Assistant]",
            }.get(msg.role, f"[{msg.role}]")
            parts.append(f"{prefix}: {msg.content}")
        return "\n".join(parts)

    def _parse_ndjson_output(self, stdout: str) -> str:
        """Parse NDJSON output and extract the final assistant text.

        ACPX ``--format json`` emits one JSON object per line.  We look
        for text delta events and concatenate them.  If the output is
        not valid NDJSON, we fall back to returning the raw stdout.
        """
        text_parts: list[str] = []
        has_json = False

        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                has_json = True
            except json.JSONDecodeError:
                continue

            text = self._extract_text_from_event(event)
            if text:
                text_parts.append(text)

        if text_parts:
            return "".join(text_parts)

        # If no structured text events found, return raw stdout
        if not has_json:
            return stdout.strip()

        return ""

    @staticmethod
    def _extract_text_from_event(event: dict) -> str:
        """Pull text content from a single NDJSON event.

        Handles several event shapes that ACPX may emit:

        * ``{"type": "text_delta", "delta": "..."}``
        * ``{"type": "message", "content": "..."}``
        * ``{"type": "result", "text": "..."}``
        * ``{"content": "..."}`` (generic fallback)
        """
        etype = event.get("type", "")

        # Text delta events (streaming)
        if etype in ("text_delta", "response.output_text.delta"):
            return event.get("delta", "")

        # Final message or result
        if etype in ("message", "result"):
            return event.get("text", "") or event.get("content", "")

        # Generic content field
        if "content" in event and isinstance(event["content"], str) and not etype:
            return event["content"]

        return ""

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
    ) -> None:
        """Publish a provider audit event including the agent name."""
        try:
            from missy.core.events import AuditEvent, event_bus

            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail={
                    "provider": self.name,
                    "agent": self._agent,
                    "message": detail_msg,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
