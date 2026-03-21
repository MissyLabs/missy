"""Structured output validation for the Missy agent.

Enforces Pydantic model schemas on LLM responses, with automatic retry and
error-feedback when the model produces malformed or invalid JSON.

Example::

    from missy.agent.structured_output import OutputSchema, StructuredOutputRunner
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int
        reasoning: str

    schema = OutputSchema(Answer)
    runner = StructuredOutputRunner(provider)
    result = runner.complete_structured(messages, schema, system="You are helpful.")
    if result.success:
        print(result.data.value)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from missy.providers.base import BaseProvider, Message

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult(Generic[T]):
    """Result of a structured output attempt.

    Attributes:
        success: ``True`` when the response was parsed and validated without errors.
        data: Parsed and validated Pydantic model instance, or ``None`` on failure.
        raw_content: The original LLM response string.
        validation_errors: Human-readable description of each validation error
            encountered during the final attempt.
        attempts: Total number of provider calls made.
    """

    success: bool
    data: T | None
    raw_content: str
    validation_errors: list[str] = field(default_factory=list)
    attempts: int = 1


class OutputSchema(Generic[T]):
    """Wraps a Pydantic model as an output schema directive for the LLM.

    Args:
        model_class: The Pydantic ``BaseModel`` subclass that responses must
            conform to.
        description: Optional human-readable description injected into the
            prompt instruction.  Defaults to the model's docstring.
        max_retries: Maximum number of additional provider calls made after
            the first attempt when validation fails (default ``2``).
        strict: When ``True``, extra fields in the response are forbidden.
            Passed to :meth:`pydantic.BaseModel.model_validate` as
            ``strict=True``.

    Example::

        class Greeting(BaseModel):
            message: str
            language: str

        schema = OutputSchema(Greeting, max_retries=1)
        instruction = schema.to_prompt_instruction()
        result = schema.parse('{"message": "hello", "language": "en"}')
    """

    def __init__(
        self,
        model_class: type[T],
        description: str = "",
        max_retries: int = 2,
        strict: bool = False,
    ) -> None:
        self.model_class = model_class
        self.description = description or (model_class.__doc__ or "").strip()
        self.max_retries = max_retries
        self.strict = strict

    def to_json_schema(self) -> dict:
        """Return the JSON Schema dict generated from the Pydantic model."""
        return self.model_class.model_json_schema()

    def to_prompt_instruction(self) -> str:
        """Return the instruction text to append to the system prompt.

        The instruction asks the model to respond with valid JSON only and
        includes the full JSON Schema for the target model.

        Returns:
            Multi-line string suitable for appending to a system prompt.
        """
        schema_json = json.dumps(self.to_json_schema(), indent=2)
        lines = [
            "You MUST respond with valid JSON matching this schema:",
            f"```json\n{schema_json}\n```",
        ]
        if self.description:
            lines.insert(0, self.description)
            lines.insert(1, "")
        lines.append("Do not include any text before or after the JSON.")
        return "\n".join(lines)

    def parse(self, content: str) -> StructuredResult[T]:
        """Parse and validate an LLM response string against the schema.

        Attempts to extract a JSON object from ``content``, then validates it
        against the Pydantic model.

        Args:
            content: Raw string returned by the LLM.

        Returns:
            A :class:`StructuredResult` with ``success=True`` and ``data``
            populated when validation passes, or ``success=False`` and
            ``validation_errors`` populated on failure.
        """
        raw = content or ""
        json_text = self._extract_json(raw)
        if json_text is None:
            return StructuredResult(
                success=False,
                data=None,
                raw_content=raw,
                validation_errors=["No JSON object found in response."],
            )

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return StructuredResult(
                success=False,
                data=None,
                raw_content=raw,
                validation_errors=[f"Invalid JSON: {exc}"],
            )

        try:
            obj = self.model_class.model_validate(parsed, strict=self.strict)
            return StructuredResult(success=True, data=obj, raw_content=raw)
        except ValidationError as exc:
            errors = self.format_validation_error(exc).splitlines()
            return StructuredResult(
                success=False,
                data=None,
                raw_content=raw,
                validation_errors=errors,
            )

    def _extract_json(self, text: str) -> str | None:
        """Extract the first JSON object or array from ``text``.

        Handles the following formats:

        - Raw JSON starting with ``{`` or ``[``
        - Fenced code blocks (````json`` or ` ``` `` alone)
        - JSON embedded anywhere in prose (scans for ``{`` / ``[``)

        Args:
            text: Raw string to search.

        Returns:
            The extracted JSON string, or ``None`` if no candidate is found.
        """
        stripped = text.strip()
        if not stripped:
            return None

        # Raw JSON — starts directly with { or [
        if stripped[0] in ("{", "["):
            return stripped

        # Fenced code block: ```json ... ``` or ``` ... ```
        block_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", stripped, re.IGNORECASE)
        if block_match:
            candidate = block_match.group(1).strip()
            if candidate:
                return candidate

        # JSON embedded in prose — find outermost { ... } or [ ... ]
        for opener, closer in (("{", "}"), ("[", "]")):
            start = stripped.find(opener)
            end = stripped.rfind(closer)
            if start != -1 and end > start:
                return stripped[start : end + 1]

        return None

    def format_validation_error(self, error: ValidationError) -> str:
        """Format a Pydantic ``ValidationError`` as a prompt for the LLM.

        Args:
            error: The validation error to format.

        Returns:
            A multi-line string describing the errors and asking the model to
            fix them.
        """
        lines = ["Your response had validation errors:"]
        for err in error.errors():
            loc = " -> ".join(str(p) for p in err["loc"]) if err["loc"] else "(root)"
            msg = err["msg"]
            lines.append(f'- field "{loc}": {msg}')
        lines.append("\nPlease fix these errors and respond again with valid JSON.")
        return "\n".join(lines)


class StructuredOutputRunner:
    """Runs provider calls with structured output enforcement.

    Wraps a :class:`~missy.providers.base.BaseProvider` and retries failed
    validation attempts by feeding error feedback back to the model.

    Args:
        provider: The provider to use for completions.

    Example::

        runner = StructuredOutputRunner(provider)
        result = runner.complete_structured(messages, schema, system="Be concise.")
        if result.success:
            process(result.data)
        else:
            logger.warning("Validation failed after %d attempts", result.attempts)
    """

    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider

    def complete_structured(
        self,
        messages: list[Message],
        schema: OutputSchema[T],
        system: str = "",
        **kwargs,
    ) -> StructuredResult[T]:
        """Call the provider and validate the response against ``schema``.

        Retries up to ``schema.max_retries`` additional times when validation
        fails, appending the formatted error as a user message each time.

        Args:
            messages: Conversation history to send to the provider.
            schema: The :class:`OutputSchema` to enforce.
            system: Optional system prompt.  The schema instruction is
                appended to this string.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`~missy.providers.base.BaseProvider.complete`.

        Returns:
            A :class:`StructuredResult` reflecting the final attempt.
        """
        from missy.providers.base import Message as _Message

        augmented_system = _build_system(system, schema)
        history = list(messages)
        attempts = 0
        last_result: StructuredResult[T] | None = None

        for attempt in range(schema.max_retries + 1):
            response = self.provider.complete(history, system=augmented_system, **kwargs)
            attempts = attempt + 1
            raw = response.content
            result = schema.parse(raw)
            result.attempts = attempts

            if result.success:
                return result

            last_result = result
            if attempt < schema.max_retries:
                # Feed validation errors back as a user message.
                error_feedback = _format_errors_as_feedback(result.validation_errors)
                history = list(history) + [
                    _Message(role="assistant", content=raw),
                    _Message(role="user", content=error_feedback),
                ]
                logger.debug(
                    "Structured output attempt %d/%d failed; retrying",
                    attempt + 1,
                    schema.max_retries + 1,
                )

        assert last_result is not None
        return last_result

    async def acomplete_structured(
        self,
        messages: list[Message],
        schema: OutputSchema[T],
        system: str = "",
        **kwargs,
    ) -> StructuredResult[T]:
        """Async version of :meth:`complete_structured`.

        Delegates to the provider's ``acomplete`` coroutine when available;
        falls back to the synchronous :meth:`~BaseProvider.complete` in a
        thread executor otherwise.

        Args:
            messages: Conversation history to send to the provider.
            schema: The :class:`OutputSchema` to enforce.
            system: Optional system prompt.
            **kwargs: Extra keyword arguments forwarded to the provider.

        Returns:
            A :class:`StructuredResult` reflecting the final attempt.
        """
        import asyncio

        from missy.providers.base import Message as _Message

        augmented_system = _build_system(system, schema)
        history = list(messages)
        attempts = 0
        last_result: StructuredResult[T] | None = None
        loop = asyncio.get_event_loop()

        for attempt in range(schema.max_retries + 1):
            attempts = attempt + 1

            if _is_async_callable(getattr(self.provider, "acomplete", None)):
                response = await self.provider.acomplete(history, system=augmented_system, **kwargs)
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda h=history: self.provider.complete(h, system=augmented_system, **kwargs),
                )

            raw = response.content
            result = schema.parse(raw)
            result.attempts = attempts

            if result.success:
                return result

            last_result = result
            if attempt < schema.max_retries:
                feedback = _format_errors_as_feedback(result.validation_errors)
                history = list(history) + [
                    _Message(role="assistant", content=raw),
                    _Message(role="user", content=feedback),
                ]

        assert last_result is not None
        return last_result


# ---------------------------------------------------------------------------
# Common schemas for Missy's built-in use cases
# ---------------------------------------------------------------------------


class TaskAnalysis(BaseModel):
    """Analysis of a user's task request."""

    task_type: str
    """One of: ``"question"``, ``"code"``, ``"research"``, ``"file_operation"``, ``"system_admin"``."""

    complexity: str
    """One of: ``"simple"``, ``"moderate"``, ``"complex"``."""

    tools_needed: list[str]
    """Predicted tool names required to complete the task."""

    approach: str
    """Brief description of the intended approach."""


class ErrorAnalysis(BaseModel):
    """Analysis of an error for self-correction."""

    error_type: str
    """One of: ``"syntax"``, ``"runtime"``, ``"logic"``, ``"permission"``, ``"network"``, ``"unknown"``."""

    root_cause: str
    """Concise description of what caused the error."""

    suggested_fix: str
    """Actionable suggestion to resolve the error."""

    can_retry: bool
    """``True`` when retrying the failed operation is likely to succeed."""


class ConversationSummary(BaseModel):
    """Structured summary of a conversation segment."""

    key_topics: list[str]
    """Main subjects discussed."""

    decisions_made: list[str]
    """Conclusions or choices reached during the conversation."""

    action_items: list[str]
    """Follow-up tasks or next steps identified."""

    entities_mentioned: list[str]
    """Named entities (people, systems, files, etc.) referenced."""

    overall_summary: str
    """Single paragraph summarising the conversation segment."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_async_callable(obj: object) -> bool:
    """Return ``True`` when ``obj`` is an awaitable callable (coroutine function)."""
    import asyncio
    import inspect

    if obj is None:
        return False
    return asyncio.iscoroutinefunction(obj) or inspect.iscoroutinefunction(obj)


def _build_system(system: str, schema: OutputSchema) -> str:
    """Combine the caller's system prompt with the schema instruction."""
    instruction = schema.to_prompt_instruction()
    if system:
        return f"{system}\n\n{instruction}"
    return instruction


def _format_errors_as_feedback(errors: list[str]) -> str:
    """Turn a list of error strings into a retry prompt for the LLM."""
    if not errors:
        return "Your response was invalid. Please respond with valid JSON."
    lines = ["Your response had validation errors:"]
    for err in errors:
        # Already formatted lines (from format_validation_error) are kept as-is;
        # plain strings get a bullet prefix.
        if err.startswith(("-", "Your", "\n")):
            lines.append(err)
        else:
            lines.append(f"- {err}")
    lines.append("\nPlease fix these errors and respond again with valid JSON.")
    return "\n".join(lines)
