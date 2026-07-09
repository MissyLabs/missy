"""A deterministic, offline provider for exercising the tool-calling benchmark path.

:class:`MockToolProvider` never makes a network call and requires no API
key, so ``missy tools benchmark run-llm <tool> --provider mock`` (and the
test suite) can exercise the full LLM-benchmark path — provider selects a
tool, arguments are extracted, scored — without depending on a real
Anthropic/OpenAI/Ollama credential being configured.

It is deliberately *not* registered in
:data:`missy.providers.registry._PROVIDER_CLASSES` — it must never be
reachable as a normal chat provider, only via explicit construction by the
benchmark CLI/tests, so it can't accidentally end up serving real user
conversations.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from missy.providers.base import BaseProvider, CompletionResponse, Message, ToolCall

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]+")


class MockToolProvider(BaseProvider):
    """Deterministic offline provider used for local/mock benchmark runs.

    Given a single tool schema, always attempts to call it: required string
    parameters are filled from quoted substrings or bare tokens in the
    prompt (falling back to a placeholder), required numeric parameters from
    the first number in the prompt (falling back to ``1``), and required
    boolean parameters to ``True``.

    Args:
        call_tool: When ``False``, never emits a tool call — useful for
            benchmarking the "provider ignores tools" failure mode.
    """

    name = "mock"

    def __init__(self, call_tool: bool = True) -> None:
        self.call_tool = call_tool

    def is_available(self) -> bool:
        return True

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        last = messages[-1].content if messages else ""
        return CompletionResponse(
            content=f"mock response to: {last[:80]}",
            model="mock-1",
            provider=self.name,
            usage={"prompt_tokens": len(last.split()), "completion_tokens": 8, "total_tokens": 0},
            raw={},
        )

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        prompt = messages[-1].content if messages else ""
        prompt_tokens = len((system + " " + prompt).split())

        if not self.call_tool or not tools:
            return CompletionResponse(
                content="I don't have a tool for that.",
                model="mock-1",
                provider=self.name,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 6,
                    "total_tokens": 0,
                },
                raw={},
                tool_calls=[],
                finish_reason="stop",
            )

        tool = tools[0]
        schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
        params = schema.get("parameters", {})
        properties: dict[str, Any] = params.get("properties", {})
        required: list[str] = params.get("required", [])

        args = {name: _guess_value(properties.get(name, {}), prompt) for name in required}

        call = ToolCall(id=str(uuid.uuid4()), name=getattr(tool, "name", ""), arguments=args)
        return CompletionResponse(
            content="",
            model="mock-1",
            provider=self.name,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": 4, "total_tokens": 0},
            raw={},
            tool_calls=[call],
            finish_reason="tool_calls",
        )


def _guess_value(prop: dict[str, Any], prompt: str) -> Any:
    """Best-effort extraction of a plausible argument value from *prompt*."""
    ptype = prop.get("type", "string")
    if ptype in ("integer", "number"):
        match = re.search(r"-?\d+(\.\d+)?", prompt)
        if match:
            return float(match.group()) if ptype == "number" else int(float(match.group()))
        return prop.get("default", 1)
    if ptype == "boolean":
        return True
    if ptype == "array":
        return prop.get("default", [])

    quoted = re.search(r"['\"]([^'\"]+)['\"]", prompt)
    if quoted:
        return quoted.group(1)
    words = _WORD_RE.findall(prompt)
    if words:
        return words[-1]
    return prop.get("default", "test")
