"""Ollama local-inference provider for the Missy framework.

Communicates with the Ollama REST API via :class:`~missy.gateway.client.PolicyHTTPClient`
so that all outbound requests pass through the active network policy.

The Ollama SDK is intentionally **not** used here; the raw HTTP API is
simple and avoids an extra dependency.  The ``/api/chat`` endpoint is
used with ``stream=false`` so that the full response is returned in a
single JSON payload.

Example::

    from missy.config.settings import ProviderConfig
    from missy.providers.ollama_provider import OllamaProvider

    config = ProviderConfig(name="ollama", model="llama3.2")
    provider = OllamaProvider(config)
    response = provider.complete([Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from missy.config.settings import ProviderConfig
from missy.core.events import AuditEvent, event_bus
from missy.core.exceptions import ProviderError
from missy.gateway.client import PolicyHTTPClient

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"

# Pattern to detect prompted tool call JSON in model output
_TOOL_CALL_RE = re.compile(
    r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)


class OllamaProvider(BaseProvider):
    """Provider implementation backed by the Ollama local-inference server.

    All HTTP traffic is routed through :class:`~missy.gateway.client.PolicyHTTPClient`
    so network policy is enforced automatically.

    Args:
        config: Provider-level configuration.  ``base_url`` defaults to
            ``http://localhost:11434``.  ``api_key`` is unused for Ollama.
    """

    name = "ollama"

    def __init__(self, config: ProviderConfig) -> None:
        self._base_url: str = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the Ollama server responds to ``GET /api/tags``.

        Returns:
            ``True`` when the tags endpoint returns HTTP 200.  Returns
            ``False`` on any network or HTTP error without raising.
        """
        try:
            client = PolicyHTTPClient(timeout=self._timeout)
            response = client.get(f"{self._base_url}/api/tags")
            return response.status_code == 200
        except Exception as exc:
            logger.debug("Ollama availability check failed: %s", exc)
            return False

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Send *messages* to the Ollama ``/api/chat`` endpoint.

        Args:
            messages: Ordered conversation turns.  All role values supported
                by Ollama (``"system"``, ``"user"``, ``"assistant"``) are
                forwarded as-is.
            **kwargs: Optional overrides.  Recognised keys:

                * ``model`` (str) - override the configured model.
                * ``temperature`` (float) - sampling temperature forwarded
                  inside the ``options`` payload.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On HTTP error, unexpected response format, or
                any transport-level failure.
        """
        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        model = kwargs.pop("model", self._model)

        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        payload: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "stream": False,
        }

        options: dict[str, Any] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if options:
            payload["options"] = options

        # Forward any remaining kwargs directly into the payload
        payload.update(kwargs)

        try:
            client = PolicyHTTPClient(
                session_id=session_id,
                task_id=task_id,
                timeout=self._timeout,
            )
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except ProviderError:
            raise
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", "invalid JSON response")
            raise ProviderError(f"Ollama returned invalid JSON: {exc}") from exc

        # Ollama chat response shape:
        # { "model": "...", "message": {"role": "assistant", "content": "..."},
        #   "prompt_eval_count": N, "eval_count": N, ... }
        message_obj = data.get("message") or {}
        content_text: str = message_obj.get("content", "")

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        self._emit_event(session_id, task_id, "allow", "completion successful")

        return CompletionResponse(
            content=content_text,
            model=data.get("model", model),
            provider=self.name,
            usage=usage,
            raw=data,
        )

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to a JSON-serialisable description list.

        Ollama does not support native function calling, so this returns a
        simplified list of name/description/parameters dicts that can be
        injected into the system prompt as a text description.

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of tool description dicts.
        """
        schemas = []
        for tool in tools:
            base_schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            params = base_schema.get("parameters", {})
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                }
            )
        return schemas

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        """Prompted tool-call fallback for Ollama (no native function calling).

        Injects tool schemas into the system prompt as a JSON description and
        instructs the model to respond with a specific JSON format when it
        wants to call a tool.  The model output is then parsed for that format.

        Expected tool-call format from model::

            {"tool_call": {"name": "tool_name", "arguments": {...}}}

        If no tool-call JSON is found the response is treated as plain text.

        Args:
            messages: Ordered conversation turns.
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` contains a single parsed
            :class:`~missy.providers.base.ToolCall`.
        """
        tool_schemas = self.get_tool_schema(tools)

        # Build an augmented system prompt with tool instructions
        tool_json = json.dumps(tool_schemas, indent=2)
        tool_instructions = (
            "You have access to the following tools. When you need to call a tool, "
            "respond ONLY with a JSON object in this exact format and nothing else:\n"
            '{"tool_call": {"name": "<tool_name>", "arguments": {<key>: <value>}}}\n\n'
            f"Available tools:\n{tool_json}"
        )

        # Merge system prompt
        augmented_system = tool_instructions
        if system:
            augmented_system = f"{system}\n\n{tool_instructions}"

        # Build messages for the underlying complete() call
        augmented_messages: list[Message] = [
            Message(role="system", content=augmented_system)
        ]
        for msg in messages:
            if msg.role != "system":
                augmented_messages.append(msg)

        response = self.complete(augmented_messages)
        raw_content = response.content.strip()

        # Try to parse a tool_call JSON block from the response
        match = _TOOL_CALL_RE.search(raw_content)
        if match:
            try:
                parsed = json.loads(match.group(0))
                tc_data = parsed.get("tool_call", {})
                tool_name = tc_data.get("name", "")
                arguments = tc_data.get("arguments", {})
                if tool_name:
                    import uuid as _uuid

                    tool_calls = [
                        ToolCall(
                            id=str(_uuid.uuid4())[:8],
                            name=tool_name,
                            arguments=arguments if isinstance(arguments, dict) else {},
                        )
                    ]
                    return CompletionResponse(
                        content="",
                        model=response.model,
                        provider=self.name,
                        usage=response.usage,
                        raw=response.raw,
                        tool_calls=tool_calls,
                        finish_reason="tool_calls",
                    )
            except (json.JSONDecodeError, Exception) as exc:
                logger.debug("Failed to parse tool call from Ollama response: %s", exc)

        # Plain text response
        return CompletionResponse(
            content=raw_content,
            model=response.model,
            provider=self.name,
            usage=response.usage,
            raw=response.raw,
            tool_calls=[],
            finish_reason="stop",
        )

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream partial response tokens from the Ollama API.

        Uses the ``/api/chat`` endpoint with ``stream=true`` and reads
        newline-delimited JSON chunks.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt string (merged into messages if
                no system message is already present).

        Yields:
            String token chunks as they arrive.

        Raises:
            ProviderError: On transport failure or malformed response.
        """
        api_messages: list[dict] = []
        has_system = any(m.role == "system" for m in messages)
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "stream": True,
        }

        try:
            client = PolicyHTTPClient(timeout=self._timeout)
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message_obj = chunk.get("message") or {}
                token = message_obj.get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Ollama stream failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
    ) -> None:
        """Publish a provider audit event to the global event bus.

        Args:
            session_id: Calling session identifier.
            task_id: Calling task identifier.
            result: One of ``"allow"`` or ``"error"``.
            detail_msg: Human-readable description to include in the event.
        """
        try:
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail={
                    "provider": self.name,
                    "model": self._model,
                    "base_url": self._base_url,
                    "message": detail_msg,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
