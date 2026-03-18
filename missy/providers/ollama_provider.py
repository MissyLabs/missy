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
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.gateway.client import PolicyHTTPClient

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"


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
        self._client: PolicyHTTPClient | None = None

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def _make_client(
        self,
        session_id: str = "",
        task_id: str = "",
    ) -> PolicyHTTPClient:
        """Return a cached PolicyHTTPClient, creating one on first call."""
        if self._client is None:
            self._client = PolicyHTTPClient(
                session_id=session_id,
                task_id=task_id,
                timeout=self._timeout,
                category="provider",
            )
        return self._client

    def is_available(self) -> bool:
        """Return ``True`` when the Ollama server responds to ``GET /api/tags``.

        Returns:
            ``True`` when the tags endpoint returns HTTP 200.  Returns
            ``False`` on any network or HTTP error without raising.
        """
        try:
            client = PolicyHTTPClient(timeout=self._timeout, category="provider")
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

        self._acquire_rate_limit()

        try:
            client = self._make_client(session_id=session_id, task_id=task_id)
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

        result = CompletionResponse(
            content=content_text,
            model=data.get("model", model),
            provider=self.name,
            usage=usage,
            raw=data,
        )
        self._record_rate_limit_usage(result)
        return result

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to Ollama's native tool schema format.

        Ollama expects tools in the OpenAI-compatible format::

            {"type": "function", "function": {"name": "...", "description": "...",
             "parameters": {"type": "object", "properties": {...}, "required": [...]}}}

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of Ollama-native tool schema dicts.
        """
        schemas = []
        for tool in tools:
            base_schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            params = base_schema.get("parameters", {})
            # Ensure parameters has a proper JSON Schema structure
            if params and "type" not in params:
                params = {"type": "object", "properties": params}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": params or {"type": "object", "properties": {}},
                    },
                }
            )
        return schemas

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        """Send messages with native tool calling via Ollama's ``tools`` parameter.

        Uses Ollama's native ``/api/chat`` tool calling support. The tool
        schemas are passed as the ``tools`` key in the request payload and
        the model returns ``tool_calls`` in the response message when it
        wants to invoke a tool.

        Args:
            messages: Ordered conversation turns.
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` contains the parsed
            :class:`~missy.providers.base.ToolCall` instances.
        """
        tool_schemas = self.get_tool_schema(tools)

        # Build messages, injecting system prompt if provided
        api_messages: list[dict[str, str]] = []
        has_system = any(m.role == "system" for m in messages)
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "tools": tool_schemas,
            "stream": False,
        }

        self._acquire_rate_limit()

        try:
            client = self._make_client()
            response = client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
        except ProviderError:
            raise
        except Exception as exc:
            self._emit_event("", "", "error", str(exc))
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data: dict[str, Any] = response.json()
        except Exception as exc:
            self._emit_event("", "", "error", "invalid JSON response")
            raise ProviderError(f"Ollama returned invalid JSON: {exc}") from exc

        message_obj = data.get("message") or {}
        content_text: str = message_obj.get("content", "")

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Parse native tool_calls from the response
        raw_tool_calls = message_obj.get("tool_calls") or []
        parsed_tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            func = tc.get("function") or {}
            tc_name = func.get("name", "")
            tc_args = func.get("arguments", {})
            if tc_name:
                parsed_tool_calls.append(
                    ToolCall(
                        id=tc.get("id", "") or tc_name[:8],
                        name=tc_name,
                        arguments=tc_args if isinstance(tc_args, dict) else {},
                    )
                )

        if parsed_tool_calls:
            self._emit_event("", "", "allow", "tool_calls")
            result = CompletionResponse(
                content=content_text,
                model=data.get("model", self._model),
                provider=self.name,
                usage=usage,
                raw=data,
                tool_calls=parsed_tool_calls,
                finish_reason="tool_calls",
            )
            self._record_rate_limit_usage(result)
            return result

        self._emit_event("", "", "allow", "completion successful")
        result = CompletionResponse(
            content=content_text,
            model=data.get("model", self._model),
            provider=self.name,
            usage=usage,
            raw=data,
            tool_calls=[],
            finish_reason="stop",
        )
        self._record_rate_limit_usage(result)
        return result

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
            client = self._make_client()
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
        """Publish a provider audit event including model and base_url."""
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
                    "model": self._model,
                    "base_url": self._base_url,
                    "message": detail_msg,
                },
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
