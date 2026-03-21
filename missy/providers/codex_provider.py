"""OpenAI Codex provider for Missy — uses ChatGPT's backend API.

This provider handles the OAuth token obtained via ``missy setup`` →
OpenAI OAuth flow.  It differs from :class:`~missy.providers.openai_provider.OpenAIProvider`
in three critical ways:

1. **Endpoint**: ``https://chatgpt.com/backend-api/codex/responses``
   (not ``api.openai.com``).  This is ChatGPT's internal backend, which
   accepts the OAuth bearer token.
2. **Request format**: OpenAI Responses API shape — ``instructions``
   (system prompt) + ``input`` (message list) rather than ``messages``.
3. **Extra header**: ``chatgpt-account-id`` extracted from the JWT.

The OAuth token is loaded automatically from
``~/.missy/secrets/openai-oauth.json`` (written by the wizard) and
refreshed if it is near expiry.  It can also be supplied directly via
the ``api_key`` field in ``ProviderConfig``.

All outbound HTTP is routed through
:class:`~missy.gateway.client.PolicyHTTPClient` so that network policy
is enforced automatically.

Configure in ``config.yaml``::

    providers:
      openai-codex:
        name: openai-codex
        model: "gpt-4o"
        timeout: 60
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from typing import Any

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.gateway.client import PolicyHTTPClient

from .base import BaseProvider, CompletionResponse, Message, ToolCall

logger = logging.getLogger(__name__)

_CODEX_BASE = "https://chatgpt.com/backend-api"
_CODEX_ENDPOINT = f"{_CODEX_BASE}/codex/responses"
_DEFAULT_MODEL = "gpt-5.2"


def _extract_account_id(token: str) -> str:
    """Pull ``chatgpt_account_id`` from the JWT payload without verification."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        padding = 4 - len(parts[1]) % 4
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * padding))
        ns = payload.get("https://api.openai.com/auth", {})
        return ns.get("chatgpt_account_id", "") or payload.get("sub", "")
    except Exception:
        return ""


def _load_oauth_token() -> str | None:
    """Load the stored OAuth access token, refreshing if needed."""
    try:
        from missy.cli.oauth import refresh_token_if_needed

        return refresh_token_if_needed()
    except Exception:
        return None


def _messages_to_input(messages: list[Message]) -> list[dict]:
    """Convert Missy Message list to Responses API input format."""
    result = []
    for msg in messages:
        if msg.role == "system":
            continue  # system prompt goes in ``instructions``
        role = "user" if msg.role == "user" else "assistant"
        result.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": msg.content}]
                if role == "user"
                else [{"type": "output_text", "text": msg.content}],
            }
        )
    return result


def _extract_system(messages: list[Message]) -> str:
    """Return the content of the first system message, or empty string."""
    for msg in messages:
        if msg.role == "system":
            return msg.content
    return ""


class CodexProvider(BaseProvider):
    """Provider that calls ChatGPT's backend API with an OAuth token.

    All HTTP traffic is routed through
    :class:`~missy.gateway.client.PolicyHTTPClient` so network policy is
    enforced automatically.

    Args:
        config: Provider config.  ``api_key`` should be the OAuth access
            token.  If omitted, the token is loaded from
            ``~/.missy/secrets/openai-oauth.json``.
    """

    name = "openai-codex"

    def __init__(self, config: ProviderConfig) -> None:
        self._api_key: str | None = config.api_key
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout or 60

    def _get_token(self) -> str:
        token = self._api_key or _load_oauth_token()
        if not token:
            raise ProviderError(
                "openai-codex: no OAuth token available. "
                "Run 'missy setup' and choose OpenAI → OAuth."
            )
        return token

    def _headers(self, token: str, account_id: str) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if account_id:
            h["chatgpt-account-id"] = account_id
        return h

    def _build_body(
        self,
        messages: list[Message],
        stream: bool = True,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        instructions = _extract_system(messages)
        input_messages = _messages_to_input(messages)
        body: dict[str, Any] = {
            "model": self._model,
            "input": input_messages,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "stream": True,  # Codex endpoint requires stream=true always
        }
        if instructions:
            body["instructions"] = instructions
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def _make_client(
        self,
        session_id: str = "",
        task_id: str = "",
    ) -> PolicyHTTPClient:
        """Construct a PolicyHTTPClient for outbound requests."""
        return PolicyHTTPClient(
            session_id=session_id,
            task_id=task_id,
            timeout=self._timeout,
            category="provider",
        )

    def _stream_sse(
        self,
        client: PolicyHTTPClient,
        body: dict[str, Any],
        token: str,
        account_id: str,
    ) -> Iterator[dict]:
        """POST to the Codex endpoint and yield parsed SSE event dicts.

        Uses ``PolicyHTTPClient.post(stream=True)`` so that network
        policy is enforced.

        Yields:
            Parsed JSON event dicts from ``data:`` lines.

        Raises:
            ProviderError: On HTTP errors or stream error events.
        """
        try:
            response = client.post(
                _CODEX_ENDPOINT,
                json=body,
                headers=self._headers(token, account_id),
                stream=True,
            )
            response.raise_for_status()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"openai-codex request failed: {exc}") from exc

        try:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw in ("", "[DONE]"):
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype in ("response.failed", "error"):
                    msg = event.get("message") or event.get("error", {}).get("message", "unknown")
                    raise ProviderError(f"openai-codex stream error: {msg}")
                yield event
        finally:
            response.close()

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Single-turn completion — collects the SSE stream and returns full text."""
        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")

        self._acquire_rate_limit()

        text = "".join(self.stream(messages))

        self._emit_event(session_id, task_id, "allow", "completion successful")

        return CompletionResponse(
            content=text,
            model=self._model,
            provider=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw={},
            finish_reason="stop",
        )

    def _extract_text_from_response(self, data: dict) -> str:
        """Extract text content from a non-streaming Responses API response."""
        # Shape: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}]}
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        return part.get("text", "")
        # Fallback: older shape
        return data.get("text", "") or data.get("content", "") or ""

    def stream(self, messages: list[Message], system: str = "", **kwargs: Any) -> Iterator[str]:
        """Stream tokens from the Codex backend via SSE.

        All HTTP is routed through :class:`PolicyHTTPClient`.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt (prepended if non-empty).

        Yields:
            Text delta chunks as they arrive.

        Raises:
            ProviderError: On transport failure or stream errors.
        """
        if system:
            messages = [Message(role="system", content=system), *messages]

        token = self._get_token()
        account_id = _extract_account_id(token)
        body = self._build_body(messages, stream=True)
        client = self._make_client()

        for event in self._stream_sse(client, body, token, account_id):
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield delta

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list,
        system: str = "",
    ) -> CompletionResponse:
        """Tool-calling completion — collects SSE stream and parses tool calls.

        All HTTP is routed through :class:`PolicyHTTPClient`.

        Args:
            messages: Ordered conversation turns.
            tools: List of :class:`~missy.tools.base.BaseTool` instances
                or pre-built schema dicts.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` contains the parsed
            :class:`~missy.providers.base.ToolCall` instances.

        Raises:
            ProviderError: On transport failure or stream errors.
        """
        self._acquire_rate_limit()

        if system:
            messages = [Message(role="system", content=system), *messages]

        token = self._get_token()
        account_id = _extract_account_id(token)
        # Convert BaseTool instances → schema dicts if needed.
        tool_schemas = (
            self.get_tool_schema(tools) if tools and not isinstance(tools[0], dict) else tools
        )
        body = self._build_body(messages, tools=tool_schemas)
        client = self._make_client()

        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        current_fn: dict = {}

        for event in self._stream_sse(client, body, token, account_id):
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                text_parts.append(event.get("delta", ""))
            elif etype == "response.output_item.added":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    current_fn = {
                        "id": item.get("call_id", item.get("id", "")),
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    }
            elif etype == "response.function_call_arguments.delta":
                current_fn["arguments"] = current_fn.get("arguments", "") + event.get("delta", "")
            elif etype == "response.function_call_arguments.done" and current_fn.get("name"):
                try:
                    args = json.loads(current_fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=current_fn["id"],
                        name=current_fn["name"],
                        arguments=args,
                    )
                )
                current_fn = {}

        finish_reason = "tool_calls" if tool_calls else "stop"

        self._emit_event("", "", "allow", f"completion: {finish_reason}")

        return CompletionResponse(
            content="".join(text_parts),
            model=self._model,
            provider=self.name,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            raw={},
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to Codex Responses API function schemas."""
        schemas = []
        for tool in tools:
            if isinstance(tool, dict):
                schemas.append(tool)
                continue
            base = tool.get_schema() if hasattr(tool, "get_schema") else {}
            schemas.append(
                {
                    "type": "function",
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    "parameters": base.get(
                        "parameters",
                        {
                            "type": "object",
                            "properties": getattr(tool, "parameters", {}),
                            "required": [],
                        },
                    ),
                }
            )
        return schemas

    def is_available(self) -> bool:
        """Return True if an OAuth token is accessible."""
        try:
            return bool(self._api_key or _load_oauth_token())
        except Exception:
            return False

    def _emit_event(
        self,
        session_id: str,
        task_id: str,
        result: str,
        detail_msg: str,
    ) -> None:
        """Publish a provider audit event including the model name."""
        try:
            from missy.core.events import AuditEvent, event_bus

            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail={"provider": self.name, "model": self._model, "message": detail_msg},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
