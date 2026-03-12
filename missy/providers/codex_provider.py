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
from typing import Any, Iterator, Optional

import httpx

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message, ToolCall, ToolResult

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


def _load_oauth_token() -> Optional[str]:
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
        result.append({
            "role": role,
            "content": [{"type": "input_text", "text": msg.content}]
            if role == "user"
            else [{"type": "output_text", "text": msg.content}],
        })
    return result


def _extract_system(messages: list[Message]) -> str:
    """Return the content of the first system message, or empty string."""
    for msg in messages:
        if msg.role == "system":
            return msg.content
    return ""


class CodexProvider(BaseProvider):
    """Provider that calls ChatGPT's backend API with an OAuth token.

    Args:
        config: Provider config.  ``api_key`` should be the OAuth access
            token.  If omitted, the token is loaded from
            ``~/.missy/secrets/openai-oauth.json``.
    """

    name = "openai-codex"

    def __init__(self, config: ProviderConfig) -> None:
        self._api_key: Optional[str] = config.api_key
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
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        instructions = _extract_system(messages)
        input_messages = _messages_to_input(messages)
        body: dict[str, Any] = {
            "model": self._model,
            "input": input_messages,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "stream": stream,
        }
        if instructions:
            body["instructions"] = instructions
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        return body

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Single-turn completion via the Codex backend (non-streaming)."""
        token = self._get_token()
        account_id = _extract_account_id(token)
        body = self._build_body(messages, stream=False)

        try:
            resp = httpx.post(
                _CODEX_ENDPOINT,
                headers=self._headers(token, account_id),
                json=body,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"openai-codex HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except Exception as exc:
            raise ProviderError(f"openai-codex request failed: {exc}") from exc

        data = resp.json()
        text = self._extract_text_from_response(data)
        return CompletionResponse(content=text, finish_reason="stop")

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

    def stream(self, messages: list[Message], **kwargs: Any) -> Iterator[str]:
        """Stream tokens from the Codex backend via SSE."""
        token = self._get_token()
        account_id = _extract_account_id(token)
        body = self._build_body(messages, stream=True)

        try:
            with httpx.stream(
                "POST",
                _CODEX_ENDPOINT,
                headers=self._headers(token, account_id),
                json=body,
                timeout=self._timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
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
                    if etype == "response.output_text.delta":
                        delta = event.get("delta", "")
                        if delta:
                            yield delta
                    elif etype in ("response.failed", "error"):
                        msg = event.get("message") or event.get("error", {}).get("message", "unknown")
                        raise ProviderError(f"openai-codex stream error: {msg}")
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"openai-codex HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc

    def complete_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        system_prompt: str = "",
        **kwargs: Any,
    ) -> CompletionResponse:
        """Tool-calling completion via the Codex backend (non-streaming)."""
        token = self._get_token()
        account_id = _extract_account_id(token)
        body = self._build_body(messages, stream=False, tools=tools)

        try:
            resp = httpx.post(
                _CODEX_ENDPOINT,
                headers=self._headers(token, account_id),
                json=body,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"openai-codex HTTP {exc.response.status_code}: {exc.response.text[:200]}") from exc

        data = resp.json()
        tool_calls: list[ToolCall] = []
        text = ""

        for item in data.get("output", []):
            if item.get("type") == "function_call":
                try:
                    args = json.loads(item.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=item.get("call_id", item.get("id", "")),
                    name=item.get("name", ""),
                    arguments=args,
                ))
            elif item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        text += part.get("text", "")

        finish_reason = "tool_calls" if tool_calls else "stop"
        return CompletionResponse(
            content=text,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )

    def get_tool_schema(self, tool_name: str, description: str, parameters: dict) -> dict:
        """Return a Responses API function tool schema."""
        return {
            "type": "function",
            "name": tool_name,
            "description": description,
            "parameters": parameters,
        }

    def is_available(self) -> bool:
        """Return True if an OAuth token is accessible."""
        try:
            return bool(self._api_key or _load_oauth_token())
        except Exception:
            return False
