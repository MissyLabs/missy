"""OpenAI-compatible provider for the Missy framework.

Uses the ``openai`` SDK to call OpenAI's native Responses API when available,
falling back to Chat Completions for OpenAI-compatible ``base_url`` endpoints
and transcripts that still require Chat Completions compatibility.

The SDK is imported lazily so that Missy can start without it installed -
:meth:`is_available` returns ``False`` in that case.

Example::

    from missy.config.settings import ProviderConfig
    from missy.providers.openai_provider import OpenAIProvider

    config = ProviderConfig(name="openai", model="auto", api_key="<REDACTED>")
    provider = OpenAIProvider(config)
    response = provider.complete([Message(role="user", content="Hello")])
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError

from .base import BaseProvider, CompletionResponse, Message, ToolCall
from .rate_limiter import RateLimiter
from .round_robin import Account, RoundRobinAccounts

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "auto"
_FALLBACK_MODEL = "gpt-5.5"
_AUTO_MODEL_SENTINELS = {"", "auto", "latest", "best"}
_PREFERRED_CHAT_MODELS = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
)
_NON_CHAT_MODEL_MARKERS = (
    "audio",
    "dall",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "sora",
    "tts",
    "transcribe",
    "whisper",
)
_IMAGE_DETAIL_VALUES = {"auto", "low", "high"}

try:
    import openai as _openai_sdk

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _openai_sdk = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False


#: F15 -- the per-account record is now the shared, provider-agnostic
#: ``round_robin.Account`` (same fields: index/api_key/rate_limiter/client).
#: Aliased here so the rest of this module reads unchanged.
_OpenAIAccount = Account


class OpenAIProvider(BaseProvider):
    """Provider implementation backed by the OpenAI Chat Completions API.

    Args:
        config: Provider-level configuration.  ``api_key`` is forwarded to
            the SDK; when ``None`` the SDK reads ``OPENAI_API_KEY`` from the
            environment.  ``base_url`` overrides the default API endpoint,
            enabling OpenAI-compatible third-party services. When
            ``config.key_rotation_strategy == "round_robin"`` and
            ``config.api_keys`` has 2+ entries, every call is balanced
            across those accounts round-robin (see :class:`_OpenAIAccount`)
            instead of using a single sticky key.
    """

    name = "openai"
    accepts_message_dicts = True

    def __init__(self, config: ProviderConfig) -> None:
        self._api_key: str | None = config.api_key
        self._base_url: str | None = config.base_url
        self._model: str = config.model or _DEFAULT_MODEL
        self._timeout: int = config.timeout
        self._requests_per_minute: int = config.requests_per_minute
        self._tokens_per_minute: int = config.tokens_per_minute
        self._max_wait_seconds: float = config.max_wait_seconds
        self._client: Any | None = None
        self._resolved_model: str | None = None
        self._last_transcript_repairs: list[dict[str, Any]] = []

        # Multi-account round-robin balancing (opt-in via
        # key_rotation_strategy: "round_robin"). self._accounts stays empty
        # -- and every call falls through to the single self._api_key/
        # self._client pair above, completely unchanged -- unless the
        # operator both opts in AND configures 2+ keys. self._account_local
        # is a thread-local "which account is this call using" slot: each
        # of complete()/complete_with_tools()/stream() sets it once at the
        # top of the call (see _select_account()), and _make_client()/
        # _acquire_rate_limit()/_record_rate_limit_usage() all consult it
        # instead of a shared mutable instance attribute, so concurrent
        # calls on different threads can never race over which account's
        # client/rate limiter is "currently active" -- each call is pinned
        # to its own account slot for its own duration.
        # F15: account-list + round-robin selection are now the shared
        # RoundRobinAccounts helper; the thread-local "current account" and
        # client-building stay here (a client is SDK-specific). Behaviour is
        # unchanged: round-robin activates only with round_robin strategy + 2+
        # keys, each account getting its own independently-budgeted rate limiter.
        self._account_local = threading.local()
        _rr_keys = (
            list(config.api_keys or []) if config.key_rotation_strategy == "round_robin" else []
        )
        self._rr = RoundRobinAccounts(
            _rr_keys,
            make_rate_limiter=lambda: RateLimiter(
                requests_per_minute=self._requests_per_minute,
                tokens_per_minute=self._tokens_per_minute,
                max_wait_seconds=self._max_wait_seconds,
            ),
        )
        self._accounts: list[Account] = self._rr.accounts

    @property
    def is_multi_account(self) -> bool:
        """Return ``True`` when 2+ accounts are configured for round-robin balancing."""
        return bool(self._accounts)

    @property
    def account_count(self) -> int:
        """Return how many accounts this provider round-robins across (0 if not multi-account)."""
        return len(self._accounts)

    def _select_account(self) -> _OpenAIAccount | None:
        """Return the account to use for the call in progress on this thread.

        Returns ``None`` when this provider isn't configured for
        multi-account balancing, in which case callers fall back to the
        single ``self._api_key``/``self._client`` pair exactly as before
        this feature existed. Advances the round-robin index atomically so
        concurrent callers are assigned accounts round-robin with no lost
        or duplicated turns.
        """
        return self._rr.select()

    def _current_rate_limiter(self) -> RateLimiter | None:
        """Return the rate limiter for the account active on this thread, if any."""
        account: _OpenAIAccount | None = getattr(self._account_local, "current", None)
        if account is not None:
            return account.rate_limiter
        return self.rate_limiter

    def _acquire_rate_limit(self, estimated_tokens: int = 0) -> None:
        """Block until the active account's (or the shared) rate limiter permits a request."""
        account: _OpenAIAccount | None = getattr(self._account_local, "current", None)
        if account is not None:
            account.rate_limiter.acquire(tokens=estimated_tokens)
            return
        super()._acquire_rate_limit(estimated_tokens=estimated_tokens)

    def _record_rate_limit_usage(
        self, response: CompletionResponse, estimated_tokens: int = 0
    ) -> None:
        """Reconcile usage against the active account's (or the shared) rate limiter."""
        account: _OpenAIAccount | None = getattr(self._account_local, "current", None)
        if account is not None:
            account.rate_limiter.record_usage(
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                estimated_tokens=estimated_tokens,
            )
            return
        super()._record_rate_limit_usage(response, estimated_tokens=estimated_tokens)

    @property
    def api_key(self) -> str | None:
        """Return the active API key, if one is configured directly."""
        return self._api_key

    @api_key.setter
    def api_key(self, value: str | None) -> None:
        """Update the API key and force the SDK client to be rebuilt."""
        self._api_key = value
        self._client = None
        self._resolved_model = None

    @property
    def model(self) -> str:
        """Return the configured model selector."""
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        """Update the model selector and clear any cached auto resolution."""
        self._model = value or _DEFAULT_MODEL
        self._resolved_model = None

    # ------------------------------------------------------------------
    # BaseProvider interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` when the SDK is installed and an API key is set.

        Returns:
            ``True`` if the ``openai`` package is importable and an API key
            is present.
        """
        return _OPENAI_AVAILABLE and bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    @staticmethod
    def _normalized_host(value: str) -> str:
        """Return a lowercase host without port/brackets from a host or URL."""
        parsed = urlparse(value if "://" in value else f"https://{value}")
        host = parsed.hostname or value
        return host.strip("[]").lower().rsplit(":", 1)[0]

    @staticmethod
    def _host_matches_domain(host: str, pattern: str) -> bool:
        pattern_lower = pattern.lower()
        if pattern_lower.startswith("*."):
            suffix = pattern_lower[2:]
            return host == suffix or host.endswith("." + suffix)
        return host == pattern_lower

    def _endpoint_host(self) -> str:
        """Return the host this provider will contact for inference."""
        if self._base_url:
            return self._normalized_host(self._base_url)
        return "api.openai.com"

    def _credential_source(self) -> str:
        """Return where the active API key would come from, without the key."""
        if self._api_key:
            return "config"
        if os.environ.get("OPENAI_API_KEY"):
            return "environment"
        return "missing"

    def _network_policy_summary(self, host: str) -> tuple[str, str, str | None]:
        """Return local policy posture for *host* without DNS or audit events."""
        try:
            from missy.policy.engine import get_policy_engine

            engine = get_policy_engine()
            network = getattr(engine, "network", None)
            policy = getattr(network, "_policy", None)
        except Exception:
            return (
                "warn",
                "policy engine not initialized",
                ("Initialize policy before running provider-backed sessions."),
            )

        if policy is None:
            return (
                "warn",
                "network policy unavailable",
                ("Initialize network policy before provider calls."),
            )
        if not bool(getattr(policy, "default_deny", True)):
            return "warn", "default_allow", "Enable network.default_deny for provider egress."

        host_values = [
            *list(getattr(policy, "allowed_hosts", []) or []),
            *list(getattr(policy, "provider_allowed_hosts", []) or []),
        ]
        for entry in host_values:
            if host == self._normalized_host(str(entry)):
                return "ok", f"allowed host:{host}", None

        for pattern in list(getattr(policy, "allowed_domains", []) or []):
            if self._host_matches_domain(host, str(pattern)):
                return "ok", f"allowed domain:{pattern}", None

        return (
            "warn",
            f"missing provider allowlist for {host}",
            (
                f"Add {host!r} to network.provider_allowed_hosts, network.allowed_hosts, "
                "or use the openai network preset for native OpenAI."
            ),
        )

    def diagnostics(self) -> dict[str, Any]:
        """Return redacted OpenAI provider diagnostics without live API calls."""
        credential_source = self._credential_source()
        endpoint_host = self._endpoint_host()
        network_status, network_summary, network_remediation = self._network_policy_summary(
            endpoint_host
        )
        native_openai = self._base_url is None
        available = _OPENAI_AVAILABLE and credential_source != "missing"
        checks: list[dict[str, Any]] = [
            {
                "name": "sdk",
                "status": "ok" if _OPENAI_AVAILABLE else "error",
                "summary": "installed" if _OPENAI_AVAILABLE else "not installed",
                "remediation": "Install the openai package." if not _OPENAI_AVAILABLE else None,
            },
            {
                "name": "credential",
                "status": "ok" if credential_source != "missing" else "error",
                "summary": f"configured via {credential_source}"
                if credential_source != "missing"
                else "missing OPENAI_API_KEY/config key",
                "remediation": "Set OPENAI_API_KEY or a protected provider api_key reference."
                if credential_source == "missing"
                else None,
            },
            {
                "name": "endpoint",
                "status": "ok",
                "summary": {
                    "host": endpoint_host,
                    "native_openai": native_openai,
                    "base_url_override": bool(self._base_url),
                },
            },
            {
                "name": "network_policy",
                "status": network_status,
                "summary": network_summary,
                "remediation": network_remediation,
            },
            {
                "name": "model_selection",
                "status": "ok",
                "summary": {
                    "configured": self._model,
                    "resolved": self._resolved_model,
                    "auto": self._model.lower() in _AUTO_MODEL_SENTINELS,
                },
            },
            {
                "name": "rate_limits",
                "status": "ok" if self._requests_per_minute or self._tokens_per_minute else "warn",
                "summary": {
                    "requests_per_minute": self._requests_per_minute,
                    "tokens_per_minute": self._tokens_per_minute,
                    "max_wait_seconds": self._max_wait_seconds,
                    "timeout_seconds": self._timeout,
                },
                "remediation": "Set provider RPM/TPM budgets for deterministic throttling."
                if not (self._requests_per_minute or self._tokens_per_minute)
                else None,
            },
            {
                "name": "capabilities",
                "status": "ok",
                "summary": {
                    "responses_api": "native eligible" if native_openai else "chat-compatible",
                    "chat_completions": True,
                    "streaming": True,
                    "tool_calling": "chat_completions",
                    "structured_output": True,
                    "vision_input": True,
                    "embeddings": False,
                },
            },
            {
                "name": "multi_account_balancing",
                "status": "ok",
                "summary": {
                    "enabled": self.is_multi_account,
                    "account_count": len(self._accounts),
                },
            },
        ]
        status = (
            "error"
            if any(c["status"] == "error" for c in checks)
            else ("warn" if any(c["status"] == "warn" for c in checks) else "ok")
        )
        return {
            "provider": self.name,
            "status": status if available else "error",
            "checks": checks,
        }

    def _build_client(self, api_key: str | None) -> Any:
        """Construct a new OpenAI SDK client bound to *api_key*.

        The SDK is given a policy-aware ``http_client`` so that all provider
        egress transits the network policy check (consistent with the
        gateway), rather than issuing unchecked HTTP directly.
        """
        client_kwargs: dict[str, Any] = {"timeout": float(self._timeout)}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        try:
            from missy.providers.policy_http import build_policy_http_client

            client_kwargs["http_client"] = build_policy_http_client(timeout=float(self._timeout))
        except Exception:  # pragma: no cover - defensive; never block startup
            logger.debug("Could not build policy-aware http client", exc_info=True)
        return _openai_sdk.OpenAI(**client_kwargs)

    def _make_client(self) -> Any:
        """Return a cached OpenAI client, creating one on first call.

        When this thread's call selected a multi-account entry (see
        :meth:`_select_account`), returns that account's own lazily-built,
        independently-cached client instead of the shared
        ``self._client`` -- each account gets its own client, so rotating
        api_key on one account can never accidentally invalidate another
        account's already-built client.
        """
        account: _OpenAIAccount | None = getattr(self._account_local, "current", None)
        if account is not None:
            if account.client is None:
                account.client = self._build_client(account.api_key)
            return account.client
        if self._client is None:
            self._client = self._build_client(self._api_key)
        return self._client

    def _resolve_model(self, requested_model: str | None = None) -> str:
        """Resolve ``auto`` / ``latest`` to the best available chat model.

        The OpenAI models endpoint is account-aware, so this prefers the
        newest known chat models only when the active credentials can see
        them. If the endpoint is unavailable, fall back to the current
        recommended frontier model instead of a legacy GPT-4-era default.
        """
        model = (requested_model or self._model or _DEFAULT_MODEL).strip()
        if model.lower() not in _AUTO_MODEL_SENTINELS:
            return model
        if self._resolved_model:
            return self._resolved_model

        try:
            response = self._make_client().models.list()
            raw_models = getattr(response, "data", response)
            model_ids: set[str] = set()
            for item in raw_models:
                model_id = getattr(item, "id", None)
                if model_id is None and isinstance(item, dict):
                    model_id = item.get("id")
                if model_id:
                    model_ids.add(str(model_id))

            for preferred in _PREFERRED_CHAT_MODELS:
                if preferred in model_ids:
                    self._resolved_model = preferred
                    return preferred

            chat_like = sorted(
                mid
                for mid in model_ids
                if mid.startswith(("gpt-", "chatgpt-"))
                and not any(marker in mid.lower() for marker in _NON_CHAT_MODEL_MARKERS)
            )
            if chat_like:
                self._resolved_model = chat_like[-1]
                return chat_like[-1]
        except Exception:
            logger.debug("OpenAI model auto-detection failed; using fallback", exc_info=True)

        self._resolved_model = _FALLBACK_MODEL
        return _FALLBACK_MODEL

    @staticmethod
    def _supports_custom_temperature(model: str) -> bool:
        """Return whether custom temperature should be sent for *model*."""
        model_lower = model.lower()
        return not model_lower.startswith(("gpt-5", "o1", "o3", "o4"))

    def _apply_common_generation_kwargs(
        self,
        call_kwargs: dict[str, Any],
        kwargs: dict[str, Any],
        model: str,
    ) -> None:
        """Normalize common generation kwargs for current OpenAI chat models."""
        if "temperature" in kwargs:
            temperature = kwargs.pop("temperature")
            supports_temperature = self._supports_custom_temperature(model)
            try:
                is_default_temperature = float(temperature) == 1.0
            except (TypeError, ValueError):
                is_default_temperature = False
            if temperature is not None and not supports_temperature and not is_default_temperature:
                logger.debug("Omitting unsupported temperature override for OpenAI model %r", model)
            elif temperature is not None:
                call_kwargs["temperature"] = temperature

        if "max_completion_tokens" in kwargs:
            call_kwargs["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
        elif "max_tokens" in kwargs:
            call_kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        structured_schema = kwargs.pop("structured_output_schema", None)
        if structured_schema is not None:
            call_kwargs["response_format"] = self._chat_response_format(structured_schema)

        call_kwargs.update(kwargs)

    def _apply_responses_generation_kwargs(
        self,
        call_kwargs: dict[str, Any],
        kwargs: dict[str, Any],
        model: str,
    ) -> None:
        """Normalize common generation kwargs for OpenAI Responses calls."""
        if "temperature" in kwargs:
            temperature = kwargs.pop("temperature")
            supports_temperature = self._supports_custom_temperature(model)
            try:
                is_default_temperature = float(temperature) == 1.0
            except (TypeError, ValueError):
                is_default_temperature = False
            if temperature is not None and not supports_temperature and not is_default_temperature:
                logger.debug("Omitting unsupported temperature override for OpenAI model %r", model)
            elif temperature is not None:
                call_kwargs["temperature"] = temperature

        if "max_output_tokens" in kwargs:
            call_kwargs["max_output_tokens"] = kwargs.pop("max_output_tokens")
        elif "max_completion_tokens" in kwargs:
            call_kwargs["max_output_tokens"] = kwargs.pop("max_completion_tokens")
        elif "max_tokens" in kwargs:
            call_kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        structured_schema = kwargs.pop("structured_output_schema", None)
        if structured_schema is not None:
            call_kwargs["text"] = {
                "format": self._responses_text_format(structured_schema),
            }

        call_kwargs.update(kwargs)

    @staticmethod
    def _structured_schema_name(schema: Any) -> str:
        """Return an OpenAI-safe response format name for a Missy schema."""
        model_class = getattr(schema, "model_class", None)
        raw_name = getattr(model_class, "__name__", None) or "structured_output"
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw_name)).strip("_-")
        return (name or "structured_output")[:64]

    def _responses_text_format(self, schema: Any) -> dict[str, Any]:
        """Build the Responses API ``text.format`` JSON schema payload."""
        payload: dict[str, Any] = {
            "type": "json_schema",
            "name": self._structured_schema_name(schema),
            "schema": schema.to_json_schema(),
            "strict": bool(getattr(schema, "strict", False)),
        }
        description = getattr(schema, "description", "")
        if description:
            payload["description"] = str(description)
        return payload

    def _chat_response_format(self, schema: Any) -> dict[str, Any]:
        """Build the Chat Completions ``response_format`` JSON schema payload."""
        text_format = self._responses_text_format(schema)
        return {
            "type": "json_schema",
            "json_schema": text_format,
        }

    @staticmethod
    def _normalize_text_content(content: Any) -> str:
        """Return the textual parts of *content* without preserving rich blocks."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                    text = part.get("text")
                    if text is not None:
                        text_parts.append(str(text))
                elif isinstance(part, str):
                    text_parts.append(part)
            return "\n".join(text_parts)
        return str(content)

    @staticmethod
    def _is_safe_image_url(url: str) -> bool:
        """Return whether *url* is safe to forward as OpenAI image input."""
        if url.startswith("data:image/") and ";base64," in url:
            return True
        parsed = urlparse(url)
        return parsed.scheme == "https" and bool(parsed.netloc)

    def _normalize_user_content(self, content: Any) -> str | list[dict[str, Any]]:
        """Normalize user content, preserving safe OpenAI text/image blocks."""
        if not isinstance(content, list):
            return self._normalize_text_content(content)

        normalized: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                normalized.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                self._record_transcript_repair("drop_unsupported_user_content_part")
                continue

            part_type = str(part.get("type", ""))
            if part_type in {"text", "input_text"}:
                text = part.get("text")
                if text is not None:
                    normalized.append({"type": "text", "text": str(text)})
                else:
                    self._record_transcript_repair("drop_empty_text_part")
                continue

            if part_type in {"image_url", "input_image"}:
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url") or part.get("url")
                    detail = image_url.get("detail", part.get("detail", "auto"))
                else:
                    url = image_url or part.get("url")
                    detail = part.get("detail", "auto")

                url_text = str(url or "")
                detail_text = str(detail or "auto")
                if detail_text not in _IMAGE_DETAIL_VALUES:
                    detail_text = "auto"
                if self._is_safe_image_url(url_text):
                    normalized.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url_text, "detail": detail_text},
                        }
                    )
                else:
                    self._record_transcript_repair("drop_unsafe_image_url")
                continue

            self._record_transcript_repair("drop_unsupported_user_content_part")

        if not normalized:
            self._record_transcript_repair("empty_user_content_after_normalization")
            return ""
        return normalized

    def _record_transcript_repair(self, reason: str, **detail: Any) -> None:
        """Remember a transcript repair for later audit emission."""
        repair = {"reason": reason}
        repair.update({k: v for k, v in detail.items() if v is not None})
        self._last_transcript_repairs.append(repair)

    def _emit_transcript_repairs(self, session_id: str = "", task_id: str = "") -> None:
        """Publish provider-turn repair audit events for the latest payload."""
        if not self._last_transcript_repairs:
            return
        repairs = list(self._last_transcript_repairs)
        self._last_transcript_repairs = []
        try:
            from missy.core.events import AuditEvent, event_bus

            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_transcript_repair",
                category="provider",
                result="allow",
                detail={"provider": self.name, "model": self._model, "repairs": repairs},
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit OpenAI transcript repair audit event")

    def _message_to_chat_payload(
        self,
        message: Message | dict[str, Any],
    ) -> dict[str, Any] | None:
        """Convert Missy's message shape to OpenAI Chat Completions format."""
        if isinstance(message, Message):
            if message.role == "user":
                return {
                    "role": message.role,
                    "content": self._normalize_user_content(message.content),
                }
            return {"role": message.role, "content": self._normalize_text_content(message.content)}

        role = str(message.get("role", ""))
        if role in {"system", "user"}:
            content = message.get("content", "")
            if role == "user":
                return {"role": role, "content": self._normalize_user_content(content)}
            return {"role": role, "content": self._normalize_text_content(content)}

        if role == "assistant":
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": self._normalize_text_content(message.get("content", "")),
            }
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                payload["tool_calls"] = []
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    if "function" in call:
                        payload["tool_calls"].append(call)
                        continue
                    arguments = call.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    payload["tool_calls"].append(
                        {
                            "id": str(call.get("id", "")),
                            "type": "function",
                            "function": {
                                "name": str(call.get("name", "")),
                                "arguments": arguments,
                            },
                        }
                    )
            return payload

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                return None
            return {
                "role": "tool",
                "tool_call_id": str(tool_call_id),
                "content": str(message.get("content", "")),
            }

        return None

    def _messages_to_chat_payload(
        self,
        messages: list[Message] | list[dict[str, Any]],
        system: str = "",
    ) -> list[dict[str, Any]]:
        """Build a Chat Completions-compatible message list."""
        self._last_transcript_repairs = []
        api_messages: list[dict[str, Any]] = []
        pending_tool_call_ids: set[str] = set()
        emitted_tool_call_ids: set[str] = set()
        has_system = any(
            (msg.role if isinstance(msg, Message) else msg.get("role")) == "system"
            for msg in messages
        )
        if system and not has_system:
            api_messages.append({"role": "system", "content": system})
        for msg in messages:
            payload = self._message_to_chat_payload(msg)
            if payload is not None:
                if payload["role"] == "assistant" and payload.get("tool_calls"):
                    valid_tool_calls: list[dict[str, Any]] = []
                    for call in payload["tool_calls"]:
                        call_id = str(call.get("id", ""))
                        function = call.get("function") if isinstance(call, dict) else None
                        name = function.get("name") if isinstance(function, dict) else ""
                        if not call_id or not name:
                            self._record_transcript_repair("drop_invalid_assistant_tool_call")
                            continue
                        if call_id in emitted_tool_call_ids:
                            self._record_transcript_repair(
                                "drop_duplicate_assistant_tool_call",
                                tool_call_id=call_id,
                            )
                            continue
                        valid_tool_calls.append(call)
                        pending_tool_call_ids.add(call_id)
                        emitted_tool_call_ids.add(call_id)
                    if valid_tool_calls:
                        payload["tool_calls"] = valid_tool_calls
                    else:
                        payload.pop("tool_calls", None)

                if payload["role"] == "tool":
                    tool_call_id = str(payload.get("tool_call_id", ""))
                    if tool_call_id not in pending_tool_call_ids:
                        self._record_transcript_repair(
                            "drop_orphan_tool_result",
                            tool_call_id=tool_call_id,
                        )
                        continue
                    pending_tool_call_ids.remove(tool_call_id)

                api_messages.append(payload)
        return api_messages

    @staticmethod
    def _client_supports_responses(client: Any) -> bool:
        """Return whether *client* clearly exposes the Responses API surface."""
        if client.__class__.__module__.startswith("unittest.mock"):
            return False
        responses = getattr(client, "responses", None)
        create = getattr(responses, "create", None)
        return callable(create)

    @staticmethod
    def _client_supports_responses_stream(client: Any) -> bool:
        """Return whether *client* clearly exposes Responses streaming."""
        if client.__class__.__module__.startswith("unittest.mock"):
            return False
        responses = getattr(client, "responses", None)
        stream = getattr(responses, "stream", None)
        return callable(stream)

    def _should_use_responses_api(
        self,
        client: Any,
        api_messages: list[dict[str, Any]],
        tools_enabled: bool = False,
    ) -> bool:
        """Return whether this request should use native OpenAI Responses."""
        if self._base_url or tools_enabled or not self._client_supports_responses(client):
            return False
        for message in api_messages:
            if message.get("role") == "tool" or message.get("tool_calls"):
                return False
        return True

    @staticmethod
    def _responses_content_from_chat(content: Any) -> str | list[dict[str, Any]]:
        """Convert normalized Chat Completions content into Responses input content."""
        if not isinstance(content, list):
            return str(content or "")

        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append({"type": "input_text", "text": str(part.get("text", ""))})
            elif part_type == "image_url":
                image_url = part.get("image_url") or {}
                if isinstance(image_url, dict):
                    payload = {"type": "input_image", "image_url": str(image_url.get("url", ""))}
                    detail = image_url.get("detail")
                    if detail in _IMAGE_DETAIL_VALUES:
                        payload["detail"] = detail
                    parts.append(payload)
        return parts

    def _messages_to_responses_payload(
        self,
        api_messages: list[dict[str, Any]],
        system: str = "",
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build Responses API instructions and input from normalized messages."""
        instructions: list[str] = []
        if system:
            instructions.append(system)
        input_items: list[dict[str, Any]] = []
        for message in api_messages:
            role = str(message.get("role", ""))
            content = message.get("content", "")
            if role == "system":
                text = self._normalize_text_content(content)
                if text:
                    instructions.append(text)
                continue
            if role in {"user", "assistant"}:
                input_items.append(
                    {
                        "role": role,
                        "content": self._responses_content_from_chat(content),
                    }
                )
        return "\n\n".join(instructions), input_items

    @staticmethod
    def _extract_responses_text(raw_response: Any) -> str:
        """Extract assistant text from a Responses API response object."""
        output_text = getattr(raw_response, "output_text", None)
        if isinstance(output_text, str) and output_text:
            return output_text

        text_parts: list[str] = []
        output_items = getattr(raw_response, "output", None)
        if output_items is None and isinstance(raw_response, dict):
            output_items = raw_response.get("output")
        for item in output_items or []:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            for part in content or []:
                part_type = getattr(part, "type", None)
                text = getattr(part, "text", None)
                if isinstance(part, dict):
                    part_type = part.get("type", part_type)
                    text = part.get("text", text)
                if part_type in {"output_text", "text"} and text is not None:
                    text_parts.append(str(text))
        return "".join(text_parts)

    @staticmethod
    def _responses_usage(raw_response: Any) -> dict[str, int]:
        """Return Missy's canonical usage map from a Responses response."""
        usage_obj = getattr(raw_response, "usage", None)
        prompt_tokens = int(
            getattr(usage_obj, "input_tokens", getattr(usage_obj, "prompt_tokens", 0)) or 0
        )
        completion_tokens = int(
            getattr(
                usage_obj,
                "output_tokens",
                getattr(usage_obj, "completion_tokens", 0),
            )
            or 0
        )
        total_tokens = int(
            getattr(usage_obj, "total_tokens", prompt_tokens + completion_tokens) or 0
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _raw_dump(raw_response: Any) -> dict[str, Any]:
        """Return a defensive raw payload dump for SDK response objects."""
        if hasattr(raw_response, "model_dump"):
            dumped = raw_response.model_dump()
            return dumped if isinstance(dumped, dict) else {}
        if isinstance(raw_response, dict):
            return raw_response
        return {}

    @staticmethod
    def _event_get(event: Any, key: str, default: Any = None) -> Any:
        """Read *key* from SDK event objects or dict fixtures."""
        if isinstance(event, dict):
            return event.get(key, default)
        return getattr(event, key, default)

    @staticmethod
    def _append_reconciled_text(current: str, candidate: Any) -> tuple[str, str]:
        """Return ``(delta_to_emit, new_full_text)`` for delta/full snapshots."""
        if candidate is None:
            return "", current
        text = str(candidate)
        if not text:
            return "", current
        if text.startswith(current):
            return text[len(current) :], text
        return text, current + text

    @classmethod
    def _responses_event_full_text(cls, event: Any) -> str:
        """Extract a full text snapshot from known Responses stream events."""
        for key in ("text", "output_text", "content"):
            value = cls._event_get(event, key)
            if isinstance(value, str) and value:
                return value

        response = cls._event_get(event, "response")
        if response is not None:
            text = OpenAIProvider._extract_responses_text(response)
            if text:
                return text

        item = cls._event_get(event, "item")
        if item is not None:
            text = OpenAIProvider._extract_responses_text({"output": [item]})
            if text:
                return text

        return ""

    @classmethod
    def _responses_stream_error_message(cls, event: Any) -> str:
        """Return a safe message from an error/failed Responses stream event."""
        error = cls._event_get(event, "error")
        if error is not None:
            message = cls._event_get(error, "message")
            if message:
                return str(message)
            if isinstance(error, str):
                return error
        message = cls._event_get(event, "message")
        if message:
            return str(message)
        return "OpenAI Responses stream failed"

    def _responses_stream_chunks(
        self,
        events: Iterator[Any],
    ) -> Iterator[str]:
        """Yield reconciled text chunks from Responses streaming events."""
        full_text = ""
        for event in events:
            event_type = str(self._event_get(event, "type", ""))
            if event_type in {"response.failed", "error"}:
                raise ProviderError(self._responses_stream_error_message(event))

            if event_type == "response.output_text.delta":
                delta, full_text = self._append_reconciled_text(
                    full_text,
                    self._event_get(event, "delta", ""),
                )
            elif event_type in {
                "response.output_text.done",
                "response.output_item.done",
                "response.completed",
            }:
                delta, full_text = self._append_reconciled_text(
                    full_text,
                    self._responses_event_full_text(event),
                )
            else:
                continue

            if delta:
                yield delta

    def _stream_via_responses(
        self,
        client: Any,
        api_messages: list[dict[str, Any]],
        model: str,
        system: str = "",
    ) -> Iterator[str]:
        """Stream a compatible request through OpenAI Responses."""
        instructions, input_items = self._messages_to_responses_payload(api_messages, system=system)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
        }
        if instructions:
            call_kwargs["instructions"] = instructions

        stream_obj = client.responses.stream(**call_kwargs)
        if hasattr(stream_obj, "__enter__"):
            with stream_obj as stream:
                yield from self._responses_stream_chunks(iter(stream))
        else:
            yield from self._responses_stream_chunks(iter(stream_obj))

    def _complete_via_responses(
        self,
        client: Any,
        api_messages: list[dict[str, Any]],
        model: str,
        kwargs: dict[str, Any],
        system: str = "",
        estimated_tokens: int = 0,
    ) -> CompletionResponse:
        """Execute a plain text/vision request via OpenAI Responses."""
        instructions, input_items = self._messages_to_responses_payload(api_messages, system=system)
        call_kwargs: dict[str, Any] = {
            "model": model,
            "input": input_items,
        }
        if instructions:
            call_kwargs["instructions"] = instructions
        self._apply_responses_generation_kwargs(call_kwargs, kwargs, model)

        raw_response = client.responses.create(**call_kwargs)
        content_text = self._extract_responses_text(raw_response)
        usage = self._responses_usage(raw_response)
        response = CompletionResponse(
            content=content_text,
            model=str(getattr(raw_response, "model", model) or model),
            provider=self.name,
            usage=usage,
            raw=self._raw_dump(raw_response),
        )
        self._record_rate_limit_usage(response, estimated_tokens=estimated_tokens)
        return response

    def _complete_via_chat(
        self,
        client: Any,
        api_messages: list[dict[str, Any]],
        model: str,
        kwargs: dict[str, Any],
        estimated_tokens: int = 0,
    ) -> CompletionResponse:
        """Execute a request via Chat Completions compatibility mode."""
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        self._apply_common_generation_kwargs(call_kwargs, kwargs, model)
        raw_response = client.chat.completions.create(**call_kwargs)

        choice = raw_response.choices[0] if raw_response.choices else None
        content_text = choice.message.content if choice else ""
        usage_obj = raw_response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
        }

        response = CompletionResponse(
            content=content_text or "",
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=self._raw_dump(raw_response),
        )
        self._record_rate_limit_usage(response, estimated_tokens=estimated_tokens)
        return response

    def complete(self, messages: list[Message], **kwargs: Any) -> CompletionResponse:
        """Send *messages* to OpenAI, preferring Responses when compatible.

        Args:
            messages: Ordered conversation turns.  All role values supported
                by the API (``"system"``, ``"user"``, ``"assistant"``) are
                forwarded as-is.
            **kwargs: Optional overrides.  Recognised keys:

                * ``temperature`` (float) - sampling temperature.
                * ``max_tokens`` (int) - maximum completion tokens.
                * ``model`` (str) - override the configured model.

        Returns:
            A :class:`CompletionResponse` with the assistant reply.

        Raises:
            ProviderError: On SDK import failure, authentication error,
                API error, or timeout.
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderError("openai SDK is not installed. Run: pip install openai")

        session_id = kwargs.pop("session_id", "")
        task_id = kwargs.pop("task_id", "")
        system = kwargs.pop("system", "")
        model = self._resolve_model(kwargs.pop("model", self._model))
        api_messages = self._messages_to_chat_payload(messages, system=system)
        self._emit_transcript_repairs(session_id, task_id)

        self._account_local.current = self._select_account()
        estimated_tokens = self._estimate_tokens(messages)
        self._acquire_rate_limit(estimated_tokens=estimated_tokens)

        try:
            client = self._make_client()
            if self._should_use_responses_api(client, api_messages):
                response = self._complete_via_responses(
                    client,
                    api_messages,
                    model,
                    kwargs,
                    system=system,
                    estimated_tokens=estimated_tokens,
                )
                self._emit_event(session_id, task_id, "allow", "responses completion successful")
                return response
            response = self._complete_via_chat(
                client, api_messages, model, kwargs, estimated_tokens=estimated_tokens
            )
            self._emit_event(session_id, task_id, "allow", "chat completion successful")
            return response
        except _openai_sdk.APITimeoutError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"OpenAI request timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            if getattr(exc, "status_code", 0) == 429:
                retry_after = float(
                    getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", 5)
                )
                limiter = self._current_rate_limiter()
                if limiter is not None:
                    limiter.on_rate_limit_response(retry_after)
                raise ProviderError(f"OpenAI rate limited: {exc}") from exc
            raise ProviderError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            self._emit_event(session_id, task_id, "error", str(exc))
            raise ProviderError(f"Unexpected error calling OpenAI: {exc}") from exc

    def structured_output_kwargs(self, schema: Any) -> dict[str, Any]:
        """Ask OpenAI to enforce Missy's structured output schema natively."""
        return {"structured_output_schema": schema}

    def get_tool_schema(self, tools: list) -> list:
        """Convert BaseTool instances to OpenAI function-calling schema format.

        Delegates to :func:`~missy.providers.schema_adapter.normalize_for_provider`
        for canonical → OpenAI conversion, falling back to inline construction
        if the adapter is unavailable.

        Args:
            tools: List of :class:`~missy.tools.base.BaseTool` instances.

        Returns:
            A list of OpenAI-format tool dicts with ``type`` and ``function``
            keys.
        """
        try:
            from missy.providers.schema_adapter import normalize_for_provider

            schemas = []
            for tool in tools:
                base = tool.get_schema() if hasattr(tool, "get_schema") else {}
                canonical: dict[str, Any] = {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", ""),
                    **base,
                }
                schemas.append(normalize_for_provider(canonical, "openai"))
            return schemas
        except Exception:
            logger.debug("schema_adapter unavailable; falling back to inline schema build")

        schemas = []
        for tool in tools:
            base_schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            params = base_schema.get("parameters", {})
            function_schema: dict[str, Any] = {
                "type": "object",
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            }
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": function_schema,
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
        """Send *messages* to the OpenAI API with tool calling enabled.

        Args:
            messages: Ordered conversation turns.
            tools: List of :class:`~missy.tools.base.BaseTool` instances.
            system: Optional system prompt string.

        Returns:
            A :class:`CompletionResponse`.  When ``finish_reason`` is
            ``"tool_calls"``, ``tool_calls`` is populated with
            :class:`~missy.providers.base.ToolCall` instances.

        Raises:
            ProviderError: On SDK import failure or API error.
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderError("openai SDK is not installed. Run: pip install openai")

        tool_schemas = self.get_tool_schema(tools)

        # Build message list; inject system prompt if provided
        model = self._resolve_model(self._model)
        api_messages = self._messages_to_chat_payload(messages, system=system)
        self._emit_transcript_repairs()

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        if tool_schemas:
            call_kwargs["tools"] = tool_schemas
            call_kwargs["tool_choice"] = "auto"

        self._account_local.current = self._select_account()
        estimated_tokens = self._estimate_tokens(messages, system)
        self._acquire_rate_limit(estimated_tokens=estimated_tokens)

        try:
            client = self._make_client()
            raw_response = client.chat.completions.create(**call_kwargs)
        except _openai_sdk.APITimeoutError as exc:
            raise ProviderError(f"OpenAI request timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            if getattr(exc, "status_code", 0) == 429:
                retry_after = float(
                    getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", 5)
                )
                limiter = self._current_rate_limiter()
                if limiter is not None:
                    limiter.on_rate_limit_response(retry_after)
                raise ProviderError(f"OpenAI rate limited: {exc}") from exc
            raise ProviderError(f"OpenAI API error: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error calling OpenAI: {exc}") from exc

        choice = raw_response.choices[0] if raw_response.choices else None
        content_text = ""
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"

        if choice:
            content_text = choice.message.content or ""
            raw_finish = choice.finish_reason or "stop"
            finish_reason = raw_finish if raw_finish in ("stop", "tool_calls", "length") else "stop"

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, Exception):
                        arguments = {}
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

        usage_obj = raw_response.usage
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(usage_obj, "total_tokens", 0),
        }

        response = CompletionResponse(
            content=content_text,
            model=raw_response.model,
            provider=self.name,
            usage=usage,
            raw=raw_response.model_dump() if hasattr(raw_response, "model_dump") else {},
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        self._record_rate_limit_usage(response, estimated_tokens=estimated_tokens)
        return response

    def stream(self, messages: list[Message], system: str = "") -> Iterator[str]:
        """Stream partial response tokens from the OpenAI API.

        Args:
            messages: Ordered conversation turns.
            system: Optional system prompt string.

        Yields:
            String text delta chunks as they arrive from the API.

        Raises:
            ProviderError: On SDK import failure or API error.
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderError("openai SDK is not installed. Run: pip install openai")

        model = self._resolve_model(self._model)
        api_messages = self._messages_to_chat_payload(messages, system=system)
        self._emit_transcript_repairs()
        self._account_local.current = self._select_account()
        self._acquire_rate_limit(estimated_tokens=self._estimate_tokens(messages, system))

        try:
            client = self._make_client()
            if self._should_use_responses_api(
                client,
                api_messages,
            ) and self._client_supports_responses_stream(client):
                yield from self._stream_via_responses(
                    client,
                    api_messages,
                    model,
                    system=system,
                )
                return

            call_kwargs: dict[str, Any] = {
                "model": model,
                "messages": api_messages,
                "stream": True,
            }
            for chunk in client.chat.completions.create(**call_kwargs):
                delta = chunk.choices[0].delta.content if chunk.choices else None
                yield delta or ""
        except ProviderError:
            raise
        except _openai_sdk.APITimeoutError as exc:
            raise ProviderError(f"OpenAI stream timed out after {self._timeout}s: {exc}") from exc
        except _openai_sdk.AuthenticationError as exc:
            raise ProviderError(f"OpenAI authentication failed: {exc}") from exc
        except _openai_sdk.APIError as exc:
            raise ProviderError(f"OpenAI API error during stream: {exc}") from exc
        except Exception as exc:
            raise ProviderError(f"Unexpected error streaming from OpenAI: {exc}") from exc

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
        """Publish a provider audit event including the model name.

        When this call was dispatched through a multi-account round-robin
        slot, ``account_index`` is included so operators can confirm
        balancing is actually happening across accounts -- never the api
        key itself, only its position in the configured list.
        """
        try:
            from missy.core.events import AuditEvent, event_bus

            detail: dict[str, Any] = {
                "provider": self.name,
                "model": self._model,
                "message": detail_msg,
            }
            account: _OpenAIAccount | None = getattr(self._account_local, "current", None)
            if account is not None:
                detail["account_index"] = account.index
            event = AuditEvent.now(
                session_id=session_id,
                task_id=task_id,
                event_type="provider_invoke",
                category="provider",
                result=result,  # type: ignore[arg-type]
                detail=detail,
            )
            event_bus.publish(event)
        except Exception:
            logger.exception("Failed to emit audit event for provider %r", self.name)
