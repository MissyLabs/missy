"""Policy-enforcing boundary for provider requests.

Adapters are deliberately small synchronous callables. They are trusted code,
but their exceptions and returned content are not trusted for persistence.
Only this module resolves credentials; the credential is passed directly to an
adapter and is never included in a request/job object or result.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class ProviderFailure(StrEnum):
    AUTHORIZATION = "authorization_denied"
    UNKNOWN_PROVIDER = "provider_not_allowlisted"
    MODEL_NOT_ALLOWED = "model_not_allowlisted"
    BUDGET = "budget_exceeded"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    REFUSAL = "refusal"
    PROVIDER_ERROR = "provider_error"
    MALFORMED = "malformed_response"
    INTERNAL = "internal_error"


class BrokerError(Exception):
    """Stable, safe failure. Never use provider/secret text in message."""

    def __init__(self, category: ProviderFailure, message: str, *, retriable: bool = False):
        super().__init__(message[:256])
        self.category = category
        self.retriable = retriable


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    secret_ref: str
    models: frozenset[str]
    max_timeout_seconds: float = 120.0
    max_output_tokens: int = 8192
    max_cost_microusd: int = 10_000_000


@dataclass(frozen=True)
class ProviderRequest:
    """Canonical provider-independent inference request."""

    messages: tuple[Mapping[str, Any], ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    tools: tuple[Mapping[str, Any], ...] = ()
    request_id: str | None = None

    def canonical_bytes(self) -> bytes:
        value = {"messages": self.messages, "settings": self.settings, "tools": self.tools}
        try:
            return json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise BrokerError(ProviderFailure.MALFORMED, "Request is not canonical JSON") from exc

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class AdapterRequest:
    registry_key: str
    model: str
    messages: tuple[Mapping[str, Any], ...]
    settings: Mapping[str, Any]
    tools: tuple[Mapping[str, Any], ...]
    timeout_seconds: float
    max_output_tokens: int
    request_digest: str


@dataclass(frozen=True)
class AdapterResponse:
    content: str
    actual_model: str | None = None
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_microusd: int | None = None
    refused: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProviderAdapter(Protocol):
    def complete(self, request: AdapterRequest, credential: str) -> AdapterResponse: ...


@dataclass(frozen=True)
class ProviderResult:
    registry_key: str
    requested_model: str
    actual_model: str | None
    provider_request_id: str | None
    request_digest: str
    content: str | None
    content_sha256: str | None
    content_truncated: bool
    input_tokens: int | None
    output_tokens: int | None
    cost_microusd: int | None
    elapsed_ms: int
    error_category: str | None = None
    retriable: bool = False
    error_message: str | None = None


SecretResolver = Callable[[str], str]
Authorizer = Callable[[str, str], bool]


class RateLimitError(Exception):
    """Adapter signal for a provider rate limit; exception text is discarded."""


_SENSITIVE = re.compile(
    r"(?i)(?:\b(?:sk-[A-Za-z0-9_-]{8,}|(?:Bearer|Basic)\s+\S+|"
    r"(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[:=]\s*\S+|"
    r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{8,}|glpat-[A-Za-z0-9_-]{8,}|"
    r"xox[baprs]-[A-Za-z0-9-]{8,})|"
    r"eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
)
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]*\Z")


class ProviderBroker:
    """Allowlisted, project-authorized provider adapter dispatcher.

    `budget_microusd`, timeout and token cap are caller-requested ceilings and
    are clamped to registry policy. `execute_fanout` sends byte-identical
    canonical messages/settings/tools to each selected target; only model and
    registry identity vary.
    """

    def __init__(
        self,
        registry: Mapping[str, ProviderSpec],
        adapters: Mapping[str, ProviderAdapter],
        secret_resolver: SecretResolver,
        authorizer: Authorizer,
        *,
        max_request_bytes: int = 1_000_000,
        max_response_chars: int = 64_000,
    ):
        self._registry = dict(registry)
        self._adapters = dict(adapters)
        self._resolve = secret_resolver
        self._authorize = authorizer
        self._max_request_bytes = max_request_bytes
        self._max_response_chars = max_response_chars

    def execute(
        self,
        *,
        project_id: str,
        registry_key: str,
        model: str,
        request: ProviderRequest,
        budget_microusd: int,
        timeout_seconds: float,
        token_cap: int,
    ) -> ProviderResult:
        start = time.monotonic()
        if not project_id or not self._authorize(project_id, registry_key):
            raise BrokerError(
                ProviderFailure.AUTHORIZATION, "Project is not authorized for this provider"
            )
        spec = self._registry.get(registry_key)
        if spec is None:
            raise BrokerError(
                ProviderFailure.UNKNOWN_PROVIDER, "Provider is not in the approved registry"
            )
        if model not in spec.models:
            raise BrokerError(
                ProviderFailure.MODEL_NOT_ALLOWED, "Model is not approved for this provider"
            )
        if (
            not isinstance(budget_microusd, int)
            or budget_microusd < 0
            or budget_microusd > spec.max_cost_microusd
        ):
            raise BrokerError(ProviderFailure.BUDGET, "Requested budget exceeds provider policy")
        if token_cap <= 0 or timeout_seconds <= 0:
            raise BrokerError(ProviderFailure.BUDGET, "Token and timeout limits must be positive")
        canonical = request.canonical_bytes()
        if len(canonical) > self._max_request_bytes:
            raise BrokerError(ProviderFailure.BUDGET, "Canonical request exceeds size limit")
        adapter = self._adapters.get(registry_key)
        if adapter is None:
            raise BrokerError(ProviderFailure.UNKNOWN_PROVIDER, "No approved adapter is registered")
        timeout = min(float(timeout_seconds), spec.max_timeout_seconds)
        tokens = min(int(token_cap), spec.max_output_tokens)
        # Resolve only after all policy checks. The credential exists only in this
        # stack frame and the adapter call; it is not part of any returned object.
        try:
            credential = self._resolve(spec.secret_ref)
            if not isinstance(credential, str) or not credential:
                raise RuntimeError("credential unavailable")
        except Exception:
            raise BrokerError(
                ProviderFailure.INTERNAL, "Approved provider credential is unavailable"
            ) from None
        call = AdapterRequest(
            registry_key,
            model,
            request.messages,
            request.settings,
            request.tools,
            timeout,
            tokens,
            request.digest,
        )
        try:
            response = adapter.complete(call, credential)
        except TimeoutError:
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.TIMEOUT,
                "Provider request timed out",
                True,
            )
        except RateLimitError:
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.RATE_LIMIT,
                "Provider rate limit reached",
                True,
            )
        except Exception:
            # Never surface adapter exception text; SDK errors often contain auth headers.
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.PROVIDER_ERROR,
                "Provider request failed",
                True,
            )
        if not isinstance(response, AdapterResponse) or not isinstance(response.content, str):
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.MALFORMED,
                "Provider returned an invalid response",
                False,
            )
        if response.refused:
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.REFUSAL,
                "Provider refused the request",
                False,
            )
        for value in (response.input_tokens, response.output_tokens, response.cost_microusd):
            if value is not None and (not isinstance(value, int) or value < 0):
                return self._failure(
                    registry_key,
                    model,
                    request.digest,
                    start,
                    ProviderFailure.MALFORMED,
                    "Provider returned invalid usage metadata",
                    False,
                )
        if response.output_tokens is not None and response.output_tokens > tokens:
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.MALFORMED,
                "Provider exceeded the output-token cap",
                False,
            )
        if response.cost_microusd is not None and response.cost_microusd > budget_microusd:
            return self._failure(
                registry_key,
                model,
                request.digest,
                start,
                ProviderFailure.BUDGET,
                "Provider-reported cost exceeded the request budget",
                False,
            )
        content = self._redact(response.content, credential)[: self._max_response_chars]
        truncated = len(self._redact(response.content, credential)) > self._max_response_chars
        return ProviderResult(
            registry_key,
            model,
            self._safe_id(response.actual_model, credential, 255),
            self._safe_id(response.request_id, credential, 255),
            request.digest,
            content,
            hashlib.sha256(content.encode()).hexdigest(),
            truncated,
            response.input_tokens,
            response.output_tokens,
            response.cost_microusd,
            int((time.monotonic() - start) * 1000),
        )

    def execute_fanout(
        self,
        *,
        project_id: str,
        targets: Sequence[tuple[str, str]],
        request: ProviderRequest,
        budget_microusd: int,
        timeout_seconds: float,
        token_cap: int,
    ) -> tuple[ProviderResult, ...]:
        """Sequentially execute the same canonical request across provider targets."""
        digest = request.digest
        results = tuple(
            self.execute(
                project_id=project_id,
                registry_key=key,
                model=model,
                request=request,
                budget_microusd=budget_microusd,
                timeout_seconds=timeout_seconds,
                token_cap=token_cap,
            )
            for key, model in targets
        )
        if any(item.request_digest != digest for item in results):
            raise BrokerError(ProviderFailure.INTERNAL, "Fanout request identity changed")
        return results

    def _redact(self, value: str, credential: str) -> str:
        if credential:
            value = value.replace(credential, "[REDACTED]")
        return _SENSITIVE.sub("[REDACTED]", value)

    @staticmethod
    def _safe_id(value: str | None, credential: str, limit: int) -> str | None:
        # Adapter metadata is untrusted. Discard unsafe fields entirely rather
        # than returning a truncated credential or partially redacted value.
        if not isinstance(value, str) or len(value) > limit:
            return None
        if credential in value or _SENSITIVE.search(value) or not _SAFE_IDENTIFIER.fullmatch(value):
            return None
        return value

    @staticmethod
    def _failure(
        key: str,
        model: str,
        digest: str,
        start: float,
        category: ProviderFailure,
        message: str,
        retriable: bool,
    ) -> ProviderResult:
        return ProviderResult(
            key,
            model,
            None,
            None,
            digest,
            None,
            None,
            False,
            None,
            None,
            None,
            int((time.monotonic() - start) * 1000),
            category.value,
            retriable,
            message[:256],
        )
