"""Direct provider regressions derived from validation run 19."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any

import pytest

from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.config.settings import ProviderConfig
from missy.core.exceptions import MissyError
from missy.providers.base import Message
from missy.providers.openai_provider import OpenAIProvider
from missy.providers.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    RateLimitRequestTooLarge,
)
from missy.providers.registry import ProviderRegistry
from missy.providers.schema_adapter import (
    canonical_from_anthropic,
    canonical_from_openai,
    normalize_for_provider,
)


class _Clock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Provider:
    name = "fake"

    def is_available(self) -> bool:
        return True


def _open_breaker(breaker: CircuitBreaker) -> None:
    with pytest.raises(RuntimeError, match="down"):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert breaker.state == CircuitState.OPEN


def test_prov_014_half_open_admits_exactly_one_concurrent_probe() -> None:
    clock = _Clock()
    breaker = CircuitBreaker("primary", threshold=1, base_timeout=10, max_timeout=40, clock=clock)
    _open_breaker(breaker)
    clock.value = 10

    start = threading.Barrier(51)
    probe_entered = threading.Event()
    release_probe = threading.Event()
    provider_calls = 0
    provider_lock = threading.Lock()

    def provider() -> str:
        nonlocal provider_calls
        with provider_lock:
            provider_calls += 1
        probe_entered.set()
        assert release_probe.wait(2)
        return "recovered"

    def caller() -> str:
        start.wait()
        try:
            return breaker.call(provider)
        except MissyError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=50) as pool:
        futures = [pool.submit(caller) for _ in range(50)]
        start.wait()
        assert probe_entered.wait(1)
        deadline = time.monotonic() + 1
        while sum(f.done() for f in futures) < 49 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert sum(f.done() for f in futures) == 49
        release_probe.set()
        results = [future.result(timeout=2) for future in futures]

    assert provider_calls == 1
    assert results.count("recovered") == 1
    assert results.count("rejected") == 49
    assert breaker.state == CircuitState.CLOSED


def test_prov_015_failed_half_open_probes_back_off_to_cap_and_success_resets() -> None:
    clock = _Clock()
    breaker = CircuitBreaker("primary", threshold=1, base_timeout=2, max_timeout=8, clock=clock)
    _open_breaker(breaker)

    for now, expected_timeout in ((2, 4), (6, 8), (14, 8), (22, 8)):
        clock.value = now
        with pytest.raises(RuntimeError, match="probe"):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("probe")))
        assert breaker.state == CircuitState.OPEN
        assert breaker._recovery_timeout == expected_timeout
        clock.value = now + expected_timeout - 0.01
        with pytest.raises(MissyError, match="OPEN"):
            breaker.call(lambda: "too early")

    clock.value = 30
    assert breaker.call(lambda: "healthy") == "healthy"
    assert breaker.state == CircuitState.CLOSED
    assert breaker._recovery_timeout == 2
    assert breaker._failure_count == 0


def test_prov_027_default_transition_revalidates_after_concurrent_disable() -> None:
    registry = ProviderRegistry()
    provider = _Provider()
    registry.register("primary", provider)  # type: ignore[arg-type]
    probe_entered = threading.Event()
    release_probe = threading.Event()

    def delayed_available(name: str, candidate: Any) -> bool:
        assert name == "primary"
        assert candidate is provider
        probe_entered.set()
        assert release_probe.wait(2)
        return True

    registry._availability_for = delayed_available  # type: ignore[method-assign]
    errors: list[Exception] = []

    def select_default() -> None:
        try:
            registry.set_default("primary")
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=select_default)
    thread.start()
    assert probe_entered.wait(1)
    registry.set_enabled("primary", False)
    release_probe.set()
    thread.join(2)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert "disabled" in str(errors[0])
    assert registry.get_default_name() is None
    assert not registry.is_enabled("primary")


def test_prov_029_concurrent_reconciliation_is_ordered_and_exactly_once() -> None:
    clock = _Clock(100)
    limiter = RateLimiter(
        requests_per_minute=0,
        tokens_per_minute=100,
        max_wait_seconds=0,
        clock=clock,
        sleeper=lambda _: None,
    )
    first = limiter.acquire(tokens=40, reconcile=True)
    second = limiter.acquire(tokens=40, reconcile=True)
    assert first is not None and second is not None
    assert limiter.token_capacity == 20

    # Response 2 arrives first. Its refund is held until response 1 settles,
    # so response scheduling cannot change bucket arithmetic.
    assert limiter.record_usage(prompt_tokens=10, reservation=second)
    assert limiter.token_capacity == 20
    assert limiter.record_usage(prompt_tokens=70, reservation=first)
    assert limiter.token_capacity == 30

    # A duplicate response cannot refund or charge the request again.
    assert not limiter.record_usage(prompt_tokens=0, reservation=second)
    assert limiter.token_capacity == 30


def test_prov_029_concurrent_acquisition_never_oversubscribes() -> None:
    clock = _Clock(10)
    limiter = RateLimiter(
        requests_per_minute=0,
        tokens_per_minute=100,
        max_wait_seconds=0,
        clock=clock,
        sleeper=lambda _: None,
    )
    start = threading.Barrier(21)

    def acquire() -> str:
        start.wait()
        try:
            limiter.acquire(tokens=10)
            return "allowed"
        except RateLimitExceeded:
            return "denied"

    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(acquire) for _ in range(20)]
        start.wait()
        outcomes = [future.result(timeout=2) for future in futures]

    assert outcomes.count("allowed") == 10
    assert outcomes.count("denied") == 10
    assert limiter.token_capacity == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"requests_per_minute": -1}, "requests_per_minute"),
        ({"tokens_per_minute": -1}, "tokens_per_minute"),
        ({"requests_per_minute": 1.5}, "requests_per_minute"),
        ({"max_wait_seconds": float("nan")}, "max_wait_seconds"),
        ({"max_wait_seconds": float("inf")}, "max_wait_seconds"),
    ],
)
def test_prov_030_invalid_configuration_rejects_early(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RateLimiter(**kwargs)


@pytest.mark.parametrize("tokens", [-1, float("nan"), float("inf"), True, "1"])
def test_prov_030_invalid_request_tokens_reject_early(tokens: Any) -> None:
    limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=100)
    with pytest.raises(ValueError, match="tokens"):
        limiter.acquire(tokens=tokens)


def test_prov_030_unlimited_oversized_and_clock_boundaries() -> None:
    unlimited = RateLimiter(requests_per_minute=None, tokens_per_minute=0)
    unlimited.acquire(tokens=10**9)

    clock = _Clock(100)
    limiter = RateLimiter(
        requests_per_minute=1,
        tokens_per_minute=100,
        max_wait_seconds=0,
        clock=clock,
        sleeper=lambda _: None,
    )
    with pytest.raises(RateLimitRequestTooLarge, match="exceeds"):
        limiter.acquire(tokens=101)

    limiter.acquire(tokens=100)
    assert limiter.request_capacity == 0
    assert limiter.token_capacity == 0

    clock.value = 90
    assert limiter.request_capacity == 0
    assert limiter.token_capacity == 0
    assert limiter._req_last_refill == 100
    assert limiter._tok_last_refill == 100

    clock.value = 10_000
    assert limiter.request_capacity == 1
    assert limiter.token_capacity == 100


@pytest.mark.parametrize("usage", [-1, float("nan"), float("inf"), True, "1"])
def test_prov_030_invalid_reported_usage_rejects_early(usage: Any) -> None:
    limiter = RateLimiter(requests_per_minute=0, tokens_per_minute=100)
    with pytest.raises(ValueError, match="prompt_tokens"):
        limiter.record_usage(prompt_tokens=usage)


def _assert_adapter_contract(exchange: dict[str, Any]) -> None:
    """Semantic oracle shared by genuine adapters and deliberate mutants."""
    schema = exchange["schema"]
    assert schema["name"] == "lookup"
    assert schema["description"] == "Look up one record"
    assert schema["parameters"]["required"] == ["record_id"]
    assert schema["parameters"]["properties"]["record_id"]["type"] == "string"
    assert [message["role"] for message in exchange["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert exchange["tool_call"]["id"] == exchange["tool_result"]["tool_call_id"]
    assert exchange["text"] == "hello"
    assert exchange["wire_model"] == exchange["requested_model"]


def test_prov_050_all_five_schema_adapters_preserve_canonical_semantics() -> None:
    canonical = {
        "name": "lookup",
        "description": "Look up one record",
        "parameters": {
            "type": "object",
            "properties": {"record_id": {"type": "string"}},
            "required": ["record_id"],
        },
    }
    original = deepcopy(canonical)
    provider = OpenAIProvider(ProviderConfig(name="openai", model="configured-model"))
    messages = provider._messages_to_chat_payload(
        [
            Message("system", "rules"),
            Message("user", "question"),
            Message("assistant", "answer"),
        ]
    )
    first, accumulated = provider._append_reconciled_text("", "hel")
    duplicate, accumulated = provider._append_reconciled_text(accumulated, "hel")
    last, accumulated = provider._append_reconciled_text(accumulated, "hello")
    assert first + duplicate + last == "hello"

    for adapter in ("anthropic", "openai", "ollama", "mistral", "gemini"):
        wire = normalize_for_provider(canonical, adapter)
        if adapter == "anthropic":
            recovered = canonical_from_anthropic(wire)
        elif adapter in {"openai", "ollama", "mistral"}:
            recovered = canonical_from_openai(wire)
        else:
            recovered = wire
        exchange = {
            "schema": recovered,
            "messages": messages,
            "tool_call": {"id": "call-7", "name": "lookup"},
            "tool_result": {"tool_call_id": "call-7", "content": "record"},
            "text": accumulated,
            "requested_model": "configured-model",
            "wire_model": provider._resolve_model("configured-model"),
        }
        _assert_adapter_contract(exchange)

    assert canonical == original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["schema"].update(
            parameters={**value["schema"]["parameters"], "required": []}
        ),
        lambda value: value["messages"].__setitem__(
            1, {**value["messages"][1], "role": "assistant"}
        ),
        lambda value: value["tool_result"].update(tool_call_id="call-other"),
        lambda value: value.update(text="hellohello"),
        lambda value: value.update(wire_model="mutant-model"),
    ],
    ids=("drop-required", "swap-role", "swap-tool-id", "duplicate-delta", "leak-model"),
)
def test_prov_050_each_seeded_mutant_is_caught(mutate: Any) -> None:
    exchange = {
        "schema": {
            "name": "lookup",
            "description": "Look up one record",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        },
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "tool_call": {"id": "call-7", "name": "lookup"},
        "tool_result": {"tool_call_id": "call-7", "content": "record"},
        "text": "hello",
        "requested_model": "configured-model",
        "wire_model": "configured-model",
    }
    mutate(exchange)
    with pytest.raises(AssertionError):
        _assert_adapter_contract(exchange)
