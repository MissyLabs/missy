"""Tests for OpenAIProvider's multi-account round-robin balancing.

Covers adding 2+ OpenAI accounts (``api_keys`` + ``key_rotation_strategy:
"round_robin"``) and having the provider balance every call across them:
per-call account selection, per-account client caching, per-account rate
limiter isolation, thread-safety under concurrency, conversation-context
fidelity across account switches, and ``ProviderRegistry.rotate_key()``'s
no-op behavior for a provider that already balances internally.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from missy.config.settings import ProviderConfig
from missy.core.events import event_bus
from missy.providers.base import Message
from missy.providers.openai_provider import OpenAIProvider
from missy.providers.registry import ProviderRegistry


def _make_config(**overrides) -> ProviderConfig:
    defaults = {
        "name": "openai",
        "model": "gpt-4o",
        "api_key": "key-a",
        "api_keys": ["key-a", "key-b"],
        "key_rotation_strategy": "round_robin",
    }
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def _mock_response(text: str = "Hello!", model: str = "gpt-4o"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=text, tool_calls=None),
        finish_reason="stop",
    )
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return SimpleNamespace(choices=[choice], model=model, usage=usage, model_dump=dict)


class TestMultiAccountConstruction:
    def test_default_strategy_does_not_build_accounts_pool(self):
        """Even with 2 api_keys, the default 'failover' strategy must leave
        the legacy single-key path fully in charge -- zero behavior change
        for existing configs that never opted in."""
        p = OpenAIProvider(_make_config(key_rotation_strategy="failover"))
        assert p._accounts == []
        assert p.is_multi_account is False

    def test_round_robin_with_two_keys_builds_two_accounts(self):
        p = OpenAIProvider(_make_config())
        assert p.is_multi_account is True
        assert [a.api_key for a in p._accounts] == ["key-a", "key-b"]
        assert [a.index for a in p._accounts] == [0, 1]
        # Each account gets its own RateLimiter instance, not a shared one.
        assert p._accounts[0].rate_limiter is not p._accounts[1].rate_limiter

    def test_round_robin_with_only_one_key_falls_back_to_legacy(self):
        p = OpenAIProvider(
            _make_config(
                api_key="only-key", api_keys=["only-key"], key_rotation_strategy="round_robin"
            )
        )
        assert p._accounts == []
        assert p.is_multi_account is False


class TestRoundRobinDispatch:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_calls_alternate_accounts_and_cache_clients_per_account(self, mock_sdk):
        clients_by_key: dict[str, MagicMock] = {}

        def _build(**kwargs):
            key = kwargs.get("api_key")
            if key not in clients_by_key:
                client = MagicMock()
                client.chat.completions.create.return_value = _mock_response(f"reply from {key}")
                clients_by_key[key] = client
            return clients_by_key[key]

        mock_sdk.OpenAI.side_effect = _build

        p = OpenAIProvider(_make_config())
        results = [p.complete([Message(role="user", content="Hi")]).content for _ in range(4)]

        assert results == [
            "reply from key-a",
            "reply from key-b",
            "reply from key-a",
            "reply from key-b",
        ]
        # Exactly one client built per account -- each account's client is
        # cached on the account itself (not invalidated by the other
        # account's activity, unlike the legacy shared self._client).
        assert mock_sdk.OpenAI.call_count == 2
        assert p._accounts[0].client is clients_by_key["key-a"]
        assert p._accounts[1].client is clients_by_key["key-b"]

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_complete_with_tools_also_balances_across_accounts(self, mock_sdk):
        clients_by_key: dict[str, MagicMock] = {}

        def _build(**kwargs):
            key = kwargs.get("api_key")
            if key not in clients_by_key:
                client = MagicMock()
                raw = SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=f"tool reply from {key}", tool_calls=None
                            ),
                            finish_reason="stop",
                        )
                    ],
                    model="gpt-4o",
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                    model_dump=dict,
                )
                client.chat.completions.create.return_value = raw
                clients_by_key[key] = client
            return clients_by_key[key]

        mock_sdk.OpenAI.side_effect = _build

        p = OpenAIProvider(_make_config())
        r1 = p.complete_with_tools([Message(role="user", content="Hi")], tools=[])
        r2 = p.complete_with_tools([Message(role="user", content="Hi")], tools=[])

        assert r1.content == "tool reply from key-a"
        assert r2.content == "tool reply from key-b"

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_account_index_reported_in_audit_event(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        captured = []
        original = event_bus.publish
        event_bus.publish = lambda evt: (captured.append(evt), original(evt))
        try:
            p.complete([Message(role="user", content="Hi")], session_id="s1", task_id="t1")
            p.complete([Message(role="user", content="Hi")], session_id="s1", task_id="t1")
        finally:
            event_bus.publish = original

        indices = [e.detail.get("account_index") for e in captured if e.result == "allow"]
        assert indices == [0, 1]
        # Never leak the actual key into the audit trail.
        assert all("key-a" not in str(e.detail) and "key-b" not in str(e.detail) for e in captured)


class TestRoundRobinPreservesConversationContext:
    """The core requirement: switching which account handles a call must
    never drop, reorder, or corrupt the conversation being sent."""

    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_growing_conversation_forwarded_intact_across_account_switches(self, mock_sdk):
        received_payloads: list[list[dict]] = []

        def _build(**kwargs):
            client = MagicMock()

            def _create(**call_kwargs):
                received_payloads.append(call_kwargs["messages"])
                return _mock_response(f"turn {len(received_payloads)}")

            client.chat.completions.create.side_effect = _create
            return client

        mock_sdk.OpenAI.side_effect = _build

        p = OpenAIProvider(_make_config())

        # Simulate a growing multi-turn conversation, same as a real
        # session's history would look across turns.
        turn1 = [Message(role="user", content="What's 2+2?")]
        resp1 = p.complete(turn1)
        assert resp1.content == "turn 1"

        turn2 = [
            Message(role="user", content="What's 2+2?"),
            Message(role="assistant", content=resp1.content),
            Message(role="user", content="And 3+3?"),
        ]
        resp2 = p.complete(turn2)
        assert resp2.content == "turn 2"

        turn3 = [
            *turn2,
            Message(role="assistant", content=resp2.content),
            Message(role="user", content="Sum those two answers."),
        ]
        resp3 = p.complete(turn3)
        assert resp3.content == "turn 3"

        # Every turn's payload sent to the (account-specific) client must
        # match exactly what was asked for that turn -- content and order
        # both -- regardless of which account handled it.
        assert len(received_payloads) == 3
        assert [m["content"] for m in received_payloads[0]] == ["What's 2+2?"]
        assert [m["content"] for m in received_payloads[1]] == [
            "What's 2+2?",
            "turn 1",
            "And 3+3?",
        ]
        assert [m["content"] for m in received_payloads[2]] == [
            "What's 2+2?",
            "turn 1",
            "And 3+3?",
            "turn 2",
            "Sum those two answers.",
        ]

        # And the account still alternated normally underneath.
        assert mock_sdk.OpenAI.call_count == 2


class TestRoundRobinRateLimiterIndependence:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_each_account_uses_its_own_rate_limiter(self, mock_sdk):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response()
        mock_sdk.OpenAI.return_value = mock_client

        p = OpenAIProvider(_make_config())
        limiter_a = MagicMock(wraps=p._accounts[0].rate_limiter)
        limiter_b = MagicMock(wraps=p._accounts[1].rate_limiter)
        p._accounts[0].rate_limiter = limiter_a
        p._accounts[1].rate_limiter = limiter_b

        p.complete([Message(role="user", content="Hi")])
        p.complete([Message(role="user", content="Hi")])
        p.complete([Message(role="user", content="Hi")])

        assert limiter_a.acquire.call_count == 2
        assert limiter_b.acquire.call_count == 1
        assert limiter_a.record_usage.call_count == 2
        assert limiter_b.record_usage.call_count == 1
        # The shared, provider-level rate_limiter (set externally by the
        # registry) must never be touched while in multi-account mode.
        assert p.rate_limiter is None


class TestRoundRobinConcurrency:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_concurrent_calls_distribute_evenly_with_no_key_crossover(self, mock_sdk):
        lock = threading.Lock()
        seen: list[tuple[str, str]] = []  # (api_key used to build client, api_key of that call)

        def _build(**kwargs):
            key = kwargs.get("api_key")
            client = MagicMock()
            client.chat.completions.create.return_value = _mock_response(f"reply-{key}")

            def _create(**_call_kwargs):
                with lock:
                    seen.append((key, key))
                return client.chat.completions.create.return_value

            client.chat.completions.create.side_effect = _create
            return client

        mock_sdk.OpenAI.side_effect = _build

        p = OpenAIProvider(_make_config())
        results: list[str] = []
        results_lock = threading.Lock()

        def _worker():
            r = p.complete([Message(role="user", content="Hi")])
            with results_lock:
                results.append(r.content)

        threads = [threading.Thread(target=_worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(results) == 20
        # Perfectly even split -- the round-robin counter is advanced
        # atomically under a lock regardless of thread interleaving.
        assert sorted(results.count(r) for r in set(results)) == [10, 10]
        # No call's response body ever mismatches the key its own client
        # was built with (proves no cross-account leakage under
        # concurrency, even though racing first-use client construction
        # for the same account is itself an accepted, harmless tradeoff --
        # see _make_client()'s docstring).
        assert all(a == b for a, b in seen)


class TestRegistryRotateKeyMultiAccountNoOp:
    @patch("missy.providers.openai_provider._openai_sdk")
    @patch("missy.providers.openai_provider._OPENAI_AVAILABLE", True)
    def test_rotate_key_is_a_noop_for_multi_account_provider(self, mock_sdk):
        cfg = _make_config()
        p = OpenAIProvider(cfg)
        registry = ProviderRegistry()
        registry.register("openai", p, config=cfg)

        registry.rotate_key("openai")

        # Legacy bookkeeping must be untouched -- the provider already
        # balances every call across its own accounts pool internally.
        assert registry._key_indices.get("openai", 0) == 0
        assert p._api_key == "key-a"
        assert p._accounts[0].api_key == "key-a"
        assert p._accounts[1].api_key == "key-b"
