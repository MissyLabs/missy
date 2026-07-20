"""Tests for the reusable RoundRobinAccounts helper (F15)."""

from __future__ import annotations

import threading

from missy.providers.round_robin import AccountView, RoundRobinAccounts


class _FakeLimiter:
    _counter = 0

    def __init__(self) -> None:
        type(self)._counter += 1
        self.id = type(self)._counter


def _rr(keys, **kw):
    return RoundRobinAccounts(keys, make_rate_limiter=_FakeLimiter, **kw)


class TestActivation:
    def test_single_key_not_activated(self) -> None:
        rr = _rr(["k1"])
        assert rr.is_multi_account is False
        assert rr.count == 0
        assert rr.select() is None

    def test_empty_keys_not_activated(self) -> None:
        assert _rr([]).is_multi_account is False
        assert _rr(None).is_multi_account is False

    def test_two_keys_activate(self) -> None:
        rr = _rr(["k1", "k2"])
        assert rr.is_multi_account is True
        assert rr.count == 2

    def test_min_accounts_override(self) -> None:
        # With min_accounts=1, a single key activates.
        assert _rr(["only"], min_accounts=1).is_multi_account is True


class TestAccounts:
    def test_public_views_have_indices_but_no_credentials(self) -> None:
        rr = _rr(["a", "b", "c"])
        assert [acc.index for acc in rr.accounts] == [0, 1, 2]
        assert not any(hasattr(acc, "api_key") for acc in rr.accounts)

    def test_public_views_are_immutable_snapshots(self) -> None:
        rr = _rr(["a", "b"])
        views = rr.accounts
        assert isinstance(views, tuple)
        assert views is not rr.accounts

    def test_client_readiness_is_credential_free(self) -> None:
        rr = _rr(["a", "b"])
        assert all(not acc.client_ready for acc in rr.accounts)


class TestSelection:
    def test_round_robins_in_order(self) -> None:
        rr = _rr(["a", "b", "c"])
        picks = [rr.select().index for _ in range(6)]
        assert picks == [0, 1, 2, 0, 1, 2]

    def test_returns_safe_account_views(self) -> None:
        rr = _rr(["secret-key-alpha", "secret-key-beta"])
        acc = rr.select()
        assert isinstance(acc, AccountView)
        assert "secret-key" not in repr(acc)
        assert not hasattr(acc, "api_key")

    def test_concurrent_selection_distributes_evenly(self) -> None:
        rr = _rr([f"k{i}" for i in range(4)])
        counts = [0, 0, 0, 0]
        lock = threading.Lock()

        def worker():
            for _ in range(250):
                acc = rr.select()
                with lock:
                    counts[acc.index] += 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # 4 threads * 250 = 1000 selects across 4 accounts -> exactly balanced,
        # proving the atomic index advance loses/duplicates no turns.
        assert counts == [250, 250, 250, 250]


class TestOpenAIIntegration:
    def test_openai_provider_uses_helper(self) -> None:
        # The provider's account list is the helper's, and single-key config
        # leaves balancing inactive (unchanged legacy behaviour).
        from missy.config.settings import ProviderConfig
        from missy.providers.openai_provider import OpenAIProvider

        single = OpenAIProvider(
            ProviderConfig(name="openai", model="auto", api_key="k", api_keys=["k"])
        )
        assert single.is_multi_account is False

        multi = OpenAIProvider(
            ProviderConfig(
                name="openai",
                model="auto",
                api_keys=["k1", "k2"],
                key_rotation_strategy="round_robin",
            )
        )
        assert multi.is_multi_account is True
        assert multi.account_count == 2
        # Selection round-robins through the shared helper.
        assert multi._select_account().index == 0
        assert multi._select_account().index == 1
        assert multi._select_account().index == 0
