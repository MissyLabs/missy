"""Tests for the reusable RoundRobinAccounts helper (F15)."""

from __future__ import annotations

import threading
import time

from missy.providers.round_robin import Account, AccountView, RoundRobinAccounts


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


# ---------------------------------------------------------------------------
# Per-account failure tracking / backoff
# ---------------------------------------------------------------------------


class TestFailureTracking:
    def test_below_threshold_stays_healthy(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=3)
        account = rr._live_accounts[0]
        rr.record_failure(account)
        rr.record_failure(account)
        assert account.unhealthy_until == 0.0
        assert rr.accounts[0].healthy is True

    def test_reaching_threshold_opens_the_account(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=3, base_backoff_seconds=10.0)
        account = rr._live_accounts[0]
        rr.record_failure(account)
        rr.record_failure(account)
        rr.record_failure(account)
        assert account.unhealthy_until > time.monotonic()
        assert rr.accounts[0].healthy is False
        assert rr.accounts[0].consecutive_failures == 3

    def test_success_clears_failure_count_and_backoff(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=3, base_backoff_seconds=10.0)
        account = rr._live_accounts[0]
        for _ in range(3):
            rr.record_failure(account)
        assert account.unhealthy_until > 0.0

        rr.record_success(account)
        assert account.consecutive_failures == 0
        assert account.backoff_seconds == 0.0
        assert account.unhealthy_until == 0.0
        assert rr.accounts[0].healthy is True

    def test_success_below_threshold_also_resets_partial_failures(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=5)
        account = rr._live_accounts[0]
        rr.record_failure(account)
        rr.record_failure(account)
        rr.record_success(account)
        assert account.consecutive_failures == 0

    def test_selection_skips_unhealthy_account(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=1, base_backoff_seconds=999.0)
        bad = rr._live_accounts[0]
        rr.record_failure(bad)  # threshold=1 -> opens immediately

        picks = [rr.select().index for _ in range(4)]
        assert picks == [1, 1, 1, 1]

    def test_selection_resumes_using_recovered_account(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=1, base_backoff_seconds=0.01)
        bad = rr._live_accounts[0]
        rr.record_failure(bad)
        assert rr.select().index == 1  # skipped while unhealthy

        time.sleep(0.02)  # backoff window elapses
        picks = {rr.select().index for _ in range(4)}
        assert 0 in picks  # account 0 is selectable again

    def test_all_accounts_unhealthy_fails_open(self) -> None:
        """Every account being in cooldown must not make selection return
        None -- a temporarily degraded provider should still be usable."""
        rr = _rr(["a", "b"], failure_threshold=1, base_backoff_seconds=999.0)
        for account in rr._live_accounts:
            rr.record_failure(account)

        result = rr.select()
        assert result is not None
        assert result.index in (0, 1)

    def test_all_unhealthy_returns_soonest_to_recover(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=1)
        accounts = rr._live_accounts
        now = time.monotonic()
        accounts[0].unhealthy_until = now + 100
        accounts[1].unhealthy_until = now + 5
        accounts[0].consecutive_failures = 1
        accounts[1].consecutive_failures = 1

        result = rr._select_live()
        assert result is accounts[1]

    def test_backoff_doubles_on_post_cooldown_probe_failure(self) -> None:
        rr = _rr(
            ["a", "b"],
            failure_threshold=1,
            base_backoff_seconds=10.0,
            max_backoff_seconds=1000.0,
        )
        account = rr._live_accounts[0]
        rr.record_failure(account)
        assert account.backoff_seconds == 10.0

        # Simulate the cooldown having already elapsed, then a fresh probe
        # also failing -- backoff should double, not restart at base.
        account.unhealthy_until = 0.0
        rr.record_failure(account)
        assert account.backoff_seconds == 20.0

    def test_backoff_is_capped_at_max(self) -> None:
        rr = _rr(
            ["a", "b"],
            failure_threshold=1,
            base_backoff_seconds=100.0,
            max_backoff_seconds=150.0,
        )
        account = rr._live_accounts[0]
        rr.record_failure(account)
        assert account.backoff_seconds == 100.0

        account.unhealthy_until = 0.0
        rr.record_failure(account)  # would double to 200 without the cap
        assert account.backoff_seconds == 150.0

    def test_default_thresholds_match_circuit_breaker_defaults(self) -> None:
        """Documented parity with CircuitBreaker's defaults (threshold=5,
        base_timeout=60, max_timeout=300) so operators reason about one
        failure-isolation policy across the codebase, not two."""
        rr = _rr(["a", "b"])
        account = rr._live_accounts[0]
        for _ in range(4):
            rr.record_failure(account)
        assert account.unhealthy_until == 0.0  # not yet at threshold 5
        rr.record_failure(account)
        assert account.backoff_seconds == 60.0

    def test_record_failure_on_unconfigured_single_account_is_a_no_op(self) -> None:
        """When balancing is inactive there are no live Account objects to
        mutate -- callers guard on the account they were given being
        non-None, but calling with a manually constructed Account must not
        raise even though it isn't tracked by this selector."""
        rr = _rr(["only"])  # below min_accounts -> inactive
        assert rr._live_accounts == []
        stray = Account(index=0, api_key="only", rate_limiter=_FakeLimiter())
        rr.record_failure(stray)  # must not raise
        assert stray.consecutive_failures == 1

    def test_accounts_property_reports_failure_count(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=5)
        rr.record_failure(rr._live_accounts[0])
        rr.record_failure(rr._live_accounts[0])
        views = {v.index: v for v in rr.accounts}
        assert views[0].consecutive_failures == 2
        assert views[0].healthy is True  # below threshold
        assert views[1].consecutive_failures == 0

    def test_concurrent_failure_recording_is_race_free(self) -> None:
        rr = _rr(["a", "b"], failure_threshold=50, base_backoff_seconds=10.0)
        account = rr._live_accounts[0]

        def worker():
            for _ in range(25):
                rr.record_failure(account)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 4 threads * 25 = 100 failures, no lost increments.
        assert account.consecutive_failures == 100
        assert account.unhealthy_until > time.monotonic()

    def test_healthy_account_selection_unaffected_by_health_tracking(self) -> None:
        """No account has ever failed -- selection order must be identical
        to plain round-robin (regression guard for the health-aware scan)."""
        rr = _rr(["a", "b", "c"])
        picks = [rr.select().index for _ in range(6)]
        assert picks == [0, 1, 2, 0, 1, 2]


class TestCodexIntegration:
    def test_codex_provider_uses_helper_for_oauth_accounts(self) -> None:
        from missy.config.settings import ProviderConfig
        from missy.providers.codex_provider import CodexProvider

        single = CodexProvider(ProviderConfig(name="openai-codex", model="gpt-5.2", api_key="k"))
        assert single.is_multi_account is False

        multi = CodexProvider(
            ProviderConfig(
                name="openai-codex",
                model="gpt-5.2",
                oauth_accounts=["work", "personal"],
                key_rotation_strategy="round_robin",
            )
        )
        assert multi.is_multi_account is True
        assert multi.account_count == 2

    def test_codex_provider_skips_unhealthy_account_after_repeated_failures(self) -> None:
        from missy.config.settings import ProviderConfig
        from missy.providers.codex_provider import CodexProvider

        provider = CodexProvider(
            ProviderConfig(
                name="openai-codex",
                model="gpt-5.2",
                oauth_accounts=["work", "personal"],
                key_rotation_strategy="round_robin",
            )
        )
        work = provider._rr._live_accounts[0]
        # Simulate "work" repeatedly failing (e.g. exhausted quota).
        for _ in range(5):
            provider._rr.record_failure(work)

        picks = [provider._select_account().api_key for _ in range(4)]
        assert picks == ["personal", "personal", "personal", "personal"]
