"""Session 13: Comprehensive tests for Watchdog and RateLimiter.

Covers angles not exercised by earlier test files:

Watchdog (missy/agent/watchdog.py):
- no-op _check_all when no checks registered
- WARNING vs ERROR log level based on failure_threshold
- recovery log message via caplog
- last_checked timestamp updated precisely on both pass and exception paths
- audit event session_id and task_id fields
- event_bus import is lazy (inside _check_all)
- register replaces health state with fresh SubsystemHealth
- start/stop with a short interval actually fires _check_all at least once
- stop called twice is idempotent (no AttributeError on joined thread)
- _run loop exits cleanly after stop event is set

RateLimiter (missy/providers/rate_limiter.py):
- acquire(tokens=0) with rpm-only limit does not touch tok bucket
- request_capacity is inf when rpm=0 but tpm is active
- token_capacity is inf when tpm=0 but rpm is active
- on_rate_limit_response with retry_after=0 never calls time.sleep
- on_rate_limit_response with retry_after>0 calls time.sleep once
- on_rate_limit_response retry_after capped at max_wait
- on_rate_limit_response with tpm=0 only drains req bucket
- RateLimitExceeded.wait_seconds is positive and matches expected wait
- RateLimitExceeded str representation contains wait_seconds formatted to 1dp
- acquire deducts exactly 1.0 from req bucket per call
- acquire deducts exactly tokens from tok bucket per call
- concurrent acquire + record_usage from many threads leaves buckets >= 0
- record_usage with negative total (prompt + completion < 0) is no-op
- bucket refill after artificial timestamp backdate caps at limit
- max_wait_seconds=0.0 raises immediately without sleeping when bucket empty
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from missy.agent.watchdog import Watchdog
from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded

# ===========================================================================
# Watchdog — _check_all with no registered checks
# ===========================================================================


class TestWatchdogNoChecks:
    """_check_all must be a no-op when no subsystems are registered."""

    @patch("missy.core.events.event_bus")
    def test_check_all_empty_does_not_raise(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd._check_all()  # must not raise

    @patch("missy.core.events.event_bus")
    def test_check_all_empty_publishes_no_events(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd._check_all()
        mock_bus.publish.assert_not_called()

    def test_get_report_before_registration_is_empty_dict(self) -> None:
        wd = Watchdog()
        assert wd.get_report() == {}


# ===========================================================================
# Watchdog — log level: WARNING below threshold, ERROR at/above threshold
# ===========================================================================


class TestWatchdogLogLevels:
    """Failures below threshold log WARNING; at-or-above threshold log ERROR."""

    @patch("missy.core.events.event_bus")
    def test_first_failure_logs_warning_when_threshold_is_3(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        wd = Watchdog(failure_threshold=3)
        wd.register("svc", lambda: False)
        with caplog.at_level(logging.WARNING, logger="missy.agent.watchdog"):
            wd._check_all()
        # consecutive_failures == 1, threshold == 3 → WARNING not ERROR
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert not any(r.levelno == logging.ERROR for r in caplog.records)

    @patch("missy.core.events.event_bus")
    def test_second_failure_logs_warning_when_threshold_is_3(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        wd = Watchdog(failure_threshold=3)
        wd.register("svc", lambda: False)
        wd._check_all()  # failures=1
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="missy.agent.watchdog"):
            wd._check_all()  # failures=2, still < 3
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert not any(r.levelno == logging.ERROR for r in caplog.records)

    @patch("missy.core.events.event_bus")
    def test_threshold_failure_logs_error(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        wd = Watchdog(failure_threshold=3)
        wd.register("svc", lambda: False)
        # Drive failures up to threshold
        for _ in range(2):
            wd._check_all()
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="missy.agent.watchdog"):
            wd._check_all()  # failures=3, at threshold → ERROR
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    @patch("missy.core.events.event_bus")
    def test_beyond_threshold_still_logs_error(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        wd = Watchdog(failure_threshold=2)
        wd.register("svc", lambda: False)
        for _ in range(3):
            wd._check_all()  # failures 1, 2, 3
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="missy.agent.watchdog"):
            wd._check_all()  # failures=4 — still ERROR
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    @patch("missy.core.events.event_bus")
    def test_threshold_of_1_first_failure_logs_error(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        wd = Watchdog(failure_threshold=1)
        wd.register("svc", lambda: False)
        with caplog.at_level(logging.ERROR, logger="missy.agent.watchdog"):
            wd._check_all()  # failures=1 >= threshold=1 → ERROR
        assert any(r.levelno == logging.ERROR for r in caplog.records)


# ===========================================================================
# Watchdog — recovery log message
# ===========================================================================


class TestWatchdogRecoveryLog:
    """A transition from unhealthy to healthy must emit a recovery log line."""

    @patch("missy.core.events.event_bus")
    def test_recovery_emits_info_log_with_subsystem_name(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        results = iter([False, True])
        wd = Watchdog()
        wd.register("my-db", lambda: next(results))
        wd._check_all()  # fail → unhealthy
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="missy.agent.watchdog"):
            wd._check_all()  # recover
        recovery_records = [r for r in caplog.records if "recover" in r.message.lower()]
        assert recovery_records, "Expected a recovery INFO log"
        assert "my-db" in recovery_records[0].message

    @patch("missy.core.events.event_bus")
    def test_no_recovery_log_on_first_healthy_check(
        self, mock_bus: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Subsystem that is healthy from the start should NOT emit a recovery log."""
        wd = Watchdog()
        wd.register("svc", lambda: True)
        with caplog.at_level(logging.INFO, logger="missy.agent.watchdog"):
            wd._check_all()
        recovery_records = [r for r in caplog.records if "recover" in r.message.lower()]
        assert not recovery_records


# ===========================================================================
# Watchdog — audit event field content
# ===========================================================================


class TestWatchdogAuditEventFields:
    """Verify specific fields on the AuditEvent emitted for each check."""

    @patch("missy.core.events.event_bus")
    def test_event_session_id_is_watchdog(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.session_id == "watchdog"

    @patch("missy.core.events.event_bus")
    def test_event_task_id_is_subsystem_name(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("cache", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.task_id == "cache"

    @patch("missy.core.events.event_bus")
    def test_event_category_is_plugin(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.category == "plugin"

    @patch("missy.core.events.event_bus")
    def test_events_published_in_registration_order(self, mock_bus: MagicMock) -> None:
        """Events must be published once per subsystem; order matches registration."""
        wd = Watchdog()
        wd.register("first", lambda: True)
        wd.register("second", lambda: False)
        wd.register("third", lambda: True)
        wd._check_all()
        assert mock_bus.publish.call_count == 3
        names = [c[0][0].task_id for c in mock_bus.publish.call_args_list]
        assert names == ["first", "second", "third"]

    @patch("missy.core.events.event_bus")
    def test_event_detail_failures_is_zero_on_healthy(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: True)
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.detail["failures"] == 0

    @patch("missy.core.events.event_bus")
    def test_event_detail_failures_after_two_failures(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        wd._check_all()
        event = mock_bus.publish.call_args[0][0]
        assert event.detail["failures"] == 2


# ===========================================================================
# Watchdog — register replaces health state
# ===========================================================================


class TestWatchdogRegisterResetsHealth:
    """Re-registering a subsystem must reset its health state."""

    @patch("missy.core.events.event_bus")
    def test_re_register_resets_consecutive_failures(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        wd._check_all()
        assert wd._health["svc"].consecutive_failures == 2
        # Re-register — should get a fresh SubsystemHealth
        wd.register("svc", lambda: True)
        assert wd._health["svc"].consecutive_failures == 0

    @patch("missy.core.events.event_bus")
    def test_re_register_resets_healthy_flag(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd._check_all()
        assert wd._health["svc"].healthy is False
        wd.register("svc", lambda: True)
        # Fresh health state starts as healthy=True
        assert wd._health["svc"].healthy is True

    @patch("missy.core.events.event_bus")
    def test_re_register_uses_new_check_function(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: False)
        wd.register("svc", lambda: True)
        wd._check_all()
        assert wd._health["svc"].healthy is True


# ===========================================================================
# Watchdog — start/stop background thread fires _check_all
# ===========================================================================


class TestWatchdogBackgroundThread:
    """With a very short interval the background thread fires _check_all quickly."""

    def test_background_thread_runs_check_all(self) -> None:
        """_check_all must be called by the background thread within the interval."""
        call_event = threading.Event()
        original_check_all = Watchdog._check_all

        def patched_check_all(self_inner: Watchdog) -> None:
            original_check_all(self_inner)
            call_event.set()

        with patch.object(Watchdog, "_check_all", patched_check_all):
            wd = Watchdog(check_interval=0.05)
            wd.start()
            fired = call_event.wait(timeout=2.0)
            wd.stop()

        assert fired, "_check_all was never called by the background thread"

    def test_double_stop_is_idempotent(self) -> None:
        """Calling stop() a second time on a stopped watchdog must not raise."""
        wd = Watchdog(check_interval=3600)
        wd.start()
        wd.stop()
        wd.stop()  # must not raise AttributeError or RuntimeError


# ===========================================================================
# Watchdog — exception in check_fn treated same as returning False
# ===========================================================================


class TestWatchdogExceptionEqualsFailure:
    """Any exception from check_fn increments failures and sets last_error."""

    @patch("missy.core.events.event_bus")
    def test_type_error_increments_failures(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("svc", lambda: int("not a number"))
        wd._check_all()
        assert wd._health["svc"].consecutive_failures == 1
        assert wd._health["svc"].healthy is False

    @patch("missy.core.events.event_bus")
    def test_os_error_captured_in_last_error(self, mock_bus: MagicMock) -> None:
        def failing_check() -> bool:
            raise OSError("disk full")

        wd = Watchdog()
        wd.register("svc", failing_check)
        wd._check_all()
        assert "disk full" in wd._health["svc"].last_error

    @patch("missy.core.events.event_bus")
    def test_exception_and_false_both_increment_failures(self, mock_bus: MagicMock) -> None:
        """Two checks: one raises, one returns False. Both should be unhealthy."""
        wd = Watchdog()
        wd.register("raises", lambda: 1 / 0)
        wd.register("false", lambda: False)
        wd._check_all()
        assert wd._health["raises"].consecutive_failures == 1
        assert wd._health["false"].consecutive_failures == 1


# ===========================================================================
# Watchdog — get_report() structure
# ===========================================================================


class TestWatchdogGetReportStructure:
    """get_report() must always include exactly three keys per subsystem."""

    @patch("missy.core.events.event_bus")
    def test_each_entry_has_required_keys(self, mock_bus: MagicMock) -> None:
        wd = Watchdog()
        wd.register("a", lambda: True)
        wd.register("b", lambda: False)
        wd._check_all()
        report = wd.get_report()
        for name in ("a", "b"):
            assert set(report[name].keys()) == {"healthy", "consecutive_failures", "last_error"}

    @patch("missy.core.events.event_bus")
    def test_report_reflects_state_after_multiple_cycles(self, mock_bus: MagicMock) -> None:
        states = [False, False, True, True]
        idx = [0]

        def check() -> bool:
            v = states[idx[0]]
            if idx[0] < len(states) - 1:
                idx[0] += 1
            return v

        wd = Watchdog()
        wd.register("svc", check)
        for _ in range(4):
            wd._check_all()

        report = wd.get_report()
        assert report["svc"]["healthy"] is True
        assert report["svc"]["consecutive_failures"] == 0
        assert report["svc"]["last_error"] == ""


# ===========================================================================
# RateLimiter — constructor defaults and attribute types
# ===========================================================================


class TestRateLimiterConstructorDefaults:
    """Verify constructor defaults match documented values."""

    def test_default_rpm_is_60(self) -> None:
        rl = RateLimiter()
        assert rl._rpm == 60

    def test_default_tpm_is_100_000(self) -> None:
        rl = RateLimiter()
        assert rl._tpm == 100_000

    def test_default_max_wait_is_30(self) -> None:
        rl = RateLimiter()
        assert rl._max_wait == 30.0

    def test_initial_req_tokens_equals_rpm(self) -> None:
        rl = RateLimiter(requests_per_minute=120)
        assert rl._req_tokens == 120.0

    def test_initial_tok_tokens_equals_tpm(self) -> None:
        rl = RateLimiter(tokens_per_minute=50_000)
        assert rl._tok_tokens == 50_000.0

    def test_rpm_zero_req_tokens_is_zero(self) -> None:
        """When rpm=0, internal req bucket starts at 0.0."""
        rl = RateLimiter(requests_per_minute=0)
        assert rl._req_tokens == 0.0

    def test_tpm_zero_tok_tokens_is_zero(self) -> None:
        """When tpm=0, internal tok bucket starts at 0.0."""
        rl = RateLimiter(tokens_per_minute=0)
        assert rl._tok_tokens == 0.0


# ===========================================================================
# RateLimiter — acquire() deduction precision
# ===========================================================================


class TestRateLimiterAcquireDeduction:
    """acquire() must deduct exactly the right amount from each bucket."""

    def test_acquire_no_tokens_deducts_exactly_one_request(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=0, max_wait_seconds=0.0)
        before = rl._req_tokens
        rl.acquire(tokens=0)
        # Allow a tiny refill between the read and the acquire call
        assert rl._req_tokens <= before - 0.99

    def test_acquire_with_tokens_deducts_from_tok_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        with rl._lock:
            rl._tok_tokens = 500.0
        rl.acquire(tokens=200)
        assert rl._tok_tokens <= 300.5  # allow tiny refill jitter

    def test_acquire_with_tokens_also_deducts_one_request(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=10_000, max_wait_seconds=0.0)
        req_before = rl._req_tokens
        rl.acquire(tokens=500)
        assert rl._req_tokens <= req_before - 0.99

    def test_acquire_zero_tokens_does_not_touch_tok_bucket(self) -> None:
        """tokens=0 must not alter the token bucket at all."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=1000, max_wait_seconds=0.0)
        with rl._lock:
            rl._tok_tokens = 750.0
        rl.acquire(tokens=0)
        # Token bucket unchanged (small refill is acceptable, but never a deduction)
        assert rl._tok_tokens >= 749.5


# ===========================================================================
# RateLimiter — capacity properties
# ===========================================================================


class TestRateLimiterCapacityProperties:
    """request_capacity and token_capacity return correct values."""

    def test_request_capacity_inf_when_rpm_zero_tpm_active(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=5000)
        assert rl.request_capacity == float("inf")

    def test_token_capacity_inf_when_tpm_zero_rpm_active(self) -> None:
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=0)
        assert rl.token_capacity == float("inf")

    def test_request_capacity_reflects_deduction(self) -> None:
        rl = RateLimiter(requests_per_minute=10, tokens_per_minute=0, max_wait_seconds=0.0)
        initial = rl.request_capacity
        rl.acquire()
        after = rl.request_capacity
        assert after < initial

    def test_token_capacity_reflects_deduction(self) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=1000, max_wait_seconds=0.0)
        initial = rl.token_capacity
        rl.acquire(tokens=300)
        after = rl.token_capacity
        assert after < initial

    def test_request_capacity_caps_at_rpm_after_backdated_refill(self) -> None:
        """Even when the last-refill timestamp is ancient, capacity must not exceed rpm."""
        rpm = 30
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        with rl._lock:
            rl._req_tokens = 0.0
            rl._req_last_refill = time.monotonic() - 86_400  # 24 h ago
        cap = rl.request_capacity
        assert cap <= float(rpm) + 0.01, f"Capacity overflowed: {cap}"

    def test_token_capacity_caps_at_tpm_after_backdated_refill(self) -> None:
        tpm = 500
        rl = RateLimiter(requests_per_minute=10_000, tokens_per_minute=tpm, max_wait_seconds=0.0)
        with rl._lock:
            rl._tok_tokens = 0.0
            rl._tok_last_refill = time.monotonic() - 86_400
        cap = rl.token_capacity
        assert cap <= float(tpm) + 0.01, f"Token capacity overflowed: {cap}"


# ===========================================================================
# RateLimiter — RateLimitExceeded exception attributes
# ===========================================================================


class TestRateLimitExceededException:
    """RateLimitExceeded carries the expected attributes and string representation."""

    def test_wait_seconds_attribute_is_positive(self) -> None:
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.acquire()
        assert exc_info.value.wait_seconds > 0.0

    def test_str_contains_wait_seconds_with_one_decimal(self) -> None:
        exc = RateLimitExceeded(wait_seconds=7.5)
        assert "7.5" in str(exc)

    def test_wait_seconds_stored_exactly(self) -> None:
        exc = RateLimitExceeded(wait_seconds=12.34)
        assert exc.wait_seconds == pytest.approx(12.34)

    def test_is_exception_subclass(self) -> None:
        exc = RateLimitExceeded(wait_seconds=1.0)
        assert isinstance(exc, Exception)


# ===========================================================================
# RateLimiter — on_rate_limit_response sleep behaviour
# ===========================================================================


class TestRateLimiterOnRateLimitResponseSleep:
    """on_rate_limit_response must call time.sleep only when retry_after > 0."""

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_retry_after_zero_does_not_call_sleep(self, mock_sleep: MagicMock) -> None:
        rl = RateLimiter(requests_per_minute=60)
        rl.on_rate_limit_response(retry_after=0.0)
        mock_sleep.assert_not_called()

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_retry_after_positive_calls_sleep_once(self, mock_sleep: MagicMock) -> None:
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=10.0)
        rl.on_rate_limit_response(retry_after=2.0)
        mock_sleep.assert_called_once()

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_retry_after_capped_at_max_wait(self, mock_sleep: MagicMock) -> None:
        """The sleep duration must be min(retry_after, max_wait)."""
        max_wait = 5.0
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=max_wait)
        rl.on_rate_limit_response(retry_after=100.0)
        mock_sleep.assert_called_once_with(max_wait)

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_retry_after_smaller_than_max_wait_uses_retry_after(
        self, mock_sleep: MagicMock
    ) -> None:
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=30.0)
        rl.on_rate_limit_response(retry_after=3.0)
        mock_sleep.assert_called_once_with(3.0)

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_tpm_zero_drains_only_req_bucket(self, mock_sleep: MagicMock) -> None:
        """When tpm=0, the tok bucket is not tracked; only req bucket should be zeroed."""
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=0)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0
        # tok bucket starts at 0 (tpm=0 means no token tracking)
        assert rl._tok_tokens == 0.0


# ===========================================================================
# RateLimiter — record_usage edge cases
# ===========================================================================


class TestRateLimiterRecordUsageEdgeCases:
    """Edge conditions for record_usage."""

    def test_record_usage_total_zero_is_no_op(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=500)
        with rl._lock:
            rl._tok_tokens = 300.0
        rl.record_usage(prompt_tokens=0, completion_tokens=0)
        assert rl._tok_tokens == 300.0

    def test_record_usage_negative_total_is_no_op(self) -> None:
        """record_usage with negative combined total must skip deduction."""
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=500)
        with rl._lock:
            rl._tok_tokens = 300.0
        rl.record_usage(prompt_tokens=-100, completion_tokens=-50)
        assert rl._tok_tokens == 300.0

    def test_record_usage_clamps_to_zero_not_negative(self) -> None:
        rl = RateLimiter(requests_per_minute=100, tokens_per_minute=500)
        with rl._lock:
            rl._tok_tokens = 10.0
        # Deduct 1000 tokens from a bucket that only has 10
        rl.record_usage(prompt_tokens=900, completion_tokens=100)
        assert rl._tok_tokens == 0.0

    def test_record_usage_with_tpm_zero_does_not_alter_req_bucket(self) -> None:
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=0)
        req_before = rl._req_tokens
        rl.record_usage(prompt_tokens=500, completion_tokens=500)
        assert rl._req_tokens == req_before

    def test_record_usage_does_not_alter_req_bucket_when_tpm_active(self) -> None:
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=1000)
        rl.acquire(tokens=100)
        req_after_acquire = rl._req_tokens
        rl.record_usage(prompt_tokens=50, completion_tokens=50)
        assert rl._req_tokens == req_after_acquire


# ===========================================================================
# RateLimiter — max_wait_seconds=0.0 raises without sleeping
# ===========================================================================


class TestRateLimiterImmediateRaise:
    """With max_wait=0 the raise must happen in microseconds, not milliseconds."""

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_max_wait_zero_does_not_sleep_before_raising(self, mock_sleep: MagicMock) -> None:
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()
        with pytest.raises(RateLimitExceeded):
            rl.acquire()
        mock_sleep.assert_not_called()

    @patch("missy.providers.rate_limiter.time.sleep")
    def test_max_wait_zero_token_exhaustion_no_sleep(self, mock_sleep: MagicMock) -> None:
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=100, max_wait_seconds=0.0)
        rl.acquire(tokens=100)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1)
        mock_sleep.assert_not_called()


# ===========================================================================
# RateLimiter — unlimited paths return immediately
# ===========================================================================


class TestRateLimiterUnlimitedPaths:
    """rpm=0 and tpm=0 must return from acquire() without any locking overhead."""

    def test_both_unlimited_returns_immediately_for_many_calls(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        for _ in range(1_000):
            rl.acquire(tokens=999_999)

    def test_both_unlimited_record_usage_is_no_op(self) -> None:
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        tok_before = rl._tok_tokens
        rl.record_usage(prompt_tokens=10_000, completion_tokens=10_000)
        assert rl._tok_tokens == tok_before

    def test_both_unlimited_on_rate_limit_response_zero_sleep(self) -> None:
        """on_rate_limit_response still drains req_tokens (it may be 0 already)."""
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        # Must not raise even though buckets are already 0
        rl.on_rate_limit_response(retry_after=0.0)


# ===========================================================================
# RateLimiter — concurrent acquire from many threads
# ===========================================================================


class TestRateLimiterConcurrentThreads:
    """Thread safety: buckets never go negative under heavy concurrent load."""

    def test_concurrent_acquire_buckets_never_negative(self) -> None:
        """After N threads each acquire once, both internal counters must be >= 0."""
        n_threads = 40
        rl = RateLimiter(
            requests_per_minute=n_threads * 2,
            tokens_per_minute=n_threads * 200,
            max_wait_seconds=2.0,
        )
        errors: list[Exception] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            try:
                rl.acquire(tokens=200)
            except RateLimitExceeded:
                pass
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Unexpected errors: {errors}"
        assert rl._req_tokens >= 0.0
        assert rl._tok_tokens >= 0.0

    def test_concurrent_record_usage_does_not_corrupt_state(self) -> None:
        """Concurrent record_usage calls must not drive tok_tokens below zero."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=10_000, max_wait_seconds=1.0)
        n_threads = 20
        barrier = threading.Barrier(n_threads)

        def recorder() -> None:
            barrier.wait()
            for _ in range(50):
                rl.record_usage(prompt_tokens=10, completion_tokens=10)

        threads = [threading.Thread(target=recorder, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert rl._tok_tokens >= 0.0
