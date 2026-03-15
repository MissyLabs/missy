"""Session 22 edge-case tests.

Covers specific boundary conditions that are unlikely to be exercised by
existing tests:

1.  Config hot-reload with a valid but changed config file.
2.  SQLite memory store with very long content (> 100 KB).
3.  Rate-limiter token refill after exactly the window duration elapses.
4.  Tool registry double-registration (same name, different instance).
5.  Circuit breaker half-open state recovery after a successful probe.
6.  Provider registry with all providers unavailable.
7.  Scheduler parse edge cases: zero-interval, invalid hour, unknown day.
8.  Context manager with zero total token budget.
9.  MCP manager shutdown when no servers are active (idempotent).
10. Cost tracker with negative token counts (defensive handling).
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# 1. Config hot-reload with valid but different config
# ===========================================================================


class TestConfigHotReloadCallback:
    """ConfigWatcher triggers the reload callback when the file mtime changes."""

    def test_reload_callback_invoked_on_mtime_change(self, tmp_path: Path) -> None:
        """Modifying the config file causes reload_fn to be called."""
        from missy.config.hotreload import ConfigWatcher

        config_file = tmp_path / "config.yaml"
        config_file.write_text("log_level: info\n")
        # ConfigWatcher._check_file_safety rejects group/world-writable files.
        config_file.chmod(0o600)

        received: list = []

        def _callback(new_cfg) -> None:
            received.append(new_cfg)

        watcher = ConfigWatcher(
            config_path=str(config_file),
            reload_fn=_callback,
            debounce_seconds=0.05,
            poll_interval=0.02,
        )

        # _do_reload does `from missy.config.settings import load_config`
        # so we patch the name in that module.
        fake_config = MagicMock()
        with patch("missy.config.settings.load_config", return_value=fake_config):
            watcher.start()
            time.sleep(0.02)
            # Writing keeps the same inode; update mtime by overwriting.
            config_file.write_text("log_level: debug\n")
            config_file.chmod(0o600)
            time.sleep(0.3)
            watcher.stop()

        assert len(received) >= 1, "Reload callback was never invoked"

    def test_reload_skipped_for_symlink(self, tmp_path: Path) -> None:
        """ConfigWatcher refuses to reload when the config path is a symlink."""
        from missy.config.hotreload import ConfigWatcher

        real_file = tmp_path / "real.yaml"
        real_file.write_text("log_level: info\n")
        link_file = tmp_path / "config.yaml"
        link_file.symlink_to(real_file)

        received: list = []

        def _callback(cfg) -> None:
            received.append(cfg)

        watcher = ConfigWatcher(
            config_path=str(link_file),
            reload_fn=_callback,
            debounce_seconds=0.0,
            poll_interval=0.02,
        )
        # _check_file_safety should reject symlinks; callback never fires.
        assert watcher._check_file_safety() is False


# ===========================================================================
# 2. Memory store with very long content
# ===========================================================================


class TestMemoryStoreVeryLongContent:
    """SQLiteMemoryStore handles content exceeding 100 KB without errors."""

    def test_add_and_retrieve_100kb_turn(self, tmp_path: Path) -> None:
        from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=str(tmp_path / "mem.db"))
        big_content = "A" * 110_000  # > 100 KB

        turn = ConversationTurn.new("sess-big", "user", big_content)
        store.add_turn(turn)

        turns = store.get_session_turns("sess-big")
        assert len(turns) == 1
        assert turns[0].content == big_content

    def test_fts_search_long_content_does_not_raise(self, tmp_path: Path) -> None:
        from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore

        store = SQLiteMemoryStore(db_path=str(tmp_path / "mem_fts.db"))
        # Insert a turn with a unique token embedded in long filler.
        big_content = "filler " * 15_000 + "UNIQUETOKEN"
        turn = ConversationTurn.new("sess-fts", "user", big_content)
        store.add_turn(turn)

        results = store.search("UNIQUETOKEN")
        assert len(results) >= 1


# ===========================================================================
# 3. Rate-limiter token refill after exact boundary
# ===========================================================================


class TestRateLimiterRefillAtBoundary:
    """Tokens are replenished correctly after exactly one window has passed."""

    def test_refill_restores_full_capacity_after_window(self) -> None:
        from missy.providers.rate_limiter import RateLimiter

        # 60 rpm means 1 token per second
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=0)

        # Drain the bucket completely.
        with limiter._lock:
            limiter._req_tokens = 0.0
            limiter._req_last_refill = time.monotonic() - 60.0  # simulate 60 s elapsed

        # After 60 s of elapsed time the bucket should refill to the rpm cap.
        with limiter._lock:
            limiter._refill()
            capacity = limiter._req_tokens

        assert capacity == pytest.approx(60.0, abs=0.01)

    def test_acquire_succeeds_after_manual_time_advance(self) -> None:
        from missy.providers.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=0)

        # Drain the bucket.
        with limiter._lock:
            limiter._req_tokens = 0.0
            # Set last_refill 30 s in the past — enough to refill ~1 token (2 rpm * 30/60).
            limiter._req_last_refill = time.monotonic() - 30.0

        # acquire() should succeed because ~1 token has accumulated.
        limiter.acquire()  # must not raise

    def test_on_rate_limit_response_drains_bucket(self) -> None:
        from missy.providers.rate_limiter import RateLimiter

        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=1_000)
        # Both buckets start full; drain them.
        limiter.on_rate_limit_response(retry_after=0.0)

        assert limiter.request_capacity == pytest.approx(0.0, abs=0.5)
        assert limiter.token_capacity == pytest.approx(0.0, abs=10.0)


# ===========================================================================
# 4. Tool registry double-register
# ===========================================================================


class TestToolRegistryDoubleRegister:
    """Registering a tool with the same name replaces the previous entry."""

    def _make_tool(self, name: str, return_value: str):
        from missy.tools.base import BaseTool, ToolPermissions, ToolResult

        # Capture return_value in a default arg to avoid closure issues.
        class _Tool(BaseTool):
            _rv = return_value

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output=self._rv)

        _Tool.name = name
        _Tool.description = f"Tool {name}"
        _Tool.permissions = ToolPermissions()
        return _Tool()

    def test_second_registration_replaces_first(self) -> None:
        from missy.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tool_v1 = self._make_tool("dupe_tool", "v1")
        tool_v2 = self._make_tool("dupe_tool", "v2")

        registry.register(tool_v1)
        registry.register(tool_v2)

        # Only one tool under that name.
        assert registry.list_tools() == ["dupe_tool"]
        # The retrieved instance is the second one.
        assert registry.get("dupe_tool") is tool_v2

    def test_double_register_then_execute_uses_latest(self) -> None:
        from missy.tools.registry import ToolRegistry

        registry = ToolRegistry()
        tool_v1 = self._make_tool("calc", "first")
        tool_v2 = self._make_tool("calc", "second")

        registry.register(tool_v1)
        registry.register(tool_v2)

        result = registry.execute("calc")
        assert result.success is True
        assert result.output == "second"


# ===========================================================================
# 5. Circuit breaker half-open state recovery
# ===========================================================================


class TestCircuitBreakerHalfOpenRecovery:
    """A successful call in half-open state transitions the circuit back to closed."""

    def _open_breaker(self, breaker) -> None:
        """Force the breaker open by exceeding the failure threshold."""
        for _ in range(breaker._threshold):
            with contextlib.suppress(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    def test_half_open_success_closes_circuit(self) -> None:
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test", threshold=2, base_timeout=0.05)
        self._open_breaker(breaker)
        assert breaker._state == CircuitState.OPEN

        # Advance time past the recovery timeout.
        with breaker._lock:
            breaker._last_failure_time = time.monotonic() - 1.0

        # The state property transitions OPEN -> HALF_OPEN when timeout elapsed.
        assert breaker.state == CircuitState.HALF_OPEN

        # A successful probe call should close the circuit.
        result = breaker.call(lambda: "ok")
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_and_doubles_timeout(self) -> None:
        from missy.agent.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker("test2", threshold=2, base_timeout=0.05, max_timeout=10.0)
        self._open_breaker(breaker)

        original_timeout = breaker._recovery_timeout

        # Advance time so the circuit goes half-open.
        with breaker._lock:
            breaker._last_failure_time = time.monotonic() - 1.0

        assert breaker.state == CircuitState.HALF_OPEN

        # Failing probe should re-open with doubled timeout.
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("probe fail")))

        assert breaker._state == CircuitState.OPEN
        assert breaker._recovery_timeout == min(original_timeout * 2, breaker._max_timeout)

    def test_open_circuit_rejects_calls(self) -> None:
        from missy.agent.circuit_breaker import CircuitBreaker
        from missy.core.exceptions import MissyError

        breaker = CircuitBreaker("test3", threshold=1, base_timeout=300.0)
        with contextlib.suppress(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        with pytest.raises(MissyError, match="OPEN"):
            breaker.call(lambda: "should not run")


# ===========================================================================
# 6. Provider registry with all providers unavailable
# ===========================================================================


class TestProviderRegistryAllUnavailable:
    """get_available() returns an empty list when every provider is down."""

    def test_empty_registry_returns_no_available(self) -> None:
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()
        assert registry.get_available() == []

    def test_all_unavailable_providers_returns_empty(self) -> None:
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()

        unavailable = MagicMock()
        unavailable.is_available.return_value = False
        unavailable.name = "mock_down"
        registry.register("mock_down", unavailable)

        available = registry.get_available()
        assert available == []

    def test_is_available_exception_treated_as_unavailable(self) -> None:
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()

        exploding = MagicMock()
        exploding.is_available.side_effect = ConnectionError("network down")
        exploding.name = "exploding"
        registry.register("exploding", exploding)

        # Should not propagate the exception.
        available = registry.get_available()
        assert available == []

    def test_mixed_available_unavailable(self) -> None:
        from missy.providers.registry import ProviderRegistry

        registry = ProviderRegistry()

        up = MagicMock()
        up.is_available.return_value = True
        up.name = "up"
        registry.register("up", up)

        down = MagicMock()
        down.is_available.return_value = False
        down.name = "down"
        registry.register("down", down)

        available = registry.get_available()
        assert available == [up]


# ===========================================================================
# 7. Scheduler parsing edge cases
# ===========================================================================


class TestSchedulerParseEdgeCases:
    """parse_schedule raises ValueError for invalid or nonsensical inputs."""

    def test_every_zero_minutes_raises(self) -> None:
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="positive"):
            parse_schedule("every 0 minutes")

    def test_daily_at_invalid_hour_raises(self) -> None:
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="Hour"):
            parse_schedule("daily at 25:00")

    def test_weekly_invalid_day_raises(self) -> None:
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="Unrecognised day"):
            parse_schedule("weekly on funday at 09:00")

    def test_empty_string_raises(self) -> None:
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError):
            parse_schedule("")

    def test_every_zero_seconds_raises(self) -> None:
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="positive"):
            parse_schedule("every 0 seconds")

    def test_daily_at_invalid_minute_raises(self) -> None:
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError, match="Minute"):
            parse_schedule("daily at 09:60")


# ===========================================================================
# 8. Context manager with zero budget
# ===========================================================================


class TestContextManagerZeroBudget:
    """A TokenBudget with total=0 produces an empty history list."""

    def test_zero_budget_returns_only_new_message(self) -> None:
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(total=0, system_reserve=0, tool_definitions_reserve=0)
        mgr = ContextManager(budget)

        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        system, messages = mgr.build_messages(
            system="sys",
            new_message="hello",
            history=history,
        )

        # With a zero budget no history turns should survive pruning.
        # The only message must be the new user message.
        assert messages[-1] == {"role": "user", "content": "hello"}
        # History is completely pruned (no turns fit in zero budget).
        assert all(m["content"] == "hello" for m in messages)

    def test_zero_budget_no_history_returns_message(self) -> None:
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(total=0, system_reserve=0, tool_definitions_reserve=0)
        mgr = ContextManager(budget)

        system, messages = mgr.build_messages(
            system="sys",
            new_message="ping",
            history=[],
        )
        assert len(messages) == 1
        assert messages[0]["content"] == "ping"

    def test_negative_budget_behaves_like_zero(self) -> None:
        """A negative total budget should not crash and yields just the new message."""
        from missy.agent.context import ContextManager, TokenBudget

        budget = TokenBudget(total=-100, system_reserve=0, tool_definitions_reserve=0)
        mgr = ContextManager(budget)

        system, messages = mgr.build_messages(
            system="sys",
            new_message="test",
            history=[{"role": "user", "content": "old msg"}],
        )
        assert messages[-1]["content"] == "test"


# ===========================================================================
# 9. MCP manager shutdown with no active servers (idempotent)
# ===========================================================================


class TestMcpManagerIdempotentShutdown:
    """McpManager.shutdown() is a no-op when no servers have been added."""

    def test_shutdown_with_no_clients_does_not_raise(self, tmp_path: Path) -> None:
        from missy.mcp.manager import McpManager

        mgr = McpManager(config_path=str(tmp_path / "mcp.json"))
        # Should complete silently with zero clients registered.
        mgr.shutdown()

    def test_double_shutdown_does_not_raise(self, tmp_path: Path) -> None:
        from missy.mcp.manager import McpManager

        mgr = McpManager(config_path=str(tmp_path / "mcp.json"))
        mgr.shutdown()
        mgr.shutdown()  # second call must also be safe

    def test_list_servers_empty_before_connect(self, tmp_path: Path) -> None:
        from missy.mcp.manager import McpManager

        mgr = McpManager(config_path=str(tmp_path / "mcp.json"))
        assert mgr.list_servers() == []

    def test_all_tools_empty_before_connect(self, tmp_path: Path) -> None:
        from missy.mcp.manager import McpManager

        mgr = McpManager(config_path=str(tmp_path / "mcp.json"))
        assert mgr.all_tools() == []

    def test_connect_all_no_config_file_is_no_op(self, tmp_path: Path) -> None:
        """connect_all() is silent when no config file exists."""
        from missy.mcp.manager import McpManager

        mgr = McpManager(config_path=str(tmp_path / "nonexistent_mcp.json"))
        mgr.connect_all()  # must not raise
        assert mgr.list_servers() == []


# ===========================================================================
# 10. Cost tracker with negative token counts
# ===========================================================================


class TestCostTrackerNegativeTokens:
    """CostTracker handles negative token counts defensively."""

    def test_negative_prompt_tokens_does_not_raise(self) -> None:
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-sonnet-4", prompt_tokens=-10, completion_tokens=5)
        # Record is returned; cost computation should not crash.
        assert rec is not None

    def test_negative_completion_tokens_does_not_raise(self) -> None:
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-sonnet-4", prompt_tokens=100, completion_tokens=-50)
        assert rec is not None

    def test_both_negative_total_tokens_does_not_raise(self) -> None:
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="gpt-4o", prompt_tokens=-5, completion_tokens=-5)
        # Totals may go negative; check_budget should still function.
        tracker.check_budget()  # unlimited budget — must not raise

    def test_negative_tokens_do_not_cause_budget_false_positive(self) -> None:
        """Negative tokens should not incorrectly trigger budget exceeded."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=1.0)
        tracker.record(model="claude-sonnet-4", prompt_tokens=-1_000_000, completion_tokens=0)
        # Total cost could be negative — budget check must not raise.
        tracker.check_budget()

    def test_zero_token_record_returns_zero_cost(self) -> None:
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-haiku-4", prompt_tokens=0, completion_tokens=0)
        assert rec.cost_usd == 0.0
        assert tracker.total_cost_usd == 0.0

    def test_unknown_model_uses_zero_pricing(self) -> None:
        """An unrecognised model name falls back to zero-cost pricing."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(
            model="totally-unknown-model-xyz", prompt_tokens=1000, completion_tokens=500
        )
        assert rec.cost_usd == 0.0
