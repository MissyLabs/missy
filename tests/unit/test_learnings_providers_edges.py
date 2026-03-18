"""Edge case tests for Learnings, ProviderRegistry, and RateLimiter.

Covers scenarios not addressed by the existing test suites:

Learnings:
  - Extract learning from successful vs. failed tool use
  - No learning inference from trivial chat interaction
  - Store and retrieve learnings by task type via SQLiteMemoryStore
  - Learnings persistence (write + reload)
  - Deduplication is NOT performed by the store (documents contract)
  - Max learnings per task type via limit parameter
  - Empty task type handling in get_learnings
  - Unicode content in lessons survives round-trip
  - Approach is capped at 5 tools regardless of input length
  - Timestamp is a valid ISO-8601 string

ProviderRegistry:
  - Register and resolve a provider
  - Resolve with fallback when primary is absent
  - Resolve non-existent provider returns None
  - Switch active provider (set_default + get_default_name)
  - set_default on unavailable provider raises ValueError
  - set_default on missing provider raises ValueError
  - Provider availability check isolates unavailable providers
  - API key rotation: round-robin through api_keys list
  - API key rotation with fewer than 2 keys is a no-op
  - API key rotation on unregistered provider is a no-op
  - Fast / premium model tier selection via ModelRouter
  - ModelRouter falls back to primary model when tier model is absent
  - Empty registry: list_providers, get_available, get all work safely
  - Provider enable/disable via from_config
  - from_config auto-allows provider base_url host into network policy
  - Overwriting a registration silently replaces the old instance

RateLimiter:
  - Allow requests well under limit with no blocking
  - Block requests over limit and raise RateLimitExceeded
  - Window refill: capacity recovers after explicit bucket drain + sleep
  - Per-instance limits are independent (two limiters do not share state)
  - Concurrent rate limit checks produce consistent counts
  - Zero-token acquire does not consume the token bucket
  - on_rate_limit_response drains both buckets; retry_after=0 is instant
  - RateLimitExceeded carries the correct wait_seconds attribute
  - request_capacity / token_capacity properties reflect live bucket state
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from missy.agent.learnings import (
    TaskLearning,
    extract_learnings,
)
from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ProviderConfig,
    ShellPolicy,
)
from missy.memory.sqlite_store import SQLiteMemoryStore
from missy.providers import registry as registry_module
from missy.providers.base import BaseProvider, CompletionResponse
from missy.providers.rate_limiter import RateLimiter, RateLimitExceeded
from missy.providers.registry import ModelRouter, ProviderRegistry

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _fake_provider(name: str = "fake", available: bool = True) -> BaseProvider:
    """Build a MagicMock that satisfies the BaseProvider interface."""
    p = MagicMock(spec=BaseProvider)
    p.name = name
    p.is_available.return_value = available
    p.complete.return_value = CompletionResponse(
        content="ok",
        model="test-model",
        provider=name,
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        raw={},
    )
    return p


def _missy_config(providers: dict | None = None) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers=providers or {},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Restore the module-level registry singleton after every test."""
    original = registry_module._registry
    yield
    registry_module._registry = original


# ===========================================================================
# Learnings — pure extraction logic
# ===========================================================================


class TestExtractLearningsSuccessfulToolUse:
    """extract_learnings produces a 'success' record when the response signals success."""

    def test_shell_exec_success(self):
        tl = extract_learnings(
            tool_names_used=["shell_exec"],
            final_response="The script ran successfully.",
            prompt="Run the deploy script",
        )
        assert tl.task_type == "shell"
        assert tl.outcome == "success"
        assert "succeeded" in tl.lesson
        assert "shell_exec" in tl.approach

    def test_file_write_success(self):
        tl = extract_learnings(
            tool_names_used=["file_write"],
            final_response="File has been written and completed.",
            prompt="Save the output to disk",
        )
        assert tl.task_type == "file"
        assert tl.outcome == "success"

    def test_web_fetch_success(self):
        tl = extract_learnings(
            tool_names_used=["web_fetch"],
            final_response="Done! The page was fetched.",
            prompt="Fetch the page",
        )
        assert tl.task_type == "web"
        assert tl.outcome == "success"
        assert tl.approach == ["web_fetch"]


class TestExtractLearningsFailedToolUse:
    """extract_learnings produces a 'failure' record when the response signals failure."""

    def test_shell_exec_failed(self):
        tl = extract_learnings(
            tool_names_used=["shell_exec"],
            final_response="The command failed with exit code 127.",
            prompt="Run missing-tool",
        )
        assert tl.outcome == "failure"
        assert "failure" in tl.lesson

    def test_file_read_error(self):
        tl = extract_learnings(
            tool_names_used=["file_read"],
            final_response="Error: permission denied.",
            prompt="Read /etc/shadow",
        )
        assert tl.outcome == "failure"

    def test_web_fetch_unable(self):
        tl = extract_learnings(
            tool_names_used=["web_fetch"],
            final_response="I was unable to fetch the resource.",
            prompt="Fetch resource",
        )
        assert tl.outcome == "failure"
        assert tl.task_type == "web"


class TestNoLearningFromTrivialInteraction:
    """A chat-only (no tools) exchange with a neutral response gives 'partial' outcome
    and the task type 'chat'.  The lesson still mentions the task type."""

    def test_trivial_chat(self):
        tl = extract_learnings(
            tool_names_used=[],
            final_response="The capital of France is Paris.",
            prompt="What is the capital of France?",
        )
        assert tl.task_type == "chat"
        assert tl.outcome == "partial"
        assert "chat" in tl.lesson
        assert tl.approach == ["direct_response"]

    def test_unknown_tool_is_chat(self):
        tl = extract_learnings(
            tool_names_used=["totally_unknown_tool"],
            final_response="Here is my analysis.",
            prompt="Analyse this",
        )
        # Unknown tool names resolve to 'chat' task type
        assert tl.task_type == "chat"

    def test_lesson_always_starts_with_task_type(self):
        tl = extract_learnings([], "Something happened.", "Do a thing")
        assert tl.lesson.startswith(tl.task_type)


class TestApproachCap:
    """The approach list must never exceed 5 entries."""

    def test_exactly_five_tools(self):
        tools = ["tool_a", "tool_b", "tool_c", "tool_d", "tool_e"]
        tl = extract_learnings(tools, "Successfully finished.", "Multi-step task")
        assert tl.approach == tools

    def test_six_tools_capped_at_five(self):
        tools = [f"tool_{i}" for i in range(6)]
        tl = extract_learnings(tools, "done", "Multi-step task")
        assert len(tl.approach) == 5
        assert tl.approach == tools[:5]

    def test_twenty_tools_capped_at_five(self):
        tools = [f"t{i}" for i in range(20)]
        tl = extract_learnings(tools, "finished", "Many steps")
        assert len(tl.approach) == 5


class TestTimestampValidity:
    """Verify the auto-generated timestamp is a valid ISO-8601 string."""

    def test_timestamp_is_non_empty_string(self):
        tl = TaskLearning(task_type="shell", approach=[], outcome="success", lesson="x")
        assert isinstance(tl.timestamp, str)
        assert len(tl.timestamp) > 0

    def test_timestamp_contains_T_separator(self):
        tl = TaskLearning(task_type="web", approach=[], outcome="partial", lesson="y")
        # ISO-8601 datetime strings contain a 'T' separator
        assert "T" in tl.timestamp

    def test_explicit_timestamp_preserved(self):
        ts = "2025-06-01T12:00:00+00:00"
        tl = TaskLearning(task_type="file", approach=[], outcome="failure", lesson="z", timestamp=ts)
        assert tl.timestamp == ts


# ===========================================================================
# Learnings — SQLiteMemoryStore persistence
# ===========================================================================


class TestLearningsPersistence:
    """Save TaskLearning objects to SQLite and verify retrieval."""

    def test_save_and_retrieve_by_task_type(self, tmp_path):
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        tl = extract_learnings(["shell_exec"], "Successfully ran.", "Deploy")
        store.save_learning(tl)
        lessons = store.get_learnings(task_type="shell")
        assert len(lessons) == 1
        assert "shell" in lessons[0]

    def test_retrieve_without_filter_returns_all_types(self, tmp_path):
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        store.save_learning(extract_learnings(["shell_exec"], "done", "p1"))
        store.save_learning(extract_learnings(["web_fetch"], "done", "p2"))
        store.save_learning(extract_learnings([], "partial", "p3"))
        all_lessons = store.get_learnings()
        assert len(all_lessons) == 3

    def test_wrong_task_type_filter_returns_empty(self, tmp_path):
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        store.save_learning(extract_learnings(["shell_exec"], "done", "x"))
        lessons = store.get_learnings(task_type="web")
        assert lessons == []

    def test_limit_parameter_respected(self, tmp_path):
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        for i in range(10):
            tl = extract_learnings(["shell_exec"], "done", f"task {i}")
            store.save_learning(tl)
        lessons = store.get_learnings(task_type="shell", limit=3)
        assert len(lessons) == 3

    def test_most_recent_first_ordering(self, tmp_path):
        """get_learnings returns lessons most-recent-first."""
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        # Insert three learnings with distinct timestamps to guarantee order
        for i in range(3):
            @dataclass
            class FakeLearning:
                task_type: str = "shell"
                outcome: str = "success"
                lesson: str = f"lesson-{i}"
                approach: list = None
                timestamp: str = f"2025-01-0{i+1}T00:00:00+00:00"

                def __post_init__(self):
                    if self.approach is None:
                        self.approach = []

            store.save_learning(FakeLearning())

        lessons = store.get_learnings(task_type="shell", limit=3)
        # Most recent is lesson-2 (timestamp 2025-01-03)
        assert lessons[0] == "lesson-2"
        assert lessons[-1] == "lesson-0"

    def test_persistence_survives_store_reload(self, tmp_path):
        """Data written to the db is visible from a new SQLiteMemoryStore instance."""
        db = str(tmp_path / "mem.db")
        store1 = SQLiteMemoryStore(db_path=db)
        tl = extract_learnings(["file_write"], "completed", "write task")
        store1.save_learning(tl)

        store2 = SQLiteMemoryStore(db_path=db)
        lessons = store2.get_learnings(task_type="file")
        assert len(lessons) == 1

    def test_empty_task_type_stored_and_retrieved(self, tmp_path):
        """A learning with an empty task_type string round-trips correctly."""
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)

        @dataclass
        class EmptyTypeLearning:
            task_type: str = ""
            outcome: str = "partial"
            lesson: str = "empty-type lesson"
            approach: list = None
            timestamp: str = "2025-01-01T00:00:00+00:00"

            def __post_init__(self):
                if self.approach is None:
                    self.approach = []

        store.save_learning(EmptyTypeLearning())
        # get_learnings with no filter should include it
        all_lessons = store.get_learnings(limit=10)
        assert "empty-type lesson" in all_lessons

    def test_unicode_in_lesson_survives_round_trip(self, tmp_path):
        """Unicode characters in lesson text must be preserved without corruption."""
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        unicode_lesson = "学習: シェル → ファイル書き込み成功 🎉"
        tl = TaskLearning(
            task_type="shell+file",
            approach=["shell_exec", "file_write"],
            outcome="success",
            lesson=unicode_lesson,
        )
        store.save_learning(tl)
        lessons = store.get_learnings(task_type="shell+file")
        assert lessons[0] == unicode_lesson

    def test_duplicate_learnings_both_stored(self, tmp_path):
        """The store does NOT deduplicate; identical lessons are stored separately.
        This documents the current contract: deduplication is the caller's concern."""
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        tl = extract_learnings(["shell_exec"], "done", "task")
        store.save_learning(tl)
        store.save_learning(tl)  # same object, second insert
        lessons = store.get_learnings(task_type="shell", limit=10)
        assert len(lessons) == 2

    def test_get_learnings_returns_strings_not_rows(self, tmp_path):
        """get_learnings must return plain strings, not sqlite3.Row objects."""
        db = str(tmp_path / "mem.db")
        store = SQLiteMemoryStore(db_path=db)
        tl = extract_learnings(["web_fetch"], "done", "fetch page")
        store.save_learning(tl)
        lessons = store.get_learnings()
        assert all(isinstance(item, str) for item in lessons)


# ===========================================================================
# ProviderRegistry — register, resolve, fallback
# ===========================================================================


class TestProviderRegistryRegisterResolve:
    def test_register_and_resolve_provider(self):
        registry = ProviderRegistry()
        p = _fake_provider("openai")
        registry.register("openai", p)
        assert registry.get("openai") is p

    def test_resolve_non_existent_returns_none(self):
        registry = ProviderRegistry()
        assert registry.get("does-not-exist") is None

    def test_register_replaces_previous(self):
        registry = ProviderRegistry()
        old = _fake_provider("v1")
        new = _fake_provider("v2")
        registry.register("slot", old)
        registry.register("slot", new)
        assert registry.get("slot") is new

    def test_resolve_with_fallback_pattern(self):
        """Callers may implement fallback by trying get() then falling back to
        the first available provider.  Verify the registry supports this pattern."""
        registry = ProviderRegistry()
        fallback = _fake_provider("fallback", available=True)
        registry.register("fallback", fallback)
        # Primary is not registered
        primary = registry.get("primary")
        result = primary or registry.get_available()[0]
        assert result is fallback

    def test_empty_registry_get_available_returns_empty_list(self):
        registry = ProviderRegistry()
        assert registry.get_available() == []

    def test_empty_registry_list_providers_returns_empty_list(self):
        registry = ProviderRegistry()
        assert registry.list_providers() == []

    def test_empty_registry_get_returns_none(self):
        registry = ProviderRegistry()
        assert registry.get("anything") is None


class TestProviderRegistrySwitchDefault:
    def test_set_and_get_default(self):
        registry = ProviderRegistry()
        p = _fake_provider("anthropic", available=True)
        registry.register("anthropic", p)
        registry.set_default("anthropic")
        assert registry.get_default_name() == "anthropic"

    def test_set_default_on_missing_provider_raises(self):
        registry = ProviderRegistry()
        with pytest.raises(ValueError, match="not registered"):
            registry.set_default("ghost")

    def test_set_default_on_unavailable_provider_raises(self):
        registry = ProviderRegistry()
        p = _fake_provider("sleepy", available=False)
        registry.register("sleepy", p)
        with pytest.raises(ValueError, match="not available"):
            registry.set_default("sleepy")

    def test_set_default_when_is_available_raises_exception(self):
        """If is_available() throws, set_default must raise ValueError."""
        registry = ProviderRegistry()
        p = _fake_provider("flaky")
        p.is_available.side_effect = ConnectionError("network unreachable")
        registry.register("flaky", p)
        with pytest.raises(ValueError, match="availability check failed"):
            registry.set_default("flaky")

    def test_switching_default_provider(self):
        registry = ProviderRegistry()
        a = _fake_provider("alpha", available=True)
        b = _fake_provider("beta", available=True)
        registry.register("alpha", a)
        registry.register("beta", b)
        registry.set_default("alpha")
        assert registry.get_default_name() == "alpha"
        registry.set_default("beta")
        assert registry.get_default_name() == "beta"

    def test_default_is_none_before_set(self):
        registry = ProviderRegistry()
        assert registry.get_default_name() is None


class TestProviderAvailability:
    def test_available_providers_filtered_correctly(self):
        registry = ProviderRegistry()
        up = _fake_provider("up", available=True)
        down = _fake_provider("down", available=False)
        registry.register("up", up)
        registry.register("down", down)
        available = registry.get_available()
        assert len(available) == 1
        assert available[0] is up

    def test_exception_in_is_available_excluded_silently(self):
        registry = ProviderRegistry()
        p = _fake_provider("crashy")
        p.is_available.side_effect = RuntimeError("boom")
        registry.register("crashy", p)
        assert registry.get_available() == []

    def test_all_providers_available(self):
        registry = ProviderRegistry()
        for i in range(3):
            registry.register(f"p{i}", _fake_provider(f"p{i}", available=True))
        assert len(registry.get_available()) == 3

    def test_none_available(self):
        registry = ProviderRegistry()
        for i in range(3):
            registry.register(f"p{i}", _fake_provider(f"p{i}", available=False))
        assert registry.get_available() == []


class TestApiKeyRotation:
    """rotate_key advances through api_keys round-robin and updates provider.api_key."""

    def _make_rotatable_provider(self) -> tuple[BaseProvider, ProviderConfig]:
        """Return a provider mock that has an api_key attribute and a config with 3 keys."""
        p = MagicMock(spec=BaseProvider)
        p.name = "rotating"
        p.is_available.return_value = True
        p.api_key = "key-0"
        config = ProviderConfig(
            name="anthropic",
            model="claude-3",
            api_key="key-0",
            api_keys=["key-0", "key-1", "key-2"],
        )
        return p, config

    def test_key_rotates_to_next(self):
        registry = ProviderRegistry()
        p, cfg = self._make_rotatable_provider()
        registry.register("r", p, config=cfg)
        registry.rotate_key("r")
        assert p.api_key == "key-1"

    def test_key_rotates_round_robin(self):
        registry = ProviderRegistry()
        p, cfg = self._make_rotatable_provider()
        registry.register("r", p, config=cfg)
        registry.rotate_key("r")  # → key-1
        registry.rotate_key("r")  # → key-2
        registry.rotate_key("r")  # → key-0 (wraps)
        assert p.api_key == "key-0"

    def test_rotation_with_fewer_than_two_keys_is_noop(self):
        registry = ProviderRegistry()
        p = _fake_provider("solo")
        p.api_key = "only-key"
        cfg = ProviderConfig(name="anthropic", model="m", api_key="only-key", api_keys=["only-key"])
        registry.register("solo", p, config=cfg)
        registry.rotate_key("solo")
        assert p.api_key == "only-key"

    def test_rotation_with_zero_keys_is_noop(self):
        registry = ProviderRegistry()
        p = _fake_provider("nokeys")
        p.api_key = "from-env"
        cfg = ProviderConfig(name="anthropic", model="m", api_keys=[])
        registry.register("nokeys", p, config=cfg)
        registry.rotate_key("nokeys")
        assert p.api_key == "from-env"

    def test_rotate_key_on_unregistered_provider_is_noop(self):
        """rotate_key for an unknown name must not raise; it logs a warning."""
        registry = ProviderRegistry()
        # Should silently do nothing
        registry.rotate_key("not-here")

    def test_rotate_key_updates_private_attribute_when_no_public_api_key(self):
        """If provider only has _api_key, that attribute is updated instead."""
        registry = ProviderRegistry()
        p = MagicMock(spec=BaseProvider)
        p.name = "private"
        # Remove api_key from spec; give only _api_key
        del p.api_key
        p._api_key = "k0"
        cfg = ProviderConfig(
            name="anthropic", model="m", api_keys=["k0", "k1"]
        )
        registry.register("private", p, config=cfg)
        registry.rotate_key("private")
        assert p._api_key == "k1"


class TestFromConfig:
    def test_disabled_provider_is_skipped(self):
        config = _missy_config(
            providers={
                "offline": ProviderConfig(
                    name="ollama",
                    model="llama3",
                    enabled=False,
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("offline") is None

    def test_unknown_provider_name_is_skipped(self):
        config = _missy_config(
            providers={
                "mystery": ProviderConfig(
                    name="not-a-real-provider-xyz",
                    model="some-model",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("mystery") is None

    def test_known_provider_is_registered(self):
        config = _missy_config(
            providers={
                "local": ProviderConfig(
                    name="ollama",
                    model="llama3.2",
                )
            }
        )
        registry = ProviderRegistry.from_config(config)
        assert registry.get("local") is not None

    def test_base_url_host_auto_added_to_network_policy(self):
        config = _missy_config(
            providers={
                "custom": ProviderConfig(
                    name="ollama",
                    model="llama3.2",
                    base_url="http://my-ollama-host:11434",
                    enabled=True,
                )
            }
        )
        ProviderRegistry.from_config(config)
        # The hostname from base_url must appear in provider_allowed_hosts
        assert "my-ollama-host" in config.network.provider_allowed_hosts

    def test_base_url_host_not_duplicated(self):
        host = "ollama.local"
        config = _missy_config(
            providers={
                "a": ProviderConfig(
                    name="ollama",
                    model="m",
                    base_url=f"http://{host}:11434",
                    enabled=True,
                ),
            }
        )
        config.network.provider_allowed_hosts = [host]
        ProviderRegistry.from_config(config)
        count = config.network.provider_allowed_hosts.count(host)
        assert count == 1, f"Host duplicated: found {count} times"


# ===========================================================================
# ModelRouter — fast / premium / primary tier selection
# ===========================================================================


class TestModelRouterTiers:
    def _cfg(self, fast: str = "", premium: str = "", model: str = "primary-model") -> ProviderConfig:
        return ProviderConfig(
            name="anthropic",
            model=model,
            fast_model=fast,
            premium_model=premium,
        )

    def test_fast_tier_returns_fast_model(self):
        router = ModelRouter()
        cfg = self._cfg(fast="fast-model", premium="premium-model")
        assert router.select_model(cfg, "fast") == "fast-model"

    def test_premium_tier_returns_premium_model(self):
        router = ModelRouter()
        cfg = self._cfg(fast="fast-model", premium="premium-model")
        assert router.select_model(cfg, "premium") == "premium-model"

    def test_primary_tier_returns_default_model(self):
        router = ModelRouter()
        cfg = self._cfg(fast="fast-model", premium="premium-model", model="default-model")
        assert router.select_model(cfg, "primary") == "default-model"

    def test_fast_tier_falls_back_when_fast_model_empty(self):
        router = ModelRouter()
        cfg = self._cfg(fast="", premium="premium-model", model="default-model")
        assert router.select_model(cfg, "fast") == "default-model"

    def test_premium_tier_falls_back_when_premium_model_empty(self):
        router = ModelRouter()
        cfg = self._cfg(fast="fast-model", premium="", model="default-model")
        assert router.select_model(cfg, "premium") == "default-model"

    def test_score_complexity_short_simple_is_fast(self):
        router = ModelRouter()
        assert router.score_complexity("What is Python?", history_length=0, tool_count=0) == "fast"

    def test_score_complexity_debug_keyword_is_premium(self):
        router = ModelRouter()
        assert router.score_complexity("debug this function") == "premium"

    def test_score_complexity_long_prompt_is_premium(self):
        router = ModelRouter()
        long_prompt = "x " * 300
        assert router.score_complexity(long_prompt) == "premium"

    def test_score_complexity_many_tools_is_premium(self):
        router = ModelRouter()
        assert router.score_complexity("help me", tool_count=4) == "premium"

    def test_score_complexity_long_history_is_premium(self):
        router = ModelRouter()
        assert router.score_complexity("continue", history_length=15) == "premium"

    def test_score_complexity_moderate_is_primary(self):
        router = ModelRouter()
        # Not short+simple, not premium signals
        assert router.score_complexity("Help me write a function", history_length=3) == "primary"


# ===========================================================================
# RateLimiter — edge cases
# ===========================================================================


class TestRateLimiterUnderLimit:
    def test_single_acquire_well_under_limit(self):
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=1.0)
        rl.acquire()  # must not raise

    def test_multiple_acquires_under_limit(self):
        rl = RateLimiter(requests_per_minute=100, max_wait_seconds=1.0)
        for _ in range(10):
            rl.acquire()

    def test_zero_token_acquire_does_not_consume_token_bucket(self):
        """acquire(tokens=0) must not deduct from the token bucket."""
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=100, max_wait_seconds=0.0)
        initial_capacity = rl.token_capacity
        rl.acquire(tokens=0)
        # Capacity should be unchanged (modulo tiny refill during the call)
        assert rl.token_capacity >= initial_capacity - 1.0  # 1 token tolerance

    def test_unlimited_limiter_never_raises(self):
        rl = RateLimiter(requests_per_minute=0, tokens_per_minute=0)
        for _ in range(100):
            rl.acquire(tokens=1_000_000)


class TestRateLimiterOverLimit:
    def test_over_request_limit_raises(self):
        rl = RateLimiter(requests_per_minute=2, max_wait_seconds=0.0)
        rl.acquire()
        rl.acquire()
        with pytest.raises(RateLimitExceeded):
            rl.acquire()

    def test_over_token_limit_raises(self):
        rl = RateLimiter(requests_per_minute=1000, tokens_per_minute=100, max_wait_seconds=0.0)
        rl.acquire(tokens=100)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(tokens=1)

    def test_rate_limit_exceeded_carries_wait_seconds(self):
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.acquire()
        assert exc_info.value.wait_seconds > 0
        assert isinstance(exc_info.value.wait_seconds, float)

    def test_rate_limit_exceeded_str_contains_wait(self):
        rl = RateLimiter(requests_per_minute=1, max_wait_seconds=0.0)
        rl.acquire()
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.acquire()
        assert "wait" in str(exc_info.value).lower()


class TestRateLimiterWindowReset:
    def test_capacity_recovers_after_drain_and_sleep(self):
        """After draining the bucket, capacity recovers proportional to elapsed time."""
        rpm = 6000  # 100 req/sec — fast refill for deterministic test
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        with rl._lock:
            rl._req_tokens = 0.0
            rl._req_last_refill = time.monotonic()

        time.sleep(0.05)  # 50ms → should refill ~5 tokens at 100 req/sec
        assert rl.request_capacity > 0.0

    def test_bucket_caps_at_maximum_after_long_idle(self):
        """Bucket must never exceed rpm even after a very long idle period."""
        rpm = 10
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        with rl._lock:
            rl._req_last_refill -= 3600.0  # simulate 1 hour idle
        cap = rl.request_capacity
        assert cap <= float(rpm) + 0.01


class TestRateLimiterPerProviderIndependence:
    """Two separate RateLimiter instances must not share state."""

    def test_independent_limiters_do_not_interfere(self):
        a = RateLimiter(requests_per_minute=2, max_wait_seconds=0.0)
        b = RateLimiter(requests_per_minute=2, max_wait_seconds=0.0)
        a.acquire()
        a.acquire()
        # a is now exhausted; b should still work
        b.acquire()
        b.acquire()
        with pytest.raises(RateLimitExceeded):
            a.acquire()

    def test_token_buckets_are_independent(self):
        a = RateLimiter(requests_per_minute=1000, tokens_per_minute=100, max_wait_seconds=0.0)
        b = RateLimiter(requests_per_minute=1000, tokens_per_minute=100, max_wait_seconds=0.0)
        a.acquire(tokens=100)  # exhaust a's token bucket
        # b's token bucket is unaffected
        b.acquire(tokens=100)


class TestRateLimiterConcurrency:
    """Thread-safe acquire under concurrency must produce consistent counts."""

    def test_concurrent_acquires_produce_correct_success_count(self):
        """Exactly rpm threads should succeed; the rest should get RateLimitExceeded."""
        rpm = 20
        n_threads = 40
        rl = RateLimiter(requests_per_minute=rpm, tokens_per_minute=0, max_wait_seconds=0.0)
        successes = []
        failures = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            try:
                rl.acquire()
                with lock:
                    successes.append(1)
            except RateLimitExceeded:
                with lock:
                    failures.append(1)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(successes) + len(failures) == n_threads
        assert len(successes) <= rpm
        assert len(failures) > 0

    def test_no_unexpected_errors_under_concurrency(self):
        """Only RateLimitExceeded may escape from concurrent acquire calls."""
        rl = RateLimiter(requests_per_minute=10, max_wait_seconds=0.05)
        unexpected = []
        lock = threading.Lock()
        barrier = threading.Barrier(20)

        def worker():
            barrier.wait()
            try:
                rl.acquire()
            except RateLimitExceeded:
                pass
            except Exception as exc:
                with lock:
                    unexpected.append(exc)

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert unexpected == []

    def test_capacity_properties_are_thread_safe(self):
        """Reading capacity properties while other threads acquire must not crash."""
        rl = RateLimiter(requests_per_minute=1000, max_wait_seconds=1.0)
        stop = threading.Event()
        errors = []

        def acquirer():
            while not stop.is_set():
                try:
                    rl.acquire()
                except RateLimitExceeded:
                    pass
                except Exception as exc:
                    errors.append(exc)

        def reader():
            while not stop.is_set():
                try:
                    _ = rl.request_capacity
                    _ = rl.token_capacity
                except Exception as exc:
                    errors.append(exc)

        threads = (
            [threading.Thread(target=acquirer, daemon=True) for _ in range(3)]
            + [threading.Thread(target=reader, daemon=True) for _ in range(2)]
        )
        for t in threads:
            t.start()
        time.sleep(0.1)
        stop.set()
        for t in threads:
            t.join(timeout=3)

        assert errors == []


class TestOnRateLimitResponse:
    def test_drains_both_buckets(self):
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=10_000, max_wait_seconds=0.0)
        rl.on_rate_limit_response(retry_after=0.0)
        assert rl._req_tokens == 0.0
        assert rl._tok_tokens == 0.0

    def test_retry_after_zero_returns_immediately(self):
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=0.0)
        start = time.monotonic()
        rl.on_rate_limit_response(retry_after=0.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_after_drain_acquire_raises(self):
        """After on_rate_limit_response, any immediate acquire must raise."""
        rl = RateLimiter(requests_per_minute=60, max_wait_seconds=0.0)
        rl.on_rate_limit_response(retry_after=0.0)
        with pytest.raises(RateLimitExceeded):
            rl.acquire()
