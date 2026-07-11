"""SR-4.8: live integration tests for AgentRuntime provider rotation/fallback.

Prior to this fix, ProviderRegistry.rotate_key() and ModelRouter had zero
production call sites (extensively unit-tested in isolation only), and
AgentRuntime._get_provider() only performed a static, start-of-run
availability check -- it never retried a mid-run provider failure, never
rotated API keys, never verified model/tool compatibility of a fallback
candidate, and never emitted an audit event for the transition.

These tests exercise AgentRuntime._call_provider_with_fallback() and its
wiring into _single_turn()/_tool_loop() against real BaseProvider
subclasses (not mocks) registered in a real ProviderRegistry, so the
circuit-breaker/rotation/fallback/audit machinery is verified end-to-end
exactly as it runs in production.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from missy.agent.circuit_breaker import CircuitBreaker, CircuitState
from missy.agent.runtime import AgentConfig, AgentRuntime
from missy.agent.cost_tracker import BudgetExceededError, CostTracker
from missy.config.settings import ProviderConfig
from missy.core.events import AuditEvent, EventBus
from missy.core.exceptions import ProviderError
from missy.observability.audit_logger import AuditLogger
from missy.providers.base import BaseProvider, CompletionResponse
from missy.providers.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Real (non-mock) provider fixtures
# ---------------------------------------------------------------------------


class _FailNTimesProvider(BaseProvider):
    """Fails with a chosen error message for the first N calls, then succeeds.

    Tracks which api_key was active (via the real ``_api_key`` attribute
    convention that missy.providers.registry.rotate_key() and every
    concrete provider actually use) on every call attempt.
    """

    def __init__(self, name: str, config: ProviderConfig, fail_times: int, error_message: str):
        self.name = name
        self._config = config
        self._api_key = config.api_key
        self._fail_times = fail_times
        self._error_message = error_message
        self.calls = 0
        self.keys_seen: list[str | None] = []
        self.received_messages: list = []
        self.received_model: str | None = None

    def is_available(self) -> bool:
        return True

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.keys_seen.append(self._api_key)
        self.received_messages = messages
        self.received_model = kwargs.get("model")
        if self.calls <= self._fail_times:
            raise ProviderError(self._error_message)
        return CompletionResponse(
            content=f"{self.name} reply",
            model=kwargs.get("model") or "default-model",
            provider=self.name,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw={},
        )


class _AlwaysFailProvider(BaseProvider):
    def __init__(self, name: str, config: ProviderConfig, error_message: str):
        self.name = name
        self._config = config
        self._api_key = config.api_key
        self.calls = 0
        self._error_message = error_message

    def is_available(self) -> bool:
        return True

    def complete(self, messages, **kwargs):
        self.calls += 1
        raise ProviderError(self._error_message)


class _HealthyProvider(BaseProvider):
    """A plain healthy provider that does NOT override complete_with_tools
    (inherits BaseProvider's default degrade-to-complete() -- i.e. not
    tool-capable, used for tool-compatibility fallback ordering tests)."""

    def __init__(self, name: str, config: ProviderConfig, accepts_message_dicts: bool = False):
        self.name = name
        self._config = config
        self._api_key = config.api_key
        self.accepts_message_dicts = accepts_message_dicts
        self.calls = 0
        self.received_messages: list = []
        self.received_model: str | None = None

    def is_available(self) -> bool:
        return True

    def complete(self, messages, **kwargs):
        self.calls += 1
        self.received_messages = messages
        self.received_model = kwargs.get("model")
        return CompletionResponse(
            content=f"{self.name} reply",
            model=kwargs.get("model") or "default-model",
            provider=self.name,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw={},
        )


class _HealthyToolCapableProvider(_HealthyProvider):
    """Explicitly overrides complete_with_tools -- tool-capable."""

    def complete_with_tools(self, messages, tools, system=""):
        self.calls += 1
        self.received_messages = messages
        return CompletionResponse(
            content=f"{self.name} tool reply",
            model="default-model",
            provider=self.name,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            raw={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_runtime(provider_name: str = "primary") -> AgentRuntime:
    """Construct an AgentRuntime with every unrelated subsystem disabled,
    so _call_provider_with_fallback's own logic is exercised in isolation."""
    rt = AgentRuntime.__new__(AgentRuntime)
    rt.config = AgentConfig(provider=provider_name, max_iterations=1)
    # _make_circuit_breaker is an instance method (not a staticmethod) so
    # it can look up this provider's own registered ProviderConfig
    # tunables (SR-4.8 residual) -- must be called on the instance.
    rt._circuit_breaker = rt._make_circuit_breaker(provider_name)
    rt._fallback_breakers = {}
    rt._rate_limiter = None
    rt._drift_detector = None
    rt._cost_tracking_enabled = False
    rt._cost_trackers = {}
    rt._identity = None
    rt._message_bus = None
    from missy.core.session import SessionManager

    rt._session_mgr = SessionManager()
    return rt


def _install_registry(*pairs: tuple[str, BaseProvider, ProviderConfig]) -> ProviderRegistry:
    registry = ProviderRegistry()
    for key, provider, config in pairs:
        registry.register(key, provider, config=config)
    return registry


def _capture_events() -> tuple[list[AuditEvent], callable]:
    """Wrap the global event_bus.publish and return (captured_list, uninstall)."""
    from missy.core.events import event_bus

    captured: list[AuditEvent] = []
    original = event_bus.publish

    def _wrapped(evt):
        original(evt)
        captured.append(evt)

    event_bus.publish = _wrapped

    def _uninstall():
        event_bus.publish = original

    return captured, _uninstall


# ---------------------------------------------------------------------------
# Key rotation on auth failure
# ---------------------------------------------------------------------------


class TestKeyRotationOnAuthFailure:
    def test_rotates_key_and_retries_same_provider_on_auth_failure(self):
        cfg = ProviderConfig(
            name="flaky", model="m", api_key="key-0", api_keys=["key-0", "key-1"]
        )
        provider = _FailNTimesProvider(
            "flaky", cfg, fail_times=1, error_message="flaky authentication failed: bad key"
        )
        registry = _install_registry(("flaky", provider, cfg))

        rt = _bare_runtime("flaky")
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            result = rt._single_turn(
                provider=provider,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        assert result.content == "flaky reply"
        # First attempt used key-0 (the failing one), the retry used key-1
        # after rotate_key() actually flipped provider._api_key -- proving
        # rotation is live-wired, not just called and ignored.
        assert provider.keys_seen == ["key-0", "key-1"]
        assert provider.calls == 2

    def test_does_not_rotate_when_only_one_key_configured(self):
        cfg = ProviderConfig(name="flaky", model="m", api_key="only-key")
        provider = _AlwaysFailProvider("flaky", cfg, "flaky authentication failed: bad key")
        healthy_cfg = ProviderConfig(name="healthy", model="m", api_key="hk")
        healthy = _HealthyProvider("healthy", healthy_cfg)
        registry = _install_registry(
            ("flaky", provider, cfg), ("healthy", healthy, healthy_cfg)
        )

        rt = _bare_runtime("flaky")
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            result = rt._single_turn(
                provider=provider,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        # Only one key -- rotate_key() is a documented no-op -- so this
        # goes straight to cross-provider fallback instead.
        assert provider.calls == 1
        assert result.content == "healthy reply"

    def test_does_not_rotate_on_rate_limit_failure(self):
        """Rotating credentials cannot fix a rate limit; must skip straight
        to cross-provider fallback even with multiple keys configured."""
        cfg = ProviderConfig(
            name="limited", model="m", api_key="key-0", api_keys=["key-0", "key-1"]
        )
        provider = _AlwaysFailProvider("limited", cfg, "limited rate limited: 429")
        healthy_cfg = ProviderConfig(name="healthy", model="m", api_key="hk")
        healthy = _HealthyProvider("healthy", healthy_cfg)
        registry = _install_registry(
            ("limited", provider, cfg), ("healthy", healthy, healthy_cfg)
        )

        rt = _bare_runtime("limited")
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            result = rt._single_turn(
                provider=provider,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        assert provider.calls == 1  # never retried on a rotated key
        assert provider._api_key == "key-0"  # unchanged
        assert result.content == "healthy reply"


# ---------------------------------------------------------------------------
# Cross-provider fallback: model isolation + transcript integrity
# ---------------------------------------------------------------------------


class TestCrossProviderFallbackTranscriptAndModel:
    def test_fallback_provider_does_not_receive_primarys_model_string(self):
        primary_cfg = ProviderConfig(name="anthropic", model="claude-primary-model", api_key="k1")
        primary = _AlwaysFailProvider("anthropic", primary_cfg, "anthropic authentication failed")
        fallback_cfg = ProviderConfig(name="openai", model="gpt-fallback-model", api_key="k2")
        fallback = _HealthyProvider("openai", fallback_cfg)
        registry = _install_registry(
            ("anthropic", primary, primary_cfg), ("openai", fallback, fallback_cfg)
        )

        rt = _bare_runtime("anthropic")
        rt.config.model = "claude-primary-model"
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            result = rt._single_turn(
                provider=primary,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        assert result.provider == "openai"
        # SR-4.8's core model-compatibility gap: forwarding the primary's
        # model id to an unrelated provider would silently request a
        # nonexistent model on the fallback's API.
        assert fallback.received_model is None

    def test_fallback_message_format_matches_target_providers_convention(self):
        """Transcript integrity: each candidate gets messages rebuilt in
        *its own* accepts_message_dicts format, not reused verbatim from
        whatever was built for the original provider."""
        primary_cfg = ProviderConfig(name="anthropic", model="m", api_key="k1")
        primary = _AlwaysFailProvider("anthropic", primary_cfg, "anthropic authentication failed")
        fallback_cfg = ProviderConfig(name="openai", model="m", api_key="k2")
        fallback = _HealthyToolCapableProvider(
            "openai", fallback_cfg, accepts_message_dicts=True
        )
        registry = _install_registry(
            ("anthropic", primary, primary_cfg), ("openai", fallback, fallback_cfg)
        )

        rt = _bare_runtime("anthropic")
        loop_messages = [{"role": "user", "content": "hello there"}]

        def _make_call(target):
            if getattr(target, "accepts_message_dicts", False) is True:
                msgs = rt._dicts_to_native_messages("sys", loop_messages)
            else:
                msgs = rt._dicts_to_messages("sys", loop_messages)

            def _call():
                return target.complete_with_tools(msgs, [], "sys")

            return _call

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            response, used = rt._call_provider_with_fallback(
                primary, _make_call, session_id="s1", task_id="t1", requires_tools=True
            )

        assert used.name == "openai"
        # openai's accepts_message_dicts=True path returns list[dict], not
        # list[Message] -- confirms per-provider reformatting actually ran.
        assert isinstance(fallback.received_messages, list)
        assert all(isinstance(m, dict) for m in fallback.received_messages)
        assert fallback.received_messages[-1]["content"] == "hello there"


# ---------------------------------------------------------------------------
# Tool/model compatibility ordering
# ---------------------------------------------------------------------------


class TestToolCompatibilityOrdering:
    def test_prefers_tool_capable_candidate_when_tools_required(self):
        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider("primary", primary_cfg, "primary authentication failed")
        plain_cfg = ProviderConfig(name="plain", model="m", api_key="k2")
        plain = _HealthyProvider("plain", plain_cfg)
        tool_cfg = ProviderConfig(name="tool_capable", model="m", api_key="k3")
        tool_capable = _HealthyToolCapableProvider("tool_capable", tool_cfg)
        # Registration order deliberately puts the non-tool-capable
        # candidate first to prove selection isn't just "first available".
        registry = _install_registry(
            ("primary", primary, primary_cfg),
            ("plain", plain, plain_cfg),
            ("tool_capable", tool_capable, tool_cfg),
        )

        rt = _bare_runtime("primary")

        def _make_call(target):
            def _call():
                return target.complete_with_tools([], [], "sys")

            return _call

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            response, used = rt._call_provider_with_fallback(
                primary, _make_call, session_id="s1", task_id="t1", requires_tools=True
            )

        assert used.name == "tool_capable"
        assert plain.calls == 0

    def test_flags_degraded_audit_event_when_no_tool_capable_candidate_exists(self):
        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider("primary", primary_cfg, "primary authentication failed")
        plain_cfg = ProviderConfig(name="plain", model="m", api_key="k2")
        plain = _HealthyProvider("plain", plain_cfg)
        registry = _install_registry(
            ("primary", primary, primary_cfg), ("plain", plain, plain_cfg)
        )

        rt = _bare_runtime("primary")
        captured, uninstall = _capture_events()
        try:
            with patch("missy.agent.runtime.get_registry", return_value=registry):
                rt._single_turn(
                    provider=primary,
                    system_prompt="sys",
                    messages=[{"role": "user", "content": "hi"}],
                    session_id="s1",
                    task_id="t1",
                )
        finally:
            uninstall()

        fallback_events = [e for e in captured if e.event_type == "agent.provider.fallback"]
        assert len(fallback_events) == 1
        # requires_tools=False for _single_turn -- degraded flag should be
        # False since tool capability isn't relevant to a plain completion.
        assert fallback_events[0].detail["tool_compatibility_degraded"] is False


# ---------------------------------------------------------------------------
# Budget accounting across the transition
# ---------------------------------------------------------------------------


class TestBudgetAccountingAcrossFallback:
    def test_exhausted_budget_blocks_fallback_attempt_entirely(self):
        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider("primary", primary_cfg, "primary authentication failed")
        healthy_cfg = ProviderConfig(name="healthy", model="m", api_key="k2")
        healthy = _HealthyProvider("healthy", healthy_cfg)
        registry = _install_registry(
            ("primary", primary, primary_cfg), ("healthy", healthy, healthy_cfg)
        )

        rt = _bare_runtime("primary")
        rt._cost_tracking_enabled = True
        rt.config.max_spend_usd = 0.01
        tracker = CostTracker(max_spend_usd=0.01)
        tracker._total_cost = 1.0  # already over budget from prior calls
        rt._cost_trackers = {"s1": tracker}
        import threading

        rt._cost_trackers_lock = threading.Lock()

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            pytest.raises(BudgetExceededError),
        ):
            rt._single_turn(
                provider=primary,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        # Budget check happens before the fallback candidate is ever tried.
        assert healthy.calls == 0


# ---------------------------------------------------------------------------
# Cooldown / circuit-breaker eligibility
# ---------------------------------------------------------------------------


class TestCooldownEligibility:
    def test_provider_with_open_breaker_is_excluded_from_fallback_candidates(self):
        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider("primary", primary_cfg, "primary authentication failed")
        broken_cfg = ProviderConfig(name="broken", model="m", api_key="k2")
        broken = _AlwaysFailProvider("broken", broken_cfg, "broken unknown error")
        healthy_cfg = ProviderConfig(name="healthy", model="m", api_key="k3")
        healthy = _HealthyProvider("healthy", healthy_cfg)
        registry = _install_registry(
            ("primary", primary, primary_cfg),
            ("broken", broken, broken_cfg),
            ("healthy", healthy, healthy_cfg),
        )

        rt = _bare_runtime("primary")
        # Pre-open broken's breaker via repeated failures, exactly as
        # production would after enough real calls fail.
        broken_breaker = CircuitBreaker("broken", threshold=1, base_timeout=3600.0)
        with pytest.raises(Exception):
            broken_breaker.call(lambda: (_ for _ in ()).throw(ProviderError("boom")))
        assert broken_breaker.state == CircuitState.OPEN
        rt._fallback_breakers["broken"] = broken_breaker

        with patch("missy.agent.runtime.get_registry", return_value=registry):
            result = rt._single_turn(
                provider=primary,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        assert result.provider == "healthy"
        assert broken.calls == 0  # never attempted -- breaker was OPEN


# ---------------------------------------------------------------------------
# All candidates exhausted -> fail closed
# ---------------------------------------------------------------------------


class TestAllCandidatesExhausted:
    def test_reraises_when_every_provider_fails(self):
        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider("primary", primary_cfg, "primary authentication failed")
        also_broken_cfg = ProviderConfig(name="also_broken", model="m", api_key="k2")
        also_broken = _AlwaysFailProvider("also_broken", also_broken_cfg, "also broken: 500")
        registry = _install_registry(
            ("primary", primary, primary_cfg), ("also_broken", also_broken, also_broken_cfg)
        )

        rt = _bare_runtime("primary")
        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            pytest.raises(ProviderError),
        ):
            rt._single_turn(
                provider=primary,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        assert primary.calls == 1
        assert also_broken.calls == 1


# ---------------------------------------------------------------------------
# Redacted audit events end-to-end (through the real AuditLogger sink)
# ---------------------------------------------------------------------------


class TestRedactedAuditEventsEndToEnd:
    def test_secret_shaped_error_text_is_redacted_before_reaching_disk(self, tmp_path):
        leaking_secret = "sk-ant-api03-FAKESECRETFAKESECRETFAKESECRETFAKESECRETFAKE1234"
        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider(
            "primary", primary_cfg, f"primary authentication failed: key={leaking_secret}"
        )
        healthy_cfg = ProviderConfig(name="healthy", model="m", api_key="k2")
        healthy = _HealthyProvider("healthy", healthy_cfg)
        registry = _install_registry(
            ("primary", primary, primary_cfg), ("healthy", healthy, healthy_cfg)
        )

        log_path = tmp_path / "audit.jsonl"
        scoped_bus = EventBus()
        AuditLogger(log_path=str(log_path), bus=scoped_bus)
        rt = _bare_runtime("primary")

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch("missy.agent.runtime.event_bus", scoped_bus),
        ):
            rt._single_turn(
                provider=primary,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )

        raw = log_path.read_text()
        assert leaking_secret not in raw
        assert "agent.provider.call_failed" in raw
        assert "agent.provider.fallback" in raw


# ---------------------------------------------------------------------------
# Policy revalidation unaffected by a mid-loop provider swap (SR-2.3)
# ---------------------------------------------------------------------------


class TestToolLoopFallbackPreservesToolPolicy:
    def test_tool_dispatch_after_fallback_still_enforces_allowed_tool_names(self):
        """SR-2.3's per-turn tool allow-set enforcement is independent of
        which provider proposed the call -- verify it still applies to a
        tool call returned by a *fallback* provider mid-loop, not just the
        originally configured one."""
        from missy.providers.base import ToolCall

        primary_cfg = ProviderConfig(name="primary", model="m", api_key="k1")
        primary = _AlwaysFailProvider("primary", primary_cfg, "primary authentication failed")

        class _MaliciousToolCallProvider(_HealthyToolCapableProvider):
            def complete_with_tools(self, messages, tools, system=""):
                self.calls += 1
                return CompletionResponse(
                    content="",
                    model="m",
                    provider=self.name,
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    raw={},
                    tool_calls=[
                        ToolCall(id="1", name="not_in_allowed_set", arguments={}),
                    ],
                    finish_reason="tool_calls",
                )

        fallback_cfg = ProviderConfig(name="fallback", model="m", api_key="k2")
        fallback = _MaliciousToolCallProvider("fallback", fallback_cfg)
        registry = _install_registry(
            ("primary", primary, primary_cfg), ("fallback", fallback, fallback_cfg)
        )

        rt = _bare_runtime("primary")
        rt._memory_store = None
        rt._context_manager = None
        rt._sanitizer = None
        rt._interactive_approval = None

        called_tool_names: list[str] = []

        def _fake_execute_tool(tc, **kwargs):
            called_tool_names.append(tc.name)
            from missy.providers.base import ToolResult

            allowed = kwargs.get("allowed_tool_names", set())
            if tc.name not in allowed:
                return ToolResult(
                    tool_call_id=tc.id, name=tc.name, content="denied: not allowed", is_error=True
                )
            return ToolResult(tool_call_id=tc.id, name=tc.name, content="ok")

        with (
            patch("missy.agent.runtime.get_registry", return_value=registry),
            patch.object(rt, "_execute_tool", side_effect=_fake_execute_tool),
        ):
            text, tool_names_used = rt._tool_loop(
                provider=primary,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                tools=[],  # allowed_tool_names ends up empty
                session_id="s1",
                task_id="t1",
                user_input="hi",
            )

        # The fallback provider's tool call was dispatched (proving the
        # loop kept running against the fallback), but the dispatch layer
        # still denied it since it wasn't in this turn's allowed set --
        # unaffected by which provider proposed it.
        assert called_tool_names == ["not_in_allowed_set"]
