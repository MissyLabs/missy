"""Hardening tests.


Tests for:
- Piper TTS subprocess timeout enforcement
- Agent runtime tool loop iteration limit and fallback
- Agent runtime tool execution retry exhaustion
- Discord gateway opcode handling edge cases
- Discord REST send_message retry logic
- Network policy edge cases
- MCP manager lifecycle edge cases
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from missy.core.exceptions import PolicyViolationError

# ── Piper TTS timeout ──────────────────────────────────────────────


class TestPiperTTSTimeout:
    """Verify the piper subprocess communicate() call has a timeout."""

    @pytest.mark.asyncio
    async def test_communicate_timeout_kills_process(self) -> None:
        """If Piper hangs, the process should be killed after 60s timeout."""
        from missy.channels.voice.tts.piper import PiperTTS

        tts = PiperTTS.__new__(PiperTTS)
        tts._loaded = True
        tts._piper_bin = "/usr/bin/piper"
        tts._model_file = "/tmp/model.onnx"
        tts._voice = "en_US-lessac-medium"
        tts._sample_rate = 22050
        tts._channels = 1

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        async def mock_wait_for(coro, *, timeout):
            if hasattr(coro, "close"):
                coro.close()
            raise TimeoutError()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=mock_wait_for),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            await tts.synthesize("hello world")

        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_success_returns_audio(self) -> None:
        """Verify synthesize works when piper returns normally."""
        from missy.channels.voice.tts.piper import PiperTTS

        tts = PiperTTS.__new__(PiperTTS)
        tts._loaded = True
        tts._piper_bin = "/usr/bin/piper"
        tts._model_file = "/tmp/model.onnx"
        tts._voice = "en_US-lessac-medium"
        tts._sample_rate = 22050
        tts._channels = 1

        pcm_data = b"\x00\x01" * 100  # Some PCM data

        mock_proc = AsyncMock()
        mock_proc.returncode = 0

        async def mock_wait_for(coro, *, timeout):
            assert timeout == 60.0
            return (pcm_data, b"")

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", side_effect=mock_wait_for),
        ):
            result = await tts.synthesize("test")

        assert result.data  # Should have WAV data
        assert result.format == "wav"

    @pytest.mark.asyncio
    async def test_synthesize_not_loaded_raises(self) -> None:
        """Synthesize before load() should raise RuntimeError."""
        from missy.channels.voice.tts.piper import PiperTTS

        tts = PiperTTS.__new__(PiperTTS)
        tts._loaded = False
        tts._piper_bin = None
        tts._model_file = None

        with pytest.raises(RuntimeError, match="load"):
            await tts.synthesize("hello")


# ── Agent runtime tool loop edge cases ─────────────────────────────


class TestAgentToolLoopEdgeCases:
    """Test edge cases in the agent runtime's tool loop."""

    def _make_runtime(self, max_iterations=3):
        from missy.agent.runtime import AgentConfig, AgentRuntime

        config = AgentConfig(
            provider="anthropic",
            max_iterations=max_iterations,
        )
        rt = AgentRuntime.__new__(AgentRuntime)
        rt.config = config
        rt._session_mgr = MagicMock()
        rt._circuit_breaker = MagicMock()
        rt._rate_limiter = None
        rt._context_manager = None
        rt._memory_store = None
        rt._cost_tracker = None
        rt._sanitizer = None
        rt._pending_recovery = []
        rt._progress = MagicMock()
        rt._interactive_approval = None
        rt._drift_detector = None
        rt._identity = None
        rt._trust_scorer = MagicMock()
        rt._attention = None
        rt._persona_manager = None
        rt._behavior = None
        rt._response_shaper = None
        rt._message_bus = None
        return rt

    def test_iteration_limit_fallback_success(self) -> None:
        """When iteration limit is hit, fallback single_turn should work."""
        from missy.providers.base import CompletionResponse, ToolCall, ToolResult

        rt = self._make_runtime(max_iterations=1)

        # Provider always requests tool calls (never stops)
        response_with_tools = CompletionResponse(
            content="thinking...",
            model="test-model",
            provider="anthropic",
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id="tc1", name="calculator", arguments={"expression": "1+1"})],
            usage={"input_tokens": 10, "output_tokens": 10},
            raw={},
        )
        rt._circuit_breaker.call = MagicMock(return_value=response_with_tools)

        rt._execute_tool = MagicMock(
            return_value=ToolResult(
                tool_call_id="tc1", name="calculator", content="2", is_error=False
            )
        )
        rt._record_cost = MagicMock()
        rt._check_budget = MagicMock()
        rt._acquire_rate_limit = MagicMock()
        rt._emit_event = MagicMock()

        # Fallback single_turn returns a final response
        fallback_resp = CompletionResponse(
            content="The answer is 2.",
            model="test-model",
            provider="anthropic",
            finish_reason="stop",
            usage={"input_tokens": 10, "output_tokens": 5},
            raw={},
        )
        rt._single_turn = MagicMock(return_value=fallback_resp)

        mock_provider = MagicMock()
        mock_provider.name = "anthropic"
        result, tools = rt._tool_loop(
            provider=mock_provider,
            system_prompt="test",
            messages=[{"role": "user", "content": "what is 1+1"}],
            tools=[MagicMock()],
            session_id="s1",
            task_id="t1",
            user_input="what is 1+1",
        )
        assert result == "The answer is 2."
        rt._single_turn.assert_called_once()

    def test_iteration_limit_fallback_failure(self) -> None:
        """When iteration limit is hit and fallback fails, return error string."""
        from missy.providers.base import CompletionResponse, ToolCall, ToolResult

        rt = self._make_runtime(max_iterations=1)

        response_with_tools = CompletionResponse(
            content="thinking...",
            model="test-model",
            provider="anthropic",
            finish_reason="tool_calls",
            tool_calls=[ToolCall(id="tc1", name="calculator", arguments={"expression": "1+1"})],
            usage={"input_tokens": 10, "output_tokens": 10},
            raw={},
        )
        rt._circuit_breaker.call = MagicMock(return_value=response_with_tools)
        rt._execute_tool = MagicMock(
            return_value=ToolResult(
                tool_call_id="tc1", name="calculator", content="2", is_error=False
            )
        )
        rt._record_cost = MagicMock()
        rt._check_budget = MagicMock()
        rt._acquire_rate_limit = MagicMock()
        rt._emit_event = MagicMock()

        # Fallback single_turn raises
        rt._single_turn = MagicMock(side_effect=Exception("provider down"))

        mock_provider = MagicMock()
        mock_provider.name = "anthropic"
        result, tools = rt._tool_loop(
            provider=mock_provider,
            system_prompt="test",
            messages=[{"role": "user", "content": "hello"}],
            tools=[MagicMock()],
            session_id="s1",
            task_id="t1",
            user_input="hello",
        )
        assert "iteration limit" in result.lower()

    def test_provider_without_complete_with_tools(self) -> None:
        """Provider lacking complete_with_tools falls back to single_turn."""
        from missy.providers.base import CompletionResponse

        rt = self._make_runtime(max_iterations=3)

        rt._record_cost = MagicMock()
        rt._check_budget = MagicMock()
        rt._acquire_rate_limit = MagicMock()
        rt._emit_event = MagicMock()

        fallback_resp = CompletionResponse(
            content="fallback response",
            model="test-model",
            provider="anthropic",
            finish_reason="stop",
            usage={"input_tokens": 5, "output_tokens": 5},
            raw={},
        )
        rt._single_turn = MagicMock(return_value=fallback_resp)

        # Create a provider that does NOT have complete_with_tools
        mock_provider = MagicMock(spec=["name", "complete"])
        mock_provider.name = "anthropic"
        result, tools = rt._tool_loop(
            provider=mock_provider,
            system_prompt="test",
            messages=[{"role": "user", "content": "hello"}],
            tools=[MagicMock()],
            session_id="s1",
            task_id="t1",
            user_input="hello",
        )
        assert result == "fallback response"
        rt._single_turn.assert_called_once()

    def test_tool_execution_transient_retry_exhaustion(self) -> None:
        """Transient tool errors should be retried then return error."""
        from missy.providers.base import ToolCall

        rt = self._make_runtime()
        rt._emit_event = MagicMock()

        mock_registry = MagicMock()
        mock_registry.execute.side_effect = TimeoutError("connection timeout")

        tc = ToolCall(id="tc1", name="shell_exec", arguments={"command": "ls"})

        with (
            patch("missy.agent.runtime.get_tool_registry", return_value=mock_registry),
            patch("time.sleep"),
        ):
            result = rt._execute_tool(tc, session_id="s1", task_id="t1")

        assert result.is_error
        # Should have been called 3 times (initial + 2 retries)
        assert mock_registry.execute.call_count == 3

    def test_tool_execution_key_error(self) -> None:
        """KeyError (tool not found) should not retry."""
        from missy.providers.base import ToolCall

        rt = self._make_runtime()
        rt._emit_event = MagicMock()

        mock_registry = MagicMock()
        mock_registry.execute.side_effect = KeyError("unknown_tool")

        tc = ToolCall(id="tc1", name="unknown_tool", arguments={})

        with patch("missy.agent.runtime.get_tool_registry", return_value=mock_registry):
            result = rt._execute_tool(tc, session_id="s1", task_id="t1")

        assert result.is_error
        assert "not found" in result.content.lower()
        assert mock_registry.execute.call_count == 1  # No retry

    def test_tool_execution_unexpected_error(self) -> None:
        """Unexpected errors should return internal error without retry."""
        from missy.providers.base import ToolCall

        rt = self._make_runtime()
        rt._emit_event = MagicMock()

        mock_registry = MagicMock()
        mock_registry.execute.side_effect = ZeroDivisionError("oops")

        tc = ToolCall(id="tc1", name="calculator", arguments={"expr": "1/0"})

        with patch("missy.agent.runtime.get_tool_registry", return_value=mock_registry):
            result = rt._execute_tool(tc, session_id="s1", task_id="t1")

        assert result.is_error
        assert "internal error" in result.content.lower()
        assert mock_registry.execute.call_count == 1


# ── Discord gateway opcode edge cases ──────────────────────────────


class TestDiscordGatewayOpcodes:
    """Test Discord gateway payload handling for various opcodes."""

    def _make_gateway(self):
        from missy.channels.discord.gateway import DiscordGatewayClient

        gw = DiscordGatewayClient.__new__(DiscordGatewayClient)
        gw._token = "Bot test-token"
        gw._on_message = AsyncMock()
        gw._gateway_url = "wss://gateway.discord.gg/?v=10&encoding=json"
        gw._session_id_audit = "discord"
        gw._task_id_audit = "gateway"
        gw._ws = AsyncMock()
        gw._running = True
        gw._sequence = None
        gw._discord_session_id = None
        gw._resume_gateway_url = None
        gw._bot_user_id = None
        gw._heartbeat_task = None
        return gw

    @pytest.mark.asyncio
    async def test_heartbeat_request_from_server(self) -> None:
        """OP_HEARTBEAT (1) should trigger an immediate heartbeat send."""
        gw = self._make_gateway()
        gw._send_heartbeat = AsyncMock()

        await gw._handle_payload({"op": 1, "d": None, "s": None, "t": None})
        gw._send_heartbeat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_heartbeat_ack(self) -> None:
        """OP_HEARTBEAT_ACK (11) should be handled silently."""
        gw = self._make_gateway()
        await gw._handle_payload({"op": 11, "d": None, "s": None, "t": None})

    @pytest.mark.asyncio
    async def test_reconnect_opcode(self) -> None:
        """OP_RECONNECT (7) should close the websocket."""
        gw = self._make_gateway()
        await gw._handle_payload({"op": 7, "d": None, "s": None, "t": None})
        gw._ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_session_not_resumable(self) -> None:
        """OP_INVALID_SESSION with d=False should clear session state."""
        gw = self._make_gateway()
        gw._discord_session_id = "old-session"
        gw._resume_gateway_url = "wss://resume.example"
        gw._sequence = 42

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": 9, "d": False, "s": None, "t": None})

        assert gw._discord_session_id is None
        assert gw._resume_gateway_url is None
        assert gw._sequence is None
        gw._ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invalid_session_resumable(self) -> None:
        """OP_INVALID_SESSION with d=True should keep session state."""
        gw = self._make_gateway()
        gw._discord_session_id = "keep-session"
        gw._resume_gateway_url = "wss://resume.example"
        gw._sequence = 42

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gw._handle_payload({"op": 9, "d": True, "s": None, "t": None})

        assert gw._discord_session_id == "keep-session"
        assert gw._sequence == 42

    @pytest.mark.asyncio
    async def test_unhandled_opcode(self) -> None:
        """Unknown opcodes should be logged but not crash."""
        gw = self._make_gateway()
        await gw._handle_payload({"op": 99, "d": None, "s": None, "t": None})

    @pytest.mark.asyncio
    async def test_sequence_tracking(self) -> None:
        """Sequence numbers from payloads should be tracked."""
        gw = self._make_gateway()
        gw._send_heartbeat = AsyncMock()

        await gw._handle_payload({"op": 1, "d": None, "s": 42, "t": None})
        assert gw._sequence == 42

    @pytest.mark.asyncio
    async def test_ready_event_populates_state(self) -> None:
        """READY dispatch should populate session state."""
        gw = self._make_gateway()

        ready_data = {
            "session_id": "new-session-id",
            "resume_gateway_url": "wss://resume.discord.gg",
            "user": {"id": "12345", "username": "MissyBot", "discriminator": "0001"},
        }
        await gw._handle_payload({"op": 0, "d": ready_data, "s": 1, "t": "READY"})

        assert gw._discord_session_id == "new-session-id"
        assert gw._resume_gateway_url == "wss://resume.discord.gg"
        assert gw._bot_user_id == "12345"

    @pytest.mark.asyncio
    async def test_resumed_event(self) -> None:
        """RESUMED dispatch should not crash."""
        gw = self._make_gateway()
        await gw._handle_payload({"op": 0, "d": {}, "s": 5, "t": "RESUMED"})

    @pytest.mark.asyncio
    async def test_receive_loop_invalid_json(self) -> None:
        """Invalid JSON in receive loop should be skipped, not crash."""
        gw = self._make_gateway()

        messages = ["not json {", '{"op": 11, "d": null, "s": null, "t": null}']

        async def async_gen():
            for m in messages:
                yield m

        gw._ws = async_gen()

        await gw._receive_loop()

    @pytest.mark.asyncio
    async def test_hello_starts_heartbeat_and_identify(self) -> None:
        """OP_HELLO should start heartbeat and send identify/resume."""
        gw = self._make_gateway()
        gw._start_heartbeat = AsyncMock()
        gw._identify_or_resume = AsyncMock()

        hello_data = {"heartbeat_interval": 41250}
        await gw._handle_payload({"op": 10, "d": hello_data, "s": None, "t": None})

        gw._start_heartbeat.assert_awaited_once_with(41.25)
        gw._identify_or_resume.assert_awaited_once()


# ── Discord REST send_message retry logic ──────────────────────────


class TestDiscordRESTRetryLogic:
    """Test Discord REST client retry and rate limit handling."""

    def _make_rest(self):
        from missy.channels.discord.rest import DiscordRestClient

        rest = DiscordRestClient.__new__(DiscordRestClient)
        rest._token = "Bot test-token"
        rest._http = MagicMock()
        return rest

    def test_send_message_success(self) -> None:
        rest = self._make_rest()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "200000000000000001"}
        mock_resp.raise_for_status.return_value = None
        rest._http.post.return_value = mock_resp

        result = rest.send_message("100000000000000001", "hello")
        assert result["id"] == "200000000000000001"

    def test_send_message_rate_limited(self) -> None:
        """429 responses should trigger retry with Retry-After header."""
        rest = self._make_rest()

        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "0.01"}

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"id": "msg-after-retry"}
        ok_resp.raise_for_status.return_value = None

        rest._http.post.side_effect = [rate_resp, ok_resp]

        with patch("time.sleep"):  # skip delays
            result = rest.send_message("100000000000000001", "hello")

        assert result["id"] == "msg-after-retry"
        assert rest._http.post.call_count == 2

    def test_add_reaction_success(self) -> None:
        """add_reaction should not raise on 204 No Content."""
        rest = self._make_rest()
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_resp.raise_for_status.return_value = None
        rest._http.put.return_value = mock_resp

        # add_reaction returns None on success; just verify no exception
        rest.add_reaction("100000000000000001", "200000000000000001", "\U0001f44d")
        rest._http.put.assert_called_once()


# ── Network policy edge cases ──────────────────────────────────────


class TestNetworkPolicyEdgeCases:
    """Test network policy engine edge cases."""

    def _make_policy(self, **kwargs):
        from missy.config.settings import NetworkPolicy

        defaults = {
            "default_deny": True,
            "allowed_cidrs": [],
            "allowed_domains": [],
            "allowed_hosts": [],
            "provider_allowed_hosts": [],
            "tool_allowed_hosts": [],
            "discord_allowed_hosts": [],
        }
        defaults.update(kwargs)
        return NetworkPolicy(**defaults)

    def test_empty_domain_raises_value_error(self) -> None:
        """Empty hostname should raise ValueError."""
        from missy.policy.network import NetworkPolicyEngine

        policy = self._make_policy()
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(ValueError):
            engine.check_host("")

    def test_wildcard_domain_match(self) -> None:
        """Wildcard domain *.example.com should match sub.example.com."""
        from missy.policy.network import NetworkPolicyEngine

        policy = self._make_policy(allowed_domains=["*.example.com"])
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("sub.example.com") is True

    def test_cidr_allowlist(self) -> None:
        """IP in allowed CIDR should be allowed."""
        from missy.policy.network import NetworkPolicyEngine

        policy = self._make_policy(allowed_cidrs=["10.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("10.1.2.3") is True

    def test_denied_ip_raises_policy_violation(self) -> None:
        """IP not in CIDR should raise PolicyViolationError."""
        from missy.policy.network import NetworkPolicyEngine

        policy = self._make_policy(allowed_cidrs=["10.0.0.0/8"])
        engine = NetworkPolicyEngine(policy)
        with pytest.raises(PolicyViolationError):
            engine.check_host("192.168.1.1")

    def test_default_allow_mode(self) -> None:
        """default_deny=False should allow everything."""
        from missy.policy.network import NetworkPolicyEngine

        policy = self._make_policy(default_deny=False)
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("evil.example.com") is True

    def test_exact_host_match(self) -> None:
        """Exact hostname in allowed_hosts should be allowed."""
        from missy.policy.network import NetworkPolicyEngine

        policy = self._make_policy(allowed_hosts=["api.example.com"])
        engine = NetworkPolicyEngine(policy)
        assert engine.check_host("api.example.com") is True


# ── MCP manager lifecycle edge cases ───────────────────────────────


class TestMCPLifecycleEdgeCases:
    """Test MCP server restart and lifecycle."""

    def test_restart_nonexistent_server(self) -> None:
        """Restarting a server that doesn't exist should be a no-op."""
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mgr._clients = {}
        mgr._lock = threading.Lock()

        mgr.restart_server("nonexistent")
        # Should not crash

    def test_health_check_no_dead_servers(self) -> None:
        """health_check with all healthy servers should not restart anything."""
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mock_client = MagicMock()
        mock_client.is_alive.return_value = True
        mgr._clients = {"test-server": mock_client}
        mgr._lock = threading.Lock()

        mgr.health_check()

    def test_shutdown_idempotent(self) -> None:
        """Calling shutdown twice should not crash."""
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mock_client = MagicMock()
        mgr._clients = {"s1": mock_client}
        mgr._lock = threading.Lock()

        mgr.shutdown()
        mock_client.disconnect.assert_called_once()

        # Second call — all clients already disconnected
        mgr.shutdown()

    def test_call_tool_invalid_name(self) -> None:
        """Tool names without __ separator should be rejected."""
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mgr._clients = {}
        mgr._lock = threading.Lock()

        result = mgr.call_tool("no-separator", {"arg": "val"})
        assert "invalid" in result.lower()

    def test_call_tool_unsafe_name(self) -> None:
        """Tool names with special characters should be rejected."""
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mgr._clients = {}
        mgr._lock = threading.Lock()

        result = mgr.call_tool("server__tool;drop table", {})
        assert "unsafe" in result.lower()

    def test_call_tool_server_not_connected(self) -> None:
        """Calling a tool on a disconnected server should return error."""
        from missy.mcp.manager import McpManager

        mgr = McpManager.__new__(McpManager)
        mgr._clients = {}
        mgr._lock = threading.Lock()

        result = mgr.call_tool("missing__tool", {})
        assert "not connected" in result.lower()


# ── Scheduler parse edge cases ─────────────────────────────────────


class TestSchedulerParserEdgeCases:
    """Test scheduler schedule parsing edge cases."""

    def test_invalid_format_raises(self) -> None:
        """Invalid schedule strings should raise ValueError."""
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError):
            parse_schedule("completely invalid schedule string")

    def test_zero_interval_raises(self) -> None:
        """Zero-minute interval should raise ValueError."""
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError):
            parse_schedule("every 0 minutes")

    def test_negative_interval_raises(self) -> None:
        """Negative interval should raise ValueError."""
        from missy.scheduler.parser import parse_schedule

        with pytest.raises(ValueError):
            parse_schedule("every -5 minutes")

    def test_valid_daily_schedule(self) -> None:
        """Valid daily schedule should return cron dict."""
        from missy.scheduler.parser import parse_schedule

        result = parse_schedule("daily at 09:00")
        assert "hour" in result
        assert result["hour"] == 9

    def test_valid_interval_schedule(self) -> None:
        """Valid interval schedule should return interval dict."""
        from missy.scheduler.parser import parse_schedule

        result = parse_schedule("every 5 minutes")
        assert "minutes" in result
