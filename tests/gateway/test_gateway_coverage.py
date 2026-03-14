"""Coverage tests for missy.gateway.client and missy.agent.watchdog.

Targets the specific uncovered lines called out in the coverage report:
  - client.py  127-130  put()
  - client.py  175-176  apost() body (async POST response + event emit)
  - client.py  189-190  __aexit__ / aclose()
  - client.py  250-266  _emit_request_event
  - watchdog.py 64       _run while-loop body
  - watchdog.py 113-114  silent except around AuditEvent publish
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.events import event_bus
from missy.core.exceptions import PolicyViolationError
from missy.gateway.client import PolicyHTTPClient
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _permissive_config() -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=False,
            allowed_cidrs=[],
            allowed_domains=[],
            allowed_hosts=[],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


def _restrictive_config(
    allowed_hosts: list[str] | None = None,
    allowed_domains: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=allowed_domains or [],
            allowed_hosts=allowed_hosts or [],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


def _mock_response(status_code: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


@pytest.fixture(autouse=True)
def _engine_and_bus() -> Generator[None, None, None]:
    """Reset policy engine to permissive + clear event bus before each test."""
    original = engine_module._engine
    init_policy_engine(_permissive_config())
    event_bus.clear()
    yield
    engine_module._engine = original
    event_bus.clear()


# ---------------------------------------------------------------------------
# put() – lines 127-130
# ---------------------------------------------------------------------------


class TestSyncPut:
    def test_put_returns_response(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "put", return_value=mock_resp):
            resp = client.put("https://api.example.com/resource/1")
        assert resp.status_code == 200

    def test_put_performs_policy_check_before_request(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "put") as mock_put:
            with pytest.raises(PolicyViolationError):
                client.put("https://denied.example.com/resource/1")
            mock_put.assert_not_called()

    def test_put_emits_network_request_event(self) -> None:
        client = PolicyHTTPClient(session_id="sess-put", task_id="task-put")
        mock_resp = _mock_response(204)
        with patch.object(httpx.Client, "put", return_value=mock_resp):
            client.put("https://api.example.com/resource/1", json={"x": 1})
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].detail["method"] == "PUT"
        assert events[0].detail["status_code"] == 204
        assert events[0].session_id == "sess-put"

    def test_put_kwargs_forwarded_to_httpx(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "put", return_value=mock_resp) as mock_put:
            client.put("https://api.example.com/resource/1", json={"key": "val"}, timeout=10)
        mock_put.assert_called_once_with(
            "https://api.example.com/resource/1", json={"key": "val"}, timeout=10
        )

    def test_put_allowed_host_succeeds(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config(allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "put", return_value=mock_resp):
            resp = client.put("https://api.example.com/resource/1")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# apost() – lines 172-174
# ---------------------------------------------------------------------------


class TestAsyncPost:
    async def test_apost_returns_response(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            resp = await client.apost("https://api.example.com/items")
        assert resp.status_code == 201

    async def test_apost_performs_policy_check_before_request(self) -> None:
        engine_module._engine = None
        init_policy_engine(_restrictive_config())
        client = PolicyHTTPClient()
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            with pytest.raises(PolicyViolationError):
                await client.apost("https://denied.example.com/items")
            mock_post.assert_not_called()

    async def test_apost_emits_network_request_event(self) -> None:
        client = PolicyHTTPClient(session_id="async-sess", task_id="async-task")
        mock_resp = _mock_response(202)
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            await client.apost("https://api.example.com/items", json={"a": "b"})
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].detail["method"] == "POST"
        assert events[0].detail["status_code"] == 202
        assert events[0].session_id == "async-sess"

    async def test_apost_kwargs_forwarded_to_httpx(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await client.apost("https://api.example.com/items", data=b"raw")
        mock_post.assert_called_once_with("https://api.example.com/items", data=b"raw")


# ---------------------------------------------------------------------------
# aclose() / __aexit__ – lines 189-190, 206-208
# ---------------------------------------------------------------------------


class TestAsyncClose:
    async def test_aexit_closes_async_client(self) -> None:
        async with PolicyHTTPClient() as client:
            _ = client._get_async_client()
            assert client._async_client is not None
        # __aexit__ calls aclose(), which sets _async_client to None
        assert client._async_client is None

    async def test_aclose_idempotent_when_no_client_created(self) -> None:
        client = PolicyHTTPClient()
        assert client._async_client is None
        await client.aclose()  # must not raise
        assert client._async_client is None

    async def test_aclose_after_request_closes_client(self) -> None:
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp
        ):
            await client.aget("https://api.example.com/")
        inner = client._async_client
        assert inner is not None
        await client.aclose()
        assert client._async_client is None

    async def test_async_context_manager_makes_client_available_inside_block(self) -> None:
        mock_resp = _mock_response(200)
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp
        ):
            async with PolicyHTTPClient() as client:
                resp = await client.aget("https://api.example.com/ping")
        assert resp.status_code == 200
        assert client._async_client is None  # closed on __aexit__


# ---------------------------------------------------------------------------
# Sync context manager (belt-and-suspenders)
# ---------------------------------------------------------------------------


class TestSyncContextManager:
    def test_enter_returns_same_instance(self) -> None:
        client = PolicyHTTPClient()
        returned = client.__enter__()
        assert returned is client
        client.__exit__(None, None, None)

    def test_exit_closes_sync_client(self) -> None:
        client = PolicyHTTPClient()
        with client:
            _ = client._get_sync_client()
            assert client._sync_client is not None
        assert client._sync_client is None

    def test_sync_context_manager_full_request_flow(self) -> None:
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp), PolicyHTTPClient() as client:
            resp = client.get("https://api.example.com/ping")
        assert resp.status_code == 200
        assert client._sync_client is None


# ---------------------------------------------------------------------------
# _emit_request_event – line 251
# ---------------------------------------------------------------------------


class TestEmitRequestEvent:
    def test_emit_records_method_url_status(self) -> None:
        client = PolicyHTTPClient(session_id="s1", task_id="t1")
        client._emit_request_event("DELETE", "https://api.example.com/x", 204)
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        evt = events[0]
        assert evt.detail["method"] == "DELETE"
        assert evt.detail["url"] == "https://api.example.com/x"
        assert evt.detail["status_code"] == 204
        assert evt.session_id == "s1"
        assert evt.task_id == "t1"
        assert evt.category == "network"
        assert evt.result == "allow"

    def test_emit_records_correct_event_type(self) -> None:
        client = PolicyHTTPClient()
        client._emit_request_event("PATCH", "https://api.example.com/y", 200)
        events = event_bus.get_events(event_type="network_request")
        assert events[0].event_type == "network_request"

    def test_emit_multiple_requests_each_recorded(self) -> None:
        client = PolicyHTTPClient(session_id="multi")
        for method, code in (("GET", 200), ("POST", 201), ("PUT", 204)):
            client._emit_request_event(method, "https://api.example.com/", code)
        events = event_bus.get_events(event_type="network_request", session_id="multi")
        assert len(events) == 3
        methods = [e.detail["method"] for e in events]
        assert methods == ["GET", "POST", "PUT"]


# ---------------------------------------------------------------------------
# Category forwarding to policy engine
# ---------------------------------------------------------------------------


class TestCategoryForwarding:
    def test_category_forwarded_on_get(self) -> None:
        client = PolicyHTTPClient(category="provider")
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/")
        mock_engine.check_network.assert_called_once_with(
            "api.example.com", "", "", category="provider"
        )

    def test_category_forwarded_on_put(self) -> None:
        client = PolicyHTTPClient(category="tool")
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "put", return_value=mock_resp):
                client.put("https://api.example.com/resource/1")
        mock_engine.check_network.assert_called_once_with(
            "api.example.com", "", "", category="tool"
        )

    async def test_category_forwarded_on_apost(self) -> None:
        client = PolicyHTTPClient(category="discord")
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(
                httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
            ):
                await client.apost("https://discord.com/api/webhooks/x")
        mock_engine.check_network.assert_called_once_with(
            "discord.com", "", "", category="discord"
        )

    def test_empty_category_forwarded_by_default(self) -> None:
        client = PolicyHTTPClient()
        with patch("missy.gateway.client.get_policy_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine
            mock_resp = _mock_response(200)
            with patch.object(httpx.Client, "get", return_value=mock_resp):
                client.get("https://api.example.com/")
        _, _, call_kwargs = mock_engine.check_network.mock_calls[0]
        assert call_kwargs.get("category", "") == ""


# ---------------------------------------------------------------------------
# _check_url malformed URL edge cases
# ---------------------------------------------------------------------------


class TestCheckUrlEdgeCases:
    def test_empty_string_raises_value_error(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("")

    def test_relative_path_raises_value_error(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("/relative/path/only")

    def test_fragment_only_raises_value_error(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("#fragment")

    def test_host_only_without_scheme_raises_value_error(self) -> None:
        client = PolicyHTTPClient()
        with pytest.raises(ValueError):
            client._check_url("example.com/path")


# ---------------------------------------------------------------------------
# Watchdog._check_all() – unhealthy/recovery/threshold transitions
# ---------------------------------------------------------------------------

# Import here so the fixture above does not interfere with watchdog internals.
from missy.agent.watchdog import Watchdog  # noqa: E402


class TestWatchdogCheckAll:
    def _make_watchdog(self, threshold: int = 3) -> Watchdog:
        return Watchdog(check_interval=9999.0, failure_threshold=threshold)

    def test_healthy_check_sets_healthy_true_and_clears_failures(self) -> None:
        wd = self._make_watchdog()
        wd.register("db", lambda: True)
        # Seed a prior failure so we verify the reset path.
        wd._health["db"].healthy = False
        wd._health["db"].consecutive_failures = 2
        wd._check_all()
        assert wd._health["db"].healthy is True
        assert wd._health["db"].consecutive_failures == 0
        assert wd._health["db"].last_error == ""

    def test_recovery_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        wd = self._make_watchdog()
        wd.register("cache", lambda: True)
        wd._health["cache"].healthy = False  # simulate prior failure
        with caplog.at_level(logging.INFO, logger="missy.agent.watchdog"):
            wd._check_all()
        assert any("recovered" in r.message for r in caplog.records)

    def test_unhealthy_check_increments_failures(self) -> None:
        wd = self._make_watchdog()
        wd.register("net", lambda: False)
        wd._check_all()
        assert wd._health["net"].healthy is False
        assert wd._health["net"].consecutive_failures == 1

    def test_failure_threshold_escalates_to_error_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        wd = self._make_watchdog(threshold=2)
        wd.register("disk", lambda: False)
        # First call: below threshold (WARNING).
        with caplog.at_level(logging.WARNING, logger="missy.agent.watchdog"):
            wd._check_all()
        warning_messages = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_messages, "Expected a WARNING log before threshold is hit"
        caplog.clear()

        # Second call: at threshold → ERROR.
        with caplog.at_level(logging.ERROR, logger="missy.agent.watchdog"):
            wd._check_all()
        error_messages = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_messages, "Expected an ERROR log once threshold is reached"

    def test_check_fn_raising_exception_counts_as_failure(self) -> None:
        wd = self._make_watchdog()
        wd.register("flaky", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        wd._check_all()
        h = wd._health["flaky"]
        assert h.healthy is False
        assert h.consecutive_failures == 1
        assert "boom" in h.last_error

    def test_last_checked_is_updated_on_healthy(self) -> None:
        wd = self._make_watchdog()
        wd.register("svc", lambda: True)
        before = time.monotonic()
        wd._check_all()
        assert wd._health["svc"].last_checked >= before

    def test_last_checked_is_updated_on_failure(self) -> None:
        wd = self._make_watchdog()
        wd.register("svc", lambda: False)
        before = time.monotonic()
        wd._check_all()
        assert wd._health["svc"].last_checked >= before

    def test_multiple_subsystems_checked_independently(self) -> None:
        wd = self._make_watchdog()
        wd.register("good", lambda: True)
        wd.register("bad", lambda: False)
        wd._check_all()
        assert wd._health["good"].healthy is True
        assert wd._health["bad"].healthy is False

    def test_check_all_emits_audit_event_for_healthy(self) -> None:
        wd = self._make_watchdog()
        wd.register("api", lambda: True)
        wd._check_all()
        events = event_bus.get_events(event_type="watchdog.health_check")
        assert len(events) == 1
        assert events[0].detail["healthy"] is True
        assert events[0].detail["subsystem"] == "api"

    def test_check_all_emits_audit_event_for_unhealthy(self) -> None:
        wd = self._make_watchdog()
        wd.register("api", lambda: False)
        wd._check_all()
        events = event_bus.get_events(event_type="watchdog.health_check")
        assert len(events) == 1
        assert events[0].result == "error"
        assert events[0].detail["healthy"] is False

    def test_audit_event_publish_exception_is_silenced(self) -> None:
        """AuditEvent publish exceptions must not bubble out of _check_all.

        _check_all imports event_bus from missy.core.events at call time, so
        we patch it on the canonical module, not on the watchdog module.
        """
        wd = self._make_watchdog()
        wd.register("svc", lambda: True)
        with patch("missy.core.events.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus exploded")
            # Should complete without raising.
            wd._check_all()
        assert wd._health["svc"].healthy is True

    def test_audit_event_publish_exception_does_not_affect_health_state(self) -> None:
        """Health state must be consistent even when audit publish fails."""
        wd = self._make_watchdog()
        wd.register("svc", lambda: False)
        with patch("missy.core.events.event_bus") as mock_bus:
            mock_bus.publish.side_effect = RuntimeError("bus exploded")
            wd._check_all()
        assert wd._health["svc"].healthy is False
        assert wd._health["svc"].consecutive_failures == 1


# ---------------------------------------------------------------------------
# Watchdog._run() – line 64 (the while-loop body executes _check_all)
# ---------------------------------------------------------------------------


class TestWatchdogRun:
    def test_run_calls_check_all_while_not_stopped(self) -> None:
        """_run() should invoke _check_all at least once before stop() is called."""
        wd = Watchdog(check_interval=0.01, failure_threshold=3)
        called = threading.Event()

        def always_healthy() -> bool:
            called.set()
            return True

        wd.register("fast", always_healthy)
        wd.start()
        called.wait(timeout=2.0)
        wd.stop()
        assert called.is_set(), "_check_all was never invoked by _run()"

    def test_start_stop_lifecycle(self) -> None:
        wd = Watchdog(check_interval=9999.0, failure_threshold=3)
        wd.register("noop", lambda: True)
        wd.start()
        assert wd._thread is not None
        assert wd._thread.is_alive()
        wd.stop()
        # After stop(), the thread should terminate.
        wd._thread.join(timeout=2.0)
        assert not wd._thread.is_alive()

    def test_get_report_reflects_run_results(self) -> None:
        wd = Watchdog(check_interval=0.01, failure_threshold=3)
        ready = threading.Event()

        def check() -> bool:
            ready.set()
            return True

        wd.register("reporter", check)
        wd.start()
        ready.wait(timeout=2.0)
        wd.stop()
        report = wd.get_report()
        assert "reporter" in report
        assert report["reporter"]["healthy"] is True
