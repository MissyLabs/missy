"""Tests for missy.gateway.client.PolicyHTTPClient."""

from __future__ import annotations

from typing import Generator
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
from missy.gateway.client import PolicyHTTPClient, create_client
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_config(
    *,
    default_deny: bool = True,
    allowed_hosts: list[str] | None = None,
    allowed_domains: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=default_deny,
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


@pytest.fixture(autouse=True)
def setup_engine() -> Generator[None, None, None]:
    """Install a permissive engine so most client tests can focus on HTTP behaviour."""
    original = engine_module._engine
    init_policy_engine(make_config(default_deny=False))
    event_bus.clear()
    yield
    engine_module._engine = original
    event_bus.clear()


def _mock_response(status_code: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# create_client factory
# ---------------------------------------------------------------------------


class TestCreateClient:
    def test_returns_policy_http_client(self):
        client = create_client()
        assert isinstance(client, PolicyHTTPClient)

    def test_parameters_forwarded(self):
        client = create_client(session_id="s1", task_id="t1", timeout=60)
        assert client.session_id == "s1"
        assert client.task_id == "t1"
        assert client.timeout == 60

    def test_default_parameters(self):
        client = create_client()
        assert client.session_id == ""
        assert client.task_id == ""
        assert client.timeout == 30


# ---------------------------------------------------------------------------
# Synchronous GET
# ---------------------------------------------------------------------------


class TestSyncGet:
    def test_get_returns_response(self):
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            resp = client.get("https://api.example.com/data")
        assert resp.status_code == 200

    def test_get_policy_denied_raises(self):
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True, allowed_hosts=[]))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            client.get("https://evil.example.com/steal")

    def test_get_policy_check_before_request(self):
        """The HTTP call must NOT happen when policy denies."""
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True))
        client = PolicyHTTPClient()
        with patch.object(httpx.Client, "get") as mock_get:
            with pytest.raises(PolicyViolationError):
                client.get("https://denied.com/path")
            mock_get.assert_not_called()

    def test_get_emits_request_event_on_success(self):
        client = PolicyHTTPClient(session_id="s1", task_id="t1")
        mock_resp = _mock_response(200)
        with patch.object(httpx.Client, "get", return_value=mock_resp):
            client.get("https://api.example.com/")
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].detail["method"] == "GET"
        assert events[0].detail["status_code"] == 200

    def test_get_kwargs_forwarded(self):
        client = PolicyHTTPClient()
        mock_resp = _mock_response()
        with patch.object(httpx.Client, "get", return_value=mock_resp) as mock_get:
            client.get("https://api.example.com/", headers={"X-Test": "1"}, timeout=5)
        mock_get.assert_called_once_with(
            "https://api.example.com/", headers={"X-Test": "1"}, timeout=5
        )


# ---------------------------------------------------------------------------
# Synchronous POST
# ---------------------------------------------------------------------------


class TestSyncPost:
    def test_post_returns_response(self):
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            resp = client.post("https://api.example.com/create", json={"k": "v"})
        assert resp.status_code == 201

    def test_post_policy_denied_raises(self):
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            client.post("https://evil.com/upload", data=b"payload")

    def test_post_emits_request_event(self):
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(httpx.Client, "post", return_value=mock_resp):
            client.post("https://api.example.com/create")
        events = event_bus.get_events(event_type="network_request")
        assert events[0].detail["method"] == "POST"


# ---------------------------------------------------------------------------
# Async GET
# ---------------------------------------------------------------------------


class TestAsyncGet:
    async def test_aget_returns_response(self):
        client = PolicyHTTPClient()
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            resp = await client.aget("https://api.example.com/data")
        assert resp.status_code == 200

    async def test_aget_policy_denied_raises(self):
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            await client.aget("https://denied.com/path")

    async def test_aget_emits_request_event(self):
        client = PolicyHTTPClient(session_id="as", task_id="at")
        mock_resp = _mock_response(200)
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_resp):
            await client.aget("https://api.example.com/")
        events = event_bus.get_events(event_type="network_request")
        assert len(events) == 1
        assert events[0].session_id == "as"


# ---------------------------------------------------------------------------
# Async POST
# ---------------------------------------------------------------------------


class TestAsyncPost:
    async def test_apost_returns_response(self):
        client = PolicyHTTPClient()
        mock_resp = _mock_response(201)
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_resp
        ):
            resp = await client.apost("https://api.example.com/create")
        assert resp.status_code == 201

    async def test_apost_policy_denied_raises(self):
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            await client.apost("https://evil.com/upload")


# ---------------------------------------------------------------------------
# URL parsing / host extraction
# ---------------------------------------------------------------------------


class TestURLParsing:
    def test_malformed_url_raises_value_error(self):
        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("not-a-url-at-all")

    def test_url_without_scheme_raises_value_error(self):
        client = PolicyHTTPClient()
        with pytest.raises(ValueError):
            client._check_url("api.example.com/path")

    def test_ipv6_host_in_url(self):
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=False))
        client = PolicyHTTPClient()
        # Should not raise – just check that it completes without error
        client._check_url("https://[::1]/path")

    def test_host_extracted_from_url_checked(self):
        """Policy should be checked with the bare hostname, not the full URL."""
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True, allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        # Should not raise
        client._check_url("https://api.example.com/some/path?q=1")

    def test_denied_host_not_in_allowlist(self):
        engine_module._engine = None
        init_policy_engine(make_config(default_deny=True, allowed_hosts=["api.example.com"]))
        client = PolicyHTTPClient()
        with pytest.raises(PolicyViolationError):
            client._check_url("https://evil.example.com/steal")


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_sync_context_manager_closes_client(self):
        with PolicyHTTPClient() as client:
            # Trigger lazy creation of the sync client
            _ = client._get_sync_client()
        # After exit the client should be None (closed)
        assert client._sync_client is None

    async def test_async_context_manager_closes_client(self):
        async with PolicyHTTPClient() as client:
            _ = client._get_async_client()
        assert client._async_client is None

    def test_close_without_client_creation_is_safe(self):
        client = PolicyHTTPClient()
        client.close()  # No sync client was ever created – should not raise.

    async def test_aclose_without_client_creation_is_safe(self):
        client = PolicyHTTPClient()
        await client.aclose()


# ---------------------------------------------------------------------------
# Client reuse (lazy singleton pattern)
# ---------------------------------------------------------------------------


class TestClientReuse:
    def test_sync_client_reused_across_calls(self):
        client = PolicyHTTPClient()
        c1 = client._get_sync_client()
        c2 = client._get_sync_client()
        assert c1 is c2

    def test_async_client_reused_across_calls(self):
        client = PolicyHTTPClient()
        c1 = client._get_async_client()
        c2 = client._get_async_client()
        assert c1 is c2
