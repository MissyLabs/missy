"""Tests for gateway client DELETE and PATCH methods (session 14)."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, patch

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
from missy.gateway.client import create_client
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine


def _make_config(
    *,
    allowed_hosts: list[str] | None = None,
) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=True,
            allowed_cidrs=[],
            allowed_domains=[],
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
def _policy(tmp_path: object) -> Generator[None]:
    cfg = _make_config(allowed_hosts=["api.example.com:443"])
    init_policy_engine(cfg)
    yield
    engine_module._engine = None


class TestSyncDeleteMethod:
    """Test synchronous HTTP DELETE through PolicyHTTPClient."""

    def test_delete_allowed_host(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(204)
        with patch.object(client, "_get_sync_client") as mock_client:
            mock_client.return_value.delete.return_value = resp
            result = client.delete("https://api.example.com:443/resource/1")
        assert result.status_code == 204

    def test_delete_denied_host(self):
        client = create_client(session_id="s1", task_id="t1")
        with pytest.raises(PolicyViolationError):
            client.delete("https://evil.example.com/resource")

    def test_delete_emits_audit_event(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(200)
        events: list = []
        def cb(e): events.append(e)  # noqa: E704
        event_bus.subscribe("network_request", cb)
        try:
            with patch.object(client, "_get_sync_client") as mock_client:
                mock_client.return_value.delete.return_value = resp
                client.delete("https://api.example.com:443/resource/1")
            assert len(events) == 1
            assert events[0].detail["method"] == "DELETE"
        finally:
            event_bus.unsubscribe("network_request", cb)

    def test_delete_sanitizes_kwargs(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(204)
        with patch.object(client, "_get_sync_client") as mock_client:
            mock_client.return_value.delete.return_value = resp
            client.delete(
                "https://api.example.com:443/r",
                follow_redirects=True,
            )
            _, call_kwargs = mock_client.return_value.delete.call_args
            assert "follow_redirects" not in call_kwargs


class TestSyncPatchMethod:
    """Test synchronous HTTP PATCH through PolicyHTTPClient."""

    def test_patch_allowed_host(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(200, json={"updated": True})
        with patch.object(client, "_get_sync_client") as mock_client:
            mock_client.return_value.patch.return_value = resp
            result = client.patch(
                "https://api.example.com:443/resource/1",
                json={"name": "new"},
            )
        assert result.status_code == 200

    def test_patch_denied_host(self):
        client = create_client(session_id="s1", task_id="t1")
        with pytest.raises(PolicyViolationError):
            client.patch("https://evil.example.com/resource", json={"x": 1})

    def test_patch_emits_audit_event(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(200)
        events: list = []
        def cb(e): events.append(e)  # noqa: E704
        event_bus.subscribe("network_request", cb)
        try:
            with patch.object(client, "_get_sync_client") as mock_client:
                mock_client.return_value.patch.return_value = resp
                client.patch("https://api.example.com:443/resource/1")
            assert len(events) == 1
            assert events[0].detail["method"] == "PATCH"
        finally:
            event_bus.unsubscribe("network_request", cb)


@pytest.mark.asyncio
class TestAsyncDeleteMethod:
    """Test asynchronous HTTP DELETE through PolicyHTTPClient."""

    async def test_adelete_allowed_host(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(204)
        with patch.object(client, "_get_async_client") as mock_client:
            mock_client.return_value.delete = AsyncMock(return_value=resp)
            result = await client.adelete("https://api.example.com:443/resource/1")
        assert result.status_code == 204

    async def test_adelete_denied_host(self):
        client = create_client(session_id="s1", task_id="t1")
        with pytest.raises(PolicyViolationError):
            await client.adelete("https://evil.example.com/resource")


@pytest.mark.asyncio
class TestAsyncPatchMethod:
    """Test asynchronous HTTP PATCH through PolicyHTTPClient."""

    async def test_apatch_allowed_host(self):
        client = create_client(session_id="s1", task_id="t1")
        resp = httpx.Response(200)
        with patch.object(client, "_get_async_client") as mock_client:
            mock_client.return_value.patch = AsyncMock(return_value=resp)
            result = await client.apatch(
                "https://api.example.com:443/resource/1",
                json={"name": "new"},
            )
        assert result.status_code == 200

    async def test_apatch_denied_host(self):
        client = create_client(session_id="s1", task_id="t1")
        with pytest.raises(PolicyViolationError):
            await client.apatch("https://evil.example.com/resource")
