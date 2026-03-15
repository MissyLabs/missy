"""Session 19: Gateway hardening tests — response size limits."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from missy.gateway.client import PolicyHTTPClient


class TestResponseSizeLimit:
    """Test _check_response_size enforcement."""

    def _make_client(self, max_bytes: int = 1024) -> PolicyHTTPClient:
        with patch("missy.gateway.client.get_policy_engine"):
            return PolicyHTTPClient(
                session_id="test",
                task_id="t1",
                max_response_bytes=max_bytes,
            )

    def test_default_max_response_bytes(self):
        with patch("missy.gateway.client.get_policy_engine"):
            client = PolicyHTTPClient()
        assert client.max_response_bytes == PolicyHTTPClient.DEFAULT_MAX_RESPONSE_BYTES

    def test_custom_max_response_bytes(self):
        client = self._make_client(max_bytes=5000)
        assert client.max_response_bytes == 5000

    def test_response_within_limit_passes(self):
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={"content-length": "500"})
        # Should not raise
        client._check_response_size(response, "http://example.com")

    def test_response_exceeding_limit_raises(self):
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={"content-length": "2048"})
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(response, "http://example.com")

    def test_response_without_content_length_passes(self):
        """Chunked responses without Content-Length should pass the header check."""
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={})
        # Should not raise — no Content-Length to check
        client._check_response_size(response, "http://example.com")

    def test_response_with_non_numeric_content_length(self):
        """Non-numeric Content-Length should be treated as unknown (pass)."""
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={"content-length": "invalid"})
        # Should not raise — non-numeric treated as 0
        client._check_response_size(response, "http://example.com")

    def test_response_exactly_at_limit(self):
        """Response exactly at limit should pass."""
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={"content-length": "1024"})
        client._check_response_size(response, "http://example.com")

    def test_response_one_over_limit(self):
        """Response one byte over limit should fail."""
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={"content-length": "1025"})
        with pytest.raises(ValueError, match="too large"):
            client._check_response_size(response, "http://example.com")

    def test_zero_content_length(self):
        """Zero Content-Length should pass."""
        client = self._make_client(max_bytes=1024)
        response = httpx.Response(200, headers={"content-length": "0"})
        client._check_response_size(response, "http://example.com")


class TestGatewayMethodIntegration:
    """Test that all HTTP methods call _check_response_size."""

    def _make_client_with_mocked_transport(self):
        """Create a client with mocked policy engine and transport."""
        with patch("missy.gateway.client.get_policy_engine"):
            client = PolicyHTTPClient(
                session_id="test",
                task_id="t1",
                max_response_bytes=100,
            )
        return client

    def test_get_checks_response_size(self):
        client = self._make_client_with_mocked_transport()
        mock_response = httpx.Response(200, headers={"content-length": "200"})

        with (
            patch.object(client, "_check_url"),
            patch.object(client, "_get_sync_client") as mock_sync,
        ):
            mock_sync.return_value.get.return_value = mock_response
            with pytest.raises(ValueError, match="too large"):
                client.get("http://example.com")

    def test_post_checks_response_size(self):
        client = self._make_client_with_mocked_transport()
        mock_response = httpx.Response(200, headers={"content-length": "200"})

        with (
            patch.object(client, "_check_url"),
            patch.object(client, "_get_sync_client") as mock_sync,
        ):
            mock_sync.return_value.post.return_value = mock_response
            with pytest.raises(ValueError, match="too large"):
                client.post("http://example.com")

    @pytest.mark.asyncio
    async def test_aget_checks_response_size(self):
        client = self._make_client_with_mocked_transport()
        mock_response = httpx.Response(200, headers={"content-length": "200"})

        with patch.object(client, "_check_url"):
            mock_async_client = MagicMock()
            mock_async_client.get = AsyncMock(return_value=mock_response)
            with (
                patch.object(client, "_get_async_client", return_value=mock_async_client),
                pytest.raises(ValueError, match="too large"),
            ):
                await client.aget("http://example.com")

    @pytest.mark.asyncio
    async def test_apost_checks_response_size(self):
        client = self._make_client_with_mocked_transport()
        mock_response = httpx.Response(200, headers={"content-length": "200"})

        with patch.object(client, "_check_url"):
            mock_async_client = MagicMock()
            mock_async_client.post = AsyncMock(return_value=mock_response)
            with (
                patch.object(client, "_get_async_client", return_value=mock_async_client),
                pytest.raises(ValueError, match="too large"),
            ):
                await client.apost("http://example.com")


class TestGatewayPoolLimits:
    """Verify pool limit configuration."""

    def test_pool_limits_set(self):
        assert PolicyHTTPClient._POOL_LIMITS.max_connections == 20
        assert PolicyHTTPClient._POOL_LIMITS.max_keepalive_connections == 10

    def test_allowed_schemes(self):
        assert {"http", "https"} == PolicyHTTPClient._ALLOWED_SCHEMES

    def test_allowed_kwargs(self):
        allowed = PolicyHTTPClient._ALLOWED_KWARGS
        assert "headers" in allowed
        assert "json" in allowed
        assert "verify" not in allowed
        assert "transport" not in allowed
        assert "auth" not in allowed
        assert "base_url" not in allowed
