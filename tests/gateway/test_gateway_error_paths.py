"""Tests for PolicyHTTPClient error handling paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.gateway.client import PolicyHTTPClient
from missy.core.exceptions import PolicyViolationError


class TestURLValidation:
    """Tests for _check_url URL parsing edge cases."""

    def test_url_too_long(self):
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url("https://example.com/" + "a" * 10000)

    def test_unsupported_scheme_ftp(self):
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ftp://example.com/file")

    def test_unsupported_scheme_file(self):
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("file:///etc/passwd")

    def test_no_host_in_url(self):
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("http://")

    def test_malformed_url_no_scheme(self):
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("just-a-string")


class TestRESTPolicy:
    """Tests for _check_rest_policy edge cases."""

    @patch("missy.gateway.client.get_policy_engine")
    def test_no_rest_policy_allows(self, mock_engine):
        engine = MagicMock()
        engine.rest_policy = None
        mock_engine.return_value = engine

        client = PolicyHTTPClient(timeout=5)
        # Should not raise
        client._check_rest_policy("example.com", "GET", "/api/test")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_deny(self, mock_engine):
        engine = MagicMock()
        engine.rest_policy.check.return_value = "deny"
        mock_engine.return_value = engine

        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(PolicyViolationError, match="REST policy denied"):
            client._check_rest_policy("example.com", "DELETE", "/resources/1")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_allow(self, mock_engine):
        engine = MagicMock()
        engine.rest_policy.check.return_value = "allow"
        mock_engine.return_value = engine

        client = PolicyHTTPClient(timeout=5)
        # Should not raise
        client._check_rest_policy("example.com", "GET", "/api/test")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_exception_degrades_gracefully(self, mock_engine):
        """When rest_policy.check() raises, the request should be allowed."""
        engine = MagicMock()
        engine.rest_policy.check.side_effect = RuntimeError("broken parser")
        mock_engine.return_value = engine

        client = PolicyHTTPClient(timeout=5)
        # Should not raise — graceful degradation
        client._check_rest_policy("example.com", "GET", "/api/test")

    @patch("missy.gateway.client.get_policy_engine")
    def test_rest_policy_check_with_various_methods(self, mock_engine):
        engine = MagicMock()
        engine.rest_policy.check.return_value = "allow"
        mock_engine.return_value = engine

        client = PolicyHTTPClient(timeout=5)
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            client._check_rest_policy("api.example.com", method, "/v1/resource")
            engine.rest_policy.check.assert_called_with("api.example.com", method, "/v1/resource")


class TestInteractiveApproval:
    """Tests for operator override in _check_url."""

    @patch("missy.gateway.client._interactive_approval", None)
    @patch("missy.gateway.client.get_policy_engine")
    def test_no_approval_instance_raises(self, mock_engine):
        mock_engine.return_value.check_network.side_effect = PolicyViolationError(
            "denied", category="network", detail="blocked host"
        )
        client = PolicyHTTPClient(timeout=5)
        with pytest.raises(PolicyViolationError):
            client._check_url("https://blocked.example.com")

    @patch("missy.gateway.client.get_policy_engine")
    def test_policy_allows_no_error(self, mock_engine):
        mock_engine.return_value.check_network.return_value = None
        client = PolicyHTTPClient(timeout=5)
        # Should not raise
        client._check_url("https://allowed.example.com", method="GET")


class TestAllowedKwargs:
    """Test that unsafe kwargs are stripped."""

    def test_allowed_kwargs_defined(self):
        assert "headers" in PolicyHTTPClient._ALLOWED_KWARGS
        assert "params" in PolicyHTTPClient._ALLOWED_KWARGS
        # Security-sensitive kwargs must NOT be in allowed set
        assert "verify" not in PolicyHTTPClient._ALLOWED_KWARGS
        assert "transport" not in PolicyHTTPClient._ALLOWED_KWARGS
        assert "base_url" not in PolicyHTTPClient._ALLOWED_KWARGS
        assert "auth" not in PolicyHTTPClient._ALLOWED_KWARGS
