"""Test that REST policy check failures result in denial (fail-closed).

Verifies the fix for the fail-open pattern where unexpected exceptions
in _check_rest_policy would silently allow requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.gateway.client import PolicyHTTPClient, PolicyViolationError


@pytest.fixture
def client():
    return PolicyHTTPClient()


class TestRestPolicyFailClosed:
    def test_unexpected_error_raises_policy_violation(self, client):
        """An unexpected exception in REST policy evaluation must deny the request."""
        mock_engine = MagicMock()
        mock_engine.rest_policy.check.side_effect = RuntimeError("internal bug")

        with patch("missy.gateway.client.get_policy_engine", return_value=mock_engine), pytest.raises(PolicyViolationError):
            client._check_rest_policy("api.example.com", "GET", "/foo")

    def test_explicit_deny_still_raises(self, client):
        """An explicit deny result still raises PolicyViolationError."""
        mock_engine = MagicMock()
        mock_engine.rest_policy.check.return_value = "deny"

        with patch("missy.gateway.client.get_policy_engine", return_value=mock_engine), pytest.raises(PolicyViolationError, match="REST policy denied"):
            client._check_rest_policy("api.example.com", "DELETE", "/")

    def test_allow_result_passes(self, client):
        """An 'allow' result does not raise."""
        mock_engine = MagicMock()
        mock_engine.rest_policy.check.return_value = "allow"

        with patch("missy.gateway.client.get_policy_engine", return_value=mock_engine):
            # Should not raise
            client._check_rest_policy("api.example.com", "GET", "/repos")

    def test_no_rest_policy_passes(self, client):
        """When no REST policy is configured, all requests pass."""
        mock_engine = MagicMock()
        mock_engine.rest_policy = None

        with patch("missy.gateway.client.get_policy_engine", return_value=mock_engine):
            # Should not raise
            client._check_rest_policy("api.example.com", "GET", "/anything")

    def test_type_error_denied(self, client):
        """TypeError during check is treated as failure → denial."""
        mock_engine = MagicMock()
        mock_engine.rest_policy.check.side_effect = TypeError("bad argument")

        with patch("missy.gateway.client.get_policy_engine", return_value=mock_engine), pytest.raises(PolicyViolationError, match="REST policy check error"):
            client._check_rest_policy("api.example.com", "POST", "/data")
