"""Tests for the policy-aware provider HTTP client (missy.providers.policy_http)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from missy.core.exceptions import PolicyViolationError
from missy.providers.policy_http import (
    _policy_request_hook,
    build_policy_http_client,
)


def _request(url: str = "https://api.anthropic.com/v1/messages") -> httpx.Request:
    return httpx.Request("POST", url)


class TestPolicyRequestHook:
    def test_allows_when_policy_permits(self):
        engine = MagicMock()
        engine.check_network_resolved.return_value = (True, "203.0.113.10")
        engine.rest_policy.check.return_value = None
        with patch("missy.policy.engine.get_policy_engine", return_value=engine):
            # Should not raise.
            _policy_request_hook(_request())
        engine.check_network_resolved.assert_called_once()
        # Called with the host and provider category.
        args, kwargs = engine.check_network_resolved.call_args
        assert args[0] == "api.anthropic.com"
        assert kwargs.get("category") == "provider"

    def test_denies_when_policy_denies(self):
        engine = MagicMock()
        engine.check_network_resolved.side_effect = PolicyViolationError(
            "denied", category="network", detail="nope"
        )
        with (
            patch("missy.policy.engine.get_policy_engine", return_value=engine),
            pytest.raises(PolicyViolationError),
        ):
            _policy_request_hook(_request("https://evil.example.com/x"))

    def test_no_host_fails_closed(self):
        req = MagicMock()
        req.url.host = ""
        with (
            patch("missy.policy.engine.get_policy_engine") as get_engine,
            pytest.raises(PolicyViolationError, match="no policy-checkable"),
        ):
            _policy_request_hook(req)
        get_engine.assert_not_called()

    def test_denies_when_engine_uninitialised(self):
        with (
            patch(
                "missy.policy.engine.get_policy_engine",
                side_effect=RuntimeError("not initialised"),
            ),
            pytest.raises(PolicyViolationError, match="not initialized"),
        ):
            _policy_request_hook(_request())

    def test_rest_policy_denial_applies_to_provider_request(self):
        engine = MagicMock()
        engine.check_network_resolved.return_value = (True, "203.0.113.10")
        engine.rest_policy.check.return_value = "deny"
        with (
            patch("missy.policy.engine.get_policy_engine", return_value=engine),
            pytest.raises(PolicyViolationError, match="REST policy denied"),
        ):
            _policy_request_hook(_request("https://api.anthropic.com/v1/../admin?token=secret"))
        engine.rest_policy.check.assert_called_once_with("api.anthropic.com", "POST", "/admin")

    def test_emits_audit_events(self):
        engine = MagicMock()
        engine.check_network_resolved.return_value = (True, "203.0.113.10")
        engine.rest_policy.check.return_value = None
        with (
            patch("missy.policy.engine.get_policy_engine", return_value=engine),
            patch("missy.providers.policy_http.event_bus") as bus,
        ):
            _policy_request_hook(_request("https://user:pass@api.anthropic.com/v1/messages?key=x"))
        assert bus.publish.called
        event = bus.publish.call_args.args[0]
        assert "user" not in event.detail["url"]
        assert "pass" not in event.detail["url"]
        assert "key=" not in event.detail["url"]


class TestBuildClient:
    def test_returns_httpx_client_with_hook(self):
        client = build_policy_http_client(timeout=15.0)
        try:
            assert isinstance(client, httpx.Client)
            hooks = client.event_hooks.get("request", [])
            assert _policy_request_hook in hooks
            assert client._transport.__class__.__name__ == "PinnedHTTPTransport"
        finally:
            client.close()
