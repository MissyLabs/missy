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
        engine.check_network.return_value = True
        with patch("missy.policy.engine.get_policy_engine", return_value=engine):
            # Should not raise.
            _policy_request_hook(_request())
        engine.check_network.assert_called_once()
        # Called with the host and provider category.
        args, kwargs = engine.check_network.call_args
        assert args[0] == "api.anthropic.com"
        assert kwargs.get("category") == "provider"

    def test_denies_when_policy_denies(self):
        engine = MagicMock()
        engine.check_network.side_effect = PolicyViolationError(
            "denied", category="network", detail="nope"
        )
        with (
            patch("missy.policy.engine.get_policy_engine", return_value=engine),
            pytest.raises(PolicyViolationError),
        ):
            _policy_request_hook(_request("https://evil.example.com/x"))

    def test_no_host_is_noop(self):
        # A request URL without a host must not invoke the policy engine.
        req = MagicMock()
        req.url.host = ""
        with patch("missy.policy.engine.get_policy_engine") as get_engine:
            _policy_request_hook(req)
        get_engine.assert_not_called()

    def test_allows_when_engine_uninitialised(self):
        # Defensive: when the policy engine is not initialised, allow but log.
        with patch(
            "missy.policy.engine.get_policy_engine",
            side_effect=RuntimeError("not initialised"),
        ):
            # Should not raise.
            _policy_request_hook(_request())

    def test_emits_audit_events(self):
        engine = MagicMock()
        engine.check_network.return_value = True
        with (
            patch("missy.policy.engine.get_policy_engine", return_value=engine),
            patch("missy.providers.policy_http.event_bus") as bus,
        ):
            _policy_request_hook(_request())
        assert bus.publish.called


class TestBuildClient:
    def test_returns_httpx_client_with_hook(self):
        client = build_policy_http_client(timeout=15.0)
        try:
            assert isinstance(client, httpx.Client)
            hooks = client.event_hooks.get("request", [])
            assert _policy_request_hook in hooks
        finally:
            client.close()
