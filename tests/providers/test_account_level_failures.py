"""wtf.md incident: an HTTP 402 on one round-robin account must be retried on
the sibling account, bench the failing account immediately, and keep the
upstream body for diagnosis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from missy.config.settings import ProviderConfig
from missy.core.exceptions import ProviderError
from missy.providers.health import (
    ProviderFailureClass,
    classify_provider_error,
    is_account_level_failure,
)
from missy.providers.round_robin import RoundRobinAccounts
from tests.agent.test_provider_fallback import (
    _bare_runtime,
    _install_registry,
    _MultiAccountFailNTimesProvider,
)


def _http_error(status: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(
        f"Client error '{status} Payment Required' for url '{request.url}'",
        request=request,
        response=response,
    )


class TestClassification:
    def test_402_is_account_level(self):
        exc = ProviderError(
            "openai-codex request failed: Client error '402 Payment Required' for url 'x'"
        )
        assert classify_provider_error(exc) == ProviderFailureClass.ACCOUNT
        assert is_account_level_failure(exc)

    def test_status_on_cause_is_detected(self):
        wrapped = ProviderError("request failed")
        wrapped.__cause__ = _http_error(402, "{}")
        assert classify_provider_error(wrapped) == ProviderFailureClass.ACCOUNT

    def test_openai_sdk_style_status_code(self):
        exc = Exception("Error code: 402 - {'error': {'code': 'insufficient_quota'}}")
        assert classify_provider_error(exc) == ProviderFailureClass.ACCOUNT

    def test_rate_limit_and_auth_unchanged(self):
        assert classify_provider_error(ProviderError("rate limited: 429")) == "rate_limit"
        assert classify_provider_error(ProviderError("authentication failed 401")) == "auth"
        assert not is_account_level_failure(ProviderError("timed out"))
        assert not is_account_level_failure(None)


class TestImmediateBackoff:
    def test_immediate_failure_benches_account_at_once(self):
        rr = RoundRobinAccounts(["a", "b"], make_rate_limiter=lambda: None)
        bad = rr._accounts[1]
        rr.record_failure(bad, immediate=True)
        picks = {rr.select().index for _ in range(10)}
        assert picks == {0}

    def test_ordinary_failure_still_needs_threshold(self):
        rr = RoundRobinAccounts(["a", "b"], make_rate_limiter=lambda: None)
        rr.record_failure(rr._accounts[1])
        picks = {rr.select().index for _ in range(4)}
        assert picks == {0, 1}


class TestCodexProviderIntegration:
    def test_402_benches_account_and_error_keeps_body(self):
        from missy.providers.codex_provider import CodexProvider

        cfg = ProviderConfig(
            name="openai-codex",
            model="gpt-5.6-sol",
            oauth_accounts=["person", "tovdc"],
            key_rotation_strategy="round_robin",
        )
        provider = CodexProvider(cfg)
        body = '{"detail": {"code": "deactivated_workspace"}}'
        with (
            patch("missy.providers.codex_provider._load_oauth_token", return_value="tok"),
            patch.object(CodexProvider, "_post_sse", side_effect=_http_error(402, body)),
        ):
            provider._prepare_call()
            account = provider._account_local.current
            try:
                list(provider._stream_sse(MagicMock(), {}, "tok", "acct"))
            except ProviderError as exc:
                message = str(exc)
            else:  # pragma: no cover
                raise AssertionError("expected ProviderError")
        assert "HTTP 402" in message and "deactivated_workspace" in message
        assert account.unhealthy_until > 0  # benched on the first 402


class TestRuntimeSiblingRetry:
    def test_402_retries_sibling_account_before_fallback(self):
        class _PaymentRequired(_MultiAccountFailNTimesProvider):
            def complete(self, messages, **kwargs):
                self.calls += 1
                account = (self.calls - 1) % self.account_count
                self.accounts_seen.append(account)
                if self.calls <= self._fail_times:
                    raise ProviderError(
                        "openai-codex request failed (HTTP 402): Client error "
                        "'402 Payment Required'"
                    )
                return _ok(self.name)

        def _ok(name):
            from missy.providers.base import CompletionResponse

            return CompletionResponse(
                content=f"{name} reply", model="m", provider=name, usage={}, raw={}
            )

        cfg = ProviderConfig(name="codex", model="m", api_key="unused")
        provider = _PaymentRequired("codex", cfg, fail_times=1)
        registry = _install_registry(("codex", provider, cfg))
        rt = _bare_runtime("codex")
        with patch("missy.agent.runtime.get_registry", return_value=registry):
            result = rt._single_turn(
                provider=provider,
                system_prompt="sys",
                messages=[{"role": "user", "content": "hi"}],
                session_id="s1",
                task_id="t1",
            )
        assert result.content == "codex reply"
        assert provider.accounts_seen == [0, 1]
