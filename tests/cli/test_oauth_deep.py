"""Deep tests for missy.cli.oauth — PKCE, JWT, token persistence, HTTP exchange.

Covers all public and private helpers with 45+ test cases across:
1. PKCE generation
2. Auth URL construction
3. JWT payload parsing
4. Account metadata extraction
5. Token save / load round-trips
6. refresh_token_if_needed state machine
7. Authorization code extraction from redirect URLs
8. Token exchange via httpx (mocked)
9. Token refresh via httpx (mocked)
10. _CallbackHandler GET handling
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import time
import urllib.parse
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import missy.cli.oauth as oauth_module
from missy.cli.oauth import (
    AUTHORIZE_URL,
    DEFAULT_CLIENT_ID,
    REDIRECT_URI,
    REFRESH_MARGIN_SECONDS,
    SCOPES,
    TOKEN_URL,
    _build_auth_url,
    _extract_account_metadata,
    _extract_code_from_url,
    _generate_pkce,
    _parse_jwt_payload,
    load_token,
    refresh_token_if_needed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jwt(payload: dict) -> str:
    """Build a minimal, unsigned JWT string from *payload*."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    body_bytes = json.dumps(payload).encode()
    body = base64.urlsafe_b64encode(body_bytes).rstrip(b"=").decode()
    return f"{header}.{body}.fakesig"


def _patch_token_file(tmp_path: Any):
    """Return a context-manager-compatible patcher for TOKEN_FILE."""
    token_file = tmp_path / "openai-oauth.json"
    return patch.object(oauth_module, "TOKEN_FILE", token_file)


# ---------------------------------------------------------------------------
# 1. PKCE generation
# ---------------------------------------------------------------------------


class TestGeneratePkce:
    def test_returns_two_strings(self):
        verifier, challenge = _generate_pkce()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_is_base64url_without_padding(self):
        verifier, _ = _generate_pkce()
        # Must be decodable as base64url with manual padding
        padded = verifier + "=" * (-len(verifier) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        assert len(decoded) == 96

    def test_verifier_has_expected_character_length(self):
        verifier, _ = _generate_pkce()
        # 96 bytes → 128 base64url characters (without padding)
        assert len(verifier) == 128

    def test_challenge_is_sha256_of_verifier(self):
        verifier, challenge = _generate_pkce()
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert challenge == expected

    def test_challenge_has_no_padding(self):
        _, challenge = _generate_pkce()
        assert "=" not in challenge

    def test_verifier_is_unique_across_calls(self):
        v1, _ = _generate_pkce()
        v2, _ = _generate_pkce()
        assert v1 != v2

    def test_challenge_no_url_unsafe_chars(self):
        _, challenge = _generate_pkce()
        assert "+" not in challenge
        assert "/" not in challenge


# ---------------------------------------------------------------------------
# 2. Auth URL construction
# ---------------------------------------------------------------------------


class TestBuildAuthUrl:
    def _parse(self, url: str) -> dict[str, str]:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        return {k: v[0] for k, v in qs.items()}

    def test_base_url_is_authorize_endpoint(self):
        url = _build_auth_url("cid", "state123", "challenge_abc")
        assert url.startswith(AUTHORIZE_URL + "?")

    def test_response_type_is_code(self):
        params = self._parse(_build_auth_url("cid", "st", "ch"))
        assert params["response_type"] == "code"

    def test_client_id_present(self):
        params = self._parse(_build_auth_url("my-client", "st", "ch"))
        assert params["client_id"] == "my-client"

    def test_redirect_uri_matches_constant(self):
        params = self._parse(_build_auth_url("cid", "st", "ch"))
        assert params["redirect_uri"] == REDIRECT_URI

    def test_scope_matches_constant(self):
        params = self._parse(_build_auth_url("cid", "st", "ch"))
        assert params["scope"] == SCOPES

    def test_code_challenge_present(self):
        params = self._parse(_build_auth_url("cid", "st", "mychallenge"))
        assert params["code_challenge"] == "mychallenge"

    def test_code_challenge_method_is_s256(self):
        params = self._parse(_build_auth_url("cid", "st", "ch"))
        assert params["code_challenge_method"] == "S256"

    def test_state_present(self):
        params = self._parse(_build_auth_url("cid", "unique_state", "ch"))
        assert params["state"] == "unique_state"

    def test_special_chars_in_client_id_are_encoded(self):
        url = _build_auth_url("id with spaces", "st", "ch")
        assert "id+with+spaces" in url or "id%20with%20spaces" in url


# ---------------------------------------------------------------------------
# 3. JWT payload parsing
# ---------------------------------------------------------------------------


class TestParseJwtPayload:
    def test_parses_valid_three_part_jwt(self):
        payload = {"sub": "user123", "email": "a@b.com"}
        token = _make_jwt(payload)
        result = _parse_jwt_payload(token)
        assert result["sub"] == "user123"
        assert result["email"] == "a@b.com"

    def test_returns_empty_dict_for_two_part_token(self):
        result = _parse_jwt_payload("header.onlyonepart")
        assert result == {}

    def test_returns_empty_dict_for_one_part_token(self):
        result = _parse_jwt_payload("onlyone")
        assert result == {}

    def test_returns_empty_dict_for_empty_string(self):
        result = _parse_jwt_payload("")
        assert result == {}

    def test_returns_empty_dict_for_malformed_base64_payload(self):
        result = _parse_jwt_payload("header.!!!notbase64!!!.sig")
        assert result == {}

    def test_returns_empty_dict_for_non_json_payload(self):
        garbage = base64.urlsafe_b64encode(b"not-json").decode().rstrip("=")
        result = _parse_jwt_payload(f"header.{garbage}.sig")
        assert result == {}

    def test_handles_payload_without_padding(self):
        # The implementation adds padding; verify it handles 1-char short segments
        payload = {"k": "v"}
        body_bytes = json.dumps(payload).encode()
        # Force a payload length that needs padding
        body = base64.urlsafe_b64encode(body_bytes).decode().rstrip("=")
        token = f"header.{body}.sig"
        result = _parse_jwt_payload(token)
        assert result.get("k") == "v"

    def test_nested_claims_preserved(self):
        payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_abc"}}
        token = _make_jwt(payload)
        result = _parse_jwt_payload(token)
        assert result["https://api.openai.com/auth"]["chatgpt_account_id"] == "acct_abc"


# ---------------------------------------------------------------------------
# 4. Account metadata extraction
# ---------------------------------------------------------------------------


class TestExtractAccountMetadata:
    def test_extracts_account_id_from_openai_namespace(self):
        payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_xyz"}}
        token = _make_jwt(payload)
        account_id, email = _extract_account_metadata(token)
        assert account_id == "acct_xyz"
        assert email == ""

    def test_falls_back_to_sub_when_no_namespace(self):
        payload = {"sub": "sub_fallback", "email": "user@test.com"}
        token = _make_jwt(payload)
        account_id, email = _extract_account_metadata(token)
        assert account_id == "sub_fallback"
        assert email == "user@test.com"

    def test_extracts_email(self):
        payload = {"sub": "s", "email": "hello@world.org"}
        token = _make_jwt(payload)
        _, email = _extract_account_metadata(token)
        assert email == "hello@world.org"

    def test_both_empty_when_token_unparseable(self):
        account_id, email = _extract_account_metadata("garbage")
        assert account_id == ""
        assert email == ""

    def test_openai_namespace_preferred_over_sub(self):
        payload = {
            "sub": "sub_value",
            "https://api.openai.com/auth": {"chatgpt_account_id": "ns_value"},
        }
        token = _make_jwt(payload)
        account_id, _ = _extract_account_metadata(token)
        assert account_id == "ns_value"

    def test_falls_back_to_sub_when_namespace_account_id_empty(self):
        payload = {
            "sub": "sub_value",
            "https://api.openai.com/auth": {"chatgpt_account_id": ""},
        }
        token = _make_jwt(payload)
        account_id, _ = _extract_account_metadata(token)
        assert account_id == "sub_value"


# ---------------------------------------------------------------------------
# 5. Token save / load
# ---------------------------------------------------------------------------


class TestSaveLoadToken:
    def test_round_trip(self, tmp_path):
        data = {"access_token": "tok", "expires_at": 9999999}
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            result = load_token()
        assert result == data

    def test_file_created_with_mode_600(self, tmp_path):
        data = {"access_token": "tok"}
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            token_file = tmp_path / "openai-oauth.json"
            file_mode = stat.S_IMODE(token_file.stat().st_mode)
        assert file_mode == 0o600

    def test_parent_directory_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "token.json"
        with patch.object(oauth_module, "TOKEN_FILE", nested):
            oauth_module._save_token({"access_token": "x"})
            assert nested.exists()

    def test_load_returns_none_when_file_missing(self, tmp_path):
        with _patch_token_file(tmp_path):
            result = load_token()
        assert result is None

    def test_load_returns_none_for_corrupt_json(self, tmp_path):
        token_file = tmp_path / "openai-oauth.json"
        token_file.write_text("{ not valid json }", encoding="utf-8")
        with patch.object(oauth_module, "TOKEN_FILE", token_file):
            result = load_token()
        assert result is None

    def test_save_is_atomic_via_tmp_file(self, tmp_path):
        """The tmp file must not exist after a successful save."""
        data = {"access_token": "z"}
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            tmp = (tmp_path / "openai-oauth.json").with_suffix(".tmp")
        assert not tmp.exists()

    def test_load_returns_all_fields(self, tmp_path):
        data = {
            "provider": "openai-oauth",
            "client_id": "cid",
            "access_token": "at",
            "refresh_token": "rt",
            "expires_at": 123456,
        }
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            result = load_token()
        assert result == data


# ---------------------------------------------------------------------------
# 6. refresh_token_if_needed
# ---------------------------------------------------------------------------


class TestRefreshTokenIfNeeded:
    def test_returns_access_token_when_not_expired(self, tmp_path):
        future = int(time.time()) + 7200
        data = {
            "access_token": "valid-token",
            "refresh_token": "rt",
            "expires_at": future,
            "client_id": "cid",
        }
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            result = refresh_token_if_needed("cid")
        assert result == "valid-token"

    def test_returns_none_when_no_token_stored(self, tmp_path):
        with _patch_token_file(tmp_path):
            result = refresh_token_if_needed()
        assert result is None

    def test_refreshes_when_token_expired(self, tmp_path):
        expired = int(time.time()) - 10
        data = {
            "access_token": "old-token",
            "refresh_token": "my-refresh",
            "expires_at": expired,
            "client_id": "cid",
        }
        new_token_resp = {
            "access_token": "new-token",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            with patch.object(oauth_module, "_do_refresh", return_value=new_token_resp):
                result = refresh_token_if_needed("cid")
        assert result == "new-token"

    def test_saves_refreshed_token(self, tmp_path):
        expired = int(time.time()) - 10
        data = {
            "access_token": "old",
            "refresh_token": "rt",
            "expires_at": expired,
            "client_id": "cid",
        }
        new_resp = {"access_token": "fresh", "expires_in": 3600}
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            with patch.object(oauth_module, "_do_refresh", return_value=new_resp):
                refresh_token_if_needed("cid")
            saved = load_token()
        assert saved["access_token"] == "fresh"

    def test_returns_stale_token_when_refresh_raises(self, tmp_path):
        expired = int(time.time()) - 10
        data = {
            "access_token": "stale",
            "refresh_token": "rt",
            "expires_at": expired,
            "client_id": "cid",
        }
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            with (
                patch.object(oauth_module, "_do_refresh", side_effect=RuntimeError("fail")),
                patch.object(oauth_module, "console"),
            ):
                result = refresh_token_if_needed("cid")
        assert result == "stale"

    def test_uses_client_id_from_stored_token_when_none_passed(self, tmp_path):
        future = int(time.time()) + 7200
        data = {
            "access_token": "tok",
            "refresh_token": "rt",
            "expires_at": future,
            "client_id": "stored-cid",
        }
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            result = refresh_token_if_needed(None)
        assert result == "tok"

    def test_token_within_margin_triggers_refresh(self, tmp_path):
        # expires_at is within the REFRESH_MARGIN_SECONDS window
        near_expiry = int(time.time()) + REFRESH_MARGIN_SECONDS - 60
        data = {
            "access_token": "near-expiry",
            "refresh_token": "rt",
            "expires_at": near_expiry,
            "client_id": "cid",
        }
        new_resp = {"access_token": "refreshed", "expires_in": 3600}
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            with patch.object(oauth_module, "_do_refresh", return_value=new_resp):
                result = refresh_token_if_needed("cid")
        assert result == "refreshed"

    def test_preserves_old_refresh_token_when_new_response_omits_it(self, tmp_path):
        expired = int(time.time()) - 10
        data = {
            "access_token": "old",
            "refresh_token": "keep-this",
            "expires_at": expired,
            "client_id": "cid",
        }
        # New response deliberately omits refresh_token
        new_resp = {"access_token": "fresh", "expires_in": 3600}
        with _patch_token_file(tmp_path):
            oauth_module._save_token(data)
            with patch.object(oauth_module, "_do_refresh", return_value=new_resp):
                refresh_token_if_needed("cid")
            saved = load_token()
        assert saved["refresh_token"] == "keep-this"


# ---------------------------------------------------------------------------
# 7. Code extraction from redirect URL
# ---------------------------------------------------------------------------


class TestExtractCodeFromUrl:
    def test_extracts_code_from_full_redirect_url(self):
        url = f"{REDIRECT_URI}?code=abc123&state=xyz"
        assert _extract_code_from_url(url) == "abc123"

    def test_returns_bare_code_when_no_query_string(self):
        assert _extract_code_from_url("barecode") == "barecode"

    def test_returns_none_for_empty_string(self):
        assert _extract_code_from_url("") is None

    def test_returns_none_for_whitespace_only(self):
        assert _extract_code_from_url("   ") is None

    def test_returns_none_when_url_has_no_code_param(self):
        url = f"{REDIRECT_URI}?state=abc&error=denied"
        assert _extract_code_from_url(url) is None

    def test_strips_leading_and_trailing_whitespace(self):
        url = f"  {REDIRECT_URI}?code=trimmed  "
        assert _extract_code_from_url(url) == "trimmed"

    def test_handles_multiple_params(self):
        url = f"{REDIRECT_URI}?state=s&code=mycode&session_state=ss"
        assert _extract_code_from_url(url) == "mycode"

    def test_returns_first_code_value(self):
        # parse_qs returns a list; we take [0]
        url = f"{REDIRECT_URI}?code=first&code=second"
        result = _extract_code_from_url(url)
        assert result == "first"


# ---------------------------------------------------------------------------
# 8. Token exchange (_exchange_code) — mock httpx
# ---------------------------------------------------------------------------


class TestExchangeCode:
    def _make_response(self, status: int, body: dict) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body
        resp.text = json.dumps(body)
        return resp

    def test_success_returns_token_dict(self):
        from missy.cli.oauth import _exchange_code

        expected = {"access_token": "at", "refresh_token": "rt", "expires_in": 3600}
        resp = self._make_response(200, expected)

        with patch("httpx.post", return_value=resp) as mock_post:
            result = _exchange_code("cid", "code123", "verifier456")

        assert result == expected
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        # Verify we posted to the correct endpoint
        assert call_kwargs.args[0] == TOKEN_URL

    def test_posts_correct_payload_fields(self):
        from missy.cli.oauth import _exchange_code

        resp = self._make_response(200, {"access_token": "tok"})
        with patch("httpx.post", return_value=resp) as mock_post:
            _exchange_code("my-cid", "auth-code", "pkce-verifier")

        _, kwargs = mock_post.call_args
        data = kwargs["data"]
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "auth-code"
        assert data["code_verifier"] == "pkce-verifier"
        assert data["client_id"] == "my-cid"
        assert data["redirect_uri"] == REDIRECT_URI

    def test_raises_runtime_error_on_non_200(self):
        from missy.cli.oauth import _exchange_code

        resp = self._make_response(400, {"error": "invalid_grant"})
        with patch("httpx.post", return_value=resp):
            with pytest.raises(RuntimeError, match="Token exchange failed"):
                _exchange_code("cid", "bad-code", "verifier")

    def test_error_message_includes_status_code(self):
        from missy.cli.oauth import _exchange_code

        resp = self._make_response(401, {})
        resp.text = "Unauthorized"
        with patch("httpx.post", return_value=resp):
            with pytest.raises(RuntimeError, match="401"):
                _exchange_code("cid", "code", "verifier")

    def test_uses_30_second_timeout(self):
        from missy.cli.oauth import _exchange_code

        resp = self._make_response(200, {"access_token": "t"})
        with patch("httpx.post", return_value=resp) as mock_post:
            _exchange_code("cid", "c", "v")

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# 9. Token refresh (_do_refresh) — mock httpx
# ---------------------------------------------------------------------------


class TestDoRefresh:
    def _make_response(self, status: int, body: dict) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body
        resp.text = json.dumps(body)
        return resp

    def test_success_returns_token_dict(self):
        from missy.cli.oauth import _do_refresh

        expected = {"access_token": "new-at", "expires_in": 3600}
        resp = self._make_response(200, expected)

        with patch("httpx.post", return_value=resp):
            result = _do_refresh("cid", "refresh-tok")

        assert result == expected

    def test_posts_correct_grant_type(self):
        from missy.cli.oauth import _do_refresh

        resp = self._make_response(200, {"access_token": "t"})
        with patch("httpx.post", return_value=resp) as mock_post:
            _do_refresh("cid", "my-refresh")

        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "my-refresh"
        assert kwargs["data"]["client_id"] == "cid"

    def test_raises_on_non_200(self):
        from missy.cli.oauth import _do_refresh

        resp = self._make_response(403, {"error": "forbidden"})
        with patch("httpx.post", return_value=resp):
            with pytest.raises(RuntimeError, match="Token refresh failed"):
                _do_refresh("cid", "rt")

    def test_posts_to_token_url(self):
        from missy.cli.oauth import _do_refresh

        resp = self._make_response(200, {"access_token": "t"})
        with patch("httpx.post", return_value=resp) as mock_post:
            _do_refresh("cid", "rt")

        assert mock_post.call_args.args[0] == TOKEN_URL

    def test_uses_30_second_timeout(self):
        from missy.cli.oauth import _do_refresh

        resp = self._make_response(200, {"access_token": "t"})
        with patch("httpx.post", return_value=resp) as mock_post:
            _do_refresh("cid", "rt")

        _, kwargs = mock_post.call_args
        assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# 10. _CallbackHandler GET handling
# ---------------------------------------------------------------------------


class TestCallbackHandler:
    """Tests for _CallbackHandler.do_GET, isolating the handler from a live server."""

    def _make_handler(self, path: str) -> oauth_module._CallbackHandler:
        """Build a _CallbackHandler instance wired to a BytesIO response buffer."""
        handler = oauth_module._CallbackHandler.__new__(oauth_module._CallbackHandler)
        handler.path = path

        buf = BytesIO()
        handler.wfile = buf
        handler.rfile = BytesIO()
        handler.server = MagicMock()

        # Collect headers into a dict for inspection
        handler._sent_headers: dict[str, str] = {}
        handler._response_code: list[int] = []

        def fake_send_response(code: int) -> None:
            handler._response_code.append(code)

        def fake_send_header(key: str, val: str) -> None:
            handler._sent_headers[key] = val

        def fake_end_headers() -> None:
            pass

        handler.send_response = fake_send_response
        handler.send_header = fake_send_header
        handler.end_headers = fake_end_headers

        return handler

    def setup_method(self):
        # Reset module-level state before each test.
        oauth_module._callback_result.clear()
        oauth_module._callback_event.clear()

    def test_code_captured_from_query_string(self):
        handler = self._make_handler("/auth/callback?code=abc&state=xyz")
        handler.do_GET()
        assert oauth_module._callback_result["code"] == "abc"
        assert oauth_module._callback_result["state"] == "xyz"

    def test_event_set_after_get(self):
        handler = self._make_handler("/auth/callback?code=abc&state=xyz")
        handler.do_GET()
        assert oauth_module._callback_event.is_set()

    def test_error_captured_from_query_string(self):
        handler = self._make_handler("/auth/callback?error=access_denied&state=xyz")
        handler.do_GET()
        assert oauth_module._callback_result["error"] == "access_denied"

    def test_code_is_none_when_missing(self):
        handler = self._make_handler("/auth/callback?state=xyz")
        handler.do_GET()
        assert oauth_module._callback_result["code"] is None

    def test_sends_200_response(self):
        handler = self._make_handler("/auth/callback?code=x&state=y")
        handler.do_GET()
        assert handler._response_code == [200]

    def test_response_content_type_is_html(self):
        handler = self._make_handler("/auth/callback?code=x&state=y")
        handler.do_GET()
        assert handler._sent_headers.get("Content-Type") == "text/html"

    def test_response_body_written_to_wfile(self):
        handler = self._make_handler("/auth/callback?code=x&state=y")
        handler.do_GET()
        body = handler.wfile.getvalue()
        assert b"Missy" in body

    def test_log_message_is_silent(self):
        handler = oauth_module._CallbackHandler.__new__(oauth_module._CallbackHandler)
        # Must not raise and must return None
        result = handler.log_message("GET /path HTTP/1.1", "200", "OK")
        assert result is None
