"""OpenAI OAuth 2.0 PKCE callback flow for Missy.

Implements the Codex / ChatGPT OAuth flow:

1. Generate PKCE code_verifier + code_challenge (S256).
2. Open browser to ``https://auth.openai.com/oauth/authorize``.
3. Listen on ``127.0.0.1:1455`` for the redirect callback.
4. Fall back to manual URL paste for headless/remote environments.
5. Exchange the authorization code for access + refresh tokens.
6. Parse the JWT access token to extract ``account_id`` and email.
7. Persist tokens to ``~/.missy/secrets/openai-oauth.json``.
8. Expose :func:`load_token` and :func:`refresh_token_if_needed` for
   runtime use by the OpenAI provider.

NOTE: OpenClaw's implementation requests only identity scopes
(``openid profile email offline_access``), which causes 403 errors on
all model calls because ``model.request`` is missing.  This
implementation requests the full required scope set.

The OAuth client ID is read from the environment variable
``OPENAI_OAUTH_CLIENT_ID``.  Users who wish to use their own registered
OAuth application can set this variable before running ``missy setup``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_BASE = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE}/oauth/token"
REDIRECT_URI = "http://127.0.0.1:1455/auth/callback"
CALLBACK_PORT = 1455

# Scopes required for model inference (fixes the known OpenClaw bug where
# only identity scopes were requested, resulting in 403 on all API calls).
SCOPES = "openid profile email offline_access model.request api.responses.write"

# OAuth client ID — override via OPENAI_OAUTH_CLIENT_ID env var.
# This is the client ID registered with OpenAI for your OAuth application.
DEFAULT_CLIENT_ID = os.environ.get("OPENAI_OAUTH_CLIENT_ID", "")

TOKEN_FILE = Path("~/.missy/secrets/openai-oauth.json").expanduser()

# Refresh tokens 5 minutes before expiry.
REFRESH_MARGIN_SECONDS = 300

# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for PKCE S256.

    The verifier is 96 random bytes encoded as base64url (128 chars).
    The challenge is SHA-256(verifier) encoded as base64url.
    """
    verifier_bytes = secrets.token_bytes(96)
    verifier = base64.urlsafe_b64encode(verifier_bytes).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _build_auth_url(client_id: str, state: str, code_challenge: str) -> str:
    """Build the authorization URL with PKCE parameters."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params)


# ---------------------------------------------------------------------------
# Local callback server
# ---------------------------------------------------------------------------

_callback_result: dict = {}
_callback_event = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures the OAuth callback parameters."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _callback_result["code"] = params.get("code", [None])[0]
        _callback_result["state"] = params.get("state", [None])[0]
        _callback_result["error"] = params.get("error", [None])[0]

        body = (
            b"<html><body><h2>Missy authentication complete.</h2>"
            b"<p>You may close this tab.</p></body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        _callback_event.set()

    def log_message(self, *_args) -> None:  # silence access log
        pass


def _start_callback_server() -> Optional[HTTPServer]:
    """Start the local callback HTTP server in a daemon thread.

    Returns the server on success, or None if the port is in use.
    """
    try:
        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _CallbackHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
    except OSError:
        return None


def _wait_for_callback(server: HTTPServer, timeout: int = 120) -> Optional[str]:
    """Block until the callback arrives or timeout expires.

    Returns the authorization code, or None on timeout/error.
    """
    got_code = _callback_event.wait(timeout=timeout)
    server.shutdown()
    if not got_code:
        return None
    if _callback_result.get("error"):
        console.print(f"  [red]OAuth error:[/] {_callback_result['error']}")
        return None
    return _callback_result.get("code")


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------


def _exchange_code(client_id: str, code: str, verifier: str) -> dict:
    """POST the authorization code to the token endpoint.

    Returns the raw token response dict.

    Raises:
        RuntimeError: If the exchange fails or the HTTP response is not 200.
    """
    import httpx

    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": verifier,
        "client_id": client_id,
    }
    resp = httpx.post(TOKEN_URL, data=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Token exchange failed: HTTP {resp.status_code} — {resp.text[:200]}"
        )
    return resp.json()


def _do_refresh(client_id: str, refresh_token: str) -> dict:
    """Exchange a refresh token for a new access token.

    Returns the raw token response dict.
    """
    import httpx

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    resp = httpx.post(TOKEN_URL, data=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Token refresh failed: HTTP {resp.status_code} — {resp.text[:200]}"
        )
    return resp.json()


# ---------------------------------------------------------------------------
# JWT parsing
# ---------------------------------------------------------------------------


def _parse_jwt_payload(token: str) -> dict:
    """Decode the JWT payload (no verification — for metadata extraction only).

    Returns the claims dict, or an empty dict on parse error.
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        padding = 4 - len(parts[1]) % 4
        padded = parts[1] + "=" * padding
        payload_bytes = base64.urlsafe_b64decode(padded)
        return json.loads(payload_bytes)
    except Exception:
        return {}


def _extract_account_metadata(access_token: str) -> tuple[str, str]:
    """Extract ``(account_id, email)`` from the JWT access token.

    Both values default to empty string if not present.
    """
    claims = _parse_jwt_payload(access_token)
    # OpenAI stores account ID under a namespaced claim.
    oa_ns = claims.get("https://api.openai.com/auth", {})
    account_id = oa_ns.get("chatgpt_account_id", "") or claims.get("sub", "")
    email = claims.get("email", "")
    return account_id, email


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def _save_token(data: dict) -> None:
    """Write token data to TOKEN_FILE atomically (mode 0o600)."""
    TOKEN_FILE.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    tmp = TOKEN_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(TOKEN_FILE)


def load_token() -> Optional[dict]:
    """Load the stored OAuth token dict, or None if not present."""
    if not TOKEN_FILE.exists():
        return None
    try:
        return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh_token_if_needed(client_id: Optional[str] = None) -> Optional[str]:
    """Return a valid access token, refreshing if within the expiry margin.

    Returns the access token string, or None if no token is stored or
    refresh fails.
    """
    token_data = load_token()
    if not token_data:
        return None

    cid = client_id or token_data.get("client_id") or DEFAULT_CLIENT_ID
    expires_at = token_data.get("expires_at", 0)

    if time.time() < expires_at - REFRESH_MARGIN_SECONDS:
        return token_data["access_token"]

    # Token expired or about to expire — refresh.
    refresh_tok = token_data.get("refresh_token")
    if not refresh_tok or not cid:
        return token_data.get("access_token")  # return stale; let caller handle 401

    try:
        new_data = _do_refresh(cid, refresh_tok)
        expires_at = int(time.time()) + int(new_data.get("expires_in", 3600))
        token_data.update(
            {
                "access_token": new_data["access_token"],
                "refresh_token": new_data.get("refresh_token", refresh_tok),
                "expires_at": expires_at,
            }
        )
        _save_token(token_data)
        return token_data["access_token"]
    except Exception as exc:
        console.print(f"  [yellow]Token refresh failed:[/] {exc}")
        return token_data.get("access_token")


# ---------------------------------------------------------------------------
# Manual paste fallback
# ---------------------------------------------------------------------------


def _extract_code_from_url(raw: str) -> Optional[str]:
    """Extract the ``code`` parameter from a full redirect URL string."""
    raw = raw.strip()
    if "?" not in raw:
        # User may have pasted just the code itself.
        return raw or None
    parsed = urllib.parse.urlparse(raw)
    params = urllib.parse.parse_qs(parsed.query)
    return params.get("code", [None])[0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_openai_oauth(client_id: Optional[str] = None) -> Optional[str]:
    """Run the full PKCE OAuth flow and return the access token on success.

    Steps:
    1. Validate client_id.
    2. Generate PKCE pair + state.
    3. Attempt local callback server; fall back to manual paste.
    4. Open browser to authorization URL.
    5. Exchange code for tokens.
    6. Persist and return access token.

    Returns:
        The OAuth access token string, or ``None`` if the flow was aborted.
    """
    cid = client_id or DEFAULT_CLIENT_ID
    if not cid:
        console.print(
            "  [red]No OAuth client ID configured.[/]\n"
            "  Set [bold]OPENAI_OAUTH_CLIENT_ID[/] in your environment, or\n"
            "  register an OAuth app at platform.openai.com and provide the client ID."
        )
        cid = click.prompt("  OpenAI OAuth client ID", default="").strip()
        if not cid:
            return None

    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)
    auth_url = _build_auth_url(cid, state, challenge)

    # Try to start local callback server.
    _callback_event.clear()
    _callback_result.clear()
    server = _start_callback_server()
    use_callback = server is not None

    if use_callback:
        console.print(
            "  [dim]Local callback server started on port 1455.[/]\n"
            "  Opening browser to OpenAI authorization page…"
        )
    else:
        console.print(
            "  [yellow]Port 1455 is in use — falling back to manual URL paste.[/]\n"
            "  (For remote/VPS: run [bold]ssh -L 1455:localhost:1455 user@host[/] first.)"
        )
        use_callback = False

    # Open browser.
    opened = webbrowser.open(auth_url)
    if not opened or not use_callback:
        console.print(f"\n  Open this URL in your browser:\n  [cyan]{auth_url}[/]\n")

    # Get authorization code.
    code: Optional[str] = None

    if use_callback:
        console.print("  [dim]Waiting up to 120 s for browser callback…[/]")
        code = _wait_for_callback(server, timeout=120)
        if not code:
            console.print("  [yellow]Callback timed out. Falling back to manual paste.[/]")

    if not code:
        raw = click.prompt(
            "  Paste the full redirect URL (or just the 'code' value) from your browser",
            default="",
        )
        code = _extract_code_from_url(raw)

    if not code:
        console.print("  [red]No authorization code received. OAuth flow aborted.[/]")
        return None

    # Exchange code for tokens.
    console.print("  [dim]Exchanging authorization code for tokens…[/]")
    try:
        token_resp = _exchange_code(cid, code, verifier)
    except RuntimeError as exc:
        console.print(f"  [red]Token exchange failed:[/] {exc}")
        return None

    access_token = token_resp.get("access_token")
    if not access_token:
        console.print("  [red]Token response missing access_token.[/]")
        return None

    refresh_tok = token_resp.get("refresh_token", "")
    expires_in = int(token_resp.get("expires_in", 3600))
    expires_at = int(time.time()) + expires_in

    account_id, email = _extract_account_metadata(access_token)

    token_data = {
        "provider": "openai-oauth",
        "client_id": cid,
        "access_token": access_token,
        "refresh_token": refresh_tok,
        "expires_at": expires_at,
        "token_type": token_resp.get("token_type", "Bearer"),
        "account_id": account_id,
        "email": email,
    }
    _save_token(token_data)

    id_display = email or account_id or "(unknown)"
    console.print(f"  [green]OAuth complete.[/] Signed in as [bold]{id_display}[/]")
    console.print(f"  [dim]Token stored at {TOKEN_FILE}[/]")

    return access_token
