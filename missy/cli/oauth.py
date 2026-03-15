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

Uses the same client ID and parameters as OpenClaw's ``@mariozechner/pi-ai``
package (``app_EMoamEEZ73f0CkXaXp7hrann``) so the flow is accepted by
OpenAI's auth server.  Override with ``OPENAI_OAUTH_CLIENT_ID`` env var
to use your own registered OAuth application.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import click
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_BASE = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
CALLBACK_PORT = 1455

# Scopes — matches the @mariozechner/pi-ai package used by OpenClaw exactly.
SCOPES = "openid profile email offline_access"

# Client ID from @mariozechner/pi-ai v0.57.1 (openclaw's OAuth dependency).
# Override via OPENAI_OAUTH_CLIENT_ID env var to use your own registered app.
DEFAULT_CLIENT_ID = os.environ.get("OPENAI_OAUTH_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")

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


def _start_callback_server() -> HTTPServer | None:
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


def _wait_for_callback(
    server: HTTPServer, timeout: int = 120, expected_state: str = "",
) -> str | None:
    """Block until the callback arrives or timeout expires.

    When *expected_state* is provided, the callback's ``state`` parameter
    is compared against it to prevent CSRF attacks.

    Returns the authorization code, or None on timeout/error/state mismatch.
    """
    got_code = _callback_event.wait(timeout=timeout)
    server.shutdown()
    if not got_code:
        return None
    if _callback_result.get("error"):
        console.print(f"  [red]OAuth error:[/] {_callback_result['error']}")
        return None
    # CSRF check: verify that the state parameter matches what we sent.
    if expected_state and _callback_result.get("state") != expected_state:
        console.print("  [red]OAuth state mismatch — possible CSRF attack. Aborting.[/]")
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
        raise RuntimeError(f"Token exchange failed: HTTP {resp.status_code} — {resp.text[:200]}")
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
        raise RuntimeError(f"Token refresh failed: HTTP {resp.status_code} — {resp.text[:200]}")
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
    except Exception as exc:
        logger.debug("JWT payload parsing failed: %s", exc)
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
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(TOKEN_FILE)


def load_token() -> dict | None:
    """Load the stored OAuth token dict, or None if not present."""
    if not TOKEN_FILE.exists():
        return None
    try:
        return json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def refresh_token_if_needed(client_id: str | None = None) -> str | None:
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


def _extract_code_from_url(raw: str) -> str | None:
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


def run_openai_oauth(client_id: str | None = None) -> str | None:
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

    # Start local callback server (best-effort).
    _callback_event.clear()
    _callback_result.clear()
    server = _start_callback_server()

    if server:
        console.print("  [dim]Callback server listening on localhost:1455.[/]")
    else:
        console.print("  [yellow]Port 1455 is in use — automatic capture unavailable.[/]")

    # Open browser and always print the URL.
    webbrowser.open(auth_url)
    console.print(
        f"\n  [bold]Authorization URL:[/]\n  [cyan]{auth_url}[/]\n\n"
        "  1. Complete sign-in in your browser.\n"
        "  2. After authorizing, your browser will redirect to localhost:1455.\n"
        "     If it shows a blank page or an error, that's fine — the code was captured.\n"
        "  3. If automatic capture doesn't work, paste the full redirect URL below.\n"
    )

    # Race: automatic callback vs. manual paste.
    # prompt runs in a thread so the callback server can win concurrently.
    code: str | None = None
    paste_result: list[str] = []
    paste_done = threading.Event()

    def _prompt_thread() -> None:
        try:
            raw = click.prompt(
                "  Paste redirect URL (or press Enter to wait for automatic capture)",
                default="",
                show_default=False,
            )
            paste_result.append(raw.strip())
        except Exception:
            paste_result.append("")
        finally:
            paste_done.set()

    prompt_t = threading.Thread(target=_prompt_thread, daemon=True)
    prompt_t.start()

    # Wait for whichever arrives first: callback or paste.
    if server:
        while not paste_done.is_set():
            if _callback_event.wait(timeout=0.5):
                break
        if _callback_event.is_set() and not _callback_result.get("error"):
            # Verify state to prevent CSRF
            if _callback_result.get("state") != state:
                console.print("  [red]OAuth state mismatch — possible CSRF attack.[/]")
            else:
                code = _callback_result.get("code")
                if code:
                    console.print("  [green]Automatic callback received.[/]")
            server.shutdown()

    # If automatic capture didn't get a code, use whatever was pasted.
    if not code:
        paste_done.wait(timeout=120)
        if server:
            server.shutdown()
        raw = paste_result[0] if paste_result else ""
        code = _extract_code_from_url(raw) if raw else None

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
