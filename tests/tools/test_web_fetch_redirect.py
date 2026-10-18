"""web_fetch never follows redirects, but must say where one points (WB-001)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from missy.tools.builtin.web_fetch import WebFetchTool


def _fetch(status: int, headers: dict[str, str]):
    response = SimpleNamespace(status_code=status, text="", headers=headers, truncated=False)
    client = MagicMock()
    client.get_capped.return_value = response
    with patch("missy.gateway.client.create_client", return_value=client):
        return WebFetchTool().execute(url="http://127.0.0.1:8000/start")


def test_redirect_error_names_the_resolved_target() -> None:
    result = _fetch(303, {"location": "/page.html"})
    assert not result.success
    assert "HTTP 303 redirect to http://127.0.0.1:8000/page.html" in result.error
    assert "not followed automatically" in result.error


def test_non_redirect_error_is_unchanged() -> None:
    result = _fetch(404, {})
    assert result.error == "HTTP 404"


def test_redirect_without_location_keeps_bare_status() -> None:
    result = _fetch(302, {})
    assert result.error == "HTTP 302"
