"""Tests for the Web TUI's memory browse/edit and provider config endpoints.

Covers the JSON API surface added for the multipage operator console:

- ``GET /api/v1/memory/recent`` (query-less browsing with pagination)
- ``GET /api/v1/memory/sessions``
- ``PUT /api/v1/memory/turns/{id}`` (operator memory edit + FTS reindex)
- ``GET /api/v1/providers/{name}`` (redacted config detail)
- ``provider.enable`` / ``provider.disable`` operator controls
"""

from __future__ import annotations

import socket
import time
from collections.abc import Generator

import httpx
import pytest

from missy.api.server import ApiConfig, ApiServer
from missy.config.settings import ProviderConfig
from missy.memory.sqlite_store import ConversationTurn, SQLiteMemoryStore
from missy.providers.registry import ProviderRegistry

API_KEY = "test-api-key-webui"
HEADERS = {"X-API-Key": API_KEY}
SECRET_KEY_MATERIAL = "sk-super-secret-value-123"


class FakeProvider:
    name = "fake"

    def __init__(self, available: bool = True):
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=0.2)
            return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Server did not start within {timeout}s: {url}")


@pytest.fixture
def memory_store(tmp_path) -> SQLiteMemoryStore:
    store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    for index in range(3):
        turn = ConversationTurn.new("sess-1", "user", f"first-session message {index}")
        turn.timestamp = f"2026-07-17T00:00:{index:02d}+00:00"
        store.add_turn(turn)
    other = ConversationTurn.new("sess-2", "assistant", "second-session reply")
    other.timestamp = "2026-07-17T00:01:00+00:00"
    store.add_turn(other)
    store.register_session("sess-1", name="first session")
    store.register_session("sess-2", name="second session")
    return store


@pytest.fixture
def provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(
        "alpha",
        FakeProvider(),
        config=ProviderConfig(name="alpha", model="model-a", api_key=SECRET_KEY_MATERIAL),
    )
    registry.register("beta", FakeProvider(), config=ProviderConfig(name="beta", model="model-b"))
    registry.set_default("alpha")
    return registry


@pytest.fixture
def server(memory_store, provider_registry) -> Generator[ApiServer, None, None]:
    port = _free_port()
    cfg = ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY, rate_limit_rpm=500)
    srv = ApiServer(
        config=cfg,
        memory_store=memory_store,
        provider_registry=provider_registry,
    )
    srv.start()
    _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
    yield srv
    srv.stop()


@pytest.fixture
def client(server) -> Generator[httpx.Client, None, None]:
    with httpx.Client(
        base_url=f"http://127.0.0.1:{server.config.port}/api/v1",
        headers=HEADERS,
        timeout=5.0,
    ) as c:
        yield c


class TestMemoryRecent:
    def test_returns_newest_first_with_total(self, client):
        resp = client.get("/memory/recent").json()
        assert resp["data"]["total"] == 4
        results = resp["data"]["results"]
        assert results[0]["content"] == "second-session reply"

    def test_pagination(self, client):
        page = client.get("/memory/recent", params={"limit": 2, "offset": 2}).json()["data"]
        assert page["total"] == 4
        assert len(page["results"]) == 2
        assert page["has_more"] is False

    def test_session_filter(self, client):
        page = client.get("/memory/recent", params={"session_id": "sess-2"}).json()["data"]
        assert page["total"] == 1
        assert page["results"][0]["session_id"] == "sess-2"

    def test_requires_auth(self, server):
        resp = httpx.get(f"http://127.0.0.1:{server.config.port}/api/v1/memory/recent", timeout=5.0)
        assert resp.status_code == 401

    def test_no_store_returns_empty(self):
        port = _free_port()
        srv = ApiServer(config=ApiConfig(host="127.0.0.1", port=port, api_key=API_KEY))
        srv.start()
        _wait_for_server(f"http://127.0.0.1:{port}/api/v1/health")
        try:
            resp = httpx.get(
                f"http://127.0.0.1:{port}/api/v1/memory/recent", headers=HEADERS, timeout=5.0
            ).json()
            assert resp["data"] == {"results": [], "total": 0, "has_more": False}
        finally:
            srv.stop()


class TestMemorySessions:
    def test_lists_recorded_sessions(self, client):
        resp = client.get("/memory/sessions").json()
        session_ids = {s["session_id"] for s in resp["data"]["sessions"]}
        assert session_ids == {"sess-1", "sess-2"}


class TestUpdateMemoryTurn:
    def test_edit_updates_content_and_marks_edited(self, client):
        turn_id = client.get("/memory/recent").json()["data"]["results"][0]["id"]
        resp = client.put(f"/memory/turns/{turn_id}", json={"content": "revised by operator"})
        assert resp.status_code == 200
        latest = client.get("/memory/recent").json()["data"]["results"][0]
        assert latest["content"] == "revised by operator"
        assert latest["edited_at"]

    def test_edit_reindexes_search(self, client):
        turn_id = client.get("/memory/recent").json()["data"]["results"][0]["id"]
        client.put(f"/memory/turns/{turn_id}", json={"content": "replacement-marker text"})
        stale = client.get("/memory/search", params={"q": "second-session reply"}).json()
        assert stale["data"]["results"] == []
        fresh = client.get("/memory/search", params={"q": "replacement-marker"}).json()
        assert [t["id"] for t in fresh["data"]["results"]] == [turn_id]

    def test_unknown_turn_returns_404(self, client):
        resp = client.put("/memory/turns/ghost", json={"content": "text"})
        assert resp.status_code == 404

    def test_empty_content_returns_400(self, client):
        turn_id = client.get("/memory/recent").json()["data"]["results"][0]["id"]
        assert client.put(f"/memory/turns/{turn_id}", json={"content": "  "}).status_code == 400
        assert client.put(f"/memory/turns/{turn_id}", json={}).status_code == 400

    def test_requires_auth(self, server):
        resp = httpx.put(
            f"http://127.0.0.1:{server.config.port}/api/v1/memory/turns/x",
            json={"content": "text"},
            timeout=5.0,
        )
        assert resp.status_code == 401


class TestProviderDetail:
    def test_detail_reports_config_without_key_material(self, client):
        resp = client.get("/providers/alpha")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["config"]["model"] == "model-a"
        assert data["config"]["api_key_configured"] is True
        assert data["is_default"] is True
        # The actual key must never appear anywhere in the response.
        assert SECRET_KEY_MATERIAL not in resp.text

    def test_unknown_provider_returns_404(self, client):
        assert client.get("/providers/ghost").status_code == 404

    def test_list_includes_enabled_and_model(self, client):
        providers = client.get("/providers").json()["data"]["providers"]
        by_name = {p["name"]: p for p in providers}
        assert by_name["alpha"]["enabled"] is True
        assert by_name["alpha"]["model"] == "model-a"
        assert by_name["beta"]["model"] == "model-b"


class TestProviderToggleControls:
    def test_disable_requires_confirmation(self, client):
        resp = client.post("/controls/provider.disable", json={"target": "beta"})
        assert resp.status_code == 409
        assert "confirmation" in resp.text.lower()

    def test_disable_and_reenable_roundtrip(self, client, provider_registry):
        resp = client.post(
            "/controls/provider.disable",
            json={"target": "beta", "confirm": "disable-provider:beta"},
        )
        assert resp.status_code == 200
        assert provider_registry.is_enabled("beta") is False
        listed = client.get("/providers").json()["data"]["providers"]
        assert next(p for p in listed if p["name"] == "beta")["enabled"] is False

        again = client.post(
            "/controls/provider.disable",
            json={"target": "beta", "confirm": "disable-provider:beta"},
        )
        assert again.status_code == 409  # already disabled

        resp = client.post(
            "/controls/provider.enable",
            json={"target": "beta", "confirm": "enable-provider:beta"},
        )
        assert resp.status_code == 200
        assert provider_registry.is_enabled("beta") is True

    def test_cannot_disable_default_provider(self, client, provider_registry):
        resp = client.post(
            "/controls/provider.disable",
            json={"target": "alpha", "confirm": "disable-provider:alpha"},
        )
        assert resp.status_code == 409
        assert provider_registry.is_enabled("alpha") is True

    def test_cannot_set_disabled_provider_default(self, client):
        client.post(
            "/controls/provider.disable",
            json={"target": "beta", "confirm": "disable-provider:beta"},
        )
        resp = client.post(
            "/controls/provider.set_default",
            json={"target": "beta", "confirm": "set-default:beta"},
        )
        assert resp.status_code == 409

    def test_unknown_provider_returns_404(self, client):
        resp = client.post(
            "/controls/provider.disable",
            json={"target": "ghost", "confirm": "disable-provider:ghost"},
        )
        assert resp.status_code == 404

    def test_controls_list_includes_toggles_with_targets(self, client):
        controls = {c["id"]: c for c in client.get("/controls").json()["data"]["controls"]}
        assert "provider.enable" in controls
        assert "provider.disable" in controls
        disable_targets = {t["name"]: t for t in controls["provider.disable"]["targets"]}
        # The default provider is never offered as a disable target.
        assert disable_targets["alpha"]["available"] is False
        assert disable_targets["beta"]["available"] is True


class TestLogsTail:
    """F18 — GET /api/v1/logs/tail (redacted application-log tail)."""

    def test_returns_tail_lines(self, client, tmp_path, monkeypatch):
        log = tmp_path / "app.log"
        log.write_text("\n".join(f"line {i}" for i in range(50)) + "\n")
        monkeypatch.setenv("MISSY_APP_LOG", str(log))
        data = client.get("/logs/tail", params={"lines": 10}).json()["data"]
        assert data["path"] == str(log)
        assert len(data["lines"]) == 10
        assert data["lines"][-1] == "line 49"

    def test_lines_capped(self, client, tmp_path, monkeypatch):
        log = tmp_path / "app.log"
        log.write_text("\n".join(f"l{i}" for i in range(20)) + "\n")
        monkeypatch.setenv("MISSY_APP_LOG", str(log))
        # Requesting more lines than exist just returns what's there.
        data = client.get("/logs/tail", params={"lines": 5000}).json()["data"]
        assert len(data["lines"]) == 20

    def test_redacts_secrets(self, client, tmp_path, monkeypatch):
        log = tmp_path / "app.log"
        # A GitHub PAT-shaped token must be censored, not returned verbatim.
        secret = "ghp_" + "a" * 36
        log.write_text(f"INFO connecting with token {secret}\nINFO done\n")
        monkeypatch.setenv("MISSY_APP_LOG", str(log))
        data = client.get("/logs/tail").json()["data"]
        joined = "\n".join(data["lines"])
        assert secret not in joined

    def test_missing_log_returns_empty(self, client, tmp_path, monkeypatch):
        monkeypatch.setenv("MISSY_APP_LOG", str(tmp_path / "does-not-exist.log"))
        data = client.get("/logs/tail").json()["data"]
        assert data["path"] is None
        assert data["lines"] == []

    def test_requires_auth(self, server, tmp_path, monkeypatch):
        monkeypatch.setenv("MISSY_APP_LOG", str(tmp_path / "app.log"))
        resp = httpx.get(f"http://127.0.0.1:{server.config.port}/api/v1/logs/tail", timeout=5.0)
        assert resp.status_code == 401
