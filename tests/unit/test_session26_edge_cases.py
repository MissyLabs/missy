"""Edge case tests for session 26: gateway client, cost tracker, checkpoint,
webhook handler, and audit logger."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import (
    FilesystemPolicy,
    MissyConfig,
    NetworkPolicy,
    PluginPolicy,
    ShellPolicy,
)
from missy.core.events import AuditEvent, EventBus, event_bus
from missy.policy import engine as engine_module
from missy.policy.engine import init_policy_engine

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_config(*, default_deny: bool = False) -> MissyConfig:
    return MissyConfig(
        network=NetworkPolicy(
            default_deny=default_deny,
            allowed_cidrs=[],
            allowed_domains=[],
            allowed_hosts=[],
        ),
        filesystem=FilesystemPolicy(),
        shell=ShellPolicy(),
        plugins=PluginPolicy(),
        providers={},
        workspace_path="/tmp",
        audit_log_path="/tmp/audit.log",
    )


# ---------------------------------------------------------------------------
# 1. Gateway client edge cases
# ---------------------------------------------------------------------------


class TestGatewayClientEdgeCases:
    """Edge cases for PolicyHTTPClient._check_url and constructor validation."""

    @pytest.fixture(autouse=True)
    def permissive_engine(self):
        original = engine_module._engine
        init_policy_engine(_make_config(default_deny=False))
        event_bus.clear()
        yield
        engine_module._engine = original
        event_bus.clear()

    def test_url_with_special_characters_in_path_is_accepted(self):
        """URLs with percent-encoded characters in the path are valid."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        # Should not raise — the host is still parseable
        # We only need _check_url to pass; actual network call is not made
        # Patch the sync client so we never hit the network
        with patch.object(client, "_get_sync_client") as mock_sync:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {}
            mock_sync.return_value.get.return_value = mock_resp
            # _check_response_size reads .headers attribute
            mock_resp.headers = MagicMock()
            mock_resp.headers.get.return_value = None
            mock_resp.content = b"ok"
            url = "https://example.com/path%20with%20spaces?q=hello%26world"
            client.get(url)  # should not raise

    def test_empty_url_raises_value_error(self):
        """An empty string is not a valid URL."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        with pytest.raises(ValueError):
            client._check_url("")

    def test_url_exceeding_8192_chars_raises_value_error(self):
        """URLs longer than 8192 characters are rejected."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        long_url = "https://example.com/" + "a" * 8200
        assert len(long_url) > 8192
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(long_url)

    def test_url_exactly_8192_chars_is_accepted(self):
        """A URL of exactly 8192 characters is at the boundary and should pass length check."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        # Build a URL that is exactly 8192 characters long
        base = "https://example.com/"
        padding = "a" * (8192 - len(base))
        url = base + padding
        assert len(url) == 8192
        # _check_url will reach policy check; just verify no ValueError for length
        # (policy engine is permissive so it won't raise PolicyViolationError)
        client._check_url(url)  # must not raise ValueError about length

    def test_url_at_8193_chars_raises(self):
        """A URL of 8193 characters exceeds the limit."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        base = "https://example.com/"
        padding = "a" * (8193 - len(base))
        url = base + padding
        assert len(url) == 8193
        with pytest.raises(ValueError, match="exceeds maximum length"):
            client._check_url(url)

    def test_timeout_zero_raises_value_error(self):
        """A timeout of 0 is not positive and must be rejected."""
        from missy.gateway.client import PolicyHTTPClient

        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=0)

    def test_negative_timeout_raises_value_error(self):
        """A negative timeout is invalid."""
        from missy.gateway.client import PolicyHTTPClient

        with pytest.raises(ValueError, match="timeout must be positive"):
            PolicyHTTPClient(timeout=-5)

    def test_positive_timeout_is_accepted(self):
        """A timeout of 1 is valid."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient(timeout=1)
        assert client.timeout == 1

    def test_url_with_no_host_raises_value_error(self):
        """A URL without a host component raises ValueError."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Cannot determine host"):
            client._check_url("https:///no-host-path")

    def test_non_http_scheme_raises_value_error(self):
        """ftp:// and other non-http(s) schemes are rejected."""
        from missy.gateway.client import PolicyHTTPClient

        client = PolicyHTTPClient()
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            client._check_url("ftp://example.com/file")


# ---------------------------------------------------------------------------
# 2. Cost tracker edge cases
# ---------------------------------------------------------------------------


class TestCostTrackerEdgeCases:
    """Edge cases for CostTracker.record and check_budget."""

    def test_record_zero_tokens_returns_record_with_zero_cost(self):
        """Recording zero tokens for a priced model yields zero cost."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="claude-sonnet-4", prompt_tokens=0, completion_tokens=0)
        assert rec.cost_usd == 0.0
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0

    def test_record_zero_tokens_still_counts_as_a_call(self):
        """A zero-token recording increments call_count."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record(model="claude-sonnet-4", prompt_tokens=0, completion_tokens=0)
        assert tracker.call_count == 1

    def test_budget_enforcement_with_exact_match(self):
        """When total cost equals the budget exactly, BudgetExceededError is raised."""
        from missy.agent.cost_tracker import BudgetExceededError, CostTracker

        # claude-sonnet-4: input $0.003/1k, output $0.015/1k
        # 1000 prompt tokens = $0.003 exactly
        tracker = CostTracker(max_spend_usd=0.003)
        tracker.record(model="claude-sonnet-4", prompt_tokens=1000, completion_tokens=0)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_budget()
        assert exc_info.value.spent >= 0.003
        assert exc_info.value.limit == 0.003

    def test_budget_not_exceeded_when_just_below_limit(self):
        """When total cost is just below the limit, no error is raised."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=1.0)
        tracker.record(model="claude-sonnet-4", prompt_tokens=100, completion_tokens=0)
        tracker.check_budget()  # must not raise

    def test_multiple_rapid_cost_recordings_are_thread_safe(self):
        """Concurrent record() calls from multiple threads produce consistent totals."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        errors: list[Exception] = []

        def record_costs():
            try:
                for _ in range(50):
                    tracker.record(
                        model="claude-sonnet-4",
                        prompt_tokens=10,
                        completion_tokens=5,
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_costs) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        # 10 threads * 50 iterations = 500 calls
        assert tracker.call_count == 500
        assert tracker.total_prompt_tokens == 500 * 10
        assert tracker.total_completion_tokens == 500 * 5

    def test_no_budget_limit_never_raises(self):
        """With max_spend_usd=0 (unlimited), check_budget never raises."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker(max_spend_usd=0.0)
        tracker.record(model="claude-opus-4", prompt_tokens=10_000, completion_tokens=10_000)
        tracker.check_budget()  # must not raise

    def test_unknown_model_records_zero_cost(self):
        """An unknown model prefix falls back to zero-cost pricing."""
        from missy.agent.cost_tracker import CostTracker

        tracker = CostTracker()
        rec = tracker.record(model="totally-unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
        assert rec.cost_usd == 0.0


# ---------------------------------------------------------------------------
# 3. Checkpoint edge cases
# ---------------------------------------------------------------------------


class TestCheckpointEdgeCases:
    """Edge cases for CheckpointManager create/update/complete."""

    @pytest.fixture()
    def cm(self, tmp_path: Path):
        from missy.agent.checkpoint import CheckpointManager

        return CheckpointManager(db_path=str(tmp_path / "checkpoints.db"))

    def test_create_with_empty_prompt(self, cm):
        """A checkpoint may be created with an empty prompt string."""
        cid = cm.create(session_id="s1", task_id="t1", prompt="")
        assert cid  # UUID is returned
        incomplete = cm.get_incomplete()
        assert len(incomplete) == 1
        assert incomplete[0]["prompt"] == ""

    def test_create_with_very_long_session_id(self, cm):
        """A very long session ID (e.g. 10 000 chars) is stored without truncation."""
        long_id = "x" * 10_000
        cm.create(session_id=long_id, task_id="t1", prompt="test")
        incomplete = cm.get_incomplete()
        assert len(incomplete) == 1
        assert incomplete[0]["session_id"] == long_id

    def test_update_with_empty_messages_list(self, cm):
        """Updating a checkpoint with an empty messages list is valid."""
        cid = cm.create(session_id="s1", task_id="t1", prompt="hello")
        cm.update(cid, loop_messages=[], tool_names_used=[], iteration=0)
        rows = cm.get_incomplete()
        assert rows[0]["loop_messages"] == []

    def test_concurrent_checkpoint_creation(self, cm):
        """Multiple threads can create checkpoints concurrently without errors."""
        errors: list[Exception] = []
        created_ids: list[str] = []
        lock = threading.Lock()

        def create_checkpoint(i: int):
            try:
                cid = cm.create(
                    session_id=f"session-{i}",
                    task_id=f"task-{i}",
                    prompt=f"Prompt number {i}",
                )
                with lock:
                    created_ids.append(cid)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=create_checkpoint, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(created_ids) == 20
        # All IDs should be distinct UUIDs
        assert len(set(created_ids)) == 20

    def test_complete_removes_from_incomplete_list(self, cm):
        """After complete(), the checkpoint no longer appears in get_incomplete()."""
        cid = cm.create(session_id="s1", task_id="t1", prompt="test")
        assert len(cm.get_incomplete()) == 1
        cm.complete(cid)
        assert cm.get_incomplete() == []

    def test_classify_fresh_checkpoint_returns_resume(self, cm):
        """A checkpoint created seconds ago is classified as 'resume'."""
        cm.create(session_id="s1", task_id="t1", prompt="fresh")
        rows = cm.get_incomplete()
        action = cm.classify(rows[0])
        assert action == "resume"

    def test_classify_stale_checkpoint_returns_abandon(self, cm):
        """A checkpoint older than 24 hours is classified as 'abandon'."""
        cid = cm.create(session_id="s1", task_id="t1", prompt="stale")
        # Backdate the created_at to 25 hours ago
        conn = cm._connect()
        old_ts = time.time() - (25 * 3600)
        conn.execute(
            "UPDATE checkpoints SET created_at = ? WHERE id = ?",
            (old_ts, cid),
        )
        conn.commit()
        rows = cm.get_incomplete()
        assert len(rows) == 1
        action = cm.classify(rows[0])
        assert action == "abandon"


# ---------------------------------------------------------------------------
# 4. Webhook handler edge cases
# ---------------------------------------------------------------------------


class TestWebhookHandlerEdgeCases:
    """Test do_POST validation logic directly via the handler class."""

    @pytest.fixture()
    def channel_and_handler_class(self):
        """Return a (channel, HandlerClass) pair for handler instantiation."""
        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        # We need to extract the Handler class that is defined inside start().
        # Instead of spinning up a real server we replicate the relevant
        # logic directly by patching.
        return channel

    def _make_handler(self, channel, body: bytes, headers: dict[str, str]):
        """Build a minimal handler-like object that exercises do_POST validation."""
        # We call the validation logic from the handler inline by replicating
        # the channel's _check_rate_limit and inspecting return codes.
        # For simplicity, test the channel-level methods directly.
        return channel, body, headers

    # -- rate-limit is allow-listed by default for 127.0.0.1 unless exhausted

    def test_empty_body_post_returns_400(self):
        """A POST with an empty body (no 'prompt' key) results in a 400 response."""
        import http.client
        import socket

        from missy.channels.webhook import WebhookChannel

        channel = WebhookChannel(host="127.0.0.1", port=0)
        # Use a real (but ephemeral) port

        # We spin up the server on an available port briefly
        with tempfile.TemporaryDirectory(), socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        channel._port = free_port
        channel.start()
        try:
            time.sleep(0.05)  # let the thread start
            conn = http.client.HTTPConnection("127.0.0.1", free_port, timeout=3)
            conn.request(
                "POST",
                "/",
                body=b"{}",
                headers={"Content-Type": "application/json", "Content-Length": "2"},
            )
            resp = conn.getresponse()
            assert resp.status == 400
        finally:
            channel.stop()

    def test_non_json_content_type_returns_415(self):
        """A POST with text/plain Content-Type is rejected with 415."""
        import http.client
        import socket

        from missy.channels.webhook import WebhookChannel

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        channel = WebhookChannel(host="127.0.0.1", port=free_port)
        channel.start()
        try:
            time.sleep(0.05)
            conn = http.client.HTTPConnection("127.0.0.1", free_port, timeout=3)
            body = b'{"prompt": "hello"}'
            conn.request(
                "POST",
                "/",
                body=body,
                headers={
                    "Content-Type": "text/plain",
                    "Content-Length": str(len(body)),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 415
        finally:
            channel.stop()

    def test_oversized_content_length_returns_413(self):
        """A POST whose Content-Length exceeds 1 MB is rejected with 413."""
        import http.client
        import socket

        from missy.channels.webhook import WebhookChannel

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        channel = WebhookChannel(host="127.0.0.1", port=free_port)
        channel.start()
        try:
            time.sleep(0.05)
            conn = http.client.HTTPConnection("127.0.0.1", free_port, timeout=3)
            # Report a Content-Length bigger than 1 MB but send nothing extra
            oversized = 1024 * 1024 + 1
            conn.request(
                "POST",
                "/",
                body=b"",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(oversized),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 413
        finally:
            channel.stop()

    def test_valid_post_is_queued(self):
        """A well-formed POST queues a ChannelMessage and returns 202."""
        import http.client
        import socket

        from missy.channels.webhook import WebhookChannel

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        channel = WebhookChannel(host="127.0.0.1", port=free_port)
        channel.start()
        try:
            time.sleep(0.05)
            body = json.dumps({"prompt": "Do something useful"}).encode()
            conn = http.client.HTTPConnection("127.0.0.1", free_port, timeout=3)
            conn.request(
                "POST",
                "/",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 202
            msg = channel.receive()
            assert msg is not None
            assert msg.content == "Do something useful"
        finally:
            channel.stop()

    def test_get_request_returns_405(self):
        """GET requests to the webhook are rejected with 405."""
        import http.client
        import socket

        from missy.channels.webhook import WebhookChannel

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free_port = s.getsockname()[1]

        channel = WebhookChannel(host="127.0.0.1", port=free_port)
        channel.start()
        try:
            time.sleep(0.05)
            conn = http.client.HTTPConnection("127.0.0.1", free_port, timeout=3)
            conn.request("GET", "/")
            resp = conn.getresponse()
            assert resp.status == 405
        finally:
            channel.stop()


# ---------------------------------------------------------------------------
# 5. Audit logger edge cases
# ---------------------------------------------------------------------------


class TestAuditLoggerEdgeCases:
    """Edge cases for AuditLogger._handle_event and query methods."""

    @pytest.fixture()
    def log_file(self, tmp_path: Path) -> Path:
        return tmp_path / "audit.jsonl"

    @pytest.fixture()
    def audit_logger(self, log_file: Path):
        """Create an AuditLogger with a fresh EventBus so it doesn't interfere."""
        from missy.observability.audit_logger import AuditLogger

        bus = EventBus()
        logger = AuditLogger(log_path=str(log_file), bus=bus)
        return logger, bus

    def _make_event(self, **kwargs) -> AuditEvent:
        defaults = {
            "session_id": "s1",
            "task_id": "t1",
            "event_type": "test.event",
            "category": "network",
            "result": "allow",
            "detail": {},
        }
        defaults.update(kwargs)
        return AuditEvent.now(**defaults)

    def test_logging_unicode_content_is_written_correctly(self, audit_logger, log_file):
        """Events containing unicode (emoji, CJK, etc.) are stored and retrievable."""
        logger, bus = audit_logger
        event = self._make_event(detail={"msg": "Hello \u4e16\u754c \U0001f600"})
        logger._handle_event(event)

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "\u4e16\u754c" in record["detail"]["msg"]

    def test_logging_nested_dict_event_is_serialised(self, audit_logger, log_file):
        """Events with deeply nested detail dicts are serialised to JSON correctly."""
        logger, bus = audit_logger
        nested_detail = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", {"key": "value"}],
                }
            }
        }
        event = self._make_event(detail=nested_detail)
        logger._handle_event(event)

        lines = log_file.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        assert record["detail"]["level1"]["level2"]["level3"][2]["key"] == "value"

    def test_get_policy_violations_with_empty_log_returns_empty_list(self, tmp_path):
        """get_policy_violations on an empty log file returns []."""
        from missy.observability.audit_logger import AuditLogger

        log_file = tmp_path / "empty.jsonl"
        log_file.touch()
        logger = AuditLogger(log_path=str(log_file), bus=EventBus())
        assert logger.get_policy_violations(limit=10) == []

    def test_get_policy_violations_with_no_file_returns_empty_list(self, tmp_path):
        """get_policy_violations when the log file does not exist returns []."""
        from missy.observability.audit_logger import AuditLogger

        log_path = str(tmp_path / "nonexistent.jsonl")
        logger = AuditLogger(log_path=log_path, bus=EventBus())
        # Do NOT write anything; file should not exist yet
        assert logger.get_policy_violations(limit=10) == []

    def test_get_recent_events_with_empty_log_returns_empty_list(self, tmp_path):
        """get_recent_events on an empty log file returns []."""
        from missy.observability.audit_logger import AuditLogger

        log_file = tmp_path / "empty2.jsonl"
        log_file.touch()
        logger = AuditLogger(log_path=str(log_file), bus=EventBus())
        assert logger.get_recent_events(limit=5) == []

    def test_security_violations_only_returns_deny_events(self, audit_logger, log_file):
        """get_policy_violations filters to result=='deny' only."""
        logger, bus = audit_logger
        allow_event = self._make_event(result="allow", event_type="network.request")
        deny_event = self._make_event(result="deny", event_type="policy.deny")
        logger._handle_event(allow_event)
        logger._handle_event(deny_event)

        violations = logger.get_policy_violations(limit=10)
        assert len(violations) == 1
        assert violations[0]["result"] == "deny"

    def test_logging_event_with_none_detail_value(self, audit_logger, log_file):
        """Events with None values in detail dict are serialised via default=str."""
        logger, bus = audit_logger
        event = self._make_event(detail={"key": None, "other": 42})
        logger._handle_event(event)

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["detail"]["key"] is None
        assert record["detail"]["other"] == 42

    def test_multiple_events_are_appended_in_order(self, audit_logger, log_file):
        """Successive _handle_event calls append lines in chronological order."""
        logger, bus = audit_logger
        for i in range(5):
            event = self._make_event(detail={"index": i}, event_type=f"test.event.{i}")
            logger._handle_event(event)

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["detail"]["index"] == i
