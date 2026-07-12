"""SR-1.1: audit event signing + verification.

The security review found the audit log completely unsigned in practice:
the only signing path (AgentRuntime._emit_event) covered 3 fields out of
8, embedded the signature inside the mutable `detail` dict, only fired
for events routed through that one method, and had zero verification
anywhere in the codebase. Live PoC in the review: editing a `deny` event
to `allow` in the JSONL file and reading it back succeeded cleanly.

These tests exercise the real AuditLogger write path (via a real
EventBus.publish(), not by calling _handle_event() directly) and the
real AgentIdentity Ed25519 keypair (not mocks), reproducing the review's
own tamper scenario against the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from missy.core.events import AuditEvent, EventBus
from missy.observability.audit_logger import (
    AuditLogger,
    _make_default_identity,
    init_audit_logger,
    verify_audit_log,
)
from missy.security.identity import AgentIdentity


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def log_path(tmp_path: Path) -> str:
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def identity() -> AgentIdentity:
    return AgentIdentity.generate()


def _publish(bus: EventBus, event_type: str, category: str, result: str, **detail):
    bus.publish(
        AuditEvent.now(
            session_id="s1",
            task_id="t1",
            event_type=event_type,
            category=category,
            result=result,
            detail=detail,
        )
    )


def _read_lines(log_path: str) -> list[dict]:
    return [json.loads(ln) for ln in Path(log_path).read_text().splitlines() if ln.strip()]


class TestSigningWritesTopLevelSignature:
    def test_signed_event_has_top_level_identity_signature(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "shell.exec", "shell", "deny", command="rm -rf /")

        records = _read_lines(log_path)
        assert len(records) == 1
        assert "identity_signature" in records[0]
        # Top-level, not nested in the mutable detail dict (the review's
        # explicit criticism of the old AgentRuntime._emit_event scheme).
        assert "identity_signature" not in records[0]["detail"]

    def test_signature_covers_every_field_not_just_three(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        """The old scheme signed only session_id/task_id/event_type.
        Tampering with `result` (the field that actually matters for a
        deny→allow attack) must now be detected."""
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "shell.exec", "shell", "deny", command="rm -rf /")

        records = _read_lines(log_path)
        records[0]["result"] = "allow"
        Path(log_path).write_text(json.dumps(records[0]) + "\n")

        results = verify_audit_log(log_path, identity)
        assert results[0].status == "tampered"

    def test_unsigned_when_no_identity_configured(self, log_path: str, bus: EventBus):
        """Direct AuditLogger construction defaults to unsigned (no
        implicit key I/O for the many read-only/test callers)."""
        AuditLogger(log_path=log_path, bus=bus, identity=None)
        _publish(bus, "network.request", "network", "allow")

        records = _read_lines(log_path)
        assert "identity_signature" not in records[0]


class TestVerifyAuditLogDetectsTheReviewsExactPoc:
    def test_deny_to_allow_tamper_is_detected(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        """Live reproduction of the security review's own PoC: sign a
        real deny event, edit it to allow after the fact exactly as the
        review's demonstration did, and confirm verification now flags
        it -- where before the fix, reading it back succeeded cleanly."""
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "shell.exec", "shell", "deny", command="rm -rf /")

        before = verify_audit_log(log_path, identity)
        assert before[0].status == "valid"

        records = _read_lines(log_path)
        records[0]["result"] = "allow"
        Path(log_path).write_text(json.dumps(records[0]) + "\n")

        after = verify_audit_log(log_path, identity)
        assert after[0].status == "tampered"
        assert after[0].event_type == "shell.exec"

    def test_untampered_log_verifies_valid(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "network.request", "network", "allow", host="example.com")
        _publish(bus, "shell.exec", "shell", "deny", command="ls")

        results = verify_audit_log(log_path, identity)
        assert [r.status for r in results] == ["valid", "valid"]
        assert [r.event_type for r in results] == ["network.request", "shell.exec"]

    def test_detail_tamper_is_detected(self, log_path: str, bus: EventBus, identity: AgentIdentity):
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "network.request", "network", "deny", host="169.254.169.254")

        records = _read_lines(log_path)
        records[0]["detail"]["host"] = "example.com"
        Path(log_path).write_text(json.dumps(records[0]) + "\n")

        results = verify_audit_log(log_path, identity)
        assert results[0].status == "tampered"

    def test_signature_deletion_reported_as_unsigned_not_valid(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        """Stripping the signature entirely must not read back as
        trivially valid (an attacker could otherwise just delete the
        field rather than forge it)."""
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "shell.exec", "shell", "deny", command="rm -rf /")

        records = _read_lines(log_path)
        records[0]["result"] = "allow"
        del records[0]["identity_signature"]
        Path(log_path).write_text(json.dumps(records[0]) + "\n")

        results = verify_audit_log(log_path, identity)
        assert results[0].status == "unsigned"

    def test_verifying_with_wrong_identity_fails(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        other = AgentIdentity.generate()
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "network.request", "network", "allow")

        results = verify_audit_log(log_path, other)
        assert results[0].status == "tampered"

    def test_malformed_line_reported_not_silently_skipped(self, log_path: str):
        Path(log_path).write_text("not valid json at all\n")
        results = verify_audit_log(log_path, AgentIdentity.generate())
        assert results[0].status == "malformed"

    def test_empty_log_returns_empty_list(self, log_path: str, identity: AgentIdentity):
        assert verify_audit_log(log_path, identity) == []

    def test_missing_log_file_returns_empty_list(self, tmp_path: Path, identity: AgentIdentity):
        missing = str(tmp_path / "does_not_exist.jsonl")
        assert verify_audit_log(missing, identity) == []

    def test_signature_key_order_in_file_does_not_matter(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        """Verification re-serializes with sort_keys=True on both sides,
        so this must not be sensitive to how a downstream tool happened
        to reformat/reorder the JSON keys on disk (only to a change in
        the actual field *values*)."""
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "network.request", "network", "allow", host="example.com")

        records = _read_lines(log_path)
        reordered = dict(reversed(list(records[0].items())))
        Path(log_path).write_text(json.dumps(reordered) + "\n")

        results = verify_audit_log(log_path, identity)
        assert results[0].status == "valid"


class TestHashChainDetectsReordering:
    """Per-line signing alone (the rest of this file) detects CONTENT
    tampering but not REORDERING: two validly-signed lines swapped in
    position, or one deleted, would leave every individual signature
    intact. Each line now carries a prev_chain_hash (SHA-256 of the exact
    bytes of the line immediately before it, covered by that line's own
    signature) so verify_audit_log() can also detect sequence rewrites.
    """

    def test_baseline_chain_is_valid_and_first_line_not_applicable(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "event.a", "test", "allow")
        _publish(bus, "event.b", "test", "allow")
        _publish(bus, "event.c", "test", "allow")

        results = verify_audit_log(log_path, identity)
        assert [r.status for r in results] == ["valid", "valid", "valid"]
        assert results[0].chain_ok is None  # first line: nothing to check against
        assert results[1].chain_ok is True
        assert results[2].chain_ok is True

    def test_swapping_two_validly_signed_lines_is_caught_by_chain_ok(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        """The exact gap per-line signing alone cannot close: a pure
        reorder with no content edits leaves every signature valid.
        """
        AuditLogger(log_path=log_path, bus=bus, identity=identity)
        _publish(bus, "event.a", "test", "allow")
        _publish(bus, "event.b", "test", "allow")
        _publish(bus, "event.c", "test", "allow")

        lines = Path(log_path).read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        Path(log_path).write_text("\n".join(lines) + "\n")

        results = verify_audit_log(log_path, identity)
        assert all(r.status == "valid" for r in results), (
            "a pure reorder must not break any individual line's signature"
        )
        assert any(r.chain_ok is False for r in results[1:]), (
            "reordering must be caught via chain_ok despite valid signatures"
        )

    def test_chain_continues_across_a_fresh_logger_instance(
        self, log_path: str, identity: AgentIdentity
    ):
        """A restarted process (fresh AuditLogger construction against the
        same existing file) must seed its chain state from the file's
        current tail, not start a disconnected new chain.
        """
        bus1 = EventBus()
        AuditLogger(log_path=log_path, bus=bus1, identity=identity)
        _publish(bus1, "before.restart", "test", "allow")

        bus2 = EventBus()
        AuditLogger(log_path=log_path, bus=bus2, identity=identity)
        _publish(bus2, "after.restart", "test", "allow")

        results = verify_audit_log(log_path, identity)
        assert [r.status for r in results] == ["valid", "valid"]
        assert results[1].chain_ok is True

    def test_legacy_lines_with_no_chain_field_are_not_applicable(
        self, log_path: str, identity: AgentIdentity
    ):
        """A log written before this feature existed has no
        prev_chain_hash field at all -- must remain valid, not be
        misclassified as tampered/broken.
        """
        record = {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "session_id": "s1",
            "task_id": "t1",
            "event_type": "legacy.event",
            "category": "test",
            "result": "allow",
            "detail": {},
            "policy_rule": None,
        }
        payload = json.dumps(record, sort_keys=True, default=str).encode("utf-8")
        record["identity_signature"] = identity.sign(payload).hex()
        Path(log_path).write_text(json.dumps(record, default=str) + "\n")

        results = verify_audit_log(log_path, identity)
        assert results[0].status == "valid"
        assert results[0].chain_ok is None

    def test_concurrent_writers_do_not_corrupt_the_chain(
        self, log_path: str, bus: EventBus, identity: AgentIdentity
    ):
        """The build-sign-write-advance sequence runs under a lock so
        concurrent publishers can't both read the same 'previous hash'
        and produce two lines chained from the same point.
        """
        import threading

        AuditLogger(log_path=log_path, bus=bus, identity=identity)

        n_threads, n_events = 6, 15

        def worker(tid: int) -> None:
            for i in range(n_events):
                _publish(bus, f"ev.{tid}.{i}", "test", "allow")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        results = verify_audit_log(log_path, identity)
        assert len(results) == n_threads * n_events
        assert all(r.status == "valid" for r in results)
        assert not any(r.chain_ok is False for r in results)


class TestInitAuditLoggerSignsByDefault:
    def test_init_audit_logger_resolves_identity_and_signs(self, tmp_path: Path, monkeypatch):
        key_path = str(tmp_path / "identity.pem")
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", key_path)

        log_path = str(tmp_path / "audit.jsonl")
        fresh_bus = EventBus()
        al = init_audit_logger(log_path=log_path)
        # Swap in a scoped bus post-hoc so this test doesn't touch the
        # process-global event_bus (matches this codebase's established
        # pattern for signature-relevant tests).
        al._bus = fresh_bus
        al._subscribe()
        _publish(fresh_bus, "network.request", "network", "allow")

        records = _read_lines(log_path)
        assert "identity_signature" in records[0]

    def test_explicit_identity_is_reused_not_re_resolved(self, tmp_path: Path):
        log_path = str(tmp_path / "audit.jsonl")
        identity = AgentIdentity.generate()
        al = init_audit_logger(log_path=log_path, identity=identity)
        assert al._identity is identity


class TestMakeDefaultIdentity:
    def test_returns_none_on_failure_without_raising(self, monkeypatch):
        import missy.security.identity as identity_module

        def _raise(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(identity_module.AgentIdentity, "load_or_generate", classmethod(_raise))
        assert _make_default_identity() is None

    def test_returns_a_real_identity_on_success(self, tmp_path: Path, monkeypatch):
        key_path = str(tmp_path / "identity.pem")
        monkeypatch.setattr("missy.security.identity.DEFAULT_KEY_PATH", key_path)
        result = _make_default_identity()
        assert isinstance(result, AgentIdentity)
        assert Path(key_path).exists()


class TestAgentIdentityLoadOrGenerate:
    def test_generates_and_saves_when_absent(self, tmp_path: Path):
        key_path = str(tmp_path / "sub" / "identity.pem")
        identity = AgentIdentity.load_or_generate(key_path)
        assert Path(key_path).exists()
        assert isinstance(identity, AgentIdentity)

    def test_loads_existing_key_rather_than_regenerating(self, tmp_path: Path):
        key_path = str(tmp_path / "identity.pem")
        first = AgentIdentity.load_or_generate(key_path)
        second = AgentIdentity.load_or_generate(key_path)
        assert first.public_key_fingerprint() == second.public_key_fingerprint()

    def test_same_identity_used_by_runtime_and_audit_logger(self, tmp_path: Path):
        """SR-1.1's core integration guarantee: AgentRuntime and
        AuditLogger must sign with the SAME keypair when both resolve
        the identity via the shared default path, or a runtime-side
        event and an audit-sink-side event would be unverifiable
        against each other."""
        key_path = str(tmp_path / "identity.pem")
        from missy.agent.runtime import AgentRuntime

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("missy.security.identity.DEFAULT_KEY_PATH", key_path)
            runtime_identity = AgentRuntime._make_identity()
            audit_identity = AgentIdentity.load_or_generate(key_path)

        assert runtime_identity.public_key_fingerprint() == audit_identity.public_key_fingerprint()
