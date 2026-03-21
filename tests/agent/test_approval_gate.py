"""Tests for missy.agent.approval — approval gate for sensitive operations."""

from __future__ import annotations

import threading

import pytest

from missy.agent.approval import (
    ApprovalDenied,
    ApprovalGate,
    ApprovalTimeout,
    PendingApproval,
)


class TestPendingApproval:
    def test_approve_sets_state(self):
        pa = PendingApproval("test action", "reason", timeout=5.0)
        pa.approve()
        assert pa._approved is True

    def test_deny_sets_state(self):
        pa = PendingApproval("test action", "reason", timeout=5.0)
        pa.deny()
        assert pa._approved is False

    def test_wait_approved(self):
        pa = PendingApproval("test", "reason", timeout=5.0)
        pa.approve()
        assert pa.wait() is True

    def test_wait_denied(self):
        pa = PendingApproval("test", "reason", timeout=5.0)
        pa.deny()
        with pytest.raises(ApprovalDenied, match="Action denied"):
            pa.wait()

    def test_wait_timeout(self):
        pa = PendingApproval("test", "reason", timeout=0.01)
        with pytest.raises(ApprovalTimeout, match="timed out"):
            pa.wait()

    def test_approve_from_thread(self):
        pa = PendingApproval("test", "reason", timeout=5.0)

        def approve_later():
            pa.approve()

        t = threading.Thread(target=approve_later)
        t.start()
        assert pa.wait() is True
        t.join()


class TestApprovalGate:
    def test_request_approved_via_handle_response(self):
        gate = ApprovalGate(default_timeout=2.0)
        # Run request in a thread since it blocks
        result = {}

        def do_request():
            try:
                gate.request("delete files", reason="cleanup", risk="high")
                result["approved"] = True
            except (ApprovalTimeout, ApprovalDenied) as e:
                result["error"] = str(e)

        t = threading.Thread(target=do_request)
        t.start()

        # Wait for the pending approval to show up
        import time

        for _ in range(50):
            pending = gate.list_pending()
            if pending:
                break
            time.sleep(0.01)

        assert len(pending) == 1
        approval_id = pending[0]["id"]
        assert gate.handle_response(f"approve {approval_id}")
        t.join(timeout=2.0)
        assert result.get("approved") is True

    def test_request_denied_via_handle_response(self):
        gate = ApprovalGate(default_timeout=2.0)
        result = {}

        def do_request():
            try:
                gate.request("delete files", risk="high")
                result["approved"] = True
            except ApprovalDenied:
                result["denied"] = True
            except ApprovalTimeout:
                result["timeout"] = True

        t = threading.Thread(target=do_request)
        t.start()

        import time

        for _ in range(50):
            pending = gate.list_pending()
            if pending:
                break
            time.sleep(0.01)

        approval_id = pending[0]["id"]
        assert gate.handle_response(f"deny {approval_id}")
        t.join(timeout=2.0)
        assert result.get("denied") is True

    def test_request_timeout(self):
        gate = ApprovalGate(default_timeout=0.01)
        with pytest.raises(ApprovalTimeout):
            gate.request("slow action")

    def test_handle_response_unrecognized(self):
        gate = ApprovalGate()
        assert gate.handle_response("random text") is False

    def test_list_pending_empty(self):
        gate = ApprovalGate()
        assert gate.list_pending() == []

    def test_send_fn_called(self):
        messages = []
        gate = ApprovalGate(send_fn=messages.append, default_timeout=0.01)
        with pytest.raises(ApprovalTimeout):
            gate.request("test action", reason="test reason", risk="low")
        assert len(messages) == 1
        assert "Approval Required" in messages[0]
        assert "test action" in messages[0]
        assert "test reason" in messages[0]

    def test_send_fn_failure_does_not_block(self):
        def bad_send(msg):
            raise RuntimeError("send failed")

        gate = ApprovalGate(send_fn=bad_send, default_timeout=0.01)
        with pytest.raises(ApprovalTimeout):
            gate.request("test action")
        # Should not crash despite send failure

    def test_no_send_fn(self):
        gate = ApprovalGate(send_fn=None, default_timeout=0.01)
        with pytest.raises(ApprovalTimeout):
            gate.request("test action")

    def test_pending_cleaned_up_after_completion(self):
        gate = ApprovalGate(default_timeout=0.01)
        with pytest.raises(ApprovalTimeout):
            gate.request("test")
        assert gate.list_pending() == []

    def test_handle_response_case_insensitive(self):
        gate = ApprovalGate(default_timeout=2.0)
        result = {}

        def do_request():
            try:
                gate.request("action")
                result["approved"] = True
            except (ApprovalTimeout, ApprovalDenied) as e:
                result["error"] = str(e)

        t = threading.Thread(target=do_request)
        t.start()

        import time

        for _ in range(50):
            pending = gate.list_pending()
            if pending:
                break
            time.sleep(0.01)

        approval_id = pending[0]["id"]
        assert gate.handle_response(f"APPROVE {approval_id}")
        t.join(timeout=2.0)
        assert result.get("approved") is True


class TestExceptions:
    def test_approval_timeout_is_exception(self):
        assert issubclass(ApprovalTimeout, Exception)

    def test_approval_denied_is_exception(self):
        assert issubclass(ApprovalDenied, Exception)
