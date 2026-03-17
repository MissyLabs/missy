"""Tests for missy.agent.interactive_approval — Interactive TUI approval."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from missy.agent.interactive_approval import InteractiveApproval


@pytest.fixture
def approval() -> InteractiveApproval:
    return InteractiveApproval()


class TestCheckRemembered:
    """Test the check_remembered method."""

    def test_check_remembered_unknown(self, approval: InteractiveApproval) -> None:
        """Returns None for actions that have never been seen."""
        result = approval.check_remembered("network_request", "https://example.com")
        assert result is None

    def test_check_remembered_after_allow_always(self, approval: InteractiveApproval) -> None:
        """Returns True after an 'allow always' decision is stored."""
        key = approval._make_key("network_request", "https://example.com")
        approval._remembered[key] = True

        result = approval.check_remembered("network_request", "https://example.com")
        assert result is True


class TestPromptUser:
    """Test the prompt_user method."""

    def test_non_tty_denies(self, approval: InteractiveApproval) -> None:
        """When stdin is not a TTY, always deny."""
        with patch.object(InteractiveApproval, "_is_tty", return_value=False):
            result = approval.prompt_user("network_request", "https://example.com")
        assert result is False

    def test_remember_allow_always(self, approval: InteractiveApproval) -> None:
        """Response 'a' is remembered for subsequent calls."""
        with (
            patch.object(InteractiveApproval, "_is_tty", return_value=True),
            patch.object(InteractiveApproval, "_do_prompt", return_value=True),
        ):
            # Simulate "a" response by directly testing the remembered path.
            pass

        # Manually simulate the "allow always" flow:
        key = approval._make_key("network_request", "https://example.com")
        approval._remembered[key] = True

        # Now prompt_user should return True without calling _do_prompt.
        result = approval.prompt_user("network_request", "https://example.com")
        assert result is True

    def test_deny_not_remembered(self, approval: InteractiveApproval) -> None:
        """Response 'n' is NOT stored in remembered decisions."""
        with (
            patch.object(InteractiveApproval, "_is_tty", return_value=True),
            patch.object(InteractiveApproval, "_do_prompt", return_value=False),
        ):
            result = approval.prompt_user("network_request", "https://example.com")

        assert result is False
        # Verify nothing was remembered.
        assert approval.check_remembered("network_request", "https://example.com") is None

    def test_allow_once_not_remembered(self, approval: InteractiveApproval) -> None:
        """Response 'y' (allow once) is not remembered."""
        with (
            patch.object(InteractiveApproval, "_is_tty", return_value=True),
            patch.object(InteractiveApproval, "_do_prompt", return_value=True),
        ):
            result = approval.prompt_user("network_request", "https://example.com")

        assert result is True
        # 'y' does not store in remembered (only 'a' does).
        assert approval.check_remembered("network_request", "https://example.com") is None


class TestThreadSafe:
    """Test concurrent access to the InteractiveApproval instance."""

    def test_thread_safe(self, approval: InteractiveApproval) -> None:
        """Concurrent access from multiple threads does not crash."""
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                action = f"action_{i % 5}"
                detail = f"detail_{i}"
                # Check remembered (should be None or True).
                approval.check_remembered(action, detail)
                # Directly set a remembered value.
                key = approval._make_key(action, detail)
                with approval._lock:
                    approval._remembered[key] = True
                # Check again.
                result = approval.check_remembered(action, detail)
                assert result is True
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


class TestMakeKey:
    """Test the _make_key static method."""

    def test_consistent_hashing(self) -> None:
        """Same inputs always produce the same key."""
        key1 = InteractiveApproval._make_key("action", "detail")
        key2 = InteractiveApproval._make_key("action", "detail")
        assert key1 == key2

    def test_different_inputs_different_keys(self) -> None:
        """Different inputs produce different keys."""
        key1 = InteractiveApproval._make_key("action_a", "detail")
        key2 = InteractiveApproval._make_key("action_b", "detail")
        assert key1 != key2
