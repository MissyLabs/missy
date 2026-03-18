"""Tests for missy.core.exceptions and bus_topics."""

from __future__ import annotations

import pytest

from missy.core import bus_topics
from missy.core.exceptions import (
    ApprovalRequiredError,
    ConfigurationError,
    MissyError,
    PolicyViolationError,
    ProviderError,
    SchedulerError,
)

# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_missy_error_is_base(self):
        assert issubclass(PolicyViolationError, MissyError)
        assert issubclass(ConfigurationError, MissyError)
        assert issubclass(ProviderError, MissyError)
        assert issubclass(SchedulerError, MissyError)
        assert issubclass(ApprovalRequiredError, MissyError)

    def test_all_are_exceptions(self):
        for cls in (MissyError, PolicyViolationError, ConfigurationError,
                    ProviderError, SchedulerError, ApprovalRequiredError):
            assert issubclass(cls, Exception)


class TestPolicyViolationError:
    def test_attributes(self):
        exc = PolicyViolationError(
            "blocked", category="network", detail="host not allowed"
        )
        assert str(exc) == "blocked"
        assert exc.category == "network"
        assert exc.detail == "host not allowed"

    def test_repr(self):
        exc = PolicyViolationError(
            "blocked", category="filesystem", detail="/etc/passwd"
        )
        r = repr(exc)
        assert "PolicyViolationError" in r
        assert "filesystem" in r
        assert "/etc/passwd" in r

    def test_can_be_caught_as_missy_error(self):
        with pytest.raises(MissyError):
            raise PolicyViolationError("x", category="shell", detail="rm")


class TestApprovalRequiredError:
    def test_basic(self):
        exc = ApprovalRequiredError("delete_file")
        assert "delete_file" in str(exc)
        assert exc.action == "delete_file"
        assert exc.reason == ""

    def test_with_reason(self):
        exc = ApprovalRequiredError("rm -rf", reason="destructive operation")
        assert "rm -rf" in str(exc)
        assert "destructive" in str(exc)
        assert exc.reason == "destructive operation"


class TestSimpleExceptions:
    def test_configuration_error(self):
        exc = ConfigurationError("bad config")
        assert str(exc) == "bad config"

    def test_provider_error(self):
        exc = ProviderError("rate limited")
        assert str(exc) == "rate limited"

    def test_scheduler_error(self):
        exc = SchedulerError("max jobs reached")
        assert str(exc) == "max jobs reached"


# ---------------------------------------------------------------------------
# Bus topics constants
# ---------------------------------------------------------------------------


class TestBusTopics:
    def test_all_topics_are_dotted_strings(self):
        topics = [
            bus_topics.CHANNEL_INBOUND,
            bus_topics.CHANNEL_OUTBOUND,
            bus_topics.AGENT_RUN_START,
            bus_topics.AGENT_RUN_COMPLETE,
            bus_topics.AGENT_RUN_ERROR,
            bus_topics.TOOL_REQUEST,
            bus_topics.TOOL_RESULT,
            bus_topics.SYSTEM_STARTUP,
            bus_topics.SYSTEM_SHUTDOWN,
            bus_topics.SECURITY_VIOLATION,
            bus_topics.SECURITY_APPROVAL_NEEDED,
            bus_topics.SECURITY_APPROVAL_RESPONSE,
        ]
        for topic in topics:
            assert isinstance(topic, str)
            assert "." in topic
            assert len(topic) > 3

    def test_topics_are_unique(self):
        topics = [
            bus_topics.CHANNEL_INBOUND,
            bus_topics.CHANNEL_OUTBOUND,
            bus_topics.AGENT_RUN_START,
            bus_topics.AGENT_RUN_COMPLETE,
            bus_topics.AGENT_RUN_ERROR,
            bus_topics.TOOL_REQUEST,
            bus_topics.TOOL_RESULT,
            bus_topics.SYSTEM_STARTUP,
            bus_topics.SYSTEM_SHUTDOWN,
            bus_topics.SECURITY_VIOLATION,
            bus_topics.SECURITY_APPROVAL_NEEDED,
            bus_topics.SECURITY_APPROVAL_RESPONSE,
        ]
        assert len(topics) == len(set(topics))

    def test_channel_topics(self):
        assert bus_topics.CHANNEL_INBOUND.startswith("channel.")
        assert bus_topics.CHANNEL_OUTBOUND.startswith("channel.")

    def test_agent_topics(self):
        assert bus_topics.AGENT_RUN_START.startswith("agent.")
        assert bus_topics.AGENT_RUN_COMPLETE.startswith("agent.")
        assert bus_topics.AGENT_RUN_ERROR.startswith("agent.")

    def test_security_topics(self):
        assert bus_topics.SECURITY_VIOLATION.startswith("security.")
        assert bus_topics.SECURITY_APPROVAL_NEEDED.startswith("security.")
