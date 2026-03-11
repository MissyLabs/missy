"""Custom exceptions for the Missy framework."""

from __future__ import annotations


class MissyError(Exception):
    """Base exception for all Missy errors."""


class PolicyViolationError(MissyError):
    """Raised when an action is blocked by a policy rule.

    Attributes:
        category: The policy category that was violated
            (e.g. 'network', 'filesystem', 'shell', 'plugin').
        detail: Human-readable description of the violation.
    """

    def __init__(self, message: str, *, category: str, detail: str) -> None:
        super().__init__(message)
        self.category = category
        self.detail = detail

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({str(self)!r}, "
            f"category={self.category!r}, detail={self.detail!r})"
        )


class ConfigurationError(MissyError):
    """Raised when the configuration is invalid or cannot be loaded."""


class ProviderError(MissyError):
    """Raised when an AI provider returns an error or is misconfigured."""


class SchedulerError(MissyError):
    """Raised when the task scheduler encounters an unrecoverable error."""


class ApprovalRequiredError(MissyError):
    """Raised when an action requires explicit operator approval before proceeding.

    Attributes:
        action: Short description of the action requiring approval.
        reason: Why approval is required.
    """

    def __init__(self, action: str, reason: str = "") -> None:
        self.action = action
        self.reason = reason
        msg = f"Approval required for action: {action!r}"
        if reason:
            msg += f" \u2014 {reason}"
        super().__init__(msg)
