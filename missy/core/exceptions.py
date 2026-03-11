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
