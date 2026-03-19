"""Vision audit event logging.

Integrates with Missy's AuditLogger to record vision-related events
including device selection, capture, analysis, and intent activation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _emit_audit_event(category: str, action: str, details: dict[str, Any]) -> None:
    """Emit a vision audit event via the global AuditLogger."""
    try:
        from missy.observability.audit_logger import get_audit_logger

        audit = get_audit_logger()
        if audit is None:
            return

        event = {
            "category": category,
            "action": action,
            "timestamp": datetime.now(UTC).isoformat(),
            **details,
        }
        audit.log(event)
    except Exception:
        # Never let audit failures break the vision pipeline
        logger.debug("Failed to emit vision audit event", exc_info=True)


def audit_vision_capture(
    *,
    device: str = "",
    source_type: str = "",
    trigger_reason: str = "user_command",
    success: bool = True,
    width: int = 0,
    height: int = 0,
    error: str = "",
) -> None:
    """Log a vision capture event."""
    _emit_audit_event(
        "vision",
        "capture",
        {
            "device": device,
            "source_type": source_type,
            "trigger_reason": trigger_reason,
            "success": success,
            "width": width,
            "height": height,
            "error": error,
        },
    )


def audit_vision_analysis(
    *,
    mode: str = "general",
    source_type: str = "",
    trigger_reason: str = "user_command",
    success: bool = True,
    error: str = "",
) -> None:
    """Log a vision analysis event."""
    _emit_audit_event(
        "vision",
        "analyze",
        {
            "mode": mode,
            "source_type": source_type,
            "trigger_reason": trigger_reason,
            "success": success,
            "error": error,
        },
    )


def audit_vision_intent(
    *,
    text: str = "",
    intent: str = "",
    confidence: float = 0.0,
    decision: str = "",
    trigger_phrase: str = "",
) -> None:
    """Log a vision intent classification event."""
    _emit_audit_event(
        "vision",
        "intent",
        {
            "text_length": len(text),
            "intent": intent,
            "confidence": round(confidence, 3),
            "decision": decision,
            "trigger_phrase": trigger_phrase,
        },
    )


def audit_vision_device_discovery(
    *,
    camera_count: int = 0,
    preferred_device: str = "",
    preferred_name: str = "",
) -> None:
    """Log a camera discovery event."""
    _emit_audit_event(
        "vision",
        "device_discovery",
        {
            "camera_count": camera_count,
            "preferred_device": preferred_device,
            "preferred_name": preferred_name,
        },
    )


def audit_vision_session(
    *,
    action: str = "",
    task_id: str = "",
    task_type: str = "",
    frame_count: int = 0,
) -> None:
    """Log a scene session lifecycle event."""
    _emit_audit_event(
        "vision",
        f"session_{action}",
        {
            "task_id": task_id,
            "task_type": task_type,
            "frame_count": frame_count,
        },
    )


def audit_vision_burst(
    *,
    device: str = "",
    count: int = 0,
    successful: int = 0,
    best_only: bool = False,
    trigger_reason: str = "user_command",
) -> None:
    """Log a burst capture event."""
    _emit_audit_event(
        "vision",
        "burst_capture",
        {
            "device": device,
            "requested_count": count,
            "successful_count": successful,
            "best_only": best_only,
            "trigger_reason": trigger_reason,
        },
    )


def audit_vision_error(
    *,
    operation: str = "",
    error: str = "",
    device: str = "",
    recoverable: bool = True,
) -> None:
    """Log a vision error event for diagnostics."""
    _emit_audit_event(
        "vision",
        "error",
        {
            "operation": operation,
            "error": error,
            "device": device,
            "recoverable": recoverable,
        },
    )
