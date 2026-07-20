"""Audit browser query helpers for the Web TUI and JSON API."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from missy.core.events import AuditEvent, EventBus, event_bus

AUDIT_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "csrf",
    "csrf_token",
    "key",
    "password",
    "secret",
    "token",
}

AUDIT_FILTER_KEYS = (
    "q",
    "event_type",
    "category",
    "result",
    "session_id",
    "task_id",
    "severity",
    "actor",
    "source",
    "subsystem",
    "action",
    "since",
    "until",
)


def query_audit_events(
    params: dict[str, Any],
    *,
    bus: EventBus = event_bus,
) -> dict[str, Any]:
    """Return redacted, filtered, paginated audit events for API responses."""
    limit = _bounded_int(params.get("limit"), default=50, minimum=1, maximum=500)
    offset = _bounded_int(params.get("offset"), default=0, minimum=0, maximum=100_000)
    scan_limit = max((limit + offset) * 10, 200)

    memory_events = [event_to_record(event) for event in bus.get_events()]
    source = "memory"
    try:
        from missy.observability.audit_logger import get_audit_logger

        file_events = get_audit_logger().get_recent_events(limit=scan_limit)
        # The process-wide logger may be reconfigured while more than one
        # API/config lifecycle is winding down. Never let a successful file
        # read hide current-process events that were synchronously published
        # to the authoritative event bus but landed in a just-rotated path.
        # Deduplicate on the common AuditEvent fields because persisted rows
        # also carry chain/signature metadata absent from the in-memory form.
        events = list(file_events)
        seen = {_audit_identity(event) for event in file_events}
        for event in memory_events:
            identity = _audit_identity(event)
            if identity not in seen:
                events.append(event)
                seen.add(identity)
        events.sort(key=lambda event: str(event.get("timestamp") or ""))
        events = events[-scan_limit:]
        source = "file+memory"
    except Exception:
        events = memory_events[-scan_limit:]

    redacted = [redact_audit_value(event) for event in events]
    matched = [event for event in redacted if audit_record_matches(event, params)]
    newest_first = list(reversed(matched))
    page = newest_first[offset : offset + limit]
    with_ids = [add_audit_event_id(event) for event in page]

    return {
        "events": with_ids,
        "count": len(with_ids),
        "total": len(matched),
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < len(matched),
        "source": source,
        "filters": {key: params[key] for key in AUDIT_FILTER_KEYS if params.get(key)},
        "facets": build_audit_facets(matched),
    }


def _audit_identity(event: dict[str, Any]) -> str:
    """Stable identity for the fields shared by file and in-memory events."""
    common = {
        key: event.get(key)
        for key in (
            "timestamp",
            "session_id",
            "task_id",
            "event_type",
            "category",
            "result",
            "detail",
            "policy_rule",
        )
    }
    # Durable audit rows are redacted before they are written, while the
    # EventBus retains the original structured detail.  Compare the same
    # redacted representation on both sides or an event containing a secret
    # appears twice when the file and in-memory snapshots are merged.
    redacted = redact_audit_value(common)
    return json.dumps(redacted, sort_keys=True, default=str, separators=(",", ":"))


def redact_audit_value(value: Any, *, key: str = "") -> Any:
    """Return a copy of an audit value with credential material removed."""
    key_l = key.lower()
    if any(marker in key_l for marker in AUDIT_SENSITIVE_KEYS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_audit_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_audit_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_audit_value(item) for item in value]
    if isinstance(value, str):
        try:
            from missy.security.secrets import secrets_detector

            return secrets_detector.redact(value)
        except Exception:
            return value
    return value


def audit_record_matches(record: dict[str, Any], params: dict[str, Any]) -> bool:
    """Return True when an audit record satisfies API query parameters."""
    detail = record.get("detail") if isinstance(record.get("detail"), dict) else {}
    exact_filters = {
        "event_type": record.get("event_type"),
        "category": record.get("category"),
        "result": record.get("result"),
        "session_id": record.get("session_id"),
        "task_id": record.get("task_id"),
        "severity": detail.get("severity"),
        "actor": detail.get("actor"),
        "source": detail.get("source"),
        "subsystem": detail.get("subsystem") or record.get("category"),
        "action": detail.get("action") or record.get("event_type"),
    }
    for param, value in exact_filters.items():
        wanted = params.get(param)
        if wanted and str(value or "").lower() != str(wanted).lower():
            return False

    query = str(params.get("q") or "").strip().lower()
    if query:
        haystack = json.dumps(record, sort_keys=True, default=str).lower()
        if query not in haystack:
            return False

    timestamp = str(record.get("timestamp") or "")
    since = str(params.get("since") or "")
    until = str(params.get("until") or "")
    if since and timestamp < since:
        return False
    return not (until and timestamp > until)


def build_audit_facets(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Build filter facets from the complete matched event set."""
    facets: dict[str, dict[str, int]] = {
        "category": {},
        "result": {},
        "severity": {},
        "subsystem": {},
    }
    for event in events:
        detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
        values = {
            "category": event.get("category"),
            "result": event.get("result"),
            "severity": detail.get("severity"),
            "subsystem": detail.get("subsystem") or event.get("category"),
        }
        for key, value in values.items():
            if value:
                value_s = str(value)
                facets[key][value_s] = facets[key].get(value_s, 0) + 1
    return facets


def add_audit_event_id(event: dict[str, Any]) -> dict[str, Any]:
    """Attach a stable ID derived from the redacted event payload."""
    payload = json.dumps(event, sort_keys=True, default=str, separators=(",", ":"))
    event_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return {"id": event_id, **event}


def event_to_record(event: AuditEvent) -> dict[str, Any]:
    """Convert an in-memory audit event to its JSONL-shaped dict form."""
    return {
        "timestamp": event.timestamp.isoformat(),
        "session_id": event.session_id,
        "task_id": event.task_id,
        "event_type": event.event_type,
        "category": event.category,
        "result": event.result,
        "detail": event.detail,
        "policy_rule": event.policy_rule,
    }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (ValueError, TypeError):
        parsed = default
    return max(minimum, min(parsed, maximum))
