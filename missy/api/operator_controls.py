"""Policy-shaped operator controls for the Web TUI/API."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from missy.api.audit_browser import redact_audit_value

if TYPE_CHECKING:
    from missy.providers.registry import ProviderRegistry
    from missy.scheduler.manager import SchedulerManager


_CONTROL_PROVIDER_SET_DEFAULT = "provider.set_default"
_CONTROL_SCHEDULER_PAUSE = "scheduler.pause_job"
_CONTROL_SCHEDULER_RESUME = "scheduler.resume_job"
_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def list_operator_controls(
    provider_registry: ProviderRegistry | None = None,
    scheduler: SchedulerManager | None = None,
) -> dict[str, Any]:
    """Return the available operator controls and target state."""
    providers = _provider_targets(provider_registry)
    pause_targets, resume_targets = _scheduler_targets(scheduler)
    return {
        "controls": [
            {
                "id": _CONTROL_PROVIDER_SET_DEFAULT,
                "label": "Set default provider",
                "description": "Switch the in-process default provider used for new API sessions.",
                "subsystem": "provider",
                "requires_confirmation": True,
                "confirmation_template": "set-default:{target}",
                "enabled": bool(provider_registry is not None and providers),
                "targets": providers,
            },
            {
                "id": _CONTROL_SCHEDULER_PAUSE,
                "label": "Pause scheduled job",
                "description": "Disable a scheduled job until an operator resumes it.",
                "subsystem": "scheduler",
                "requires_confirmation": True,
                "confirmation_template": "pause-job:{target}",
                "enabled": bool(scheduler is not None and pause_targets),
                "targets": pause_targets,
            },
            {
                "id": _CONTROL_SCHEDULER_RESUME,
                "label": "Resume scheduled job",
                "description": "Re-enable a paused scheduled job.",
                "subsystem": "scheduler",
                "requires_confirmation": True,
                "confirmation_template": "resume-job:{target}",
                "enabled": bool(scheduler is not None and resume_targets),
                "targets": resume_targets,
            },
        ]
    }


def execute_operator_control(
    control_id: str,
    body: dict[str, Any],
    *,
    provider_registry: ProviderRegistry | None = None,
    scheduler: SchedulerManager | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Execute a confirmed operator control.

    Returns ``(status_code, response_data, audit_detail)``. Callers are
    responsible for attaching actor/source metadata and publishing the audit
    event.
    """
    if control_id == _CONTROL_PROVIDER_SET_DEFAULT:
        return _execute_provider_set_default(body, provider_registry=provider_registry)
    if control_id in {_CONTROL_SCHEDULER_PAUSE, _CONTROL_SCHEDULER_RESUME}:
        return _execute_scheduler_control(control_id, body, scheduler=scheduler)

    detail = _audit_detail(control_id, "unknown", reason="unknown_control")
    return 404, {"message": "Unknown operator control"}, detail


def _execute_provider_set_default(
    body: dict[str, Any],
    *,
    provider_registry: ProviderRegistry | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    control_id = _CONTROL_PROVIDER_SET_DEFAULT
    target = str(body.get("target") or "").strip()
    detail = _audit_detail(control_id, target)
    if provider_registry is None:
        detail["reason"] = "provider_registry_unavailable"
        return 503, {"message": "Provider registry is not attached"}, detail
    if not _SAFE_TARGET_RE.fullmatch(target):
        detail["reason"] = "invalid_target"
        return 400, {"message": "Invalid provider target"}, detail

    expected_confirmation = f"set-default:{target}"
    provided_confirmation = str(body.get("confirm") or "")
    if provided_confirmation != expected_confirmation:
        detail["reason"] = "confirmation_required"
        detail["confirmation_template"] = "set-default:{target}"
        return (
            409,
            {
                "message": "Explicit confirmation is required",
                "confirmation": expected_confirmation,
            },
            detail,
        )

    try:
        names = set(provider_registry.list_providers())
    except Exception as exc:
        detail["reason"] = "provider_list_failed"
        detail["error"] = _safe_error(exc)
        return 503, {"message": "Provider registry is unavailable"}, detail
    if target not in names:
        detail["reason"] = "unknown_provider"
        return 404, {"message": f"Provider {target!r} is not registered"}, detail

    previous = ""
    try:
        previous = provider_registry.get_default_name() or ""
    except Exception:
        previous = ""
    detail["previous"] = previous

    provider = provider_registry.get(target)
    try:
        available = bool(provider is not None and provider.is_available())
    except Exception as exc:
        detail["reason"] = "availability_check_failed"
        detail["error"] = _safe_error(exc)
        return 409, {"message": f"Provider {target!r} availability check failed"}, detail
    if not available:
        detail["reason"] = "provider_unavailable"
        return 409, {"message": f"Provider {target!r} is not available"}, detail

    try:
        provider_registry.set_default(target)
    except Exception as exc:
        detail["reason"] = "set_default_failed"
        detail["error"] = _safe_error(exc)
        return 409, {"message": _safe_error(exc)}, detail

    detail["reason"] = "confirmed"
    detail["current"] = target
    return (
        200,
        {
            "control": control_id,
            "target": target,
            "previous": previous,
            "current": target,
        },
        detail,
    )


def _execute_scheduler_control(
    control_id: str,
    body: dict[str, Any],
    *,
    scheduler: SchedulerManager | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    target = str(body.get("target") or "").strip()
    detail = _audit_detail(control_id, target)
    if scheduler is None:
        detail["reason"] = "scheduler_unavailable"
        return 503, {"message": "Scheduler is not attached"}, detail
    if not _SAFE_TARGET_RE.fullmatch(target):
        detail["reason"] = "invalid_target"
        return 400, {"message": "Invalid scheduler target"}, detail

    action = "pause" if control_id == _CONTROL_SCHEDULER_PAUSE else "resume"
    expected_confirmation = f"{action}-job:{target}"
    provided_confirmation = str(body.get("confirm") or "")
    if provided_confirmation != expected_confirmation:
        detail["reason"] = "confirmation_required"
        detail["confirmation_template"] = f"{action}-job:{{target}}"
        return (
            409,
            {
                "message": "Explicit confirmation is required",
                "confirmation": expected_confirmation,
            },
            detail,
        )

    try:
        jobs = list(scheduler.list_jobs())
    except Exception as exc:
        detail["reason"] = "scheduler_list_failed"
        detail["error"] = _safe_error(exc)
        return 503, {"message": "Scheduler is unavailable"}, detail

    job = next(
        (candidate for candidate in jobs if str(getattr(candidate, "id", "")) == target), None
    )
    if job is None:
        detail["reason"] = "unknown_job"
        return 404, {"message": f"Scheduled job {target!r} is not registered"}, detail

    job_name = str(getattr(job, "name", ""))
    enabled = bool(getattr(job, "enabled", False))
    detail["name"] = job_name
    detail["previous_enabled"] = enabled
    if control_id == _CONTROL_SCHEDULER_PAUSE and not enabled:
        detail["reason"] = "job_already_paused"
        return 409, {"message": f"Scheduled job {target!r} is already paused"}, detail
    if control_id == _CONTROL_SCHEDULER_RESUME and enabled:
        detail["reason"] = "job_already_enabled"
        return 409, {"message": f"Scheduled job {target!r} is already enabled"}, detail

    try:
        if control_id == _CONTROL_SCHEDULER_PAUSE:
            scheduler.pause_job(target)
            current_enabled = False
        else:
            scheduler.resume_job(target)
            current_enabled = True
    except Exception as exc:
        detail["reason"] = "scheduler_mutation_failed"
        detail["error"] = _safe_error(exc)
        return 409, {"message": _safe_error(exc)}, detail

    detail["reason"] = "confirmed"
    detail["current_enabled"] = current_enabled
    return (
        200,
        {
            "control": control_id,
            "target": target,
            "name": job_name,
            "previous_enabled": enabled,
            "current_enabled": current_enabled,
        },
        detail,
    )


def _provider_targets(provider_registry: ProviderRegistry | None) -> list[dict[str, Any]]:
    if provider_registry is None:
        return []
    targets: list[dict[str, Any]] = []
    default_name = ""
    try:
        default_name = provider_registry.get_default_name() or ""
    except Exception:
        default_name = ""
    try:
        names = provider_registry.list_providers()
    except Exception:
        return []
    for name in names:
        provider = provider_registry.get(name)
        available = False
        try:
            available = bool(provider is not None and provider.is_available())
        except Exception:
            available = False
        targets.append(
            {
                "name": name,
                "available": available,
                "is_current": name == default_name,
                "confirmation": f"set-default:{name}",
            }
        )
    return targets


def _scheduler_targets(
    scheduler: SchedulerManager | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if scheduler is None:
        return [], []
    try:
        jobs = list(scheduler.list_jobs())
    except Exception:
        return [], []
    pause_targets: list[dict[str, Any]] = []
    resume_targets: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(getattr(job, "id", ""))
        if not _SAFE_TARGET_RE.fullmatch(job_id):
            continue
        enabled = bool(getattr(job, "enabled", False))
        name = str(getattr(job, "name", "") or job_id)
        schedule = str(getattr(job, "schedule", ""))
        provider = str(getattr(job, "provider", ""))
        base = {
            "name": job_id,
            "label": name,
            "schedule": schedule,
            "provider": provider,
            "available": True,
        }
        pause_targets.append(
            {
                **base,
                "available": enabled,
                "is_current": not enabled,
                "state": "enabled" if enabled else "paused",
                "action_label": "Pause",
                "confirmation": f"pause-job:{job_id}",
            }
        )
        resume_targets.append(
            {
                **base,
                "available": not enabled,
                "is_current": enabled,
                "state": "paused" if not enabled else "enabled",
                "action_label": "Resume",
                "confirmation": f"resume-job:{job_id}",
            }
        )
    return pause_targets, resume_targets


def _audit_detail(control_id: str, target: str, **extra: Any) -> dict[str, Any]:
    subsystem = "control"
    if control_id.startswith("provider."):
        subsystem = "provider"
    elif control_id.startswith("scheduler."):
        subsystem = "scheduler"
    return redact_audit_value(
        {
            "subsystem": subsystem,
            "action": control_id,
            "target": target,
            "severity": "info",
            **extra,
        }
    )


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    return str(redact_audit_value(text))[:240]
