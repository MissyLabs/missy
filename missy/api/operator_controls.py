"""Policy-shaped operator controls for the Web TUI/API."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from missy.api.audit_browser import redact_audit_value

if TYPE_CHECKING:
    from missy.providers.registry import ProviderRegistry
    from missy.scheduler.manager import SchedulerManager
    from missy.tools.benchmark.benchmark_store import BenchmarkStore
    from missy.tools.intelligence.candidate_store import CandidateStore


_CONTROL_PROVIDER_SET_DEFAULT = "provider.set_default"
_CONTROL_SCHEDULER_PAUSE = "scheduler.pause_job"
_CONTROL_SCHEDULER_RESUME = "scheduler.resume_job"
_CONTROL_SCHEDULER_REMOVE = "scheduler.remove_job"
_CONTROL_CANDIDATE_IMPORT_BENCHMARKS = "tool_candidate.import_benchmarks"
_CONTROL_CANDIDATE_APPROVE = "tool_candidate.approve"
_CONTROL_CANDIDATE_ENABLE = "tool_candidate.enable"
_CONTROL_CANDIDATE_DENY = "tool_candidate.deny"
_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def list_operator_controls(
    provider_registry: ProviderRegistry | None = None,
    scheduler: SchedulerManager | None = None,
    candidate_store: CandidateStore | None = None,
) -> dict[str, Any]:
    """Return the available operator controls and target state."""
    providers = _provider_targets(provider_registry)
    pause_targets, resume_targets = _scheduler_targets(scheduler)
    remove_targets = _scheduler_remove_targets(scheduler)
    candidate_targets = _candidate_targets(candidate_store)
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
            {
                "id": _CONTROL_SCHEDULER_REMOVE,
                "label": "Remove scheduled job",
                "description": "Permanently delete a scheduled job.",
                "subsystem": "scheduler",
                "requires_confirmation": True,
                "confirmation_template": "remove-job:{target}",
                "destructive": True,
                "enabled": bool(scheduler is not None and remove_targets),
                "targets": remove_targets,
            },
            {
                "id": _CONTROL_CANDIDATE_IMPORT_BENCHMARKS,
                "label": "Import candidate benchmarks",
                "description": "Copy stored benchmark summaries into a tool candidate review record.",
                "subsystem": "tool_candidate",
                "requires_confirmation": True,
                "confirmation_template": "import-candidate-benchmarks:{target}",
                "enabled": bool(candidate_store is not None and candidate_targets["benchmarkable"]),
                "targets": candidate_targets["benchmarkable"],
            },
            {
                "id": _CONTROL_CANDIDATE_APPROVE,
                "label": "Approve tool candidate",
                "description": "Approve a benchmarked candidate without enabling runtime use.",
                "subsystem": "tool_candidate",
                "requires_confirmation": True,
                "confirmation_template": "approve-candidate:{target}",
                "enabled": bool(candidate_store is not None and candidate_targets["approvable"]),
                "targets": candidate_targets["approvable"],
            },
            {
                "id": _CONTROL_CANDIDATE_ENABLE,
                "label": "Enable tool candidate",
                "description": "Enable an approved candidate for future controlled runtime loading.",
                "subsystem": "tool_candidate",
                "requires_confirmation": True,
                "confirmation_template": "enable-candidate:{target}",
                "enabled": bool(candidate_store is not None and candidate_targets["enableable"]),
                "targets": candidate_targets["enableable"],
            },
            {
                "id": _CONTROL_CANDIDATE_DENY,
                "label": "Deny tool candidate",
                "description": "Disable a candidate with an operator review reason.",
                "subsystem": "tool_candidate",
                "requires_confirmation": True,
                "confirmation_template": "deny-candidate:{target}",
                "destructive": True,
                "enabled": bool(candidate_store is not None and candidate_targets["denyable"]),
                "targets": candidate_targets["denyable"],
            },
        ]
    }


def execute_operator_control(
    control_id: str,
    body: dict[str, Any],
    *,
    provider_registry: ProviderRegistry | None = None,
    scheduler: SchedulerManager | None = None,
    candidate_store: CandidateStore | None = None,
    benchmark_store: BenchmarkStore | None = None,
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
    if control_id == _CONTROL_SCHEDULER_REMOVE:
        return _execute_scheduler_remove(body, scheduler=scheduler)
    if control_id == _CONTROL_CANDIDATE_IMPORT_BENCHMARKS:
        return _execute_candidate_import_benchmarks(
            body,
            candidate_store=candidate_store,
            benchmark_store=benchmark_store,
        )
    if control_id == _CONTROL_CANDIDATE_APPROVE:
        return _execute_candidate_transition(
            control_id,
            body,
            candidate_store=candidate_store,
            target_state_value="approved",
            confirmation_prefix="approve-candidate",
        )
    if control_id == _CONTROL_CANDIDATE_ENABLE:
        return _execute_candidate_transition(
            control_id,
            body,
            candidate_store=candidate_store,
            target_state_value="enabled",
            confirmation_prefix="enable-candidate",
        )
    if control_id == _CONTROL_CANDIDATE_DENY:
        return _execute_candidate_transition(
            control_id,
            body,
            candidate_store=candidate_store,
            target_state_value="disabled",
            confirmation_prefix="deny-candidate",
            require_notes=True,
        )

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


def _execute_scheduler_remove(
    body: dict[str, Any],
    *,
    scheduler: SchedulerManager | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    control_id = _CONTROL_SCHEDULER_REMOVE
    target = str(body.get("target") or "").strip()
    detail = _audit_detail(control_id, target)
    if scheduler is None:
        detail["reason"] = "scheduler_unavailable"
        return 503, {"message": "Scheduler is not attached"}, detail
    if not _SAFE_TARGET_RE.fullmatch(target):
        detail["reason"] = "invalid_target"
        return 400, {"message": "Invalid scheduler target"}, detail

    expected_confirmation = f"remove-job:{target}"
    provided_confirmation = str(body.get("confirm") or "")
    if provided_confirmation != expected_confirmation:
        detail["reason"] = "confirmation_required"
        detail["confirmation_template"] = "remove-job:{target}"
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
    detail["name"] = job_name

    try:
        scheduler.remove_job(target)
    except Exception as exc:
        detail["reason"] = "scheduler_mutation_failed"
        detail["error"] = _safe_error(exc)
        return 409, {"message": _safe_error(exc)}, detail

    detail["reason"] = "confirmed"
    return (
        200,
        {
            "control": control_id,
            "target": target,
            "name": job_name,
            "removed": True,
        },
        detail,
    )


def _execute_candidate_import_benchmarks(
    body: dict[str, Any],
    *,
    candidate_store: CandidateStore | None,
    benchmark_store: BenchmarkStore | None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    from missy.tools.intelligence import CandidateBenchmarkReconciler

    control_id = _CONTROL_CANDIDATE_IMPORT_BENCHMARKS
    target = str(body.get("target") or "").strip()
    detail = _audit_detail(control_id, target)
    if candidate_store is None:
        detail["reason"] = "candidate_store_unavailable"
        return 503, {"message": "Candidate store is not attached"}, detail
    if benchmark_store is None:
        detail["reason"] = "benchmark_store_unavailable"
        return 503, {"message": "Benchmark store is not attached"}, detail
    if not _SAFE_TARGET_RE.fullmatch(target):
        detail["reason"] = "invalid_target"
        return 400, {"message": "Invalid candidate target"}, detail

    expected_confirmation = f"import-candidate-benchmarks:{target}"
    provided_confirmation = str(body.get("confirm") or "")
    if provided_confirmation != expected_confirmation:
        detail["reason"] = "confirmation_required"
        detail["confirmation_template"] = "import-candidate-benchmarks:{target}"
        return (
            409,
            {
                "message": "Explicit confirmation is required",
                "confirmation": expected_confirmation,
            },
            detail,
        )

    try:
        reconciler = CandidateBenchmarkReconciler(
            candidate_store=candidate_store,
            benchmark_store=benchmark_store,
            min_samples=int(body.get("min_samples") or 3),
            min_composite=float(body.get("min_composite") or 0.4),
            min_safety=float(body.get("min_safety") or 1.0),
            min_schema_score=float(body.get("min_schema_score") or 0.8),
        )
    except (TypeError, ValueError) as exc:
        detail["reason"] = "invalid_threshold"
        detail["error"] = _safe_error(exc)
        return 400, {"message": "Invalid benchmark threshold"}, detail

    try:
        result = reconciler.reconcile_candidate(
            target,
            tool_name=str(body.get("tool_name") or "").strip() or None,
            actor="operator",
        )
    except ValueError as exc:
        detail["reason"] = "benchmark_data_missing"
        detail["error"] = _safe_error(exc)
        return 404, {"message": _safe_error(exc)}, detail
    except Exception as exc:  # noqa: BLE001
        detail["reason"] = "benchmark_import_failed"
        detail["error"] = _safe_error(exc)
        return 409, {"message": _safe_error(exc)}, detail

    if result is None:
        detail["reason"] = "unknown_candidate"
        return 404, {"message": f"Candidate {target!r} not found"}, detail

    candidate = result.candidate
    decisions = [
        {
            "provider": d.provider,
            "enabled": d.enabled,
            "reason": d.reason,
            "run_count": d.run_count,
            "composite": d.summary.composite,
        }
        for d in result.decisions
    ]
    detail.update(
        {
            "reason": "confirmed",
            "name": candidate.name,
            "state": candidate.state.value,
            "providers": decisions,
        }
    )
    return (
        200,
        {
            "control": control_id,
            "target": target,
            "candidate": candidate.to_dict(),
            "decisions": decisions,
        },
        detail,
    )


def _execute_candidate_transition(
    control_id: str,
    body: dict[str, Any],
    *,
    candidate_store: CandidateStore | None,
    target_state_value: str,
    confirmation_prefix: str,
    require_notes: bool = False,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    from missy.tools.intelligence import ToolLifecycleState

    target = str(body.get("target") or "").strip()
    detail = _audit_detail(control_id, target)
    if candidate_store is None:
        detail["reason"] = "candidate_store_unavailable"
        return 503, {"message": "Candidate store is not attached"}, detail
    if not _SAFE_TARGET_RE.fullmatch(target):
        detail["reason"] = "invalid_target"
        return 400, {"message": "Invalid candidate target"}, detail

    expected_confirmation = f"{confirmation_prefix}:{target}"
    provided_confirmation = str(body.get("confirm") or "")
    if provided_confirmation != expected_confirmation:
        detail["reason"] = "confirmation_required"
        detail["confirmation_template"] = f"{confirmation_prefix}:{{target}}"
        return (
            409,
            {
                "message": "Explicit confirmation is required",
                "confirmation": expected_confirmation,
            },
            detail,
        )

    notes = str(body.get("notes") or body.get("reason") or "").strip()
    if require_notes and not notes:
        detail["reason"] = "review_reason_required"
        return 400, {"message": "A review reason is required"}, detail

    try:
        target_state = ToolLifecycleState(target_state_value)
        updated = candidate_store.transition(
            target,
            target_state,
            notes=notes,
            actor="operator",
        )
    except ValueError as exc:
        detail["reason"] = "invalid_lifecycle_transition"
        detail["error"] = _safe_error(exc)
        return 409, {"message": _safe_error(exc)}, detail
    except Exception as exc:  # noqa: BLE001
        detail["reason"] = "candidate_transition_failed"
        detail["error"] = _safe_error(exc)
        return 409, {"message": _safe_error(exc)}, detail

    if updated is None:
        detail["reason"] = "unknown_candidate"
        return 404, {"message": f"Candidate {target!r} not found"}, detail

    detail.update(
        {
            "reason": "confirmed",
            "name": updated.name,
            "state": updated.state.value,
            "notes": notes,
        }
    )
    return (
        200,
        {
            "control": control_id,
            "target": target,
            "candidate": updated.to_dict(),
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


def _candidate_targets(candidate_store: CandidateStore | None) -> dict[str, list[dict[str, Any]]]:
    if candidate_store is None:
        return {"benchmarkable": [], "approvable": [], "enableable": [], "denyable": []}
    try:
        candidates = candidate_store.list_all(limit=200)
    except Exception:
        return {"benchmarkable": [], "approvable": [], "enableable": [], "denyable": []}

    targets = {"benchmarkable": [], "approvable": [], "enableable": [], "denyable": []}
    for candidate in candidates:
        base = {
            "name": candidate.id,
            "label": candidate.name,
            "candidate_name": candidate.name,
            "state": candidate.state.value,
            "owner": candidate.owner,
            "available": True,
            "is_current": False,
        }
        if candidate.state.value in {"proposed", "experimental"}:
            targets["benchmarkable"].append(
                {
                    **base,
                    "action_label": "Import",
                    "confirmation": f"import-candidate-benchmarks:{candidate.id}",
                }
            )
        if candidate.state.value == "benchmarked":
            targets["approvable"].append(
                {
                    **base,
                    "action_label": "Approve",
                    "confirmation": f"approve-candidate:{candidate.id}",
                }
            )
        if candidate.state.value == "approved":
            targets["enableable"].append(
                {
                    **base,
                    "action_label": "Enable",
                    "confirmation": f"enable-candidate:{candidate.id}",
                }
            )
        if candidate.state.value != "disabled":
            targets["denyable"].append(
                {
                    **base,
                    "action_label": "Deny",
                    "confirmation": f"deny-candidate:{candidate.id}",
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


def _scheduler_remove_targets(scheduler: SchedulerManager | None) -> list[dict[str, Any]]:
    if scheduler is None:
        return []
    try:
        jobs = list(scheduler.list_jobs())
    except Exception:
        return []
    targets: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(getattr(job, "id", ""))
        if not _SAFE_TARGET_RE.fullmatch(job_id):
            continue
        name = str(getattr(job, "name", "") or job_id)
        schedule = str(getattr(job, "schedule", ""))
        provider = str(getattr(job, "provider", ""))
        targets.append(
            {
                "name": job_id,
                "label": name,
                "schedule": schedule,
                "provider": provider,
                "available": True,
                "is_current": False,
                "state": "enabled" if getattr(job, "enabled", False) else "paused",
                "action_label": "Remove",
                "confirmation": f"remove-job:{job_id}",
            }
        )
    return targets


def _audit_detail(control_id: str, target: str, **extra: Any) -> dict[str, Any]:
    subsystem = "control"
    if control_id.startswith("provider."):
        subsystem = "provider"
    elif control_id.startswith("scheduler."):
        subsystem = "scheduler"
    elif control_id.startswith("tool_candidate."):
        subsystem = "tool_candidate"
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
