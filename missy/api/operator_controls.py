"""Policy-shaped operator controls for the Web TUI/API."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from missy.api.audit_browser import redact_audit_value

if TYPE_CHECKING:
    from missy.providers.registry import ProviderRegistry


_CONTROL_PROVIDER_SET_DEFAULT = "provider.set_default"
_SAFE_TARGET_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def list_operator_controls(provider_registry: ProviderRegistry | None = None) -> dict[str, Any]:
    """Return the available operator controls and target state."""
    providers = _provider_targets(provider_registry)
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
            }
        ]
    }


def execute_operator_control(
    control_id: str,
    body: dict[str, Any],
    *,
    provider_registry: ProviderRegistry | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Execute a confirmed operator control.

    Returns ``(status_code, response_data, audit_detail)``. Callers are
    responsible for attaching actor/source metadata and publishing the audit
    event.
    """
    if control_id != _CONTROL_PROVIDER_SET_DEFAULT:
        detail = _audit_detail(control_id, "unknown", reason="unknown_control")
        return 404, {"message": "Unknown operator control"}, detail

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


def _audit_detail(control_id: str, target: str, **extra: Any) -> dict[str, Any]:
    return redact_audit_value(
        {
            "subsystem": "provider" if control_id.startswith("provider.") else "control",
            "action": control_id,
            "target": target,
            "severity": "info",
            **extra,
        }
    )


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip() or type(exc).__name__
    return str(redact_audit_value(text))[:240]
