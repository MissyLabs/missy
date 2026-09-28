"""Operator-approved Nomad offload and benchmark workload templates."""

from __future__ import annotations

import copy
import re
import uuid
from typing import Any

from missy.config.settings import NomadConfig
from missy.nomad.errors import NomadValidationError
from missy.nomad.models import NomadJobRequest

_TEMPLATE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


def _template(config: NomadConfig, name: str) -> dict[str, Any]:
    if not _TEMPLATE_NAME.fullmatch(str(name)):
        raise NomadValidationError("workload_template has an invalid name.")
    template = config.workload_templates.get(name)
    if not isinstance(template, dict):
        raise NomadValidationError(
            f"Workload template {name!r} is not explicitly enabled by the operator."
        )
    return template


def build_offload_request(
    config: NomadConfig,
    template_name: str,
    *,
    parameters: dict[str, str] | None = None,
    internal_parameters: dict[str, str] | None = None,
    input_artifacts: dict[str, str] | None = None,
    output_artifacts: dict[str, str] | None = None,
    idempotency_key: str = "",
    job_id: str = "",
    purpose_suffix: str = "",
    result_schema: dict[str, Any] | None = None,
) -> NomadJobRequest:
    """Render only an operator-defined template with constrained caller data."""
    template = _template(config, template_name)
    parameters = parameters or {}
    if not isinstance(parameters, dict):
        raise NomadValidationError("parameters must be an object.")
    allowed = set(template.get("allowed_parameters") or [])
    required = set(template.get("required_parameters") or [])
    supplied = set(parameters)
    if supplied - allowed:
        raise NomadValidationError(
            "Unsupported workload parameter(s): " + ", ".join(sorted(supplied - allowed)) + "."
        )
    if required - supplied:
        raise NomadValidationError(
            "Missing required workload parameter(s): "
            + ", ".join(sorted(required - supplied))
            + "."
        )
    request_data = copy.deepcopy(template["request"])
    forbidden = {"parameters", "input_artifacts", "output_artifacts", "idempotency_key", "job_id"}
    if forbidden & set(request_data):
        raise NomadValidationError(
            f"Configured workload template {template_name!r} contains runtime-only fields."
        )
    rendered_parameters = dict(parameters)
    rendered_parameters.update(internal_parameters or {})
    request_data.update(
        {
            "job_type": "batch",
            "parameters": rendered_parameters,
            "input_artifacts": input_artifacts or {},
            "output_artifacts": output_artifacts or {},
            "idempotency_key": idempotency_key,
            "job_id": job_id,
        }
    )
    if purpose_suffix:
        base = str(request_data.get("purpose") or template_name)
        request_data["purpose"] = f"{base} {purpose_suffix}"
    if result_schema is not None:
        request_data["result_schema"] = result_schema
    request = NomadJobRequest.from_mapping(request_data)
    request.validate(config)
    return request


def new_task_id() -> str:
    return f"task-{uuid.uuid4()}"
