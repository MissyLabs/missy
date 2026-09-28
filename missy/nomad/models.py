"""Nomad workload request validation and safe JSON job construction."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from missy.config.settings import NomadConfig
from missy.nomad.errors import NomadValidationError

_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_DIGEST_RE = re.compile(r"@sha256:[0-9a-fA-F]{64}$")
_SECRET_NAME_RE = re.compile(
    r"(?:secret|token|password|passwd|api[_-]?key|credential|auth(?:orization)?)", re.I
)
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
_SECRET_REF_RE = re.compile(r"^nomad-var://([A-Za-z0-9_./-]{1,240})#([A-Za-z0-9_]{1,64})$")


def _safe_text(value: Any, name: str, *, maximum: int = 512, allow_empty: bool = False) -> str:
    text = str(value or "").strip()
    if not text and not allow_empty:
        raise NomadValidationError(f"{name} must not be empty.")
    if len(text) > maximum or any(char in text for char in "\x00\r"):
        raise NomadValidationError(f"{name} is invalid or exceeds {maximum} characters.")
    return text


def image_registry(image: str) -> str:
    repository = image.split("@", 1)[0]
    first = repository.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first.lower()
    return "docker.io"


@dataclass
class NomadJobRequest:
    """Structured, bounded input for one Missy-owned Nomad job."""

    purpose: str
    image: str
    job_type: str = "service"
    job_id: str = ""
    namespace: str = ""
    node_pool: str = ""
    datacenter: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    secret_references: dict[str, str] = field(default_factory=dict)
    cpu_mhz: int = 500
    memory_mb: int = 512
    disk_mb: int = 256
    count: int = 1
    architecture: str = ""
    required_node_attributes: dict[str, str] = field(default_factory=dict)
    ports: list[dict[str, Any]] = field(default_factory=list)
    service: dict[str, Any] | None = None
    max_run_seconds: int = 3600
    retry_attempts: int = 0
    expected_output: str = ""
    input_artifacts: dict[str, str] = field(default_factory=dict)
    output_artifacts: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)
    result_schema: dict[str, Any] | None = None
    cleanup_policy: str = "retain"
    disposable: bool = False
    external_prerequisites: list[str] = field(default_factory=list)
    idempotency_key: str = ""
    stateful: bool = False
    persistence_plan: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> NomadJobRequest:
        if not isinstance(value, dict):
            raise NomadValidationError("request must be an object.")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - known
        if unknown:
            raise NomadValidationError(f"Unknown workload field(s): {', '.join(sorted(unknown))}.")
        try:
            return cls(**value)
        except TypeError as exc:
            raise NomadValidationError(f"Invalid workload request: {exc}") from exc

    def validate(self, config: NomadConfig, *, allow_unverified_stateful: bool = False) -> None:
        self.purpose = _safe_text(self.purpose, "purpose", maximum=240)
        self.image = _safe_text(self.image, "image", maximum=512)
        self.job_type = _safe_text(self.job_type, "job_type", maximum=16).lower()
        if self.job_type not in {"service", "batch"}:
            raise NomadValidationError("job_type must be 'service' or 'batch'.")
        if self.job_id:
            self.job_id = _safe_text(self.job_id, "job_id", maximum=63).lower()
            if not self.job_id.startswith("missy-") or not _NAME_RE.fullmatch(self.job_id):
                raise NomadValidationError(
                    "job_id must be a DNS-safe name beginning with 'missy-'."
                )
        if config.require_image_digest and not _DIGEST_RE.search(self.image):
            raise NomadValidationError("image must be pinned with an @sha256 digest.")
        registry = image_registry(self.image)
        if config.approved_registries and registry not in {
            item.lower() for item in config.approved_registries
        }:
            raise NomadValidationError(f"Container registry {registry!r} is not approved.")

        for name, value, limit in (
            ("cpu_mhz", self.cpu_mhz, config.max_cpu_mhz),
            ("memory_mb", self.memory_mb, config.max_memory_mb),
            ("disk_mb", self.disk_mb, config.max_disk_mb),
            ("count", self.count, config.max_group_count),
        ):
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise NomadValidationError(f"{name} must be an integer.") from exc
            if parsed <= 0 or parsed > limit:
                raise NomadValidationError(f"{name} must be between 1 and {limit}.")
            setattr(self, name, parsed)
        self.max_run_seconds = int(self.max_run_seconds)
        if (
            self.job_type == "batch"
            and not 1 <= self.max_run_seconds <= config.max_wall_time_seconds
        ):
            raise NomadValidationError(
                f"max_run_seconds must be between 1 and {config.max_wall_time_seconds}."
            )
        self.retry_attempts = int(self.retry_attempts)
        if not 0 <= self.retry_attempts <= config.max_retry_attempts:
            raise NomadValidationError(
                f"retry_attempts must be between 0 and {config.max_retry_attempts}."
            )

        self.command = _safe_text(self.command, "command", maximum=1024, allow_empty=True)
        if self.command and self.command not in config.allowed_job_commands:
            raise NomadValidationError(
                f"command {self.command!r} is not explicitly allowed for Nomad jobs."
            )
        if len(self.args) > 128:
            raise NomadValidationError("args may contain at most 128 entries.")
        self.args = [
            _safe_text(arg, "argument", maximum=4096, allow_empty=True) for arg in self.args
        ]
        if any(_SECRET_NAME_RE.search(arg) for arg in self.args):
            raise NomadValidationError(
                "Secret-shaped command arguments are forbidden; use secret_references."
            )
        if len(self.environment) > 64:
            raise NomadValidationError("environment may contain at most 64 entries.")
        clean_environment: dict[str, str] = {}
        for key, value in self.environment.items():
            key_text = _safe_text(key, "environment key", maximum=128)
            if _SECRET_NAME_RE.search(key_text):
                raise NomadValidationError(
                    f"Secret-shaped environment variable {key_text!r} must use an approved runtime secret mechanism."
                )
            clean_environment[key_text] = _safe_text(
                value, f"environment value for {key_text}", maximum=4096, allow_empty=True
            )
        self.environment = clean_environment
        if not isinstance(self.secret_references, dict) or len(self.secret_references) > 32:
            raise NomadValidationError(
                "secret_references must be an object containing at most 32 entries."
            )
        clean_references: dict[str, str] = {}
        for env_name, reference in self.secret_references.items():
            name = str(env_name).strip()
            ref = str(reference).strip()
            if not _ENV_NAME_RE.fullmatch(name) or not _SECRET_NAME_RE.search(name):
                raise NomadValidationError(
                    "secret_references keys must be secret-shaped uppercase environment names."
                )
            if name in self.environment:
                raise NomadValidationError(
                    f"Secret reference {name!r} conflicts with a plain environment variable."
                )
            if not _SECRET_REF_RE.fullmatch(ref) or not any(
                ref.startswith(prefix) for prefix in config.approved_secret_reference_prefixes
            ):
                raise NomadValidationError(
                    f"Secret reference for {name!r} is outside approved Nomad Variables paths."
                )
            clean_references[name] = ref
        self.secret_references = clean_references
        self.parameters = _validate_string_mapping(
            self.parameters, "parameters", reject_secret_names=True
        )
        self.input_artifacts = _validate_artifacts(self.input_artifacts, "input_artifacts", config)
        self.output_artifacts = _validate_artifacts(
            self.output_artifacts, "output_artifacts", config
        )
        self.result_schema = _validate_result_schema(self.result_schema)
        self.required_node_attributes = _validate_string_mapping(
            self.required_node_attributes, "required_node_attributes"
        )
        self.cleanup_policy = _safe_text(self.cleanup_policy, "cleanup_policy", maximum=32).lower()
        if self.cleanup_policy not in {"retain", "stop_after_verified"}:
            raise NomadValidationError("cleanup_policy must be 'retain' or 'stop_after_verified'.")
        if not isinstance(self.disposable, bool):
            raise NomadValidationError("disposable must be a boolean.")
        if len(self.external_prerequisites) > 16:
            raise NomadValidationError("external_prerequisites may contain at most 16 entries.")
        self.external_prerequisites = [
            _safe_text(item, "external prerequisite", maximum=512)
            for item in self.external_prerequisites
        ]
        self.expected_output = _safe_text(
            self.expected_output, "expected_output", maximum=512, allow_empty=True
        )
        self.idempotency_key = _safe_text(
            self.idempotency_key, "idempotency_key", maximum=128, allow_empty=True
        )
        if self.stateful:
            required = {
                "volume",
                "node_id",
                "backup",
                "restore",
                "migration",
                "data_loss_boundary",
            }
            if not isinstance(self.persistence_plan, dict) or not required.issubset(
                self.persistence_plan
            ):
                raise NomadValidationError(
                    "Stateful jobs require a persistence_plan containing volume, backup, "
                    "restore, migration, and data_loss_boundary."
                )
            if not allow_unverified_stateful:
                raise NomadValidationError(
                    "Stateful job submission is not enabled until the referenced volume is "
                    "verified on eligible nodes; placement discovery may still be used."
                )
        if self.service is not None and self.job_type != "service":
            raise NomadValidationError("service registration is only valid for service jobs.")
        _validate_ports(self.ports)
        _validate_service(self.service, self.ports)


def _validate_string_mapping(
    value: dict[str, str], name: str, *, reject_secret_names: bool = False
) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > 64:
        raise NomadValidationError(f"{name} must be an object containing at most 64 entries.")
    clean: dict[str, str] = {}
    for key, item in value.items():
        clean_key = _safe_text(key, f"{name} key", maximum=128)
        if reject_secret_names and _SECRET_NAME_RE.search(clean_key):
            raise NomadValidationError(
                f"Secret-shaped {name} key {clean_key!r} requires an approved runtime secret mechanism."
            )
        clean[clean_key] = _safe_text(item, f"{name} value for {clean_key}", maximum=2048)
    return clean


def _validate_artifacts(value: dict[str, str], name: str, config: NomadConfig) -> dict[str, str]:
    clean = _validate_string_mapping(value, name)
    for identifier in clean.values():
        if not any(identifier.startswith(prefix) for prefix in config.approved_artifact_prefixes):
            raise NomadValidationError(
                f"{name} identifier {identifier!r} is outside approved artifact storage."
            )
    return clean


def _validate_result_schema(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"required", "properties"}:
        raise NomadValidationError("result_schema accepts only required and properties fields.")
    required = value.get("required", [])
    properties = value.get("properties", {})
    if (
        not isinstance(required, list)
        or len(required) > 32
        or not all(isinstance(item, str) and item for item in required)
    ):
        raise NomadValidationError("result_schema.required must be a list of field names.")
    if not isinstance(properties, dict) or len(properties) > 32:
        raise NomadValidationError("result_schema.properties must be an object.")
    allowed_types = {"string", "number", "integer", "boolean", "object", "array"}
    clean_properties: dict[str, dict[str, str]] = {}
    for name, definition in properties.items():
        field_name = _safe_text(name, "result schema field", maximum=128)
        if not isinstance(definition, dict) or set(definition) != {"type"}:
            raise NomadValidationError("Each result_schema property must contain exactly one type.")
        field_type = str(definition["type"])
        if field_type not in allowed_types:
            raise NomadValidationError(f"Unsupported result schema type {field_type!r}.")
        clean_properties[field_name] = {"type": field_type}
    if not set(required).issubset(clean_properties):
        raise NomadValidationError(
            "result_schema.required fields must appear in result_schema.properties."
        )
    return {"required": list(required), "properties": clean_properties}


def _validate_ports(ports: list[dict[str, Any]]) -> None:
    if not isinstance(ports, list) or len(ports) > 16:
        raise NomadValidationError("ports must be a list containing at most 16 entries.")
    labels: set[str] = set()
    for port in ports:
        if not isinstance(port, dict) or set(port) - {"label", "to", "static"}:
            raise NomadValidationError("Each port accepts only label, to, and static fields.")
        label = _safe_text(port.get("label"), "port label", maximum=32)
        if not _NAME_RE.fullmatch(label) or label in labels:
            raise NomadValidationError(f"Invalid or duplicate port label: {label!r}.")
        labels.add(label)
        for key in ("to", "static"):
            if key in port and port[key] is not None:
                value = int(port[key])
                if not 1 <= value <= 65535:
                    raise NomadValidationError(f"Port {key} must be between 1 and 65535.")
                port[key] = value
        port["label"] = label


def _validate_service(service: dict[str, Any] | None, ports: list[dict[str, Any]]) -> None:
    if service is None:
        return
    if not isinstance(service, dict):
        raise NomadValidationError("service must be an object.")
    allowed = {"name", "port", "check_path", "check_interval_seconds", "check_timeout_seconds"}
    if set(service) - allowed:
        raise NomadValidationError("service contains unsupported fields.")
    labels = {port["label"] for port in ports}
    port_label = _safe_text(service.get("port"), "service port", maximum=32)
    if port_label not in labels:
        raise NomadValidationError("service.port must reference a declared port label.")
    service["name"] = _safe_text(service.get("name"), "service name", maximum=63)
    service["port"] = port_label
    path = _safe_text(service.get("check_path", "/"), "check_path", maximum=512)
    if not path.startswith("/"):
        raise NomadValidationError("service.check_path must begin with '/'.")
    service["check_path"] = path
    interval = int(service.get("check_interval_seconds", 10))
    timeout = int(service.get("check_timeout_seconds", 2))
    if not 2 <= interval <= 3600 or not 1 <= timeout < interval:
        raise NomadValidationError("service check timeout must be positive and below its interval.")
    service["check_interval_seconds"] = interval
    service["check_timeout_seconds"] = timeout


def choose_scope(request: NomadJobRequest, config: NomadConfig) -> tuple[str, str, str]:
    namespace = request.namespace or config.default_namespace
    node_pool = request.node_pool or config.default_node_pool
    datacenter = request.datacenter or config.default_datacenter
    for label, value, allowed in (
        ("namespace", namespace, config.allowed_namespaces),
        ("node_pool", node_pool, config.allowed_node_pools),
        ("datacenter", datacenter, config.allowed_datacenters),
    ):
        if not value:
            raise NomadValidationError(f"No {label} was provided and no default is configured.")
        if value not in allowed:
            raise NomadValidationError(f"{label} {value!r} is outside Missy's authorized scope.")
    request.namespace, request.node_pool, request.datacenter = namespace, node_pool, datacenter
    return namespace, node_pool, datacenter


def build_job(
    request: NomadJobRequest, config: NomadConfig, *, tracking_id: str = ""
) -> dict[str, Any]:
    """Build a minimal Nomad JSON job from a validated structured request."""
    request.validate(config)
    namespace, node_pool, datacenter = choose_scope(request, config)
    tracking_id = tracking_id or str(uuid.uuid4())
    if not request.job_id:
        slug = re.sub(r"[^a-z0-9]+", "-", request.purpose.lower()).strip("-")[:40]
        request.job_id = f"missy-{slug or 'workload'}-{tracking_id[:8]}"

    config_block: dict[str, Any] = {"image": request.image, "cap_drop": ["ALL"]}
    if request.command:
        config_block["command"] = request.command
    if request.args:
        config_block["args"] = request.args
    task_environment = dict(request.environment)
    if request.parameters:
        task_environment["MISSY_PARAMETERS_JSON"] = json.dumps(
            request.parameters, sort_keys=True, separators=(",", ":")
        )
    if request.input_artifacts:
        task_environment["MISSY_INPUT_ARTIFACTS_JSON"] = json.dumps(
            request.input_artifacts, sort_keys=True, separators=(",", ":")
        )
    if request.output_artifacts:
        task_environment["MISSY_OUTPUT_ARTIFACTS_JSON"] = json.dumps(
            request.output_artifacts, sort_keys=True, separators=(",", ":")
        )
    task: dict[str, Any] = {
        "Name": "workload",
        "Driver": "docker",
        "User": "65534:65534",
        "Config": config_block,
        "Env": task_environment or None,
        "Resources": {"CPU": request.cpu_mhz, "MemoryMB": request.memory_mb},
        "LogConfig": {"MaxFiles": 3, "MaxFileSizeMB": 10},
    }
    if request.secret_references:
        templates: list[dict[str, Any]] = []
        for index, (env_name, reference) in enumerate(sorted(request.secret_references.items())):
            match = _SECRET_REF_RE.fullmatch(reference)
            if match is None:  # validate() above guarantees this branch is unreachable
                raise NomadValidationError("Invalid secret reference.")
            variable_path, variable_key = match.groups()
            templates.append(
                {
                    "DestPath": f"secrets/missy-secret-{index}.env",
                    "EmbeddedTmpl": (
                        f'{{{{ with nomadVar "{variable_path}" }}}}'
                        f'{env_name}={{{{ index . "{variable_key}" | toJSON }}}}'
                        "{{ end }}"
                    ),
                    "Envvars": True,
                    "ChangeMode": "restart",
                    "Perms": "0600",
                }
            )
        task["Templates"] = templates
    networks: list[dict[str, Any]] | None = None
    if request.ports:
        dynamic: list[dict[str, Any]] = []
        reserved: list[dict[str, Any]] = []
        for port in request.ports:
            target = reserved if port.get("static") else dynamic
            item: dict[str, Any] = {"Label": port["label"], "To": port.get("to")}
            if port.get("static"):
                item["Value"] = port["static"]
            target.append(item)
        networks = [{"Mode": "bridge", "DynamicPorts": dynamic, "ReservedPorts": reserved}]
    if request.service:
        svc = request.service
        task["Services"] = [
            {
                "Name": svc["name"],
                "PortLabel": svc["port"],
                "Provider": "nomad",
                "Checks": [
                    {
                        "Name": f"{svc['name']}-http",
                        "Type": "http",
                        "Path": svc["check_path"],
                        "Interval": svc["check_interval_seconds"] * 1_000_000_000,
                        "Timeout": svc["check_timeout_seconds"] * 1_000_000_000,
                    }
                ],
            }
        ]
    constraints = None
    if request.architecture:
        constraints = [
            {"LTarget": "${attr.cpu.arch}", "Operand": "=", "RTarget": request.architecture}
        ]
    group: dict[str, Any] = {
        "Name": "workload",
        "Count": request.count,
        "Constraints": constraints,
        "Tasks": [task],
        "EphemeralDisk": {"SizeMB": request.disk_mb},
        "Networks": networks,
    }
    if request.job_type == "batch":
        group["RestartPolicy"] = {
            "Attempts": request.retry_attempts,
            "Interval": 60 * 1_000_000_000,
            "Delay": 5 * 1_000_000_000,
            "Mode": "fail",
        }
        group["ReschedulePolicy"] = {
            "Attempts": request.retry_attempts,
            "Interval": 3600 * 1_000_000_000,
            "Unlimited": False,
        }
        group["MaxRunDuration"] = request.max_run_seconds * 1_000_000_000
    else:
        group["RestartPolicy"] = {
            "Attempts": 3,
            "Interval": 10 * 60 * 1_000_000_000,
            "Delay": 15 * 1_000_000_000,
            "Mode": "delay",
        }
        group["ReschedulePolicy"] = {
            "Attempts": 3,
            "Interval": 60 * 60 * 1_000_000_000,
            "Delay": 30 * 1_000_000_000,
            "DelayFunction": "exponential",
            "MaxDelay": 15 * 60 * 1_000_000_000,
            "Unlimited": False,
        }
        group["Update"] = {
            "MaxParallel": 1,
            "HealthCheck": "checks",
            "MinHealthyTime": 10 * 1_000_000_000,
            "HealthyDeadline": 5 * 60 * 1_000_000_000,
            "ProgressDeadline": 10 * 60 * 1_000_000_000,
            "AutoRevert": False,
            "AutoPromote": False,
        }
    return {
        "Job": {
            "ID": request.job_id,
            "Name": request.job_id,
            "Type": request.job_type,
            "Namespace": namespace,
            "NodePool": node_pool,
            "Datacenters": [datacenter],
            "Meta": {
                "owner": config.owner,
                "managed-by": "missy",
                "tracking-id": tracking_id,
                "purpose": request.purpose,
                "idempotency-key": request.idempotency_key,
            },
            "TaskGroups": [group],
        }
    }


def spec_hash(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
