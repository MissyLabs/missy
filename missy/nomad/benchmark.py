"""Validation and expansion of reproducible Nomad benchmark definitions."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from missy.config.settings import NomadConfig
from missy.nomad.errors import NomadValidationError
from missy.nomad.workloads import build_offload_request

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/@:-]{0,255}$")


@dataclass
class BenchmarkDefinition:
    """A bounded, operator-templated benchmark matrix."""

    name: str
    workload_template: str
    workload_version: str
    input_artifacts: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, str] = field(default_factory=dict)
    output_prefix: str = ""
    result_schema: dict[str, Any] | None = None
    warmup_runs: int = 0
    repetitions: int = 1
    parallelism: int = 1
    idempotency_key: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> BenchmarkDefinition:
        if not isinstance(value, dict):
            raise NomadValidationError("benchmark definition must be an object.")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - known
        if unknown:
            raise NomadValidationError(
                "Unknown benchmark field(s): " + ", ".join(sorted(unknown)) + "."
            )
        try:
            return cls(**value)
        except TypeError as exc:
            raise NomadValidationError(f"Invalid benchmark definition: {exc}") from exc

    def validate(self, config: NomadConfig) -> None:
        self.name = str(self.name).strip()
        if not self.name or len(self.name) > 128 or any(c in self.name for c in "\x00\r\n"):
            raise NomadValidationError("benchmark name is invalid.")
        self.workload_version = str(self.workload_version).strip()
        if not _VERSION_RE.fullmatch(self.workload_version):
            raise NomadValidationError(
                "workload_version must be an immutable version, digest, or commit identifier."
            )
        for field_name in ("warmup_runs", "repetitions", "parallelism"):
            try:
                setattr(self, field_name, int(getattr(self, field_name)))
            except (TypeError, ValueError) as exc:
                raise NomadValidationError(f"{field_name} must be an integer.") from exc
        if not 0 <= self.warmup_runs <= config.max_benchmark_runs:
            raise NomadValidationError("warmup_runs exceeds the configured benchmark limit.")
        total = self.warmup_runs + self.repetitions
        if self.repetitions < 1 or total > config.max_benchmark_runs:
            raise NomadValidationError(
                f"warmups plus repetitions must be between 1 and {config.max_benchmark_runs}."
            )
        if not 1 <= self.parallelism <= config.max_benchmark_parallelism:
            raise NomadValidationError(
                f"parallelism must be between 1 and {config.max_benchmark_parallelism}."
            )
        if not self.output_prefix:
            raise NomadValidationError(
                "output_prefix is required so benchmark results use durable approved storage."
            )
        if not any(
            self.output_prefix.startswith(prefix) for prefix in config.approved_artifact_prefixes
        ):
            raise NomadValidationError("output_prefix is outside approved artifact storage.")

    def requests(self, config: NomadConfig, benchmark_id: str) -> list[dict[str, Any]]:
        self.validate(config)
        result: list[dict[str, Any]] = []
        total = self.warmup_runs + self.repetitions
        slug = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")[:24] or "benchmark"
        for index in range(total):
            warmup = index < self.warmup_runs
            run_number = index + 1 if warmup else index - self.warmup_runs + 1
            kind = "warmup" if warmup else "measured"
            job_id = f"missy-bench-{slug}-{benchmark_id[-8:]}-{index + 1}"
            internal_parameters = {
                "missy_benchmark_id": benchmark_id,
                "missy_workload_version": self.workload_version,
                "missy_run_kind": kind,
                "missy_run_number": str(run_number),
            }
            output = {
                "result": f"{self.output_prefix.rstrip('/')}/{benchmark_id}/{kind}-{run_number}.json"
            }
            request = build_offload_request(
                config,
                self.workload_template,
                parameters=self.parameters,
                internal_parameters=internal_parameters,
                input_artifacts=self.input_artifacts,
                output_artifacts=output,
                idempotency_key=(
                    f"{self.idempotency_key}:{kind}:{run_number}"
                    if self.idempotency_key
                    else f"benchmark:{benchmark_id}:{kind}:{run_number}"
                ),
                job_id=job_id,
                purpose_suffix=f"{kind} {run_number}",
                result_schema=self.result_schema,
            )
            result.append(
                {
                    "kind": kind,
                    "run_number": run_number,
                    "request": dict(request.__dict__),
                }
            )
        return result


def new_benchmark_id() -> str:
    return f"benchmark-{uuid.uuid4()}"
