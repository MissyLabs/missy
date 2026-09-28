"""Bounded, data-only worker for approved RepoEval fixtures.

No checkout, tool, patch, or test execution. External job isolation and signed
approval verification are mandatory; this module cannot provide either.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .contracts import canonical_json
from .evaluation import evaluate_patch_repair, evaluate_tool_call
from .provider import ProviderBroker, ProviderRequest

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}\Z")
_CLASSES = {
    "repository-orientation": ("missy.orientation-facts", {"orientation-facts.json"}),
    "tool-call": ("missy.exact-tool-call", {"calculator-tool.json", "case.json", "oracle.json"}),
    "patch-generation": ("missy.patch-oracle", {"repair.py", "test_cases.json", "oracle.json"}),
}
_REFUSALS = frozenset(
    {
        "invalid_request",
        "invalid_manifest",
        "manifest_digest_mismatch",
        "approval_required",
        "unsupported_manifest",
        "unsupported_validator",
        "payload_set_mismatch",
        "invalid_payload",
        "input_limit",
        "payload_digest_mismatch",
        "timeout_limit",
        "token_limit",
        "budget_limit",
        "invalid_provider_target",
        "provider_not_in_manifest",
    }
)


@dataclass(frozen=True)
class WorkerRequest:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    payloads: Mapping[str, bytes]  # prompt.md and named fixtures, never paths
    project_id: str
    registry_key: str
    model: str
    budget_microusd: int
    timeout_seconds: int
    token_cap: int


@dataclass(frozen=True)
class WorkerResult:
    status: str
    reason: str
    manifest_sha256: str | None = None
    request_sha256: str | None = None
    response_sha256: str | None = None
    report_sha256: str | None = None
    validator_id: str | None = None
    score: float = 0.0

    def report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "manifest_sha256": self.manifest_sha256,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "validator_id": self.validator_id,
            "score": self.score,
        }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: bytes) -> Any:
    return json.loads(
        data.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError())
    )


def _preflight(
    req: WorkerRequest, approve: Callable[[str, Mapping[str, Any]], bool]
) -> tuple[dict, dict[str, bytes], str]:
    if (
        not isinstance(req, WorkerRequest)
        or not isinstance(req.manifest_sha256, str)
        or not _SHA.fullmatch(req.manifest_sha256)
    ):
        raise ValueError("invalid_request")
    try:
        encoded = canonical_json(req.manifest)
        if len(encoded) > 32768 or not isinstance(req.manifest, Mapping):
            raise ValueError("invalid_manifest")
        manifest = _json(encoded)  # detach from mutable external mappings
    except (TypeError, ValueError, OverflowError, UnicodeError):
        raise ValueError("invalid_manifest") from None
    if _sha(encoded) != req.manifest_sha256:
        raise ValueError("manifest_digest_mismatch")
    # Trusted callback checks signed attestation or immutable digest allowlist.
    # An approval field inside the manifest never carries authority.
    try:
        allowed = approve(req.manifest_sha256, manifest) is True
    except Exception:
        allowed = False
    if not allowed:
        raise ValueError("approval_required")
    try:
        task, execution = manifest["task"], manifest["execution"]
        sandbox = manifest["sandbox"]
        klass = task["class"]
        fixtures = task["fixture_digests"]
        validator = manifest["validation"]["validators"]
        if (
            klass not in _CLASSES
            or not isinstance(fixtures, dict)
            or set(fixtures) != _CLASSES[klass][1]
            or not isinstance(validator, list)
            or len(validator) != 1
            or validator[0]["id"] != _CLASSES[klass][0]
            or validator[0]["version"] != "1"
            or validator[0]["required"] is not True
            or not isinstance(execution, dict)
            or any(
                execution[k] != v
                for k, v in {
                    "warmups": 0,
                    "repetitions": 1,
                    "parallelism": 1,
                    "max_attempts": 1,
                }.items()
            )
            or type(execution["timeout_seconds"]) is not int
            or not 1 <= execution["timeout_seconds"] <= 120
            or sandbox["network_policy"] != "offline"
            or not isinstance(sandbox["image_digest"], str)
            or not _IMAGE.fullmatch(sandbox["image_digest"])
            or sandbox["image_digest"].startswith("registry.invalid/")
            or any(
                type(sandbox[key]) is not int or not lo <= sandbox[key] <= hi
                for key, lo, hi in (
                    ("cpu_mhz", 50, 4000),
                    ("memory_mb", 64, 4096),
                    ("disk_mb", 64, 4096),
                )
            )
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest["repository"]["commit_sha"])
        ):
            raise ValueError("unsupported_manifest")
        parameters = {
            "repository-orientation": {"oracle": "fixtures/orientation-facts.json"},
            "tool-call": {"oracle": "fixtures/oracle.json"},
            "patch-generation": {
                "source": "fixtures/repair.py",
                "tests": "fixtures/test_cases.json",
                "oracle": "fixtures/oracle.json",
            },
        }
        if validator[0]["parameters"] != parameters[klass]:
            raise ValueError("unsupported_validator")
        if klass == "tool-call" and task.get("tool_schema_uris") != [
            "workloads/missy/tool-call-correctness/fixtures/calculator-tool.json"
        ]:
            raise ValueError("unsupported_validator")
        if not isinstance(req.payloads, Mapping) or set(req.payloads) != {"prompt.md", *fixtures}:
            raise ValueError("payload_set_mismatch")
        payloads = {}
        total = 0
        for name in sorted(req.payloads):
            data = req.payloads[name]
            if not isinstance(data, bytes):
                raise ValueError("invalid_payload")
            total += len(data)
            if total > 1_000_000:
                raise ValueError("input_limit")
            expected = task["prompt_sha256"] if name == "prompt.md" else fixtures[name]
            if (
                not isinstance(expected, str)
                or not _SHA.fullmatch(expected)
                or _sha(data) != expected
            ):
                raise ValueError("payload_digest_mismatch")
            payloads[name] = data
        if (
            type(req.timeout_seconds) is not int
            or not 1 <= req.timeout_seconds <= execution["timeout_seconds"]
        ):
            raise ValueError("timeout_limit")
        if type(req.token_cap) is not int or not 1 <= req.token_cap <= 8192:
            raise ValueError("token_limit")
        if type(req.budget_microusd) is not int or not 0 <= req.budget_microusd <= 10_000_000:
            raise ValueError("budget_limit")
        if not all(
            isinstance(x, str) and 0 < len(x) <= 128
            for x in (req.project_id, req.registry_key, req.model)
        ):
            raise ValueError("invalid_provider_target")
        # Bundled examples intentionally have providers=[]; not executable as-is.
        providers = manifest["providers"]
        if not isinstance(providers, list) or not any(
            isinstance(x, dict)
            and x.get("registry_key") == req.registry_key
            and x.get("model") == req.model
            for x in providers
        ):
            raise ValueError("provider_not_in_manifest")
        return manifest, payloads, klass
    except ValueError:
        raise
    except (KeyError, TypeError, AttributeError):
        raise ValueError("unsupported_manifest") from None


def _arithmetic(expression: str) -> Fraction:
    """Restricted arithmetic AST, never eval or tool invocation."""
    if not isinstance(expression, str) or len(expression) > 160:
        raise ValueError
    tree = ast.parse(expression, mode="eval")
    nodes = 0

    def visit(node: ast.AST, depth: int = 0) -> Fraction:
        nonlocal nodes
        nodes += 1
        if depth > 8 or nodes > 32:
            raise ValueError
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 1_000_000:
                raise ValueError
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.USub, ast.UAdd):
            value = visit(node.operand, depth + 1)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Div) and not b:
                raise ValueError
            result = (
                a + b
                if isinstance(node.op, ast.Add)
                else a - b
                if isinstance(node.op, ast.Sub)
                else a * b
                if isinstance(node.op, ast.Mult)
                else a / b
            )
            if abs(result) > 1_000_000_000 or result.denominator > 10**15:
                raise ValueError
            return result
        raise ValueError

    return visit(tree.body)


def _operands(expression: str) -> tuple[str, ...]:
    """Require reviewed task operands, not just an arbitrary way to say 45."""
    if not isinstance(expression, str) or len(expression) > 160:
        raise ValueError
    return tuple(
        sorted(
            str(node.value)
            for node in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(node, ast.Constant) and type(node.value) in (int, float)
        )
    )


def _validate(
    klass: str, response: str, manifest: dict, payloads: dict[str, bytes]
) -> tuple[bool, str]:
    if klass == "patch-generation":
        # Inspection is not proof of applied/tested repair. Never execute patches.
        finding = evaluate_patch_repair(
            response, ["repair.py"], {"max_bytes": 64000, "max_files": 1}
        )
        reason = finding["facts"].get("reason", "sandbox_unavailable")
        return False, "sandbox_unavailable" if reason == "oracle_not_proven" else reason
    try:
        candidate = _json(response.encode("utf-8"))
        if klass == "repository-orientation":
            oracle = _json(payloads["orientation-facts.json"])
            if oracle["source_commit"] != manifest["repository"]["commit_sha"]:
                return False, "fixture_identity_mismatch"
            return (
                candidate == oracle,
                "orientation_exact_match" if candidate == oracle else "orientation_mismatch",
            )
        schema = _json(payloads["calculator-tool.json"])
        case = _json(payloads["case.json"])
        oracle = _json(payloads["oracle.json"])
        if (
            not isinstance(candidate, dict)
            or case["expected_call"]["name"] != schema["name"]
            or oracle["checks"][0]["expected"] != schema["name"]
            or case["expected_numeric_result"] != oracle["checks"][2]["expected_value"]
        ):
            return False, "fixture_invalid"
        if evaluate_tool_call(candidate, schema)["status"] != "passed":
            return False, "tool_call_invalid"
        expression = candidate["arguments"]["expression"]
        reference = case["expected_call"]["arguments"]["expression"]
        actual_value = _arithmetic(expression)
        reference_value = _arithmetic(reference)
        correct = (
            _operands(expression) == _operands(reference)
            and actual_value == reference_value
            and reference_value == Fraction(str(case["expected_numeric_result"]))
        )
        return correct, "tool_call_equivalent" if correct else "tool_call_wrong_result"
    except (UnicodeError, TypeError, ValueError, KeyError, IndexError, OverflowError, SyntaxError):
        return False, "invalid_response_or_fixture"


def execute_worker(
    req: WorkerRequest, *, approve: Callable[[str, Mapping[str, Any]], bool], broker: ProviderBroker
) -> WorkerResult:
    """One approved manifest, one provider request, one deterministic validator.

    Outer job must enforce CPU/memory/disk/wallclock limits; Python cannot
    forcibly interrupt a synchronous provider adapter. No raw artifacts saved.
    """
    try:
        manifest, payloads, klass = _preflight(req, approve)
    except Exception as exc:
        reason = (
            str(exc) if type(exc) is ValueError and str(exc) in _REFUSALS else "invalid_request"
        )
        return WorkerResult("refused", reason)
    digest, validator_id = req.manifest_sha256, _CLASSES[klass][0]
    try:
        prompt = payloads["prompt.md"].decode("utf-8")
        tools = (_json(payloads["calculator-tool.json"]),) if klass == "tool-call" else ()
        request = ProviderRequest(
            messages=({"role": "user", "content": prompt},),
            settings={"seed": manifest["execution"]["seed"]},
            tools=tools,
        )
        request_sha = request.digest
    except (UnicodeError, ValueError, TypeError, KeyError):
        return WorkerResult("refused", "invalid_provider_input", digest, validator_id=validator_id)
    started = time.monotonic()
    try:
        answer = broker.execute(
            project_id=req.project_id,
            registry_key=req.registry_key,
            model=req.model,
            request=request,
            budget_microusd=req.budget_microusd,
            timeout_seconds=req.timeout_seconds,
            token_cap=req.token_cap,
        )
    except Exception:
        answer = None  # SDK exceptions may contain secrets; never persist them.
    if answer is None:
        result = WorkerResult(
            "failed", "provider_error", digest, request_sha, validator_id=validator_id
        )
    elif time.monotonic() - started > req.timeout_seconds:
        result = WorkerResult(
            "failed", "deadline_exceeded", digest, request_sha, validator_id=validator_id
        )
    elif (
        not hasattr(answer, "error_category")
        or not hasattr(answer, "content")
        or answer.error_category is not None
        or answer.content is None
    ):
        result = WorkerResult(
            "failed", "provider_failure", digest, request_sha, validator_id=validator_id
        )
    elif (
        not isinstance(answer.content, str)
        or not isinstance(answer.content_sha256, str)
        or not _SHA.fullmatch(answer.content_sha256)
        or getattr(answer, "request_digest", None) != request_sha
        or getattr(answer, "registry_key", None) != req.registry_key
        or getattr(answer, "requested_model", None) != req.model
        or getattr(answer, "content_truncated", True)
        or len(answer.content.encode("utf-8", errors="replace")) > 64000
        or _sha(answer.content.encode("utf-8", errors="replace")) != answer.content_sha256
    ):
        result = WorkerResult(
            "failed", "response_integrity_failed", digest, request_sha, validator_id=validator_id
        )
    else:
        ok, reason = _validate(klass, answer.content, manifest, payloads)
        result = WorkerResult(
            "passed" if ok else "failed",
            reason,
            digest,
            request_sha,
            answer.content_sha256,
            validator_id=validator_id,
            score=1.0 if ok else 0.0,
        )
    return WorkerResult(
        **{**result.__dict__, "report_sha256": _sha(canonical_json(result.report()))}
    )
