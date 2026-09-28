"""Bounded offline evaluators. Tool calls and patches are never executed here."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator, SchemaError


def _result(name: str, ok: bool, **facts: Any) -> dict:
    return {
        "id": name,
        "status": "passed" if ok else "failed",
        "score": 1.0 if ok else 0.0,
        "facts": facts,
    }


def evaluate_orientation(response: Any, expected_facts: Mapping[str, Any]) -> dict:
    try:
        parsed = json.loads(response) if isinstance(response, str) else response
    except (ValueError, TypeError):
        return _result("orientation-facts", False, reason="invalid_json")
    if not isinstance(parsed, dict):
        return _result("orientation-facts", False, reason="not_object")
    matched = sorted(k for k, v in expected_facts.items() if parsed.get(k) == v)
    missing = sorted(set(expected_facts) - set(matched))
    return _result(
        "orientation-facts",
        not missing,
        matched=matched,
        missing=missing,
        expected_count=len(expected_facts),
    )


def _contains_ref(value: Any) -> bool:
    """Reject references rather than permit untrusted schemas to fetch remote resources."""
    if isinstance(value, Mapping):
        return (
            "$ref" in value
            or "$dynamicRef" in value
            or any(_contains_ref(x) for x in value.values())
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_ref(x) for x in value)
    return False


def evaluate_tool_call(response: Any, tool_schema: Mapping[str, Any]) -> dict:
    try:
        call = (
            json.loads(response, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if isinstance(response, str)
            else response
        )
    except (ValueError, TypeError):
        return _result("tool-call", False, reason="invalid_json", executed=False)
    if (
        not isinstance(call, dict)
        or set(call) != {"name", "arguments"}
        or not isinstance(call["name"], str)
    ):
        return _result("tool-call", False, reason="invalid_envelope", executed=False)
    if not isinstance(tool_schema, Mapping) or not isinstance(tool_schema.get("name"), str):
        return _result("tool-call", False, reason="invalid_tool_schema", executed=False)
    # Provider-neutral contracts use input_schema; reviewed Missy fixtures use
    # parameters. Never silently default a missing schema to permissive {}.
    schemas = [tool_schema[k] for k in ("input_schema", "parameters") if k in tool_schema]
    if len(schemas) != 1 or not isinstance(schemas[0], Mapping) or _contains_ref(schemas[0]):
        return _result("tool-call", False, reason="invalid_tool_schema", executed=False)
    try:
        Draft202012Validator.check_schema(schemas[0])
        arguments_valid = Draft202012Validator(schemas[0]).is_valid(call["arguments"])
    except (SchemaError, TypeError, ValueError):
        return _result("tool-call", False, reason="invalid_tool_schema", executed=False)
    selected = call["name"] == tool_schema["name"]
    return _result(
        "tool-call",
        selected and arguments_valid,
        selected=call["name"],
        arguments_valid=arguments_valid,
        executed=False,
    )


def _safe_path(path: str) -> bool:
    return (
        bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and all(part not in ("", ".", "..") for part in path.split("/"))
        and not any(ord(c) < 32 or ord(c) == 127 for c in path)
    )


def evaluate_patch_repair(
    patch: str, expected_files: list[str], limits: Mapping[str, int] | None = None
) -> dict:
    """Inspect paths only. A diff is not evidence of an applied or tested repair."""
    limits = dict(limits or {})
    max_bytes = min(limits.get("max_bytes", 200000), 1000000)
    max_files = min(limits.get("max_files", 32), 128)
    if not isinstance(patch, str) or len(patch.encode("utf-8")) > max_bytes:
        return _result(
            "patch-repair",
            False,
            reason="size_limit",
            applied=False,
            tested=False,
            repair_verified=False,
            commands_executed=False,
        )
    if "\x00" in patch:
        return _result(
            "patch-repair",
            False,
            reason="nul_byte",
            applied=False,
            tested=False,
            repair_verified=False,
            commands_executed=False,
        )
    paths = []
    for line in patch.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].split("\t", 1)[0]
            if path == "/dev/null":
                continue
            path = path.removeprefix("a/").removeprefix("b/")
            if not _safe_path(path):
                return _result(
                    "patch-repair",
                    False,
                    reason="unsafe_path",
                    applied=False,
                    tested=False,
                    repair_verified=False,
                    commands_executed=False,
                )
            paths.append(path)
    files = sorted(set(paths))
    facts = {
        "files": files,
        "applied": False,
        "tested": False,
        "repair_verified": False,
        "commands_executed": False,
    }
    if len(files) > max_files:
        return _result("patch-repair", False, reason="file_limit", files_count=len(files), **facts)
    if not files:
        return _result("patch-repair", False, reason="no_diff_files", **facts)
    unexpected = sorted(set(files) - set(expected_files))
    if unexpected:
        return _result(
            "patch-repair", False, reason="unexpected_files", unexpected_files=unexpected, **facts
        )
    return _result("patch-repair", False, reason="oracle_not_proven", unexpected_files=[], **facts)
