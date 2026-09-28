"""Integrity, schema-gap, and safety checks for curated offline Missy cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2] / "missy/repoeval"
WORKLOAD_ROOT = ROOT / "workloads" / "missy"
SCHEMA = json.loads((ROOT / "schemas" / "workload.schema.json").read_text())
EXPECTED = {
    "repository-orientation": "repository-orientation",
    "tool-call-correctness": "tool-call",
    "patch-test-repair": "patch-generation",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("case_id,task_class", EXPECTED.items())
def test_definition_is_schema_shaped_but_explicitly_non_executable(
    case_id: str, task_class: str
) -> None:
    directory = WORKLOAD_ROOT / case_id
    definition = json.loads((directory / "workload.json").read_text())
    errors = list(Draft202012Validator(SCHEMA).iter_errors(definition))

    # Schema v1 rejects the empty provider list. Its regex accepts the
    # placeholder image syntax, but syntax alone does not authorize an image.
    assert definition["task"]["class"] == task_class
    assert definition["providers"] == []
    assert {"providers"} == {
        "/".join(map(str, error.absolute_path))
        for error in errors
        if list(error.absolute_path) == ["providers"]
    }
    assert all(list(error.absolute_path) == ["providers"] for error in errors)
    assert definition["sandbox"]["image_digest"].startswith("registry.invalid/")


@pytest.mark.parametrize("case_id", EXPECTED)
def test_prompt_and_fixture_digests_match_definition(case_id: str) -> None:
    directory = WORKLOAD_ROOT / case_id
    definition = json.loads((directory / "workload.json").read_text())
    prompt = directory / "prompt.md"
    assert definition["task"]["prompt_sha256"] == _digest(prompt)
    for name, expected_digest in definition["task"]["fixture_digests"].items():
        fixture = directory / "fixtures" / name
        assert fixture.is_file(), name
        assert expected_digest == _digest(fixture), name


def test_three_classes_have_reviewed_source_identity() -> None:
    for case_id in EXPECTED:
        definition = json.loads((WORKLOAD_ROOT / case_id / "workload.json").read_text())
        assert definition["repository"]["repository_id"] == "MissyLabs/missy"
        assert definition["repository"]["commit_sha"] == "cd378928b8609cf48f83cc75393a18086bb2c7a1"
        assert definition["sandbox"]["network_policy"] == "offline"


def test_fixtures_do_not_encode_shell_or_tool_execution() -> None:
    all_text = "\n".join(
        path.read_text()
        for path in WORKLOAD_ROOT.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md", ".py"}
    ).lower()
    assert "shell_exec" not in all_text
    assert "subprocess.run(" not in all_text
    assert "os.system(" not in all_text
    assert "execute it" in (WORKLOAD_ROOT / "tool-call-correctness" / "prompt.md").read_text()


def test_repair_oracle_is_declarative_and_fixtures_are_data_only() -> None:
    repair = WORKLOAD_ROOT / "patch-test-repair" / "fixtures"
    source = (repair / "repair.py").read_text()
    oracle = json.loads((repair / "oracle.json").read_text())
    cases = json.loads((repair / "test_cases.json").read_text())
    assert "return value + percentage" in source
    assert oracle["required_patch"]["return_expression"] == "value * percentage / 100"
    assert len(cases["cases"]) == 4
    assert "shell" not in json.dumps(oracle).lower()


def test_schema_v1_drafts_remain_non_runnable_without_explicit_approval() -> None:
    # The bundled examples have no approved provider; syntax of a placeholder
    # image digest does not convey authority to execute it.
    for case_id in EXPECTED:
        definition = json.loads((WORKLOAD_ROOT / case_id / "workload.json").read_text())
        assert not definition["providers"]
        assert definition["sandbox"]["image_digest"].startswith("registry.invalid/")
