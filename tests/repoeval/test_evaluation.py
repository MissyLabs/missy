import json
import unittest
from pathlib import Path

from missy.repoeval.contracts import comparability_key, compare_runs
from missy.repoeval.evaluation import (
    evaluate_orientation,
    evaluate_patch_repair,
    evaluate_tool_call,
)
from missy.repoeval.report import build_comparison_report


class EvaluationTests(unittest.TestCase):
    def test_orientation_exact_expected_facts(self):
        self.assertEqual(
            evaluate_orientation('{"entry":"src/main.py"}', {"entry": "src/main.py"})["status"],
            "passed",
        )
        self.assertEqual(evaluate_orientation("no", {})["status"], "failed")

    def test_tool_call_validated_not_executed(self):
        schema = {
            "name": "lookup",
            "input_schema": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer"}},
                "additionalProperties": False,
            },
        }
        out = evaluate_tool_call({"name": "lookup", "arguments": {"id": 2}}, schema)
        self.assertEqual(out["status"], "passed")
        self.assertFalse(out["facts"]["executed"])
        self.assertEqual(
            evaluate_tool_call({"name": "lookup", "arguments": {"id": True}}, schema)["status"],
            "failed",
        )

    def test_tool_call_schema_constraints_and_invalid_schemas(self):
        schema = {
            "name": "lookup",
            "input_schema": {
                "type": "object",
                "required": ["mode", "items"],
                "additionalProperties": False,
                "properties": {
                    "mode": {"type": "string", "enum": ["read"]},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "items": {"type": "integer", "minimum": 1},
                    },
                },
            },
        }
        valid = {"name": "lookup", "arguments": {"mode": "read", "items": [1]}}
        self.assertEqual(evaluate_tool_call(valid, schema)["status"], "passed")
        for bad in (
            {"mode": "write", "items": [1]},
            {"mode": "read", "items": []},
            {"mode": "read", "items": [0]},
            {"mode": "read", "items": [1, 2, 3]},
            {"mode": "read", "items": [1], "extra": 1},
        ):
            self.assertEqual(
                evaluate_tool_call({**valid, "arguments": bad}, schema)["status"], "failed"
            )
        self.assertEqual(
            evaluate_tool_call(valid, {"name": "lookup"})["facts"]["reason"], "invalid_tool_schema"
        )
        self.assertEqual(
            evaluate_tool_call(valid, {"name": "lookup", "input_schema": {"type": "impossible"}})[
                "facts"
            ]["reason"],
            "invalid_tool_schema",
        )
        self.assertEqual(
            evaluate_tool_call(
                valid,
                {"name": "lookup", "input_schema": {"$ref": "https://example.invalid/schema"}},
            )["facts"]["reason"],
            "invalid_tool_schema",
        )
        self.assertEqual(evaluate_tool_call({**valid, "extra": 1}, schema)["status"], "failed")
        self.assertEqual(
            evaluate_tool_call('{"name":"lookup","arguments":NaN}', schema)["status"], "failed"
        )

    def test_reviewed_fixture_parameter_schema_is_enforced(self):
        fixture = (
            Path(__file__).resolve().parents[2]
            / "missy/repoeval/workloads/missy/tool-call-correctness/fixtures/calculator-tool.json"
        )
        schema = json.loads(fixture.read_text())
        self.assertEqual(
            evaluate_tool_call(
                {"name": "calculator", "arguments": {"expression": "250 * 18 / 100"}}, schema
            )["status"],
            "passed",
        )
        self.assertEqual(
            evaluate_tool_call({"name": "calculator", "arguments": {}}, schema)["status"], "failed"
        )
        self.assertEqual(
            evaluate_tool_call({"name": "calculator", "arguments": {"expression": 12}}, schema)[
                "status"
            ],
            "failed",
        )

    def test_patch_paths_allowlisted_bounded_and_never_executes(self):
        patch = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n"
        out = evaluate_patch_repair(patch, ["src/a.py"])
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["score"], 0.0)
        self.assertEqual(out["facts"]["reason"], "oracle_not_proven")
        self.assertFalse(out["facts"]["commands_executed"])
        self.assertFalse(out["facts"]["applied"])
        self.assertFalse(out["facts"]["tested"])
        self.assertFalse(out["facts"]["repair_verified"])
        self.assertEqual(evaluate_patch_repair(patch, ["other.py"])["status"], "failed")
        self.assertEqual(
            evaluate_patch_repair("--- a/../escape\n+++ b/../escape\n", ["../escape"])["facts"][
                "reason"
            ],
            "unsafe_path",
        )
        self.assertEqual(
            evaluate_patch_repair("x" * 100, ["x"], {"max_bytes": 10})["facts"]["reason"],
            "size_limit",
        )

    def test_patch_noop_malicious_and_unverified_never_score_repair(self):
        cases = (
            "--- a/src/a.py\n+++ b/src/a.py\n",  # no hunk
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+old\n",  # no-op
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-old\n+new\n",  # untested
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-safe\n+__import__('os').system('echo bad')\n",
        )
        for patch in cases:
            out = evaluate_patch_repair(patch, ["src/a.py"])
            self.assertEqual(out["status"], "failed")
            self.assertEqual(out["score"], 0.0)
            self.assertFalse(out["facts"]["commands_executed"])
        self.assertEqual(
            evaluate_patch_repair("--- a/src/./a.py\n+++ b/src/./a.py\n", ["src/a.py"])["facts"][
                "reason"
            ],
            "unsafe_path",
        )
        self.assertEqual(
            evaluate_patch_repair("--- a/src/a.py\n+++ b/src/a.py\x1b\n", ["src/a.py"])["facts"][
                "reason"
            ],
            "unsafe_path",
        )

    def test_report_splits_and_marks_incomparable(self):
        base = {
            "comparability_rules_version": "1.0",
            "repository": {"repository_id": "r", "commit_sha": "a", "snapshot_id": "s"},
            "task": {"class": "x"},
            "workload_id": "w",
            "workload_version": "1",
            "definition_sha256": "c",
            "sandbox": {
                "image_digest": "img",
                "network_policy": "n",
                "cpu_mhz": 100,
                "memory_mb": 128,
                "disk_mb": 256,
                "timeout_seconds": 10,
            },
            "evaluator_version": "e",
            "validators": [],
            "provider_independent_settings_sha256": "f",
            "tool_schemas_sha256": "g",
        }
        key = comparability_key(base)
        a = {**base, "run_id": "run-a", "comparability_key": key}
        b = {**base, "run_id": "run-b", "comparability_key": key, "provider": {"model": "other"}}
        c = {
            **base,
            "run_id": "run-c",
            "comparability_key": "0" * 64,
            "repository": {"repository_id": "r", "commit_sha": "different", "snapshot_id": "s"},
        }
        report = build_comparison_report([c, b, a])
        self.assertEqual(report["groups"][0]["run_ids"], ["run-a", "run-b"])
        self.assertEqual(report["incomparable"][0]["run_id"], "run-c")
        self.assertFalse(compare_runs(a, c).comparable)
