import json
import math
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from missy.repoeval.contracts import (
    canonical_json,
    canonical_provider_input,
    comparability_key,
    compare_runs,
    definition_hash,
)


def manifest(**changes):
    value = {
        "comparability_rules_version": "1.0",
        "repository": {"repository_id": "r", "commit_sha": "a" * 40, "snapshot_id": "s"},
        "task": {
            "class": "orientation",
            "prompt_sha256": "b" * 64,
            "fixture_digests": {},
            "tool_schema_uris": [],
        },
        "workload_id": "w",
        "workload_version": "1",
        "definition_sha256": "c" * 64,
        "sandbox": {
            "image_digest": "img@sha256:" + "d" * 64,
            "network_policy": "none",
            "cpu_mhz": 100,
            "memory_mb": 128,
            "disk_mb": 256,
            "timeout_seconds": 5,
        },
        "evaluator_version": "e1",
        "validators": [],
        "provider_independent_settings_sha256": "e" * 64,
        "tool_schemas_sha256": "f" * 64,
    }
    value.update(changes)
    return value


class ContractTests(unittest.TestCase):
    def test_run_manifest_v1_keeps_new_metadata_optional(self):
        root = Path(__file__).resolve().parents[2] / "missy/repoeval"
        schema = json.loads((root / "schemas/run-manifest.schema.json").read_text())
        example = json.loads((root / "schemas/examples/run-manifest.json").read_text())
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(example)), [])
        self.assertEqual(schema["$id"], "urn:repoeval-foundry:schema:run-manifest:v1")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0")
        for field in (
            "idempotency_key_sha256",
            "provider_independent_settings_sha256",
            "tool_schemas_sha256",
            "evaluator_version",
            "validators",
        ):
            self.assertNotIn(field, schema["required"])
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(example)), [])
        for field in (
            "idempotency_key_sha256",
            "provider_independent_settings_sha256",
            "tool_schemas_sha256",
            "evaluator_version",
            "validators",
        ):
            example.pop(field, None)
        self.assertEqual(list(Draft202012Validator(schema).iter_errors(example)), [])

    def test_canonical_key_order_and_unicode(self):
        self.assertEqual(
            canonical_json({"z": 1, "é": "x", "a": 2}), '{"a":2,"z":1,"é":"x"}'.encode()
        )

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            canonical_json({"x": math.nan})

    def test_provider_input_shared_envelope(self):
        x = canonical_provider_input("p", [], [], {"temperature": 0})
        self.assertEqual(
            x, {"prompt": "p", "messages": [], "tool_schemas": [], "settings": {"temperature": 0}}
        )

    def test_definition_hash_ignores_provider_and_description(self):
        a = {"id": "x", "description": "one", "providers": [{"model": "a"}], "task": {"x": 1}}
        b = {**a, "description": "two", "providers": [{"model": "b"}]}
        self.assertEqual(definition_hash(a), definition_hash(b))

    def test_provider_and_repetition_do_not_affect_comparability(self):
        a = manifest(provider={"model": "a"}, repetition={"number": 1})
        b = manifest(provider={"model": "b"}, repetition={"number": 7})
        key = comparability_key(a)
        a["comparability_key"] = key
        b["comparability_key"] = key
        self.assertTrue(compare_runs(a, b).comparable)

    def test_mismatch_is_incomparable(self):
        self.assertIn(
            "repository.commit_sha",
            compare_runs(
                manifest(),
                manifest(repository={"repository_id": "r", "commit_sha": "z", "snapshot_id": "s"}),
            ).reasons,
        )

    def test_invalid_claimed_key_refused(self):
        a = manifest(comparability_key="0" * 64)
        self.assertIn("comparability_key_invalid", compare_runs(a, manifest()).reasons)
