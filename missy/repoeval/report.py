"""Deterministic comparison reports with explicit incomparable runs."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import compare_runs


def build_comparison_report(runs: Sequence[Mapping[str, Any]]) -> dict:
    groups = defaultdict(list)
    incomparable = []
    for run in sorted(runs, key=lambda x: str(x.get("run_id", ""))):
        key = run.get("comparability_key")
        if not isinstance(key, str) or len(key) != 64:
            incomparable.append(
                {"run_id": run.get("run_id"), "reason": ["missing_or_invalid_comparability_key"]}
            )
        else:
            groups[key].append(run)
    accepted = []
    for key, members in sorted(groups.items()):
        valid = []
        for run in members:
            cmp = compare_runs(members[0], run)
            if cmp.comparable and cmp.key == key:
                valid.append(run)
            else:
                incomparable.append(
                    {
                        "run_id": run.get("run_id"),
                        "reason": list(cmp.reasons) or ["claimed_key_mismatch"],
                    }
                )
        if valid:
            accepted.append(
                {
                    "comparability_key": key,
                    "run_ids": [r.get("run_id") for r in valid],
                    "runs": valid,
                }
            )
    return {
        "schema_version": "1.0",
        "groups": accepted,
        "incomparable": sorted(incomparable, key=lambda x: str(x.get("run_id", ""))),
    }
