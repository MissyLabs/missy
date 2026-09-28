"""Worker contract tests never contact a provider or run repository code."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from missy.repoeval.contracts import sha256_json
from missy.repoeval.provider import ProviderResult
from missy.repoeval.worker import WorkerRequest, execute_worker

ROOT = Path(__file__).resolve().parents[2] / "missy/repoeval/workloads/missy"
CASES = {"repository-orientation", "tool-call-correctness", "patch-test-repair"}


class StubBroker:
    def __init__(self, content=None, *, tamper=False, failure=False):
        self.content = content
        self.tamper = tamper
        self.failure = failure
        self.calls = []

    def execute(self, **kw):
        self.calls.append(kw)
        if self.failure:
            raise RuntimeError("sk-DO-NOT-LEAK-secret-token")
        request = kw["request"]
        content = self.content
        return ProviderResult(
            kw["registry_key"],
            kw["model"],
            kw["model"],
            None,
            request.digest,
            content,
            hashlib.sha256(content.encode()).hexdigest() if not self.tamper else "0" * 64,
            False,
            1,
            1,
            1,
            5,
        )


def case(name):
    directory = ROOT / name
    manifest = json.loads((directory / "workload.json").read_text())
    manifest["providers"] = [{"registry_key": "approved-stub", "model": "stub-model"}]
    manifest["sandbox"]["image_digest"] = "registry.example/approved-worker@sha256:" + "1" * 64
    payloads = {"prompt.md": (directory / "prompt.md").read_bytes()}
    for fixture in manifest["task"]["fixture_digests"]:
        payloads[fixture] = (directory / "fixtures" / fixture).read_bytes()
    return WorkerRequest(
        manifest,
        sha256_json(manifest),
        payloads,
        "project",
        "approved-stub",
        "stub-model",
        1000,
        20,
        512,
    )


def approve_digest(digest, manifest):
    return digest == sha256_json(manifest)


def output(name):
    if name == "repository-orientation":
        return (ROOT / name / "fixtures/orientation-facts.json").read_text()
    if name == "tool-call-correctness":
        return '{"name":"calculator","arguments":{"expression":"250 * 18 / 100"}}'
    return '--- a/repair.py\n+++ b/repair.py\n@@ -1,3 +1,3 @@\n def percent_of(value: float, percentage: float) -> float:\n     """Return percentage percent of value."""\n-    return value + percentage\n+    return value * percentage / 100\n'


@pytest.mark.parametrize("name", ["repository-orientation", "tool-call-correctness"])
def test_pass_without_executing_any_tool(name):
    broker = StubBroker(output(name))
    request = case(name)
    result = execute_worker(request, approve=approve_digest, broker=broker)
    assert result.status == "passed"
    assert result.score == 1
    assert result.manifest_sha256 == sha256_json(request.manifest)
    assert len(result.report_sha256) == 64
    assert len(broker.calls) == 1
    assert result.response_sha256 == hashlib.sha256(output(name).encode()).hexdigest()
    assert "content" not in result.report()


def test_patch_never_claims_applied_or_tested():
    broker = StubBroker(output("patch-test-repair"))
    result = execute_worker(case("patch-test-repair"), approve=approve_digest, broker=broker)
    assert result.status == "failed" and result.reason == "sandbox_unavailable"
    assert result.score == 0
    assert len(broker.calls) == 1


@pytest.mark.parametrize("name", CASES)
def test_bundled_drafts_without_provider_cannot_execute(name):
    req = case(name)
    req.manifest["providers"] = []
    req = WorkerRequest(
        req.manifest,
        sha256_json(req.manifest),
        req.payloads,
        req.project_id,
        req.registry_key,
        req.model,
        req.budget_microusd,
        req.timeout_seconds,
        req.token_cap,
    )
    broker = StubBroker(output(name))
    result = execute_worker(req, approve=approve_digest, broker=broker)
    assert result.status == "refused" and result.reason == "provider_not_in_manifest"
    assert broker.calls == []


def test_approval_mandatory_even_if_manifest_claims_approval():
    req = case("tool-call-correctness")
    req.manifest["approved"] = True
    req = WorkerRequest(
        req.manifest,
        sha256_json(req.manifest),
        req.payloads,
        req.project_id,
        req.registry_key,
        req.model,
        req.budget_microusd,
        req.timeout_seconds,
        req.token_cap,
    )
    broker = StubBroker(output("tool-call-correctness"))
    result = execute_worker(req, approve=lambda *_: False, broker=broker)
    assert result.reason == "approval_required" and broker.calls == []


def test_malformed_approval_or_unexpected_preflight_error_does_not_leak_text():
    req = case("tool-call-correctness")
    broker = StubBroker(output("tool-call-correctness"))

    def secret_error(*_):
        raise RuntimeError("Bearer secret-credential")

    assert execute_worker(req, approve=secret_error, broker=broker).reason == "approval_required"
    assert broker.calls == []

    invalid = WorkerRequest(
        req.manifest,
        "0" * 64,
        req.payloads,
        req.project_id,
        req.registry_key,
        req.model,
        req.budget_microusd,
        req.timeout_seconds,
        req.token_cap,
    )
    assert (
        execute_worker(invalid, approve=approve_digest, broker=broker).reason
        == "manifest_digest_mismatch"
    )


@pytest.mark.parametrize(
    "change,expected",
    [
        ("digest", "manifest_digest_mismatch"),
        ("fixture", "payload_digest_mismatch"),
        ("extra", "payload_set_mismatch"),
        ("timeout", "timeout_limit"),
        ("attempts", "unsupported_manifest"),
        ("validator", "unsupported_validator"),
        ("image", "unsupported_manifest"),
        ("resources", "unsupported_manifest"),
    ],
)
def test_preflight_rejects_modified_or_unbounded_input(change, expected):
    req = case("tool-call-correctness")
    attrs = copy.deepcopy(req.__dict__)
    if change == "digest":
        attrs["manifest_sha256"] = "0" * 64
    if change == "fixture":
        attrs["payloads"]["case.json"] += b"\n"
    if change == "extra":
        attrs["payloads"]["untrusted.py"] = b"pass"
    if change == "timeout":
        attrs["timeout_seconds"] = 121
    if change == "attempts":
        attrs["manifest"]["execution"]["max_attempts"] = 2
    if change == "validator":
        attrs["manifest"]["validation"]["validators"][0]["parameters"]["oracle"] = "../../secret"
    if change == "image":
        attrs["manifest"]["sandbox"]["image_digest"] = (
            "registry.invalid/repoeval@sha256:" + "1" * 64
        )
    if change == "resources":
        attrs["manifest"]["sandbox"]["memory_mb"] = 1_000_000
    if change in {"attempts", "validator", "image", "resources"}:
        attrs["manifest_sha256"] = sha256_json(attrs["manifest"])
    broker = StubBroker(output("tool-call-correctness"))
    result = execute_worker(WorkerRequest(**attrs), approve=approve_digest, broker=broker)
    assert result.status == "refused" and result.reason == expected
    assert broker.calls == []


@pytest.mark.parametrize(
    "content,expected",
    [
        ('{"name":"shell","arguments":{"expression":"250*18/100"}}', "tool_call_invalid"),
        (
            '{"name":"calculator","arguments":{"expression":"__import__(\\"os\\")"}}',
            "invalid_response_or_fixture",
        ),
        (
            '{"name":"calculator","arguments":{"expression":"250 * 18 / 100 + 1"}}',
            "tool_call_wrong_result",
        ),
        (
            '{"name":"calculator","arguments":{"expression":"40 + 5"}}',
            "tool_call_wrong_result",
        ),
        ("not-json", "invalid_response_or_fixture"),
    ],
)
def test_bad_tool_calls_fail_without_execution(content, expected):
    result = execute_worker(
        case("tool-call-correctness"), approve=approve_digest, broker=StubBroker(content)
    )
    assert result.status == "failed" and result.reason == expected


def test_broker_errors_and_digest_mismatch_fail_closed_without_secret_leakage():
    req = case("repository-orientation")
    failed = execute_worker(req, approve=approve_digest, broker=StubBroker(failure=True))
    corrupt = execute_worker(
        req,
        approve=approve_digest,
        broker=StubBroker(output("repository-orientation"), tamper=True),
    )
    assert (failed.status, failed.reason) == ("failed", "provider_error")
    assert (corrupt.status, corrupt.reason) == ("failed", "response_integrity_failed")
    assert "sk-DO-NOT-LEAK" not in repr(failed)


def test_patch_unsafe_path_refused_and_provider_output_not_exposed():
    result = execute_worker(
        case("patch-test-repair"),
        approve=approve_digest,
        broker=StubBroker("--- a/../../secret\n+++ b/../../secret\n"),
    )
    assert result.status == "failed" and result.reason == "unsafe_path"
    assert "../../secret" not in repr(result)
