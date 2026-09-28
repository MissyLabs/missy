"""Entrypoint fixtures are offline and do not enable a production backend."""

import hashlib
import json
from pathlib import Path

import pytest

from missy.repoeval.contracts import canonical_json
from missy.repoeval.entrypoint import SealedBundle, run_once
from missy.repoeval.provider import ProviderResult

ROOT = Path(__file__).resolve().parents[2] / "missy/repoeval/workloads/missy/repository-orientation"


def output():
    return (ROOT / "fixtures/orientation-facts.json").read_text()


class Store:
    def __init__(self, bundle):
        self.bundle = bundle
        self.calls = []

    def resolve(self, digest):
        self.calls.append(digest)
        return self.bundle


class Broker:
    def __init__(self):
        self.calls = []

    def execute(self, **kw):
        self.calls.append(kw)
        content = output()
        return ProviderResult(
            kw["registry_key"],
            kw["model"],
            kw["model"],
            None,
            kw["request"].digest,
            content,
            hashlib.sha256(content.encode()).hexdigest(),
            False,
            1,
            1,
            1,
            5,
        )


def fixture():
    manifest = json.loads((ROOT / "workload.json").read_text())
    manifest["providers"] = [{"registry_key": "approved-stub", "model": "stub-model"}]
    manifest["sandbox"]["image_digest"] = "registry.example/worker@sha256:" + "1" * 64
    payloads = {"prompt.md": (ROOT / "prompt.md").read_bytes()}
    payloads.update(
        {
            name: (ROOT / "fixtures" / name).read_bytes()
            for name in manifest["task"]["fixture_digests"]
        }
    )
    bundle = SealedBundle(
        canonical_json(manifest),
        payloads,
        "approved-project",
        "approved-stub",
        "stub-model",
        1000,
        20,
        512,
    )
    return hashlib.sha256(bundle.manifest_bytes).hexdigest(), bundle


def launch(tmp_path, reference, bundle, *, approve=lambda *_: True, broker=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, result = tmp_path / "manifest.sha256", tmp_path / "result.json"
    source.write_bytes(reference)
    store = Store(bundle)
    broker = broker or Broker()
    status = run_once(store=store, approve=approve, broker=broker, reference=source, result=result)
    return status, json.loads(result.read_bytes()), store, broker


def test_no_backends_refuses_without_reading_a_reference(tmp_path):
    result = tmp_path / "result.json"
    assert run_once(reference=tmp_path / "missing", result=result) == 2
    assert json.loads(result.read_bytes()) == {"status": "refused", "reason": "backend_unavailable"}


def test_exact_approved_fixture_finishes_without_shell_or_network(tmp_path):
    digest, bundle = fixture()
    status, result, store, broker = launch(tmp_path, (digest + "\n").encode(), bundle)
    assert status == 0
    assert result["status"] == "passed"
    assert result["manifest_sha256"] == digest
    assert store.calls == [digest] and len(broker.calls) == 1
    assert "content" not in result and "payloads" not in result


@pytest.mark.parametrize(
    "reference",
    [
        b"",
        b"https://host/manifest",
        b"A" * 64,
        b"0" * 64 + b"\nextra",
        b"0" * 64 + b" ",
        b"0" * 64 + b"\n\n",
    ],
)
def test_reference_rejects_non_digest_before_store(tmp_path, reference):
    _, bundle = fixture()
    status, result, store, broker = launch(tmp_path, reference, bundle)
    assert status == 2 and result["reason"] == "invalid_reference"
    assert not store.calls and not broker.calls


def test_symlink_reference_refused(tmp_path):
    digest, bundle = fixture()
    (tmp_path / "real").write_text(digest)
    (tmp_path / "manifest.sha256").symlink_to(tmp_path / "real")
    result = tmp_path / "result.json"
    assert (
        run_once(
            store=Store(bundle),
            approve=lambda *_: True,
            broker=Broker(),
            reference=tmp_path / "manifest.sha256",
            result=result,
        )
        == 2
    )
    assert json.loads(result.read_bytes())["reason"] == "invalid_reference"


def test_digest_custody_and_approval_fail_closed(tmp_path):
    digest, bundle = fixture()
    status, result, _, broker = launch(tmp_path, digest.encode(), bundle, approve=lambda *_: False)
    assert (status, result["reason"]) == (2, "approval_required")
    assert not broker.calls

    altered = SealedBundle(
        bundle.manifest_bytes + b" ",
        bundle.payloads,
        bundle.project_id,
        bundle.registry_key,
        bundle.model,
        1000,
        20,
        512,
    )
    status, result, _, broker = launch(tmp_path / "other", digest.encode(), altered)
    assert (status, result["reason"]) == (2, "custody_unavailable")
    assert not broker.calls


def test_manifest_does_not_supply_provider_authority(tmp_path):
    digest, bundle = fixture()
    wrong = SealedBundle(
        bundle.manifest_bytes,
        bundle.payloads,
        bundle.project_id,
        "other",
        bundle.model,
        1000,
        20,
        512,
    )
    status, result, _, broker = launch(tmp_path, digest.encode(), wrong)
    assert (status, result["reason"]) == (2, "invalid_manifest")
    assert not broker.calls


def test_result_is_write_once_even_on_second_attempt(tmp_path):
    digest, bundle = fixture()
    path = tmp_path / "manifest.sha256"
    path.write_text(digest)
    result = tmp_path / "result.json"
    assert (
        run_once(
            store=Store(bundle),
            approve=lambda *_: True,
            broker=Broker(),
            reference=path,
            result=result,
        )
        == 0
    )
    before = result.read_bytes()
    assert (
        run_once(
            store=Store(bundle),
            approve=lambda *_: True,
            broker=Broker(),
            reference=path,
            result=result,
        )
        == 2
    )
    assert result.read_bytes() == before


def test_store_error_does_not_leak_secret(tmp_path):
    digest, _ = fixture()

    class BrokenStore:
        def resolve(self, _):
            raise RuntimeError("Bearer secret")

    reference, result = tmp_path / "manifest.sha256", tmp_path / "result.json"
    reference.write_text(digest)
    assert (
        run_once(
            store=BrokenStore(),
            approve=lambda *_: True,
            broker=Broker(),
            reference=reference,
            result=result,
        )
        == 2
    )
    assert result.read_bytes() == b'{"reason":"custody_unavailable","status":"refused"}\n'
