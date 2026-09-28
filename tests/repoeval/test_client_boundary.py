"""Keep local core claims separate from Missy's gated remote HTTP client."""

from missy.repoeval import FoundryAPI, FoundryService, Principal, scan_repository
from missy.repoeval.control import FoundryError
from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool


def test_core_exports_native_missy_import_paths() -> None:
    assert FoundryAPI.__module__ == "missy.repoeval.api"
    assert FoundryService.__module__ == "missy.repoeval.control"
    assert Principal.__module__ == "missy.repoeval.control"
    assert scan_repository.__module__ == "missy.repoeval.scanner"


def test_core_does_not_grant_remote_client_a_reviewed_staging_plan() -> None:
    # The current core can attest its own planning constraints, not the remote
    # client's additional capacity, staging-placement, egress and audit checks.
    # Do not fabricate them to turn on the client start path.
    core_shaped_plan = {
        "id": "plan-example",
        "state": "planned",
        "project_id": "example",
        "workload": {},
        "definition_sha256": "a" * 64,
        "comparability_key": "b" * 64,
    }
    assert not RepoevalFoundryTool._approved(core_shaped_plan)


def test_empty_core_refuses_unregistered_snapshot_and_has_no_dispatcher() -> None:
    service = FoundryService(repositories={}, providers=set(), limits={})
    principal = Principal("local", "example", frozenset({"read", "execute", "snapshot:start"}))
    api = FoundryAPI(service, lambda headers: principal if headers.get("X-Test") == "yes" else None)
    assert api.handle("GET", "/api/projects/example/repositories", {}, {}).status == 401
    assert api.handle("GET", "/api/projects/example/repositories", {"X-Test": "yes"}).body == {
        "ok": True,
        "data": [],
    }
    refused = api.handle(
        "POST",
        "/api/projects/example/snapshots",
        {"X-Test": "yes", "Idempotency-Key": "offline-key-123"},
        {"repository_id": "unregistered", "commit_sha": "a" * 40},
    )
    assert refused.status == 400
    assert refused.body["error"]["category"] == "invalid_input"
    assert service.dispatcher is None
    assert service.allow_demo_dispatch is False
    assert issubclass(FoundryError, ValueError)
