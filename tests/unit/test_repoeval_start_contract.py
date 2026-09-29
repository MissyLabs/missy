"""A real Foundry coordinator reply consumed by the Missy client at this head."""

import json
from pathlib import Path

from missy.tools.builtin.repoeval_tools import RepoevalFoundryTool, _identity_digest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "schemas/fixtures/benchmark-start-contract.json"
SCHEMA = ROOT / "schemas/benchmark-start-ack.schema.json"


class FakeResponse:
    status_code = 202

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeClient:
    category = "tool"

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post_limited(self, url, _limit, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.payload)


def test_foundry_start_fixture_reaches_missy_client_without_claiming_completion(tmp_path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert fixture["version"] == 1
    assert fixture["schema"] == "schemas/benchmark-start-ack.schema.json"
    assert fixture["http_status"] == 202
    # The Foundry producer validates this schema with Draft 2020-12. Missy
    # does not require jsonschema at runtime or in its dev environment: the
    # client contract below checks the actual fixture and wire acknowledgement.
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(schema["properties"]["data"]["required"]) == set(fixture["example_response"]["data"])
    sibling = ROOT.parent / "repoeval-foundry/schemas"
    if sibling.is_dir():
        assert (
            json.loads((sibling / "fixtures/benchmark-start-contract.json").read_text()) == fixture
        )
        assert json.loads((sibling / "benchmark-start-ack.schema.json").read_text()) == schema

    payload = fixture["example_response"]
    data = payload["data"]
    project, plan_id, key = data["project_id"], data["plan_id"], "shared-start-fixture"
    assert data["id"] == "run-" + _identity_digest([project, plan_id, key])[:24]
    client = FakeClient(payload)
    token = tmp_path / "token"
    token.write_text("test-only-token\n", encoding="ascii")
    token.chmod(0o600)
    tool = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id=project,
        allowed_hosts=["foundry.example"],
        http_client=client,
        api_available=True,
        token_file=str(token),
    )
    tool._plans[plan_id] = {
        "state": "planned",
        "placement": {"pool": "staging"},
        "policy_checks": dict.fromkeys(
            ("repository", "image", "providers", "budget", "quota", "egress", "audit"), True
        ),
        "workload": {
            "providers": [{"registry_key": "approved"}],
            "execution": {"repetitions": 1},
            "artifacts": {"required_kinds": ["result"]},
        },
    }
    result = tool.execute(
        action="start",
        plan_id=plan_id,
        idempotency_key=key,
        acknowledge_project_scope=True,
    )
    assert result.success
    assert result.output["acknowledged"] is True
    assert result.output["execution_complete"] is False
    assert result.output["resource"] == data
    assert len(client.calls) == 1
    assert client.calls[0][0].endswith("/projects/project/benchmark/start")
    assert client.calls[0][1]["headers"]["Idempotency-Key"] == key
