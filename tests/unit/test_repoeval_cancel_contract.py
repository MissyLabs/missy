"""Versioned Foundry/Missy wire fixture: acceptance is not completion.

The local fixture and schema are mirrored from Foundry. A sibling clone is
optional locally and is never required in CI or at runtime.
"""

import json
from pathlib import Path

import pytest

from missy.tools.builtin.repoeval_tools import RepoevalFoundryMutateTool, RepoevalFoundryTool

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "schemas/fixtures/cancellation-contract.json"
SCHEMA = ROOT / "schemas/cancel-ack.schema.json"


class FakeResponse:
    def __init__(self, data, status):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


class FakeClient:
    category = "tool"

    def __init__(self):
        self.responses = []

    def post_limited(self, *_args, **_kwargs):
        return self.responses.pop(0)


@pytest.fixture
def client_tool(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-only-token\n", encoding="ascii")
    token.chmod(0o600)
    client = FakeClient()
    instance = RepoevalFoundryTool(
        base_url="https://foundry.example/api",
        project_id="alpha",
        allowed_hosts=["foundry.example"],
        http_client=client,
        api_available=True,
        token_file=str(token),
    )
    return instance, client


def test_mirrored_cancellation_contract_fixture():
    fixture = json.loads(FIXTURE.read_text())
    schema = json.loads(SCHEMA.read_text())
    assert fixture["version"] == 1
    assert fixture["schema"] == "schemas/cancel-ack.schema.json"
    assert fixture["http_status"] == 202
    assert set(fixture["accepted_states"]) == set(schema["properties"]["state"]["enum"])
    assert fixture["example_response"] == {
        "ok": True,
        "data": {"id": "run-example", "state": "cancel_pending", "cancel_requested": True},
    }
    sibling = ROOT.parent / "repoeval-foundry/schemas"
    if sibling.is_dir():
        assert json.loads((sibling / "fixtures/cancellation-contract.json").read_text()) == fixture
        assert json.loads((sibling / "cancel-ack.schema.json").read_text()) == schema


@pytest.mark.parametrize("state", ["cancel_pending", "cancelled", "failed"])
def test_each_foundry_cancel_ack_state_is_accepted_without_claiming_completion(state, client_tool):
    fixture = json.loads(FIXTURE.read_text())
    instance, client = client_tool
    mutation = RepoevalFoundryMutateTool(instance)
    reply = fixture["example_response"].copy()
    reply["data"] = {**reply["data"], "state": state}
    client.responses.append(FakeResponse(reply, fixture["http_status"]))
    result = mutation.execute(action="cancel", run_id="run-example")
    assert result.success
    assert result.output == {
        "acknowledged": True,
        "execution_complete": False,
        "resource": reply["data"],
        "route_project_id": "alpha",
        "scope_source": "authenticated_project_route",
    }


@pytest.mark.parametrize(
    "override",
    [
        {"state": "running"},
        {"id": "run-other"},
        {"cancel_requested": False},
        {"cancel_requested": None},
        {"project_id": "foreign"},
    ],
)
def test_invalid_cancel_ack_fails_closed(override, client_tool):
    fixture = json.loads(FIXTURE.read_text())
    instance, client = client_tool
    reply = fixture["example_response"].copy()
    reply["data"] = {**reply["data"], **override}
    client.responses.append(FakeResponse(reply, 202))
    assert (
        not RepoevalFoundryMutateTool(instance)
        .execute(action="cancel", run_id="run-example")
        .success
    )
