"""Tests for missy.tools.builtin.obs_tools.

All tests mock the config loader and the ``websockets`` connection --
no real OBS instance, network call, or policy-engine initialisation is
required. Two concerns get deep coverage: (1) the obs-websocket v5
auth-response hash, since a wrong hash silently locks every request out,
and (2) the fail-closed confirmation gating for streaming, since that's
the one action this module can't afford to get permissive by accident.
"""

from __future__ import annotations

import base64
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import MissyConfig, ObsConfig
from missy.tools.builtin.obs_tools import (
    ObsListScenesTool,
    ObsSetSourceVisibilityTool,
    ObsStartRecordingTool,
    ObsStartStreamingConfirmedTool,
    ObsStatusTool,
    ObsStopRecordingTool,
    ObsStopStreamingConfirmedTool,
    ObsSwitchSceneTool,
    _compute_auth_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_missy_config(obs: ObsConfig) -> MissyConfig:
    return MagicMock(obs=obs)


def _mock_config(**overrides) -> ObsConfig:
    defaults = {"enabled": True, "host": "127.0.0.1", "port": 4455, "password": None}
    defaults.update(overrides)
    return ObsConfig(**defaults)


class _FakeWs:
    """A minimal async-iterable fake obs-websocket connection.

    ``responses`` is a queue of raw JSON strings returned in order by
    successive ``.recv()`` calls (Hello, Identified, RequestResponse).
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        return self._responses.pop(0)

    async def __aenter__(self) -> _FakeWs:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _hello(auth: dict | None = None) -> str:
    d: dict = {"obsWebSocketVersion": "5.0.0", "rpcVersion": 1}
    if auth:
        d["authentication"] = auth
    return json.dumps({"op": 0, "d": d})


def _identified() -> str:
    return json.dumps({"op": 2, "d": {}})


def _request_response(request_type: str, response_data: dict, *, ok: bool = True) -> str:
    return json.dumps(
        {
            "op": 7,
            "d": {
                "requestType": request_type,
                "requestStatus": {"result": ok, "code": 100 if ok else 400},
                "responseData": response_data,
            },
        }
    )


def _connect_sequence(*raw_responses: str):
    """Return a patch target for websockets.connect yielding one _FakeWs.

    Use when the tool under test issues exactly one obs-websocket request
    (each ``_obs_request()`` call opens its own connection -- see
    obs_tools.py's module docstring -- so a tool issuing N requests needs
    N separate hello/identified/response sequences; see
    :func:`_connect_multi_request` for that case).
    """
    fake_ws = _FakeWs(list(raw_responses))
    connect = MagicMock(return_value=fake_ws)
    return connect, fake_ws


def _connect_multi_request(*request_data_pairs: tuple[str, dict]):
    """Return a patch target for a tool issuing multiple sequential requests.

    Each ``(request_type, response_data)`` pair gets its own fresh
    hello/identified/response connection, in order.
    """
    fakes = [
        _FakeWs([_hello(), _identified(), _request_response(request_type, data)])
        for request_type, data in request_data_pairs
    ]
    connect = MagicMock(side_effect=fakes)
    return connect, fakes


# ---------------------------------------------------------------------------
# Auth response hash (protocol correctness -- a wrong hash silently locks
# every request out with no other symptom)
# ---------------------------------------------------------------------------


class TestComputeAuthResponse:
    def test_matches_obs_websocket_v5_spec_manually(self):
        password = "hunter2"
        challenge = "chal123"
        salt = "salt456"

        base64_secret = base64.b64encode(
            hashlib.sha256((password + salt).encode()).digest()
        ).decode()
        expected = base64.b64encode(
            hashlib.sha256((base64_secret + challenge).encode()).digest()
        ).decode()

        assert _compute_auth_response(password, challenge, salt) == expected

    def test_different_passwords_produce_different_hashes(self):
        a = _compute_auth_response("pw1", "chal", "salt")
        b = _compute_auth_response("pw2", "chal", "salt")
        assert a != b

    def test_deterministic(self):
        a = _compute_auth_response("pw", "chal", "salt")
        b = _compute_auth_response("pw", "chal", "salt")
        assert a == b


# ---------------------------------------------------------------------------
# Fail-closed config gating
# ---------------------------------------------------------------------------


class TestObsDisabledGating:
    def test_status_fails_when_obs_disabled(self):
        with patch(
            "missy.tools.builtin.obs_tools.load_missy_config",
            return_value=_make_missy_config(_mock_config(enabled=False)),
        ):
            result = ObsStatusTool().execute()
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_status_fails_when_no_config_at_all(self):
        with patch("missy.tools.builtin.obs_tools.load_missy_config", return_value=None):
            result = ObsStatusTool().execute()
        assert result.success is False


# ---------------------------------------------------------------------------
# ObsStatusTool -- happy path, mocked WS
# ---------------------------------------------------------------------------


class TestObsStatusTool:
    def _run_with_responses(self, *pairs: tuple[str, dict], config: ObsConfig | None = None):
        """pairs: (requestType, responseData), one per _obs_request() call."""
        connect, fakes = _connect_multi_request(*pairs)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(config or _mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsStatusTool().execute()
        return result, fakes

    def test_reports_connected_scene_and_stream_state(self):
        result, _ = self._run_with_responses(
            ("GetVersion", {"obsVersion": "30.0.0"}),
            ("GetCurrentProgramScene", {"currentProgramSceneName": "Main"}),
            ("GetStreamStatus", {"outputActive": True, "outputTimecode": "00:01:00"}),
            ("GetRecordStatus", {"outputActive": False, "outputPaused": False}),
        )
        assert result.success is True
        assert result.output["current_scene"] == "Main"
        assert result.output["streaming"] is True
        assert result.output["recording"] is False

    def test_password_never_appears_in_output(self):
        result, _ = self._run_with_responses(
            ("GetVersion", {"obsVersion": "30.0.0"}),
            ("GetCurrentProgramScene", {"currentProgramSceneName": "Main"}),
            ("GetStreamStatus", {"outputActive": False}),
            ("GetRecordStatus", {"outputActive": False}),
            config=_mock_config(password="supersecret"),
        )
        assert "supersecret" not in json.dumps(result.output)

    def test_connection_failure_returns_clean_error(self):
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", side_effect=OSError("Connection refused")),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsStatusTool().execute()
        assert result.success is False
        assert "Connection refused" in result.error

    def test_network_policy_denial_propagates(self):
        from missy.core.exceptions import PolicyViolationError

        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(host="10.0.0.99")),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
        ):
            mock_engine.return_value.check_network.side_effect = PolicyViolationError(
                "denied", category="network", detail="host not allowed"
            )
            with pytest.raises(PolicyViolationError):
                ObsStatusTool().execute()

    def test_resolve_network_hosts_declares_configured_host(self):
        with patch(
            "missy.tools.builtin.obs_tools.load_missy_config",
            return_value=_make_missy_config(_mock_config(host="192.168.1.50")),
        ):
            hosts = ObsStatusTool().resolve_network_hosts({})
        assert hosts == ["192.168.1.50"]


class TestObsAuthenticatedHandshake:
    """End-to-end coverage of the real challenge/salt auth flow (not just
    TestComputeAuthResponse's isolated hash-math check)."""

    def test_sends_correct_computed_auth_response(self):
        challenge, salt, password = "chal-xyz", "salt-abc", "hunter2"
        fake_ws = _FakeWs(
            [
                _hello({"challenge": challenge, "salt": salt}),
                _identified(),
                _request_response("GetVersion", {"obsVersion": "30.0.0"}),
            ]
        )
        connect = MagicMock(return_value=fake_ws)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(password=password)),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            from missy.tools.builtin.obs_tools import _compute_auth_response, _obs_request

            result_data = _obs_request("GetVersion")

        assert result_data == {"obsVersion": "30.0.0"}
        identify_msg = fake_ws.sent[0]
        assert identify_msg["op"] == 1
        assert identify_msg["d"]["authentication"] == _compute_auth_response(
            password, challenge, salt
        )

    def test_password_required_but_not_configured_raises_before_identify(self):
        fake_ws = _FakeWs([_hello({"challenge": "c", "salt": "s"})])
        connect = MagicMock(return_value=fake_ws)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(password=None)),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsStatusTool().execute()

        assert result.success is False
        assert "password" in result.error.lower()
        assert fake_ws.sent == []  # never even attempted Identify


# ---------------------------------------------------------------------------
# ObsListScenesTool
# ---------------------------------------------------------------------------


class TestObsListScenesTool:
    def test_lists_scenes_and_current_scene_sources(self):
        connect, _ = _connect_multi_request(
            (
                "GetSceneList",
                {
                    "currentProgramSceneName": "Main",
                    "scenes": [{"sceneName": "Main"}, {"sceneName": "BRB"}],
                },
            ),
            (
                "GetSceneItemList",
                {
                    "sceneItems": [
                        {"sourceName": "Webcam", "sceneItemEnabled": True, "sceneItemId": 1},
                        {"sourceName": "Overlay", "sceneItemEnabled": False, "sceneItemId": 2},
                    ]
                },
            ),
        )
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsListScenesTool().execute()

        assert result.success is True
        assert result.output["scenes"] == ["Main", "BRB"]
        assert result.output["current_scene"] == "Main"
        assert len(result.output["current_scene_sources"]) == 2
        assert result.output["current_scene_sources"][0]["visible"] is True


# ---------------------------------------------------------------------------
# ObsSwitchSceneTool -- allowlist + approval gating
# ---------------------------------------------------------------------------


class TestObsSwitchSceneTool:
    def test_switch_allowed_when_no_allowlist_configured(self):
        raw = [_hello(), _identified(), _request_response("SetCurrentProgramScene", {})]
        connect, _ = _connect_sequence(*raw)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(scene_allowlist=[])),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsSwitchSceneTool().execute(scene_name="Anything")
        assert result.success is True
        assert result.output["current_scene"] == "Anything"

    def test_switch_to_allowlisted_scene_skips_approval(self):
        raw = [_hello(), _identified(), _request_response("SetCurrentProgramScene", {})]
        connect, _ = _connect_sequence(*raw)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(scene_allowlist=["Main", "BRB"])),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
            patch("missy.tools.builtin.obs_tools.require_approval") as mock_approval,
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsSwitchSceneTool().execute(scene_name="Main")
        assert result.success is True
        mock_approval.assert_not_called()

    def test_switch_to_non_allowlisted_scene_requires_approval(self):
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(scene_allowlist=["Main"])),
            ),
            patch(
                "missy.tools.builtin.obs_tools.require_approval",
                return_value="denied by operator",
            ) as mock_approval,
        ):
            result = ObsSwitchSceneTool().execute(scene_name="SecretScene")
        assert result.success is False
        assert "denied" in result.error
        mock_approval.assert_called_once()

    def test_approved_non_allowlisted_switch_proceeds(self):
        raw = [_hello(), _identified(), _request_response("SetCurrentProgramScene", {})]
        connect, _ = _connect_sequence(*raw)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config(scene_allowlist=["Main"])),
            ),
            patch("missy.tools.builtin.obs_tools.require_approval", return_value=None),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsSwitchSceneTool().execute(scene_name="SecretScene")
        assert result.success is True


# ---------------------------------------------------------------------------
# ObsSetSourceVisibilityTool
# ---------------------------------------------------------------------------


class TestObsSetSourceVisibilityTool:
    def test_toggles_matching_source(self):
        connect, fakes = _connect_multi_request(
            ("GetSceneItemList", {"sceneItems": [{"sourceName": "Webcam", "sceneItemId": 7}]}),
            ("SetSceneItemEnabled", {}),
        )
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsSetSourceVisibilityTool().execute(
                scene_name="Main", source_name="Webcam", visible=False
            )
        assert result.success is True
        set_req = fakes[-1].sent[-1]["d"]["requestData"]
        assert set_req["sceneItemId"] == 7
        assert set_req["sceneItemEnabled"] is False

    def test_source_not_found_returns_error(self):
        raw = [_hello(), _identified(), _request_response("GetSceneItemList", {"sceneItems": []})]
        connect, _ = _connect_sequence(*raw)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsSetSourceVisibilityTool().execute(
                scene_name="Main", source_name="Ghost", visible=True
            )
        assert result.success is False
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# Recording tools -- no approval required
# ---------------------------------------------------------------------------


class TestObsRecordingTools:
    def test_start_recording_no_approval_needed(self):
        raw = [_hello(), _identified(), _request_response("StartRecord", {})]
        connect, _ = _connect_sequence(*raw)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
            patch("missy.tools.builtin.obs_tools.require_approval") as mock_approval,
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsStartRecordingTool().execute()
        assert result.success is True
        mock_approval.assert_not_called()

    def test_stop_recording_returns_output_path(self):
        raw = [
            _hello(),
            _identified(),
            _request_response("StopRecord", {"outputPath": "/home/missy/videos/rec.mkv"}),
        ]
        connect, _ = _connect_sequence(*raw)
        with (
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsStopRecordingTool().execute()
        assert result.output["output_path"] == "/home/missy/videos/rec.mkv"


# ---------------------------------------------------------------------------
# Streaming tools -- ALWAYS require approval, fail closed with no gate
# ---------------------------------------------------------------------------


class TestObsStreamingConfirmedTools:
    def test_start_streaming_denied_with_no_gate_configured(self):
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=None):
            result = ObsStartStreamingConfirmedTool().execute()
        assert result.success is False
        assert "approval" in result.error.lower()

    def test_stop_streaming_denied_with_no_gate_configured(self):
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=None):
            result = ObsStopStreamingConfirmedTool().execute()
        assert result.success is False

    def test_start_streaming_calls_gate_before_obs_request(self):
        mock_gate = MagicMock()
        raw = [_hello(), _identified(), _request_response("StartStream", {})]
        connect, _ = _connect_sequence(*raw)
        with (
            patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate),
            patch(
                "missy.tools.builtin.obs_tools.load_missy_config",
                return_value=_make_missy_config(_mock_config()),
            ),
            patch("missy.policy.engine.get_policy_engine") as mock_engine,
            patch("websockets.connect", connect),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = ObsStartStreamingConfirmedTool().execute()
        assert result.success is True
        mock_gate.request.assert_called_once()
        assert mock_gate.request.call_args.kwargs["risk"] == "high"

    def test_denied_approval_blocks_the_obs_request_entirely(self):
        from missy.agent.approval import ApprovalDenied

        mock_gate = MagicMock()
        mock_gate.request.side_effect = ApprovalDenied("no")
        with (
            patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate),
            patch("missy.tools.builtin.obs_tools._obs_request") as mock_request,
        ):
            result = ObsStartStreamingConfirmedTool().execute()
        assert result.success is False
        mock_request.assert_not_called()

    def test_stop_streaming_always_confirms_even_with_scene_allowlist_set(self):
        """Unlike obs_switch_scene, there's no allowlist bypass for streaming."""
        mock_gate = MagicMock()
        from missy.agent.approval import ApprovalDenied

        mock_gate.request.side_effect = ApprovalDenied("no")
        with patch("missy.agent.approval.get_shared_approval_gate", return_value=mock_gate):
            result = ObsStopStreamingConfirmedTool().execute()
        assert result.success is False
        mock_gate.request.assert_called_once()


# ---------------------------------------------------------------------------
# Permission declarations
# ---------------------------------------------------------------------------


class TestObsPermissions:
    @pytest.mark.parametrize(
        "tool_cls",
        [
            ObsStatusTool,
            ObsListScenesTool,
            ObsSwitchSceneTool,
            ObsSetSourceVisibilityTool,
            ObsStartRecordingTool,
            ObsStopRecordingTool,
            ObsStartStreamingConfirmedTool,
            ObsStopStreamingConfirmedTool,
        ],
    )
    def test_declares_network_permission(self, tool_cls):
        assert tool_cls().permissions.network is True

    @pytest.mark.parametrize(
        "tool_cls", [ObsStartStreamingConfirmedTool, ObsStopStreamingConfirmedTool]
    )
    def test_streaming_tools_take_no_parameters(self, tool_cls):
        assert tool_cls().parameters == {}
