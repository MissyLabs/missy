"""Tests for missy.tools.builtin.vtube_tools.

Mocks the config loader and the ``websockets`` connection. Deep coverage
on the auth-token flow: a fresh token must be requested, used, AND
persisted to vault -- and never appear in any tool output -- since that's
the one path in this module most likely to leak the one real secret it
handles.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from missy.config.settings import MissyConfig, VtubeConfig
from missy.tools.builtin.vtube_tools import (
    VtubeLoadModelTool,
    VtubeSetParameterTool,
    VtubeStatusTool,
    VtubeTriggerHotkeyTool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_missy_config(vtube: VtubeConfig) -> MissyConfig:
    return MagicMock(vtube=vtube)


def _mock_config(**overrides) -> VtubeConfig:
    defaults = {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 8001,
        "auth_token": "existing-token",
        "plugin_name": "Missy",
        "plugin_developer": "MissyLabs",
    }
    defaults.update(overrides)
    return VtubeConfig(**defaults)


class _FakeVtsWs:
    """Fake VTS connection: replies based on requested messageType, not order."""

    def __init__(self, responses_by_type: dict[str, dict]) -> None:
        self._responses = responses_by_type
        self.sent: list[dict] = []
        self._pending_reply: dict | None = None

    async def send(self, raw: str) -> None:
        msg = json.loads(raw)
        self.sent.append(msg)
        message_type = msg["messageType"]
        data = self._responses.get(message_type, {})
        self._pending_reply = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": msg["requestID"],
            "messageType": message_type.replace("Request", "Response"),
            "data": data,
        }

    async def recv(self) -> str:
        reply = self._pending_reply
        self._pending_reply = None
        return json.dumps(reply)

    async def __aenter__(self) -> _FakeVtsWs:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _connect(responses_by_type: dict[str, dict]):
    fake_ws = _FakeVtsWs(responses_by_type)
    connect = MagicMock(return_value=fake_ws)
    return connect, fake_ws


def _patched(config: VtubeConfig, responses: dict[str, dict]):
    """Context manager stack: config, policy engine (allow), websockets.connect."""
    connect, fake_ws = _connect(responses)
    return (
        patch(
            "missy.tools.builtin.vtube_tools.load_missy_config",
            return_value=_make_missy_config(config),
        ),
        patch("missy.policy.engine.get_policy_engine"),
        patch("websockets.connect", connect),
    ), fake_ws


# ---------------------------------------------------------------------------
# Fail-closed config gating
# ---------------------------------------------------------------------------


class TestVtubeDisabledGating:
    def test_status_fails_when_disabled(self):
        with patch(
            "missy.tools.builtin.vtube_tools.load_missy_config",
            return_value=_make_missy_config(_mock_config(enabled=False)),
        ):
            result = VtubeStatusTool().execute()
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_status_fails_when_no_config(self):
        with patch("missy.tools.builtin.vtube_tools.load_missy_config", return_value=None):
            result = VtubeStatusTool().execute()
        assert result.success is False


# ---------------------------------------------------------------------------
# VtubeStatusTool -- authenticated happy path with an EXISTING token
# ---------------------------------------------------------------------------


class TestVtubeStatusTool:
    def test_reports_current_model(self):
        responses = {
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "CurrentModelRequest": {
                "modelLoaded": True,
                "modelName": "Missy_v1",
                "modelID": "abc123",
                "live2DModelName": "missy.model3.json",
            },
        }
        config = _mock_config(auth_token="existing-token")
        patches, fake_ws = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeStatusTool().execute()

        assert result.success is True
        assert result.output["model_name"] == "Missy_v1"
        assert result.output["model_loaded"] is True

        # Re-authenticates with the pre-configured token -- never requests
        # a fresh one when one is already configured.
        auth_msg = next(m for m in fake_ws.sent if m["messageType"] == "AuthenticationRequest")
        assert auth_msg["data"]["authenticationToken"] == "existing-token"
        assert not any(m["messageType"] == "AuthenticationTokenRequest" for m in fake_ws.sent)

    def test_auth_rejected_returns_clean_error(self):
        responses = {
            "AuthenticationRequest": {"authenticated": False, "reason": "token revoked"},
        }
        config = _mock_config()
        patches, _ = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeStatusTool().execute()

        assert result.success is False
        assert "token revoked" in result.error

    def test_resolve_network_hosts_declares_configured_host(self):
        with patch(
            "missy.tools.builtin.vtube_tools.load_missy_config",
            return_value=_make_missy_config(_mock_config(host="192.168.1.60")),
        ):
            hosts = VtubeStatusTool().resolve_network_hosts({})
        assert hosts == ["192.168.1.60"]


# ---------------------------------------------------------------------------
# First-time authorization: acquire a fresh token, persist to vault, never
# leak it in the tool output
# ---------------------------------------------------------------------------


class TestVtubeFreshTokenAcquisition:
    def test_acquires_and_persists_fresh_token_when_none_configured(self):
        responses = {
            "AuthenticationTokenRequest": {"authenticationToken": "brand-new-secret-token"},
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "CurrentModelRequest": {"modelLoaded": False, "modelName": "", "modelID": ""},
        }
        config = _mock_config(auth_token=None)
        patches, fake_ws = _patched(config, responses)

        with (
            patches[0],
            patches[1] as mock_engine,
            patches[2],
            patch("missy.security.vault.Vault") as mock_vault_cls,
        ):
            mock_engine.return_value.check_network.return_value = True
            mock_vault = MagicMock()
            mock_vault_cls.return_value = mock_vault
            result = VtubeStatusTool().execute()

        assert result.success is True
        # The fresh token was used to authenticate...
        auth_msg = next(m for m in fake_ws.sent if m["messageType"] == "AuthenticationRequest")
        assert auth_msg["data"]["authenticationToken"] == "brand-new-secret-token"
        # ...persisted to vault...
        mock_vault.set.assert_called_once_with("vtube_studio_token", "brand-new-secret-token")
        # ...and never appears anywhere in the tool's output.
        assert "brand-new-secret-token" not in json.dumps(result.output)

    def test_vault_persist_failure_does_not_fail_the_call(self):
        """Losing the token to a Vault write error shouldn't undo an
        otherwise-successful VTS call -- it just means the pop-up recurs
        next time."""
        responses = {
            "AuthenticationTokenRequest": {"authenticationToken": "tok"},
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "CurrentModelRequest": {"modelLoaded": False, "modelName": "", "modelID": ""},
        }
        config = _mock_config(auth_token=None)
        patches, _ = _patched(config, responses)

        with (
            patches[0],
            patches[1] as mock_engine,
            patches[2],
            patch("missy.security.vault.Vault", side_effect=RuntimeError("disk full")),
        ):
            mock_engine.return_value.check_network.return_value = True
            result = VtubeStatusTool().execute()

        assert result.success is True

    def test_denied_pop_up_returns_clean_error(self):
        responses = {"AuthenticationTokenRequest": {}}  # no authenticationToken key = denied
        config = _mock_config(auth_token=None)
        patches, _ = _patched(config, responses)

        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeStatusTool().execute()

        assert result.success is False
        assert "token" in result.error.lower()


# ---------------------------------------------------------------------------
# VtubeLoadModelTool -- name -> ID resolution
# ---------------------------------------------------------------------------


class TestVtubeLoadModelTool:
    def test_loads_model_by_name(self):
        responses = {
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "AvailableModelsRequest": {
                "availableModels": [
                    {"modelName": "Missy_v1", "modelID": "id-1"},
                    {"modelName": "Missy_v2", "modelID": "id-2"},
                ]
            },
            "ModelLoadRequest": {},
        }
        config = _mock_config()
        patches, fake_ws = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeLoadModelTool().execute(model_name="Missy_v2")

        assert result.success is True
        assert result.output["model_id"] == "id-2"
        load_msg = next(m for m in fake_ws.sent if m["messageType"] == "ModelLoadRequest")
        assert load_msg["data"]["modelID"] == "id-2"

    def test_unknown_model_name_lists_available(self):
        responses = {
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "AvailableModelsRequest": {
                "availableModels": [{"modelName": "OnlyOne", "modelID": "x"}]
            },
        }
        config = _mock_config()
        patches, _ = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeLoadModelTool().execute(model_name="Ghost")

        assert result.success is False
        assert "OnlyOne" in result.error


# ---------------------------------------------------------------------------
# VtubeTriggerHotkeyTool -- name -> ID resolution
# ---------------------------------------------------------------------------


class TestVtubeTriggerHotkeyTool:
    def test_triggers_hotkey_by_name(self):
        responses = {
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "HotkeysInCurrentModelRequest": {
                "availableHotkeys": [{"name": "Wave", "hotkeyID": "hk-1"}]
            },
            "HotkeyTriggerRequest": {},
        }
        config = _mock_config()
        patches, fake_ws = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeTriggerHotkeyTool().execute(hotkey_name="Wave")

        assert result.success is True
        trigger_msg = next(m for m in fake_ws.sent if m["messageType"] == "HotkeyTriggerRequest")
        assert trigger_msg["data"]["hotkeyID"] == "hk-1"

    def test_unknown_hotkey_lists_available(self):
        responses = {
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "HotkeysInCurrentModelRequest": {
                "availableHotkeys": [{"name": "Wave", "hotkeyID": "hk-1"}]
            },
        }
        config = _mock_config()
        patches, _ = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeTriggerHotkeyTool().execute(hotkey_name="Nonexistent")

        assert result.success is False
        assert "Wave" in result.error


# ---------------------------------------------------------------------------
# VtubeSetParameterTool
# ---------------------------------------------------------------------------


class TestVtubeSetParameterTool:
    def test_sets_parameter_value(self):
        responses = {
            "AuthenticationRequest": {"authenticated": True, "reason": "ok"},
            "InjectParameterDataRequest": {},
        }
        config = _mock_config()
        patches, fake_ws = _patched(config, responses)
        with patches[0], patches[1] as mock_engine, patches[2]:
            mock_engine.return_value.check_network.return_value = True
            result = VtubeSetParameterTool().execute(parameter_id="MouthOpen", value=0.75)

        assert result.success is True
        assert result.output["value"] == 0.75
        inject_msg = next(
            m for m in fake_ws.sent if m["messageType"] == "InjectParameterDataRequest"
        )
        param = inject_msg["data"]["parameterValues"][0]
        assert param["id"] == "MouthOpen"
        assert param["value"] == 0.75


# ---------------------------------------------------------------------------
# Permission declarations
# ---------------------------------------------------------------------------


class TestVtubePermissions:
    @pytest.mark.parametrize(
        "tool_cls",
        [VtubeStatusTool, VtubeLoadModelTool, VtubeTriggerHotkeyTool, VtubeSetParameterTool],
    )
    def test_declares_network_permission(self, tool_cls):
        assert tool_cls().permissions.network is True
