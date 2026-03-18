"""Tests for missy.channels.voice.pairing.PairingManager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from missy.channels.voice.pairing import PairingManager
from missy.channels.voice.registry import EdgeNode


def _make_registry():
    """Create a mock DeviceRegistry."""
    reg = MagicMock()
    reg.get_node.return_value = None
    return reg


def _make_node(node_id="test-node", paired=False, **kwargs):
    """Create a mock EdgeNode."""
    defaults = {
        "node_id": node_id,
        "friendly_name": "Test Node",
        "room": "office",
        "ip_address": "192.168.1.10",
        "hardware_profile": {},
        "paired": paired,
        "status": "offline",
        "policy_mode": "full",
        "audio_logging": False,
        "audio_log_dir": "",
        "audio_log_retention_days": 7,
    }
    defaults.update(kwargs)
    return EdgeNode(**defaults)


class TestInitiatePairing:
    @patch("missy.channels.voice.pairing.event_bus")
    def test_creates_new_node(self, mock_bus):
        reg = _make_registry()
        mgr = PairingManager(reg)
        node_id = mgr.initiate_pairing(
            node_id="node-1",
            friendly_name="Kitchen",
            room="kitchen",
            ip_address="192.168.1.20",
            hardware_profile={"mic_type": "respeaker"},
        )
        assert node_id == "node-1"
        reg.add_node.assert_called_once()
        mock_bus.publish.assert_called_once()

    @patch("missy.channels.voice.pairing.event_bus")
    def test_generates_uuid_when_empty(self, mock_bus):
        reg = _make_registry()
        mgr = PairingManager(reg)
        node_id = mgr.initiate_pairing(
            node_id="",
            friendly_name="Auto",
            room="room",
            ip_address="10.0.0.1",
            hardware_profile={},
        )
        assert len(node_id) > 0
        assert "-" in node_id  # UUID format

    @patch("missy.channels.voice.pairing.event_bus")
    def test_idempotent_existing_node(self, mock_bus):
        reg = _make_registry()
        reg.get_node.return_value = _make_node("existing")
        mgr = PairingManager(reg)
        node_id = mgr.initiate_pairing(
            node_id="existing",
            friendly_name="X",
            room="r",
            ip_address="1.2.3.4",
            hardware_profile={},
        )
        assert node_id == "existing"
        reg.add_node.assert_not_called()


class TestApprovePairing:
    @patch("missy.channels.voice.pairing.event_bus")
    def test_approve_pending_node(self, mock_bus):
        reg = _make_registry()
        reg.get_node.return_value = _make_node("node-1", paired=False)
        reg.generate_token.return_value = "secret-token-123"
        mgr = PairingManager(reg)
        token = mgr.approve_pairing("node-1")
        assert token == "secret-token-123"
        reg.approve_node.assert_called_once_with("node-1")
        mock_bus.publish.assert_called_once()

    def test_approve_not_found(self):
        reg = _make_registry()
        mgr = PairingManager(reg)
        with pytest.raises(ValueError, match="not found"):
            mgr.approve_pairing("nonexistent")

    @patch("missy.channels.voice.pairing.event_bus")
    def test_approve_already_paired(self, mock_bus):
        reg = _make_registry()
        reg.get_node.return_value = _make_node("node-1", paired=True)
        mgr = PairingManager(reg)
        with pytest.raises(ValueError, match="already paired"):
            mgr.approve_pairing("node-1")


class TestRejectPairing:
    @patch("missy.channels.voice.pairing.event_bus")
    def test_reject_pending(self, mock_bus):
        reg = _make_registry()
        reg.get_node.return_value = _make_node("node-1", paired=False)
        mgr = PairingManager(reg)
        mgr.reject_pairing("node-1")
        reg.remove_node.assert_called_once_with("node-1")

    @patch("missy.channels.voice.pairing.event_bus")
    def test_reject_not_found_noop(self, mock_bus):
        reg = _make_registry()
        mgr = PairingManager(reg)
        mgr.reject_pairing("nonexistent")  # No error
        reg.remove_node.assert_not_called()

    @patch("missy.channels.voice.pairing.event_bus")
    def test_reject_already_paired(self, mock_bus):
        reg = _make_registry()
        reg.get_node.return_value = _make_node("node-1", paired=True)
        mgr = PairingManager(reg)
        with pytest.raises(ValueError, match="already paired"):
            mgr.reject_pairing("node-1")


class TestListPending:
    def test_delegates_to_registry(self):
        reg = _make_registry()
        reg.list_pending.return_value = [_make_node("p1"), _make_node("p2")]
        mgr = PairingManager(reg)
        pending = mgr.list_pending()
        assert len(pending) == 2


class TestUnpairNode:
    @patch("missy.channels.voice.pairing.event_bus")
    def test_unpair_existing(self, mock_bus):
        reg = _make_registry()
        reg.get_node.return_value = _make_node("node-1", paired=True)
        mgr = PairingManager(reg)
        mgr.unpair_node("node-1")
        reg.remove_node.assert_called_once_with("node-1")
        mock_bus.publish.assert_called_once()

    @patch("missy.channels.voice.pairing.event_bus")
    def test_unpair_not_found_noop(self, mock_bus):
        reg = _make_registry()
        mgr = PairingManager(reg)
        mgr.unpair_node("nonexistent")
        reg.remove_node.assert_not_called()
