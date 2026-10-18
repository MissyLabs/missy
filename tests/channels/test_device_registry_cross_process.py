"""CLI device changes must survive the gateway's heartbeat writes (premortem item 11)."""

from __future__ import annotations

from missy.channels.voice.registry import DeviceRegistry, EdgeNode


def _node() -> EdgeNode:
    return EdgeNode(node_id="n1", friendly_name="Kitchen", room="kitchen", ip_address="10.0.0.5")


def test_cli_mute_survives_gateway_heartbeat_and_is_visible(tmp_path) -> None:
    path = str(tmp_path / "devices.json")
    gateway = DeviceRegistry(path)
    gateway.load()
    gateway.add_node(_node())

    cli = DeviceRegistry(path)  # a separate `missy devices policy` process
    cli.load()
    cli.update_node("n1", policy_mode="muted")

    gateway.mark_online("n1", "10.0.0.5")  # heartbeat on the long-lived instance

    assert gateway.get_node("n1").policy_mode == "muted"
    fresh = DeviceRegistry(path)
    fresh.load()
    assert fresh.get_node("n1").policy_mode == "muted"
    assert fresh.get_node("n1").status == "online"


def test_cli_unpair_is_seen_by_gateway(tmp_path) -> None:
    path = str(tmp_path / "devices.json")
    gateway = DeviceRegistry(path)
    gateway.load()
    gateway.add_node(_node())

    cli = DeviceRegistry(path)
    cli.load()
    cli.remove_node("n1")

    assert gateway.get_node("n1") is None
