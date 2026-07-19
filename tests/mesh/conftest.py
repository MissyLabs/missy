"""Shared fixtures for mesh tests (F01)."""

from __future__ import annotations

import pytest

from missy.mesh.identity import encode_public_key, local_peer_id, local_public_key_raw
from missy.security.identity import AgentIdentity


class Node:
    """Bundle of an identity + its derived peer_id + encoded public key."""

    def __init__(self) -> None:
        self.identity = AgentIdentity.generate()
        self.peer_id = local_peer_id(self.identity)
        self.public_key = encode_public_key(local_public_key_raw(self.identity))


@pytest.fixture
def node_a() -> Node:
    return Node()


@pytest.fixture
def node_b() -> Node:
    return Node()


@pytest.fixture
def node_c() -> Node:
    return Node()
