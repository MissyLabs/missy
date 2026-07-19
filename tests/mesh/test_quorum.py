"""Tests for the distributed policy quorum (F01)."""

from __future__ import annotations

from missy.mesh.peer_registry import PeerRegistry
from missy.mesh.quorum import PolicyQuorum


def _registry_with_voters(*nodes, trust=800):
    reg = PeerRegistry()
    for n in nodes:
        reg.add_peer(n.public_key, capabilities={"policy.vote"}, trust=trust)
    return reg


class TestQuorum:
    def test_threshold_met(self, node_a, node_b) -> None:
        reg = _registry_with_voters(node_a, node_b)
        q = PolicyQuorum(reg)
        votes = [
            PolicyQuorum.build_vote(
                identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
            ),
            PolicyQuorum.build_vote(
                identity=node_b.identity, sender=node_b.peer_id, proposal_id="P", approve=True
            ),
        ]
        result = q.evaluate("P", votes, threshold=2)
        assert result.approved
        assert result.approvals == 2
        assert set(result.voters) == {node_a.peer_id, node_b.peer_id}

    def test_threshold_not_met(self, node_a, node_b) -> None:
        reg = _registry_with_voters(node_a, node_b)
        votes = [
            PolicyQuorum.build_vote(
                identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
            ),
        ]
        result = PolicyQuorum(reg).evaluate("P", votes, threshold=2)
        assert not result.approved
        assert result.approvals == 1
        assert "insufficient" in result.reason

    def test_forged_vote_discarded(self, node_a, node_b, node_c) -> None:
        # node_c signs a vote but claims to be node_a → signature check fails.
        reg = _registry_with_voters(node_a, node_b)
        forged = PolicyQuorum.build_vote(
            identity=node_c.identity, sender=node_a.peer_id, proposal_id="P", approve=True
        )
        genuine = PolicyQuorum.build_vote(
            identity=node_b.identity, sender=node_b.peer_id, proposal_id="P", approve=True
        )
        result = PolicyQuorum(reg).evaluate("P", [forged, genuine], threshold=2)
        assert not result.approved
        assert result.approvals == 1  # only the genuine vote counted

    def test_vote_without_capability_discarded(self, node_a, node_b) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key)  # no policy.vote grant
        reg.add_peer(node_b.public_key, capabilities={"policy.vote"})
        votes = [
            PolicyQuorum.build_vote(
                identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
            ),
            PolicyQuorum.build_vote(
                identity=node_b.identity, sender=node_b.peer_id, proposal_id="P", approve=True
            ),
        ]
        result = PolicyQuorum(reg).evaluate("P", votes, threshold=2)
        assert result.approvals == 1  # node_a's vote dropped

    def test_duplicate_votes_counted_once(self, node_a, node_b) -> None:
        reg = _registry_with_voters(node_a, node_b)
        v1 = PolicyQuorum.build_vote(
            identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
        )
        v2 = PolicyQuorum.build_vote(
            identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
        )
        result = PolicyQuorum(reg).evaluate("P", [v1, v2], threshold=2)
        assert result.approvals == 1  # same voter, counted once

    def test_wrong_proposal_id_ignored(self, node_a) -> None:
        reg = _registry_with_voters(node_a)
        vote = PolicyQuorum.build_vote(
            identity=node_a.identity, sender=node_a.peer_id, proposal_id="OTHER", approve=True
        )
        result = PolicyQuorum(reg).evaluate("P", [vote], threshold=1)
        assert result.approvals == 0

    def test_min_trust_filter(self, node_a, node_b) -> None:
        reg = PeerRegistry()
        reg.add_peer(node_a.public_key, capabilities={"policy.vote"}, trust=100)
        reg.add_peer(node_b.public_key, capabilities={"policy.vote"}, trust=900)
        q = PolicyQuorum(reg, min_trust=500)
        votes = [
            PolicyQuorum.build_vote(
                identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
            ),
            PolicyQuorum.build_vote(
                identity=node_b.identity, sender=node_b.peer_id, proposal_id="P", approve=True
            ),
        ]
        result = q.evaluate("P", votes, threshold=1)
        assert result.voters == [node_b.peer_id]  # low-trust node_a excluded

    def test_rejections_counted(self, node_a, node_b) -> None:
        reg = _registry_with_voters(node_a, node_b)
        votes = [
            PolicyQuorum.build_vote(
                identity=node_a.identity, sender=node_a.peer_id, proposal_id="P", approve=True
            ),
            PolicyQuorum.build_vote(
                identity=node_b.identity, sender=node_b.peer_id, proposal_id="P", approve=False
            ),
        ]
        result = PolicyQuorum(reg).evaluate("P", votes, threshold=2)
        assert result.approvals == 1 and result.rejections == 1
        assert not result.approved
