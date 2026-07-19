"""Distributed policy quorum for the Leyline mesh (F01).

A capability-*widening* action on the mesh (e.g. granting a peer a new
capability, or accepting a mesh-wide policy change) must not be unilaterally
decidable by a single node — otherwise one compromised node could widen the
whole mesh's authority. :class:`PolicyQuorum` collects Ed25519-signed votes,
verifies each against the :class:`PeerRegistry` (so a forged or unknown-peer
vote is discarded), de-duplicates by voter, and requires a configurable
threshold of *distinct, authenticated, trusted* approvals before the action is
allowed. Fail-closed: too few valid approvals → denied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from missy.mesh.envelope import SignedEnvelope


@dataclass
class QuorumResult:
    """Outcome of evaluating a set of votes for a proposal."""

    approved: bool
    approvals: int
    rejections: int
    required: int
    voters: list[str]
    reason: str = ""


class PolicyQuorum:
    """Evaluates signed votes on a mesh proposal against a threshold."""

    def __init__(self, registry, *, min_trust: int = 0) -> None:
        self._registry = registry
        self._min_trust = min_trust

    def evaluate(
        self,
        proposal_id: str,
        votes: list[SignedEnvelope],
        *,
        threshold: int,
        capability: str = "policy.vote",
    ) -> QuorumResult:
        """Return a :class:`QuorumResult` for ``proposal_id``.

        A vote counts only if: its signature verifies against the sender's
        registered key; the sender holds the ``capability`` grant; the sender
        meets ``min_trust``; the vote is for this ``proposal_id``; and the
        sender has not already voted (first vote per peer wins).
        """
        seen: dict[str, bool] = {}
        for env in votes:
            if env.kind != "policy.vote":
                continue
            if env.payload.get("proposal_id") != proposal_id:
                continue
            ok, _reason = env.verify(self._registry)
            if not ok:
                continue
            peer = self._registry.get(env.sender)
            if peer is None or self._registry.is_allowed(env.sender, capability) is False:
                continue
            if peer.trust < self._min_trust:
                continue
            if env.sender in seen:
                continue  # first vote per peer is authoritative
            seen[env.sender] = bool(env.payload.get("approve", False))

        approvals = sum(1 for v in seen.values() if v)
        rejections = sum(1 for v in seen.values() if not v)
        approved = approvals >= threshold
        return QuorumResult(
            approved=approved,
            approvals=approvals,
            rejections=rejections,
            required=threshold,
            voters=sorted(seen),
            reason="ok" if approved else "insufficient approvals",
        )

    @staticmethod
    def build_vote(*, identity, sender: str, proposal_id: str, approve: bool) -> SignedEnvelope:
        """Construct a signed ``policy.vote`` envelope."""
        payload: dict[str, Any] = {"proposal_id": proposal_id, "approve": approve}
        return SignedEnvelope.create(
            identity=identity, sender=sender, kind="policy.vote", payload=payload
        )
