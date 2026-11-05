use crate::{Purpose, VerifiedEnvelope, envelope::validate_hex};
use anyhow::{Result, ensure};
use missy_core::Scope;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vote {
    pub proposal_digest: String,
    pub approve: bool,
}
#[derive(Clone, Debug, Serialize)]
pub struct QuorumResult {
    pub approved: bool,
    pub approvals: usize,
    pub rejections: usize,
    pub required: usize,
    pub voters: Vec<String>,
}

/// A local proposal policy. This is not peer-supplied, and satisfying it never
/// modifies grants or configuration automatically.
pub struct Quorum {
    recipient: String,
    scope: Scope,
    proposal_digest: String,
    eligible: BTreeSet<String>,
    threshold: usize,
    expires: u64,
    votes: BTreeMap<String, (bool, u64)>,
}
impl Quorum {
    pub fn new(
        recipient: String,
        scope: Scope,
        proposal_digest: String,
        eligible: BTreeSet<String>,
        threshold: usize,
        expires: u64,
    ) -> Result<Self> {
        validate_hex(&recipient, 64)?;
        validate_hex(&proposal_digest, 64)?;
        scope.validate()?;
        ensure!(
            !eligible.is_empty()
                && eligible.len() <= 256
                && threshold > 0
                && threshold <= eligible.len(),
            "invalid quorum threshold"
        );
        for id in &eligible {
            validate_hex(id, 64)?;
            ensure!(id != &recipient, "self-vote forbidden");
        }
        Ok(Self {
            recipient,
            scope,
            proposal_digest,
            eligible,
            threshold,
            expires,
            votes: BTreeMap::new(),
        })
    }
    pub fn add(&mut self, verified: VerifiedEnvelope, now: u64) -> Result<()> {
        let body = verified.body();
        ensure!(
            now < self.expires
                && now < body.expiry
                && body.purpose == Purpose::PolicyVote
                && body.recipient == self.recipient
                && body.scope == self.scope,
            "vote context or expiry mismatch"
        );
        let vote: Vote = serde_json::from_value(body.payload.clone())?;
        ensure!(
            vote.proposal_digest == self.proposal_digest && self.eligible.contains(&body.sender),
            "proposal or voter mismatch"
        );
        ensure!(!self.votes.contains_key(&body.sender), "duplicate voter");
        self.votes
            .insert(body.sender.clone(), (vote.approve, body.expiry));
        Ok(())
    }
    pub fn result(&self, now: u64) -> QuorumResult {
        let approvals = self
            .votes
            .values()
            .filter(|(approve, expiry)| *approve && now < *expiry)
            .count();
        let rejections = self
            .votes
            .values()
            .filter(|(approve, expiry)| !*approve && now < *expiry)
            .count();
        QuorumResult {
            approved: now < self.expires && approvals >= self.threshold,
            approvals,
            rejections,
            required: self.threshold,
            voters: self.votes.keys().cloned().collect(),
        }
    }
}
