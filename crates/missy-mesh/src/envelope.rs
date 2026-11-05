use crate::identity::{Identity, public_key};
use anyhow::{Result, ensure};
use ed25519_dalek::Signature;
use missy_core::Scope;
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const MAX_ENVELOPE_BYTES: usize = 64 * 1024;
pub const MAX_TTL_SECONDS: u64 = 300;
pub const CLOCK_SKEW_SECONDS: u64 = 30;
const DOMAIN: &[u8] = b"missy-native-mesh-envelope-v1\0";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    MemoryRead,
    MemoryWrite,
    Delegate,
    Gossip,
    PolicyVote,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    MemoryRead,
    MemoryUpdate,
    DelegateRequest,
    Gossip,
    PolicyVote,
}
impl Purpose {
    pub fn capability(self) -> Capability {
        match self {
            Self::MemoryRead => Capability::MemoryRead,
            Self::MemoryUpdate => Capability::MemoryWrite,
            Self::DelegateRequest => Capability::Delegate,
            Self::Gossip => Capability::Gossip,
            Self::PolicyVote => Capability::PolicyVote,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnvelopeBody {
    pub version: u32,
    pub sender: String,
    pub recipient: String,
    pub scope: Scope,
    pub purpose: Purpose,
    pub timestamp: u64,
    pub nonce: String,
    pub expiry: u64,
    pub payload: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SignedEnvelope {
    pub body: EnvelopeBody,
    pub signature: String,
}

/// Cannot be constructed, deserialized, or cloned by callers. A verified value
/// represents one durable nonce admission on one receiving identity.
pub struct VerifiedEnvelope {
    pub(crate) envelope: SignedEnvelope,
}
impl VerifiedEnvelope {
    pub fn body(&self) -> &EnvelopeBody {
        &self.envelope.body
    }
}
impl SignedEnvelope {
    pub fn sign(
        identity: &Identity,
        recipient: String,
        scope: Scope,
        purpose: Purpose,
        payload: Value,
        now: u64,
        ttl: u64,
    ) -> Result<Self> {
        ensure!(ttl > 0 && ttl <= MAX_TTL_SECONDS, "invalid envelope TTL");
        let mut nonce = [0u8; 16];
        OsRng.fill_bytes(&mut nonce);
        let body = EnvelopeBody {
            version: 1,
            sender: identity.peer_id(),
            recipient,
            scope,
            purpose,
            timestamp: now,
            nonce: hex::encode(nonce),
            expiry: now
                .checked_add(ttl)
                .ok_or_else(|| anyhow::anyhow!("expiry overflow"))?,
            payload,
        };
        body.validate(now)?;
        let signature = identity.sign(&body.signing_bytes()?);
        let envelope = Self { body, signature };
        envelope.to_json()?;
        Ok(envelope)
    }
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() <= MAX_ENVELOPE_BYTES, "envelope exceeds 64KiB");
        // Reject duplicate JSON keys, including nested payloads, before typed parsing.
        let value = crate::json::parse(bytes)?;
        Ok(serde_json::from_value(value)?)
    }
    pub fn to_json(&self) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(self)?;
        ensure!(bytes.len() <= MAX_ENVELOPE_BYTES, "envelope exceeds 64KiB");
        Ok(bytes)
    }
    pub(crate) fn authenticate(&self, key: &str) -> Result<()> {
        ensure!(self.signature.len() == 128, "invalid signature length");
        let sig = Signature::from_slice(&hex::decode(&self.signature)?)?;
        public_key(key)?.verify_strict(&self.body.signing_bytes()?, &sig)?;
        Ok(())
    }
}
impl EnvelopeBody {
    pub fn signing_bytes(&self) -> Result<Vec<u8>> {
        let value = serde_json::to_value(self)?;
        let canonical = crate::json::canonical(&value)?;
        ensure!(
            canonical.len() <= MAX_ENVELOPE_BYTES,
            "signing payload exceeds bound"
        );
        Ok([DOMAIN, &canonical].concat())
    }
    pub(crate) fn validate(&self, now: u64) -> Result<()> {
        ensure!(self.version == 1, "unsupported mesh version");
        self.scope.validate()?;
        for id in [&self.sender, &self.recipient] {
            validate_hex(id, 64)?;
        }
        validate_hex(&self.nonce, 32)?;
        ensure!(self.sender != self.recipient, "mesh reflection denied");
        ensure!(
            self.timestamp > 0 && self.timestamp <= now.saturating_add(CLOCK_SKEW_SECONDS),
            "future or invalid timestamp"
        );
        ensure!(
            self.expiry > self.timestamp
                && self.expiry - self.timestamp <= MAX_TTL_SECONDS
                && self.expiry > now,
            "expired or invalid envelope lifetime"
        );
        self.signing_bytes()?;
        Ok(())
    }
}
pub(crate) fn validate_hex(s: &str, len: usize) -> Result<()> {
    ensure!(
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid canonical hex identifier"
    );
    Ok(())
}
