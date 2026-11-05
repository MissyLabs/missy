use crate::private::split;
use anyhow::{Context, Result, ensure};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rand::{RngCore, rngs::OsRng};
use sha2::{Digest, Sha256};
use std::path::Path;
use zeroize::Zeroizing;

/// Private Ed25519 identity, deliberately neither Debug nor Serialize.
pub struct Identity(SigningKey);
impl Identity {
    /// Explicit initialization only. Parent must already be private (0700).
    /// Existing keys are never overwritten or regenerated.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let (dir, name) = split(path.as_ref())?;
        let mut seed = Zeroizing::new([0u8; 32]);
        OsRng.fill_bytes(seed.as_mut());
        let key = Self(SigningKey::from_bytes(&seed));
        dir.create(&name, seed.as_ref())?;
        Ok(key)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let (dir, name) = split(path.as_ref())?;
        let bytes = Zeroizing::new(
            dir.read(&name, 32)?
                .context("mesh identity missing; initialize explicitly")?,
        );
        let seed: Zeroizing<[u8; 32]> = Zeroizing::new(
            bytes
                .as_slice()
                .try_into()
                .context("invalid identity length")?,
        );
        Ok(Self(SigningKey::from_bytes(&seed)))
    }
    pub fn public_key(&self) -> String {
        hex::encode(self.0.verifying_key().as_bytes())
    }
    pub fn peer_id(&self) -> String {
        peer_id(&self.0.verifying_key())
    }
    pub(crate) fn sign(&self, bytes: &[u8]) -> String {
        hex::encode(self.0.sign(bytes).to_bytes())
    }
}
pub fn peer_id(key: &VerifyingKey) -> String {
    hex::encode(Sha256::digest(key.as_bytes()))
}
pub(crate) fn public_key(encoded: &str) -> Result<VerifyingKey> {
    ensure!(
        encoded.len() == 64
            && encoded
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "public key must be lowercase hex"
    );
    let bytes: [u8; 32] = hex::decode(encoded)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("public key length"))?;
    let key = VerifyingKey::from_bytes(&bytes)?;
    ensure!(!key.is_weak(), "weak Ed25519 key denied");
    Ok(key)
}
