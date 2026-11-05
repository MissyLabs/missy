//! The default vectors encode hashed lexical token frequencies, NOT neural semantics.
use anyhow::{Result, ensure};
use sha2::{Digest, Sha256};

/// Implementations must be deterministic, local, bounded, and provide a stable model/version ID.
/// A real local model can implement this later; no model is downloaded or invoked by this crate.
pub trait Embedder: Send + Sync {
    fn name(&self) -> &str;
    fn dimension(&self) -> usize;
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}
pub struct HashingEmbedder;
impl Embedder for HashingEmbedder {
    fn name(&self) -> &str {
        "sha256-token-frequency-lexical-v1-not-neural"
    }
    fn dimension(&self) -> usize {
        256
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.dimension()];
        for token in tokens(text) {
            let hash = Sha256::digest(token.as_bytes());
            vector[usize::from(hash[0])] += 1.0;
        }
        normalize(vector, self.dimension())
    }
}
pub(crate) fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(128).flat_map(char::to_lowercase).collect())
        .collect()
}
pub(crate) fn normalize(mut vector: Vec<f32>, dimension: usize) -> Result<Vec<f32>> {
    ensure!(
        vector.len() == dimension && vector.iter().all(|v| v.is_finite()),
        "invalid embedding vector"
    );
    let norm = vector
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    Ok(vector)
}
pub(crate) fn encode(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|v| v.to_le_bytes()).collect()
}
pub(crate) fn decode(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    ensure!(
        bytes.len() == dimension * 4,
        "stored vector length corruption"
    );
    let vector: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    ensure!(
        vector.iter().all(|v| v.is_finite()),
        "stored vector corruption"
    );
    Ok(vector)
}
