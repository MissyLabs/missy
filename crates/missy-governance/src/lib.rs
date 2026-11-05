//! Durable operator-controlled governance. Approval adds a gate and never bypasses
//! the inner executor's policy. No process-wide sandbox or implicit trust grants.
mod approvals;
mod documents;
mod guarded;
mod registry;
mod trust;
use anyhow::Result;
pub use approvals::*;
pub use documents::*;
pub use guarded::*;
pub use registry::{ApprovalRegistry, AuditEvent};
use serde::Serialize;
use sha2::{Digest, Sha256};
pub use trust::*;

/// Stable JSON hashing: recursively sorted object keys, significant array order.
pub fn fingerprint<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    fn canonical(value: serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let ordered: std::collections::BTreeMap<_, _> =
                    map.into_iter().map(|(k, v)| (k, canonical(v))).collect();
                serde_json::Value::Object(ordered.into_iter().collect())
            }
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.into_iter().map(canonical).collect())
            }
            other => other,
        }
    }
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(
        &canonical(serde_json::to_value(value)?),
    )?)))
}
pub(crate) fn now() -> Result<i64> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs()
        .try_into()?)
}
pub(crate) fn identifier(value: &str, label: &str) -> Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= 512 && !value.chars().any(char::is_control),
        "invalid {label}"
    );
    Ok(())
}
pub(crate) fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
