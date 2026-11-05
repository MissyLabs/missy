use crate::{
    ApprovalRegistry, fingerprint, identifier,
    registry::{audit, scope_key},
};
use anyhow::{Result, ensure};
use missy_core::Scope;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
/// Advisory only. Maximum trust never implies approval or a policy grant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TrustScore {
    pub score: f64,
    pub version: i64,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationOutcome {
    Succeeded,
    Failed,
    Unknown,
}
impl ApprovalRegistry {
    pub fn trust(&self, scope: &Scope) -> Result<TrustScore> {
        let conn = self.connect()?;
        let key = scope_key(scope)?;
        Ok(conn
            .query_row(
                "SELECT score,version FROM trust WHERE scope_key=?1",
                [key],
                |r| {
                    Ok(TrustScore {
                        score: r.get(0)?,
                        version: r.get(1)?,
                    })
                },
            )
            .optional()?
            .unwrap_or(TrustScore {
                score: 0.0,
                version: 0,
            }))
    }
    /// Explicit, finite operator delta with optimistic concurrency. Failed or
    /// unknown outcomes cannot raise trust. Success alone changes nothing.
    pub fn update_trust(
        &self,
        scope: &Scope,
        expected_version: i64,
        delta: f64,
        outcome: OperationOutcome,
        operator_id: &str,
    ) -> Result<TrustScore> {
        identifier(operator_id, "operator ID")?;
        ensure!(
            delta.is_finite() && (-2.0..=2.0).contains(&delta),
            "trust delta must be finite and within [-2,2]"
        );
        ensure!(
            outcome == OperationOutcome::Succeeded || delta <= 0.0,
            "failed/unknown operation cannot increase trust"
        );
        let key = scope_key(scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let old = tx
            .query_row(
                "SELECT score,version FROM trust WHERE scope_key=?1",
                [&key],
                |r| {
                    Ok(TrustScore {
                        score: r.get(0)?,
                        version: r.get(1)?,
                    })
                },
            )
            .optional()?
            .unwrap_or(TrustScore {
                score: 0.0,
                version: 0,
            });
        ensure!(old.version == expected_version, "trust version conflict");
        ensure!(old.score.is_finite(), "stored trust score is nonfinite");
        let next = TrustScore {
            score: (old.score + delta).clamp(-1.0, 1.0),
            version: old
                .version
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("trust version overflow"))?,
        };
        tx.execute(
            "INSERT INTO trust(scope_key,score,version) VALUES(?1,?2,?3)
            ON CONFLICT(scope_key) DO UPDATE SET score=excluded.score,version=excluded.version",
            params![key, next.score, next.version],
        )?;
        audit(
            &tx,
            &key,
            "trust",
            "trust_operator_updated",
            &fingerprint(&(next, outcome))?,
            Some(operator_id),
        )?;
        tx.commit()?;
        Ok(next)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_scoped_explicit_updates_and_no_failed_grants() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let a = ApprovalRegistry::open(dir.path().join("db"))?;
        let scope = Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        };
        assert_eq!(a.trust(&scope)?.score, 0.0);
        for delta in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 3.0] {
            assert!(
                a.update_trust(&scope, 0, delta, OperationOutcome::Succeeded, "op")
                    .is_err()
            );
        }
        assert!(
            a.update_trust(&scope, 0, 0.1, OperationOutcome::Failed, "op")
                .is_err()
        );
        assert!(
            a.update_trust(&scope, 0, 0.1, OperationOutcome::Unknown, "op")
                .is_err()
        );
        assert_eq!(
            a.update_trust(&scope, 0, 2.0, OperationOutcome::Succeeded, "op")?
                .score,
            1.0
        );
        assert!(
            a.update_trust(&scope, 0, -0.1, OperationOutcome::Failed, "op")
                .is_err()
        );
        assert_eq!(
            a.update_trust(&scope, 1, -2.0, OperationOutcome::Failed, "op")?
                .score,
            -1.0
        );
        let mut other = scope;
        other.session = "other".into();
        assert_eq!(a.trust(&other)?.score, 0.0);
        Ok(())
    }
}
