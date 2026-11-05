use crate::{
    ApprovalRegistry, digest, fingerprint, identifier, now,
    registry::{audit, scope_key},
};
use anyhow::{Result, ensure};
use missy_core::{Scope, ToolCall, ToolContext};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ApprovalBinding {
    pub scope: Scope,
    pub turn_id: String,
    pub call_id: String,
    pub tool_name: String,
    pub arguments_hash: String,
    /// Hash policy/config/provider/tool/persona snapshot, recomputed on resume,
    /// never accepted from model input or an unverified restored checkpoint.
    pub context_fingerprint: String,
}
impl ApprovalBinding {
    pub fn new(ctx: &ToolContext, call: &ToolCall, context_fingerprint: &str) -> Result<Self> {
        let binding = Self {
            scope: ctx.scope.clone(),
            turn_id: ctx.turn_id.clone(),
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            arguments_hash: fingerprint(&call.arguments)?,
            context_fingerprint: context_fingerprint.into(),
        };
        binding.validate()?;
        Ok(binding)
    }
    pub fn validate(&self) -> Result<()> {
        self.scope.validate()?;
        identifier(&self.turn_id, "turn ID")?;
        identifier(&self.call_id, "call ID")?;
        identifier(&self.tool_name, "tool name")?;
        ensure!(
            digest(&self.arguments_hash) && digest(&self.context_fingerprint),
            "expected lowercase SHA-256 fingerprints"
        );
        Ok(())
    }
    pub fn hash(&self) -> Result<String> {
        self.validate()?;
        fingerprint(self)
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Denied,
    Revoked,
    Expired,
    Consumed,
}
impl ApprovalStatus {
    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "pending" => Self::Pending,
            "approved" => Self::Approved,
            "denied" => Self::Denied,
            "revoked" => Self::Revoked,
            "expired" => Self::Expired,
            "consumed" => Self::Consumed,
            _ => anyhow::bail!("invalid approval status"),
        })
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Approve,
    Deny,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApprovalRecord {
    pub id: String,
    pub binding: ApprovalBinding,
    pub binding_hash: String,
    pub status: ApprovalStatus,
    pub created_at: i64,
    pub expires_at: i64,
    /// Not a bearer credential. Exact scoped binding and database state remain
    /// authoritative; consume clears the token in the same transaction.
    pub token_id: Option<String>,
}
/// The core error is the runtime-recognized definite pre-dispatch pause signal.
pub type ApprovalRequiredError = missy_core::ApprovalRequired;
#[derive(Debug, Clone, thiserror::Error)]
#[error("operator approval refused: {status:?} (request {request_id})")]
pub struct ApprovalRefusedError {
    pub request_id: String,
    pub status: ApprovalStatus,
}
#[derive(Debug)]
pub struct ConsumedApproval {
    pub request_id: String,
    pub token_id: String,
    pub binding_hash: String,
}

fn load(conn: &rusqlite::Connection, scope: &str, id: &str) -> Result<Option<ApprovalRecord>> {
    let raw = conn
        .query_row(
            "SELECT id,binding,binding_hash,status,created_at,expires_at,token_id
        FROM approval WHERE scope_key=?1 AND id=?2",
            params![scope, id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, Option<String>>(6)?,
                ))
            },
        )
        .optional()?;
    raw.map(
        |(id, binding, binding_hash, status, created_at, expires_at, token_id)| {
            Ok(ApprovalRecord {
                id,
                binding: serde_json::from_str(&binding)?,
                binding_hash,
                status: ApprovalStatus::parse(&status)?,
                created_at,
                expires_at,
                token_id,
            })
        },
    )
    .transpose()
}
fn expire(
    tx: &rusqlite::Transaction<'_>,
    key: &str,
    record: &mut ApprovalRecord,
    time: i64,
) -> Result<()> {
    if time >= record.expires_at
        && matches!(
            record.status,
            ApprovalStatus::Pending | ApprovalStatus::Approved
        )
    {
        tx.execute(
            "UPDATE approval SET status='expired',token_id=NULL WHERE id=?1",
            [&record.id],
        )?;
        record.status = ApprovalStatus::Expired;
        record.token_id = None;
        audit(
            tx,
            key,
            &record.id,
            "approval_expired",
            &record.binding_hash,
            None,
        )?;
    }
    Ok(())
}
impl ApprovalRegistry {
    /// Idempotent entire binding. Never silently reissue terminal decisions.
    pub fn request(&self, binding: &ApprovalBinding, ttl_seconds: u64) -> Result<ApprovalRecord> {
        self.request_at(binding, ttl_seconds, now()?)
    }
    fn request_at(
        &self,
        binding: &ApprovalBinding,
        ttl_seconds: u64,
        time: i64,
    ) -> Result<ApprovalRecord> {
        ensure!(
            (1..=86400).contains(&ttl_seconds),
            "approval TTL must be 1..86400 seconds"
        );
        let hash = binding.hash()?;
        let key = scope_key(&binding.scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<String> = tx
            .query_row(
                "SELECT id FROM approval WHERE binding_hash=?1",
                [&hash],
                |r| r.get(0),
            )
            .optional()?;
        let record = if let Some(id) = existing {
            let mut record =
                load(&tx, &key, &id)?.ok_or_else(|| anyhow::anyhow!("approval scope mismatch"))?;
            expire(&tx, &key, &mut record, time)?;
            record
        } else {
            let id = uuid::Uuid::new_v4().to_string();
            let expires_at = time
                .checked_add(ttl_seconds as i64)
                .ok_or_else(|| anyhow::anyhow!("invalid expiry"))?;
            tx.execute("INSERT INTO approval(id,scope_key,binding_hash,binding,status,created_at,expires_at)
                VALUES(?1,?2,?3,?4,'pending',?5,?6)", params![id,key,hash,serde_json::to_string(binding)?,time,expires_at])?;
            audit(&tx, &key, &id, "approval_requested", &hash, None)?;
            load(&tx, &key, &id)?.expect("inserted approval")
        };
        tx.commit()?;
        Ok(record)
    }
    pub fn get(&self, scope: &Scope, id: &str) -> Result<Option<ApprovalRecord>> {
        let key = scope_key(scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut record = load(&tx, &key, id)?;
        if let Some(record) = &mut record {
            expire(&tx, &key, record, now()?)?;
        }
        tx.commit()?;
        Ok(record)
    }
    pub fn list(&self, scope: &Scope, limit: usize) -> Result<Vec<ApprovalRecord>> {
        let key = scope_key(scope)?;
        let conn = self.connect()?;
        let ids = conn
            .prepare(
                "SELECT id FROM approval WHERE scope_key=?1 ORDER BY created_at DESC,id LIMIT ?2",
            )?
            .query_map(params![key, limit.min(1000) as i64], |r| {
                r.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ids.iter()
            .map(|id| {
                self.get(scope, id)?
                    .ok_or_else(|| anyhow::anyhow!("approval disappeared"))
            })
            .collect()
    }
    /// Authenticated operator surface only. expected_binding_hash binds the
    /// decision to the exact inspected request; identity strings are audit hashes.
    pub fn decide(
        &self,
        scope: &Scope,
        id: &str,
        expected_binding_hash: &str,
        decision: Decision,
        operator_id: &str,
    ) -> Result<ApprovalRecord> {
        identifier(operator_id, "operator ID")?;
        let key = scope_key(scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut record =
            load(&tx, &key, id)?.ok_or_else(|| anyhow::anyhow!("approval not found in scope"))?;
        ensure!(
            record.binding_hash == expected_binding_hash,
            "approval binding mismatch"
        );
        expire(&tx, &key, &mut record, now()?)?;
        if record.status != ApprovalStatus::Pending {
            tx.commit()?;
            return Err(ApprovalRefusedError {
                request_id: id.into(),
                status: record.status,
            }
            .into());
        }
        let (status, event, token) = match decision {
            Decision::Approve => (
                "approved",
                "approval_approved",
                Some(uuid::Uuid::new_v4().to_string()),
            ),
            Decision::Deny => ("denied", "approval_denied", None),
        };
        tx.execute("UPDATE approval SET status=?1,decided_at=?2,token_id=?3 WHERE id=?4 AND status='pending'",
            params![status,now()?,token,id])?;
        audit(
            &tx,
            &key,
            id,
            event,
            &record.binding_hash,
            Some(operator_id),
        )?;
        let record = load(&tx, &key, id)?.expect("existing approval");
        tx.commit()?;
        Ok(record)
    }
    pub fn revoke(
        &self,
        scope: &Scope,
        id: &str,
        expected_binding_hash: &str,
        operator_id: &str,
    ) -> Result<()> {
        identifier(operator_id, "operator ID")?;
        let key = scope_key(scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut record =
            load(&tx, &key, id)?.ok_or_else(|| anyhow::anyhow!("approval not found in scope"))?;
        ensure!(
            record.binding_hash == expected_binding_hash,
            "approval binding mismatch"
        );
        expire(&tx, &key, &mut record, now()?)?;
        if !matches!(
            record.status,
            ApprovalStatus::Pending | ApprovalStatus::Approved
        ) {
            tx.commit()?;
            return Err(ApprovalRefusedError {
                request_id: id.into(),
                status: record.status,
            }
            .into());
        }
        tx.execute(
            "UPDATE approval SET status='revoked',token_id=NULL WHERE id=?1",
            [id],
        )?;
        audit(
            &tx,
            &key,
            id,
            "approval_revoked",
            &record.binding_hash,
            Some(operator_id),
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn consume(&self, binding: &ApprovalBinding) -> Result<ConsumedApproval> {
        self.consume_at(binding, now()?)
    }
    fn consume_at(&self, binding: &ApprovalBinding, time: i64) -> Result<ConsumedApproval> {
        let hash = binding.hash()?;
        let key = scope_key(&binding.scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let id: Option<String> = tx
            .query_row(
                "SELECT id FROM approval WHERE scope_key=?1 AND binding_hash=?2",
                params![key, hash],
                |r| r.get(0),
            )
            .optional()?;
        let id = id.ok_or_else(|| anyhow::anyhow!("exact approval binding not found"))?;
        let mut record = load(&tx, &key, &id)?.expect("existing approval");
        ensure!(record.binding == *binding, "approval binding mismatch");
        expire(&tx, &key, &mut record, time)?;
        let refusal = match record.status {
            ApprovalStatus::Pending => Some(anyhow::Error::new(ApprovalRequiredError {
                request_id: id.clone(),
            })),
            ApprovalStatus::Approved => None,
            status => Some(anyhow::Error::new(ApprovalRefusedError {
                request_id: id.clone(),
                status,
            })),
        };
        if let Some(error) = refusal {
            tx.commit()?;
            return Err(error);
        }
        let token = record
            .token_id
            .ok_or_else(|| anyhow::anyhow!("approved decision missing token"))?;
        let changed = tx.execute(
            "UPDATE approval SET status='consumed',consumed_at=?1,token_id=NULL
            WHERE id=?2 AND status='approved' AND token_id=?3",
            params![time, id, token],
        )?;
        ensure!(changed == 1, "approval already consumed");
        audit(&tx, &key, &id, "approval_consumed", &hash, None)?;
        tx.commit()?;
        Ok(ConsumedApproval {
            request_id: id,
            token_id: token,
            binding_hash: hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn binding() -> ApprovalBinding {
        ApprovalBinding {
            scope: Scope {
                tenant: "tenant".into(),
                user: "user".into(),
                channel: "channel".into(),
                session: "session".into(),
            },
            turn_id: "turn".into(),
            call_id: "call".into(),
            tool_name: "shell".into(),
            arguments_hash: fingerprint(&serde_json::json!({"command":"secret command"})).unwrap(),
            context_fingerprint: fingerprint(&"policy and persona v1").unwrap(),
        }
    }
    #[test]
    fn durable_atomic_single_use_across_instances_and_redacted_audit() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("db");
        let a = ApprovalRegistry::open(&path)?;
        let b = ApprovalRegistry::open(&path)?;
        let binding = binding();
        let pending = a.request(&binding, 60)?;
        assert!(
            a.consume(&binding)
                .unwrap_err()
                .is::<ApprovalRequiredError>()
        );
        b.decide(
            &binding.scope,
            &pending.id,
            &pending.binding_hash,
            Decision::Approve,
            "operator-secret-name",
        )?;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let mut handles = vec![];
        for registry in [a.clone(), b.clone()] {
            let barrier = barrier.clone();
            let binding = binding.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                registry.consume(&binding).is_ok()
            }));
        }
        barrier.wait();
        let wins = handles
            .into_iter()
            .map(|h| h.join().unwrap() as usize)
            .sum::<usize>();
        assert_eq!(wins, 1);
        assert_eq!(
            a.get(&binding.scope, &pending.id)?.unwrap().status,
            ApprovalStatus::Consumed
        );
        assert_eq!(a.request(&binding, 60)?.status, ApprovalStatus::Consumed);
        let audit = serde_json::to_string(&a.audit(&binding.scope, 0, 100)?)?;
        assert!(!audit.contains("operator-secret-name") && !audit.contains("secret command"));
        assert_eq!(a.audit(&binding.scope, 0, 100)?.len(), 3);
        Ok(())
    }
    #[test]
    fn mismatch_expiry_denial_and_revocation_fail_closed() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let a = ApprovalRegistry::open(dir.path().join("db"))?;
        let binding = binding();
        let pending = a.request(&binding, 60)?;
        let mut other = binding.clone();
        other.scope.session = "elsewhere".into();
        assert!(a.get(&other.scope, &pending.id)?.is_none());
        assert!(
            a.decide(
                &other.scope,
                &pending.id,
                &pending.binding_hash,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        assert!(
            a.decide(
                &binding.scope,
                &pending.id,
                &"f".repeat(64),
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        a.decide(
            &binding.scope,
            &pending.id,
            &pending.binding_hash,
            Decision::Approve,
            "op",
        )?;
        for mutation in [0, 1, 2, 3, 4, 5] {
            let mut other = binding.clone();
            match mutation {
                0 => other.arguments_hash = fingerprint(&"changed args")?,
                1 => other.context_fingerprint = fingerprint(&"changed policy")?,
                2 => other.turn_id = "other".into(),
                3 => other.tool_name = "other_tool".into(),
                4 => other.scope.user = "other".into(),
                _ => other.call_id = "other".into(),
            }
            assert!(a.consume(&other).is_err());
        }
        a.revoke(&binding.scope, &pending.id, &pending.binding_hash, "op")?;
        assert!(
            a.consume(&binding)
                .unwrap_err()
                .is::<ApprovalRefusedError>()
        );
        let mut expired = binding.clone();
        expired.call_id = "expired".into();
        let p = a.request_at(&expired, 1, now()? - 2)?;
        assert!(
            a.decide(
                &expired.scope,
                &p.id,
                &p.binding_hash,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        assert_eq!(
            a.get(&expired.scope, &p.id)?.unwrap().status,
            ApprovalStatus::Expired
        );
        let mut exp_approved = binding.clone();
        exp_approved.call_id = "expired-after-approval".into();
        let p = a.request(&exp_approved, 1)?;
        a.decide(
            &exp_approved.scope,
            &p.id,
            &p.binding_hash,
            Decision::Approve,
            "op",
        )?;
        assert!(a.consume_at(&exp_approved, p.expires_at).is_err());
        let mut denied = binding.clone();
        denied.call_id = "denied".into();
        let p = a.request(&denied, 60)?;
        a.decide(&denied.scope, &p.id, &p.binding_hash, Decision::Deny, "op")?;
        assert!(a.consume(&denied).is_err());
        assert!(
            a.decide(
                &denied.scope,
                &p.id,
                &p.binding_hash,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn operator_decision_race_has_exactly_one_winner() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("db");
        let a = ApprovalRegistry::open(&path)?;
        let binding = binding();
        let record = a.request(&binding, 60)?;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let mut handles = vec![];
        for decision in [Decision::Approve, Decision::Deny] {
            let registry = ApprovalRegistry::open(&path)?;
            let barrier = barrier.clone();
            let scope = binding.scope.clone();
            let record = record.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                registry
                    .decide(&scope, &record.id, &record.binding_hash, decision, "op")
                    .is_ok()
            }));
        }
        barrier.wait();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| h.join().unwrap() as usize)
                .sum::<usize>(),
            1
        );
        if a.get(&binding.scope, &record.id)?.unwrap().status == ApprovalStatus::Approved {
            a.revoke(&binding.scope, &record.id, &record.binding_hash, "op")?;
        }
        assert!(a.consume(&binding).is_err());
        assert!(
            a.decide(
                &binding.scope,
                &record.id,
                &record.binding_hash,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn revoke_versus_consume_is_serialized_and_durable() -> Result<()> {
        for _ in 0..8 {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("db");
            let a = ApprovalRegistry::open(&path)?;
            let b = ApprovalRegistry::open(&path)?;
            let binding = binding();
            let record = a.request(&binding, 60)?;
            a.decide(
                &binding.scope,
                &record.id,
                &record.binding_hash,
                Decision::Approve,
                "op",
            )?;
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
            let worker_barrier = barrier.clone();
            let other_binding = binding.clone();
            let other_record = record.clone();
            let revoke = std::thread::spawn(move || {
                worker_barrier.wait();
                b.revoke(
                    &other_binding.scope,
                    &other_record.id,
                    &other_record.binding_hash,
                    "op",
                )
                .is_ok()
            });
            barrier.wait();
            let consumed = a.consume(&binding).is_ok();
            let revoked = revoke.join().unwrap();
            assert_ne!(consumed, revoked);
            let reopened = ApprovalRegistry::open(&path)?;
            assert_eq!(
                reopened.get(&binding.scope, &record.id)?.unwrap().status,
                if consumed {
                    ApprovalStatus::Consumed
                } else {
                    ApprovalStatus::Revoked
                }
            );
            assert!(reopened.consume(&binding).is_err());
        }
        Ok(())
    }
    #[test]
    fn audit_failure_rolls_back_token_consumption() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let a = ApprovalRegistry::open(dir.path().join("db"))?;
        let binding = binding();
        let record = a.request(&binding, 60)?;
        a.decide(
            &binding.scope,
            &record.id,
            &record.binding_hash,
            Decision::Approve,
            "op",
        )?;
        a.connect()?.execute_batch("CREATE TRIGGER fail_audit BEFORE INSERT ON audit
            WHEN NEW.event='approval_consumed' BEGIN SELECT RAISE(ABORT,'injected audit failure'); END;")?;
        assert!(a.consume(&binding).is_err());
        let still_approved = a.get(&binding.scope, &record.id)?.unwrap();
        assert_eq!(still_approved.status, ApprovalStatus::Approved);
        assert!(still_approved.token_id.is_some());
        a.connect()?.execute_batch("DROP TRIGGER fail_audit;")?;
        a.consume(&binding)?;
        assert!(a.consume(&binding).is_err());
        Ok(())
    }
}
