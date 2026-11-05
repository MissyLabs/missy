//! Persona and patch proposals are inert UTF-8 documents. SQLite is the atomic
//! authoritative revision store, avoiding unsafe partial DB-plus-file updates.
//! Approving a patch records review, but does not apply or execute it. An explicit
//! policy-aware patch executor must separately stage/validate/apply any code.
use crate::{
    ApprovalRegistry, Decision, digest, fingerprint, identifier, now,
    registry::{audit, scope_key},
};
use anyhow::{Result, ensure};
use missy_core::Scope;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DocumentKind {
    Persona,
    SystemPrompt,
    Patch,
}
impl DocumentKind {
    fn key(self) -> &'static str {
        match self {
            Self::Persona => "persona",
            Self::SystemPrompt => "system_prompt",
            Self::Patch => "patch",
        }
    }
    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "persona" => Self::Persona,
            "system_prompt" => Self::SystemPrompt,
            "patch" => Self::Patch,
            _ => anyhow::bail!("invalid document kind"),
        })
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentRevision {
    pub kind: DocumentKind,
    pub version: i64,
    pub content: String,
    pub content_hash: String,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Pending,
    Approved,
    Denied,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentProposal {
    pub id: String,
    pub kind: DocumentKind,
    pub base_version: i64,
    pub content: String,
    pub content_hash: String,
    pub context_fingerprint: String,
    pub proposal_hash: String,
    pub status: ProposalStatus,
    pub created_at: i64,
}
/// An immutable per-turn snapshot. Parent must bind its hash into runtime context
/// fingerprint; newly approved revisions are read only at a new turn boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PromptSnapshot {
    pub persona: Option<DocumentRevision>,
    pub system_prompt: Option<DocumentRevision>,
    pub prompt_hash: String,
}

fn current(
    conn: &rusqlite::Connection,
    key: &str,
    kind: DocumentKind,
) -> Result<Option<DocumentRevision>> {
    Ok(conn
        .query_row(
            "SELECT version,content,content_hash FROM document_revision
        WHERE scope_key=?1 AND kind=?2 ORDER BY version DESC LIMIT 1",
            params![key, kind.key()],
            |r| {
                Ok(DocumentRevision {
                    kind,
                    version: r.get(0)?,
                    content: r.get(1)?,
                    content_hash: r.get(2)?,
                })
            },
        )
        .optional()?)
}
fn proposal(conn: &rusqlite::Connection, key: &str, id: &str) -> Result<Option<DocumentProposal>> {
    let raw=conn.query_row("SELECT id,kind,base_version,content,content_hash,context_fingerprint,proposal_hash,status,created_at
        FROM document_proposal WHERE scope_key=?1 AND id=?2",params![key,id],|r|
        Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,i64>(2)?,r.get::<_,String>(3)?,r.get::<_,String>(4)?,
            r.get::<_,String>(5)?,r.get::<_,String>(6)?,r.get::<_,String>(7)?,r.get::<_,i64>(8)?))).optional()?;
    raw.map(
        |(
            id,
            kind,
            base_version,
            content,
            content_hash,
            context_fingerprint,
            proposal_hash,
            status,
            created_at,
        )| {
            Ok(DocumentProposal {
                id,
                kind: DocumentKind::parse(&kind)?,
                base_version,
                content,
                content_hash,
                context_fingerprint,
                proposal_hash,
                status: match status.as_str() {
                    "pending" => ProposalStatus::Pending,
                    "approved" => ProposalStatus::Approved,
                    "denied" => ProposalStatus::Denied,
                    _ => anyhow::bail!("invalid proposal status"),
                },
                created_at,
            })
        },
    )
    .transpose()
}
impl ApprovalRegistry {
    pub fn document(&self, scope: &Scope, kind: DocumentKind) -> Result<Option<DocumentRevision>> {
        current(&self.connect()?, &scope_key(scope)?, kind)
    }
    pub fn document_proposal(&self, scope: &Scope, id: &str) -> Result<Option<DocumentProposal>> {
        proposal(&self.connect()?, &scope_key(scope)?, id)
    }
    pub fn document_proposals(&self, scope: &Scope, limit: usize) -> Result<Vec<DocumentProposal>> {
        let key = scope_key(scope)?;
        let conn = self.connect()?;
        let ids=conn.prepare("SELECT id FROM document_proposal WHERE scope_key=?1 ORDER BY created_at DESC,id LIMIT ?2")?
            .query_map(params![key,limit.min(1000) as i64],|r|r.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        ids.iter()
            .map(|id| {
                proposal(&conn, &key, id)?.ok_or_else(|| anyhow::anyhow!("proposal disappeared"))
            })
            .collect()
    }
    /// Staging has no active effect, even for persona or prompt edits.
    pub fn propose_document(
        &self,
        scope: &Scope,
        kind: DocumentKind,
        base_version: i64,
        content: &str,
        context_fingerprint: &str,
    ) -> Result<DocumentProposal> {
        ensure!(
            digest(context_fingerprint),
            "context must be lowercase SHA-256 fingerprint"
        );
        ensure!(
            !content.trim().is_empty() && content.len() <= 1024 * 1024 && !content.contains('\0'),
            "document must be nonempty UTF-8 text, at most 1MiB, without NUL"
        );
        ensure!(base_version >= 0, "invalid base version");
        let key = scope_key(scope)?;
        let content_hash = fingerprint(content)?;
        let proposal_hash = fingerprint(&(
            "missy-document-v1",
            &key,
            kind,
            base_version,
            &content_hash,
            context_fingerprint,
        ))?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        ensure!(
            current(&tx, &key, kind)?.map_or(0, |r| r.version) == base_version,
            "document base version conflict"
        );
        let id = uuid::Uuid::new_v4().to_string();
        tx.execute("INSERT INTO document_proposal(id,scope_key,kind,base_version,content,content_hash,
            context_fingerprint,proposal_hash,status,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,'pending',?9)",
            params![id,key,kind.key(),base_version,content,content_hash,context_fingerprint,proposal_hash,now()?])?;
        audit(&tx, &key, &id, "document_proposed", &proposal_hash, None)?;
        let proposal = proposal(&tx, &key, &id)?.expect("inserted proposal");
        tx.commit()?;
        Ok(proposal)
    }
    /// Authenticated explicit operator approval and revision write are atomic.
    /// A revision conflict or changed policy/config context rejects approval.
    /// Return value is a stored revision, NOT an instruction to mutate an active
    /// turn, live source file, executing process, or existing runtime policy.
    pub fn decide_document(
        &self,
        scope: &Scope,
        id: &str,
        expected_proposal_hash: &str,
        current_context_fingerprint: &str,
        decision: Decision,
        operator_id: &str,
    ) -> Result<DocumentProposal> {
        identifier(operator_id, "operator ID")?;
        let key = scope_key(scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let pending = proposal(&tx, &key, id)?
            .ok_or_else(|| anyhow::anyhow!("proposal not found in scope"))?;
        ensure!(
            pending.status == ProposalStatus::Pending,
            "proposal already decided"
        );
        ensure!(
            pending.proposal_hash == expected_proposal_hash,
            "proposal hash mismatch"
        );
        ensure!(
            pending.context_fingerprint == current_context_fingerprint,
            "proposal context fingerprint changed"
        );
        ensure!(
            pending.content_hash == fingerprint(&pending.content)?,
            "proposal content hash mismatch"
        );
        let (status, event) = match decision {
            Decision::Approve => {
                ensure!(
                    current(&tx, &key, pending.kind)?.map_or(0, |r| r.version)
                        == pending.base_version,
                    "document base version conflict"
                );
                let version = pending
                    .base_version
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("document version overflow"))?;
                tx.execute("INSERT INTO document_revision(scope_key,kind,version,content,content_hash) VALUES(?1,?2,?3,?4,?5)",
                    params![key,pending.kind.key(),version,pending.content,pending.content_hash])?;
                ("approved", "document_approved")
            }
            Decision::Deny => ("denied", "document_denied"),
        };
        tx.execute("UPDATE document_proposal SET status=?1 WHERE scope_key=?2 AND id=?3 AND status='pending'",params![status,key,id])?;
        audit(
            &tx,
            &key,
            id,
            event,
            &pending.proposal_hash,
            Some(operator_id),
        )?;
        let result = proposal(&tx, &key, id)?.expect("existing proposal");
        tx.commit()?;
        Ok(result)
    }
    pub fn prompt_snapshot(&self, scope: &Scope) -> Result<PromptSnapshot> {
        let key = scope_key(scope)?;
        let mut conn = self.connect()?;
        let tx = conn.transaction()?;
        let persona = current(&tx, &key, DocumentKind::Persona)?;
        let system_prompt = current(&tx, &key, DocumentKind::SystemPrompt)?;
        let prompt_hash = fingerprint(&("missy-prompt-v1", &key, &persona, &system_prompt))?;
        tx.commit()?;
        Ok(PromptSnapshot {
            persona,
            system_prompt,
            prompt_hash,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staged_scoped_versioned_atomic_and_never_mutates_snapshot() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("db");
        let a = ApprovalRegistry::open(&path)?;
        let scope = Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        };
        let before = a.prompt_snapshot(&scope)?;
        let context = fingerprint(&"policy config")?;
        let p = a.propose_document(
            &scope,
            DocumentKind::Persona,
            0,
            "operator reviewed persona",
            &context,
        )?;
        assert_eq!(a.prompt_snapshot(&scope)?, before);
        let mut other = scope.clone();
        other.user = "other".into();
        assert!(a.document_proposal(&other, &p.id)?.is_none());
        assert!(
            a.decide_document(
                &other,
                &p.id,
                &p.proposal_hash,
                &context,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        assert!(
            a.decide_document(
                &scope,
                &p.id,
                &p.proposal_hash,
                &fingerprint(&"changed policy")?,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        let stale = a.propose_document(
            &scope,
            DocumentKind::Persona,
            0,
            "conflicting proposal",
            &context,
        )?;
        a.decide_document(
            &scope,
            &p.id,
            &p.proposal_hash,
            &context,
            Decision::Approve,
            "op",
        )?;
        assert!(
            a.decide_document(
                &scope,
                &stale.id,
                &stale.proposal_hash,
                &context,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        assert!(
            a.decide_document(
                &scope,
                &p.id,
                &p.proposal_hash,
                &context,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        assert_eq!(before.persona, None);
        let reopened = ApprovalRegistry::open(path)?;
        let after = reopened.prompt_snapshot(&scope)?;
        assert_ne!(after.prompt_hash, before.prompt_hash);
        assert_eq!(after.persona.unwrap().version, 1);
        let patch = a.propose_document(
            &scope,
            DocumentKind::Patch,
            0,
            "inert patch content",
            &context,
        )?;
        a.decide_document(
            &scope,
            &patch.id,
            &patch.proposal_hash,
            &context,
            Decision::Approve,
            "op",
        )?;
        assert_eq!(
            a.prompt_snapshot(&scope)?.prompt_hash,
            reopened.prompt_snapshot(&scope)?.prompt_hash
        );
        assert_eq!(
            a.document(&scope, DocumentKind::Patch)?.unwrap().content,
            "inert patch content"
        );
        let audit = serde_json::to_string(&a.audit(&scope, 0, 100)?)?;
        assert!(
            !audit.contains("operator reviewed persona") && !audit.contains("inert patch content")
        );
        Ok(())
    }
    #[test]
    fn audit_failure_rolls_back_revision_and_operator_decision() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let a = ApprovalRegistry::open(dir.path().join("db"))?;
        let scope = Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        };
        let context = fingerprint(&"config")?;
        let pending = a.propose_document(
            &scope,
            DocumentKind::SystemPrompt,
            0,
            "reviewed system prompt",
            &context,
        )?;
        a.connect()?.execute_batch("CREATE TRIGGER fail_document_audit BEFORE INSERT ON audit
            WHEN NEW.event='document_approved' BEGIN SELECT RAISE(ABORT,'injected audit failure'); END;")?;
        assert!(
            a.decide_document(
                &scope,
                &pending.id,
                &pending.proposal_hash,
                &context,
                Decision::Approve,
                "op"
            )
            .is_err()
        );
        assert!(a.document(&scope, DocumentKind::SystemPrompt)?.is_none());
        assert_eq!(
            a.document_proposal(&scope, &pending.id)?.unwrap().status,
            ProposalStatus::Pending
        );
        a.connect()?
            .execute_batch("DROP TRIGGER fail_document_audit;")?;
        a.decide_document(
            &scope,
            &pending.id,
            &pending.proposal_hash,
            &context,
            Decision::Deny,
            "op",
        )?;
        assert!(a.document(&scope, DocumentKind::SystemPrompt)?.is_none());
        Ok(())
    }
}
