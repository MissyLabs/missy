//! Operator-only code evolution and inert persona experiments.
//! No ToolExecutor implementation, automatic source changes, restart or deployment.
#![cfg(target_os = "linux")]
mod patch;
mod persona;
mod repository;
use anyhow::{Context, Result, ensure};
use missy_core::Scope;
pub use persona::*;
pub use repository::EvolutionPolicy;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    time::Duration,
};

pub(crate) fn hash(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}
pub(crate) fn binding<T: Serialize + ?Sized>(v: &T) -> Result<String> {
    Ok(hash(serde_json::to_vec(v)?))
}
pub(crate) fn scope_key(scope: &Scope) -> Result<String> {
    scope.validate()?;
    binding(scope)
}
pub(crate) fn operator(scope: &Scope, id: &str, confirm: bool) -> Result<()> {
    scope.validate()?;
    ensure!(
        confirm && id == scope.user && !id.is_empty(),
        "explicit authenticated scope operator and confirmation required"
    );
    Ok(())
}

#[derive(Clone)]
pub struct EvolutionRegistry {
    path: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileChange {
    pub path: String,
    pub before: String,
    pub after: String,
    pub mode: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvolutionPlan {
    pub id: String,
    pub scope_hash: String,
    pub repo: PathBuf,
    pub base_head: String,
    pub branch: String,
    pub title: String,
    pub diff: String,
    pub diff_hash: String,
    pub policy_hash: String,
    pub files: Vec<FileChange>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvolutionProposal {
    pub plan: EvolutionPlan,
    pub plan_hash: String,
    /// proposed, approved, denied, applying, applied, rolling_back,
    /// rolled_back or recovery_required. Unknown effects are never replayed.
    pub status: String,
    pub commit: Option<String>,
    pub rollback_commit: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvolutionAudit {
    pub sequence: i64,
    pub proposal_id: String,
    pub event: String,
    pub operator: Option<String>,
    pub detail: String,
}

impl EvolutionRegistry {
    /// Parent must already exist, be operator-owned and mode 0700. No live defaults.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(
            path.is_absolute() && path.file_name().is_some(),
            "absolute durable database path required"
        );
        let parent = path.parent().context("database parent")?;
        repository::safe_directory(parent)?;
        let m = fs::metadata(parent)?;
        ensure!(
            m.mode() & 0o077 == 0,
            "database parent must be private (0700)"
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)?;
        let m = file.metadata()?;
        ensure!(
            m.is_file()
                && m.nlink() == 1
                && m.mode() & 0o077 == 0
                && m.uid() == unsafe { libc::geteuid() },
            "private regular operator-owned database required"
        );
        let this = Self {
            path: path.to_owned(),
        };
        let c = this.connect()?;
        c.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS evolution(id TEXT PRIMARY KEY, scope TEXT NOT NULL, plan TEXT NOT NULL, hash TEXT NOT NULL, status TEXT NOT NULL, reviewed_by TEXT, commit_id TEXT, rollback_commit TEXT);
            CREATE TABLE IF NOT EXISTS evolution_journal(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evolution_audit(sequence INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, id TEXT NOT NULL, event TEXT NOT NULL, operator TEXT, detail TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS immutable_evolution BEFORE UPDATE OF plan,hash,scope ON evolution BEGIN SELECT RAISE(ABORT,'immutable evolution plan'); END;
            CREATE TABLE IF NOT EXISTS experiment(scope TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY(scope,id));
            CREATE TABLE IF NOT EXISTS experiment_assignment(scope TEXT NOT NULL, experiment TEXT NOT NULL, principal TEXT NOT NULL, variant INTEGER NOT NULL, PRIMARY KEY(scope,experiment,principal));
            CREATE TABLE IF NOT EXISTS experiment_outcome(scope TEXT NOT NULL, experiment TEXT NOT NULL, event TEXT NOT NULL, variant INTEGER NOT NULL, success INTEGER NOT NULL, refused INTEGER NOT NULL, PRIMARY KEY(scope,experiment,event));")?;
        Ok(this)
    }
    pub(crate) fn connect(&self) -> Result<Connection> {
        let m = fs::symlink_metadata(&self.path)?;
        ensure!(
            m.is_file()
                && m.nlink() == 1
                && m.mode() & 0o077 == 0
                && m.uid() == unsafe { libc::geteuid() },
            "unsafe database"
        );
        let c = Connection::open_with_flags(
            &self.path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        c.busy_timeout(Duration::from_secs(10))?;
        c.execute_batch("PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;")?;
        Ok(c)
    }
    pub fn propose(
        &self,
        scope: &Scope,
        policy: &EvolutionPolicy,
        repo: impl AsRef<Path>,
        diff: &str,
        title: &str,
    ) -> Result<EvolutionProposal> {
        ensure!(
            !title.trim().is_empty() && title.len() <= 512 && !title.contains(['\n', '\r', '\0']),
            "invalid title"
        );
        let key = scope_key(scope)?;
        let repo = repository::Repository::open(policy, repo.as_ref(), false)?;
        repo.clean()?;
        let files = patch::prepare(diff, &repo, policy)?;
        let plan = EvolutionPlan {
            id: uuid::Uuid::new_v4().to_string(),
            scope_hash: key.clone(),
            repo: repo.path.clone(),
            base_head: repo.head()?,
            branch: repo.branch()?,
            title: title.to_owned(),
            diff: diff.to_owned(),
            diff_hash: hash(diff),
            policy_hash: binding(policy)?,
            files,
        };
        let plan_hash = binding(&plan)?;
        let mut c = self.connect()?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        tx.execute(
            "INSERT INTO evolution(id,scope,plan,hash,status) VALUES(?1,?2,?3,?4,'proposed')",
            params![plan.id, key, serde_json::to_string(&plan)?, plan_hash],
        )?;
        Self::audit_row(&tx, &key, &plan.id, "proposed", None, &plan_hash)?;
        tx.commit()?;
        Ok(EvolutionProposal {
            plan,
            plan_hash,
            status: "proposed".into(),
            commit: None,
            rollback_commit: None,
        })
    }
    pub fn proposal(&self, scope: &Scope, id: &str) -> Result<EvolutionProposal> {
        Self::load(&self.connect()?, &scope_key(scope)?, id)
    }
    fn load(c: &Connection, key: &str, id: &str) -> Result<EvolutionProposal> {
        let (p,h,s,commit,rollback_commit): (String,String,String,Option<String>,Option<String>) = c.query_row("SELECT plan,hash,status,commit_id,rollback_commit FROM evolution WHERE scope=?1 AND id=?2",params![key,id],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))).optional()?.context("proposal not found in scope")?;
        let plan: EvolutionPlan = serde_json::from_str(&p)?;
        ensure!(
            plan.scope_hash == key
                && plan.id == id
                && binding(&plan)? == h
                && hash(&plan.diff) == plan.diff_hash,
            "plan integrity mismatch"
        );
        Ok(EvolutionProposal {
            plan,
            plan_hash: h,
            status: s,
            commit,
            rollback_commit,
        })
    }
    pub fn review(
        &self,
        scope: &Scope,
        id: &str,
        exact_plan_hash: &str,
        operator_id: &str,
        approve: bool,
    ) -> Result<EvolutionProposal> {
        operator(scope, operator_id, true)?;
        let key = scope_key(scope)?;
        let mut c = self.connect()?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let p = Self::load(&tx, &key, id)?;
        ensure!(
            p.status == "proposed" && p.plan_hash == exact_plan_hash,
            "pending exact-hash review required"
        );
        let status = if approve { "approved" } else { "denied" };
        tx.execute(
            "UPDATE evolution SET status=?1,reviewed_by=?2 WHERE id=?3",
            params![status, operator_id, id],
        )?;
        Self::audit_row(&tx, &key, id, status, Some(operator_id), exact_plan_hash)?;
        tx.commit()?;
        self.proposal(scope, id)
    }
    pub fn audit(&self, scope: &Scope) -> Result<Vec<EvolutionAudit>> {
        let c = self.connect()?;
        let key = scope_key(scope)?;
        let mut s=c.prepare("SELECT sequence,id,event,operator,detail FROM evolution_audit WHERE scope=?1 ORDER BY sequence")?;
        Ok(s.query_map([key], |r| {
            Ok(EvolutionAudit {
                sequence: r.get(0)?,
                proposal_id: r.get(1)?,
                event: r.get(2)?,
                operator: r.get(3)?,
                detail: r.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?)
    }
    pub(crate) fn audit_row(
        c: &Connection,
        key: &str,
        id: &str,
        event: &str,
        operator: Option<&str>,
        detail: &str,
    ) -> Result<()> {
        c.execute(
            "INSERT INTO evolution_audit(scope,id,event,operator,detail) VALUES(?1,?2,?3,?4,?5)",
            params![key, id, event, operator, detail],
        )?;
        Ok(())
    }
    pub fn recovery_journal(&self, scope: &Scope, id: &str) -> Result<Option<serde_json::Value>> {
        self.proposal(scope, id)?;
        let c = self.connect()?;
        let raw: Option<String> = c
            .query_row(
                "SELECT payload FROM evolution_journal WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        raw.map(|s| Ok(serde_json::from_str(&s)?)).transpose()
    }
    /// Only invoke from an authenticated operator surface. Review never invokes this.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        scope: &Scope,
        id: &str,
        exact_plan_hash: &str,
        pinned_head: &str,
        policy: &EvolutionPolicy,
        operator_id: &str,
        confirm: bool,
    ) -> Result<EvolutionProposal> {
        self.execute(
            scope,
            id,
            exact_plan_hash,
            pinned_head,
            policy,
            operator_id,
            confirm,
            false,
        )
    }
    /// Creates a new inverse commit; never reset/stash/checkout unrelated work.
    #[allow(clippy::too_many_arguments)]
    pub fn rollback(
        &self,
        scope: &Scope,
        id: &str,
        exact_plan_hash: &str,
        pinned_head: &str,
        policy: &EvolutionPolicy,
        operator_id: &str,
        confirm: bool,
    ) -> Result<EvolutionProposal> {
        self.execute(
            scope,
            id,
            exact_plan_hash,
            pinned_head,
            policy,
            operator_id,
            confirm,
            true,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        scope: &Scope,
        id: &str,
        exact_plan_hash: &str,
        pinned_head: &str,
        policy: &EvolutionPolicy,
        operator_id: &str,
        confirm: bool,
        rollback: bool,
    ) -> Result<EvolutionProposal> {
        operator(scope, operator_id, confirm)?;
        ensure!(
            policy.allow_git_commits && policy.operator_user == operator_id,
            "policy denies operator Git writes"
        );
        let key = scope_key(scope)?;
        let mut c = self.connect()?;
        let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let p = Self::load(&tx, &key, id)?;
        ensure!(
            p.plan_hash == exact_plan_hash && p.plan.policy_hash == binding(policy)?,
            "exact plan/policy hash required"
        );
        let expected = if rollback {
            p.commit.as_deref().context("no applied commit")?
        } else {
            &p.plan.base_head
        };
        ensure!(
            p.status == if rollback { "applied" } else { "approved" },
            "invalid state; unknown effects require manual recovery"
        );
        ensure!(pinned_head == expected, "pinned HEAD mismatch");
        let reviewer: String =
            tx.query_row("SELECT reviewed_by FROM evolution WHERE id=?1", [id], |r| {
                r.get(0)
            })?;
        ensure!(reviewer == operator_id, "reviewing operator required");
        let repo = repository::Repository::open(policy, &p.plan.repo, true)?;
        repo.clean()?;
        ensure!(
            repo.head()? == expected && repo.branch()? == p.plan.branch,
            "repository HEAD/branch changed"
        );
        let files: Vec<FileChange> = p
            .plan
            .files
            .iter()
            .map(|f| {
                if rollback {
                    FileChange {
                        path: f.path.clone(),
                        before: f.after.clone(),
                        after: f.before.clone(),
                        mode: f.mode,
                    }
                } else {
                    f.clone()
                }
            })
            .collect();
        repo.verify_files(&files, policy)?;
        let phase = if rollback { "rolling_back" } else { "applying" };
        let journal = serde_json::json!({"version":1,"phase":phase,"plan_hash":p.plan_hash,"repo":p.plan.repo,"old_head":expected,"branch":p.plan.branch,"files":files,"warning":"Manual recovery only after uncertain effect. Do not replay apply."});
        tx.execute(
            "INSERT OR REPLACE INTO evolution_journal(id,payload) VALUES(?1,?2)",
            params![id, journal.to_string()],
        )?;
        tx.execute(
            "UPDATE evolution SET status=?1 WHERE id=?2",
            params![phase, id],
        )?;
        Self::audit_row(&tx, &key, id, phase, Some(operator_id), exact_plan_hash)?;
        tx.commit()?;
        let result = repo.install(
            &files,
            expected,
            &p.plan.branch,
            &format!(
                "[missy-evolve] {}{}",
                if rollback { "rollback " } else { "" },
                p.plan.title
            ),
            |detail| {
                let c = self.connect()?;
                c.execute(
                    "UPDATE evolution_journal SET payload=?1 WHERE id=?2",
                    params![detail.to_string(), id],
                )?;
                Ok(())
            },
            journal,
        );
        match result {
            Ok(commit) => {
                let tx = c.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                if rollback {
                    tx.execute(
                        "UPDATE evolution SET status='rolled_back',rollback_commit=?1 WHERE id=?2",
                        params![commit, id],
                    )?;
                } else {
                    tx.execute(
                        "UPDATE evolution SET status='applied',commit_id=?1 WHERE id=?2",
                        params![commit, id],
                    )?;
                }
                Self::audit_row(
                    &tx,
                    &key,
                    id,
                    if rollback { "rolled_back" } else { "applied" },
                    Some(operator_id),
                    &commit,
                )?;
                tx.execute("DELETE FROM evolution_journal WHERE id=?1", [id])?;
                tx.commit()?;
                self.proposal(scope, id)
            }
            Err(e) => {
                c.execute(
                    "UPDATE evolution SET status='recovery_required' WHERE id=?1",
                    [id],
                )?;
                Self::audit_row(
                    &c,
                    &key,
                    id,
                    "recovery_required",
                    Some(operator_id),
                    &e.to_string(),
                )?;
                Err(e.context("effect may be partial; recovery journal retained, manual operator recovery required"))
            }
        }
    }
}
