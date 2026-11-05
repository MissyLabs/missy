use missy_core::Scope;
use missy_evolution::*;
use missy_governance::{ApprovalRegistry, Decision, DocumentKind};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::PathBuf,
    process::Command,
};
use tempfile::TempDir;

const DIFF: &str =
    "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-old\n+new\n";
fn scope() -> Scope {
    Scope {
        tenant: "test".into(),
        user: "operator".into(),
        channel: "fixture".into(),
        session: "one".into(),
    }
}
fn git(repo: &std::path::Path, args: &[&str]) -> String {
    let o = Command::new("/usr/bin/git")
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", "/nonexistent")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_NAME", "test")
        .env("GIT_AUTHOR_EMAIL", "test@localhost")
        .env("GIT_COMMITTER_NAME", "test")
        .env("GIT_COMMITTER_EMAIL", "test@localhost")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgSign=false",
        ])
        .args(args)
        .current_dir(repo)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout).unwrap().trim().into()
}
struct Fixture {
    _temp: TempDir,
    repo: PathBuf,
    db: PathBuf,
    store: EvolutionRegistry,
    policy: EvolutionPolicy,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let repo = temp.path().join("repo");
        fs::create_dir(&repo).unwrap();
        git(&repo, &["init", "-q", "-b", "fixture"]);
        fs::write(repo.join("a.txt"), "old\n").unwrap();
        fs::write(repo.join("b.txt"), "untouched\n").unwrap();
        git(&repo, &["add", "--", "a.txt", "b.txt"]);
        git(&repo, &["commit", "-q", "-m", "base"]);
        let db = temp.path().join("evolution.db");
        let store = EvolutionRegistry::open(&db).unwrap();
        let policy = EvolutionPolicy {
            read_roots: vec![repo.clone()],
            write_roots: vec![repo.clone()],
            operator_user: "operator".into(),
            allow_git_commits: true,
        };
        Self {
            _temp: temp,
            repo,
            db,
            store,
            policy,
        }
    }
    fn propose(&self) -> EvolutionProposal {
        self.store
            .propose(&scope(), &self.policy, &self.repo, DIFF, "synthetic edit")
            .unwrap()
    }
    fn approved(&self) -> EvolutionProposal {
        let p = self.propose();
        self.store
            .review(&scope(), &p.plan.id, &p.plan_hash, "operator", true)
            .unwrap()
    }
}
#[test]
fn persistent_scoped_hash_review_apply_rollback_exact() {
    let f = Fixture::new();
    let p = f.propose();
    assert_eq!(p.plan.diff_hash.len(), 64);
    assert_eq!(fs::read_to_string(f.repo.join("a.txt")).unwrap(), "old\n");
    let other = Scope {
        user: "other".into(),
        ..scope()
    };
    assert!(f.store.proposal(&other, &p.plan.id).is_err());
    assert!(
        f.store
            .review(&scope(), &p.plan.id, "wrong", "operator", true)
            .is_err()
    );
    assert!(
        f.store
            .review(&scope(), &p.plan.id, &p.plan_hash, "model", true)
            .is_err()
    );
    let reopened = EvolutionRegistry::open(&f.db).unwrap();
    reopened
        .review(&scope(), &p.plan.id, &p.plan_hash, "operator", true)
        .unwrap();
    assert!(
        reopened
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                false
            )
            .is_err()
    );
    assert!(
        reopened
            .apply(
                &scope(),
                &p.plan.id,
                "wrong",
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    let a = reopened
        .apply(
            &scope(),
            &p.plan.id,
            &p.plan_hash,
            &p.plan.base_head,
            &f.policy,
            "operator",
            true,
        )
        .unwrap();
    assert_eq!(a.status, "applied");
    assert_eq!(fs::read_to_string(f.repo.join("a.txt")).unwrap(), "new\n");
    assert!(git(&f.repo, &["status", "--porcelain"]).is_empty());
    assert!(
        reopened
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    assert!(
        reopened
            .rollback(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    let r = reopened
        .rollback(
            &scope(),
            &p.plan.id,
            &p.plan_hash,
            a.commit.as_deref().unwrap(),
            &f.policy,
            "operator",
            true,
        )
        .unwrap();
    assert_eq!(r.status, "rolled_back");
    assert_eq!(fs::read_to_string(f.repo.join("a.txt")).unwrap(), "old\n");
    assert_eq!(
        fs::read_to_string(f.repo.join("b.txt")).unwrap(),
        "untouched\n"
    );
    assert_ne!(
        r.rollback_commit.as_deref(),
        Some(p.plan.base_head.as_str())
    );
    assert!(git(&f.repo, &["status", "--porcelain"]).is_empty());
    assert!(
        reopened
            .recovery_journal(&scope(), &p.plan.id)
            .unwrap()
            .is_none()
    );
    assert_eq!(reopened.audit(&scope()).unwrap().len(), 6);
}
#[test]
fn repository_changes_are_never_stashed_or_overwritten() {
    let f = Fixture::new();
    let p = f.approved();
    fs::write(f.repo.join("b.txt"), "unrelated work\n").unwrap();
    assert!(
        f.store
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    assert_eq!(
        fs::read_to_string(f.repo.join("b.txt")).unwrap(),
        "unrelated work\n"
    );
    fs::write(f.repo.join("b.txt"), "untouched\n").unwrap();
    git(
        &f.repo,
        &["commit", "--allow-empty", "-q", "-m", "new head"],
    );
    assert!(
        f.store
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    assert_eq!(
        f.store.proposal(&scope(), &p.plan.id).unwrap().status,
        "approved"
    );
}
#[test]
fn rollback_refuses_unrelated_commit_or_dirty_file() {
    let f = Fixture::new();
    let p = f.approved();
    let a = f
        .store
        .apply(
            &scope(),
            &p.plan.id,
            &p.plan_hash,
            &p.plan.base_head,
            &f.policy,
            "operator",
            true,
        )
        .unwrap();
    fs::write(f.repo.join("b.txt"), "do not erase\n").unwrap();
    assert!(
        f.store
            .rollback(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                a.commit.as_deref().unwrap(),
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    assert_eq!(
        fs::read_to_string(f.repo.join("b.txt")).unwrap(),
        "do not erase\n"
    );
    git(&f.repo, &["add", "b.txt"]);
    git(&f.repo, &["commit", "-q", "-m", "unrelated"]);
    assert!(
        f.store
            .rollback(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                a.commit.as_deref().unwrap(),
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
}
#[test]
fn malicious_paths_links_and_grammar_fail_closed() {
    let f = Fixture::new();
    for diff in [
        DIFF.replace("a.txt", "../a.txt"),
        DIFF.replace("a.txt", ".git/config"),
        DIFF.replace("@@ -1 +1 @@", "@@ -1,9 +1 @@"),
        DIFF.replace("-old", "-different"),
        DIFF.replace("-old", "évil"),
        DIFF.replace("--- a/a.txt", "new file mode 100644\n--- /dev/null"),
    ] {
        assert!(
            f.store
                .propose(&scope(), &f.policy, &f.repo, &diff, "bad")
                .is_err()
        );
    }
    fs::remove_file(f.repo.join("a.txt")).unwrap();
    symlink("b.txt", f.repo.join("a.txt")).unwrap();
    assert!(
        f.store
            .propose(&scope(), &f.policy, &f.repo, DIFF, "symlink")
            .is_err()
    );
    fs::remove_file(f.repo.join("a.txt")).unwrap();
    fs::hard_link(f.repo.join("b.txt"), f.repo.join("a.txt")).unwrap();
    assert!(
        f.store
            .propose(&scope(), &f.policy, &f.repo, DIFF, "hardlink")
            .is_err()
    );
}
#[test]
fn multi_file_apply_and_rollback() {
    let f = Fixture::new();
    let diff = format!(
        "{DIFF}diff --git a/b.txt b/b.txt\n--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-untouched\n+changed\n"
    );
    let p = f
        .store
        .propose(&scope(), &f.policy, &f.repo, &diff, "two files")
        .unwrap();
    f.store
        .review(&scope(), &p.plan.id, &p.plan_hash, "operator", true)
        .unwrap();
    let a = f
        .store
        .apply(
            &scope(),
            &p.plan.id,
            &p.plan_hash,
            &p.plan.base_head,
            &f.policy,
            "operator",
            true,
        )
        .unwrap();
    assert_eq!(
        fs::read_to_string(f.repo.join("b.txt")).unwrap(),
        "changed\n"
    );
    f.store
        .rollback(
            &scope(),
            &p.plan.id,
            &p.plan_hash,
            a.commit.as_deref().unwrap(),
            &f.policy,
            "operator",
            true,
        )
        .unwrap();
    assert_eq!(fs::read_to_string(f.repo.join("a.txt")).unwrap(), "old\n");
    assert_eq!(
        fs::read_to_string(f.repo.join("b.txt")).unwrap(),
        "untouched\n"
    );
}
#[test]
fn private_store_and_policy_changes_reject() {
    let f = Fixture::new();
    let p = f.approved();
    let mut altered = f.policy.clone();
    altered.allow_git_commits = false;
    assert!(
        f.store
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &altered,
                "operator",
                true
            )
            .is_err()
    );
    let linked = f._temp.path().join("linked.db");
    symlink(&f.db, &linked).unwrap();
    assert!(EvolutionRegistry::open(&linked).is_err());
    fs::set_permissions(&f.db, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(EvolutionRegistry::open(&f.db).is_err());
}
#[test]
fn hatching_experiments_stable_scoped_untrusted_and_operator_promoted() {
    let f = Fixture::new();
    let governance = ApprovalRegistry::open(f._temp.path().join("governance.db")).unwrap();
    let context = "a".repeat(64);
    let h = propose_hatching(
        &scope(),
        HatchingAnswers {
            name: "Test".into(),
            tone: "measured".into(),
            purpose: "synthetic fixture".into(),
            provider_label: "offline".into(),
        },
        &governance,
        &context,
    )
    .unwrap();
    assert!(
        h.config_proposal["requires_operator_configuration"]
            .as_bool()
            .unwrap()
    );
    assert!(
        governance
            .document(&scope(), DocumentKind::Persona)
            .unwrap()
            .is_none()
    );
    let e = ExperimentRegistry::new(f.store.clone());
    let (experiment, hash) = e
        .create(
            &scope(),
            vec![
                PersonaVariant {
                    name: "A".into(),
                    prompt: "Calm".into(),
                },
                PersonaVariant {
                    name: "B".into(),
                    prompt: "Terse".into(),
                },
            ],
        )
        .unwrap();
    let assignment = e.assign(&scope(), &experiment.id).unwrap();
    let reloaded = ExperimentRegistry::new(EvolutionRegistry::open(&f.db).unwrap());
    assert_eq!(
        assignment,
        reloaded.assign(&scope(), &experiment.id).unwrap()
    );
    e.record_outcome(&scope(), &experiment.id, "event-one", true, false)
        .unwrap();
    e.record_outcome(&scope(), &experiment.id, "event-one", false, true)
        .unwrap();
    let comparison = e.compare(&scope(), &experiment.id).unwrap();
    assert_eq!(comparison.iter().map(|c| c.observations).sum::<u64>(), 1);
    assert!(comparison.iter().all(|c| !c.trusted_for_promotion));
    let other = Scope {
        user: "other".into(),
        ..scope()
    };
    assert!(e.assign(&other, &experiment.id).is_err());
    assert!(
        e.propose_promotion(
            &scope(),
            &experiment.id,
            "bad",
            "A",
            &governance,
            0,
            &context,
            "operator",
            true
        )
        .is_err()
    );
    assert!(
        e.propose_promotion(
            &scope(),
            &experiment.id,
            &hash,
            "A",
            &governance,
            0,
            &context,
            "model",
            true
        )
        .is_err()
    );
    let p = e
        .propose_promotion(
            &scope(),
            &experiment.id,
            &hash,
            "A",
            &governance,
            0,
            &context,
            "operator",
            true,
        )
        .unwrap();
    assert!(
        governance
            .document(&scope(), DocumentKind::Persona)
            .unwrap()
            .is_none()
    );
    governance
        .decide_document(
            &scope(),
            &p.id,
            &p.proposal_hash,
            &context,
            Decision::Approve,
            "operator",
        )
        .unwrap();
    assert_eq!(
        governance
            .document(&scope(), DocumentKind::Persona)
            .unwrap()
            .unwrap()
            .content,
        "Calm"
    );
}

#[test]
fn uncertain_effect_keeps_recovery_journal_and_never_replays() {
    let f = Fixture::new();
    let p = f.approved();
    let c = rusqlite::Connection::open(&f.db).unwrap();
    c.execute_batch("CREATE TRIGGER synthetic_checkpoint_failure BEFORE UPDATE ON evolution_journal BEGIN SELECT RAISE(ABORT,'synthetic durability failure'); END;").unwrap();
    assert!(
        f.store
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    let reloaded = EvolutionRegistry::open(&f.db).unwrap();
    assert_eq!(
        reloaded.proposal(&scope(), &p.plan.id).unwrap().status,
        "recovery_required"
    );
    assert!(
        reloaded
            .recovery_journal(&scope(), &p.plan.id)
            .unwrap()
            .is_some()
    );
    assert_eq!(fs::read_to_string(f.repo.join("a.txt")).unwrap(), "old\n");
    c.execute_batch("DROP TRIGGER synthetic_checkpoint_failure;")
        .unwrap();
    assert!(
        reloaded
            .apply(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
    assert!(
        reloaded
            .rollback(
                &scope(),
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &f.policy,
                "operator",
                true
            )
            .is_err()
    );
}
#[test]
fn includes_filters_and_git_metadata_links_rejected() {
    let f = Fixture::new();
    let config = f.repo.join(".git/config");
    let old = fs::read_to_string(&config).unwrap();
    fs::write(&config, format!("{old}\n[include]\npath = /tmp/not-read\n")).unwrap();
    assert!(
        f.store
            .propose(&scope(), &f.policy, &f.repo, DIFF, "include")
            .is_err()
    );
    fs::write(&config, &old).unwrap();
    symlink("/tmp", f.repo.join(".git/objects/untrusted")).unwrap();
    assert!(
        f.store
            .propose(&scope(), &f.policy, &f.repo, DIFF, "metadata link")
            .is_err()
    );
}
