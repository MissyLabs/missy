//! Crash-like fixture: inject a journal failure after first atomic source swap.
use missy_core::Scope;
use missy_evolution::{EvolutionPolicy, EvolutionRegistry};
use std::{fs, os::unix::fs::PermissionsExt, process::Command};
#[test]
fn partial_multi_file_effect_retains_original_backup_and_blocks_replay() {
    let t = tempfile::tempdir().unwrap();
    fs::set_permissions(t.path(), fs::Permissions::from_mode(0o700)).unwrap();
    let repo = t.path().join("repo");
    fs::create_dir(&repo).unwrap();
    let git = |args: &[&str]| {
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
            .current_dir(&repo)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    };
    git(&["init", "-q", "-b", "fixture"]);
    fs::write(repo.join("a.txt"), "old\n").unwrap();
    fs::write(repo.join("b.txt"), "old\n").unwrap();
    git(&["add", "."]);
    git(&["commit", "-qm", "base"]);
    let db = t.path().join("state.db");
    let registry = EvolutionRegistry::open(&db).unwrap();
    let scope = Scope {
        tenant: "test".into(),
        user: "operator".into(),
        channel: "fixture".into(),
        session: "one".into(),
    };
    let policy = EvolutionPolicy {
        read_roots: vec![repo.clone()],
        write_roots: vec![repo.clone()],
        operator_user: "operator".into(),
        allow_git_commits: true,
    };
    let diff = ["a.txt", "b.txt"]
        .iter()
        .map(|p| format!("diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-old\n+new\n"))
        .collect::<String>();
    let p = registry
        .propose(&scope, &policy, &repo, &diff, "partial fixture")
        .unwrap();
    registry
        .review(&scope, &p.plan.id, &p.plan_hash, "operator", true)
        .unwrap();
    let c = rusqlite::Connection::open(&db).unwrap();
    c.execute_batch("CREATE TRIGGER fail_after_swap BEFORE UPDATE ON evolution_journal WHEN json_array_length(NEW.payload,'$.installed') > 0 BEGIN SELECT RAISE(ABORT,'synthetic post-swap failure'); END;").unwrap();
    assert!(
        registry
            .apply(
                &scope,
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &policy,
                "operator",
                true
            )
            .is_err()
    );
    assert_eq!(
        registry.proposal(&scope, &p.plan.id).unwrap().status,
        "recovery_required"
    );
    assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "new\n");
    assert_eq!(fs::read_to_string(repo.join("b.txt")).unwrap(), "old\n");
    let j = registry
        .recovery_journal(&scope, &p.plan.id)
        .unwrap()
        .unwrap();
    let backup = j["backups"][0]["backup"].as_str().unwrap();
    assert_eq!(fs::read_to_string(repo.join(backup)).unwrap(), "old\n");
    assert!(repo.join(".git/index.lock").exists());
    c.execute_batch("DROP TRIGGER fail_after_swap;").unwrap();
    assert!(
        registry
            .apply(
                &scope,
                &p.plan.id,
                &p.plan_hash,
                &p.plan.base_head,
                &policy,
                "operator",
                true
            )
            .is_err()
    );
    assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "new\n");
}
