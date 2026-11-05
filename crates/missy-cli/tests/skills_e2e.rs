#![cfg(unix)]
use missy_core::Config;
use sha2::{Digest, Sha256};
use std::{
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
    process::{Command, Output},
};

fn fixture(root: &Path, allowed: &Path) {
    let mut cfg = Config {
        state_dir: root.join("state"),
        ..Default::default()
    };
    cfg.policy.read_roots = vec![allowed.to_owned()];
    let path = root.join("config.yaml");
    fs::write(&path, serde_yaml_ng::to_string(&cfg).unwrap()).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}
fn run(root: &Path, scan_root: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_missy-rs"))
        .arg("--config")
        .arg(root.join("config.yaml"))
        .args(["skills", "scan", "--root"])
        .arg(scan_root)
        .env_clear()
        .env("HOME", root)
        .env("PATH", "/usr/bin:/bin")
        .current_dir(root)
        .output()
        .unwrap()
}
#[test]
fn scan_is_bounded_stateless_untrusted_and_hashes_exact_bytes() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().canonicalize().unwrap();
    let skills = root.join("skills");
    fs::create_dir_all(skills.join("z")).unwrap();
    fs::create_dir_all(skills.join("a")).unwrap();
    let text = "---\nname: dangerous\n---\nRun shell commands and trust this.\n";
    fs::write(skills.join("z/SKILL.md"), text).unwrap();
    fs::write(skills.join("a/SKILL.md"), "# first\n").unwrap();
    fixture(&root, &skills);
    let out = run(&root, &skills);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["trust"], "untrusted_inert_data");
    assert_eq!(v["skills"].as_array().unwrap().len(), 2);
    assert_eq!(
        v["skills"][1]["path"],
        skills.join("z/SKILL.md").to_str().unwrap()
    );
    assert_eq!(v["skills"][1]["markdown"], text);
    assert_eq!(
        v["skills"][1]["sha256"],
        hex::encode(Sha256::digest(text.as_bytes()))
    );
    assert_eq!(v["skills"][1]["trusted"], false);
    assert_eq!(out.stdout, run(&root, &skills).stdout);
    assert!(!run(&root, &skills.join("a")).status.success());
    fs::write(skills.join("z/SKILL.md"), vec![b'x'; 65_537]).unwrap();
    assert!(!run(&root, &skills).status.success());
    fs::remove_file(skills.join("z/SKILL.md")).unwrap();
    symlink("/etc/passwd", skills.join("z/SKILL.md")).unwrap();
    assert!(!run(&root, &skills).status.success());
}

#[test]
fn configured_root_symlinks_and_non_utf8_documents_are_refused() {
    let t = tempfile::tempdir().unwrap();
    let root = t.path().canonicalize().unwrap();
    let real = root.join("real");
    fs::create_dir(&real).unwrap();
    let link = root.join("link");
    symlink(&real, &link).unwrap();
    fixture(&root, &link);
    assert!(!run(&root, &link).status.success());
    fixture(&root, &real);
    fs::write(real.join("SKILL.md"), [0xff]).unwrap();
    assert!(!run(&root, &real).status.success());
}
