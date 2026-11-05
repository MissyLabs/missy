use super::*;
use std::os::unix::fs::{PermissionsExt, symlink};

fn policy(root: &Path) -> Policy {
    Policy::new(PolicyConfig {
        write_roots: vec![root.into()],
        max_file_bytes: 64,
        ..Default::default()
    })
    .unwrap()
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn delete_exact_bytes_and_empty_file_no_journal_left() {
    let tmp = tempfile::tempdir().unwrap();
    let p = policy(tmp.path());
    let path = tmp.path().join("file");
    for bytes in [b"exact\0bytes\xff".as_slice(), b"".as_slice()] {
        std::fs::write(&path, bytes).unwrap();
        p.delete_file(&path, &digest(bytes)).unwrap();
        assert!(!path.exists());
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
    }
}
#[test]
fn delete_hash_validation_precedes_any_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let p = policy(tmp.path());
    let path = tmp.path().join("file");
    std::fs::write(&path, b"right").unwrap();
    for hash in [
        String::new(),
        "a".repeat(63),
        "a".repeat(65),
        "z".repeat(64),
        digest(b"wrong"),
    ] {
        assert!(p.delete_file(&path, &hash).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"right");
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
    }
    p.delete_file(&path, &digest(b"right").to_ascii_uppercase())
        .unwrap();
}
#[test]
fn delete_denies_unsafe_paths_links_types_size_and_read_only_roots() {
    let tmp = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let path = tmp.path().join("file");
    std::fs::write(&path, b"safe").unwrap();
    std::fs::write(outside.path().join("file"), b"safe").unwrap();
    symlink(&path, tmp.path().join("symlink")).unwrap();
    symlink(outside.path(), tmp.path().join("parent-link")).unwrap();
    std::fs::hard_link(&path, tmp.path().join("hardlink")).unwrap();
    std::fs::create_dir(tmp.path().join("directory")).unwrap();
    std::fs::write(tmp.path().join("large"), [0; 65]).unwrap();
    let fifo = cstr(tmp.path().join("fifo").as_os_str()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
    std::fs::create_dir(tmp.path().join(".missy-delete-protected")).unwrap();
    std::fs::write(tmp.path().join(".missy-delete-protected/file"), b"safe").unwrap();
    let p = policy(tmp.path());
    for denied in [
        path.clone(),
        tmp.path().join("hardlink"),
        tmp.path().join("symlink"),
        tmp.path().join("parent-link/file"),
        tmp.path().join("directory"),
        tmp.path().join("missing/file"),
        tmp.path().join("large"),
        tmp.path().join("fifo"),
        tmp.path().join(".missy-delete-protected/file"),
        tmp.path().join("./file"),
        tmp.path().join("../file"),
        outside.path().join("file"),
        tmp.path().to_path_buf(),
        PathBuf::from("relative"),
    ] {
        assert!(
            p.delete_file(&denied, &digest(b"safe")).is_err(),
            "{denied:?}"
        );
    }
    std::fs::remove_file(tmp.path().join("hardlink")).unwrap();
    let read_only = Policy::new(PolicyConfig {
        read_roots: vec![tmp.path().into()],
        max_file_bytes: 64,
        ..Default::default()
    })
    .unwrap();
    assert!(read_only.delete_file(&path, &digest(b"safe")).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"safe");
    assert_eq!(std::fs::read(outside.path().join("file")).unwrap(), b"safe");
}
#[test]
fn delete_requires_owned_file_and_private_parent() {
    let tmp = tempfile::tempdir().unwrap();
    let p = policy(tmp.path());
    let path = tmp.path().join("file");
    std::fs::write(&path, b"safe").unwrap();
    std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
    assert!(p.delete_file(&path, &digest(b"safe")).is_err());
    std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    if unsafe { libc::geteuid() } == 0 {
        let name = cstr(path.as_os_str()).unwrap();
        assert_eq!(unsafe { libc::chown(name.as_ptr(), 12345, 12345) }, 0);
        assert!(p.delete_file(&path, &digest(b"safe")).is_err());
        assert_eq!(unsafe { libc::chown(name.as_ptr(), 0, 0) }, 0);
    }
    assert_eq!(std::fs::read(&path).unwrap(), b"safe");
}
#[test]
fn delete_substituted_inode_even_same_bytes_is_restored_not_deleted() {
    for symlink_replacement in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let p = policy(tmp.path());
        let path = tmp.path().join("file");
        let saved = tmp.path().join("saved");
        std::fs::write(&path, b"safe").unwrap();
        let result = p.delete_file_inner(
            &path,
            &digest(b"safe"),
            || {
                std::fs::rename(&path, &saved).unwrap();
                if symlink_replacement {
                    symlink(&saved, &path).unwrap();
                } else {
                    std::fs::write(&path, b"safe").unwrap();
                }
            },
            || {},
        );
        assert!(result.unwrap_err().to_string().contains("restored"));
        assert_eq!(
            std::fs::symlink_metadata(&path).unwrap().is_symlink(),
            symlink_replacement
        );
        assert_eq!(std::fs::read(&saved).unwrap(), b"safe");
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 2);
    }
}
#[test]
fn delete_failed_restore_preserves_unknown_entry_and_never_overwrites_original() {
    let tmp = tempfile::tempdir().unwrap();
    let p = policy(tmp.path());
    let path = tmp.path().join("file");
    let saved = tmp.path().join("saved");
    std::fs::write(&path, b"safe").unwrap();
    let result = p.delete_file_inner(
        &path,
        &digest(b"safe"),
        || {
            std::fs::rename(&path, &saved).unwrap();
            std::fs::write(&path, b"unknown").unwrap();
        },
        || {
            std::fs::write(&path, b"new original").unwrap();
        },
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("outcome uncertain")
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"new original");
    let journal = std::fs::read_dir(tmp.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .as_bytes()
                .starts_with(b".missy-delete-")
        })
        .unwrap();
    assert_eq!(journal.metadata().unwrap().mode() & 0o777, 0o700);
    assert_eq!(std::fs::read(journal.join("entry")).unwrap(), b"unknown");
    assert_eq!(std::fs::read(&saved).unwrap(), b"safe");
}

#[test]
fn delete_rechecks_bytes_after_quarantine_and_restores_changed_original() {
    let tmp = tempfile::tempdir().unwrap();
    let p = policy(tmp.path());
    let path = tmp.path().join("file");
    std::fs::write(&path, b"safe").unwrap();
    // A previously opened descriptor remains writable after the pathname moves.
    let mut writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    let result = p.delete_file_inner(
        &path,
        &digest(b"safe"),
        || {},
        || {
            writer.write_all(b"edit").unwrap();
            writer.sync_all().unwrap();
        },
    );
    assert!(result.unwrap_err().to_string().contains("restored"));
    assert_eq!(std::fs::read(&path).unwrap(), b"edit");
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
}
