use crate::{FileChange, hash};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            fs::{MetadataExt, PermissionsExt},
            process::CommandExt,
        },
    },
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
};

/// Explicit operator configuration. This crate has no implicit allowed repository.
/// Roots and repository must be real canonical directories owned by the current
/// OS user. `write_roots` must be narrower than or equal to `read_roots`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvolutionPolicy {
    pub read_roots: Vec<PathBuf>,
    pub write_roots: Vec<PathBuf>,
    pub operator_user: String,
    pub allow_git_commits: bool,
}
impl EvolutionPolicy {
    fn check_repo(&self, repo: &Path, write: bool) -> Result<()> {
        ensure!(
            !self.operator_user.is_empty(),
            "configured operator required"
        );
        for root in self.read_roots.iter().chain(&self.write_roots) {
            ensure!(
                root.is_absolute() && fs::canonicalize(root)? == *root,
                "canonical explicit allowroots required"
            );
            safe_directory(root)?;
        }
        ensure!(
            self.read_roots.iter().any(|r| repo.starts_with(r)),
            "repository outside read allowroots"
        );
        ensure!(
            self.write_roots
                .iter()
                .all(|w| self.read_roots.iter().any(|r| w.starts_with(r))),
            "write roots outside read policy"
        );
        if write {
            ensure!(
                self.write_roots.iter().any(|r| repo.starts_with(r)),
                "repository outside write allowroots"
            );
        }
        Ok(())
    }
    pub(crate) fn check_file(&self, repo: &Path, path: &str) -> Result<()> {
        ensure!(crate::patch::path_valid(path), "unsafe relative path");
        let full = repo.join(path);
        ensure!(
            self.read_roots.iter().any(|r| full.starts_with(r))
                && self.write_roots.iter().any(|r| full.starts_with(r)),
            "file outside explicit read/write roots"
        );
        Ok(())
    }
}
fn cstr(s: impl AsRef<std::ffi::OsStr>) -> Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    Ok(std::ffi::CString::new(s.as_ref().as_bytes())?)
}
fn directory(path: &Path) -> Result<File> {
    ensure!(path.is_absolute(), "absolute directory required");
    let mut fd = File::open("/")?;
    for c in path.components() {
        match c {
            Component::RootDir => {}
            Component::Normal(name) => {
                let name = cstr(name)?;
                let raw = unsafe {
                    libc::openat(
                        fd.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                ensure!(
                    raw >= 0,
                    "unsafe or inaccessible directory component: {}",
                    std::io::Error::last_os_error()
                );
                fd = unsafe { File::from_raw_fd(raw) };
                let m = fd.metadata()?;
                // Sticky /tmp is permitted as an ancestor, never as a repo itself.
                ensure!(
                    m.mode() & 0o022 == 0 || m.mode() & libc::S_ISVTX != 0,
                    "writable directory ancestor"
                );
            }
            _ => anyhow::bail!("noncanonical directory"),
        }
    }
    Ok(fd)
}
pub(crate) fn safe_directory(path: &Path) -> Result<()> {
    let fd = directory(path)?;
    let m = fd.metadata()?;
    ensure!(
        m.is_dir() && m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o022 == 0,
        "directory must be owned and not group/world writable"
    );
    Ok(())
}
fn safe_git_tree(path: &Path, depth: usize, budget: &mut usize) -> Result<()> {
    ensure!(depth <= 32 && *budget > 0, "Git metadata tree too large");
    for entry in fs::read_dir(path)? {
        *budget = budget
            .checked_sub(1)
            .context("Git metadata tree too large")?;
        let entry = entry?;
        let metadata = fs::symlink_metadata(entry.path())?;
        ensure!(
            metadata.uid() == unsafe { libc::geteuid() } && metadata.mode() & 0o022 == 0,
            "unsafe Git metadata owner/permissions"
        );
        if metadata.is_dir() {
            safe_git_tree(&entry.path(), depth + 1, budget)?;
        } else {
            ensure!(
                metadata.is_file() && metadata.nlink() == 1,
                "symlink/hardlink/nonregular Git metadata unsupported"
            );
        }
    }
    Ok(())
}
fn regular_at(parent: &File, name: &str, write: bool, create: bool, mode: u32) -> Result<File> {
    let name = cstr(name)?;
    let raw = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            (if write { libc::O_RDWR } else { libc::O_RDONLY })
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC
                | (if create {
                    libc::O_CREAT | libc::O_EXCL
                } else {
                    0
                }),
            mode,
        )
    };
    ensure!(raw >= 0, "unsafe file: {}", std::io::Error::last_os_error());
    let f = unsafe { File::from_raw_fd(raw) };
    let m = f.metadata()?;
    ensure!(
        m.is_file()
            && m.nlink() == 1
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o022 == 0,
        "regular owned non-hardlinked file required"
    );
    Ok(f)
}
pub(crate) struct Repository {
    pub path: PathBuf,
    root: File,
    gitdir: File,
    _lock: File,
}
impl Repository {
    pub fn open(policy: &EvolutionPolicy, path: &Path, write: bool) -> Result<Self> {
        ensure!(
            path.is_absolute() && fs::canonicalize(path)? == path,
            "canonical repository path required"
        );
        policy.check_repo(path, write)?;
        safe_directory(path)?;
        let root = directory(path)?;
        let git_path = path.join(".git");
        safe_directory(&git_path)?;
        let gitdir = directory(&git_path)?;
        safe_git_tree(&git_path, 0, &mut 100_000)?;
        ensure!(
            !git_path.join("commondir").exists()
                && !git_path.join("objects/info/alternates").exists(),
            "linked worktrees/alternate objects unsupported"
        );
        // Never ask Git to interpret an attacker-controlled include/filter/helper
        // config. This deliberately supports ordinary local disposable repos,
        // not arbitrary developer configurations. No hooks/filters are executed.
        let mut config_file = regular_at(&gitdir, "config", false, false, 0)?;
        ensure!(
            config_file.metadata()?.len() <= 65536,
            "oversized Git config"
        );
        let mut config = String::new();
        config_file.read_to_string(&mut config)?;
        let mut section = String::new();
        for line in config.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
                continue;
            }
            if line.starts_with('[') {
                ensure!(
                    line == "[core]" || line == "[user]",
                    "unsupported Git config section; no includes, filters, remotes or extensions allowed"
                );
                section = line.to_owned();
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .context("unsupported Git config syntax")?;
            let key = key.trim().to_ascii_lowercase();
            let value = value.trim();
            ensure!(
                !value.contains(['\\', '\0']),
                "unsupported Git config continuation"
            );
            match (section.as_str(), key.as_str()) {
                ("[core]", "repositoryformatversion") => {
                    ensure!(value == "0", "unsupported repository format")
                }
                ("[core]", "bare") => ensure!(value == "false", "bare repositories unsupported"),
                ("[core]", "filemode" | "logallrefupdates") => {
                    ensure!(["true", "false"].contains(&value), "invalid Git boolean")
                }
                ("[user]", "name" | "email") => {}
                _ => anyhow::bail!("unsupported local Git config key"),
            }
        }
        // Persistent advisory lock; never unlink it, as that allows two lock inodes.
        let lock_name = "missy-evolution.lock";
        let lock = match regular_at(&gitdir, lock_name, true, true, 0o600) {
            Ok(f) => f,
            Err(_) => regular_at(&gitdir, lock_name, true, false, 0o600)?,
        };
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "repository evolution lock busy"
        );
        let this = Self {
            path: path.to_owned(),
            root,
            gitdir,
            _lock: lock,
        };
        ensure!(
            this.git(&["rev-parse", "--show-prefix"], None, None)?
                .iter()
                .all(|b| *b == b'\n'),
            "repository root required"
        );
        this.branch()?;
        Ok(this)
    }
    fn command(&self, index: Option<&Path>) -> Command {
        let mut c = Command::new("/usr/bin/git");
        c.env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", "/nonexistent")
            .env("LC_ALL", "C")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_ATTR_NOSYSTEM", "1")
            .env("GIT_NO_REPLACE_OBJECTS", "1")
            .env("GIT_DIR", "/proc/self/cwd/.git")
            .env("GIT_WORK_TREE", "/proc/self/cwd")
            .env("GIT_AUTHOR_NAME", "Missy operator evolution")
            .env("GIT_AUTHOR_EMAIL", "operator@localhost")
            .env("GIT_COMMITTER_NAME", "Missy operator evolution")
            .env("GIT_COMMITTER_EMAIL", "operator@localhost")
            .args([
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "commit.gpgSign=false",
                "-c",
                "core.quotePath=true",
            ]);
        if let Some(p) = index {
            c.env("GIT_INDEX_FILE", p);
        }
        let fd = self.root.as_raw_fd();
        // Pin cwd by descriptor. No pathname lookup between check and subprocess.
        unsafe {
            c.pre_exec(move || {
                if libc::fchdir(fd) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        c
    }
    fn git(&self, args: &[&str], input: Option<&[u8]>, index: Option<&Path>) -> Result<Vec<u8>> {
        let mut command = self.command(index);
        command
            .args(args)
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn()?;
        if let Some(bytes) = input {
            child.stdin.take().context("git stdin")?.write_all(bytes)?;
        }
        let output = child.wait_with_output()?;
        ensure!(
            output.status.success(),
            "git {} failed: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(output.stdout)
    }
    pub fn head(&self) -> Result<String> {
        Ok(
            String::from_utf8(self.git(&["rev-parse", "--verify", "HEAD"], None, None)?)?
                .trim()
                .to_owned(),
        )
    }
    pub fn branch(&self) -> Result<String> {
        let b = String::from_utf8(self.git(&["symbolic-ref", "-q", "HEAD"], None, None)?)?
            .trim()
            .to_owned();
        ensure!(
            b.starts_with("refs/heads/") && !b.contains(['\n', '\0', ' ']),
            "normal named branch required"
        );
        Ok(b)
    }
    pub fn clean(&self) -> Result<()> {
        let m = self.gitdir.metadata()?;
        let current = directory(&self.path.join(".git"))?.metadata()?;
        ensure!(
            m.ino() == current.ino() && m.dev() == current.dev(),
            "Git directory changed"
        );
        let current = directory(&self.path)?.metadata()?;
        let m = self.root.metadata()?;
        ensure!(
            m.ino() == current.ino() && m.dev() == current.dev(),
            "repository path changed"
        );
        ensure!(
            self.git(
                &[
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                    "--ignore-submodules=none"
                ],
                None,
                None
            )?
            .is_empty(),
            "repository tracked/untracked status must be clean"
        );
        for marker in [
            "MERGE_HEAD",
            "CHERRY_PICK_HEAD",
            "REVERT_HEAD",
            "rebase-apply",
            "rebase-merge",
            "index.lock",
            "HEAD.lock",
        ] {
            ensure!(
                !self.path.join(".git").join(marker).exists(),
                "repository has active Git operation"
            );
        }
        regular_at(&self.gitdir, "index", false, false, 0)?;
        let flags = self.git(&["ls-files", "-v"], None, None)?;
        ensure!(
            flags
                .split(|b| *b == b'\n')
                .filter(|l| !l.is_empty())
                .all(|l| l[0] == b'H'),
            "assume-unchanged/skip-worktree/unmerged index entries unsupported"
        );
        Ok(())
    }
    fn parent(&self, path: &str) -> Result<(File, String)> {
        ensure!(crate::patch::path_valid(path), "unsafe path");
        let p = Path::new(path);
        let mut fd = self.root.try_clone()?;
        if let Some(parent) = p.parent() {
            for component in parent.components() {
                let name = cstr(component.as_os_str())?;
                let raw = unsafe {
                    libc::openat(
                        fd.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                ensure!(raw >= 0, "symlink/unsafe source parent");
                fd = unsafe { File::from_raw_fd(raw) };
                let m = fd.metadata()?;
                ensure!(
                    m.uid() == unsafe { libc::geteuid() } && m.mode() & 0o022 == 0,
                    "unsafe source parent ownership"
                );
            }
        }
        Ok((
            fd,
            p.file_name()
                .context("filename")?
                .to_str()
                .context("UTF-8 filename")?
                .to_owned(),
        ))
    }
    pub fn read_file(&self, path: &str) -> Result<(String, u32)> {
        let (parent, name) = self.parent(path)?;
        let mut f = regular_at(&parent, &name, false, false, 0)?;
        ensure!(
            f.metadata()?.len() <= 4 * 1024 * 1024,
            "source file too large"
        );
        let mode = f.metadata()?.mode() & 0o777;
        let mut text = String::new();
        f.read_to_string(&mut text)?;
        // Exact tracked-file check prevents ignored files from being modified.
        let entry = self.git(&["ls-files", "--stage", "--", path], None, None)?;
        let entry = String::from_utf8(entry)?;
        ensure!(
            (entry.starts_with("100644 ") || entry.starts_with("100755 "))
                && entry.lines().count() == 1
                && entry.contains(" 0\t"),
            "single stage-zero regular tracked file required"
        );
        Ok((text, mode))
    }
    pub fn verify_files(&self, files: &[FileChange], policy: &EvolutionPolicy) -> Result<()> {
        for f in files {
            policy.check_file(&self.path, &f.path)?;
            let (text, mode) = self.read_file(&f.path)?;
            ensure!(
                text == f.before && mode == f.mode,
                "source snapshot changed: {}",
                f.path
            );
        }
        Ok(())
    }
    pub fn install(
        &self,
        files: &[FileChange],
        old_head: &str,
        branch: &str,
        title: &str,
        mut checkpoint: impl FnMut(&serde_json::Value) -> Result<()>,
        mut journal: serde_json::Value,
    ) -> Result<String> {
        self.clean()?;
        ensure!(
            self.head()? == old_head && self.branch()? == branch,
            "repository moved"
        );
        // Build an isolated index and commit before touching any tracked file.
        // Plumbing only: no hooks, filters, project commands, or shell execution.
        let scratch = tempfile::Builder::new()
            .prefix("missy-evolve-stage-")
            .tempdir_in(self.path.join(".git"))?;
        let index = scratch.path().join("index");
        self.git(&["read-tree", old_head], None, Some(&index))?;
        for file in files {
            let blob = String::from_utf8(self.git(
                &["hash-object", "-w", "--stdin", "--no-filters"],
                Some(file.after.as_bytes()),
                None,
            )?)?
            .trim()
            .to_owned();
            let mode = if file.mode & 0o111 != 0 {
                "100755"
            } else {
                "100644"
            };
            self.git(
                &[
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    mode,
                    &blob,
                    &file.path,
                ],
                None,
                Some(&index),
            )?;
        }
        let tree = String::from_utf8(self.git(&["write-tree"], None, Some(&index))?)?
            .trim()
            .to_owned();
        let commit = String::from_utf8(self.git(
            &["commit-tree", &tree, "-p", old_head],
            Some(format!("{title}\n").as_bytes()),
            None,
        )?)?
        .trim()
        .to_owned();
        journal["new_commit"] = commit.clone().into();
        journal["scratch_index"] = index.to_string_lossy().to_string().into();
        journal["installed"] = serde_json::json!([]);
        journal["backups"] = serde_json::json!([]);
        checkpoint(&journal)?;
        // Exclusive Git index lock protects cooperative Git writers too.
        let mut index_lock = regular_at(&self.gitdir, "index.lock", true, true, 0o600)?;
        let index_bytes = fs::read(&index)?;
        index_lock.write_all(&index_bytes)?;
        index_lock.sync_all()?;
        // Multi-file installation is journaled, not falsely advertised as one
        // filesystem transaction. Each file exchange is atomic with its backup.
        // A crash or mismatch is fail-closed and leaves manual recovery evidence.
        for file in files {
            let (parent, name) = self.parent(&file.path)?;
            let mut original = regular_at(&parent, &name, false, false, 0)?;
            let meta = original.metadata()?;
            let mut text = String::new();
            original.read_to_string(&mut text)?;
            ensure!(text == file.before, "source changed before exchange");
            let temp = format!(".missy-evolve-{}", uuid::Uuid::new_v4());
            let mut staged = regular_at(&parent, &temp, true, true, 0o600)?;
            staged.write_all(file.after.as_bytes())?;
            staged.set_permissions(fs::Permissions::from_mode(file.mode))?;
            staged.sync_all()?;
            let backup = Path::new(&file.path)
                .with_file_name(&temp)
                .to_string_lossy()
                .to_string();
            journal["backups"].as_array_mut().context("journal array")?.push(serde_json::json!({"path":file.path,"backup":backup,"before_hash":hash(&file.before),"after_hash":hash(&file.after)}));
            checkpoint(&journal)?;
            let name_c = cstr(&name)?;
            let temp_c = cstr(&temp)?;
            let current = regular_at(&parent, &name, false, false, 0)?.metadata()?;
            ensure!(
                current.ino() == meta.ino() && current.dev() == meta.dev(),
                "source inode changed"
            );
            let result = unsafe {
                libc::syscall(
                    libc::SYS_renameat2,
                    parent.as_raw_fd(),
                    temp_c.as_ptr(),
                    parent.as_raw_fd(),
                    name_c.as_ptr(),
                    libc::RENAME_EXCHANGE,
                )
            };
            ensure!(
                result == 0,
                "atomic file exchange failed: {}",
                std::io::Error::last_os_error()
            );
            parent.sync_all()?;
            let saved = regular_at(&parent, &temp, false, false, 0)?.metadata()?;
            ensure!(
                saved.ino() == meta.ino() && saved.dev() == meta.dev(),
                "raced source exchange; inspect recovery journal"
            );
            journal["installed"]
                .as_array_mut()
                .context("journal array")?
                .push(file.path.clone().into());
            checkpoint(&journal)?;
        }
        ensure!(
            self.head()? == old_head && self.branch()? == branch,
            "HEAD changed during installation"
        );
        self.git(&["update-ref", branch, &commit, old_head], None, None)?;
        journal["ref_updated"] = true.into();
        checkpoint(&journal)?;
        let a = cstr("index.lock")?;
        let b = cstr("index")?;
        // Existing index must still be a regular single-link file.
        regular_at(&self.gitdir, "index", false, false, 0)?;
        ensure!(
            unsafe {
                libc::renameat(
                    self.gitdir.as_raw_fd(),
                    a.as_ptr(),
                    self.gitdir.as_raw_fd(),
                    b.as_ptr(),
                )
            } == 0,
            "index installation failed"
        );
        self.gitdir.sync_all()?;
        // Delete only the exact temporary backups we created, after success.
        for backup in journal["backups"].as_array().context("backups")? {
            let path = backup["backup"].as_str().context("backup path")?;
            let (parent, name) = self.parent(path)?;
            let name = cstr(name)?;
            ensure!(
                unsafe { libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), 0) } == 0,
                "backup cleanup failed"
            );
            parent.sync_all()?;
        }
        self.clean()?;
        ensure!(self.head()? == commit, "unexpected final HEAD");
        Ok(commit)
    }
}
