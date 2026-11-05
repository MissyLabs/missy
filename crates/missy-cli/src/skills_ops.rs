//! Explicit, stateless inspection of untrusted skill documentation.
use anyhow::{Context, Result, ensure};
use clap::Subcommand;
use missy_core::Config;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Subcommand)]
pub enum SkillsCommand {
    /// Inspect SKILL.md files as inert data under an exact configured read root.
    Scan {
        #[arg(long)]
        root: PathBuf,
    },
}

#[derive(Serialize)]
struct Skill {
    root: PathBuf,
    path: PathBuf,
    sha256: String,
    markdown: String,
    trusted: bool,
}

pub fn execute(cfg: &Config, command: SkillsCommand) -> Result<()> {
    match command {
        SkillsCommand::Scan { root } => {
            ensure!(
                root.is_absolute()
                    && root
                        .components()
                        .all(|c| matches!(c, Component::RootDir | Component::Normal(_))),
                "root must be an absolute canonical path"
            );
            let normalized: PathBuf = root.components().collect();
            ensure!(
                normalized.as_os_str() == root.as_os_str(),
                "root must be an absolute canonical path"
            );
            ensure!(
                cfg.policy
                    .read_roots
                    .iter()
                    .any(|p| p.as_os_str() == root.as_os_str()),
                "root must exactly match a configured policy read root"
            );
            let skills = scan(&root)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "root": root, "trust": "untrusted_inert_data", "skills": skills
                }))?
            );
            Ok(())
        }
    }
}

#[cfg(unix)]
fn scan(root: &Path) -> Result<Vec<Skill>> {
    use std::{
        ffi::{CStr, CString, OsString},
        fs::File,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::ffi::{OsStrExt, OsStringExt},
        },
    };
    fn openat(parent: &File, name: &std::ffi::OsStr, flags: libc::c_int) -> Result<File> {
        let name = CString::new(name.as_bytes())?;
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(unsafe { File::from_raw_fd(fd) })
    }
    fn entries(dir: &File) -> Result<Vec<OsString>> {
        // Open a separate file description so directory offsets cannot leak between scans.
        let stream_fd = openat(
            dir,
            std::ffi::OsStr::new("."),
            libc::O_RDONLY | libc::O_DIRECTORY,
        )?;
        use std::os::fd::IntoRawFd;
        let fd = stream_fd.into_raw_fd();
        let stream = unsafe { libc::fdopendir(fd) };
        if stream.is_null() {
            unsafe {
                libc::close(fd);
            }
            return Err(std::io::Error::last_os_error().into());
        }
        struct Dir(*mut libc::DIR);
        impl Drop for Dir {
            fn drop(&mut self) {
                unsafe {
                    libc::closedir(self.0);
                }
            }
        }
        let stream = Dir(stream);
        let mut out = Vec::new();
        loop {
            // POSIX requires clearing errno to distinguish EOF from error.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            unsafe {
                *libc::__errno_location() = 0;
            }
            #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
            unsafe {
                *libc::__error() = 0;
            }
            let ent = unsafe { libc::readdir(stream.0) };
            if ent.is_null() {
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(0) {
                    return Err(err.into());
                }
                break;
            }
            let name = unsafe { CStr::from_ptr((*ent).d_name.as_ptr()) }.to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            out.push(OsString::from_vec(name.to_vec()));
            ensure!(out.len() <= 1024, "skill directory entry bound exceeded");
        }
        out.sort();
        Ok(out)
    }
    let mut base = File::open("/")?;
    for c in root.components() {
        if let Component::Normal(name) = c {
            base = openat(&base, name, libc::O_RDONLY | libc::O_DIRECTORY)
                .context("opening skill root without symlinks")?;
        }
    }
    let mut pending = vec![(base, root.to_owned(), 0usize)];
    let (mut dirs, mut count, mut bytes) = (0usize, 0usize, 0usize);
    let mut skills = Vec::new();
    while let Some((dir, path, depth)) = pending.pop() {
        dirs += 1;
        ensure!(dirs <= 512, "skill directory bound exceeded");
        let names = entries(&dir)?;
        count += names.len();
        ensure!(count <= 8192, "skill entry bound exceeded");
        for name in names {
            let c = CString::new(name.as_bytes())?;
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            if unsafe {
                libc::fstatat(
                    dir.as_raw_fd(),
                    c.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } != 0
            {
                return Err(std::io::Error::last_os_error().into());
            }
            let kind = unsafe { stat.assume_init() }.st_mode & libc::S_IFMT;
            ensure!(kind != libc::S_IFLNK, "symlink in skill root");
            if name == "SKILL.md" {
                ensure!(kind == libc::S_IFREG, "skill must be a regular file");
            }
            if kind == libc::S_IFDIR {
                ensure!(
                    depth < 8 && dirs + pending.len() < 512,
                    "skill directory/depth bound exceeded"
                );
                let child = openat(&dir, &name, libc::O_RDONLY | libc::O_DIRECTORY)?;
                pending.push((child, path.join(name), depth + 1));
            } else if kind == libc::S_IFREG && name == "SKILL.md" {
                ensure!(skills.len() < 128, "skill count bound exceeded");
                let file = openat(&dir, &name, libc::O_RDONLY | libc::O_NONBLOCK)?;
                ensure!(file.metadata()?.is_file(), "skill must be a regular file");
                let mut content = Vec::new();
                file.take(65_537).read_to_end(&mut content)?;
                bytes += content.len();
                ensure!(
                    content.len() <= 65_536 && bytes <= 1_048_576,
                    "skill byte bound exceeded"
                );
                let hash = hex::encode(Sha256::digest(&content));
                skills.push(Skill {
                    root: root.to_owned(),
                    path: path.join(name),
                    sha256: hash,
                    markdown: String::from_utf8(content).context("skill must be UTF-8")?,
                    trusted: false,
                });
            }
        }
    }
    skills.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(skills)
}

#[cfg(not(unix))]
fn scan(_: &Path) -> Result<Vec<Skill>> {
    anyhow::bail!("secure skill scan unsupported on this platform")
}
