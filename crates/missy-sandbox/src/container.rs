//! Hardened Docker builder only. This module NEVER launches a runtime or pulls images.
use crate::{SandboxConfig, landlock::open_root, validate_path};
use anyhow::{Context, Result, bail};
use missy_core::PolicyConfig;
use missy_policy::Policy;
use serde::{Deserialize, Serialize};
use std::{
    os::unix::{ffi::OsStrExt, fs::FileTypeExt},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerConfig {
    /// Operator-approved Docker-compatible runtime, never PATH lookup.
    pub runtime: PathBuf,
    /// Full immutable prebuilt image reference: repository@sha256:<64 lowercase hex>.
    pub image: String,
    /// Non-root numeric uid:gid used inside the container.
    pub uid: u32,
    pub gid: u32,
    pub pids_limit: u32,
    pub memory_bytes: u64,
    pub cpus: u32,
}

/// Pure argv plan. Consumers own bounded runtime execution and MUST use the exact
/// cleanup argv on cancellation/drop, not `docker ... $(...)` or broad cleanup.
#[derive(Clone, Debug)]
pub struct ContainerCommand {
    pub runtime: PathBuf,
    pub name: String,
    pub args: Vec<String>,
    pub cleanup_args: Vec<String>,
}

/// `approved_runtime` is a separately trusted operator allowlist value, not user
/// tool arguments. Exact payload argv is still checked by missy-policy.
pub fn build_container_command(
    policy: &PolicyConfig,
    sandbox: &SandboxConfig,
    config: &ContainerConfig,
    approved_runtime: &Path,
    argv: &[String],
    mcp: bool,
) -> Result<ContainerCommand> {
    sandbox.validate(policy)?;
    let policy = Policy::new(policy.clone())?;
    if mcp {
        policy.check_mcp_command(argv)?;
    } else {
        policy.check_command(argv)?;
    }
    validate_path(&config.runtime)?;
    validate_path(approved_runtime)?;
    if config.runtime != approved_runtime {
        bail!("container runtime is not the operator-approved absolute path");
    }
    let (repository, digest) = config
        .image
        .split_once("@sha256:")
        .context("container image requires immutable sha256 digest")?;
    if repository.is_empty()
        || repository.starts_with('-')
        || !repository
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"/._:-".contains(&c))
        || digest.len() != 64
        || !digest
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        bail!("invalid immutable container image reference");
    }
    if config.uid == 0
        || config.gid == 0
        || config.pids_limit == 0
        || config.pids_limit > 4096
        || config.memory_bytes < 16 * 1024 * 1024
        || config.memory_bytes > 64 * 1024 * 1024 * 1024
        || config.cpus == 0
        || config.cpus > 64
    {
        bail!("container requires nonroot uid/gid and bounded pids/memory/CPU limits");
    }
    let name = format!("missy-sandbox-{}", uuid::Uuid::new_v4().simple());
    let mut args: Vec<String> = [
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--read-only",
        "--ipc=none",
        "--log-driver=none",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    args.extend([
        "--name".into(),
        name.clone(),
        "--user".into(),
        format!("{}:{}", config.uid, config.gid),
        "--pids-limit".into(),
        config.pids_limit.to_string(),
        "--memory".into(),
        config.memory_bytes.to_string(),
        "--memory-swap".into(),
        config.memory_bytes.to_string(),
        "--cpus".into(),
        config.cpus.to_string(),
        "--ulimit".into(),
        "core=0:0".into(),
        "--workdir=/".into(),
    ]);
    // A write mount deliberately overrides the same read mount; write-only paths
    // are readable in containers, unlike the separate native read/write rights.
    for (roots, readonly) in [(&sandbox.read_roots, true), (&sandbox.write_roots, false)] {
        for root in roots {
            validate_mount(root)?;
            if readonly && sandbox.write_roots.contains(root) {
                continue;
            }
            let path = root
                .to_str()
                .context("container mount path must be UTF-8")?;
            if path.as_bytes().iter().any(|c| b",\n\r".contains(c)) {
                bail!("container mount path contains unsafe mount-option delimiter");
            }
            args.extend([
                "--mount".into(),
                format!(
                    "type=bind,src={path},dst={path},bind-propagation=rprivate{}",
                    if readonly {
                        ",readonly,bind-recursive=readonly"
                    } else {
                        ""
                    }
                ),
            ]);
        }
    }
    // Ignore untrusted image ENTRYPOINT; execute precisely the approved payload.
    args.extend(["--entrypoint".into(), argv[0].clone(), config.image.clone()]);
    args.extend_from_slice(&argv[1..]);
    Ok(ContainerCommand {
        runtime: config.runtime.clone(),
        name: name.clone(),
        args,
        cleanup_args: vec!["rm".into(), "--force".into(), "--".into(), name],
    })
}

fn validate_mount(root: &Path) -> Result<()> {
    open_root(root)?; // Includes O_NOFOLLOW intermediate components.
    // Never expose kernel controls, device trees, standard daemon/socket trees.
    for forbidden in [
        "/proc",
        "/sys",
        "/dev",
        "/run",
        "/var/run",
        "/var/lib/docker",
        "/var/lib/containerd",
    ] {
        let forbidden = Path::new(forbidden);
        if root.starts_with(forbidden) || forbidden.starts_with(root) {
            bail!("container mount overlaps protected host tree");
        }
    }
    // Reject nonstandard Docker sockets too, wherever a mount would expose them.
    // Bounded snapshot only: trusted administrators must not mutate mounts/roots
    // between planning and daemon mount setup (the daemon opens paths separately).
    let mut stack = vec![root.to_owned()];
    let mut entries = 0;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir)? {
            entries += 1;
            if entries > 100_000 {
                bail!("container mount inspection exceeds bounded entry limit");
            }
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_socket() || entry.file_name().as_bytes() == b"docker.sock" {
                bail!("container mount would expose host socket");
            }
            if kind.is_dir() {
                stack.push(entry.path());
            }
        }
    }
    Ok(())
}
