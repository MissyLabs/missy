//! Operator-reviewed source is compiled offline, never automatically executed.
use crate::*;
use anyhow::{Context, Result, ensure};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::Path,
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustSource {
    pub id: String,
    pub source: String,
    pub definition: ToolDefinition,
    pub args: Vec<String>,
    pub capabilities: PluginCapabilities,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustSourceProposal {
    pub hash: String,
    pub scope: Scope,
    pub source: RustSource,
    pub reviewed: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RustBuildConfig {
    pub rustc: PathBuf,
    pub rustc_sha256: String,
    pub sysroot: PathBuf,
    pub linker: PathBuf,
    pub linker_sha256: String,
    /// Toolchain/loader directories only, not HOME/workspace/credential trees.
    pub read_roots: Vec<PathBuf>,
    /// Explicit native library search paths, inside read_roots.
    pub native_library_roots: Vec<PathBuf>,
    /// Operator-pinned link startup/runtime arguments, never model-provided.
    pub linker_args: Vec<String>,
    pub timeout_ms: u64,
    pub max_output_bytes: usize,
}
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn hash(scope: &Scope, source: &RustSource) -> Result<String> {
    Ok(sha(&serde_json::to_vec(&(
        "missy-rust-source-v1",
        scope,
        source,
    ))?))
}
impl RustSource {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.source.trim().is_empty()
                && self.source.len() <= 49152
                && !self.source.contains('\0'),
            "Rust source must be nonempty UTF-8, at most 48KiB without NUL"
        );
        // Validate same schema/args/capabilities as eventual plugin. No existing
        // ELF is needed during a data-only proposal.
        self.manifest(
            PathBuf::from("/usr/bin/missy_source_placeholder"),
            "0".repeat(64),
        )
        .validate()
    }
    fn manifest(&self, executable: PathBuf, sha256: String) -> PluginManifest {
        PluginManifest {
            id: self.id.clone(),
            definition: self.definition.clone(),
            command: CommandTemplate {
                executable,
                sha256,
                args: self.args.clone(),
            },
            capabilities: self.capabilities.clone(),
        }
    }
}
impl RustBuildConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=120000).contains(&self.timeout_ms)
                && (1..=1048576).contains(&self.max_output_bytes),
            "bounded compiler timeout/output required"
        );
        ensure!(
            !self.read_roots.is_empty() && self.read_roots.len() <= 16,
            "explicit bounded toolchain read roots required"
        );
        crate::sandbox::validate_root(&self.sysroot)?;
        for path in &self.read_roots {
            crate::sandbox::validate_root(path)?;
        }
        ensure!(
            self.native_library_roots.len() <= 16 && self.linker_args.len() <= 32,
            "bounded linker configuration required"
        );
        for path in &self.native_library_roots {
            crate::sandbox::validate_root(path)?;
            ensure!(
                self.read_roots.iter().any(|root| path.starts_with(root)),
                "native library root must be explicitly readable"
            );
        }
        ensure!(
            self.linker_args
                .iter()
                .all(|arg| !arg.is_empty() && arg.len() <= 4096 && !arg.contains('\0')),
            "invalid operator linker argument"
        );
        ensure!(
            self.read_roots
                .iter()
                .any(|root| self.sysroot.starts_with(root)),
            "sysroot must be explicitly readable"
        );
        for path in [&self.rustc, &self.linker] {
            ensure!(
                path.is_absolute() && path.file_name().is_some(),
                "explicit compiler/linker path required"
            );
        }
        Ok(())
    }
}
impl ToolRegistry {
    fn source_path(&self, scope: &Scope, id: &str) -> Result<PathBuf> {
        scope.validate()?;
        validate_id(id)?;
        Ok(self.path().join(format!(
            "source-{}-{id}.json",
            sha(&serde_json::to_vec(scope)?)
        )))
    }
    fn load_source(&self, scope: &Scope, id: &str) -> Result<RustSourceProposal> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(self.source_path(scope, id)?)?;
        let m = file.metadata()?;
        ensure!(
            m.is_file()
                && m.nlink() == 1
                && m.uid() == unsafe { libc::geteuid() }
                && m.mode() & 0o077 == 0
                && m.len() <= 131072,
            "source proposal must be private bounded regular file"
        );
        let mut bytes = vec![];
        file.take(131073).read_to_end(&mut bytes)?;
        let proposal: RustSourceProposal = serde_json::from_slice(&bytes)
            .map_err(|_| anyhow::anyhow!("invalid source proposal record"))?;
        proposal.source.validate()?;
        ensure!(
            proposal.scope == *scope
                && proposal.source.id == id
                && proposal.hash == hash(scope, &proposal.source)?,
            "source scope/id/hash drift"
        );
        Ok(proposal)
    }
    fn save_source(&self, p: &RustSourceProposal, new: bool) -> Result<()> {
        let mut temp = tempfile::NamedTempFile::new_in(self.path())?;
        temp.as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))?;
        temp.write_all(&serde_json::to_vec(p)?)?;
        temp.as_file().sync_all()?;
        let path = self.source_path(&p.scope, &p.source.id)?;
        if new {
            temp.persist_noclobber(path)?;
        } else {
            temp.persist(path)?;
        }
        File::open(self.path())?.sync_all()?;
        Ok(())
    }
    pub fn propose_rust_source(
        &self,
        scope: &Scope,
        source: RustSource,
    ) -> Result<RustSourceProposal> {
        let _lock = self.lock()?;
        scope.validate()?;
        source.validate()?;
        let p = RustSourceProposal {
            hash: hash(scope, &source)?,
            scope: scope.clone(),
            source,
            reviewed: false,
        };
        self.save_source(&p, true)?;
        Ok(p)
    }
    pub fn rust_source(&self, scope: &Scope, id: &str) -> Result<RustSourceProposal> {
        let _lock = self.lock()?;
        self.load_source(scope, id)
    }
    pub fn review_rust_source(
        &self,
        scope: &Scope,
        id: &str,
        exact_hash: &str,
        approve: bool,
    ) -> Result<()> {
        let _lock = self.lock()?;
        let mut p = self.load_source(scope, id)?;
        ensure!(p.hash == exact_hash, "exact source hash approval mismatch");
        p.reviewed = approve;
        self.save_source(&p, false)
    }
    pub async fn build_rust_source(
        &self,
        scope: &Scope,
        id: &str,
        exact_hash: &str,
        config: &RustBuildConfig,
        confirm: bool,
        cancellation: CancellationToken,
    ) -> Result<PluginProposal> {
        ensure!(
            confirm,
            "source build requires explicit operator confirmation"
        );
        config.validate()?;
        crate::sandbox::check_available()?;
        let reviewed = {
            let _lock = self.lock()?;
            self.load_source(scope, id)?
        };
        ensure!(
            reviewed.reviewed && reviewed.hash == exact_hash,
            "review source exact hash before compiling"
        );
        ensure!(
            !cancellation.is_cancelled(),
            "source build cancelled before spawn"
        );
        let compiler = crate::sandbox::pinned_native(
            &CommandTemplate {
                executable: config.rustc.clone(),
                sha256: config.rustc_sha256.clone(),
                args: vec![],
            },
            256 * 1024 * 1024,
        )?;
        let linker = crate::sandbox::pinned_native(
            &CommandTemplate {
                executable: config.linker.clone(),
                sha256: config.linker_sha256.clone(),
                args: vec![],
            },
            128 * 1024 * 1024,
        )?;
        let temp = tempfile::Builder::new()
            .prefix("missy-reviewed-rust-")
            .tempdir()?;
        let source_path = temp.path().join("main.rs");
        fs::write(&source_path, &reviewed.source.source)?;
        let linker_path = temp.path().join("pinned_linker");
        // Write sealed reviewed linker bytes into private child build directory;
        // no ambient PATH compiler/linker discovery or cargo build.rs exists.
        let mut linker_copy = File::create(&linker_path)?;
        let mut linker_reader = linker.try_clone()?;
        use std::io::{Seek, SeekFrom};
        linker_reader.seek(SeekFrom::Start(0))?;
        std::io::copy(&mut linker_reader, &mut linker_copy)?;
        linker_copy.set_permissions(fs::Permissions::from_mode(0o700))?;
        drop(linker_copy);
        let output = temp.path().join("plugin_elf");
        crate::sandbox::compile_rust(
            compiler,
            config,
            temp.path(),
            &source_path,
            &linker_path,
            &output,
            cancellation,
        )
        .await?;
        // Compiler may be malicious. Enforce regular private bounded ELF output
        // using exactly the same sealed-image check as actual plugin dispatch.
        let bytes = read_output(&output)?;
        let artifact_hash = sha(&bytes);
        let manifest_path = self.path().join(format!("built_{id}_{artifact_hash}"));
        let manifest = reviewed
            .source
            .manifest(manifest_path.clone(), artifact_hash);
        let _lock = self.lock()?;
        let current = self.load_source(scope, id)?;
        ensure!(
            current.reviewed && current.hash == reviewed.hash,
            "source review revoked or changed during compile"
        );
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o700)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&manifest_path)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        crate::sandbox::pinned_executable(&manifest.command)?;
        let receipt = serde_json::json!({"format":"missy-native-source-build-v1","scope":scope,
            "source_hash":reviewed.hash,"compiler_configuration_sha256":sha(&serde_json::to_vec(config)?),
            "rustc_sha256":config.rustc_sha256,"linker_sha256":config.linker_sha256,
            "artifact_sha256":manifest.command.sha256,"sandbox":"landlock-seccomp-offline",
            "artifact_executed":false,"binary_reviewed":false,"enabled":false});
        let mut receipt_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(
                self.path()
                    .join(format!("build_{id}_{}.json", manifest.command.sha256)),
            )?;
        receipt_file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
        receipt_file.sync_all()?;
        File::open(self.path())?.sync_all()?;
        drop(_lock);
        // Separate prebuilt registry record is PENDING, never enabled/reviewed.
        self.propose(scope, manifest)
            .context("build artifact retained but pending manifest publication failed")
    }
}
fn read_output(path: &Path) -> Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let m = file.metadata()?;
    ensure!(
        m.is_file() && m.nlink() == 1 && m.len() <= 32 * 1024 * 1024,
        "compiler output is not bounded regular ELF"
    );
    let mut bytes = vec![];
    file.take(32 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 32 * 1024 * 1024 && bytes.starts_with(b"\x7fELF"),
        "compiler did not produce bounded native ELF"
    );
    Ok(bytes)
}
