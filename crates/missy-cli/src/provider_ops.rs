//! Explicit provider profile operations. Never prints credential values.
use anyhow::{Result, ensure};
use clap::Subcommand;
use missy_providers::{CredentialResolver, EnvironmentCredentials, HttpProvider};
use serde_json::json;
use std::path::Path;
use tokio_util::sync::CancellationToken;

#[derive(Subcommand)]
pub enum ProviderCommand {
    /// List configured native profiles without contacting providers.
    List,
    /// Plan a restart-only primary profile change. Apply atomically with --confirm.
    Switch {
        provider: String,
        #[arg(long)]
        confirm: bool,
    },
    /// Validate profile and required environment references without inference.
    Check { provider: String },
    /// List API key reference availability, never key values.
    Keypool { provider: String },
    /// Refresh the explicitly configured OAuth profile into its private vault.
    OAuthRefresh { provider: String },
}
pub async fn execute(path: &Path, command: ProviderCommand) -> Result<()> {
    let config = crate::load_config(path)?;
    if matches!(&command, ProviderCommand::List) {
        let profiles = profiles(&config);
        println!(
            "{}",
            json!({"providers":profiles,"routing":config.routing,"values_redacted":true})
        );
        return Ok(());
    }
    if let ProviderCommand::Switch { provider, confirm } = &command {
        return switch(path, &config, provider, *confirm);
    }
    let (ProviderCommand::Check { provider }
    | ProviderCommand::Keypool { provider }
    | ProviderCommand::OAuthRefresh { provider }) = &command
    else {
        unreachable!()
    };
    let profile = config
        .providers
        .iter()
        .find(|p| &p.name == provider)
        .ok_or_else(|| anyhow::anyhow!("configured provider profile not found"))?;
    let native = HttpProvider::new(
        profile.clone(),
        missy_policy::PolicyHttpClient::new(config.policy)?,
    )?;
    let mut references: Vec<_> = profile
        .api_key_env
        .iter()
        .chain(profile.api_key_envs.iter())
        .collect();
    if let Some(oauth) = &profile.oauth {
        references.push(&oauth.refresh_token_env);
        references.extend(oauth.access_token_env.iter());
    }
    let states: Vec<_> = references
        .iter()
        .map(|reference| {
            let available = EnvironmentCredentials.resolve(reference).is_ok_and(|s| {
                !s.is_empty() && s.len() <= 16_384 && s.bytes().all(|b| b.is_ascii_graphic())
            });
            json!({"reference":reference,"available":available})
        })
        .collect();
    match command {
        ProviderCommand::Check { .. } => {
            ensure!(
                profile.oauth.is_some() || states.iter().all(|s| s["available"] == true),
                "one or more configured credential references are unavailable or invalid"
            );
            println!(
                "{}",
                json!({"valid":true,"references":states,"oauth_references_are_bootstrap_only":profile.oauth.is_some(),"vault_status":"not_inspected","network_used":false,"values_redacted":true})
            );
        }
        ProviderCommand::Keypool { .. } => {
            println!(
                "{}",
                json!({"references":states,"rotation":profile.account_rotation,"cooldown":"in_process","values_redacted":true})
            );
        }
        ProviderCommand::OAuthRefresh { .. } => {
            let cancellation = CancellationToken::new();
            let refresh = native.refresh_oauth(cancellation.clone());
            tokio::pin!(refresh);
            tokio::select! {
                biased;
                signal = crate::production::shutdown_signal() => {
                    signal?;
                    cancellation.cancel();
                    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), &mut refresh).await;
                    anyhow::bail!("OAuth refresh interrupted; if refresh reached the issuer, operator credential recovery may be required");
                }
                result = &mut refresh => result?,
            }
            println!(
                "{}",
                json!({"refreshed":true,"persisted":true,"inference_used":false,"values_redacted":true})
            );
        }
        ProviderCommand::List | ProviderCommand::Switch { .. } => unreachable!(),
    }
    Ok(())
}

fn profiles(config: &missy_core::Config) -> Vec<serde_json::Value> {
    config.providers.iter().map(|p| json!({
        "name":p.name,"kind":p.kind,"model":p.model,"stream":p.stream,
        "auth":if p.oauth.is_some(){"oauth"}else if p.api_key_env.is_some() || !p.api_key_envs.is_empty(){"api_key"}else{"none"},
        "account_count":p.api_key_envs.len().max(usize::from(p.api_key_env.is_some())),
        "account_rotation":p.account_rotation,"routing":p.routing,
    })).collect()
}
fn switch(path: &Path, config: &missy_core::Config, provider: &str, confirm: bool) -> Result<()> {
    use std::{
        fs::{File, OpenOptions},
        io::{Read, Write},
        os::{
            fd::AsRawFd,
            unix::fs::{MetadataExt, OpenOptionsExt},
        },
    };
    fn read(path: &Path) -> Result<Vec<u8>> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(path)?;
        let m = file.metadata()?;
        ensure!(
            m.is_file()
                && m.uid() == unsafe { libc::geteuid() }
                && m.nlink() == 1
                && m.mode() & 0o077 == 0
                && m.len() <= 2_097_152,
            "switch requires owned private single-link regular configuration"
        );
        let mut bytes = vec![];
        file.take(2_097_153).read_to_end(&mut bytes)?;
        ensure!(bytes.len() <= 2_097_152, "configuration too large");
        Ok(bytes)
    }
    let before = zeroize::Zeroizing::new(read(path)?);
    let canonical = std::fs::canonicalize(path)?;
    let parent = canonical
        .parent()
        .ok_or_else(|| anyhow::anyhow!("config parent required"))?;
    let m = std::fs::symlink_metadata(parent)?;
    ensure!(
        m.is_dir()
            && !m.file_type().is_symlink()
            && m.uid() == unsafe { libc::geteuid() }
            && m.mode() & 0o022 == 0,
        "config parent must be owned and not writable by others"
    );
    let mut next = config.clone();
    let index = next
        .providers
        .iter()
        .position(|p| p.name == provider)
        .ok_or_else(|| anyhow::anyhow!("configured provider profile not found"))?;
    let selected = next.providers.remove(index);
    next.providers.insert(0, selected);
    // Switch is explicit ordered routing, preserving task/tier eligibility and
    // every profile, price, limiter, credential reference, and policy boundary.
    next.routing.mode = missy_core::RoutingMode::Ordered;
    let priority = next
        .providers
        .iter()
        .map(|p| p.routing.priority)
        .min()
        .unwrap_or(0);
    next.providers[0].routing.priority = priority;
    next.validate()?;
    missy_providers::ProviderPool::with_routing(
        next.providers.clone(),
        next.policy.clone(),
        next.routing.clone(),
    )?;
    ensure!(
        crate::production::fingerprint(config)?
            == crate::production::fingerprint(&crate::load_config(path)?)?,
        "configuration changed while preparing switch"
    );
    if confirm {
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
            .open(canonical.with_extension("yaml.lock"))?;
        let m = lock.metadata()?;
        ensure!(
            m.is_file()
                && m.nlink() == 1
                && m.uid() == unsafe { libc::geteuid() }
                && m.mode() & 0o077 == 0,
            "invalid config mutation lock"
        );
        ensure!(
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0,
            "another config mutation is active"
        );
        let temp = parent.join(format!(".provider-switch-{}", uuid::Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(&temp)?;
            let encoded = zeroize::Zeroizing::new(serde_yaml_ng::to_string(&next)?);
            file.write_all(encoded.as_bytes())?;
            file.sync_all()?;
            crate::production::fingerprint(&crate::load_config(&temp)?)?;
            ensure!(
                read(path)?.as_slice() == before.as_slice(),
                "configuration changed during switch"
            );
            std::fs::rename(&temp, &canonical)?;
            File::open(parent)?.sync_all()?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result?;
    }
    println!(
        "{}",
        json!({"selected":provider,"applied":confirm,"restart_required":true,"running_services_changed":false,"hot_reload_supported":false,"mode":"ordered","values_redacted":true})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    fn config() -> missy_core::Config {
        missy_core::Config {
            providers: ["first", "second"]
                .into_iter()
                .map(|name| missy_core::ProviderConfig {
                    name: name.into(),
                    model: "fixture".into(),
                    api_key_env: Some("NEVER_RESOLVE_OR_PRINT_ME".into()),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }
    #[test]
    fn profile_listing_does_not_resolve_or_disclose_credentials() {
        let value = serde_json::to_string(&profiles(&config())).unwrap();
        assert!(!value.contains("NEVER_RESOLVE_OR_PRINT_ME"));
        assert!(!value.contains("base_url"));
        assert!(value.contains("account_count") && value.contains("routing"));
    }
    #[test]
    fn atomic_restart_only_switch_plan_and_apply() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = dir.path().join("config.yaml");
        let mut config = config();
        config.routing.mode = missy_core::RoutingMode::Weighted;
        let original = serde_yaml_ng::to_string(&config).unwrap();
        std::fs::write(&path, &original).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let config = crate::load_config(&path).unwrap();
        switch(&path, &config, "second", false).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
        switch(&path, &config, "second", true).unwrap();
        let updated = crate::load_config(&path).unwrap();
        assert_eq!(updated.providers[0].name, "second");
        assert_eq!(updated.routing.mode, missy_core::RoutingMode::Ordered);
        assert_eq!(updated.providers.len(), 2);
        assert_eq!(
            updated.providers[0].api_key_env,
            config.providers[1].api_key_env
        );
        assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        assert!(switch(&path, &updated, "missing", true).is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }
    #[test]
    fn switch_refuses_symlink_and_nonprivate_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(&path, serde_yaml_ng::to_string(&config()).unwrap()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(switch(&path, &config(), "first", true).is_err());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let link = dir.path().join("link.yaml");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(switch(&link, &config(), "first", true).is_err());
    }
}
