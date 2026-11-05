//! Explicit native OAuth refresh. No login discovery, ambient auth files or CLI.
use super::*;
use missy_core::OAuthConfig;
use serde::{Deserialize, Serialize};
pub(super) fn validate(config: &OAuthConfig) -> Result<()> {
    let url = url::Url::parse(&config.token_url)
        .map_err(|_| anyhow::anyhow!("invalid OAuth token endpoint"))?;
    ensure!(
        url.scheme() == "https"
            || url.scheme() == "http"
                && url
                    .host_str()
                    .is_some_and(|h| h == "127.0.0.1" || h == "localhost" || h == "[::1]"),
        "OAuth requires HTTPS or explicitly policy-allowed loopback fixture HTTP"
    );
    ensure!(
        url.username().is_empty()
            && url.password().is_none()
            && url.query().is_none()
            && url.fragment().is_none(),
        "OAuth endpoint cannot contain credentials/query/fragment"
    );
    ensure!(
        !config.client_id.is_empty()
            && config.client_id.len() <= 256
            && config.timeout_seconds > 0
            && config.timeout_seconds <= 120,
        "invalid OAuth profile bounds"
    );
    validate_env_name(&config.refresh_token_env)?;
    if let Some(reference) = &config.access_token_env {
        validate_env_name(reference)?;
    }
    ensure!(
        config.scopes.len() <= 32
            && config.scopes.iter().all(|s| !s.is_empty()
                && s.len() <= 256
                && s.bytes().all(|b| b.is_ascii_graphic())),
        "invalid OAuth scopes"
    );
    ensure!(
        config.token_vault.is_absolute(),
        "OAuth token_vault must be an explicit absolute private path"
    );
    Ok(())
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Tokens {
    access: String,
    refresh: String,
    expires_at_ms: u64,
}
fn valid(secret: &str) -> bool {
    !secret.is_empty() && secret.len() <= 16384 && secret.bytes().all(|b| b.is_ascii_graphic())
}
fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
async fn open(config: &OAuthConfig) -> Result<missy_security::Vault> {
    loop {
        let selected = config.token_vault.clone();
        if let Some(vault) =
            tokio::task::spawn_blocking(move || missy_security::Vault::try_open(selected))
                .await
                .map_err(|_| anyhow::anyhow!("OAuth vault worker failed"))??
        {
            return Ok(vault);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
fn profile_key(config: &OAuthConfig, name: &str) -> Result<String> {
    Ok(format!(
        "oauth-{}",
        hex::encode(Sha256::digest(serde_json::to_vec(&(
            name,
            &config.token_url,
            &config.client_id,
            &config.scopes
        ))?))
    ))
}
/// Invalidate only the rejected generation, never a concurrent newer refresh.
pub(super) async fn invalidate(config: &OAuthConfig, name: &str, rejected: &str) -> Result<()> {
    let vault = open(config).await?;
    let key = profile_key(config, name)?;
    if let Some(bytes) = vault.get(&key)? {
        let mut tokens: Tokens = serde_json::from_slice(&bytes)
            .map_err(|_| anyhow::anyhow!("OAuth vault token record invalid"))?;
        if tokens.access == rejected {
            tokens.expires_at_ms = 0;
            vault.set(&key, &serde_json::to_vec(&tokens)?)?;
        }
    }
    Ok(())
}
pub(super) async fn credential(
    config: &OAuthConfig,
    name: &str,
    client: &PolicyHttpClient,
    resolver: &dyn CredentialResolver,
    force_refresh: bool,
) -> Result<String> {
    let vault = open(config).await?;
    let key = profile_key(config, name)?;
    let previous = vault
        .get(&key)?
        .map(|bytes| serde_json::from_slice::<Tokens>(&bytes))
        .transpose()
        .map_err(|_| anyhow::anyhow!("OAuth vault token record invalid"))?;
    if let Some(tokens) = &previous
        && !force_refresh
        && tokens.expires_at_ms > now().saturating_add(30_000)
    {
        ensure!(valid(&tokens.access), "OAuth vault access token invalid");
        return Ok(tokens.access.clone());
    }
    if previous.is_none()
        && !force_refresh
        && let Some(reference) = &config.access_token_env
    {
        let access = resolver
            .resolve(reference)
            .map_err(|_| anyhow::anyhow!("OAuth access credential unavailable"))?;
        ensure!(valid(&access), "OAuth access credential invalid");
        let refresh = resolver
            .resolve(&config.refresh_token_env)
            .map_err(|_| anyhow::anyhow!("OAuth refresh credential unavailable"))?;
        ensure!(valid(&refresh), "OAuth refresh credential invalid");
        // Persist the generation so rejection survives process restart.
        vault.set(
            &key,
            &serde_json::to_vec(&Tokens {
                access: access.clone(),
                refresh,
                expires_at_ms: u64::MAX,
            })?,
        )?;
        return Ok(access);
    }
    let refresh = match previous {
        Some(tokens) => tokens.refresh,
        None => resolver
            .resolve(&config.refresh_token_env)
            .map_err(|_| anyhow::anyhow!("OAuth refresh credential unavailable"))?,
    };
    ensure!(valid(&refresh), "OAuth refresh credential invalid");
    let body = {
        let mut form = url::form_urlencoded::Serializer::new(String::new());
        form.append_pair("grant_type", "refresh_token")
            .append_pair("client_id", &config.client_id)
            .append_pair("refresh_token", &refresh);
        if !config.scopes.is_empty() {
            form.append_pair("scope", &config.scopes.join(" "));
        }
        form.finish().into_bytes()
    };
    let response = client
        .clone()
        .with_timeout(Duration::from_secs(config.timeout_seconds))?
        .request(
            "POST",
            &config.token_url,
            vec![(
                "content-type".into(),
                "application/x-www-form-urlencoded".into(),
            )],
            Some(body),
        )
        .await
        .map_err(|_| anyhow::anyhow!("OAuth token transport or policy failure"))?;
    ensure!(
        (200..300).contains(&response.status),
        "OAuth refresh rejected with HTTP status {}",
        response.status
    );
    let value =
        strict_json(&response.body).map_err(|_| anyhow::anyhow!("OAuth token response invalid"))?;
    ensure!(
        string(&value, "token_type")?.eq_ignore_ascii_case("bearer"),
        "unsupported OAuth token type"
    );
    let access = string(&value, "access_token")?.to_owned();
    ensure!(valid(&access), "OAuth token response credential invalid");
    let refresh = value
        .get("refresh_token")
        .map(|v| {
            v.as_str()
                .ok_or_else(|| anyhow::anyhow!("invalid rotated refresh token"))
        })
        .transpose()?
        .unwrap_or(&refresh)
        .to_owned();
    ensure!(valid(&refresh), "rotated refresh token invalid");
    let expiry = number(&value, "expires_in")?;
    ensure!(
        (31..=31_536_000).contains(&expiry),
        "OAuth expiry outside supported bounds"
    );
    vault.set(
        &key,
        &serde_json::to_vec(&Tokens {
            access: access.clone(),
            refresh,
            expires_at_ms: now().saturating_add(expiry.saturating_mul(1000)),
        })?,
    )?;
    Ok(access)
}
