//! Opt-in, resource-allowlisted Discord REST tools. No gateway credentials.
//! Uploads MUST be wrapped in the parent's durable, scoped side-effect approval
//! guard. This executor cannot itself attest that a human approved a ToolContext.
use anyhow::{Result, anyhow, bail};
use async_trait::async_trait;
use missy_core::{
    PolicyConfig, ToolCall, ToolContext, ToolDefinition, ToolDispatchRefused, ToolExecutor,
};
use missy_policy::{HttpResponse, MAX_HTTP_REQUEST_BYTES, Policy, PolicyHttpClient};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    future::Future,
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use url::Url;

const API: &str = "https://discord.com/api/v10";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiscordToolsConfig {
    pub enabled: bool,
    pub lookup_enabled: bool,
    pub upload_enabled: bool,
    /// Trusted local configuration only. Never sourced from tool arguments.
    pub token_env: String,
    pub bot_user_id: String,
    pub allowed_guilds: Vec<String>,
    pub allowed_users: Vec<String>,
    /// channel ID -> guild ID, explicit operator attestation of membership.
    pub allowed_channels: BTreeMap<String, String>,
    pub max_file_bytes: usize,
    pub api_base: String,
    /// Explicit test-only opt-in; only loopback fixed-base URLs are accepted.
    pub fixture_mode: bool,
}
impl Default for DiscordToolsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lookup_enabled: false,
            upload_enabled: false,
            token_env: "MISSY_DISCORD_TOOLS_TOKEN".into(),
            bot_user_id: String::new(),
            allowed_guilds: vec![],
            allowed_users: vec![],
            allowed_channels: BTreeMap::new(),
            max_file_bytes: 3 * 1024 * 1024,
            api_base: API.into(),
            fixture_mode: false,
        }
    }
}

pub struct DiscordTools {
    config: DiscordToolsConfig,
    policy: Policy,
    http: PolicyHttpClient,
    // A bounded same-process duplicate barrier supplements, never replaces, the
    // runtime's durable side-effect journal. Credential never implements Debug.
    attempted: Mutex<HashSet<String>>,
}
impl std::fmt::Debug for DiscordTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscordTools")
            .field("enabled", &self.config.enabled)
            .finish_non_exhaustive()
    }
}
#[derive(Debug)]
pub struct UploadOutcomeUnknown;
impl std::fmt::Display for UploadOutcomeUnknown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Discord upload outcome UNKNOWN; do not replay this operation")
    }
}
impl std::error::Error for UploadOutcomeUnknown {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LookupArgs {
    guild_id: String,
    user_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UploadArgs {
    guild_id: String,
    channel_id: String,
    path: String,
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    caption: String,
}
fn refused(reason: &str) -> anyhow::Error {
    ToolDispatchRefused {
        reason: reason.into(),
    }
    .into()
}
fn snowflake(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 20
        || s.starts_with('0')
        || !s.bytes().all(|b| b.is_ascii_digit())
        || s.parse::<u64>().ok().filter(|n| *n > 0).is_none()
    {
        bail!("invalid Discord snowflake");
    }
    Ok(())
}
fn credential(config: &DiscordToolsConfig) -> Result<String> {
    let t = std::env::var(&config.token_env)
        .map_err(|_| refused("dedicated Discord token unavailable"))?;
    if t.len() < 8
        || t.len() > 512
        || !t
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
    {
        return Err(refused("dedicated Discord token invalid"));
    }
    Ok(t)
}
fn bounded_string(v: &Value, key: &str, max: usize) -> Option<String> {
    v.get(key)?
        .as_str()
        .filter(|s| s.len() <= max)
        .map(str::to_owned)
}
/// 25 ASCII characters, length-framed inputs bind every scope component, turn,
/// and call. Never derive nonce solely from provider-controlled call ID.
pub fn upload_nonce(ctx: &ToolContext, call_id: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"missy-discord-upload-v1");
    for s in [
        &ctx.scope.tenant,
        &ctx.scope.user,
        &ctx.scope.channel,
        &ctx.scope.session,
        &ctx.turn_id,
        &call_id.to_owned(),
    ] {
        hash.update((s.len() as u64).to_be_bytes());
        hash.update(s.as_bytes());
    }
    hex::encode(hash.finalize())[..25].into()
}
impl DiscordTools {
    pub fn new(mut policy: PolicyConfig, config: DiscordToolsConfig) -> Result<Self> {
        if !config.token_env.starts_with("MISSY_DISCORD_TOOLS_")
            || config.token_env.len() > 128
            || !config
                .token_env
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
        {
            bail!("Discord token must use a dedicated MISSY_DISCORD_TOOLS_ environment variable");
        }
        let base = Url::parse(&config.api_base).map_err(|_| anyhow!("invalid Discord API base"))?;
        if config.api_base != API
            && (!config.fixture_mode
                || base.scheme() != "http"
                || !matches!(base.host_str(), Some("127.0.0.1" | "[::1]" | "::1"))
                || base.path() != "/api/v10")
        {
            bail!("Discord API base is fixed except explicit loopback fixtures");
        }
        if !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            bail!("Discord API base cannot contain credentials, query or fragment");
        }
        if config.enabled {
            snowflake(&config.bot_user_id)?;
            if !policy
                .allowed_origins
                .contains(&base.origin().ascii_serialization())
            {
                bail!("Discord origin must be explicitly allowed by policy");
            }
        }
        if config.max_file_bytes == 0 || config.max_file_bytes > MAX_HTTP_REQUEST_BYTES - 16384 {
            bail!("Discord file limit exceeds bounded multipart capacity");
        }
        for id in config.allowed_guilds.iter().chain(&config.allowed_users) {
            snowflake(id)?;
        }
        for (c, g) in &config.allowed_channels {
            snowflake(c)?;
            snowflake(g)?;
            if !config.allowed_guilds.contains(g) {
                bail!("Discord channel guild must be allowlisted");
            }
        }
        // Bound the descriptor-based read itself, not just a subsequent Vec check.
        policy.max_file_bytes = policy.max_file_bytes.min(config.max_file_bytes);
        let policy = Policy::new(policy)?;
        Ok(Self {
            http: PolicyHttpClient::from_policy(policy.clone()),
            policy,
            config,
            attempted: Mutex::new(HashSet::new()),
        })
    }
    fn check(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        if !self.config.enabled {
            return Err(refused("Discord tools disabled"));
        }
        ctx.scope.validate().map_err(|_| refused("invalid scope"))?;
        if ctx.turn_id.is_empty()
            || ctx.turn_id.len() > 256
            || call.id.is_empty()
            || call.id.len() > 256
        {
            return Err(refused("invalid correlation identifiers"));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(refused("cancelled before dispatch"));
        }
        let guild = match call.name.as_str() {
            "discord_lookup_user" if self.config.lookup_enabled => {
                let a: LookupArgs = serde_json::from_value(call.arguments.clone())
                    .map_err(|_| refused("invalid lookup arguments"))?;
                snowflake(&a.user_id).map_err(|_| refused("invalid user ID"))?;
                if !self.config.allowed_users.contains(&a.user_id) {
                    return Err(refused("Discord user denied"));
                }
                a.guild_id
            }
            "discord_upload_file" if self.config.upload_enabled => {
                let a: UploadArgs = serde_json::from_value(call.arguments.clone())
                    .map_err(|_| refused("invalid upload arguments"))?;
                if self.config.allowed_channels.get(&a.channel_id) != Some(&a.guild_id) {
                    return Err(refused("Discord channel/guild denied"));
                }
                if a.caption.chars().count() > 2000 || a.caption.len() > 8000 {
                    return Err(refused("Discord caption exceeds 2000 characters"));
                }
                a.guild_id
            }
            _ => return Err(refused("Discord capability disabled or unknown")),
        };
        if !self.config.allowed_guilds.contains(&guild) {
            return Err(refused("Discord guild denied"));
        }
        Ok(())
    }
    async fn get(&self, ctx: &ToolContext, path: &str, token: &str) -> Result<HttpResponse> {
        let url = format!("{}{path}", self.config.api_base);
        tokio::select! { biased;
            _ = ctx.cancellation.cancelled() => bail!("Discord read cancelled"),
            r = self.http.request("GET", &url, vec![("authorization".into(), format!("Bot {token}"))], None) => r.map_err(|_| anyhow!("Discord read transport failed"))
        }
    }
    async fn lookup(&self, ctx: &ToolContext, call: &ToolCall, token: &str) -> Result<Value> {
        let a: LookupArgs = serde_json::from_value(call.arguments.clone())?;
        let r = self
            .get(
                ctx,
                &format!("/guilds/{}/members/{}", a.guild_id, a.user_id),
                token,
            )
            .await?;
        check_status(&r)?;
        let v: Value = serde_json::from_slice(&r.body)
            .map_err(|_| anyhow!("Discord lookup response invalid"))?;
        let u = &v["user"];
        if u["id"].as_str() != Some(&a.user_id) {
            bail!("Discord lookup identity mismatch");
        }
        let username = bounded_string(u, "username", 128)
            .ok_or_else(|| anyhow!("Discord username invalid"))?;
        // Explicit DTO projection: email, phone, tokens, arbitrary upstream fields
        // and headers are never exposed, even when fixtures inject them.
        Ok(
            json!({"trust":"untrusted", "data":{"guild_id":a.guild_id,"user_id":a.user_id,
            "username":username,"global_name":bounded_string(u,"global_name",128),
            "nickname":bounded_string(&v,"nick",128),"bot":u["bot"].as_bool().unwrap_or(false)}}),
        )
    }
    async fn upload(&self, ctx: &ToolContext, call: &ToolCall, token: &str) -> Result<Value> {
        let a: UploadArgs = serde_json::from_value(call.arguments.clone())?;
        let bytes = self
            .policy
            .read_file(&a.path)
            .map_err(|_| refused("Discord file policy denied"))?;
        if bytes.is_empty() || bytes.len() > self.config.max_file_bytes {
            return Err(refused("Discord file size denied"));
        }
        let raw = a
            .filename
            .as_deref()
            .or_else(|| Path::new(&a.path).file_name().and_then(|p| p.to_str()))
            .ok_or_else(|| refused("missing filename"))?;
        let filename = sanitize_filename(raw)?;
        let mime = file_mime(&filename, &bytes)?;
        if a.mime_type.as_deref().is_some_and(|m| m != mime) {
            return Err(refused("Discord MIME mismatch"));
        }
        let nonce = upload_nonce(ctx, &call.id);
        let payload = json!({"content":a.caption,"nonce":nonce,"enforce_nonce":true,
            "allowed_mentions":{"parse":[]},"attachments":[{"id":0,"filename":filename}]});
        // Deterministic unpredictable-enough boundary, explicitly collision checked.
        let boundary = format!("missy-{}", hex::encode(Sha256::digest(&bytes)));
        if bytes
            .windows(boundary.len())
            .any(|w| w == boundary.as_bytes())
        {
            return Err(refused("multipart boundary collision"));
        }
        let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\nContent-Type: application/json\r\n\r\n{}\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"files[0]\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n", payload).into_bytes();
        body.extend_from_slice(&bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        if body.len() > MAX_HTTP_REQUEST_BYTES {
            return Err(refused("Discord multipart request too large"));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(refused("cancelled before dispatch"));
        }
        {
            let mut seen = self
                .attempted
                .lock()
                .map_err(|_| refused("Discord duplicate guard unavailable"))?;
            if seen.contains(&nonce) {
                return Err(UploadOutcomeUnknown.into());
            }
            if seen.len() >= 65536 {
                return Err(refused(
                    "Discord duplicate guard full; rotate executor safely",
                ));
            }
            seen.insert(nonce.clone());
        }
        let url = format!(
            "{}/channels/{}/messages",
            self.config.api_base, a.channel_id
        );
        let started = AtomicBool::new(false);
        let request = self.http.request(
            "POST",
            &url,
            vec![
                ("authorization".into(), format!("Bot {token}")),
                (
                    "content-type".into(),
                    format!("multipart/form-data; boundary={boundary}"),
                ),
            ],
            Some(body),
        );
        tokio::pin!(request);
        let response = tokio::select! { biased;
            _ = ctx.cancellation.cancelled() => {
                if !started.load(Ordering::Relaxed) {
                    if let Ok(mut seen) = self.attempted.lock() { seen.remove(&nonce); }
                    return Err(refused("cancelled before dispatch"));
                }
                return Err(UploadOutcomeUnknown.into());
            },
            r = std::future::poll_fn(|cx| {
                started.store(true, Ordering::Relaxed);
                request.as_mut().poll(cx)
            }) => r
        };
        if let Ok(r) = response {
            if r.status == 429 {
                return Err(rate_limit(&r));
            }
            if (200..300).contains(&r.status)
                && let Ok(v) = serde_json::from_slice::<Value>(&r.body)
                && let Ok(out) =
                    self.validate_message(&v, &a.channel_id, &nonce, &filename, bytes.len())
            {
                return Ok(out);
            }
        }
        // A failed POST is never repeated. Bounded read-only reconciliation can
        // prove delivery, but absence is NOT proof of no effect.
        if !ctx.cancellation.is_cancelled()
            && let Ok(r) = self
                .get(
                    ctx,
                    &format!("/channels/{}/messages?limit=25", a.channel_id),
                    token,
                )
                .await
            && r.status == 200
            && let Ok(Value::Array(messages)) = serde_json::from_slice(&r.body)
            && messages.len() <= 25
        {
            for v in messages {
                if let Ok(out) =
                    self.validate_message(&v, &a.channel_id, &nonce, &filename, bytes.len())
                {
                    return Ok(out);
                }
            }
        }
        Err(UploadOutcomeUnknown.into())
    }
    fn validate_message(
        &self,
        v: &Value,
        channel: &str,
        nonce: &str,
        filename: &str,
        size: usize,
    ) -> Result<Value> {
        let id = v["id"]
            .as_str()
            .ok_or_else(|| anyhow!("missing message ID"))?;
        snowflake(id)?;
        if v["channel_id"].as_str() != Some(channel)
            || v["author"]["id"].as_str() != Some(&self.config.bot_user_id)
            || v["author"]["bot"].as_bool() != Some(true)
            || v["nonce"].as_str() != Some(nonce)
        {
            bail!("Discord response correlation mismatch");
        }
        let attachments = v["attachments"]
            .as_array()
            .filter(|a| a.len() == 1)
            .ok_or_else(|| anyhow!("invalid attachments"))?;
        let a = &attachments[0];
        snowflake(
            a["id"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid attachment ID"))?,
        )?;
        if a["filename"].as_str() != Some(filename) || a["size"].as_u64() != Some(size as u64) {
            bail!("Discord attachment mismatch");
        }
        let raw_url = a["url"]
            .as_str()
            .filter(|s| s.len() <= 2048)
            .ok_or_else(|| anyhow!("invalid attachment URL"))?;
        let url = Url::parse(raw_url)?;
        if url.scheme() != "https"
            || !matches!(
                url.host_str(),
                Some("cdn.discordapp.com" | "media.discordapp.net")
            )
            || !url.username().is_empty()
            || url.password().is_some()
            || url.port().is_some()
            || url.fragment().is_some()
            || !url.path().starts_with(&format!("/attachments/{channel}/"))
        {
            bail!("Discord attachment URL denied");
        }
        // CDN URL is returned as data, NEVER fetched. Signed query is necessary
        // for download and is not an authentication credential for our bot.
        Ok(
            json!({"trust":"untrusted","data":{"message_id":id,"channel_id":channel,"nonce":nonce,
            "attachments":[{"id":a["id"],"filename":filename,"size":size,"url":raw_url}]}}),
        )
    }
}
fn rate_limit(r: &HttpResponse) -> anyhow::Error {
    let retry = serde_json::from_slice::<Value>(&r.body)
        .ok()
        .and_then(|v| v["retry_after"].as_f64())
        .filter(|n| n.is_finite() && *n >= 0.0 && *n <= 86400.0);
    anyhow!("Discord rate limited (429); retry_after_seconds={retry:?}; no automatic retry")
}
fn check_status(r: &HttpResponse) -> Result<()> {
    if r.status == 429 {
        return Err(rate_limit(r));
    }
    if !(200..300).contains(&r.status) {
        bail!("Discord read returned status {}", r.status);
    }
    Ok(())
}
fn sanitize_filename(raw: &str) -> Result<String> {
    let leaf = raw.rsplit(['/', '\\']).next().unwrap_or_default();
    if leaf.is_empty() || leaf.len() > 120 || leaf.starts_with('.') {
        return Err(refused("invalid upload filename"));
    }
    let s: String = leaf
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    Ok(s)
}
fn file_mime(name: &str, b: &[u8]) -> Result<&'static str> {
    let ext = name
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mime = match ext.as_str() {
        "png" if b.starts_with(b"\x89PNG\r\n\x1a\n") => "image/png",
        "jpg" | "jpeg" if b.starts_with(b"\xff\xd8\xff") => "image/jpeg",
        "gif" if b.starts_with(b"GIF87a") || b.starts_with(b"GIF89a") => "image/gif",
        "webp" if b.starts_with(b"RIFF") && b.get(8..12) == Some(b"WEBP") => "image/webp",
        "wav" if b.starts_with(b"RIFF") && b.get(8..12) == Some(b"WAVE") => "audio/wav",
        "pdf" if b.starts_with(b"%PDF-") => "application/pdf",
        "zip" if b.starts_with(b"PK\x03\x04") || b.starts_with(b"PK\x05\x06") => "application/zip",
        "mp4" if b.get(4..8) == Some(b"ftyp") => "video/mp4",
        _ => return Err(refused("unsupported file extension or magic mismatch")),
    };
    Ok(mime)
}
#[async_trait]
impl ToolExecutor for DiscordTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut out = vec![];
        if !self.config.enabled {
            return out;
        }
        if self.config.lookup_enabled {
            out.push(ToolDefinition {name:"discord_lookup_user".into(),description:"Read an exact allowlisted guild member; safe public profile fields only.".into(),side_effect:false,
            parameters:json!({"type":"object","additionalProperties":false,"required":["guild_id","user_id"],"properties":{"guild_id":{"type":"string"},"user_id":{"type":"string"}}})});
        }
        if self.config.upload_enabled {
            out.push(ToolDefinition {name:"discord_upload_file".into(),description:"Upload one policy-readable binary to an explicitly allowed guild/channel. Requires durable scoped approval. UNKNOWN must never be replayed.".into(),side_effect:true,
            parameters:json!({"type":"object","additionalProperties":false,"required":["guild_id","channel_id","path"],"properties":{"guild_id":{"type":"string"},"channel_id":{"type":"string"},"path":{"type":"string"},"filename":{"type":"string"},"mime_type":{"type":"string"},"caption":{"type":"string","maxLength":2000}}})});
        }
        out
    }
    fn configuration_fingerprint(&self) -> String {
        let mut h = Sha256::new();
        h.update(serde_json::to_vec(&self.config).unwrap_or_default());
        h.update(serde_json::to_vec(self.policy.config()).unwrap_or_default());
        // Rotation invalidates existing capability approvals without disclosing token.
        match credential(&self.config) {
            Ok(t) => h.update(Sha256::digest(t.as_bytes())),
            Err(_) => h.update(b"missing-credential"),
        }
        hex::encode(h.finalize())
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        self.check(ctx, call)?;
        let _ = credential(&self.config)?;
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.check(ctx, call)?;
        let token = credential(&self.config)?;
        let mut output = match call.name.as_str() {
            "discord_lookup_user" => self.lookup(ctx, call, &token).await,
            "discord_upload_file" => self.upload(ctx, call, &token).await,
            _ => Err(refused("unknown Discord tool")),
        }?;
        redact(&mut output, &token);
        Ok(output)
    }
}
fn redact(value: &mut Value, token: &str) {
    match value {
        Value::String(s) => *s = s.replace(token, "[REDACTED]"),
        Value::Array(values) => {
            for v in values {
                redact(v, token);
            }
        }
        Value::Object(values) => {
            for v in values.values_mut() {
                redact(v, token);
            }
        }
        _ => (),
    }
}
