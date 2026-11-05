//! Opt-in Discord transport. Every outbound path is policy-bound.
mod journal;
mod voice;
pub use voice::{DiscordGatewayVoice, DiscordVoiceGatewayHook, discord_voice_bridge};
#[cfg(test)]
mod tests;
use anyhow::{Result, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use journal::Journal;
use missy_core::{DiscordConfig, PolicyConfig, Scope};
use missy_policy::PolicyHttpClient;
use missy_runtime::Runtime;
use missy_store::TurnStatus;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc, time::Duration};
use tokio_tungstenite::tungstenite::Message;
const REST: &str = "https://discord.com/api/v10";
const MAX_CHUNKS: usize = 16;

/// No credentials, session IDs, Gateway URLs or raw JSON are returned.
#[derive(Debug, Clone, Serialize)]
pub struct DiscordProbe {
    pub bot_id: String,
    pub username: String,
    pub session_start_total: u64,
    pub session_start_remaining: u64,
    pub session_start_reset_after_ms: u64,
    pub max_concurrency: u64,
}
/// Read-only authentication/policy preflight. No Gateway session, command
/// registration, messages, mutations or config-file reads.
pub async fn probe_discord(config: DiscordConfig, policy: PolicyConfig) -> Result<DiscordProbe> {
    validate_config(&config)?;
    probe_at(&PolicyHttpClient::new(policy)?, &token(&config)?, REST).await
}
async fn probe_at(http: &PolicyHttpClient, token: &str, rest: &str) -> Result<DiscordProbe> {
    let me = get_json(http, token, &format!("{rest}/users/@me")).await?;
    ensure!(
        me["bot"] == true,
        "Discord credential is not a bot credential"
    );
    let bot_id = me["id"]
        .as_str()
        .filter(|id| snowflake(id))
        .ok_or_else(|| anyhow::anyhow!("invalid Discord bot identity"))?
        .to_owned();
    let username = me["username"]
        .as_str()
        .filter(|s| !s.is_empty() && s.len() <= 128)
        .ok_or_else(|| anyhow::anyhow!("invalid Discord bot username"))?
        .to_owned();
    let gateway = get_json(http, token, &format!("{rest}/gateway/bot")).await?;
    let limits = &gateway["session_start_limit"];
    let number = |key: &str| -> Result<u64> {
        limits[key]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("invalid Discord session limits"))
    };
    Ok(DiscordProbe {
        bot_id,
        username,
        session_start_total: number("total")?,
        session_start_remaining: number("remaining")?,
        session_start_reset_after_ms: number("reset_after")?,
        max_concurrency: number("max_concurrency")?,
    })
}
#[derive(Default, Serialize, Deserialize)]
struct Resume {
    session: Option<String>,
    sequence: Option<u64>,
    url: Option<String>,
    bot: String,
}
/// Legacy signature retained, but never silently opens an implicit live path.
pub async fn serve_discord(_: Arc<Runtime>, _: DiscordConfig, _: PolicyConfig) -> Result<()> {
    bail!("Discord requires an explicit private state directory; use serve_discord_with_state_dir")
}
pub async fn serve_discord_with_state_dir(
    runtime: Arc<Runtime>,
    config: DiscordConfig,
    policy: PolicyConfig,
    state_dir: impl AsRef<Path>,
) -> Result<()> {
    serve_inner(runtime, config, policy, state_dir, None).await
}
pub async fn serve_discord_with_voice(
    runtime: Arc<Runtime>,
    config: DiscordConfig,
    policy: PolicyConfig,
    state_dir: impl AsRef<Path>,
    voice: DiscordVoiceGatewayHook,
) -> Result<()> {
    serve_inner(runtime, config, policy, state_dir, Some(voice)).await
}
async fn serve_inner(
    runtime: Arc<Runtime>,
    config: DiscordConfig,
    policy: PolicyConfig,
    state_dir: impl AsRef<Path>,
    mut voice: Option<DiscordVoiceGatewayHook>,
) -> Result<()> {
    ensure!(config.enabled, "Discord is not enabled");
    validate_config(&config)?;
    let token = token(&config)?;
    let http = Arc::new(PolicyHttpClient::new(policy)?);
    let journal = Arc::new(Journal::open(state_dir.as_ref())?);
    ensure!(
        !journal.has_uncertain()?,
        "Discord delivery outcome uncertain; inspect private discord.sqlite outbox before restarting; never blindly replay"
    );
    let identity = probe_at(&http, &token, REST).await?;
    ensure!(
        identity.session_start_remaining > 0,
        "Discord session-start budget exhausted; wait for reset before starting"
    );
    journal.bind_bot(&identity.bot_id)?;
    journal.session_budget(
        identity.session_start_remaining,
        identity.session_start_reset_after_ms,
    )?;
    let mut resume = journal.resume()?;
    resume.bot = identity.bot_id;
    let discovery = get_json(&http, &token, &format!("{REST}/gateway/bot")).await?;
    let base = discovery["url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Discord discovery returned no Gateway URL"))?
        .to_owned();
    let mut workers = tokio::task::JoinSet::new();
    workers.spawn(execution_worker(runtime, config.clone(), journal.clone()));
    workers.spawn(delivery_worker(
        http.clone(),
        token.clone(),
        config.clone(),
        journal.clone(),
        REST.into(),
        resume.bot.clone(),
    ));
    workers.spawn(interaction_ack_worker(
        http.clone(),
        config.clone(),
        journal.clone(),
        REST.into(),
    ));
    let mut retry = 1u64;
    loop {
        let attempt = tokio::select! {
            result = connection(&http, &token, &config, &base, &mut resume, &journal, &mut voice) => result,
            stopped = workers.join_next() => {
                if let Some(Ok(Err(error))) = stopped { return Err(error); }
                bail!("Discord durable worker stopped; intake stopped");
            },
        };
        if let Err(error) = attempt
            && error.downcast_ref::<FatalClose>().is_some()
        {
            bail!("Discord Gateway rejected authentication/intents; channel stopped");
        }
        if let Some(voice) = &mut voice {
            voice.reset().await;
        }
        // No raw transport error logging: errors can contain secret URL paths.
        tokio::time::sleep(Duration::from_secs(retry)).await;
        retry = retry.saturating_mul(2).min(60);
    }
}
fn token(config: &DiscordConfig) -> Result<String> {
    ensure!(
        config.token_env.starts_with("MISSY_")
            && config
                .token_env
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
        "Discord requires an explicitly dedicated MISSY_* credential variable; shared bot environment variables are refused"
    );
    let token = std::env::var(&config.token_env)
        .map_err(|_| anyhow::anyhow!("Discord token environment variable missing"))?;
    ensure!(
        (32..=256).contains(&token.len()) && token.bytes().all(|b| b.is_ascii_graphic()),
        "invalid Discord token"
    );
    Ok(token)
}
fn auth(token: &str) -> Vec<(String, String)> {
    vec![("Authorization".into(), format!("Bot {token}"))]
}
fn snowflake(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 20
        && value.bytes().all(|b| b.is_ascii_digit())
        && value.parse::<u64>().is_ok_and(|v| v > 0)
}
pub fn validate_config(config: &DiscordConfig) -> Result<()> {
    ensure!(
        config.require_mention,
        "Discord requires mention protection"
    );
    ensure!(
        config.allowed_users.len() <= 10000
            && config.allowed_channels.len() <= 10000
            && config.allowed_guilds.len() <= 10000,
        "Discord allowlist too large"
    );
    for id in config
        .allowed_users
        .iter()
        .chain(&config.allowed_channels)
        .chain(&config.allowed_guilds)
    {
        ensure!(snowflake(id), "Discord allowlists must contain numeric IDs");
    }
    ensure!(
        !config.allowed_users.is_empty(),
        "Discord requires an explicit user allowlist"
    );
    ensure!(
        config.allow_dm
            || (!config.allowed_guilds.is_empty() && !config.allowed_channels.is_empty()),
        "Discord requires guild/channel allowlists or DM opt-in"
    );
    Ok(())
}
#[derive(Debug)]
struct FatalClose;
impl std::fmt::Display for FatalClose {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("fatal Discord close")
    }
}
impl std::error::Error for FatalClose {}
async fn connection(
    http: &PolicyHttpClient,
    token: &str,
    config: &DiscordConfig,
    base: &str,
    resume: &mut Resume,
    journal: &Journal,
    voice: &mut Option<DiscordVoiceGatewayHook>,
) -> Result<()> {
    let mut url = url::Url::parse(resume.url.as_deref().unwrap_or(base))?;
    ensure!(url.scheme() == "wss", "Discord Gateway must use TLS");
    url.set_query(Some("v=10&encoding=json"));
    let ws = http
        .connect_websocket(url.as_str(), vec![])
        .await
        .map_err(|_| anyhow::anyhow!("Discord Gateway connection failed"))?;
    gateway_session_inner(ws, token, config, resume, journal, voice).await
}
#[cfg(test)]
async fn gateway_session(
    ws: missy_policy::PolicyWebSocket,
    token: &str,
    config: &DiscordConfig,
    resume: &mut Resume,
    journal: &Journal,
) -> Result<()> {
    gateway_session_inner(ws, token, config, resume, journal, &mut None).await
}
async fn gateway_session_inner(
    mut ws: missy_policy::PolicyWebSocket,
    token: &str,
    config: &DiscordConfig,
    resume: &mut Resume,
    journal: &Journal,
    voice: &mut Option<DiscordVoiceGatewayHook>,
) -> Result<()> {
    let hello = tokio::time::timeout(Duration::from_secs(30), ws.next())
        .await?
        .ok_or_else(|| anyhow::anyhow!("no Gateway hello"))??;
    let hello: Value = serde_json::from_str(hello.to_text()?)?;
    ensure!(hello["op"] == 10, "invalid Gateway hello");
    let millis = hello["d"]["heartbeat_interval"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("invalid heartbeat"))?;
    ensure!(
        (1000..=120_000).contains(&millis),
        "heartbeat outside bounds"
    );
    let identify = if let (Some(session), Some(sequence)) = (&resume.session, resume.sequence) {
        json!({"op":6,"d":{"token":token,"session_id":session,"seq":sequence}})
    } else {
        journal.reserve_identify(journal::now_ms())?;
        json!({"op":2,"d":{"token":token,"intents":if voice.is_some() {37377 | 128} else {37377},"properties":{"os":"linux","browser":"missy-rust","device":"missy-rust"}}})
    };
    send_frame(&mut ws, Message::Text(identify.to_string().into())).await?;
    let mut timer = tokio::time::interval(Duration::from_millis(millis));
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut awaiting_ack = false;
    loop {
        tokio::select! {
            command = async { match voice { Some(hook) => hook.commands.recv().await, None => std::future::pending().await } } => {
                if let Some(command) = command { voice.as_mut().expect("voice hook").command(&mut ws, command).await?; }
                else { *voice = None; }
            }
            _ = timer.tick() => {
                ensure!(!awaiting_ack, "Gateway heartbeat ACK deadline exceeded");
                send_frame(&mut ws, Message::Text(json!({"op":1,"d":resume.sequence}).to_string().into())).await?; awaiting_ack = true;
            }
            frame = ws.next() => {
                let frame = frame.ok_or_else(|| anyhow::anyhow!("Gateway disconnected"))??;
                if let Message::Close(close) = frame {
                    if let Some(close) = close { let code: u16 = close.code.into();
                        if matches!(code,4004|4010|4011|4012|4013|4014) { return Err(FatalClose.into()); }
                        if matches!(code,4007|4009) { reset_resume(resume, journal)?; }
                    } bail!("Gateway closed");
                }
                if let Message::Ping(payload) = frame { send_frame(&mut ws,Message::Pong(payload)).await?; continue; }
                if !frame.is_text() { continue; }
                ensure!(frame.len() <= 128 * 1024, "Gateway event too large");
                let event: Value = serde_json::from_str(frame.to_text()?)?;
                match event["op"].as_u64() {
                    Some(11) => awaiting_ack = false,
                    Some(1) => { send_frame(&mut ws,Message::Text(json!({"op":1,"d":resume.sequence}).to_string().into())).await?; }
                    Some(7) => bail!("Gateway requested reconnect"),
                    Some(9) => { if event["d"] != true { reset_resume(resume, journal)?; } bail!("Gateway invalid session"); }
                    Some(0) => {
                        let mut next = Resume { session: resume.session.clone(), sequence: resume.sequence, url: resume.url.clone(), bot: resume.bot.clone() };
                        if let Some(sequence) = event["s"].as_u64() { next.sequence = Some(sequence); }
                        if let Some(voice) = voice { voice.event(&event, &resume.bot).await?; }
                        let incoming = match event["t"].as_str() {
                            Some("READY") => {
                                ensure!(event["d"]["user"]["id"].as_str() == Some(resume.bot.as_str()), "Gateway bot identity mismatch");
                                next.session = event["d"]["session_id"].as_str().filter(|s| s.len() <= 256).map(str::to_owned);
                                next.url = event["d"]["resume_gateway_url"].as_str().filter(|s| s.len() <= 2048).map(str::to_owned); None
                            },
                            Some("MESSAGE_CREATE") => incoming(config, &resume.bot, &event["d"]),
                            Some("INTERACTION_CREATE") => interaction(config, &resume.bot, &event["d"]),
                            _ => None,
                        };
                        // Commit intake + sequence atomically BEFORE advancing heartbeats.
                        // Queue saturation reconnects without acknowledging this event.
                        journal.intake(incoming.as_ref(), &next, journal::now_ms())?; *resume = next;
                    }, _ => {}
                }
            }
        }
    }
}
fn reset_resume(resume: &mut Resume, journal: &Journal) -> Result<()> {
    *resume = Resume {
        bot: resume.bot.clone(),
        ..Resume::default()
    };
    journal.intake(None, resume, journal::now_ms())
}
async fn send_frame(ws: &mut missy_policy::PolicyWebSocket, message: Message) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), ws.send(message)).await??;
    Ok(())
}
#[derive(Clone, Serialize, Deserialize)]
enum Action {
    Ask(String),
    Status(String),
    Cancel(String),
    Resume(String),
}
#[derive(Clone, Serialize, Deserialize)]
struct Interaction {
    id: String,
    application: String,
    token: String,
}
#[derive(Clone, Serialize, Deserialize)]
struct Incoming {
    scope: Scope,
    request_id: String,
    channel: String,
    action: Action,
    interaction: Option<Interaction>,
}
fn scope(config: &DiscordConfig, user: &str, channel: &str, guild: Option<&str>) -> Option<Scope> {
    if ![user, channel].iter().all(|s| snowflake(s))
        || !config.allowed_users.iter().any(|s| s == user)
    {
        return None;
    }
    let tenant = match guild {
        Some(g)
            if snowflake(g)
                && config.allowed_guilds.iter().any(|s| s == g)
                && config.allowed_channels.iter().any(|s| s == channel) =>
        {
            format!("discord:g:{g}")
        }
        None if config.allow_dm => "discord:dm".into(),
        _ => return None,
    };
    let scope = Scope {
        tenant,
        user: format!("discord:{user}"),
        channel: format!("discord:{channel}"),
        session: format!("channel:{channel}:user:{user}"),
    };
    scope.validate().ok()?;
    Some(scope)
}
fn still_allowed(config: &DiscordConfig, message: &Incoming) -> bool {
    let Some(user) = message.scope.user.strip_prefix("discord:") else {
        return false;
    };
    let guild = message.scope.tenant.strip_prefix("discord:g:");
    (guild.is_some() || message.scope.tenant == "discord:dm")
        && scope(config, user, &message.channel, guild).as_ref() == Some(&message.scope)
}
fn incoming(config: &DiscordConfig, bot: &str, value: &Value) -> Option<Incoming> {
    if value["author"]["bot"] == true || !value["webhook_id"].is_null() || !snowflake(bot) {
        return None;
    }
    let id = value["id"].as_str().filter(|s| snowflake(s))?;
    let channel = value["channel_id"].as_str()?;
    let scope = scope(
        config,
        value["author"]["id"].as_str()?,
        channel,
        guild_field(value)?,
    )?;
    if !value["mentions"]
        .as_array()?
        .iter()
        .any(|m| m["id"].as_str() == Some(bot))
    {
        return None;
    }
    let content = value["content"].as_str()?;
    if content.is_empty() || content.len() > super::MAX_PROMPT_BYTES {
        return None;
    }
    let mut prompt = content.to_owned();
    if !value["attachments"].is_null() && !value["attachments"].is_array() {
        return None;
    }
    if let Some(attachments) = value["attachments"].as_array() {
        if attachments.len() > 10 {
            return None;
        }
        let mut metadata = Vec::new();
        let mut total = 0u64;
        for a in attachments {
            let id = a["id"].as_str().filter(|s| snowflake(s))?;
            let name = a["filename"].as_str().filter(|s| s.len() <= 256)?;
            let size = a["size"].as_u64().filter(|n| *n <= 25 * 1024 * 1024)?;
            total = total.checked_add(size)?;
            if total > 50 * 1024 * 1024 {
                return None;
            }
            // No URL, proxy URL or attacker-supplied MIME forwarding/fetching.
            metadata.push(json!({"id":id,"filename":name,"size":size}));
        }
        if !metadata.is_empty() {
            prompt.push_str(&format!(
                "\n[Untrusted attachment metadata only; contents not fetched or available: {}]",
                serde_json::to_string(&metadata).ok()?
            ));
        }
    }
    if prompt.len() > super::MAX_PROMPT_BYTES {
        return None;
    }
    Some(Incoming {
        scope,
        request_id: format!("discord:{id}"),
        channel: channel.into(),
        action: Action::Ask(prompt),
        interaction: None,
    })
}
/// Only called for authenticated INTERACTION_CREATE on the policy-pinned Gateway.
/// No public HTTP interaction handler exists to bypass Gateway authentication.
fn interaction(config: &DiscordConfig, bot: &str, value: &Value) -> Option<Incoming> {
    if !snowflake(bot)
        || value["type"] != 2
        || value["application_id"].as_str()? != bot
        || value["data"]["name"] != "missy"
    {
        return None;
    }
    let guild = guild_field(value)?;
    let author = if guild.is_some() {
        &value["member"]["user"]
    } else {
        &value["user"]
    };
    if author["bot"] == true {
        return None;
    }
    let channel = value["channel_id"].as_str()?;
    let scope = scope(config, author["id"].as_str()?, channel, guild)?;
    let id = value["id"].as_str().filter(|s| snowflake(s))?;
    let token = value["token"].as_str().filter(|s| {
        (16..=512).contains(&s.len())
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    })?;
    let options = value["data"]["options"].as_array()?;
    if options.len() != 1 || options[0]["type"] != 1 {
        return None;
    }
    let command = options[0]["name"].as_str()?;
    let args = options[0]["options"].as_array()?;
    if args.len() != 1 || args[0]["type"] != 3 {
        return None;
    }
    let text = args[0]["value"].as_str()?;
    let action = match command {
        "ask"
            if args[0]["name"] == "prompt"
                && !text.trim().is_empty()
                && text.len() <= super::MAX_PROMPT_BYTES =>
        {
            Action::Ask(text.into())
        }
        "status" | "cancel" | "resume"
            if args[0]["name"] == "turn"
                && !text.is_empty()
                && text.len() <= 128
                && text
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)) =>
        {
            match command {
                "status" => Action::Status(text.into()),
                "cancel" => Action::Cancel(text.into()),
                _ => Action::Resume(text.into()),
            }
        }
        _ => return None,
    };
    Some(Incoming {
        scope,
        request_id: format!("discord:i:{id}"),
        channel: channel.into(),
        action,
        interaction: Some(Interaction {
            id: id.into(),
            application: bot.into(),
            token: token.into(),
        }),
    })
}
fn guild_field(value: &Value) -> Option<Option<&str>> {
    if value["guild_id"].is_null() {
        Some(None)
    } else {
        value["guild_id"].as_str().map(Some)
    }
}
async fn execution_worker(
    runtime: Arc<Runtime>,
    config: DiscordConfig,
    journal: Arc<Journal>,
) -> Result<()> {
    let mut tasks = tokio::task::JoinSet::new();
    let mut active = std::collections::HashSet::new();
    loop {
        while let Some(task) = tasks.try_join_next() {
            let (id, result) =
                task.map_err(|_| anyhow::anyhow!("Discord execution worker failed"))?;
            active.remove(&id);
            result?;
        }
        if let Some(message) = journal.pending()? {
            if !still_allowed(&config, &message) {
                journal.block(&message.request_id)?;
                continue;
            }
            if active.contains(&message.request_id) {
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            }
            // Control operations remain responsive even while all execution slots
            // are occupied. Long asks/resumes have a separate bounded task set.
            if matches!(message.action, Action::Ask(_) | Action::Resume(_))
                && !journal.rate_denied(&message.request_id)?
            {
                if active.len() < 8 {
                    active.insert(message.request_id.clone());
                    let rt = runtime.clone();
                    let j = journal.clone();
                    tasks.spawn(async move {
                        let result = async {
                            if let Some(text) = execute(&rt, &message, &j).await? {
                                j.result(&message, &text)?;
                            }
                            Ok::<(), anyhow::Error>(())
                        }
                        .await;
                        (message.request_id, result)
                    });
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
                continue;
            }
            let text = if journal.rate_denied(&message.request_id)? {
                Some(
                    "Principal rate limit exceeded (30 requests per minute). No turn was executed."
                        .into(),
                )
            } else {
                execute(&runtime, &message, &journal).await?
            };
            if let Some(text) = text {
                journal.result(&message, &text)?;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
async fn execute(
    runtime: &Runtime,
    message: &Incoming,
    journal: &Journal,
) -> Result<Option<String>> {
    match &message.action {
        Action::Ask(prompt) => {
            let admission = runtime
                .store()
                .admit(&message.scope, &message.request_id, prompt)?;
            if admission.turn.status == TurnStatus::Admitted {
                // Admitted turns have not executed a tool. Running/interrupted work
                // is NEVER implicitly resumed after a crash.
                let _ = runtime
                    .resume(message.scope.clone(), &admission.turn.id)
                    .await;
            }
            turn_result(runtime, message, &admission.turn.id)
        }
        Action::Status(id) => Ok(Some(match runtime.status(&message.scope, id) {
            Ok(turn) => format!("Turn {}: {:?}", turn.id, turn.status),
            Err(_) => "Turn not found in this principal/channel scope.".into(),
        })),
        Action::Cancel(id) => Ok(Some(if runtime.cancel(&message.scope, id).is_ok() {
            format!("Cancellation requested for turn {id}; inspect status to confirm termination.")
        } else {
            "Turn not found or not cancellable in this scope.".into()
        })),
        Action::Resume(id) => {
            runtime.store().recover(&message.scope)?;
            let Ok(turn) = runtime.status(&message.scope, id) else {
                return Ok(Some("Turn not found in this scope.".into()));
            };
            if matches!(
                turn.status,
                TurnStatus::Completed | TurnStatus::Failed | TurnStatus::Cancelled
            ) {
                return turn_result(runtime, message, id);
            }
            if !journal.claim_resume(&message.request_id)? {
                if turn.status == TurnStatus::Admitted {
                    return Ok(Some(format!(
                        "Resume attempt for turn {id} was interrupted before execution. Submit a new explicit /missy resume in this scope; the original occurrence will not be replayed."
                    )));
                }
                return turn_result(runtime, message, id);
            }
            if let Ok(result) = runtime.resume(message.scope.clone(), id).await {
                return Ok(Some(format!(
                    "Turn {}: {}\n{}",
                    result.turn_id, result.status, result.content
                )));
            }
            Ok(Some("Resume not performed. Inspect scoped status; an active lease, capacity, terminal state or unresolved tool outcome can block resume.".into()))
        }
    }
}
fn turn_result(runtime: &Runtime, message: &Incoming, id: &str) -> Result<Option<String>> {
    runtime.store().recover(&message.scope)?;
    let turn = runtime.status(&message.scope, id)?;
    Ok(match turn.status {
        TurnStatus::Running | TurnStatus::Admitted => None,
        TurnStatus::Completed => Some(format!(
            "{}\n[Turn {}]",
            turn.response
                .as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or("Turn finished without text."),
            id
        )),
        TurnStatus::Interrupted => Some(format!(
            "Turn {id} interrupted. Use /missy status or explicit /missy resume in this same user/channel scope. No automatic replay was attempted."
        )),
        TurnStatus::Failed | TurnStatus::Cancelled => Some(format!(
            "Turn {id}: {:?}. Inspect scoped history for details.",
            turn.status
        )),
    })
}
fn chunks(text: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut chunk = String::new();
    let mut size = 0;
    for c in text.chars() {
        if size + c.len_utf16() > 1900 {
            output.push(std::mem::take(&mut chunk));
            size = 0;
            if output.len() == MAX_CHUNKS - 1 {
                output.push("[Response truncated at transport limit; inspect scoped history for the full result.]".into());
                return output;
            }
        }
        chunk.push(c);
        size += c.len_utf16();
    }
    if !chunk.is_empty() {
        output.push(chunk);
    }
    output
}
fn nonce(request: &str, index: usize) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!("missy-discord-v1:{request}:{index}").as_bytes())
    )[..24]
        .to_owned()
}
async fn get_json(http: &PolicyHttpClient, token: &str, url: &str) -> Result<Value> {
    let response = http
        .request("GET", url, auth(token), None)
        .await
        .map_err(|_| anyhow::anyhow!("Discord policy-bound HTTP request failed"))?;
    ensure!(
        (200..300).contains(&response.status),
        "Discord HTTP authentication/policy/permission request failed"
    );
    serde_json::from_slice(&response.body).map_err(|_| anyhow::anyhow!("invalid Discord HTTP JSON"))
}
async fn delivery_worker(
    http: Arc<PolicyHttpClient>,
    token: String,
    config: DiscordConfig,
    journal: Arc<Journal>,
    rest: String,
    bot: String,
) -> Result<()> {
    loop {
        ensure!(
            !journal.has_uncertain()?,
            "Discord delivery outcome uncertain; intake stopped. Inspect private discord.sqlite outbox before any retry; never blindly replay"
        );
        if let Some(delivery) = journal.next_delivery()? {
            if !still_allowed(&config, &delivery.message) {
                journal.block(&delivery.message.request_id)?;
                continue;
            }
            deliver(&http, &token, &rest, &bot, &journal, delivery).await?;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
async fn deliver(
    http: &PolicyHttpClient,
    token: &str,
    rest: &str,
    bot: &str,
    journal: &Journal,
    delivery: journal::Delivery,
) -> Result<()> {
    if let Some(interaction) = &delivery.message.interaction {
        return deliver_interaction(http, rest, journal, &delivery, interaction).await;
    }
    let channel = &delivery.message.channel;
    if delivery.state == "sending" {
        // Discord nonce retention is finite. Never blindly POST after an unknown
        // result, even on restart. Bounded read reconciliation or explicit review.
        let found = get_json(
            http,
            token,
            &format!("{rest}/channels/{channel}/messages?limit=100"),
        )
        .await;
        if let Ok(messages) = found
            && let Some(messages) = messages.as_array()
        {
            for message in messages {
                if message["author"]["id"].as_str() == Some(bot)
                    && message["channel_id"].as_str() == Some(channel.as_str())
                    && message["nonce"].as_str() == Some(delivery.nonce.as_str())
                    && message["content"].as_str() == Some(delivery.text.as_str())
                    && let Some(id) = message["id"].as_str().filter(|s| snowflake(s))
                {
                    return journal.receipt(&delivery.nonce, id);
                }
            }
        }
        return journal.uncertain(&delivery.nonce);
    }
    journal.sending(&delivery.nonce)?;
    let payload = json!({"content":delivery.text,"nonce":delivery.nonce,"enforce_nonce":true,"allowed_mentions":{"parse":[],"replied_user":false}});
    let mut headers = auth(token);
    headers.push(("Content-Type".into(), "application/json".into()));
    let response = http
        .request(
            "POST",
            &format!("{rest}/channels/{channel}/messages"),
            headers,
            Some(serde_json::to_vec(&payload)?),
        )
        .await;
    if let Ok(response) = response {
        if response.status == 429 {
            let seconds = serde_json::from_slice::<Value>(&response.body)
                .ok()
                .and_then(|v| v["retry_after"].as_f64())
                .unwrap_or(1.0);
            if seconds.is_finite() && (0.0..=60.0).contains(&seconds) {
                journal.retry(
                    &delivery.nonce,
                    journal::now_ms().saturating_add((seconds.max(0.5) * 1000.0) as i64),
                )?;
                return Ok(());
            }
        }
        if (200..300).contains(&response.status)
            && let Ok(value) = serde_json::from_slice::<Value>(&response.body)
            && value["channel_id"].as_str() == Some(channel.as_str())
            && value["author"]["id"].as_str() == Some(bot)
            && value["nonce"].as_str() == Some(delivery.nonce.as_str())
            && let Some(id) = value["id"].as_str().filter(|s| snowflake(s))
        {
            return journal.receipt(&delivery.nonce, id);
        }
    }
    Ok(()) // persisted sending: one read reconciliation on next pass/restart
}
async fn interaction_ack_worker(
    http: Arc<PolicyHttpClient>,
    config: DiscordConfig,
    journal: Arc<Journal>,
    rest: String,
) -> Result<()> {
    loop {
        if let Some(message) = journal.pending_ack()? {
            if !still_allowed(&config, &message) {
                journal.block(&message.request_id)?;
                continue;
            }
            let i = message
                .interaction
                .as_ref()
                .expect("journal interaction filter");
            // Persist BEFORE callback. Ambiguity is safe: callback identity is unique,
            // and already-acknowledged responses never cause a followup message.
            journal.acking(&message.request_id)?;
            let _ = http
                .request(
                    "POST",
                    &format!("{rest}/interactions/{}/{}/callback", i.id, i.token),
                    vec![("Content-Type".into(), "application/json".into())],
                    Some(serde_json::to_vec(&json!({"type":5,"data":{"flags":64}}))?),
                )
                .await;
            journal.acked(&message.request_id)?;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
async fn deliver_interaction(
    http: &PolicyHttpClient,
    rest: &str,
    journal: &Journal,
    delivery: &journal::Delivery,
    interaction: &Interaction,
) -> Result<()> {
    journal.sending(&delivery.nonce)?;
    // Original response PATCH is idempotent. Never public fallback/followup POST.
    let original = format!(
        "{rest}/webhooks/{}/{}/messages/@original",
        interaction.application, interaction.token
    );
    let headers = vec![("Content-Type".into(), "application/json".into())];
    let response = http.request("PATCH", &original, headers, Some(serde_json::to_vec(&json!({"content":delivery.text,"allowed_mentions":{"parse":[],"replied_user":false}}))?)).await;
    if let Ok(response) = response
        && (200..300).contains(&response.status)
        && let Ok(value) = serde_json::from_slice::<Value>(&response.body)
        && let Some(id) = value["id"].as_str().filter(|s| snowflake(s))
    {
        return journal.receipt(&delivery.nonce, id);
    }
    // Expired tokens/missing permissions/unknown results remain durably visible.
    journal.uncertain(&delivery.nonce)
}
