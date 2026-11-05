//! Optional Discord Gateway, using only the policy client's pinned WS and REST.
use anyhow::{Result, bail, ensure};
use futures_util::{SinkExt, StreamExt};
use missy_core::{DiscordConfig, PolicyConfig, Scope};
use missy_policy::PolicyHttpClient;
use missy_runtime::Runtime;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
const REST: &str = "https://discord.com/api/v10";
const MAX_CHUNKS: usize = 16;
#[derive(Default)]
struct Resume {
    session: Option<String>,
    sequence: Option<u64>,
    url: Option<String>,
    bot: String,
}
pub async fn serve_discord(
    runtime: Arc<Runtime>,
    config: DiscordConfig,
    policy: PolicyConfig,
) -> Result<()> {
    ensure!(config.enabled, "Discord is not enabled");
    validate_config(&config)?;
    let token = std::env::var(&config.token_env)
        .map_err(|_| anyhow::anyhow!("Discord token environment variable missing"))?;
    ensure!(
        (32..=256).contains(&token.len()) && token.bytes().all(|b| b.is_ascii_graphic()),
        "invalid Discord token"
    );
    let http = Arc::new(PolicyHttpClient::new(policy)?);
    let discovery = http
        .request_json("GET", &format!("{REST}/gateway/bot"), auth(&token), None)
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Discord gateway discovery failed; check network policy and credentials"
            )
        })?;
    let base = discovery["url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Discord discovery returned no gateway URL"))?
        .to_owned();
    let (tx, rx) = mpsc::channel(32);
    let worker = tokio::spawn(process_messages(runtime, http.clone(), token.clone(), rx));
    let mut resume = Resume::default();
    let mut retry = 1u64;
    loop {
        let attempt = connection(&http, &token, &config, &base, &mut resume, &tx).await;
        if worker.is_finished() {
            bail!("Discord message worker stopped");
        }
        if let Err(error) = attempt
            && error.downcast_ref::<FatalClose>().is_some()
        {
            worker.abort();
            bail!("Discord Gateway rejected authentication/intents; channel stopped");
        }
        tokio::time::sleep(Duration::from_secs(retry)).await;
        retry = retry.saturating_mul(2).min(60);
    }
}
fn auth(token: &str) -> Vec<(String, String)> {
    vec![("Authorization".into(), format!("Bot {token}"))]
}
fn snowflake(value: &str) -> bool {
    !value.is_empty() && value.len() <= 20 && value.bytes().all(|b| b.is_ascii_digit())
}
pub fn validate_config(config: &DiscordConfig) -> Result<()> {
    ensure!(
        config.require_mention,
        "Discord requires explicit mentions; disabling mention protection is unsupported"
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
        "Discord requires explicit guild/channel allowlists or DM opt-in"
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
    tx: &mpsc::Sender<Incoming>,
) -> Result<()> {
    let mut url = url::Url::parse(resume.url.as_deref().unwrap_or(base))?;
    ensure!(url.scheme() == "wss", "Discord Gateway must use TLS");
    url.set_query(Some("v=10&encoding=json"));
    let ws = http.connect_websocket(url.as_str(), vec![]).await?;
    gateway_session(ws, token, config, resume, tx).await
}
async fn gateway_session(
    mut ws: missy_policy::PolicyWebSocket,
    token: &str,
    config: &DiscordConfig,
    resume: &mut Resume,
    tx: &mpsc::Sender<Incoming>,
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
        json!({"op":2,"d":{"token":token,"intents":37377,"properties":{"os":"linux","browser":"missy-rust","device":"missy-rust"}}})
    };
    send_frame(&mut ws, Message::Text(identify.to_string().into())).await?;
    let mut timer = tokio::time::interval(Duration::from_millis(millis));
    timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut awaiting_ack = false;
    let mut rates: HashMap<String, (Instant, u32)> = HashMap::new();
    loop {
        tokio::select! {
            _ = timer.tick() => {
                ensure!(!awaiting_ack, "Gateway heartbeat ACK deadline exceeded");
                send_frame(&mut ws, Message::Text(json!({"op":1,"d":resume.sequence}).to_string().into())).await?; awaiting_ack = true;
            }
            frame = ws.next() => {
                let frame = frame.ok_or_else(|| anyhow::anyhow!("Gateway disconnected"))??;
                if let Message::Close(close) = frame {
                    if let Some(close) = close { let code: u16 = close.code.into();
                        if matches!(code,4004|4010|4011|4012|4013|4014) { return Err(FatalClose.into()); }
                        if matches!(code,4007|4009) { *resume = Resume::default(); }
                    } bail!("Gateway closed");
                }
                if let Message::Ping(payload) = frame { send_frame(&mut ws,Message::Pong(payload)).await?; continue; }
                if !frame.is_text() { continue; }
                let event: Value = serde_json::from_str(frame.to_text()?)?;
                if let Some(s) = event["s"].as_u64() { resume.sequence = Some(s); }
                match event["op"].as_u64() {
                    Some(11) => awaiting_ack = false,
                    Some(1) => { send_frame(&mut ws,Message::Text(json!({"op":1,"d":resume.sequence}).to_string().into())).await?; }
                    Some(7) => bail!("Gateway requested reconnect"),
                    Some(9) => { if event["d"] != true { *resume = Resume::default(); } bail!("Gateway invalid session"); }
                    Some(0) => match event["t"].as_str() {
                        Some("READY") => {
                            resume.session = event["d"]["session_id"].as_str().map(str::to_owned);
                            resume.url = event["d"]["resume_gateway_url"].as_str().map(str::to_owned);
                            resume.bot = event["d"]["user"]["id"].as_str().unwrap_or("").to_owned();
                        }
                        Some("MESSAGE_CREATE") => if let Some(message) = incoming(config, &resume.bot, &event["d"]) {
                            let rate = rates.entry(message.scope.user.clone()).or_insert((Instant::now(),0));
                            if rate.0.elapsed() >= Duration::from_secs(60) { *rate = (Instant::now(),0); }
                            if rate.1 >= 30 { continue; }
                            rate.1 += 1;
                            let _ = tx.try_send(message);
                        },
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
}
async fn send_frame(ws: &mut missy_policy::PolicyWebSocket, message: Message) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(30), ws.send(message)).await??;
    Ok(())
}
struct Incoming {
    scope: Scope,
    request_id: String,
    prompt: String,
    channel: String,
}
fn incoming(config: &DiscordConfig, bot: &str, value: &Value) -> Option<Incoming> {
    if value["author"]["bot"] == true || !value["webhook_id"].is_null() || !snowflake(bot) {
        return None;
    }
    let user = value["author"]["id"].as_str()?;
    let channel = value["channel_id"].as_str()?;
    let id = value["id"].as_str()?;
    if ![user, channel, id].iter().all(|s| snowflake(s))
        || !config.allowed_users.iter().any(|s| s == user)
    {
        return None;
    }
    let tenant = match value["guild_id"].as_str() {
        Some(g)
            if snowflake(g)
                && config.allowed_guilds.iter().any(|s| s == g)
                && config.allowed_channels.iter().any(|s| s == channel) =>
        {
            format!("discord:g:{g}")
        }
        None if config.allow_dm => "discord:dm".to_owned(),
        _ => return None,
    };
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
    let scope = Scope {
        tenant,
        user: format!("discord:{user}"),
        channel: format!("discord:{channel}"),
        session: format!("channel:{channel}:user:{user}"),
    };
    scope.validate().ok()?;
    Some(Incoming {
        scope,
        request_id: format!("discord:{id}"),
        prompt: content.to_owned(),
        channel: channel.to_owned(),
    })
}
async fn process_messages(
    runtime: Arc<Runtime>,
    http: Arc<PolicyHttpClient>,
    token: String,
    mut rx: mpsc::Receiver<Incoming>,
) {
    while let Some(message) = rx.recv().await {
        let text = match runtime
            .run(message.scope, message.prompt, message.request_id)
            .await
        {
            Ok(result) => {
                let Ok(value) = serde_json::to_value(&result) else {
                    continue;
                };
                if value["deduplicated"] == true {
                    continue;
                }
                value["content"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("Turn finished without text. Inspect its scoped status.")
                    .to_owned()
            }
            Err(_) => "Turn could not execute. No automatic replay was attempted.".to_owned(),
        };
        for chunk in chunks(&text) {
            if send_message(&http, &token, &message.channel, &chunk)
                .await
                .is_err()
            {
                break;
            }
        }
    }
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
async fn send_message(
    http: &PolicyHttpClient,
    token: &str,
    channel: &str,
    text: &str,
) -> Result<()> {
    ensure!(snowflake(channel), "invalid Discord channel");
    let payload = serde_json::to_vec(
        &json!({"content":text,"allowed_mentions":{"parse":[],"replied_user":false}}),
    )?;
    for _ in 0..3 {
        let mut headers = auth(token);
        headers.push(("Content-Type".into(), "application/json".into()));
        let response = http
            .request(
                "POST",
                &format!("{REST}/channels/{channel}/messages"),
                headers,
                Some(payload.clone()),
            )
            .await?;
        if response.status == 429 {
            let value: Value = serde_json::from_slice(&response.body)?;
            let seconds = value["retry_after"].as_f64().unwrap_or(1.0);
            ensure!(
                seconds.is_finite() && (0.0..=60.0).contains(&seconds),
                "Discord rate wait out of bounds"
            );
            tokio::time::sleep(Duration::from_secs_f64(seconds.max(0.1))).await;
            continue;
        }
        ensure!(
            (200..300).contains(&response.status),
            "Discord message delivery failed"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
        return Ok(());
    }
    bail!("Discord REST retry limit exceeded")
}
#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> DiscordConfig {
        DiscordConfig {
            enabled: true,
            allowed_users: vec!["1".into()],
            allowed_guilds: vec!["2".into()],
            allowed_channels: vec!["3".into()],
            ..DiscordConfig::default()
        }
    }
    fn message() -> Value {
        json!({"id":"5","guild_id":"2","channel_id":"3","author":{"id":"1","bot":false},"mentions":[{"id":"4"}],"content":"<@4> hi"})
    }
    #[test]
    fn allowlists_mentions_bot_and_isolation() {
        let config = cfg();
        let value = message();
        assert_eq!(
            incoming(&config, "4", &value).unwrap().scope.tenant,
            "discord:g:2"
        );
        for (key, replacement) in [
            ("guild_id", json!("9")),
            ("channel_id", json!("9")),
            ("mentions", json!([])),
            ("author", json!({"id":"1","bot":true})),
        ] {
            let mut denied = value.clone();
            denied[key] = replacement;
            assert!(incoming(&config, "4", &denied).is_none());
        }
        let mut dm = value.clone();
        dm.as_object_mut().unwrap().remove("guild_id");
        assert!(incoming(&config, "4", &dm).is_none());
        let mut allow = config;
        allow.allow_dm = true;
        assert_eq!(
            incoming(&allow, "4", &dm).unwrap().scope.tenant,
            "discord:dm"
        );
        assert!(validate_config(&DiscordConfig::default()).is_err());
    }
    #[test]
    fn chunk_utf16_limits() {
        let output = chunks(&"😀".repeat(40_000));
        assert!(output.len() <= MAX_CHUNKS);
        assert!(output.iter().all(|s| s.encode_utf16().count() <= 2000));
    }
    #[tokio::test]
    async fn mock_gateway_identify_dispatch_ack_and_resume() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for resume_attempt in [false, true] {
                let (stream, _) = listener.accept().await.unwrap();
                let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
                ws.send(Message::Text(
                    json!({"op":10,"d":{"heartbeat_interval":1000}})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
                let identify: Value =
                    serde_json::from_str(ws.next().await.unwrap().unwrap().to_text().unwrap())
                        .unwrap();
                assert_eq!(identify["op"], if resume_attempt { 6 } else { 2 });
                if resume_attempt {
                    assert_eq!(identify["d"]["session_id"], "session-secret");
                    assert_eq!(identify["d"]["seq"], 2);
                }
                ws.send(Message::Text(json!({"op":0,"t":"READY","s":1,"d":{"session_id":"session-secret","resume_gateway_url":"wss://gateway.discord.gg","user":{"id":"4"}}}).to_string().into())).await.unwrap();
                ws.send(Message::Text(
                    json!({"op":0,"t":"MESSAGE_CREATE","s":2,"d":message()})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
                let heartbeat: Value =
                    serde_json::from_str(ws.next().await.unwrap().unwrap().to_text().unwrap())
                        .unwrap();
                assert_eq!(heartbeat["op"], 1);
                ws.send(Message::Text(json!({"op":11,"d":null}).to_string().into()))
                    .await
                    .unwrap();
                ws.send(Message::Text(json!({"op":7,"d":null}).to_string().into()))
                    .await
                    .unwrap();
            }
        });
        let http = PolicyHttpClient::new(PolicyConfig {
            allowed_origins: vec![format!("http://{address}")],
            allow_private_network: true,
            ..PolicyConfig::default()
        })
        .unwrap();
        let (tx, mut rx) = mpsc::channel(4);
        let mut resume = Resume::default();
        for _ in 0..2 {
            let ws = http
                .connect_websocket(&format!("ws://{address}"), vec![])
                .await
                .unwrap();
            assert!(
                tokio::time::timeout(
                    Duration::from_secs(5),
                    gateway_session(ws, "test-token", &cfg(), &mut resume, &tx)
                )
                .await
                .unwrap()
                .is_err()
            );
            assert_eq!(rx.recv().await.unwrap().request_id, "discord:5");
        }
        server.await.unwrap();
    }
}
