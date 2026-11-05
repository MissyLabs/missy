//! Bounded text Gateway opcode 4 bridge. No second bot login.
use super::*;
use async_trait::async_trait;
use missy_discord_voice::{GatewayVoiceAdapter, GatewayVoiceCredentials};
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
pub struct DiscordGatewayVoice {
    tx: mpsc::Sender<Command>,
    ready: watch::Receiver<bool>,
}
pub struct DiscordVoiceGatewayHook {
    pub(super) commands: mpsc::Receiver<Command>,
    ready: watch::Sender<bool>,
    pending: HashMap<u64, Pending>,
    bot: Option<u64>,
    active: HashMap<u64, (u64, CancellationToken)>,
}
impl Drop for DiscordVoiceGatewayHook {
    fn drop(&mut self) {
        for (_, (_, cancel)) in self.active.drain() {
            cancel.cancel();
        }
    }
}
pub(super) enum Command {
    Join {
        guild: u64,
        channel: u64,
        bot: u64,
        reply: oneshot::Sender<Result<GatewayVoiceCredentials>>,
    },
    Leave {
        guild: u64,
        bot: u64,
        reply: oneshot::Sender<Result<()>>,
    },
}
struct Pending {
    channel: u64,
    bot: u64,
    session: Option<String>,
    endpoint: Option<String>,
    token: Option<String>,
    reply: oneshot::Sender<Result<GatewayVoiceCredentials>>,
}
pub fn discord_voice_bridge() -> (Arc<DiscordGatewayVoice>, DiscordVoiceGatewayHook) {
    let (tx, commands) = mpsc::channel(16);
    let (ready, rx) = watch::channel(false);
    (
        Arc::new(DiscordGatewayVoice { tx, ready: rx }),
        DiscordVoiceGatewayHook {
            commands,
            ready,
            pending: HashMap::new(),
            bot: None,
            active: HashMap::new(),
        },
    )
}
#[async_trait]
impl GatewayVoiceAdapter for DiscordGatewayVoice {
    async fn negotiate(
        &self,
        guild: u64,
        channel: u64,
        bot: u64,
    ) -> Result<GatewayVoiceCredentials> {
        let mut ready = self.ready.clone();
        tokio::time::timeout(Duration::from_secs(30), async {
            while !*ready.borrow_and_update() {
                ready
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("Gateway bridge stopped"))?;
            }
            Ok::<_, anyhow::Error>(())
        })
        .await
        .map_err(|_| anyhow::anyhow!("Gateway not ready"))??;
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Join {
                guild,
                channel,
                bot,
                reply: tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Gateway bridge closed"))?;
        tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .map_err(|_| anyhow::anyhow!("Gateway events timeout"))?
            .map_err(|_| anyhow::anyhow!("Gateway reset"))?
    }
    async fn leave(&self, guild: u64, bot: u64) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Command::Leave {
                guild,
                bot,
                reply: tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Gateway closed"))?;
        tokio::time::timeout(Duration::from_secs(5), rx)
            .await
            .map_err(|_| anyhow::anyhow!("leave timeout"))?
            .map_err(|_| anyhow::anyhow!("leave interrupted"))?
    }
}
impl DiscordVoiceGatewayHook {
    pub(super) async fn reset(&mut self) {
        self.ready.send_replace(false);
        self.pending.clear();
        self.bot = None;
        for (_, (_, cancel)) in self.active.drain() {
            cancel.cancel();
        }
        while let Ok(c) = self.commands.try_recv() {
            drop(c);
        }
    }
    pub(super) async fn command(
        &mut self,
        ws: &mut missy_policy::PolicyWebSocket,
        c: Command,
    ) -> Result<()> {
        match c {
            Command::Join {
                guild,
                channel,
                bot,
                reply,
            } => {
                if self.bot != Some(bot)
                    || self.pending.contains_key(&guild)
                    || self.pending.len() >= 16
                {
                    let _ = reply.send(Err(anyhow::anyhow!("voice request denied")));
                    return Ok(());
                }
                self.pending.insert(
                    guild,
                    Pending {
                        channel,
                        bot,
                        session: None,
                        endpoint: None,
                        token: None,
                        reply,
                    },
                );
                send_frame(ws,Message::Text(json!({"op":4,"d":{"guild_id":guild.to_string(),"channel_id":channel.to_string(),"self_mute":false,"self_deaf":false}}).to_string().into())).await?;
            }
            Command::Leave { guild, bot, reply } => {
                if self.bot != Some(bot) {
                    let _ = reply.send(Err(anyhow::anyhow!("bot mismatch")));
                    return Ok(());
                }
                self.pending.remove(&guild);
                send_frame(ws,Message::Text(json!({"op":4,"d":{"guild_id":guild.to_string(),"channel_id":null,"self_mute":false,"self_deaf":false}}).to_string().into())).await?;
                let _ = reply.send(Ok(()));
            }
        }
        Ok(())
    }
    pub(super) async fn event(&mut self, event: &Value, bot: &str) -> Result<()> {
        let name = event["t"].as_str().unwrap_or("");
        let d = &event["d"];
        if name == "READY" || name == "RESUMED" {
            if name == "READY" {
                ensure!(d["user"]["id"].as_str() == Some(bot), "bot mismatch");
                self.pending.clear();
            }
            self.bot = Some(bot.parse()?);
            self.ready.send_replace(true);
            return Ok(());
        }
        if !matches!(name, "VOICE_STATE_UPDATE" | "VOICE_SERVER_UPDATE") {
            return Ok(());
        }
        let Some(guild) = d["guild_id"].as_str().and_then(|v| v.parse::<u64>().ok()) else {
            return Ok(());
        };
        if let Some((channel, cancel)) = self.active.get(&guild)
            && (name == "VOICE_SERVER_UPDATE"
                || (name == "VOICE_STATE_UPDATE"
                    && d["user_id"].as_str() == Some(bot)
                    && d["channel_id"].as_str().and_then(|s| s.parse::<u64>().ok())
                        != Some(*channel)))
        {
            cancel.cancel();
            self.active.remove(&guild);
        }
        let Some(p) = self.pending.get_mut(&guild) else {
            return Ok(());
        };
        if p.reply.is_closed() {
            self.pending.remove(&guild);
            return Ok(());
        }
        if name == "VOICE_STATE_UPDATE" {
            if d["user_id"].as_str() != Some(bot) {
                return Ok(());
            }
            if d["channel_id"].as_str().and_then(|v| v.parse::<u64>().ok()) != Some(p.channel) {
                self.pending.remove(&guild);
                return Ok(());
            }
            p.session = d["session_id"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 256)
                .map(str::to_owned);
        } else {
            p.endpoint = d["endpoint"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 253)
                .map(str::to_owned);
            p.token = d["token"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 1024)
                .map(str::to_owned);
            if p.endpoint.is_none() || p.token.is_none() {
                self.pending.remove(&guild);
                return Ok(());
            }
        }
        if p.session.is_some() && p.endpoint.is_some() && p.token.is_some() {
            let p = self.pending.remove(&guild).unwrap();
            let session_cancel = CancellationToken::new();
            self.active
                .insert(guild, (p.channel, session_cancel.clone()));
            let _ = p.reply.send(Ok(GatewayVoiceCredentials {
                guild_id: guild,
                channel_id: p.channel,
                user_id: p.bot,
                endpoint: p.endpoint.unwrap(),
                session_id: p.session.unwrap(),
                token: p.token.unwrap(),
                session_cancel,
            }));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn authenticated_gateway_pairing_and_reset() {
        let (_, mut hook) = discord_voice_bridge();
        hook.event(&json!({"t":"READY","d":{"user":{"id":"1"}}}), "1")
            .await
            .unwrap();
        let (reply, rx) = oneshot::channel();
        hook.pending.insert(
            2,
            Pending {
                channel: 3,
                bot: 1,
                session: None,
                endpoint: None,
                token: None,
                reply,
            },
        );
        hook.event(&json!({"t":"VOICE_STATE_UPDATE","d":{"guild_id":"2","channel_id":"3","user_id":"99","session_id":"attacker"}}), "1").await.unwrap();
        assert!(hook.pending[&2].session.is_none());
        hook.event(&json!({"t":"VOICE_SERVER_UPDATE","d":{"guild_id":"2","endpoint":"test.invalid","token":"synthetic"}}),"1").await.unwrap();
        hook.event(&json!({"t":"VOICE_STATE_UPDATE","d":{"guild_id":"2","channel_id":"3","user_id":"1","session_id":"synthetic"}}),"1").await.unwrap();
        let credentials = rx.await.unwrap().unwrap();
        assert_eq!(credentials.channel_id, 3);
        assert!(!credentials.session_cancel.is_cancelled());
        hook.reset().await;
        assert!(credentials.session_cancel.is_cancelled());
        assert!(hook.pending.is_empty());
    }
    #[tokio::test]
    async fn opcode_four_websocket_fixture() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let peer = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
            let msg = ws.next().await.unwrap().unwrap();
            let value: Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
            assert_eq!(value["op"], 4);
            assert_eq!(value["d"]["channel_id"], "3");
        });
        let config = missy_core::PolicyConfig {
            allow_private_network: true,
            allowed_origins: vec![format!("http://{addr}")],
            ..Default::default()
        };
        let http = PolicyHttpClient::new(config).unwrap();
        let mut ws = http
            .connect_websocket(&format!("ws://{addr}"), vec![])
            .await
            .unwrap();
        let (_, mut hook) = discord_voice_bridge();
        hook.bot = Some(1);
        let (tx, _) = oneshot::channel();
        hook.command(
            &mut ws,
            Command::Join {
                guild: 2,
                channel: 3,
                bot: 1,
                reply: tx,
            },
        )
        .await
        .unwrap();
        peer.await.unwrap();
    }
}
