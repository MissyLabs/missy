use crate::{DiscordVoiceConfig, DiscordVoiceService, VoiceSessionStatus};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock, Weak};
#[async_trait]
pub trait VoiceApprovalGate: Send + Sync {
    async fn approve(&self, ctx: &ToolContext, call: &ToolCall) -> Result<bool>;
}
pub struct DiscordVoiceSayTools {
    service: Arc<OnceLock<Weak<DiscordVoiceService>>>,
    config: DiscordVoiceConfig,
    _keepalive: Option<Arc<DiscordVoiceService>>,
    gate: Arc<dyn VoiceApprovalGate>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Say {
    guild_id: u64,
    channel_id: u64,
    text: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionArgs {
    guild_id: u64,
    channel_id: u64,
}
impl DiscordVoiceSayTools {
    pub fn new(service: Arc<DiscordVoiceService>, gate: Arc<dyn VoiceApprovalGate>) -> Self {
        let config = service.config().clone();
        let cell = OnceLock::new();
        let _ = cell.set(Arc::downgrade(&service));
        Self {
            service: Arc::new(cell),
            config,
            gate,
            _keepalive: Some(service),
        }
    }
    /// Build runtime tool catalogs before service construction, then attach the
    /// exact configured service once. Never changes config/capability lineage.
    pub fn pending(config: DiscordVoiceConfig, gate: Arc<dyn VoiceApprovalGate>) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            service: Arc::new(OnceLock::new()),
            config,
            gate,
            _keepalive: None,
        })
    }
    pub fn attach(&self, service: Arc<DiscordVoiceService>) -> Result<()> {
        ensure!(
            serde_json::to_vec(&self.config)? == serde_json::to_vec(service.config())?,
            "voice service config mismatch"
        );
        self.service
            .set(Arc::downgrade(&service))
            .map_err(|_| anyhow::anyhow!("voice service already attached"))
    }
    fn service(&self) -> Result<Arc<DiscordVoiceService>> {
        self.service
            .get()
            .and_then(Weak::upgrade)
            .context("voice service not attached/running")
    }
    fn request(&self, ctx: &ToolContext, call: &ToolCall) -> Result<(Say, u64)> {
        ctx.scope.validate()?;
        ensure!(
            call.name == "discord_voice_say"
                && serde_json::to_vec(&call.arguments)?.len() <= 40_000,
            "invalid voice say request"
        );
        let a: Say = serde_json::from_value(call.arguments.clone())?;
        ensure!(
            !a.text.trim().is_empty() && a.text.len() <= 32768 && !a.text.contains('\0'),
            "invalid voice text"
        );
        let users: Vec<_> = self
            .config
            .bindings
            .iter()
            .filter(|b| {
                b.consent
                    && b.guild_id == a.guild_id
                    && b.channel_id == a.channel_id
                    && b.scope == ctx.scope
            })
            .collect();
        ensure!(
            users.len() == 1,
            "exact unique scope voice binding required"
        );
        Ok((a, users.first().context("missing voice binding")?.user_id))
    }
}
impl DiscordVoiceSayTools {
    fn session_request(&self, ctx: &ToolContext, call: &ToolCall) -> Result<(SessionArgs, u64)> {
        ctx.scope.validate()?;
        ensure!(
            matches!(
                call.name.as_str(),
                "discord_voice_join" | "discord_voice_leave" | "discord_voice_status"
            ),
            "unknown voice operation"
        );
        let args: SessionArgs = serde_json::from_value(call.arguments.clone())?;
        let bindings: Vec<_> = self
            .config
            .bindings
            .iter()
            .filter(|b| {
                b.consent
                    && b.guild_id == args.guild_id
                    && b.channel_id == args.channel_id
                    && b.scope == ctx.scope
            })
            .collect();
        ensure!(
            bindings.len() == 1,
            "exact unique authenticated voice binding required"
        );
        self.service()?;
        Ok((args, bindings[0].user_id))
    }
}
#[async_trait]
impl ToolExecutor for DiscordVoiceSayTools {
    fn configuration_fingerprint(&self) -> String {
        format!(
            "discord-voice-say-v1:{}",
            serde_json::to_string(&self.config).unwrap_or_default()
        )
    }
    fn definitions(&self) -> Vec<ToolDefinition> {
        let mut defs=vec![ToolDefinition{name:"discord_voice_say".into(),description:"Queue approved speech into an already approved Discord voice session, bound to the exact caller scope. Never joins or retries speech automatically; queued is not played.".into(),side_effect:true,parameters:json!({"type":"object","additionalProperties":false,"properties":{"guild_id":{"type":"integer","minimum":1},"channel_id":{"type":"integer","minimum":1},"text":{"type":"string","minLength":1,"maxLength":32768}},"required":["guild_id","channel_id","text"]})}];
        for (name, description, side_effect) in [
            (
                "discord_voice_join",
                "Join the exact operator-consented voice binding after durable human approval; never auto-retry an uncertain join.",
                true,
            ),
            (
                "discord_voice_leave",
                "Leave the exact authenticated voice binding after durable approval.",
                true,
            ),
            (
                "discord_voice_status",
                "Read voice-session state for the exact authenticated caller binding. No join or capture.",
                false,
            ),
        ] {
            defs.push(ToolDefinition{name:name.into(),description:description.into(),side_effect,parameters:json!({"type":"object","additionalProperties":false,"properties":{"guild_id":{"type":"integer","minimum":1},"channel_id":{"type":"integer","minimum":1}},"required":["guild_id","channel_id"]})});
        }
        defs
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        let result = async {
            if call.name != "discord_voice_say" {
                self.session_request(ctx, call)?;
                ensure!(
                    !ctx.cancellation.is_cancelled(),
                    "voice operation cancelled before dispatch"
                );
                return Ok(());
            }
            let (a, _) = self.request(ctx, call)?;
            ensure!(
                self.service()?.status(a.guild_id).await
                    == Some(VoiceSessionStatus::ConnectingOrActive),
                "voice session not joined"
            );
            Ok::<_, anyhow::Error>(())
        }
        .await;
        result.map_err(|e| {
            missy_core::ToolDispatchRefused {
                reason: e.to_string(),
            }
            .into()
        })
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        self.preflight(ctx, call).await?;
        if call.name != "discord_voice_say" {
            let (args, actor) = self.session_request(ctx, call)?;
            let service = self.service()?;
            if call.name == "discord_voice_status" {
                let state = match service.status(args.guild_id).await {
                    Some(VoiceSessionStatus::ConnectingOrActive) => "connecting_or_active",
                    Some(VoiceSessionStatus::Ended) => "ended",
                    None => "not_joined",
                };
                return Ok(
                    json!({"guild_id":args.guild_id,"channel_id":args.channel_id,"state":state}),
                );
            }
            ensure!(
                self.gate.approve(ctx, call).await?,
                "durable exact voice operation approval required"
            );
            if call.name == "discord_voice_join" {
                service
                    .join_cancellable(
                        args.guild_id,
                        args.channel_id,
                        actor,
                        &DiscordVoiceService::join_approval(args.guild_id, args.channel_id),
                        ctx.cancellation.child_token(),
                    )
                    .await?;
            } else {
                service
                    .leave(
                        args.guild_id,
                        args.channel_id,
                        actor,
                        &DiscordVoiceService::leave_approval(args.guild_id, args.channel_id),
                    )
                    .await?;
            }
            return Ok(
                json!({"guild_id":args.guild_id,"channel_id":args.channel_id,"operation":call.name,"completed":true}),
            );
        }
        let (a, actor) = self.request(ctx, call)?;
        ensure!(
            self.gate.approve(ctx, call).await?,
            "durable exact voice speech approval required"
        );
        self.service()?
            .speak_text(
                a.guild_id,
                a.channel_id,
                actor,
                &a.text,
                ctx.cancellation.child_token(),
            )
            .await?;
        Ok(json!({"queued":true,"played":false,"guild_id":a.guild_id,"channel_id":a.channel_id}))
    }
}
