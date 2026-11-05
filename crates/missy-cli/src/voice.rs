//! Server-side voice capability selection, never a client-controlled scope.
use anyhow::{Result, bail};
use async_trait::async_trait;
use missy_voice::{Device, PolicyMode, VoiceAgent};
use serde::Deserialize;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceApplication {
    pub server: missy_voice::VoiceConfig,
    pub stt: missy_voice::CommandConfig,
    pub tts: missy_voice::CommandConfig,
}
pub struct RuntimeVoice {
    pub full: Arc<missy_runtime::Runtime>,
    pub safe: Arc<missy_runtime::Runtime>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscordVoiceApplication {
    pub server: missy_discord_voice::DiscordVoiceConfig,
    pub stt: missy_voice::CommandConfig,
    pub tts: missy_voice::CommandConfig,
}
#[async_trait]
impl missy_discord_voice::DiscordVoiceAgent for RuntimeVoice {
    async fn respond(
        &self,
        binding: &missy_discord_voice::VoiceBinding,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<String> {
        if !binding.consent {
            bail!("speaker has no explicit voice consent");
        }
        let mode = match binding.mode {
            missy_discord_voice::VoiceMode::Normal => PolicyMode::Full,
            missy_discord_voice::VoiceMode::NoTools | missy_discord_voice::VoiceMode::SafeChat => {
                PolicyMode::SafeChat
            }
        };
        let device = Device {
            node_id: format!(
                "discord:{}:{}:{}",
                binding.guild_id, binding.channel_id, binding.user_id
            ),
            friendly_name: "consented Discord voice participant".into(),
            room: binding.channel_id.to_string(),
            scope: Some(binding.scope.clone()),
            mode,
            paired: true,
            requested_at: 0,
        };
        <Self as VoiceAgent>::respond(self, &device, prompt, cancel).await
    }
}
struct CancelOnDrop {
    runtime: Arc<missy_runtime::Runtime>,
    scope: missy_core::Scope,
    turn_id: String,
    armed: bool,
}
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.runtime.cancel(&self.scope, &self.turn_id);
        }
    }
}
#[async_trait]
impl VoiceAgent for RuntimeVoice {
    async fn respond(
        &self,
        device: &Device,
        prompt: &str,
        cancel: CancellationToken,
    ) -> Result<String> {
        let scope = device
            .scope
            .clone()
            .ok_or_else(|| anyhow::anyhow!("device has no approved principal"))?;
        let runtime = match device.mode {
            PolicyMode::Full => &self.full,
            PolicyMode::SafeChat => &self.safe,
            PolicyMode::Muted => bail!("device is muted"),
        };
        let start = runtime
            .start(
                scope.clone(),
                prompt.to_owned(),
                format!("voice:{}", uuid::Uuid::new_v4()),
            )
            .await?;
        let turn_id = start.turn_id;
        let mut guard = CancelOnDrop {
            runtime: runtime.clone(),
            scope: scope.clone(),
            turn_id: turn_id.clone(),
            armed: true,
        };
        loop {
            let t = runtime.status(&scope, &turn_id)?;
            match t.status {
                missy_store::TurnStatus::Completed => {
                    guard.armed = false;
                    return Ok(t.response.unwrap_or_default());
                }
                missy_store::TurnStatus::Cancelled
                | missy_store::TurnStatus::Failed
                | missy_store::TurnStatus::Interrupted => {
                    guard.armed = false;
                    bail!("voice turn stopped; inspect its scoped status");
                }
                _ => {}
            }
            tokio::select! {_=cancel.cancelled()=>{runtime.cancel(&scope,&turn_id)?;bail!("voice turn cancellation requested");},_=tokio::time::sleep(std::time::Duration::from_millis(100))=>{}}
        }
    }
}
pub fn mode(value: &str) -> Result<PolicyMode> {
    match value {
        "full" => Ok(PolicyMode::Full),
        "safe-chat" => Ok(PolicyMode::SafeChat),
        "muted" => Ok(PolicyMode::Muted),
        _ => bail!("mode must be full, safe-chat or muted"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use missy_core::{Completion, Message, Provider, Scope, ToolDefinition, Usage};
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Blocked(AtomicBool);
    #[async_trait]
    impl Provider for Blocked {
        fn name(&self) -> &str {
            "voice-test"
        }
        async fn complete(
            &self,
            _: &[Message],
            _: &[ToolDefinition],
            cancel: CancellationToken,
        ) -> Result<Completion> {
            self.0.store(true, Ordering::SeqCst);
            cancel.cancelled().await;
            Ok(Completion {
                message: Message::text("assistant", "cancelled"),
                usage: Usage::default(),
                finish_reason: "stop".into(),
            })
        }
    }
    #[tokio::test]
    async fn dropped_voice_callback_cancels_detached_runtime_turn() {
        let d = tempfile::tempdir().unwrap();
        let provider = Arc::new(Blocked(AtomicBool::new(false)));
        let store = Arc::new(missy_store::Store::open(d.path().join("runtime.sqlite3")).unwrap());
        let runtime = Arc::new(
            missy_runtime::Runtime::new(
                store.clone(),
                provider.clone(),
                Arc::new(crate::extensions::NoTools),
                "test".into(),
                Default::default(),
            )
            .unwrap(),
        );
        let adapter = Arc::new(RuntimeVoice {
            full: runtime.clone(),
            safe: runtime.clone(),
        });
        let scope = Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "voice".into(),
            session: "device".into(),
        };
        let device = Device {
            node_id: "device".into(),
            friendly_name: "fixture".into(),
            room: "".into(),
            scope: Some(scope.clone()),
            mode: PolicyMode::Full,
            paired: true,
            requested_at: 0,
        };
        let callback = tokio::spawn(async move {
            adapter
                .respond(&device, "request", CancellationToken::new())
                .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while !provider.0.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        callback.abort();
        let _ = callback.await;
        let turn = store.list_turns(&scope, 10).unwrap().remove(0);
        assert!(store.cancellation_requested(&scope, &turn.id).unwrap());
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                if runtime.status(&scope, &turn.id).unwrap().status
                    == missy_store::TurnStatus::Cancelled
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }
}
