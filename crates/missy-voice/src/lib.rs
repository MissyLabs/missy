//! Native, loopback-only edge voice transport. No recording devices are opened.
//!
//! Device scope and mode come exclusively from an explicitly approved private
//! registry. The embedding application MUST dispatch `SafeChat` to a separately
//! constructed no-tools runtime with its own capability fingerprint. The trait
//! makes that choice explicit; it does not grant voice clients a runtime handle.
mod command;
mod registry;
mod server;

pub use command::{CommandConfig, CommandStt, CommandTts};
pub use registry::{Device, DeviceRegistry, PolicyMode};
pub use server::{VoiceConfig, VoiceServer};

use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
}
impl AudioFormat {
    pub fn validate(self) -> Result<()> {
        anyhow::ensure!(
            (8_000..=48_000).contains(&self.sample_rate),
            "invalid sample rate"
        );
        anyhow::ensure!((1..=2).contains(&self.channels), "invalid channel count");
        Ok(())
    }
}

pub struct SpeechAudio {
    /// WAV container, not an implicitly encoded or captured microphone stream.
    pub wav: Vec<u8>,
    pub sample_rate: u32,
}

#[async_trait]
pub trait Stt: Send + Sync {
    async fn transcribe(
        &self,
        pcm: &[u8],
        format: AudioFormat,
        cancellation: CancellationToken,
    ) -> Result<String>;
}

#[async_trait]
pub trait Tts: Send + Sync {
    async fn synthesize(&self, text: &str, cancellation: CancellationToken) -> Result<SpeechAudio>;
}

#[async_trait]
pub trait VoiceAgent: Send + Sync {
    /// `device` is registry-owned, never client-provided scope or mode. For
    /// SafeChat use a no-tools runtime, not a prompt instructing tools be hidden.
    async fn respond(
        &self,
        device: &Device,
        prompt: &str,
        cancellation: CancellationToken,
    ) -> Result<String>;
}
