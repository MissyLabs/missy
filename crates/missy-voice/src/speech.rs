//! Strict PCM decoding shared by local playback and avatar lip synchronization.
use crate::SpeechAudio;
use anyhow::{Result, ensure};
use async_trait::async_trait;
use missy_core::{ToolCall, ToolContext};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Implementations must enforce output policy AND durable exact-call approval.
/// A model-provided `confirm` property is not consent. No implicit default device.
#[async_trait]
#[allow(clippy::too_many_arguments)]
pub trait SpeechOutput: Send + Sync {
    fn configuration_fingerprint(&self) -> String;
    fn validate_output(&self, sink: &str, volume_pct: u8) -> Result<()>;
    async fn play(
        &self,
        ctx: &ToolContext,
        call: &ToolCall,
        sink: &str,
        volume_pct: u8,
        audio: &SpeechAudio,
        cancellation: CancellationToken,
        started: tokio::sync::oneshot::Sender<tokio::time::Instant>,
    ) -> Result<()>;
}

pub struct PcmWav<'a> {
    pub pcm: &'a [u8],
    pub sample_rate: u32,
    pub channels: u16,
}
impl PcmWav<'_> {
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(
            self.pcm.len() as f64 / (self.sample_rate as f64 * self.channels as f64 * 2.0),
        )
    }
    /// 50ms normalized RMS envelope, including a trailing partial chunk.
    /// Digital silence stays zero; nonzero peaks are compressed to 0.85.
    pub fn mouth_envelope(&self) -> Vec<f64> {
        let chunk = (self.sample_rate as usize / 20).max(1) * self.channels as usize * 2;
        let levels: Vec<f64> = self
            .pcm
            .chunks(chunk)
            .map(|bytes| {
                let (sum, n) =
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .fold((0.0, 0usize), |(sum, n), b| {
                            let x = i16::from_le_bytes([b[0], b[1]]) as f64 / 32768.0;
                            (sum + x * x, n + 1)
                        });
                (sum / n.max(1) as f64).sqrt()
            })
            .collect();
        let peak = levels.iter().copied().fold(0.0, f64::max);
        levels
            .into_iter()
            .map(|v| {
                if peak == 0.0 {
                    0.0
                } else {
                    (v / peak).powf(0.6) * 0.85
                }
            })
            .collect()
    }
}
pub fn decode_pcm_wav(bytes: &[u8]) -> Result<PcmWav<'_>> {
    ensure!(
        bytes.len() >= 44 && bytes.len() <= 10 * 1024 * 1024,
        "WAV size invalid"
    );
    ensure!(
        &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "WAV container required"
    );
    let len = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    ensure!(
        len.checked_add(8) == Some(bytes.len()),
        "WAV RIFF length mismatch"
    );
    let mut pos = 12usize;
    let mut format = None;
    let mut pcm = None;
    while pos < bytes.len() {
        ensure!(bytes.len() - pos >= 8, "truncated WAV chunk");
        let n = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let start = pos + 8;
        let end = start
            .checked_add(n)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| anyhow::anyhow!("truncated WAV payload"))?;
        match &bytes[pos..pos + 4] {
            b"fmt " => {
                ensure!(format.is_none() && n >= 16, "invalid duplicate WAV format");
                let f = &bytes[start..end];
                let encoding = u16::from_le_bytes(f[..2].try_into()?);
                let channels = u16::from_le_bytes(f[2..4].try_into()?);
                let rate = u32::from_le_bytes(f[4..8].try_into()?);
                let byte_rate = u32::from_le_bytes(f[8..12].try_into()?);
                let align = u16::from_le_bytes(f[12..14].try_into()?);
                let bits = u16::from_le_bytes(f[14..16].try_into()?);
                ensure!(
                    encoding == 1
                        && (1..=2).contains(&channels)
                        && (8000..=48000).contains(&rate)
                        && bits == 16
                        && align == channels * 2
                        && byte_rate == rate * channels as u32 * 2,
                    "16-bit mono/stereo PCM WAV required"
                );
                format = Some((rate, channels));
            }
            b"data" => {
                ensure!(pcm.is_none(), "duplicate WAV data");
                pcm = Some(&bytes[start..end]);
            }
            _ => {}
        }
        pos = end
            .checked_add(n % 2)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| anyhow::anyhow!("missing WAV chunk padding"))?;
    }
    let (sample_rate, channels) = format.ok_or_else(|| anyhow::anyhow!("missing WAV format"))?;
    let pcm = pcm.ok_or_else(|| anyhow::anyhow!("missing WAV data"))?;
    ensure!(
        !pcm.is_empty() && pcm.len().is_multiple_of(channels as usize * 2),
        "invalid PCM alignment"
    );
    Ok(PcmWav {
        pcm,
        sample_rate,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AudioFormat, command::encode_wav};
    #[test]
    fn strict_pcm_and_partial_envelope() {
        let bytes = encode_wav(
            &vec![0; 1602],
            AudioFormat {
                sample_rate: 16000,
                channels: 1,
            },
        );
        let wav = decode_pcm_wav(&bytes).unwrap();
        assert_eq!(wav.mouth_envelope(), vec![0.0, 0.0]);
        let mut corrupt = bytes.clone();
        corrupt[4] = 0;
        assert!(decode_pcm_wav(&corrupt).is_err());
        let loud = encode_wav(
            &[0xff, 0x7f].repeat(801),
            AudioFormat {
                sample_rate: 16000,
                channels: 1,
            },
        );
        assert_eq!(
            decode_pcm_wav(&loud).unwrap().mouth_envelope(),
            vec![0.85, 0.85]
        );
    }
}
