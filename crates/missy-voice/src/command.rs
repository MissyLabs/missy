//! Explicit trusted native CLI adapters. These do not install engines or models.
//! whisper.cpp `whisper-cli -m /abs/model -f {input} -otxt -of {output}`
//! writes `{output}.txt`. Piper receives text on stdin and must use
//! `--model /abs/model --output_file {output}` to write a WAV container.
use crate::{AudioFormat, SpeechAudio, Stt, Tts};
use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{io::AsyncWriteExt, process::Command};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandConfig {
    pub executable: PathBuf,
    /// Trusted operator config, not client input. Placeholders must be entire
    /// arguments. No shell, interpolation, user-supplied argv or inherited env.
    pub args: Vec<String>,
    pub timeout_seconds: u64,
}
impl CommandConfig {
    fn validate(&self, stt: bool) -> Result<()> {
        ensure!(
            self.executable.is_absolute(),
            "voice executable must be absolute"
        );
        let name = self
            .executable
            .file_name()
            .and_then(|n| n.to_str())
            .context("invalid executable name")?;
        ensure!(
            ![
                "sh", "bash", "dash", "zsh", "fish", "env", "python", "python3", "node", "perl",
                "ruby", "sudo", "doas"
            ]
            .contains(&name),
            "native speech engine required, not a command interpreter"
        );
        let m = fs::metadata(&self.executable)?;
        ensure!(
            m.is_file() && m.mode() & 0o111 != 0 && m.mode() & 0o022 == 0,
            "speech executable must be a non-group/world-writable executable file"
        );
        let mut file = OpenOptions::new().read(true).open(&self.executable)?;
        let mut magic = [0; 4];
        file.read_exact(&mut magic)?;
        ensure!(
            magic == *b"\x7fELF",
            "only native ELF speech engines are supported (no scripts)"
        );
        ensure!(
            (1..=600).contains(&self.timeout_seconds) && self.args.len() <= 128,
            "invalid command limits"
        );
        ensure!(
            self.args.iter().any(|a| a == "{output}"),
            "output placeholder required"
        );
        if stt {
            ensure!(
                self.args.iter().any(|a| a == "{input}"),
                "input placeholder required"
            );
        }
        for arg in &self.args {
            ensure!(
                arg.len() <= 4096 && !arg.contains('\0'),
                "invalid command argument"
            );
            if arg.contains('{') || arg.contains('}') {
                ensure!(
                    arg == "{output}" || stt && arg == "{input}",
                    "placeholder must be a complete, supported argument"
                );
            }
        }
        Ok(())
    }
}

pub struct CommandStt {
    config: CommandConfig,
}
impl CommandStt {
    pub fn new(config: CommandConfig) -> Result<Self> {
        config.validate(true)?;
        Ok(Self { config })
    }
}
pub struct CommandTts {
    config: CommandConfig,
}
impl CommandTts {
    pub fn new(config: CommandConfig) -> Result<Self> {
        config.validate(false)?;
        Ok(Self { config })
    }
}

#[async_trait]
impl Stt for CommandStt {
    async fn transcribe(
        &self,
        pcm: &[u8],
        format: AudioFormat,
        cancellation: CancellationToken,
    ) -> Result<String> {
        format.validate()?;
        ensure!(
            !pcm.is_empty()
                && pcm.len() <= 10 * 1024 * 1024
                && pcm.len().is_multiple_of(2 * format.channels as usize),
            "invalid PCM input"
        );
        let dir = private_temp()?;
        let input = dir.path().join("input.wav");
        let prefix = dir.path().join("transcript");
        write_private(&input, &encode_wav(pcm, format))?;
        write_private(&prefix.with_extension("txt"), b"")?;
        run(
            &self.config,
            dir.path(),
            Some(&input),
            &prefix,
            None,
            cancellation,
        )
        .await?;
        let bytes = read_bounded(&prefix.with_extension("txt"), 64 * 1024)?;
        Ok(String::from_utf8(bytes)
            .context("speech transcript is not UTF-8")?
            .trim()
            .to_owned())
    }
}

#[async_trait]
impl Tts for CommandTts {
    async fn synthesize(&self, text: &str, cancellation: CancellationToken) -> Result<SpeechAudio> {
        ensure!(
            text.len() <= 256 * 1024 && !text.contains('\0'),
            "invalid TTS text"
        );
        let dir = private_temp()?;
        let output = dir.path().join("speech.wav");
        write_private(&output, b"")?;
        run(
            &self.config,
            dir.path(),
            None,
            &output,
            Some(text),
            cancellation,
        )
        .await?;
        let wav = read_bounded(&output, 10 * 1024 * 1024)?;
        let sample_rate = wav_sample_rate(&wav)?;
        Ok(SpeechAudio { wav, sample_rate })
    }
}

async fn run(
    config: &CommandConfig,
    cwd: &Path,
    input: Option<&Path>,
    output: &Path,
    text: Option<&str>,
    cancellation: CancellationToken,
) -> Result<()> {
    ensure!(!cancellation.is_cancelled(), "voice command cancelled");
    // Revalidate at execution too: an operator replacing a configured binary
    // with an interpreter script must not silently change the execution model.
    config.validate(input.is_some())?;
    let mut command = Command::new(&config.executable);
    for arg in &config.args {
        match arg.as_str() {
            "{input}" => {
                command.arg(input.context("unexpected input placeholder")?);
            }
            "{output}" => {
                command.arg(output);
            }
            _ => {
                command.arg(arg);
            }
        }
    }
    command
        .current_dir(cwd)
        .env_clear()
        .env("LC_ALL", "C")
        .stdin(if text.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    // A dedicated process group permits cancellation cleanup of CLI workers,
    // rather than just the immediate child. No pre_exec closure or shell.
    command.as_std_mut().process_group(0);
    let mut child = command
        .spawn()
        .context("failed to start native speech command")?;
    let _group = ProcessGroup(child.id().context("speech process has no pid")?);
    let operation = async {
        if let Some(text) = text {
            let mut stdin = child.stdin.take().context("speech stdin unavailable")?;
            stdin.write_all(text.as_bytes()).await?;
            stdin.write_all(b"\n").await?;
            stdin.shutdown().await?;
        }
        let status = child.wait().await?;
        ensure!(status.success(), "native speech command failed");
        Ok::<(), anyhow::Error>(())
    };
    tokio::select! {
        biased;
        _ = cancellation.cancelled() => anyhow::bail!("native speech command cancelled"),
        result = tokio::time::timeout(Duration::from_secs(config.timeout_seconds), operation) => result.context("native speech command timed out")??,
    }
    Ok(())
}

use std::os::unix::process::CommandExt;
struct ProcessGroup(u32);
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        // SAFETY: process group belongs to our child, created using setpgid(0).
        // Negative pid targets this one group, never the caller's group.
        unsafe {
            libc::kill(-(self.0 as i32), libc::SIGKILL);
        }
    }
}
fn private_temp() -> Result<tempfile::TempDir> {
    let dir = tempfile::Builder::new().prefix("missy-voice-").tempdir()?;
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700))?;
    Ok(dir)
}
fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(
        file.metadata()?.is_file() && file.metadata()?.len() <= limit as u64,
        "invalid speech output file"
    );
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= limit, "speech output too large");
    Ok(bytes)
}
/// Serialize already-received little-endian PCM. No capture or audio devices.
pub fn encode_wav(pcm: &[u8], format: AudioFormat) -> Vec<u8> {
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32 + pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&format.channels.to_le_bytes());
    wav.extend_from_slice(&format.sample_rate.to_le_bytes());
    wav.extend_from_slice(&(format.sample_rate * u32::from(format.channels) * 2).to_le_bytes());
    wav.extend_from_slice(&(format.channels * 2).to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}
fn wav_sample_rate(wav: &[u8]) -> Result<u32> {
    ensure!(
        wav.len() >= 44 && &wav[..4] == b"RIFF" && &wav[8..12] == b"WAVE",
        "speech output must be WAV"
    );
    let mut offset = 12usize;
    let mut rate = None;
    let mut data = false;
    while offset + 8 <= wav.len() {
        let length = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into()?) as usize;
        let body = offset + 8;
        let end = body.checked_add(length).context("invalid WAV chunk")?;
        ensure!(end <= wav.len(), "truncated WAV");
        if &wav[offset..offset + 4] == b"fmt " {
            ensure!(
                length >= 16 && u16::from_le_bytes(wav[body..body + 2].try_into()?) == 1,
                "only PCM WAV output supported"
            );
            let channels = u16::from_le_bytes(wav[body + 2..body + 4].try_into()?);
            let sample_rate = u32::from_le_bytes(wav[body + 4..body + 8].try_into()?);
            AudioFormat {
                sample_rate,
                channels,
            }
            .validate()?;
            ensure!(
                u16::from_le_bytes(wav[body + 14..body + 16].try_into()?) == 16,
                "TTS must produce 16-bit PCM"
            );
            rate = Some(sample_rate);
        }
        if &wav[offset..offset + 4] == b"data" {
            ensure!(length > 0 && length.is_multiple_of(2), "invalid WAV data");
            data = true;
        }
        offset = end.checked_add(length % 2).context("invalid WAV chunk")?;
    }
    ensure!(data, "WAV missing audio data");
    rate.context("WAV missing sample format")
}
