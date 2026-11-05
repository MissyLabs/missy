use anyhow::{Result, bail, ensure};
use missy_core::ToolContext;
use missy_policy::{Policy, PolicyHttpClient};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SingingConfig {
    pub base_url: String,
    pub max_polls: u32,
    pub poll_interval_ms: u64,
}
impl Default for SingingConfig {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            max_polls: 300,
            poll_interval_ms: 1000,
        }
    }
}
impl SingingConfig {
    pub fn validate(&self) -> Result<()> {
        super::validate_base(&self.base_url, false)?;
        ensure!(
            (1..=600).contains(&self.max_polls) && (10..=5000).contains(&self.poll_interval_ms),
            "singing poll bounds"
        );
        Ok(())
    }
}
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SingingRequest {
    pub lyrics: String,
    pub tags: String,
    pub duration: f64,
    pub bpm: u16,
    pub keyscale: String,
    pub time_signature: String,
    pub language: String,
    pub seed: u32,
    pub steps: u16,
    pub sampler_cfg: f64,
    pub lyric_cfg: f64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u16,
    pub min_p: f64,
    pub generate_audio_codes: bool,
    pub validate_only: bool,
    pub save_path: Option<PathBuf>,
}
impl Default for SingingRequest {
    fn default() -> Self {
        Self {
            lyrics: String::new(),
            tags: String::new(),
            duration: 30.0,
            bpm: 120,
            keyscale: "C major".into(),
            time_signature: "4".into(),
            language: "en".into(),
            seed: 0,
            steps: 8,
            sampler_cfg: 1.0,
            lyric_cfg: 2.0,
            temperature: 0.85,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            generate_audio_codes: true,
            validate_only: false,
            save_path: None,
        }
    }
}
impl SingingRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.lyrics.len() <= 16384 && self.tags.len() <= 2048,
            "lyrics/tags too large"
        );
        ensure!(
            self.validate_only
                || (!self.lyrics.trim().is_empty()
                    && !self.tags.trim().is_empty()
                    && self.save_path.is_some()),
            "lyrics, tags and save_path required"
        );
        ensure!(
            (5.0..=180.0).contains(&self.duration)
                && (10..=300).contains(&self.bpm)
                && (1..=50).contains(&self.steps),
            "song duration/tempo/steps out of range"
        );
        ensure!(
            ["2", "3", "4", "6"].contains(&self.time_signature.as_str()),
            "invalid time signature"
        );
        let (note, mode) = self
            .keyscale
            .split_once(' ')
            .ok_or_else(|| anyhow::anyhow!("invalid key"))?;
        ensure!(
            [
                "C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#",
                "Bb", "B"
            ]
            .contains(&note)
                && ["major", "minor"].contains(&mode),
            "invalid key"
        );
        ensure!("ar az bg bn ca cs da de el en es fa fi fr he hi hr ht hu id is it ja ko la lt ms ne nl no pa pl pt ro ru sa sk sr sv sw ta te th tl tr uk unknown ur vi yue zh".split(' ').any(|l|l == self.language),"unsupported language");
        ensure!(
            (0.0..=30.0).contains(&self.sampler_cfg)
                && (0.0..=30.0).contains(&self.lyric_cfg)
                && (0.0..=2.0).contains(&self.temperature)
                && (0.0..=1.0).contains(&self.top_p)
                && (0.0..=1.0).contains(&self.min_p)
                && self.top_k <= 100,
            "sampling bounds"
        );
        Ok(())
    }
}
pub fn workflow(r: &SingingRequest, prefix: &str) -> Value {
    json!({
        "model":{"class_type":"UNETLoader","inputs":{"unet_name":"acestep_v1.5_turbo.safetensors","weight_dtype":"default"}},
        "clip":{"class_type":"DualCLIPLoader","inputs":{"clip_name1":"qwen_0.6b_ace15.safetensors","clip_name2":"qwen_4b_ace15.safetensors","type":"ace","device":"cpu"}},
        "vae":{"class_type":"VAELoader","inputs":{"vae_name":"ace_1.5_vae.safetensors"}},
        "condition":{"class_type":"TextEncodeAceStepAudio1.5","inputs":{"clip":["clip",0],"tags":r.tags,"lyrics":r.lyrics,"seed":r.seed,"bpm":r.bpm,"duration":r.duration,"timesignature":r.time_signature,"language":r.language,"keyscale":r.keyscale,"generate_audio_codes":r.generate_audio_codes,"cfg_scale":r.lyric_cfg,"temperature":r.temperature,"top_p":r.top_p,"top_k":r.top_k,"min_p":r.min_p}},
        "negative":{"class_type":"ConditioningZeroOut","inputs":{"conditioning":["condition",0]}},
        "sampling":{"class_type":"ModelSamplingAuraFlow","inputs":{"model":["model",0],"shift":3.0}},
        "latent":{"class_type":"EmptyAceStep1.5LatentAudio","inputs":{"seconds":r.duration,"batch_size":1}},
        "sampler":{"class_type":"KSampler","inputs":{"model":["sampling",0],"positive":["condition",0],"negative":["negative",0],"latent_image":["latent",0],"seed":r.seed,"steps":r.steps,"cfg":r.sampler_cfg,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
        "decode":{"class_type":"VAEDecodeAudio","inputs":{"samples":["sampler",0],"vae":["vae",0]}},
        "output":{"class_type":"SaveAudio","inputs":{"audio":["decode",0],"filename_prefix":prefix}}
    })
}
pub async fn execute(
    policy: &Policy,
    http: &PolicyHttpClient,
    config: &SingingConfig,
    ctx: &ToolContext,
    r: SingingRequest,
) -> Result<Value> {
    r.validate()?;
    if let Some(path) = &r.save_path {
        super::check_output(policy, path)?;
    }
    let base = config.base_url.trim_end_matches('/');
    let nodes = request(http, ctx, "GET", &format!("{base}/object_info"), None).await?;
    for node in [
        "ConditioningZeroOut",
        "DualCLIPLoader",
        "EmptyAceStep1.5LatentAudio",
        "KSampler",
        "ModelSamplingAuraFlow",
        "SaveAudio",
        "TextEncodeAceStepAudio1.5",
        "UNETLoader",
        "VAEDecodeAudio",
        "VAELoader",
    ] {
        ensure!(
            nodes.get(node).is_some(),
            "required ACE-Step node missing: {node}"
        );
    }
    for (folder, files) in [
        ("diffusion_models", vec!["acestep_v1.5_turbo.safetensors"]),
        (
            "text_encoders",
            vec!["qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors"],
        ),
        ("vae", vec!["ace_1.5_vae.safetensors"]),
    ] {
        let v = request(http, ctx, "GET", &format!("{base}/models/{folder}"), None).await?;
        let list = v
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("model inventory invalid"))?;
        for f in files {
            ensure!(
                list.iter().any(|v| v.as_str() == Some(f)),
                "required ACE-Step model missing"
            );
        }
    }
    if r.validate_only {
        return Ok(
            json!({"ready":true,"backend":"ace-step-1.5","gpu_verified":false,"model_inventory_verified":true}),
        );
    }
    // No model-specified graph, fullpath copy, URL, node type or output prefix.
    let prefix = format!("missy_singing_{}", uuid::Uuid::new_v4().simple());
    let response = request(
        http,
        ctx,
        "POST",
        &format!("{base}/prompt"),
        Some(json!({"prompt":workflow(&r,&prefix),"client_id":uuid::Uuid::new_v4().to_string()})),
    )
    .await?;
    let id = response
        .get("prompt_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("missing Comfy prompt id"))?;
    ensure!(
        !id.is_empty()
            && id.len() <= 128
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "invalid prompt id"
    );
    for _ in 0..config.max_polls {
        let history = request(http, ctx, "GET", &format!("{base}/history/{id}"), None).await?;
        if let Some(entry) = history.get(id) {
            ensure!(
                entry.pointer("/status/status_str").and_then(Value::as_str) != Some("error"),
                "ACE-Step execution failed"
            );
            let audio = entry.pointer("/outputs/output/audio");
            let descriptor = audio.and_then(|a| {
                a.as_array()
                    .and_then(|v| v.first())
                    .or_else(|| a.as_object().map(|_| a))
            });
            if let Some(d) = descriptor {
                let filename = d
                    .get("filename")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow::anyhow!("audio filename absent"))?;
                ensure!(
                    filename.starts_with(&prefix)
                        && filename.len() <= 255
                        && !filename.contains(['/', '\\'])
                        && !filename.contains(".."),
                    "untrusted audio filename"
                );
                let subfolder = d.get("subfolder").and_then(Value::as_str).unwrap_or("");
                ensure!(subfolder.is_empty(), "unexpected audio subfolder");
                ensure!(
                    d.get("type").and_then(Value::as_str).unwrap_or("output") == "output",
                    "audio must be output"
                );
                let mut url = url::Url::parse(&format!("{base}/view"))?;
                url.query_pairs_mut()
                    .append_pair("filename", filename)
                    .append_pair("subfolder", "")
                    .append_pair("type", "output");
                let response = tokio::select! { _ = ctx.cancellation.cancelled() => bail!("singing cancelled; remote job may continue"), r = http.get(url.as_str()) => r? };
                ensure!(response.status == 200, "audio retrieval failed");
                let (metadata, mime) = validate_audio(&response.body)?;
                let duration = metadata["duration_seconds"].as_f64().unwrap_or(0.0);
                ensure!(
                    (duration - r.duration).abs() <= 3.0,
                    "generated audio duration differs from requested score"
                );
                let artifact =
                    super::publish(policy, r.save_path.clone().unwrap(), &response.body, mime)?;
                return Ok(
                    json!({"backend":"ace-step-1.5","prompt_id":id,"mix":artifact,"audio":metadata,"vocal_stem":null,"instrumental_stem":null,"phoneme_timings":null,"lip_sync_safe":false}),
                );
            }
        }
        tokio::select! { _ = ctx.cancellation.cancelled() => bail!("singing cancelled; remote job may continue"), _ = tokio::time::sleep(Duration::from_millis(config.poll_interval_ms)) => {} }
    }
    bail!("singing completion deadline exceeded; remote job may continue")
}
async fn request(
    http: &PolicyHttpClient,
    ctx: &ToolContext,
    method: &str,
    url: &str,
    body: Option<Value>,
) -> Result<Value> {
    tokio::select! { _ = ctx.cancellation.cancelled() => bail!("singing cancelled; remote job may continue"), r = http.request_json(method,url,vec![],body) => r }
}
pub fn validate_wav(bytes: &[u8]) -> Result<Value> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))?;
    let spec = reader.spec();
    ensure!(
        (1..=2).contains(&spec.channels)
            && (8000..=192000).contains(&spec.sample_rate)
            && spec.sample_format == hound::SampleFormat::Int
            && spec.bits_per_sample <= 32,
        "unsupported audio format"
    );
    let frames = reader.duration();
    ensure!(
        frames > 0 && frames <= spec.sample_rate * 181,
        "audio duration out of bounds"
    );
    let mut count = 0u64;
    for s in reader.samples::<i32>() {
        s?;
        count += 1;
    }
    ensure!(
        count == frames as u64 * spec.channels as u64,
        "truncated WAV"
    );
    Ok(
        json!({"sample_rate":spec.sample_rate,"channels":spec.channels,"frames":frames,"duration_seconds":frames as f64/spec.sample_rate as f64}),
    )
}
pub fn validate_audio(bytes: &[u8]) -> Result<(Value, &'static str)> {
    if !bytes.starts_with(b"fLaC") {
        return Ok((validate_wav(bytes)?, "audio/wav"));
    }
    let mut reader = claxon::FlacReader::new(std::io::Cursor::new(bytes))?;
    let info = reader.streaminfo();
    ensure!(
        (1..=2).contains(&info.channels) && (8000..=192000).contains(&info.sample_rate),
        "unsupported FLAC parameters"
    );
    let frames = info
        .samples
        .ok_or_else(|| anyhow::anyhow!("FLAC duration required"))?;
    ensure!(
        frames > 0 && frames <= info.sample_rate as u64 * 181,
        "FLAC duration bounds"
    );
    let expected = frames * info.channels as u64;
    let mut count = 0;
    for s in reader.samples() {
        s?;
        count += 1;
        ensure!(count <= expected, "FLAC decoded sample overflow");
    }
    ensure!(count == expected, "truncated FLAC");
    Ok((
        json!({"sample_rate":info.sample_rate,"channels":info.channels,"frames":frames,"duration_seconds":frames as f64/info.sample_rate as f64}),
        "audio/flac",
    ))
}
pub fn schema() -> Value {
    super::schema(
        json!({"lyrics":{"type":"string","maxLength":16384},"tags":{"type":"string","maxLength":2048},"duration":{"type":"number","minimum":5,"maximum":180},"bpm":{"type":"integer","minimum":10,"maximum":300},"keyscale":{"type":"string"},"time_signature":{"enum":["2","3","4","6"]},"language":{"type":"string"},"seed":{"type":"integer","minimum":0,"maximum":4294967295u64},"steps":{"type":"integer","minimum":1,"maximum":50},"sampler_cfg":{"type":"number"},"lyric_cfg":{"type":"number"},"temperature":{"type":"number"},"top_p":{"type":"number"},"top_k":{"type":"integer"},"min_p":{"type":"number"},"generate_audio_codes":{"type":"boolean"},"validate_only":{"type":"boolean"},"save_path":{"type":"string"}}),
        &[],
    )
}
