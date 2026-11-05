//! Native bounded specialties. Embedders own durable approval and journaling.
#![cfg(target_os = "linux")]
mod calculator;
mod documents;
mod foundry;
mod music;
mod singing;
use anyhow::{Result, bail, ensure};
use async_trait::async_trait;
pub use documents::{DocumentBackendConfig, DocumentExecutablePin};
pub use foundry::FoundryConfig;
use missy_core::{PolicyConfig, ToolCall, ToolContext, ToolDefinition, ToolExecutor};
use missy_policy::{Policy, PolicyHttpClient};
pub use music::{Note, Score, Waveform};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
pub use singing::SingingConfig;
use std::path::PathBuf;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpecialtyConfig {
    pub enabled: bool,
    pub singing: Option<SingingConfig>,
    pub foundry: Option<FoundryConfig>,
    pub enable_music: bool,
    pub enable_documents: bool,
    pub document_backend: Option<DocumentBackendConfig>,
    // Zero selects conservative defaults.
    pub max_document_bytes: usize,
    pub max_audio_seconds: u32,
}
pub struct SpecialtyTools {
    policy: Policy,
    http: PolicyHttpClient,
    config: SpecialtyConfig,
    foundry_state: tokio::sync::Mutex<foundry::FoundryState>,
}
impl SpecialtyTools {
    pub fn new(policy: PolicyConfig, mut config: SpecialtyConfig) -> Result<Self> {
        if config.max_document_bytes == 0 {
            config.max_document_bytes = 1024 * 1024;
        }
        if config.max_audio_seconds == 0 {
            config.max_audio_seconds = 60;
        }
        ensure!(
            config.max_document_bytes <= 4 * 1024 * 1024,
            "invalid document cap"
        );
        ensure!(config.max_audio_seconds <= 120, "invalid audio cap");
        if let Some(c) = &config.singing {
            c.validate()?;
        }
        if let Some(c) = &config.foundry {
            c.validate()?;
        }
        if let Some(c) = &config.document_backend {
            c.validate()?;
        }
        let policy = Policy::new(policy)?;
        let http = PolicyHttpClient::from_policy(policy.clone());
        Ok(Self {
            policy,
            http,
            config,
            foundry_state: tokio::sync::Mutex::new(foundry::FoundryState::default()),
        })
    }
    /// Native byte-input OCR. Caller must enforce durable capture approval.
    /// No screen/file discovery; private staging obeys configured file policy.
    pub async fn ocr_png(&self, ctx: &ToolContext, png: &[u8]) -> Result<String> {
        ensure!(
            self.config.enabled && self.config.enable_documents,
            "OCR disabled"
        );
        let c = self
            .config
            .document_backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("OCR backend not configured"))?;
        documents::ocr_png(&self.policy, c, self.config.max_document_bytes, ctx, png).await
    }
    pub fn screen_ocr_configured(&self) -> bool {
        self.config.enabled
            && self.config.enable_documents
            && self
                .config
                .document_backend
                .as_ref()
                .is_some_and(|c| c.tesseract.is_some())
    }
    fn parse(&self, call: &ToolCall) -> Result<Request> {
        ensure!(self.config.enabled, "specialty tools disabled");
        Ok(match call.name.as_str() {
            "calculator" => Request::Calculator(serde_json::from_value(call.arguments.clone())?),
            "singing_generate" if self.config.singing.is_some() => {
                let r: singing::SingingRequest = serde_json::from_value(call.arguments.clone())?;
                r.validate()?;
                Request::Singing(r)
            }
            "repoeval_foundry_read" | "repoeval_foundry_mutate" | "repoeval_foundry"
                if self.config.foundry.is_some() =>
            {
                ensure!(
                    call.name == "repoeval_foundry_read"
                        || self.config.foundry.as_ref().unwrap().allow_mutations,
                    "Foundry mutation surface disabled"
                );
                let r: foundry::FoundryRequest = serde_json::from_value(call.arguments.clone())?;
                r.validate(self.config.foundry.as_ref().unwrap(), &call.name)?;
                Request::Foundry(r)
            }
            "music_compose" if self.config.enable_music => {
                let r: MusicRequest = serde_json::from_value(call.arguments.clone())?;
                r.score.validate(self.config.max_audio_seconds)?;
                ensure!(r.wav_output != r.midi_output, "output paths must differ");
                Request::Music(r)
            }
            "document_extract" if self.config.enable_documents => {
                Request::Document(serde_json::from_value(call.arguments.clone())?)
            }
            "document_pdf_extract" | "document_ocr"
                if self.config.enable_documents && self.config.document_backend.is_some() =>
            {
                let r: documents::DocumentRequest = serde_json::from_value(call.arguments.clone())?;
                r.validate(&call.name, self.config.document_backend.as_ref().unwrap())?;
                Request::NativeDocument(call.name.clone(), r)
            }
            _ => bail!("unknown or disabled specialty tool"),
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Calculation {
    expression: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MusicRequest {
    score: Score,
    wav_output: PathBuf,
    midi_output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DocumentRequest {
    input: PathBuf,
}
enum Request {
    NativeDocument(String, documents::DocumentRequest),
    Calculator(Calculation),
    Singing(singing::SingingRequest),
    Foundry(foundry::FoundryRequest),
    Music(MusicRequest),
    Document(DocumentRequest),
}
pub(crate) fn schema(properties: Value, required: &[&str]) -> Value {
    json!({"type":"object","additionalProperties":false,"properties":properties,"required":required})
}
fn def(name: &str, description: &str, parameters: Value, side_effect: bool) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: description.into(),
        parameters,
        side_effect,
    }
}
#[async_trait]
impl ToolExecutor for SpecialtyTools {
    fn definitions(&self) -> Vec<ToolDefinition> {
        if !self.config.enabled {
            return vec![];
        }
        let mut defs = vec![def(
            "calculator",
            "Native bounded finite real arithmetic + - * / // % ** and parentheses. No code evaluation, complex or bitwise support.",
            schema(
                json!({"expression":{"type":"string","maxLength":4096}}),
                &["expression"],
            ),
            false,
        )];
        if self.config.singing.is_some() {
            defs.push(def("singing_generate", "Actual ACE-Step 1.5 sung full-mix generation via operator-configured ComfyUI. No isolated stems or lip-sync timings. validate_only inspects node/model readiness. No TTS substitution.", singing::schema(), true));
        }
        if let Some(c) = &self.config.foundry {
            defs.push(def("repoeval_foundry_read", "Project/resource-scoped Foundry list, reviewed plan, status, artifacts, compare and report. Verified IDs; never clone or locally execute repository code.", foundry::schema(false), false));
            if c.allow_mutations {
                defs.push(def("repoeval_foundry_mutate", "Foundry snapshot/start/cancel for operator-allowlisted resources. Requires outer durable approval, explicit project acknowledgement and reviewed staging plan for start.", foundry::schema(true), true));
                defs.push(def("repoeval_foundry", "Compatibility Foundry read/plan/snapshot/start/cancel surface. All invocations conservatively require approval.", foundry::schema(true), true));
            }
        }
        if self.config.enable_music {
            defs.push(def("music_compose", "Render bounded explicit note score to real PCM WAV and standard MIDI. Deterministic oscillator synthesis, not vocals; no playback.", schema(json!({"score":music::schema(),"wav_output":{"type":"string"},"midi_output":{"type":"string"}}), &["score","wav_output","midi_output"]), true));
        }
        if self.config.enable_documents {
            defs.push(def("document_extract", "Read policy-authorized UTF-8 Markdown/text file and extract headings/fenced blocks without executing content. PDF/OCR not supported.", schema(json!({"input":{"type":"string"}}), &["input"]), false));
            if let Some(c) = &self.config.document_backend {
                defs.extend(documents::definitions(c));
            }
        }
        defs
    }
    fn configuration_fingerprint(&self) -> String {
        hex::encode(Sha256::digest(
            serde_json::to_vec(&(self.policy.config(), &self.config, self.definitions()))
                .expect("serializable specialty config"),
        ))
    }
    async fn preflight(&self, ctx: &ToolContext, call: &ToolCall) -> Result<()> {
        ensure!(!ctx.cancellation.is_cancelled(), "specialty cancelled");
        match self.parse(call)? {
            Request::Music(r) => {
                check_output(&self.policy, &r.wav_output)?;
                check_output(&self.policy, &r.midi_output)?;
            }
            Request::Singing(r) => {
                if let Some(path) = r.save_path {
                    check_output(&self.policy, &path)?;
                }
            }
            Request::Calculator(r) => {
                calculator::calculate(&r.expression)?;
            }
            _ => {}
        }
        Ok(())
    }
    async fn execute(&self, ctx: &ToolContext, call: &ToolCall) -> Result<Value> {
        ensure!(!ctx.cancellation.is_cancelled(), "specialty cancelled");
        match self.parse(call)? {
            Request::NativeDocument(name, r) => {
                documents::execute(
                    &self.policy,
                    self.config.document_backend.as_ref().unwrap(),
                    self.config.max_document_bytes,
                    ctx,
                    &name,
                    r,
                )
                .await
            }
            Request::Calculator(r) => Ok(json!({"result":calculator::calculate(&r.expression)?})),
            Request::Singing(r) => {
                singing::execute(
                    &self.policy,
                    &self.http,
                    self.config.singing.as_ref().unwrap(),
                    ctx,
                    r,
                )
                .await
            }
            Request::Foundry(r) => {
                let mut state = tokio::select! { _ = ctx.cancellation.cancelled() => bail!("Foundry cancelled"), s = self.foundry_state.lock() => s };
                foundry::execute(
                    &self.http,
                    self.config.foundry.as_ref().unwrap(),
                    &mut state,
                    ctx,
                    r,
                )
                .await
            }
            Request::Music(r) => {
                check_output(&self.policy, &r.wav_output)?;
                check_output(&self.policy, &r.midi_output)?;
                let (wav, midi) = music::render(&r.score)?;
                let wav = publish(&self.policy, r.wav_output, &wav, "audio/wav")?;
                let midi = publish(&self.policy, r.midi_output, &midi, "audio/midi")?;
                Ok(
                    json!({"wav":wav,"midi":midi,"synthesis":"deterministic oscillator, not vocals","duration_seconds":r.score.duration_seconds()}),
                )
            }
            Request::Document(r) => self.document(r.input),
        }
    }
}
impl SpecialtyTools {
    fn document(&self, input: PathBuf) -> Result<Value> {
        ensure!(
            matches!(
                input.extension().and_then(|e| e.to_str()),
                Some("md" | "markdown" | "txt")
            ),
            "only UTF-8 Markdown/text supported"
        );
        let bytes = self.policy.read_file(&input)?;
        ensure!(
            bytes.len() <= self.config.max_document_bytes,
            "document exceeds cap"
        );
        let text = String::from_utf8(bytes)?;
        ensure!(!text.contains('\0'), "binary document denied");
        let mut headings = Vec::new();
        let mut code = Vec::new();
        let mut fence: Option<(char, usize, String)> = None;
        for (n, line) in text.lines().enumerate() {
            let t = line.trim_start();
            let c = t.chars().next().unwrap_or(' ');
            let count = t.chars().take_while(|x| *x == c).count();
            if let Some((f, len, _)) = &fence {
                if c == *f && count >= *len && t[count..].trim().is_empty() {
                    let (_, _, body) = fence.take().unwrap();
                    code.push(body);
                } else {
                    fence.as_mut().unwrap().2.push_str(line);
                    fence.as_mut().unwrap().2.push('\n');
                }
            } else if matches!(c, '`' | '~') && count >= 3 {
                fence = Some((c, count, String::new()));
            } else {
                let level = t.bytes().take_while(|c| *c == b'#').count();
                if (1..=6).contains(&level) && t.as_bytes().get(level) == Some(&b' ') {
                    headings.push(json!({"line":n+1,"level":level,"text":t[level..].trim()}));
                }
            }
        }
        Ok(
            json!({"input":input,"text":text,"headings":headings,"code_blocks":code,"unterminated_fence":fence.is_some(),"format":"utf8_markdown_or_text"}),
        )
    }
}
pub(crate) fn publish(policy: &Policy, path: PathBuf, bytes: &[u8], mime: &str) -> Result<Value> {
    policy.write_file(&path, bytes)?;
    Ok(
        json!({"path":path,"bytes":bytes.len(),"sha256":hex::encode(Sha256::digest(bytes)),"mime_type":mime}),
    )
}
pub(crate) fn check_output(policy: &Policy, path: &std::path::Path) -> Result<()> {
    use std::path::Component;
    ensure!(
        path.is_absolute()
            && path
                .components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
            && !path
                .as_os_str()
                .as_encoded_bytes()
                .split(|c| *c == b'/')
                .any(|s| s == b"." || s == b"..")
            && policy
                .config()
                .write_roots
                .iter()
                .any(|r| path.starts_with(r) && path != r),
        "artifact path outside write policy"
    );
    Ok(())
}
pub(crate) fn validate_base(raw: &str, allow_api: bool) -> Result<()> {
    let u = url::Url::parse(raw)?;
    ensure!(
        matches!(u.scheme(), "http" | "https")
            && u.host_str().is_some()
            && u.username().is_empty()
            && u.password().is_none()
            && u.query().is_none()
            && u.fragment().is_none(),
        "invalid endpoint"
    );
    ensure!(
        u.path() == "/" || (allow_api && matches!(u.path(), "/api" | "/api/")),
        "endpoint must be origin (or /api for Foundry)"
    );
    Ok(())
}
#[cfg(test)]
mod tests;
