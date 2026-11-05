//! Audio-derived animation, not simulated microphone tracking. One authenticated
//! policy-pinned VTube connection; monotonic deadlines and explicit mouth reset.
use crate::{IntegrationTools, desktop_ws::Vtube};
use anyhow::{Context, Result, ensure};
use missy_core::{ToolCall, ToolContext};
use missy_voice::{SpeechOutput, Tts, speech::decode_pcm_wav};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};
use tokio::sync::oneshot;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Speak {
    pub text: String,
    pub sink: String,
    pub volume_pct: u8,
    pub parameter: String,
}
impl IntegrationTools {
    /// SpeechOutput must consume the exact durable vtube_speak approval once;
    /// it covers output and animation together, not two independent approvals.
    pub fn with_speech(mut self, tts: Arc<dyn Tts>, output: Arc<dyn SpeechOutput>) -> Self {
        self.fingerprint.push_str(&format!(
            ":speech-v1:{}",
            output.configuration_fingerprint()
        ));
        self.speech = Some((tts, output));
        self
    }
}
impl Vtube {
    pub(crate) fn validate_speech(&self, a: &Speak) -> Result<()> {
        let range = self
            .cfg
            .parameters
            .get(&a.parameter)
            .context("mouth parameter not granted")?;
        ensure!(
            self.conn.cfg.allow_mutations && range.min <= 0.0 && range.max >= 0.85,
            "mouth range must permit 0..0.85"
        );
        ensure!(
            !a.text.trim().is_empty() && a.text.len() <= 8192 && !a.text.contains('\0'),
            "invalid speech text"
        );
        Ok(())
    }
    pub(crate) async fn speak(
        &self,
        ctx: &ToolContext,
        call: &ToolCall,
        a: Speak,
        tts: &dyn Tts,
        output: &dyn SpeechOutput,
    ) -> Result<Value> {
        self.validate_speech(&a)?;
        output.validate_output(&a.sink, a.volume_pct)?;
        let cancel = ctx.cancellation.child_token();
        let audio = tokio::select! {biased;_=cancel.cancelled()=>anyhow::bail!("speech cancelled"),r=tokio::time::timeout(Duration::from_secs(30),tts.synthesize(&a.text,cancel.child_token()))=>r.context("speech synthesis timeout")??};
        let wav = decode_pcm_wav(&audio.wav)?;
        ensure!(
            wav.duration() <= Duration::from_secs(120),
            "speech exceeds 120 seconds"
        );
        let levels = wav.mouth_envelope();
        let duration = wav.duration();
        let mut ws = tokio::select! {biased;_=cancel.cancelled()=>anyhow::bail!("speech cancelled"),r=tokio::time::timeout(Duration::from_secs(5),self.connect())=>r.context("avatar connection timeout")??};
        let (started, rx) = oneshot::channel();
        let mut did_start = false;
        let play = output.play(
            ctx,
            call,
            &a.sink,
            a.volume_pct,
            &audio,
            cancel.child_token(),
            started,
        );
        let animation = async {
            // The adapter signals child start after approval and process spawn,
            // not before. This is process-start synchronization, not a hardware
            // DAC clock guarantee; the backend may buffer playback.
            let start = rx.await.context("playback did not start")?;
            did_start = true;
            for (i, value) in levels.iter().enumerate() {
                tokio::time::sleep_until(start + Duration::from_millis(i as u64 * 50)).await;
                // Skip stale animation after backend stalls; never burst old
                // envelope values into a recovered connection.
                if tokio::time::Instant::now() > start + Duration::from_millis(i as u64 * 50 + 150)
                {
                    continue;
                }
                self.request(&mut ws,"InjectParameterDataRequest",json!({"faceFound":false,"mode":"set","parameterValues":[{"id":a.parameter,"value":value,"weight":1.0}]})).await?;
            }
            tokio::time::sleep_until(start + duration).await;
            Ok::<_, anyhow::Error>(())
        };
        let result = tokio::select! {biased;_=cancel.cancelled()=>Err(anyhow::anyhow!("speech cancelled")),r=tokio::time::timeout(Duration::from_secs(125),async{tokio::try_join!(play,animation).map(|_|())})=>r.context("speech playback timeout").and_then(|r|r)};
        cancel.cancel();
        // Best effort bounded cleanup, including failure/cancellation. Never
        // retries uncertain playback or reopens another animation connection.
        if !did_start {
            result?;
            anyhow::bail!("playback never started");
        }
        let reset=tokio::time::timeout(Duration::from_secs(2),self.request(&mut ws,"InjectParameterDataRequest",json!({"faceFound":false,"mode":"set","parameterValues":[{"id":a.parameter,"value":0.0,"weight":1.0}]}))).await;
        result?;
        reset.context("mouth reset timeout")??;
        Ok(
            json!({"played":true,"sink":a.sink,"parameter":a.parameter,"envelope_frames":levels.len(),"duration_ms":duration.as_millis()}),
        )
    }
}
