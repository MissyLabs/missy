use async_trait::async_trait;
use missy_core::{Scope, ToolCall, ToolContext, ToolExecutor};
use missy_voice::{
    AudioApprovalGate, AudioDeviceConfig, AudioDeviceTools, AudioFormat, SpeechAudio, Stt, Tts,
    audio_device::{AudioDeviceKind, AudioExecutable},
    speech::SpeechOutput,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokio_util::sync::CancellationToken;
struct Gate(AtomicUsize);
#[async_trait]
impl AudioApprovalGate for Gate {
    async fn approve(&self, _: &ToolContext, _: &ToolCall) -> anyhow::Result<bool> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    }
}
struct Engine;
fn wav() -> Vec<u8> {
    let mut b = b"RIFF".to_vec();
    b.extend_from_slice(&2036u32.to_le_bytes());
    b.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
    b.extend_from_slice(&16000u32.to_le_bytes());
    b.extend_from_slice(&32000u32.to_le_bytes());
    b.extend_from_slice(b"\x02\0\x10\0data");
    b.extend_from_slice(&2000u32.to_le_bytes());
    b.extend_from_slice(&[0, 1].repeat(1000));
    b
}
#[async_trait]
impl Tts for Engine {
    async fn synthesize(&self, _: &str, _: CancellationToken) -> anyhow::Result<SpeechAudio> {
        Ok(SpeechAudio {
            wav: wav(),
            sample_rate: 16000,
        })
    }
}
#[async_trait]
impl Stt for Engine {
    async fn transcribe(
        &self,
        pcm: &[u8],
        f: AudioFormat,
        _: CancellationToken,
    ) -> anyhow::Result<String> {
        assert_eq!(pcm.len(), 32000);
        assert_eq!(f.sample_rate, 16000);
        Ok("synthetic input only".into())
    }
}
fn ctx() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "p".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "turn".into(),
        cancellation: CancellationToken::new(),
    }
}
fn call(args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: "audio-call".into(),
        name: "audio_device".into(),
        arguments: args,
    }
}
fn fixture() -> (tempfile::TempDir, AudioDeviceConfig) {
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join("audio-fixture");
    let source = dir.path().join("fixture.rs");
    std::fs::write(&source, include_str!("fixtures/audio_device.rs")).unwrap();
    assert!(
        std::process::Command::new("rustc")
            .arg("--edition=2024")
            .arg(&source)
            .arg("-o")
            .arg(&bin)
            .status()
            .unwrap()
            .success()
    );
    let pin = AudioExecutable {
        sha256: format!("{:x}", Sha256::digest(std::fs::read(&bin).unwrap())),
        path: bin,
    };
    let cfg = AudioDeviceConfig {
        enabled: true,
        pulse_server: "unix:/isolated-test-only".into(),
        pactl: Some(pin.clone()),
        paplay: Some(pin.clone()),
        parec: Some(pin),
        sinks: [("virtual".into(), AudioDeviceKind::Virtual)].into(),
        sources: [("virtual".into(), AudioDeviceKind::Virtual)].into(),
        virtual_routes: vec!["virtual".into()],
        max_volume_pct: 40,
        ..Default::default()
    };
    (dir, cfg)
}
#[tokio::test]
async fn fixture_discovery_routing_output_and_bounded_input() {
    let (_dir, cfg) = fixture();
    let gate = Arc::new(Gate(AtomicUsize::new(0)));
    let tools = AudioDeviceTools::new(cfg.clone())
        .unwrap()
        .with_tts(Arc::new(Engine))
        .with_stt(Arc::new(Engine));
    assert!(
        tools
            .execute(
                &ctx(),
                &call(json!({"action":"discover","direction":"input"}))
            )
            .await
            .is_err()
    );
    let tools = tools.with_approval(gate.clone());
    let list = tools
        .execute(
            &ctx(),
            &call(json!({"action":"discover","direction":"input"})),
        )
        .await
        .unwrap();
    assert_eq!(list["devices"].as_array().unwrap().len(), 1);
    assert!(!list.to_string().contains("ungranted"));
    assert_eq!(
        tools
            .execute(
                &ctx(),
                &call(json!({"action":"test_route","sink":"virtual"}))
            )
            .await
            .unwrap()["played"],
        false
    );
    assert_eq!(
        tools
            .execute(
                &ctx(),
                &call(json!({"action":"create_route","sink":"virtual"}))
            )
            .await
            .unwrap()["default_changed"],
        false
    );
    tools
        .execute(
            &ctx(),
            &call(json!({"action":"set_volume","sink":"virtual","volume_pct":20})),
        )
        .await
        .unwrap();
    tools.execute(&ctx(),&call(json!({"action":"speak","sink":"virtual","volume_pct":20,"text":"not a shell; $(no)"}))).await.unwrap();
    assert_eq!(
        tools
            .execute(
                &ctx(),
                &call(json!({"action":"listen","source":"virtual","seconds":1}))
            )
            .await
            .unwrap()["transcript"],
        "synthetic input only"
    );
    let used = gate.0.load(Ordering::SeqCst);
    for args in [
        json!({"action":"set_volume","sink":"virtual","volume_pct":81}),
        json!({"action":"listen","source":"virtual","seconds":31}),
        json!({"action":"test_route","sink":"--default"}),
        json!({"action":"discover","direction":"input","confirm":true}),
    ] {
        assert!(tools.execute(&ctx(), &call(args)).await.is_err());
    }
    assert_eq!(gate.0.load(Ordering::SeqCst), used);
    let mut cfg = cfg;
    cfg.sinks
        .insert("physical".into(), AudioDeviceKind::Physical);
    assert!(
        AudioDeviceTools::new(cfg)
            .unwrap()
            .validate_output("physical", 10)
            .is_err()
    );
}
