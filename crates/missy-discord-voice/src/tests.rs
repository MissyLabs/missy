use super::*;
use crate::protocol::*;
use serde_json::json;
#[test]
fn energy_gate_speech_silence_quiet_noise() {
    assert!(!audio::speech_energy(&[]));
    assert!(!audio::speech_energy(&[0; 1920]));
    assert!(!audio::speech_energy(&[10; 1920]));
    assert!(audio::speech_energy(&[1000; 1920]));
}
use std::{collections::BTreeSet, io::Cursor, path::PathBuf};
pub(crate) fn config() -> DiscordVoiceConfig {
    DiscordVoiceConfig {
        native_dave: missy_dave::NativeDaveConfig {
            library_path: PathBuf::from("/explicit/libdave.so"),
            sha256: "a".repeat(64),
            version: "1.2.2/cpp".into(),
            source_commit: "6396c90aedbb9a6ee117ea9ce1ce9832af6a3375".into(),
        },
        enabled: true,
        bot_user_id: 1,
        bindings: vec![VoiceBinding {
            guild_id: 2,
            channel_id: 3,
            user_id: 4,
            consent: true,
            mode: VoiceMode::SafeChat,
            scope: missy_core::Scope {
                tenant: "t".into(),
                user: "u".into(),
                channel: "c".into(),
                session: "s".into(),
            },
        }],
        endpoint_hosts: BTreeSet::from(["voice.discord.gg".into()]),
        udp_port_min: 1000,
        udp_port_max: 65535,
        allow_private_udp: false,
        max_utterance_seconds: 10,
        silence_ms: 1500,
        max_users: 8,
        max_reply_seconds: 30,
    }
}
#[test]
fn exact_consent_scope_and_bounds() {
    let mut c = config();
    assert!(c.validate().is_ok());
    assert!(c.binding(2, 3, 4).is_some());
    assert!(c.binding(2, 3, 5).is_none());
    assert!(c.binding(2, 5, 4).is_none());
    c.bindings[0].consent = false;
    assert!(c.binding(2, 3, 4).is_none());
    c.max_users = 0;
    assert!(c.validate().is_err());
}
#[test]
fn udp_policy() {
    let c = config();
    for ip in [
        "127.0.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "0.0.0.0",
        "224.0.0.1",
        "100.64.0.1",
        "198.18.0.1",
        "192.0.2.1",
        "::1",
        "::ffff:8.8.8.8",
        "2001:db8::1",
    ] {
        assert!(c.udp_address(ip, 5000).is_err(), "{ip}");
    }
    assert!(c.udp_address("8.8.8.8", 5000).is_ok());
    assert!(c.udp_address("8.8.8.8", 80).is_err());
}
#[tokio::test(start_paused = true)]
async fn heartbeat_stale_wrong_ack_reset() {
    let mut h = Heartbeat::default();
    h.hello(100).unwrap();
    assert!(h.tick(-1).unwrap().is_some());
    assert!(h.ack(&json!({"t":99})).is_err());
    tokio::time::advance(std::time::Duration::from_millis(101)).await;
    assert!(h.tick(1).is_err());
    h.hello(100).unwrap();
    assert!(h.tick(1).unwrap().is_some());
    h.ack(&json!({"t":2})).unwrap();
    tokio::time::advance(std::time::Duration::from_millis(101)).await;
    assert!(h.tick(1).unwrap().is_some());
}
#[test]
fn replay_window() {
    let mut r = Replay::default();
    assert!(r.accept(100));
    assert!(r.accept(99));
    assert!(!r.accept(99));
    assert!(!r.accept(1));
    assert!(r.accept(200));
    assert!(!r.accept(100));
}
#[test]
fn rtp_authentication_replay_silence() {
    let mut tx = RtpCipher::new(&[7; 32], 42).unwrap();
    let mut rx = RtpCipher::new(&[7; 32], 3).unwrap();
    let wire = tx.encode(SILENCE).unwrap();
    let mut bad = wire.clone();
    bad[8] ^= 1;
    assert!(rx.decode(&bad).is_err());
    assert_eq!(rx.decode(&wire).unwrap(), (42, SILENCE.to_vec()));
    assert!(rx.decode(&wire).is_err());
    let mut wrong = RtpCipher::new(&[8; 32], 3).unwrap();
    assert!(wrong.decode(&tx.encode(&[1, 2, 3]).unwrap()).is_err());
    for n in 0..32 {
        assert!(rx.decode(&vec![0; n]).is_err());
    }
}
#[tokio::test]
async fn udp_discovery_fixture() {
    let server = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let addr = server.local_addr().unwrap();
    let peer = tokio::spawn(async move {
        let mut b = [0; 128];
        let (n, a) = server.recv_from(&mut b).await.unwrap();
        assert_eq!(&b[..n], discovery_request(42));
        b[0..2].copy_from_slice(&2u16.to_be_bytes());
        b[8..18].copy_from_slice(b"127.0.0.1\0");
        b[72..74].copy_from_slice(&a.port().to_be_bytes());
        server.send_to(&b[..74], a).await.unwrap();
    });
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    socket.connect(addr).await.unwrap();
    socket.send(&discovery_request(42)).await.unwrap();
    let mut b = [0; 128];
    let n = socket.recv(&mut b).await.unwrap();
    assert_eq!(discovery_response(&b[..n], 42).unwrap().0, "127.0.0.1");
    assert!(discovery_response(&b[..n], 43).is_err());
    peer.await.unwrap();
}
#[test]
fn opus_pcm_wav_bounds() {
    let samples: Vec<i16> = (0..1920)
        .map(|i| ((i as f32 / 10.0).sin() * 8000.0) as i16)
        .collect();
    let mut enc =
        opus2::Encoder::new(48_000, opus2::Channels::Stereo, opus2::Application::Voip).unwrap();
    let packet = enc.encode_vec(&samples, 4000).unwrap();
    let mut dec = opus2::Decoder::new(48_000, opus2::Channels::Stereo).unwrap();
    let mut pcm = [0; 1920];
    assert_eq!(dec.decode(&packet, &mut pcm, false).unwrap(), 960);
    assert_eq!(audio::mono(&pcm).len(), 1920);
    let mut bytes = Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..320 {
            w.write_sample(100i16).unwrap();
        }
        w.finalize().unwrap();
    }
    let frames = audio::wav_frames(bytes.get_ref(), 1).unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), 1920);
    assert!(audio::wav_frames(&vec![0; 1_000_000], 1).is_err());
}
#[test]
fn credentials_debug_redacted() {
    let c = GatewayVoiceCredentials {
        guild_id: 1,
        channel_id: 2,
        user_id: 3,
        endpoint: "secret-host".into(),
        session_id: "secret-session".into(),
        token: "secret-token".into(),
        session_cancel: tokio_util::sync::CancellationToken::new(),
    };
    assert!(!format!("{c:?}").contains("secret"));
}
