//! Explicit native acceptance: real WS/UDP, libdave MLS and Opus, synthetic audio.
//! The external sender is the pinned upstream test helper, never production code.
use super::*;
use std::{
    ffi::c_void,
    num::NonZeroU16,
    path::Path,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};

type Handle = *mut c_void;
type Create = unsafe extern "C" fn(u64) -> Handle;
type Destroy = unsafe extern "C" fn(Handle);
type Sender = unsafe extern "C" fn(Handle, *mut *mut u8, *mut usize);
type Add = unsafe extern "C" fn(Handle, u32, *mut u8, usize, *mut *mut u8, *mut usize);
type Split = unsafe extern "C" fn(
    Handle,
    *mut u8,
    usize,
    *mut *mut u8,
    *mut usize,
    *mut *mut u8,
    *mut usize,
);

struct External {
    handle: Handle,
    destroy: Destroy,
    sender: Sender,
    add: Add,
    split: Split,
    _library: libloading::Library,
}
impl Drop for External {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.handle);
        }
    }
}
impl External {
    fn load(channel: u64) -> Result<Self> {
        let path = std::env::var("MISSY_TEST_LIBDAVE_HELPER")?;
        ensure!(
            Path::new(&path).is_absolute(),
            "absolute upstream test helper required"
        );
        // SAFETY: operator-selected test-only binary, signatures from the pinned
        // upstream external_sender_wrapper.h; the owning library outlives handle.
        unsafe {
            let library = libloading::Library::new(path)?;
            let create = *library.get::<Create>(b"daveExternalSenderCreate\0")?;
            let result = Self {
                handle: create(channel),
                destroy: *library.get::<Destroy>(b"daveExternalSenderDestroy\0")?,
                sender: *library
                    .get::<Sender>(b"daveExternalSenderGetMarshalledExternalSender\0")?,
                add: *library.get::<Add>(b"daveExternalSenderProposeAdd\0")?,
                split: *library.get::<Split>(b"daveExternalSenderSplitCommitWelcome\0")?,
                _library: library,
            };
            ensure!(
                !result.handle.is_null(),
                "external sender allocation failed"
            );
            Ok(result)
        }
    }
    unsafe fn take(data: *mut u8, len: usize) -> Result<Vec<u8>> {
        ensure!(
            !data.is_null() && (1..=1_048_576).contains(&len),
            "invalid upstream fixture output"
        );
        // Official test helper uses malloc in CopyVectorToOutputBuffer.
        let bytes = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        unsafe {
            libc::free(data.cast());
        }
        Ok(bytes)
    }
    fn sender(&self) -> Result<Vec<u8>> {
        let (mut data, mut len) = (ptr::null_mut(), 0);
        unsafe {
            (self.sender)(self.handle, &mut data, &mut len);
            Self::take(data, len)
        }
    }
    fn add(&self, package: &mut [u8]) -> Result<Vec<u8>> {
        let (mut data, mut len) = (ptr::null_mut(), 0);
        unsafe {
            (self.add)(
                self.handle,
                0,
                package.as_mut_ptr(),
                package.len(),
                &mut data,
                &mut len,
            );
            Self::take(data, len)
        }
    }
    fn split(&self, combined: &mut [u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        let (mut commit, mut commit_len, mut welcome, mut welcome_len) =
            (ptr::null_mut(), 0, ptr::null_mut(), 0);
        unsafe {
            (self.split)(
                self.handle,
                combined.as_mut_ptr(),
                combined.len(),
                &mut commit,
                &mut commit_len,
                &mut welcome,
                &mut welcome_len,
            );
            Ok((
                Self::take(commit, commit_len)?,
                Self::take(welcome, welcome_len)?,
            ))
        }
    }
}

struct SpeechFixture {
    stt_calls: AtomicUsize,
    tts_calls: AtomicUsize,
}
#[async_trait]
impl Stt for SpeechFixture {
    async fn transcribe(
        &self,
        pcm: &[u8],
        format: AudioFormat,
        _: CancellationToken,
    ) -> Result<String> {
        assert_eq!(format.sample_rate, 48_000);
        assert_eq!(format.channels, 1);
        assert!(pcm.len() >= 28_800 && pcm.len() <= 960_000);
        let samples: Vec<_> = pcm
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert!(
            audio::speech_energy(&samples),
            "STT must see decoded non-silent PCM"
        );
        self.stt_calls.fetch_add(1, Ordering::SeqCst);
        Ok("synthetic speech request".into())
    }
}
#[async_trait]
impl Tts for SpeechFixture {
    async fn synthesize(
        &self,
        text: &str,
        _: CancellationToken,
    ) -> Result<missy_voice::SpeechAudio> {
        assert_eq!(text, "synthetic speech response");
        self.tts_calls.fetch_add(1, Ordering::SeqCst);
        let mut bytes = std::io::Cursor::new(Vec::new());
        {
            let mut wav = hound::WavWriter::new(
                &mut bytes,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            for i in 0..4800 {
                wav.write_sample(((i as f32 * 0.05).sin() * 8000.0) as i16)?;
            }
            wav.finalize()?;
        }
        Ok(missy_voice::SpeechAudio {
            wav: bytes.into_inner(),
            sample_rate: 48_000,
        })
    }
}
struct NoToolsFixture(mpsc::Sender<VoiceBinding>);
#[async_trait]
impl DiscordVoiceAgent for NoToolsFixture {
    async fn respond(
        &self,
        binding: &VoiceBinding,
        prompt: &str,
        _: CancellationToken,
    ) -> Result<String> {
        assert_eq!(binding.mode, crate::VoiceMode::NoTools);
        assert_eq!(binding.user_id, 4);
        assert_eq!(binding.scope.user, "u");
        assert_eq!(prompt, "synthetic speech request");
        self.0.send(binding.clone()).await?;
        Ok("synthetic speech response".into())
    }
}
struct GatewayFixture;
#[async_trait]
impl GatewayVoiceAdapter for GatewayFixture {
    async fn negotiate(
        &self,
        guild: u64,
        channel: u64,
        user: u64,
    ) -> Result<GatewayVoiceCredentials> {
        assert_eq!((guild, channel, user), (2, 3, 1));
        Ok(GatewayVoiceCredentials {
            guild_id: guild,
            channel_id: channel,
            user_id: user,
            endpoint: "fixture.invalid".into(),
            session_id: "synthetic-session".into(),
            token: "synthetic-token".into(),
            session_cancel: CancellationToken::new(),
        })
    }
    async fn leave(&self, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
}
type ServerWs = tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>;
async fn receive_op(ws: &mut ServerWs, expected: u8) -> Result<Message> {
    loop {
        let frame = timeout(Duration::from_secs(5), ws.next())
            .await?
            .ok_or_else(|| anyhow::anyhow!("fixture WS closed"))??;
        match &frame {
            Message::Text(text) => {
                let value: Value = serde_json::from_str(text)?;
                if value["op"] == 3 {
                    ws.send(message(6, value["d"].clone())).await?;
                    continue;
                }
                ensure!(
                    value["op"] == expected,
                    "unexpected fixture JSON opcode: {}",
                    value["op"]
                );
                return Ok(frame);
            }
            Message::Binary(bytes) => {
                ensure!(
                    bytes.first() == Some(&expected),
                    "unexpected fixture binary opcode"
                );
                return Ok(frame);
            }
            Message::Ping(bytes) => {
                ws.send(Message::Pong(bytes.clone())).await?;
            }
            _ => bail!("unexpected fixture WS frame"),
        }
    }
}
async fn server_binary(ws: &mut ServerWs, op: u8, body: &[u8]) -> Result<()> {
    let mut bytes = vec![0, 1, op];
    bytes.extend_from_slice(body);
    ws.send(Message::Binary(bytes.into())).await?;
    Ok(())
}

#[tokio::test]
#[ignore = "requires explicit pinned native libdave and upstream external-sender helper"]
async fn native_full_voice_ws_udp_speech_roundtrip() -> Result<()> {
    timeout(Duration::from_secs(15), native_roundtrip(false)).await?
}
#[tokio::test]
#[ignore = "requires explicit pinned native libdave and upstream external-sender helper"]
async fn native_signalling_downgrade_terminates_session() -> Result<()> {
    timeout(Duration::from_secs(15), native_roundtrip(true)).await?
}
async fn native_roundtrip(downgrade: bool) -> Result<()> {
    let mut config = crate::tests::config();
    config.native_dave.library_path = std::env::var("MISSY_TEST_LIBDAVE_PATH")?.into();
    config.native_dave.sha256 = std::env::var("MISSY_TEST_LIBDAVE_SHA256")?;
    config.bindings[0].mode = crate::VoiceMode::NoTools;
    config.silence_ms = 200;
    config.allow_private_udp = true;
    let mut denied = config.bindings[0].clone();
    denied.user_id = 5;
    denied.consent = false;
    config.bindings.push(denied);
    let mut wrong = config.bindings[0].clone();
    wrong.user_id = 7;
    wrong.scope.user = "other".into();
    config.bindings.push(wrong);
    let udp = UdpSocket::bind("127.0.0.1:0").await?;
    let udp_address = udp.local_addr()?;
    config.udp_port_min = udp_address.port();
    config.udp_port_max = udp_address.port();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let http = Arc::new(PolicyHttpClient::new(missy_core::PolicyConfig {
        allow_private_network: true,
        allowed_origins: vec![format!("http://{address}")],
        ..Default::default()
    })?);
    let speech = Arc::new(SpeechFixture {
        stt_calls: AtomicUsize::new(0),
        tts_calls: AtomicUsize::new(0),
    });
    let (agent_tx, mut agent_rx) = mpsc::channel(2);
    let agent = Arc::new(NoToolsFixture(agent_tx));
    let credentials = GatewayFixture.negotiate(2, 3, 1).await?;
    let gateway_cancel = credentials.session_cancel.clone();
    let cancel = CancellationToken::new();
    let (ready_tx, ready_rx) = oneshot::channel();
    let native_config = config.native_dave.clone();
    let task = tokio::spawn(run_at_loopback_url(
        Arc::new(config),
        http,
        credentials,
        speech.clone(),
        speech.clone(),
        agent,
        cancel.clone(),
        ready_tx,
        format!("ws://{address}/"),
    ));
    // Aborts on assertion/error too: no detached session task may survive a test.
    struct AbortOnDrop(JoinHandle<Result<()>>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let mut task = AbortOnDrop(task);
    let (stream, _) = listener.accept().await?;
    let mut ws = tokio_tungstenite::accept_async(stream).await?;
    let identify = receive_op(&mut ws, 0).await?;
    let identify: Value = serde_json::from_str(identify.to_text()?)?;
    assert_eq!(identify["d"]["max_dave_protocol_version"], 1);
    assert_eq!(identify["d"]["user_id"], "1");
    ws.send(message(8, json!({"heartbeat_interval":120000})))
        .await?;
    ws.send(message(
        2,
        json!({"ssrc":99,"ip":"127.0.0.1","port":udp_address.port(),"modes":[MODE]}),
    ))
    .await?;
    let mut discovery = [0; 128];
    let (n, client_address) =
        timeout(Duration::from_secs(3), udp.recv_from(&mut discovery)).await??;
    assert_eq!(&discovery[..n], &protocol::discovery_request(99));
    discovery[..2].copy_from_slice(&2u16.to_be_bytes());
    discovery[8..18].copy_from_slice(b"127.0.0.1\0");
    discovery[72..74].copy_from_slice(&client_address.port().to_be_bytes());
    udp.send_to(&discovery[..74], client_address).await?;
    let selected = receive_op(&mut ws, 1).await?;
    let selected: Value = serde_json::from_str(selected.to_text()?)?;
    assert_eq!(selected["d"]["data"]["mode"], MODE);
    ws.send(message(11, json!({"user_ids":["1","4"]}))).await?;
    ws.send(message(
        4,
        json!({"mode":MODE,"secret_key":vec![7u8;32],"dave_protocol_version":1}),
    ))
    .await?;
    let package = receive_op(&mut ws, 26).await?;
    ensure!(
        package.into_data().len() > 1,
        "missing service MLS key package"
    );

    let external = External::load(3)?;
    let sender = external.sender()?;
    let mut peer =
        missy_dave::DaveSession::new(&native_config, NonZeroU16::new(1).unwrap(), 4, 3, None)?;
    peer.set_external_sender(&sender)?;
    let mut peer_package = peer.create_key_package()?;
    server_binary(&mut ws, 25, &sender).await?;
    let proposal = external.add(&mut peer_package)?;
    assert_eq!(proposal.first(), Some(&0));
    server_binary(&mut ws, 27, &proposal).await?;
    let combined = receive_op(&mut ws, 28).await?;
    let mut combined = combined.into_data()[1..].to_vec();
    let (commit, welcome) = external.split(&mut combined)?;
    peer.process_welcome(&welcome, Some(&[1, 4]))?;
    peer.prepare_transition(0, 1)?;
    peer.execute_transition(0)?;
    let mut commit_body = vec![0, 0];
    commit_body.extend_from_slice(&commit);
    server_binary(&mut ws, 29, &commit_body).await?;
    timeout(Duration::from_secs(3), ready_rx).await??;

    // Numeric protocol fields must not wrap into valid versions/transitions.
    let mut protocol_peer = Dave::new(&native_config, 1, 7, 3)?;
    assert!(
        protocol_peer
            .json(21, &json!({"protocol_version":65537,"transition_id":0}))
            .is_err()
    );
    assert!(
        protocol_peer
            .json(21, &json!({"protocol_version":1,"transition_id":65536}))
            .is_err()
    );
    assert!(
        protocol_peer
            .json(21, &json!({"protocol_version":0,"transition_id":0}))
            .is_err()
    );
    assert!(
        protocol_peer
            .json(22, &json!({"transition_id":65536}))
            .is_err()
    );

    let mut encoder =
        opus2::Encoder::new(48_000, opus2::Channels::Stereo, opus2::Application::Voip)?;
    let samples: Vec<i16> = (0..1920)
        .map(|i| ((i as f32 * 0.04).sin() * 9000.0) as i16)
        .collect();
    let opus = encoder.encode_vec(&samples, 4000)?;
    let mut sender_cipher = RtpCipher::new(&[7; 32], 45)?;
    // Wire SSRC says user 7, but DAVE authenticates user 4. Must never reach STT.
    for (ssrc, user) in [(42, 7), (43, 5), (44, 88)] {
        ws.send(message(
            5,
            json!({"ssrc":ssrc,"user_id":user.to_string(),"speaking":1}),
        ))
        .await?;
        tokio::time::sleep(Duration::from_millis(25)).await;
        let mut tx = RtpCipher::new(&[7; 32], ssrc)?;
        for _ in 0..20 {
            let encrypted = peer.encrypt_opus(ssrc, &opus)?;
            udp.send_to(&tx.encode(&encrypted)?, client_address).await?;
        }
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        speech.stt_calls.load(Ordering::SeqCst),
        0,
        "wrong sender/no consent reached STT"
    );
    assert!(agent_rx.try_recv().is_err());
    ws.send(message(5, json!({"ssrc":45,"user_id":"4","speaking":1})))
        .await?;
    // Ensure signalling has been processed before UDP; separate transports have
    // no ordering guarantee. Replayed/invalid/cleartext payloads are dropped.
    tokio::time::sleep(Duration::from_millis(40)).await;
    // Enough packets to trigger an utterance if *any* rejected class leaks
    // through. One authentic packet plus duplicates must remain sub-threshold.
    let encrypted = peer.encrypt_opus(45, &opus)?;
    let replayed = sender_cipher.encode(&encrypted)?;
    udp.send_to(&replayed, client_address).await?;
    for _ in 0..20 {
        let fresh = sender_cipher.encode(&peer.encrypt_opus(45, &opus)?)?;
        let mut bad_nonce = fresh.clone();
        let last = bad_nonce.len() - 1;
        bad_nonce[last] ^= 1;
        udp.send_to(&bad_nonce, client_address).await?;
        udp.send_to(&replayed, client_address).await?;
        udp.send_to(&sender_cipher.encode(&opus)?, client_address)
            .await?;
        let mut tampered_dave = peer.encrypt_opus(45, &opus)?;
        tampered_dave[0] ^= 1;
        udp.send_to(&sender_cipher.encode(&tampered_dave)?, client_address)
            .await?;
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        speech.stt_calls.load(Ordering::SeqCst),
        0,
        "bad nonce/replay/plaintext/tampered media reached STT"
    );
    assert!(agent_rx.try_recv().is_err());
    for i in 0..20 {
        let wire = sender_cipher.encode(&peer.encrypt_opus(45, &opus)?)?;
        udp.send_to(&wire, client_address).await?;
        if i == 0 {
            udp.send_to(&wire, client_address).await?;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let binding = timeout(Duration::from_secs(3), agent_rx.recv())
        .await?
        .ok_or_else(|| anyhow::anyhow!("no speech turn"))?;
    assert_eq!(binding.mode, crate::VoiceMode::NoTools);
    assert_eq!(speech.stt_calls.load(Ordering::SeqCst), 1);
    let mut receiver = RtpCipher::new(&[7; 32], 42)?;
    let mut decoder = opus2::Decoder::new(48_000, opus2::Channels::Stereo)?;
    let mut wire = [0; 4096];
    let n = timeout(Duration::from_secs(3), udp.recv(&mut wire)).await??;
    let (ssrc, encrypted) = receiver.decode(&wire[..n])?;
    assert_eq!(ssrc, 99);
    let opus = peer.decrypt_opus(1, &encrypted)?;
    assert_ne!(opus, protocol::SILENCE);
    let mut pcm = [0; 1920];
    assert_eq!(decoder.decode(&opus, &mut pcm, false)?, 960);
    assert!(
        audio::speech_energy(&pcm),
        "response must decode to actual audio"
    );
    assert_eq!(speech.tts_calls.load(Ordering::SeqCst), 1);
    let first_packet = Instant::now();
    let mut silence_packets = 0;
    for index in 1..10 {
        let n = timeout(Duration::from_secs(2), udp.recv(&mut wire)).await??;
        let (ssrc, encrypted) = receiver.decode(&wire[..n])?;
        assert_eq!(ssrc, 99);
        let opus = peer.decrypt_opus(1, &encrypted)?;
        if index >= 5 {
            assert_eq!(opus, protocol::SILENCE);
            silence_packets += 1;
        } else {
            assert_eq!(decoder.decode(&opus, &mut pcm, false)?, 960);
        }
    }
    assert_eq!(
        silence_packets, 5,
        "exactly five encrypted terminal silence frames"
    );
    assert!(
        first_packet.elapsed() >= Duration::from_millis(120),
        "outbound audio must be paced, not burst"
    );
    for expected in [1, 0] {
        let speaking = receive_op(&mut ws, 5).await?;
        let speaking: Value = serde_json::from_str(speaking.to_text()?)?;
        assert_eq!(speaking["d"]["speaking"], expected);
        assert_eq!(speaking["d"]["ssrc"], 99);
    }
    // Both trust invalidation and signalling downgrade tear down the real loop.
    if downgrade {
        ws.send(message(21, json!({"protocol_version":0,"transition_id":3})))
            .await?;
    } else {
        gateway_cancel.cancel();
    }
    let result = timeout(Duration::from_secs(2), &mut task.0).await??;
    let expected = if downgrade {
        "DAVE downgrade denied"
    } else {
        "Gateway voice session invalidated"
    };
    assert!(result.unwrap_err().to_string().contains(expected));
    Ok(())
}
