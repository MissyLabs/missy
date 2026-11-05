use super::*;
struct Speech;
#[async_trait]
impl Stt for Speech {
    async fn transcribe(&self, pcm: &[u8], _: AudioFormat, _: CancellationToken) -> Result<String> {
        Ok(format!("{}", pcm[0]))
    }
}
#[async_trait]
impl Tts for Speech {
    async fn synthesize(&self, _: &str, _: CancellationToken) -> Result<missy_voice::SpeechAudio> {
        Ok(missy_voice::SpeechAudio {
            wav: Vec::new(),
            sample_rate: 48_000,
        })
    }
}
struct Agent(mpsc::Sender<(u64, String)>);
#[async_trait]
impl DiscordVoiceAgent for Agent {
    async fn respond(
        &self,
        binding: &VoiceBinding,
        prompt: &str,
        _: CancellationToken,
    ) -> Result<String> {
        self.0.send((binding.user_id, prompt.into())).await.unwrap();
        Ok("reply".into())
    }
}
struct Gateway;
struct WavSpeech;
struct SayGate;
struct DenyGate;
#[async_trait]
impl crate::VoiceApprovalGate for DenyGate {
    async fn approve(&self, _: &missy_core::ToolContext, _: &missy_core::ToolCall) -> Result<bool> {
        Ok(false)
    }
}
#[tokio::test]
async fn model_join_leave_status_require_exact_scope_and_durable_approval() {
    use missy_core::ToolExecutor;
    let (tx, _) = mpsc::channel(1);
    let config = crate::tests::config();
    let scope = config.bindings[0].scope.clone();
    let service = Arc::new(
        DiscordVoiceService::new(
            config,
            Arc::new(PolicyHttpClient::new(Default::default()).unwrap()),
            Arc::new(Gateway),
            Arc::new(Speech),
            Arc::new(WavSpeech),
            Arc::new(Agent(tx)),
        )
        .unwrap(),
    );
    let tools = crate::DiscordVoiceSayTools::new(service, Arc::new(DenyGate));
    let mut ctx = missy_core::ToolContext {
        scope,
        turn_id: "fixture".into(),
        cancellation: CancellationToken::new(),
    };
    for name in [
        "discord_voice_join",
        "discord_voice_leave",
        "discord_voice_status",
    ] {
        assert!(tools.definitions().iter().any(|d| d.name == name));
        let call = missy_core::ToolCall {
            id: "fixture".into(),
            name: name.into(),
            arguments: json!({"guild_id":2,"channel_id":3}),
        };
        if name == "discord_voice_status" {
            assert_eq!(
                tools.execute(&ctx, &call).await.unwrap()["state"],
                "not_joined"
            );
        } else {
            assert!(
                tools
                    .execute(&ctx, &call)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("durable exact")
            );
        }
        let old = ctx.scope.user.clone();
        ctx.scope.user = "foreign".into();
        assert!(tools.preflight(&ctx, &call).await.is_err());
        ctx.scope.user = old;
        let mut override_call = call;
        override_call.arguments["actor"] = json!(4);
        assert!(tools.preflight(&ctx, &override_call).await.is_err());
    }
}
#[async_trait]
impl crate::VoiceApprovalGate for SayGate {
    async fn approve(&self, _: &missy_core::ToolContext, _: &missy_core::ToolCall) -> Result<bool> {
        Ok(true)
    }
}
#[tokio::test]
async fn say_tool_requires_exact_scope_and_never_allows_actor_override() {
    use missy_core::ToolExecutor;
    let (tx, _) = mpsc::channel(1);
    let config = crate::tests::config();
    let scope = config.bindings[0].scope.clone();
    let service = Arc::new(
        DiscordVoiceService::new(
            config,
            Arc::new(PolicyHttpClient::new(Default::default()).unwrap()),
            Arc::new(Gateway),
            Arc::new(Speech),
            Arc::new(WavSpeech),
            Arc::new(Agent(tx)),
        )
        .unwrap(),
    );
    let (output, mut rx) = mpsc::channel(2);
    let cancel = CancellationToken::new();
    let child = cancel.clone();
    let task = tokio::spawn(async move {
        child.cancelled().await;
        Ok(())
    });
    service.sessions.lock().await.insert(
        2,
        Session {
            channel: 3,
            cancel: cancel.clone(),
            task,
            text_output: output,
        },
    );
    let tools = crate::DiscordVoiceSayTools::new(service, Arc::new(SayGate));
    let mut context = missy_core::ToolContext {
        scope,
        turn_id: "test".into(),
        cancellation: CancellationToken::new(),
    };
    let mut call = missy_core::ToolCall {
        id: "say".into(),
        name: "discord_voice_say".into(),
        arguments: json!({"guild_id":2,"channel_id":3,"text":"synthetic"}),
    };
    assert_eq!(
        tools.execute(&context, &call).await.unwrap()["queued"],
        true
    );
    assert!(rx.recv().await.is_some());
    context.scope.user = "different".into();
    assert!(tools.preflight(&context, &call).await.is_err());
    call.arguments["actor"] = json!(4);
    assert!(tools.preflight(&context, &call).await.is_err());
    cancel.cancel();
}
#[async_trait]
impl Tts for WavSpeech {
    async fn synthesize(&self, _: &str, _: CancellationToken) -> Result<missy_voice::SpeechAudio> {
        let mut b = b"RIFF".to_vec();
        b.extend_from_slice(&1956u32.to_le_bytes());
        b.extend_from_slice(b"WAVEfmt \x10\0\0\0\x01\0\x01\0");
        b.extend_from_slice(&48000u32.to_le_bytes());
        b.extend_from_slice(&96000u32.to_le_bytes());
        b.extend_from_slice(b"\x02\0\x10\0data");
        b.extend_from_slice(&1920u32.to_le_bytes());
        b.extend_from_slice(&[0, 1].repeat(960));
        Ok(missy_voice::SpeechAudio {
            wav: b,
            sample_rate: 48000,
        })
    }
}
#[tokio::test]
async fn text_speech_is_bound_bounded_and_does_not_autojoin() {
    let (tx, _) = mpsc::channel(1);
    let service = DiscordVoiceService::new(
        crate::tests::config(),
        Arc::new(PolicyHttpClient::new(Default::default()).unwrap()),
        Arc::new(Gateway),
        Arc::new(Speech),
        Arc::new(WavSpeech),
        Arc::new(Agent(tx)),
    )
    .unwrap();
    assert!(
        service
            .speak_text(2, 3, 4, "not joined", CancellationToken::new())
            .await
            .is_err()
    );
    let (output, mut rx) = mpsc::channel(2);
    let cancel = CancellationToken::new();
    let child = cancel.clone();
    let task = tokio::spawn(async move {
        child.cancelled().await;
        Ok(())
    });
    service.sessions.lock().await.insert(
        2,
        Session {
            channel: 3,
            cancel: cancel.clone(),
            task,
            text_output: output,
        },
    );
    service
        .speak_text(2, 3, 4, "synthetic speech", CancellationToken::new())
        .await
        .unwrap();
    let frames = rx.recv().await.unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), 1920);
    assert!(
        service
            .speak_text(2, 3, 99, "bad actor", CancellationToken::new())
            .await
            .is_err()
    );
    assert!(
        service
            .speak_text(2, 99, 4, "bad channel", CancellationToken::new())
            .await
            .is_err()
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(
        service
            .speak_text(2, 3, 4, "cancel", cancelled)
            .await
            .is_err()
    );
    assert!(rx.try_recv().is_err());
    cancel.cancel();
}
#[async_trait]
impl GatewayVoiceAdapter for Gateway {
    async fn negotiate(&self, _: u64, _: u64, _: u64) -> Result<GatewayVoiceCredentials> {
        panic!("unapproved operation reached Gateway")
    }
    async fn leave(&self, _: u64, _: u64) -> Result<()> {
        panic!("unapproved operation reached Gateway")
    }
}
#[tokio::test]
async fn exact_approvals_denied_before_network() {
    let (tx, _) = mpsc::channel(1);
    let service = DiscordVoiceService::new(
        crate::tests::config(),
        Arc::new(PolicyHttpClient::new(Default::default()).unwrap()),
        Arc::new(Gateway),
        Arc::new(Speech),
        Arc::new(Speech),
        Arc::new(Agent(tx)),
    )
    .unwrap();
    assert!(service.join(2, 3, 4, "yes").await.is_err());
    let cancel = CancellationToken::new();
    cancel.cancel();
    assert!(
        service
            .join_cancellable(2, 3, 4, &DiscordVoiceService::join_approval(2, 3), cancel)
            .await
            .is_err()
    );
    assert!(
        service
            .speak_text(2, 3, 4, "not joined", CancellationToken::new())
            .await
            .is_err()
    );
    assert!(
        service
            .join(2, 3, 99, &DiscordVoiceService::join_approval(2, 3))
            .await
            .is_err()
    );
    assert!(service.leave(2, 3, 4, "yes").await.is_err());
    assert!(
        service
            .leave(2, 3, 99, &DiscordVoiceService::leave_approval(2, 3))
            .await
            .is_err()
    );
}
#[tokio::test]
async fn separate_user_queues_scopes_and_consent() {
    let mut config = crate::tests::config();
    let mut b = config.bindings[0].clone();
    b.user_id = 5;
    b.scope.user = "other".into();
    config.bindings.push(b.clone());
    b.user_id = 6;
    b.consent = false;
    config.bindings.push(b);
    let (tx, mut rx) = mpsc::channel(4);
    let (out, _) = mpsc::channel(2);
    let cancel = CancellationToken::new();
    let (mut collectors, guard) = workers(
        &config,
        2,
        3,
        Arc::new(Speech),
        Arc::new(Speech),
        Arc::new(Agent(tx)),
        cancel,
        out,
    )
    .unwrap();
    assert_eq!(collectors.len(), 2);
    assert!(!collectors.contains_key(&6));
    collectors
        .get_mut(&4)
        .unwrap()
        .tx
        .try_send(vec![7])
        .unwrap();
    collectors
        .get_mut(&5)
        .unwrap()
        .tx
        .try_send(vec![9])
        .unwrap();
    let mut seen = std::collections::BTreeMap::new();
    for _ in 0..2 {
        let (user, prompt) = timeout(Duration::from_secs(1), rx.recv())
            .await
            .unwrap()
            .unwrap();
        seen.insert(user, prompt);
    }
    assert_eq!(seen[&4], "7");
    assert_eq!(seen[&5], "9");
    drop(guard);
}
#[tokio::test]
async fn voice_resume_websocket_fixture() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let peer = tokio::spawn(async move {
        // First WS connection drops before Hello. Resume retries transport,
        // preserving seq_ack rather than renegotiating or replaying speech.
        let (first, _) = listener.accept().await.unwrap();
        let mut first = tokio_tungstenite::accept_async(first).await.unwrap();
        let _ = first.next().await;
        drop(first);
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        let frame = ws.next().await.unwrap().unwrap();
        let value: Value = serde_json::from_str(frame.to_text().unwrap()).unwrap();
        assert_eq!(value["op"], 7);
        assert_eq!(value["d"]["seq_ack"], 42);
        ws.send(message(8, json!({"heartbeat_interval":1000})))
            .await
            .unwrap();
        ws.send(message(9, json!({}))).await.unwrap();
    });
    let http = PolicyHttpClient::new(missy_core::PolicyConfig {
        allow_private_network: true,
        allowed_origins: vec![format!("http://{address}")],
        ..Default::default()
    })
    .unwrap();
    let c = GatewayVoiceCredentials {
        guild_id: 1,
        channel_id: 2,
        user_id: 3,
        endpoint: "synthetic".into(),
        session_id: "synthetic".into(),
        token: "synthetic".into(),
        session_cancel: CancellationToken::new(),
    };
    let (_, hb) = retry_resume(
        &http,
        &format!("ws://{address}"),
        &c,
        42,
        &CancellationToken::new(),
    )
    .await
    .unwrap();
    assert_eq!(hb.interval_ms, Some(1000));
    peer.await.unwrap();
}
#[tokio::test]
async fn worker_drop_cancels() {
    let cancel = CancellationToken::new();
    let c = cancel.clone();
    let task = tokio::spawn(async move {
        c.cancelled().await;
    });
    let workers = Workers {
        tasks: vec![task],
        cancel: cancel.clone(),
    };
    drop(workers);
    assert!(cancel.is_cancelled());
}

struct DropWitness(Arc<std::sync::atomic::AtomicUsize>);
impl Drop for DropWitness {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}
struct BlockingSpeech {
    started: tokio::sync::Notify,
    dropped: Arc<std::sync::atomic::AtomicUsize>,
}
#[async_trait]
impl Stt for BlockingSpeech {
    async fn transcribe(&self, _: &[u8], _: AudioFormat, _: CancellationToken) -> Result<String> {
        let _witness = DropWitness(self.dropped.clone());
        self.started.notify_one();
        // Deliberately ignores cooperative cancellation. Worker ownership must
        // still drop the in-flight future instead of detaching it.
        std::future::pending().await
    }
}
#[tokio::test]
async fn worker_drop_aborts_inflight_uncooperative_stt() {
    let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let speech = Arc::new(BlockingSpeech {
        started: tokio::sync::Notify::new(),
        dropped: dropped.clone(),
    });
    let (agent_tx, mut agent_rx) = mpsc::channel(1);
    let (out, mut output_rx) = mpsc::channel(1);
    let cancel = CancellationToken::new();
    let (collectors, guard) = workers(
        &crate::tests::config(),
        2,
        3,
        speech.clone(),
        Arc::new(Speech),
        Arc::new(Agent(agent_tx)),
        cancel.clone(),
        out,
    )
    .unwrap();
    collectors[&4].tx.send(vec![1]).await.unwrap();
    timeout(Duration::from_secs(1), speech.started.notified())
        .await
        .unwrap();
    drop(guard);
    timeout(Duration::from_secs(1), async {
        while dropped.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(cancel.is_cancelled());
    assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert!(agent_rx.recv().await.is_none());
    assert!(output_rx.recv().await.is_none());
}

#[tokio::test]
async fn service_drop_aborts_owned_session_and_closes_socket() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let client = tokio::net::TcpStream::connect(address).await.unwrap();
    let (mut peer, _) = listener.accept().await.unwrap();
    let (tx, _) = mpsc::channel(1);
    let service = DiscordVoiceService::new(
        crate::tests::config(),
        Arc::new(PolicyHttpClient::new(Default::default()).unwrap()),
        Arc::new(Gateway),
        Arc::new(Speech),
        Arc::new(Speech),
        Arc::new(Agent(tx)),
    )
    .unwrap();
    let cancel = CancellationToken::new();
    let (started, running) = oneshot::channel();
    let task = tokio::spawn(async move {
        let _socket = client;
        started.send(()).unwrap();
        std::future::pending::<Result<()>>().await
    });
    service.sessions.lock().await.insert(
        2,
        Session {
            channel: 3,
            text_output: mpsc::channel(2).0,
            cancel: cancel.clone(),
            task,
        },
    );
    running.await.unwrap();
    drop(service);
    assert!(cancel.is_cancelled());
    let mut byte = [0; 1];
    use tokio::io::AsyncReadExt;
    assert_eq!(
        timeout(Duration::from_secs(1), peer.read(&mut byte))
            .await
            .unwrap()
            .unwrap(),
        0,
        "owned socket must close after session abort"
    );
}

#[tokio::test]
async fn loopback_test_adapter_rejects_remote_and_url_metadata() {
    for url in [
        "ws://example.com/",
        "ws://localhost/",
        "ws://192.168.1.13/",
        "wss://127.0.0.1/",
        "ws://user@127.0.0.1/",
        "ws://127.0.0.1/?token=test",
        "ws://127.0.0.1/#fragment",
    ] {
        let (tx, _) = mpsc::channel(1);
        let (ready, _) = oneshot::channel();
        let c = GatewayVoiceCredentials {
            guild_id: 2,
            channel_id: 3,
            user_id: 1,
            endpoint: "synthetic".into(),
            session_id: "synthetic".into(),
            token: "synthetic".into(),
            session_cancel: CancellationToken::new(),
        };
        let result = run_at_loopback_url(
            Arc::new(crate::tests::config()),
            Arc::new(PolicyHttpClient::new(Default::default()).unwrap()),
            c,
            Arc::new(Speech),
            Arc::new(Speech),
            Arc::new(Agent(tx)),
            CancellationToken::new(),
            ready,
            url.into(),
        )
        .await;
        let expected = if url.contains('@') || url.contains('?') || url.contains('#') {
            "invalid test URL metadata"
        } else {
            "test URL must be literal loopback ws"
        };
        assert_eq!(result.unwrap_err().to_string(), expected);
    }
}

#[test]
fn production_endpoint_remains_wss_only_and_exact_allowlisted() {
    let c = crate::tests::config();
    assert_eq!(
        endpoint(&c, "voice.discord.gg:443").unwrap(),
        "wss://voice.discord.gg/?v=8"
    );
    for denied in [
        "127.0.0.1:8080",
        "ws://voice.discord.gg",
        "voice.discord.gg.attacker.invalid",
        "voice.discord.gg/?v=8",
        "user@voice.discord.gg",
        "voice.discord.gg#fragment",
    ] {
        assert!(endpoint(&c, denied).is_err(), "{denied}");
    }
}
