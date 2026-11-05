use super::*;
#[test]
fn native_flac_decoder_accepts_complete_verbatim_frame_rejects_truncation() {
    // Minimal synthetic FLAC: 8000 Hz, mono signed16, 16 silent samples,
    // STREAMINFO followed by one independent verbatim subframe. No encoder,
    // fixtures downloaded from the network or workstation audio required.
    fn crc8(bytes: &[u8]) -> u8 {
        let mut crc = 0u8;
        for b in bytes {
            crc ^= b;
            for _ in 0..8 {
                crc = if crc & 0x80 != 0 {
                    (crc << 1) ^ 7
                } else {
                    crc << 1
                };
            }
        }
        crc
    }
    fn crc16(bytes: &[u8]) -> u16 {
        let mut crc = 0u16;
        for b in bytes {
            crc ^= (*b as u16) << 8;
            for _ in 0..8 {
                crc = if crc & 0x8000 != 0 {
                    (crc << 1) ^ 0x8005
                } else {
                    crc << 1
                };
            }
        }
        crc
    }
    let mut bytes = b"fLaC".to_vec();
    bytes.extend_from_slice(&[0x80, 0, 0, 34]);
    bytes.extend_from_slice(&16u16.to_be_bytes());
    bytes.extend_from_slice(&16u16.to_be_bytes());
    bytes.extend_from_slice(&[0; 6]);
    let packed: u64 = (8000u64 << 44) | (15u64 << 36) | 16;
    bytes.extend_from_slice(&packed.to_be_bytes());
    bytes.extend_from_slice(&[0; 16]);
    let mut frame = vec![0xff, 0xf8, 0x60, 0x08, 0, 15];
    frame.push(crc8(&frame));
    frame.push(2);
    frame.extend_from_slice(&[0; 32]);
    frame.extend_from_slice(&crc16(&frame).to_be_bytes());
    bytes.extend_from_slice(&frame);
    let (metadata, mime) = singing::validate_audio(&bytes).unwrap();
    assert_eq!(mime, "audio/flac");
    assert_eq!(metadata["frames"], 16);
    assert!(singing::validate_audio(&bytes[..bytes.len() - 1]).is_err());
    let end = bytes.len() - 1;
    bytes[end] ^= 1;
    assert!(singing::validate_audio(&bytes).is_err());
}
use missy_core::Scope;
use std::{
    io::{Read, Write},
    net::TcpListener,
    thread,
};
fn ctx() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "test".into(),
            user: "test".into(),
            channel: "test".into(),
            session: "test".into(),
        },
        turn_id: "test".into(),
        cancellation: tokio_util::sync::CancellationToken::new(),
    }
}
fn policy(dir: &std::path::Path) -> PolicyConfig {
    PolicyConfig {
        read_roots: vec![dir.into()],
        write_roots: vec![dir.into()],
        max_file_bytes: 16 * 1024 * 1024,
        ..Default::default()
    }
}
fn call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: "test".into(),
        name: name.into(),
        arguments,
    }
}
#[test]
fn calculator_real_arithmetic_and_refusals() {
    for (expr, result) in [
        ("(2 ** 10) / 4", 256.0),
        ("-2**2", -4.0),
        ("2**3**2", 512.0),
        ("-7//3", -3.0),
        ("-7%3", 2.0),
        ("1.5e2 + .5", 150.5),
        ("2**-2", 0.25),
    ] {
        assert_eq!(calculator::calculate(expr).unwrap(), result, "{expr}");
    }
    for expr in [
        "__import__('os')",
        "2&3",
        "2j",
        "sqrt(4)",
        "1/0",
        "1e999",
        "(-1)**.5",
        "9**9999",
        "(2",
        "1 2",
    ] {
        assert!(calculator::calculate(expr).is_err(), "{expr}");
    }
    assert!(calculator::calculate(&"(".repeat(1000)).is_err());
}
fn score() -> Score {
    Score {
        bpm: 120,
        ticks_per_beat: 480,
        notes: vec![Note {
            pitch: 69,
            start_tick: 0,
            duration_ticks: 480,
            velocity: 100,
        }],
        waveform: Waveform::Sine,
    }
}
#[test]
fn native_music_decodes_and_contains_expected_tone() {
    let s = score();
    let (wav, midi) = music::render(&s).unwrap();
    assert_eq!(music::render(&s).unwrap(), (wav.clone(), midi.clone()));
    let mut reader = hound::WavReader::new(std::io::Cursor::new(&wav)).unwrap();
    assert_eq!(reader.spec().sample_rate, 22050);
    assert_eq!(reader.duration(), 11025);
    let samples: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    assert!(samples.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / samples.len() as f64 > 1e7);
    let crossings = samples.windows(2).filter(|s| s[0] <= 0 && s[1] > 0).count();
    assert!((218..=222).contains(&crossings));
    let smf = midly::Smf::parse(&midi).unwrap();
    assert_eq!(smf.tracks.len(), 1);
    assert!(smf.tracks[0].iter().any(|e|matches!(e.kind,midly::TrackEventKind::Midi{message:midly::MidiMessage::NoteOn{key,..},..} if key.as_int() == 69)));
    assert!(singing::validate_audio(&wav).is_ok());
    assert!(singing::validate_audio(&wav[..wav.len() - 2]).is_err());
    let mut invalid = s.clone();
    invalid.notes.push(invalid.notes[0].clone());
    assert!(invalid.validate(60).is_err());
    invalid = s;
    invalid.notes[0].duration_ticks = u32::MAX;
    assert!(invalid.validate(60).is_err());
}
#[tokio::test]
async fn default_denied_policy_artifacts_and_documents() {
    let dir = tempfile::tempdir().unwrap();
    let p = policy(dir.path());
    let disabled = SpecialtyTools::new(p.clone(), SpecialtyConfig::default()).unwrap();
    assert!(disabled.definitions().is_empty());
    assert!(
        disabled
            .execute(&ctx(), &call("calculator", json!({"expression":"2+2"})))
            .await
            .is_err()
    );
    let t = SpecialtyTools::new(
        p,
        SpecialtyConfig {
            enabled: true,
            enable_music: true,
            enable_documents: true,
            ..Default::default()
        },
    )
    .unwrap();
    let wav = dir.path().join("tone.wav");
    let midi = dir.path().join("tone.mid");
    let r = t
        .execute(
            &ctx(),
            &call(
                "music_compose",
                json!({"score":score(),"wav_output":wav,"midi_output":midi}),
            ),
        )
        .await
        .unwrap();
    let bytes = std::fs::read(&wav).unwrap();
    assert_eq!(r["wav"]["sha256"], hex::encode(Sha256::digest(&bytes)));
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&wav).unwrap().permissions().mode() & 0o777,
        0o600
    );
    let doc = dir.path().join("doc.md");
    std::fs::write(&doc, "# Title\n```rust\nprint()\n```\n## Second\n").unwrap();
    let r = t
        .execute(&ctx(), &call("document_extract", json!({"input":doc})))
        .await
        .unwrap();
    assert_eq!(r["headings"].as_array().unwrap().len(), 2);
    assert_eq!(r["code_blocks"][0], "print()\n");
    let link = dir.path().join("link.md");
    std::os::unix::fs::symlink("/etc/passwd", &link).unwrap();
    assert!(
        t.execute(&ctx(), &call("document_extract", json!({"input":link})))
            .await
            .is_err()
    );
    assert!(
        t.execute(
            &ctx(),
            &call("calculator", json!({"expression":"2","extra":true}))
        )
        .await
        .is_err()
    );
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    assert!(
        t.execute(&cancelled, &call("calculator", json!({"expression":"2"})))
            .await
            .is_err()
    );
}
// Synthetic loopback HTTP only. No external services, environment edits or network.
struct Exchange {
    path: String,
    body_check: Option<Box<dyn Fn(Value) + Send>>,
    status: u16,
    body: Vec<u8>,
    dynamic_body: Option<Box<dyn Fn() -> Vec<u8> + Send>>,
}
fn exchange(path: &str, status: u16, body: Value) -> Exchange {
    Exchange {
        path: path.into(),
        body_check: None,
        status,
        body: serde_json::to_vec(&body).unwrap(),
        dynamic_body: None,
    }
}
fn server(exchanges: Vec<Exchange>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let job = thread::spawn(move || {
        for e in exchanges {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut bytes = Vec::new();
            let header_end;
            loop {
                let mut b = [0];
                stream.read_exact(&mut b).unwrap();
                bytes.push(b[0]);
                if bytes.ends_with(b"\r\n\r\n") {
                    header_end = bytes.len();
                    break;
                }
                assert!(bytes.len() < 65536);
            }
            let headers = String::from_utf8(bytes.clone()).unwrap();
            let path = headers
                .lines()
                .next()
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap();
            assert!(
                path == e.path
                    || (e.path.ends_with('*') && path.starts_with(e.path.trim_end_matches('*'))),
                "expected {}, got {path}",
                e.path
            );
            let len: usize = headers
                .lines()
                .find_map(|s| {
                    s.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .map(|s| s.trim().parse().unwrap())
                })
                .unwrap_or(0);
            bytes.resize(header_end + len, 0);
            stream.read_exact(&mut bytes[header_end..]).unwrap();
            if let Some(check) = e.body_check {
                check(serde_json::from_slice(&bytes[header_end..]).unwrap());
            }
            let body = if let Some(generate) = e.dynamic_body {
                generate()
            } else {
                e.body
            };
            write!(
                stream,
                "HTTP/1.1 {} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                e.status,
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (format!("http://{addr}"), job)
}
fn http(base: &str) -> PolicyHttpClient {
    PolicyHttpClient::new(PolicyConfig {
        allowed_origins: vec![base.into()],
        allow_private_network: true,
        ..Default::default()
    })
    .unwrap()
}
fn workload() -> Value {
    json!({"schema_version":"1.0","id":"demo","version":"1","repository":{"repository_id":"owner/repo","commit_sha":"a".repeat(40),"snapshot_id":"snapshot-1"},"sandbox":{"image_digest":format!("image@sha256:{}","a".repeat(64))},"execution":{"repetitions":1,"parallelism":1,"timeout_seconds":60,"max_attempts":1},"providers":[{"id":"provider"}],"artifacts":{"required_kinds":["result"]}})
}
fn foundry_config(base: &str) -> FoundryConfig {
    FoundryConfig {
        base_url: base.into(),
        project_id: "project".into(),
        token_env: "TEST_TOKEN_UNUSED".into(),
        repository_ids: vec!["owner/repo".into()],
        run_ids: vec!["run-1".into(), "run-2".into()],
        snapshot_ids: vec!["snapshot-1".into()],
        allow_mutations: true,
        reviewed_workload_sha256: vec![foundry::identity(&workload())],
        idempotency_prefix: "test-".into(),
    }
}
#[tokio::test]
async fn foundry_real_legacy_routes_plan_and_reserved_start() {
    let w = workload();
    let plan = json!({"id":"plan-1","project_id":"project","state":"planned","placement":{"pool":"staging"},"policy_checks":{"repository":true,"image":true,"providers":true,"budget":true,"quota":true,"egress":true,"audit":true},"workload":w});
    let run = format!(
        "run-{}",
        &foundry::identity(&json!(["project", "plan-1", "test-123456"]))[..24]
    );
    let child = format!("run-{}", &foundry::identity(&json!([run, 0, 0]))[..24]);
    let reserved = json!({"id":run,"project_id":"project","plan_id":"plan-1","state":"reserved","job_id":null,"cancel_requested":false,"required_artifacts":["result"],"children":[{"run_id":child,"provider_index":0,"repetition":0,"state":"reserved","job_id":format!("foundry-{child}")}]});
    let mut plan_exchange = exchange(
        "/api/projects/project/benchmark/plan",
        200,
        json!({"ok":true,"data":plan}),
    );
    plan_exchange.body_check = Some(Box::new(move |b| {
        assert_eq!(b, json!({"workload":workload()}))
    }));
    let (base, job) = server(vec![
        exchange(
            "/api/projects/project/repositories",
            200,
            json!({"ok":true,"data":["owner/repo"]}),
        ),
        plan_exchange,
        exchange(
            "/api/projects/project/benchmark/start",
            202,
            json!({"ok":true,"data":reserved}),
        ),
    ]);
    let mut c = foundry_config(&format!("{base}/api"));
    c.validate().unwrap();
    let h = http(&base);
    let mut state = foundry::FoundryState::default();
    let start:foundry::FoundryRequest = serde_json::from_value(json!({"action":"start","plan_id":"plan-1","idempotency_key":"test-123456","acknowledge_project_scope":true})).unwrap();
    assert!(
        foundry::execute_authenticated(&h, &c, &mut state, &ctx(), start.clone(), "fixture-token")
            .await
            .is_err()
    );
    for r in [
        foundry::FoundryRequest::List {},
        foundry::FoundryRequest::Plan {
            workload: workload(),
        },
        start,
    ] {
        foundry::execute_authenticated(&h, &c, &mut state, &ctx(), r, "fixture-token")
            .await
            .unwrap();
    }
    job.join().unwrap();
    c.reviewed_workload_sha256.clear();
    assert!(
        foundry::FoundryRequest::Plan {
            workload: workload()
        }
        .validate(&c, "repoeval_foundry_read")
        .is_err()
    );
}
#[tokio::test]
async fn singing_exact_ace_graph_and_decoded_audio_publication() {
    let nodes = [
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
    ]
    .iter()
    .map(|k| (k.to_string(), json!({})))
    .collect::<serde_json::Map<_, _>>();
    let prefix = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let p = prefix.clone();
    let mut prompt = exchange("/prompt", 200, json!({"prompt_id":"prompt-1"}));
    prompt.body_check = Some(Box::new(move |body| {
        let graph = &body["prompt"];
        assert_eq!(graph["condition"]["inputs"]["lyrics"], "Fixture lyrics");
        assert_eq!(graph["model"]["class_type"], "UNETLoader");
        *p.lock().unwrap() = graph["output"]["inputs"]["filename_prefix"]
            .as_str()
            .unwrap()
            .into();
    }));
    let p = prefix.clone();
    let mut history = exchange("/history/prompt-1", 200, json!({}));
    history.dynamic_body = Some(Box::new(move || {
        serde_json::to_vec(&json!({"prompt-1":{"status":{"status_str":"success"},"outputs":{"output":{"audio":[{"filename":format!("{}_00001_.wav",p.lock().unwrap()),"type":"output","subfolder":""}]}}}})).unwrap()
    }));
    let mut s = score();
    s.notes[0].duration_ticks = 4800;
    let audio = music::render(&s).unwrap().0;
    let mut view = exchange("/view?filename=missy_singing_*", 200, json!({}));
    view.body = audio.clone();
    let (base, job) = server(vec![
        exchange("/object_info", 200, Value::Object(nodes)),
        exchange(
            "/models/diffusion_models",
            200,
            json!(["acestep_v1.5_turbo.safetensors"]),
        ),
        exchange(
            "/models/text_encoders",
            200,
            json!(["qwen_0.6b_ace15.safetensors", "qwen_4b_ace15.safetensors"]),
        ),
        exchange("/models/vae", 200, json!(["ace_1.5_vae.safetensors"])),
        prompt,
        history,
        view,
    ]);
    let dir = tempfile::tempdir().unwrap();
    let p = Policy::new(policy(dir.path())).unwrap();
    let h = http(&base);
    let config = SingingConfig {
        base_url: base,
        max_polls: 1,
        poll_interval_ms: 10,
    };
    let r = singing::SingingRequest {
        lyrics: "Fixture lyrics".into(),
        tags: "piano voice".into(),
        duration: 5.0,
        save_path: Some(dir.path().join("song.wav")),
        ..Default::default()
    };
    let graph = singing::workflow(&r, "prefix");
    assert_eq!(graph.as_object().unwrap().len(), 10);
    assert_eq!(graph["condition"]["inputs"]["lyrics"], r.lyrics);
    let output = singing::execute(&p, &h, &config, &ctx(), r).await.unwrap();
    assert_eq!(output["audio"]["duration_seconds"], 5.0);
    assert_eq!(output["lip_sync_safe"], false);
    assert!(output["vocal_stem"].is_null());
    assert_eq!(std::fs::read(dir.path().join("song.wav")).unwrap(), audio);
    job.join().unwrap();
}
#[test]
fn foundry_scope_and_response_rejections() {
    let c = foundry_config("https://example.invalid/api");
    for args in [
        json!({"action":"status","resource_type":"run","resource_id":"../secret"}),
        json!({"action":"cancel","run_id":"run-1","acknowledge_project_scope":false}),
        json!({"action":"compare","run_ids":["run-1","run-1"]}),
        json!({"action":"snapshot","repository_id":"owner/repo","commit_sha":"a".repeat(40),"idempotency_key":"wrong-key","acknowledge_project_scope":true}),
    ] {
        let r: foundry::FoundryRequest = serde_json::from_value(args).unwrap();
        assert!(r.validate(&c, "repoeval_foundry").is_err());
    }
    let r = foundry::FoundryRequest::Artifacts {
        run_id: "run-1".into(),
    };
    assert!(
        foundry::validate_response(
            &c,
            &r,
            &json!({"project_id":"other","run_id":"run-1","artifacts":[]})
        )
        .is_err()
    );
    let r = foundry::FoundryRequest::Compare {
        run_ids: vec!["run-1".into(), "run-2".into()],
    };
    assert!(
        foundry::validate_response(
            &c,
            &r,
            &json!({"project_id":"project","run_ids":["run-1","run-2"],"comparable":true})
        )
        .is_err()
    );
    assert!(
        serde_json::from_value::<foundry::FoundryRequest>(
            json!({"action":"list","url":"https://evil.invalid"})
        )
        .is_err()
    );
    assert!(validate_base("http://user:secret@example.invalid", false).is_err());
}
#[test]
fn foundry_compare_report_and_mutation_contracts() {
    let c = foundry_config("https://example.invalid/api");
    let compare = foundry::FoundryRequest::Compare {
        run_ids: c.run_ids.clone(),
    };
    let valid = json!({"project_id":"project","comparison_id":format!("comparison-{}","a".repeat(24)),"run_ids":c.run_ids,"comparable":true,"pairs":[{"left_run_id":"run-1","right_run_id":"run-2","comparable":true,"key":"a".repeat(64),"reasons":[]}]});
    foundry::validate_response(&c, &compare, &valid).unwrap();
    let mut invalid = valid.clone();
    invalid["pairs"][0]["right_run_id"] = json!("run-1");
    assert!(foundry::validate_response(&c, &compare, &invalid).is_err());
    invalid = valid;
    invalid["comparable"] = json!(false);
    assert!(foundry::validate_response(&c, &compare, &invalid).is_err());
    let report = foundry::FoundryRequest::Report {
        run_ids: c.run_ids.clone(),
    };
    let valid = json!({"project_id":"project","report_id":format!("report-{}","a".repeat(24)),"run_ids":c.run_ids,"status":"draft","published":false,"schema_version":"1.0","groups":[{"comparability_key":"a".repeat(64),"run_ids":["run-1"]}],"incomparable":[{"run_id":"run-2","reason":["comparability_key_invalid"]}]});
    foundry::validate_response(&c, &report, &valid).unwrap();
    let mut invalid = valid.clone();
    invalid["published"] = json!(true);
    assert!(foundry::validate_response(&c, &report, &invalid).is_err());
    invalid = valid;
    invalid["incomparable"][0]["reason"] = json!(["invented_reason"]);
    assert!(foundry::validate_response(&c, &report, &invalid).is_err());
    let cancel = foundry::FoundryRequest::Cancel {
        run_id: "run-1".into(),
        acknowledge_project_scope: true,
    };
    foundry::validate_response(
        &c,
        &cancel,
        &json!({"id":"run-1","state":"cancel_pending","cancel_requested":true}),
    )
    .unwrap();
    assert!(
        foundry::validate_response(
            &c,
            &cancel,
            &json!({"id":"run-1","state":"cancel_pending","cancel_requested":false})
        )
        .is_err()
    );
    let snapshot = foundry::FoundryRequest::Snapshot {
        repository_id: "owner/repo".into(),
        commit_sha: "a".repeat(40),
        idempotency_key: "test-123456".into(),
        acknowledge_project_scope: true,
    };
    foundry::validate_response(&c,&snapshot,&json!({"id":"snapshot-1","project_id":"project","repository_id":"owner/repo","commit_sha":"a".repeat(40),"state":"requested"})).unwrap();
    let mut insecure = c;
    insecure.base_url = "http://example.invalid".into();
    assert!(insecure.validate().is_err());
}
#[tokio::test]
async fn foundry_bad_refresh_revokes_cached_plan_and_scope_change_revokes_grants() {
    let (base, job) = server(vec![
        exchange(
            "/projects/project/repositories",
            200,
            json!({"ok":true,"data":["owner/repo"]}),
        ),
        exchange(
            "/projects/project/repositories",
            500,
            json!({"secret":"do not echo"}),
        ),
    ]);
    let c = foundry_config(&base);
    let h = http(&base);
    let mut state = foundry::FoundryState::default();
    let context = ctx();
    foundry::execute_authenticated(
        &h,
        &c,
        &mut state,
        &context,
        foundry::FoundryRequest::List {},
        "fixture-token",
    )
    .await
    .unwrap();
    assert!(
        foundry::execute_authenticated(
            &h,
            &c,
            &mut state,
            &context,
            foundry::FoundryRequest::List {},
            "fixture-token"
        )
        .await
        .is_err()
    );
    assert!(
        foundry::execute_authenticated(
            &h,
            &c,
            &mut state,
            &context,
            foundry::FoundryRequest::Plan {
                workload: workload()
            },
            "fixture-token"
        )
        .await
        .is_err()
    );
    let mut other = ctx();
    other.scope.session = "different".into();
    assert!(
        foundry::execute_authenticated(
            &h,
            &c,
            &mut state,
            &other,
            foundry::FoundryRequest::Plan {
                workload: workload()
            },
            "fixture-token"
        )
        .await
        .is_err()
    );
    job.join().unwrap();
}
#[tokio::test]
async fn foundry_credential_echo_rejected_and_unknown_fields_denied() {
    let (base, job) = server(vec![exchange(
        "/projects/project/repositories",
        200,
        json!({"ok":true,"data":["fixture-token"]}),
    )]);
    let c = foundry_config(&base);
    let h = http(&base);
    let mut state = foundry::FoundryState::default();
    assert!(
        foundry::execute_authenticated(
            &h,
            &c,
            &mut state,
            &ctx(),
            foundry::FoundryRequest::List {},
            "fixture-token"
        )
        .await
        .is_err()
    );
    job.join().unwrap();
    assert!(
        serde_json::from_value::<singing::SingingRequest>(
            json!({"validate_only":true,"comfyui_host":"evil.invalid"})
        )
        .is_err()
    );
    assert!(singing::validate_audio(b"fLaCgarbage").is_err());
    assert!(singing::validate_audio(b"RIFFnot-audio").is_err());
}
