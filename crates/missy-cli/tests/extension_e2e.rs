//! Real native binary, loopback HTTP fixtures, disposable state only.
use missy_core::{Config, ProviderConfig, Scope};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    process::{Command, Output},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};
struct Fixture {
    origin: String,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}
impl Fixture {
    fn new(mut respond: impl FnMut(usize, &Value) -> Value + Send + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut socket, _)) => {
                        socket
                            .set_read_timeout(Some(Duration::from_secs(5)))
                            .unwrap();
                        let request = read_request(&mut socket);
                        let n = {
                            let mut r = captured.lock().unwrap();
                            let n = r.len();
                            r.push(request.clone());
                            n
                        };
                        let body = serde_json::to_vec(&respond(n, &request)).unwrap();
                        write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
                        socket.write_all(&body).unwrap();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => panic!("fixture accept: {e}"),
                }
            }
        });
        Self {
            origin,
            requests,
            stop,
            worker: Some(worker),
        }
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let r = self.worker.take().unwrap().join();
        if !thread::panicking() {
            r.unwrap();
        }
    }
}
fn read_request(s: &mut TcpStream) -> Value {
    let mut h = Vec::new();
    let mut b = [0];
    while !h.ends_with(b"\r\n\r\n") {
        s.read_exact(&mut b).unwrap();
        h.push(b[0]);
        assert!(h.len() < 65536);
    }
    let h = String::from_utf8(h).unwrap();
    assert!(h.starts_with("POST /v1/chat/completions "));
    let len: usize = h
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse().unwrap())
        })
        .unwrap();
    assert!(len < 2_000_000);
    let mut body = vec![0; len];
    s.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}
fn reply(text: &str, calls: Vec<Value>) -> Value {
    json!({"model":"fixture","choices":[{"finish_reason":if calls.is_empty(){"stop"}else{"tool_calls"},"message":{"role":"assistant","content":text,"tool_calls":calls}}],"usage":{"prompt_tokens":100,"completion_tokens":20}})
}
fn call(id: &str, name: &str, args: Value) -> Value {
    json!({"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}})
}
fn result(request: &Value, id: &str) -> Value {
    let m = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "tool" && m["tool_call_id"] == id)
        .unwrap_or_else(|| panic!("missing result {id}: {request}"));
    let envelope: Value = serde_json::from_str(m["content"].as_str().unwrap()).unwrap();
    assert_eq!(envelope["untrusted"], true);
    assert_eq!(envelope["truncated"], false);
    let data = envelope["data"].as_str().unwrap();
    assert_eq!(
        envelope["sha256"],
        hex::encode(Sha256::digest(data.as_bytes()))
    );
    serde_json::from_str(data).unwrap()
}
fn invoke(root: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_missy-rs"))
        .arg("--config")
        .arg(root.join("config.yaml"))
        .args(args)
        .env_clear()
        .env("HOME", root)
        .env("PATH", "/usr/bin:/bin")
        .env("MISSY_TEST_KEY", "extension-fixture-only")
        .current_dir(root)
        .output()
        .unwrap()
}
fn ok(o: Output) -> Value {
    assert!(
        o.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    serde_json::from_slice(&o.stdout).unwrap()
}
fn private(path: &Path, value: &impl serde::Serialize) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, serde_yaml_ng::to_string(value).unwrap()).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
}
fn config(root: &Path, p: &Fixture) -> Config {
    let mut c = Config {
        state_dir: root.join("state"),
        providers: vec![ProviderConfig {
            name: "fixture".into(),
            kind: "openai".into(),
            base_url: format!("{}/v1", p.origin),
            model: "fixture".into(),
            api_key_env: Some("MISSY_TEST_KEY".into()),
            input_microusd_per_million: Some(1000),
            output_microusd_per_million: Some(2000),
            ..Default::default()
        }],
        ..Default::default()
    };
    c.policy.allowed_origins = vec![p.origin.clone()];
    c.policy.allow_private_network = true;
    c.policy.max_file_bytes = 65536;
    c
}
fn guard(root: &Path, c: &mut Config) {
    let p = root.join("guard.yaml");
    let names = if c.policy.allow_file_delete {
        vec!["file_delete"]
    } else {
        vec!["vision_analyze", "vision_scene"]
    };
    private(
        &p,
        &json!({"sensitive_tool_names":names,"context_fingerprint":"0".repeat(64),"approval_ttl_seconds":300}),
    );
    c.extensions.approvals_config = Some(p);
}
fn approve(root: &Path) -> Value {
    let all = ok(invoke(root, &["approvals", "list"]));
    let rows: Vec<_> = all
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["status"] == "pending")
        .collect();
    assert_eq!(rows.len(), 1, "{all}");
    let r = rows[0].clone();
    assert!(
        !invoke(
            root,
            &[
                "approvals",
                "approve",
                r["id"].as_str().unwrap(),
                "--binding-hash",
                &"f".repeat(64),
                "--operator",
                "fixture"
            ]
        )
        .status
        .success()
    );
    ok(invoke(
        root,
        &[
            "approvals",
            "approve",
            r["id"].as_str().unwrap(),
            "--binding-hash",
            r["binding_hash"].as_str().unwrap(),
            "--operator",
            "fixture",
        ],
    ));
    r
}
// Complete synthetic 1x1 RGB PNG, with CRCs and zlib Adler checksum.
fn png() -> Vec<u8> {
    fn crc(b: &[u8]) -> u32 {
        let mut c = !0u32;
        for &b in b {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = (c >> 1) ^ (0xedb88320u32 & (0u32.wrapping_sub(c & 1)));
            }
        }
        !c
    }
    fn chunk(p: &mut Vec<u8>, k: &[u8; 4], d: &[u8]) {
        p.extend_from_slice(&(d.len() as u32).to_be_bytes());
        let s = p.len();
        p.extend_from_slice(k);
        p.extend_from_slice(d);
        let c = crc(&p[s..]);
        p.extend_from_slice(&c.to_be_bytes());
    }
    let mut p = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut p, b"IHDR", &[0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0]);
    chunk(
        &mut p,
        b"IDAT",
        &[0x78, 1, 1, 4, 0, 0xfb, 0xff, 0, 255, 0, 0, 3, 1, 1, 0],
    );
    chunk(&mut p, b"IEND", &[]);
    p
}
fn unbase64(s: &str) -> Vec<u8> {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (mut out, mut acc, mut bits) = (vec![], 0u32, 0);
    for b in s.bytes().take_while(|b| *b != b'=') {
        acc = (acc << 6) | alphabet.iter().position(|c| *c == b).unwrap() as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

#[test]
fn vision_binary_exact_approval_png_disclosure_and_durable_priced_cap() {
    let root = tempfile::tempdir().unwrap();
    let image = root.path().join("private-synthetic.png");
    std::fs::write(&image, png()).unwrap();
    let path = image.clone();
    let p = Fixture::new(move |n, _| match n {
        0 => reply(
            "",
            vec![call(
                "vision-original",
                "vision_analyze",
                json!({"imagepath":path,"question":"Describe."}),
            )],
        ),
        1 => reply("A red pixel was reported.", vec![]),
        2 => reply(
            "",
            vec![call(
                "vision-cap",
                "vision_analyze",
                json!({"imagepath":path,"question":"Describe."}),
            )],
        ),
        _ => reply("UNEXPECTED REPLAY", vec![]),
    });
    let v = Fixture::new(
        |_, _| json!({"model":"vision-fixture","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"A red pixel."}}],"usage":{"prompt_tokens":123,"completion_tokens":7}}),
    );
    let mut c = config(root.path(), &p);
    guard(root.path(), &mut c);
    c.policy.read_roots = vec![root.path().into()];
    c.policy.allowed_origins.push(v.origin.clone());
    let vc = root.path().join("vision.yaml");
    private(
        &vc,
        &json!({"enabled":true,"kind":"openai","base_url":format!("{}/v1",v.origin),"model":"vision-fixture","api_key_env":"MISSY_TEST_KEY","max_output_tokens":16,"image_input_tokens_estimate":1,"input_microusd_per_million":1000000,"output_microusd_per_million":2000000,"max_call_cost_microusd":1100,"max_total_cost_microusd":1100}),
    );
    c.extensions.vision_config = Some(vc);
    private(&root.path().join("config.yaml"), &c);
    let pending = ok(invoke(
        root.path(),
        &["ask", "Inspect fixture", "--request-id", "vision-one"],
    ));
    assert_eq!(pending["status"], "interrupted");
    assert_eq!(pending["resumable"], true);
    assert_eq!(p.count(), 1);
    assert_eq!(v.count(), 0);
    let approved = approve(root.path());
    assert_eq!(v.count(), 0);
    let id = pending["turn_id"].as_str().unwrap();
    let done = ok(invoke(root.path(), &["resume", id]));
    assert_eq!(done["status"], "completed");
    assert_eq!(p.count(), 2, "regenerated call");
    assert_eq!(v.count(), 1);
    let request = v.requests.lock().unwrap()[0].clone();
    assert_eq!(request["model"], "vision-fixture");
    let content = &request["messages"][1]["content"];
    assert_eq!(content[0]["text"], "Describe.");
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    let bytes = unbase64(url.strip_prefix("data:image/png;base64,").unwrap());
    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert_eq!(&bytes[16..24], &[0, 0, 0, 1, 0, 0, 0, 1]);
    assert!(!request.to_string().contains(image.to_str().unwrap()));
    let receipt = result(&p.requests.lock().unwrap()[1], "vision-original");
    assert_eq!(receipt["untrusted"], true);
    assert_eq!(receipt["analysis"], "A red pixel.");
    assert_eq!(receipt["usage"]["cost_microusd"], 137);
    assert_eq!(receipt["usage"]["cost_known"], true);
    assert_eq!(receipt["accounting"]["settled"], true);
    let receipts = ok(invoke(root.path(), &["approvals", "list"]));
    assert!(
        receipts
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["id"] == approved["id"] && r["status"] == "consumed")
    );
    assert!(!invoke(root.path(), &["resume", id]).status.success());
    assert_eq!(
        ok(invoke(
            root.path(),
            &["ask", "Inspect fixture", "--request-id", "vision-one"]
        ))["turn_id"],
        id
    );
    assert_eq!(p.count(), 2);
    assert_eq!(v.count(), 1);
    let cap = ok(invoke(
        root.path(),
        &[
            "ask",
            "Inspect second fixture",
            "--request-id",
            "vision-two",
        ],
    ));
    assert_eq!(cap["status"], "interrupted");
    approve(root.path());
    let denied = ok(invoke(
        root.path(),
        &["resume", cap["turn_id"].as_str().unwrap()],
    ));
    assert_ne!(denied["status"], "completed");
    let state = missy_store::Store::open(c.state_dir.join("runtime.sqlite3")).unwrap();
    let scope = Scope {
        tenant: "local".into(),
        user: "operator".into(),
        channel: "cli".into(),
        session: "default".into(),
    };
    let stored = state
        .list_turns(&scope, 10)
        .unwrap()
        .into_iter()
        .find(|t| t.id == cap["turn_id"].as_str().unwrap())
        .unwrap();
    assert!(
        stored
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("budget"),
        "{stored:?}"
    );
    assert_eq!(p.count(), 3);
    assert_eq!(v.count(), 1, "cap allowed disclosure");
    let db = rusqlite::Connection::open(c.state_dir.join("vision-budget.sqlite3")).unwrap();
    let rows: Vec<(i64, i64, String)> = db
        .prepare("SELECT reserved,actual,state FROM vision_calls")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].0 <= 1100 && rows[0].0 >= 137);
    assert_eq!(rows[0].1, 137);
    assert_eq!(rows[0].2, "settled");
}

#[test]
fn file_delete_binary_exact_approval_then_duplicate_does_not_delete_recreated_file() {
    let root = tempfile::tempdir().unwrap();
    let workspace = root.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let victim = workspace.join("fixture.txt");
    let bytes = b"exact original fixture";
    std::fs::write(&victim, bytes).unwrap();
    let args = json!({"path":victim,"expected_sha256":hex::encode(Sha256::digest(bytes))});
    let p = Fixture::new(move |n, _| {
        if n == 0 {
            reply(
                "",
                vec![call("delete-original", "file_delete", args.clone())],
            )
        } else {
            reply("Deleted.", vec![])
        }
    });
    let mut c = config(root.path(), &p);
    c.policy.write_roots = vec![workspace];
    c.policy.allow_file_delete = true;
    guard(root.path(), &mut c);
    private(&root.path().join("config.yaml"), &c);
    let pending = ok(invoke(
        root.path(),
        &["ask", "Delete fixture", "--request-id", "delete-one"],
    ));
    assert_eq!(pending["status"], "interrupted");
    assert_eq!(std::fs::read(&victim).unwrap(), bytes);
    assert_eq!(p.count(), 1);
    approve(root.path());
    let id = pending["turn_id"].as_str().unwrap();
    assert_eq!(
        ok(invoke(root.path(), &["resume", id]))["status"],
        "completed"
    );
    assert!(!victim.exists());
    assert_eq!(p.count(), 2);
    let receipt = result(&p.requests.lock().unwrap()[1], "delete-original");
    assert_eq!(receipt["trust"], "untrusted");
    assert_eq!(receipt["data"]["deleted"], true);
    std::fs::write(&victim, bytes).unwrap();
    assert_eq!(
        ok(invoke(
            root.path(),
            &["ask", "Delete fixture", "--request-id", "delete-one"]
        ))["turn_id"],
        id
    );
    assert!(!invoke(root.path(), &["resume", id]).status.success());
    assert_eq!(std::fs::read(&victim).unwrap(), bytes);
    assert_eq!(p.count(), 2);
}

#[test]
fn history_binary_dynamic_receipt_expansion_and_four_component_scope_isolation() {
    let root = tempfile::tempdir().unwrap();
    let source = "LOCAL_RECALL é𐐀 exact UTF-8 fixture.";
    let expected = source.to_owned();
    let p = Fixture::new(move |n, r| match n {
        0 => {
            let names: Vec<_> = r["tools"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap())
                .collect();
            assert!(names.contains(&"history_search") && names.contains(&"memory_expand"));
            assert!(!names.contains(&"context_shunt"));
            reply(
                "",
                vec![call(
                    "local",
                    "history_search",
                    json!({"query":"LOCAL_RECALL"}),
                )],
            )
        }
        1 => {
            let s = result(r, "local");
            assert_eq!(s["trust"], "untrusted_data");
            assert_eq!(s["window"]["kind"], "latest_exact_scope_turns");
            let rows = s["results"].as_array().unwrap();
            assert_eq!(rows.len(), 1, "{s}");
            let row = &rows[0];
            assert_eq!(row["text"], expected);
            assert_eq!(
                row["content_sha256"],
                hex::encode(Sha256::digest(expected.as_bytes()))
            );
            reply(
                "",
                vec![call(
                    "expand",
                    "memory_expand",
                    json!({"item_id":row["item_id"],"content_sha256":row["content_sha256"],"turn_id":row["turn_id"],"start_byte":0,"end_byte":expected.len()}),
                )],
            )
        }
        2 => {
            let e = result(r, "expand");
            assert_eq!(e["text"], expected);
            assert_eq!(e["trust"], "untrusted_data");
            reply(
                "",
                vec![call(
                    "foreign",
                    "history_search",
                    json!({"query":"FOREIGN_RECALL"}),
                )],
            )
        }
        3 => {
            assert_eq!(result(r, "foreign")["results"], json!([]));
            reply(
                "",
                vec![call(
                    "memory",
                    "history_search",
                    json!({"query":"MEMORY_ONLY"}),
                )],
            )
        }
        4 => {
            assert_eq!(result(r, "memory")["results"], json!([]));
            reply("Scoped recall completed.", vec![])
        }
        _ => reply("UNEXPECTED REPLAY", vec![]),
    });
    let mut c = config(root.path(), &p);
    let history = root.path().join("history.yaml");
    private(&history, &json!({"history":{"enabled":true}}));
    c.extensions.history_config = Some(history);
    private(&root.path().join("config.yaml"), &c);
    let store = missy_store::Store::open(c.state_dir.join("runtime.sqlite3")).unwrap();
    let scope = Scope {
        tenant: "local".into(),
        user: "operator".into(),
        channel: "cli".into(),
        session: "default".into(),
    };
    store.admit(&scope, "seed-local", source).unwrap();
    for field in ["tenant", "user", "channel", "session"] {
        let mut s = scope.clone();
        match field {
            "tenant" => s.tenant = "foreign".into(),
            "user" => s.user = "foreign".into(),
            "channel" => s.channel = "foreign".into(),
            _ => s.session = "foreign".into(),
        };
        store
            .admit(
                &s,
                "seed-foreign",
                &format!("FOREIGN_RECALL_SECRET_{field}"),
            )
            .unwrap();
    }
    store
        .memory_put(
            &scope,
            "memory-only",
            "MEMORY_ONLY_SECRET",
            "fixture",
            "untrusted",
        )
        .unwrap();
    drop(store);
    let done = ok(invoke(
        root.path(),
        &[
            "ask",
            "Recall fixture evidence",
            "--request-id",
            "history-one",
        ],
    ));
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(p.count(), 5);
    for r in p.requests.lock().unwrap().iter() {
        let text = r.to_string();
        assert!(!text.contains("FOREIGN_RECALL_SECRET"));
        assert!(!text.contains("MEMORY_ONLY_SECRET"));
    }
    assert!(c.extensions.approvals_config.is_none());
}
#[test]
fn integration_new_mutation_names_cannot_be_omitted_from_cli_guard() {
    // Both constructors are offline. These named grants are meaningful but no
    // request may reach either service or the completion provider at startup.
    for omitted in ["incus_operation", "comfyui_artifact"] {
        let root = tempfile::tempdir().unwrap();
        let p = Fixture::new(|_, _| reply("UNEXPECTED PROVIDER REQUEST", vec![]));
        let service = Fixture::new(|_, _| json!({"unexpected":"integration network access"}));
        let mut c = config(root.path(), &p);
        c.policy.allowed_origins.push(service.origin.clone());
        let integrations = root.path().join("integrations.yaml");
        private(
            &integrations,
            &json!({
                "incus":{"connection":{"enabled":true,"endpoint":service.origin,"allow_mutations":true},"project":"fixture","instances":["fixture"],
                    "operations":{"snapshot":{"action":"snapshot_create","instance":"fixture","snapshot":"safe-fixture"}}},
                "comfyui":{"connection":{"enabled":true,"endpoint":service.origin,"allow_mutations":true},"max_tracked_jobs":10,"max_artifact_bytes":1024,
                    "allowed_classes":["CLIPTextEncode"],"workflows":{"fixture":{"graph":{"1":{"class_type":"CLIPTextEncode","inputs":{"text":"fixture"}}},"prompt_node":"1"}}}
            }),
        );
        c.extensions.integrations_config = Some(integrations);
        let names: Vec<_> = [
            "incus_mutate",
            "incus_operation",
            "comfyui_submit",
            "comfyui_artifact",
        ]
        .into_iter()
        .filter(|name| *name != omitted)
        .collect();
        let g = root.path().join("guard.yaml");
        private(
            &g,
            &json!({"sensitive_tool_names":names,"context_fingerprint":"0".repeat(64),"approval_ttl_seconds":300}),
        );
        c.extensions.approvals_config = Some(g);
        private(&root.path().join("config.yaml"), &c);
        let out = invoke(
            root.path(),
            &[
                "ask",
                "No effects permitted",
                "--request-id",
                "startup-denial",
            ],
        );
        assert!(
            !out.status.success(),
            "guard omission {omitted} was accepted"
        );
        assert!(
            String::from_utf8_lossy(&out.stderr)
                .contains("every integration mutation must be in sensitive_tool_names"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(p.count(), 0);
        assert_eq!(service.count(), 0);
    }
}
