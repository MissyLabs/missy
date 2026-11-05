use missy_core::{PolicyConfig, Scope, ToolCall, ToolContext, ToolDispatchRefused, ToolExecutor};
use missy_discord_tools::{DiscordTools, DiscordToolsConfig, UploadOutcomeUnknown, upload_nonce};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, Once},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

const TOKEN: &str = "fixture_token_not_live_12345";
fn env_setup() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // One test-only variable, written once before any test reads it. Never
        // modifies the gateway's token or inspects any real credential.
        unsafe {
            std::env::set_var("MISSY_DISCORD_TOOLS_FIXTURE_TOKEN", TOKEN);
        }
    });
}
fn ctx() -> ToolContext {
    ToolContext {
        scope: Scope {
            tenant: "t".into(),
            user: "u".into(),
            channel: "c".into(),
            session: "s".into(),
        },
        turn_id: "turn".into(),
        cancellation: Default::default(),
    }
}
fn call(name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: "call".into(),
        name: name.into(),
        arguments: args,
    }
}
struct Request {
    head: String,
    body: Vec<u8>,
}
async fn fixture<F>(
    count: usize,
    handler: F,
) -> (
    String,
    Arc<Mutex<Vec<Request>>>,
    tokio::task::JoinHandle<()>,
)
where
    F: Fn(&Request) -> (u16, Value) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/api/v10", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let copy = requests.clone();
    let task = tokio::spawn(async move {
        for _ in 0..count {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = vec![];
            let mut buf = [0u8; 8192];
            let end = loop {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buf[..n]);
                if let Some(i) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    break i + 4;
                }
            };
            let head = String::from_utf8(bytes[..end].to_vec()).unwrap();
            let len: usize = head
                .lines()
                .find_map(|s| {
                    s.to_lowercase()
                        .strip_prefix("content-length:")
                        .map(|s| s.trim().parse().unwrap())
                })
                .unwrap_or(0);
            while bytes.len() < end + len {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buf[..n]);
            }
            let req = Request {
                head,
                body: bytes[end..end + len].to_vec(),
            };
            let (status, value) = handler(&req);
            copy.lock().unwrap().push(req);
            let body = serde_json::to_vec(&value).unwrap();
            socket.write_all(format!("HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
        }
    });
    (base, requests, task)
}
fn config(base: String) -> DiscordToolsConfig {
    env_setup();
    DiscordToolsConfig {
        enabled: true,
        lookup_enabled: true,
        upload_enabled: true,
        token_env: "MISSY_DISCORD_TOOLS_FIXTURE_TOKEN".into(),
        bot_user_id: "10".into(),
        allowed_guilds: vec!["20".into()],
        allowed_users: vec!["30".into()],
        allowed_channels: BTreeMap::from([("40".into(), "20".into())]),
        api_base: base,
        fixture_mode: true,
        ..Default::default()
    }
}
fn policy(c: &DiscordToolsConfig, root: &std::path::Path) -> PolicyConfig {
    PolicyConfig {
        allowed_origins: vec![
            url::Url::parse(&c.api_base)
                .unwrap()
                .origin()
                .ascii_serialization(),
        ],
        allow_private_network: true,
        read_roots: vec![root.into()],
        max_file_bytes: 4 * 1024 * 1024,
        ..Default::default()
    }
}
fn msg(nonce: &str, filename: &str, size: usize) -> Value {
    json!({"id":"50","channel_id":"40","nonce":nonce,"author":{"id":"10","bot":true},"attachments":[{"id":"60","filename":filename,"size":size,"url":"https://cdn.discordapp.com/attachments/40/60/art.png?ex=fixture"}]})
}
fn nonce_from(req: &Request) -> String {
    let s = String::from_utf8_lossy(&req.body);
    let p = s.find("{\"allowed_mentions\"").unwrap();
    let e = s[p..].find("\r\n").unwrap() + p;
    serde_json::from_str::<Value>(&s[p..e]).unwrap()["nonce"]
        .as_str()
        .unwrap()
        .into()
}

#[tokio::test]
async fn actual_binary_multipart_and_safe_result() {
    let data = b"\x89PNG\r\n\x1a\n\0\xffbinary\0";
    let (base, requests, task) = fixture(1, move |req| {
        assert!(req.head.starts_with("POST /api/v10/channels/40/messages "));
        assert!(req.head.contains(&format!("authorization: Bot {TOKEN}")));
        assert!(!req.head.lines().next().unwrap().contains(TOKEN));
        assert!(req.body.windows(data.len()).any(|w| w == data));
        let s = String::from_utf8_lossy(&req.body);
        assert!(s.contains("\"enforce_nonce\":true"));
        assert!(s.contains("\"parse\":[]"));
        assert!(s.contains("filename=\"art.png\""));
        assert!(!s.contains(TOKEN));
        (200, msg(&nonce_from(req), "art.png", data.len()))
    })
    .await;
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("art.png");
    std::fs::write(&file, data).unwrap();
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let c = call(
        "discord_upload_file",
        json!({"guild_id":"20","channel_id":"40","path":file,"caption":"test"}),
    );
    let out = tools.execute(&ctx(), &c).await.unwrap();
    assert_eq!(out["data"]["attachments"][0]["size"], data.len());
    assert_eq!(out["data"]["nonce"], upload_nonce(&ctx(), "call"));
    assert!(
        tools
            .execute(&ctx(), &c)
            .await
            .unwrap_err()
            .downcast_ref::<UploadOutcomeUnknown>()
            .is_some()
    );
    task.await.unwrap();
    assert_eq!(requests.lock().unwrap().len(), 1);
    assert!(!format!("{tools:?}").contains(TOKEN));
    assert!(!tools.configuration_fingerprint().contains(TOKEN));
    assert!(
        tools
            .definitions()
            .iter()
            .find(|d| d.name == "discord_upload_file")
            .unwrap()
            .side_effect
    );
}

#[tokio::test]
async fn lookup_is_read_only_and_projects_safe_dto() {
    let (base,requests,task)=fixture(1,|req| {assert!(req.head.starts_with("GET /api/v10/guilds/20/members/30 "));(200,json!({"nick":"member","user":{"id":"30","username":"person","email":"secret@example.test","phone":"SECRET","token":TOKEN}}))}).await;
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let out = tools
        .execute(
            &ctx(),
            &call(
                "discord_lookup_user",
                json!({"guild_id":"20","user_id":"30"}),
            ),
        )
        .await
        .unwrap();
    assert_eq!(out["data"]["username"], "person");
    let text = out.to_string();
    assert!(!text.contains("email"));
    assert!(!text.contains("SECRET"));
    assert!(!text.contains(TOKEN));
    assert!(
        !tools
            .definitions()
            .iter()
            .find(|d| d.name == "discord_lookup_user")
            .unwrap()
            .side_effect
    );
    task.await.unwrap();
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn cancelled_after_post_dispatch_is_unknown_without_retry() {
    let context = ctx();
    let cancel = context.cancellation.clone();
    let (base, requests, task) = fixture(1, move |_| {
        cancel.cancel();
        (200, json!({}))
    })
    .await;
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a.png");
    std::fs::write(&file, b"\x89PNG\r\n\x1a\nx").unwrap();
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let e = tools
        .execute(
            &context,
            &call(
                "discord_upload_file",
                json!({"guild_id":"20","channel_id":"40","path":file}),
            ),
        )
        .await
        .unwrap_err();
    assert!(e.downcast_ref::<UploadOutcomeUnknown>().is_some());
    task.await.unwrap();
    assert_eq!(requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn response_channel_bot_nonce_size_and_cdn_must_all_match() {
    for bad in ["channel", "bot", "nonce", "size", "url", "id"] {
        let (base, _, task) = fixture(2, move |req| {
            if req.head.starts_with("GET") {
                return (200, json!([]));
            }
            let mut value = msg(&nonce_from(req), "a.png", 9);
            match bad {
                "channel" => value["channel_id"] = json!("41"),
                "bot" => value["author"]["id"] = json!("11"),
                "nonce" => value["nonce"] = json!("wrong"),
                "size" => value["attachments"][0]["size"] = json!(10),
                "url" => value["attachments"][0]["url"] = json!("https://evil.example/secret"),
                _ => value["id"] = json!("18446744073709551616"),
            }
            (200, value)
        })
        .await;
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("a.png");
        std::fs::write(&file, b"\x89PNG\r\n\x1a\nx").unwrap();
        let cfg = config(base);
        let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
        assert!(
            tools
                .execute(
                    &ctx(),
                    &call(
                        "discord_upload_file",
                        json!({"guild_id":"20","channel_id":"40","path":file})
                    )
                )
                .await
                .unwrap_err()
                .downcast_ref::<UploadOutcomeUnknown>()
                .is_some()
        );
        task.await.unwrap();
    }
}

#[tokio::test]
async fn exact_credential_echo_redacted_and_rotation_changes_fingerprint() {
    let (base, _, task) = fixture(1, |_| (200, json!({"user":{"id":"30","username":TOKEN}}))).await;
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let out = tools
        .execute(
            &ctx(),
            &call(
                "discord_lookup_user",
                json!({"guild_id":"20","user_id":"30"}),
            ),
        )
        .await
        .unwrap();
    assert_eq!(out["data"]["username"], "[REDACTED]");
    task.await.unwrap();
    let mut cfg = config("http://127.0.0.1:9/api/v10".into());
    cfg.token_env = "MISSY_DISCORD_TOOLS_ROTATION_FIXTURE".into();
    // Unique variable used by this test only; no real environment values read.
    unsafe {
        std::env::set_var(&cfg.token_env, "synthetic_before_rotation");
    }
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg.clone()).unwrap();
    let before = tools.configuration_fingerprint();
    unsafe {
        std::env::set_var(&cfg.token_env, "synthetic_after_rotation");
    }
    assert_ne!(before, tools.configuration_fingerprint());
}

#[tokio::test]
async fn uncertain_post_reconciles_read_only_never_reposts() {
    let (base, requests, task) = fixture(2, |req| {
        if req.head.starts_with("POST") {
            (503, json!({"error":TOKEN}))
        } else {
            (200, json!([]))
        }
    })
    .await;
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a.png");
    std::fs::write(&file, b"\x89PNG\r\n\x1a\ncontent").unwrap();
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let c = call(
        "discord_upload_file",
        json!({"guild_id":"20","channel_id":"40","path":file}),
    );
    let e = tools.execute(&ctx(), &c).await.unwrap_err();
    assert!(e.downcast_ref::<UploadOutcomeUnknown>().is_some());
    assert!(!e.to_string().contains(TOKEN));
    assert!(tools.execute(&ctx(), &c).await.is_err());
    task.await.unwrap();
    assert_eq!(
        requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.head.starts_with("POST"))
            .count(),
        1
    );
}

#[tokio::test]
async fn failed_post_can_be_proven_successful_without_repeat() {
    let saved = Arc::new(Mutex::new(String::new()));
    let copy = saved.clone();
    let (base, _, task) = fixture(2, move |req| {
        if req.head.starts_with("POST") {
            *copy.lock().unwrap() = nonce_from(req);
            (500, json!({}))
        } else {
            (200, json!([msg(&copy.lock().unwrap(), "a.png", 9)]))
        }
    })
    .await;
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a.png");
    std::fs::write(&file, b"\x89PNG\r\n\x1a\nx").unwrap();
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "discord_upload_file",
                    json!({"guild_id":"20","channel_id":"40","path":file})
                )
            )
            .await
            .is_ok()
    );
    task.await.unwrap();
}

#[tokio::test]
async fn no_effect_denials_are_typed_before_any_network() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = config("http://127.0.0.1:9/api/v10".into());
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    for args in [
        json!({"guild_id":"20","channel_id":"41","path":"/nope"}),
        json!({"guild_id":"21","channel_id":"40","path":"/nope"}),
    ] {
        assert!(
            tools
                .execute(&ctx(), &call("discord_upload_file", args))
                .await
                .unwrap_err()
                .downcast_ref::<ToolDispatchRefused>()
                .is_some()
        );
    }
    let file = tmp.path().join("a.png");
    std::fs::write(&file, b"\x89PNG\r\n\x1a\nx").unwrap();
    let link = tmp.path().join("link.png");
    std::os::unix::fs::symlink(&file, &link).unwrap();
    assert!(
        tools
            .execute(
                &ctx(),
                &call(
                    "discord_upload_file",
                    json!({"guild_id":"20","channel_id":"40","path":link})
                )
            )
            .await
            .unwrap_err()
            .downcast_ref::<ToolDispatchRefused>()
            .is_some()
    );
    let context = ctx();
    context.cancellation.cancel();
    assert!(
        tools
            .execute(
                &context,
                &call(
                    "discord_upload_file",
                    json!({"guild_id":"20","channel_id":"40","path":file})
                )
            )
            .await
            .unwrap_err()
            .downcast_ref::<ToolDispatchRefused>()
            .is_some()
    );
}

#[tokio::test]
async fn limits_magic_and_rate_limit_no_retry() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config("http://127.0.0.1:9/api/v10".into());
    cfg.max_file_bytes = 8;
    let file = tmp.path().join("a.png");
    std::fs::write(&file, b"\x89PNG\r\n\x1a\nlarge").unwrap();
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let c = call(
        "discord_upload_file",
        json!({"guild_id":"20","channel_id":"40","path":file}),
    );
    assert!(
        tools
            .execute(&ctx(), &c)
            .await
            .unwrap_err()
            .downcast_ref::<ToolDispatchRefused>()
            .is_some()
    );
    let (base, _, task) = fixture(1, |_| (429, json!({"retry_after":1.25,"secret":TOKEN}))).await;
    let cfg = config(base);
    let tools = DiscordTools::new(policy(&cfg, tmp.path()), cfg).unwrap();
    let e = tools.execute(&ctx(), &c).await.unwrap_err();
    assert!(e.to_string().contains("1.25"));
    assert!(!e.to_string().contains(TOKEN));
    task.await.unwrap();
}

#[test]
fn defaults_endpoints_and_nonce_scope() {
    let tools = DiscordTools::new(PolicyConfig::default(), DiscordToolsConfig::default()).unwrap();
    assert!(tools.definitions().is_empty());
    let cfg = DiscordToolsConfig {
        api_base: "https://evil.example/api/v10".into(),
        ..Default::default()
    };
    assert!(DiscordTools::new(PolicyConfig::default(), cfg).is_err());
    let cfg = DiscordToolsConfig {
        token_env: "DISCORD_TOKEN".into(),
        ..Default::default()
    };
    assert!(DiscordTools::new(PolicyConfig::default(), cfg).is_err());
    let a = ctx();
    let mut b = ctx();
    b.scope.user = "another".into();
    assert_ne!(upload_nonce(&a, "call"), upload_nonce(&b, "call"));
    b = ctx();
    b.turn_id = "other".into();
    assert_ne!(upload_nonce(&a, "call"), upload_nonce(&b, "call"));
    assert_eq!(upload_nonce(&a, "call").len(), 25);
    let cfg = DiscordToolsConfig {
        fixture_mode: true,
        api_base: "http://127.0.0.1:9/api/v10".into(),
        enabled: true,
        bot_user_id: "10".into(),
        ..Default::default()
    };
    assert!(DiscordTools::new(PolicyConfig::default(), cfg).is_err());
}
