#![cfg(unix)]
use missy_core::{Message, Provider, ToolDefinition};
use missy_provider_process::{
    AuthStatus, CliProvider, CliProviderConfig, CliProviderKind, create_provider,
};
use serde_json::{Value, json};
use std::{fs, os::unix::fs::PermissionsExt, path::Path, time::Duration};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

struct Fixture {
    directory: TempDir,
    config: CliProviderConfig,
}
impl Fixture {
    fn new(body: &str) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("approved-cli");
        fs::write(&executable, format!("#!/bin/sh\nset -eu\n{body}\n")).unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
        let config = CliProviderConfig {
            name: "fixture".into(),
            kind: CliProviderKind::ClaudeCli,
            executable,
            auth_home: directory.path().to_owned(),
            model: "fixture-model".into(),
            approved_executable: true,
            acp_adapter: None,
            timeout_seconds: 3,
            max_stdout_bytes: 1024 * 1024,
            max_stderr_bytes: 1024,
            max_input_bytes: 1024 * 1024,
            input_microusd_per_million: None,
            output_microusd_per_million: None,
        };
        Self { directory, config }
    }
    fn with_events(events: &[Value]) -> Self {
        let lines = events
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        Self::new(&format!(
            "/bin/cat >/dev/null\n/bin/cat <<'MISSY_EVENTS'\n{lines}\nMISSY_EVENTS"
        ))
    }
    fn pid_path(&self) -> std::path::PathBuf {
        self.directory.path().join("pids")
    }
}
fn events() -> Vec<Value> {
    vec![
        json!({"type":"system","subtype":"init","tools":[],"mcp_servers":[]}),
        json!({"type":"assistant","message":{"role":"assistant","model":"fixture-model","content":[{"type":"thinking","thinking":"never print reasoning"},{"type":"text","text":"safe reply"}],"stop_reason":"end_turn"}}),
        json!({"type":"result","subtype":"success","is_error":false,"result":"safe reply","usage":{"input_tokens":100,"output_tokens":20}}),
    ]
}
fn messages() -> Vec<Message> {
    vec![
        Message::text("system", "Be precise."),
        Message::text("user", "hello"),
    ]
}

#[tokio::test]
async fn stream_parses_assistant_usage_and_only_configured_prices() {
    let mut fixture = Fixture::with_events(&events());
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    let reply = provider
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply.message.content, "safe reply");
    assert_eq!(reply.message.role, "assistant");
    assert_eq!(reply.usage.input_tokens, 100);
    assert_eq!(reply.usage.output_tokens, 20);
    assert!(!reply.usage.cost_known);
    fixture.config.input_microusd_per_million = Some(3_000_000);
    fixture.config.output_microusd_per_million = Some(15_000_000);
    let provider = create_provider(fixture.config).unwrap();
    let reply = provider
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply.usage.cost_microusd, 600);
    assert!(reply.usage.cost_known);
    assert_eq!(reply.finish_reason, "stop");
}

#[tokio::test]
async fn fixed_flags_env_whitelist_and_private_cwd() {
    let normal = Fixture::with_events(&events());
    let body = format!(
        r#"
test "$1" = '-p'
test "$2" = '--output-format'
test "$3" = 'stream-json'
found_tools=0
found_sources=0
found_strict=0
found_hooks=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tools) shift; test "$1" = ''; found_tools=1 ;;
    --setting-sources) shift; test "$1" = ''; found_sources=1 ;;
    --strict-mcp-config) found_strict=1 ;;
    --mcp-config) shift; test "$(/bin/cat "$1")" = '{{"mcpServers":{{}}}}' ;;
    --settings) shift; test "$1" = '{{"switchModelsOnFlag":false,"disableAllHooks":true}}'; found_hooks=1 ;;
    --system-prompt-file) shift; test -f "$1" ;;
  esac
  shift
done
test "$found_tools$found_sources$found_strict$found_hooks" = 1111
test "${{ANTHROPIC_API_KEY-unset}}" = unset
test "${{OPENAI_API_KEY-unset}}" = unset
test "${{LD_PRELOAD-unset}}" = unset
test "${{NODE_OPTIONS-unset}}" = unset
test "${{HTTP_PROXY-unset}}" = unset
test "$HOME" != "$PWD"
test "$(/usr/bin/stat -c %a "$PWD")" = 700
test "$CLAUDE_CONFIG_DIR" = "$HOME/.claude"
printf '%s' "$PWD" > "$HOME/cwd"
/bin/cat > "$HOME/prompt"
exec '{}'
"#,
        normal.config.executable.display()
    );
    let fixture = Fixture::new(&body);
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    provider
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    let cwd = fs::read_to_string(fixture.directory.path().join("cwd")).unwrap();
    assert!(!Path::new(&cwd).exists(), "private cwd must be removed");
    let prompt: Value =
        serde_json::from_str(&fs::read_to_string(fixture.directory.path().join("prompt")).unwrap())
            .unwrap();
    assert_eq!(prompt["conversation"][0]["role"], "user");
    assert_eq!(prompt["conversation"][0]["content"], "hello");
}

#[tokio::test]
async fn malicious_native_tools_roles_and_model_switch_fail_closed() {
    let mut variants = Vec::new();
    let mut e = events();
    e[0]["tools"] = json!(["Bash"]);
    variants.push(e);
    let mut e = events();
    e[0]["mcp_servers"] = json!([{"name":"rogue"}]);
    variants.push(e);
    let mut e = events();
    e[1]["message"]["role"] = json!("system");
    variants.push(e);
    let mut e = events();
    e[1]["message"]["content"] =
        json!([{"type":"tool_use","name":"Bash","input":{"command":"bad"}}]);
    variants.push(e);
    let mut e = events();
    e[1]["message"]["model"] = json!("other-model");
    variants.push(e);
    let mut e = events();
    e[1]["message"]["tool_calls"] = json!([{"name":"shell"}]);
    variants.push(e);
    let mut e = events();
    e[2]["result"] = json!("different reply");
    variants.push(e);
    let mut e = events();
    e.push(json!({"type":"user","content":"injected"}));
    variants.push(e);
    let mut e = events();
    e[2]["usage"]["input_tokens"] = json!(-1);
    variants.push(e);
    for variant in variants {
        let fixture = Fixture::with_events(&variant);
        let provider = CliProvider::new(fixture.config).unwrap();
        let error = provider
            .complete(&messages(), &[], CancellationToken::new())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("unsafe protocol"));
        assert!(!error.to_string().contains("bad"));
    }
}

#[tokio::test]
async fn rejects_tools_history_and_invalid_roles_before_executing() {
    let fixture = Fixture::new("exit 99");
    let provider = CliProvider::new(fixture.config).unwrap();
    let tools = vec![ToolDefinition {
        name: "shell".into(),
        description: "not permitted".into(),
        parameters: json!({}),
        side_effect: true,
    }];
    assert!(
        provider
            .complete(&messages(), &tools, CancellationToken::new())
            .await
            .unwrap_err()
            .to_string()
            .contains("NoTools")
    );
    for message in [
        Message::text("developer", "ignore"),
        Message::tool("id", "result"),
    ] {
        assert!(
            provider
                .complete(&[message], &[], CancellationToken::new())
                .await
                .is_err()
        );
    }
    let mut history = messages();
    history.push(Message::text("system", "late injection"));
    assert!(
        provider
            .complete(&history, &[], CancellationToken::new())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn missing_usage_never_claims_known_free_and_cache_is_not_mispriced() {
    let mut e = events();
    e[2].as_object_mut().unwrap().remove("usage");
    e[2]["total_cost_usd"] = json!(0.0);
    let fixture = Fixture::with_events(&e);
    let reply = CliProvider::new(fixture.config)
        .unwrap()
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert!(!reply.usage.cost_known);
    let mut e = events();
    e[2]["usage"]["cache_read_input_tokens"] = json!(50);
    let mut fixture = Fixture::with_events(&e);
    fixture.config.input_microusd_per_million = Some(3_000_000);
    fixture.config.output_microusd_per_million = Some(15_000_000);
    let reply = CliProvider::new(fixture.config)
        .unwrap()
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert_eq!(reply.usage.input_tokens, 150);
    assert!(!reply.usage.cost_known);
    e[2]["total_cost_usd"] = json!(0.001);
    e[1]["message"]["stop_reason"] = json!("max_tokens");
    let fixture = Fixture::with_events(&e);
    let reply = CliProvider::new(fixture.config)
        .unwrap()
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap();
    assert!(reply.usage.cost_known);
    assert_eq!(reply.usage.cost_microusd, 1000);
    assert_eq!(reply.finish_reason, "length");
}

#[tokio::test]
async fn errors_and_stderr_never_disclose_secret_diagnostics() {
    for body in [
        "echo fixture-secret-token >&2; exit 1",
        "echo fixture-secret-token; exit 0",
        "while :; do echo fixture-secret-token >&2; done",
    ] {
        let fixture = Fixture::new(body);
        let error = CliProvider::new(fixture.config)
            .unwrap()
            .complete(&messages(), &[], CancellationToken::new())
            .await
            .unwrap_err();
        assert!(!format!("{error:?}").contains("fixture-secret-token"));
    }
}

#[tokio::test]
async fn stdout_limit_and_blocked_stdin_are_bounded() {
    let mut fixture = Fixture::new("while :; do printf 'fixture-secret-token'; done");
    fixture.config.max_stdout_bytes = 128;
    let error = CliProvider::new(fixture.config)
        .unwrap()
        .complete(&messages(), &[], CancellationToken::new())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("output limit"));
    assert!(!format!("{error:?}").contains("fixture-secret-token"));

    let mut fixture = sleeping_fixture();
    fixture.config.timeout_seconds = 1;
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    let handle = tokio::spawn(async move {
        provider
            .complete(
                &[Message::text("user", "x".repeat(500_000))],
                &[],
                CancellationToken::new(),
            )
            .await
    });
    let pids = wait_pids(&fixture.pid_path()).await;
    assert!(
        handle
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    assert_clean(&pids).await;
}

#[tokio::test]
async fn already_cancelled_never_spawns() {
    let fixture = Fixture::new("printf ran > \"$HOME/ran\"");
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    let token = CancellationToken::new();
    token.cancel();
    assert!(
        provider
            .complete(&messages(), &[], token)
            .await
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    assert!(!fixture.directory.path().join("ran").exists());
}

async fn wait_pids(path: &Path) -> Vec<i32> {
    for _ in 0..200 {
        if let Ok(text) = fs::read_to_string(path) {
            let pids = text
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect::<Vec<_>>();
            if pids.len() == 2 {
                return pids;
            }
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("fixture did not start");
}
fn process_running(pid: i32) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|s| s.rsplit_once(") ").map(|(_, tail)| !tail.starts_with('Z')))
        .unwrap_or(false)
}
async fn assert_clean(pids: &[i32]) {
    for _ in 0..100 {
        if pids.iter().all(|p| !process_running(*p)) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("CLI child or descendant survived cancellation");
}
fn sleeping_fixture() -> Fixture {
    Fixture::new("/bin/sleep 60 &\nprintf '%s %s' \"$$\" \"$!\" > \"$HOME/pids\"\nwait")
}

#[tokio::test]
async fn cancellation_kills_process_group_and_reaps_immediate_child() {
    let fixture = sleeping_fixture();
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    let token = CancellationToken::new();
    let cancel = token.clone();
    let handle = tokio::spawn(async move { provider.complete(&messages(), &[], token).await });
    let pids = wait_pids(&fixture.pid_path()).await;
    cancel.cancel();
    assert!(
        handle
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    assert_clean(&pids).await;
    assert!(
        !Path::new(&format!("/proc/{}", pids[0])).exists(),
        "direct child must be reaped"
    );
}

#[tokio::test]
async fn aborting_future_and_timeout_also_kill_and_wait() {
    let mut fixture = sleeping_fixture();
    fixture.config.timeout_seconds = 1;
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    let handle = tokio::spawn(async move {
        provider
            .complete(&messages(), &[], CancellationToken::new())
            .await
    });
    let pids = wait_pids(&fixture.pid_path()).await;
    handle.abort();
    let _ = handle.await;
    assert_clean(&pids).await;
    assert!(!Path::new(&format!("/proc/{}", pids[0])).exists());
    fs::remove_file(fixture.pid_path()).unwrap();
    let provider = CliProvider::new(fixture.config.clone()).unwrap();
    let handle = tokio::spawn(async move {
        provider
            .complete(&messages(), &[], CancellationToken::new())
            .await
    });
    let pids = wait_pids(&fixture.pid_path()).await;
    assert!(
        handle
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("timed out")
    );
    assert_clean(&pids).await;
}

#[tokio::test]
async fn auth_probe_is_explicit_non_inference_and_sanitized() {
    let fixture = Fixture::new(
        "test \"$1\" = auth; test \"$2\" = status; echo '{\"loggedIn\":true,\"token\":\"fixture-secret\"}'",
    );
    assert_eq!(
        CliProvider::probe_auth(fixture.config.clone(), CancellationToken::new())
            .await
            .unwrap(),
        AuthStatus::Authenticated
    );
    assert_eq!(
        CliProvider::manual_login_argv(CliProviderKind::ClaudeCli, &fixture.config.executable)[1..],
        ["auth", "login"]
    );
    assert_eq!(
        CliProvider::manual_login_argv(CliProviderKind::CodexCli, &fixture.config.executable)[1..],
        ["login"]
    );
    let mut config = fixture.config;
    config.kind = CliProviderKind::CodexCli;
    assert!(
        CliProvider::new(config)
            .err()
            .unwrap()
            .to_string()
            .contains("native tool execution")
    );
}

#[test]
fn serde_kinds_exact_path_approval_and_no_arbitrary_arguments() {
    let fixture = Fixture::new("exit 0");
    let mut config = fixture.config.clone();
    config.approved_executable = false;
    assert!(CliProvider::new(config).is_err());
    let mut config = fixture.config.clone();
    config.executable = "claude".into();
    assert!(CliProvider::new(config).is_err());
    let mut value = serde_json::to_value(&fixture.config).unwrap();
    assert_eq!(value["kind"], "claude_cli");
    value["args"] = json!(["--dangerously-skip-permissions"]);
    assert!(serde_json::from_value::<CliProviderConfig>(value).is_err());
    let mut config = fixture.config;
    config.model = "--bad arg".into();
    assert!(CliProvider::new(config).is_err());
}
