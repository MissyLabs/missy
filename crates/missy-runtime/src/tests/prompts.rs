use super::*;
use std::sync::Mutex;

struct Resolver(Mutex<HashMap<String, String>>);
impl ScopedPromptResolver for Resolver {
    fn resolve(&self, scope: &Scope, base: &str) -> Result<String> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .get(&scope.user)
            .cloned()
            .unwrap_or_else(|| base.into()))
    }
    fn configuration_fingerprint(&self) -> String {
        "scoped-test-registry-v1".into()
    }
}
fn resolver() -> Arc<Resolver> {
    Arc::new(Resolver(Mutex::new(HashMap::from([
        ("a".into(), "persona alpha v1".into()),
        ("b".into(), "persona beta".into()),
    ]))))
}
#[tokio::test]
async fn two_scopes_share_runtime_but_pin_distinct_prompts() {
    let provider = Arc::new(Script::new(vec![
        completion("one", "stop"),
        completion("two", "stop"),
        completion("three", "stop"),
    ]));
    let (_dir, rt) = setup(provider.clone(), tools(), RuntimeConfig::default());
    let resolver = resolver();
    let rt = rt.with_prompt_resolver(resolver.clone()).unwrap();
    let fingerprint = rt.configuration_fingerprint().unwrap();
    let first = rt
        .run(scope("a"), "request".into(), "one".into())
        .await
        .unwrap();
    rt.run(scope("b"), "request".into(), "two".into())
        .await
        .unwrap();
    resolver
        .0
        .lock()
        .unwrap()
        .insert("a".into(), "persona alpha v2".into());
    assert_eq!(fingerprint, rt.configuration_fingerprint().unwrap());
    let duplicate = rt
        .run(scope("a"), "request".into(), "one".into())
        .await
        .unwrap();
    assert!(duplicate.deduplicated);
    assert_eq!(duplicate.turn_id, first.turn_id);
    rt.run(scope("a"), "new request".into(), "three".into())
        .await
        .unwrap();
    let messages = provider.messages.lock().unwrap();
    assert!(messages[0][0].content.starts_with("persona alpha v1\n"));
    assert!(messages[1][0].content.starts_with("persona beta\n"));
    assert!(messages[2][0].content.starts_with("persona alpha v2\n"));
    assert!(
        rt.store()
            .admission_metadata(&scope("b"), &first.turn_id)
            .is_err()
    );
}
#[tokio::test]
async fn admission_pins_prompt_and_memory_before_capacity_and_survives_restart() {
    let provider = Arc::new(Script::new(vec![completion("done", "stop")]));
    let cfg = RuntimeConfig {
        max_concurrent_turns: 1,
        memory_context_enabled: true,
        ..RuntimeConfig::default()
    };
    let (dir, rt) = setup(provider.clone(), tools(), cfg.clone());
    rt.store()
        .memory_put(
            &scope("a"),
            "db",
            "database old selected fact",
            "user",
            "untrusted",
        )
        .unwrap();
    let resolver = resolver();
    let rt = rt.with_prompt_resolver(resolver.clone()).unwrap();
    let permit = rt.inner.permits.clone().acquire_owned().await.unwrap();
    let queued = rt
        .run(scope("a"), "database request".into(), "queued".into())
        .await
        .unwrap();
    assert_eq!(queued.status, "admitted");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    let before = rt
        .store()
        .admission_metadata(&scope("a"), &queued.turn_id)
        .unwrap();
    resolver
        .0
        .lock()
        .unwrap()
        .insert("a".into(), "persona alpha v2".into());
    rt.store()
        .memory_put(
            &scope("a"),
            "db",
            "database new fact must not drift",
            "user",
            "untrusted",
        )
        .unwrap();
    drop(permit);
    drop(rt);
    let rt = Runtime::new(
        Arc::new(Store::open(dir.path().join("runtime.db")).unwrap()),
        provider.clone(),
        tools(),
        "trusted system".into(),
        cfg,
    )
    .unwrap()
    .with_prompt_resolver(resolver)
    .unwrap();
    let done = rt.resume(scope("a"), &queued.turn_id).await.unwrap();
    assert_eq!(done.status, "completed");
    assert_eq!(
        before,
        rt.store()
            .admission_metadata(&scope("a"), &queued.turn_id)
            .unwrap()
    );
    let messages = provider.messages.lock().unwrap();
    assert!(messages[0][0].content.starts_with("persona alpha v1\n"));
    let memory = messages[0]
        .iter()
        .find(|m| m.content.contains("untrusted_scoped_memory"))
        .unwrap();
    assert_eq!(memory.role, "user");
    assert!(memory.content.contains("old selected fact"));
    assert!(!memory.content.contains("new fact must not drift"));
    assert!(!messages[0][0].content.contains("old selected fact"));
}
#[tokio::test]
async fn continuation_restart_reuses_pinned_prompt_not_new_revision() {
    let provider = Arc::new(Script::new(vec![
        completion("first ", "length"),
        Err(anyhow!("offline")),
        completion("second", "stop"),
    ]));
    let cfg = RuntimeConfig {
        max_cost_microusd: 0,
        ..RuntimeConfig::default()
    };
    let (dir, rt) = setup(provider.clone(), tools(), cfg.clone());
    let resolver = resolver();
    let rt = rt.with_prompt_resolver(resolver.clone()).unwrap();
    let paused = rt
        .run(scope("a"), "original".into(), "request".into())
        .await
        .unwrap();
    assert_eq!(paused.status, "interrupted");
    let before = rt
        .load(
            &scope("a"),
            &rt.status(&scope("a"), &paused.turn_id).unwrap(),
        )
        .unwrap();
    resolver
        .0
        .lock()
        .unwrap()
        .insert("a".into(), "persona alpha v2".into());
    drop(rt);
    let rt = Runtime::new(
        Arc::new(Store::open(dir.path().join("runtime.db")).unwrap()),
        provider.clone(),
        tools(),
        "trusted system".into(),
        cfg,
    )
    .unwrap()
    .with_prompt_resolver(resolver)
    .unwrap();
    let done = rt.resume(scope("a"), &paused.turn_id).await.unwrap();
    assert_eq!(done.status, "completed");
    assert_eq!(done.content, "first second");
    let after = rt
        .load(
            &scope("a"),
            &rt.status(&scope("a"), &paused.turn_id).unwrap(),
        )
        .unwrap();
    assert_eq!(before.prompt_snapshot, after.prompt_snapshot);
    assert!(after.continuations >= before.continuations);
    for messages in provider.messages.lock().unwrap().iter() {
        assert!(messages[0].content.starts_with("persona alpha v1\n"));
    }
}
#[tokio::test]
async fn exact_tools_pin_scope_snapshot_and_preserve_dedup_without_provider() {
    let provider = Arc::new(Script::new(vec![]));
    let (_dir, rt) = setup(
        provider.clone(),
        tools(),
        RuntimeConfig {
            memory_context_enabled: true,
            ..RuntimeConfig::default()
        },
    );
    rt.store()
        .memory_put(
            &scope("a"),
            "value",
            "value remembered fact",
            "user",
            "untrusted",
        )
        .unwrap();
    let resolver = resolver();
    let rt = rt.with_prompt_resolver(resolver.clone()).unwrap();
    let first = rt
        .run_tools(scope("a"), vec![call("exact")], "literal".into())
        .await
        .unwrap();
    resolver
        .0
        .lock()
        .unwrap()
        .insert("a".into(), "persona alpha v2".into());
    let again = rt
        .run_tools(scope("a"), vec![call("exact")], "literal".into())
        .await
        .unwrap();
    assert!(again.deduplicated);
    assert_eq!(again.turn_id, first.turn_id);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        rt.admitted_prompt(&scope("a"), &first.turn_id)
            .unwrap()
            .unwrap()
            .text,
        "persona alpha v1"
    );
}
