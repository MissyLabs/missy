use super::*;
use crate::{
    crdt::MAX_CAUSAL_COUNTER,
    envelope::{CLOCK_SKEW_SECONDS, MAX_TTL_SECONDS},
};
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
};
use serde_json::json;
use std::{
    collections::BTreeSet,
    fs,
    net::SocketAddr,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
};
use tempfile::TempDir;
use tower::ServiceExt;

fn scope() -> Scope {
    Scope {
        tenant: "tenant".into(),
        user: "user".into(),
        channel: "channel".into(),
        session: "session".into(),
    }
}
fn all_caps() -> BTreeSet<Capability> {
    [
        Capability::MemoryRead,
        Capability::MemoryWrite,
        Capability::Delegate,
        Capability::Gossip,
        Capability::PolicyVote,
    ]
    .into()
}
fn mkdir(path: &Path) {
    fs::create_dir(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
struct Lab {
    _temp: TempDir,
    alice: Identity,
    bob: Identity,
    carol: Identity,
    config: MeshConfig,
}
impl Lab {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        fs::set_permissions(root, fs::Permissions::from_mode(0o700)).unwrap();
        let alice = Identity::create(root.join("alice.key")).unwrap();
        let bob = Identity::create(root.join("bob.key")).unwrap();
        let carol = Identity::create(root.join("carol.key")).unwrap();
        let config = MeshConfig {
            version: 1,
            identity_path: root.join("bob.key"),
            state_dir: root.join("bob-state"),
            authorized_peers: vec![Self::peer(&alice), Self::peer(&carol)],
            listen: "127.0.0.1:0".parse().unwrap(),
            http_policy: missy_core::PolicyConfig::default(),
        };
        Self {
            _temp: temp,
            alice,
            bob,
            carol,
            config,
        }
    }
    fn peer(identity: &Identity) -> Peer {
        Peer {
            public_key: identity.public_key(),
            endpoint: None,
            grants: vec![Grant {
                scope: scope(),
                capabilities: all_caps(),
            }],
        }
    }
    fn node(&self) -> MeshNode {
        MeshNode::open(self.config.clone()).unwrap()
    }
    fn sign(&self, purpose: Purpose, payload: Value, now: u64) -> SignedEnvelope {
        SignedEnvelope::sign(
            &self.alice,
            self.bob.peer_id(),
            scope(),
            purpose,
            payload,
            now,
            60,
        )
        .unwrap()
    }
}
fn resign(identity: &Identity, env: &mut SignedEnvelope) {
    env.signature = identity.sign(&env.body.signing_bytes().unwrap());
}
fn verify(node: &MeshNode, env: &SignedEnvelope, now: u64) -> Result<VerifiedEnvelope> {
    node.verify_json(&env.to_json()?, now)
}

#[test]
fn identity_is_explicit_private_persistent_and_never_regenerated() {
    let lab = Lab::new();
    let path = lab.config.identity_path.clone();
    assert_eq!(
        Identity::load(&path).unwrap().public_key(),
        lab.bob.public_key()
    );
    assert!(Identity::create(&path).is_err());
    assert!(Identity::load(path.with_file_name("missing.key")).is_err());
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(Identity::load(&path).is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let link = path.with_file_name("link.key");
    symlink(&path, &link).unwrap();
    assert!(Identity::load(link).is_err());
    let parent = lab._temp.path().join("link-dir");
    symlink(lab._temp.path(), &parent).unwrap();
    assert!(Identity::load(parent.join("bob.key")).is_err());
    let hard = path.with_file_name("hard.key");
    fs::hard_link(&path, &hard).unwrap();
    assert!(Identity::load(&path).is_err());
}

#[test]
fn peers_are_persistent_and_operator_scoped_no_tofu_or_escalation() {
    let lab = Lab::new();
    let node = lab.node();
    assert_eq!(node.peers().unwrap().len(), 2);
    let removed = lab.alice.peer_id();
    assert!(node.peer_remove(&removed).unwrap());
    assert!(verify(&node, &lab.sign(Purpose::Gossip, json!({}), 100), 100).is_err());
    drop(node);
    let node = lab.node();
    assert_eq!(node.peers().unwrap().len(), 1);
    node.peer_add(Lab::peer(&lab.alice)).unwrap();
    let mut escalation = Lab::peer(&lab.alice);
    escalation.grants[0].scope.session = "other".into();
    assert!(node.peer_add(escalation).is_err());
    let stranger = Identity::create(lab._temp.path().join("stranger.key")).unwrap();
    assert!(node.peer_add(Lab::peer(&stranger)).is_err());
    let mut allowed = Lab::peer(&lab.alice);
    allowed.grants[0].capabilities.clear();
    node.peer_add(allowed).unwrap();
    assert!(verify(&node, &lab.sign(Purpose::Gossip, json!({}), 100), 100).is_err());
    assert!(lab.node_open_result().is_err()); // exclusive state lock, not competing registries
}
impl Lab {
    fn node_open_result(&self) -> Result<MeshNode> {
        MeshNode::open(self.config.clone())
    }
}

#[test]
fn strict_signatures_all_security_fields_and_payload_are_authenticated() {
    let lab = Lab::new();
    let node = lab.node();
    let env = lab.sign(Purpose::Gossip, json!({"x":1,"nested":{"a":2,"b":3}}), 100);
    let mut variants = Vec::new();
    let mut e = env.clone();
    e.body.sender = lab.carol.peer_id();
    variants.push(e);
    let mut e = env.clone();
    e.body.recipient = lab.carol.peer_id();
    variants.push(e);
    let mut e = env.clone();
    e.body.scope.session = "different".into();
    variants.push(e);
    let mut e = env.clone();
    e.body.purpose = Purpose::DelegateRequest;
    variants.push(e);
    let mut e = env.clone();
    e.body.timestamp = 101;
    variants.push(e);
    let mut e = env.clone();
    e.body.expiry = 159;
    variants.push(e);
    let mut e = env.clone();
    e.body.nonce = "11".repeat(16);
    variants.push(e);
    let mut e = env.clone();
    e.body.payload["x"] = json!(2);
    variants.push(e);
    let mut e = env.clone();
    e.body.version = 2;
    variants.push(e);
    for e in variants {
        assert!(verify(&node, &e, 100).is_err());
    }
    assert!(verify(&node, &env, 100).is_ok());
    assert!(verify(&node, &env, 100).is_err());
}

#[test]
fn durable_replay_denies_cross_scope_cross_purpose_and_restart() {
    let mut lab = Lab::new();
    let mut second = scope();
    second.session = "second".into();
    lab.config.authorized_peers[0].grants.push(Grant {
        scope: second.clone(),
        capabilities: all_caps(),
    });
    let node = lab.node();
    let env = lab.sign(Purpose::Gossip, json!({}), 100);
    verify(&node, &env, 100).unwrap();
    let mut changed = env.clone();
    changed.body.scope = second;
    changed.body.purpose = Purpose::MemoryRead;
    resign(&lab.alice, &mut changed);
    assert!(
        verify(&node, &changed, 100)
            .unwrap_err_text()
            .contains("replay")
    );
    drop(node);
    let node = lab.node();
    assert!(verify(&node, &env, 100).is_err());
    let next = lab.sign(Purpose::Gossip, json!({}), 101);
    verify(&node, &next, 101).unwrap();
    assert!(verify(&node, &lab.sign(Purpose::Gossip, json!({}), 100), 100).is_err());
}
trait ErrorText {
    fn unwrap_err_text(self) -> String;
}
impl<T> ErrorText for Result<T> {
    fn unwrap_err_text(self) -> String {
        match self {
            Ok(_) => panic!("expected failure"),
            Err(e) => e.to_string(),
        }
    }
}

#[test]
fn expiry_recipient_reflection_crosspeer_and_scope_are_fail_closed() {
    let lab = Lab::new();
    let node = lab.node();
    let env = lab.sign(Purpose::Gossip, json!({}), 100);
    assert!(verify(&node, &env, 160).is_err());
    let mut e = env.clone();
    e.body.timestamp = 100 + CLOCK_SKEW_SECONDS + 1;
    e.body.expiry = e.body.timestamp + 60;
    resign(&lab.alice, &mut e);
    assert!(verify(&node, &e, 100).is_err());
    let mut e = env.clone();
    e.body.expiry = e.body.timestamp + MAX_TTL_SECONDS + 1;
    resign(&lab.alice, &mut e);
    assert!(verify(&node, &e, 100).is_err());
    let mut e = env.clone();
    e.body.recipient = lab.carol.peer_id();
    resign(&lab.alice, &mut e);
    assert!(verify(&node, &e, 100).is_err());
    let mut e = env.clone();
    e.body.sender = lab.carol.peer_id();
    resign(&lab.alice, &mut e);
    assert!(verify(&node, &e, 100).is_err());
    let mut e = env.clone();
    e.body.scope.tenant = "ungranted".into();
    resign(&lab.alice, &mut e);
    assert!(verify(&node, &e, 100).is_err());
    let mut e = env.clone();
    e.body.sender = lab.bob.peer_id();
    e.body.recipient = lab.bob.peer_id();
    resign(&lab.bob, &mut e);
    assert!(verify(&node, &e, 100).is_err());
    assert!(
        SignedEnvelope::sign(
            &lab.alice,
            lab.alice.peer_id(),
            scope(),
            Purpose::Gossip,
            json!({}),
            100,
            60
        )
        .is_err()
    );
    assert!(verify(&node, &env, 100).is_ok());
}

#[test]
fn bounded_json_rejects_duplicates_floats_unknown_fields_and_excess_depth() {
    let lab = Lab::new();
    let env = lab.sign(Purpose::Gossip, json!({"a":1,"b":2}), 100);
    let data = String::from_utf8(env.to_json().unwrap()).unwrap();
    let duplicate = data.replace("\"a\":1", "\"a\":1,\"a\":2");
    assert!(SignedEnvelope::from_json(duplicate.as_bytes()).is_err());
    let duplicate = data.replace("\"version\":1", "\"version\":1,\"version\":1");
    assert!(SignedEnvelope::from_json(duplicate.as_bytes()).is_err());
    let float = data.replace("\"a\":1", "\"a\":1.0");
    assert!(SignedEnvelope::from_json(float.as_bytes()).is_err());
    let unknown = data.replace("\"version\":1", "\"version\":1,\"admin\":true");
    assert!(SignedEnvelope::from_json(unknown.as_bytes()).is_err());
    let reordered = data.replace("\"a\":1,\"b\":2", "\"b\":2,\"a\":1");
    let reordered = SignedEnvelope::from_json(reordered.as_bytes()).unwrap();
    reordered.authenticate(&lab.alice.public_key()).unwrap();
    let mut deep = json!(1);
    for _ in 0..20 {
        deep = json!([deep]);
    }
    assert!(
        SignedEnvelope::sign(
            &lab.alice,
            lab.bob.peer_id(),
            scope(),
            Purpose::Gossip,
            deep,
            100,
            60
        )
        .is_err()
    );
    assert!(SignedEnvelope::from_json(&vec![b' '; MAX_ENVELOPE_BYTES + 1]).is_err());
    assert!(
        SignedEnvelope::sign(
            &lab.alice,
            lab.bob.peer_id(),
            scope(),
            Purpose::Gossip,
            json!({"data":"a".repeat(MAX_ENVELOPE_BYTES)}),
            100,
            60
        )
        .is_err()
    );
}

#[test]
fn config_removal_overrides_persisted_grants_and_private_config_is_required() {
    let mut lab = Lab::new();
    let node = lab.node();
    drop(node);
    lab.config.authorized_peers[0].grants[0]
        .capabilities
        .remove(&Capability::Delegate);
    let node = lab.node();
    assert!(
        verify(
            &node,
            &lab.sign(
                Purpose::DelegateRequest,
                json!({"task_id":"x","task":"hi"}),
                100
            ),
            100
        )
        .is_err()
    );
    drop(node);
    let path = lab._temp.path().join("config.json");
    fs::write(&path, serde_json::to_vec(&lab.config).unwrap()).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    MeshConfig::load(&path).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(MeshConfig::load(&path).is_err());
    lab.config.listen = "0.0.0.0:7777".parse().unwrap();
    assert!(lab.config.validate().is_err());
}

fn update(value: Value, counter: u32) -> MemoryUpdate {
    MemoryUpdate {
        key: "shared".into(),
        value,
        deleted: false,
        timestamp: 100,
        counter,
    }
}
#[test]
fn crdt_ties_are_deterministic_even_same_identity_stamp_conflicts() {
    let lab = Lab::new();
    let mut entries = Vec::new();
    for (peer, value, counter) in [
        (&lab.alice, json!("a"), 0),
        (&lab.carol, json!("b"), 0),
        (&lab.alice, json!("z"), 0),
        (&lab.alice, json!("causal"), 1),
    ] {
        entries.push(Entry {
            scope: scope(),
            peer_id: peer.peer_id(),
            update: update(value, counter),
        });
    }
    let mut forward = LwwMap::default();
    for e in &entries {
        forward.apply_entry(e.clone(), 100).unwrap();
    }
    let mut reverse = LwwMap::default();
    for e in entries.iter().rev() {
        reverse.apply_entry(e.clone(), 100).unwrap();
    }
    assert_eq!(forward, reverse);
    for e in &entries {
        forward.apply_entry(e.clone(), 100).unwrap();
    }
    assert_eq!(forward, reverse);
    // Enumerate every permutation, including identical peer/stamp payloads.
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let indexes = [a, b, c, d];
                    if indexes.into_iter().collect::<BTreeSet<_>>().len() != 4 {
                        continue;
                    }
                    let mut replica = LwwMap::default();
                    for i in indexes {
                        replica.apply_entry(entries[i].clone(), 100).unwrap();
                    }
                    assert_eq!(replica, forward);
                }
            }
        }
    }
    let mut same = LwwMap::default();
    same.apply_entry(entries[0].clone(), 100).unwrap();
    same.apply_entry(entries[2].clone(), 100).unwrap();
    assert_eq!(same.get(&scope(), "shared"), Some(&json!("z")));
    let mut tomb = entries[2].clone();
    tomb.update.deleted = true;
    tomb.update.value = Value::Null;
    same.apply_entry(tomb, 100).unwrap();
    assert!(same.get(&scope(), "shared").is_none());
    same.apply_entry(entries[0].clone(), 100).unwrap();
    assert!(same.get(&scope(), "shared").is_none());
}

#[test]
fn signed_crdt_persists_scope_tombstones_and_bounds_causal_timestamp() {
    let lab = Lab::new();
    let node = lab.node();
    let env = lab.sign(
        Purpose::MemoryUpdate,
        serde_json::to_value(update(json!({"note":"hello"}), 0)).unwrap(),
        100,
    );
    let verified = verify(&node, &env, 100).unwrap();
    assert!(node.apply_memory(verified, 100).unwrap());
    assert_eq!(
        node.memory_get(&scope(), "shared").unwrap(),
        Some(json!({"note":"hello"}))
    );
    let mut other = scope();
    other.session = "isolated".into();
    assert!(node.memory_get(&other, "shared").unwrap().is_none());
    drop(node);
    let node = lab.node();
    assert!(node.memory_get(&scope(), "shared").unwrap().is_some());
    let mut bad = update(Value::Null, MAX_CAUSAL_COUNTER + 1);
    let env = lab.sign(
        Purpose::MemoryUpdate,
        serde_json::to_value(&bad).unwrap(),
        100,
    );
    assert!(
        node.apply_memory(verify(&node, &env, 100).unwrap(), 100)
            .is_err()
    );
    bad.counter = 0;
    bad.timestamp = 101;
    let env = lab.sign(
        Purpose::MemoryUpdate,
        serde_json::to_value(&bad).unwrap(),
        100,
    );
    assert!(
        node.apply_memory(verify(&node, &env, 100).unwrap(), 100)
            .is_err()
    );
    let mut tomb = update(Value::Null, 1);
    tomb.deleted = true;
    let env = lab.sign(
        Purpose::MemoryUpdate,
        serde_json::to_value(tomb).unwrap(),
        100,
    );
    node.apply_memory(verify(&node, &env, 100).unwrap(), 100)
        .unwrap();
    drop(node);
    let node = lab.node();
    assert!(node.memory_get(&scope(), "shared").unwrap().is_none());
}

#[test]
fn quorum_counts_unique_authenticated_scoped_voters_and_exact_proposals() {
    let lab = Lab::new();
    let node = lab.node();
    let proposal = "aa".repeat(32);
    let eligible: BTreeSet<_> = [lab.alice.peer_id(), lab.carol.peer_id()].into();
    assert!(
        Quorum::new(
            lab.bob.peer_id(),
            scope(),
            proposal.clone(),
            eligible.clone(),
            0,
            200
        )
        .is_err()
    );
    assert!(
        Quorum::new(
            lab.bob.peer_id(),
            scope(),
            proposal.clone(),
            eligible.clone(),
            3,
            200
        )
        .is_err()
    );
    let mut quorum = Quorum::new(
        lab.bob.peer_id(),
        scope(),
        proposal.clone(),
        eligible,
        2,
        200,
    )
    .unwrap();
    let payload = serde_json::to_value(Vote {
        proposal_digest: proposal,
        approve: true,
    })
    .unwrap();
    let alice = lab.sign(Purpose::PolicyVote, payload.clone(), 100);
    quorum
        .add(verify(&node, &alice, 100).unwrap(), 100)
        .unwrap();
    assert!(!quorum.result(100).approved);
    let duplicate = lab.sign(Purpose::PolicyVote, payload.clone(), 100);
    assert!(
        quorum
            .add(verify(&node, &duplicate, 100).unwrap(), 100)
            .is_err()
    );
    let wrong = SignedEnvelope::sign(
        &lab.carol,
        lab.bob.peer_id(),
        scope(),
        Purpose::PolicyVote,
        json!({"proposal_digest":"bb".repeat(32),"approve":true}),
        100,
        60,
    )
    .unwrap();
    assert!(
        quorum
            .add(verify(&node, &wrong, 100).unwrap(), 100)
            .is_err()
    );
    let carol = SignedEnvelope::sign(
        &lab.carol,
        lab.bob.peer_id(),
        scope(),
        Purpose::PolicyVote,
        payload,
        100,
        60,
    )
    .unwrap();
    quorum
        .add(verify(&node, &carol, 100).unwrap(), 100)
        .unwrap();
    let result = quorum.result(100);
    assert!(result.approved);
    assert_eq!(result.approvals, 2);
    assert_eq!(result.voters.len(), 2);
    assert!(!quorum.result(160).approved); // individual vote lifetime, before proposal expiry
    assert!(!quorum.result(200).approved);
    // Quorum status does not widen local authority.
    let mut actual = node.peers().unwrap();
    let mut expected = lab.config.authorized_peers.clone();
    actual.sort_by_key(|p| p.id().unwrap());
    expected.sort_by_key(|p| p.id().unwrap());
    assert_eq!(actual, expected);
}

#[test]
fn delegation_is_opaque_trusted_scope_namespaced_request_and_no_tool_fields() {
    let lab = Lab::new();
    let node = lab.node();
    let env = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"task_1","task":"Summarize this fixture."}),
        100,
    );
    let request = node
        .delegation(verify(&node, &env, 100).unwrap(), 100)
        .unwrap();
    assert_eq!(request.scope(), &scope());
    assert!(request.request_id().starts_with("mesh_"));
    assert_eq!(request.task(), "Summarize this fixture.");
    let again = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"task_1","task":"Summarize this fixture."}),
        100,
    );
    let again = node
        .delegation(verify(&node, &again, 100).unwrap(), 100)
        .unwrap();
    assert_eq!(request.request_id(), again.request_id());
    let bad = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"task_1","task":"hi","tool":"shell","arguments":{"cmd":"bad"}}),
        100,
    );
    assert!(
        node.delegation(verify(&node, &bad, 100).unwrap(), 100)
            .is_err()
    );
    let bad = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"x","task":"hi","scope":{"tenant":"other"}}),
        100,
    );
    assert!(
        node.delegation(verify(&node, &bad, 100).unwrap(), 100)
            .is_err()
    );
    let bad = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"x","task":"hi","budget":999}),
        100,
    );
    assert!(
        node.delegation(verify(&node, &bad, 100).unwrap(), 100)
            .is_err()
    );
}

#[tokio::test]
async fn ingress_is_loopback_signed_bounded_replay_safe_and_backpressured() {
    let lab = Lab::new();
    let node = lab.node();
    let now = unix_now().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let app = node.router(tx).unwrap();
    let env = lab.sign(Purpose::Gossip, json!({"fixture":true}), now);
    let bytes = env.to_json().unwrap();
    let request = |body: Vec<u8>, remote: &str| {
        let mut req = Request::post("/mesh/envelopes")
            .body(Body::from(body))
            .unwrap();
        req.extensions_mut()
            .insert(ConnectInfo(remote.parse::<SocketAddr>().unwrap()));
        req
    };
    assert_eq!(
        app.clone()
            .oneshot(request(bytes.clone(), "192.0.2.1:9000"))
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        app.clone()
            .oneshot(request(b"{}".to_vec(), "127.0.0.1:9000"))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        app.clone()
            .oneshot(request(bytes.clone(), "127.0.0.1:9000"))
            .await
            .unwrap()
            .status(),
        StatusCode::ACCEPTED
    );
    assert_eq!(
        app.clone()
            .oneshot(request(bytes.clone(), "127.0.0.1:9000"))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(rx.recv().await.unwrap().body().sender, lab.alice.peer_id());
    assert_eq!(
        app.clone()
            .oneshot(request(bytes, "127.0.0.1:9000"))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        app.oneshot(request(
            vec![b' '; MAX_ENVELOPE_BYTES + 1],
            "127.0.0.1:9000"
        ))
        .await
        .unwrap()
        .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

#[tokio::test]
async fn outbound_uses_configured_endpoint_and_denies_policy_without_network() {
    let mut lab = Lab::new();
    lab.config.authorized_peers[0].endpoint = Some("http://127.0.0.1:1/mesh/envelopes".into());
    let node = lab.node();
    let env = node
        .sign(
            lab.alice.peer_id(),
            scope(),
            Purpose::Gossip,
            json!({}),
            unix_now().unwrap(),
            60,
        )
        .unwrap();
    assert!(
        node.send(&env)
            .await
            .unwrap_err_text()
            .contains("origin denied")
    );
    let inbound = lab.sign(Purpose::Gossip, json!({}), unix_now().unwrap());
    assert!(node.send(&inbound).await.is_err());
    let mut peer = Lab::peer(&lab.alice);
    peer.endpoint = Some("https://example.com/arbitrary".into());
    assert!(node.peer_add(peer).is_err());
}

#[test]
fn corrupted_state_and_insecure_state_are_not_repaired() {
    let lab = Lab::new();
    let node = lab.node();
    drop(node);
    let path = lab.config.state_dir.join("mesh.json");
    fs::write(&path, b"{}").unwrap();
    assert!(lab.node_open_result().is_err());
    assert_eq!(fs::read(&path).unwrap(), b"{}");
    fs::remove_file(&path).unwrap();
    fs::set_permissions(&lab.config.state_dir, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(lab.node_open_result().is_err());
}

#[test]
fn private_parent_must_exist_and_identity_state_mismatch_fails() {
    let lab = Lab::new();
    let parent = lab._temp.path().join("private");
    mkdir(&parent);
    Identity::create(parent.join("identity")).unwrap();
    let node = lab.node();
    drop(node);
    let mut changed = lab.config.clone();
    changed.identity_path = lab._temp.path().join("alice.key");
    assert!(MeshNode::open(changed).is_err());
    assert!(Identity::create(lab._temp.path().join("missing/identity")).is_err());
}

struct MockProvider {
    calls: std::sync::atomic::AtomicUsize,
    request_effect: bool,
}
#[async_trait::async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mesh-fixture"
    }
    async fn complete(
        &self,
        _: &[missy_core::Message],
        _: &[missy_core::ToolDefinition],
        _: tokio_util::sync::CancellationToken,
    ) -> Result<missy_core::Completion> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let message = if self.request_effect {
            missy_core::Message {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![missy_core::ToolCall {
                    id: "effect-1".into(),
                    name: "mock_effect".into(),
                    arguments: json!({}),
                }],
                tool_call_id: None,
            }
        } else {
            missy_core::Message::text(
                "assistant",
                "Local fixture completed. This transcript is not in the mesh receipt.",
            )
        };
        Ok(missy_core::Completion {
            message,
            usage: missy_core::Usage {
                input_tokens: 10,
                output_tokens: 10,
                cost_microusd: 1,
                cost_known: true,
            },
            finish_reason: if self.request_effect {
                "tool_calls"
            } else {
                "stop"
            }
            .into(),
        })
    }
}
struct ApprovalTools {
    calls: std::sync::atomic::AtomicUsize,
}
#[async_trait::async_trait]
impl ToolExecutor for ApprovalTools {
    fn definitions(&self) -> Vec<missy_core::ToolDefinition> {
        vec![missy_core::ToolDefinition {
            name: "mock_effect".into(),
            description: "fixture approval gate".into(),
            parameters: json!({"type":"object","properties":{},"additionalProperties":false}),
            side_effect: true,
        }]
    }
    async fn preflight(&self, _: &missy_core::ToolContext, _: &missy_core::ToolCall) -> Result<()> {
        Err(missy_core::ApprovalRequired {
            request_id: "fixture_operator_approval".into(),
        }
        .into())
    }
    async fn execute(
        &self,
        _: &missy_core::ToolContext,
        _: &missy_core::ToolCall,
    ) -> Result<Value> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(json!({"mock":true}))
    }
}
fn budget() -> RuntimeConfig {
    RuntimeConfig {
        max_steps: 4,
        max_tokens: 1000,
        max_turn_seconds: 30,
        max_cost_microusd: 100,
        max_input_bytes: 16 * 1024,
        max_concurrent_turns: 2,
        ..RuntimeConfig::default()
    }
}
fn mock_runtime(
    lab: &Lab,
    effect: bool,
) -> (Arc<MeshRuntime>, Arc<MockProvider>, Arc<ApprovalTools>) {
    let store =
        Arc::new(missy_store::Store::open(lab._temp.path().join("runtime.sqlite")).unwrap());
    let provider = Arc::new(MockProvider {
        calls: std::sync::atomic::AtomicUsize::new(0),
        request_effect: effect,
    });
    let tools = Arc::new(ApprovalTools {
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    let rt = Runtime::new(
        store,
        provider.clone(),
        tools.clone(),
        "Local fixture policy".into(),
        budget(),
    )
    .unwrap();
    (
        Arc::new(MeshRuntime::try_from_runtime(Arc::new(rt)).unwrap()),
        provider,
        tools,
    )
}
#[tokio::test]
async fn end_to_end_signed_delegation_runtime_idempotency_and_private_receipts() {
    let lab = Lab::new();
    let node = lab.node();
    let (runtime, provider, tools) = mock_runtime(&lab, false);
    let now = unix_now().unwrap();
    let payload =
        json!({"task_id":"durable_fixture_task","task":"Summarize fixture, no external work"});
    let env = lab.sign(Purpose::DelegateRequest, payload.clone(), now);
    let id = receipt_id(&env).unwrap();
    let verified = verify(&node, &env, now).unwrap();
    assert_eq!(
        node.receipt(&scope(), &id).unwrap().unwrap().status,
        "admitted"
    );
    let first = node.dispatch(verified, &runtime).await.unwrap();
    assert_eq!(first.status, "completed");
    assert!(first.turn_id.is_some());
    let env = lab.sign(Purpose::DelegateRequest, payload, now);
    let second = node
        .dispatch(verify(&node, &env, now).unwrap(), &runtime)
        .await
        .unwrap();
    assert_eq!(first.turn_id, second.turn_id);
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(tools.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    let encoded = serde_json::to_string(&first).unwrap();
    assert!(!encoded.contains("transcript"));
    assert!(!encoded.contains("fixture completed"));
    let mut other = scope();
    other.user = "other".into();
    assert!(node.receipt(&other, &id).unwrap().is_none());
    drop(node);
    let node = lab.node();
    assert_eq!(
        node.receipt(&scope(), &id).unwrap().unwrap().turn_id,
        first.turn_id
    );
}
#[tokio::test]
async fn end_to_end_delegation_cannot_bypass_runtime_operator_approval() {
    let lab = Lab::new();
    let node = lab.node();
    let (runtime, _, tools) = mock_runtime(&lab, true);
    let now = unix_now().unwrap();
    let env = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"approval_fixture","task":"Request mock effect"}),
        now,
    );
    let receipt = node
        .dispatch(verify(&node, &env, now).unwrap(), &runtime)
        .await
        .unwrap();
    assert_ne!(receipt.status, "completed");
    assert_eq!(tools.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(receipt.turn_id.is_some());
}

#[tokio::test]
async fn in_process_signed_http_to_runtime_worker_preserves_scope_and_receipt() {
    let lab = Lab::new();
    let node = lab.node();
    let (runtime, provider, tools) = mock_runtime(&lab, false);
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);
    let app = node.router(tx).unwrap();
    let now = unix_now().unwrap();
    let env = lab.sign(
        Purpose::DelegateRequest,
        json!({"task_id":"http_fixture","task":"Local fixture"}),
        now,
    );
    let id = receipt_id(&env).unwrap();
    let mut request = Request::post("/mesh/envelopes")
        .body(Body::from(env.to_json().unwrap()))
        .unwrap();
    request
        .extensions_mut()
        .insert(ConnectInfo("127.0.0.1:9876".parse::<SocketAddr>().unwrap()));
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    let verified = rx.recv().await.unwrap();
    assert_eq!(verified.body().scope, scope());
    let receipt = node.dispatch(verified, &runtime).await.unwrap();
    assert_eq!(receipt.status, "completed");
    assert_eq!(
        node.receipt(&scope(), &id).unwrap().unwrap().turn_id,
        receipt.turn_id
    );
    assert_eq!(provider.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(tools.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    let body = http_body_util::BodyExt::collect(response.into_body())
        .await
        .unwrap()
        .to_bytes();
    assert!(body.is_empty());
}

#[test]
fn verified_tokens_are_rechecked_after_local_grant_revocation() {
    let lab = Lab::new();
    let node = lab.node();
    let env = lab.sign(
        Purpose::MemoryUpdate,
        serde_json::to_value(update(json!(1), 0)).unwrap(),
        100,
    );
    let token = verify(&node, &env, 100).unwrap();
    node.peer_remove(&lab.alice.peer_id()).unwrap();
    assert!(node.apply_memory(token, 100).is_err());
    assert!(node.memory_get(&scope(), "shared").unwrap().is_none());
}

#[test]
fn replay_capacity_never_evicts_a_live_nonce_to_admit_another() {
    let lab = Lab::new();
    let node = lab.node();
    let env = lab.sign(Purpose::Gossip, json!({}), 100);
    verify(&node, &env, 100).unwrap();
    node.mutate(|state| {
        for i in 1..MAX_REPLAY_ENTRIES {
            state
                .replay
                .insert(format!("{}:{i:032x}", lab.carol.peer_id()), 160);
        }
        Ok(())
    })
    .unwrap();
    let next = lab.sign(Purpose::Gossip, json!({}), 100);
    assert!(
        verify(&node, &next, 100)
            .unwrap_err_text()
            .contains("capacity")
    );
    assert!(
        verify(&node, &env, 100)
            .unwrap_err_text()
            .contains("replay")
    );
}
#[test]
fn explicit_mesh_budget_rejects_unlimited_or_excessive_runtime() {
    validate_budget(&budget()).unwrap();
    let mut cfg = budget();
    cfg.max_cost_microusd = 0;
    assert!(validate_budget(&cfg).is_err());
    let mut cfg = budget();
    cfg.max_steps = 33;
    assert!(validate_budget(&cfg).is_err());
    let mut cfg = budget();
    cfg.max_tokens = 100001;
    assert!(validate_budget(&cfg).is_err());
    let mut cfg = budget();
    cfg.max_turn_seconds = 301;
    assert!(validate_budget(&cfg).is_err());
    let mut cfg = budget();
    cfg.max_input_bytes = 65536;
    assert!(validate_budget(&cfg).is_err());
}
#[test]
fn private_snapshot_reopens_after_wire_node_count_is_exceeded() {
    let lab = Lab::new();
    let node = lab.node();
    for _ in 0..160 {
        verify(&node, &lab.sign(Purpose::Gossip, json!({}), 100), 100).unwrap();
    }
    assert_eq!(node.status().unwrap().replay_entries, 160);
    drop(node);
    let node = lab.node();
    assert_eq!(node.status().unwrap().replay_entries, 160);
}
