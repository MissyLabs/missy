# Native mesh

Native Ed25519 identities, signed recipient-bound scoped envelopes, explicit
capability grants, durable replay admission, deterministic LWW memory and
authenticated quorum. No Python bridge, TOFU, wildcard scopes or implicit key
generation. Implemented and tested with local disposable fixtures only, **not
accepted or deployed live network federation**.

## Operator configuration and persistence

`MeshConfig::load(path)` reads a separate private JSON file, not the main Missy
config or legacy Python state. Required fields:

* `version`: `1`.
* `identity_path`: absolute path to an existing Ed25519 seed file.
* `state_dir`: absolute, explicit private mesh state directory. The final
  directory can be created, but its parent must exist.
* `authorized_peers`: array of `Peer { public_key, endpoint, grants }`.
  `public_key` is lowercase hex of the 32-byte Ed25519 public key. The stable
  peer ID is SHA-256 of those raw public-key bytes. An endpoint is `null` or an
  operator-set HTTPS URL ending exactly `/mesh/envelopes`, without credentials,
  query or fragment. Numeric loopback HTTP is allowed for fixtures.
* `grants`: each peer has an array of `{ scope, capabilities }`, with the shared
  `Scope { tenant, user, channel, session }` matched exactly. Capabilities are
  `memory_read`, `memory_write`, `delegate`, `gossip`, `policy_vote`. Empty grants
  mean no authority.
* `listen`: loopback socket address, for example `127.0.0.1:8790`.
* `http_policy`: native `PolicyConfig`; exact origin allowlist, DNS and private
  network restrictions apply independently to the fixed peer endpoint.

All private directories must be owned by the effective uid and deny group/other
access (normally 0700); files normally 0600. Descriptor-relative traversal
rejects symlink parents, symlink files, non-regular entries and hard-linked
private files. No permission repair or recursive chmod occurs. Key creation is
**explicit** `Identity::create(path)`, with exclusive create, a preexisting
private parent and fsync. `Identity::load` never regenerates missing keys.

`MeshNode::open(config)` loads existing identity and mesh state. It exclusively
locks state for its lifetime and binds the stored identity ID to the key. First
open seeds the peer registry from private config. Subsequent peer mutations are
persisted. `peer_add` can only use keys, endpoints and exact-scope grants already
authorized in private config. Config remains an upper bound even over old
persisted grants, so removing authority from config takes effect on reopen.
There is no HTTP peer-add/grant endpoint, and quorum never mutates grants.

State uses atomic replace and directory fsync, capped at 4MiB. Live replay
entries are never evicted for capacity. Limits: 256 peers, 128 grants per peer,
8192 replay entries, 8192 receipts and 4096 memory rows including tombstones.
Exhaustion fails closed. Receipt/tombstone archival is deliberately **not** an
automatic deletion feature. Back up identity and state together; rolling state
back also rolls back replay history. There is no anti-rollback hardware anchor.

## Signing and admission API

`MeshNode::sign(recipient, scope, purpose, payload, unix_seconds, ttl_seconds)`
requires an explicitly allowed peer and exact local scope/purpose capability.
`SignedEnvelope::sign(&Identity, ...)` is the lower-level signing primitive.
`Purpose` serializes as `memory_read`, `memory_update`, `delegate_request`,
`gossip`, `policy_vote` and maps to one fixed capability, never a caller-chosen
capability name. `SignedEnvelope::{to_json,from_json}` provide bounded codecs.

Envelope fields are `body { version, sender, recipient, scope, purpose,
timestamp, nonce, expiry, payload }` plus `signature`. Signature covers all body
fields under domain `missy-native-mesh-envelope-v1` plus a NUL separator. Mesh
canonical JSON is sorted-key, compact UTF-8, integer-only data; **not a claim of
RFC8785 compatibility or legacy wire compatibility**. Duplicate keys, floats,
unknown envelope fields, >64KiB wire data, >16-level canonical depth and >4096
JSON nodes are rejected. Nonces are cryptographically random 128-bit lowercase
hex. TTL is 1..300 seconds; forward timestamp skew is capped at 30 seconds.

`verify_json(bytes, now)` performs strict Ed25519 verification, configured key
lookup, recipient/reflection checks, exact scope/capability check, expiry and
durable sender+nonce replay admission. This is **not a read-only diagnostic**:
it consumes the nonce and fsyncs an admitted receipt before returning an opaque
`VerifiedEnvelope`. Nonce reuse is denied across purpose and scope; restart does
not reset admission. Backward wall-clock movement below the persisted high-water
mark fails closed. A failed signature never consumes a nonce. A consumed valid
nonce is not restored if later application dispatch rejects the payload.

`receipt_id(&SignedEnvelope)` returns a canonical signed-body digest.
`receipt(&Scope, id)` is local-only and exact-scope. Receipt contains sender,
scope, purpose, status and optional runtime turn ID, **not transcript/tool
output**. `admitted` after a crash is not proof that execution happened, and is
not permission to redispatch. `not_completed` also requires journal
reconciliation rather than assuming effects did not occur.

## Runtime and HTTP integration

`router(mpsc::Sender<VerifiedEnvelope>)` builds an Axum router with only
`POST /mesh/envelopes`, a 64KiB request limit, signature authentication and a
bounded inbox. It requires Axum `ConnectInfo<SocketAddr>` and checks the source
is loopback. `serve(inbox)` binds only configured loopback and supplies this
connection metadata. `serve_with_runtime(Arc<MeshRuntime>)` additionally runs a
sequential bounded worker: memory updates are merged; delegation is sent through
the normal Runtime; other purposes are recorded not completed and require a
separate explicit local consumer. No service launches during construction or
tests. HTTP returns only 202 admission, 401 rejection, 403 non-loopback,
413 oversize or 503 capacity. A 202 is **not task completion**. No read-memory,
status, transcript, arbitrary URL, approval or direct-tool HTTP endpoint exists.

`send(&SignedEnvelope)` authenticates the local signature and sends only to the
operator-configured recipient endpoint through native `PolicyHttpClient`, with
15-second timeout and normal origin/DNS/TLS/redirect/proxy defenses. An inbound
payload can never select an outbound URL. There is no auto-discovery or unsigned
broadcast. Inbound loopback is a deliberate limitation, not an Internet-facing
federation deployment recommendation.

Delegation DTO is exactly `DelegatedTask { task_id, task }`. Remote scope,
tools, arguments, budget or capability overrides are unknown fields and denied.
`delegation(VerifiedEnvelope, now)` returns an opaque `DelegationRequest` using
the locally granted exact scope. The stable Runtime request ID hashes sender,
recipient, scope and task ID into `mesh_<sha256>`. New envelopes for the same
task deduplicate in Runtime; changing its prompt does not create a fresh ID.
`run_delegation(request, &MeshRuntime)` rechecks grants and expiry and calls
ordinary Runtime, never ToolExecutor directly.

`MeshRuntime::try_from_runtime(Arc<Runtime>)` validates the actual runtime's
`configuration()`; `MeshRuntime::new(store, provider, tools, system, budget)` can
build a dedicated runtime. Both require explicit local bounded configuration:
steps <=32, tokens <=100000, turn <=300 seconds, input <=16384 bytes,
concurrency <=8 and cost budget in 1..1000000 microUSD. Zero/unlimited cost is
rejected. The ordinary Runtime remains responsible for tool approval gates,
leases, side-effect journal and durable task admission. Granting delegation is
still granting the peer ability to submit prompts to those locally selected
tools and provider; it is not a sandbox or a substitute for safe tool policy.

## CRDT and quorum

`MemoryUpdate { key, value, deleted, timestamp, counter }` is accepted only in a
verified `memory_update` envelope; timestamp must equal the authenticated
envelope timestamp. Values <=16KiB, counter <=1000000, keys 1..256 bytes.
Scope is part of the map key. Total winner order is `(timestamp, peer_id,
counter, deleted, canonical_value_bytes)`: peer ID breaks cross-peer timestamp
ties; counter orders same-identity same-second writes; tombstone and canonical
value bytes make same-ID/same-stamp conflicts arrival-independent. Tombstones
are retained. Network-provided snapshots cannot assert another writer's stamp.
`apply_memory` rechecks current authority and persists before returning.

`Quorum::new(recipient, scope, proposal_digest, eligible_peer_ids, threshold,
expiry)` is local policy. `add(VerifiedEnvelope, now)` accepts only unique
authenticated exact-scope `policy_vote` messages for that recipient and exact
proposal digest, with a typed bool `approve`. Threshold zero or above eligible
count is invalid. Quorum expiry and each vote expiry gate approval. A vote from
an arbitrary string identity, duplicate voter, ungranted peer or different
proposal cannot count. Quorum results are advisory and grant no new authority.

## Validation

Run `cargo test -p missy-mesh --lib` and
`cargo clippy -p missy-mesh --all-targets -- -D warnings`. Tests use disposable
temp dirs, generated real Ed25519 keys, Axum in-process fixtures and mocked
provider/tools. They cover signature tampering, replay across restart/scope,
TTL, wrong recipient/reflection/cross-peer, explicit grants/config revocation,
JSON bounds/duplicates, CRDT permutations/tombstones/persistence and quorum
unique-voter thresholds. Runtime fixtures exercise real durable Runtime
admission/dedup and prove an approval-required tool never executes. No outbound
test endpoint is contacted: the transport test is denied by policy before any
network connection. No live keys, peers, tools or service are modified.
