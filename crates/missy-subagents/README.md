# Explicit bounded subagents

`AgentPool::new(BTreeMap<String, Arc<Runtime>>, Limits)` admits an operator-authored
`DagSpec`, not model-produced spawn instructions. Each `AgentSpec` selects an
already configured provider ID and declares `id`, `parent_id`, `depth`, `prompt`
and optional `depends_on`. Top-level child depth is 1. Parent completion is an
implicit dependency; other dependencies express complementary parallel work.
Cycles, unknown parents/providers, duplicate IDs, invalid depths, invalid prompts
and unknown fields are rejected before root admission. Hard ceilings are depth 4,
16 children, 4 workers and 600 seconds. Runtime capacity remains binding.

`pool.run(&registry, trusted_scope, root_request_id, spec, cancellation)` returns a
serializable `RootReceipt` with original durable child request/turn identities,
per-child outcomes, usage and aggregate reservations. Scope must come from a
trusted operator/authenticated adapter, never a model. Children inherit exact
tenant/user/channel and receive independently derived session suffixes. No child
can choose a scope or add tools, providers, policies or capabilities. Parent and
sibling history is not copied. Explicit dependency results are JSON-framed as
untrusted data; no output is parsed as configuration or orchestration instructions.
Principal-owned personal memory, if explicitly enabled in the supplied runtime,
retains its existing policy; the pool does not enable cross-session history.

## Budget and interruption semantics

Every runtime must have a positive finite cost bound. CLI builders must also use
known configured prices. Full child runtime step/token/cost ceilings are
cumulatively reserved in private SQLite **before** admission. Reservations are
never refunded, including failure, cancellation and unknown outcomes. Consequently
parallel siblings cannot each spend the entire root allowance, and actual known
usage also blocks further admissions if larger than reservations. These are
admission/runtime limits, not a promise that an external provider cannot bill for
an already dispatched completion exceeding its declared price/token bounds.

The immutable root hash binds the complete DAG, all pool runtime configuration
fingerprints and root limits. A changed prompt/provider/config under the same
scope/request ID is rejected. A completed root replays its persisted receipt and
original child identities without executing again. Recovery rehydrates completed
runtime turns whose receipt write was interrupted by original identity. Uncertain
active children are cancelled and marked interrupted; their dependents and all
still-planned work are blocked. There is no automatic replay or resume across an
uncertain tool/provider effect.

Cancellation, deadline expiry and dropping the root future call `Runtime::cancel`
on actual admitted turn IDs. A native per-root nonblocking file lock prevents
another coordinator from taking over an active root. Receipts use SQLite WAL/FULL
durability. Native files use NOFOLLOW, reject symlink ancestors, hardlinks,
exposed files, foreign applications and nonprivate parent directories. Registry
parent must be current-user owned and mode 0700; files 0600 or stricter. This
protects against other principals, not a malicious owner rewriting the database.

Verification: `cargo test -p missy-subagents --offline`. Fixtures use real
`Runtime`/`Store`, atomic provider concurrency tracking, cancellation and durable
restart behavior. No live providers, agents or services are used by these tests.
