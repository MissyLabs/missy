# Native governance

This crate is an additional authorization gate, not a policy override or OS
sandbox. Tests use temporary state. It never reads or edits live Python state,
active persona files, running policy, source files or desktop sessions.

## Exact operator APIs

`ApprovalRegistry::open(path)` opens a durable SQLite registry. Parent directory
must exist, be service-owned and not writable by untrusted users. Unix database
files require private permissions, service ownership and no symlink/hardlink.
Hostile same-UID processes and root are outside this local trust boundary.

- `list(&Scope, limit)` and `get(&Scope, id)` return scoped request records.
- `decide(&Scope,id,expected_binding_hash,Decision::Approve|Deny,operator_id)`
  decides pending requests once. `revoke(&Scope,id,expected_binding_hash,operator_id)`
  revokes pending/approved requests. Terminal decisions cannot be reissued.
- `request(&ApprovalBinding,ttl_seconds)` is idempotent for the entire exact
  binding; TTL is 1..86400 seconds and retries do not extend it.
- `consume(&ApprovalBinding)` atomically checks scope/binding/status/expiration,
  clears the opaque decision token, sets consumed and appends an audit event.
  It commits before dispatch. Failures never undo consumption or regrant it.
- `audit(&Scope,after_sequence,limit)` is scoped and stores only fixed events,
  object IDs, binding hashes and hashed operator identity, never raw arguments,
  document content, arbitrary reason text or operator identity strings.

Bindings cover every Scope field, turn ID, call ID, tool name, canonical arguments
hash and context fingerprint. JSON object key order is irrelevant; array order
is significant. Raw arguments are not persisted. No wildcard approvals or trust
inference exists. Approve/deny/revoke methods must be called only by an
authenticated operator CLI/API, never exposed as model-callable tools.
`operator_id` is an audit label, not authentication. Database decisions are not
portable cryptographically signed capabilities; opaque tokens are not credentials
and must not be forwarded as a way to override a policy-aware executor.

## Executor/runtime wiring

`GuardedTools::new(Arc<dyn ToolExecutor>,registry,GuardConfig)` wraps an unchanged
policy-aware executor. `GuardConfig` has `sensitive_tool_names:BTreeSet<String>`,
`context_fingerprint:String` (lowercase SHA-256), `approval_ttl_seconds:u64`.
The trusted embedding runtime must hash relevant config, policy, provider/tool
settings and prompt snapshot into this context. Guard adds its full configuration
and inner executor fingerprint. Inner executors must override their fingerprint
when effective policy is broader than definitions. Changed inner configuration
fails closed and requires rebuilding the wrapper from trusted current config.

`preflight` invokes inner preflight, then creates/inspects the durable approval.
Pending returns `missy_core::ApprovalRequired{request_id}`, also exported as
`ApprovalRequiredError`. Repeated pending calls preserve request ID. Approved
preflight never consumes. Terminal states return `ApprovalRefusedError`.
Runtime must preflight the entire batch before journaling/dispatch, durably save
exact deferred calls, pause pending approvals, and resume those calls, not newly
generated calls. Preflight refusal is definite nonexecution, not UNKNOWN.

`execute` rechecks inner preflight and consumes before inner execute. All errors
in this predispatch preamble are sanitized `missy_core::ToolDispatchRefused`,
allowing runtime to record Failed rather than UNKNOWN. Errors from inner execute
are never wrapped, as they may represent uncertain effects. Inner policy checks
remain mandatory; approval never enables a disabled capability. Revocation races
are linearized at consumption: revocation first blocks, consumption first
authorizes dispatch and cannot be retroactively cancelled. `annotation(name)`
and sensitive tool descriptions expose the additional gate without claiming an
approval overrides policy.

## Persona, prompt, patch and trust

SQLite is the atomic authoritative document store. There is no false claim that
unrelated files and SQLite commit atomically.

- `propose_document(scope,DocumentKind::{Persona,SystemPrompt,Patch},base_version,
  content,context_fingerprint)` stores bounded inert UTF-8 text.
- `document_proposal(scope,id)` and `document_proposals(scope,limit)` expose text
  for authorized operator review, not logging.
- `decide_document(scope,id,expected_proposal_hash,current_context_fingerprint,
  Decision,operator_id)` atomically records explicit approval and a new revision.
  Changed context/hash, stale base version, wrong scope or repeat decisions fail.
- `document(scope,kind)` reads a revision; `prompt_snapshot(scope)` reads an
  immutable combined persona/system prompt snapshot and `prompt_hash`.

Runtime must load prompt snapshots only at a new-turn boundary, bind the hash
into its context fingerprint, reject changed snapshots on resume, and never
hot-replace an active turn's prompt after an approval. Approved patches are
reviewed text, not code execution. Applying them needs a separate requested
operation routed through the appropriate policy-aware executor and approval.

`trust(scope)` is advisory normalized [-1,1] plus version.
`update_trust(scope,expected_version,delta,OperationOutcome,operator_id)` uses
finite bounded deltas and optimistic concurrency. Failed/unknown outcomes cannot
increase trust. Scores never grant approval or modify policy.

Landlock is deliberately not implemented. No process is kernel-sandboxed by
these APIs. Run `cargo test -p missy-governance` for disposable tests covering
cross-instance single use, decision/revoke races, expiry, binding/scope changes,
transaction rollback, preflight, retained inner policy, bounded trust and atomic
document revisions without active snapshot mutation.
