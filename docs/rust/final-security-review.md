# Focused final security review

Reviewed 2026-11-05. Scope: new personal memory, evolution, operator DAG subagents,
capture, specialty tools, and runtime observation. Native DAVE cryptography and
parent-owned CLI integration were explicitly outside this review. This was a
time-bounded adversarial review, not a claim of exhaustive security verification.

## Demonstrated defect: cross-principal FTS ranking statistics

**Status: fixed with parent authorization.** The original personal-memory search
filtered returned rows by authenticated tenant and user but ordered them with
`bm25(personal_fts)`. SQLite FTS5 computes document frequencies over the entire
virtual table, not the namespace-filtered result set. Thus foreign private
contents influenced observable ordering and the top result even though foreign
rows themselves were correctly filtered.

Exact reproduction against the public Rust API:

1. Alice stores ID `a` with `alpha alpha`, ID `b` with `beta beta`.
2. Searching `alpha beta` returns `[a,b]`.
3. Bob stores 100 private records containing `alpha alpha`.
4. Alice's same search returns `[b,a]` without any Alice mutation.

The added test
`foreign_private_term_frequencies_cannot_change_personal_ranking` failed against
the original implementation with precisely that reversal. This is an observable
private-term-statistics side channel, not demonstrated disclosure of entire
foreign records.

The fix builds a transient in-memory FTS5 index containing only the authenticated
principal's validated records and computes BM25 there. This preserves Unicode
FTS matching and useful relevance ranking without foreign document frequencies.
The input corpus remains bounded by the existing 512-record / 2 MiB limits;
excess or corrupt principal records fail closed. The fingerprint now explicitly
binds the namespace-local ranking policy. Existing persistent schema and
namespace authority are unchanged. The regression checks full ordering and
limit-one results across foreign inserts, updates and deletes.

Authorized source changes were restricted to
`crates/missy-personal/src/lib.rs` and
`crates/missy-personal/tests/personal.rs`.

## Other observations

- Personal get/update/delete use tenant+user predicates, source-principal checks,
  and integrity validation. No global or other-user fallback was found. Existing
  corruption, Unicode, bounded-output, permission, sidecar and link tests passed.
- Retrieval's separate hybrid search computes lexical document frequencies from
  `load_chunks`' exact-scope corpus, not a global FTS index. The demonstrated
  personal defect was not found in that implementation.
- Evolution proposal loading checks exact scope, ID, plan checksum and diff
  checksum. ID-only updates seen later in the workflow operate after scoped
  loading and use globally unique primary-key IDs, within transactions where
  applicable. I did not demonstrate an unknown-ID cross-scope mutation.
- Subagent roots durably reserve full child bounds before dispatch, retain
  reservations after failures, derive child sessions while preserving parent
  tenant/user/channel, reject immutable-DAG/configuration changes, and avoid
  automatic recovery replay. Actual mock-provider tests for parent-future drop,
  explicit cancellation, aggregate limits and restart behavior passed.
- The currently observed Foundry implementation resets cached listed repositories
  and reviewed plans when the full caller scope or credential digest changes.
  Therefore the initially suspected shared-cache cross-user authorization reuse
  is not a demonstrated defect in this tree. Config is held immutably by the
  executor; current policy/config participate in its fingerprint. A single cache
  can invalidate another scope's staged plan, but does not preserve its authority
  for the other scope. Concurrent specialty edits were owned by another agent.
- Capture revalidates executable digests on actual process execution and live
  device/Xauthority properties during dispatch. Xauthority bytes are read fresh;
  their rotation is not itself a configured executable-authority change. Process
  execution is through a verified descriptor, isolated from ambient desktop
  configuration. Tests used synthetic media and disposable authenticated Xvfb,
  not an interactive desktop or a real camera.
- Runtime observation events contain fresh trace/span IDs, timestamps, event kind
  and success, not scopes, prompts, credentials or tool payloads. Observer panic
  containment and actual tool/settlement event behavior passed targeted tests.

## Validation evidence

Before the fix, the focused five-package test invocation completed successfully:

`cargo +1.99.1 test --locked -p missy-personal -p missy-subagents -p missy-evolution -p missy-specialty -p missy-capture -- --test-threads=2`

It ran capture 6, evolution 10, personal 11, specialty 6 and subagents 8 tests.
Those tests did not cover the demonstrated ranking defect.

After adding the regression, the exact filtered personal regression failed on
the original source as expected. After the fix:

- `cargo +1.99.1 test --locked -p missy-personal`: all 12 tests passed.
- `cargo +1.99.1 clippy --locked -p missy-personal --all-targets -- -D warnings`:
  passed.
- `cargo +1.99.1 test --locked -p missy-runtime observability`: 2 passed.

No external-network fixture was used. No live service, camera, active desktop,
deployment, commit or DAVE source was modified. Other agents were editing the
shared checkout concurrently; these are results for the code observed and tested
during this review, not an immutable release attestation.
