# Native security and bounded tools production notes

## Implemented API and trust boundary

`missy-security` exports `preprocess_input(&str) -> SecurityView`,
`inspect_untrusted(&str) -> SecurityView`, `redact_output(&str) -> String`,
`redact_with_literals(&str, &[&str]) -> String`, and
`PromptDriftDetector::{default, register, verify}`.

`SecurityView` contains `text`, `findings` (rule IDs and classes, never matched credentials),
`quarantined`, `truncated`, and SHA-256 of the **original** UTF-8 bytes. Functions do not change
their input, persist files, read environment variables, contact accounts, authorize tools, or grant
capabilities. Persist raw request/tool evidence first under authenticated scope, then derive a
separate model/output projection. Never rewrite historical evidence with its redacted projection.
Authorized raw access must remain distinct from model context and ordinary delivery.

Runtime integration belongs to `missy-runtime::context`, not the tool executor. Initial input,
recalled history/memory, tool results, and continuation/replay must cross projection before every
provider call. JSON string values **and object keys** are untrusted data. Scanning only serialized
JSON hides line boundaries. Provider output must cross redaction before delivery, including cached
or resumed output. Security projections must not rewrite the exact arguments of approved calls or
manufacture successful tool receipts. Prompt-integrity registration/checking is runtime-owned.

## Legacy comparison and deliberate precision

Inspected `missy/security/{sanitizer,secrets,censor,drift}.py` and
`missy/tools/builtin/{web_fetch,file_read,calculator,__init__}.py`.

* Legacy `censor_response` is credential-only, not profanity/topic censorship. Native preserves
  that meaning. Recognizable credential formats, explicit assignments, bearer/basic headers,
  presigned signatures, credential URLs and complete PEM private-key material are redacted.
  Overlapping spans are merged. Public SSH keys and prose about password complexity are benign.
* Legacy sanitizer includes broad signals such as any HTML comment, isolated "jailbreak", and
  remembering something for later. Native intentionally does not copy those false positives.
  Translation/roleplay, OS labels, license comments, install instructions and discussing injection
  detection remain usable. Explicit instruction replacement, role spoofing, prompt exfiltration,
  concealed payloads and bounded encoded variants produce advisory quarantine findings.
* Fullwidth ASCII and zero-width controls normalize only the derived view. Raw bytes/digest do not
  change. Scans cap at4MiB, views at64KiB. Oversized scans withhold fail-closed; view truncation is
  marked and UTF-8 safe. One-level base64/hex inspection caps at2048 tokens of8192 bytes each.
* `PromptDriftDetector` checks exact trusted-template integrity, not semantic model drift. Unknown
  IDs fail closed, unlike legacy. Utility tests alone are not proof of runtime enforcement.

Regex heuristics are **not** an injection firewall. Arbitrary nested encodings, every language,
ROT13, all Unicode confusables, unknown credential formats and novel semantic attacks remain
possible. Exact-origin DNS-pinned HTTP, descriptor-relative file roots, exact argv approval,
authenticated scopes and durable approvals/receipts remain the actual boundary. Approved
executables/MCP have service-user authority, not an OS sandbox. Same-UID attackers changing mounts,
moving opened directories, replacing trusted executables or retaining writable descriptors are
outside this userspace boundary. Run under an unprivileged dedicated account and OS isolation.

## New native tool composition

| Tool | Inputs | Boundary |
| --- | --- | --- |
| `web_search` | `query`, optional `limit`1..10 | Existing PolicyHttpClient, explicit `https://html.duckduckgo.com` origin required. HTTP GET, effectful approval/receipt path. No account, redirects or automatic result-page fetches. |
| `read_document` | Exactly one `path`/`url`; `format`text/html/json; `max_chars`1..64000 | Existing file-root policy or HTTP gateway. Conservatively effectful even for local files so URL use cannot bypass approval. |

`calculator` already belongs to SpecialtyTools; no duplicate builtin registration was retained.
Search parses bounded DuckDuckGo HTML and unwraps result redirect links without fetching. Non-HTTP
and authenticated result URLs are rejected. Challenge/unrecognized markup returns an explicit
failure status, not an authoritative empty result set. Result URLs do not grant fetch permissions.
Document input caps at1MiB, UTF-8 only. HTML uses scraper/html5ever, never executes JavaScript or
loads external assets, omits script/style/template/noscript/SVG/hidden nodes and decodes entities.
CSS rendering is not implemented. PDF/ZIP/binary formats fail explicitly, not fabricated OCR.

Source HTML is scanned before DOM extraction, including hidden comments. Search/document return
an evidence envelope inside the normal untrusted wrapper:

```text
{trust:"untrusted", data:{raw_evidence:{source_utf8:original_source}, model_view:{...}}}
```

The raw source retains exact UTF-8 bytes including quarantined payloads. The complete envelope is
bounded to4MiB serialized; model_view obeys runtime.max_tool_output_bytes independently. Runtime
must persist the envelope **before clipping**, then recursively expose only model_view, never
raw_evidence. Existing web_fetch/file_read raw receipts likewise require runtime projection before
model use. Runtime owners were notified of this contract. Digest-only evidence was rejected.

Integration is verified against the actual Runtime and Store in
`crates/missy-tools/tests/runtime_security.rs`: a real BuiltinTools document call passes catalog
schema, retains exact original HTML/credential bytes in scope-protected tool_evidence, delivers
only the redacted extracted view, and reopens/deduplicates without changing evidence. Wrong
types, unknown fields and out-of-range search limits fail the actual runtime schema path. These
acceptance tests contact neither provider nor network.

Runtime/core owners60605f37 and651123c9 received API signatures, the envelope contract and new
effectful tool names before integration. No new core config fields were needed. Existing origins,
roots, private-network and runtime limits apply. Runtime context recursively projects input/history/
tool JSON and envelopes; final/cached/error output crosses redaction. Trusted system bytes live
in immutable Runtime configuration and are checkpoint-hashed on save and checked on load
(`system_prompt_hash`), so standalone PromptDriftDetector need not duplicate that mechanism.

## Actual policy/vault fixes

1. Private `.missy-delete-*` recovery journals were protected only from deletion, permitting broad
   read/write roots to read or corrupt uncertain-delete evidence. Every file access now rejects
   delete/write journal components; listings hide their names. Operator recovery remains possible.
2. Vault creation now requests0700 at mkdir, removing create-then-chmod's permissive interval.
3. First-time sign_export calls serialize identity-key creation under the vault mutation lock.

Existing tests cover symlink/hardlink/FIFO denial, directory substitution races, digest-conditional
delete/inode substitution/recovery, exact argv/clear environment, process-group cancellation,
redirect denial, all-address DNS validation/pinning, private/mapped/mixed address denial, bounded
headers/bodies and streaming total/idle deadlines. Atomic file write is replacement, not content
CAS. Deletion under hostile same-UID concurrency can preserve an uncertain recovery journal.

## Verification and residuals

Disposable temp directories and loopback mocks only:

```text
CARGO_BUILD_JOBS=2 cargo test -p missy-security -p missy-policy -p missy-tools -- --test-threads=2
CARGO_BUILD_JOBS=2 cargo clippy -p missy-security -p missy-tools -p missy-policy --all-targets -- -D warnings
cargo fmt --check -p missy-security -p missy-tools -p missy-policy
```

Verified checkpoint: **76 tests passed** (policy39, security14, tools23 including integration
suites); focused clippy all-targets with `-D warnings`, rustfmt check and scoped git diff check
passed. The final execution report is authoritative after further concurrent changes. Fixtures exercise
benign text, encoded/fullwidth/zero-width injection, complete credential redaction, UTF-8/scan
bounds, drift, concurrent signing, journal protection, schema/scope/format rejection, preserved
raw source, local HTTP, denied search before connect and nonexecuting HTML parsing.

No production replacement, accounts, host config, devices, live secrets or external live search
were accessed. Upstream search markup compatibility remains fixture-only. Documents exceeding
parser/receipt caps fail rather than silently losing source evidence. PDF/OCR/binary support is
not implemented. Raw evidence is sensitive: its retention and authorized raw retrieval must never
be confused with the redacted model/delivery interface. Heuristic tests do not prove universal
injection prevention or absence of secret leakage.
