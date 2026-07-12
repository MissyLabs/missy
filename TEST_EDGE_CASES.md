# TEST_EDGE_CASES

## Run: 2026-07-10 08:40 UTC — validation-harness overhaul session

Edge cases added or newly covered this session (see `TEST_RESULTS.md`
for full per-checkpoint test counts and `BUILD_STATUS.md`/
`AUDIT_SECURITY.md` for the underlying findings):

- acpx delegate: operator `base_url` config attempting to reintroduce
  native tools, permissive permissions, or redirect the sandboxed cwd
  back into a real repository (all stripped, verified against real
  argv construction).
- acpx delegate: response entirely a fabricated transcript continuation
  after leaked-marker stripping (must fail closed, not return empty
  content silently).
- acpx delegate: quoted `[Assistant]:`-style text in a legitimate
  *current* user request must not be mistaken for delegate-generated
  leakage (the strip only ever runs on delegate output, never input).
- acpx delegate: malicious/injected instructions embedded in prior
  conversation history must stay structurally confined before the
  current-turn boundary marker.
- acpx delegate: configured timeout far exceeding the safe upper bound
  (999,999s) must clamp to 600s with a warning logged.
- Discord: a message triggering a voice/image/screencast command AND
  failing guild/DM authorization must reach none of the three special
  dispatchers (combined denial test, not just per-dispatcher).
- Discord: `require_mention: true` guild policy must not block slash
  commands (they have no `@mention` text to check).
- Discord: two different users issuing `/ask` in sequence must receive
  two different session IDs, not share one.
- Memory tools: a store lookup that raises an exception vs. one that
  genuinely finds nothing must produce textually distinguishable error
  messages (`internal error`/`unverified` vs. `not found`).
- Incus tools: a minimal real-shaped JSON payload with only 2 fields
  must not gain synthesized fields; an empty list must stay empty, not
  get padded; a network list without a `lo` entry must not have one
  fabricated when re-rendered.
- Browser tools: the exact two error strings the validation harness
  observed (`unshare(CLONE_NEWPID): EPERM`,
  `Protocol error (Browser.enable)`) must classify as a sandbox/kernel
  launch failure with remediation text that never suggests
  `--no-sandbox`/`SYS_ADMIN`/privileged containers.
- Discord pairing: `!pair accept <id>` sent by *any* DM sender
  (including the pending requester's own ID) must never approve a
  pairing — the pending state must be left untouched.

Older focus (prior tool-intelligence overhaul session, preserved for
history):
- frequent-request detection without noisy false positives
- generated tool candidates remain disabled until approved
- generated tool permissions are least-privilege
- tool metadata includes provenance, schema, tests, versions, and lifecycle state
- benchmark runs compare OpenAI, Ollama, Anthropic, and mock/local providers where available
- provider-specific enablement respects benchmark thresholds
- unsafe or flaky tools are flagged or disabled per provider
- tool schema incompatibilities are detected and reported
- benchmark failures do not enable tools accidentally
- future overhaul compatibility for Discord, scheduling, and provider routing
