# Native Discord production conversion

The text Gateway and REST tools use the native runtime, not the retained Python
channel. They require explicit bot credential references and numeric operator
allowlists. There was no account login or real Discord connection during this
conversion. All protocol verification used disposable loopback fixtures.

## Intake, order and provenance

Authenticated Gateway events are persisted with their sequence before heartbeat
acknowledgment. Durable request IDs bind the exact admitted message, full scope,
and prepared attachment context. Same-conversation asks execute FIFO; status and
cancel operations bypass queued long work. Independent users/channels continue.
DMs require explicit opt-in and user authorization, but do not require a mention.
Guild mention protection is enabled by default and configurable only by operator.

Quoted reply text is inspected before mixed prompt framing, labeled historical,
and never becomes another task. Original reference JSON stays private in intake.
Optional text/ZIP enrichment is disabled by default. When enabled, it downloads
only exact channel/attachment-bound Discord CDN paths, through DNS-pinned policy
HTTP, without a bot Authorization header, redirects or archive extraction.
ZIP traversal, symlinks, duplicates, expansion ratio, member count, member sizes
and aggregate bytes are bounded. Source bytes are retained as immutable private
base64 evidence before derived text is presented to the model. Unsafe content is
quarantined, not mistaken for instructions. Prepared context is immutable across
restart, even if a signed CDN URL later expires.

Images/binary documents use model-callable `discord_fetch_attachment` with exact
guild/channel/message/attachment IDs and a policy-authorized output path. The
executor authenticates message lookup, validates the CDN descriptor and byte
length, then publishes bounded bytes. It accepts no model-supplied source URL.
Existing `vision_analyze` and document tools can then process the authorized file.

`discord_read_messages` returns bounded inert channel history; `raw_evidence`
remains exact while `model_view` is projected/redacted. Discord callers cannot
change guild/channel. Explicit non-Discord CLI/API operator scopes may access
configured channels. Channel messages are shared channel evidence, not private
cross-user personal memory. `discord_add_reaction` is approval-gated and never
retries an uncertain write. Multipart uploads retain their durable exact approval
and scope-bound nonces.

## Delivery and shutdown

Outbox preparation binds the full result hash, rejecting divergent repeat output.
Replies carry a non-pinging original-message reference. A shared per-origin/bot
REST governor honors429/global and bucket reset headers conservatively. Reads
retry429 within a finite bound; writes never blindly repeat after uncertainty.
Successful sends retain receipts, while an interrupted send uses read-only nonce
reconciliation. Absence of a matching message is not proof of failure.

`discord-outbox inspect` exposes hashes and identifiers, not message contents or
credentials. Offline exact-binding reconciliation can attest a proven Discord
message ID or explicitly abandon without resend, with operator identity and
evidence digest. Journal locking refuses concurrent transport access. There is no
normal manual-SQL recovery step and no "reset everything and hope" option.

Supervised readiness follows authenticated READY/RESUMED plus durable checkpoint,
not merely a socket opening. Cancellation stops intake/workers and drains bounded
in-flight operations. A drain deadline remains an error with durable unresolved
state. Shared voice uses one authenticated Gateway bridge, not a second bot login.

Native guild slash commands are `/missy ask`, `status`, `cancel`, `resume`, `help`.
Command registration is an explicit confirmed operator operation, not a startup
effect. Native operator approvals replace legacy chat reaction approvals: an emoji
or model-supplied consent never widens authority.

## Executed evidence

Focused native tests exercised actual mock HTTP/WebSocket traffic, authenticated
ready/sequence, process-reopen deduplication, FIFO/control admission, prepared
context immutability, quoted injection quarantine, ZIP/CDN negatives, exact raw
evidence retention, single PUT reactions, outbox result immutability, no-replay
reconciliation and cancellation. Parent exact final workspace results are recorded
separately in the production validation artifact. None of these are claims of
live Discord permissions, voice quality or hardware acceptance.
