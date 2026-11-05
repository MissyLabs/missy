# Native production lifecycle

`missy-rs --config /absolute/config.yaml gateway start` composes configured HTTP,
Discord, edge voice, operations and continuous durable scheduler workers. No Python
launcher, automatic migration or ambient legacy state. HTTP requires
`gateway.http_enabled`; Discord uses `discord.enabled`; voice/operations use
extension paths. Scheduler requires explicit trusted four-part scopes, never
an unscoped scan. Services share one full runtime. Voice NoTools and configured
reasoning-only subagents deliberately use capability-reduced separate runtimes.

Private atomic `run/gateway_status.json` records disabled/starting/ready/failed.
`gateway status` checks ownership lock and timestamp as well as readiness and exits
nonzero if unhealthy. HTTP/voice sockets bind before readiness. Discord waits for
authenticated readiness within startup timeout. Any required service exit stops
the supervisor. Status contains no raw errors, transcripts or credential values.
SIGTERM/SIGINT stop intake, cancel/drain work and bound total drain. Expired
deadline is failure, not a fabricated clean shutdown. Durable runtime/scheduler
IDs survive restarts; interrupted uncertain effects are not blindly replayed.

## Honest reload

No in-place hot reload. Policy, credentials, approval bindings and lineage cannot
safely be swapped underneath admitted turns. Every five seconds and on SIGHUP the
supervisor validates/fingerprints config and referenced extension YAML. Changed
config becomes `restart_required`; invalid config `invalid_config_restart_refused`;
readiness is false. Running configuration is unchanged. `gateway check-reload`
provides the explicit operator check, nonzero for differences. Environment rotation
also requires restart. Referenced binary/model assets retain extension-level pins,
not supervisor reload semantics.

## Operator packaging

Nothing here was installed or restarted. Build for target architecture with locked
dependencies, at most two jobs. Provision libopus and explicitly configured speech,
media/browser commands; DAVE remains an explicitly pinned native dependency.

1. Provision dedicated `missy-native` account, private `/etc/missy-native` and
   `/var/lib/missy-native`, not the desktop user or legacy state. Migrate offline
   into a separate destination. Review `examples/rust-production.yaml`.
2. Config/extension files must be service-UID-owned mode0600. Keep the EnvironmentFile
   private. A system manager may read root-owned environment files. Never put
   credentials into unit text or process arguments.
3. `sh packaging/rust/release.sh stage PREFIX VERSION BINARY SHA256` checksum-checks
   and probes --version in a new immutable slot. Obtain SHA256 independently;
   hashing caller-supplied bytes is not publisher authentication.
4. `activate PREFIX VERSION` atomically selects current, preserves previous. It
   never invokes systemctl, edits config/state or changes running processes.
   Run config-check and doctor as the service UID. Review the distinct systemd
   unit before operator-authorized installation/cutover. Avoid duplicate Discord
   ownership. After authorized start, check status, authenticated health/diagnostics.

`rollback PREFIX` selects the previous checksum-verified binary ONLY, not schema
or config downgrade. Back up while stopped before upgrade; verify restore and
backward compatibility offline. Restore a matching separate backup for incompatible
schema changes before starting an old binary.

The service baseline omits PrivateDevices/PrivateTmp/ProtectHome/
MemoryDenyWriteExecute/syscall filters that break approved audio, X11/CDP, local
models, Landlock or extension filesystem paths. Unprivileged does not mean sandbox
certification. Add only explicitly required device groups/path access. No broad
desktop access. Mesh listener and bounded worker are supervised with the same
runtime. Discord voice shares the text Gateway, stable prebuilt tool catalog and
late-bound service. Optional operator-owned `joins` entries contain guild, channel,
actor and the exact service join approval. Empty joins means no unattended joining;
no model request can set consent or pick an actor. Service shutdown is retained even
if a later startup join fails. Audio/scene/desktop composition retains dynamic exact
mutation approvals from the corresponding extensions.

`extensions.triggers_config` supports `poll_seconds` and `entries` containing
`scope`, `config: TriggerConfig`, optional `source: ObservationSource`. File/disk
sources require exact allowed roots; load and durable scoped-idle observations
are explicit. Heartbeat/schedule can use no source. Unsupported or ungranted
observations fail startup, not silently idle. `extensions.workloads_config` starts
its bounded durable due-work poller, using the same explicit config/state policy.
The trigger file also accepts explicit `maintenance` entries `{scope,config}` with
MaintenanceConfig. Provider-free idle digest maintenance requires enabled plus
operator_consent, retains scoped provenance and does not execute recalled requests.

## Offline operator lifecycle

`config validate` parses the config and referenced extension files without opening
runtime state. `config backup /absolute/private/snapshot.json` creates a new
mode0600 snapshot; it contains the private main YAML, not extension files, vaults,
environment credentials or databases. `config diff CANDIDATE` and `config plan
CANDIDATE` print changed field names only, never values. `config rollback SNAPSHOT
--confirm` checks the pathname binding, checksum, private permissions and restored
configuration before atomically replacing the main YAML. It can recover an invalid
current YAML. It never signals, restarts or hot-reloads a running gateway. A restored
config still needs an explicit restart, and extension files require separate
version-matched backups. Concurrent cooperating rollback writers use a private lock.

`discord-outbox inspect` runs offline under the Discord ownership lock and prints
uncertain/sending records without message contents. `discord-outbox resolve NONCE
--binding HASH --operator ID --evidence TEXT [--receipt MESSAGE_ID] --confirm`
records proven delivery or explicitly abandons without retry. It never sends a
message. The binding changes when the reviewed record changes. Stop the Discord
owner before reconciliation; no second writer is allowed.

`plugins source-propose PRIVATE_YAML`, `source-show ID`, `source-review ID --hash
HASH --approve`, and `source-build ID --hash HASH --build-config PRIVATE_YAML
--confirm` expose the native reviewed-source lifecycle. Building uses the pinned
offline compiler sandbox and creates a pending binary proposal, not approval or
execution. Review and enable that binary separately. SIGTERM/SIGINT cancel builds.

The `discord`, `discord-voice` and `voice serve` entrypoints use the same supervisor
as `gateway start`: configured services are retained, one state owner, authenticated
readiness, cancellation during startup and bounded shutdown. The explicit
`discord-voice` join overrides configured automatic joins for that invocation.
Approvals bind parsed referenced extension content, not only their pathnames;
composition refuses extension drift while building a runtime. Orchestration requires
at least two concurrent turn slots, with children sharing the runtime's durable
limits. Delegation provider IDs resolve explicit operator-approved native profiles,
not aliases of the parent's fallback pool. Reasoning-only children remain NoTools.
