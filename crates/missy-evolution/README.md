# Native operator-reviewed evolution

Inert hatching/persona proposals, immutable A/B experiments and explicitly invoked
Git-backed code application/rollback. No runtime ToolExecutor, restart, deploy,
push, shell hooks, project commands or proposed-source execution.

## Workflow

Open an absolute SQLite path in a private operator-owned 0700 directory. Existing
database must be regular, non-hardlinked, operator-owned and 0600. Supply explicit
canonical `EvolutionPolicy` read/write roots, operator_user and allow_git_commits.

`propose(scope, policy, repo, diff, title)` records an immutable full plan binding
scope, canonical repo, branch, base HEAD, diff SHA-256, policy SHA-256 and exact
before/after snapshots/modes. Review the full plan. `review(scope, id,
exact_plan_hash, operator_id, approve)` reviews precisely that plan and is inert.

Only an authenticated operator surface may invoke `apply(scope, id,
exact_plan_hash, pinned_head, policy, operator_id, true)`. Operator must match
trusted scope.user, policy.operator_user and the reviewing operator. The CLI
must authenticate/authorize its operator independently of model-generated text.
`rollback` has the same shape and pins HEAD to the applied commit; it creates an
inverse commit, never resets/stashes/checks out unrelated work. Any newer commit,
dirty tracked file or untracked file prevents apply and rollback.

SQLite uses immediate transactions and FULL durability. Effects are journaled
before source installation. Errors after intent become `recovery_required`;
crashes leave `applying`/`rolling_back`. Neither state can be replayed. Inspect
`recovery_journal` and audit: full snapshots, old HEAD, branch, planned commit,
exchanged-file backups. Manual operator recovery is required, not an automatic
reset or speculative retry. Scratch index paths are transient; full durable
snapshots and retained source backups are recovery evidence.

## Deliberately narrow supported subset

Linux only. Supply an isolated **disposable work repository**, not a live checkout.
Requires normal `.git` directory, named branch, clean tracked/untracked status,
regular Git metadata, no linked worktrees, alternates, pending Git operation,
assume-unchanged or skip-worktree. Local config permits baseline `[core]` settings
and `[user] name/email` only. Includes, filters, remotes, extensions and arbitrary
config are rejected to avoid interpretation of executable/helper configuration.

Strict unified-diff parser supports existing tracked regular UTF-8 text-file
modifications only: <=64 files, <=1 MiB patch, <=4 MiB/file, newline-terminated
source. Rejects binary, quoted/absolute/parent/.git paths, renames, additions,
deletions, modes, unknown lines and mismatched/fuzzy hunks. Filenames use ASCII
alphanumeric/dot/underscore/hyphen components. Unsupported input must be
regenerated within this subset, never silently reinterpreted.

## Transaction and security boundaries

Persistent advisory repo lock; directory descriptors, O_NOFOLLOW, ownership and
nlink checks; Git cwd pinned using fchdir before exec. Empty child environment,
absolute Git binary, disabled hooks/fsmonitor/signing, isolated staging index,
no filters/shell or project execution. Before changing source, create blobs and
a commit with Git plumbing. Install files using `renameat2(RENAME_EXCHANGE)`,
retain original inodes until refs/index finish, compare-and-swap branch HEAD.

**Multi-file writes are journaled, not globally atomic for outside readers.**
Each file exchange is atomic, but multiple files/SQLite/ref/index cannot share a
single filesystem transaction. Unknown effects stop conservatively, retaining
evidence, rather than overwriting possibly unrelated work. A noncooperative
same-OS-user writer can defeat advisory locking; this is not a sandbox against
the operator/root. Use isolated repos with no concurrent writers or live readers.
Project tests/builds are arbitrary execution and must be run separately in
operator-chosen isolation before review. Patch validity does not imply correctness.

## Persona and hatching

`propose_hatching(scope,HatchingAnswers{name,tone,purpose,provider_label},
governance,context_hash)` produces pending persona and inert config JSON only.
No credentials/config writes, provider probes, memory seeds or service starts.
Parent owns interactive wizard questions and displays proposals.

`ExperimentRegistry::new(EvolutionRegistry)` supports create/experiment/assign/
record_outcome/compare. Immutable scoped variants, persisted deterministic
principal-bound assignment, deduplicated untrusted scores. Exact scope includes
tenant/user/channel/session; sharing experiments across users is not implicit.
Scores can never promote variants or authorize source changes.

`propose_promotion` requires confirmed operator and exact experiment hash, and
creates only pending governance DocumentProposal. Existing ApprovalRegistry
decide_document remains the final human approval/revision gate. Existing persona
history/base-version/context checks and PromptSnapshot remain authoritative.
Parent loads approved scoped snapshots at the next runtime turn boundary.

## Validation

`cargo test -p missy-evolution -- --test-threads=1`

Tests use only temporary synthetic Git fixtures: durable scope/exact-hash review,
confirmed pinned apply, multi-file inverse rollback, preservation of unrelated
edits, dirty/new HEAD rejection, policy/private store checks, symlink/hardlink,
path/grammar/config/metadata rejection, injected durability failure with retained
journal/no replay, hatching inactivity, stable assignment, comparison/dedup and
human-governed promotion. No live service/config/repository is changed.
