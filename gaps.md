# Missy — Implementation Gaps

Gap analysis comparing `prompt.md` requirements against the current implementation.

---

## Missing CLI Commands (prompt.md lines 578-596)

| Required Command | Status |
|---|---|
| `missy gateway start` | **Missing** — no `gateway` subgroup exists |
| `missy gateway status` | **Missing** — no `gateway` subgroup exists |
| `missy doctor` | **Missing** — no health diagnostics command |

---

## Missing Documentation Files (prompt.md lines 388-410)

These docs are explicitly required but don't exist:

- `ARCHITECTURE.md`
- `CONFIG_REFERENCE.md`
- `PROVIDERS.md`
- `SCHEDULER.md`
- `SKILLS_AND_PLUGINS.md`
- `MEMORY_AND_PERSISTENCE.md`
- `TESTING.md`
- `TROUBLESHOOTING.md`
- `docs/implementation/module-map.md`
- `docs/implementation/agent-loop.md`
- `docs/implementation/policy-engine.md`
- `docs/implementation/network-client.md`
- `docs/implementation/provider-abstraction.md`
- `docs/implementation/scheduler-execution.md`
- `docs/implementation/audit-events.md`
- `docs/implementation/persistence-schema.md`
- `docs/implementation/manifest-schema.md`

Only `docs/THREAT_MODEL.md` and `docs/implementation/discord-channel.md` exist.

---

## Missing Features

### Tooling (prompt.md line 439)
- `mypy` is not in `pyproject.toml` dev dependencies — prompt requires `ruff + mypy`

### Systemd / service-mode (prompt.md line 468, Phase 4 line 517)
- No `.service` files, no systemd examples anywhere

### Separate per-category network allowlists (prompt.md lines 173-176)
- Prompt requires distinct allowlists for: LLM providers, tool/plugin requests, skills, fetch/web actions, Discord gateway/REST
- Current implementation has a single unified network allowlist

### Approval model for high-risk actions (prompt.md line 615)
- "require explicit confirmation before privileged tool or plugin execution" — not implemented

### Safe-chat-only / no-tools mode (prompt.md line 615)
- "configurable per channel or session source" — no per-session capability mode exists

### Separate secrets directory (prompt.md lines 192-196)
- Prompt requires 5 separate directories: config, runtime state, logs, workspace, secrets
- Current `init` only creates `~/.missy/` and `~/workspace/` with no dedicated secrets dir

### Example bundled first-party skills (prompt.md line 626)
- Only a `calculator` *tool* exists (`missy/tools/builtin/calculator.py`)
- No bundled *skills* (the `skills/` subsystem has no bundled examples)

### Discord thread-aware reply handling (prompt.md line 301)
- Mentioned as required; unclear if implemented

### Discord typing indicators (prompt.md line 336)
- "typing indicators or a comparable in-progress UX signal" — unclear if implemented

### Discord attachment/media policy gating (prompt.md line 380)
- "attachments and media handling must be policy-gated and sandboxed" — unclear if implemented

### Per-provider policy model (prompt.md line 614)
- "each LLM provider can be individually enabled, disabled, rate-limited, timed out, and bound to approved endpoints only"
- Only timeout is per-provider; no enable/disable flag or per-provider endpoint binding

### Discord example configs in docs (prompt.md lines 629-634)
- Required examples: DM pairing mode, guild mention-only, allowlisted support channel, multi-account, `allowBots="mentions"` mode
- None of these are documented

---

## Minor Discrepancy

### Weekly schedule format (prompt.md line 563)
- Prompt specifies: `"weekly on Monday at HH:MM"`
- Parser supports: `"weekly on Monday HH:MM"` (no `at`) — parser will reject the `at` variant

---

## Priority Summary

| Priority | Gap |
|---|---|
| High | `missy gateway start/status` and `missy doctor` CLI commands |
| High | 17 missing documentation files |
| Medium | Per-category network allowlists |
| Medium | Approval model for high-risk actions |
| Medium | Per-session capability modes (safe-chat-only, no-tools) |
| Medium | systemd service files |
| Low | `mypy` in dev dependencies |
| Low | Separate secrets directory |
| Low | Bundled first-party skills |
| Low | Weekly schedule `at` keyword support |
| Low | Discord UX gaps (threads, typing indicators, attachment gating) |
| Low | Per-provider enable/disable and endpoint binding |
| Low | Discord example configurations in docs |
