# AUDIT_SECURITY

- Timestamp: 2026-07-08 10:07:13

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and Web TUI scan
```
/home/missy/missy/LAST_SESSION_SUMMARY.md:7:- Extended `missy/api/diagnostics.py` with gateway, Discord, provider/tool
/home/missy/missy/LAST_SESSION_SUMMARY.md:8:  network scope, REST policy, and scheduling policy readiness checks.
/home/missy/missy/LAST_SESSION_SUMMARY.md:9:- Added optional redacted remediation hints to diagnostics checks.
/home/missy/missy/LAST_SESSION_SUMMARY.md:15:  output, and token redaction.
/home/missy/missy/LAST_SESSION_SUMMARY.md:16:- Updated required loop artifacts for the Web TUI primary focus.
/home/missy/missy/LAST_SESSION_SUMMARY.md:45:- Run/session streaming viewer is still not implemented.
/home/missy/missy/LAST_SESSION_SUMMARY.md:46:- Live diagnostics probes should be added carefully behind policy and timeout
/home/missy/missy/LAST_SESSION_SUMMARY.md:54:policy-gated operator controls API for safe enable/disable workflows.
/home/missy/missy/fixes.md:14:### 1.1 🔴 Token bucket is never pre-charged — TPM limiting is effectively disabled
/home/missy/missy/fixes.md:15:- **Where:** `missy/providers/anthropic_provider.py:142` and `:266`; `missy/providers/base.py:227-238`.
/home/missy/missy/fixes.md:16:- **Problem:** `complete()` / `complete_with_tools()` call `self._acquire_rate_limit()`
/home/missy/missy/fixes.md:17:  with **no `estimated_tokens` argument**, so `RateLimiter.acquire(tokens=0)` only spends
/home/missy/missy/fixes.md:19:  `record_usage()`. A burst of large-context requests can all pass `acquire()`
/home/missy/missy/fixes.md:22:  blow past provider tokens-per-minute limits and trigger 429s (or spend) under load.
/home/missy/missy/fixes.md:24:  tokenizer) and pass to `_acquire_rate_limit(estimated_tokens=...)`. Then use
/home/missy/missy/fixes.md:26:  Do the same in `openai_provider.py` and `ollama_provider.py`.
/home/missy/missy/fixes.md:29:- **Where:** `missy/providers/registry.py:236` — `instance.rate_limiter = RateLimiter()`.
/home/missy/missy/fixes.md:30:- **Problem:** Every provider gets the default `RateLimiter(60 rpm, 100_000 tpm)`. There is
/home/missy/missy/fixes.md:40:- **Where:** `missy/providers/rate_limiter.py:149-164`.
/home/missy/missy/fixes.md:42:  The RateLimiter is shared across threads (async runtime, scheduler, Discord). A synchronous
/home/missy/missy/fixes.md:45:- **Impact:** One provider's 429 can freeze the whole agent for up to `max_wait` (30s).
/home/missy/missy/fixes.md:47:  next `acquire()` compute the wait naturally, or push the retry-after as a "not before"
/home/missy/missy/fixes.md:51:- **Where:** `anthropic_provider.py:158-159`, `openai_provider.py:141`.
/home/missy/missy/fixes.md:63:- **Where:** `missy/scheduler/manager.py:332-355` (`_run_job` active-hours gate) combined with
/home/missy/missy/fixes.md:75:- **Where:** `missy/scheduler/jobs.py:108-122` (`should_retry` ignores `error`), field defined at `:57`.
/home/missy/missy/fixes.md:76:- **Problem:** `ScheduledJob.retry_on` defaults to `["network", "provider_error"]` and is
/home/missy/missy/fixes.md:78:  **every** error type up to `max_attempts`. Non-retryable errors (e.g. auth failure, invalid
/home/missy/missy/fixes.md:87:- **Problem:** The config schema advertises `max_jobs` (0 = unlimited) but `SchedulerManager`
/home/missy/missy/fixes.md:107:- **Fix:** Update `next_run` from `self._scheduler.get_job(...)` in a `finally` block.
/home/missy/missy/fixes.md:113:### 3.1 🟠 API session registry and memory-store turn counts drift
/home/missy/missy/fixes.md:115:  `_handle_list_sessions:588-597` (overlays DB counts).
/home/missy/missy/fixes.md:117:  turns persist in SQLite. After a restart, `/chat` with a previously valid `session_id` returns
/home/missy/missy/fixes.md:120:- **Impact:** Confusing client behavior; history is orphaned relative to the session lifecycle.
/home/missy/missy/fixes.md:121:- **Fix:** Back the session registry with the memory store's `sessions` table (there's already a
/home/missy/missy/fixes.md:122:  `register_session`/`list_sessions` API), or on 404 fall back to the DB session if turns exist.
/home/missy/missy/fixes.md:130:- **Impact:** `missy cost` under-reports call count on long sessions; budget summary `call_count`
/home/missy/missy/fixes.md:141:- **Fix:** Re-raise (or surface via an audit event + return status) on persistence failure for
/home/missy/missy/fixes.md:161:- **Problem:** When `sandbox.enabled: true` but Docker isn't accessible, `get_sandbox` logs a
/home/missy/missy/fixes.md:163:  isolation** (just a scrubbed env). An operator who enabled sandboxing for safety silently loses
/home/missy/missy/fixes.md:166:  host despite `enabled: true`.
/home/missy/missy/fixes.md:167:- **Fix:** Add a `require_isolation`/`strict` flag. When sandbox is enabled and Docker is
/home/missy/missy/fixes.md:173:- **Problem:** The `SandboxConfig` promises `network_disabled`, `memory_limit`, `cpu_limit`,
/home/missy/missy/fixes.md:182:### 4.3 🟠 REST-policy L7 rules are only enforced inside `PolicyHTTPClient`
/home/missy/missy/fixes.md:183:- **Where:** `missy/gateway/client.py:355-357` calls `_check_rest_policy` only when a `method` is
/home/missy/missy/fixes.md:186:  (`anthropic_provider.py:84`), bypassing `PolicyHTTPClient` entirely. So network policy + REST
/home/missy/missy/fixes.md:187:  policy + response-size caps + audit events do **not** apply to the largest volume of outbound
/home/missy/missy/fixes.md:189:  HTTP" is not true for provider calls.
/home/missy/missy/fixes.md:190:- **Impact:** Egress controls and audit are blind to provider traffic; a compromised base_url or
/home/missy/missy/fixes.md:191:  proxy setting can exfiltrate without policy review.
/home/missy/missy/fixes.md:192:- **Fix:** Pass a policy-aware `http_client` into the provider SDKs (both Anthropic and OpenAI SDKs
/home/missy/missy/fixes.md:201:- **Fix:** On injection detection, emit a `security` audit event and either skip the run or require
/home/missy/missy/fixes.md:202:  an approval gate (`ApprovalGate`) before executing.
/home/missy/missy/fixes.md:212:- **Fix:** Pass stable AAD (e.g. `b"missy-vault-v1:" + uid`) to ChaCha20Poly1305 so cross-context
/home/missy/missy/fixes.md:213:  ciphertext substitution fails authentication.
/home/missy/missy/fixes.md:224:### 4.7 🟡 API `/chat` mutates shared runtime provider globally per-request
/home/missy/missy/fixes.md:225:- **Where:** `api/server.py:518-522` — `runtime.switch_provider(provider_override)`.
/home/missy/missy/fixes.md:226:- **Problem:** `switch_provider` changes the runtime's active provider process-wide. Two concurrent
/home/missy/missy/fixes.md:227:  `/chat` requests with different `provider` values race, and one request's override leaks into the
/home/missy/missy/fixes.md:229:- **Fix:** Pass the provider per-run (as a `run()` argument) rather than mutating shared state, or
/home/missy/missy/fixes.md:230:  serialize/scope provider selection per session.
/home/missy/missy/fixes.md:252:- **Where:** `missy/providers/registry.py:257-280`.
/home/missy/missy/fixes.md:258:### 5.4 🟡 `_extract_all_programs` blanket-rejects `<<` but the tool advertises redirection
/home/missy/missy/fixes.md:259:- **Where:** `missy/policy/shell.py:135` (`_SUBSHELL_MARKERS` includes `"<<"`) vs.
/home/missy/missy/fixes.md:270:### 6.1 🟢 Feature: Provider-tier & rate-limit configuration surface (`providers.*.limits`)
/home/missy/missy/fixes.md:274:  providers:
/home/missy/missy/fixes.md:275:    anthropic:
/home/missy/missy/fixes.md:281:  Wire into `ProviderRegistry.from_config` to build a correctly-sized `RateLimiter`, add
/home/missy/missy/fixes.md:282:  `missy providers limits` CLI to display current buckets (`request_capacity`/`token_capacity`),
/home/missy/missy/fixes.md:283:  and estimate prompt tokens before `acquire()` so TPM is actually honored. Emit an audit event
/home/missy/missy/fixes.md:287:### 6.2 🟢 Feature: Egress audit report (`missy audit egress`) + universal gateway routing
/home/missy/missy/fixes.md:288:- **Motivation:** Item 4.3 — provider SDKs bypass the gateway, so there's no single egress ledger.
/home/missy/missy/fixes.md:290:  1. Inject a policy-aware `httpx` client into every provider SDK so all outbound HTTP flows
/home/missy/missy/fixes.md:291:     through `PolicyHTTPClient` (network + REST policy + size cap + `network_request` audit events).
/home/missy/missy/fixes.md:292:  2. Add `missy audit egress [--since] [--host]` that aggregates `network_request` audit events
/home/missy/missy/fixes.md:294:     `~/.missy/audit.jsonl`.
/home/missy/missy/fixes.md:296:  operators a real egress dashboard for a security-first product.
/home/missy/missy/fixes.md:298:### 6.3 🟢 (Optional) Feature: Persistent, restart-safe API sessions
/home/missy/missy/fixes.md:299:- **Motivation:** Item 3.1 — API sessions are memory-only and orphan their DB history on restart.
/home/missy/missy/fixes.md:300:- **Proposal:** Back `_SessionRegistry` with the memory store's `sessions` table (load on startup,
/home/missy/missy/fixes.md:301:  write-through on create/touch/delete). Add a `last_provider` column so `/chat` can resume with the
/home/missy/missy/fixes.md:302:  correct provider after a restart.
/home/missy/missy/fixes.md:312:4. **4.3** provider traffic bypasses gateway (security/audit, high) → enables 6.2.
/home/missy/missy/fixes.md:313:5. **3.1 / 3.3** session drift + silent job-persistence loss (data consistency).
/home/missy/missy/fixes.md:316:*Note:* several fixes have existing tests under `tests/` (policy, scheduler, providers,
/home/missy/missy/fixes.md:317:memory). Update/extend those suites alongside each change; coverage threshold is 90%.
/home/missy/missy/pyproject.toml:1:[build-system]
/home/missy/missy/pyproject.toml:2:requires = ["setuptools>=68", "wheel"]
/home/missy/missy/pyproject.toml:3:build-backend = "setuptools.build_meta"
/home/missy/missy/pyproject.toml:8:description = "A policy-enforced AI agent framework"
/home/missy/missy/pyproject.toml:9:requires-python = ">=3.11"
/home/missy/missy/pyproject.toml:14:    "anthropic>=0.25",
/home/missy/missy/pyproject.toml:15:    "openai>=1.25",
/home/missy/missy/pyproject.toml:17:    "apscheduler>=3.10",
/home/missy/missy/pyproject.toml:22:    "websockets>=12.0",
/home/missy/missy/pyproject.toml:48:# Discord voice transport/session. Requires system ffmpeg installed.
/home/missy/missy/pyproject.toml:60:# Browser automation (Firefox via Playwright) and GTK/X11 accessibility tools.
/home/missy/missy/pyproject.toml:71:[tool.setuptools.package-data]
/home/missy/missy/pyproject.toml:72:missy = ["py.typed", "channels/screencast/web/*.html"]
/home/missy/missy/pyproject.toml:74:[tool.setuptools.packages.find]
/home/missy/missy/pyproject.toml:78:[tool.pytest.ini_options]
/home/missy/missy/pyproject.toml:82:    "ignore:websockets.legacy is deprecated:DeprecationWarning",
/home/missy/missy/pyproject.toml:87:[tool.black]
/home/missy/missy/pyproject.toml:91:[tool.ruff]
/home/missy/missy/pyproject.toml:95:[tool.ruff.lint]
/home/missy/missy/pyproject.toml:99:[tool.ruff.lint.isort]
/home/missy/missy/pyproject.toml:102:[tool.coverage.run]
/home/missy/missy/pyproject.toml:106:[tool.coverage.report]
/home/missy/missy/pyproject.toml:110:[tool.mypy]
/home/missy/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
/home/missy/missy/HUMANIZE_STATUS.md:31:| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
/home/missy/missy/HUMANIZE_STATUS.md:35:| H_I | Mood state with decay | not_started | First humanize implementation target in sessions 8-9. |
/home/missy/missy/HUMANIZE_STATUS.md:39:- Initialized required loop tracking documents.
/home/missy/missy/HUMANIZE_STATUS.md:41:- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
/home/missy/missy/HUMANIZE_STATUS.md:45:- Session 2 added the A2 layered tool policy pipeline with profile bundles, group expansion, glob matching, inline `-tool` denies, `alsoAllow`, fail-warning unknown allowlists, and structured trace records.
/home/missy/missy/HUMANIZE_STATUS.md:46:- Session 2 wired `AgentRuntime._get_tools()` to resolve tools through the pipeline and record `_last_tool_policy_decision` for audit/debugging.
/home/missy/missy/HUMANIZE_STATUS.md:47:- Session 2 added `tests/policy/test_tool_policy_pipeline.py` and runtime coverage for policy decisions in `tests/agent/test_runtime_streaming.py`.
/home/missy/missy/HUMANIZE_STATUS.md:48:- Session 3 added config parsing for `tools.*`, `tools.byProvider`, `tools.byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools`.
/home/missy/missy/HUMANIZE_STATUS.md:49:- Session 3 added `build_configured_tool_policy_layers()` and `collect_tool_policy_groups()` so runtime policy resolution now consumes YAML-backed provider/global/agent/sandbox/subagent layers.
/home/missy/missy/HUMANIZE_STATUS.md:50:- Session 3 routed parsed tool policies into CLI-created runtimes for ask/run/gateway/API paths and documented the YAML surface in `docs/configuration.md`.
/home/missy/missy/HUMANIZE_STATUS.md:51:- Session 3 added config, policy-pipeline, and runtime tests for those surfaces, then verified the full test suite and full-repo ruff.
/home/missy/missy/HUMANIZE_STATUS.md:55:1. Harden A1 by routing provider/tool-loop stream events through `AgentSubscription` where Missy's providers expose stream events, not only the simple `run_stream()` path.
/home/missy/missy/HUMANIZE_STATUS.md:56:2. Add the A7 `BlockChunker` and connect it to A1 flush points so pre-tool text can be delivered through Discord/CLI/Web in order.
/home/missy/missy/HUMANIZE_STATUS.md:58:4. Add channel/group policy sources on top of the A2 pipeline when Discord/CLI/Web channel identity context is available.
/home/missy/missy/examples/systemd/README.md:8:## Prerequisites
/home/missy/missy/examples/systemd/README.md:16:- Any required API keys must be set as environment variables. You can use a
/home/missy/missy/examples/systemd/README.md:51:sudo systemctl enable --now missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:54:This enables the service to start on boot and starts it immediately.
/home/missy/missy/examples/systemd/README.md:91:sudo systemctl disable missy-gateway@youruser
/home/missy/missy/examples/systemd/README.md:102:| `NoNewPrivileges=true` | Prevents the process from gaining new privileges (e.g. via setuid). |
/home/missy/missy/examples/systemd/README.md:106:| `ReadWritePaths=...` | Grants write access to `~/.missy` (for audit logs, jobs, memory) and `~/workspace` (for agent output). |
/home/missy/missy/examples/systemd/README.md:120:- Check that `config.yaml` includes the required domains in `allowed_domains` or `allowed_hosts`.
/home/missy/missy/examples/systemd/README.md:124:- Check that `discord.enabled: true` is set in `config.yaml`.
/home/missy/missy/install.sh:29:    echo "Error: Python 3.11+ is required." >&2
/home/missy/missy/install.sh:37:    echo "Error: git is required." >&2
/home/missy/missy/README.md:5:Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.
/home/missy/missy/README.md:13:Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.
/home/missy/missy/README.md:18:- **No plugins** unless you approve them individually
/home/missy/missy/README.md:19:- **Every action** logged as structured JSONL with full audit trail
/home/missy/missy/README.md:20:- **Every audit event** signed with the agent's Ed25519 identity
/home/missy/missy/README.md:29:- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
/home/missy/missy/README.md:30:- **API key rotation** — multiple keys per provider, round-robin distribution
/home/missy/missy/README.md:31:- **Model tiers** — `fast_model` for quick tasks, `premium_model` for complex reasoning, auto-routed by ModelRouter
/home/missy/missy/README.md:32:- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
/home/missy/missy/README.md:33:- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
/home/missy/missy/README.md:34:- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
/home/missy/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy/README.md:42:- **Cost tracking** — per-session budget caps with `max_spend_usd`
/home/missy/missy/README.md:44:- **Checkpoint recovery** — WAL-mode SQLite checkpointing; `missy recover` resumes incomplete sessions
/home/missy/missy/README.md:45:- **Failure tracking** — per-tool consecutive failure counts with automatic strategy rotation
/home/missy/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy/README.md:53:- **REST API** — Agent-as-a-Service endpoint (`missy api start`) with loopback binding, API key auth, rate limiting
/home/missy/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
/home/missy/missy/README.md:63:- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
/home/missy/missy/README.md:64:- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
/home/missy/missy/README.md:65:- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
/home/missy/missy/README.md:66:- **Landlock LSM** — Linux kernel-level filesystem enforcement via Landlock syscalls, complementing userspace policy
/home/missy/missy/README.md:67:- **Security scanner** — `missy security scan` audits installation for permission issues, config hygiene, exposed secrets
/home/missy/missy/README.md:68:- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
/home/missy/missy/README.md:72:- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
/home/missy/missy/README.md:73:- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
/home/missy/missy/README.md:74:- **Webhooks** — HTTP ingress with HMAC auth, rate limiting, payload validation
/home/missy/missy/README.md:75:- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth
/home/missy/missy/README.md:76:- **Screencast** — browser-based screen capture channel with token authentication and session management
/home/missy/missy/README.md:80:- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
/home/missy/missy/README.md:81:- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
/home/missy/missy/README.md:82:- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
/home/missy/missy/README.md:85:- **Persona system** — YAML-backed agent identity/tone/style with backup, rollback, and audit logging
/home/missy/missy/README.md:93:- **Multi-provider** — Anthropic/OpenAI/Ollama image message formatting
/home/missy/missy/README.md:95:- **CLI tools** — `missy vision capture|inspect|review|doctor|health|benchmark|validate|memory`
/home/missy/missy/README.md:98:- **Browser tools** — Playwright-based Firefox automation (`pip install -e ".[desktop]"`)
/home/missy/missy/README.md:99:- **X11 tools** — window management and application launching
/home/missy/missy/README.md:100:- **Accessibility** — AT-SPI toolkit integration for GUI interaction
/home/missy/missy/README.md:103:- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
/home/missy/missy/README.md:106:- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`
/home/missy/missy/README.md:109:- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
/home/missy/missy/README.md:110:- **Application logs** — rotating Python/provider diagnostics at `~/.missy/missy.log` (`missy logs tail`)
/home/missy/missy/README.md:112:- **Cost tracking** — per-session spend monitoring with configurable caps
/home/missy/missy/README.md:122:This clones to `~/.local/share/missy`, creates a venv, installs, and symlinks `missy` into `~/.local/bin`. Requires Python 3.11+ and git.
/home/missy/missy/README.md:124:## Quick Start
/home/missy/missy/README.md:130:The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:
/home/missy/missy/README.md:134:missy run    # interactive session
/home/missy/missy/README.md:152:missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
/home/missy/missy/README.md:164:pip install -e ".[discord_voice]" # discord.py[voice] + voice recv; requires system ffmpeg
/home/missy/missy/README.md:165:pip install -e ".[dev]"           # pytest, ruff, mypy, hypothesis, coverage tools
/home/missy/missy/README.md:182:(network,     (Anthropic, OpenAI,        (built-in tools,
/home/missy/missy/README.md:183: filesystem,   Ollama + fallback)         skills, plugins,
/home/missy/missy/README.md:199: Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
/home/missy/missy/README.md:205:Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.
/home/missy/missy/README.md:217:  default_deny: true
/home/missy/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy/README.md:234:  enabled: false
/home/missy/missy/README.md:237:providers:
/home/missy/missy/README.md:238:  anthropic:
/home/missy/missy/README.md:239:    name: anthropic
/home/missy/missy/README.md:246:  enabled: false
/home/missy/missy/README.md:266:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy/README.md:267:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy/README.md:268:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy/README.md:269:missy providers list                # List providers and availability
/home/missy/missy/README.md:270:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy/README.md:277:# Security & audit
/home/missy/missy/README.md:278:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy/README.md:279:missy audit security                # Policy violations
/home/missy/missy/README.md:286:missy presets list                  # Show built-in network presets
/home/missy/missy/README.md:289:missy discord status | probe | register-commands | audit
/home/missy/missy/README.md:293:missy devices list | pair | unpair | status | policy
/home/missy/missy/README.md:295:# MCP & skills
/home/missy/missy/README.md:297:missy skills                        # List registered skills
/home/missy/missy/README.md:298:missy skills scan                   # Discover SKILL.md files
/home/missy/missy/README.md:302:missy vision health | benchmark | validate | memory
/home/missy/missy/README.md:317:missy sessions list | rename | cleanup
/home/missy/missy/README.md:344:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy/README.md:354:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy/README.md:370:| [Getting Started](https://missylabs.github.io/getting-started/) | 5 | Install, quickstart, wizard, first conversation |
/home/missy/missy/README.md:371:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy/README.md:373:| [Architecture](https://missylabs.github.io/architecture/) | 10 | Runtime, context, circuit breaker, progress, playbook, sleep mode, synthesizer, attention, message bus |
/home/missy/missy/README.md:376:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy/README.md:377:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy/README.md:384:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy/README.md:392:├── agent/           Runtime, circuit breaker, context, playbook, consolidation,
/home/missy/missy/README.md:393:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy/README.md:396:├── channels/        CLI, Discord, webhooks, voice (WebSocket), screencast (browser)
/home/missy/missy/README.md:404:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy/README.md:405:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy/README.md:406:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy/README.md:409:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy/README.md:410:├── plugins/         Security-gated external plugin loader
/home/missy/missy/README.md:411:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy/BUILD_STATUS.md:1:# Build Status
/home/missy/missy/BUILD_STATUS.md:8:stdlib `ApiServer` still owns the local Web UI and JSON API, with API-key auth,
/home/missy/missy/BUILD_STATUS.md:9:browser login sessions, CSRF checks for unsafe browser API calls, hardened
/home/missy/missy/BUILD_STATUS.md:10:cookies, no-store responses, and server-side redaction.
/home/missy/missy/BUILD_STATUS.md:12:This session deepened the operator diagnostics/doctor slice:
/home/missy/missy/BUILD_STATUS.md:14:- Added gateway diagnostics for the policy-enforcing HTTP client, response size
/home/missy/missy/BUILD_STATUS.md:15:  cap, and active policy binding.
/home/missy/missy/BUILD_STATUS.md:16:- Added Discord readiness diagnostics from the active policy/config object:
/home/missy/missy/BUILD_STATUS.md:18:  network prerequisites, and Discord voice tool visibility.
/home/missy/missy/BUILD_STATUS.md:21:- Exposed policy network scope checks for provider/tool allowlists and REST
/home/missy/missy/BUILD_STATUS.md:30:| Existing JSON API auth | preserved | `X-API-Key` and bearer token auth still guard `/api/v1/*`. |
/home/missy/missy/BUILD_STATUS.md:32:| Web session handling | extracted | `WebSession` and `WebSessionStore` live in `missy/api/web_sessions.py`. |
/home/missy/missy/BUILD_STATUS.md:33:| CSRF protection | implemented | Required for unsafe API calls made with browser session cookies; denials are audited. |
/home/missy/missy/BUILD_STATUS.md:34:| Operator dashboard | improved | Runtime, providers, tools, sessions, security posture, audit trail, and diagnostics are shown. |
/home/missy/missy/BUILD_STATUS.md:35:| Audit log browser API | improved | Authenticated `/api/v1/audit` supports filters, facets, redaction, IDs, totals, offsets, and `has_more`. |
/home/missy/missy/BUILD_STATUS.md:37:| Diagnostics API | improved | `/api/v1/diagnostics` now reports Web, providers, tools, memory, policy, gateway, Discord, scheduler, and runtime posture. |
/home/missy/missy/BUILD_STATUS.md:39:| Console security tests | expanded | API suite covers auth, CSRF, audit, diagnostics redaction, gateway posture, and Discord policy readiness. |
/home/missy/missy/BUILD_STATUS.md:44:  session lifecycle, audit querying, and diagnostics view models are separated.
/home/missy/missy/BUILD_STATUS.md:45:- Dashboard JavaScript calls `/api/v1/status`, `/providers`, `/tools`,
/home/missy/missy/BUILD_STATUS.md:46:  `/sessions`, `/diagnostics`, and `/audit` with same-origin credentials and
/home/missy/missy/BUILD_STATUS.md:48:- Diagnostics are pure read-only view models built from injected server
/home/missy/missy/BUILD_STATUS.md:49:  dependencies plus the active policy engine config reference.
/home/missy/missy/BUILD_STATUS.md:50:- Diagnostic checks redact summaries and remediation text before returning data
/home/missy/missy/BUILD_STATUS.md:52:- `LOOP_INSTRUCTIONS.md` remains modified from outside this session and was not
/home/missy/missy/BUILD_STATUS.md:64:1. Continue extracting Web TUI rendering and frontend assets out of
/home/missy/missy/BUILD_STATUS.md:66:2. Add safe policy-gated controls for providers, tools, scheduled jobs,
/home/missy/missy/BUILD_STATUS.md:68:3. Add run/session viewer with streaming output, tool calls, errors, model
/home/missy/missy/BUILD_STATUS.md:71:   reachability checks that do not bypass policy.
/home/missy/missy/LOOP_HEALTH.md:5:- Branch: overhaul/web-tui-20260708-122250
/home/missy/missy/LOOP_HEALTH.md:6:- Primary focus: complete web TUI and operator console overhaul
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:8:and Odin remain references for operator ergonomics, diagnostics, auditability,
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:16:| Secure local Web UI entrypoint | started | `/login` and `/` implemented with cookie sessions and CSRF. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:17:| Explicit authentication/session handling | improved | Browser session storage is extracted into `missy/api/web_sessions.py`. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:18:| Polished dashboard | started | Runtime, providers, tools, sessions, security posture, audit trail, and diagnostics are shown. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:19:| Session/run viewer | not_started | Needs streaming output, tool calls, errors, costs, routing, fallback, resume context. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:20:| Audit log browser | improved | `/api/v1/audit` supports filters, facets, file/memory sources, redaction, IDs, totals, offsets, and `has_more`; UI has filters, pagination, and details. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:21:| Diagnostics/doctor views | improved | `/api/v1/diagnostics` now covers Web, providers, tools, memory, policy, gateway, Discord, scheduler, runtime, and remediation hints. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:22:| Safe operator controls | not_started | Must be policy-gated, default-deny, audited, and confirmation guarded. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:24:| Backend Web TUI security | improved | Auth, CSRF, rate limit, hardened headers, audit events, redaction, XSS-resistant dashboard rendering, and redacted audit search are in place. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:31:| A2 | Layered tool policy pipeline | hardened | Policy surfaces include current security updates and are now reflected in Discord diagnostics. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:32:| A3 | Mutation fingerprinting + sticky lastToolError | implemented | Repeated identical failing tool calls are fingerprinted and surfaced to the model. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:35:| A6 | Per-provider tool schema normalization | live | Provider schema methods delegate to `normalize_for_provider()` with fallbacks. |
/home/missy/missy/OPENCLAW_GAP_ANALYSIS.md:47:small safe controls API with policy-gated enable/disable operations and audit
/home/missy/missy/HUMANIZE_TEST_PLAN.md:9:- Mock LLM/provider calls. Behavioral tests should assert prompt fragments, state transitions, audit entries, cooldown decisions, or emitted channel timing calls.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:10:- Keep security and reliability separate from style: humanistic behaviors must not bypass policy, mutate tool results, or hide errors.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:22:  - Block flush at `text_end` and before tool execution.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:27:- A2 policy coverage: `tests/policy/test_tool_policy_pipeline.py`
/home/missy/missy/HUMANIZE_TEST_PLAN.md:29:  - Glob allow rules and inline `-tool` deny syntax compose in one layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:30:  - `alsoAllow` can restore matching tools after a restrictive layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:31:  - Unknown plugin-only allowlists warn without hiding core tools.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:32:  - Standard profile → provider → global → agent → group → sandbox → subagent layer ordering records trace labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:33:  - Config-backed provider/global/agent/sandbox/subagent layers preserve ordering and source labels.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:34:  - Custom `tools.groups` definitions extend the built-in group map.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:37:  - `tools.*`, `tools.byProvider`, nested `byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools` parse from YAML.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:38:  - Invalid tool profiles fail with a configuration error.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:40:  - `AgentRuntime._get_tools()` records a `ToolPolicyDecision` and filters `safe-chat` through the A2 profile layer.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:42:  - `AgentRuntime._get_tools()` consumes config-backed global and agent policy surfaces.
/home/missy/missy/HUMANIZE_TEST_PLAN.md:48:| H_A | Delay calculation respects length/complexity/mood/channel caps; quick/fast/asap bypasses long sleeps; channel typing indicator ordering is mocked. |
/home/missy/missy/HUMANIZE_TEST_PLAN.md:54:| H_G | Apology appears for a tool failure once; gratitude and hedging do not duplicate in the same exchange. |
/home/missy/missy/HUMANIZE_AUDIT.md:3:Rotation policy: keep this file under 5 MB. Move older entries to timestamped archive files before appending more.
/home/missy/missy/HUMANIZE_AUDIT.md:7:| 2026-04-27T16:09:36Z | humanize.loop.initialized | allow | Initialized audit file for the OpenClaw/humanize loop. No opt-in humanistic behavior was activated this session. |
/home/missy/missy/HUMANIZE_AUDIT.md:8:| 2026-04-27T16:09:36Z | openclaw.a1.subscription | allow | Added streaming state machine primitives that can support future timing, tone, apology, and mood integrations without changing tool correctness. |
```
