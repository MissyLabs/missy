# AUDIT_SECURITY

- Timestamp: 2026-07-09 14:21:56

## Expected common security and operations docs
- present: README.md
- present: docs/security.md
- present: docs/operations.md
- present: docs/discord.md

## Security and tool-intelligence scan
```
/home/missy/missy-loops/missy/pyproject.toml:2:requires = ["setuptools>=68", "wheel"]
/home/missy/missy-loops/missy/pyproject.toml:3:build-backend = "setuptools.build_meta"
/home/missy/missy-loops/missy/pyproject.toml:8:description = "A policy-enforced AI agent framework"
/home/missy/missy-loops/missy/pyproject.toml:14:    "anthropic>=0.25",
/home/missy/missy-loops/missy/pyproject.toml:15:    "openai>=1.25",
/home/missy/missy-loops/missy/pyproject.toml:17:    "apscheduler>=3.10",
/home/missy/missy-loops/missy/pyproject.toml:60:# Browser automation (Firefox via Playwright) and GTK/X11 accessibility tools.
/home/missy/missy-loops/missy/pyproject.toml:71:[tool.setuptools.package-data]
/home/missy/missy-loops/missy/pyproject.toml:74:[tool.setuptools.packages.find]
/home/missy/missy-loops/missy/pyproject.toml:78:[tool.pytest.ini_options]
/home/missy/missy-loops/missy/pyproject.toml:87:[tool.black]
/home/missy/missy-loops/missy/pyproject.toml:91:[tool.ruff]
/home/missy/missy-loops/missy/pyproject.toml:95:[tool.ruff.lint]
/home/missy/missy-loops/missy/pyproject.toml:99:[tool.ruff.lint.isort]
/home/missy/missy-loops/missy/pyproject.toml:102:[tool.coverage.run]
/home/missy/missy-loops/missy/pyproject.toml:106:[tool.coverage.report]
/home/missy/missy-loops/missy/pyproject.toml:110:[tool.mypy]
/home/missy/missy-loops/missy/HUMANIZE_AUDIT.md:3:Rotation policy: keep this file under 5 MB. Move older entries to timestamped archive files before appending more.
/home/missy/missy-loops/missy/HUMANIZE_AUDIT.md:7:| 2026-04-27T16:09:36Z | humanize.loop.initialized | allow | Initialized audit file for the OpenClaw/humanize loop. No opt-in humanistic behavior was activated this session. |
/home/missy/missy-loops/missy/HUMANIZE_AUDIT.md:8:| 2026-04-27T16:09:36Z | openclaw.a1.subscription | allow | Added streaming state machine primitives that can support future timing, tone, apology, and mood integrations without changing tool correctness. |
/home/missy/missy-loops/missy/HUMANIZE_AUDIT.md:9:| 2026-04-27T18:32:16Z | openclaw.a2.tool_policy | allow | Added layered tool availability filtering with trace labels. This gates future humanistic memory tools without changing execution fail-closed policy. |
/home/missy/missy-loops/missy/HUMANIZE_AUDIT.md:10:| 2026-04-27T18:53:28Z | openclaw.a2.config_policy | allow | Routed YAML-backed provider/global/agent/sandbox/subagent tool policy layers into runtime exposure decisions. Execution policy remains fail-closed in the registry. |
/home/missy/missy-loops/missy/HATCHING_LOG.md:36:- `verify_providers` — API key detection across providers
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:7:- Added benchmark-to-candidate reconciliation via
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:9:- Added `missy tools candidates import-benchmarks <candidate_id>` with
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:10:  provider threshold options and a benchmark tool-name override.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:11:- Extended benchmark provider summaries with schema-score and tool-call quality
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:13:- Reconciled benchmark imports now update candidate benchmark summaries,
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:14:  provider-enabled flags, and audited review metadata without approving or
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:15:  enabling a tool.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:16:- Fixed the CLI enable pre-check to require `approved`, matching the
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:17:  store-level lifecycle gate.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:19:- Added unit and CLI tests for benchmark import behavior and error paths.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:24:python3 -m pytest tests/tools/test_benchmark_reconciler.py tests/tools/test_candidate_store.py tests/tools/test_benchmark.py tests/tools/test_provider_gate.py tests/cli/test_tool_provider_cli.py -q
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:45:- Web/API operator controls do not yet expose candidate lifecycle actions or
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:46:  benchmark import.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:47:- Enabled candidates still need a controlled runtime loading path with
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:48:  schema/provenance/policy/test gates.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:49:- Provider fallback recommendations exist in CLI/provider gate code but are
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:50:  not yet surfaced in runtime responses when a tool is gated off.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:51:- Candidate review can import schema-score aggregates, but provider-family
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:52:  schema compatibility reporting is still limited.
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:56:Add Web/API candidate controls for list/show/import-benchmarks/approve/enable/
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:57:deny, reusing `CandidateStore` and `CandidateBenchmarkReconciler` rather than
/home/missy/missy-loops/missy/LAST_SESSION_SUMMARY.md:58:duplicating lifecycle logic.
/home/missy/missy-loops/missy/AUDIT_SECURITY.md:11:## Security and tool-intelligence scan
/home/missy/missy-loops/missy/CLAUDE.md:7:**Missy** is a security-first, self-hosted local agentic AI assistant for Linux. Production-grade agent platform with strict security controls, policy enforcement, and full auditability. Python 3.11+.
/home/missy/missy-loops/missy/CLAUDE.md:23:python3 -m pytest tests/unit/test_policy_engine.py -v     # single file
/home/missy/missy-loops/missy/CLAUDE.md:36:Secure-by-default: all capabilities (shell, plugins, network) are disabled until explicitly enabled in `~/.missy/config.yaml`.
/home/missy/missy-loops/missy/CLAUDE.md:42:  → missy setup (wizard.py + oauth.py + anthropic_auth.py)
/home/missy/missy-loops/missy/CLAUDE.md:48:       ├─ PolicyEngine (policy/engine.py) + RestPolicy (policy/rest_policy.py)
/home/missy/missy-loops/missy/CLAUDE.md:49:       ├─ AgentIdentity (security/identity.py) — Ed25519 keypair, signs audit events
/home/missy/missy-loops/missy/CLAUDE.md:50:       ├─ TrustScorer (security/trust.py) — 0-1000 reliability tracking per tool/provider
/home/missy/missy-loops/missy/CLAUDE.md:57:       ├─ ProviderRegistry + ModelRouter (providers/registry.py)
/home/missy/missy-loops/missy/CLAUDE.md:58:       ├─ RateLimiter (providers/rate_limiter.py)
/home/missy/missy-loops/missy/CLAUDE.md:59:       ├─ PolicyHTTPClient (gateway/client.py) + InteractiveApproval (agent/interactive_approval.py)
/home/missy/missy-loops/missy/CLAUDE.md:60:       ├─ ToolRegistry (tools/registry.py) + built-in tools
/home/missy/missy-loops/missy/CLAUDE.md:62:       ├─ SkillDiscovery (skills/discovery.py) — SKILL.md dynamic skill loading
/home/missy/missy-loops/missy/CLAUDE.md:68:       ├─ ApprovalGate (agent/approval.py)
/home/missy/missy-loops/missy/CLAUDE.md:74:       ├─ FailureTracker (agent/failure_tracker.py) — per-tool failure counts + strategy rotation
/home/missy/missy-loops/missy/CLAUDE.md:77:       ├─ CodeEvolutionManager (agent/code_evolution.py) — self-evolving code with approval + git rollback
/home/missy/missy-loops/missy/CLAUDE.md:78:       ├─ StructuredOutput (agent/structured_output.py) — Pydantic schema enforcement on LLM responses
/home/missy/missy-loops/missy/CLAUDE.md:85:       ├─ SecurityScanner (security/scanner.py) — installation security auditing
/home/missy/missy-loops/missy/CLAUDE.md:108:**Policy Engine (`missy/policy/`)** — Multi-layer enforcement facade:
/home/missy/missy-loops/missy/CLAUDE.md:109:- `NetworkPolicyEngine`: CIDR blocks, domain suffix matching, per-category host allowlists (provider, tool, discord)
/home/missy/missy-loops/missy/CLAUDE.md:112:- `RestPolicy`: L7 HTTP method + path glob rules per host (e.g. allow GET /repos/**, deny DELETE /**)
/home/missy/missy-loops/missy/CLAUDE.md:113:- Network presets (`missy/policy/presets.py`): `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy-loops/missy/CLAUDE.md:115:**Gateway (`missy/gateway/client.py`)** — `PolicyHTTPClient` wraps httpx; single enforcement point for ALL outbound HTTP. Every request checked against network policy + REST policy before dispatch. `InteractiveApproval` TUI prompts operator on denied requests (y/n/a with session memory).
/home/missy/missy-loops/missy/CLAUDE.md:117:**Providers (`missy/providers/`)** — `BaseProvider` defines the interface (`Message`, `CompletionResponse`, `ToolCall`, `ToolResult`). Implementations: `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`. `ProviderRegistry` handles resolution with fallback. `ProviderConfig` supports API key rotation (`api_keys` list), `fast_model`/`premium_model` tiers.
/home/missy/missy-loops/missy/CLAUDE.md:121:- `DiscordChannel`: Full WebSocket Gateway API with access control (DM allowlist, guild/role policies), slash commands (`/ask`, `/status`, `/model`, `/help`)
/home/missy/missy-loops/missy/CLAUDE.md:124:- `VoiceChannel`: WebSocket server (default port 8765) accepting connections from edge nodes (ReSpeaker, Raspberry Pi). Protocol: JSON control frames + binary PCM audio. Device pairing with PBKDF2-hashed tokens. Per-node policy modes: `full`, `safe-chat`, `muted`. STT via faster-whisper, TTS via piper binary.
/home/missy/missy-loops/missy/CLAUDE.md:128:- `ContextManager`: Token budget (default 30k) with reserves for system prompt, tool definitions, memory fraction (15%), learnings fraction (5%). Prunes oldest history first.
/home/missy/missy-loops/missy/CLAUDE.md:131:- `AttentionSystem`: 5 brain-inspired subsystems — `AlertingAttention` (urgency keywords), `OrientingAttention` (topic extraction), `SustainedAttention` (focus continuity), `SelectiveAttention` (context filtering), `ExecutiveAttention` (tool prioritization)
/home/missy/missy-loops/missy/CLAUDE.md:132:- `Playbook`: Auto-captures successful tool patterns (task_type + tool_sequence hash), injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals. JSON persistence at `~/.missy/playbook.json`.
/home/missy/missy-loops/missy/CLAUDE.md:133:- `ProgressReporter`: Protocol with `NullReporter`, `AuditReporter`, `CLIReporter`. Called in tool loop for structured progress events.
/home/missy/missy-loops/missy/CLAUDE.md:134:- `InteractiveApproval`: Real-time Rich TUI for policy-denied operations (y=allow once, n=deny, a=allow always). Session-scoped memory. Non-TTY auto-denies.
/home/missy/missy-loops/missy/CLAUDE.md:135:- `DoneCriteria`: Generates verification prompts injected after each tool-call round
/home/missy/missy-loops/missy/CLAUDE.md:136:- `Learnings`: Extracts task_type/outcome/lesson from tool-augmented runs, persisted in SQLite
/home/missy/missy-loops/missy/CLAUDE.md:137:- `PromptPatchManager`: Self-tuning prompt patches with approval workflow (proposed/approved/rejected)
/home/missy/missy-loops/missy/CLAUDE.md:139:- `ApprovalGate`: Human-in-the-loop approval for sensitive operations
/home/missy/missy-loops/missy/CLAUDE.md:142:- `FailureTracker`: Per-tool consecutive failure counts. Injects strategy-rotation prompts after repeated failures.
/home/missy/missy-loops/missy/CLAUDE.md:145:- `CodeEvolutionManager`: Self-evolving code modification engine with approval workflow, git-backed rollback, and `missy evolve` CLI.
/home/missy/missy-loops/missy/CLAUDE.md:146:- `StructuredOutput`: Pydantic model schema enforcement on LLM responses with automatic retry on validation failure.
/home/missy/missy-loops/missy/CLAUDE.md:152:**MCP (`missy/mcp/`)** — `McpManager` manages MCP server connections. Config at `~/.missy/mcp.json`. Tools are namespaced as `server__tool`. Auto-restarts dead servers via `health_check()`. Digest pinning (`missy mcp pin`) records SHA-256 of tool manifests; mismatches refuse to load.
/home/missy/missy-loops/missy/CLAUDE.md:154:**Skills (`missy/skills/`)** — `SkillDiscovery` scans directories for SKILL.md files (cross-agent portable skill format with YAML frontmatter). `missy skills scan` lists discovered skills. Fuzzy search by name/description.
/home/missy/missy-loops/missy/CLAUDE.md:156:**Scheduler (`missy/scheduler/`)** — APScheduler-backed job management with JSON persistence at `~/.missy/jobs.json`. Parser converts human-friendly schedules to cron expressions.
/home/missy/missy-loops/missy/CLAUDE.md:163:- `AgentIdentity`: Ed25519 keypair at `~/.missy/identity.pem`. Signs audit events. JWK export.
/home/missy/missy-loops/missy/CLAUDE.md:164:- `TrustScorer`: 0-1000 reliability tracking per tool/provider/MCP server. Success (+10), failure (-50), violation (-200). Warns below threshold.
/home/missy/missy-loops/missy/CLAUDE.md:165:- `PromptDriftDetector`: SHA-256 hashes system prompts at start, verifies before each provider call. Emits `security.prompt_drift` audit event on tamper.
/home/missy/missy-loops/missy/CLAUDE.md:166:- `ContainerSandbox`: Optional Docker-based isolation for tool execution. Per-session containers with `--network=none`, memory/CPU limits. Config: `container: { enabled: true }`.
/home/missy/missy-loops/missy/CLAUDE.md:167:- `LandlockPolicy`: Linux Landlock LSM filesystem policy enforcement via ctypes syscalls. Kernel-level read/write path restrictions complementing userspace policy engine.
/home/missy/missy-loops/missy/CLAUDE.md:168:- `SecurityScanner`: Installation security auditor (`missy security scan`). Checks file permissions, config hygiene, exposed secrets, and reports severity-ranked findings.
/home/missy/missy-loops/missy/CLAUDE.md:181:- Agent tools: `vision_capture`, `vision_burst`, `vision_analyze`, `vision_devices`, `vision_scene`
/home/missy/missy-loops/missy/CLAUDE.md:184:**Memory (`missy/memory/`)** — `SQLiteMemoryStore` at `~/.missy/memory.db` with FTS5 search. Stores conversation turns and learnings. `cleanup()` removes turns older than N days. Optional `VectorMemoryStore` with FAISS semantic search (`pip install -e ".[vector]"`). `GraphMemoryStore` provides SQLite-backed entity-relationship graph memory with rule-based pattern matching for structured knowledge retrieval.
/home/missy/missy-loops/missy/CLAUDE.md:194:**Observability (`missy/observability/`)** — `AuditLogger` writes structured JSONL to `~/.missy/audit.jsonl`. `OtelExporter` sends traces/metrics to an OTLP endpoint when enabled.
/home/missy/missy-loops/missy/CLAUDE.md:202:| Audit log | `~/.missy/audit.jsonl` |
/home/missy/missy-loops/missy/CLAUDE.md:207:| Device registry | `~/.missy/devices.json` |
/home/missy/missy-loops/missy/CLAUDE.md:215:| Persona audit log | `~/.missy/persona_audit.jsonl` |
/home/missy/missy-loops/missy/CLAUDE.md:218:| Skills directory | `~/.missy/skills/` |
/home/missy/missy-loops/missy/CLAUDE.md:228:config_version: 2                    # schema version (auto-migrated on startup)
/home/missy/missy-loops/missy/CLAUDE.md:231:  default_deny: true
/home/missy/missy-loops/missy/CLAUDE.md:233:    - anthropic
/home/missy/missy-loops/missy/CLAUDE.md:238:  provider_allowed_hosts: []      # per-category overrides
/home/missy/missy-loops/missy/CLAUDE.md:239:  tool_allowed_hosts: []
/home/missy/missy-loops/missy/CLAUDE.md:249:      action: "deny"
/home/missy/missy-loops/missy/CLAUDE.md:256:  enabled: false
/home/missy/missy-loops/missy/CLAUDE.md:259:plugins:
/home/missy/missy-loops/missy/CLAUDE.md:260:  enabled: false
/home/missy/missy-loops/missy/CLAUDE.md:261:  allowed_plugins: []
/home/missy/missy-loops/missy/CLAUDE.md:263:providers:
/home/missy/missy-loops/missy/CLAUDE.md:264:  anthropic:
/home/missy/missy-loops/missy/CLAUDE.md:265:    name: anthropic
/home/missy/missy-loops/missy/CLAUDE.md:272:    enabled: true
/home/missy/missy-loops/missy/CLAUDE.md:275:  enabled: true
/home/missy/missy-loops/missy/CLAUDE.md:280:  enabled: false
/home/missy/missy-loops/missy/CLAUDE.md:286:  otel_enabled: false
/home/missy/missy-loops/missy/CLAUDE.md:293:  enabled: false
/home/missy/missy-loops/missy/CLAUDE.md:308:  # See missy/channels/discord/config.py for full schema
/home/missy/missy-loops/missy/CLAUDE.md:311:  enabled: false
/home/missy/missy-loops/missy/CLAUDE.md:318:  enabled: true
/home/missy/missy-loops/missy/CLAUDE.md:329:audit_log_path: "~/.missy/audit.jsonl"
/home/missy/missy-loops/missy/CLAUDE.md:338:missy setup --no-prompt             Non-interactive setup (--provider, --api-key-env, --model)
/home/missy/missy-loops/missy/CLAUDE.md:339:missy ask PROMPT                    Single-turn query (--provider, --session)
/home/missy/missy-loops/missy/CLAUDE.md:340:missy run                           Interactive REPL session (--provider)
/home/missy/missy-loops/missy/CLAUDE.md:341:missy providers list                List configured providers and availability
/home/missy/missy-loops/missy/CLAUDE.md:342:missy providers switch NAME         Switch active provider at runtime
/home/missy/missy-loops/missy/CLAUDE.md:343:missy skills                        List registered skills
/home/missy/missy-loops/missy/CLAUDE.md:344:missy skills scan                   Scan for SKILL.md files (--path)
/home/missy/missy-loops/missy/CLAUDE.md:345:missy presets list                  Show built-in network policy presets
/home/missy/missy-loops/missy/CLAUDE.md:346:missy plugins                       List plugins and their status
/home/missy/missy-loops/missy/CLAUDE.md:349:missy schedule add                  Add scheduled job (--name, --schedule, --task, --provider)
/home/missy/missy-loops/missy/CLAUDE.md:355:missy audit security                Show recent security events (--limit)
/home/missy/missy-loops/missy/CLAUDE.md:356:missy audit recent                  Show recent audit events (--limit, --category)
/home/missy/missy-loops/missy/CLAUDE.md:364:missy discord audit                 Show Discord-specific audit events (--limit)
/home/missy/missy-loops/missy/CLAUDE.md:371:missy approvals list                List pending approval requests
/home/missy/missy-loops/missy/CLAUDE.md:380:missy mcp pin NAME                  Pin tool manifest SHA-256 digest for verification
/home/missy/missy-loops/missy/CLAUDE.md:386:missy devices policy NODE_ID        Set node policy mode (--mode full|safe-chat|muted)
/home/missy/missy-loops/missy/CLAUDE.md:420:missy vision benchmark              Run vision capture performance benchmarks
/home/missy/missy-loops/missy/CLAUDE.md:432:missy persona log                   Show persona change audit log (--limit)
/home/missy/missy-loops/missy/CLAUDE.md:451:Desktop extra also needs: `sudo apt install python3-pyatspi` for accessibility tools.
/home/missy/missy-loops/missy/CLAUDE.md:455:Full docs site: **https://missylabs.github.io/** — 80+ pages covering getting started, configuration, security, architecture, CLI reference, channels, providers, extending, edge nodes, operations, and Leyline P2P network. Source at `/home/missy/missylabs.github.io/` (MkDocs Material, deployed via GitHub Actions).
/home/missy/missy-loops/missy/CLAUDE.md:459:Tests under `tests/` with subdirectories: `agent/`, `api/`, `channels/`, `cli/`, `config/`, `core/`, `gateway/`, `integration/`, `mcp/`, `memory/`, `observability/`, `plugins/`, `policy/`, `providers/`, `scheduler/`, `security/`, `skills/`, `tools/`, `unit/`, `vision/`. 480+ test files, 20,000+ tests, coverage threshold 90% (configured in `pyproject.toml`).
/home/missy/missy-loops/missy/CLAUDE.md:469:- **Server side** (this repo): `missy/channels/voice/server.py` is the authoritative WebSocket protocol implementation. `missy/channels/voice/registry.py` handles device registration, PBKDF2 token hashing, and node management. `missy/channels/voice/edge_client.py` is the original reference client (PipeWire, manual push-to-talk, local testing only).
/home/missy/missy-loops/missy/CLAUDE.md:508:- Edge node management commands (`missy devices list/status/pair/unpair/policy`) in this repo interact with the `DeviceRegistry` that missy-edge authenticates against.
/home/missy/missy-loops/missy/README.md:5:Missy is a production-grade agentic platform that runs entirely on your hardware. Default-deny network, filesystem sandboxing, shell whitelisting, encrypted vault, and structured audit logging — every capability is locked down until you explicitly allow it. Connect any AI provider. Deploy voice nodes throughout your home. Automate with scheduled jobs. Extend with tools, skills, and plugins.
/home/missy/missy-loops/missy/README.md:13:Most AI assistants trust the network, trust the model, and trust the plugins. Missy trusts nothing by default.
/home/missy/missy-loops/missy/README.md:18:- **No plugins** unless you approve them individually
/home/missy/missy-loops/missy/README.md:19:- **Every action** logged as structured JSONL with full audit trail
/home/missy/missy-loops/missy/README.md:20:- **Every audit event** signed with the agent's Ed25519 identity
/home/missy/missy-loops/missy/README.md:29:- **Multi-provider** — Anthropic (Claude), OpenAI (GPT), Ollama (local models) with automatic fallback and runtime hot-swap (`missy providers switch`)
/home/missy/missy-loops/missy/README.md:30:- **API key rotation** — multiple keys per provider, round-robin distribution
/home/missy/missy-loops/missy/README.md:32:- **Agentic runtime** — tool-augmented loops with done-criteria verification, learnings extraction, and self-tuning prompt patches
/home/missy/missy-loops/missy/README.md:33:- **AI Playbook** — auto-captures successful tool patterns, injects proven approaches into context, auto-promotes patterns with 3+ successes to skill proposals
/home/missy/missy-loops/missy/README.md:34:- **Attention system** — 5 brain-inspired subsystems (alerting, orienting, sustained, selective, executive) that track urgency, extract topics, maintain focus, and prioritize tools
/home/missy/missy-loops/missy/README.md:39:- **Interactive approval TUI** — real-time Rich terminal prompt for policy-denied operations (allow once / deny / allow always)
/home/missy/missy-loops/missy/README.md:40:- **Circuit breaker** — automatic backoff on provider failures (threshold=5, exponential to 300s)
/home/missy/missy-loops/missy/README.md:45:- **Failure tracking** — per-tool consecutive failure counts with automatic strategy rotation
/home/missy/missy-loops/missy/README.md:48:- **Code evolution** — self-evolving code modification engine with approval workflow and git-backed rollback
/home/missy/missy-loops/missy/README.md:49:- **Structured output** — Pydantic schema enforcement on LLM responses with automatic retry
/home/missy/missy-loops/missy/README.md:56:- **Multi-layer policy engine** — network (CIDR + domain + host), filesystem (per-path R/W), shell (command whitelist), L7 REST (HTTP method + path per host)
/home/missy/missy-loops/missy/README.md:57:- **Network presets** — `presets: ["anthropic", "github"]` auto-expands to correct hosts/domains/CIDRs
/home/missy/missy-loops/missy/README.md:58:- **Gateway enforcement** — all HTTP flows through `PolicyHTTPClient` with DNS rebinding protection, redirect blocking, scheme restrictions, interactive approval
/home/missy/missy-loops/missy/README.md:60:- **Prompt drift detection** — SHA-256 hashes system prompts, detects tampering between tool loop iterations
/home/missy/missy-loops/missy/README.md:63:- **Agent identity** — Ed25519 keypair at `~/.missy/identity.pem`, signs audit events, JWK export
/home/missy/missy-loops/missy/README.md:64:- **Trust scoring** — 0-1000 reliability tracking per tool/provider/MCP server with threshold warnings
/home/missy/missy-loops/missy/README.md:65:- **Container sandbox** — optional Docker-based isolation for tool execution (`--network=none`, memory/CPU limits)
/home/missy/missy-loops/missy/README.md:66:- **Landlock LSM** — Linux kernel-level filesystem enforcement via Landlock syscalls, complementing userspace policy
/home/missy/missy-loops/missy/README.md:67:- **Security scanner** — `missy security scan` audits installation for permission issues, config hygiene, exposed secrets
/home/missy/missy-loops/missy/README.md:68:- **MCP digest pinning** — SHA-256 verification of tool manifests; mismatches refuse to load
/home/missy/missy-loops/missy/README.md:72:- **CLI** — interactive REPL and single-shot queries with Rich formatting, capability modes (full/safe-chat/no-tools)
/home/missy/missy-loops/missy/README.md:73:- **Discord** — full Gateway WebSocket API, slash commands (`/ask`, `/status`, `/model`), DM allowlist, guild/role policies, image analysis
/home/missy/missy-loops/missy/README.md:75:- **Voice** — WebSocket server for edge nodes, faster-whisper STT, Piper TTS, device registry with PBKDF2 auth
/home/missy/missy-loops/missy/README.md:80:- **MCP servers** — connect external tool servers via `~/.missy/mcp.json`, auto-restart, digest pinning
/home/missy/missy-loops/missy/README.md:81:- **SKILL.md discovery** — scan directories for cross-agent portable skill definitions (`missy skills scan`)
/home/missy/missy-loops/missy/README.md:82:- **Tools, skills, plugins** — three extension tiers with increasing isolation and permission requirements
/home/missy/missy-loops/missy/README.md:85:- **Persona system** — YAML-backed agent identity/tone/style with backup, rollback, and audit logging
/home/missy/missy-loops/missy/README.md:93:- **Multi-provider** — Anthropic/OpenAI/Ollama image message formatting
/home/missy/missy-loops/missy/README.md:95:- **CLI tools** — `missy vision capture|inspect|review|doctor|health|benchmark|validate|memory`
/home/missy/missy-loops/missy/README.md:98:- **Browser tools** — Playwright-based Firefox automation (`pip install -e ".[desktop]"`)
/home/missy/missy-loops/missy/README.md:99:- **X11 tools** — window management and application launching
/home/missy/missy-loops/missy/README.md:100:- **Accessibility** — AT-SPI toolkit integration for GUI interaction
/home/missy/missy-loops/missy/README.md:103:- **Config presets** — `presets: ["anthropic", "github"]` replaces manual host lists
/home/missy/missy-loops/missy/README.md:106:- **Non-interactive setup** — `missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt`
/home/missy/missy-loops/missy/README.md:109:- **Audit logger** — every policy decision, provider call, and tool execution as JSONL, signed by agent identity
/home/missy/missy-loops/missy/README.md:110:- **Application logs** — rotating Python/provider diagnostics at `~/.missy/missy.log` (`missy logs tail`)
/home/missy/missy-loops/missy/README.md:130:The setup wizard walks you through configuring API keys, providers, network policy, and workspace paths. Once complete:
/home/missy/missy-loops/missy/README.md:152:missy setup --provider anthropic --api-key-env ANTHROPIC_API_KEY --no-prompt
/home/missy/missy-loops/missy/README.md:165:pip install -e ".[dev]"           # pytest, ruff, mypy, hypothesis, coverage tools
/home/missy/missy-loops/missy/README.md:182:(network,     (Anthropic, OpenAI,        (built-in tools,
/home/missy/missy-loops/missy/README.md:183: filesystem,   Ollama + fallback)         skills, plugins,
/home/missy/missy-loops/missy/README.md:199: Network ──► AuditLogger (signed) ──► ~/.missy/audit.jsonl
/home/missy/missy-loops/missy/README.md:205:Every outbound request — from providers, tools, plugins, MCP servers, Discord — passes through `PolicyHTTPClient`. No exceptions.
/home/missy/missy-loops/missy/README.md:217:  default_deny: true
/home/missy/missy-loops/missy/README.md:219:    - anthropic                  # auto-expands to api.anthropic.com + anthropic.com
/home/missy/missy-loops/missy/README.md:234:  enabled: false
/home/missy/missy-loops/missy/README.md:237:providers:
/home/missy/missy-loops/missy/README.md:238:  anthropic:
/home/missy/missy-loops/missy/README.md:239:    name: anthropic
/home/missy/missy-loops/missy/README.md:246:  enabled: false
/home/missy/missy-loops/missy/README.md:267:missy setup --no-prompt             # Non-interactive (--provider, --api-key-env, --model)
/home/missy/missy-loops/missy/README.md:268:missy ask PROMPT                    # Single-turn query (--provider, --session, --mode)
/home/missy/missy-loops/missy/README.md:269:missy run                           # Interactive REPL (--provider, --mode)
/home/missy/missy-loops/missy/README.md:270:missy providers list                # List providers and availability
/home/missy/missy-loops/missy/README.md:271:missy providers switch NAME         # Hot-swap active provider
/home/missy/missy-loops/missy/README.md:273:missy plugins                       # List plugins and their status
/home/missy/missy-loops/missy/README.md:279:# Security & audit
/home/missy/missy-loops/missy/README.md:280:missy audit recent                  # Recent events (--limit, --category)
/home/missy/missy-loops/missy/README.md:281:missy audit security                # Policy violations
/home/missy/missy-loops/missy/README.md:283:missy approvals list                # Pending human-in-the-loop approval requests
/home/missy/missy-loops/missy/README.md:297:missy discord status | probe | register-commands | audit
/home/missy/missy-loops/missy/README.md:301:missy devices list | pair | unpair | status | policy
/home/missy/missy-loops/missy/README.md:303:# MCP & skills
/home/missy/missy-loops/missy/README.md:305:missy skills                        # List registered skills
/home/missy/missy-loops/missy/README.md:306:missy skills scan                   # Discover SKILL.md files
/home/missy/missy-loops/missy/README.md:310:missy vision health | benchmark | validate | memory
/home/missy/missy-loops/missy/README.md:352:missy devices policy ID --mode full|safe-chat|muted
/home/missy/missy-loops/missy/README.md:362:python3 -m pytest tests/ -k "test_policy" -v         # Filter by name
/home/missy/missy-loops/missy/README.md:379:| [Configuration](https://missylabs.github.io/configuration/) | 7 | Full YAML reference, network/fs/shell policy, presets, providers |
/home/missy/missy-loops/missy/README.md:382:| [CLI Reference](https://missylabs.github.io/cli/) | 20 | Every command group, including gateway, discord, approvals, patches, sandbox, sessions |
/home/missy/missy-loops/missy/README.md:384:| [Providers](https://missylabs.github.io/providers/) | 5 | Anthropic, OpenAI, Ollama, runtime switching |
/home/missy/missy-loops/missy/README.md:385:| [Extending](https://missylabs.github.io/extending/) | 4 | Tools, plugins, MCP servers, SKILL.md |
/home/missy/missy-loops/missy/README.md:392:Developer-facing references in [`docs/`](docs/) — architecture, implementation deep-dives, persistence schema, module map.
/home/missy/missy-loops/missy/README.md:401:│                    attention, progress, approval, persona, behavior, hatching,
/home/missy/missy-loops/missy/README.md:412:├── policy/          Network, filesystem, shell, REST L7 policy engines + presets
/home/missy/missy-loops/missy/README.md:413:├── providers/       Anthropic, OpenAI, Ollama + registry with fallback & hot-swap
/home/missy/missy-loops/missy/README.md:414:├── scheduler/       APScheduler integration, human schedule parser
/home/missy/missy-loops/missy/README.md:417:├── skills/          Skill registry + SKILL.md discovery
/home/missy/missy-loops/missy/README.md:418:├── plugins/         Security-gated external plugin loader
/home/missy/missy-loops/missy/README.md:419:├── tools/           Built-in tools + registry (18+ tools)
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:9:| A1 | Streaming subscription state machine | tested | Core module and focused tests added; lightly wired to `AgentRuntime.run_stream()`. Needs channel/tool-loop integration. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:10:| A2 | Layered tool policy pipeline | hardened | `missy/policy/tool_policy_pipeline.py` is wired into `AgentRuntime._get_tools()` for runtime capability profiles and config-backed provider/global/agent/sandbox/subagent policy surfaces. Channel/group policy sources remain future hardening. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:13:| A5 | Auth profile cooldown + fallback | not_started | Provider registry/rate limiter work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:14:| A6 | Per-provider tool schema normalization | not_started | Schema adapter work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:18:| A10 | Sub-agent depth + child caps | not_started | SubAgentRunner persistence/tool policy work remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:29:| H_C | Persistent personal memory | not_started | Memory schema/CLI remains. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:31:| H_E | Genuine disagreement and pushback | not_started | Prompt fragment and audit logging remain. |
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:41:- Updated `AgentRuntime.run_stream()` to pass provider chunks through `AgentSubscription`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:45:- Session 2 added the A2 layered tool policy pipeline with profile bundles, group expansion, glob matching, inline `-tool` denies, `alsoAllow`, fail-warning unknown allowlists, and structured trace records.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:46:- Session 2 wired `AgentRuntime._get_tools()` to resolve tools through the pipeline and record `_last_tool_policy_decision` for audit/debugging.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:47:- Session 2 added `tests/policy/test_tool_policy_pipeline.py` and runtime coverage for policy decisions in `tests/agent/test_runtime_streaming.py`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:48:- Session 3 added config parsing for `tools.*`, `tools.byProvider`, `tools.byModel`, `tools.groups`, `agents.<id>.tools`, `agents.<id>.subagents.tools`, and `sandbox.tools`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:49:- Session 3 added `build_configured_tool_policy_layers()` and `collect_tool_policy_groups()` so runtime policy resolution now consumes YAML-backed provider/global/agent/sandbox/subagent layers.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:50:- Session 3 routed parsed tool policies into CLI-created runtimes for ask/run/gateway/API paths and documented the YAML surface in `docs/configuration.md`.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:51:- Session 3 added config, policy-pipeline, and runtime tests for those surfaces, then verified the full test suite and full-repo ruff.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:55:1. Harden A1 by routing provider/tool-loop stream events through `AgentSubscription` where Missy's providers expose stream events, not only the simple `run_stream()` path.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:56:2. Add the A7 `BlockChunker` and connect it to A1 flush points so pre-tool text can be delivered through Discord/CLI/Web in order.
/home/missy/missy-loops/missy/HUMANIZE_STATUS.md:58:4. Add channel/group policy sources on top of the A2 pipeline when Discord/CLI/Web channel identity context is available.
/home/missy/missy-loops/missy/fixes.md:14:### 1.1 🔴 Token bucket is never pre-charged — TPM limiting is effectively disabled
/home/missy/missy-loops/missy/fixes.md:15:- **Where:** `missy/providers/anthropic_provider.py:142` and `:266`; `missy/providers/base.py:227-238`.
/home/missy/missy-loops/missy/fixes.md:16:- **Problem:** `complete()` / `complete_with_tools()` call `self._acquire_rate_limit()`
/home/missy/missy-loops/missy/fixes.md:22:  blow past provider tokens-per-minute limits and trigger 429s (or spend) under load.
/home/missy/missy-loops/missy/fixes.md:26:  Do the same in `openai_provider.py` and `ollama_provider.py`.
/home/missy/missy-loops/missy/fixes.md:29:- **Where:** `missy/providers/registry.py:236` — `instance.rate_limiter = RateLimiter()`.
/home/missy/missy-loops/missy/fixes.md:30:- **Problem:** Every provider gets the default `RateLimiter(60 rpm, 100_000 tpm)`. There is
/home/missy/missy-loops/missy/fixes.md:40:- **Where:** `missy/providers/rate_limiter.py:149-164`.
/home/missy/missy-loops/missy/fixes.md:42:  The RateLimiter is shared across threads (async runtime, scheduler, Discord). A synchronous
/home/missy/missy-loops/missy/fixes.md:45:- **Impact:** One provider's 429 can freeze the whole agent for up to `max_wait` (30s).
/home/missy/missy-loops/missy/fixes.md:51:- **Where:** `anthropic_provider.py:158-159`, `openai_provider.py:141`.
/home/missy/missy-loops/missy/fixes.md:63:- **Where:** `missy/scheduler/manager.py:332-355` (`_run_job` active-hours gate) combined with
/home/missy/missy-loops/missy/fixes.md:75:- **Where:** `missy/scheduler/jobs.py:108-122` (`should_retry` ignores `error`), field defined at `:57`.
/home/missy/missy-loops/missy/fixes.md:76:- **Problem:** `ScheduledJob.retry_on` defaults to `["network", "provider_error"]` and is
/home/missy/missy-loops/missy/fixes.md:87:- **Problem:** The config schema advertises `max_jobs` (0 = unlimited) but `SchedulerManager`
/home/missy/missy-loops/missy/fixes.md:107:- **Fix:** Update `next_run` from `self._scheduler.get_job(...)` in a `finally` block.
/home/missy/missy-loops/missy/fixes.md:113:### 3.1 🟠 API session registry and memory-store turn counts drift
/home/missy/missy-loops/missy/fixes.md:120:- **Impact:** Confusing client behavior; history is orphaned relative to the session lifecycle.
/home/missy/missy-loops/missy/fixes.md:121:- **Fix:** Back the session registry with the memory store's `sessions` table (there's already a
/home/missy/missy-loops/missy/fixes.md:141:- **Fix:** Re-raise (or surface via an audit event + return status) on persistence failure for
/home/missy/missy-loops/missy/fixes.md:161:- **Problem:** When `sandbox.enabled: true` but Docker isn't accessible, `get_sandbox` logs a
/home/missy/missy-loops/missy/fixes.md:163:  isolation** (just a scrubbed env). An operator who enabled sandboxing for safety silently loses
/home/missy/missy-loops/missy/fixes.md:166:  host despite `enabled: true`.
/home/missy/missy-loops/missy/fixes.md:167:- **Fix:** Add a `require_isolation`/`strict` flag. When sandbox is enabled and Docker is
/home/missy/missy-loops/missy/fixes.md:173:- **Problem:** The `SandboxConfig` promises `network_disabled`, `memory_limit`, `cpu_limit`,
/home/missy/missy-loops/missy/fixes.md:182:### 4.3 🟠 REST-policy L7 rules are only enforced inside `PolicyHTTPClient`
/home/missy/missy-loops/missy/fixes.md:183:- **Where:** `missy/gateway/client.py:355-357` calls `_check_rest_policy` only when a `method` is
/home/missy/missy-loops/missy/fixes.md:186:  (`anthropic_provider.py:84`), bypassing `PolicyHTTPClient` entirely. So network policy + REST
/home/missy/missy-loops/missy/fixes.md:187:  policy + response-size caps + audit events do **not** apply to the largest volume of outbound
/home/missy/missy-loops/missy/fixes.md:189:  HTTP" is not true for provider calls.
/home/missy/missy-loops/missy/fixes.md:190:- **Impact:** Egress controls and audit are blind to provider traffic; a compromised base_url or
/home/missy/missy-loops/missy/fixes.md:191:  proxy setting can exfiltrate without policy review.
/home/missy/missy-loops/missy/fixes.md:192:- **Fix:** Pass a policy-aware `http_client` into the provider SDKs (both Anthropic and OpenAI SDKs
/home/missy/missy-loops/missy/fixes.md:201:- **Fix:** On injection detection, emit a `security` audit event and either skip the run or require
/home/missy/missy-loops/missy/fixes.md:202:  an approval gate (`ApprovalGate`) before executing.
/home/missy/missy-loops/missy/fixes.md:224:### 4.7 🟡 API `/chat` mutates shared runtime provider globally per-request
/home/missy/missy-loops/missy/fixes.md:225:- **Where:** `api/server.py:518-522` — `runtime.switch_provider(provider_override)`.
/home/missy/missy-loops/missy/fixes.md:226:- **Problem:** `switch_provider` changes the runtime's active provider process-wide. Two concurrent
/home/missy/missy-loops/missy/fixes.md:227:  `/chat` requests with different `provider` values race, and one request's override leaks into the
/home/missy/missy-loops/missy/fixes.md:229:- **Fix:** Pass the provider per-run (as a `run()` argument) rather than mutating shared state, or
/home/missy/missy-loops/missy/fixes.md:230:  serialize/scope provider selection per session.
/home/missy/missy-loops/missy/fixes.md:252:- **Where:** `missy/providers/registry.py:257-280`.
/home/missy/missy-loops/missy/fixes.md:258:### 5.4 🟡 `_extract_all_programs` blanket-rejects `<<` but the tool advertises redirection
/home/missy/missy-loops/missy/fixes.md:259:- **Where:** `missy/policy/shell.py:135` (`_SUBSHELL_MARKERS` includes `"<<"`) vs.
/home/missy/missy-loops/missy/fixes.md:270:### 6.1 🟢 Feature: Provider-tier & rate-limit configuration surface (`providers.*.limits`)
/home/missy/missy-loops/missy/fixes.md:274:  providers:
/home/missy/missy-loops/missy/fixes.md:275:    anthropic:
/home/missy/missy-loops/missy/fixes.md:282:  `missy providers limits` CLI to display current buckets (`request_capacity`/`token_capacity`),
/home/missy/missy-loops/missy/fixes.md:283:  and estimate prompt tokens before `acquire()` so TPM is actually honored. Emit an audit event
/home/missy/missy-loops/missy/fixes.md:287:### 6.2 🟢 Feature: Egress audit report (`missy audit egress`) + universal gateway routing
/home/missy/missy-loops/missy/fixes.md:288:- **Motivation:** Item 4.3 — provider SDKs bypass the gateway, so there's no single egress ledger.
/home/missy/missy-loops/missy/fixes.md:290:  1. Inject a policy-aware `httpx` client into every provider SDK so all outbound HTTP flows
/home/missy/missy-loops/missy/fixes.md:291:     through `PolicyHTTPClient` (network + REST policy + size cap + `network_request` audit events).
```
