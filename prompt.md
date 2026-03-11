You are running inside a persistent build loop environment controlled by a shell script.

The repository state persists between runs. Your job is to continue progress across sessions until the project is fully complete.

LOOP OPERATING MODEL

- The project directory is: ~/git/missy
- You are invoked repeatedly in separate sessions.
- A session may end before the project is complete.
- The next session must inspect the repository and continue from where the previous one stopped.
- Do not restart from scratch.
- Do not ask the user questions unless absolutely necessary.
- Prioritize real implementation over discussion.

MANDATORY SESSION BEHAVIOR

At the start of every session you must:
1. Inspect the repository.
2. Read BUILD_STATUS.md if it exists.
3. Read LAST_SESSION_SUMMARY.md if it exists.
4. Read any test, audit, and results files that already exist.
5. Determine the highest-value next tasks.
6. Continue implementation immediately.

You MUST maintain these project artifacts:

- ~/git/missy/BUILD_STATUS.md
- ~/git/missy/BUILD_RESULTS.md
- ~/git/missy/AUDIT_SECURITY.md
- ~/git/missy/AUDIT_CONNECTIVITY.md
- ~/git/missy/TEST_RESULTS.md
- ~/git/missy/TEST_EDGE_CASES.md
- ~/git/missy/LAST_SESSION_SUMMARY.md

BUILD_STATUS.md must always include:
- completed steps
- architecture status
- remaining tasks
- blockers
- next actions for the next session

LAST_SESSION_SUMMARY.md must always include:
- what was implemented this session
- what remains
- what should be done first next session

COMPLETION RULE

Only create ~/git/missy/COMPLETE.md when ALL of the following are true:
- all planned phases are implemented
- the project is runnable
- core CLI works
- provider abstraction works
- scheduler works
- policy engine works
- audit logging works
- docs exist
- tests have been run
- security artifacts exist
- implementation documentation exists
- Discord integration exists and is documented

SESSION PRIORITIES

In order of priority:
1. write real code
2. harden security
3. write tests
4. run tests
5. fix failures
6. update docs
7. update BUILD_STATUS.md and LAST_SESSION_SUMMARY.md

GIT DISCIPLINE

- Prefer many small commits.
- Commit after meaningful progress.
- Keep the repository in a resumable state.

ANTI-STAGNATION RULES

If you are stuck:
- inspect failing files
- simplify the design
- implement the secure MVP path
- leave notes in BUILD_STATUS.md
- keep moving forward

Do not spend the session only planning. Produce concrete code and file changes.

==================================================
MISSY PROJECT BUILD
==================================================

You are Claude Code. Build a brand-new local-first Python project called Missy.

Missy is a security-first, self-hosted, agentic assistant for Linux, inspired by the strengths of:
- https://github.com/nagisanzenin/skyclaw
- https://github.com/openclaw/openclaw

This must be a clean-room Python redesign, not a mechanical port.

PRIMARY GOAL

Create Missy as a production-grade Python agent platform that:
1. Runs locally on a Linux host inside my environment.
2. Uses Python throughout as the primary implementation language.
3. Preserves the important end-user capabilities and operator ergonomics of OpenClaw-style assistants and SkyClaw-style autonomous execution loops.
4. Has an extremely explicit and hardened security model.
5. Supports multiple LLM providers:
   - OpenAI
   - Anthropic
   - Ollama
6. Supports:
   - tools
   - skills
   - plugins
   - scheduling / cron-like recurring tasks
   - memory
   - session handling
   - policy enforcement
   - network egress restriction at the core of the runtime
7. Includes a production-ready Discord integration inspired by OpenClaw's channel model.
8. Is fully rebranded as Missy across code, docs, binaries, config, service files, and examples.

This should be a serious local system, not a toy demo.

MANDATORY DESIGN INTENT

Missy should combine these conceptual strengths:

From SkyClaw, inherit the spirit of:
- an autonomous agent loop
- think -> act -> verify -> retry patterns
- task decomposition
- resilience / self-healing
- provider fallback concepts
- skills as structured capabilities
- high observability
- local execution
- strong failure handling

From OpenClaw, inherit the spirit of:
- local-first assistant architecture
- a gateway/control-plane concept
- channels / sessions / inbox routing concepts
- user-facing assistant workflows
- skills/plugin extensibility
- scheduling / cron support
- operational CLI
- security guardrails for inbound access
- agent isolation and scoped execution

Do NOT blindly mirror their file layout or code. This is a Python-native redesign.

CRITICAL SECURITY REQUIREMENTS

Missy must be built with security as a first-class architectural concern, not added later.

Design and implement an explicit security profile with the following properties:

1. Default-deny network egress
   - The runtime must assume outbound network is blocked unless explicitly allowed.
   - The whitelist mechanism must be part of the core runtime policy layer.
   - Support CIDR allowlists, especially:
     - 10.0.0.0/8
   - Support exact hostnames.
   - Support wildcard hostname rules, for example:
     - *.github.com
   - Support separate allowlists for:
     - LLM providers
     - tool/plugin requests
     - skills
     - fetch/web actions
     - Discord gateway / REST endpoints
   - Enforce policy before any outbound request is made.

2. Policy-driven execution
   - All tools, plugins, and skills must declare:
     - required permissions
     - network requirements
     - filesystem requirements
     - risk level
   - The runtime must validate those requirements against policy before execution.

3. Filesystem sandboxing
   - Restrict writable paths to a configured workspace.
   - Prevent path traversal.
   - Resolve and validate symlinks before access.
   - Separate:
     - config directory
     - runtime state directory
     - logs directory
     - workspace directory
     - secrets directory

4. Secrets handling
   - Never hardcode secrets.
   - Read secrets from environment variables or a dedicated local secrets file with strict permissions.
   - Redact secrets from logs.
   - Prevent tool output from echoing secrets into chat responses when possible.
   - Discord bot tokens must be sourced securely from environment variables or a local secrets store, never committed plaintext.

5. Command execution hardening
   - All shell execution must be opt-in and policy-gated.
   - Prefer subprocess invocation without shell where possible.
   - If shell is allowed, log it and constrain it.
   - Support a configurable denylist and allowlist for commands.
   - Support timeout, max output, and working-directory constraints.

6. Plugin/skill isolation
   - Skills and plugins must not receive unrestricted ambient authority.
   - They should execute through a controlled capability API.
   - Prefer a manifest-based model where plugins request permissions explicitly.
   - Make it possible to disable all third-party plugins globally.

7. Inbound trust model
   - Treat all inbound user input as untrusted.
   - Support approval/pairing concepts for untrusted senders or channels.
   - Support explicit allowlists per channel/session source.
   - Support read-only / no-tools / safe-chat-only modes.

8. Auditability
   - Every privileged action must be logged with:
     - who/what requested it
     - which policy allowed it
     - timestamp
     - session/task ID
     - outcome
   - Include a human-readable security audit command.

9. Secure defaults
   - Missy must boot in the safest practical mode.
   - Dangerous features must require explicit enablement.

10. Threat model
   - Document the threat model clearly:
     - prompt injection
     - malicious skills/plugins
     - data exfiltration
     - SSRF
     - local privilege misuse
     - secrets leakage
     - tool abuse
     - scheduler abuse
     - Discord bot loops / spoofing / impersonation
     - channel impersonation
     - remote content poisoning

ARCHITECTURE REQUIREMENTS

Design Missy as a maintainable multi-module Python application.

Suggested top-level architecture:

- missy/
  - core/
  - agent/
  - gateway/
  - providers/
  - tools/
  - skills/
  - plugins/
  - scheduler/
  - memory/
  - policy/
  - channels/
    - discord/
  - cli/
  - config/
  - observability/
  - security/
  - tests/

Required subsystems:
1. Gateway / control plane
2. Agent runtime
3. Provider abstraction
4. Tool framework
5. Skills system
6. Plugin system
7. Scheduler
8. Memory
9. Config system
10. Observability
11. Discord channel integration

DISCORD INTEGRATION REQUIREMENTS

Research-informed design goal: match the spirit of OpenClaw's Discord channel integration while implementing a Python-native version.

Missy must include a first-party Discord integration with these capabilities:

1. Connectivity model
   - Official Discord gateway support for inbound events.
   - Discord REST support for outbound messaging, reactions, thread replies, and slash command registration where needed.
   - Health checks, reconnect logic, heartbeat handling, and session resumption or clean re-login behavior.

2. Supported surfaces
   - DMs
   - guild channels
   - optional thread-aware reply handling
   - slash commands
   - interaction components if practical in MVP, otherwise scaffold them clearly

3. Session/routing behavior
   - DMs should default to a pairing / approval workflow for unknown users.
   - Guild channels should support allowlist-based routing and per-guild/per-channel policy.
   - Support require-mention behavior per guild or channel.
   - Support ignore-other-mentions behavior so messages mentioning other users/roles but not the bot can be dropped.
   - Route inbound Discord messages into normal Missy sessions instead of a special ad hoc path.

4. Access control
   - Support DM policies:
     - pairing
     - allowlist
     - open
     - disabled
   - Support guild policy / group policy with allowlists.
   - Support explicit allowFrom / users / roles / channels style configuration.
   - Support safe-chat-only, no-tools, and fully-capable modes per Discord source.

5. Bot loop prevention
   - By default, ignore bot-authored messages.
   - If bot messages are allowed, support a safer mentions-only mode for bot messages.
   - Own-bot filtering must be specific to the bound bot account, not every configured bot account in the runtime.
   - Add explicit anti-loop protections and audit events.

6. Multi-account support
   - Support Discord accounts / account IDs so multiple bot identities can exist if needed.
   - Bind sessions/channels to a specific bot account.
   - Prevent cross-account own-message filtering bugs.

7. UX behaviors
   - Support acknowledgement reactions while processing, with layered config resolution.
   - Support typing indicators or a comparable in-progress UX signal.
   - Support reply-to mode options.
   - Preserve message readability for multiline replies.

8. Configuration
   - Support token from environment or secure local secret reference.
   - Do not require plaintext token in committed config.
   - Support per-account config overrides.
   - Support intent-related validation and startup diagnostics.

9. Required Discord setup guidance in docs
   - Create a Discord application and bot.
   - Enable Message Content Intent.
   - Enable Server Members Intent when role allowlists or name-to-ID resolution are needed.
   - Presence intent should be optional.
   - Document OAuth scopes, permissions, invite flow, and troubleshooting.
   - Document restart/reconnect steps after changing intents.

10. Slash commands and interactions
   - Include native slash command support in the design.
   - Slash commands should run in isolated command sessions while still routing back to the main conversation/session when appropriate.
   - Include at least a minimal command catalog for MVP, such as:
     - /ask
     - /status
     - /model
     - /help
   - If interactive components are not fully implemented in MVP, scaffold the interfaces and document them.

11. Diagnostics
   - Include Discord-specific doctor / audit / status commands.
   - Include permission probes, routing diagnostics, and health reporting.
   - Include logs for:
     - gateway connected / disconnected
     - intents mismatch
     - pairing waits
     - allowlist denials
     - require-mention filtering
     - bot-message filtering
     - reaction/typing failures
     - slash command registration failures

12. Security-specific Discord constraints
   - The Discord integration must still respect the global network allowlist.
   - Discord endpoints must be explicitly allowlisted.
   - Discord attachments and media handling must be policy-gated and sandboxed.
   - Discord integration must not bypass approval gates, filesystem policy, shell policy, plugin policy, or provider policy.

IMPLEMENTATION DOCUMENTATION REQUIREMENTS

Missy must include full implementation documentation, not just user docs.

Required documentation set:
- README.md
- ARCHITECTURE.md
- SECURITY.md
- OPERATIONS.md
- CONFIG_REFERENCE.md
- PROVIDERS.md
- DISCORD.md
- SCHEDULER.md
- SKILLS_AND_PLUGINS.md
- MEMORY_AND_PERSISTENCE.md
- TESTING.md
- TROUBLESHOOTING.md
- docs/implementation/
  - module-map.md
  - agent-loop.md
  - policy-engine.md
  - network-client.md
  - discord-channel.md
  - provider-abstraction.md
  - scheduler-execution.md
  - audit-events.md
  - persistence-schema.md
  - manifest-schema.md

Documentation requirements:
- explain architecture decisions
- explain module boundaries and data flow
- document config schema and examples
- document threat model and hardening posture
- document operational lifecycle
- document how Discord integration works end-to-end
- document scheduler execution path
- document policy enforcement path
- document tests and how to run them
- document migration strategy for persisted data
- document plugin and skill manifests with versioning and permissions
- document audit event schema and examples
- include sequence diagrams or ASCII flow diagrams where useful

IMPLEMENTATION REQUIREMENTS

Use Python as the primary language.

Preferred implementation guidance:
- Python 3.12+
- typed code throughout
- pydantic for schemas/config if useful
- FastAPI only if needed for a local gateway API
- Typer or Click for CLI
- asyncio where appropriate
- pytest for tests
- ruff + mypy
- SQLite for local metadata/state if appropriate
- APScheduler or an equivalent for scheduling if it fits the design
- httpx for outbound HTTP with policy enforcement hooks
- a Python Discord library only if it can be cleanly wrapped behind Missy's channel abstraction; otherwise implement a minimal direct client carefully
- avoid overengineering, but do not build a toy

Where reasonable, prefer:
- capability-based interfaces
- explicit schemas
- no hidden magic
- security over convenience

FEATURE PARITY INTENT

Missy should include a meaningful first version of:
- local assistant runtime
- CLI for setup and operation
- session/task execution
- model/provider selection
- provider fallback support
- tools
- skills
- plugins
- memory
- scheduling
- audit logs
- security audit command
- local config wizard or setup helper
- service-mode operation on Linux (systemd-friendly)
- channel-ready architecture
- Discord channel integration

REBRANDING REQUIREMENTS

Everything must be named Missy, not OpenClaw or SkyClaw.

Use:
- project name: Missy
- Python package namespace: missy
- CLI command: missy
- systemd service names: missy-gateway.service or similar
- config files: missy.toml / missy.yaml as appropriate
- docs, comments, examples, banners, tests: all Missy-branded

DELIVERABLES

Phase 1:
- Analyze the two upstream repos conceptually
- Extract the important architectural ideas to preserve
- Produce a clean-room Missy architecture plan
- Produce a threat model and security model
- Produce a build plan
- Produce a Discord integration architecture plan

Phase 2:
- Scaffold the full Python project
- Create config models, policy models, provider interfaces, gateway skeleton, agent loop skeleton, tool registry, skills loader, plugin manager, scheduler manager, memory layer, audit logger, Discord channel abstractions, and CLI

Phase 3:
- Implement a functional MVP that can:
  - start locally
  - accept a task from CLI
  - call one of the supported LLM providers
  - execute policy-approved tools
  - load skills
  - schedule a recurring task
  - persist state locally
  - emit audit logs
  - connect to Discord
  - receive approved Discord DMs
  - receive approved guild mentions
  - send Discord replies safely

Phase 4:
- Harden the implementation
- Add tests
- Add docs
- Add systemd examples
- Add an example security policy
- Add example whitelist rules including:
  - 10.0.0.0/8
  - *.github.com
  - exact provider endpoints
  - exact Discord endpoints
- Add a security audit command
- Add Discord diagnostics and doctor commands

REQUIRED OUTPUT STYLE

Do not just brainstorm. Actually create the project.

For each step:
1. State what you are implementing.
2. Create or modify files.
3. Explain the reasoning briefly.
4. Keep moving until there is a coherent runnable Missy project.

If information is missing, make sensible engineering decisions and document them.
Do not stall on unnecessary questions.
Prefer delivering a secure MVP over waiting for perfect completeness.

EXPLICIT FEATURE DETAILS

Missy must support these provider concepts:
- OpenAI provider
- Anthropic provider
- Ollama provider
- a common chat/completion abstraction
- optional provider failover sequence
- provider-level timeout and retry policy

Missy must support these policy concepts:
- network allowlist / deny-by-default
- host wildcard support
- CIDR support
- per-tool and per-plugin permission checks
- shell execution controls
- filesystem path restrictions
- scheduling permission controls
- Discord endpoint allowlisting and attachment/media policy checks

Missy must support these scheduler concepts:
- parse "every X minutes"
- parse "daily at HH:MM"
- parse "weekly on Monday at HH:MM"
- store jobs durably
- allow list/show/delete/pause/resume
- run jobs through normal agent execution pipeline

Missy must support these skills/plugin concepts:
- local bundled skills
- workspace-local skills
- plugin manifests
- clear permission declarations
- safe loading behavior
- versioned interfaces where practical

Missy must support these CLI concepts:
- missy init
- missy run
- missy ask
- missy gateway start
- missy gateway status
- missy schedule add
- missy schedule list
- missy schedule pause
- missy schedule resume
- missy schedule remove
- missy audit security
- missy audit discord
- missy doctor
- missy providers list
- missy skills list
- missy plugins list
- missy discord register-commands
- missy discord status
- missy discord probe

ADDITIONAL REQUIREMENTS

- Prefer first-party local integrations over browser automation whenever possible.
- Every network-capable module must route outbound requests through a single central policy-enforced network client so that no code pathan bypass the whitelist accidentally.
- Include example tests proving that blocked domains, blocked CIDRs, forbidden filesystem paths, and forbidden shell commands are denied by policy.
- Create a SECURITY.md with threat model, trust boundaries, safe deployment posture, secrets handling guidance, plugin/skill risk model, scheduler abuse considerations, Discord channel abuse considerations, and operator hardening steps.
- Create an OPERATIONS.md with deployment guidance for an internal Linux host, systemd setup, log locations, upgrade flow, backup guidance, incident response basics, Discord deployment notes, and safe rollback guidance.
- Include a secure example configuration showing:
  - default-deny network
  - 10.0.0.0/8 allowed
  - *.github.com allowed
  - exact OpenAI / Anthropic / Ollama endpoints
  - exact Discord endpoints
  - restricted writable workspace paths
  - disabled third-party plugins by default
- Include tests that prove scheduled jobs are also subject to normal policy checks and cannot bypass network, shell, filesystem, plugin, or Discord restrictions.
- Include an explicit provider policy model so each LLM provider can be individually enabled, disabled, rate-limited, timed out, and bound to approved endpoints only.
- Include support for safe-chat-only mode, no-tools mode, and fully capable mode, configurable per channel or session source.
- Include an approval model for higher-risk actions so the runtime can require explicit confirmation before privileged tool or plugin execution.
- Include structured audit events for:
  - network allow / deny
  - shell allow / deny
  - filesystem allow / deny
  - plugin allow / deny
  - scheduler job creation / execution / failure
  - provider invocation / failure / fallback
  - Discord connect / disconnect / deny / reply / filter / pairing / command
- Include migration-safe local persistence design so config, memory, schedules, and audit history can evolve cleanly over time.
- Include a documented plugin/skill manifest schema with versioning, compatibility rules, and permission declarations.
- Include example bundled first-party skills that demonstrate safe local automation patterns without broad unrestricted access.
- Include implementation docs for every major module, not just top-level user guides.
- Include Discord configuration examples for:
  - DM pairing mode
  - guild mention-only mode
  - allowlisted support channel
  - multi-account setup
  - safer allowBots="mentions" mode

NON-GOALS / CONSTRAINTS

- Do not make this dependent on cloud-only infrastructure.
- Do not assume outbound internet should be generally open.
- Do not make plugin execution implicitly trusted.
- Do not prioritize flashy UI over secure local operation.
- Do not leave security as TODOs when they can be concretely implemented now.
- Do not mirror upstream code mechanically; reinterpret the architecture into Python.

FINAL EXPECTATION

I want a serious, secure, Python-based local assistant platform named Missy, inspired by SkyClaw and OpenClaw, with:
- local Linux operation
- OpenAI / Anthropic / Ollama support
- strong tool/skill/plugin architecture
- scheduling support
- explicit network whitelist enforcement
- strong auditability
- secure defaults
- full implementation documentation
- Discord integration comparable in spirit to OpenClaw's Discord channel support

Start by creating the architecture, threat model, repository structure, and initial project files, then continue implementing the MVP.
