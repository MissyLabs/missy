# CLI command reference

Moved verbatim from `CLAUDE.md`. `missy --help` and `missy <group> --help` are authoritative.


```
missy init                          Create default config at ~/.missy/config.yaml
missy setup                         Interactive setup wizard (API keys, OAuth)
missy setup --no-prompt             Non-interactive setup (--provider, --api-key-env, --model)
missy ask PROMPT                    Single-turn query (--provider, --session)
missy run                           Interactive REPL session (--provider)
missy providers list                List configured providers and availability
missy providers switch NAME         Switch active provider (a running `missy gateway start` daemon's live default if reachable via --host/--port, else a local-only, non-persisted fallback)
missy providers auth [NAME]         Refresh OpenAI API-key or OAuth/Codex credentials (--method api-key|oauth, --account NAME for an additional OpenAI OAuth account)
missy providers oauth-accounts      List every stored OpenAI OAuth account (email/account-id/token-expiry, no secrets) and whether round-robin balancing is active
missy skills                        List registered skills
missy skills scan                   Scan for SKILL.md files (--path)
missy presets list                  Show built-in network policy presets
missy plugins                       List plugins and their status
missy doctor                        System health check

missy schedule add                  Add scheduled job (--name, --schedule, --task, --provider, --capability-mode; defaults to safe-chat, not full)
missy schedule list                 List all scheduled jobs
missy schedule pause JOB_ID         Pause a job
missy schedule resume JOB_ID        Resume a paused job
missy schedule remove JOB_ID        Remove a job

missy audit security                Show recent security events (--limit)
missy audit recent                  Show recent audit events (--limit, --category)

missy gateway start                 Start the gateway server (--host, --port); also launches the Web TUI (see api: config)
missy gateway status                Show gateway status

missy discord status                Show Discord channel configuration
missy discord probe                 Test Discord bot token connectivity
missy discord register-commands     Register slash commands (--guild-id, --global)
missy discord audit                 Show Discord-specific audit events (--limit)

missy vault set KEY VALUE           Store an encrypted secret
missy vault get KEY                 Retrieve a secret
missy vault list                    List stored key names
missy vault delete KEY              Delete a secret

missy approvals list                List pending approval requests from a running gateway (--host, --port, --api-key)
missy approvals approve ID          Approve a pending request by ID
missy approvals deny ID             Deny a pending request by ID

missy patches list                  List prompt patches
missy patches approve PATCH_ID      Approve a proposed patch (--force if the injection scanner flags it)
missy patches reject PATCH_ID       Reject a proposed patch

missy mcp list                      List configured MCP servers
missy mcp add NAME                  Connect to an MCP server (--command, --url)
missy mcp remove NAME               Disconnect and remove an MCP server
missy mcp pin NAME                  Pin tool manifest SHA-256 digest for verification

missy devices list                  List all registered edge nodes
missy devices pair                  Approve a pending pairing request (--node-id, --mode full|safe-chat|muted)
missy devices unpair NODE_ID        Remove a paired edge node
missy devices status                Show online/offline status of all nodes
missy devices policy NODE_ID        Set node policy mode (--mode full|safe-chat|muted)

missy voice status                  Show voice channel config and STT/TTS status
missy voice test NODE_ID            Test TTS synthesis for an edge node (--text)

missy config backups                List config backups
missy config diff                   Diff current config vs latest backup
missy config rollback               Restore config from latest backup
missy config plan                   Show what changed since last backup

missy sandbox status                Show Docker container sandbox config and availability

missy security scan                 Audit installation for security issues (permissions, config, secrets)

missy api start                     Start the Agent-as-a-Service REST API (--host, --port)
missy api status                    Show API server configuration and status

missy evolve list                   List code evolution proposals
missy evolve show EVOLUTION_ID      Show details of a code evolution proposal
missy evolve approve EVOLUTION_ID   Approve a proposed code evolution
missy evolve reject EVOLUTION_ID    Reject a proposed code evolution
missy evolve apply EVOLUTION_ID     Apply an approved code evolution
missy evolve rollback EVOLUTION_ID  Roll back an applied code evolution

missy sessions list                 List active sessions
missy sessions rename SESSION       Rename a session
missy sessions cleanup              Delete old conversation history (--older-than, --dry-run)

missy vision devices                Enumerate and diagnose available USB cameras
missy vision capture                Capture frames (--device, --output, --count, --width, --height)
missy vision inspect                Run visual quality assessment (--file, --screenshot, --device)
missy vision review                 LLM-powered visual analysis (--mode general|puzzle|painting|inspection, --file, --context)
missy vision doctor                 Run vision subsystem diagnostics
missy vision health                 Show capture statistics and device health status
missy vision benchmark              Run vision capture performance benchmarks
missy vision validate               Validate vision pipeline end-to-end
missy vision memory                 Show vision scene memory status and history

missy hatch                         First-run bootstrap wizard (--non-interactive)

missy persona show                  Display current persona configuration
missy persona edit                  Edit persona fields (--name, --tone, --identity)
missy persona reset                 Reset persona to factory defaults
missy persona backups               List available persona backups
missy persona diff                  Show diff between current persona and latest backup
missy persona rollback              Restore persona from latest backup
missy persona log                   Show persona change audit log (--limit)

missy cost                          Show cost tracking config and budget status (--session)
missy recover                       List incomplete checkpoints from previous sessions (--abandon-all, --resume ID, --provider)
```
