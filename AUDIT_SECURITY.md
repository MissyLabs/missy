# AUDIT_SECURITY

- Timestamp: 2026-07-07 18:49:30 EDT

## Security Posture

- Discord diagnostics report token presence only and do not print token values.
- Discord voice tools now participate in Discord capability-mode visibility and still execute through the normal `ToolRegistry` permission and audit path.
- Runtime voice bindings remain account/guild scoped and fail closed on missing or ambiguous scope.
- `missy discord diagnostics` surfaces policy readiness for `discord.com` and `gateway.discord.gg` so operators can see default-deny network gaps before probing live APIs.
- Recent Discord audit rendering is bounded by `--limit` and truncates detail output.

## Relevant Security Controls Preserved

- Secrets come from environment, vault references, or protected local files and remain redacted.
- Discord REST calls use `PolicyHTTPClient`.
- Bot-loop prevention, own-message filtering, DM policy, guild policy, allowlists, and require-mention filtering remain in the channel access-control path.
- Discord voice manager bindings are cleared on startup failure and shutdown.
- Privileged voice actions remain policy-gated tools rather than direct channel shortcuts.

## Remaining Security Work

- Harden accepted Discord image attachments with size limits, MIME validation, image-dimension checks, filename hygiene, URL fetch policy, and richer audit detail.
- Add live Gateway heartbeat/resume diagnostics when a safe service status channel exists.
- Continue verifying Discord slash command registration failures produce actionable audit/operator output.
