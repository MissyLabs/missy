# Native watchdog

The watchdog is disabled unless an operator supplies a trusted, private `extensions.watchdog_config` file with `enabled: true`. It checks only explicitly configured probes. It never restarts or repairs anything. Database checks use `PRAGMA quick_check` against the runtime's existing store. HTTP checks use the same outbound policy gateway as other native traffic, with no redirect or allowlist exemption. Configure the exact origin in the existing policy and explicitly allow private networking if needed. URLs must not contain credentials, queries, or fragments. A 2xx response is healthy. The body is not recorded. Raw TCP port and process probes are not supported.

Example extension file:

```yaml
enabled: true
scope: {tenant: operator, user: operator, channel: operations, session: watchdog}
poll_seconds: 30
failure_threshold: 2
probes:
  - {kind: database, name: database}
  - {kind: http, name: local_api, url: 'http://127.0.0.1:8080/health'}
```

Every probe records bounded health state and transitions in the private `scheduler.sqlite3`; response payloads, URLs, and transport errors are not recorded in those health records. An optional `trigger` is an existing `TriggerConfig` with `kind: {kind: watchdog}`. It fires only after a probe reaches the configured consecutive-failure threshold and only with explicit `enabled` and `operator_consent`, a bounded configured prompt and cooldown. Such a trigger invokes the existing scoped runtime, may incur provider charges, and retains the runtime's existing permissions, approvals, and budgets. Omit it for observation without inference. Recovery resets consecutive failures. Changes require the normal gateway restart and trigger consent binding rules.
