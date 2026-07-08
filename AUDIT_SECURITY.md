# AUDIT_SECURITY

- Timestamp: 2026-07-08 10:57:27 EDT

## Current Security Posture

- Web TUI access remains authenticated with the configured API key.
- Browser sessions use hardened cookies and per-session CSRF tokens for unsafe
  API calls and logout.
- Safe controls fail closed for unknown controls, invalid targets, missing
  dependencies, missing confirmation, unavailable providers, unknown scheduler
  jobs, and invalid scheduler job state.
- Scheduler pause/resume controls mutate only through the attached scheduler
  manager and do not create a parallel scheduler path.
- Control allow/deny results are audited as structured `web.control` events and
  pass through existing redaction.
- Existing response security headers remain in place: no-store, nosniff, frame
  deny, and CSP for the Web console.

## Checked This Session

- Provider control behavior remains covered by tests.
- Scheduler control listing and mutation behavior is covered by focused API
  tests.
- Confirmation-denied scheduler controls are audited as deny events.
- Successful scheduler pause/resume controls are audited as allow events.
- Full pytest and full-repo ruff passed.

## Still Important

- Add tool, channel, and experimental-feature controls behind the same
  confirmation, policy, CSRF, and audit model.
- Add the run/session viewer without exposing unredacted prompt, tool, provider,
  or secret material.
- Add browser smoke/visual coverage when a browser test dependency is available.
