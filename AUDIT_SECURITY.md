# AUDIT_SECURITY

- Timestamp: 2026-07-08 10:19:19 EDT

## Current posture

- Web TUI entry still requires API-key login or authenticated API credentials.
- Browser sessions use HttpOnly SameSite cookies and CSRF tokens for unsafe API
  methods.
- New operator controls are server-enforced, not UI-trusted:
  - unknown controls deny
  - invalid provider target names deny
  - missing exact confirmation denies
  - unregistered providers deny
  - unavailable providers deny
  - allow and deny outcomes emit `web.control` audit events
- Control details are redacted through the existing audit redaction helper
  before publication.
- The Controls panel only invokes the same authenticated JSON API; it does not
  grant client-side authority.

## Remaining security work

- Add policy-backed enable/disable controls for tools, jobs, channels, and
  experimental features.
- Keep extending server-side authorization around any future privileged control.
- Add visual/browser smoke coverage for XSS-resistant rendering and responsive
  layout once browser test dependencies are available.
- Continue hardening run/session viewer output against tool-output injection
  before streaming transcript support lands.
