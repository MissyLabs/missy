# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 13:56:58 EDT

Expected connectivity posture:

- Default-deny network where practical.
- Native OpenAI provider traffic should target exact OpenAI API endpoints
  through the policy-aware SDK HTTP client.
- OpenAI-compatible `base_url` traffic remains explicit and must be allowed by
  configured host policy.
- Responses routing must not broaden outbound access; it reuses the existing
  OpenAI SDK client construction.
- Exact local Web TUI bind address and origin policy.
- Exact benchmark and non-OpenAI provider endpoints where enabled.
- No unreviewed broad outbound access.
