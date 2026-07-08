# AUDIT_CONNECTIVITY

- Timestamp: 2026-07-08 13:29:18 EDT

Expected OpenAI connectivity posture:

- Default-deny network where practical.
- OpenAI provider access limited to configured provider hosts, normally
  `api.openai.com`, plus explicitly configured OpenAI-compatible `base_url`
  hosts.
- OpenAI SDK traffic should use Missy's policy-aware HTTP client when the SDK
  accepts the injected client.
- Unsafe image URL schemes are stripped before provider calls to reduce
  provider-side fetch abuse and local path leakage.
- Future Responses API migration must retain exact endpoint allowlisting and
  must not bypass provider network policy.
