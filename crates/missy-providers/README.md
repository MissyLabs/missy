# missy-providers

Real async non-streaming `missy_core::Provider` implementations for OpenAI
Chat Completions, Anthropic Messages, and Ollama Chat. No Python, vendor SDK,
direct HTTP client, fallback default endpoint, or live credential is used.

## Public API

- `ProviderPool::new(Vec<ProviderConfig>, PolicyConfig) -> anyhow::Result<Self>`
- `ProviderPool::with_client(Vec<ProviderConfig>, PolicyHttpClient) -> Result<Self>`
- `HttpProvider::new(ProviderConfig, PolicyHttpClient) -> Result<Self>`
- `OpenAiProvider`, `AnthropicProvider`, `OllamaProvider`: same adapter constructor.
- `ProviderPool::costs_known() -> bool` checks configured prices, not future usage.
- `ProviderPool::require_known_prices() -> Result<()>` for finite-budget startup.
- `Provider::configuration_fingerprint()` hashes ordered configuration and policy,
  including endpoint, model, pricing and credential reference name. It never reads
  the credential value. Credential rotation does not invalidate a checkpoint.

For OpenAI and Anthropic set `base_url` to the API root including `/v1`.
For Ollama use the origin. Endpoint suffixes are `/chat/completions`, `/messages`,
and `/api/chat` respectively. Only an explicit allowed origin can be contacted;
local endpoints additionally require `allow_private_network: true`.

Credentials resolve only `api_key_env` names, per request. No implicit variables,
inline secrets, URL userinfo, query tokens, proxy or redirects are accepted.
Errors never contain resolved credentials, response bodies or raw transport errors.

## Bounds and validation

Policy HTTP transport bounds bodies to 4 MiB. Providers also bound request bodies
to 4 MiB, configured timeouts to 1..600 seconds, calls to 128 per response, and
output token usage to the configured maximum. Cancellation drops the in-flight
request, and is never retried on another provider. Requests use `stream:false`.
Length limits and unsupported finish reasons return errors, never partial success.

Assistant is the only permitted output role. Unknown input roles, misplaced system
messages, ambiguous/duplicate JSON object keys, missing usage, duplicate, invalid
or historically reused tool IDs, unresolved/duplicate results, unknown tools,
non-object/malformed arguments, and JSON Schema violations fail closed. JSON
Schema uses `jsonschema` with default features disabled; nonlocal references are
rejected, so schema validation cannot bypass the policy HTTP gateway.

Ollama does not require wire IDs. Missing IDs are assigned collision-resistant
UUID-based local IDs; supplied IDs are still validated. Tool results are translated
to Ollama's `tool_name` field using validated historical IDs.

## Cost and failover limitations

Both prices must be explicitly supplied in **microUSD per million tokens**.
`Some(0)` declares a free model; absent rates set `Usage.cost_known=false`.
Integer cost arithmetic uses u128 intermediate values, checked overflow, and
rounds the combined input/output cost up to one microUSD. Missing/malformed token
usage fails, including for nominally free models.

Cached-input pricing is not supported: nonzero OpenAI or Anthropic cached token
usage fails rather than silently mispricing the response. Unsupported multimodal,
thinking, streaming, vendor CLI/OAuth and response blocks also fail explicitly.

Failover is sequential, at most one request per eligible configured provider.
Failures immediately open exponential cooldown circuits (500 ms initial, capped
at 30 seconds). Calls during cooldown skip the unhealthy provider rather than
sleeping, so cancellation cannot be trapped in a backoff sleep. A later call
probes after cooldown. No tool is ever executed or replayed by this crate.

**A failed remote request may still have been billed.** After any failed attempt,
an eventual fallback completion sets `cost_known=false`, even if its own usage is
valid. All-failed requests return an error, and may have unknown billed usage.
Runtime finite-budget enforcement must reject unknown costs before tool execution.
These post-response checks do **not** establish a strict pre-request monetary cap:
that requires prompt-token reservation, worst-case output reservation and provider
billing reconciliation. Do not describe this crate as guaranteeing a hard spend cap.

`cargo test -p missy-providers` runs isolated real local TCP fixtures. It never
loads credentials or contacts public provider endpoints.
