# Native remote MCP

`HttpMcpTools::connect(PolicyHttpClient, Vec<HttpMcpConfig>).await` discovers
explicitly configured HTTP servers and returns a `missy_core::ToolExecutor`.
Construct it before publishing the tool catalog. Names are exactly
`<configured-name>__<remote-tool-name>`. Every definition is effectful, including
tools carrying `readOnlyHint`; callers must apply their normal durable approval
and side-effect journaling boundary.

`HttpMcpConfig` has `name`, `endpoint`, exact `origin`, required
`manifest_sha256`, `timeout_seconds`, `transport` (`streamable_http` default or
`legacy_sse`), `credential_headers_env`, `max_response_bytes`,
`max_argument_bytes`, `protocol_version`, and optional `message_endpoint`.
Defaults are protocol `2025-03-26`, 30 seconds per complete discovery/execution,
1 MiB response budget and 64 KiB argument budget. Config deserialization denies
unknown fields. Use `manifest_digest(&tools_list_result)` on an independently
reviewed catalog to calculate the required SHA256 pin, never auto-trust a first
network response. Object keys sort recursively; array order is preserved.

## Transport and trust boundary

All connections go through `PolicyHttpClient::request_stream`, inheriting exact
origin allowlisting, all-answer DNS validation and pinned resolution, private
network policy, no proxy, no redirects, bounded headers and total/idle deadlines.
The MCP config additionally binds the endpoint to its declared exact origin.
Credentials come only from the explicitly named environment variables and only
in Authorization, X-Api-Key or Api-Key HTTP headers. No inline credentials,
cookies, routing headers, URL userinfo, URL query, fragments or alternate URL
schemes are accepted. Credentials are snapshotted at connect, are not Debug
printed, and are never forwarded to a server-advertised arbitrary endpoint.

The opaque configuration fingerprint hashes the entire ordered config, the
entire policy, and digests of the snapshotted credential values. Thus a restart
with rotated credentials invalidates prior execution identity; environment names
alone are not enough. No raw credential appears in the returned fingerprint.
Reconnect to apply rotation. This fingerprint is an execution-identity digest,
not a credential store or public authentication verifier.

Streamable HTTP supports JSON responses and a complete JSON-RPC event from a
genuinely open SSE body without waiting for EOF. Initialization requires an
agreed protocol and server-issued session ID, sent on all following requests;
unexpected replacement IDs and mismatched JSON-RPC IDs fail closed. A fresh
initialized session is used for each execution. `tools/list` is verified before
and after `tools/call` in that same session.

Legacy SSE maintains its GET body while posting requests to the operator's
explicitly approved `message_endpoint`. Relative endpoint events resolve against
the configured GET URL, but their normalized absolute URL must exactly equal
that configured endpoint and origin before any authenticated POST. The persistent
stream has cumulative byte and complete-operation deadline limits. **Legacy
servers requiring dynamically generated query-bearing message URLs are not
supported**. They fail closed rather than silently extending credential scope.

JSON-RPC batching, pagination, server-initiated requests/notifications, resumable
event replay, session migration and unbounded long-lived client sessions are not
implemented. Unsupported messages fail closed. The SSE parser accepts LF/CRLF
lines and comments, rejects invalid UTF-8, truncation and oversized event streams.

## Arguments and outcome semantics

Input schemas must be closed objects. A strict supported JSON Schema 2020-12
keyword set is validated by `jsonschema`; external and local references, dialect
overrides and unsupported keywords are rejected. The dependency's HTTP/file
retrieval features are disabled. Arguments have depth, node and serialized-byte
limits, and are validated at preflight and again at execute.

Local gate failures are `ToolDispatchRefused`, before transport/effect dispatch.
After beginning an execution transport, any timeout, cancellation, protocol
failure, pin drift, remote error or oversized output becomes the typed
`HttpMcpOutcomeUnknown`. This intentionally errs toward uncertainty even if an
initialization failed before the call. Never automatically retry this error.

Disposable mock HTTP tests exercise JSON, persistent legacy SSE, open Streamable
SSE, session and RPC correlation, pin changes, schema/name preflight, malformed
SSE, environment header scoping, endpoint injection, output limits and uncertain
cancellation/timeouts. No production endpoint or credential is needed.
