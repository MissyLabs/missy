# missy-policy

Linux userspace capability policy. Not an OS sandbox. Default configuration grants
no filesystem roots, network origins, shell or MCP capabilities. No live-state
defaults and no Python runtime dependency.

## Public API

- `Policy::new(PolicyConfig) -> anyhow::Result<Policy>`
- `Policy::{read_file, write_file}` accept absolute `impl AsRef<Path>` paths.
  Reads return bounded bytes; writes take `&[u8]` and atomically replace a file.
- `Policy::list_files(&Path) -> Result<Vec<String>>` returns sorted immediate
  names, bounded to 4096 entries/256 KiB. No recursive traversal.
- `Policy::check_command(&[String])` requires shell opt-in and an exact argv entry.
- `Policy::check_mcp_command(&[String])` independently requires MCP opt-in and an
  exact argv entry. Neither method executes anything.
- `PolicyHttpClient::new(PolicyConfig) -> Result<Self>` or `from_policy(Policy)`.
- `PolicyHttpClient::with_timeout(Duration) -> Result<Self>` consumes the client;
  bounds are 1 to 600 seconds, default 30 seconds.
- `PolicyHttpClient::with_chunk_timeout(Duration) -> Result<Self>` configures the
  streaming idle-read timeout, independently bounded to 1..600 seconds (default
  30). Buffered requests retain their existing total timeout behavior.
- `request(method: &str, url: &str, headers: Vec<(String,String)>,
  body: Option<Vec<u8>>) -> Result<HttpResponse>` is async.
- `request_stream(method: &str, url: &str, headers: Vec<(String,String)>,
  body: Option<Vec<u8>>) -> Result<PolicyHttpStream>` is async. The stream exposes
  public `status: u16`, `headers: Vec<(String,String)>`, and async
  `chunk(&mut self) -> Result<Option<Vec<u8>>>`. It does not expose reqwest.
- `get(url: &str) -> Result<HttpResponse>` is async.
- `request_json` has the same parameters as `request` except an optional JSON
  Value body, returning a Value. It rejects non-2xx with a status-only error.
- `connect_websocket(url: &str, headers: Vec<(String,String)>)
  -> Result<PolicyWebSocket>` is async. `wss` maps to the configured `https`
  origin; `ws` maps to `http`. It returns a bounded tungstenite stream over the
  pinned TCP connection and verified TLS where applicable.
- Both policy/client expose `config() -> &PolicyConfig` for fingerprints.

`HttpResponse` has public `status: u16`, `headers: Vec<(String,String)>` and
`body: Vec<u8>`. Authentication response headers are filtered, but response bodies
and other headers remain untrusted application data, not safe logging material.
Dropping an in-flight request/handshake future cancels its transport. The caller
owns WebSocket heartbeat/read deadlines, message processing and stream closure.

Streaming HTTP uses the same request validation, DNS pinning and response header
filtering as buffered HTTP. Its absolute deadline starts before DNS/connection
setup and includes time between reads. Every chunk read is bounded by the earlier
of that deadline and the idle-read timeout. No reqwest body-wide timeout is used
for streaming. For legacy SSE long polls, callers may set both timeouts up to 600
seconds. Heartbeats cannot extend the total deadline. All delivered chunks count
toward the hard 4 MiB cumulative response limit, even for indefinite/chunked SSE.
EOF, errors and cancelled pending chunk reads close the stream permanently;
subsequent reads return `None`. Dropping the stream also closes its non-pooled
response. A caller retaining a stream without reading must still drop it when
done; no background task reads or buffers events. `Debug` reports counts/status
only, never headers, URL, body bytes or the internal transport.

## Enforced boundaries

Filesystem roots must exist at construction and are canonicalized and opened
component-by-component with `openat(O_NOFOLLOW|O_DIRECTORY)`. Operations use root
directory capabilities and never reopen resolved paths. Relative paths and any
literal `.`/`..` components are rejected. Symlinks on operation paths, nonregular
files and multiply linked files are denied. Writes use an exclusive random temp
entry in the pinned parent directory, mode 0600 and effective-user ownership,
file fsync, atomic rename and directory fsync. Existing foreign-owned write
destinations are denied. Parent directories are not implicitly created. File
limits come from configuration, with a 64 MiB hard maximum.

Network allowlist entries are exact scheme, canonical hostname and effective
port origins, without paths, queries, fragments, wildcards or credentials. Both
HTTP and WebSocket route through the same DNS validation. Every returned DNS
answer must pass before any connection. All public/private mixtures and mapped
IPv6 are rejected. Private, loopback, link-local and special-use ranges require
an explicit private-network flag in addition to the exact origin grant.
Conservative IPv6 checks also reject transition/documentation ranges by default.
HTTP uses a fresh no-proxy client with DNS overrides pinning the validated
snapshot, retaining the original hostname for TLS and Host. WebSocket opens TCP
only to validated addresses before TLS/handshake. Neither follows redirects.

HTTP request and response bodies are bounded to 4 MiB, including chunked
responses. Headers, URL length, methods and routing/framing headers are bounded or
restricted. Errors omit raw URLs, upstream response bodies, headers and transport
error chains so credentials do not escape through diagnostic formatting.

## Limits, not marketing

Same-UID adversaries able to move already-opened directories out of a root,
modify executable images, inspect process memory, or change mounts are not
contained. Use distinct service users and an OS sandbox for those threats.
Exact allowlisting `/bin/sh -c ...` or another interpreter explicitly authorizes
that program's behavior; this crate never performs interpolation or inserts a
shell, but it cannot make an explicitly allowed interpreter harmless. Tools must
spawn only approved argv and implement process-group cleanup on cancellation.
Atomic writes replace inodes; they are not suitable for SQLite database updates
or generic read-modify-write locking. The shell/MCP process and its descendants
are not network-contained by this policy: allowed processes retain OS authority.

Tests use temporary directories and opt-in loopback listeners only:
`cargo test -p missy-policy`.
