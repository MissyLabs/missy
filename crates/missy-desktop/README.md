# missy-desktop

Native Rust X11 (`x11rb`), typed AT-SPI D-Bus (`zbus`) and Chromium CDP browser
automation. CDP works with an explicitly provisioned Playwright Chromium endpoint;
there is no Python wrapper, arbitrary JavaScript tool, subprocess browser driver,
camera, implicit DISPLAY, session bus discovery, or implicit personal tab selection.

## Public integration

`DesktopTools::new(DesktopConfig, PolicyHttpClient, Arc<dyn DesktopApproval>)`
implements `ToolExecutor`. All defaults deny connections. Every connection endpoint
is explicit and constructor validation performs no I/O. `DesktopApproval` is a
required async trait with `authorize(&ToolContext, &ToolCall) -> Result<()>` and
`configuration_fingerprint() -> String`. Wire it to durable exact-call/scope approval.
Do not supply a permissive implementation without an enforcing outer guard.

`approve_all` defaults true. Input/navigation/AT-SPI actions and screenshots always
require approval even when that flag is false. Screenshots additionally require
`enable_screenshots`; mutations require `enable_input`. Reads enumerate only
allowlisted windows/applications. Screenshot definitions have `side_effect = true`
for privacy-sensitive dispatch/journaling. Ordinary enumeration/text reads are false.
No returned page text, accessibility names or image bytes are trusted instructions.

The parent can enforce **all** tools through its compositional approval guard. When
doing so it must protect every definition, including reads, and bind the exact call
and scope before delegation. `execute` repeats `preflight` deliberately; approval
must be idempotent, durable and valid for both checks.

## X11 isolation and authentication

Configure an isolated server's absolute Unix socket and an explicit mode-0600 file
containing exactly 16 raw MIT-MAGIC-COOKIE-1 cookie bytes. This is **not** an
Xauthority file. Neither DISPLAY nor XAUTHORITY/HOME is consulted. The separate
Xserver must use matching server authentication. Do not point at an active desktop.
Exact class AND title allowlists apply; names are not cryptographic application
identities, and malicious clients on the same server can spoof them. X11 is not a
per-window security boundary. `isolated_session = true` acknowledges this requirement.

Window enumeration returns title, WM_CLASS, geometry and depth. Screenshots are
bounded raw `application/x-x11-zpixmap` image artifacts with base64, dimensions,
depth, byte order, bits-per-pixel, padding and visual masks, not mislabelled PNGs.
Root screenshots are forbidden. XTest accepts `key`/`button` press or release,
window-relative `move` (`x`,`y`), `drag` (`x`,`y`,`to_x`,`to_y`,`code` button 1..3),
and `type_text` (`text`, 1..256 printable ASCII bytes). All require an already-focused
allowlisted client or its focused descendant; the executor never steals focus.
Coordinates are geometry-bounded and translated to the configured screen root,
not caller-selected root coordinates. Occluded pointer endpoints are denied.
Focus/allowlist/geometry checks and dispatch occur under a short server grab on the
**isolated server only**. A drag is an immediate press/move/release, not a timed
gesture. Typing resolves existing Latin keysyms and Shift_L, rejects active keys,
modifiers/locks/buttons, and never modifies keymaps or installs virtual keyboards.
Unsupported text/layouts fail before typing. Global shortcuts and other X11 client behavior
are why the isolated Xserver is mandatory. An independent five-second connection
watchdog shuts down stalled sockets, bounding background blocking work. A timeout
or cancellation can race a dispatched input and must be treated as UNKNOWN, never
automatically replayed. Raw key/button press/release still requires two separately
approved calls; drag/type_text contain bounded complete pairs in a single call.

## Accessibility

Configure an explicit `unix:path=/...` accessibility bus, isolated from the real
session. Exact application AND window names are required; ambiguous selections
fail closed. Query/action references are pinned to the application's unique bus
owner and must remain descendants of the chosen window. Name/role/action lists,
depth, node count and strings are bounded. Actions require a path discovered within
the bounded scoped tree, never arbitrary D-Bus service/method dispatch. The parent
deadline bounds the async bus operation. AT-SPI names are untrusted and spoofable;
the bus isolation is a real requirement, not cosmetic configuration.

`desktop_accessibility_get_text` uses typed `org.a11y.atspi.Text.GetText` with
explicit `start`/`end` satisfying `0 <= start <= end <= 4096`. Returned UTF-8 text
is byte-bounded as well. `desktop_accessibility_set_value` uses typed
`org.a11y.atspi.EditableText.SetTextContents` with `value` at most 8192 UTF-8 bytes
and no NUL. It is a text operation, not numeric Value or arbitrary D-Bus dispatch.
Both take exact `application`, `window`, and descendant `path`, pin the unique bus
owner, and re-resolve selection and bounded descendant membership immediately
before dispatch. Set requires `enable_input` and outer per-call approval. AT-SPI
offers no atomic cross-process tree transaction: a malicious application can still
reparent its objects after validation. Do not treat this protocol as a security
boundary against hostile peers; the isolated bus/session remains mandatory.

## Browser boundary

Configure a canonical HTTP(S) CDP service origin, exact page `target_id` and exact
canonical allowed HTTP(S) page origins. Include the CDP service origin in the
`PolicyHttpClient` network allowlist; enable private network access only when
needed. Discovery uses policy HTTP and WebSocket connections use its pinned DNS/TLS
path. Returned WebSocket endpoints must match the configured authority and exact
`/devtools/page/<target>` path. IDs are monotonically increasing and unknown,
duplicate or malformed reply IDs fail closed. Events and output are bounded.

Only fixed, bounded scripts exist: location, title/text, CSS selector click,
input/textarea fill, and selector visibility polling. `browser_get_url` reads the
pinned target. `browser_wait` takes `selector`, `state` (`visible` or `hidden`),
and `timeout_ms` (1..15000); it polls at 100ms, defines visible by nonzero layout
dimensions plus computed display/visibility, and treats absent elements as hidden.
The outer desktop deadline can shorten this bound. This is visibility, not
interactability or an opacity/occlusion guarantee. `browser_evaluate` accepts only
`expression_id` = `url`, `title`, or `ready_state`; arbitrary JavaScript, functions,
and legacy evaluate arguments are intentionally rejected. `browser_close` requires
input opt-in/approval and sends `Page.close` solely through the configured page's
socket. It accepts no URL/target argument and never terminates a browser process.
Arguments are JSON encoded, never interpolated as source.
Every renderer script checks origin in its own turn. Requests (including redirects)
are intercepted with Fetch, frame navigation events are checked, and final location
is checked before releasing data. Navigation currently requires settling at the
exact requested URL; same-origin canonicalizing redirects may fail conservatively.
On an error, the connection drops; Fetch interception may leave requests paused.
Discard/reset the isolated page after a failure rather than reconnecting a personal
browser or assuming no navigation occurred. Successful calls disable interception
and close their temporary socket.

**CDP is remote code execution authority over the browser host.** Client URL policy
is NOT a browser-host firewall. Service workers, already-active requests, WebSockets,
other targets, a hostile CDP server and renderer behavior cannot be contained by
this adapter. Run the browser in a dedicated container/network namespace with
outbound authorities independently denied unless explicitly allowed. Never expose
its CDP port publicly. `isolated_browser_network = true` is a required operator
attestation that this independent boundary exists; it does not create one. Unknown
schemes/origins are denied, including about:blank/data/file URLs. Provision the
target initially at an allowed URL in that isolated browser.

## Verification boundaries

Tests use a fresh authenticated Xvfb selected by `-displayfd`, a private D-Bus mock
and fake HTTP/WebSocket CDP peers. They never inherit DISPLAY or create windows on
an existing desktop. XTest pointer move, drag, and Latin typing are exercised only
on the fresh authenticated Xvfb test window, including actual delivered key/button
events, unchanged focus on denial, coordinate bounds, and released modifiers.
The private D-Bus mock verifies typed text read/edit and rejects out-of-tree and
reparented paths. Fixed CDP transactions cover wait, timeouts, safe expression IDs,
get_url, exact-page close, and origin/redirect checks.
Passing these is protocol/isolation acceptance, not a claim
that an actual desktop accessibility stack or production browser was exercised.
