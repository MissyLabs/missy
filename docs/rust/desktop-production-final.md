# Final native desktop status and screen-reading parity

Repository conversion only. Acceptance uses a newly created authenticated Xvfb
server, mock endpoints, synthetic pixel matrices and generated text PNGs. No
existing desktop, account, display authentication or application is controlled.

## Typed approval-gated status

`desktop_status` takes an empty object and rejects unknown fields before approval.
It is catalogued only with at least one explicitly configured desktop backend.
It always requires desktop approval, including when `approve_all` is false.
The outer CLI exact durable guard additionally protects all desktop definitions.

`DesktopStatusReport` includes capture/input/vision/OCR/launch capabilities, each
backend's configuration/isolation attestation, and endpoint health. Only the
configured isolated X11 endpoint is probed, via authenticated handshake. There
is no enumeration, screenshot, focus query or input. Browser and accessibility
are `not_probed`, not assumed healthy; unconfigured endpoints are `not_configured`.
X11 failures are `unavailable` with a fixed bounded diagnostic, never raw errors,
paths, cookies or arbitrary endpoint response text.

Isolation is operator-attested, not independently proved by the handshake.
Reachability does not establish meaningful capture content or input safety.
The report says `ambient_discovery=false` and `input_dispatched=false`. No ambient
DISPLAY/XAUTHORITY/DBus/CDP discovery occurs.

## Blank-frame refusal before OCR or vision

`x11_read_screen` converts native pixels to RGB, excluding padding/alpha, then
rejects uniform/near-uniform frames before PNG output and before OCR or billable
vision. The legacy threshold is retained: every RGB channel's standard deviation
below 1.5. A fixed `blank_capture` error says content is unknown and no OCR/vision
was invoked. It does not pretend that lock/off state was observed. Malformed
pixels fail closed. Raw `desktop_screenshot` remains a raw artifact, not OCR.

## Actual OCR coordinates, not model guesses

`ScreenOcr` is an injection seam with no desktop-to-specialty dependency. The CLI
`desktop_ocr::build` attaches `SpecialtyTools::ocr_png` only when specialties,
documents and operator-pinned Tesseract are explicitly configured. Parent main
composes it with DesktopTools. OCR-only reads work without vision. Governed
ScreenInterpreter output remains separately labelled `analysis`.

The native backend sanitizes original PNG bytes without resizing, enforces
pixel/byte budgets, rechecks executable SHA-256 on every call, and invokes fixed
English Tesseract `--psm 3 tsv` through existing child-only Landlock execution.
There is no executable discovery, shell, arbitrary flags or model OCR. Runtime
read roots are explicit. TCP denial is the actual sandbox guarantee, not total
network isolation. Sandbox cancellation/timeout/output overflow/future drop
kill and reap the owned subprocess group.

Staging must be canonical, private (mode excludes group/other), and inside both
policy read/write roots. Each call has a private temporary directory; image
publication uses policy.write_file, and temporary staging is removed on drop.
Byte OCR accepts no caller-selected file path. Parent capture approval protects
disclosure.

Strict TSV parsing yields actual word labels, confidence, bounding boxes and
centers. Coordinate space is `native_window_pixels`, local to the captured
client window, not display/root or resized model coordinates. Confidence below
40, empty words and zero-size boxes are omitted. Invalid/out-of-image/overflow
coordinates, nonfinite confidence and malformed TSV are rejected. Limits:
1 MiB TSV, 4096 rows, 128 anchors, 128 characters per label. The CLI scans joined
word text for credential-shaped spans and conservatively redacts all labels if
found. OCR is imperfect and all output remains untrusted.

Availability distinguishes `not_configured`, `available` (possibly zero words),
and `unavailable`. Backend/TSV failure returns no anchors and a fixed error code,
not raw exception text or fabricated labels. Backend identity is fingerprinted.

## Focused validation

```text
cargo test -p missy-desktop -p missy-specialty -j 2
cargo test -p missy-cli --test desktop_ocr_e2e -j 2
```

Synthetic matrices cover black/white/solid-colored/near-uniform rejection,
empty/invalid data and useful variation. Fresh Xvfb asserts interpreter and OCR
invocation counts stay zero for a uniform window, then draws known variation
and checks interpretation plus literal TSV anchors. Status checks cover isolated
handshake reachability, missing endpoint, approval denial and unknown fields.
OCR-only backend failure is explicitly unknown with bounded error output.

Real sandboxed Tesseract acceptance generates a 1000x200 PNG with DejaVu Sans 64
`MISSY FIXTURE TEXT` at (20,60). It checks actual words and bounding boxes:
`MISSY` (26,60)-(214,109), `FIXTURE` (241,61)-(500,109), `TEXT`
(524,61)-(686,108). CLI acceptance checks first center (120,84), path policy,
pin mismatch, disabled config, cancellation and staging cleanup. Exact fixture
assertions intentionally expose changes to the installed font/OCR artifacts;
production OCR does not guarantee universal pixel-identical recognition.

Printable ASCII typing deliberately remains limited to 256 bytes with no keymap
changes. No active desktop validation or arbitrary Unicode parity is claimed.
