# Native explicit live capture

`CaptureTools::new(PolicyConfig, CaptureConfig, Arc<dyn CaptureApproval>)` is
the sole public construction API. It implements `ToolExecutor`; the convenience
`capture_run(ctx, call)` goes through the same gate, not a typed approval bypass.
Linux only. Requires Landlock ABI 5 or later, pinned native FFmpeg/FFprobe and
private, same-UID mode-0700 staging. Never silently falls back without confinement.

## Operator configuration and external approval

Camera and screencast are independently **disabled unless explicitly enabled**.
`CaptureConfig` requires explicit executable absolute canonical paths and SHA-256,
runtime library/binary read roots, staging root and exact camera paths. Defaults
when deserializing: timeout 120000ms, output 32MiB, pixels 1920x1080, FPS 30.
The policy file budget must cover the configured output budget.

The three tool names are `camera_photo`, `camera_clip`, `screencast`. Every
definition is `side_effect=true`. Arguments share `output`, `consent:true`,
`width`/`height` (default 640x480), `fps` (default 15), `duration_seconds`
(default 1, maximum 90). Camera tools additionally require exact allowlisted
`device`, limited to `/dev/videoNUMBER` V4L2 character devices with Linux major 81.
Photo output is one PNG frame; clips are silent MPEG-4 MP4. Dimensions must be
even, no more than 1920x1080. FPS is 1..30 subject to lower operator maximum.
There are no arbitrary format/filter/command/source/URL options.

Caller `consent` is a capability request, **not proof of human consent**.
`CaptureApproval::authorize(context, call)` MUST integrate a durable exterior
human approval system tied to the exact complete call, context scope and turn,
and executor configuration fingerprint. The trait has no allow-all implementation.
The parent runtime must enforce its own durable approval and UNKNOWN-effect
journal in addition to this gate. Authorization errors propagate unchanged so
`ApprovalRequired` retains its semantics. Rejected pre-dispatch validation is
`ToolDispatchRefused`. After capture dispatch all failures, cancellation, timeout
and dropped futures are potentially UNKNOWN effects; **never automatically retry**.

Preflight checks all arguments, output policy and source identity without opening
the device/connecting to X11. Execute repeats validation, takes an identity
snapshot, asks exact-call approval, and checks identity again before dispatch.
No production test-source bypass is exposed. Read-only camera sysfs discovery is
already provided by `missy-media`; this crate does not autodiscover any source.

## X11 and audio safety

`TrustedX11 { display, xauthority, isolated_session:true }` is trusted operator
configuration only, never a model argument. Only canonical local `:NUMBER`
displays numbered at least 100 are permitted. The application must supply a
**fresh separately provisioned isolated session**, not an existing desktop.
The flag is an operator attestation, not cryptographic proof of isolation.
The Unix socket must belong to the service UID. The canonical Xauthority must
be bounded, owned, regular and mode0600 or stricter. It is copied into private
staging; only that explicit XAUTHORITY enters the cleared child environment.
FFmpeg receives `unix:NUMBER+0,0` explicitly. Ambient DISPLAY, WAYLAND_DISPLAY,
XAUTHORITY, PulseAudio/PipeWire variables and shell environment are never inherited.
The application must never grant the active user's desktop cookie here.

`audio` and `audio_consent` default false and only false is accepted. **No audio
backend exists**: even a separate consent boolean cannot enable an unimplemented
microphone capture path. Inputs map only video; `-an -sn -dn` and post-probe
exactly-one-video-stream checks prevent hidden audio. Future audio support must
have a separate operator opt-in, allowlisted source and durable human consent.

## Child confinement and output verification

Only generated argv is executed, without a shell or PATH resolution. Binary
hashes are reverified on every execution and the open verified inode is executed
through its descriptor. Child-only Landlock grants runtime read paths, private
staging read/write, and only the chosen V4L2 device read/write/IOCTL. O_RDWR and
VIDIOC ioctls are necessary for FFmpeg's V4L2 driver; no `/dev` tree is granted.
All other device ioctls and TCP bind/connect are denied. RLIMIT_FSIZE, CPU/core
limits, FFmpeg output limit, bounded combined stdout/stderr drain and wall timeout
bound execution. A private process group is killed/reaped on error, cancellation
and future Drop. The leader is not reaped before group cleanup (PID reuse safety).

FFprobe counts decoded frames, verifies dimensions, codec, stream count and actual
duration. A second FFmpeg complete strict decode rejects damaged/truncated media.
Only then are bounded bytes atomically published through Policy's descriptor-
relative no-symlink mode0600 file writer. Returned metadata contains actual byte
size, SHA-256, actual duration/frame count, codec/format metadata and `audio:false`.
No successful capture is inferred from process spawn or a placeholder filename.

Landlock does not restrict Unix-domain IPC, UDP, display protocol behavior or
V4L2 ioctl meanings. X11 access is session-wide, not window-confined. The cleared
environment, narrow generated argv, trusted configured endpoint and private cookie
are important controls. This does not claim seccomp/VM isolation. Root, same-UID
attackers able to mutate configured roots/mounts or the trusted X server/device
are outside this userspace boundary. Runtime read roots must contain no secrets.

## Tests

`cargo test -p missy-capture -- --test-threads=1` uses only the internal `cfg(test)`
FFmpeg `testsrc` fixture and a **new authenticated Xvfb**. It never opens any
camera/microphone, connects to an existing display or uses ambient desktop state.
Tests cover actual photo/clip/hash/full decode, fresh isolated screencast through
exact-call approval, denial, postapproval cookie drift, malformed arguments,
path symlinks, pin drift, timeout, cancellation, future drop and output bounds.
Physical V4L2 device compatibility is intentionally not live-tested.
