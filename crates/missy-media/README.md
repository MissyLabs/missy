# Native file-only media executor

`MediaTools::new(policy: PolicyConfig, config: MediaConfig) -> Result<MediaTools>`
implements `ToolExecutor`. Register behind the runtime's durable approval gate.
Every artifact-writing tool has `side_effect = true`. The executor does not
self-approve. `vision_devices` is the only read-only tool.

## Implemented tools

* `video_edit`: typed `trim`, `transcode`, `concat`, `overlay`, `extract_frame`,
  `extract_audio`, and `storyboard` operations. Output video is intentionally
  **silent MPEG-4 MP4**, 15 fps and YUV420p. Audio extraction produces mono
  16 kHz PCM16 WAV, not speaker output. Frames are PNG.
* `video_storyboard`: assembles existing PNG/JPEG files. Files produced by a
  separately approved Comfy integration can be consumed through the same read
  policy. No image generation, arbitrary remote service calls, transitions,
  music generation, or inferred/fabricated shots.
* `image_edit`: PNG/JPEG copy, resize, crop, rotate (90/180/270), overlay.
  Rust `image` decoders check dimensions, pixel count and allocation limits.
* `vision_capture`: requires explicit input **file**, not a device or desktop.
  Decodes and re-encodes the existing image. Never captures real hardware.
* `vision_devices`: fixed read-only `/sys/class/video4linux` discovery.
  It never opens `/dev`, starts capture or accesses a microphone.

No Python bridge, arbitrary command execution, raw filters, model-controlled
executable arguments, STT/TTS implementation, vision model, synthetic singing,
or pretend music generation. Typed public `VideoRequest`, `VideoOperation`,
`ImageRequest`, `ImageOperation`, `Rotation`, `Artifact`, and metadata types are
exported for native integrations.

## Explicit operator configuration

`MediaConfig` has no implicit executable discovery or permissive defaults:

* `ffmpeg` and `ffprobe`: `ExecutablePin { path, sha256 }`, canonical absolute
  executable path and approved SHA-256 (64 hexadecimal characters). Regular
  executable files may not be group/world writable. Both are verified on
  construction **and before every invocation**. The operator obtains and
  approves these hashes out of band; a model cannot choose them.
* `staging_root`: existing canonical absolute directory. Each call creates an
  independent mode-0700 temporary directory beneath it, removed by RAII on
  success, failure, cancellation and future drop.
* `runtime_read_roots`: explicit canonical runtime directories below `/usr`,
  normally `/usr/bin` and `/usr/lib`. Never grant documents or secret trees.
  These permissions apply to the child, not to model file reads.
* `timeout_ms`: 1..300000 per child invocation.
* `max_duration_seconds`: 1..600. Input, aggregate and output durations checked.
* `max_pixels`: 1..16777216. Video dimensions additionally limited to even
  dimensions 2..1920. Native image dimension ceiling is 8192.
* `max_output_bytes`: 1024..67108864, also subject to policy file-size limit.

`PolicyConfig` read/write roots are separately authoritative. The internal
native-execution capability is generated for the exact, typed per-call argv;
it does not enable the user's generic shell tools. Changes to operator config,
file policy or catalog change `configuration_fingerprint`.

## File and process boundary

1. Source files go through `Policy::read_file`: bounded regular file,
   descriptor-relative walks, `O_NOFOLLOW` throughout, no symlink/hardlink input.
2. Bytes are copied to fixed, private staging filenames. Images are normalized
   with the bounded Rust decoder first. The child sees neither original paths
   nor policy read/write roots.
3. Explicit input demuxers restrict video to MOV/MP4, Matroska/WebM and WAV.
   No playlist, concat demuxer, image2 patterns or nested URL format is accepted.
   Protocols are limited to `file,pipe`. MOV external data references and
   absolute-path dereferencing are explicitly disabled.
4. Every native invocation uses `missy-sandbox::SandboxedCommand`, actual
   child-only Landlock ABI >= 4 with TCP denial, cleared environment, null
   stdin, exact generated argv policy validation, combined-output cap and
   timeout. Read permission covers only explicit runtime roots and the private
   call directory. Write permission covers only that call directory.
5. Process ownership is inherited from the sandbox adapter: a separate group,
   reserved leader PID (`waitid WNOWAIT`), kill plus reap on timeout,
   cancellation, future drop, output overflow, errors and completion. No shell.
6. FFprobe verifies actual codec, streams, dimensions and finite bounded
   duration. FFmpeg fully decodes the bounded output to its null muxer before
   publication. Duration agreement rejects size-limit truncation rather than
   claiming a partial edit succeeded. PNG/JPEG edits re-decode before publishing.
7. Only then does `Policy::write_file` atomically publish mode-0600 bytes to the
   explicitly requested destination, rejecting symlink destinations. Existing
   regular user-owned files in policy-authorized write roots may be atomically
   replaced; this is intentional and requires outer mutation approval.

Returned artifact JSON contains the actual path, byte length, SHA-256 and
verified media metadata, not a fabricated success marker.

### Limits of the guarantee

The approved executable, linked runtime libraries, operator configuration and
same-UID/administrator filesystem integrity are trusted. Hashing immediately
before spawn is not an inode-pinned `fexecve`: an administrator/same-UID process
able to replace approved executables during execution remains outside this
boundary, consistent with the policy crate's threat model. Landlock does not
claim UDP or Unix-socket isolation; the media API permits no networking flags
or protocols. Native decoders are bounded by allocation flags, duration,
dimensions, output limits and wall-clock timeout, not a formal memory/CPU quota.
Publication is atomic per artifact, not a multi-artifact transaction. Input
reads are capped by `PolicyConfig.max_file_bytes` (hard ceiling 64 MiB).

## Validation

`cargo test -p missy-media` creates tiny synthetic colored videos with trusted
test-only lavfi constants and no hardware. It exercises all seven FFmpeg edit
operations, verifies artifact hashes/codecs/durations, decodes frame pixels,
tests image operations, policy and symlink refusal, disguised playlists, exact
typed-field rejection, executable hash mismatch, decompression bounds, actual
Landlock denial, process timeout, cancellation and dropped-future cleanup paths.
No expensive render, real camera, microphone, desktop interaction or playback.
