# Native file-only media executor

`MediaTools::new(policy: PolicyConfig, config: MediaConfig) -> Result<MediaTools>`
implements `ToolExecutor`. Register behind the runtime's durable approval gate.
Every artifact-writing tool has `side_effect = true`. The executor does not
self-approve. `vision_devices` is the only read-only tool.

## Implemented tools

* `video_edit`: typed `trim`, `transcode`, `concat`, `overlay`, `extract_frame`,
  `extract_audio`, `storyboard`, `audio_mux`, `audio_mix`, `speed`, `text`, and
  `crossfade` operations. Existing video operations remain intentionally
  **silent MPEG-4 MP4**, 15 fps and YUV420p. Audio is present only through explicit
  new audio operations or a required `preserve_audio: true` on speed/text/crossfade.
  Audio extraction produces mono
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

## Legacy video_edit compatibility mapping

The tool name and approval boundary stay `video_edit`. This is native typed
compatibility, not a Python call-through or acceptance of legacy free-form flags.

| Legacy option | Native request |
| --- | --- |
| `audio`, `audio_mode=replace`, `audio_file`, `loop` | `audio_mux`, `input`, `audio_file`, required `loop` boolean and finite `gain` 0..4 |
| `audio`, `audio_mode=mix` | `audio_mix`, same fields plus required finite `original_gain` 0..4; absent original track degrades to replacement |
| Audio duration follows video | Omit/null `duration_ms` to use video duration, or positive bounded duration no longer than video; short audio pads with silence, explicit loop repeats it |
| `speed`, `factor` | `speed`, finite factor 0.25..4 and required `preserve_audio`; real audio uses chained atempo stages 0.5..2, pitch preserved |
| `text`, `font_file`, `font_size`, `position` | `text`, explicit policy-readable TTF/OTF `font_file`, font size 8..256, nine legacy position names; 4px margins instead of legacy 20px |
| Caption `start`, `end` seconds | Required `start_ms`, `end_ms`, positive interval within actual input duration |
| `font_color`, `box_color`, `box_opacity`, `box` | Required colors from white/black/red/blue/yellow, finite opacity 0..1, boolean box; no color or coordinate expressions |
| `concat`, `transition=crossfade`, `transition_duration` | `crossfade`, exactly two `inputs`, even bounded `width`/`height`, positive `duration_ms` <=5000 and <= shortest actual clip, required `preserve_audio` |
| Legacy implicit audio preservation | Not implicit: existing operations remain silent, new speed/text/crossfade require an explicit boolean |

Mux/mix stream-copy the video and encode AAC stereo 48kHz at 128kbit/s.
Speed/text/crossfade encode MPEG-4 video with optional AAC, not NVENC/H.265.
Crossfade uses normalized resolution/fps/timebase; if explicitly preserving
audio and one clip lacks a track, bounded silence fills that clip before acrossfade.
No audio is synthesized when neither clip has audio. Output duration is checked.
Legacy arbitrary x/y expressions, automatic font discovery/size, arbitrary colors,
CRF/encoder selection, last-frame sentinel and implicit output naming are not
accepted. Existing bounded trim/transcode/extract mappings are unchanged.

Caption UTF-8 text (1..4096 bytes, no NUL) is written to a private textfile and
rendered with `expansion=none`; punctuation, percent expressions, quotes, colons
and filter syntax stay literal. The approved font is read through Policy and
copied to a fixed private filename; there is no font resolver, environment font
path, or implicit font grant. Only generated numeric/enum expressions enter the
filter graph. Caption staging roots with graph metacharacters are rejected.
No new config fields were needed: existing config remains source/JSON compatible;
the expanded schema is already part of the configuration fingerprint.

All streams, including expected AAC, are FFprobed and fully decoded before atomic
publication. Interrupted command timeouts/cancellation return an unknown-outcome
error and are never retried here. Dropped futures are killed/reaped by the sandbox;
the caller's durable journal must keep their outcome unknown until reconciliation.

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
test-only lavfi constants and no hardware. It exercises the original seven FFmpeg edit
operations, verifies artifact hashes/codecs/durations, decodes frame pixels,
tests image operations, policy and symlink refusal, disguised playlists, exact
typed-field rejection, executable hash mismatch, decompression bounds, actual
Landlock denial, process timeout, cancellation and dropped-future cleanup paths.
No expensive render, real camera, microphone, desktop interaction or playback.
New-operation tests exercise 13 real mux/mix/loop/speed/text/crossfade cases,
decode video and audio bytes to hashes, assert stream presence and durations,
compare actual loop/mix audio, exercise literal dangerous-looking caption text,
mixed missing-audio crossfades and full-overlap boundaries, and reject excessive
gains, non-finite speeds, missing preservation flags and denied audio/font reads.
