# Video Generation Tool — Audit, Fixes, and Rewrite Plan

Audit date: 2026-07-15. Target: `missy/tools/builtin/video_generate.py` (`video_generate` tool),
backed by the external ComfyUI server at `127.0.0.1:8199` (systemd unit `comfyui.service`,
ComfyUI 0.28.0, PyTorch 2.5.1+cu121, RTX 3070 8 GB).

Status: **implemented** — everything in "The Plan" below is built and live-verified unless
explicitly marked *(deferred)*.

---

## 1. Current-state scan

What exists today (commit `6b28c90`):

- Two backends: `svd` (SVD-XT image-to-video via `ImageOnlyCheckpointLoader` /
  `SVD_img2vid_Conditioning`) and `animatediff` (AnimateDiff-Evolved text-to-video on SD1.5).
- Both mux frames via VideoHelperSuite's `VHS_VideoCombine` (h264 MP4, crf 19), then copy the
  result to `~/.missy/videos/` (or HTTP `/view` fallback).
- Polls `/history/{prompt_id}` every 2 s until complete/error/timeout (default 600 s).
- Declares real network/filesystem targets via `resolve_network_hosts()` /
  `resolve_filesystem_targets()` so the policy engine enforces actual values.
- Models on disk: `svd_xt.safetensors`, `v1-5-pruned-emaonly.safetensors`, `mm_sd_v15_v2.ckpt`.

GPU acceleration: **confirmed working** — `/system_stats` reports `cuda:0 NVIDIA GeForce RTX 3070`
with cudaMallocAsync; ComfyUI was launched without any `--cpu` flag. The gap is not that GPU is
unused, but that the tool never *verifies* this: pointed at a CPU-only ComfyUI it would silently
generate at ~50x cost.

## 2. Implementation gaps found

| # | Gap | Severity |
|---|-----|----------|
| G1 | **No audio at all** — every produced MP4 is silent; no way to generate or mux a soundtrack. | high (goal) |
| G2 | **No GPU verification** — tool never checks `/system_stats`; a CPU-only server is silently accepted. | high (goal) |
| G3 | **Random seed not echoed back** — `seed=0` picks a random seed that is never returned, so a good result can never be reproduced or refined ("same seed, more steps" is impossible). | high |
| G4 | **Timeout leaks the job** — on timeout the tool returns an error but the workflow keeps hogging the GPU; no `/interrupt` or `/queue` delete is issued. | high |
| G5 | **No input validation/clamping** — `motion_bucket_id=5000`, `width=513`, `video_frames=400` are all submitted as-is and fail deep inside ComfyUI with opaque node errors (SVD dims must be /8; frame counts have model-specific sane ranges). | medium |
| G6 | **Upload MIME always `image/png`** — a JPEG source is uploaded with the wrong content type. | low |
| G7 | **`save_path` collisions overwrite silently.** | low |
| G8 | **Motion module not overridable** — `_DEFAULT_MOTION_MODULE` is hardcoded; `checkpoint` only overrides the base checkpoint. | low |
| G9 | **No sampler/scheduler control** — hardcoded `euler/karras` (svd) and `euler/normal` (animatediff); animatediff notably benefits from `dpmpp_2m/karras`. | medium |
| G10 | **No model-availability preflight** — a missing checkpoint surfaces as a raw ComfyUI validation blob instead of an actionable "download X to Y" message. | medium |
| G11 | **Poor progress/queue visibility** — poll loop can't distinguish "queued behind another job" from "running". *(partially addressed: the timeout path now inspects `/queue` and reports whether the job was still pending or mid-run; live streaming progress reporting deferred)* | low |

## 3. Quality gaps (why output quality is capped today)

| # | Gap | Fix |
|---|-----|-----|
| Q1 | Best available model is 2023-era SVD/AnimateDiff. State of the art that fits an 8 GB card is **Wan 2.2 TI2V 5B** (text-to-video *and* image-to-video, 24 fps native, dramatically better temporal coherence and prompt adherence). | New `wan` backend. |
| Q2 | SVD output is 25 frames @ 6 fps — visibly choppy slide-show motion. | RIFE frame interpolation (core `FrameInterpolate` node, `rife_v4.26.safetensors`) — 25 frames @ 6 fps → 97 frames @ 24 fps. |
| Q3 | AnimateDiff runs bare SD1.5: no FreeU, weak sampler, 20 steps. | `FreeU_V2` patch + `dpmpp_2m/karras` + 25-step default. |
| Q4 | 512×512 / 1024×576 max output; no upscale path. | Optional `upscale=true`: RealESRGAN x2 (`ImageUpscaleWithModel`) before interpolation. |
| Q5 | crf 19 leaves visible artifacts on high-motion content; encoder is CPU x264 only. | crf 17 default, `video_format` param exposes `h264-mp4` / `h265-mp4` / `nvenc_h264-mp4` (NVENC = GPU encode). |

## 4. Audio design (G1)

ComfyUI 0.28 core ships Stable Audio Open 1.0 support (`ConditioningStableAudio`,
`EmptyLatentAudio`, `VAEDecodeAudio`) and `VHS_VideoCombine` accepts an optional `audio`
input — so audio generation and muxing happen **inside the same workflow graph** as the video,
in one server round-trip, GPU-accelerated end to end.

Parameters, valid with **every** backend:

- `audio_prompt` — text description of the soundtrack ("gentle rain and distant thunder",
  "upbeat synthwave"). Builds an audio branch selected by `audio_model`.
- `audio_model` — **`stable-audio-3`** (default, recommended) or **`stable-audio-open-1.0`** (legacy).
- `audio_path` — mux an existing local audio file instead (uploaded to ComfyUI's input dir,
  loaded via `LoadAudio`). Mutually exclusive with `audio_prompt`.

**`stable-audio-3`** (Stable Audio 3.0 medium base, open-weight, May 2026) — per ComfyUI's official
`audio_stable_audio_3_medium_base` template's direct (no-reprompt) path:
`CheckpointLoaderSimple(stable_audio_3_medium_base.safetensors)` +
`CLIPLoader(t5gemma_b_b_ul2.safetensors, type="stable_audio")` → `CLIPTextEncode` (pos/neg) →
`KSampler(sampler="lcm", scheduler="simple")` → `VAEDecodeAudio` → `VHS_VideoCombine.audio`.
SA3 has **no** `ConditioningStableAudio` node — clip length comes from `EmptyLatentAudio` alone.
Every node class_type already exists in ComfyUI, but running SA3 also needs a ComfyUI build whose
loaders recognize the SA3 checkpoint architecture (ComfyUI 0.28 predates it) plus the two SA3 model
files; the tool's preflight names either if missing. Models: `Comfy-Org/stable-audio-3` →
`models/checkpoints/stable_audio_3_medium_base.safetensors` + `models/text_encoders/t5gemma_b_b_ul2.safetensors`.

**`stable-audio-open-1.0`** (legacy fallback) — the original branch:
`CheckpointLoaderSimple(stable-audio-open-1.0.safetensors)` → `CLIPTextEncode` (pos/neg) →
`ConditioningStableAudio(seconds_total = final video duration)` → `EmptyLatentAudio` →
`KSampler(dpmpp_3m_sde_gpu)` → `VAEDecodeAudio` → `VHS_VideoCombine.audio`. Models:
`Comfy-Org/stable-audio-open-1.0_repackaged` → `models/checkpoints/stable-audio-open-1.0.safetensors`,
plus `google-t5/t5-base`'s `model.safetensors` saved as `models/text_encoders/t5_base.safetensors`.
Both models load the encoder via `CLIPLoader(type="stable_audio")` (the checkpoint embeds none).

Audio duration is derived from the *final* frame count and fps (i.e. after interpolation), so
the soundtrack always matches the clip length. `audio_negative_prompt` defaults to `"noise, distortion, low quality"`.

## 5. The plan

### 5.1 New models (downloaded to `~/comfyui/models/`)

| File | Dir | Source | Purpose |
|------|-----|--------|---------|
| `stable_audio_3_medium_base.safetensors` | `checkpoints/` | Comfy-Org/stable-audio-3 | audio generation (default, SA3) |
| `t5gemma_b_b_ul2.safetensors` | `text_encoders/` | Comfy-Org/stable-audio-3 | SA3 audio text encoder |
| `stable-audio-open-1.0.safetensors` | `checkpoints/` | Comfy-Org/stable-audio-open-1.0_repackaged | audio generation (legacy, SA1) |
| `t5_base.safetensors` | `text_encoders/` | google-t5/t5-base (model.safetensors) | SA1 audio text encoder |
| `rife_v4.26.safetensors` | `frame_interpolation/` | Comfy-Org/frame_interpolation | RIFE interpolation |
| `film_net_fp16.safetensors` | `frame_interpolation/` | Comfy-Org/frame_interpolation | FILM alternative |
| `RealESRGAN_x2plus.pth` | `upscale_models/` | xinntao/Real-ESRGAN v0.2.1 | 2x upscale |
| `wan2.2_ti2v_5B_fp16.safetensors` | `diffusion_models/` | Comfy-Org/Wan_2.2_ComfyUI_Repackaged | wan backend |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | Comfy-Org/Wan_2.2_ComfyUI_Repackaged | wan text encoder |
| `wan2.2_vae.safetensors` | `vae/` | Comfy-Org/Wan_2.2_ComfyUI_Repackaged | wan VAE |

No new custom nodes are required — interpolation, upscale, audio, and Wan are all core
ComfyUI 0.28 nodes; the existing AnimateDiff-Evolved + VideoHelperSuite packs stay as-is.

### 5.2 Rewritten tool architecture (`video_generate.py`, ground-up rewrite)

```
execute()
  ├─ _validate_and_clamp()      # G5: per-backend dim snapping (/8 svd+ad, /32 wan),
  │                             #     frame/fps/motion_bucket/cfg clamps, actionable errors
  ├─ _preflight()               # G2: GET /system_stats → require CUDA/ROCm/MPS device
  │                             #     (allow_cpu=True escape hatch); returns gpu info for output
  │                             # G10: GET /models/{folder} → verify required model files
  │                             #     exist, else error naming the file, dir, and source repo
  ├─ upload image / audio file  # G6: MIME by extension
  ├─ _build_*_workflow()        # svd | animatediff | wan — each returns (graph, image_ref)
  ├─ _append_upscale()          # Q4: optional RealESRGAN before interpolation
  ├─ _append_interpolation()    # Q2: RIFE FrameInterpolate, multiplier auto per backend
  ├─ _append_audio_branch()     # G1: Stable Audio Open gen or LoadAudio mux
  ├─ _append_video_combine()    # Q5: VHS_VideoCombine, crf 17, selectable format,
  │                             #     audio input wired when present
  ├─ submit + _wait_for_completion()   # G11: timeout message reports queued vs mid-run
  │     └─ on timeout: _cancel(prompt_id)   # G4: POST /interrupt + /queue delete
  └─ _retrieve_video()          # G7: collision-safe save path
```

Every workflow builder stays a module-level pure function returning an API-format graph dict
(the existing test style), composed via explicit node-id offsets so branches never collide.

### 5.3 Backends

| backend | mode | model | native defaults |
|---------|------|-------|-----------------|
| `wan` (**new, recommended**) | text→video *and* image→video (`image_path` optional) | Wan 2.2 TI2V 5B (`UNETLoader` + `ModelSamplingSD3(shift=8)` + `Wan22ImageToVideoLatent`), **loaded fp8** — fp16 weights OOM'd the RTX 3070's sampler at default resolution in live testing | 832×480, 81 frames @ 24 fps, 20 steps, cfg 5, `uni_pc/simple`, standard Wan negative prompt |
| `svd` | image→video | SVD-XT | 1024×576, 25 frames @ 6 fps, 20 steps, cfg 2.5, `euler/karras`, auto-interpolate 4x → 24 fps |
| `animatediff` | text→video | SD1.5 + mm_sd_v15_v2 | 512×512, 16 frames @ 8 fps, 25 steps, cfg 7.5, `dpmpp_2m/karras`, FreeU_V2, auto-interpolate 2x → 16 fps |

`wan` on the 3070 (8 GB) relies on ComfyUI's automatic weight offloading into the 64 GB system
RAM; defaults are sized down from the model's 1280×704×121 native config to keep single-clip
latency in single-digit minutes. Default timeout becomes backend-aware (`timeout=0` → auto:
3600 s wan, 1200 s others).

### 5.4 New/changed parameters

New: `audio_prompt`, `audio_negative_prompt`, `audio_path`, `audio_steps`, `audio_cfg`,
`interpolate` (0 = auto per backend, 1 = off, 2–8 = multiplier), `upscale` (bool),
`sampler`, `scheduler`, `motion_module`, `video_format` (`h264-mp4` | `h265-mp4` |
`nvenc_h264-mp4`), `crf`, `allow_cpu`.

Changed: `backend` gains `"wan"` (and `wan` accepts optional `image_path`); `timeout=0` = auto.

Output dict (G3 fixed — seed always echoed): `path`, `backend`, `frames` (final),
`frames_generated`, `fps` (final), `duration_seconds`, `width`, `height`, `seed`, `steps`,
`audio` (`{"source": "generated"|"file", ...}` or `null`), `gpu` (`{"name", "vram_total_gb"}`),
`size_bytes`, `prompt_id`, `elapsed_seconds`.

### 5.5 Tests

Rewrite `tests/tools/test_video_generate.py` in the same mocked-`PolicyHTTPClient` style:
builder graph shape/wiring for all three backends, audio branch composition (generated + file +
mutual exclusion), interpolation/upscale node insertion, validation clamps, GPU preflight
accept/reject, timeout-cancels-job, seed echo, collision-safe save, schema.

### 5.6 Live verification (real ComfyUI, real GPU)

1. `svd` + `audio_prompt` + auto-interpolation → MP4 with audio stream, 24 fps (ffprobe).
2. `animatediff` with FreeU/dpmpp_2m → MP4.
3. `wan` text→video and image→video at defaults → MP4 @ 24 fps.
4. `upscale=true` run → doubled resolution.
5. Confirm VRAM in use during generation (`nvidia-smi`), seed echo, timeout cancel path.

### 5.7 Docs

Update the Video Generation section of `CLAUDE.md` (new backends, audio, models, parameters)
and keep this file as the audit/design record.

## 6. Live verification results (RTX 3070, ComfyUI 0.28.0, 2026-07-15)

All runs GPU-verified (100% CUDA utilization observed via `nvidia-smi` during sampling; the
tool's own `/system_stats` preflight reported the cuda:0 device in every result).

| Run | Result |
|-----|--------|
| animatediff + `audio_prompt` (jazz piano) | ✅ 46 s → 512×512, 16→31 frames @ 16 fps, h264 + **generated AAC track** |
| svd i2v + `audio_path` (WAV) + auto 4x RIFE | ✅ 160 s → 1024×576, 25→97 frames @ 24 fps, h264 + muxed AAC track |
| wan t2v + `audio_prompt`, defaults | ✅ 134 s → 832×480, 81 frames @ 24 fps, h264 + generated AAC track |
| wan i2v (`image_path`), 33 frames | ✅ 832×480 @ 24 fps |
| `upscale=true` + `video_format=nvenc_h264-mp4` | ✅ 512→1024 (ffprobe-confirmed) NVENC h264 |
| `video_format=h265-mp4` | ✅ HEVC stream produced |

Two fixes discovered live (both folded into the implementation):

1. The repackaged Stable Audio checkpoint embeds **no text encoder** — `t5_base.safetensors`
   (google-t5/t5-base `model.safetensors`) must be loaded separately via
   `CLIPLoader(type="stable_audio")`, exactly as in ComfyUI's official audio example.
2. Wan 2.2 5B **fp16 weights OOM the 8 GB card's sampler** at the default 832×480×81 — the
   UNET is now loaded with `weight_dtype="fp8_e4m3fn"`, which completes with headroom
   (~7.6 GB peak) at negligible quality cost.

## 7. Explicitly out of scope (generation)

- Managing the ComfyUI process itself (stays a systemd-managed external service).
- Lip-sync / speech TTS tracks (Missy's Piper TTS is a separate subsystem; muxing a Piper WAV
  via `audio_path` already works and covers the practical need).
- Wan 2.2 14B (needs ≥16 GB VRAM even at fp8) and streaming progress events over WebSocket.

---

# Part II — Video Editing (`video_edit` tool)

Audit date: 2026-07-16. Second pass over the video tool set. Part I (generation) is
implemented and live-verified; the remaining capability gap is that **Missy can create
clips but cannot do anything with them afterwards**.

## 8. Editing gaps found

| # | Gap | Severity |
|---|-----|----------|
| E1 | **No splicing** — two or more generated clips cannot be joined into one video. `video_generate` caps a single clip at ~5 s (svd/animatediff) or ~5 s wan default; longer content requires concatenation, which today means the operator shelling out manually. | high (goal) |
| E2 | **No text overlay** — no way to caption, title, or watermark a video. | high (goal) |
| E3 | **No basic edits** — no trim/cut, no speed change, no resize of an existing file. "Improve based on feedback" for anything temporal (too long, wrong pacing) currently means a full, slow re-generation. | high (goal) |
| E4 | **Concat is not naive** — generated clips vary in resolution/fps/audio presence (svd@24fps-interp vs wan@24fps vs animatediff@16fps; some have audio tracks, some are silent). A raw `concat` demuxer join produces broken files; inputs must be normalized (scale/pad/fps/audio) first. | design constraint |
| E5 | **ComfyUI is the wrong backend for editing** — deterministic cut/join/overlay work needs no GPU sampling and no model; running it through a ComfyUI graph adds latency, VRAM pressure and fragility for zero benefit. | design decision |

## 9. Design

**New tool `video_edit`** (`missy/tools/builtin/video_edit.py`), backed by **ffmpeg/ffprobe**
as the external tool (E5) — `/usr/bin/ffmpeg` 6.1.1 system package, invoked as a direct
subprocess (list argv, no shell, sanitized environment following `tts_speak.py`'s
`_SAFE_*_ENV_VARS` precedent). Verified present in this build: `drawtext` (libfreetype),
`concat`, `xfade`/`acrossfade`, `atempo`, `libx264`/`libx265`, and `h264_nvenc`/`hevc_nvenc`
for GPU-accelerated encoding on the same RTX 3070 the generation backend uses.

One tool, an `operation` parameter (mirrors the single-tool-multiple-backends shape of
`video_generate`):

| operation | purpose | key parameters |
|-----------|---------|----------------|
| `concat` | splice 2+ videos into one (E1) | `inputs` (list, order preserved), `transition` (`"none"` \| `"crossfade"`), `transition_duration` |
| `trim` | frame-accurate cut (E3) | `input`, `start`, `end` or `duration` (re-encode, not `-c copy`, for frame accuracy) |
| `text` | overlay text (E2) | `input`, `text`, `position` (9 presets or `x`/`y` expressions), `font_size` (0 = auto ≈ height/12), `font_color`, `box` + `box_color`/`box_opacity`, `start`/`end` (timed captions via `enable=between(t,a,b)`), `font_file` |
| `speed` | change playback speed (E3) | `input`, `factor` 0.25–4.0 (video `setpts`, audio pitch-preserving chained `atempo`) |
| `resize` | rescale (E3) | `input`, `width`/`height` (0 = derive from aspect; snapped to even) |

Design points:

- **Concat normalization (E4)**: every input is probed with `ffprobe -print_format json`;
  the target canvas is the first input's resolution (max-fps of the set). Each input goes
  through `scale=W:H:force_original_aspect_ratio=decrease` + `pad` (letterbox, never
  distort) + `setsar=1` + `fps=F`, then the `concat` filter. Audio: if *any* input has an
  audio stream, silent `anullsrc` tracks are synthesized for the ones that don't (trimmed
  to that input's duration) so `concat=v=1:a=1` always has matched pairs; if *no* input has
  audio the output is video-only. `transition="crossfade"` chains pairwise
  `xfade=fade` / `acrossfade` with offsets computed from the probed durations.
- **drawtext escaping**: the text is written to a **temp textfile** and passed via
  drawtext's `textfile=` option — sidestepping ffmpeg's notoriously fragile filter-graph
  quoting entirely; arbitrary user text (colons, quotes, percent signs) cannot break or
  inject into the filter graph. Default font: DejaVuSans-Bold, located by `_find_font()`
  probing standard paths (`fonts/truetype/dejavu`, liberation, freefont), overridable via
  `font_file`.
- **Encoding**: same surface as `video_generate` — `video_format` = `h264-mp4` (default) /
  `h265-mp4` / `nvenc_h264-mp4` (NVENC GPU encode), `crf` (default 17; nvenc maps to `-cq`),
  `+faststart`, `yuv420p`, audio AAC 192k. Output defaults to a timestamped
  `~/.missy/videos/edit_<ts>.mp4` with the same never-overwrite numeric-suffix collision
  handling as `video_generate`.
- **Output dict**: `path`, `operation`, `width`, `height`, `fps`, `duration_seconds`,
  `has_audio`, `size_bytes`, `inputs` (count), `encoder`, `elapsed_seconds` — the result is
  ffprobe'd after encoding, so the reported numbers describe the actual file, not the request.
- **Chaining**: complex edits compose by calling the tool repeatedly (trim → text → concat),
  exactly like `video_generate`'s "iteration is just calling the tool again" contract.

## 10. Security / policy integration

- `permissions = ToolPermissions(shell=True, filesystem_read=True, filesystem_write=True)`.
- `resolve_shell_command()` returns `"ffmpeg && ffprobe"` (both binaries run every
  invocation) — the same SR-1.5 convention as `tts_speak.py`, so `ShellPolicy` checks the
  real host binaries, not a meaningless `command` kwarg. With shell policy disabled or the
  binaries not allow-listed, the tool is denied fail-closed.
- `resolve_filesystem_targets()` declares the real per-call values: reads = `inputs`/
  `input`/`font_file`, writes = `save_path` or the default videos dir.
- ffmpeg runs with argv lists (never `shell=True`), a minimal environment, `-nostdin`,
  and a hard timeout (default 600 s, parameter-overridable); stderr tail is surfaced in
  errors.
- Input files must exist and probe as media before any encode starts (actionable errors,
  matching the generation tool's preflight philosophy).

## 11. Tests + verification plan

- `tests/tools/test_video_edit.py`, same style as `test_video_generate.py`: pure builder
  functions (`_build_*_command`) asserted structurally without ffmpeg; execute() paths with
  monkeypatched `subprocess.run`/probe; validation rejections; resolver coverage.
- Live verification (real ffmpeg 6.1.1): synthesize test clips (`testsrc2`/`sine`), then
  concat (mixed fps/audio inputs + crossfade), trim, text overlay, speed, resize, NVENC —
  each output ffprobe-verified (streams, duration, dimensions).

## 12. Live verification results (ffmpeg 6.1.1, 2026-07-16)

Synthetic inputs: 640×480@24 4 s (audio), 1024×576@12 3 s (**silent**), 640×480@24 3 s
(audio) — deliberately mixed resolution/fps/audio-presence.

| Run | Result |
|-----|--------|
| concat all 3 mixed inputs | ✅ 10.0 s (= 4+3+3), 640×480 @ 24 fps, audio present (silent track synthesized for the middle clip) |
| concat crossfade 0.75 s | ✅ 6.25 s (= 4+3−0.75), xfade + acrossfade |
| trim 1.0→3.0 s | ✅ exactly 2.0 s output |
| text overlay, timed 0.5–3.5 s | ✅ frame-extracted at t=2 (rendered) and t=3.9 (absent) — pixel-verified |
| speed 2x | ✅ 4.0 s → 2.083 s, audio pitch preserved |
| resize width=1280, NVENC | ✅ 1280×960 (aspect derived), `h264_nvenc` GPU encode |
| `video_format=h265-mp4` | ✅ ffprobe: `hevc` stream |

Two issues discovered live (both folded into the implementation + regression tests):

1. **drawtext silently renders nothing when the text contains `%`** (its default
   expansion mode treats `%` as the start of a `%{...}` function; ffmpeg still exits 0,
   so the failure is invisible without pixel-checking the output). Fixed with
   `expansion=none` — the tool overlays literal text, never expansion templates.
2. **`x`/`y`/color parameters were an injection surface**: they enter the filter graph
   verbatim, so a crafted value (`x=0,movie=/etc/passwd`) could smuggle a whole new
   filter — including sources reading files the filesystem policy never saw. Now
   whitelist-validated (`_EXPR_SAFE`/`_COLOR_SAFE`: no `:`/`,`/`;`/`=`/quotes/backslash);
   the text content itself was already immune via the textfile route.

## 13. Explicitly out of scope (editing)

- A general ffmpeg passthrough (arbitrary filter graphs) — that's `shell_exec` + operator
  policy territory, not a structured tool.
- Subtitle files (SRT/ASS burn-in), picture-in-picture, chroma key.
- Re-encoding-free (`-c copy`) fast paths — frame accuracy and normalization are worth the
  encode cost at these clip lengths. *(Part III adds one deliberate exception: the `audio`
  mux operation stream-copies the video, since remuxing an audio track is not an edit of
  the video stream and a lossless copy is strictly better there.)*

---

# Part III — Storyboard Coherence + Still/Audio Primitives

Audit date: 2026-07-19. Third pass over the video tool set. Parts I (generation) and II
(editing) are implemented and live-verified; F16 added `video_storyboard` (multi-scene
orchestration). This pass audits the storyboard against its own documented contract and
against the state of practice for multi-scene generation, and fixes what it finds.

Research note: the community-standard technique for coherent multi-scene Wan 2.2 video is
**last-frame chaining** — generate scene N, extract its final frame, and feed it as the
`start_image` of scene N+1's image-to-video run, then concatenate (see ComfyUI's official
Wan 2.2 workflows and the widely-shared multi-scene chaining workflow built on exactly this
pattern). Missy's `wan` backend already supports i2v via `image_path`; what was missing was
(a) a way to extract a frame from a clip at all, and (b) the storyboard plumbing to chain it.

## 14. Gaps found

| # | Gap | Severity |
|---|-----|----------|
| S1 | **Scene `caption` is silently ignored** — the storyboard schema documents a per-scene `caption`, but `execute()` never overlays it. A documented parameter that does nothing. | high (bug) |
| S2 | **Per-scene `transition` mis-honored** — only `scenes[-1].get("transition")` is consulted, and whatever the *last* scene says is applied to *every* join. A `transition` on any other scene is ignored outright. | medium (bug) |
| S3 | **No per-scene overrides, no per-scene seed echo** — scenes can't carry their own `seed`/`negative_prompt`/`image_path`/`audio_prompt`/`steps`/`cfg`/`video_frames`, and the per-scene seeds `video_generate` echoes are discarded, so a storyboard with one bad scene cannot regenerate just that scene. This contradicts the tools' own "iteration is just calling again" contract. | high |
| S4 | **No visual continuity between scenes** — every scene is generated independently from text, so characters/settings drift freely across cuts. The `wan` backend's i2v mode enables last-frame chaining (the standard technique), but nothing implements it. | high (goal) |
| S5 | **No way to extract a still frame from a video** — blocks S4, and also blocks pointing `vision_analyze` at a specific moment of a generated clip for the review-then-refine loop. | medium |
| S6 | **No way to lay one continuous soundtrack over an assembled video** — audio only enters at generation time, per clip. A storyboard given a shared `audio_prompt` generates N *disjoint* mini-soundtracks that get stitched at the cuts. | medium |
| S7 | Storyboard `title_seconds`/`transition` are unvalidated (arbitrary strings/negatives pass straight through). | low |

## 15. Design

**`video_edit` gains two operations** (same builder-function + policy conventions as Part II):

| operation | purpose | key parameters |
|-----------|---------|----------------|
| `extract_frame` | export one frame as PNG/JPEG (S5; enables S4 and vision review) | `input`, `at` (seconds; **-1 = last frame**, computed from probed duration/fps as `duration - max(0.1, 1.5/fps)` so low-fps clips still land on a real frame), `save_path` (extension picks the codec; default timestamped `.png` under `~/.missy/videos/`) |
| `audio` | mux an audio file onto an existing video (S6) | `input`, `audio_file`, `audio_mode` (`"replace"` default \| `"mix"` — amix with the existing track, `normalize=0`, `duration=first`), `loop` (loop a short track to the video length via `-stream_loop -1`) |

`audio` design points: the video stream is `-c:v copy`ed (bit-exact, fast — remuxing audio is
not a video edit; `video_format`/`crf` are ignored and documented as such); the audio is
`apad`ded and `-shortest`ed so the output duration always equals the video's regardless of
the track being shorter or longer; audio-only inputs are probed with a widened `_probe`
(`expect="audio"`) since the Part II prober hard-required a video stream. `mix` falls back to
`replace` when the video has no audio track.

**`video_storyboard` rework**:

- **Captions (S1)**: after a scene's clip is generated (and trimmed), a non-empty `caption`
  overlays it via `video_edit` `text` (position `bottom`, boxed, whole-clip duration). A
  caption failure degrades gracefully to the uncaptioned clip (same posture as trim).
- **Transitions (S2)**: `scenes[i].transition` now means *the join into scene i* (scene 0's is
  ignored); the storyboard-level `transition` param is the default for scenes that don't say.
  Uniform joins → one `concat` call exactly as before; mixed joins → left-fold pairwise
  `concat` calls, each with its own transition. Values validated against `{none, crossfade}`.
- **Per-scene overrides + seed echo (S3)**: scenes may carry `seed`, `negative_prompt`,
  `image_path`, `audio_prompt`, `steps`, `cfg`, `video_frames` — a whitelist, overriding the
  shared kwargs for that scene only. The output's `scenes` list now records each scene's
  echoed `seed`, `prompt`, and `path`, so any single scene can be reproduced or re-rolled.
- **Continuity (S4)**: `continuity=True` extracts the last frame of each scene's *final* clip
  (post-trim/caption — the exact frame the viewer sees before the cut) via `extract_frame`
  and passes it as the next scene's `image_path` (a scene's own `image_path` wins). Requires
  an i2v-capable backend: allowed for `wan`/`svd`, rejected with an actionable error for
  `animatediff`/`wan14b`. For `svd`, scene 0 must supply its own `image_path` (svd always
  needs one); for `wan`, scene 0 runs t2v as usual. A frame-extraction failure is a hard
  error (silently dropping continuity mid-storyboard would defeat its purpose).
- **Soundtrack (S6)**: new `audio_path` storyboard param — after assembly (and title), the
  file is muxed over the whole video via the new `audio` op (`audio_mode`/`loop`
  forwarded). Per-scene generated audio still works (scene `audio_prompt` override / shared
  kwargs) for per-scene sfx/ambience; one continuous *generated* soundtrack would require an
  audio-only generation path in `video_generate` — deferred, see §18.
- **Validation (S7)**: `transition` validated; `title_seconds` clamped to 0.5–30 s.

## 16. Tests

- `test_video_edit.py`: `extract_frame` builder shape + last-frame `at` derivation + execute
  path (monkeypatched run/probe) + validation (missing input, bad `at`); `audio` builder
  (replace/mix/loop argv), duration semantics flags, audio-file probing, validation
  (missing/`non-audio` file), `-c:v copy` assertion; widened `_probe` audio mode.
- `test_video_storyboard.py`: caption → per-scene `text` call; caption failure degrades;
  mixed per-scene transitions → pairwise concats with correct transition each; uniform →
  single concat; override whitelist forwarded (and non-whitelisted scene keys *not*
  forwarded); per-scene seed echo in output; `continuity=True` → `extract_frame` + chained
  `image_path` (and scene-own `image_path` wins; rejected for non-i2v backends; extraction
  failure aborts); `audio_path` → final `audio` mux; `title_seconds` clamp; invalid
  transition rejected.

## 17. Live verification results (ffmpeg 6.1.1 + ComfyUI 0.28 on the 5080 box, 2026-07-19)

| Run | Result |
|-----|--------|
| `extract_frame` at t=1.0 and at=-1 on a 4 s/24 fps clip | ✅ two 640×480 PNGs, byte-distinct (`cmp`); at=-1 derived to 3.9 s |
| `audio` replace: 1.5 s MP3 onto 4 s silent video | ✅ AAC track padded to ~4.0 s; video stream MD5 **identical** to the input's (bit-exact `-c:v copy`) |
| `audio` loop: same 1.5 s track, `loop=True` | ✅ audio stream spans ~4.0 s |
| `audio` mix onto a video with an existing track | ✅ blended track, duration unchanged |
| Real 2-scene `wan` storyboard: `continuity=True`, per-scene captions + durations, crossfade join, title | ✅ scene 1's last frame extracted (`frame_*.png`) and fed to scene 2's i2v generation; per-scene edit chain (trim → caption) visible in artifacts; final 448×256 @ 24 fps MP4, duration exactly 1.375+1.375−0.5 = 2.25 s; **both scene seeds echoed** in the output for re-rolling |

Run-15 validation also exercised VIDEDIT-006's long literal caption (`%`, quotes, commas,
ampersand, colons). It exposed that the height-only automatic font size could place the
beginning and end outside the frame even though drawtext rendered the string safely. Automatic
sizing now measures the selected FreeType font (with a conservative dependency-free fallback)
and reduces the size until every line fits inside the frame; explicit `font_size` remains exact.

One behavior noted live (correct, kept): a scene `duration` longer than the generated clip
trims to the clip's actual length (trim cannot extend); the crossfade offset math then uses
the *probed* durations, so the final length stays exact.

## 18. Explicitly out of scope (Part III)

- Audio-only generation (`video_generate` producing a standalone soundtrack file for
  storyboard-wide *generated* music) — needs a `SaveAudio`-terminated ComfyUI graph and a
  different output contract; `audio_path` covers the assembled-soundtrack need today.
- Wan 2.2 first-last-frame conditioning (`WanFirstLastFrameToVideo`) — jointly conditions a
  scene on both endpoints for authored-feeling transitions; strictly better than plain
  last-frame chaining but needs the 14B i2v expert pair (not present on either box) and a
  per-scene *pair* planning contract. Revisit when the 14B i2v experts are downloaded.
- Per-scene backend mixing (e.g. svd for scene 1, wan for the rest).
