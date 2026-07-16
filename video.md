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

Two new parameters, valid with **every** backend:

- `audio_prompt` — text description of the soundtrack ("gentle rain and distant thunder",
  "upbeat synthwave"). Adds a Stable Audio Open branch:
  `CheckpointLoaderSimple(stable-audio-open-1.0.safetensors)` → `CLIPTextEncode` (pos/neg) →
  `ConditioningStableAudio(seconds_total = final video duration)` → `EmptyLatentAudio` →
  `KSampler` → `VAEDecodeAudio` → `VHS_VideoCombine.audio`.
- `audio_path` — mux an existing local audio file instead (uploaded to ComfyUI's input dir,
  loaded via `LoadAudio`). Mutually exclusive with `audio_prompt`.

Models: `Comfy-Org/stable-audio-open-1.0_repackaged` (non-gated) →
`models/checkpoints/stable-audio-open-1.0.safetensors`, plus `google-t5/t5-base`'s
`model.safetensors` saved as `models/text_encoders/t5_base.safetensors` (the checkpoint does not
embed a text encoder; it is loaded via `CLIPLoader(type="stable_audio")`, exactly as in ComfyUI's
official audio example).

Audio duration is derived from the *final* frame count and fps (i.e. after interpolation), so
the soundtrack always matches the clip length. `audio_negative_prompt` defaults to `"noise, distortion, low quality"`.

## 5. The plan

### 5.1 New models (downloaded to `~/comfyui/models/`)

| File | Dir | Source | Purpose |
|------|-----|--------|---------|
| `stable-audio-open-1.0.safetensors` | `checkpoints/` | Comfy-Org/stable-audio-open-1.0_repackaged | audio generation |
| `t5_base.safetensors` | `text_encoders/` | google-t5/t5-base (model.safetensors) | audio text encoder |
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

## 7. Explicitly out of scope

- Managing the ComfyUI process itself (stays a systemd-managed external service).
- Lip-sync / speech TTS tracks (Missy's Piper TTS is a separate subsystem; muxing a Piper WAV
  via `audio_path` already works and covers the practical need).
- Wan 2.2 14B (needs ≥16 GB VRAM even at fp8) and streaming progress events over WebSocket.
