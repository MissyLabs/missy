# Native desktop, capture, media, vision and specialty production contracts

This is the conversion artifact for the media/desktop slice. It records actual
behavior, acceptance evidence and explicit differences from legacy Python. It is
not a deployment record. No running bot, real camera, microphone, ambient display,
personal browser or existing graphical session was used during this work.

## Tool surface and approval wiring

All constructors remain compatible with their existing signatures. New optional
configuration fields default to disabled. The runtime must register exact tool
definitions and preserve `side_effect` approval/journal semantics. Desktop and
capture additionally enforce injected, scope-bound durable human approval. A
model-provided consent boolean is not the approval authority.

* `camera_burst` and legacy-name `vision_burst`: existing `CaptureTools` constructor.
  Explicit `outputs` array (1..16 unique PNG paths), `interval_ms` (100..10000),
  exact allowlisted `device`, `consent:true`, existing dimensions/fps fields. No
  ambient camera selection. One exact approval binds the entire burst. Per-frame
  executable/device/output snapshots detect post-approval changes. Cancellation
  stops remaining frames. Already captured files may remain after partial failure;
  the runtime must never automatically retry an uncertain capture. Inter-frame
  delay is measured after each completed capture, not a real-time camera clock.
* `browser_fill_many`: `fields:[{selector,value}]`, optional `submit_selector`.
  All selectors are resolved and typed before first mutation. Native prototype
  value setters plus input/change events support controlled forms. Submit is in
  the same renderer turn. Duplicate selectors, arbitrary fields and oversized
  arrays/values are refused. Calls are mutations and always approval-gated.
* `browser_extract`: typed `mode:text|links|table|form` and `selector`; bounded
  DOM extraction. Form mode returns metadata, not field values/passwords. The
  existing `browser_evaluate` remains fixed read IDs only. Arbitrary JavaScript
  is deliberately not exposed. Exact configured target/origin checks and an
  independently isolated browser network remain mandatory, including for the
  new typed workflows. CDP interception is not a browser-wide network firewall.
* `desktop_focus_window`: exact window ID, allowed class/title, explicitly
  isolated X11 socket and cookie. Raises and sets focus natively, verifies focus.
* `desktop_launch_app`, `x11_launch`, `install_software_confirmed`: optional
  `DesktopConfig.launch:DesktopLaunchConfig`. Config contains `isolated_session`,
  `applications` and `packages` maps, explicit `environment`, canonical
  `working_directory`, and `max_children` (1..16). Each map value is
  `DesktopCommand {executable:DesktopExecutablePin {path,sha256},args}`. Tool args
  select only an exact operator-bound app/package key, never executable/argv.
  No PATH search, inherited desktop session, shell, or automatic sudo. Package
  keys may include exact versions; package command argv is fully operator
  supplied. All names require enable_input and approval. Executable inode is
  pinned by verified descriptor through spawn. Application children are bounded
  and supervised until executor shutdown. Install waits for command completion;
  timeout/cancel is an uncertain mutation requiring reconciliation and external
  operational validation. This facility is not a container or a privilege sandbox:
  only bind reviewed trusted binaries and an actually isolated session/user.
* `x11_read_screen`: additionally inject
  `DesktopTools::with_screen_interpreter(Arc<dyn ScreenInterpreter>)`.
  Trait: `async fn interpret(&self, ctx:&ToolContext,
  png:Vec<u8>, question:&str)->Result<Value>`, plus fingerprint. Captures only the
  approved allowlisted window, converts real X11 stride, byte order and channel
  masks to PNG, then returns actual interpreter analysis, not a relabeled image.
  A host adapter can call `VisionTools::analyze_png(ctx,&png,question)` after the
  parent capture/disclosure approval. This API sanitizes/decodes and retains the
  normal backend budget ledger. Without the interpreter this tool is absent.
* `vision_monitor`: explicit image **file**, question, samples (1..60), interval_ms
  (100..60000), cooldown_ms (0..300000), change_threshold (finite 0..1), optional
  analyze_first. Five-minute maximum polling span. A separately approved producer
  may atomically refresh the file. This is not a covert background camera. Use
  the scheduler for recurring runs with durable execution/approval. Pixels are
  decoded/sanitized and normalized to 32x32 RGB; mean absolute channel difference
  is normalized 0..1. Compare to the last analyzed baseline so gradual changes
  accumulate. Billing only occurs on threshold plus cooldown. Maximum 32 analyses
  per run, normal per-call/total backend ledger, cancellation-aware sleeps. Process
  restart does not preserve an in-flight polling run or its baseline.
* `vision_scene`: now actually implements legacy scene-session actions `create`,
  `add_observation`, `update_state`, `summarize`, `close`, using the private configured
  vision SQLite ledger. Sessions are isolated by exact tenant/user/channel/session
  tuple, task ID scoped within that tuple. Empty task ID selects that scope's latest
  open session; create generates a UUID. Supported task_type values are general,
  puzzle, painting, inspection. `state_updates` is a bounded JSON object string.
  Observations are append-only (256/session), state/observation contents remain
  explicitly untrusted caller assertions. Close prevents further mutation.
  Summaries return latest 32 observations plus total/truncation indication. No
  provider output automatically becomes authoritative state. Existing imagepath /
  max_objects strict model scene analysis remains available without an action.
* `video_storyboard` accepts existing images, or prompted scenes when a generator
  is injected. `video_generate_storyboard` is the explicit scene-generation name.
  Inject `MediaTools::with_scene_generator(Arc<dyn SceneGenerator>)`. Trait:
  `async generate(&self,ctx:&ToolContext,request:&SceneGenerationRequest)->Result<PathBuf>`
  and fingerprint. Request public fields: prompt, output:PathBuf, width, height.
  Model args: `scenes:[{prompt,output}],output,seconds_per_image,width,height`.
  The trusted host adapter must use governed integration jobs, binding fixed
  operator workflow/output node configuration into its fingerprint. Integration
  provides `IntegrationTools::approved_scene` for approved Comfy submit, bounded
  history polling and scope-owned artifact download. There is no generated-image
  placeholder. Media code checks returned exact path, fully decodes each image,
  requires requested dimensions, assembles and fully decodes a real MP4. Existing
  images-only assembly is accurately described as such when no generator exists.
* `document_pdf_extract` and `document_ocr`: optional
  `SpecialtyConfig.document_backend:DocumentBackendConfig` with optional pdftotext /
  tesseract `DocumentExecutablePin {path,sha256}`, staging_root,
  runtime_read_roots and timeout_ms (<=60000). Existing enable_documents also
  required. PDF accepts policy-readable file and max_pages (1..100, default20);
  actual pinned pdftotext text extraction. OCR accepts explicit PNG/JPEG, decodes
  and rewrites pixels before pinned English Tesseract. No camera/screen discovery.
  Landlock ABI>=4 required, TCP denied, no shell, arbitrary flags, languages or
  broad secret-directory reads. Runtime roots must be canonical subdirs of /usr
  including required OCR trained data. Input/output cap uses max_document_bytes.
  Scanned-PDF page rasterization and OCR of PDFs is not automatic; extract/render
  pages separately. Existing `document_extract` remains UTF-8 text/Markdown.

## Validation and evidence

Focused command (never full workspace):

`CARGO_BUILD_JOBS=2 cargo test -p missy-desktop -p missy-vision -p missy-media -p missy-specialty -p missy-capture -- --test-threads=2`

Acceptance fixtures include:

* authenticated new Xvfb server and new allowlisted window only: native focus,
  existing XTest/deny behavior, actual X11 pixels -> PNG -> synthetic interpreter;
* exact pinned `/usr/bin/true` application fixture, arbitrary argv/unknown app /
  pin drift denial; never launches a GUI or invokes a package manager;
* synthetic Node DOM classes executing fixed multi-field fill/submit and table
  extraction including JavaScript-looking value data, without evaluation escape;
* generated fixture scenes -> real FFmpeg MP4 -> decoded red and blue frames;
* synthetic capture burst frames, typed bounds/duplicate outputs, no device use;
* unchanged-file monitor makes no provider request; atomic synthetic pixel change
  invokes loopback fixture provider once and cooldown suppresses later calls;
* scene session create/observe/state/reopen/summarize/close with cross-user denial;
* hand-built PDF -> actual sandboxed pdftotext text, synthetic drawn text PNG ->
  actual sandboxed Tesseract recognition, never real display content;
* robust browser PNG full decoder checks now reject a previously accepted fake
  screenshot with an invalid IDAT CRC. Tests use encoder-generated valid PNG.

The final whole-slice focused run passed 62 tests, including the synthetic DOM
script acceptance. Do not infer deployed readiness from a local fixture pass. Live Comfy,
external vision credentials, production browser isolation, package install, real
hardware access and the inaccessible Robby host were not acceptance-tested here.
