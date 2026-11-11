# AutoCut upstream and editing pipeline upgrade plan

Requested 11 November 2026. Work starts from AutoCut `92d2f93d` on
`codex/pipeline-upgrades-2026-11-11`. The upstream review is pinned to
FilmCraft `25ceb25f`. This document is the execution checklist and records
validation and any remaining work.

## Acceptance and operating rules

- Preserve existing projects, 903 command registrations, AI model defaults,
  Malayalam and English cloning, 360 metadata, privacy controls, RAEA,
  Photos and publishing adapters. Keep old project and script formats readable.
- Implement changes through normal engine commands and CLI/MCP entry points.
  New editable state persists and supports undo. Reject invalid input before
  mutation. Shared GPU and browser jobs use the existing resource locks.
- Develop and measure with synthetic media and isolated projects. A real
  project is opened read-only for compatibility comparisons. User originals,
  credentials, catalogue queues and published videos are production data.
- Render representative before/after frames and exercise actual entry points.
  Every crash fix carries a regression. Run focused checks during development,
  one complete `cargo xtask ci` before delivery, then native build and smoke tests.
- Stage a signed AutoCut app and CLI with rollback copies. Follow the existing
  session delivery workflow after verification. Keep engine tools such as RAEA
  as separate local installations and never bundle their runtimes.
- A checkbox is complete only after implementation and its acceptance checks.
  Record reasons and unresolved prerequisites for any item that cannot pass.

## 1. Upstream first

- [x] **U1 — current hardening batch.** Review all 14 commits
  `a64b3d37..25ceb25f`. Port graphics/shadow correctness, pin-chain limits,
  decoder and input bounds, reserved GIF output, waveform allocation limits,
  Frame Hold identity uniqueness, project-manager traversal, parity/server
  bounds and regular-file reads. Preserve fork APIs and local tests.
  Acceptance: hostile inputs return errors or bounded output, old valid
  fixtures remain correct, original command schemas stay compatible.
- [x] **U2 — responsive scrubbing.** Adapt `97bb466c` and its prerequisites to
  the fork's app/audio layout. Audio device opens run on the scrub worker,
  grains reuse a stream, sequence/sample-rate changes refresh the source,
  and playback releases the scrub device. Acceptance: a fake slow device
  cannot block playhead input, release is bounded, CPU/macOS smoke check.
- [x] **U3 — caption-track choice and MP3.** Port deferred caption export
  selection (`9fc8988d`, `80b7a9ad`) and clean-room MP3 export
  (`9b7967d2`) with UI, CLI and preset compatibility.
  Acceptance: explicitly chosen subtitle track exports correctly, legacy
  defaults work, MP3 imports and decodes with correct duration and channels.
- [x] **U4 — 10-bit H.264 and encoder performance.** Integrate the cohesive
  High 10 and High 10 Intra work (`d24ce15e`) and compatible optimization/
  thread-control prerequisites. Audit every public interface and existing
  thread cap. Acceptance: legacy 8-bit remains valid, 10-bit exports contain
  the requested profile and precision, external decoder comparisons pass,
  output bitrate and timed CPU benchmarks are recorded.

## 2. Catalogue to timeline

- [x] **P1 — FrameAtlas edit-plan import.** Add an engine/CLI workflow using
  stable catalogue identities, scene/speech keys, source part filenames and
  exact timing. Validate references through FrameAtlas, request only chosen
  originals in an editing folder, retain source provenance, then construct
  an editable sequence through existing timeline commands.
  Acceptance: normal/portrait/photo/360/multipart sources, changed scene
  references, expired selections, missing sources, cancellation and retries.
  An import never edits the user's catalogue metadata or duplicates a
  completed source retrieval.

## 3. Colour consistency

- [x] **P2 — source-aware measured colour.** Extend `clip.autoColor` using
  source transfer/primaries and camera metadata. Normalize HDR for an SDR
  target, preserve intended HDR for an HDR target, and add explicit screen
  protection regions for device footage. Keep bounded measured correction
  and before/after reports.
  Acceptance: phone HDR, ordinary SDR, 360 and dashcam fixtures preserve
  highlights/greys/skin and screen contrast, repeated passes do not stack,
  strength overrides, undo and project reopen all work.

## 4. Audio continuity across revisions

- [x] **P3 — editable audio stems and continuity QA.** Retain named dialogue,
  music, source and effects roles through chapter/revision assembly. Add a
  deterministic continuous music-bed policy, proper crossfades and a
  source-audio mute policy for dashcam footage. Extend the current audio QA
  to report abrupt boundary changes and local loudness jumps.
  Acceptance: mixed old/new chapters keep separate stems, narration is
  intact, original muted speech/music stays muted, boundaries render without
  discontinuities, intentional silence has a recorded exemption.

## 5. Reusable renders

- [x] **P4 — chapter and VFX cache.** Fingerprint source content, exact ranges,
  project/sequence settings, engine version, effects, colour, captions,
  audio policy and external composition parameters. Cache complete renders
  atomically with manifests. Reuse unchanged work and invalidate dependent
  work after a change. Keep saved projects separate from disposable cache.
  Acceptance: a second run hits cache, a changed chapter misses only relevant
  entries, audio/colour/size/rate edits invalidate correctly, cancellation,
  concurrent writers, missing media and corrupt cache entries recover safely.

## 6. RAEA previews and quality

- [x] **P5 — RAEA preflight and review bundle.** Extend the helper to emit
  representative contact sheets, motion/frozen-section findings, duration,
  alpha, text bounds and sound cue checks. Support a lower-cost review
  render before the 4K master and connect to the reusable-render cache.
  Acceptance: deliberately frozen, truncated, wrong-rate, missing-alpha and
  clipped-text samples are detected, deliberate holds can be declared,
  sound cues align, scene and transparent transition handoffs reopen.

## 7. Delivery QA as the standard path

- [x] **P6 — integrated delivery recipe.** A single reusable recipe runs
  changed-line pronunciation/hear-back checks, coverage of privacy around
  cuts, motion/frozen-frame checks, stem continuity, colour/screen checks,
  subtitles and safe zones before preparing the final export.
  Acceptance: blocking failures stop delivery and report exact timestamps;
  warnings and skipped checks are explicit. Horizontal YouTube videos use
  sidecar subtitles by default, Shorts can burn captions, multilingual
  files retain aligned cue IDs. Use the existing voice cache and persistent
  pronunciation rules. Upload settings follow the approved project policy.

## 8. Final verification and delivery

- [x] **V1.** Focused regressions and the single full workspace gate pass.
- [x] **V2.** Existing saved project frame comparisons, command preservation,
  local model/catalogue queries, a 4K export and source/audio round-trip pass.
- [x] **V3.** Native desktop workflows are exercised and screenshots inspected.
- [x] **V4.** Commit coherent phases, verify the AutoCut remote target, stage
  rollback copies and install the signed tested app/CLI. Resume the saved
  clean desktop project and verify its hash and playhead.
- [x] **V5.** Update agent guide and pipeline documentation with commands,
  defaults, limitations and examples. Publish a concise completion report.

## Execution log

- 11 November. Created the plan and started U1. The two existing untracked
  local notes remain outside this work. Upstream has 14 new non-merge
  commits at the pinned boundary.

- U1–U4 ported into the fork layout (validation in progress). The monolithic
  export/timeline/app modules are retained. The cohesive first-party H.264
  subtree includes High 10, rate-control repayment, scalar/CABAC/quantizer
  optimizations and explicit thread pools. Generated upstream codec fixture
  files are excluded. Export thread caps are carried into the encoder pool.
- External FFmpeg 9.0.2 / ffprobe installed through Homebrew solely as test
  oracles. They are not called by editing/export product paths or bundled.
- First focused export run: 151 passed, three integration failures. Fixed
  MP3 being subject to PCM sample-format validation and the format count
  assertion; rerunning before proceeding to the pipeline phases.

- Upstream focused checks now pass: export (154 tests plus one ignored),
  H.264 (including High 10 external decoder comparisons and fixed-slice
  thread determinism), MP3 oracle, input/render safety crates, two slow
  scrub-worker tests, two scrub UI tests and five caption export UI tests.
  Full workspace Clippy/CI and native smoke checks remain for delivery.
- P1 implementation started: loopback FrameAtlas adapter, durable retrieval
  receipts, cancellation and an undoable engine edit-plan importer that
  preserves source/scene/speech/part references and mutes dashcam sources.

- P1–P6 implementation and documentation are now present, awaiting the full
  gate and real entry-point checks. Added an actual-source narration check
  (`tts.checkClip`) so QA cannot approve a new synthesized take while an old
  audio file remains on the timeline.
- Engine iteration: 1,206 tests passed; two recipe-list expectations needed
  the additional built-in recipe. Those assertions are updated. The new
  recipe is appended, preserving the ordering of existing recipes.
- The shared GPU is currently occupied by catalogue scene analysis; no VFX
  render has been started on top of it. CPU checks and documentation continue.

- Native validation passed: seven sampled frames from two saved production projects are
  byte-identical to the previous build; neither project file changed. All 903 existing
  commands and eight AI voice/model catalogues are retained. Eleven commands are added;
  only the export and Auto Color schemas intentionally gain options.
- A generated device-screen fixture preserves protected pixels within one code value
  while correcting the surround. Multi-loop music uses non-overlapping A/B tracks,
  retains the partner on rebuild, and undoes atomically. The continuity fixture detects
  a loudness jump and honours an intentional-break exemption.
- Real 4K H.264 delivery, two aligned SRT tracks, byte-identical cache reuse, High 10
  Intra/yuv420p10le verification and native MP3 reimport passed. Actual-source narration
  QA detects a script changed without replacing its timeline audio.
- FrameAtlas's live read-only stats query passed. Retrieval resume/cancellation uses a
  loopback mock, not production downloads. Native imports and focused tests cover a
  portrait photo, retained named camera parts, point-citation ranges, provenance,
  missing/stale sources and one-step undo. No catalogue annotations or queues changed.
- RAEA review, cache hit, true 3840×2160 master and 18-frame 4K alpha transition passed
  native preflight. The alpha sequence saved and reopened with its 30 fps timing.
  RAEA v0.1.0 ignores size overrides for fixed layouts: the helper now sets the output
  stage, the sample lays out from useVideoConfig, and wrong dimensions block import.
  Both contact sheets and the native caption-track selector screenshot were inspected.
  The shared GPU lock was released after these tests.
- Cache dependency review retained both active transition handles and conservative full
  audio dependencies (processor history, tails, lookahead and ducking). Unrelated later
  video edits still reuse earlier chapters. Corrupt artifacts miss and concurrent writers
  fail safely. An interrupted publication is unpublished; a leftover pending directory
  gives its exact path for cleanup rather than being imported as a valid render.
- Local benchmark: three 60-frame 640×360 software exports, 2 Mbit/s requested, two-thread
  limit. Median wall time was 0.405 s before / 0.474 s after; measured file rates were
  2.137 / 1.953 Mbit/s. The encoder tracks its target more closely; this small, concurrent
  workstation run is not evidence of a speed gain. Export includes startup/compositing,
  and the new encoder also respects the thread cap instead of using the global pool.
- Final full workspace gate is running after the native-test fixes. Earlier gate attempts
  were superseded after a recipe-test expectation and the multi-loop integration failure;
  they are not reported as completed passes.

## Practical limits

- Screen masks are static until tracked/keyframed; they cannot recover clipped originals.
  HDR uses the existing native colour pipeline; SDR compositing remains RGBA8.
- Motion and privacy checks sample frames. Declared text bounds are geometry checks,
  not OCR, and audible-cue checks establish signal, not sound-design quality. Visual and
  listening review remain necessary. Missing required models block the standard gate;
  explicit project-policy skips are recorded.
- RAEA stays separately installed under its own licence, outside AutoCut's package.
  Explicit external dependencies must include imported modules, fonts, media and sounds.
  No upload, publication or AI disclosure setting is changed by delivery.prepare.
- Production media, project files, catalogue state and unrelated local notes were preserved.

- Final gate review caught an existing mask-job cancellation observation race under load:
  the cancel flag was set, but jobs.list exposed only worker completion. Added an additive
  cancelRequested field, a deterministic pending-worker regression and the corresponding
  project-replacement assertion. Cancellation does not claim premature completion.
  The focused regression passes; the final full gate uses four test threads and eight
  build jobs to keep the shared workstation responsive.
- Project-level render resources are now in cache identity too: embedded LUT text, project
  settings, source graphics and motion tracks. A dedicated regression proves that changing
  a library LUT invalidates the cache without changing a clip reference.

- A complete checkpoint gate passed (`ci-green.log`): formatting, warnings-as-errors Clippy,
  workspace tests, layering, 93 attributed assets and all 57 WebAssembly crates. The
  release gate is repeated after the final review additions: the FrameAtlas HTTP error
  body now retains expired-selection recovery instructions, and cache identity includes
  build features, target and flags. Synthetic configuration tests also cover an unrelated
  non-Unicode environment value and a clean error for invalid selected build flags.
- The exported native 4K file also passed the positive audible-cue check, including
  exact duration, dimensions and frame rate; absent/silent cues are covered by regressions.

- **Release validation passed:** the completed `ci-release.log` gate exited 0 with 6,222
  passing tests across 418 suites (89 intentionally ignored), including all 1,227 engine
  tests. Clippy treats warnings as errors; layering, all 93 indexed assets and all 57
  WebAssembly crates pass. The last HTTP/configuration regressions are in this run.
- Built the shipping app and CLI with the workspace's unified feature graph. Native and
  WebAssembly renderer fingerprints are distinct. The exact shipping binaries repeated
  the seven saved-project frame matches, all eight AI catalogue comparisons and native
  4K/cache/High 10/MP3/two-language SRT tests successfully. All 914 commands are present.
- A real 2160×3840 Short with burned captions and sidecar passed delivery and publishing
  preflight. The gate first correctly blocked a caption behind YouTube's bottom controls;
  moving it to the safe middle allowed delivery. The rendered contact sheet was inspected.
- Code is ready for the tested source commit, signed package and rollback-safe installation.
  V4 and the final delivery record remain pending until installation is observed.

## Completed delivery

All upstream and pipeline phases are implemented and verified. The installed source build is
`d7ce0e60` (AV20), followed only by this documentation record. It preserves all 903 previous
commands, adds 11, retains the eight AI catalogues and keeps publishing/disclosure policy intact.

The signed local AutoCut app and CLI are installed at their normal paths. The saved EXEED RX
v04 project reopened in Editing at tick 7,256,390,400,000; its file hash, clean state and playhead
were preserved. The new continuity command also ran successfully through the
live app bridge. Rollback copies are `target/macos-package/AutoCut-before-AV20-2026-11-11.app`
and `target/release/filmcraft-cli-before-AV20-2026-11-11`. Packaging used local ad-hoc signing,
matching the established development workflow; RAEA and FFmpeg are not bundled.

The verified remote is `moheshmohan/AutoCut`, using `autocut`, with fast-forward publication
to `main` and `feat/autocut`. The final documentation commit records this deployment; its
source code is identical to the fully tested installed build.

Future editing sessions should start with [the native pipeline guide](../docs/local-pipeline-upgrades.md)
and [RAEA workflow](../docs/raea-workflow.md). Catalogue assembly, screen protection, editable
stems/continuous music, render reuse, RAEA review bundles and checked delivery are ready to use.
