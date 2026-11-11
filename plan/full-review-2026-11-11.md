# AutoCut full review — 11 November 2026

Baseline: `cdf63e5b`, installed source `d7ce0e60`. Review branch:
`codex/full-review-2026-11-11`. Existing unrelated local notes stay untouched.

## Scope and approach

Reviewed project/history rollback, subclip placement and colour measurement, audio stems and
scrub worker, export/delivery, render cache, catalogue retrieval, the RAEA handoff and relevant
UI integration. Reproductions use generated media and local mock services. Saved production
projects are read-only compatibility samples, never writable test fixtures.

Severity describes the observed consequence in these workflows, rather than a security rating.
A full review cannot establish that every possible media file or workflow is bug-free.

## Confirmed findings and fixes

| ID | Severity | Trigger and observed bug | Change |
| --- | --- | --- | --- |
| R1 | High | Export a WAV through a hard-link alias of the open project: the project becomes RIFF audio. Frame export, captions and QA reports can also replace imported originals. | Reject project/original destinations, including existing native hard/symbolic-link aliases, before movie/frame/caption/report writes. |
| R2 | High | `delivery.prepare` checks an extensionless requested path, but the exporter appends `.mp4` and replaces an existing master; publishing then checks the wrong filename. | Resolve the exact exporter path before collision checks, cache handling and sidecar naming. |
| R3 | High | Delivery overwrites an existing subtitle, manifest or QA report. Its legacy exporter sidecar can overwrite an unrelated SRT before the per-track sidecars are written. | Precheck delivery outputs; exclusively create SRTs/manifests; omit the legacy sidecar. QA reports use numbered revisions and return `reportPath`. |
| R4 | High | Delivery approves a missing original wrapped in a subclip; nested sequences are not traversed. | Bounded dependency traversal checks referenced items, missing files, offline flags, cycles and nested sequence bounds. |
| R5 | High | An original changes after a preview was cached: delivery/privacy can inspect stale preview pixels. | Refresh referenced media and inspect full-resolution originals; measured colour and audio continuity also use originals. |
| R6 | Medium | Assigning audio stems replaces Essential Sound wholesale, losing mute, repair, gain and ducking treatment. | Update the role/type while preserving processing and mute settings. |
| R7 | Medium | Portrait subclips receive a fallback 1920×1080 motion anchor. Auto colour rejects subclips or uses incorrect pixel dimensions for screen masks. | Resolve the parent media and interpreted source size for placement, measurement and masks. |
| R8 | Medium | A rejected atomic edit restores the content/history but makes a saved project dirty. | Restore the saved-state marker after rollback while preserving revision invalidation. |
| R9 | Medium | A corrupt cache-hit copy leaves a partial final output; a hit into a missing parent folder fails. | Stream to a synced temporary file, verify size/digest, then publish without replacement; create parent directories. |
| R10 | Medium | Invalid cache sequence parameters silently fall back to the active sequence. An unused missing editing proxy blocks original-media caching. | Validate explicit identities and exclude editing proxies from original-media render identity. |
| R11 | Medium | Different catalogue scenes from one original request retrieval again, colliding with the retained original. | Freshly resolve each citation and reuse verified retained bytes for the same asset/source filename; preserve distinct trims and separate multipart files. |
| R12 | Medium | A completed RAEA render disappears when native probe or QA raises an exception. An invalid expected-spec shape fails only after rendering. | Retain the finished asset before probe/QA, validate the spec first, and emit no import manifest on failure. |
| R13 | Medium | Slow audio-device opening accumulates unbounded pending scrub mixers during dragging. | Bounded wake channel and coalesced latest grain/configuration; release acknowledgements are bounded too. |
| R14 | Medium | A disabled still or static background exempts a frozen live video from delivery motion warnings. | Exempt only sampled intervals whose enabled picture clips are all static and have no overlapping transition. |

## Evidence and validation

Evidence lives under the gitignored `target/full-review-2026-11-11/` directory.

- Initial engine regression run reproduced eight failures; the project-alias case was also
  reproduced through the installed native CLI using an owned synthetic project.
- Installed native CLI reproductions confirmed original-frame replacement and the extensionless
  delivery/master replacement (`legacy-extra-reproductions.json`).
- Initial RAEA recovery tests reproduced three failures. All three now pass without a GPU or
  external renderer (`python3 -m unittest discover -s examples/raea -p test_render.py`).
- Fifteen engine regressions, one catalogue mock-service regression and one bounded-scrub
  regression cover the fixes. Existing audio release tests also cover delayed opening/closing.
- The final `cargo xtask ci` completed successfully: 6,239 tests passed, zero failed and 89 ignored
  across 418 suites. The engine contributed 1,242 passing tests. Formatting, warnings-as-errors
  Clippy, layering, attribution for all 93 assets and all 57 WebAssembly crate checks passed
  (`ci.log`, `validation-summary.json`). The separately printed translation report is excluded
  from the test count. An initial Clippy attempt caught a collapsible conditional; it was fixed
  before the complete green run.
- Native candidate safety checks pass for project aliases, source-frame/report destinations,
  extensionless delivery, existing subtitles and numbered QA report retries (`native-safety-proof.json`).
- All 914 commands remain available; eight AI/model catalogues match the installed version.
  The only command-schema change is the optional `captions.export.overwrite` parameter
  (`compatibility.json`).
- Seven sampled frames from two saved Chameleon/FX projects match the installed renderer exactly;
  both saved files remain unchanged. Original screen pixels stay within one channel level while
  the unprotected region receives the requested correction (`native-proof.json`).
- Real native 3840×2160 and 2160×3840 exports pass, including horizontal sidecars, portrait burned
  captions, two aligned language tracks, byte-identical cached output, High 10 Intra confirmed by
  external ffprobe, and native MP3 reimport (`native-proof.json`, `native-short-proof.json`).
- The isolated desktop smoke test passes LUT export and caption-track selection; screenshots were
  visually inspected. Five playhead commands took 0.09 seconds (`desktop-proof.json`).

## Practical limits

- QA is sampled evidence: it cannot prove every frame, face, pronunciation or device screen is
  correct. Full-resolution inspection and listening remain part of production review.
- No production Photos/YouTube upload, cloud retrieval, model download or GPU generation is part
  of this review. Catalogue integration uses a local mock service; the other catalogue session
  keeps its GPU lock.
- Runtime validation targets this Apple Silicon Mac. The complete gate includes portable tests
  and WebAssembly checks; native Windows/Linux device drivers are not exercised here.
- Existing UI translation gaps still fall back to English, as recorded by the gate's translation
  report. Completing all interface-language translations is outside these correctness fixes.
- Destination guards cover the engine movie/frame/caption/report paths reviewed here. They are
  not a replacement for OS permissions or a general file-system sandbox, and do not claim an
  exhaustive audit of every low-level writer or numbered image-sequence destination.

## Completion checklist

- [x] Reproduce and fix confirmed findings; distinguish hypotheses from bugs.
- [x] Review high-risk paths and record practical limitations.
- [x] Full workspace gate and native 4K/editing workflows pass.
- [x] Commit, publish to AutoCut and install the verified build with rollback.

## Delivery record

Release source: [`14dfc06d`](https://github.com/moheshmohan/AutoCut/commit/14dfc06dd0703d6b73abe3ccf3a7d7a8fa32d22a)
(AV21). It was pushed atomically to `autocut/main` and `autocut/feat/autocut`; both refs were read
back and verified. Subsequent review-record changes are documentation only.

The macOS package carries that source commit and bundle id `com.moheshmohan.autocut`. App, CLI and
DMG signatures passed verification; packaging uses the established local ad-hoc signing workflow.
The packaged desktop passed a second isolated smoke test. RAEA and FFmpeg are not bundled.

Installed app: `target/macos-package/AutoCut.app`; installed CLI: `target/release/filmcraft-cli`.
Rollback copies: `target/macos-package/AutoCut-before-AV21-2026-11-11.app` and
`target/release/filmcraft-cli-before-AV21-2026-11-11`.

The existing EXEED saved session reopened with its original file hash, active sequence 2074,
Editing workspace and playhead 7256390400000. It remains clean. The installed app exposes all
914 commands, the RAEA image-sequence FPS parameter and the voice catalogue. No production media
or project edits are part of this review (`installed.json`, `session-before-install.json`).
