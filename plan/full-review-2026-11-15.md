# AutoCut review — 15 November 2026

Scope: everything added since `main` (`ea52bcff`), which is AV17–AV22 at `6ba4c4fb`: 266 files,
about 19,800 added lines. Reviewed on `codex/full-review-2026-11-11`. An earlier pass over part of
this work is in [full-review-2026-11-11.md](full-review-2026-11-11.md).

## How it was reviewed

Read in full: the new engine modules (`render_cache`, `delivery`, `audio_continuity`,
`animation_preflight`, `catalog_edit`, `source_check`, `output_safety`), every changed engine
source file, the CLI's `catalog` and `project_lock`, the desktop app's scrub audio output, the
RAEA helper, and the media reader, auto-save and MCP server changes.

Skimmed only: the ported encoders (`h264enc`, `mp3enc`), which upstream tests against external
decoders, and the smaller upstream ports in `render`, `interchange`, `captions`, `project` and
`tts-text`. A defect in those would not have been found by this review.

The new commands were then run, read-only, on four saved projects with the installed CLI.

## Findings

| # | Severity | What was wrong | Outcome |
|---|---|---|---|
| 1 | High | **The delivery gate failed finished edits.** `sequence.checkContinuity` reports any 100 ms loudness change over 6 dB at a clip boundary, and `delivery.preflight` failed on every one. The South Korea Expanded Final has 267 of them in 702 boundaries, the EXEED edit 89 in 257, none a click. `delivery.prepare` would have refused both. | Fixed. A click (a step between neighbouring samples) fails; a loudness jump alone is a warning. |
| 2 | High | **Edits with more than 2000 clip boundaries failed the gate unmeasured.** The check refuses more than 2000 boundaries, and the gate had no way to pass a selection. | Fixed. The gate measures 2000 boundaries at a time. |
| 2a | High | **Two caption tracks cut into different cues failed the gate.** The South Korea Expanded Final carries 249 English and 338 Malayalam cues; the gate required every track to match the first cue for cue. A translation is rarely cut where its source is. | Fixed. A difference is a review warning that gives each track's cue count and last cue end. |
| 3 | High | **The render cache refused large projects.** The cache identity held every selected item's JSON and is capped at 8 MiB. The EXEED project's identity was 8,303,281 bytes of the 8,388,608 allowed. One more edit, or a larger project, and `render.cachePlan` fails after hashing every source. | Fixed. The identity holds each item's digest. |
| 4 | High | **A cache failure after the export failed the delivery.** With no cache folder, a full cache disk or a held lock, `delivery.prepare` exported the master, failed on `render.cacheCommit`, wrote no subtitles or manifest, and refused the retry because the master existed. A plan that could not be made stopped the delivery before it started. | Fixed. The delivery goes ahead and says why in `cache.note`. |
| 5 | Medium | **A failed catalogue download could not be retried.** The failed job stayed in the receipt and was polled again on every run. Separately, a successful poll of a failed job was treated as a failed call, so the code that handles a failed job never ran. | Fixed. The job is forgotten and the next run asks again. |
| 6 | Medium | **Deleting the "Continuous Music B" track blocked every later music bed** ("saved music partner track is missing"). | Fixed. A new partner track is made. |
| 7 | Low | **A failed RAEA run kept its empty revision folder**, so the same name failed on the next run. | Fixed. |
| 8 | Low (build time) | **Editing the UI or automation crate rebuilt the engine.** The engine's build script hashed and watched every crate, including the seven built on top of it. | Fixed. Those crates are left out of the hash; nothing in them can change what the engine renders. |
| 9 | Medium | After the export, any other failure in `delivery.prepare` (a subtitle export, the publish pre-flight, the manifest) still leaves the master without a manifest, and the retry is refused. | Open. Needs a decision: resume an incomplete delivery, or remove the master on failure. |
| 10 | Medium | `tts.checkClip` hears a clip's own range but compares it with the narration's whole text, so a trimmed or razor-split narration clip reads as unclear and fails the gate. | Open, not reproduced. |
| 11 | Low | Every `render.cachePlan` hashes every source file in full (14 s for the EXEED project's 101 sources), and a delivery does it twice. | Open. A size-and-time shortcut would weaken the content guarantee. |
| 12 | Low | `clip.autoColor` ignores `protectScreens` entries for clips it is not grading. | Open. |
| 13 | Low | `render.cacheCommit` leaves one `<key>.lock` file per cache key. | Open. |

Findings 1 and 2a change what the gate blocks, which is a product decision as much as a bug fix.
Both checks failed a finished, delivered edit. If a loudness jump or a caption difference should
block delivery after all, the places are `continuity` and the `subtitleAlignment` check in
`crates/engine/src/delivery.rs`.

## Checked and found sound

- Appending new effects after the intrinsic ones (`effects.apply`, presets, Lumetri presets): the
  renderer filters standard effects from the list, so their order relative to Motion does not
  change the picture.
- `sequence.continuousMusic` writes its fade keyframes in source time, which is what clip
  keyframes use.
- `catalog cancel` sends the job id as text; FrameAtlas converts it.
- The scrub output's coalesced queue cannot lose a wake-up or keep more than one pending grain.
- Added non-test code has no `unwrap`, `expect`, `panic!` or `unsafe`.
- The render cache folder is inside the Media Cache, so the existing age and size policy and
  `mediaCache.clean` cover it.

## Evidence

Before the fixes, with the installed CLI (`target/release/filmcraft-cli`, AV21), read-only:

| Project | Check | Result |
|---|---|---|
| South Korea - Expanded Final (4K) | `sequence.checkContinuity` | 702 boundaries, 267 findings, 0 clicks |
| EXEED RX - natural edit v04 | `sequence.checkContinuity` | 257 boundaries, 89 findings, 0 clicks |
| Halloween Reel | `sequence.checkContinuity` | 30 boundaries, 0 findings |
| EXEED RX - natural edit v04 | `render.cachePlan` | 14 s, identity 8,303,281 of 8,388,608 bytes |

After the fixes, with a development build of the CLI, read-only:

| Project | Check | Result |
|---|---|---|
| EXEED RX - natural edit v04 | `render.cachePlan` | identity 25,328 bytes (103 item digests, 101 sources) |
| South Korea - Expanded Final (4K) | `delivery.preflight` with privacy and narration off | audio continuity: warning (0 clicks, 267 jumps, 702 boundaries); mix: pass (true peak −1.8 dBTP); sources: pass |

That preflight run took 8 min 52 s for the hour-long 4K edit (it samples 600 frames from the
originals). It ran before the caption fix, and its only failing check was `subtitleAlignment`,
which is what led to finding 2a. It was not run again after that fix; the new test covers it.

## Validation

- New regression tests, each written for the case that failed (they were not run against the old
  code): `a_large_edit_still_has_a_small_identity`,
  `long_edits_are_measured_whole_and_only_clicks_block`,
  `differently_cut_caption_tracks_are_a_review_warning`,
  `delivery_goes_ahead_without_a_cache_folder`, `a_deleted_partner_track_is_made_again`,
  `a_failed_retrieval_is_asked_for_again_on_the_next_run`,
  `crates_built_on_the_engine_are_found_through_one_another`,
  `test_failed_render_frees_its_revision_name`.
- Touching `crates/ui-egui/src/lib.rs` no longer recompiles the engine; touching
  `crates/render/src/lib.rs` still does.
- `cargo xtask ci --changed --base HEAD` passed: formatting, Clippy with warnings as errors on the
  engine, the CLI and the nine crates built on them, 1,332 tests (1,248 in the engine, none
  failed, 9 ignored), layering, the 93 asset attributions and the 57 WebAssembly crate checks.
- The full `cargo xtask ci` passed on `4ebe1d80`, which is these fixes (AV23) plus the upstream
  ports AV24–AV33: formatting, Clippy with warnings as errors on the whole workspace, 6,333
  tests (none failed, 89 ignored), the translation report, layering, the 93 asset attributions
  and the 57 WebAssembly crate checks.

## Not done

- No production upload, model download or GPU generation.
- Findings 9–13 are open.

## After the review

- The fixes are commit AV23. The upstream survey is in
  [upstream-sync-2026-11-15.md](upstream-sync-2026-11-15.md); its 32 commits worth taking were
  ported as AV24–AV33.
- The app and the CLI were rebuilt from `4ebe1d80` and installed on 15 November. The build
  before it is kept as `target/macos-package/AutoCut-before-AV33-2026-11-15.app` and
  `target/release/filmcraft-cli-before-AV33-2026-11-15`. Both builds list the same 914 commands
  with the same parameters.
- With the installed CLI, read-only, on EXEED RX - natural edit v04: `sequence.checkContinuity`
  reports 89 findings, none a click, and the `render.cachePlan` identity is 25,328 bytes.
