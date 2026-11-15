# Upstream triage — 15 November 2026

Written as a survey, before anything was ported. What was integrated since is under
[Results](#results) at the end.

Upstream `storytold/filmcraft` (`origin`) is at `e6df3186` (v0.20.1). The last reviewed upstream
commit was `25ceb25f` (AV20, [pipeline-upgrades-2026-11.md](pipeline-upgrades-2026-11.md) U1).
There are 68 non-merge commits in `25ceb25f..origin/main`. AutoCut is at `6ba4c4fb`.

"Applies" is `git cherry-pick -n` of each commit alone on `6ba4c4fb`, in a scratch worktree.
"Clean" means textually clean: nothing was compiled or tested. All TAKE and CARE commits were
also stacked in upstream order with no new conflicts.

Totals: **26 TAKE**, **7 CARE** (worth having, needs a real port), **3 ALREADY**, **32 SKIP**.

## What the fork would gain

| Area | Commits | What it closes or adds |
|---|---|---|
| Crash and hang on hostile files | `161b8f55`, `5f37739c`, `e6df3186`, `1709a321`, `0e848080`, `cdf05234`, `4bec22bd`, `824a3258`, `4c0d2ad7`, `32308121` | Deeply nested FCPXML/FCP7 overflows the stack (a crash); a giant `.fcproj` is read whole before any check; APNG/HDR/EXR allocate before their limits; an AV1 chunk can show hundreds of 4K pictures; a seek can read samples one by one, uncancellable; remix planning can loop without end. |
| Files written or deleted where they should not be | `ca4560be`, `e33f1da5`, `9c436d27` | `graphics.template.remove` deletes any template-shaped file a path names; Project Manager writes through dangling links and copies link targets; narration edits and Edit Original follow a stored network path. |
| Memory | `a9251b55`, `617e0aa7`, `0ba402b4` (CARE), `0d8eea99` (CARE) | Render Audio builds the whole WAV in memory; absurd mixer latency allocates without bound; the GOP cache stays over budget on long-GOP 4K; project-driven audio buffer sizes are unchecked. |
| Wrong results | `0ed546f8`, `a02bec68` (CARE), `82f22cfd`, `ff7fc8d4` (CARE), `9599eb50` (CARE) | Resampled audio differs with where a read starts (matters for chapter renders and the render cache); APV 4:4:4 exports carry 4:2:2 chroma; a proxy, scene-detect or voice-over job that finishes after another project opens writes into it; a wrong stored media width multiplies blur radii. |
| Editing and UI | `ed1c2c3d`, `16cbb5af`, `7ac54e97` | Reveal in Project scrolls to the clip; right-click reaches a narrow gap's menu. |
| Export | `3ed8caab` (CARE), `046d6b0b` | The export says which encoder ran and warns when hardware encoding fell back to software; OpenEXR ZIP is faster. |
| Tests only | `3bcd485b`, `a6c0f459`, `b42c5258`, `c0efc301` | Numbered WebP sequence import (the RAEA path), TIFF oracle fixtures, comments. |
| Low priority | `70dd1a53` (CARE) | Default screenshot, log and voice-over folders in a private temp folder. |

## Every commit

| commit | subject | decision | applies | note |
|---|---|---|---|---|
| `decb070b` | dev_icon: MimeType line | ALREADY | clean, empty | fork has it as `79f5036c` |
| `619b3609` | CI: xtask ci on PRs | ALREADY | conflict | fork has it as `b787c150` |
| `4c22bb01` | CI: no-LTO test binaries | ALREADY | clean, empty | fork has it as `28fcaddd` |
| `0d4a48a7` | CI: GPU tests on lavapipe | SKIP | clean | CI |
| `ff7fc8d4` | engine: size proxy / scene-detect work from the media | CARE (medium) | conflicts in `proxies.rs` | keep the fork's `pub(crate) check_proxy` and its `.lrv` tolerance; drop the deferred job-queue context |
| `a9251b55` | engine: stream Render Audio to disk | TAKE | 1 hunk in `previews.rs` | keep the fork's `MixStream` argument, alive across segments |
| `9599eb50` | render: cap effect scale from a wrong stored width | CARE (small–medium) | 4 context conflicts | five fork-only `FxCtx` initialisers need `mask_scale`; GPU mask ops must use the uncapped scale (redaction masks) |
| `046d6b0b` | export: EXR ZIP reuses deflate tables | TAKE | clean | output bytes unchanged |
| `824a3258` | audio-dsp, aac, dng: bound loops and copying | TAKE | 1 hunk in `dng/writer.rs` | hand-apply or drop the DNG hunk |
| `b42c5258` | export: deflate test comment | TAKE | clean stacked | after `046d6b0b` |
| `cdf05234` | av1: refuse absurd shown-picture counts | TAKE | 3 layout hunks | mechanical port; the last in-tree AV1 fix upstream made |
| `3fec758c` | Release v0.20.0 | SKIP | conflict | version |
| `c0efc301` | export: deflate test comments | TAKE | clean stacked | after `046d6b0b` |
| `617e0aa7` | render: refuse absurd mixer latency | TAKE | test adjacency | |
| `4c0d2ad7` | audio-dsp: validate stored remix plans | TAKE | clean | a stored plan that breaks the rules is dropped on load: check saved projects that use Remix |
| `c3d98bce` | ci: Harden-Runner | SKIP | conflict | CI |
| `70dd1a53` | private temp folder for screenshots, logs, voice-over | CARE (small–medium) | 1 hunk in `narration.rs` | `media_dir` becomes fallible: `generated_dir` and five callers propagate |
| `af5e1348`, `2bb0273d` | craft-update pin | SKIP | conflict | updater |
| `3285bbf7` | tests: private temp folders | SKIP | conflicts | test hygiene, adds `tempfile` |
| `4bec22bd` | codecs: bound the random-access search | TAKE | product clean | port the test into the inline `mod tests` |
| `7a27c230` | Flatpak updates, 0.20.1 | SKIP | conflict | packaging |
| `0d8eea99` | audio: bound project-driven buffer sizes | CARE (medium) | 7 hunks | real clash with the fork's `MixStream`, time remap and pitch-stretch paths |
| `0ba402b4` | codecs: evict interior GOP cache frames | CARE (medium) | 4 hunks in `gop.rs` | port the eviction rule together with "the caller keeps the frame it decoded" |
| `e42e2c91` | CI: nightly release dry run | SKIP | clean | CI |
| `db60ee45`, `504fe9fc`, `f7db6cdb` | codecs, containers, captions, time from craft-libs | SKIP | conflicts (about 190 files) | relocation: would delete the fork's in-tree crates |
| `e33f1da5` | Project Manager: dangling links, no links followed | TAKE | clean | |
| `ca4560be` | graphics template removed only via a library entry | TAKE | test adjacency | |
| `32308121` | tts-text: cap dictionary pronunciations and phonemes | TAKE | product clean | English phonemizer only; Malayalam untouched |
| `82f22cfd` | bind ingest jobs, scene detect, voice-over takes to their project | TAKE | clean after `ff7fc8d4` | the fork already has `document_epoch` |
| `9c436d27` | stored-path policy in narration edits, Edit Original, Update Metadata | TAKE | clean | one line in fork-heavy `narration.rs` |
| `0e848080` | media: decode limits before allocating | TAKE | clean | on the image-sequence import path |
| `5f37739c` | interchange: bound diagnostics, AAF, DTD, OMF | TAKE | 1 hunk in `fcp7.rs` | |
| `a6c0f459` | media: TIFF oracle fixtures | TAKE (optional) | clean | test only |
| `a02bec68` | export: APV 4:4:4 carries full chroma | CARE (small) | fork export is one file | rewrite the branch with the fork's `to_yuv444_10` |
| `0ed546f8` | codecs: resampled audio independent of read start | TAKE | clean | audio will not be bit-identical to earlier builds for sources off the sequence rate |
| `cb9783ba` | CI (Windows) | SKIP | conflicts | fold its `private_temp_dir` lint hunk into `70dd1a53` |
| `ff8046f5` | craft-libs pin (HEVC 4:2:2) | SKIP | conflict | relocation |
| `3ed8caab` | export: say which encoder ran, warn on fallback | CARE (medium) | 7 files | fork-only `CaptionSei` and `ParallelEncoder` must report `hardware()` or VideoToolbox exports read as "software"; about ten `Report { … }` literals need `..Default::default()` |
| `180ab6bb`, `08ada45d`, `36652749`, `fe744cbc`, `06e32b24`, `3c564b9c`, `53836ee9`, `c27afcf2`, `221f0425` | VP8, EDL, MP3, ProRes, MJPEG from craft-libs | SKIP | conflict | relocation and its follow-ups |
| `ec0ab982`, `fbc689cd` | updater: release notes through the desktop host | SKIP | file absent | updater |
| `a1f5697c`, `4d82e518`, `f7ee663c`, `edcc4ee1` | CI and coverage | SKIP | | CI |
| `8dc97adb`, `c504ddae` | CLI test executable from nextest | SKIP | conflict | test infrastructure |
| `3bcd485b` | test: numbered WebP sequence import | TAKE | clean | test only |
| `21be72b5` | UI test waits for queued proxy jobs | SKIP | clean | needs the deferred job queue |
| `356cc274`, `adf97f6c` | docs: ProRes | SKIP | conflict | docs for skipped work |
| `ed1c2c3d` | Reveal in Project scrolls to the clip | TAKE | struct context | add the `reveal` field and set it in `lib.rs` |
| `16cbb5af` | Reveal brings back a sideways-scrolled list | TAKE | clean stacked | after `ed1c2c3d` |
| `161b8f55` | interchange: bound XML nesting and node counts | TAKE | clean stacked | after `5f37739c` |
| `1709a321` | cap project files at 256 MiB | TAKE | 2 hunks | the largest `.fcproj` under `~/Movies/AutoCut` is 30.6 MB |
| `7ac54e97` | context menus target narrow gaps | TAKE | file layout | hand-apply to `panels/timeline.rs` |
| `e6df3186` | interchange: retry FCPXML media cut by layout depth | TAKE | clean stacked | after `161b8f55` |

## Suggested order, each batch validated on its own

1. **Leaf-crate hardening**: `824a3258`, `4c0d2ad7`, `cdf05234`, `32308121`, `0e848080`, `a6c0f459`
   (audio-dsp, aac, av1, dng, tts-text, media). Their tests plus `cargo xtask wasm`.
2. **Interchange**: `5f37739c`, `161b8f55`, `e6df3186`.
3. **Export, simple**: `046d6b0b`, `b42c5258`, `c0efc301`, then `a02bec68`. An EXR ZIP byte-compare
   and an APV 444 round trip.
4. **Engine safety**: `e33f1da5`, `ca4560be`, `9c436d27`, `1709a321`, `a9251b55`, `3bcd485b`.
5. **Engine jobs**: `ff7fc8d4`, then `82f22cfd`.
6. **Codecs**: `4bec22bd`, `0ed546f8`, then `0ba402b4`. A decode check on a long-GOP 4K file.
7. **Render and audio**: `617e0aa7`, `9599eb50`, `0d8eea99`. Repeat the saved-project frame
   comparison and an audio round trip.
8. **Encoder reporting**: `3ed8caab`. A real VideoToolbox export and a forced fallback.
9. **UI**: `ed1c2c3d`, `16cbb5af`, `7ac54e97`.
10. **Low priority**: `70dd1a53`.

## Still deferred from before `25ceb25f`

- The updater (`87c8e669`, `f1a06dfe`, and now `af5e1348`, `ec0ab982`, `fbc689cd`, `7a27c230`).
- The app-wide naming change (`41977bab`); the graphics start and adapter ladder
  (`a65d1e97`, `0dba3b1e`, `2d536e6a`); egui 0.37, docking, the IME composer, craft-ui.
- File splits and strict-Clippy rewrites. Nine of this range's ports conflict only because the
  fork keeps export, timeline, app and test files unsplit.
- The job queue (`42b3f83a`) and jobs reading their own file copy (`bee5ce7c`, `9208a305`).
- The H.264 encoder search batch appears closed: the fork's `crates/h264enc` differs from
  upstream at `25ceb25f` by ten added lines.

## The relocation changes the next sync

Upstream has now moved aac, ac3, flac, mpa, ogg, opus, apv, av1, avi, h264, hevc, mpeg2v, mpegts,
mxf, vp8, vp9, prores and mp3enc out to `storytold/craft-libs`, and thinned captions and time.
The fork keeps these crates in-tree, so fixes to them will no longer appear in
`storytold/filmcraft`. The next sync has to triage `craft-libs` as well. `cdf05234` and
`824a3258` are the last such fixes visible in this repository.

The next upstream review starts at `e6df3186`.

## Results

Integrated on 15 November 2026 on branch `claude/upstream-sync-2026-11-15-ab0269`, cut from
`6513574b` (this triage, on top of AV23 `68c8a862`). Upstream commits were applied with
`git cherry-pick -n` where they applied and by hand where the fork's layout differs; each batch
is one commit whose message lists the upstream hashes. Nothing was merged from `origin`, and
nothing was pushed.

All 32 TAKE and CARE commits are in: the 25 TAKE and 7 CARE rows of the table above. (The totals
at the top say 26 TAKE and 32 SKIP; the table has 25 and 33. `cb9783ba` is a SKIP row whose one
`private_temp_dir` hunk is folded into AV33.) None was deferred.

| Commit | Batch | Upstream commits |
|---|---|---|
| `296c1668` AV24 | Leaf-crate hardening | `824a3258`, `cdf05234`, `4c0d2ad7`, `32308121`, `0e848080`, `a6c0f459` |
| `68ddfa6e` AV25 | Interchange | `5f37739c`, `161b8f55`, `e6df3186` |
| `8a7c8ee1` AV26 | Export | `046d6b0b`, `b42c5258`, `c0efc301`, `a02bec68` |
| `f967a76f` AV27 | Engine safety | `a9251b55`, `e33f1da5`, `ca4560be`, `9c436d27`, `3bcd485b`, `1709a321` |
| `210be5bb` AV28 | Engine jobs | `ff7fc8d4`, `82f22cfd` |
| `c99b280f` AV29 | Codecs | `4bec22bd`, `0ed546f8`, `0ba402b4` |
| `2dfec023` AV30 | Render and audio | `617e0aa7`, `9599eb50`, `0d8eea99` |
| `fab647de` AV31 | Encoder reporting | `3ed8caab` |
| `e1cd6d43` AV32 | UI | `ed1c2c3d`, `16cbb5af`, `7ac54e97` |
| `29eb83b5` AV33 | Low priority | `70dd1a53` (with the `private_temp_dir` builder hunk of `cb9783ba`) |

### Parts of a commit that were not taken

- `824a3258`: the DNG writer's `write` depth check and its `should_panic` test. They need
  `try_write`, which came to upstream with lossy-JPEG DNG writing and is not in the fork. `pack`
  refuses a depth outside 1..=16 and the tag merge is one pass, both with their tests.
- `ff7fc8d4`: the job-queue context (the queue ticket, and the worker opening its own copy of the
  file). The job queue (`42b3f83a`, `bee5ce7c`, `9208a305`) stays deferred; the source duration is
  read from the source the job already holds.
- `82f22cfd`: its one-line change to `crates/ui-egui/src/app/playback/audio_recovery_tests.rs`, a
  file the fork does not have.
- `3ed8caab`: the two `log` lines in the export crate (`log::info!` of the encoder, `log::warn!`
  of the fallback). The fork's export crate does not depend on `log`, and adding it would change
  `Cargo.lock`. The report, the Events panel and the CLI still carry the warning, and the platform
  crate already logs why a hardware encoder declined.
- Tests that sat next to an upstream hunk but belong to commits the fork does not have were left
  out: `a_huge_link_group_does_not_write_quadratic_links` (FCP7), `gradient_stop_keeps_its_alpha`
  (graphics templates), `demo_audio_has_no_hiss` and `tone_phase_is_exact_late` (generators),
  `gpu_adjustment_step_matches_cpu_between_layers` and the readback tests (GPU),
  `transform_with_its_own_shutter_angle_stays_on_the_cpu` (gpufx).

### How the ports were adapted

- **AV24** `cdf05234`: placed by hand in the fork's AV1 decoder, which keeps
  `Output::Existing { … }` as a struct variant and `Slot<Arc<Picture>>`. `32308121`: tests moved
  into the inline `mod tests`.
- **AV25** `5f37739c`: the fork keeps FCP7 `clip_item` in one function, so the offline
  placeholder is a small `offline_media` helper beside it. `161b8f55` came without an FCP7 test;
  `deeply_nested_bins_stop_at_the_container_depth_limit` covers the container limit.
- **AV26** `a02bec68`: rewritten in the fork's `export/src/lib.rs` with its
  `to_yuv444_10(…, kr, kb, …)` and the in-tree `filmcraft_apv`; a short picture buffer is an
  encode error.
- **AV27** `a9251b55`: the fork's `MixStream` is still made once per job and passed to every mix
  call, so it lives across segments. `1709a321`: `open_project` keeps `expand_home`, and the
  bounded read sits on the fork's `open_regular_file`.
- **AV28** `ff7fc8d4`: `pub(crate) check_proxy(s, item, path)` keeps its signature and its `.lrv`
  tolerance (`PROXY_SHORT_FRAMES`); the source-duration check is `check_proxy_against`.
- **AV29** `0ba402b4`: both halves together. The fork has no raw-frame `Taken` type, so the frame
  a decode keeps is a `(pts, Arc<VideoFrame>)` pair, and an exact request never takes a draft frame
  of its own decode. `0ed546f8`: its MXF test writes the fixture with the in-tree `filmcraft_mxf`.
- **AV30** `617e0aa7`: the fork's graph key carries the stream, so a refused graph is remembered
  by its structure hash. `9599eb50`: the five fork-only `FxCtx` initialisers have `mask_scale`;
  both GPU layer contexts in `plan.rs` cap `px_scale` as the CPU path does; the fork's GPU mask
  ops (`FxOp::Masked`, `OpacityMask`: redaction masks) are placed by the uncapped `mask_scale`.
  `gpu_mask_ops_are_placed_by_the_uncapped_scale` fails without that (the mask edge lands at 8
  working pixels instead of 128). `0d8eea99`: `read_speed` (the stored speed held to 100×) is the
  one speed every path maps media with: varispeed, the media span of Time Remapping, and the
  transition handles. Maintain Audio Pitch takes 1/16× to 16× only, so the stretch and its origin
  never see a capped speed. `mix_item` keeps the stream arguments and takes the stored clip as
  `timing`; the Vorbis and audio-file sources keep their PCM cache. Non-finite stored speeds read
  at the cap.
- **AV31** `3ed8caab`: the fork-only `CaptionSei` answers `hardware()` for the encoder it wraps
  and `ParallelEncoder` answers `VideoToolbox`. The fork's own `Report { … }` literals (16 files
  in engine, export and the UI tests) take `..Default::default()`.
- **AV32**: `ed1c2c3d` sets the reveal request in `ui-egui/src/lib.rs`; `7ac54e97` is applied to
  the unsplit `panels/timeline.rs`. Upstream's Reveal test reads the selection highlight's own
  automation element (`project.item.<id>.selected`, from `ef802a4c`, which the fork does not
  have). Here it reads the row or card the fork registers: the scroll is checked the same way, and
  the selection is asserted beside it.
- **AV33** `70dd1a53`: `generated_dir` is fallible like `media_dir`, and its five callers
  (narration, neural music, elements, two in generated audio) propagate the error.

### Checks

- Each batch passed `cargo xtask ci --changed --base HEAD` before its commit: `cargo fmt --check`,
  Clippy with `-D warnings` on the changed crates and their dependents, the changed crates' tests,
  `layers`, `assets`, and `wasm` where an L0–L4 crate was touched.
- `cargo xtask ci --changed --base 6513574b` passes at `29eb83b5`, the last code commit: the 18
  changed crates' tests (232 suites, 3,754 passed, none failed, 41 ignored), Clippy on them and
  the 38 crates that depend on them, `layers`, `assets` (93 attributed) and `wasm` (57 crates).
- One test is timing-dependent and is not part of this work:
  `catalog::retrieval_tests::review_catalogue_scenes_reuse_one_verified_retained_original`
  (`apps/filmcraft-cli/src/catalog.rs`, untouched here) failed once in a repeat of the whole-range
  gate and passes on a rerun; run alone it failed 2 times in 40. See the follow-ups.
- Builds used `CARGO_TARGET_DIR=target/agent-upstream-sync` in this worktree, with the release
  profile's LTO off and 16 codegen units, as the earlier syncs did. The full `cargo xtask ci` was
  not run here: it is the main session's, after the merge.
- Command ids: the 914 ids of `filmcraft-cli commands --json` at `6513574b` are all present at the
  tip, and none was added.
- Test names: every `#[test]` function name at `6513574b` (6,248) is still there, and 87 were
  added (6,335). One name changed, upstream's own rename with the behaviour:
  `file_links_count_and_a_too_deep_folder_is_an_error` is now
  `file_links_are_skipped_and_a_too_deep_folder_is_an_error` (`e33f1da5`).
- Encoder reporting on this Mac, exporting the built-in demo project: Hardware encoding Auto
  reports `encoder: "VideoToolbox"` with no warning; Auto with two-pass VBR reports `"software"`
  and `hardware encoding declined (VideoToolbox: two-pass encoding); the software H.264 encoder
  ran`, also on stderr; Off reports `"software"` with no warning.
- ffmpeg and ffprobe are not installed, so the external-oracle tests skip, as before.

### Still to do before a push

- The one full `cargo xtask ci` on the merged result, then the user's OK. Push only to `autocut`.
- Not done here, because they need the user's projects and media under `~/Movies/AutoCut`, which
  this work did not touch: the saved-project frame comparison and an audio round trip (batch 7),
  a decode check on a long-GOP 4K file (batch 6), and opening saved projects that use Remix
  (`4c0d2ad7` drops a stored plan that breaks the rules on load).
- Expected differences: audio of sources off the sequence rate is not bit-identical to earlier
  builds (`0ed546f8`), so such audio rendered before this sync (a chapter render, a cached
  render) will not match the same audio rendered after it; a clip stored above 100× reads its
  audio at 100×.

### Follow-ups noticed, not changed

- Fixed-name fallbacks in the shared temp folder remain where upstream left them or where the code
  is the fork's own: `FilmCraft Previews` (`previews.rs`), `FilmCraft Feedback` (`feedback.rs`) and
  `filmcraft-video-templates` (`video_templates.rs`). `private_temp_dir` now exists for them.
- The Project panel registers no automation element for a selected row or card
  (`project.item.<id>.selected`, upstream `ef802a4c`), so an agent reads the selection from the
  session state, not from `ui.elements`.
- The flaky catalogue test above: its local HTTP server listens non-blocking, and on macOS the
  accepted socket inherits that, so its first `read_line` can return `WouldBlock` before the
  client has written (`catalog.rs:381`); the read timeout does not apply to a non-blocking socket.
  `stream.set_nonblocking(false)` after `accept` would settle it. It can fail the full gate too.
- The next sync also has to triage `storytold/craft-libs` (see above).
