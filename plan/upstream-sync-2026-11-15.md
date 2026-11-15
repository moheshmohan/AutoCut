# Upstream triage — 15 November 2026

A survey, not a port: nothing in this file has been integrated yet.

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
