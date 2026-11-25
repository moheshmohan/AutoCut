# Upstream sync — 24 November 2026

Upstream `storytold/filmcraft` (`origin`) is at `ba6fe105`. The last reviewed upstream commit was
`009b72ca` ([upstream-sync-2026-11-22.md](upstream-sync-2026-11-22.md)). There are 8 non-merge
commits in `009b72ca..origin/main` (fetched 24 November; nothing newer) and two merges
(`e7345f6e`, `7584ca78`, pins only). AutoCut was at `18e6f49b` (PX4.5; local `main`).

This sync also triaged `storytold/craft-libs` for the first time: the shared repository upstream
moved its codecs, containers, captions, time and interchange code and its still decoders into
(4–16 November 2026). The fork keeps all of those in-tree, so fixes made there never reached it.

Totals: of the 8 upstream commits, **2 taken** (AV62, AV63) and **6 skipped** (three `craft-libs`
plumbing commits, three release-CI commits). The Russian catalog was taken as AV64. From
`craft-libs`, the crash / hostile-input fixes that apply to the fork's in-tree copies were ported
as **AV65–AV76** (12 commits), together with 15 older upstream FilmCraft fixes to the same crates
that an earlier sync had deferred. No `craft-libs` dependency was added to any `Cargo.toml` (AV76
only moves the existing `craft-docpath` pin).

## What the fork gains

| Area | Commits | What it adds or fixes |
|---|---|---|
| Speed (export, GPU) | `ba6fe105` (AV62) | With GPU Rendering on, an SDR export that needs nothing done to the picture after rendering (no alpha, no fitting or placement, no image, name or timecode overlay) reads back the GPU's finished 8-bit sRGB frame (an export-only LUT resolve pass, byte-for-byte `over_black_rgba8`) instead of the half-float accumulator converted on the CPU. Read-backs wait on their own submission instead of the shared device's latest. Measured here (below): 3 layers + effects, 1080p H.264 through VideoToolbox, 1.34 s → 0.99 s; CPU time per frame 20–21 ms → 9 ms in both H.264 scenes. |
| Speed (titles) | `117876b0` (AV63) | Responsive-design pins resolve their target layers once per layout: a 200,000-layer pin chain 1.80 s → 0.21 s. |
| Russian | upstream `ru.tsv` (AV64) | Russian covers the whole interface: 3,747 entries (upstream's 3,738 plus 9 AutoCut entries); strings shown in English 3,932 → 707 of 4,300. |
| Hardening (video) | craft-libs + FilmCraft (AV65–AV67) | H.264: picture order count and picture-number overflow, IDCT input range, temporal-direct vectors, slice QP, avcC validation, huge ue(v) fields, overlapping slices, at most 64 pictures per call, gray row instead of a panic without threads, 32-bit length overflows. HEVC: QP and weight ranges, hvcC validation, ue(v) sums, 64 pictures per call. AV1: tile-metadata copy budget, tile-count limits, unterminated LEB128, a worker panic is an error instead of a hang. VP9: a failed superframe drops its queued repeats. APV: absurd bit depths and QPs, alpha frame sized after its tiles. DNx: repeated or decreasing scan lines refused, 1 GiB frame cap. |
| Hardening (audio, containers) | craft-libs + FilmCraft (AV68–AV71) | AAC: non-finite spectra, a damaged in-band PCE no longer changes the layout, rate index past the table. FLAC: predictions past i32. WAV/AIFF headers: hostile channels and bits saturate. ISO BMFF: QuickTime v2 sound entries, file-wide sample budget, stsc-bounded reservation, PCM chunks over 16 MiB split. MXF: empty tracks, timecode overflow, repeated track references, 256 MiB metadata budget, AES3 reads only the frames asked. AVI: child chunks held to their parent. Matroska: frame budget, track-entry cap, content-encoding chain cap, stripped-header cap, 64 MiB resync budget, EBML header cap, laced blocks over 16 MiB skipped, warnings capped, timestamp overflow. Ogg: recovery budget for bad-CRC pages, packet-count cap, size overflow, warnings capped, granule range, range checks before allocating. |
| Hardening (stills, interchange, text) | craft-libs (AV72–AV76) | Targa: 768 MiB decode budget, fallible buffers. EXR: compressed chunks too short for their lines refused before the picture is allocated. Still export: fallible frame-sized buffers. OMF: depth-limited visits spend the budget, shared search budget. FCPXML and FCP7: nest cycles refused. Fonts: capped file reads, outline recording capped at 100,000 commands. DNG: lossless JPEG bounded by its data, per-chunk decode budget (shared tile bytes charged, overlaps still allowed), LinearizationTable count, `raw::unpack` sizes. `craft-docpath` pin: non-Unicode document paths keep the namespace rules. |

## Every upstream commit

| commit | subject | decision | applies | note |
|---|---|---|---|---|
| `ba6fe105` | export: GPU-rendered SDR exports read back the finished sRGB RGBA8 frame (#2119) | TAKE | conflicts in all 10 files; `registry.rs` absent | AV62, ported by hand (below) |
| `117876b0` | project: resolve pin target indices once per layout (#4904) | TAKE | three-way merge, clean | AV63 |
| `2a4c3ead` | media: use shared animated-image signature detection (#5159) | SKIP | — | craft-libs (`craft-sniff`). Not a bug fix: the fork's `Format::sniff` already needs the whole signature (`starts_with`, `get(8..12)`); upstream's new test passes on the fork's code (checked by reading; not added) |
| `17221b5d` | 778: use the shared writer for 16-bit RGBA TIFF stills (#5192) | SKIP | — | craft-libs (`craft-tiff`); its test pins the output to the `image` writer's bytes, which the fork still uses |
| `5ac81d0d` | 778: pin FilmCraft to one shared craft-libs revision | SKIP | — | pins only |
| `25979b15` | ci(release): forward ARTCRAFT_SERVICES_DISPATCH_TOKEN secret (#5201) | SKIP | — | release CI |
| `9b61bcee` | ci(release): publish scheduled nightly prereleases (#5207) | SKIP | — | release CI |
| `48d4c952` | revert: restore scheduled release.yml dry-run on main (#5209) | SKIP | — | release CI |

The two merges (`e7345f6e`, `7584ca78`) carry only pin changes (the latter also the `craft-sniff`
change of `2a4c3ead`).

## The Russian catalog (AV64)

As AV49 took Bosnian: a key-wise union on (context, source) of upstream's `ru.tsv` at
`origin/main` (3,738 entries; the file is the same at `009b72ca`) and the fork's (368). Upstream's
entries, order and header are kept, and for the 76 entries both had with different wording
upstream's translation is taken (283 were identical). The 9 entries only the fork had are
appended under a comment line: Quick Short…, Templates, Save as Video Template…, Recipes, Auto
Redact Faces and Plates…, Save Feedback Bundle…, Ken Burns, Lumetri Color, Lumetri Scopes. The
sidecar is upstream's (one line added about the AutoCut entries) and the `ATTRIBUTION.md` row takes
upstream's wording; `docs/localization-parity.md` gains a Russian row. The i18n tests (22,
including `catalogs_are_well_formed`) and `settings_ui` (15) pass.

## craft-libs triage

`storytold/craft-libs` was cloned read-only (at `249a41bd9`, 23 November) into the worktree's
`target/`. For each crate upstream FilmCraft uses, every commit after the crate's first (extraction)
commit was listed: **969 commits**. 515 were classified one by one (the full table, with the
fork file checked for each, is [upstream-sync-2026-11-24-craft-libs.tsv](upstream-sync-2026-11-24-craft-libs.tsv));
454 were set aside by crate (below). Four triage agents read the diffs and checked each
`crash` / `correctness` row they marked "applies" against the fork's code; three port lanes then
re-checked every item before porting it.

### Which craft-libs crate is which in-tree crate

| craft-libs crate | there since | the fork's in-tree code | commits |
|---|---|---|---|
| `craft-video` | 11 Nov | `h264`, `h264enc`, `hevc`, `vp9`, `av1`, `vp8`, `mpeg2v`, `apv`, `prores`, `dnx`; `codecs/src/video.rs` | 70 |
| `craft-audio` | 4 Nov | `aac`, `ac3`, `mpa`, `mp3enc`, `opus`, `flac`; `codecs/src/audio.rs`, `media/src/wav.rs`, `interchange/src/wav.rs`, `export/src/pcm.rs` | 129 |
| `craft-media`, `craft-media-source`, `craft-y4m`, `craft-sniff` | 20 Oct–16 Nov | `avi`, `isobmff`, `matroska`, `mpegts`, `mxf`, `ogg`, `bitstream`; `media/src/{cancel,reader,animated}.rs`, `codecs/src/y4m.rs` | 64 |
| `craft-dpx`, `craft-tga`, `craft-exr`, `craft-hdr`, `craft-tiff`, `craft-png` | 10–22 Nov | `media/src/{dpx,tga}.rs`, `media/src/still/{exr,hdr,ycbcr_tiff}`, `dng/src/tiff.rs`, `export/src/{raster,still}.rs` | 45 |
| `craft-timeline`, `craft-captions`, `craft-aaf`, `craft-cfb`, `craft-docpath`, `craft-fcproj-format` | 16 Oct–15 Nov | `time`, `interchange` (EDL, ALE, FCP7, FCPXML, OTIO, AAF, OMF), `captions`, `cfb`, the `craft-docpath` pin, `project` | 64 |
| `craft-control-protocol`, `craft-jsonrpc`, `craft-font-catalog` | 25 Oct–15 Nov | `automation/src/server.rs`, the app's `control_server.rs`, the MCP line reader; `text/src/{fonts,sfnt,render}.rs` | 37 |
| `craft-raw` | 11 Nov | `dng` (DNG only; the camera-raw formats are not in the fork) | 106 |
| `craft-psd`, `craft-jpeg` | 15 Oct–11 Nov | none: the fork reads JPEG through `image` and has no PSD decoder | 104, n/a by crate |
| `craft-ui`, `craft-update`, `craft-web-api`, `craft-web-host` | 9 Oct–5 Nov | the fork's own UI (its `craft-ui` pin is older and stays), no updater, its own web host | 350, skipped by rule (shared UI, updater) |

### Counts (the 515 classified)

| class | commits | | applies to the fork | commits |
|---|---|---|---|---|
| crash / hostile input | 88 | | applies | 112 |
| correctness | 89 | | already in the fork | 64 |
| performance | 19 | | not in the fork (n/a) | 339 |
| feature | 143 | | | |
| refactor, move, tests, docs, pins | 176 | | | |

Recommendations: 71 `port` (the crash and correctness fixes that apply), 8 `port-small`
(performance or small features), 32 `list`, 404 `skip`. A few commits touch two groups and are
counted in both (`3dadce4d7c`, `d26e89917a`).

### Ported (AV65–AV76)

Every crash fix has a regression test; each lane reverted its fix in the working tree, saw the
test fail (or panic, hang or over-allocate), and restored it. The exceptions are named in the
commit messages and under Risks (tests of new functions, the 32-bit-only length tests, a
cherry-picked AES3 test, the LinearizationTable check, which gives the same error either way).
The commit messages list the tests.

| Commit | craft-libs | older FilmCraft fixes taken with it (`-x`) | notes |
|---|---|---|---|
| AV65 H.264 | `6fa8323991`, `56a1f2b7c4`, `9814469613`, `227cfee7e0` (part), `409b0036f8`, `0226f1d03e` | `cb5577d0` (H.264 part; adds `BitReader::read_ue_plus`), `c42e803e`, `66487b0a` (H.264 part), `8dc619b7` | fork-only: the temporal-direct and implicit-weight POC differences saturate too (both trees overflowed there once POCs saturate) |
| AV66 HEVC | `1175abec2e` | `cb5577d0` (HEVC part), `66487b0a` (HEVC part) | the slice-header hunks of `cb5577d0` placed by hand (the fork's parser is unsplit) |
| AV67 AV1, VP9, APV, DNx | `227cfee7e0` (VP9), `16564be34b`, `7f8a4eed58`, `4e92a95e41` | `e90bb50f`, `06eeff43`, `62c16fcb`, `5163f259` | |
| AV68 AAC, FLAC, PCM | `d7a40bc110`, `ac8c5aae5d` / `8291279c8b` / `542ca365e2` (rate parts), `5617261fdc`, `119f6e252e`, `b99a64fc5b` | — | craft-libs' FLAC test passed on the fork as is; a 32-bit case it let through was added |
| AV69 ISO BMFF, MXF, AVI | `f0ad6c1e1d`, `ecf0ad47ad` (stbl), `7474826f6f` (ISO BMFF), `d3bc790aba` (part), `c31fc02ab9`, `de1daefe8c` (part) | `3ef97d85`, `076aa3c9`, `100577c6` | the file-wide sample budget fitted the fork's demuxer without a rewrite |
| AV70 Matroska | `32b2fd8adf`, `ecf0ad47ad` (Matroska), `1e4271e369`, `7474826f6f` (Matroska) | — | four more overflows found by craft-libs' tests (timestamps, TimestampScale, duration, cue positions) fixed as in its import `07047b722`; a new `Error::TooLarge` variant |
| AV71 Ogg | `6865897f14` (part), `27f2204603` (part), `e7814b1727` | `0498cfc0`, `52eb9d76`, `89001f7c`, `0d3b37e2` | the sequence-gap and chained-stream behaviour changes left out (correctness) |
| AV72 Targa, EXR, export buffers | `d2bb64c4ac`, `59ce036a8f`, `95aba513e7` | — | budget 768 MiB so an 8192×8192 32-bit Targa still opens; the EXR fix (a ~200-byte file committed 1 GiB) was found by the triage, not a craft-libs commit |
| AV73 OMF, FCPXML, FCP7 | `d26e89917a` (OMF part), `47036b5e63`, `a44e1b7273` | — | the FCP7 self-reference cycle was found by the triage (craft-libs has it too); fixed the same way |
| AV74 fonts | `23a63f5c76`, `14bccd4e09` | — | public `covers_text` keeps a 512 MiB cap (a system colour-emoji font is 183 MiB); 64 MiB stays on the every-face probe |
| AV75 DNG | `c6f4234682`, `8a067d58d0`, `ccabfee3b5`, `9b87ff0a0e`, `5e5c0cf874` | — | craft-libs' refusal of overlapping tiles not taken (real files store identical tiles once); shared bytes are charged per tile instead |
| AV76 `craft-docpath` | `dccbdc949d`, `44260b9c50`, `3c5e0201e0` | — | the existing pin moves from `82a24b006e` to `3c5e0201e0` (a commit that changes only `crates/docpath`); only that `Cargo.lock` line changes |

### Listed, not ported (recommendation for the next sync)

Correctness fixes that apply, in the order worth taking:

1. **AAC** (`crates/aac`, `crates/codecs/src/audio.rs`): `2b35d24324` (a PCE shaped like 3.0–5.1 keeps
   decoder order, so the centre lands on the left), `555be7c20c` (failed access units leave no state;
   reset restarts the noise generator; HE-AAC is upsampled without the SBR filterbanks' 481-sample
   delay and plays about 20 ms early), `ca99f91a18` (CBR fill can pass the decoder buffer),
   `098eb2e43a` (export floors AAC at 32 kbit/s total: 5.1 at 48 kHz or stereo at 96 kHz encodes
   silence), `e98f3d47b3` (LATM channel counts and PCE), the rest of `542ca365e2` / `ac8c5aae5d`
   (strict element placement, writer refusals, the 0x2b7 sync extension).
2. **Drop-frame time code** (`crates/time` `parse_timecode`, `97080c57f8`): `01;00;00;30` takes the
   non-drop path. Small.
3. **AAF** (`crates/interchange/src/aaf`): `3728fd1ffb` (source offsets at the referenced slot's rate;
   sound offsets are wrong in foreign AAFs and in the fork's own exports — AAFs the fork wrote before
   would then read differently), `70fabd64dc` (unnamed clips take the master's name), `d6134e51cb`
   (relative locators percent-decoded).
4. **Captions** (`crates/captions`): `86e59ebc08` (ten fixes: TTML whitespace and control characters,
   WebVTT arrow escapes and voice names, SRT lines that look like timings, SCC same-text cues and
   pop-on load, CEA-708 windows, TTML literal angle brackets and cue ids), `fdbb9c2471` (TTML body
   styles, `inherit`, literal entities), `0dcef63a88` (signed SCC / MCC fields).
5. **Containers**: MPEG-TS `471abca087` (one access unit per unit, recovery points are key; upstream
   FilmCraft `f2a01712` + `a76f1ab0`), `9a08c77871` (PCR from the PMT's PCR_PID, per-stream wrap,
   transport-error and scrambled packets, duplicates), `45ddc4ffa6`, `9d21479d7a` (program-stream
   re-reads near the end; MXF writer edit-rate check); AVI decimal stream ids (`53b937ab`) and index
   recovery (`e382b599`, `c5897a30`, `f99d53dd`); Ogg sequence gaps and chained links
   (`27f2204603`, `6865897f14` rest); the MP4 `roll` sample group for AAC (`3dadce4d7c`; verify the
   priming claim first); `FileReader` truncation and relative-path checks (`e44720af47`,
   `a1f8141307`); Targa pixel data ending at the footer (`1985afe943`).
6. **Uncompressed video** (`crates/codecs/src/video.rs`): QuickTime 24-bit rows padded to even bytes
   (`f57db543da`; odd widths come out sheared) and AVI BI_BITFIELDS masks (`c3ad96bfd8`). Small.
7. **Opus** concealment (`861c5e172d`), damaged-FLAC timing (`f19f131247`), AIFF-C compression codes
   (`d26e89917a`, the `wav.rs` part), DJI short DNG strips (`60f7c5d74f`, `786c5571f0`).

`port-small`: the bit reader's fast paths (`b25d6e4797`, `ee76ffe8d8`, `62eae4e4cc`, bit-identical
with differential tests), Ogg CRC slice-by-16 (`44b4a23137` = FilmCraft `6763f9bf`), Y4M frame
times by `tick_of` (`3de39fd3a5`), FLAC wasted bits and sample-rate codes (`82f63f3a4f`,
`7c88c201a5`), AAF info on flattened selectors (`584d87f9e1`).

`list` (features and large performance work): **HEVC 4:2:2** (`bb57d422d0`; the fork refuses
`chroma_format_idc != 1`, so 4:2:2 camera footage does not decode), AV1 cross-frame wavefront
(`62e09222f1`, ~1.7× with many threads), HEVC dequantise-while-decoding (`c25999c8a4`), picture
memory budgets and frame limits (VP9, MPEG-2, H.264, ProRes, VP8), lossy-JPEG DNG and DNG opcode
lists, the shaping plan cache.

Also noticed, outside the craft-libs lists: `craft-cfb`'s extraction brought hardening the fork's
`crates/cfb` lacks (a chain loop shorter than the stream reads back as repeated data; a FAT sector
listed twice; a 1 GiB stream cap); the HEVC picture reports only the luma bit depth (a legal Main 10
stream with 8-bit luma and 10-bit chroma would get wrong chroma; `77f244cb96`); and further upstream
FilmCraft commits the 11-08 sync deferred in these crates (`9463ed6f` AVI bounds and decoded-packet
caches, `157cc6b5` recovery points on the decoder side, MXF `041670ea` / `4ad7474c`, the audio perf
series listed under `44b4a23137` in the TSV).

## Results

Integrated on 24 November 2026 on branch `worktree-agent-a593f66f4fbec9f24`, cut from `18e6f49b`.
AV62–AV64 were made here; AV65–AV76 in three port lanes (their own worktrees and target folders,
each gated on its own) and cherry-picked here in order, without conflicts. Nothing was merged from
`origin`, and nothing was pushed.

| Commit | Port | Source |
|---|---|---|
| `097caabf` AV62 | GPU SDR exports read back the finished sRGB frame | `ba6fe105` |
| `58c6825c` AV63 | Pins resolve their targets once per layout | `117876b0` |
| `3a1f6974` AV64 | Russian interface catalog | upstream `ru.tsv` |
| `7c558f75` AV65 | H.264 hostile-input fixes | craft-libs, FilmCraft (above) |
| `964f3cf3` AV66 | HEVC hostile-input fixes | |
| `10e329f2` AV67 | AV1, VP9, APV, DNx hostile-input fixes | |
| `5b038471` AV68 | AAC, FLAC, PCM header fixes | |
| `d27fd77b` AV69 | ISO BMFF, MXF, AVI bounds | |
| `7f87dd12` AV70 | Matroska bounds | |
| `c372265a` AV71 | Ogg bounds | |
| `64d2b11e` AV72 | Targa, EXR, still-export buffers | |
| `d777dff5` AV73 | OMF budget, FCPXML / FCP7 cycles | |
| `9995a23e` AV74 | Font read and outline caps | |
| `c451706e` AV75 | DNG decode bounds | |
| `1ee6d198` AV76 | `craft-docpath` pin | |

### How AV62 was adapted

The fork's `plan_frame` and `is_cpu_frame` take no device caps and its `FrameRenderer` has no
`caps()`; its compositor has no GPU adjustment-layer step, so those parts of upstream's tests are
left out. The fork checks itself that every transition of a plan is drawn on the device (else the
CPU renders the frame); the float and the 8-bit export paths share that as `draw_export`. The new
output read-back retires its staging buffer on a failed wait or a short frame (upstream's
behaviour); the accumulator read-back keeps the fork's buffer handling (upstream's retire-on-failure
for it is another commit) and only waits on its own submission, with a timeout on the callback.
Upstream's `copy_mapped` (another commit) is not taken. The tests are upstream's, adapted: the LUT
resolve against all 65,536 halves and the rounding thresholds on every backend (here Apple M3 Max,
Metal: byte-exact), the 8-bit frame equal to the float frame over black (plain, transformed, the
bench's effects stack on six demo scenes, every GL transition and Push nested or not), the output
read-back retiring and recovering, the export pipeline's choice and fallbacks (`gpu_hook_scenarios`
8a–8g), and a Push transition exported on the GPU without a fallback.

### Measured (Apple M3 Max, macOS)

`bench --section export --repeat 3 --hw auto` (release; 1080p, 191 frames from an 8 s H.264
fixture), the base build and the AV62 build run alternately twice (6 runs each); load average 5–6
from other work on the machine.

| scene | before: best / median | after: best / median | CPU per frame before → after |
|---|---|---|---|
| H.264 + AAC (MP4), VideoToolbox encode | 1.372 / 1.377 s | 1.367 / 1.382 s | 20.3 → 9.2 ms |
| H.264 + AAC, 3 layers + effects | 1.337 / 1.349 s | 0.993 / 0.994 s | 21.3 → 9.0 ms |
| ProRes 422 (our encoder) | 2.963 / 3.023 s | 2.772 / 2.849 s | 200 → 187 ms |

Every frame was rendered on the GPU in both builds (573 per scene, no fallback). The GPU
read-back's CPU conversion goes from 8.1 ms to 0.4 ms per frame (a row copy). The plain H.264 export
is bound by the encoder here, so its wall time does not move; the layered one is not, and is 1.35×
faster. What applies on macOS: the 8-bit read-back and the per-submission waits (every GPU export on
any encoder). What does not: NVENC (upstream's 17.6 s → 11.1 s figure is a 1080p HEVC NVENC export on
Windows); here the hardware encoder is VideoToolbox. A 4K export, or HEVC through VideoToolbox, was
not measured.

### Checks

A lane gate, not the full `cargo xtask ci` (that is the main session's, after the merge). With
`CARGO_TARGET_DIR=target/agent-sync`, dev profile, one cargo at a time, at `1ee6d198` (AV76):

| Command | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --no-fail-fast` on the 25 changed crates but the UI (`filmcraft`, `filmcraft-cli`, aac, apv, av1, avi, bitstream, codecs, dng, dnx, export, flac, gpu, h264, hevc, interchange, isobmff, matroska, media, mxf, ogg, project, text, vp9) and `filmcraft-engine` (the `craft-docpath` pin) | 141 suites, 3,239 passed, 0 failed, 36 ignored (engine lib 1,447 of them) |
| `cargo test --no-fail-fast -p filmcraft-ui-egui` | 111 suites, 820 passed, 0 failed, 9 ignored |
| `cargo clippy` of effectcraft, engine and CLI with the `effectcraft` features, `-D warnings` | clean |
| `cargo test -p filmcraft-cli --features filmcraft-cli/effectcraft --test cli -- effectcraft` | 1 passed (33 filtered out) |
| `cargo xtask layers` | ok |
| `cargo xtask assets` | ok (93 assets attributed) |
| `cargo xtask wasm` | ok (58 crates) |

- Each commit was also gated on its own as it was made: AV62 with clippy on gpu and export and their
  tests (gpu 64 + 1, export 165 + 7), then a workspace clippy; AV63 with the project tests (159);
  AV64 with the i18n tests (22), `settings_ui` (15) and `assets`. The lanes ran fmt, clippy on their
  crates (also without default features where the crate has them), their crates' tests and
  `codecs`, `layers`, `assets` and `wasm` before each commit; their counts are in their reports.
- ffmpeg is installed on this machine now (the lanes found it), so the codec oracle and conformance
  tests ran (for example H.264 conformance 38, HEVC 43, VP9 50).
- Not run: the tests of crates nothing here changed (their code is linted by the workspace clippy),
  the release-profile runs `cargo xtask ci --changed` would make (the root `Cargo.toml` changed,
  which sends it to the full gate), and anything on Linux, Windows or a 32-bit target (the
  `8dc619b7` length tests reproduce only on 32-bit).

### No-loss proof against `18e6f49b`

- **Command ids**: `filmcraft-cli commands --json` lists 954 commands at `18e6f49b` (built from it before
  any change) and 954 at `1ee6d198`: the same ids, and no entry differs (labels, menus, shortcuts,
  params).
- **Tests**: `#[test]` and `#[tokio::test]` functions found by a script over every `.rs` file (one
  per file and name): 6,680 → 6,831. None removed or renamed; 151 added (Ogg 20, Matroska 17, H.264
  19, HEVC 14, AV1 11, ISO BMFF 7, gpu 7, Targa 5, DNG 6, MXF 7, interchange 8, and the rest).
- **`pub` / `pub(crate)` functions**: 8,960 → 8,969. None removed; added `render_rgba8`,
  `render_export_rgba8_prepared`, `resolve_export_rgba8`, `read_ue_plus`, `tile_metadata_bytes`,
  `reserve_mbs`, `reaches_active`, `frame_prefix_len`, `decode_aes3_frames`. The `FrameRenderer`
  trait gains `render_rgba8` with a default body, so every existing renderer still builds.
- **Files**: none deleted or renamed (`git diff --diff-filter=DR 18e6f49b HEAD` is empty); 2 added
  (`crates/apv/tests/{alpha,hostile_args}.rs`) besides this record and its TSV. `Cargo.toml` changes
  only in AV76 (the `craft-docpath` rev) and `crates/bitstream/Cargo.toml` in AV65 (upstream
  `8dc619b7` splits proptest's dev-dependency for wasm32).

### What behaves differently after this sync

- GPU exports: an SDR export with nothing to place or overlay gets the GPU's 8-bit frame; with
  alpha, HDR, fitting, pixel aspect, a resample or an overlay, the float frame as before. A frame the
  GPU cannot draw is still the CPU's and counts as a fallback.
- Russian shows almost the whole interface translated; 76 strings change wording to upstream's.
- Hostile or damaged media, interchange files and fonts that used to panic, overflow, hang or
  allocate without bound now give an error, a warning or a bounded result. Some limits are new and
  could refuse an extreme but valid file: Matroska over 2^16 track entries or laced blocks over
  16 MiB; Targa over 768 MiB decoded; at most 64 pictures per H.264/HEVC decode call; OMF and
  FCPXML documents past their work budgets import partly with a warning.
- `craft-docpath` at `3c5e0201e0`: non-Unicode document paths on Windows device namespaces and Unix
  roots resolve more strictly (the engine already refused non-Unicode results).

### Risks

- AV62 was checked headless (tests and the bench); exports from the running app were not looked
  at, and the app on port 9877 was left alone. The LUT parity was checked on Metal only.
- The lanes read craft-libs' current sources rather than per-commit diffs where the sandbox refused
  git in the clone; items not in a commit message's list could have been missed in the large import
  commits (`0226f1d03e`, `1175abec2e`, `16564be34b`, `7f8a4eed58`).
- The two `catch_unwind` moves (H.264 `run_job`, HEVC picture state) have no test of their own.
- AV74's 512 MiB load cap is only reproducible with a font file larger than that; tested through the
  new function.

The next upstream review starts at `ba6fe105`; the next craft-libs review at `249a41bd9`.

## Ported for AutoCut 0.13.0 (AV78–AV90, 24 November 2026)

The "Listed, not ported" correctness fixes 1–7 and the `port-small` speed items were ported the
same day in three more lanes (own worktrees and targets, lane gates each), cherry-picked onto main
after AV77, PX8.4 and AV91. HEVC 4:2:2 (`bb57d422d0`) waits for the next release (the user's
decision); the other `list` items stay listed.

| Commit | Port | craft-libs / upstream |
|---|---|---|
| AV78 | AAC: PCE speaker order, failed units leave no state, HE-AAC 481-sample SBR delay, LATM channels and PCE, strict element placement, 0x2b7 extension | `2b35d24324`, `555be7c20c`, `e98f3d47b3`, rest of `542ca365e2` / `ac8c5aae5d` |
| AV79 | AAC encoder: fill inside the decoder buffer, exact CBR limits, writers return `Result`, export floor max(6000, rate/4) bit/s per channel | `ca99f91a18`, `098eb2e43a` (AAC), writer parts of `542ca365e2` / `ac8c5aae5d` |
| AV80 | Opus concealment, damaged-FLAC timing, AIFF-C codes, FLAC wasted bits and rate codes | `861c5e172d`, `f19f131247`, `d26e89917a` (wav), `82f63f3a4f`, `7c88c201a5` |
| AV81 | Drop-frame time code carry | `97080c57f8` |
| AV82 | Compound files: chain loops, repeated FAT sectors, 1 GiB stream cap; strict root mini-stream chain | `f759a35db8`, craft-cfb extraction `06af3f2d4` |
| AV83 | AAF source starts at the referenced slot's rate (old FilmCraft-written files recognised and read as before), master names, locator URLs, flattened-selector report | `3728fd1ffb`, `70fabd64dc`, `d6134e51cb`, `584d87f9e1` |
| AV84 | Captions: TTML, WebVTT, SRT, SCC, MCC, EBU STL fixes | `86e59ebc08`, `fdbb9c2471`, `0dcef63a88` |
| AV85 | MPEG-TS access units, recovery points, PCR_PID, damaged/scrambled/repeated packets; MXF writer edit rate | `471abca087` (= upstream `f2a01712`, `a76f1ab0`, `157cc6b5`, `cd531316`), `9a08c77871`, `45ddc4ffa6`, `9d21479d7a` |
| AV86 | AVI decimal stream ids, idx1 recovery, bounded decoded-packet caches | upstream `53b937ab`, `e382b599`, `c5897a30`, `f99d53dd`, `9463ed6f` |
| AV87 | Ogg lost pages, chained links reusing a serial, trailing Opus packets | rest of `6865897f14`, `27f2204603` |
| AV88 | MP4 `roll` group for AAC; `FileReader` truncation and relative paths | `3dadce4d7c` (muxer), `e44720af47`, `a1f8141307` |
| AV89 | Targa footer, DJI DNG strips, QuickTime 24-bit row padding, AVI BI_BITFIELDS masks, HEVC chroma bit depth | `1985afe943`, `60f7c5d74f`, `786c5571f0`, `f57db543da`, `c3ad96bfd8`, `77f244cb96` |
| AV90 | Bit reader fast paths, Ogg CRC slice-by-16, Y4M frame times | `b25d6e4797`, `ee76ffe8d8`, `62eae4e4cc`, `44b4a23137` (= upstream `6763f9bf`), `3de39fd3a5` |

Left out, with reasons in the commit messages: per-frame AAC encoder buffering and craft-libs
builder APIs (no counterpart here), `frames_between`, craft-cfb's lenient-reading features, the
`VideoFrame` refactor, Ogg FLAC/Speex/Theora handling. Noticed, not changed: the AAF tape
timecode offset still converts at the file slot's rate (the fork writes 0 there), and
`crates/captions/tests/it/ttml_clock_overflow.rs` is never compiled (no `main.rs`).

## Ported for AutoCut 0.15 (AV110–AV115, 25 November 2026)

The `list` item "picture memory budgets and frame limits" was ported in one lane (own worktree and
target, lane gate), behaviour only (the file moves left out), in the fork's per-crate layout: each
decoder that needs a budget has a small private `budget` module (`PictureShape`) and exports its
`DECODE_MEMORY_BUDGET`, since `mpeg2v` does not depend on `filmcraft-bitstream`.

| Commit | Port | craft-libs |
|---|---|---|
| AV110 | H.264 `Decoder::set_max_picture_luma_samples`; unlike upstream's plain sample count the limit counts bytes (every plane, two bytes a sample above 8 bits), so High 10 and 4:2:2 pictures get the memory of the 8-bit 4:2:0 picture | `4cca9be2ab` |
| AV111 | ProRes `DecodeLimits`, `decode_frame_with_limits`, `decode_frame_into_with_limits` | `4cca9be2ab` |
| AV112 | VP8 `Decoder::with_max_luma_samples` | `07a4cd0fc7` |
| AV113 | MPEG-2 sequence memory budget (`Decoder::set_memory_budget`) and the mutation test over the synthetic field / 16x8 / dual-prime streams; also a settable area cap (`set_max_picture_luma_samples`, `probe_with_limit`) | `5e892217ed` (budget part) |
| AV114 | VP9 picture-store budget; unlike upstream's `threads + 2` it counts what the decoder's threading really keeps (eight references, at most six frames in their post stage, at most 14 output pictures), with frames at their real band sizes; also a settable luma cap | `6ac2a6a34a` (budget part) |
| AV115 | `filmcraft_codecs::limits`: one process-wide `DecoderLimits { max_luma_samples, memory_budget }` (`set_decoder_limits`, like the hardware decoding switch) that every software H.264, VP8, VP9, MPEG-2 and ProRes decoder the media layer makes reads, for a future workflow past 8K; no UI | — |

Defaults keep today's limits (H.264 2^25 as bytes, VP8 2^26, MPEG-2 2^25, ProRes 1 GiB with no area
limit; upstream's stricter ProRes 300 MiB and VP8 36M defaults were not taken). The budgets are new.
On 64-bit targets they refuse nothing that opens today: MPEG-2 uses upstream's 4 GiB (its area cap
needs at most 1.3 GB), VP9 12 GiB (the most it can hold within its size limits is about 12.1 GB,
4:4:4 12-bit at 2^26 luma samples with 18 or more threads; every level 6.2 frame in every profile
at most about 6.5 GB, 8K UHD 4:4:4 12-bit about 5.9 GB). On 32-bit targets both are 1 GiB, which
refuses VP9 8-bit 4:2:0 above about 29 million luma samples (8K UHD) and MPEG-2 4:4:4 above about
27.5 million on wasm32. HEVC, AV1, APV and DNx keep their fixed caps and do not read the setting.

### Container improvements (AV160–AV165)

The `list` items for the MP4 and Matroska containers and the shared request cancellation were
ported in one lane (own worktree and target, lane gate), behaviour only, into the fork's
`filmcraft-isobmff`, `filmcraft-matroska` and `filmcraft-media` crates.

| Commit | Port | craft-libs |
|---|---|---|
| AV160 | MP4/MOV chapter lists: `WriterOptions::chapters` writes a QuickTime chapter track and a Nero `chpl` list, `Mp4File::chpl_chapters` / `read_chapters` read them; exports write the sequence's chapter markers (MP4, MOV, M4A; an In/Out range counted from its start). `Mp4Writer` now holds its options boxed, so it does not grow with them | `7480fc2ac4` |
| AV161 | Progressive Matroska index: `OpenOptions::index_limit`, `MkvFile::index_end` / `extend_index` (resumable after a failed read, no element read twice), `Demuxer::build_index` completes a limited index; element header peeks capped at the parent; the EBML magic needs 4 bytes, not the 64 KiB window. Nothing in the fork limits the index yet (the web build could) | `5b58973bc6` |
| AV162 | QuickTime `logs` box read and written as `VideoParams::log_transfer_function` (refused at `add_track` when the reader would not accept it); `TrackSpec::colour` writes a video track's Matroska `Colour` element. The laced-remux test change was not taken: the fork's muxer laces audio whatever its timestamps, so its `laced_remux` passes without it (noted as a fork muxer bug, not fixed). Nothing maps the `logs` identifier to `ColorSpace::AppleLog` yet | `ce31f42e76` |
| AV163 | `WriterOptions::compatible_brands` (`Vec<FourCc>`; unlike upstream each brand is written once, and more than 256 distinct extra brands is an error); AV1 MP4 exports list `av01` (AV1-ISOBMFF §2.1). `MkvWriter` writes a dependent BlockGroup's real ReferenceBlock delta (`el_sint`); unlike upstream's 0, a track's first frame keeps -1 | `057b0e3e01` |
| AV164 | `Mp4Writer` counts chunk offsets from the file's first byte (a file begun mid-writer is valid, `moov` at the end or in front); the header, sample, timecode, chapter, faststart and `moov` position sums are checked (`advance`), and unlike upstream a refused sample is checked before it is written. The fork's exports start at 0, so no export changes | `63712c33e6`, `098eb2e43a`, `e5fa1ac2f9` (Mp4Writer parts) |
| AV165 | `with_cancel`, `with_draft` and `with_catch_up` restore the hint they replaced on unwind too, as `with_background` and `Context::run` already did; the fork's `Context` (all four hints) is kept rather than upstream's cancel-only one | `5d25fa3d22` |

Visible to users: AV160 (chapter markers become MP4/MOV/M4A chapters) and AV163 (AV1 MP4 exports
list `av01`); the rest is container API and crash guards. Left out, with reasons in the commit
messages: imported chapters becoming markers (`MediaInfo` has no chapter list; MKV chapters are not
imported either), mapping the `logs` identifier to `ColorSpace::AppleLog` (a media-layer change),
upstream's laced-remux test change (not needed by the fork's muxer). Noticed, not changed: the
fork's `MkvWriter` laces audio frames whatever their timestamps (laced frames after the first read
back at the block time plus multiples of DefaultDuration, or the block time without one), and
`Mp4Writer::write_sample` computes `sample_count as u32 + 1` for a sync sample, which overflows past
2^32 - 1 samples in one track (upstream's `too_many_samples` guard, from another commit).

### Still images (AV116 onwards)

The `list` items on still images (EXR, TIFF YCbCr, CinemaDNG in MXF, DNG, AVIF), ported in one
lane (own worktree and target, lane gate), behaviour only, into the fork's own crates: no
craft-libs crate and no new dependency.

| Commit | Port | craft-libs |
|---|---|---|
| AV116 | YCbCr TIFF: unusable YCbCrCoefficients / ReferenceBlackWhite fall back to their defaults (they were refused), libtiff's green (greys stay neutral when the weights do not sum to 1), a file too small for its picture refused before the data units are allocated, fallible allocations, overlapping strips refused. JPEG-compressed YCbCr stays the image crate's | `603d43da85` |
| AV117 | MXF RGBA descriptors (CinemaDNG): depth from the PixelLayout (R, G, B equal, else 0), levels from ComponentMinRef / ComponentMaxRef. The MXF video source keeps reading the levels as a Y'CbCr range only from CDCI descriptors, so a ProRes 4444 track with an RGBA descriptor and full-range RGB codes is not forced to full-range Y'CbCr (no change from today) | `fc54b56b65` |
| AV118 | EXR channel choice (`still/exr/channels.rs`): the main picture (`main_picture`, unprefixed RGB(A), else Y(A)); when there is none, upstream's `color_channels` ranking (the richest one-layer group, ties to the unnamed layer then alphabetically; a lone channel is luminance), so renderer multilayer files and depth-only files open (they were refused). YA is luminance with alpha; the pixel format names a named layer | `e7b247c09f` (and `color_channels` of `83eb8f4422`) |
| AV119 | EXR multi-part and tiled files in the fork's own reader (upstream wraps the `exr` crate, which is not in the fork's lock): every part's header listed (at most 1024 parts), the part with the picture chosen as in AV118 (deep and unknown parts never), the earlier parts' tables skipped by `chunkCount`, only that part's chunks read; tiles of the full-resolution level (one level, mip-map or rip-map), each tile once, decoded straight into place by bands; every chunk's position, part number, coordinates and size checked before the picture is allocated. PIZ, B44 and DWA stay refused: the public file layout does not specify their bitstreams and the clean-room rule rules out reading another decoder | `83eb8f4422` (part listing, part choice, one-part decode, limits) |
| AV120 | EXR sequences written from the compositor's float pictures (`Pipeline::frame_linear`, GPU or CPU), half or 32-bit float (`ExportSettings::exr_depth`, `exrDepth` on `file.exportMedia`, a Bit depth row in the export panel), PXR24 added to the compressions (lossless for half, 24 bits of a float), and the chromaticities of the frames' primaries: an HDR or wide-gamut sequence's linear BT.2020 working space unless `sdr`, else BT.709. Values above 1 and below 0 survive (the RGBA8 path clipped them). B44, PIZ and DWA not written (no public bitstream specification); single stills keep the RGBA8 half writer | `3b8e426a26` |
| AV121 | AVIF stills in the fork's HEIF reader (`codecs/src/heif.rs`): `av01` items (and grids of them) decoded by `filmcraft-av1` single-threaded under `catch_unwind`, `av1C` configuration OBUs first, the item's `colr nclx` over the sequence header, matrix 0 as GBR, matrix 10 refused; alpha auxiliary images (`auxC` alpha, `auxl`, `prem`) for AVIF and HEIC, left out (opaque, as before) when they do not decode or are transformed unlike the colour; `.avif` an import extension. `Decoder::set_shown_budget` (lower the 1 GiB per-chunk cap, never raise it) set to the item's declared picture. Upstream's grid and transform handling was already in the fork; its fuzz target and binary fixtures not taken (the tests make their AVIFs with ffmpeg / libsvtav1 and compare the planes with libdav1d) | `59b4f1e8c2` |
| AV122 | Faster DNG decode and development, from the same work in upstream FilmCraft's own `crates/dng` (the fork's lineage of the crate: `705e9a17` `ProcessOptions`, `14e22013` tint sign, `7055d45a` the speed-ups and `reference` module): 13-bit code + difference lookup and 8-byte refills in lossless JPEG, tiles decoded in parallel bands straight into place, a single strip used without a copy, normalisation tables, row-by-row demosaic / crop / colour, and `ProcessOptions::target_size` (1/2, 1/4, 1/8 size without demosaic). Merged with the fork's own bounds: every chunk keeps its AV75 decode budget and aggregate charge, the band holds only its own chunks. Not taken: the optional serde derives (`e9b12176`, a new dependency edge in the lock); `target_size` is not yet asked for by playback | `eb5844d4da` (FilmCraft `7055d45a`) |
| AV123 | DNG opcode lists (DNG 1.7.1.0 chapter 7: OpcodeList1/2/3, the 14 opcodes, the skip rule, optional / mandatory flags, preview-only opcodes), from FilmCraft `7e6f075e`, with its later opt-in (`fd0a455d`: `ProcessOptions::apply_opcodes`, off by default "until the interpretations are compared with real camera files"), the opt-in in `process::develop_with` (from `46c9d91e`) and the test helper fix `4e5d4df1`. Files without lists, and every file by default, develop exactly as before; nothing in AutoCut turns the lists on yet | `c31578e1ff`, `cb3f76848b` (FilmCraft `7e6f075e`) |
| AV124 | Lossy JPEG DNG (Compression 34892, 8-bit LinearRaw, one or three components). Upstream adds the `image` crate to `filmcraft-dng`; here the L0 crate keeps no image codec (and the lock no new edge): `set_lossy_jpeg_decoder` takes a baseline JPEG decoder, which `filmcraft-media` registers (the image crate's, under its allocation limits) before every DNG it develops. The decoder runs under `catch_unwind` with a budget of about its tile; its components must equal SamplesPerPixel (`46c9d91e`); two bits per 8 × 8 block as the plausibility floor. The test writer takes precoded chunks (`Compression::Precoded`, `chunk_samples`) instead of encoding JPEG itself; upstream's lossy tests run in `filmcraft-media` | `589d84d0f4` (FilmCraft `00c93a3e`) |
