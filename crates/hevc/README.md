# filmcraft-hevc

Clean-room, pure-Rust (no `unsafe`) H.265 / HEVC decoder, implemented from the public ITU-T
Rec. H.265 (08/2021, ISO/IEC 23008-2) specification. It is an L0 crate: it depends only on
`filmcraft-bitstream`, `thiserror`, and optionally `rayon` (default feature `threads`), and builds
for `wasm32-unknown-unknown`.

All syntax tables (the CABAC `initValue` tables arranged per `initType` following Table 9-4, default
scaling lists, `intraPredAngle` / `invAngle`, the chroma QP table, the deblocking beta'/tC' table,
the luma / chroma interpolation filter taps and the 32x32 DCT matrix) were extracted from the
specification text by [`tools/extract_tables.py`](tools/extract_tables.py) (it only reads the text
of the ITU-T PDF, see the script header); the output is `src/spec_tables.rs`. Scan orders are
derived by the algorithms of 6.5.3-6.5.5. No code was taken or ported from FFmpeg, libde265, x265 or
the HM reference software; ffmpeg / libx265 / VideoToolbox are only used as external test oracles
and fixture generators.

## Status

| Area | Status |
|---|---|
| NAL / Annex-B / `hvcC` length-prefixed input | done |
| VPS, SPS (VUI, HRD skipped, scaling lists, st/lt RPS, PCM), PPS (tiles, WPP, deblocking control, scaling lists, list modification) | done |
| Slice segment header (all fields: st/lt RPS, list modification, pred weight table, entry points, header extension) | done |
| POC, RPS marking, missing-reference generation, RefPicList construction, DPB output / bumping (C.5.2) | done |
| CABAC: all Main / Main 10 syntax elements, WPP context sync, dependent slice context restore, tiles | done |
| Coding quadtree, CU (skip / intra / inter, AMP, PCM, transquant bypass), PU, transform tree | done |
| Intra: 35 modes, reference substitution, [1 2 1] filtering, strong intra smoothing, constrained intra | done |
| Inter: merge (spatial, temporal, combined bi-pred, zero), AMVP, TMVP, 8-tap / 4-tap interpolation, default and explicit weighted prediction | done |
| Scaling (flat and scaling lists), DCT 4..32, DST 4x4, transform skip, sign data hiding | done |
| Deblocking (bS, slice / tile boundary rules, PCM / bypass exemption), SAO (band / edge, boundary rules) | done |
| Frame-level multithreading (CTB-row granularity dependencies) | done |
| Draft mode (non-reference pictures skip deblocking / SAO; reduced-resolution playback only) | done |
| Tiles / WPP | parsed and decoded sequentially within a picture (parallelism is across pictures) |
| 4:2:2 (Main 4:2:2 10 / RExt profile without RExt tools, what cameras and libx265 write): chroma geometry from the SPS chroma format in the picture store, intra, inter, transform tree, deblocking and SAO; Table 8-3 intra mode mapping; stacked chroma transform blocks with their second cbf flags; chroma QP = Min(qPi, 51); chroma motion vectors per axis; 8-12-bit | done (bit-exact on the 8-, 10- and 12-bit libx265 fixtures) |
| 4:4:4 (Main 4:4:4 / Main 4:4:4 10, RExt profiles without RExt tools, what libx265 writes): chroma at full resolution through intra (chroma modes per prediction block, 4x4 chroma blocks, neighbour filtering and the mode-dependent scan of 8x8 chroma blocks), inter (4-tap chroma filter at the luma vector), 32x32 chroma transforms and scaling lists, the 4:4:4 chroma cbfs at every depth, deblocking on the 8x8 chroma grid and SAO; 8-12-bit | done (bit-exact on 41 8-, 10- and 12-bit libx265 fixtures) |
| Monochrome (Main 4:0:0 / 4:0:0 10 / 4:0:0 12, `chroma_format_idc` 0): luma only, no chroma syntax (modes, cbfs, residuals, PCM samples, SAO, weights) or planes; `Picture::chroma` is `C400` with empty `u` / `v` planes, and the chroma depth is the luma depth | done (bit-exact on 14 libx265 fixtures) |
| 12-bit (Main 12, Main 4:2:2 12, Main 4:4:4 12, Monochrome 12, RExt profiles without RExt tools): QpBdOffset 24, interpolation shifts 4 / 2, weighted prediction offsets << 4, deblocking tc / beta << 4, SAO band shift 7 with unscaled offsets (`log2_sao_offset_scale` 0) | done (bit-exact on 27 libx265 fixtures, and on 4 coarse x265 streams against the encoder's own reconstruction) |
| Separate colour planes (`separate_colour_plane_flag`: three monochrome-coded planes with their own slices) | not supported (`Error::Unsupported`) |
| Range-extension coding tools (RDPCM, cross-component prediction, extended precision, persistent Rice adaptation, bypass alignment, chroma QP offset lists, SAO offset scaling, transform-skip rotation / contexts / sizes, intra smoothing off, high-precision weighted prediction offsets) | not yet (`Error::Unsupported`) |
| SCC, multilayer (SHVC / MV-HEVC), 3D extensions | not supported (layers > 0 are ignored) |

## API

```rust
use filmcraft_hevc::{Decoder, Plane};

let mut dec = Decoder::new(); // or Decoder::with_threads(n), Decoder::from_hvcc(&hvcc)?
for (pts, access_unit) in access_units {
    for pic in dec.decode(access_unit, pts)? {
        // pic.width / pic.height (cropped), pic.bit_depth, pic.chroma (ChromaFormat::C400 / C420 /
        // C422 / C444; pic.chroma_width / chroma_height: the u / v plane size, 0 for C400),
        // pic.y / pic.u / pic.v: Plane::U8(Vec<u8>) for 8-bit, Plane::U16(Vec<u16>) above,
        // pic.y_stride / pic.uv_stride, pic.pts, pic.poc, pic.key (IRAP), pic.color (VUI), pic.sar
    }
}
for pic in dec.flush() { /* ... */ }
if let Some(err) = dec.take_error() { /* error reported by a decoding thread */ }
// dec.set_draft(true): draft mode (playback at reduced resolution only)
```

`decode` expects whole access units (a buffer with several complete access units — e.g. an entire
Annex-B file — also works). RASL pictures following a CRA that starts the stream are skipped, as
ffmpeg does. `Decoder::stats()` returns counters of what was decoded (slice types, IDR/CRA, tiles,
WPP, weighted prediction, TMVP, SAO, block kinds...).

### Architecture

Same design as `filmcraft-h264`: the calling thread parses parameter sets and slice headers and does
POC / RPS / DPB bookkeeping; the CTU layer of each picture runs as a job (rayon pool). A job
reconstructs into private planes, deblocks CTB row r once row r+1 is decoded, applies SAO to row r
once row r+1 is deblocked, and publishes final rows (samples + 16x16 compressed motion for TMVP)
into the shared `Frame` (`OnceLock` per CTB row). Motion compensation of later pictures blocks per
row on exactly the reference rows it reads. Samples are stored as `u8` for 8-bit pictures (half
the memory traffic in prediction reads, in-loop filters and output) and as `u16` above 8 bits: the
slice decoder and the filters are generic over the sample type (`picture::Sample`), monomorphised
once per type.

Unavailable references keep distinct frame IDs and POCs while sharing one immutable gray backing
for the active picture format. This backing is created only when a reference is missing, replaced
on geometry, chroma-format or plane-depth changes, and released on flush. Jobs still reading an
older format keep its rows alive. Each backing retains the full coded picture dimensions.

Picture sizes are capped by the samples of all three planes (`params::MAX_PICTURE_SAMPLES`, those
of a 4:2:0 picture of 2^26 luma samples): a 4:2:2 picture, whose chroma planes are twice as big,
may have three quarters of the luma samples (8K fits), a 4:4:4 picture half of them (8192 x 4096
and 8K UHD fit). Monochrome pictures are capped at the 4:2:0 luma limit (the per-block state
follows the luma size).

Draft mode (`Decoder::set_draft(true)`; FilmCraft enables it with Settings ▸ Playback ▸ Draft
decoding while the Program monitor plays at 1/2 or 1/4): sub-layer non-reference pictures
(TRAIL_N, RASL_N, ...) of the highest temporal sub-layer — no later picture predicts from them,
and TMVP reads only reference pictures — skip deblocking and SAO and are flagged
`Picture::draft`; every other picture is unchanged. libx265's non-reference B pictures are such
pictures.

Modules: `params` (VPS/SPS/PPS/tile layout), `slice` (NAL + slice header), `dpb` (POC, RPS, lists,
bumping), `cabac`, `slicedec` (CTU syntax + reconstruction), `mvpred` (merge / AMVP / TMVP),
`intra`, `inter`, `transform`, `filter` (deblocking, SAO, row publication).

## Tests

`cargo test -p filmcraft-hevc` (fixtures are generated on first use into `target/fixtures/hevc/`
with ffmpeg; tests print a message and skip when ffmpeg is absent).

- Unit tests: CABAC engine round trip against an encoder model and context initialisation, scan
  orders, DCT matrix properties, QpC, intra predictors, transform DC / bounded-region shortcuts,
  MV scaling.
- Synthetic streams (`src/synth_tests.rs`): a small CABAC encoder writes streams with what libx265
  never produces — tiles (uniform and explicit spacing, loop filter across tiles on / off), PCM
  (with and without in-loop filtering), multiple slices, dependent slice segments (starting inside
  and at the start of a tile), long-term reference pictures, reference list modification. Expected
  output is known exactly where filters leave PCM samples alone; every stream is also compared with
  ffmpeg.
- Robustness: hvcC input with pts round trip; randomly corrupted, truncated and garbage input never
  panics or deadlocks (single- and multi-threaded).
- Conformance (`tests/conformance.rs`): every fixture is decoded with 1 thread, 3 threads and every
  core and compared sample-exactly, plane by plane, with `ffmpeg -f rawvideo -pix_fmt yuv420p /
  yuv422p / yuv444p / gray` at 8 bits and their `10le` / `12le` forms (monochrome pictures must have
  empty chroma planes); on mismatch
  the first differing frame, plane, sample, CTB and 8x8 block are reported. Output pts/POC order is
  checked too, and that no picture is flagged draft without draft mode. `compare_detects_mismatch`,
  `compare_detects_a_422_chroma_mismatch` and `compare_detects_444_and_monochrome_mismatches`
  guard the harness itself. Where ffmpeg is wrong (12-bit streams at the coarsest quantizers with
  cu_qp_delta), the x265 command-line encoder writes the stream and its reconstruction (`--recon`)
  is the reference (`coarse_12_bit_streams_match_the_encoders_reconstruction`). Draft mode (`draft_mode_changes_only_flagged_non_reference_pictures`):
  on B-pyramid / weighted fixtures, single- and frame-threaded, every unflagged picture is
  bit-exact and only flagged pictures differ.

### Fixture matrix (all bit-exact, single- and multi-threaded)

libx265 unless noted (x265 enables WPP by default, so most streams carry entry points).

| fixture | source / size | configuration |
|---|---|---|
| intra_main / intra_main10 | testsrc2+noise 352x288 | keyint=1, 8 / 10-bit |
| intra_nosao_nodbk | testsrc2+noise 352x288 | intra, no SAO, no deblocking |
| ultrafast / veryfast / medium / slow | 352x288 | presets (slow: rect + AMP) |
| medium_main10 / slow_main10 | 352x288 | Main 10 presets |
| p_only | 352x288 | bframes=0 |
| bframes | 352x288, 30 frames | bframes=8, b-pyramid, ref=4 |
| no_sao / no_deblock / deblock_m3_3 | 352x288 | loop filter variants |
| no_wpp | 352x288 | WPP off |
| rect_amp / min_cu16 | mandelbrot / testsrc2 | rect + AMP, min CU 16 |
| tskip | 352x288 | transform skip |
| weightp / weightp_main10 | fade-in | weightp + weightb |
| crf0 / crf51 / qp1_main10 | | quantizer extremes |
| lossless / cu_lossless | 176x144 | cu_transquant_bypass |
| slices4 | 352x288 | 4 slices per picture |
| scaling_list | 352x288 | default scaling lists |
| ctu16 / ctu32_tu4 | 352x288 | CTB 16, CTB 32 with deep TU trees, max TU 16 |
| no_signhide / constrained_intra / no_tmvp / no_strong_intra / max_merge1 | 352x288 | tool toggles |
| open_gop | 352x288, 40 frames | CRA + RASL / RADL, keyint 12 |
| keyint5_idr | 352x288 | closed GOP, IDR every 5 |
| odd_1918x1078 | 1918x1078 | conformance-window cropping |
| hd_1080p | 1920x1080 | preset fast |
| uhd_4k_main10 | 3840x2160 | Main 10 ultrafast |
| vt_main / vt_main10 | 640x360 | Apple VideoToolbox (hevc_videotoolbox), macOS only |
| intra_422 / intra_422_10 / intra_422_nosao_nodbk / intra_422_10_nosao_nodbk | 352x288 | 4:2:2 8 / 10-bit intra, with and without the in-loop filters |
| medium_422_10 / slow_422 / p_only_422_10 / bframes_422 / bframes_422_10 | 352x288 | 4:2:2 presets, P only, B pyramid (8 and 10-bit) |
| no_sao_422_10 / no_deblock_422 / deblock_m3_3_422_10 / rect_amp_422_10 / tskip_422_10 / weightp_422_10 | 352x288 | 4:2:2 tool toggles |
| qp_422_10 / crf51_422_10 / lossless_422_10 / cu_lossless_422 | 352x288 / 176x144 | 4:2:2 adaptive QP (cu_qp_delta), quantizer extremes, lossless, per-CU bypass |
| ctu32_tu4_422_10 / ctu16_422_10 / scaling_list_422_10 / constrained_intra_422_10 / slices4_422_10 | 352x288 | 4:2:2 block structure, scaling lists, constrained intra, slices |
| odd_202x118_422 / odd_330x117_422_10 / odd_98x61_422_10 / odd_1918x1078_422 / odd_1918x1078_422_10 | | 4:2:2 sizes that are not multiples of 8 or 16 (odd heights: 4:2:2 crops by lines) |
| uhd_4k_422_10 | 3840x2160 | 4:2:2 10-bit ultrafast |
| intra_420_12 / intra_420_12_nosao_nodbk / medium_420_12 / slow_420_12 / bframes_420_12 / weightp_420_12 / tskip_420_12 / qp1_420_12 / lossless_420_12 / scaling_list_420_12 / odd_1918x1078_420_12 | 352x288 / 176x144 / 1918x1078 | Main 12: intra with and without the in-loop filters, presets, B pyramid, weighted prediction, transform skip, the finest quantizer (QP 1, Qp' 25), lossless, scaling lists, cropping |
| intra_422_12 / medium_422_12 / bframes_422_12 / lossless_422_12 / odd_330x117_422_12 | | Main 4:2:2 12 |
| intra_444_12 / medium_444_12 / bframes_444_12 / weightp_444_12 / tskip_444_12 / lossless_444_12 / qp_444_12 / odd_201x117_444_12 | | Main 4:4:4 12 (qp_444_12: adaptive QP with chroma QP offsets 5 / -6) |
| intra_400_12 / medium_400_12 / lossless_400_12 | | Monochrome 12 |
| recon_crf51_420_12 / recon_crf45_422_12 / recon_crf51_444_12 / recon_crf48_400_12 | 352x288 | x265 command-line encoder at CRF 45-51 (cu_qp_delta at the coarsest quantizers), compared with the encoder's own reconstruction (`--recon`): ffmpeg 9 decodes these to other pictures (some flat mid-grey) without an error |
| intra_444 / intra_444_10 / intra_444_nosao_nodbk / intra_444_10_nosao_nodbk / no_strong_intra_444 / constrained_intra_444 | 352x288 | 4:4:4 8 / 10-bit intra, with and without the in-loop filters, strong smoothing off, constrained intra |
| medium_444 / medium_444_10 / slow_444 / p_only_444_10 / bframes_444 / bframes_444_10 | 352x288 | 4:4:4 presets, P only, B pyramid |
| no_sao_444 / no_deblock_444_10 / deblock_m3_3_444 / rect_amp_444_10 / tskip_444 / tskip_444_10 / weightp_444 / weightp_444_10 | 352x288 | 4:4:4 tool toggles |
| qp_444_10 / crf0_444 / crf51_444_10 / lossless_444 / lossless_444_10 / cu_lossless_444 | 352x288 / 176x144 | 4:4:4 adaptive QP with chroma QP offsets, quantizer extremes, lossless, per-CU bypass |
| scaling_list_444_10 / ctu16_444 / ctu32_tu4_444_10 / slices4_444_10 | 352x288 | 4:4:4 scaling lists (32x32 chroma), block structure, slices |
| odd_201x117_444 / odd_331x99_444_10 / odd_1918x1078_444_10 | | 4:4:4 at odd widths and heights |
| intra_400 / intra_400_10 / intra_400_nosao_nodbk / medium_400 / slow_400_10 / bframes_400_10 | 352x288 | monochrome intra, presets, B pyramid |
| tskip_400 / weightp_400 / lossless_400 / cu_lossless_400_10 / odd_201x117_400 | 352x288 / 176x144 / 201x117 | monochrome tool toggles, lossless, odd size |
| synthetic (7 + 4 + 2 streams) | 96x64 | tiles, PCM, slices, dependent segments, LTR, list modification; 4:2:2, 4:4:4 and monochrome PCM at 8-12 bits (mixed depths), deblocked, cropped by odd lines |

`cargo test -p filmcraft-hevc --test conformance coverage_report -- --ignored --nocapture` prints the
decoder statistics per fixture.

## Performance

`cargo test --release -p filmcraft-hevc --test perf -- --ignored --nocapture` (bit-exactness is
verified first). Apple M4 Pro (14 cores), measured while the machine was heavily shared (load
average 50-100), so these are lower bounds:

| stream | 1 thread | 14 threads |
|---|---|---|
| 1080p Main, x265 medium CRF 24, 60 frames | ~75 fps | ~225 fps |
| 1080p Main 4:2:2 10, x265 medium CRF 24, 60 frames (2026-11-25, load average ~12) | ~94 fps | ~285 fps |
| 1080p Main 4:4:4 10, x265 medium CRF 24, 30 frames (2026-11-25, load average ~10) | ~78 fps | ~150 fps |
| 2160p Main 10, x265 fast CRF 26, 20 frames | ~20 fps | ~95 fps |

No intrinsics: the hot loops are written to auto-vectorise. M4.8
profiling (`cargo xtask bench`, macOS `sample`) found the inverse transform at over half of the
decoder's CPU time: it evaluated the full matrix product with strided table reads. It now sums
contiguous basis rows scaled by the non-zero coefficients only, with the block size a compile-time
constant so the loops vectorise (exact integer sums, so bit-exact; `transform::tests` checks it
against the direct product on random sparse and dense blocks). SAO edge offset has a check-free
loop for samples whose neighbours are inside the CTB. Together: ~60 % less CPU per frame
(1080p 78 → 33 ms, 2160p 320 → 129 ms of CPU per frame through the media stack, alternating A/B), see
[docs/performance.md](../../docs/performance.md).
`cargo run --release -p filmcraft-hevc --example hevcdec -- in.hevc out.yuv` decodes a file
(`HEVC_THREADS=n`, `HEVC_LOOPS=k`, `HEVC_DRAFT=1`).

M4.10 (2026-10-03), CPU cycles per frame counted by the kernel (`/usr/bin/time -l`, all threads
summed; mean of two interleaved base / after rounds at load average 190-320) on the
`cargo xtask bench` decode clips (x265 preset fast, CRF 22, testsrc2 + grain), bit-exact:

| stream | threads | Mcycles / frame before | after | after, draft mode |
|---|---|---|---|---|
| 1080p (120 frames) | 1 | 90.9 | **67.3** (−26 %) | 61.2 |
| 1080p | 14 | 100.2 | **75.1** (−25 %) | 67.3 |
| 2160p (72 frames) | 1 | 362.6 | **262.3** (−28 %) | 232.5 |
| 2160p | 14 | 398.3 | **284.8** (−28 %) | 260.1 |

What changed: motion compensation zeroed two 10 KB windows per prediction block per component
and list (11 % of the time in `memset`) and filtered with dynamic-width scalar sums; it now keeps
reusable scratch windows and runs taps outer / outputs inner over fixed block widths. SAO
looked its offset up per sample in a table (20 % of the time); the edge / band offset is now
selected per lane from the neighbour-sign sum / band index, so the runs vectorise. Uni / bi
output clips with min / max instead of `clamp` (whose bound assert kept it scalar). Runs of
bypass bins (signs, Rice suffixes, SAO / intra fields) decode with one division, and
context-coded bins are branch-free. Single-threaded 4K now splits into CABAC residual parsing
~28 %, reference window copies ~11 %, interpolation ~11 %, deblocking + SAO + row publication
~15 %, inverse transform ~8 %.

Residual decoding (2026-11-25, from craft-libs c25999c8a4 / FilmCraft #3188): each coefficient is
scaled (8.6.3) as it is decoded and kept in a list of (index, value), instead of a pass over the
coefficient bounding box after a clear of the whole block; only the bounding box is cleared
before the list is scattered into it (the inverse transform reads nothing else); DC-only blocks
add the inverse-DC value straight to the prediction; sig_coeff_flag contexts come from a table
per sub-block (scan, coded-neighbour pattern) and the last position from inverse scan tables;
merge candidates stay in the fixed five-entry list. Output is byte-identical on the 76 fixture
streams that decode. CPU cycles per frame, one thread, release, alternating rounds (kernel counters):

| stream | before | after |
|---|---|---|
| bench_1080p (Main, x265 medium CRF 24) | 27.9 M | **26.8 M** (−4 %) |
| bench_1080p_422_10 (Main 4:2:2 10, medium CRF 24) | 44.0 M | **41.4 M** (−6 %) |
| 1080p Main, x265 fast CRF 12 (~160 Mbit/s) | 322.5 M | **291.7 M** (−10 %) |

## Known gaps

- The range extension coding tools (implicit / explicit RDPCM, cross-component prediction,
  extended precision, persistent Rice adaptation, bypass alignment, chroma QP offset lists, SAO
  offset scaling) are rejected with `Error::Unsupported`, and so are separately coded colour planes.
  4:2:2, 4:4:4 and monochrome streams that use none of those tools (what cameras and libx265 write)
  decode.
- Tiles and WPP substreams are decoded sequentially (entry points are parsed but the decoder finds
  substream starts itself); parallelism is at the picture level.
- Error concealment is minimal: missing references are replaced by grey pictures, undecodable
  slices leave grey / partial content.
- No fixture covers the `pic_output_flag = 0` path or decoding-order-only CRA handling after EOS
  beyond the synthetic streams; libx265 never emits PCM, tiles or long-term references (covered by
  the synthetic streams only).
