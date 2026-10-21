# GPU use

> **Last reviewed:** 2026-10-21 · **Last updated:** 2026-10-21 · **Change:** major (new file: what runs where, the fallbacks and how to force the CPU; GPU export on by default; masked effects, dissolves of clips that do not fill the frame, the detectors, Whisper, Kokoro and OpenVoice on the GPU) · **Target:** Adobe Premiere Pro 2026 (26.5.2)

The owner's decision (2026-10-21): **the GPU wherever it is supported.** GPU output does not have
to be bit-identical to the CPU's, only within the golden / parity tolerances (8-bit p99 ≤ 6, mean
< 1.5 away from layer outlines; the showcase scene is at 50–88 dB PSNR). The CPU stays the
reference implementation and the fallback for everything: a GPU that is missing, fails, panics
or cannot draw something never fails a job or the app, the CPU does that piece instead.

Hardware decoding and encoding (VideoToolbox, Media Foundation, VA-API, NVDEC, NVENC) are
covered in [hardware-parity.md](hardware-parity.md) and `crates/platform/README.md`; this page is
about compositing, effects and the on-device models.

## What runs where

| Path | Device (before 2026-10-21 → now) | Fallback | Force the CPU |
|---|---|---|---|
| Program monitor playback (desktop) | GPU (`GpuCompositor`, wgpu) | per layer / frame: what the plan leaves to the CPU (below); WebGL2-class devices without compute shaders render effects on the CPU | `FILMCRAFT_CPU_COMPOSITE=1` |
| Export rendering | CPU (GPU opt-in) → **GPU by default** (`GpuRendering::Auto`: export settings, every built-in preset, settings saved without the field, `file.exportMedia`, the CLI, AutoCut's showcase and shorts) | per frame: no adapter, a device error or loss (the CPU for the rest of the session), a renderer panic, a frame the plan hands back as one CPU image, and, up to 1080p, a frame of one picture without effects or masks with only titles over it and no transition (`plan::cpu_is_quicker`: the CPU draws it at least as quickly as a GPU composite read back; such batches keep every export thread; at 4K the GPU is quicker and keeps them); per layer as below | `gpuRendering: "off"`, `--gpu-rendering off`, Export ▸ "Composite on the GPU when available" |
| Render previews (render in to out) | CPU → **GPU** (an export, `GpuRendering` default) | as export | as export (`FILMCRAFT_CPU_COMPOSITE` does not apply) |
| The CLI's GPU export | none: `cargo build -p filmcraft-cli` had no wgpu backend, `--gpu-rendering auto` rendered on the CPU → **Metal** (macOS), DX12 / Vulkan (Windows), Vulkan / GL (Linux) | as export | as export |
| Standard effects on media clips | GPU: 34 effect ids (`render::gpufx::GPU_EFFECTS`: colour, levels, Lumetri basic / creative / vignette, Gaussian / directional / camera blur, sharpen, unsharp, crop, transform, flips, mirror, offset, vignette, video limiter, ASC CDL, channel mix…) | an effect outside the list, a non-finite parameter or a Transform that minifies → the clip's layer is rendered on the CPU and drawn by the GPU | — |
| Masked effects (the redaction blur), opacity masks | CPU layer → **GPU** (`FxOp::Masked`, `FxOp::OpacityMask`) | more than 65 536 mask vertices or non-finite mask values → CPU layer | — |
| Blurs behind portraits (blur fill), redaction blur | GPU (box passes, running sums for large radii) | — | — |
| Opacity and motion keyframes | GPU (evaluated on the CPU at the frame's time, drawn by the GPU) | — | — |
| Cross dissolve, dip to black / white | GPU | — | — |
| Cross dissolve between clips that do not fill the frame (scaled, moved, transparent) | GPU but **wrong** (A at full, B over it: up to 164 levels off) → **GPU, the CPU's linear mix** (a two-input step) | — | — |
| Push, whip / camera, light-leak, flow, GL Transitions, Luma Wipes, Blend Dissolves | GPU (two-input steps, WGSL ports with Rust references) | a shader the device rejects, a flow transition without its flow field → that frame on the CPU | — |
| Other transitions (obsolete / legacy ones such as Page Turn) | CPU (whole frame) | — | — |
| Titles, text, shapes (graphic clips) | rasterised on the CPU (cached per clip and time), placed by the GPU as a layer | graphic clips with standard effects or masks: CPU layer | — |
| Adjustment layers | CPU (whole frame) | — | — |
| HDR / wide-gamut sequences | CPU (whole frame) | — | — |
| HDR / wide-gamut media on a Rec. 709 sequence | GPU (input transform in WGSL) | log media: CPU layer | — |
| Colour LUTs (Lumetri Input / Creative LUT, curves, wheels, HSL) | CPU layer (`GpuLut` exists with parity tests, not in the live path) | — | — |
| Nested sequences | plain ones inlined into the GPU plan; others rendered on the CPU as a layer | — | — |
| Frame blending / optical-flow speed changes | CPU layer | — | — |
| Source monitor, item thumbnails, media browser, multicam grid | CPU (`render_item`: decode and scale one frame) | — | — |
| Scopes, Reference monitor, Export-mode preview | CPU (`render_sequence` at 250–1000 px, scope maths on the CPU) | — | — |
| Audio waveforms, meters, mixing, audio effects | CPU | — | — |
| Auto Reframe analysis, scene detection, mask tracking, flow fields of flow transitions | CPU | — | — |
| Face and plate detectors (automatic redaction) | CPU → **GPU on macOS** (fused WGSL passes through wgpu, `crates/vision/src/gpu.rs`) | no adapter / other platforms: CPU; a picture that fails or panics: again on the CPU; three failures in a row or a device error: the CPU from then on | `FILMCRAFT_VISION_CPU=1` |
| Whisper transcription (`transcript.generate`, AutoCut, voice lab) | CPU → **Metal** (candle) | load fails: CPU; a transcription that fails or panics: again on the CPU, kept there | `FILMCRAFT_SPEECH_CPU=1` |
| Whisper strict check of narration (Malayalam) | Metal | CPU | `FILMCRAFT_SPEECH_CPU=1` |
| Kokoro voices | CPU → **Metal** (candle) | weights the GPU cannot hold: CPU; a line that fails, panics or is not finite: again on the CPU, kept there | `FILMCRAFT_KOKORO_CPU=1` |
| OpenVoice tone-colour converter (timbre-only clones) | CPU → **Metal** (candle) | as Kokoro | `FILMCRAFT_OPENVOICE_CPU=1` |
| VoxCPM natural clones | Metal | CPU | `FILMCRAFT_VOXCPM_CPU=1` |
| IndicF5 (Malayalam clones) | Metal | CPU | `FILMCRAFT_F5_CPU=1` |
| ACE-Step neural music | Metal | CPU | `FILMCRAFT_ACESTEP_CPU=1` |

Metal (candle) and the WGSL detectors are macOS only; Windows and Linux keep the CPU for the
models (no CUDA). The wasm build compiles none of it: the model features are off there and the
GPU compositor uses the browser's WebGPU / WebGL2.

The machine-wide GPU lock (`filmcraft-cli locks`, `crates/locks`) is taken by the large models
(VoxCPM, IndicF5, ACE-Step) so parallel sessions take turns; Whisper, Kokoro, OpenVoice and the
detectors are small and do not take it.

## Seeing what fell back

`perf.stats` → `export.gpu`:

- `frames`, `fallbacks`: frames the GPU export renderer composited, and frames it left to the CPU.
- `cpuWork`: what GPU plans (exports and the monitors) rendered on the CPU, by reason, counted per
  frame: `frame: …` (the whole frame: adjustment layer, HDR sequence, a transition this GPU cannot
  draw, one picture without effects up to 1080p) and `layer: …` (one layer drawn by the GPU from a CPU image: `effect <id>`, `graphic (text
  and shapes rasterised on the CPU)`, `nested sequence`, colour-managed log media…).

On the 4K showcase (`Every Robot I've Met`) before the masked effects moved to the GPU, a 6 s
range rendered 180 GPU frames with 180 `layer: masked gaussian_blur` (the redaction) and 1 266
`layer: graphic` (7 titles and shapes a frame); now the redaction is a GPU effect.

## Measured

Apple M3 Max, macOS, release builds, 2026-10-21. Parallel lanes kept the load average at 20–60,
so CPU timings are inflated (GPU timings much less so); exports were run in turns, before and
after, on the same project copy (`showcase-v5.fcproj`, the YouTube 2160p / 1080p presets with
VideoToolbox encoding).

**Models** (per call; the GPU's first call, which compiles its kernels, not counted):

| Model | CPU | GPU | Speed-up | Equivalence |
|---|---|---|---|---|
| YuNet faces, long side 1920 / 1280 / 640 | 235 / 115 / 81 ms (42 ms at 1920 on a quiet machine) | 29 / 19 / 11 ms | 5–8× (1.5× against a quiet CPU) | the same 220 faces on 16 frames of the showcase's footage, boxes 0.000 px apart |
| LPD-YuNet plates, 960 | 281 ms (94 quiet) | 40 ms | 7× (2.4× quiet) | raw outputs within 9.3e-7 (relative) |
| Whisper-base, 58 s of narration in 7 clips | 40.5 s | 4.4 s | 9.1× | the same words and word times in all 7 |
| Kokoro (Heart), a 36.7 s script | 43.6 s | 5.7 s | 7.6× | Whisper hears the same words (WER 4.1 % both); waveforms 48.5 dB apart |
| OpenVoice conversion, 12 s | 12.0 s | 1.6 s | 7.4× | 96 dB apart; tone colours within 9e-6 |
| VoxCPM2, IndicF5, ACE-Step | — | Metal already | — | (unchanged) |

candle's Metal backend was tried for the detectors first: five times *slower* than the CPU (a unit
as generic tensor operations whose broadcasts are slow on Metal), hence the WGSL kernels.

**Automatic redaction** (`clip.autoRedact`, four clips of the 4K showcase, 210 analysed frames,
fresh, dry run, the same 91 tracks): 33.6–36.6 s wall and 41–45 s of CPU with the CPU detectors,
5.0–7.4 s and 7.5 s of CPU with the GPU's (5–7×).

**Exports** of the showcase (A/B in turns, two rounds each, load average 27–30; the baseline is the
build before this work, whose defaults rendered on the CPU):

| Range | What it shows | CPU (before) | GPU default (now) |
|---|---|---|---|
| 4K, 6 s (180 frames) | a redacted clip (blur behind tracked face masks), titles | 12.3–12.7 s, 21 s CPU | **7.7–8.2 s**, 17.7 s CPU (1.55×) |
| 4K, 15.6 s (468 frames) | scaled clips, blur fills, transitions, titles | 23.2–25.1 s, 55 s CPU | **18.4–18.6 s**, 52 s CPU (1.3×) |
| 4K, 10 s (300 frames) | one 1080p clip zoomed to 4K, titles | 15.2–20.2 s (load 37–47) | 16.0–19.9 s (the same) |
| 1080p, 10 s (300 frames) | one 1080p clip with a slow zoom, titles: left to the CPU (`cpu_is_quicker`) | 3.7–4.1 s, 9.0 s CPU | 2.9–3.8 s, 9.1 s CPU (+1.4 s of GPU start-up before the export) |

Before the masked blur moved to the GPU, the 6 s 4K range took 11.8 s on the GPU (39 s of CPU)
against 3.4–5.1 s on the CPU at the time: the redaction was a 4K CPU layer uploaded every frame.
Before the plain frames went back to the CPU, the 1080p range took 7.0–8.3 s.

## Gaps left on the GPU

- Effects outside `GPU_EFFECTS` (Mosaic, keys such as Ultra Key, distortions, Lumetri curves /
  wheels / HSL secondary and LUTs, stylize effects): a CPU layer each.
- Titles and text are rasterised on the CPU (cached; an animated title re-rasterises per frame).
- Adjustment layers and HDR / wide-gamut sequences render whole frames on the CPU.
- Obsolete / legacy transitions; flow transitions need their flow field, computed on the CPU.
- Thumbnails, the Source monitor, scopes and the Reference monitor render on the CPU.
- Decoded frames are uploaded from CPU memory (no zero-copy from VideoToolbox into wgpu), and the
  GPU export reads every frame back as half floats, converted to f32 and then to 8 bits on the
  CPU, on at most four renderers: the read-back and conversions are why one picture without
  effects is as quick on the CPU. Converting to the encoder's 8-bit (or Y'CbCr) format on the GPU
  would cut the read-back by four.
- An export with GPU rendering creates its four renderers (shaders and pipelines) before its first
  frame, about 1.4 s, even when every frame then stays on the CPU.
- Mosaic (the redaction's other style) is not a GPU effect.
- The models run on the GPU only on macOS.

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-21 | major | Created: inventory of every compute-heavy path and its device, fallbacks, switches; GPU export on by default; masked effects, dissolves of clips that do not fill the frame, the detectors, Whisper transcription, Kokoro and OpenVoice moved to the GPU |
