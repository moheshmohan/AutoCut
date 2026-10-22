# filmcraft-voice

Voice isolation for FilmCraft: **DeepFilterNet 3** (Hendrik Schröter et al., Interspeech 2023;
MIT OR Apache-2.0) in plain Rust. It is the neural model behind the **Voice Isolation** and
**Enhance Voice** audio effects (`filmcraft_audio_dsp::effects::voice`, `enhance_voice`), the
Essential Sound repair options *Isolate Voice* and *Enhance Voice*, `audio.isolateVoice`,
`audio.enhanceVoice` and `filmcraft-cli audio enhance`. L2; pure Rust (dependencies:
`filmcraft-audio-dsp`, `filmcraft-vision` for its ONNX reader, `miniz_oxide`, `serde`,
`thiserror`; no `-sys` crate, no `cc` build).

## Model

Downloaded by `audio.downloadVoiceModel` after the user agrees, pinned to a commit and SHA-256
(`catalog`), never bundled or committed:

| file | size | source | licence |
|---|---|---|---|
| `DeepFilterNet3_onnx.tar.gz` (encoder, ERB decoder, DF decoder as ONNX; `config.ini`) | 7,983,136 B | `Rikorose/DeepFilterNet` at `d375b2d8309e0935d165700c91da9de862a99c31`, `models/` | MIT OR Apache-2.0, © 2021 Hendrik Schröter |

It goes into `<data dir>/models/deepfilternet3/`. Hosts call `install_loader(models_dir)` once;
the first effect that needs the model loads it (46 ms) and every later one shares the weights.
Without it the effects run their DSP fallback (a decision-directed Wiener filter on the same
frames, with the same 40 ms latency).

## Why not libDF

DeepFilterNet's own Rust library runs the ONNX files on `tract`. `tract-linalg` compiles
hand-written assembly kernels with the C toolchain (`cc`) in its build script, and `psm` (through
`liquid`) does the same at build time, so it is not pure Rust. This crate implements the network
by hand instead (`net`, about 3.3 M multiply-adds per 10 ms frame as plain loops the compiler
vectorises), and takes the weights out of the same ONNX files (`weights`): each layer is found
by the node that computes it, and its operator, groups, kernel, strides, padding and tensor shapes
are checked against what the code implements, so another file is refused, never misread.
`df3` follows libDF's streaming implementation (`tract.rs`, `lib.rs`: features, normalisation,
the local-SNR stages, the deep filter).

## Measured (2026-10-22, M3 Max, release build, the machine shared with other builds)

| | |
|---|---|
| matches libDF (`DfTract`, defaults) on 20 s of test signal | the difference is 92.6 dB below the signal; worst sample 5·10⁻⁶ |
| the model stream, mono | 0.21 ms per 10 ms hop: real-time factor 0.021 (libDF on tract: 0.017) |
| the Voice Isolation effect, 48 kHz | RTF 0.027 mono, 0.031 stereo (32× real time); latency 1920 samples |
| the effect at 44.1 kHz stereo (resampled to 48 kHz and back) | RTF 0.048; latency 1802 samples |
| speech-like test signal, SI-SDR | clean: 15.4 dB out (gaps gated); white noise 0 dB: −0.3 → 12.7 dB (+13.0); pink noise and hum 5 dB: 3.5 → 9.3 dB (+5.8) |

Tests: `cargo test -p filmcraft-voice` runs the plumbing with random weights (block sizes,
per-stream state, hostile input, archive fuzzing). With the model,
`FILMCRAFT_VOICE_MODELS=<models dir>` adds the SI-SDR test; with also
`FILMCRAFT_VOICE_ORACLE=<dir>` holding `noisy.f32` and libDF's `enhanced.f32` (mono 48 kHz
little-endian f32), the comparison with libDF. `examples/voice_bench.rs` measures the speed and
`--dump <dir>` writes `noisy.f32`; libDF's output is made outside this repository (a scratch
crate on `deep_filter` with `tract`, as an external oracle only: never linked or shipped).

## Next

- Speech isolation is speech against everything else: DeepFilterNet keeps every voice, so it
  does not separate one speaker from a crowd talking.
- De-reverberation by weighted prediction error (WPE) as a DeReverb mode is not done yet.
