# filmcraft-tts

Text to speech for FilmCraft's Text to Speech panel (narrations, L2). Engine commands (`tts.*`) and
clip behaviour: `crates/engine/src/narration.rs`.

- `Voice` trait (script in, mono `f32` out at the voice's rate: 24 kHz, 44.1 or 48 kHz for natural clones), the voice catalogue (`voices`, `voice`), and
  the limits (64 KB script, 30-minute narration, pace 0.5–2×, pitch ±12 semitones).
- `script`: pause markers `[pause 1s]` / `[pause 500ms]` (case-insensitive, capped at 10 s;
  malformed markers are read as text).
- `formant`: the built-in voices **Basic Female** and **Basic Male**, an original source–filter
  formant synthesizer: spelling rules → phone targets → glottal pulse train through three two-pole
  resonators, plus filtered noise for fricatives and bursts. No download, works in the web build.

- `kokoro` (feature `kokoro`): **Kokoro-82M neural voices** in pure Rust on candle (CPU): ALBERT
  text encoder, StyleTTS 2 prosody predictor (durations, F0, energy) and decoder, iSTFTNet
  vocoder with a harmonic-plus-noise source. Weights are read from the official PyTorch
  checkpoint with candle's restricted pickle reader (no code from the file runs); voice packs from
  their zip archive. `config.json` is checked against the architecture this code implements.
- `neural` (feature `kokoro`): the natural voices as `Voice`s: script → `filmcraft-tts-text`
  (normalizer, CMUdict + letter-to-sound, ≤ 400-symbol chunks, pauses) → Kokoro per chunk → exact
  silence for pauses, 120 ms between sentences → vocal pitch with `filmcraft-audio-dsp`'s pitch
  shifter (Kokoro has no pitch input). Pace is Kokoro's speed.
- `catalog`: the natural-voice package (Kokoro-82M, 9 US English voices graded C+ or better on the
  model card, CMUdict; 336 MB), every file pinned to a revision and SHA-256. Downloaded by the
  engine (`tts.downloadVoices`) after the user confirms; never bundled or committed.

- `openvoice` (feature `kokoro`): the **OpenVoice V2 tone-colour converter** in pure Rust on
  candle — a VITS synthesizer without a text encoder: a reference encoder (six 3×3/2 conv2d →
  GRU → 256-d *tone colour*), a posterior encoder, four mean-only affine couplings conditioned on
  the colour, and a HiFi-GAN decoder at 22.05 kHz. `tone_colour` reads a speaker's timbre from
  their speech; `convert` re-voices speech from one colour to another, keeping words, timing and
  intonation. Implemented from the paper, the MIT reference code, `converter/config.json` and the
  checkpoint's tensor names (weight norm folded on load, including 2-D convolutions).
- `clone` (feature `kokoro`): **cloned voices**. A profile (`<data>/voices/<name>.json`) holds the
  speaker's tone colour, the Kokoro voice it speaks through (by default the one whose colour is
  closest to the speaker's) and that voice's colour; `clone:<name>` speaks Kokoro's reading
  re-voiced into the speaker's timbre. Speech is found in the reference by an adaptive energy
  gate; 3–120 s of it is used. Profiles record consent: `create` refuses without it.
  `examples/clone_check.rs` measures a clone — on the author's voice the timbre similarity to the
  speaker rose from 0.50 (base voice) to 0.90 (cloned).
- `catalog` also pins the cloning packages: OpenVoice V2 `converter/config.json` and
  `checkpoint.pth` (131 MB, MIT, MyShell), and the two VoxCPM releases (below), downloaded by
  `tts.downloadCloneModel {"model": …}` after the user confirms size, source and licence.

- `voxcpm` (feature `voxcpm`): **VoxCPM zero-shot voice cloning** in pure Rust on candle — a
  port of the Apache-2.0 reference inference (OpenBMB, `src/voxcpm` at cce58a5). VoxCPM is a
  tokenizer-free, diffusion-autoregressive TTS: text tokens and a voice sample's continuous
  latents go through a MiniCPM-4 text-semantic LM; a residual acoustic LM refines its state; a
  local DiT turns both into the next *patch* of latents (four 25 Hz frames) by flow matching
  (Euler on the sway schedule, CFG-Zero★ guidance); LocEnc feeds the patch back; a stop head ends
  the utterance; the causal AudioVAE decodes. One code path runs **VoxCPM1.5** (0.8 B, 44.1 kHz,
  1.95 GB) and **VoxCPM2** (2.3 B, 16 kHz in / 48 kHz out, 4.96 GB; the default).
  - `config`: `config.json` checked (sizes capped, divisibility, an unknown architecture refused).
  - `weights`: a safetensors reader that checks the header and seeks per tensor (no `unsafe`
    mmap, no whole-file buffer; bf16 → f32), and the AudioVAE checkpoint through the Kokoro
    pickle loader (weight norm folded).
  - `tokenizer`: a minimal SentencePiece-style BPE over `tokenizer.json` (`▁` normalizer, ranked
    merges, byte fallback, the reference's CJK split), refusing any other tokenizer type; its ids
    equal the Hugging Face library's (live test).
  - `text`: `filmcraft-tts-text`'s normaliser (numbers, years — "2022" → "twenty twenty-two" —,
    money, dates, abbreviations, letter-spelled acronyms, "Go1" → "Go one") with the punctuation
    put back, pause markers as exact silence, sentences grouped into ≤ 240-character utterances.
  - `minicpm`, `local`, `vae`, `model`: the stacks (RMSNorm, LongRoPE, grouped-query attention
    with a KV cache, SiLU MLP), LocEnc / LocDiT and the solver, the AudioVAE (depthwise
    convolutions and Snake as loops over threads, the dense ones as matrix products; run in
    windows with left context above the receptive field, so memory is bounded and the result
    equals one pass), and the prompt layout, prefill and patch loop (stop head, at most six
    patches per text token plus ten, regenerated with the next seed when an utterance runs into
    that cap; at most the model's 8192 positions).
  - Devices: on macOS the models run on the GPU through candle's Metal backend (the fused
    attention kernel, padded around candle's wrong result for 2–8 queries; VoxCPM2's linear
    weights stay bf16 there: 5 GB instead of 9), elsewhere on the CPU; `FILMCRAFT_VOXCPM_CPU`
    forces the CPU.
  - Checked against the PyTorch reference run as an oracle with the same noise: the first
    patches agree to 1e-4 (VoxCPM1.5 zero-shot and continuation; VoxCPM2 reference +
    continuation); later ones drift only where the scalar quantizer rounds a borderline value.
- `clone` also makes **natural clones** (profile version 2, engine `voxcpm`): a *voice sample*
  is kept next to the profile (`<name>.wav`: the best continuous 10–25 s of speech between
  pauses, long pauses shortened, −20 LUFS, at the encoder's rate) with its transcript. VoxCPM2
  uses it as an isolated reference (no transcript needed); VoxCPM1.5 continues a shorter one
  (5–10 s), which needs its exact transcript (Whisper or given) and gets 18 dB of noise
  reduction. `clone:<name>` dispatches on the profile's engine; version 1 profiles load
  unchanged; the narration cache keys a natural clone on its sample's bytes and settings.
- `wav`: WAV reading (16/24/32-bit PCM, float, any channels; hostile files are errors) and
  writing for voice samples.

- `cache`: the **narration cache** — synthesized speech kept on disk so a script is read by a
  voice once, across sessions and rebuilds (a cloned voice runs at about 2.6× real time: a
  25-line show is minutes per rebuild otherwise). The engine keeps it in
  `<data dir>/narration-cache` (native frontends turn it on; `Session::use_narration_cache`) and
  looks there after its in-memory cache, before synthesizing, for `tts.create`, `tts.render`,
  `tts.edit` and `tts.preview`.
  - Key: SHA-256 of the cache format version, the voice's identity (`voice_identity`: its id and
    the pinned model files it speaks with; for a cloned voice also its profile's id, version,
    creation time, base voice and a hash of its tone colours), the pitch and pace (exact bits) and
    the script. A re-made cloned voice, a new model revision or a format bump never hits old
    entries; they age out.
  - Entry: `<key>.fcnc`, a 28-byte header (magic, version, rate, sample count, FNV-1a checksum)
    and the samples as `f32`, bit-exact. Written to a temporary file and renamed into place.
  - Capped at 2 GB: after each write the least recently used entries (modification time,
    refreshed on every hit) are removed until the rest fits; stale temporary files go too.
  - An unreadable, truncated or corrupt entry is removed and counts as a miss; a cache that
    cannot be written only means the voice speaks again. The cache never fails a synthesis.

Everything is deterministic (the vocoder's noise is seeded). Without features the crate depends
only on `serde`, `thiserror` and `sha2` (and checks for wasm).

## Measured quality and speed (M7.12, 2026-10-08)

50 original sentences (`tests/data/sentences.txt`: numbers, money, times, dates, phone numbers,
acronyms, names) spoken by each voice through the full pipeline, transcribed by Whisper-base
(FilmCraft's own, `filmcraft-speech`), word error rate after lowercasing and removing punctuation
(number formatting such as "six" vs "6" still counts as an error):

| Voice | WER | | Voice | WER |
|---|---|---|---|---|
| Heart | 2.1 % | | Puck | 2.8 % |
| Aoede | 2.4 % | | Michael | 3.2 % |
| Nicole | 2.8 % | | Kore | 3.4 % |
| Bella | 3.0 % | | Fenrir, Sarah | 3.6 % |
| Basic Female (built in) | 96.6 % | | | |

Speed: 2.2–2.3× real time on one desktop CPU (Linux x86-64, load average ≈ 15 from parallel
builds), e.g. 186 s of speech in 84 s; model load 0.5 s. Reproduce with the `narrate` example and
`filmcraft-speech`'s `transcribe` example.

## Natural cloning: measured quality and speed (2026-10-16)

The author's own recording (a phone video, 0–120 s, Indian English, room noise: speech about
−25 dBFS over a −38 dBFS floor), cloned through `tts.cloneVoice`, three test sentences each
(`crates/engine/examples/clone_eval.rs`), on an Apple M3 Max:

| Voice | Similarity, ECAPA | Similarity, OpenVoice colour | WER | Real-time factor |
|---|---|---|---|---|
| Timbre only (Kokoro + OpenVoice, the existing clone) | 0.15 | 0.852 | 1.6 % | 1.09 (CPU) |
| **Natural, VoxCPM2** (reference, 24.5 s sample) | **0.56** | 0.777 | 4.8 % | **0.79** (Metal), 4.8 (CPU) |
| Natural, VoxCPM1.5 (continuation, 24.5 s sample) | 0.49 | 0.807 | 123 % | 0.62 (Metal) |
| Natural, VoxCPM1.5 (continuation, 9.6 s sample) | 0.45 | 0.782 | 198 % | 0.62 (Metal), ≈ 2.3 (CPU) |
| The speaker: another stretch of the recording | 0.85–0.94 | 0.96–0.98 | | |

- *Similarity* is the cosine against held-out speech of the speaker (the recording minus the
  voice sample). **ECAPA** is SpeechBrain's ECAPA-TDNN speaker-verification embedding
  (`spkrec-ecapa-voxceleb`, Apache-2.0, run as an external measuring instrument, not shipped);
  the **OpenVoice colour** is the converter's own reference encoder, which favours the timbre-only
  clone by construction (its converter targets exactly that embedding). By ECAPA the timbre-only
  clone is a different speaker (0.15); VoxCPM2 is recognisably the same one (0.56; SpeechBrain's
  own verification threshold is 0.25).
- *WER*: Whisper small against the sentence ("Gitex"/"GTX" for "GITEX" counts as an error).
- VoxCPM1.5's continuation is fragile on such a recording: with a sample whose transcript is not
  exact (Whisper hears this accent imperfectly) or that is noisy, it keeps "finishing" the sample
  before or instead of the text. On a clean sample with an exact transcript it is accurate (a
  VoxCPM sample continued by VoxCPM1.5: WER 0 %). Hence VoxCPM2 in reference mode is the default.
- Speed: VoxCPM1.5 runs ~0.6× real time on the M3 Max GPU, VoxCPM2 ~0.8×; on the CPU (candle's
  gemm) ~2.3× and ~4.8×. Loading takes 2–5 s; a voice sample is encoded once per profile.

## Limitations

- The built-in voices are **robotic** and nearly unintelligible to a recogniser (above); they are
  the no-download fallback and the test voice.
- English (United States) only. Words outside CMUdict go through letter-to-sound rules and are
  often mispronounced (see `crates/tts-text/README.md`); homographs take the first pronunciation.
- About 2× faster than real time on the CPU: a one-minute narration takes about 30 s. The UI runs
  synthesis as a background job.
- Kokoro's model card states the weights are Apache-2.0 but the repository has no LICENSE file.
- Vocal pitch shifts the voice's base pitch (and its formants slightly); pace scales every duration.
- A *timbre-only* cloned voice copies **timbre**, not accent or rhythm: those are still Kokoro's
  (US English). Natural (VoxCPM) clones copy accent and rhythm too, but are large downloads
  (2–5 GB) and want a GPU: on the CPU VoxCPM2 is about 5× slower than real time.
- Natural clones are English-normalised (the text front end); VoxCPM2 itself speaks 30 languages.
  VoxCPM1.5 needs a clean voice sample with an exact transcript (above). Natural voices reproduce
  the sample's recording colour (room, microphone) along with the voice.

## Tests

`cargo test -p filmcraft-tts`: determinism, valid samples (finite, peak ≤ 0.8, audible RMS), pause
markers produce exactly the requested number of zero samples, pace 2× / 0.5× gives 0.5× / 2× the
length (±5 % / ±10 %), measured pitch (autocorrelation) of the female voice is ≥ 1.5× the male, and
+6 semitones raises it by √2 (±0.12); empty, emoji-only, non-Latin, pause-only and over-long scripts
are refused before rendering; hostile scripts never panic. The narration cache: a second reading
of the same script and settings synthesizes nothing (a counting voice) and is bit-exact; another
pace, pitch, voice or script misses; damaged entries (empty, truncated, flipped bits, NaN samples,
absurd sizes, wrong magic, trailing bytes) are removed and synthesized again; the cap evicts the
least recently used; an unwritable cache never fails a synthesis; a cloned voice's identity
changes when its profile is made again (feature `kokoro`).

VoxCPM (feature `voxcpm`): config, tokenizer, safetensors and WAV readers fuzzed (truncations,
byte flips, doctored headers, NaN); attention (batched, and the Metal kernel) against plain loops;
KV-cache stepping equals a full pass; windowed AudioVAE equals one pass and is causal; a tiny
random VoxCPM 1 and 2 speak deterministically, with and without prompts, cap runaway
utterances and refuse texts past their context; natural profiles round-trip and are checked; the
voice sample is the densest stretch between pauses at −20 LUFS. With a release in
`FILMCRAFT_VOXCPM_DIR`: the tokenizer's ids equal Hugging Face's, the reference oracle comparison
(`VOXCPM_ORACLE`), and per-part timings (`live_voxcpm_parts`).

Model-dependent tests run when `FILMCRAFT_KOKORO_DIR` names a downloaded package directory
(determinism, whole frames, pace changes length, refusals); the loaders have hostile-input tests
(truncated or oversized voice data, wrong byte order, NaN, non-zip files, mismatched config).

## References

- VoxCPM: OpenBMB, github.com/OpenBMB/VoxCPM (Apache-2.0; `src/voxcpm` read and ported), the
  `config.json` files of openbmb/VoxCPM1.5 and openbmb/VoxCPM2; MiniCPM-4 (OpenBMB); flow
  matching (Lipman et al., ICLR 2023); CFG-Zero★ (Fan et al., 2025); Snake (Ziyin et al.,
  NeurIPS 2020). Weights: OpenBMB, Apache-2.0.

- G. E. Peterson and H. L. Barney, "Control Methods Used in a Study of the Vowels", JASA 24 (1952):
  the published average formant frequencies used for the vowels.
- D. H. Klatt, "Software for a cascade/parallel formant synthesizer", JASA 67 (1980): the standard
  two-pole digital resonator form. No code was read or copied; everything else is original.
- Kokoro (implemented from papers, `config.json` and the checkpoint's tensor names and shapes; no
  reference code read): Y. A. Li et al., "StyleTTS 2" (NeurIPS 2023); T. Kaneko et al., "iSTFTNet"
  (ICASSP 2022); Z. Lan et al., "ALBERT" (ICLR 2020); J. Kong et al., "HiFi-GAN" (NeurIPS 2020);
  X. Wang et al., "Neural source-filter waveform models" (2019); L. Ziyin et al., "Neural Networks
  Fail to Learn Periodic Functions and How to Fix It" (Snake, NeurIPS 2020). Weights: hexgrad,
  Kokoro-82M, Apache-2.0.
