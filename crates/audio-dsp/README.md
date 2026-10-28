# filmcraft-audio-dsp

Real-time audio DSP for FilmCraft. Layer **L1**, no dependencies but the L0 time crate (exact
`Tick` beat maps), compiles for `wasm32-unknown-unknown`, no `unsafe`. Clean-room: written from the public standards
(ITU-R BS.1770-4, EBU R128 / Tech 3341 / Tech 3342) and textbook DSP (RBJ cookbook biquads,
Householder FDN, phase vocoder, spectral subtraction with the statistical late-reverberation
model, WSOLA time stretching after Verhelst & Roelands, ICASSP 1993).

## Conventions

- Planar `f32` audio: `&mut [&mut [f32]]`, one slice per channel, equal lengths.
- Effects are built (allocating) with `(sample_rate, channels)`; `process`, `set_param` and
  `reset` never allocate.
- Parameters change between blocks and are smoothed per sample; output does not depend on how
  the stream is cut into blocks (tested for every effect with 0/1/random-size blocks).
- Recursive state is flushed below ~1e-30, so no denormals without CPU FTZ flags.

## Mix check (`mixcheck`)

How a mix holds up on small speakers, measured (streaming, any block size, planar `f32`):

- `BandMeter`: the energy share below 120 Hz, 120–500 Hz, 500 Hz–1 kHz, 1–4 kHz, 4–6 kHz and
  above 6 kHz, and the level of each octave band (31.5 Hz…16 kHz) relative to the whole, from
  4096-point Hann-windowed FFT frames with half overlap, channels summed. Tested: a tone lands in
  its band, white noise rises 3 dB an octave.
- `speech_ratio(speech, other, gate, hop)`: over loudness series (momentary, every 100 ms), where
  the speech is at or above its gate (`speech_gate`: 15 LU under its integrated loudness, not
  under −50 LUFS), how far it stood above the other signal — median and quietest tenth; a silent
  other signal counts at −70 LUFS.
- `GainReductionMeter`: a limiter's work, from the signal before and after it (time-aligned) in
  10 ms blocks: the largest reduction, the 99th percentile, time over 1 and 2 dB.
- `ClipMeter`: peak, samples at or over full scale (hot, but not clipping in a float mix) and
  flat tops (runs of 5 equal samples at or above 0.25: a clipped source; `clipped()` from 3).

The engine's `audio.analyzeMix` runs them over a sequence's stems.

## Synthesis (`synth`)

`synth::generate(&Params, sample_rate)` → planar stereo: procedural sound for edits, from
band-limited (PolyBLEP) oscillators, pink/white noise, swept biquads, envelopes and the FDN
reverb — no samples, no models, the same parameters and seed always giving the same audio.

- **Effects** (peak-normalised): `whoosh` (band-passed noise swelling up and down, panned
  across), `impact` (a sub boom dropping from ~110 to ~38 Hz with a noise crack and a long tail),
  `riser` (rising noise and pitch, tremolo speeding up, ending on a cut), `drone` (a slowly
  breathing low chord).
- **Music** (`Kind::Music`): made by the soundtrack engine below.
- Lengths are clamped (30 s for effects, 15 minutes for music), allocation is fallible, and
  nothing non-finite leaves the module.

The engine's `audio.generate` writes them as 48 kHz stereo WAVs and places them on the timeline.

## Soundtrack engine (`music`)

`music::generate(&Request, sample_rate)` → a `Track`: music of an **exact length** (to the
sample: `round(length · rate)` samples of a `Tick` length) that **ends on a bar**, in four stems
that sum to the mix, with its taxonomy and exact beat map. Deterministic: the same request (seed
included) always gives the same audio. Loudness-normalised to −20 LUFS (a bed under a −16 LUFS
voice) and soft-limited.

- **Taxonomy** (`taxonomy`): 9 moods (epic, tense, uplifting, calm, happy, dark, dreamy,
  energetic, sad; synonyms such as "suspense" or "chill" parse), 6 genres (cinematic, electronic,
  pop, ambient, lofi, piano), 30 use cases, and 18 mood × genre styles each with a tempo range,
  an energy and use cases; `suggest(&Query)` ranks them for a brief.
- **Exact time** (`tempo`): the tempo is held in thousandths of a BPM and beat `k` falls on tick
  `round(k · 60 000 · TPS / milli_bpm)` — no drift on long tracks; every note's sample comes from
  its tick.
- **Arranger** (`arrange`): the length fixes the whole bars before the final hit, leaving a
  ring-out of at least two beats (1.2–3 s, under a bar more). A dynamic programme over (bar, last
  section) searches the block graph — intro, verse, build, chorus, bridge, outro blocks of 1–8
  bars; a build resolves into a chorus; the intro is first, the outro or a chorus last — for the
  path filling those bars exactly, closest to the requested energy (a level shaped as an arc, or a
  curve of up to 64 points), with the intro and outro in proportion; the seed nudges near-ties so
  variations differ. No padding, looping or fade to fit: the final chord lands on the downbeat
  and rings out to the last sample.
- **Composer** (`compose`): a progression per mood for verses, choruses and bridges, a dominant
  before the final hit (a cadence), and per-instrument patterns over each bar's chord and energy
  (five tiers from sparse to full): drum lines and fills, risers into choruses, crashes opening
  them; bass lines; a seeded melody (2-bar motifs fitted to the chords, phrases resolving on the
  root or third, long tones in bridges); arpeggios; pads with common tones held across bar lines;
  keys comping or broken piano chords. Lo-fi swings; electronic pumps (side-chain) on the beat.
- **Instruments** (`instruments`): an `InstrumentBank` makes an `Instrument` (note-on / note-off
  events, blocks of dry + reverb-send audio) for each `Patch`. Every patch maps to a General MIDI
  program and every drum to a GM percussion key, so a SoundFont bank plays the same score through
  the same traits: `filmcraft-soundfont` (L2; rustysynth, MIT, pure Rust, behind a structural check
  of the `.sf2`) does, level-matched to these voices through `mix_hints` and their loudness.
  The built-in `OscillatorBank`: PolyBLEP saw / triangle / sine-saw voices with unison, two-operator
  FM (electric piano, bells), additive partials (piano), ADSR or struck envelopes, filter
  envelopes, vibrato; five drum kits (taiko, 909-style, rounded, dusty, soft) and risers/impacts.
  No samples, soundfonts or models are bundled.
- **Rendering** (`render`): every part streams through its instrument (blocks split at note events,
  so timing is sample-exact) into drums / bass / melody / pad stems, each with its own FDN room
  (the mix alone shares one — the reverb is linear); one per-sample gain (loudness, then the soft
  limiter) is applied to the mix and to every stem, so the stems still sum to it. Stems render on
  scoped threads where the platform has them (one thread on the web), with the same result.
- Limits: 0.05 s–15 minutes, stems up to 5 minutes; stem levels −60..+12 dB or muted;
  allocation is fallible; hostile numbers are clamped or refused, never a panic.

`resample::resample_sinc` converts whole buffers offline with a windowed-sinc low-pass (no
aliasing going down, no imaging going up) — for model input/output rates (24 ↔ 22.05 kHz).

## Loudness (`loudness`)

`LoudnessMeter::new(sample_rate, channels)` → `process(&[&[f32]])` / `process_interleaved`, then
`momentary()`, `short_term()`, `integrated()`, `loudness_range()`, `sample_peak()`,
`true_peak()` / `true_peak_dbtp()`, `summary()`.

- K-weighting derived for any sample rate by bilinear transform (matches the 48 kHz BS.1770 table).
- Gated integrated loudness (−70 LUFS / −10 LU) and LRA (−70 / −20 LU, 10th–95th percentile) use a
  0.01 LU histogram with exact energy sums: fixed memory for programmes of any length.
- True peak: 4× (2× at 88.2/96 kHz) polyphase Kaiser-sinc interpolation.
- Oracle-tested against ffmpeg's `ebur128` filter (`tests/loudness_oracle.rs`, skipped without ffmpeg):
  M/S within 0.0005 LU, I within 0.007 LU, LRA within 0.04 LU and true peak within 0.045 dB (0.053 dB of the analytic value on an fs/4 tone) on
  sine, pink-noise, speech-like and gated stereo signals at 44.1/48/96 kHz.
- `normalize_gain_db(measured, target)` and `normalize_gain_db_peak_limited(...)` for Normalize /
  Essential Sound loudness auto-match.

## Effects (`effects`)

All implement `AudioEffect` (`id`, `params`, `set_param`, `param`, `reset`, `process`, `latency`).
`effects()` lists `EffectInfo { id, name, category, params, create }`; each `ParamSpec` has
id, name, range, default, `Unit`, log-scale hint and choice labels so UIs can be generated.

| id | effect |
|---|---|
| `parametric_eq` | 8 bands, each peaking / shelf / LP / HP / notch / band-pass, RBJ biquads in TDF-II |
| `simple_eq` | low shelf, mid peak, high shelf |
| `compressor` | peak/RMS detector, soft knee, attack/release, make-up; stereo-linked |
| `gate` | downward expander / gate with range and hold |
| `limiter` | 5 ms look-ahead, true-peak-aware, ceiling guaranteed (latency reported) |
| `delay` | feedback delay, ms or tempo-synced divisions, mix |
| `reverb` | 8×8 Householder FDN, pre-delay, RT60 decay, HF damping, size, mix |
| `amplify`, `channel_volume`, `balance` | gain, per-channel gain, balance / constant-power pan |
| `invert`, `swap_channels`, `fill_left`, `fill_right` | polarity and routing |
| `dehum` | 50/60 Hz notch comb with up to 10 harmonics |
| `denoise` | STFT spectral gating, minimum-statistics noise floor (latency = FFT size) |
| `pitch_shifter` | ±12 semitones, phase vocoder with peak shifting + phase locking (latency = FFT size) |
| `deesser` | high-pass sidechain at `frequency` (2–12 kHz) detects sibilance relative to the broadband level (`threshold`); a dynamic high shelf cuts the band by up to `reduction` dB (1 ms / 60 ms, stereo-linked); exact identity when idle |
| `dereverb` | STFT late-reverb suppression: λ_r(t) = e^{−2Δ·T_d}·λ_x(t − T_d) (Δ = 3 ln10 / `rt60`, T_d ≈ 50 ms), over-subtraction 1 + `amount`, gain floor down to −18 dB, smoothed gains (latency = FFT size) |
| `speech_enhance` | Enhance Speech DSP chain: 80 Hz HPF, de-mud cut (250 / 350 Hz −3 dB), presence (2.5 / 4 kHz +4 dB) and air (+2 dB @ 10 kHz) boosts by `tone`, downward expander below −50 dBFS (3:1, ≤ 18 dB), 3:1 soft-knee compressor above −24 dBFS, +3 dB make-up, `mix` |
| `stereo_width` | mid/side width 0–200 % (100 % = identity, 0 % = mono) |
| `graphic_eq_10/20/30` | constant-Q peaking bands at octave / half-octave / third-octave ISO centres (Q from the bandwidth), ±24 dB, master gain |
| `parametric_eq_full` | Premiere's Parametric Equalizer: master gain, high-pass and low-pass (12–48 dB/oct Butterworth cascades), low shelf, 5 peaking bands, high shelf |
| `notch_filter` | 6 constant-bandwidth cuts (`FilterType::Cut` = 1 + (g − 1)·BP: the width stays f/Q however deep), Narrow / Very Narrow / Super Narrow |
| `scientific_filter` | Bessel / Butterworth / Chebyshev I / elliptic, LP / HP / BP / BS, order 1–12 (`design`: analog prototype → frequency transform at pre-warped edges → bilinear → biquads; elliptic via Landen-transformation Jacobi functions after Orfanidis) |
| `fft_filter` | 8-point gain curve (log-frequency, linear or smoothstep), applied zero-phase by √Hann STFT at 75 % overlap (latency = FFT size) |
| `dynamics_rack` | Dynamics: auto gate (hold) → expander → compressor (auto make-up) → limiter (soft clip), stereo-linked; `transfer_db` gives the static curve |
| `multiband_compressor` | 4 bands with LR4 crossovers and all-pass phase compensation (bands sum to an all-pass: flat magnitude), per-band threshold/ratio/attack/release/gain/solo/bypass, output limiter, channel link |
| `tube_compressor` | RMS detector, 10 dB soft knee, program-dependent release, gentle `tanh(x/2)·2` saturation |
| `chorus_flanger` | chorus (3 voices, 15–35 ms) or flanger (0.3–5 ms, feedback), transience emphasis, mix |
| `flanger` | initial/final delay sweep, stereo phasing, ±feedback, inverted / special / sinusoidal modes |
| `phaser` | 2–12 swept first-order all-passes, depth (octaves below the upper frequency), feedback, stereo phase difference |
| `analog_delay` | Tape / Tape+Tube / Analog: low-passed, saturating feedback (bounded up to 200 %), trash, stereo spread |
| `multitap_delay` | 4 independent feedback taps, levels, mix |
| `convolution_reverb` | uniformly partitioned overlap-save convolution (256-sample partitions, latency 256) with 7 impulses generated here (noise with frequency-dependent exponential decay + early reflections, unit energy); room size (resampling), LF/HF damping, pre-delay, width |
| `surround_reverb` | early reflections + FDN late field, centre input, low/high cut, width, dry/wet (stereo rendering) |
| `click_remover` | LPC residual outlier detection (k·σ, σ from the median), AR forward/backward interpolation of ≤ 64-sample regions (latency 640) |
| `channel_mixer`, `stereo_expander`, `mute` | 2×2 matrix with inversion; mid/side expand + centre pan; ramped mute |
| `distortion`, `guitar_suite` | waveshaper (soft/hard/tube/foldback, asymmetry, tone); compressor → distortion → amp/cabinet voicing → filter |
| `mastering`, `vocal_enhancer` | EQ → reverb → exciter → widener → loudness maximiser (transparent at defaults); fixed Male/Female/Music voicings |
| `binauralizer` | ±angle virtual speakers through a spherical-head model (Woodworth ITD + Brown–Duda head shadow) |
| `ambisonics_panner` | rotates a ±30° source pair (pan/tilt/roll) and re-pans to stereo; identity at 0/0/0 |
| `loudness_meter` | pass-through BS.1770 meter (`LoudnessMeterFx::meter`) |

`AudioEffect::response_db(freq)` (EQs and filters) and `transfer_db(band, input_db)` (dynamics)
expose the analytic curves for the graphical effect editors. Modulation effects start their LFO
at the beginning of a chain, so a random-access render may differ in LFO phase from continuous
playback.

`effects::premiere_tests` checks every effect: neutral settings are (delayed) identity, output is
deterministic and latency constant (impulse lands at `latency()`), plus a level / frequency-response
test on generated tones, noise or impulses and a realtime-factor floor.

Essential Sound builds on these (Repair: `denoise`, high-pass via `parametric_eq`, `dehum`,
`deesser`, `dereverb`; Clarity: `compressor`, `simple_eq`, `speech_enhance`; Creative: `reverb`,
`stereo_width`).

**Enhance Speech decision.** No machine-learning model is bundled: we found no openly licensed
speech-enhancement model that is also practical to ship here (pure Rust, wasm, asset rules).
DeepFilterNet (MIT/Apache-2.0, with a Rust implementation on `tract`) is the candidate for a future
optional integration behind a trait outside this dependency-free crate, once its weights' licence and
size are vetted as an asset. Until then `speech_enhance` is the DSP chain above.

Measured on synthetic signals (`effects::essential::tests`, 48 kHz):

| Effect | Signal | Result |
|---|---|---|
| `deesser` | 150 Hz harmonic voice + 5–10 kHz noise bursts | sibilance −5.1 dB (8 dB max) / −9.8 dB (16 dB max) during bursts; voice band ±0.01 dB; non-sibilant passages ±0.001 dB |
| `dereverb` (100 %) | harmonic syllables (100 ms / 400 ms gaps) + Polack-model late reverb, RT60 0.8 s | tail-to-burst energy −5.0 → −10.1 dB (5.0 dB better); burst energy −1.7 dB |
| `speech_enhance` | voice + fricative noise syllables over −60 dBFS noise | presence/mud tilt +5.5 dB; noise between syllables −8.7 dB; overall level +1.0 dB |
| Dialogue chain | `denoise` + `parametric_eq` (2 HP bands) + `dehum` + `deesser` + `dereverb` + `compressor` + `simple_eq` + `speech_enhance` + `reverb`, 60 s stereo | 1.36 s = **44× realtime** on one core (release, Apple Silicon; `cargo test --release -p filmcraft-audio-dsp dialogue_chain_realtime -- --ignored --nocapture`) |

## Ducking analysis (`ducking`)

Pure functions for Essential Sound auto-ducking:

- `envelope_db(channels, sample_rate, hop_s, window_s)`: centred-window RMS level per hop (dBFS,
  channel-averaged power, floor −120).
- `activity(env_db, hop_s, threshold_db, min_on_s, bridge_s)`: regions above the threshold; gaps
  shorter than `bridge_s` are merged first, then regions shorter than `min_on_s` are dropped.
- `sensitivity_threshold_db(s)`: Sensitivity 0–10 → −20 − 4·s dBFS.
- `duck_keyframes(regions, clip_start, clip_end, base_db, reduce_db, fade_s)`: (time, dB) volume
  keyframes — ramp down over `fade_s` before each region, hold, ramp up after; overlapping ramps
  merge; clamped to the clip (a ramp cut by an edge gets the ramp's value at the edge); strictly
  increasing times.

## Time stretch that keeps pitch (`stretch`)

Clip ▸ Speed/Duration ▸ **Maintain Audio Pitch**: a clip at 0.25…4× (any speed from 1/16 to 16×)
plays at its new length with its own pitch. `TimeStretch::new(sample_rate, channels, speed)` →
`render(read, o0, out)`: output sample `o` plays input position `o · speed`, `read(start, len)`
supplies planar input (any position; silence outside the media is the caller's).
`stretch(&[&[f32]], rate, speed)` does whole buffers (`round(len / speed)` samples).

- **WSOLA** (waveform-similarity overlap-add): 30 ms periodic-Hann frames at a 15 ms synthesis hop
  (the windows sum to one); each frame's input start is moved by up to ±12.5 ms from where the
  speed puts its centre to the place most like the natural continuation of the previous frame —
  normalised cross-correlation summed over **all channels**, so every channel splices at the same
  samples (inter-channel delays and polarity are kept exactly). Search at ~12 kHz (box-decimated),
  refined at the full rate around the two best coarse lags; a 1 % preference for the nominal
  position breaks the ties of periodic signals. Integer positions: speed 1 is the identity.
- **Renderable from anywhere, deterministic.** The chain of choices restarts every 128 frames
  (1.92 s): a range renders from at most three chains, never from the start of the clip, and the
  result does not depend on how the output is cut into requests, in which order, or whether the
  stretcher is fresh (tested bit-exact). At each restart (a 13-frame junction) the output moves to
  the new chain where the two choose the same input (speech: within a few frames), otherwise in
  twelve equal steps of at most 1/24 period on a sustained tone, so a tone neither clicks nor dips.
- **Bounded:** the memo holds 24 chains of frame positions; the input window is 2¹⁶ samples plus a
  frame's search region per channel (per-frame reads above ~3.3×, where frames do not overlap in
  the input). Speeds outside 1/16…16 or not finite, rates outside 1–768 kHz, 0 or > 64 channels and
  positions beyond ±2⁴⁶ are errors; non-finite input reads as silence.
- **Measured** (`cargo test --release -p filmcraft-audio-dsp stretch -- --include-ignored --nocapture`):
  a 440 Hz tone at 0.25, 0.5, 1.5, 2 and 4× peaks within 1 Hz of 440 (≤ 0.7 Hz), ≥ 99.5 % of its
  energy within 10 Hz, level within 0.3 dB, 10 ms level never more than 0.6 dB down across
  junctions, no sample step over 1.1× a clean tone's; it ends within the tolerance of where the
  speed puts its end. A 0.5 ms inter-channel delay survives every 100 ms of stretched speech.
  Against the phase-vocoder route (the phase-locked `pitch_shifter` by 1/speed, then a sinc rate
  change) on speech-like audio — envelope distance (1/6-octave levels of 43 ms frames against the
  input at the mapped time) and voiced-frame harmonicity (input 0.40):

  | speed | WSOLA distance | PV distance | WSOLA harmonicity | PV harmonicity |
  |---|---|---|---|---|
  | 0.5 | 4.06 dB | 6.98 dB | 0.478 | 0.515 |
  | 0.75 | 3.76 dB | 3.76 dB | 0.452 | 0.481 |
  | 1.5 | 3.99 dB | 6.69 dB | 0.409 | 0.360 |
  | 2 | 4.99 dB | 8.96 dB | 0.431 | 0.282 |

  Speed (60 s of 48 kHz stereo speech-like audio, release, Apple Silicon under a load of ~25):
  0.5× → 120 s out in 0.44 s (270× real time), 0.75× 216×, 1.5× 170–316×, 2× 175–260×, 4×
  153–217×; a seek (fresh stretcher, 1024 samples) 4–9 ms mid-segment, ≤ 30 ms on a junction.

## Prosody (`prosody`)

Speech intonation, measured and reshaped: mono `f32` at 4–768 kHz, at most two hours at once.
Bad rates and settings are errors, non-finite samples are silence, empty input gives an empty
result. Clean-room from the published algorithms: YIN (de Cheveigné & Kawahara, JASA 2002), the
candidate path of Boersma (IFA Proceedings 17, 1993), TD-PSOLA (Moulines & Charpentier, Speech
Communication 9, 1990) and the intensity-peak syllable count of de Jong & Wempe (Behavior
Research Methods 41, 2009).

- **F0 tracker** — `track(x, rate, &PitchSettings { min_hz: 60, max_hz: 500, voicing: 0.4 })` →
  `PitchTrack { hop: 0.01, f0: Vec<Option<f32>>, confidence, level_db }`, frame `i` centred on
  `i · 10 ms`. Rates over 32 kHz are low-passed and decimated by a whole factor to 16–32 kHz.
  Per frame the cumulative-mean-normalised difference `d′(τ)` of a 25 ms window (the
  cross-correlation from one complex FFT, energies from prefix sums); every local minimum under
  0.85 between the shortest and longest period is a candidate (six at most), refined by a
  parabola through `d`. A dynamic-programming path over all frames picks unvoiced or a
  candidate: a candidate scores `1 − d′` less 0.04 per octave under the frame's highest one
  (multiples of the period are as periodic as the period), unvoiced `1 − voicing` plus up to 1
  near silence (30–40 dB under the loud frames, or under −70 dBFS); a voiced step costs 0.6 per
  octave, a voicing change 0.2. Voiced runs under 30 ms are dropped. `confidence` is the frame's
  periodicity, `1 − d′` at its best candidate.
  Tested on glottal-like voices (harmonics at −6 dB/octave through two formants) and sines at
  16, 24, 44.1 and 48 kHz with noise 20–34 dB down: median error under 1 % (65, 110, 140, 230,
  420 Hz; 700 Hz with a raised range), every frame within 4 % on glides 90 → 260 → 120 Hz (no
  octave jumps), over 95 % of voiced frames found and of silent frames left unvoiced. A minute of
  48 kHz audio tracks in about 0.13 s (an optimised build).
- **Reshaping** — `psola(x, rate, &track, &ratios)` multiplies each frame's pitch by its ratio
  (kept within ½…2), the duration unchanged: analysis marks one period apart at the peaks of a
  1 ms moving average (±20 % around where the period puts them), synthesis marks spaced by the
  analysis period at the nearest mark over the ratio, each taking that mark's grain (a raised
  cosine back to the previous mark and on to the next, so at ratio 1 the grains add back to the
  input exactly); the local energy (30 ms) is matched to the original's and each voiced run is
  cross-faded with it over 10 ms at both ends. Unvoiced audio, runs under 40 ms and runs left at
  ratio 1 are the input's samples. `reshape(x, rate, &Shape, &PitchSettings)` builds the ratios
  (`shape_ratios`) from:

  | `Shape` field | range | default | does |
  |---|---|---|---|
  | `range` | 0.5–2.0 | 1.0 | each voiced frame's deviation from the utterance's median F0 scaled in the log domain (on a 50 ms median-smoothed contour) |
  | `shift` | −12…12 st | 0 | the whole contour moved |
  | `question_rise` | −12…12 st | 0 | added over the last half second of voicing (gaps to 150 ms bridged) before `question_end` (seconds; default the end), a smooth ramp; the rest of a voiced run past the end holds it |
  | `emphasis_db` | 0–6 dB | 0 | stressed syllables lifted: F0 above the two syllables either side (full lift at 3 st) or louder than the median syllable (full at 6 dB), a raised-cosine gain from the dip before to the dip after, never past −0.2 dBFS |

  A neutral shape returns the input unchanged. Tested: the re-tracked contour's 10–90 % range is
  the requested multiple within 5 % (measured within 0.5 % for 0.6, 1.3, 1.5, 1.6 and 2.0 at 16,
  24 and 48 kHz), the median within 3 %, the length exact, energy within 1 dB (measured
  ±0.04 dB), no sample step over twice the input's largest, a shift within 0.25 st, a question's
  end within 0.6 st of the rise asked, unvoiced hiss either side bit-identical. On the owner's
  IndicF5 narrations range 1.6 took the 10–90 % range from 3.0 to 4.4 st and from 3.8 to 5.9 st.
- **Expressiveness** — `expressiveness(x, rate, &PitchSettings)` (or `expressiveness_of` with a
  track): `f0_median_hz`, `f0_range_st` (10th–90th percentile), `f0_movement_st_s` (mean absolute
  slope between voiced frames 50 ms apart), `final_contour_st` (the last 100 ms of voicing
  against the 400 ms before: + for a rising end), `energy_variation_db` (standard deviation of the
  syllable peaks' speech-band levels), `syllable_rate` (peaks per second of speech),
  `voiced_fraction`, `speech_seconds`. `syllables(x, rate, &track)`: peaks of the 300 Hz–3 kHz
  level (30 ms around each frame) in speech, voiced or within 20 ms of it, at most 25 dB under
  the loud frames, standing 2 dB over the dips towards their neighbours, at least 80 ms apart.
  Tested on syllable trains (5 a second, levels alternating 6 dB, F0 falling 6 st in 2 s):
  rate within 0.8/s, energy variation within 1.2 dB of 3 dB, movement within 1.2 of 3 st/s,
  range within 0.8 of 4.8 st.

## Audio synchronisation (`sync`)

`find_offset(a, b, sample_rate, &SyncOptions)` returns the lag `L` with `b[n] ≈ g·a[n + L]`, a
sub-sample estimate, and two trust figures (`confidence` = peak / RMS of the correlation,
`distinct` = peak / next peak > 50 ms away; `reliable()` = ≥ 8 and ≥ 1.5).

- Coarse: DC removal + normalisation, Blackman-windowed-sinc decimation to ≤ 8 kHz (further for
  long recordings, so the transform stays ≤ 4 M points), GCC-PHAT-β (`R = A·B*/|A·B*|^0.75`,
  40 Hz – 0.4·rate band) with both signals packed into one complex FFT.
- Refine: the same correlation at the full rate on the loudest common window (≤ 2¹⁷ samples), ±2
  coarse steps around the coarse peak, parabolic interpolation.
- Tested within one sample on speech-like multi-microphone signals (gains −20…+10 dB, SNR down to
  3 dB, coloured microphones, fractional delays: worst 0.4 samples integer / 0.11 samples
  sub-sample), a 0 dB SNR room with a strong echo, inverted polarity, unrelated signals
  (unreliable) and `max_lag`. Two 10-minute recordings: 2.4 s (release, one core).

## Known limits

- Effects assume ≤ the channel count they were built with; extra channels pass through
  (channel-generic gain effects process all of them).
- DeNoise adapts continuously; a perfectly stationary tone is (by design) treated as noise after
  ~1.5 s. There is no explicit "learn noise print" mode yet.
- Filter type / choice changes switch immediately (continuous parameters are smoothed).
- Time stretch splices at whole samples: a sustained pure tone comes out up to ~3 cents sharp or
  flat (0.7 Hz at 440 Hz) and drifts in phase across each junction; speech and ambience are not
  affected audibly. Transients are repeated (slowing) or skipped (speeding) as in any time-domain
  stretch; speeds outside 1/16…16 are refused (the render plays them varispeed).
