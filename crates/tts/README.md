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

- `kokoro` (feature `kokoro`): **Kokoro-82M neural voices** in pure Rust on candle (Metal on
  macOS, the CPU elsewhere and as the fallback; `FILMCRAFT_KOKORO_CPU` keeps the CPU): ALBERT
  text encoder, StyleTTS 2 prosody predictor (durations, F0, energy) and decoder, iSTFTNet
  vocoder with a harmonic-plus-noise source. Weights are read from the official PyTorch
  checkpoint with candle's restricted pickle reader (no code from the file runs); voice packs from
  their zip archive. `config.json` is checked against the architecture this code implements.
  A line that fails, panics or yields samples that are not finite on the GPU is spoken again on
  the CPU, which keeps the voice from then on.
- `neural` (feature `kokoro`): the natural voices as `Voice`s: script → `filmcraft-tts-text`
  (normalizer, CMUdict + letter-to-sound, ≤ 400-symbol chunks, pauses) → Kokoro per chunk → exact
  silence for pauses, 120 ms between sentences → vocal pitch with `filmcraft-audio-dsp`'s pitch
  shifter (Kokoro has no pitch input). Pace is Kokoro's speed.
- `catalog`: the natural-voice package (Kokoro-82M, 9 US English voices graded C+ or better on the
  model card, CMUdict; 336 MB), every file pinned to a revision and SHA-256. Downloaded by the
  engine (`tts.downloadVoices`) after the user confirms; never bundled or committed.

- `openvoice` (feature `kokoro`): the **OpenVoice V2 tone-colour converter** in pure Rust on
  candle (Metal on macOS with the CPU as the fallback, as Kokoro; `FILMCRAFT_OPENVOICE_CPU`) — a VITS synthesizer without a text encoder: a reference encoder (six 3×3/2 conv2d →
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
- `f5` (feature `indicf5`): **IndicF5** (AI4Bharat, MIT) in pure Rust on candle — F5-TTS
  fine-tuned on eleven Indian languages, Malayalam among them, with the Vocos vocoder
  (charactr, MIT; IndicF5 ships its weights unchanged — checked bit-equal to
  `charactr/vocos-mel-24khz`). Non-autoregressive: the whole utterance's mel is generated at
  once by flow matching, infilling after a reference clip whose mel and transcript go in front,
  so the voice continues the clip's speaker. Ported from the MIT reference code shipped with the
  release (`f5_tts/model/{cfm,modules}.py`, `backbones/dit.py`, `infer/utils_infer.py`),
  x-transformers' rotary embedding and the `vocos` package.
  - `config`: the architecture read from the tensor shapes (the release's `config.json` only
    names the Hugging Face wrapper) and checked: 100 mels, 1024-point FFT, 64-wide heads,
    caps on every size.
  - `vocab`: `vocab.txt` read as Python does (universal newlines, last character of each line
    dropped) and the reference's tokenization (`;` → `,`, curly quotes, a space before an
    ASCII word, one token per character; unknown characters reported, not silently spaces).
  - `mel`: torchaudio's mel (periodic Hann, reflection-padded centre STFT, HTK filterbank,
    `log(max(·, 1e-5))`) and `torch.istft(center=True)`, both in f64.
  - `dit`: the DiT — text embedding (sinusoid + four ConvNeXt-V2 blocks with GRN), input
    embedding (noisy mel, masked reference mel, text; grouped convolutional position embedding
    with Mish), sinusoidal time MLP, 22 AdaLN-Zero blocks whose rotary embedding turns only
    the first 64 channels (head 0) as the reference applies it, final AdaLN. Guidance's two
    passes run as one batch; the text embeddings are made once per utterance.
  - `vocos`: ConvNeXt backbone and ISTFT head.
  - `model`: references (RMS raised to 0.1 as the reference does, transcript ending `".  "`),
    the duration, and the sampler: seeded noise, 32 Euler steps on the sway schedule
    (`s = −1`), guidance 2, reference frames put back.
  - Duration: the reference's frames plus as many again as the text is longer than the
    transcript. The reference counts UTF-8 bytes; that is the character ratio when both are in
    one script, but an English sample speaking Malayalam (three bytes a letter) would then
    speak three times too slowly (17.9 s instead of 6.5 s for one test line), so characters
    are counted.
  - Devices: Metal on macOS, else the CPU (`FILMCRAFT_F5_CPU` forces it). candle's fused Metal
    attention reads an input that does not start its buffer from the wrong place (it passes
    the element offset as a byte offset), so q, k and v are each a tensor of their own (a
    regression test compares the GPU with the CPU).

- **Malayalam** (`voice_for(id, "ml-IN", …)`): a natural clone speaks Malayalam through IndicF5. The workflow (clone, reference, transcript, numbers, trim): [docs/malayalam-narration.md](../../docs/malayalam-narration.md).
  - `malayalam` (no feature): the text front end. Spelling made canonical (two-part vowel
    signs composed, old ZWJ chillus made atomic, zero-width joiners dropped, Malayalam digits);
    numbers in one of two styles (`Numbers`): **English** (the narration default) says years,
    money, decimals and numbers from 100 up the English way in Malayalam script, as Malayali
    tech creators speak — "2021" → "ട്വന്റി ട്വന്റി വൺ", "2008" → "ടു തൗസന്റ് എയ്റ്റ്", "1969" →
    "നൈന്റീൻ സിക്സ്റ്റി നൈൻ", "$1,600" → "സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളർ", "2,500" → "ടു തൗസന്റ്
    ഫൈവ് ഹണ്ട്രഡ്", "170,000" → "വൺ ലാഖ് സെവന്റി തൗസന്റ്" (lakh and crore) — and small counts
    in Malayalam words ("2 ദിവസം" → "രണ്ട് ദിവസം"); a four-digit number from 1100 to 2099
    without a group comma is a year; **Malayalam** says every number in Malayalam words, Indian
    system ("2013" → "രണ്ടായിരത്തി പതിമൂന്ന്"). The long Malayalam words garbled the start of
    lines in IndicF5's reading; the user prefers the English style. Suffixes join by sandhi in
    both ("2013-ൽ" → "…പതിമൂന്നിൽ" / "ട്വന്റി തേർട്ടീനിൽ", "2021-ൽ" → "ട്വന്റി ട്വന്റി
    വണ്ണിൽ": a chillu takes its consonant back, doubled after a short one-syllable word; a
    vowel takes the glide യ or വ: "2020-ലും" → "ട്വന്റി ട്വന്റിയിലും"), ranges ("… മുതൽ …
    വരെ"), percent, money (a currency symbol, or a currency word after the amount), decimals;
    digits joined to capitals are a code ("4K" → "ഫോർ കെ"). `spell_numbers` rewrites only a
    script's numbers (pause markers kept): the engine spells a Malayalam narration's numbers in
    its `numbers` style before the voice reads it, and keys the narration cache on that text,
    so scripts without digits keep their cached readings. Checked against the v5 showcase
    script, whose numbers were converted to English by hand: with the digits put back, the
    front end says every one as the hand conversion does. Latin acronyms read with the letters' Malayalam names ("AI" → "എഐ"),
    digits in codes with English names ("G1" → "ജി വൺ"); other Latin words kept and reported;
    emoji and unknown symbols dropped and reported (`report`); sentences grouped into utterances
    that fit the model, a long one split at commas, else between words; pause markers exact.
    `detect` says whether a script is written in Malayalam; `syllables` counts its syllables.
  - `clone`: the *Malayalam reference* (`<name>.indicf5.json` and the clip
    `<name>.indicf5.wav`, 24 kHz): the best 10–15 s between pauses of the best recording there
    is — one given for it (`tts.voiceReference` with project items), else the voice sample's
    full-band copy (`<name>.full.wav`, which natural clones now keep at the recording's rate),
    else the voice sample itself — and its words, heard once by the speech model (English) or
    given by the user (and correctable). Tied to the sample's SHA-256 (a voice made again needs
    a new one); the English profile and its sample are never touched. **Trimmed to a sentence
    end** (`trim_cross_reference`, `tts.voiceReference {"trimToSentence": true}`): IndicF5
    continues the clip, so a clip that stops mid-sentence garbles the first words of every
    line. The cut goes at the quietest 20 ms frame (levels smoothed over three frames; nearer
    frames preferred, 0.5 dB per 100 ms) within 8 % of the speech (at least 0.8 s) of where the
    transcript's words put the latest sentence end (syllables up to it over all, across the
    speech's span), keeping at least `SENTENCE_REFERENCE_MIN_SECONDS` (8 s; the automatic
    pick keeps 10–15 s); the words are cut to that sentence, or taken as given (words that
    begin the transcript and end a sentence pick that sentence). `start` / `end` cut at given
    times (with the kept words). The cut is faded over 12 ms; the clip before the first trim is
    kept as `<name>.indicf5.untrimmed.wav` / `.json`. The author's hand-made cut (14.6 s ending
    mid-sentence → 9.3 s ending at "…ഇല്ല.") is what the user preferred. The recording is English:
    IndicF5 is prompted across languages. `clone:<name>` in Malayalam reads each utterance after
    it (pace is IndicF5's own speed, pitch the pitch shifter; takes reseed), with natural pauses
    between utterances (`styles::gap_ms`, below; 250 ms each before 2026-10-28,
    `Reading::natural_pauses` off), under the GPU lock. The duration is held within 4.6–6 syllables a second
    (Malayalam read aloud runs about 5–6): the reference's characters-per-second estimate is
    far off across languages and when the speech model adds words that were never said. The
    reference's words reach IndicF5 as written (Malayalam through the Malayalam front end,
    English through the English one; until 2026-10-20 Malayalam went through the English one,
    which dropped every virama: the cause of the garbled sentence starts, below). How it reads
    is a `clone::Reading` (the pause after the reference, packing, sampler, lead-in; the
    default is the measured best), every utterance with its own seed
    (`MalayalamVoice::seed(take, utterance, attempt)`).
  - `styles` (no feature) and `clone`'s style references: **voice styles** (F5-TTS's
    multi-style generation: one reference audio and text per style). IndicF5 reads in the
    emotion and prosody of the clip it continues, so a natural clone keeps a reference per style
    beside its main (plain) one: `<name>.indicf5.<style>.wav` / `.json` for `conversational`,
    `warm`, `excited`, `amazed`, `curious`, `calm`, `dramatic` (`clone::set_style`,
    `tts.voiceStyles`; 5–16 s of the speaker's own speech, cut at the last sentence end its words
    imply, faded, tied to the voice sample like the main reference; setting or removing one keeps
    the voice's references as a version, `clone::versions` / `rollback`, and a promoted sample
    keeps them). A line's delivery (`Params::delivery`) reads after its style's reference, else
    the nearest the voice has (`styles::resolve`: amazed → excited → conversational, excited →
    amazed → conversational, curious → conversational, conversational ↔ warm, calm → warm,
    dramatic → calm → warm), else the main one. Each style has a **sampler preset**
    (`styles::preset`: temperature — the scale of the flow-matching sampler's initial noise,
    Matcha-TTS's `temperature`, 1 leaves the noise untouched bit for bit — emphasis, the
    classifier-free guidance `f5::Settings::cfg`, speed on top of the pace, and expressiveness —
    each utterance's intonation reshaped after synthesis with `filmcraft_audio_dsp::prosody::reshape`
    (TD-PSOLA, length unchanged: pitch range = expressiveness about the median, a `?` ending
    raised 2 st, stressed syllables lifted 3 dB per unit over 1; at 1 the audio is untouched, and
    audio it cannot reshape is kept as read); `Params`' `temperature` (0.3–1.5), `emphasis` (1–4),
    `speed` (0.7–1.4) and `expressiveness` (0.5–2) override it; plain has none
    (the reading is exactly what it was). **Inline tags** (`script::parse_marked`,
    `malayalam::parts`): `[excited] … [calm] …` switch the style for what follows (an utterance
    never spans a tag or a line break), `[plain]` / `[neutral]` go back, `[breath]` is a
    breath-length pause of the clip's room tone; tags are never read aloud, the strict check
    hears each part against its own words, and a retake keeps its part's reference and sampler.
    **Natural pauses** (`styles::gap_ms`): the pause after an utterance follows how it ends — a
    full stop 320 ms, `?` 400, `!` 340, `…` 560, a comma 180, a split between words 110, a line
    break 480, a blank line 760 — scaled by its style (excited ×0.8 … calm ×1.25, dramatic
    ×1.35), ±12 % seeded by the take and the utterance (deterministic); none after a pause
    marker, which is exact. The narration cache keys a styled reading on the style references it
    reads after (`cache::reading_identity_for`, `clone::malayalam_reading_identity`) and on the
    delivery and sampler settings it was given, so a plain reading of one utterance keeps the key
    it always had and a styled reading never reuses a plain one; a reading of more than one
    utterance keys on the pauses' version (`styles::VERSION`).
  - `heard` (no feature): **the strict check**. `Voice::synthesize_checked` reads under it:
    the Malayalam voice hears every utterance with a `heard::Hearer` (the engine's: the
    Malayalam speech model, `filmcraft_speech::models::MALAYALAM_MODEL`), scores it
    (`heard::score`: the character error rate after normalising — NFC's two-part vowels,
    chillus as consonant + virama, ZWJ/ZWNJ, punctuation and spaces dropped — and the first six
    syllables of every sentence on their own, speech heard in front of a sentence counting
    double), scans its signal (`heard::signal`: empty, invalid, clipped, a silence over 0.9 s
    mid-reading, a burst at the start) and reads a failing utterance again
    (`heard::read_until_clear`): a new seed, then (from the third reading) twice the sampling
    steps (`Reading::for_attempt`), at most four readings, the best kept, every try reported with
    its seed. Long readings are heard in pieces of at most 12 s cut at pauses (Whisper's decoder
    holds about 440 tokens; Malayalam takes about two a character).
  - `catalog::INDICF5`: `model.safetensors` (1 402 789 408 bytes) and `vocab.txt` (11 283),
    MIT. The official repository is gated (sign-in and a click-through); FilmCraft downloads
    anonymously, so the files come from a public mirror pinned to a commit whose bytes have the
    official SHA-256.
  - `cache::voice_identity_for`: English is exactly `voice_identity` (English entries keep
    their keys); Malayalam keys on the profile, the sample, the reference clip (SHA-256) and
    words, the IndicF5 pins and the front end's version (`MALAYALAM_VERSION` 3 since the
    sentence-start fix: old readings are read again). The engine adds the strict check's
    recogniser and version to a checked reading's key, and keeps the check's report beside the
    entry (`DiskCache::put_note`, `<key>.json`, removed with it).

- `clone` also makes **natural clones** (profile version 2, engine `voxcpm`): a *voice sample*
  is kept next to the profile (`<name>.wav`: the best continuous 10–25 s of speech between
  pauses, long pauses shortened, −20 LUFS, at the encoder's rate) with its transcript. VoxCPM2
  uses it as an isolated reference (no transcript needed); VoxCPM1.5 continues a shorter one
  (5–10 s), which needs its exact transcript (Whisper or given) and gets 18 dB of noise
  reduction. `clone:<name>` dispatches on the profile's engine; version 1 profiles load
  unchanged; the narration cache keys a natural clone on its sample's bytes and settings.
- **Delivery** (`Params::style`): how a line is spoken, as a few plain words ("warm, friendly,
  smiling, natural storytelling"). The engine maps the narration's delivery (`plain`,
  `conversational`, `warm`, `excited`, `amazed`, `curious`, `calm`, `dramatic`;
  `filmcraft_project::Delivery`) to these words. VoxCPM2 takes them as its spoken-style control —
  the description in parentheses directly in front of each utterance
  (`voxcpm::text::style_prefix`, brackets stripped, at most 120 characters) — and only after a
  reference-only prompt, as the reference implementation does (a continuation prompt dictates
  the delivery itself). Other voices ignore it. A styled reading has its own cache key; the plain
  ones keep theirs. Measured on a 9 s intro line with the default natural clone, the pitch range
  (10th–90th percentile) goes from 3.0 semitones plain to 5.4 *excited* (the speaker's own
  recording: 4.4), and Whisper reads the same words: the style is followed, not spoken.
- `wav`: WAV reading (16/24/32-bit PCM, float, any channels; hostile files are errors) and
  writing for voice samples.

- `voice_lab` (feature `kokoro`): **better references for a natural clone**, found in the
  speaker's own clips and auditioned (the engine's `voice.*` commands and `filmcraft-cli voice`;
  how to record a seed for a re-clone: [docs/voice-recording.md](../../docs/voice-recording.md)).
  - `signal`: speech activity (10 ms frames, an adaptive threshold), phrases between pauses of at
    least 0.25 s, and a stretch's quality: signal to noise against the pauses around it,
    clipping (flat tops), a decay-time estimate after word ends (reverberation), clarity (the
    level contrast of syllables and dips), tonal background in the pauses (narrow spectral peaks:
    music or a hum), bandwidth; a light clean-up (70 Hz high-pass, optional noise reduction,
    trimmed edges) and references joined with a pause.
  - `scan` / `rank`: 6–12 s chunks checked against the speaker (OpenVoice tone colour, cosine ≥
    0.85), the speaker's runs heard (Whisper's words and language), whole-phrase windows scored
    per engine (VoxCPM 10–30 s, English preferred; IndicF5 6–12 s, Malayalam preferred) on
    similarity, quality, whole sentences, length and language; at most two per recording.
    `held_out`: genuine speech of the speaker that is no candidate and no current reference.
  - `audition`: the fixed test lines (English and Malayalam, hard starts and loanwords), a
    reading scored on similarity to the held-out speech, CER and sentence-start CER
    ([`heard::score`]) and the signal scan; the ranking.
  - `styles`: the style finder. IndicF5 candidates ending at a sentence end
    (`Candidate::ends_sentence`, from `sentence_edges`) measured for prosody
    (`filmcraft_audio_dsp::prosody::expressiveness`: pitch level, range and movement, energy
    variation, syllable rate, final contour), placed on the speaker's own scale (`norms`: robust
    median and spread over his candidates, pulled towards `PRIORS` — his measured speech — with
    the weight of four candidates), scored per delivery (`scores`: calm, plain, conversational,
    warm, excited, amazed, curious; the rules in the module docs) and shared out by a softmax
    (`assess`: the style and its confidence); `pick` lists the best per style (share × the
    reference score), never two overlapping stretches of a clip, none under a 0.2 share. The
    engine's `voice.findStyles` (CLI `voice styles`) hears the picks with both Malayalam models;
    `voice.prosody` (CLI `voice prosody`) measures and reshapes any recording.
  - `clone::promote` / `versions` / `rollback`: a reference replaced with the old one kept
    (`<name>.versions/r<revision>/`), consent carried over, the profile's `revision` counted (and
    in the prompt cache's key); the narration cache already keys on the reference's bytes.
    `clone::audition_voice` / `audition_malayalam`: a voice speaking from any clip, no profile.

- `cache`: the **narration cache** — synthesized speech kept on disk so a script is read by a
  voice once, across sessions and rebuilds (a cloned voice runs at about 2.6× real time: a
  25-line show is minutes per rebuild otherwise). The engine keeps it in
  `<data dir>/narration-cache` (native frontends turn it on; `Session::use_narration_cache`) and
  looks there after its in-memory cache, before synthesizing, for `tts.create`, `tts.render`,
  `tts.edit` and `tts.preview`.
  - Key: SHA-256 of the cache format version, the voice's identity (`voice_identity`: its id and
    the pinned model files it speaks with; for a cloned voice also its profile's id, version,
    creation time, base voice and a hash of its tone colours), the pitch and pace (exact bits), the
    delivery's style words when there are any, and the script. A re-made cloned voice, a new model revision or a format bump never hits old
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

On Metal (M3 Max, 2026-10-21, load average about 20 from parallel lanes): a 36.7 s script in
5.7 s against 43.6 s on the CPU (7.6×), the same words heard by Whisper (WER 4.1 % both), the
waveforms 48.5 dB apart (float noise; `gpu_speaks_like_the_cpu`). OpenVoice converts 12 s in
1.6 s against 12.0 s (7.4×), 96 dB apart (`live_openvoice_gpu_matches_the_cpu`). The narration
cache keys do not name the device: a take sounds the same from either.

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

## Malayalam (IndicF5): checked and measured (2026-10-19)

**Against the reference.** The release's own PyTorch code run on the CPU as an oracle
(`oracle/f5_oracle.py`-style, a venv under the lane's target directory, never shipped) with the
port's noise, a 7.6 s clip of the author's voice sample and a Malayalam line
(`live_matches_the_reference_oracle`):

| Compared | CPU | Metal |
|---|---|---|
| Tokens of transcript + line (149) | equal | equal |
| State after Euler step 1 / 2 (values up to 4.6) | 2.4e-7 / 6.7e-7 | 2.4e-7 / 6.3e-7 |
| Final mel after 32 steps, max / mean (values up to 17) | 5.9e-4 / 4e-6 | 9.0e-3 / 3e-5 |
| Vocos on the same mel (amplitude up to 1.2) | 2.1e-5 | 3.0e-5 |
| Reference mel | 9.9e-5 in magnitude | |

The log mel of the reference differs by up to 0.2 only in the bands above 8 kHz, which a 16 kHz
sample leaves near the 1e-5 floor: there the reference's single-precision FFT rounding is a
large share of the value (this port's FFT is f64). Fed the reference's own mel, the network
matches as above.

**Quality.** The author's natural clone (`clone:mohesh-natural`, made from a phone video in
Indian English), the two test lines through `tts.preview` (Apple M3 Max; the GPU shared with
other sessions and a local LLM, so times are high). Similarity is the cosine against held-out
speech of the speaker (the recording minus the voice sample): ECAPA is SpeechBrain's
`spkrec-ecapa-voxceleb` (Apache-2.0, an external measuring instrument), *colour* the OpenVoice
converter's reference encoder (`examples/voice_similarity.rs`). *Heard* is the share of the
line's consonants Whisper-base (`ml`) hears, compared across scripts as the showcase does —
Whisper-base writes Malayalam in Tamil script; a garbled reading scores near 0.

| IndicF5 reference | ECAPA (line 1 / 2) | Colour | Heard | Rate (syll./s) |
|---|---|---|---|---|
| **14.4 s of the original recording, 24 kHz, Whisper's words** (`tts.voiceReference` with the video) | **0.77 / 0.67** | **0.86 / 0.80** | 0.68 / 0.61 | 6.3 / 6.9 |
| 10.8 s of VoxCPM2 speaking English, exact words | 0.62 / 0.55 | 0.76 / 0.69 | 0.76 / 0.58 | 6.3 / 6.6 |
| 15.6 s of the 16 kHz voice sample | 0.45 / 0.44 | 0.74 / 0.67 | 0.72 / 0.67 | 5.9 / 5.6 |
| 7.6 s of the 16 kHz voice sample | 0.49 / 0.38 | | | 4.2 / 4.0 |
| VoxCPM2 speaking the lines in English (for comparison) | 0.55 / 0.59 | 0.83 / 0.73 | | |
| The speaker: another stretch of the recording | 0.92 | 0.97 | | |

- The full-band recording is what carries the voice: IndicF5 speaking Malayalam from it is
  closer to the speaker than VoxCPM2 speaking English (ECAPA 0.72 against 0.57 on average). A
  16 kHz sample makes IndicF5 speak band-limited and less like the speaker (0.45), hence the
  full-band copy natural clones now keep, and `tts.voiceReference` with a recording for voices
  made before. A longer clip helps (7 s → 15 s: 0.48 → 0.62 on the 24 kHz variant).
- The timbre carried over well enough that the OpenVoice tone-colour converter was not added as
  a second stage.
- Whisper-base heard the lines as Malayalam (in Tamil script), e.g. line 2 as
  "…நடந்தத்தைத்திடிக்சல் … ஒரு மன்ஷினைப் படுத்தனை தோன்றி" (*naṭanta … oru manuṣyane … tōnni*);
  line 1's years came back as digits.
- 16 Euler steps instead of 32 measured the same (ECAPA 0.74 / 0.65 against 0.76 / 0.65) in
  half the time; 32, the reference's, is kept.
- Speed: one step is two passes over reference + line (about 2 000 frames for a 14 s clip and
  a 5 s line): 0.4 s at 1 300 frames and 0.8 s at 2 000 on an idle M3 Max GPU, so about
  25 s a line (real-time factor 4–6); the reference clip's length sets most of it. A
  rebuild reads from the narration cache.

## Natural spoken Malayalam: the script lint and the fact lock (2026-10-20)

A good voice still sounds robotic reading written Malayalam, so the script is checked too
([docs/malayalam-style.md](../../docs/malayalam-style.md)):

- [`ml_script`](src/ml_script.rs): `lint` finds sentences over 14 spoken words (a number said the
  English way or a spelled acronym is one word), formal words with a spoken form (a curated table,
  `FORMAL`: written → spoken, a hint where it depends on context, a fix only when the replacement
  takes the same suffixes), numbers from 100 up in Malayalam words, Latin letters, and (a hint)
  sentences with no verb ending; `fix` makes the safe replacements and keeps the rest of the text
  byte for byte. `numbers` reads every number a script says: digits, Malayalam words (an atom
  parser with the joining, glided and elided forms scripts write, "നൂറ്റമ്പതിലധികം" 150,
  "ഇരുപത്തിയേഴായിരം" 27 000, never "ആയിരുന്നു"), English words in Malayalam script ("ട്വന്റി
  തേർട്ടീനിൽ" 2013, "എം സീറോ ടു" 2). Tested by reading back every number `malayalam::number`,
  `english_number` and `english_year` write.
- [`ml_facts`](src/ml_facts.rs): the facts of an English line (numbers by value, including "two
  and a half thousand" and halves of years; names; acronyms; units after a number; months) must
  be in its Malayalam, and no number added. Names match by sound: consonant-class skeletons of both
  sides, the English one found within up to three Malayalam words with one difference in four;
  a plan's respellings give the sound.
- Measured on a real 52-part showcase script against its English plan: the robotic first draft
  got 54 warnings and the natural rewrite 7 (no formal word); the fact lock found exactly the three
  facts the rewrite dropped and no false alarm in either.

## Malayalam: sentence starts and the strict check (measured 2026-10-20)

The author, a native speaker, heard gibberish in the v5 Malayalam narration, mostly where a
sentence begins. Measured with the strict check as the oracle (`examples/ml_lab.rs`; the
recogniser is `whisper-medium-ml`, chosen in `crates/speech/README.md`), on ten test lines (the
three sample lines and seven lines of the v5 robot script, 21 sentences), in the author's cloned
voice and reference (`clone:mohesh-natural`, the 9.3 s clip ending "…ഇല്ല."), two takes each
(their own seeds), Apple M3 Max GPU. *First starts*: the first six syllables of each line (where
a generation begins after the reference); *later starts*: the other sentences of a line (inside
the same generation); a start is garbled above 0.3.

**The cause: the reference's words did not match the clip.** IndicF5 is given the clip's mel
and its words, then the new text; it aligns the words to the clip and goes on. The voice passed
the words through the English text front end, which drops every virama (U+0D4D): "ഫൈവ് പോയിന്റ്
ത്രീ … ഇല്ല." reached the model as "ഫൈവ പോയിന റ ത രീ … ഇല ല." — 138 characters of which many
were not what the clip says. The model then said the reference's last word again before the
line: 12 of the 20 old readings start with "ഇല്ല", "ലാ" or "ലല്ല" (heard: "ഇല്ല ഇതാണ് ബോസ്റ്റൺ
ഡൈനാമിക്സിന്റെ സ്പോർട്ട്…", "ലാ ട്വഞ്ച് ട്വഞ്ച് ഈ വൺ ഒട്ടോബർ…"); not one of the readings with the
words intact does. Sentences later in a line were never affected (0 of 22 garbled). The fix: a
transcript written in Malayalam goes through the Malayalam front end (canonical spelling,
numbers spelled), one in English through the English one (regression test
`the_reference_words_reach_indicf5_intact`; the cache's `MALAYALAM_VERSION` 3, so old readings
are read again).

| Reading (10 lines × 2 takes unless said) | Mean CER | First starts: mean, garbled | Later starts: mean, garbled | Lines passing the check | Synthesis |
|---|---|---|---|---|---|
| before (English front end on the words) | 0.121 | 0.413, 10 of 20 | 0.057, 0 of 22 | 10 of 20 | 372 s |
| **words intact** (the default) | **0.086** | **0.051, 0 of 20** | 0.110, 1 of 22 | **19 of 20** | 350 s |
| words intact, 300 ms of the clip's room tone after it | 0.083 | 0.021, 0 of 20 | 0.118, 2 of 22 | 18 of 20 | 342 s |
| words intact, 300 ms of silence after it (50 ms is the reference implementation's) | 0.092 | 0.052, 0 of 20 | 0.101, 2 of 22 | 18 of 20 | 356 s |
| words intact, every sentence its own utterance | 0.096 | 0.078, 2 of 20 | 0.092, 2 of 22 | 16 of 20 | 561 s |
| words intact, 64 sampling steps (32 is the reference's) | 0.081 | 0.051, 0 of 20 | 0.119, 1 of 22 | 19 of 20 | 737 s |
| words intact, a lead-in ("ശരി.") read first and cut away (11 readings) | 0.129 | 0.477, 11 of 11 | 0.136, 1 of 11 | 0 of 11 | 223 s |
| words intact, 16 sampling steps (1 take) | 0.093 | 0.071, 0 of 10 | 0.146, 1 of 11 | 9 of 10 | 83 s |
| words intact, guidance 1.5 (2 is the reference's; 1 take) | 0.110 | 0.041, 0 of 10 | 0.179, 2 of 11 | 8 of 10 | 194 s |
| words intact, guidance 2.5 (1 take) | 0.085 | 0.042, 0 of 10 | 0.096, 1 of 11 | 9 of 10 | 190 s |
| words intact, sway −0.5 (−1 is the reference's; 1 take) | 0.104 | 0.052, 0 of 10 | 0.128, 1 of 11 | 9 of 10 | 173 s |
(*Garbled*: a start over 0.3; *passing*: the default policy, CER ≤ 0.25 and every start ≤ 0.3 —
a reading that fails it is read again. The few garbled later starts with the words intact are a
word the model added ("പക്ഷേ ഇവിടെ…") or loanwords the recogniser spells otherwise ("അപ്ലോഡർ"
for "ആപ്പിൽ ഓർഡർ"). The signal scan flagged none of the 100 readings of the first five rows.)

- **The pause after the reference** (50 ms of silence, the reference implementation's; 300 ms
  of silence; 300 ms of the clip's room tone — the author's phone recording has a −30 dB floor
  and the clip ends on a word): no measurable difference once the words match. The default stays
  50 ms; `Reading::tail_ms` / `room_tone` keep the choice.
- **One utterance per sentence** instead of the whole line (packed while it fits): no better,
  and 60 % slower (every sentence pays for the reference again). Lines stay packed
  (`Reading::per_sentence`).
- **The sampler** (sampling steps 16 / 32 / 64, guidance 1.5 / 2 / 2.5, sway −0.5 / −1): once
  the words match, none moves the starts beyond the noise of 10–20 readings; 64 steps take
  twice as long for the same, 16 steps lose a little, guidance 1.5 adds errors later in the
  line. The reference's 32 steps, guidance 2 and sway −1 stay; the strict check's third and
  fourth readings use 64 steps as another path through the sampler.
- **A lead-in** ("ശരി.", read first and cut away at its pause): harmful. IndicF5 runs it into
  the sentence without a pause, so the cut leaves it heard ("ശരി ഇതാണ്…", 11 of 11 first starts
  garbled — which also shows the check catching a word in front of a sentence). Not used
  (`Reading::lead_in`, measurement only).
- **Compared with the reference implementation** (IndicF5's `infer/utils_infer.py`): the
  duration (ours counts characters and holds 4.6–6 syllables a second), the transcript's
  `". "` ending, the reference raised to RMS 0.1 and the 50 ms pause match; it cross-fades
  batches over 150 ms where we trim silence and leave 250 ms between utterances (a join, not a
  start); it never rewrites the reference's words — the difference that mattered.

**Confirmed on fresh seeds** (takes 2–4 of the same ten lines, 30 readings each, never seen
while choosing):

| Reading | Mean CER | First starts: mean, garbled | Later starts: mean, garbled | Lines passing the check | Readings that start with words never asked for |
|---|---|---|---|---|---|
| before | 0.190 | 0.608, 22 of 30 | 0.109, 3 of 33 | 6 of 30 | 22 of 30 ("ഇല്ല", "പബിൽ"… — the end of the reference: "…കേപ്പബിലിറ്റീസ് ഇല്ല.") |
| **the default now** | **0.088** | **0.040, 0 of 30** | 0.097, 2 of 33 | **28 of 30** | **0 of 30** |

The two readings that fail the check with the defaults are the recogniser's: it wrote
"ട്വന്റി തേർട്ടീനിൽ" as "2013-ല്" and heard "ആപ്പിൽ ഓർഡർ" as "ആപ്പി ഓടർ"; the check reads them
again (a false retake costs one reading). The three sample lines in the app's own path
(`tts.preview`, strict check on, the author's voice) were all heard clearly on their first
reading (CER 0.08, 0.07, 0.07).

**The strict check.** Every utterance is heard, scored and read again until clear
([`heard`]): the character error rate after normalising (≤ 0.25), the first six syllables of
every sentence (≤ 0.3; a character heard in front of a sentence counts double — the leaked
"ഇല്ല" is the garble), and a signal scan (empty, invalid samples, clipping, a silence over
0.9 s between words, a burst more than 9 dB over the speech in the first 200 ms; advisory,
each fails the reading). Thresholds from the readings above: with the words intact no first
start and 1–2 of 22 later starts score over 0.3, the worst CER is 0.23; the old readings' leaked
words score 0.4–1. A failing utterance is read again with its own seed
(`MalayalamVoice::seed(take, utterance, attempt)`), from the third reading with 64 sampling steps;
at most four readings, the best kept
(start garble weighs double), every try reported with its seed, so one sentence can be read
again alone and reproduced. These are `Policy::default()`; the engine's `checkPolicy` (`tts.create`
and the other narration commands: up to `MAX_TAKES` = 8 readings, thresholds 0–1) asks for a
stricter one per narration, and caches its readings under a key that names it.

## Better references: the author's library (measured 2026-10-21)

`filmcraft-cli voice refs --voice clone:mohesh-natural <Google Photos cache> ~/Movies/AutoCut/*.fcproj`
(Apple M3 Max): 765 files, 729 clips with sound read (569 minutes of audio) in 16 minutes; 499
of FilmCraft's own sounds and one copy of a render (a review build cached back from Google
Photos) skipped. 2711 speech chunks checked against the voice's reference: the author's own
speech scored 0.85–0.95 (tone-colour cosine), everyone else at most 0.80, the cloned voice
itself (in that render, before renders were skipped) 0.82–0.86. Only three clips hold the
author's speech: `IMG_3337.MOV` (the clone's source), `IMG_3338.MOV` (the same session, 239 s,
Malayalam) and an English clip (126 s, the cleanest at 29 dB signal to noise); the rest of the
library is expo footage of other people and machines.

`voice audition --takes 3`: the current references and the best three candidates per engine
(and the variants) read the test lines three times each (12 readings), scored against 119 s of
held-out speech from all three clips:

| Engine | Reference | Similarity | CER | Start CER |
|---|---|---|---|---|
| VoxCPM2 | **current**: `IMG_3337.MOV` 56.1–80.7 s (24.5 s) | 0.878 | 0.039 | 0.106 |
| VoxCPM2 | `IMG_3337.MOV` 49.1–63.7 s | **0.897** | **0.036** | **0.098** |
| VoxCPM2 | it and `IMG_3338.MOV` 41.9–56.1 s joined | 0.886 | 0.059 | 0.094 |
| VoxCPM2 | `IMG_3338.MOV` 41.9–56.1 s | 0.865 | 0.042 | 0.066 |
| VoxCPM2 | the English clip, 0.3–27.7 s | 0.864 | 0.095 | 0.121 |
| VoxCPM2 | the same, cleaned (70 Hz high-pass, 6 dB noise reduction) | 0.839 | 0.050 | 0.146 |
| IndicF5 | **current**: `IMG_3337.MOV` 57.4–66.7 s, hand-typed words | 0.909 | 0.073 | 0.066 |
| IndicF5 | `IMG_3337.MOV` 56.1–63.7 s, the speech model's words | **0.918** | **0.055** | **0.040** |
| IndicF5 | `IMG_3338.MOV` 41.9–50.2 s | 0.913 | 0.055 | 0.093 |
| IndicF5 | `IMG_3337.MOV` 102.0–110.1 s | 0.905 | 0.094 | 0.087 |
| IndicF5 | the best, cleaned | 0.881 | 0.063 | 0.070 |

From one take to the next the same reference moves by about 0.01 in similarity and 0.05–0.1 in
start CER, so only the VoxCPM2 similarity gain (+0.02, in every take) stands clearly above the
noise; the best cuts are other stretches of the clone's own session. Two references joined did
not beat the better of them; a light clean-up made both engines' readings less like the speaker
(it is not used). The English clip, the cleanest recording, scores lower partly because the
held-out speech is mostly that session's microphone and room: similarity follows the recording
as well as the voice. So the library holds no clearly better recording of the speaker: a
purpose-made one is the way to a better clone ([docs/voice-recording.md](../../docs/voice-recording.md)).

## Delivery styles: the author's clips and his narrations (measured 2026-10-28)

**How flat is the synthesis?** `filmcraft-cli voice prosody` (`filmcraft_audio_dsp::prosody`,
24 kHz) on five IndicF5 narrations of the 3D Printing Saga (`Narration 1, 2, 3, 10, 20.wav`) and
five stretches of the author's own Malayalam (`IMG_3337.MOV` 56.1–63.7 s, `IMG_3338.MOV`
41.9–50.2 s, the current reference's region of `20250729-124548.mov` 132.4–141.2 s, the voice-overs
`IMG_4184.MOV` 26.4–34.7 s and `IMG_4188.MOV` 0.1–10.1 s), medians:

| | pitch | range (10–90 %) | movement | energy variation | syllable rate | voiced |
|---|---|---|---|---|---|---|
| narrations | 91 Hz | **3.0 st** (2.5–3.8) | 12.4 st/s | 4.7 dB | 4.4 /s | 0.60 |
| his speech | 92 Hz | **4.0 st** (3.1–4.4) | 9.9 st/s | 4.1 dB | 4.4 /s | 0.66 |

The narrations' intonation is about a quarter narrower than his speech — and than the very clip
they continue (4.2 st) — at the same pace, while their pitch wobbles more from one 50 ms to the
next (movement) and their syllables vary more in level: flat phrasing over a busier surface. Over
all 29 of his candidates (run 2 below) the range centre is 4.05 st.

**Reshaped** (`--range 1.3,1.6 --hear`, `target/agent-prosody/samples/`): `Narration 1.wav`
3.0 → 3.5 (1.3) and 4.4 st (1.6), `Narration 3.wav` 3.8 → 4.7 and 5.9 st; the words heard again
by both Malayalam models differ from what they heard in the original by a character error rate
of 0.02–0.07 (the same model hearing two readings of one line differs by about as much), so the
words survive; the length is unchanged.

**His clips, sorted by delivery** (`voice styles` over the Google Photos cache and the AutoCut
projects' media: 1046 files, 1012 clips with sound, 744 minutes, 24 minutes on an M3 Max with the
hearing). Against the voice's reference alone, his 3D-printing voice-overs and the current
reference's own source clip scored only 0.72–0.84 (another room and microphone), so five
stretches of them were given as `--speaker-ref` and the check kept at 0.80: his Malayalam clips
then scored 0.84–0.94 at best; the other clips over 0.80 were English (English recordings of
his, and `Every Robot I_ve Met - Pro.mp4`, a copy of an AutoCut render — the cloned voice — that
the render check does not know), and English is not taken. His Malayalam turned up in eight clips — `IMG_3337.MOV` and `IMG_3338.MOV` (the clone's
session), `20250729-124548.mov` (the current reference's), the voice-overs `IMG_4184.MOV`,
`IMG_4185.MOV` and `IMG_4188.MOV`, a 3D-printing walk-through `copy_FAA0F66C-….mov` (273 s) and
`IMG_5479.MOV` (15 s: its words do not say whose voice it is — listen); `IMG_4189.MOV` (0.78 at best) and `20250729-122424.mov` (no whole phrase ending at
a sentence end) gave none. 34 candidates (whole phrases, 6–12 s, ending at a sentence end; 29
more did not end at one) on his scale — range 3.9 ± 1.0 st, movement 9.4 ± 2.0 st/s, energy
variation 4.2 ± 1.2 dB, 4.8 ± 0.6 syllables/s — the best per style:

| style | clip | seconds | confidence | the two models agree | why |
|---|---|---|---|---|---|
| excited | `IMG_3338.MOV` | 149.9–158.0 | 0.95 | 0.80 | range 4.6 st, energy variation 5.8 dB, 5.5 syllables/s |
| amazed | `20250729-124548.mov` | 44.0–51.2 | 0.95 | 0.84 | range 5.1 st at 99 Hz, 3.4 syllables/s |
| curious | `IMG_4184.MOV` | 34.9–42.4 | 0.97 | 0.63 | ends 8.9 st up (a continuation rise as much as a question: listen) |
| conversational | `copy_FAA0F66C-….mov` | 165.1–174.0 | 0.86 | 0.63 | everything near his middle |
| plain | `IMG_4185.MOV` | 26.7–33.9 | 0.81 | 0.80 | range 2.2 st at an ordinary 5 syllables/s |
| calm | `IMG_3338.MOV` | 41.9–50.2 | 0.64 | 0.72 | range 3.1 st, movement 7.1 st/s, 4.0 syllables/s |
| warm | none good | — | 0.34 at best | — | his clips are tech talk and walk-throughs: nothing lower, slower and melodic |

The two Malayalam models agree on 0.6–0.8 of the characters of most picks (both stumble on his
English loanwords); where they agree under 0.5 the pick says `check` — one calm pick's Malayalam
Whisper heard a single word for 12 s, so its transcript is IndicWhisper's.

## Limitations

- The built-in voices are **robotic** and nearly unintelligible to a recogniser (above); they are
  the no-download fallback and the test voice.
- English (United States) only. Words outside CMUdict go through letter-to-sound rules and are
  often mispronounced (see `crates/tts-text/README.md`); homographs take the first pronunciation.
- About 2× faster than real time on the CPU (a one-minute narration takes about 30 s), about 6×
  on the GPU. The UI runs synthesis as a background job.
- Kokoro's model card states the weights are Apache-2.0 but the repository has no LICENSE file.
- Vocal pitch shifts the voice's base pitch (and its formants slightly); pace scales every duration.
- A *timbre-only* cloned voice copies **timbre**, not accent or rhythm: those are still Kokoro's
  (US English). Natural (VoxCPM) clones copy accent and rhythm too, but are large downloads
  (2–5 GB) and want a GPU: on the CPU VoxCPM2 is about 5× slower than real time.
- Malayalam: the reference clip's words come from the speech model, and Whisper-base hears
  Indian English on a phone video imperfectly (it added a sentence that was never said to the
  author's clip). IndicF5 aligns its reading to those words, so wrong ones garble the first
  syllables of a line (measured above); `tts.voiceReference` hears the clip with the Malayalam
  speech model and warns when its words do not match (`transcriptCheck`), and `transcript`
  corrects them. The strict check is only as good as its recogniser: `whisper-medium-ml` spells
  some loanwords its own way (a few false retakes) and could miss a garble that happens to
  sound like the line; the author's ear is the final judge. IndicF5 is about 4–6× slower than real time on the M3 Max GPU (a CPU is far
  slower: 22× in the oracle run). Latin words left in a Malayalam script are read as best the
  model can (reported in `notes`). Only Malayalam is wired up of IndicF5's eleven languages.
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

IndicF5 (feature `indicf5`): the vocabulary (Python's line rules) and the reference's
tokenization; the mel (silence on the floor, a tone in its band), the f64 FFT against a DFT and
the inverse STFT inverting the STFT; depthwise convolution, Mish, the position table and the
first-head rotary against loops (CPU and GPU); a tiny random release speaks deterministically,
reseeds, follows speed, refuses texts past 4096 frames, stops when asked; the GPU equals the
CPU; hostile references (short, long, silent, NaN, no words) and broken releases (vocabulary
mismatch, foreign weights, garbage) are errors; the Malayalam reference is made once per sample
(full-band source preferred, words corrected, stale when the sample changes, damaged clips
refused) and keys Malayalam only; the Malayalam voice speaks scripts with pauses, takes, pace
and pitch; the speaking rate is held; a reference trimmed to its last sentence end on
synthetic speech (syllables with −20 dB dips, sentences with −30 dB pauses: the cut within 80 ms
of the pause that leaves 8 s, the words cut to match, the untrimmed clip kept once), given words
and times, and hostile trims (NaN, negative, past the end, too short, no sentence end, silence,
no reference) refused with the files unchanged. The front end: numbers in both styles (0 to
crores, years 1100–2099, money, decimals, codes), suffix sandhi after Malayalam and English
words, the hand-converted v5 script, `spell_numbers` keeping pauses, ranges, symbols, acronyms,
canonical spelling, syllables, utterance splitting and hostile scripts. With a release in `FILMCRAFT_F5_DIR`: the oracle comparison (`F5_ORACLE`) and per-part
timings (`live_f5_parts`).

The voice lab (feature `kokoro`, synthetic speech, fake models): phrases cut at pauses; quality
telling clean from noisy, clipped, reverberant and music-backed speech; windows within each
engine's bounds that never cut a word; sentence edges by words and by pauses; a scan of two
speakers finding only the one asked for (and hearing nothing when nobody matches); ranking spread
over recordings; held-out speech away from every reference; audition order; hostile audio
(empty, NaN, infinite, full scale). Styles: each archetype (z-scores a spread or two out) gets
its style, a slight rise is no question, the scale is the speaker's own (one speaker's wide is
another's narrow; few candidates lean on the priors; the spread never collapses), picks ranked
and never overlapping, empty styles left empty, hostile features harmless, and synthetic lively
speech (wide, fast, punchy) told from flat speech end to end through the prosody measures. Versions: a promoted sample and Malayalam clip kept and
rolled back (the narration cache's identity changing and coming back), consent and the
Malayalam reference carried over, refused promotions changing nothing.

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
