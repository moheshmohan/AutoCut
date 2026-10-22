# filmcraft-speech

Speech-to-text for FilmCraft's Text panel ▸ Transcript (L2). User-facing behaviour, commands and
model handling: [docs/transcripts.md](../../docs/transcripts.md).

- `Transcriber` trait (mono 16 kHz in, word-timed `filmcraft_project::Transcript` out) and
  `FixedTranscriber` for tests and agents.
- `models`: Whisper model catalogue (pinned revisions, SHA-256, sizes, licences) and, with feature
  `download`, the verified downloader (ureq + rustls + RustCrypto). Weights are never bundled.
- `whisper` (feature `whisper`): Whisper inference on candle (CPU, pure Rust).
- `mel`, `vad`, `diarize`: log-mel front end, voice activity (Silero, `silero`, else energy) and
  word-bound tightening, speaker labels (`speakers`: ECAPA embeddings, `ecapa` and `fbank`, feature
  `whisper`; MFCC clustering as the fallback).
- `understand`: segments, pauses, fillers (English and Malayalam), hesitations and retakes of a
  talking clip (`media.understandSpeech`).

Both features are off by default and are never enabled for wasm (`cargo xtask wasm` checks the
crate without them).

- `Transcriber::hear`: the words of a clip as plain text in one language, no word times — what
  the strict check of narration asks. Whisper decodes each 30 s window greedily without
  timestamps (`<|notimestamps|>`, the model's suppress tokens, the repetition guard), as models
  fine-tuned without timestamps read best; `Whisper::hear_tokens` gives the tokens.
- `Whisper::load_on(dir, id, gpu)`: on the GPU (Metal on macOS; `FILMCRAFT_SPEECH_CPU` keeps the
  CPU). `load_gpu_first` is what transcription uses: the GPU with the CPU as the fallback (a model
  the GPU cannot load is loaded on the CPU; a transcription that fails or panics there runs again
  on the CPU, which keeps the model). Whisper-base on an M3 Max: 58 s of narration in 4.4 s on the
  GPU against 40.5 s on the CPU (load average about 20), the same words and word times
  ([docs/gpu.md](../../docs/gpu.md)). A PyTorch checkpoint (`pytorch_model.bin`) is read with candle's restricted pickle
  reader when there is no `model.safetensors` (measurement only: the catalogue pins safetensors).
- Large-v3's 128 mel bands go through the same front end (the filterbank is computed for any
  band count); checked against Hugging Face transformers: the same text on two clips.

## Malayalam: the recogniser that judges narration (measured 2026-10-20)

The strict check of Malayalam narration (`crates/tts/README.md`) needs a recogniser that writes
Malayalam in Malayalam script, English loanwords and spoken numbers included ("ഫൈവ് പോയിന്റ്
ത്രീ", "ട്വന്റി ട്വന്റി വൺ"), and hears a garbled syllable as garble. Whisper-base, the app's
default, writes Malayalam in Tamil script or translates it. Candidates that run as Whisper in this
crate, openly licensed (no NC/ND), measured with `crates/tts/examples/ml_lab.rs` on an M3 Max GPU
(character error rate after the check's normalisation, `filmcraft_tts::heard`):

- **the reference clip**: the author's 9.3 s Malayalam reference, his own words typed by hand
  (code-mixed: "ഫൈവ് പോയിന്റ് ത്രീ ഫ്ലാഷ്, … ഫ്ലാഷ് ബേസിക്കലി ഒരു ലോവർ മോഡൽ ആണ്");
- **FLEURS**: the first 30 utterances of FLEURS' Malayalam test split (Google, CC BY 4.0;
  read Wikipedia sentences, never committed);
- **IndicF5 readings**: 40 readings of ten narration lines by the author's cloned voice (20 with
  the old reference words, garbled starts and all; 20 after the fix), scored against the
  script.

| Candidate (licence, size) | FLEURS CER | Reference clip CER | Readings, intact words: CER, passing | Readings, old words: first starts flagged (12 leaked a word) |
|---|---|---|---|---|
| **`whisper-medium-ml`**: Thennal D K's Whisper medium fine-tune (Apache-2.0, 3.06 GB) | 0.046 | **0.136** | **0.086, 19 of 20** | 10 |
| Whisper large-v3-turbo fine-tuned on IndicVoices (BettySara, Apache-2.0, 3.2 GB) | 0.059 | 0.192 | 0.108, 19 of 20 | 13¹ |
| IndicWhisper Malayalam (AI4Bharat Vistaar, MIT, 1.5 GB) | **0.025** | 0.512 (spoken numbers as digits) | 0.145, 13 of 20 | 15¹ |
| Whisper small fine-tune (VRCLC, Apache-2.0, 0.97 GB) | 0.083 | 0.520 | 0.160, 13 of 20 | 18¹ |
| Whisper large-v3 LoRA on FLEURS (rontroy, Apache-2.0, 3.1 GB) | 0.096 | 0.480 (numbers as digits, repeats) | | |
| Whisper large-v3-turbo (OpenAI, MIT, 1.6 GB) | 0.918 (Devanagari and Tamil script) | 0.968 | | |
| Whisper large-v3 (OpenAI, MIT, 3.1 GB) | 1.075 (translates into English) | 1.000 | | |
| Whisper base, the app's default (OpenAI, MIT) | (Tamil script) | 1.000 ("5.3, 5.3, basically a lower model") | | |

¹ More flagged than leaked: the recogniser's own errors on clean starts count too (a false
retake each). *Passing*: the strict check's default policy.
The FLEURS numbers are characters weighted; the fine-tunes trained on FLEURS' train split (not
its test). Lines with a dash were not run on the readings (the clip numbers rule them out). The
oracle runs below found the large-v3 port right: transformers writes the same mixed scripts.

**Chosen: `whisper-medium-ml`** — Thennal D K's fine-tune of Whisper medium (Apache-2.0; trained
on Common Voice, FLEURS, IMaSC, MSC, ULCA and Indic TTS). It is the only candidate that hears
the code-mixed reference clip (IndicWhisper, best on FLEURS, writes spoken English numbers as
digits and loses the loanwords; large-v3 and large-v3-turbo answer in Devanagari or Tamil
script). The catalogue pins the safetensors conversion Hugging Face's bot made of the release
(pull request 2, commit `c65085c5`, SHA-256 `72042c80…`, 3 055 544 304 bytes) and Whisper
medium's own `tokenizer.json` (the fine-tune ships the same vocabulary without it, checked equal
id for id). On the GPU it hears about 30 tokens a second (Malayalam takes about two tokens a
character: a 9 s clip is 273 tokens, about 9 s).

**Checked against the reference.** `tools/whisper_oracle.py` (Hugging Face transformers 4.46 on
the CPU, a venv under the lane's target directory, never shipped) decodes the same way;
`malayalam_hearing_matches_the_reference_oracle` (ignored; `WHISPER_ORACLE`) compares: the
reference clip and five FLEURS clips, 1 577 tokens, identical token for token on the CPU.

## Voice activity: Silero VAD (2026-10-22)

`silero::Silero` runs Silero VAD v5's 16 kHz network in plain Rust (no model framework: it is
1.2 MB and 0.7 M multiply-adds per 32 ms) from the authors' own `silero_vad_16k.safetensors`
(MIT, `models::SILERO_VAD`, pinned to commit `fb7d7c7f`). Checked against the authors' recipe
for those weights (`tinygrad_model.py`) written with PyTorch's layers
(`tools/speech_oracle.py silero`, `tests/speech_models.rs`): Silero's own 60 s test clip, 1 875
chunks, largest difference 1.6e-6, the same speech decision on every chunk; 0.24 s for the minute
on one M3 Max core. (Their ONNX export is a slightly different graph: it agrees on 99.3 % of the
decisions.) `vad::speech_segments` uses it when downloaded, else the energy detector.

## Speaker labels: ECAPA-TDNN (2026-10-22)

`ecapa::Ecapa` is SpeechBrain's ECAPA-TDNN (VoxCeleb, Apache-2.0, `models::ECAPA_MODEL`) on
candle, its Fbank front end in `fbank`. Against SpeechBrain's own classes on two clips (Silero's
test clip, SpeechBrain's example): the same frame counts, Fbank within 1e-2 dB, embeddings at
cosine 1.000000 on the CPU and on Metal; a batch embeds each clip as alone.

`speakers` labels turns: 3 s windows every 1.5 s over the voice activity segments,
average-linkage clustering on cosine similarity with an automatic count (merge while the closest
clusters are more alike than 0.33; under 4 % of the speech folds into its nearest; one pass to the
nearest centre), majority-vote smoothing over 0.75 s, turns under 0.6 s join a neighbour.
Measured with `examples/diarize_lab.rs` on mixes dealt from known speakers (Kokoro voices made by
`tts.preview`, and the owner's own 27 s voice sample), 0.25–0.9 s between turns, diarization error
rate with no collar (0.25 s collar in brackets):

| Mix | Length | ECAPA (default) | MFCC fallback |
|---|---|---|---|
| A: owner + Kokoro michael | 78 s, 19 turns | 2 speakers, **2.4 %** (1.6 %) | 2 speakers, 2.9 % |
| B: michael + fenrir + bella | 135 s, 37 turns | 3 speakers, **2.2 %** (1.6 %) | 2 speakers, 34.9 % |
| C: owner + puck + sarah | 112 s, 27 turns | 3 speakers, **3.0 %** (2.4 %) | 3 speakers, 3.2 % |
| D: michael + fenrir + puck (three male Kokoro voices) | 130 s, 34 turns | 2 speakers, 32.2 % | 1 speaker, 65.2 % |
| E: michael alone | 49 s | 1 speaker, 2.2 % | 1, 2.7 % |
| F: the owner alone | 30 s | 1 speaker, 6.5 % | 1, 6.2 % |

What remains on A–C, E and F is where speech starts and stops (Silero against the turns' edges),
not confusion. The threshold was swept from 0.25 to 0.6 with 1.5, 3 and 4.5 s windows: the owner's
real recording is the least alike to itself (1.5 s windows: median cosine 0.41) and splits into
several speakers above 0.35, while two of Kokoro's male voices are as alike as 0.73 and merge
below 0.5; 0.33 on 3 s windows splits no real speaker and separates every voice but those two.
Different voices (the owner against Kokoro's, and Kokoro voices of the other sex) measure
0.06–0.13 median, 0.38 at most.

## Whisper large-v3-turbo (2026-10-22)

`whisper-large-v3-turbo` loads in the same code: 32 encoder layers, 4 decoder layers, 128 mel
bands, 51 866 tokens (its own `tokenizer.json`), and the alignment heads of its
`generation_config.json` give the word times. It is the English option of
`media.understandSpeech` and the recogniser of the English narration check.

## Malayalam narration: a second opinion and a voice gate (2026-10-22)

The strict check now asks IndicWhisper (`indicwhisper-ml`, MIT) when Whisper medium ml rejects a
reading, and with `agreement: both` (the default) a line is read again only when both disagree with
the script. IndicWhisper writes spoken numbers as digits, so they are spelled the way the script
spells them before comparing. The voice gate embeds each reading with ECAPA and compares it with
the cloned voice's own sample. Re-checked on the 82 checked readings (83 lines) of the owner's
narration cache (`engine/examples/narration_recheck.rs`, on a copy of the cache):

| | Lines flagged |
|---|---|
| Whisper medium ml (the cache's verdict) | 1 |
| IndicWhisper, digits as written | 27 (spoken numbers written as digits) |
| IndicWhisper, digits spelled as the script | 13 |
| both (what `agreement: both` reads again) | **1** |
| either | 13 |
| voice gate at 0.30 (similarity min 0.37, median 0.58, max 0.65) | **0** |

The second opinion costs nothing on a line the first passes (it is asked only after a rejection).
Different voices measure 0.06–0.13 median against each other (above), so a reading that came
out in someone else's voice falls under the gate's 0.30 while every reading of the owner's clone
stays above it.

## English narration: the strict check (2026-10-22)

English cloned narration (VoxCPM) is heard back by Whisper large-v3-turbo (small when turbo is
not downloaded) and read again with the next seed until clear, as Malayalam is. What was heard
goes through the same text normaliser as the script (a recogniser writes "2023" and "80 %"; the
voice read "twenty twenty-three" and "eighty percent"). Ten lines of the owner's clone: nine clear
on the first reading (CER 0–0.09, the recogniser's own slips such as "busk" for "box"), one read
again after "an robot … working" for "a robot … walking" (CER 0.17); voice similarity 0.64–0.77.
`heard::Policy::english()`: CER at most 0.15, sentence starts at most 0.5, three readings.

## Understanding raw talking clips (2026-10-22)

`media.understandSpeech` transcribes **speech only, chunk by chunk** (`Transcriber::
transcribe_chunks`, `vad::chunks`): voice activity segments joined into chunks of at most 12 s
for Malayalam (the decoder holds about 12 s of it: two tokens a character) and 28 s for English,
each decoded without timestamps as `hear` does and its words timed by cross-attention alignment.
The first live run used 30 s timestamp windows instead: the Malayalam fine-tune, trained without
timestamps, wrote a few words a window (one "word" lasted 22 s); chunked, the same clips gave
456 and 633 words at a mean confidence of 0.95.

Live, the owner's two Malayalam talks (IMG_3337, 242 s; IMG_3338, 319 s; a phone filming a laptop
screen while he talks), `filmcraft-cli rough-cut … --language ml` on the M3 Max with the GPU shared
with other jobs: 13.7 min for both; one speaker each (ECAPA); shot hint `voice` (speech, no face:
right); fillers 13 and 17 (ആ, എന്ന് പറഞ്ഞാൽ and hesitations); no retakes (none was said); the rough
cut keeps 541.4 of 561.4 s (23 pauses and 3 fillers cut, 28 pieces, every cut between words).

An English clip with known retakes (Kokoro, 34.5 s: a false start, a line said twice, a 2.5 s
pause), large-v3-turbo: both retakes found and the complete takes kept; the cut is 16.7 s and
heard back word for word ("…goes up to five times. The screen is really bright, even outside in
the sun. …"), the last words the recogniser had missed in the clip ("thanks for watching") kept
because voice activity runs on from the last word.

## References

Implemented from the published description of the model: A. Radford et al., "Robust Speech
Recognition via Large-Scale Weak Supervision" (OpenAI, 2022), the model cards and configuration
files published with the weights (`config.json`, `generation_config.json`, `tokenizer.json`), and
the candle tensor library's public API. Weights: OpenAI Whisper, MIT licence.
