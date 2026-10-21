# filmcraft-speech

Speech-to-text for FilmCraft's Text panel ▸ Transcript (L2). User-facing behaviour, commands and
model handling: [docs/transcripts.md](../../docs/transcripts.md).

- `Transcriber` trait (mono 16 kHz in, word-timed `filmcraft_project::Transcript` out) and
  `FixedTranscriber` for tests and agents.
- `models`: Whisper model catalogue (pinned revisions, SHA-256, sizes, licences) and, with feature
  `download`, the verified downloader (ureq + rustls + RustCrypto). Weights are never bundled.
- `whisper` (feature `whisper`): Whisper inference on candle (CPU, pure Rust).
- `mel`, `vad`, `diarize`: log-mel front end, word-bound tightening, MFCC-clustering speaker labels.

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

## References

Implemented from the published description of the model: A. Radford et al., "Robust Speech
Recognition via Large-Scale Weak Supervision" (OpenAI, 2022), the model cards and configuration
files published with the weights (`config.json`, `generation_config.json`, `tokenizer.json`), and
the candle tensor library's public API. Weights: OpenAI Whisper, MIT licence.
