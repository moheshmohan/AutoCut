# Transcripts and text-based editing

FilmCraft's Text panel ▸ **Transcript** tab shows the dialogue of the open sequence as text. Select
words to mark In/Out, then extract or lift them; remove filler words and long pauses in one step;
turn the transcript into captions. Every action is an engine command (`transcript.*`), so the CLI,
the control channel and MCP agents can do the same.

## Model

- A **transcript** belongs to a media item (`Project::transcripts`, saved in the `.fcproj` since
  schema v9). It lists **words** with media-time bounds (`Tick`s), an optional speaker index and a
  confidence, plus the speaker names and the language. Because the times are media time, the
  transcript stays valid however the clip is trimmed, moved, sped up or reused.
- The **sequence transcript** is derived, never stored (`filmcraft_edit::transcript::sequence_words`):
  audio tracks are read top first; a word is heard through the first enabled clip whose range
  covers the word's midpoint (duplicates of the same dialogue on lower tracks read once); disabled,
  reversed and frame-hold clips contribute nothing.
- Speaker names come from the clip transcripts, so renaming "Speaker 1" in every transcript renames
  it across the sequence.

## Commands

| Command | What it does |
|---|---|
| `transcript.generate` | Transcribe media items (`items`, else the Project selection, else the media of the sequence's audio clips). Params: `model` (default `whisper-base`), `language` (`auto` = detect), `diarize`, `maxSpeakers`. One undo step. |
| `transcript.set` | Store a transcript you bring (JSON: `language`, `speakers`, `words` with `text`/`start`/`end`/`speaker`); it is sorted and made well formed. |
| `transcript.delete` | Remove transcripts. |
| `transcript.inspect` | The sequence transcript: words (index, text, sequence times, speaker, clip), paragraphs, speakers, the word at the playhead. |
| `transcript.search` | Word-index ranges matching a phrase (case and punctuation ignored; the last word may be a prefix). |
| `transcript.select` | Mark In/Out around words `from..=to` (frame-snapped outward) and move the playhead there. |
| `transcript.extract` / `transcript.lift` | Extract (ripple) or lift the words' frames on the targeted tracks. |
| `transcript.renameSpeaker` | Rename a speaker by name (every transcript) or by index in one `item`. |
| `transcript.removeFillers` | Ripple-delete filler words (`fillers`, default um/uh/erm/…; phrases such as "you know" allowed). |
| `transcript.removePauses` | Ripple-delete pauses longer than `minSeconds`, keeping `keepSeconds` of air on both sides. |
| `transcript.createCaptions` | Lay the words out as captions on a new caption track (`maxChars`, `lines`, `minSeconds`, `maxSeconds`, `gapFrames`). |
| `transcript.models` / `transcript.downloadModel` | List the speech models (size, licence, installed; recognisers and the support models `silero-vad` and `ecapa-voxceleb`) / download one. |
| `transcript.removeRetakes` | Ripple-delete every take that a retake drops (needs `media.understandSpeech`). |

## Understanding raw talking clips

| Command | What it does |
|---|---|
| `media.understandSpeech` | A background job (`wait: true` blocks; `jobs.list` reports it under `speech`) that, per media item, finds where speech is (Silero VAD, else the energy detector), the words (Whisper: the Malayalam model for `language: "ml"`, large-v3-turbo for `"en"`, when downloaded, else `model` or the preferences' model), who speaks (ECAPA embeddings, else MFCC), sentences with speaker, text, words, confidence and fluency, pauses, filler words and hesitations, retakes with the take to keep, and a talking-head / voice / b-roll hint (a face seen while someone speaks, when the face detector is downloaded). Speech is transcribed chunk by chunk (voice activity, at most 12 s for Malayalam and 28 s for English), each chunk decoded without timestamps and its words timed by alignment. Stored with the project beside the transcript (`Project::speech`; one undo step). Params: `items`, `language`, `model`, `diarize`, `maxSpeakers`, `faces`, `minPause`, `fillers`. Holds the GPU lock while it hears. |
| `media.inspectSpeech` | The stored analyses (times in seconds and ticks). |
| `sequence.roughCut` | A new sequence of the items' kept speech, back to back: pauses longer than `minPause` (0.6 s) that the voice activity detector agrees are silent, fillers and dropped takes cut, `keepPause` (0.12 s) of silence kept at each cut, every cut between words, whole frames, V1 and A1 linked; speech running on from a word with no word written (one the recogniser missed) stays. One undo step. |
| `sequence.speakerTrack` | Colour each clip of the active sequence by the speaker who talks most in it. |
| `autocut.roughCut` | The AutoCut job: import `paths` (or take `items`), understand, rough-cut, and `export` when given. `filmcraft-cli rough-cut <files…> [--language ml] [--export out.mp4]` and the MCP tools `speech_understand` / `rough_cut` drive it. |

**Fillers.** English: um, uh, umm, uhm, erm, er, ah, hmm, mm, mhm. Malayalam: the hesitation sounds
ആ, ആആ, ആഹ്, ഉം, ഹ്മ്മ് and the discourse fillers എന്ന് പറഞ്ഞാൽ ("that is to say") and അതായത്
("that is"), and their Manglish spellings (aa, ennu paranjaal, athayath…). Sounds that are also
words (ആ is also "that", "ah", "er", "mm") count only when a pause of at least 0.25 s sets them
apart. Whisper writes few hesitations, so speech the voice activity detector heard between words
(0.15–1.2 s) with no word written is reported as a **hesitation**; `removeHesitations` cuts them.

**Retakes.** A sentence that repeats one of the three before it (character similarity of the texts
without fillers ≥ 0.6, lengths within 60 %), or a short one whose words are how the next starts
(prefix similarity ≥ 0.75), within 30 s, by the same speaker. A chain of takes is one retake; the
take kept is the most fluent (the words' mean confidence, less 0.08 a filler, 0.12 a restart, 0.25
for clipping; an abandoned start never wins; the later take wins a near tie).

## Speech recognition

Recognition goes through the `Transcriber` trait (`crates/speech`). The built-in recogniser is
OpenAI's Whisper, run in pure Rust on [candle](https://github.com/huggingface/candle) (Metal on
macOS, else the CPU), with timestamp decoding, language detection, word times from cross-attention
alignment and word confidences from the decoder (see the `filmcraft_speech::whisper` module docs).
Speakers are told apart by ECAPA-TDNN speaker embeddings over the voice activity segments
(`filmcraft_speech::speakers`) when the speaker model is downloaded, else by clustering per-chunk
MFCC statistics (`filmcraft_speech::diarize`, no model). Measurements:
[`crates/speech/README.md`](../crates/speech/README.md).

Both are **optional features**, off by default and never built for the web:

- `whisper` (on `filmcraft-speech`, `filmcraft-engine`, and the `filmcraft` / `filmcraft-cli`
  apps, where it also enables downloads): candle inference.
- `download` (`speech-download` on the engine): HTTPS downloads with rustls + RustCrypto and the
  operating system's certificate verifier.

Without `whisper`, and with no recogniser installed, `transcript.generate` and Transcribe Sequence
are disabled, with "speech-to-text is not available in this build" as the reason (`describe`,
`command_list {"enabled_only": true}` and the menus show it); with Automatically transcribe clips
on, `file.import` reports the same reason as a `transcription: …` entry in its `errors`.
Without `speech-download`, `transcript.downloadModel` is disabled the same way. Transcripts can
still be imported with `transcript.set` and edited with every other command. Hosts and tests can
install any recogniser in `Session::transcriber`, which enables transcription in any build.

### Models

Weights are **never** bundled or committed. They are downloaded on request into
`<data dir>/models/<id>/` (see `filmcraft_engine::autosave::default_data_dir`), each file pinned to a
revision of OpenAI's Hugging Face repositories and checked against its SHA-256:

| Id | Languages | Licence |
|---|---|---|
| `whisper-tiny` | multilingual | MIT (OpenAI); HF conversion Apache-2.0 |
| `whisper-base` (default) | multilingual | MIT (OpenAI); HF conversion Apache-2.0 |
| `whisper-small` | multilingual | MIT (OpenAI); HF conversion Apache-2.0 |
| `whisper-medium-ml` | Malayalam (fine-tune of Whisper medium by Thennal D K; 3.06 GB) | Apache-2.0; base model MIT (OpenAI) |
| `whisper-large-v3-turbo` | multilingual, the English option (809 M parameters, 1.6 GB) | MIT (OpenAI) |
| `indicwhisper-ml` | Malayalam, the narration check's second opinion (AI4Bharat Vistaar; 1.5 GB) | MIT |
| `silero-vad` | voice activity (Silero VAD v5, 1.2 MB) | MIT |
| `ecapa-voxceleb` | speaker embeddings (SpeechBrain ECAPA-TDNN, 83 MB) | Apache-2.0 |

`whisper-medium-ml` is what hears Malayalam narration back (the strict check, see
[malayalam-narration.md](malayalam-narration.md)); it also transcribes Malayalam, English
loanwords written in Malayalam script. Its file list pins Thennal D K's repository (the
safetensors conversion, pull request 2) and Whisper medium's own `tokenizer.json`. The survey that
chose it is in [`crates/speech/README.md`](../crates/speech/README.md).

### Testing

Unit and engine tests use a fake `Transcriber` (`FixedTranscriber`), so CI needs no model. The
end-to-end test `crates/speech/tests/whisper_model.rs` (feature `whisper`) runs only when weights are
in `target/models/<id>/` (or `$FILMCRAFT_MODELS_DIR`) and speech samples (mono 16 kHz f32 with a
reference `.txt`) are in `target/fixtures/speech/`; it reports the word error rate and otherwise
prints SKIPPED. Measured on 2026-10-01: `whisper-tiny`, English, 12.2 % WER over 797 words of
the local speech samples (LibriSpeech read speech and dialogue clips) (about 9.5 minutes for the run in a release build on an
Apple-silicon laptop CPU).

## Limits

- `transcript.generate` runs synchronously inside the command; `media.understandSpeech` is the
  background job (progress, cancellable).
- Overlapping speech is labelled with one speaker; two very similar voices (two of Kokoro's male
  voices, cosine up to 0.73) can merge into one.
- Track items that refer to a subclip are looked up by the subclip's id, so a transcript made for
  the parent media is not shown through subclip clips yet.
