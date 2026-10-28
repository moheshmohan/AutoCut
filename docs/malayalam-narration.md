# Malayalam narration

A cloned voice can narrate in Malayalam: the speaker's own voice, reading a script written in
Malayalam script, through IndicF5 (AI4Bharat's F5-TTS for Indian languages, run in pure Rust on
the GPU). This page is the workflow, step by step, with what each step is for. The details are in
[`crates/tts/README.md`](../crates/tts/README.md) (the voice, the front end, measurements) and
[`crates/autocut/README.md`](../crates/autocut/README.md) (Malayalam showcases).

Every step is an engine command, so it works the same from the app, `filmcraft-cli exec`, the
control channel and MCP. The examples use the CLI; `<data>` is FilmCraft's data directory
(`~/Library/Application Support/FilmCraft` on macOS).

## 1. Clone the voice

A *natural* clone (VoxCPM2) of the speaker, from a recording of them talking (a phone video is
fine; 30 s or more of speech, little music):

```sh
filmcraft-cli exec tts.downloadCloneModel '{"model": "voxcpm-2", "wait": true}'
filmcraft-cli --project p.fcproj exec tts.cloneVoice '{"name": "me", "consent": true, "items": [12], "wait": true}'
```

`consent: true` records that the speaker agreed. The voice is `clone:me`; it speaks English
through VoxCPM2 and Malayalam through IndicF5:

```sh
filmcraft-cli exec tts.downloadCloneModel '{"model": "indicf5", "wait": true}'
```

And the Malayalam speech model, which hears every Malayalam narration back (the strict check,
step 6) and checks the reference's words (step 3): Whisper medium fine-tuned on Malayalam by
Thennal D K (Apache-2.0), 3.06 GB, every file pinned and SHA-256 checked:

```sh
filmcraft-cli exec transcript.downloadModel '{"model": "whisper-medium-ml"}'
```

Without it Malayalam narration is still read, unchecked, and every result says so.

## 2. The reference

IndicF5 speaks by *continuing* a reference clip of the speaker: it hears the clip and its words,
then goes on in the same voice with the new text. The clip decides how much the result sounds like
the speaker, so give it the best recording there is, full band (not the 16 kHz voice sample):

```sh
filmcraft-cli --project p.fcproj exec tts.voiceReference '{"voice": "clone:me", "items": [12]}'
```

It takes the best 10–15 s between pauses and keeps it as `<data>/voices/me.indicf5.wav`, with its
words in `me.indicf5.json`. Without `items` the first Malayalam narration makes one from the
voice's own recording. `tts.voiceReference {"voice": "clone:me"}` shows what it has.

## 3. The transcript

IndicF5 aligns the new text to the reference's words, so they must be the words actually said.
The speech model (Whisper) hears Indian English imperfectly and has added sentences that were
never said; wrong words blur the first syllables of every line. Give the exact words instead.
They can come from anywhere: listen and type them, or give the clip to another tool (for
example, ask Gemini for a verbatim transcription, in Malayalam script as spoken, English words
included). Then:

```sh
filmcraft-cli exec tts.voiceReference '{"voice": "clone:me", "transcript": "ഏഹ്, ഫൈവ് പോയിന്റ് ത്രീ ഫ്ലാഷ്, ..."}'
```

The clip stays; only its words change (and the narration cache's key with them).

**The words are checked against the clip.** With the Malayalam speech model installed,
`tts.voiceReference` hears the clip and scores its words against what it heard
(`transcriptCheck`: `matches` or `mismatch`, the character error rate `cer`, what was heard).
More than 0.3 is a `warning`: the words are not what the clip says (they run past the cut, or
belong to another stretch). The author's 9.3 s clip with its hand-typed words scores 0.14 (the
filler "ഏഹ്" is not heard; loanwords are spelled a little differently). `"checkTranscript":
false` skips it.

The words reach IndicF5 exactly as written. Until 2026-10-20 they went through the English text
front end, which dropped every virama ("ഫൈവ് പോയിന്റ് ത്രീ" became "ഫൈവ പോയിന റ ത രീ"); the words
then no longer matched the clip, and IndicF5 said the clip's last word again at the start of
every utterance — heard as "ഇല്ല ഇവിടെ നിന്നാണ്…" after a clip ending "…ഇല്ല.". That was the
gibberish at the start of sentences.

## 4. Trim the reference to a sentence end

A reference picked by length usually stops mid-sentence. IndicF5 continues the clip, so a
sentence left hanging garbles the first words of every generated line. Cut it where a sentence
ends:

```sh
filmcraft-cli exec tts.voiceReference '{"voice": "clone:me", "trimToSentence": true}'
```

The cut goes at the quietest point near the last sentence end the transcript implies (where its
words put the last `.`, `?` or `!` in the speech), keeping at least 8 s, and the transcript is cut
to that sentence. A clip ending at a sentence carries the voice from 8 s on: the author's 14.6 s
reference ended mid-sentence and garbled the first words of lines; the 9.3 s cut of it that ended
at "…ഇല്ല." was preferred (by ear).

Other ways to cut:

- `{"trimToSentence": true, "transcript": "<the words up to a sentence end>"}`: cut at that
  sentence (the words must begin the transcript).
- `{"start": 1.2, "end": 10.5, "transcript": "<the words between>"}`: cut at given times (seconds
  within the current clip); the words are required because the cut takes some away.

The clip before the first trim is kept as `me.indicf5.untrimmed.wav` (and `.json`), and
`{"redo": true}` makes a new reference from the recording.

## 5. Write the script, with numbers as digits

Write the narration in Malayalam script, English words too ("റോബോട്ട്", "ജൈടെക്സ്"): Latin words
are read as best the model can and reported in the result's `notes`. Acronyms in capitals are
spelled with the letters' names ("AI" → "എഐ"), codes with digits in English ("G1" → "ജി വൺ").

Write numbers as **digits**. They are said the way Malayali tech creators say them: years, prices
and numbers from 100 up in English, in Malayalam script, small counts in Malayalam:

| Written | Said |
|---|---|
| 2021 | ട്വന്റി ട്വന്റി വൺ |
| 2013-ൽ | ട്വന്റി തേർട്ടീനിൽ |
| 2021-ന്റെ | ട്വന്റി ട്വന്റി വണ്ണിന്റെ |
| 2008 | ടു തൗസന്റ് എയ്റ്റ് |
| 1969 | നൈന്റീൻ സിക്സ്റ്റി നൈൻ |
| $1,600 | സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളർ |
| 2,500 | ടു തൗസന്റ് ഫൈവ് ഹണ്ട്രഡ് |
| 170,000-ലധികം | വൺ ലാഖ് സെവന്റി തൗസന്റിലധികം |
| 2.5 | ടു പോയിന്റ് ഫൈവ് |
| 4K | ഫോർ കെ |
| 2 ദിവസം | രണ്ട് ദിവസം |
| 85% | എൺപത്തിയഞ്ച് ശതമാനം |

A suffix written after the digits (`2013-ൽ`, `2020-ലും`, `$2,700-നാണ്`) joins the spoken word
with the sandhi Malayalam writes. A four-digit number from 1100 to 2099 without a comma is a year;
write a count with its comma (`1,234`). The subtitles show the digits as written.

The long Malayalam number words ("രണ്ടായിരത്തി ഇരുപത്തിയൊന്ന്") garbled the start of lines, and
the English way is how the author speaks; for Malayalam words instead, pass `"numbers":
"malayalam"`:

```sh
filmcraft-cli --project p.fcproj exec tts.create '{"text": "2013-ൽ ഞാൻ ...", "voice": "clone:me", "language": "ml-IN", "numbers": "malayalam"}'
```

`tts.create`, `tts.edit`, `tts.preview` and `tts.render` take `numbers` (`english`, the default,
or `malayalam`); a narration keeps it, so an edit reads the same way.

## 6. Listen, then narrate

```sh
filmcraft-cli exec tts.preview '{"text": "ട്വന്റി തേർട്ടീൻ മുതൽ ...", "voice": "clone:me", "language": "ml-IN", "path": "/tmp/check.wav"}'
```

Each line takes about 25 s on an M3 Max GPU (the reference's length sets most of it); readings
are kept in the narration cache, so a rebuild reads nothing twice. A script without digits keeps
its cached readings whichever number style is chosen. `take: 1, 2, …` reads a line afresh.

**The strict check (on by default).** `tts.preview`, `tts.create`, `tts.edit` and `tts.render`
hear every Malayalam utterance back with the Malayalam speech model and score it: the character
error rate over the utterance, and the error rate of the first six syllables of each of its
sentences on their own — where IndicF5's garble was. An utterance over the thresholds (0.25
overall, 0.3 at a sentence start), or whose signal is broken (empty, clipped, a long silence
mid-reading, a burst of level at the start), is read again with its own seed, up to four
readings, and the best is kept.

**A stricter check.** When readings that passed still sound wrong (a mispronounced word costs
little of a long utterance's character error rate), ask for a stricter check with `checkPolicy`:

```sh
filmcraft-cli exec tts.create '{"text": "…", "voice": "clone:me", "language": "ml-IN",
  "checkPolicy": {"readings": 6, "cer": 0.12, "startCer": 0.2}}'
```

`readings` (1–8) is how many times an unclear utterance is read at most, `cer` (0–1) the highest
character error rate over an utterance that passes and `startCer` (0–1) the highest at the start
of a sentence; a field left out keeps the default (Malayalam 4, 0.25, 0.3; English 3, 0.15, 0.5),
anything else is refused with what it should be. A policy other than the default is part of
the cache key, so a stricter check never reuses a reading that only passed a lenient one: a
reading cached under the default whose every utterance was heard clearly enough for the stricter
policy is kept as it is; another is heard once more as a whole and kept only if it passes;
the rest are read again. More readings take longer: each is a full IndicF5 generation and a
hearing. Results carry `check`:

```json
{"status": "clear", "model": "whisper-medium-ml check v1", "policy": {"readings": 4, "cer": 0.25, "startCer": 0.3}, "utterances": [
  {"text": "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്.", "heard": "ഇതാണ് ബോസ്റ്റൺ ഡൈനാമിക്സിന്റെ സ്പോട്ട്",
   "cer": 0.0, "startCer": 0.0, "takes": 1, "kept": 0, "seed": 1, "passed": true, "signal": [],
   "tries": [{"attempt": 0, "seed": 1, "heard": "…", "cer": 0.0, "startCer": 0.0, "starts": [0.0], "signal": []}]}]}
```

`status` is `clear`, `unclear` (an utterance never passed: listen to it, or read it with
another `take`), `skipped` (no speech model: `reason`) or `off`. Every utterance has its own
seed, recorded with each try, so one bad sentence is read again alone and can be reproduced.
Checked readings are cached under their own key with their report, so a rebuild still reports.
`"check": "off"` reads without hearing. The Text to Speech panel shows the outcome in the
status bar.

## 7. Voice styles: emotion, temperature, emphasis, speed, expressiveness

IndicF5 continues the reference clip, so the clip sets the emotion and prosody of everything it
reads: one calm reference reads every line calm. A voice keeps **one reference per style** beside
its main (plain) one — a stretch of your own Malayalam spoken that way — as F5-TTS's multi-style
generation does:

```sh
# an excited stretch of one of your clips (seconds in it), with its exact words
filmcraft-cli voice style set --voice clone:me --style excited --file ~/Movies/IMG_4184.MOV \
  --start 41.2 --end 52.8 --transcript "അത് ശരിക്കും ഒരു മാജിക് പോലെ തോന്നി! …"
# or a project item: --project p.fcproj --item 1666
filmcraft-cli voice style list --voice clone:me         # what it has, and what each emotion reads after
filmcraft-cli voice style remove --voice clone:me --style calm
```

(`exec tts.voiceStyles '{"voice": "clone:me", "set": "excited", "path": …, "start": …, "end": …,
"transcript": …}'` is the command.) Styles are `conversational`, `warm`, `excited`, `amazed`,
`curious`, `calm` and `dramatic`; `plain` replaces the main reference (6–15.5 s, the old one kept
as a version). **From the style finder**: a pick of its report is taken as it is — its style,
file (else project item), stretch and words —

```sh
filmcraft-cli voice style set --voice clone:me --report styles.json --candidate excited-1
# the same, with the words corrected, or as another style:
filmcraft-cli voice style set --voice clone:me --report styles.json --candidate calm-1 --transcript "…" --style warm
```

(`{"report": "styles.json", "candidate": "excited-1"}`, or the pick itself as `"entry": {…}`;
the request's own fields win). A pick already ends at a sentence end, so it is kept whole
(`--trim` cuts it at its words' last sentence end anyway). The stretch is cut at the last sentence end its words imply
(end the words with the sentence's full stop; `--no-trim` keeps it whole) and must leave 5–16 s.
Without `--transcript` both Malayalam speech models of the strict check (whisper-medium-ml and
IndicWhisper) hear the words; the result shows both and how far apart they are, and warns when
they disagree — give the exact words then, IndicF5 aligns its reading to them. Setting or removing
a style keeps the voice's references as a version first (`voice versions`, `voice rollback`).

**Which reference a line reads after.** A narration's delivery (the Text to Speech panel's
*Emotion*) reads after its style's reference; a style the voice has no reference for reads after
the nearest one it has, else the main reference:

| Delivery | Then | Then |
|---|---|---|
| excited | amazed | conversational |
| amazed | excited | conversational |
| curious | conversational | |
| conversational | warm | |
| warm | conversational | |
| calm | warm | |
| dramatic | calm | warm |

**Inline tags** switch the style within one narration: `[excited] അത് പ്രിന്റ് ആയി! [calm] പക്ഷേ
പിന്നെ… ` reads the first sentence excited and the rest calm; `[plain]` goes back to the main
reference and `[breath]` is a breath-length pause of the reference's room tone. Tags are never
read aloud or shown in subtitles; the strict check hears each part against its own words; a line
break also starts a new utterance.

**Temperature, emphasis, speed, expressiveness.** Each style has a preset for IndicF5's sampler
and the intonation after it; a narration may set them (`tts.create`, `tts.edit`, `tts.preview`, `tts.render`; the panel's sliders; a showcase
plan's `script.voice`):

| Setting | Range | Plain | What it does |
|---|---|---|---|
| `temperature` | 0.3–1.5 | 1.0 | the flow-matching sampler's starting noise: lower is steadier and flatter, higher livelier and riskier |
| `emphasis` | 1–4 | 2.0 | classifier-free guidance: how closely the voice follows the words and its reference |
| `speed` | 0.7–1.4 | 1.0 | speaking speed on top of `pace` (the natural rate is held at 4.6–6 syllables a second first) |
| `expressiveness` | 0.5–2 | 1.0 | after the voice reads a line, its pitch range: each utterance's deviations from its median pitch scaled by it (`filmcraft_audio_dsp::prosody::reshape`, TD-PSOLA, length unchanged), a question's end raised 2 semitones, stressed syllables lifted 3 dB per unit over 1; 1 leaves the reading exactly as synthesized |

IndicF5's readings of the owner span about 3.0 semitones of pitch (10th–90th percentile) against
4.0 in his own speech, so the livelier styles widen it:

| Style | temperature | emphasis | speed | expressiveness | pauses |
|---|---|---|---|---|---|
| conversational | 1.0 | 2.0 | 1.02 | 1.1 | ×0.9 |
| warm | 0.95 | 2.0 | 0.98 | 1.15 | ×1.0 |
| excited | 1.1 | 2.4 | 1.06 | 1.3 | ×0.8 |
| amazed | 1.1 | 2.3 | 1.0 | 1.35 | ×0.85 |
| curious | 1.05 | 2.1 | 1.0 | 1.2 | ×0.95 |
| calm | 0.85 | 1.8 | 0.95 | 1.0 | ×1.25 |
| dramatic | 0.9 | 2.2 | 0.92 | 1.2 | ×1.35 |

`null` (or leaving it out) goes back to the preset; a value out of range is refused with the
range. English narrations take none of them.

**Pauses.** Between utterances the pause follows how the last one ended — a full stop 320 ms, `?`
400, `!` 340, `…` 560, a comma 180, a line break 480, a blank line 760 — stretched by its style
(excited shorter, calm and dramatic longer) and varied ±12 % per take and utterance (the same take
always pauses the same). A pause marker (`[pause 1s]`) is exact.

**The cache.** A plain line of one utterance keeps the cache key it always had; a styled line keys
on the references it reads after, its delivery and its settings, so changing a style's reference
or a setting reads it again and a styled reading is never served for a plain one. A line of more
than one utterance is read again once (its pauses changed on 2026-10-28).

**Measured (2026-10-28, M3 Max GPU, on a copy of the profile).** The style finder's picks set as
they are (`--report styles.json --candidate excited-1 | calm-1 | amazed-1`): excited `IMG_3338.MOV`
149.9–158.0 s, calm `IMG_3338.MOV` 41.9–50.2 s, amazed `20250729-124548.mov` 44.0–51.2 s; the main
reference is the voice's own (`20250729-124548.mov` 132.4–141.2 s, words as the speech model heard
them). Five lines of `script-ml-part2.json` read in each way (`voice_styles_lab read`, the strict
check on), measured with the prosody tracker (`voice_styles_lab table`: pitch range = 10th–90th
percentile, semitones; syllable peaks a second of speech; the strict check's character error rate
of each utterance's first reading and of the one kept):

| Reading | pitch range (st) | movement (st/s) | syllables/s | energy variation (dB) | first-reading CER | kept CER | readings |
|---|---|---|---|---|---|---|---|
| main reference, plain | 4.58 | 11.3 | 4.44 | 4.5 | 0.108 | 0.114 | 6 |
| excited reference, the model's own sampler | 3.58 | 12.3 | 5.70 | 5.5 | 0.053 | 0.053 | 5 |
| excited (reference + preset) | 4.16 | 14.8 | 5.20 | 5.1 | 0.096 | 0.062 | 8 |
| calm reference, the model's own sampler | 2.79 | 13.5 | 4.83 | 4.5 | 0.073 | 0.073 | 5 |
| calm (reference + preset) | 3.13 | 12.4 | 4.63 | 5.4 | 0.082 | 0.045 | 6 |
| main, temperature 0.7 | 2.66 | 11.9 | 4.51 | 4.2 | 0.085 | 0.074 | 9 |
| main, temperature 1.3 | 5.47 | 12.0 | 4.29 | 4.5 | 0.188 | 0.103 | 7 |
| main, emphasis 1.6 | 3.32 | 11.1 | 4.31 | 4.5 | 0.130 | 0.080 | 7 |
| main, emphasis 2.6 | 5.41 | 16.6 | 4.46 | 4.5 | 0.146 | 0.090 | 8 |
| main, expressiveness 1.3 | 5.46 | 11.9 | 4.31 | 4.7 | 0.101 | 0.096 | 6 |
| main, expressiveness 1.6 | 5.05 | 13.7 | 4.24 | 4.9 | 0.129 | 0.108 | 6 |

His own speech in the references: main 4.22 st, 4.0 syllables/s; excited 4.56 st, 5.5
syllables/s, 5.8 dB; calm 3.07 st, 4.0 syllables/s, 7.1 st/s of movement; amazed 5.11 st, 3.4
syllables/s.

- **The reference carries the delivery.** The excited reference read 28 % faster than the main one
  (5.70 against 4.44 syllables/s; his clip 5.5) with more energy variation (5.5 against 4.5 dB;
  his clip 5.8); the calm reference read with the narrowest pitch (2.79 st; his clip 3.07). Its
  rate stays at 4.6–6 syllables/s (the natural-rate clamp lifts the calm clip's 4.0 to 4.6).
- **IndicF5 compresses the pitch** of a lively reference (3.58 st against his 4.56): the excited
  preset (expressiveness 1.3 among it) brings it back to 4.16.
- **Temperature and emphasis widen the pitch** monotonically (temperature 0.7 → 1.3: 2.66 → 5.47
  st; emphasis 1.6 → 2.6: 3.32 → 5.41 st), and both cost first readings above the model's own
  (first-reading CER 0.188 at temperature 1.3, 0.146 at emphasis 2.6, against 0.108): the strict
  check reads those again (kept 0.103, 0.090). Keep them near the presets.
- **Expressiveness 1.3** widened the main reference's readings from 4.58 to 5.46 st at no cost in
  what was heard (0.101 first, against 0.108); at 1.6 the tracker saw no wider (5.05 st) and the
  first readings were heard less well (0.129).
- Five lines a reading: differences of a few hundredths in the error rates are noise. Listen to
  the readings: `target/agent-voice-styles/samples/<reading>/line<k>.wav` of the run.

## 8. A Malayalam showcase

A showcase plan with `"language": "ml"` narrates its Malayalam `intro`, `lines` and `outro` in
Malayalam under the strict check and makes Malayalam subtitles; what was heard, line by line
(text, heard, `cer`, `startCer`, readings, the seeds kept), goes into the build's
`narration_check` and next to the export as `<out>.narration-check.json`, and every line never
heard clearly is noted (`--no-narration-check` turns the check off). `"narrationCheck":
{"readings": 6, "cer": 0.12, "startCer": 0.2}` (or `--narration-check
readings=6,cer=0.12,startCer=0.2`) makes the check stricter for every Malayalam line (`tts.create`'s
`checkPolicy`, above). Each part's `delivery` is its voice style (section 7), and `script.voice`
sets how a part is read — `{"intro": {"emphasis": 2.4}, "3": {"temperature": 1.1, "speed":
1.05}}` — over `script.voiceDefault` (every part). `"numbers": "malayalam"` (or `--numbers
malayalam`) changes how digits are said:

```sh
filmcraft-cli showcase --report analysis.json --script-in plan-ml.json --language ml \
  --voice clone:me --places-from-locations locations.json --export robots-ml.mp4
```

`--places-from-locations` takes the places `filmcraft-cli photos locations --album "<title>"`
read from Google Photos (Google strips them from downloads) for each entry's lower third.

## Natural spoken Malayalam

The voice reads what is written: a script in written, textbook Malayalam sounds robotic however
good the voice. Write it the way a Malayali tech creator talks: short sentences, English tech
words in Malayalam script (വെൽക്കം ചെയ്യുന്നു, വിസിറ്റേഴ്സ്, ലോഞ്ച് ചെയ്തു), spoken endings
(ഉണ്ടാക്കിയതാ, തോന്നൂ, പറയണേ), digits for numbers, acronyms letter by letter. The rules, with
before-and-after examples, are [malayalam-style.md](malayalam-style.md). Three tools keep a script
that way:

```sh
filmcraft-cli ml-lint script-ml.json --plan plan.json          # what sounds written, and facts lost
filmcraft-cli ml-script --plan plan.json --out script-ml.json  # a first draft by the local model
filmcraft-cli ml-script --plan plan.json --remember script-ml.json  # your reviewed lines, as examples
```

(`script.lintMalayalam` and `autocut.malayalamScript` are the commands.) Your approved lines are
kept in `<data>/autocut/malayalam-examples.json`, on your machine only, and shown to the model
first next time. A Malayalam showcase checks its script before the first line is voiced and
prints the findings; `--strict-script` stops the build on them.

## Files

| File | What |
|---|---|
| `<data>/voices/me.json`, `me.wav` | the clone and its voice sample (English) |
| `<data>/voices/me.indicf5.wav`, `.json` | the Malayalam reference clip, its words, where it came from |
| `<data>/voices/me.indicf5.untrimmed.wav`, `.json` | the reference before its first trim |
| `<data>/voices/me.indicf5.<style>.wav`, `.json` | a voice style's reference (`excited`, `calm`, …), its words, where it came from |
| `<data>/voices/me.versions/` | the voice's earlier references (`voice versions`, `voice rollback`) |
| `<data>/narration-cache/` | readings, keyed by voice, reference, words and settings; a checked reading's report beside it (`<key>.json`) |
| `<data>/models/whisper-medium-ml/` | the Malayalam speech model (the strict check) |
| `<out>.narration-check.json` | a Malayalam showcase's narration as the strict check heard it |
