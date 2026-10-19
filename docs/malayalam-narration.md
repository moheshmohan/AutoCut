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

## 7. A Malayalam showcase

A showcase plan with `"language": "ml"` narrates its Malayalam `intro`, `lines` and `outro` in
Malayalam, hears them back and makes Malayalam subtitles; `"numbers": "malayalam"` (or
`--numbers malayalam`) changes how digits are said:

```sh
filmcraft-cli showcase --report analysis.json --script-in plan-ml.json --language ml \
  --voice clone:me --places-from-locations locations.json --export robots-ml.mp4
```

`--places-from-locations` takes the places `filmcraft-cli photos locations --album "<title>"`
read from Google Photos (Google strips them from downloads) for each entry's lower third.

## Files

| File | What |
|---|---|
| `<data>/voices/me.json`, `me.wav` | the clone and its voice sample (English) |
| `<data>/voices/me.indicf5.wav`, `.json` | the Malayalam reference clip, its words, where it came from |
| `<data>/voices/me.indicf5.untrimmed.wav`, `.json` | the reference before its first trim |
| `<data>/narration-cache/` | readings, keyed by voice, reference, words and settings |
