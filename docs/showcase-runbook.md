# Robot showcase runbook

One round of the "every robot I've met" showcase, end to end, with the commands that do each step.
A round takes the previous round's reviewed plan, the new clips added to the Google Photos album,
and the review notes, and produces the English and Malayalam videos, checked, uploaded to the album
and exported in 4K. Everything below is `filmcraft-cli` (build it with `cargo build --release -p
filmcraft-cli`; the binary is `target/release/filmcraft-cli`). How each piece works:
[crates/autocut/README.md](../crates/autocut/README.md); every flag: `filmcraft-cli help`.

## Before the first round (once)

- Google Photos: `filmcraft-cli photos auth --albums` (`crates/gphotos/README.md`).
- Local models: `ollama pull qwen2.5vl:7b` (analysis, the picker agent, the privacy check, QA) and
  `ollama pull qwen3:8b` (scripts).
- The face and plate detectors: `filmcraft-cli exec redact.downloadModels wait=true` (4.4 MB).
- A cloned voice for both languages (`tts.cloneVoice`; Malayalam through IndicF5:
  `exec tts.downloadCloneModel model=indicf5`, with a Malayalam reference trimmed to a sentence
  end by `tts.voiceReference` — the steps are in [malayalam-narration.md](malayalam-narration.md)).
- The Malayalam speech model, which hears every Malayalam line back (the strict check, on by
  default): `exec transcript.downloadModel model=whisper-medium-ml` (3.06 GB). Then
  `exec tts.voiceReference voice=clone:<name>` says whether the reference's words match its clip
  (`transcriptCheck`).
- Malayalam on screen and in subtitles needs the craft-fonts build input (`CRAFT_FONTS_DIR`, see
  `CLAUDE.md`).
- The browser is shared: check `filmcraft-cli locks` first. The picker agent takes the `chrome`
  lock itself; a session that drives Chrome by hand holds it (`locks hold chrome`).

## The files of a round

Keep each round's files side by side and never overwrite the previous round's (below, round 6 in
`~/Movies/AutoCut`, `R=~/Movies/AutoCut`):

| file | what |
|---|---|
| `Robot Showcase v6.fcproj` | the project: a copy of v5's, new clips imported into it |
| `report.json` | the clip analysis (`analyze --out`, or `showcase --report-out`) |
| `plan-v5.json` | last round's reviewed plan (entries + script) |
| `notes-v6.json` | this round's review notes (below) |
| `script-ml-v6.json` | the Malayalam lines: `{"intro", "outro", "lines": {"<entry label>": "…"}}` |
| `plan-v6.json`, `plan-v6-ml.json` | this round's plans, written by `showcase-plan` |

```sh
cp "$R/Robot Showcase v5.fcproj" "$R/Robot Showcase v6.fcproj"   # the copy is what we change
P="$R/Robot Showcase v6.fcproj"
```

## 1. The new material from the album

Pick the album whole (the screen agent drives Google's picker), download its new videos and photos
and import them into the project; items the project already holds are reused, not imported twice
(`--videos-only` leaves the photos out):

```sh
filmcraft-cli --project "$P" photos pick --album "Robot showcase" --save
```

Analyse the new items (their item ids are in the pick's output; items analysed before come from
the analysis cache; a photo is looked at once):

```sh
filmcraft-cli --project "$P" analyze robot --items 4841,2630,2631,4977 --out "$R/report-new.json"
```

### Photos of robots never filmed

A robot you only photographed belongs in the show too: add the photo (JPEG, HEIC or PNG, as the
phone took it) to the album. The pick downloads Google's original at full resolution (a HEIC is
decoded by FilmCraft itself; anything over 8192 px on its long side is held at 8192), turned
upright by its EXIF orientation and dated by Google's record of it (else its EXIF date; Google
takes the position out, so its place comes from `photos locations` in step 2, as a clip's does).
The analysis locates the robot in it once and keeps its box (`subject` in the report).

- **An entry of photos.** A robot with video is shown in its video; one with only photos is an
  entry of its photos (up to three), each held as long as its words need (at most 20 s) with a
  slow pan and zoom that starts wide and ends framed on the robot (or the reverse) — eased in and
  out, about 2.5 % closer a second, the next photo going the other way. A 4:3 or landscape photo
  fills the 16:9 frame; a portrait photo is shown whole over its blurred copy (the pillars the
  portrait clips get) when a crop would cut its robot (or none was found), else it fills the
  frame with the move aimed at the robot. The pillars stay through the transitions into and out
  of a picture shown whole, and dip to black with it into a new year. Photos have no sound; the
  narration is timed over them like over a clip.
- **In the notes** (step 4) a photo shot is `{"item": 4977, "photo": true, "hold": 4.0, "move":
  "in"}` — `hold` in seconds (1–20; without it, its share of the entry), `move` one of `in`,
  `out`, `pan`, `none` (without it, in and out in turn), `fit` (`full` for the pillars, `fill`) and
  `focus: [x, y]` (the point to aim at, 0..1 of the photo) as for clips. An item the project holds
  as a still is a photo without saying so. An entry may mix photos and clips:
  `"Pepper": {"shots": [{"item": 2407}, {"item": 4977, "hold": 3, "move": "pan"}]}`; `add` makes an
  entry of photos: `{"label": "Toy robot", "unix": "2021-02-03 10:00:00", "shots": [{"item":
  4978, "photo": true}], "line": "…"}`.
- **Checks:** `sheet` shows a photo as one tile; QA does not take a photo's silence for a dip in
  the shots' sound; the privacy pass looks at a photo once and keeps the blur for its whole hold.
- **Any still, by hand:** select it on the timeline and Clip ▸ Video Options ▸ Ken Burns
  (`exec clip.kenBurns '{"clip": 812, "subject": [0.55, 0.2, 0.25, 0.5], "move": "out"}'`).

## 2. Where each clip was filmed

Google strips the location from the clips its API and picker hand out, so `photos locations` reads
each album item's place from the Google Photos page (the screen agent opens each item's Info panel;
it holds the `chrome` lock; `--max-items 5` is a dry run, `--resume` goes on after an interruption):

```sh
filmcraft-cli --project "$P" photos locations --album "Robot showcase" --out "$R/locations-v6.json" --save
```

It writes `[{fileName, date, gmtOffset, place, estimated, …}]` to `--out` and, with `--project`,
stores each place with the project's item (its `Location` field, and `Location Estimated` when
Google only guessed it). An entry's `place` ("Busan, South Korea") is what its lower third shows
(it starts no chapter of its own). Three ways to use them, the most reviewable first:

- `showcase-plan … --places-from-locations "$R/locations-v6.json"` (step 4) writes the place of
  each entry without one into the plan, from its first shot (matched by file name), and lists the
  entries it filled (`placesFilled`); a `place` in the notes always wins;
- `showcase … --places-from-locations <file>` does the same at build time;
- the build also fills places from the items' stored `Location` by itself.

(The code: `filmcraft_autocut::places::{Places::from_json, fill, from_items}`, and
`plan_patch::fill_places` for the plan as written.)

## 3. Look at the clips

Contact sheets of source clips, to choose windows and fits (`item:from-to` limits a row to a
stretch; `--at` takes exact times; portrait clips get tall tiles; a photo is one tile):

```sh
filmcraft-cli --project "$P" sheet --items 2407,2417:0-30,14 --n 8 --out "$R/sheet.jpg"
filmcraft-cli --project "$P" sheet --items 2417 --at 2,10.5,18,33 --side 480 --out "$R/dilly.jpg"
```

## 4. The review notes, into the plan

The notes are JSON keyed by entry **label** (numbers change; labels do not). The format is in
[`crates/autocut/src/plan_patch.rs`](../crates/autocut/src/plan_patch.rs); the usual edits:

```json
{
  "timezone": "+04:00",
  "entries": {
    "Pudu CC1": {"place": "Sydney, Australia", "line": "In Sydney that October, …",
                 "chips": ["4-in-1 cleaner", "700–1,000 m²/h", "Pudu, Shenzhen"]},
    "Disinfection robot": {"shotEdits": {"14": {"from": 0.0, "fit": "full"}}},
    "Restaurant serving robot": {
      "label": "Dilly S serving robot",
      "shots": [{"item": 2417, "plain": true, "from": 2.0, "to": 10.5, "fit": "full"},
                {"item": 2417, "plain": true, "from": 18.0, "to": 33.0, "speed": 3.0, "mute": true}]
    },
    "Micropolis M02": {
      "shots": [{"item": 2407}],
      "split": [{"label": "Micropolis M01", "unix": "2022-01-23 13:40:35",
                 "shots": [{"item": 2408}, {"item": 2409, "from": 19.0, "to": 23.5}],
                 "line": "And this is its big brother, …", "delivery": "amazed"}]
    },
    "Promobot": {"redact": {"targets": ["DP World logo"]}, "delivery": "calm"},
    "Some old entry": {"remove": true}
  },
  "add": [{"label": "Unitree Go2", "kind": "quadruped", "unix": "2024-11-15 16:47:16",
           "description": "…", "place": "Dubai", "shots": [{"item": 4841, "plain": [4, 40], "mute": true}],
           "line": "…", "chips": ["Unitree Go2", "15 kg", "From $1,600"]}],
  "script": {"pronounce": {"Go2": "Go two"}},
  "outro": {"logo": "~/Movies/AutoCut/branding/logo.png", "channel": "…", "seconds": 12}
}
```

- A shot by `item` alone is the plan's analysed shot of that clip; with `plain` it is counted in
  view across its window (`plain: [a, b]` without fixing the window), its length read from the
  project. `from`/`to`/`speed`/`fit`/`mute`/`volume`/`focus` are the shot's controls; a photo's are
  `hold`/`move`/`fit`/`focus` ([Photos of robots never filmed](#photos-of-robots-never-filmed)).
- `split` makes a second entry from one (two robots in one entry); `add` brings new robots.
- The outro's robot count ("Forty-three", or `{Count}`) follows the entries.
- Top-level fields merge into the plan: the look (`outro`, `style`, `transitions`, …) and, for the
  Malayalam narration, `"numbers": "malayalam"` (digits said as Malayalam words; by default they
  are said the English way, as Malayali creators say them — see
  [malayalam-narration.md](malayalam-narration.md)).

```sh
filmcraft-cli showcase-plan --base "$R/plan-v5.json" --patch "$R/notes-v6.json" --project "$P" \
    --places-from-locations "$R/locations-v6.json" \
    --out "$R/plan-v6.json" --lines "$R/script-ml-v6.json"
```

It writes `plan-v6.json` and `plan-v6-ml.json`, prints the counts, names every robot entry left
without a line and the entries that took their place from the locations file; the Malayalam plan
fails naming each spoken entry its lines file lacks. Several `--patch` files apply in turn. (The v5
round's 200-line Python script is one notes file: `showcase-plan` reproduces its `plan-v5.json` and
`plan-v5-ml.json` exactly.) Write the Malayalam lines with numbers as digits; the voice, its
reference and how numbers are said: [malayalam-narration.md](malayalam-narration.md).

### The Malayalam lines, spoken not written

Write `script-ml-v6.json` in natural spoken Malayalam ([malayalam-style.md](malayalam-style.md)),
or start from a draft by the local model, then read and fix it:

```sh
filmcraft-cli ml-script --plan "$R/plan-v6.json" --out "$R/script-ml-v6.json"   # a draft (qwen3:8b)
filmcraft-cli ml-lint "$R/script-ml-v6.json" --plan "$R/plan-v6.json"           # sounds written? facts lost?
filmcraft-cli ml-script --plan "$R/plan-v6.json" --remember "$R/script-ml-v6.json"   # once approved
```

`ml-lint` names each line that sounds written (long sentences, formal words, numbers in
Malayalam words, Latin letters) and each number, name, unit or month of the English it lost;
`--fix --out` makes the safe replacements. `--remember` keeps the approved lines as examples for
the next round's draft. The build checks the script again before voicing it (`--strict-script`:
any warning stops it).

## 5. Build both languages, with privacy

```sh
filmcraft-cli --project "$P" showcase --report "$R/report.json" --script-in "$R/plan-v6.json" \
    --languages en,ml --lines-ml "$R/script-ml-v6.json" --voice clone:<name> \
    --redact faces,plates --title "Every Robot I've Met" \
    --export "$R/Every Robot I've Met - v6.mp4" --save
```

One run builds the English show, then the Malayalam one from the same plan
(`… - v6 (Malayalam).mp4`), each in its own sequence of the project, each with subtitles
(`.srt`, `.vtt`), YouTube chapters (`.chapters.txt`), a thumbnail (`.thumbnail.jpg`) and QA.
Visitors' faces and plates are blurred, robots left sharp, and the plan's named targets (logos) are
hidden. Two caches in the data directory make the second language and every rebuild fast:

- the narration cache: a line spoken before is not synthesized again;
- the redaction cache: detections, the vision check's answers and the targets' looks, by file
  content (measured: 11 clips, 152 s the first time, 0.13 s the same again, 34 s with every window
  moved by half a second as the other language's cut moves them). `redact --fresh` ignores it.

Heavy runs: the GPU is shared (`locks`), and a build with a cloned voice takes a while; try notes
on a small plan (`--voice` left out builds without narration) before the full round.

**The Malayalam narration is heard back.** Every Malayalam utterance is heard by the Malayalam
speech model and read again until clear (the strict check, [malayalam-narration.md](malayalam-narration.md)
step 6); the build prints `narration check: N of M Malayalam lines heard clearly` and writes
`… (Malayalam).narration-check.json` next to the export: per line its text, what was heard, the
character error rate (`cer`), the worst sentence start (`startCer`), the readings made, and per
utterance the seed kept and every try. Listen to every line with `"passed": false` (the build
notes them too); read one again with another take by editing it (`tts.edit … take=1`), or
rewrite it. `--no-narration-check` builds without the check.

## 6. QA

Each export gets, beside it:

| file | what |
|---|---|
| `<out>.qa.json` | findings (joins, black frames, black sides around transitions, loudness and voice level, the mix on small speakers, the robot in every shot, privacy), per-entry privacy counts (`privacy`: faces, plates, named places) and the mix per stem (`mixAnalysis`) |
| `<out>.qa.png` | a frame of every entry under its number, red/amber when it has an error/warning |
| `<out>.privacy.png` | a frame of every clip with masks, where most of it is hidden |
| `<out>.narration-check.json` | the Malayalam narration as the strict check heard it, line by line (Malayalam builds) |

Read the summary line the build prints (`showcase: QA: …`), then look at both sheets. A privacy
warning ("a redaction mask … covers 34 % of …") means a blur that size hides the shot: give that
entry or shot `"redact": false`, narrow its targets, or choose another window. The mix's
warnings for small speakers are below ([Mix for small speakers](#mix-for-small-speakers)).

## 7. Check frames of the result

Any moment of an export, in one labelled sheet (no project needed):

```sh
filmcraft-cli frames "$R/Every Robot I've Met - v6.mp4" --at 601,603,605,607 --scale 0.5 --out "$R/check.jpg"
```

Fix by editing the notes and run steps 4–5 again; the caches make the rebuild cost only what
changed.

## 8. Upload to the album

Add `--upload --upload-album "Robot showcase"` to the build (both languages are uploaded, the
second titled "… (Malayalam)"), or upload a finished export:

```sh
filmcraft-cli photos upload "$R/Every Robot I've Met - v6.mp4" --album "Robot showcase"
```

## 9. The 4K final

The builds are 1080p sequences in the saved project; export each at 4K (the footage is rendered
from its sources, titles and graphics drawn at 4K). The sequence ids are in the build's output
(`languages[i].sequence`):

```sh
filmcraft-cli --project "$P" exec sequence.open item=<sequence id> --save
filmcraft-cli --project "$P" export "$R/Every Robot I've Met - v6 4K.mp4" --preset "YouTube 2160p 4K Ultra HD"
```

`--upload-youtube` on the build (or `filmcraft-cli publish run youtube <file>`) sends a video to
YouTube with its chapters and thumbnail, private unless `--youtube-privacy` says otherwise.

## Mix for small speakers

The shows are watched on TVs, laptops and phones, often quietly. Their speakers play little below
about 120 Hz, and words are understood at low volume from what they carry between 1 and 4 kHz.
The first finished show sounded bassy and distorted on a TV and had to be turned up to be
understood; measured per stem, the cause was the cloned narration (37 % of its energy below
120 Hz, 1 % at 1–4 kHz) and its peaks, which made the Mix's limiter take up to 3.7 dB off the
whole mix. So every build is now mixed **speech first** (`--mix speech`, the default; the details
and the before/after numbers: [crates/autocut/README.md](../crates/autocut/README.md#mix-for-small-speakers)):

- the narration, on A2: high-passed at 80 Hz, its boom and mud cut and its presence (1–4 kHz)
  lifted as far as its measured tone needs (the cloned voices: all the way), de-essed,
  compressed 3:1 and peak-limited, its fader making up the level the chain took. `--mix gentle`
  keeps the voice closer to its speaker (no low-mid cut, the lift at 2–3.5 kHz, above the
  vowels' formants); the author preferred the clearer default on their TV (2026-10-21), and
  `filmcraft-cli voice mix-check` measures how far each chain moves a voice;
- the music (A3), the shots' sound (A1) and the effects (A4, A5) high-passed (40, 100 and 120 Hz)
  with less bass, the music ducked 18 dB under the voice and dipped 4 dB around 2.5 kHz while it
  speaks, the effects and the transitions' sounds quieter;
- the Mix normalised as before (`--loudness social`: −14 LUFS, −1 dBTP), its limiter now taking
  under 1 dB off the peaks.

Everything is ordinary track effects and keyframes: open the Audio Track Mixer to adjust them.
`--mix cinematic` (or `"mix": "cinematic"` in the plan) keeps the earlier full-range balance.

Measure a build, or any sequence (the active one, or `sequence`; `from`/`to` for a stretch):

```sh
filmcraft-cli --project "$P" exec audio.analyzeMix '{"sequence": <id>}'
```

It reports per stem the loudness, true peak, band shares (`sub` below 120 Hz, `presence` 1–4 kHz)
and octave levels, the Mix's limiter's gain reduction (and a track limiter's), how far the
narration stands above the music and above everything else while it speaks, and clipping in each
narration clip. QA runs it on every export and warns when:

| warning | what to do |
|---|---|
| the narration less than 12 LU above the music while it speaks | duck the music deeper (Essential Sound ▸ Ducking on its clip, then Generate) or lower it |
| the Mix's limiter takes more than 2 dB off the peaks | hold the loudest stem's peaks on its own track (its level, compressor or limiter); `exec audio.analyzeMix` names each stem's true peak |
| more than 25 % of the mix's energy below 120 Hz | high-pass the stem it names (the speech mix does; a `--mix cinematic` build does not) |
| a narration clip is clipped (flat tops) | read the line again (`tts.edit … take=1`), or lower it before it reaches the mix |

**A show built before** (a final already exported) is mixed again without rebuilding it, on a copy
of its project:

```sh
cp "$P" "$R/Robot Showcase v6 (speech mix).fcproj"
P2="$R/Robot Showcase v6 (speech mix).fcproj"
filmcraft-cli --project "$P2" showcase-remix --sequence <id> --loudness social --save
filmcraft-cli --project "$P2" exec audio.analyzeMix '{"sequence": <id>, "from": 40, "to": 100}'
filmcraft-cli --project "$P2" export "$R/Every Robot I've Met - v6 4K.mp4" --preset "YouTube 2160p 4K Ultra HD"
```

(`showcase-remix` opens that sequence, takes off the effects an earlier mix put on A1–A5,
levels, ducks and treats every stem again, sets the transitions' sounds to the profile's level
and normalises; running it twice gives the same mix. To hear the difference first, export a
minute of each: `export before.mp4 --preset "YouTube 1080p Full HD" --start 40 --end 100` on the
old project and the same on the new.)
