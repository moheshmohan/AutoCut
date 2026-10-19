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

Pick the album whole (the screen agent drives Google's picker), download its new videos and import
them into the project; clips the project already holds are reused, not imported twice:

```sh
filmcraft-cli --project "$P" photos pick --album "Robot showcase" --videos-only --save
```

Analyse the new clips (their item ids are in the pick's output; clips analysed before come from the
analysis cache):

```sh
filmcraft-cli --project "$P" analyze robot --items 4841,2630,2631 --out "$R/report-new.json"
```

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
stretch; `--at` takes exact times; portrait clips get tall tiles):

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
  project. `from`/`to`/`speed`/`fit`/`mute`/`volume`/`focus` are the shot's controls.
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

## 6. QA

Each export gets, beside it:

| file | what |
|---|---|
| `<out>.qa.json` | findings (joins, black frames, loudness and voice level, the robot in every shot, privacy) and per-entry privacy counts (`privacy`: faces, plates, named places) |
| `<out>.qa.png` | a frame of every entry under its number, red/amber when it has an error/warning |
| `<out>.privacy.png` | a frame of every clip with masks, where most of it is hidden |

Read the summary line the build prints (`showcase: QA: …`), then look at both sheets. A privacy
warning ("a redaction mask … covers 34 % of …") means a blur that size hides the shot: give that
entry or shot `"redact": false`, narrow its targets, or choose another window.

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
