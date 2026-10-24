# Video pipeline: from a Google Photos album to YouTube and Instagram

The whole AutoCut pipeline, step by step, with the commands that run it. The worked example
throughout is **"Every Robot I've Met"** (H4hacks, 2026-10): 48 robots from 2013 to 2025, narrated
in Malayalam in the author's cloned voice, exported in 4K, published on YouTube
(https://www.youtube.com/watch?v=7FYPOxDYWBc), with a 34-second Instagram Reel trailer. Its input
files are in [`examples/every-robot/`](../examples/every-robot/) (no media: the clips stay in
Google Photos and AutoCut's cache).

Every command below is `filmcraft-cli` (`cargo build --release -p filmcraft-cli`, or the copy in
the packaged app, `target/macos-package/bin/filmcraft-cli`). Every flag: `filmcraft-cli help`.
The revision round in more depth (review notes, photos, places, the Malayalam style):
[showcase-runbook.md](showcase-runbook.md). A different kind of video — a maker's build, with his
own takes, a version counter and graphics after the build — is §10. How each piece works:
[crates/autocut/README.md](../crates/autocut/README.md). For a quick vertical Short or Reel of
any clips, without a showcase (`filmcraft-cli quick <folder>`: a template picked and filled from
the scene index and the speech): [quick-templates.md](quick-templates.md).

```text
 album ─► pick ─► analyse ─► plan ─► script ─► voice ─► build (1080p review) ─► QA + privacy
   ▲                                                         │
   └──────────── review notes (next round) ◄─────────────────┘
                                                             ▼
          final (4K) ─► YouTube (private) ─► subtitles ─► publish ─► trailer (Reel) ─► Instagram
```

## 0. Once per machine

| What | Command / where |
|---|---|
| Build | `cargo build --release -p filmcraft-cli` (or `packaging/macos/package.sh --arch aarch64` for the app) |
| Local models (Ollama) | `ollama pull qwen2.5vl:7b` (analysis, picker agent, privacy check, QA) and `ollama pull qwen3:8b` (scripts) |
| Face and plate detectors, and the person and pose models for faces in profile | `filmcraft-cli exec redact.downloadModels wait=true` (4.4 MB + 35.8 MB; `profiles=false` for the detectors alone) |
| Google Photos | `filmcraft-cli photos auth --albums` (needs `client_secret.json` from the Google Cloud project, see `crates/gphotos/README.md`) |
| YouTube | `filmcraft-cli youtube auth --default`: pick the channel (H4hacks) in Google's account chooser |
| Cloned voice (English) | `exec tts.cloneVoice` from the speaker's own clip, with consent; check and improve it with the voice lab (§3) |
| Cloned voice (Malayalam) | `exec tts.downloadCloneModel model=indicf5`, then `exec tts.voiceReference voice=clone:<name>` ([malayalam-narration.md](malayalam-narration.md)) |
| Malayalam speech model (hears every line back) | `exec transcript.downloadModel model=whisper-medium-ml` (3 GB) |
| Malayalam on screen | build with `CRAFT_FONTS_DIR` set (see `CLAUDE.md`); without it, keep burned-in text in English |
| Instagram | `instagram setup --app-id <Instagram app ID>` (you paste the app secret), then `instagram auth` (§9) |
| Shared machine | `filmcraft-cli locks` before driving Chrome or the GPU by hand; `locks hold chrome --minutes 30 --note "…"` while you do |

The example used: voice `clone:mohesh-natural` (VoxCPM2 for English, IndicF5 for Malayalam),
Google Cloud project `autocut-photos`, the channel H4hacks, Instagram @moheshmohan.

## 1. Seed: the album

You curate the material in Google Photos: make an album and add every clip and photo the video
should draw from. The example's album is **"Robot showcase"**; a second album of stock footage of
Dubai fed the outro. Google lets apps read only albums they made, so AutoCut opens Google's picker
and its local screen agent selects the album whole.

## 2. The first cut in one command

For a new video, one command picks the album, analyses it, plans, writes and voices the script,
builds, checks and exports:

```sh
filmcraft-cli showcase --album "Robot showcase" --prompt "a video of every robot I've met since 2013" \
    --voice clone:mohesh-natural --redact faces,plates --style tech --timeline --loudness social \
    --export ~/Movies/AutoCut/robots-v1.mp4 --project ~/Movies/AutoCut/robots.fcproj --save
```

It also writes the analysis (`--report-out report.json`) for later rounds. Review builds stay at
1080p; the approved final is rendered in 4K (§7).

## 3. Step by step (what the robots video went through)

The example took five review rounds. Each round works on a copy of the previous project and plan
(`R=~/Movies/AutoCut`, `P="$R/Robot Showcase v5.fcproj"`).

### 3.1 Pick and analyse

```sh
filmcraft-cli --project "$P" photos pick --album "Robot showcase" --save        # new items, imported
filmcraft-cli --project "$P" analyze robot --items 4841,2630,2631 --out "$R/report-new.json"
```

A local vision model finds the subject in sampled frames, reads its text, names it and keeps the
best moment per clip (cached: a clip is analysed once). Photos are analysed once and held with a
slow pan and zoom aimed at the robot. Example: [`v5/report-robots.json`](../examples/every-robot/v5/report-robots.json).

### 3.2 Where each clip was filmed

Google strips the location from downloads, so the screen agent reads each item's place from the
Google Photos page:

```sh
filmcraft-cli --project "$P" photos locations --album "Robot showcase" --out "$R/locations.json" --save
```

Example: [`v5/album-locations.txt`](../examples/every-robot/v5/album-locations.txt) (Busan, Jumeirah
Beach, Expo 2020…). The places become each entry's lower third.

### 3.3 Look at the clips

```sh
filmcraft-cli --project "$P" sheet --items 2407,2417:0-30 --n 8 --out "$R/sheet.jpg"
```

Choose windows, portrait fits and focus points from the sheets.

### 3.4 The plan

`showcase --report … --plan` prints the entries and the local model's draft script without
building. The author reviews it; review notes go into the plan with `showcase-plan` (shot windows,
`fit: "full"` for portrait clips whose action a crop would cut, `speed` for fast-forwards,
`redact` targets such as a sponsor's logo, `place`, new entries, removals, pronunciations):

```sh
filmcraft-cli showcase-plan --base "$R/plan-v5.json" --patch "$R/notes-v6.json" --project "$P" \
    --places-from-locations "$R/locations.json" --out "$R/plan-v6.json" --lines "$R/script-ml-v6.json"
```

Example plans: [`v5/plan-v5.json`](../examples/every-robot/v5/plan-v5.json) (English) and
[`v5/plan-v5-ml.json`](../examples/every-robot/v5/plan-v5-ml.json) (Malayalam). Lessons kept in
them: GITEX is said "Jai-tex" (`"pronounce"`), robot 12's mist and robot 35's arm need
`fit: "full"`, the DP World logo on the Promobot is a named `redact` target.

### 3.5 The script (Malayalam)

Write natural spoken Malayalam (short sentences, English tech words in Malayalam script, years and
prices said the English way). Draft, lint, keep:

```sh
filmcraft-cli ml-script --plan "$R/plan-v6.json" --out "$R/script-ml-v6.json"      # draft (qwen3:8b)
filmcraft-cli ml-lint "$R/script-ml-v6.json" --plan "$R/plan-v6.json"              # written-sounding lines, lost facts
filmcraft-cli ml-script --plan "$R/plan-v6.json" --remember "$R/script-ml-v6.json" # once approved
```

Example: [`v5/script-ml-v6.json`](../examples/every-robot/v5/script-ml-v6.json), side by side with
the English in [`v5/malayalam-script-v6.md`](../examples/every-robot/v5/malayalam-script-v6.md).
Style rules: [malayalam-style.md](malayalam-style.md).

### 3.6 The voice

Spoken lines are cached, so a rebuild voices only changed lines. Every Malayalam line is heard back
by the Malayalam speech model and read again until clear (the strict check). To find a better
reference in the speaker's own clips and compare by ear:

```sh
filmcraft-cli voice refs --voice clone:mohesh-natural ~/Library/Application\ Support/FilmCraft/google-photos/cache
filmcraft-cli voice audition --voice clone:mohesh-natural --report refs.json --top 3
filmcraft-cli voice promote --voice clone:mohesh-natural --report audition.json voxcpm-1   # the one chosen
filmcraft-cli voice versions --voice clone:mohesh-natural                                   # rollback r0 if needed
```

The example's English reference is r1 (a clip the author confirmed by ear); the best way to a
better clone is a purpose-made recording: [voice-recording.md](voice-recording.md).

## 4. Build the review cut (1080p)

```sh
filmcraft-cli --project "$P" showcase --report "$R/report-robots.json" --script-in "$R/plan-v5-ml.json" \
    --title "Every Robot I've Met" --subtitle "2013 – 2025" --voice clone:mohesh-natural --language ml \
    --strict-script --mood uplifting --style tech --timeline --loudness social \
    --soundfont generaluser-gs --redact faces,plates --export "$R/preview-ml.mp4" --save > "$R/build.json"
```

`--languages en,ml --lines-ml script.json` builds English and Malayalam in one run. Keep the JSON
the build prints (`> build.json`): the trailer reads it. The mix is speech first for small speakers
by default (`--mix gentle` keeps a cloned voice closer to its speaker; `--mix cinematic` is the
full-range balance). GPU rendering is on by default (`--gpu-rendering off` on `export` for the CPU).

## 5. QA and privacy

Beside every export: `<out>.qa.json` and `<out>.qa.png` (joins, black frames, black sides around
transitions, loudness, the mix on small speakers, the robot in every shot), `<out>.privacy.png`
(the masks), `<out>.narration-check.json` (Malayalam lines as heard).

- **Privacy.** Faces and plates are blurred, robots left sharp. A face inside a robot's box is
  asked about (a visitor next to Ameca is still a visitor). The final picture is checked for faces
  no mask covers (`sequence.checkPrivacy`, every 0.5 s and at each shot start); an unblurred person
  is a QA error, and uploads refuse a file whose QA lists one (`--allow-privacy-findings` only after
  you have looked: the example's reel flagged Ameca's own chin).
- **Look at frames** of any moment of an export: `filmcraft-cli frames "$R/preview-ml.mp4" --at 601,603,605 --out "$R/check.jpg"`.
- **Listen** to every Malayalam line with `"passed": false` in the narration check.

## 6. Review and the next round

Upload the review cut to the album, watch it (on the TV too: the example's mix was made for that),
and write the notes:

```sh
filmcraft-cli photos upload "$R/preview-ml.mp4" --album "Every Robot I've Met"
```

Turn the notes into `notes-v7.json` and run §3.4–§5 again; the analysis, narration and redaction
caches make a round cost only what changed. The example's notes became
[`v5/build_plan_v5.py`](../examples/every-robot/v5/build_plan_v5.py) (now a `showcase-plan` notes file).

## 7. The final (4K)

The same build with `--resolution 4k` (3840×2160, titles and graphics drawn at that size, the
YouTube 2160p preset). The example's final, Malayalam only:

```sh
filmcraft-cli showcase --report "$R/report-robots.json" --script-in "$R/plan-v5-ml.json" --project "$P" --save \
    --resolution 4k --mix speech --title "Every Robot I've Met" --subtitle "2013 – 2025" \
    --voice clone:mohesh-natural --language ml --strict-script --mood uplifting --style tech --timeline \
    --loudness social --soundfont generaluser-gs --redact faces,plates \
    --export "$R/Every Robot I've Met - Final (Malayalam, 4K, clear mix).mp4" > "$R/build-final.json"
```

The example's 12-minute final rendered at about 24 fps on an M3 Max (15 minutes), before the
2026-10-21 speed-ups (GPU compositing by default, two hardware encoders at 4K, faster VP9
decoding: [performance.md](performance.md)); the redaction and narration caches make a rebuild
mostly rendering. Its report:
[`v5/build-final2-ml.json`](../examples/every-robot/v5/build-final2-ml.json); its QA and chapters:
[`final/`](../examples/every-robot/final/).

## 8. YouTube

```sh
filmcraft-cli youtube upload "$R/Every Robot I've Met - Final (Malayalam, 4K, clear mix).mp4" \
    --title "…" --description "…" --tags "robots,robotics,…" --category 28 --privacy private
```

Chapters (`<out>.chapters.txt`) are appended to the description and `<out>.thumbnail.jpg` becomes
the thumbnail. The example's title, description and tags: [`v5/youtube-final.json`](../examples/every-robot/v5/youtube-final.json).
Then, in YouTube Studio (by hand, or an agent in the browser holding the `chrome` lock):

- **Subtitles.** Embedded captions (CEA-608/708) cannot carry Malayalam, so AutoCut skips them for
  non-Latin text: set the video's language, then Subtitles ▸ Upload file ▸ With timing ▸ the
  `<out>.srt` ([`final/final-ml-4k.srt`](../examples/every-robot/final/final-ml-4k.srt)).
- **Don't mark it as synthetic/altered content** unless it is (the author's choice for this video).
- **End screen, publish** — the author's call.
- **After publishing**, a fix to the picture can only be done with Studio's Editor (Blur: face blur
  and custom blur are permanent once applied), or by uploading a new video.

## 9. Trailer for Instagram (Reel)

A 34-second 9:16 trailer from the finished show: the most striking robot under "48 robots · 2013 →
2025" with the show's first narration sentence, robots in date order cut on the beat, labels inside
Instagram's safe zone, an end card. Work on a copy of the project:

```sh
cp "$P" "$R/trailer.fcproj"
filmcraft-cli --project "$R/trailer.fcproj" trailer --sequence 9755 \
    --plan "$R/plan-v5-ml.json" --report "$R/build-final.json" \
    --hook "Ameca, again" --hook-at 41.6 --include "Boston Dynamics Spot" \
    --at "My first robot=1347@8.6" --at "Reception robot=42@6.0" --at "Micropolis M02=49.3" \
    --exclude "Robot arms with OnRobot grippers" \
    --logo "$R/branding/h4hacks-logo.png" --channel "@H4hacks" \
    --link "https://www.youtube.com/watch?v=7FYPOxDYWBc" \
    --export "$R/Every Robot I've Met - Reel trailer v3.mp4" --save
```

Quote every URL in zsh (`?` is a glob there). Beside the video: `.cover.jpg`, `.caption.txt`,
`.frames.jpg`, `.qa.json`. Look at the frames and the QA's privacy findings before posting (the
example dropped the OnRobot shot: two visitors in profile). The caption used:
[`reel/caption.txt`](../examples/every-robot/reel/caption.txt) — on Instagram, write the YouTube
channel's name without "@" (an @ tags whatever Instagram account has that name).

Check it against Instagram's spec, then publish:

```sh
filmcraft-cli publish check instagramReel "$R/Every Robot I've Met - Reel trailer v3.mp4"
filmcraft-cli instagram publish "$R/Every Robot I've Met - Reel trailer v3.mp4" --caption "$(cat caption.txt)" --cover-at 1.4
```

**Instagram's API and AutoCut's connection (2026-10):** AutoCut signs in with *Instagram Login*
(`instagram setup` + `instagram auth`; Meta only accepts an https redirect, so the business-login
redirect is `https://localhost:8714/` and you paste the address the browser lands on back into the
terminal). With Instagram Login, Meta takes a Reel only from a **public URL** it downloads
(`--video-url https://…`); direct upload from the file is for apps using *Facebook Login for
Business* (an Instagram account linked to a Facebook Page). Until AutoCut stages files at a
temporary link or signs in with Facebook Login, post the Reel from the phone (the file is in the
Google Photos album) or pass `--video-url`.

## 10. Recipe: a maker's build video ("Akku Tesla")

Another kind of video, built with the same tools on 2026-10-24: **"Akku Tesla"**, a toy Tesla
Model 3 given working lights by an ESP32 board, in three versions (a web page, car-shaped
controls, a touch remote), 5:51, Malayalam. What makes it different from the robots show: the
maker speaks in some clips (his own takes are kept, not narrated over), the cloned voice narrates
the B-roll, the counter counts *versions* of one car, and graphics, callouts and effects go on
after the build. The files of that build are in `~/Movies/AutoCut/Akku Tesla/`.

**The project and the album.** One album holds every clip and photo ("akku tesla toy", 52 items).

```sh
D="$HOME/Movies/AutoCut/Akku Tesla"; P="$D/Akku Tesla.fcproj"
filmcraft-cli new "$P"                                   # or let --project make it
filmcraft-cli --project "$P" photos pick --album "akku tesla toy" --save
```

The picker answers the album's exact title after a few seconds; the agent waits for it (up to
30 s) and, with Chrome ▸ View ▸ Developer ▸ Allow JavaScript from Apple Events on, ticks each tile
in the page ([showcase-runbook.md §1](showcase-runbook.md#1-the-new-material-from-the-album)).

**What is in the clips.** Who speaks, in which language, where the words and pauses are, and a
scene index with contact sheets to choose windows from:

```sh
filmcraft-cli --project "$P" --save exec media.understandSpeech '{"language": "auto", "wait": true}' > "$D/speech-auto.json"
filmcraft-cli --project "$P" --save exec media.understandSpeech '{"items": [50, 51, 52], "language": "ml", "wait": true}' > "$D/speech-ml.json"
filmcraft-cli --project "$P" --save scenes analyze --detail full --sheet "$D/scenes" > "$D/scenes.json"
filmcraft-cli --project "$P" sheet --items 50:0-60,52 --n 8 --out "$D/sheets/build.jpg"
```

**The plan.** No analysis is needed when you choose the shots yourself: an empty report names the
subject, and the plan's entries carry the shots (`showcase-plan --patch` with `add` entries makes
them from item numbers, [showcase-runbook.md](showcase-runbook.md)). The Akku Tesla plan
(`plan-v1-ml.json`) has twelve entries:

- **Scene entries** for everything that is not a version — the opening montage, the maker's
  intro, the board, opening the toy, the pins, the LEDs, the first code, the finale:
  `{"label": "XIAO ESP32-C3", "kind": "scene", "event": "XIAO ESP32-C3", "caption": "Seeed Studio
  XIAO ESP32-C3", "shots": [...]}`. A scene is not counted, gets its `caption` instead of a name
  lower third, and opens its `event`'s chapter.
- **The maker's own takes**: a shot whose sound is the maker talking keeps it, raised with
  `"volume": 9` (dB over the level the mix gives the shots; 7 for shorter remarks), and its entry's
  script line is `""` — no narration over him. Every other shot is `"mute": true`.
- **Cloned narration** for the other entries: the Malayalam lines in `script.lines` (spoken,
  checked line by line by the Malayalam speech model; listen to samples first,
  [malayalam-narration.md](malayalam-narration.md)), an empty `intro` when the show opens on the
  maker's own words, and the `outro`'s call to action.
- **The version counter**: the three version entries have `"kind": "toy car"`, and the plan says
  `"counter": "version"`, so the counter reads "VERSION #1/#2/#3", the closing card "3 VERSIONS",
  while QA looks for the subject the report names (`{"subject": "toy car", "clips": []}`); or
  `"subjectCheck": false` where no frame shows the subject. (Built with `version` as the subject,
  QA warned "no version found" for each of the ten version shots.)
- **Spec chips** per version in `script.chips` (`"9": ["XIAO ESP32-C3", "4 LEDs", "Web page"]`), and
  photos as shots (`"photo": true, "hold": 5`).

```sh
echo '{"subject": "toy car", "clips": []}' > "$D/report-empty.json"
filmcraft-cli --project "$P" --save showcase --report "$D/report-empty.json" --script-in "$D/plan-v1-ml.json" \
    --title "AKKU TESLA" --subtitle "ESP32 · LEDs · Wi-Fi" --voice clone:mohesh-natural --language ml \
    --strict-script --mood uplifting --style tech --mix speech --loudness social --soundfont generaluser-gs \
    --enhance-voice --stabilize --match-color --export "$D/Akku Tesla - review 1 (1080p).mp4" --qa > "$D/build1.json"
```

`--enhance-voice` cleans the maker's takes (never the narration); `--stabilize` and
`--match-color` suit phone footage of a workbench.

**After the build: the extras.** One `run` file of commands on the built project (keep a copy of
the project first: `cp "$P" "$D/Akku Tesla (before extras).fcproj"`), then export again. The
Akku Tesla extras, with what they needed:

| what | command |
|---|---|
| the maker's takes a little lower in the final mix | `{"id": "clip.audioGain", "params": {"clips": [356, 382], "mode": "adjust", "db": -5}}` |
| the music ducks under his voice | `{"id": "track.setAutoDuck", "params": {"track": "A3", "under": ["A1"], "amount": 10, "attack": 0.2, "release": 0.6}}` |
| callouts that follow the board, the phone page, the remote | `{"id": "track.label", "params": {"clip": 392, "box": [0.42, 0.24, 0.15, 0.38], "at": 79.4, "text": "XIAO ESP32-C3", "style": "callout", "accent": "#29E0FF", "wait": true}}` (`"style": "tag"` for a short one); `track.spotlight` dims around one |
| effects on the opening shots and the hero shot | `{"id": "effects.apply", "params": {"clips": [369], "effect": "cr_light_sweep"}}` (also `cr_glitch`, `cr_chromatic_aberration`, `cr_tech_hud`) |
| a speed ramp that keeps the shot's length | `{"id": "clip.speedRamp", "params": {"clips": [472], "preset": "hero", "fit": "length"}}` — `fit: "length"` plays more of the source (or eases the curve) instead of halving the clip and leaving a gap; `"preset": "none"` puts the speed back |
| the finale pulses on the beat | `{"id": "sequence.beatPulse", "params": {"range": {"start": 324, "end": 339}, "effect": "zoomPulse", "strength": 0.6}}` |
| pin labels, a code card, a diagram | `{"id": "graphics.newText", "params": {"text": "D0  ◀  LEFT BLINKER", "position": [1220, 300], "size": 60, "font": "DIN Alternate", "fontStyle": "Bold", "seconds": 6.1, "track": "auto", "time": 46992960000000, "newClip": true}}`, then `graphics.set` (`fill_color`, `background`), `graphics.animateText` (`slideLeft` by line, `typewriter`), `graphics.textStyle` (`glow`) — `"track": "auto"` puts labels that overlap in time on tracks of their own; a `newText` that fails leaves nothing for the styling after it to restyle |
| captions that read over a white car | `{"id": "captions.setAnimatedStyle", "params": {"track": "C1", "preset": "box", "options": {"wordsPerPage": 5, "activeColor": "#29E0FF", "keywords": ["വൈഫൈ"], "position": "bottom", "margin": 0.28, "panel": true}}}` |
| the loudness again, last | `{"id": "audio.normalizeLoudness", "params": {"target": "social", "truePeak": -1.0}}` |

```sh
cp "$P" "$D/Akku Tesla (before extras).fcproj"
filmcraft-cli --project "$P" --save --keep-going run "$D/post2.jsonl" > "$D/post2.out"
filmcraft-cli --project "$P" export "$D/Akku Tesla - review 2 (1080p).mp4" --preset "YouTube 1080p Full HD"
filmcraft-cli frames "$D/Akku Tesla - review 2 (1080p).mp4" --at 182,188.5,234.7 --out "$D/check3/sheet.jpg"
filmcraft-cli photos upload "$D/Akku Tesla - review 2 (1080p).mp4" --description "Akku Tesla — review 2 (1080p)"
```

Clip ids come from the build's output (`layout`: each entry's shots and their clips) or
`inspect sequence`. Read `post2.out` line by line (`"ok": false` lines say what failed), and look
at frames of the moments you changed. The upload goes to the library (into an album only one that
FilmCraft made: `--album`, [§6](#6-review-and-the-next-round)). Review at 1080p, the final in 4K
(§7).

## 11. Where everything is

| What | Where |
|---|---|
| Projects, exports, QA, subtitles | `~/Movies/AutoCut/` (`Robot Showcase v5.fcproj`, `Every Robot I've Met - …`) |
| Clip cache (read-only) | `~/Library/Application Support/FilmCraft/google-photos/cache/` |
| Caches (analysis, narration, redaction) | `~/Library/Application Support/FilmCraft/{autocut/analysis-cache,narration-cache,redact-cache}` |
| Voices and their versions | `~/Library/Application Support/FilmCraft/voices/` |
| The example's inputs | [`examples/every-robot/`](../examples/every-robot/) |
| Revision rounds in depth | [showcase-runbook.md](showcase-runbook.md) |
| Malayalam narration | [malayalam-narration.md](malayalam-narration.md), [malayalam-style.md](malayalam-style.md) |
| GPU and performance | [gpu.md](gpu.md), [performance.md](performance.md) |

## Gotchas the example ran into

- ffmpeg is not needed (nor installed): `frames` and `sheet` make the stills, `filmcraft-cli
  export --format aac` the audio-only copies.
- Review at 1080p, render the approved final in 4K.
- A file sent to a phone over Remote Control is capped at 30 MB: share review videos through the
  Google Photos album.
- Two heavy jobs at once (a 4K render and the full test gate, say) slow each other a lot; run them
  one at a time.
- The narration mix that sounds best on headphones is not the one that is clear on a TV at low
  volume: listen on the TV (the speech-first mix is the default for that reason).
- (Akku Tesla) Google's picker shows an album's search results only after a few seconds; the album
  pick waits up to 30 s, and with Chrome's "Allow JavaScript from Apple Events" ticks every tile
  in the page instead of by eyes.
- (Akku Tesla) A counter that counts something no frame shows (versions) belongs in the plan's
  `"counter"`, not in the subject: QA looks for the subject in every shot.
- (Akku Tesla) Graphics that overlap in time need tracks of their own: `"track": "auto"`. A
  `graphics.newText` that fails no longer lets the styling after it restyle the previous title,
  but read the run's output for `"ok": false` all the same.
- (Akku Tesla) `clip.speedRamp` with `ripple: false` used to halve a shot (a mean speed of 2×) and
  leave a gap; `fit: "length"` keeps its length, and `preset: "none"` puts its speed back.
- (Akku Tesla) Animated captions over white footage need `"panel": true`.
