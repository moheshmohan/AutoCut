# filmcraft-autocut

AutoCut: prompt-driven automated editing on top of the FilmCraft engine. Ask in plain language
("remove the silences and add captions", "cut 0:10 to 0:25", "make a 60-second highlight reel
about the launch") and it plans and runs ordinary engine commands — every edit lands in History
and undoes like a human's.

## How it works

```
prompt ──► classify (local LLM, Ollama, JSON mode)
             │  routine intent + confident ──► deterministic planner ──► validated steps ──► execute
             │  custom / low confidence ────► needsSupervisor report: the document digest
             │                                  (and captioned frames, local VLM, on request)
             │                                  goes back to the caller — the MCP client
             │                                  (e.g. Claude) — which plans the edit itself
             │                                  with command_list / command_run / command_batch
             ▼
        validate against filmcraft-engine's command registry + an edit-family allowlist
             ▼
        execute through the Executor trait ──► headless Session, or the live app (bridge)
```

- **Local first.** Routine intents never touch the network: phi4-mini classifies, the planner
  emits commands like `transcript.removePauses` / `sequence.extract` / `transcript.createCaptions`.
- **Supervised over MCP, no API keys.** Complex prompts are handed back with `needsSupervisor`
  plus the compact document digest (`digest.rs`) and, when asked (`frames: true` / `--frames`),
  VLM-captioned frames (`describe.rs`, moondream/SmolVLM class). The supervisor is whoever is
  driving the editor over MCP — Claude, via `command_list` / `command_run` / `command_batch` —
  planning with the same tools a person's UI uses.
- **Nothing exotic.** AutoCut never edits project state directly; it only runs engine commands,
  so automation, the UI and agents share one undoable truth.

## Intents

| Intent | Commands it plans |
|---|---|
| `remove_silence` | `transcript.generate`? → `transcript.removePauses` |
| `remove_fillers` | `transcript.generate`? → `transcript.removeFillers` |
| `add_captions` | `transcript.generate`? → `transcript.createCaptions` (characters per line capped at what one rendered line holds on the sequence's frame — the TikTok style is one line of at most 24) |
| `cut_range` | `markers.markIn/markOut` → `sequence.extract`/`lift` → `markers.clearInOut` |
| `keep_range` | tail-then-head `sequence.extract` (order keeps times valid) |
| `highlight_reel` | time cuts (`markers.markIn/markOut` + `sequence.extract`, last-first) on the complement of the best stretches — each kept passage keeps a lead-in and a tail from the silence around it, passages under a second apart are joined, no sliver under 0.4 s is cut, the silent head and tail go too — then `trim.applyDefaultTransition {frames: 6}` puts a short dissolve and crossfade on every cut, so nothing jumps |
| `enhance_voice` | `essentialSound.setType` + preset (Clean Up Noisy Dialogue / Podcast Voice / …) + repair amounts + `autoMatch` loudness (+ `generateDucking`) |
| `add_voice_over` | `tts.create` (built-in voice; Kokoro neural voices behind `neural-voices`); `fit` previews and paces the delivery to cover the sequence |
| `export_video` | `file.exportMedia` with the destination's built-in preset: youtube, youtube4k, tiktok/reels/vertical (9:16), vimeo, prores, gif, audio, aac |
| `slow_motion` | `clip.speedDuration` with `interpolation: opticalFlow` — motion-compensated new frames (CapCut-style smooth slow-mo, the engine's own Lucas-Kanade flow), rippling to make room |
| `showcase` | `autocut.makeShowcase {subject, since?, wait}` — the showcase job (below): the clips are analysed and the plan written, nothing is built until it is reviewed. Prompts like "make a video of every robot since 2020" or "a montage of all my vintage cars" are recognised without the model (`intent::showcase_request`), and need no open sequence |
| `custom` | nobody: the report comes back with `needsSupervisor` |

Highlight scoring (`highlight.rs`) is classical: sentence/pause segmentation, topic-keyword and
salience scoring, duration shaping — no model needed to pick the cuts.

Transcription over the bridge can outlast the app's 60 s reply limit: a reply timeout from
`transcript.generate` means "still transcribing", and the runner polls for the words (up to 30
minutes) instead of planning without them.

## Showcases: analyse, build, check

A showcase is "every robot I have filmed, in order": many clips, one narrated, counted video.

1. **`analyze`** (`analyze.rs`): for each video item, frames are sampled through the engine
   (`media.frame`) and a grounding vision model (Qwen2.5-VL, `AUTOCUT_ANALYSIS_MODEL`) is asked
   frame by frame to *locate* the subject — a box or none. A box says whether, how big and
   where, so the best moment is where the subject is largest. On the two best frames, fetched
   larger (1280 px), the model then transcribes the legible text, sorted by place: `on_subject`
   (printed, painted or stuck on the robot: "ARDET", "Swiss-Mile", "Go1", "temi") and `nearby`
   (signs, booths, shop windows, and anything on a screen: "ETH zürich RSL", "QUOKKA", a
   sponsor's "DP WORLD" on a tablet). Lines are cleaned and capped (12 per place, 60 characters
   each) and kept in the report as `text` (absent in older reports, which still load). One more
   request per clip names and describes it, with that text in the prompt; a proper name in the
   label survives (`sanitize_label_with`) only if it is spelled out — in any case, spacing or
   punctuation — by text read on the robot, or by a sign that introduces one ("Ameca AI
   Interaction Guide", "Meet …", "… Robot"), or else is a known robot or maker (`KNOWN_NAMES`,
   the fallback allowlist), since a 7B model invents brands ("Quokka robot" from the kiosk
   behind it). A shop, company or sponsor name on a sign or screen may go in the description,
   never in the label. Capture dates come from `media.captureDate` (Google Photos picker
   sidecars, or the file's own metadata). `--relabel earlier.json` reads the text and describes
   again from the earlier report's best moments, without locating again.

   **Analysed once** (`analysis_cache.rs`): each clip's result is kept in
   `<data dir>/autocut/analysis-cache/`, keyed by what the file is — its Google Photos media id
   (the picker's sidecar, or the download-cache folder it sits in), else its size and the
   engine's fingerprint of its first and last MiB, so a moved or renamed file is the same clip —
   plus the subject, the vision model, `--frames`/`--screen` and a version bumped with the
   analysis. Going over a library again re-analyses nothing it has seen (`cached` in the output
   and the job summary counts the clips that came from the cache); `--fresh` (`fresh: true` for
   the jobs and `showcase_analyze`) analyses every clip again. Only an analysis in which every
   frame was read and every model call answered is kept; entries are written atomically, capped
   (20 000, the oldest go first), and a damaged one is ignored.

   **Photos** (still image items: JPEG, HEIC/HEIF, PNG…) are analysed too, from their one
   picture: located once (at 640 px), read once (1280 px), described as a photo ("This photo
   shows a robot"); the report marks them `photo: true` and keeps the subject's box where it is
   largest as `subject` (`[x, y, w, h]`, 0..1; kept for clips too), and a photo's EXIF position as
   `gps` when the file has one. They are cached like clips (one frame, no screening, in the key).

   Last, an **identity pass** (`identity.rs`) tells which clips show the same robot, since labels
   alone miscount: one robot filmed on three days of an expo gets three different generic labels.
   Candidate pairs — the same kind, filmed within 14 days, not already sharing a specific label —
   are shown to the vision model as one image, the two clips' best frames side by side; it
   describes both robots, then answers `{"same": bool, "confidence": 0..1}` (asked plainly, a 7B
   model says yes to nearly any pair). Neighbours in time are asked first, pairs already joined
   through others are skipped, and the number of comparisons is capped (`--compare N`, default
   150; 0 joins by name only). Pairs it is sure of (confidence ≥ 0.7), and clips with the same
   specific label within a week, are joined (union-find) into identities: each clip that shows
   the subject gets `identity` (1, 2, … in time order) in the report; older reports, without it,
   still load.
2. **`showcase`** (`showcase.rs`): clips that show the subject are grouped into entries, each
   robot counted once: clips with the same `identity` are one robot (across days too), and
   different identities never merge; in reports without identities, by label as before — the
   same specific name within a week (an expo's other days), or the same day with a generic
   label of the same kind. An entry shows up to three shots, the clips where the robot is
   longest in view (length × coverage), in time order — its videos, or its photos when it was
   only photographed ([Photos](#photos-a-slow-pan-and-zoom-towards-the-robot)) — under the most
   specific label among its clips (a name read on the robot first), dated when it was first
   filmed. The local LLM
   (`AUTOCUT_SCRIPT_MODEL`, falling back to `AUTOCUT_LLM_MODEL`) writes the voice-over from the
   facts, a few entries per request (template lines fill anything it does not deliver, and a
   copied item heading is dropped so it is not read out). The build runs ordinary engine
   commands into a new 16:9 sequence, every cut on a frame:
   - **cold open**: a montage of robots from across the years, cut on the music's beat, the
     title over it, a riser into an impact;
   - **per entry**: the narration (any voice — `clone:<name>` included), then a few seconds of
     the robot's own picture and sound before the next; shots cut where the analysis saw the
     robot (a portrait or 4:3 clip fills the 16:9 frame, cropped on the robot's centre from the
     analysis with a slow 6 % tilt across it, so a tall robot is seen whole and the video never
     looks vertical; `--no-fill` shows it whole over a blurred, dimmed copy instead — the pillars,
     which stay under the picture through its transitions, reaching past the cut as far as the
     transition does, cross-dissolving into the next picture's pillars, and dipping to black with
     the picture into a new year (the dip goes on V1, the pictures fading out and in over it,
     since a dip's black on V2 would cover the pillars at once); slowed only
     when the words would outrun the footage, its sound then slowed with it at its own pitch —
     Speed/Duration's Maintain Audio Pitch), each shot's sound faded in and crossfaded; a
     name/date lower third; a dramatic transition from a rotating set (cross zoom, whip, flash,
     push, glitch, light leak, zoom blur; dip to black at a new year) with a whoosh; the year
     as a title card with an impact;
   - **polish**: every shot pushes in slowly — Motion scale keyframes from as fitted to 1.5 %
     larger a second (6 % at most), eased out of the first and into the last, in media time
     so a slowed shot keeps the whole move; a filled shot's tilt is Motion position keyframes
     eased the same way (with `--no-fill`, the blurred copy behind it stays still). Lower thirds slide in from the left (3 % of the width, 0.4 s, decelerating) while
     fading in (0.25 s), and leave the same way (Motion position and Opacity keyframes on the
     graphic clip). `--stabilize` (`Options::stabilize`) puts Warp Stabilizer
     (`warp_stabilizer`: smooth motion, position/scale/rotation, crop and auto-scale) on every
     shot but the blurred outro — the analysis has no camera-motion measure, so all footage is
     treated as handheld; the stabilizer decodes each shot once at 480 px wide and tracks it
     on first render (about the shot's length in decoding, cached for the export). The engine
     has no automatic white balance or tone; its colour tool is Color Match
     (`lumetri.applyMatch`), so `--match-color` (`Options::match_color`) matches an entry's
     other clips to the middle of its first full-frame shot (an entry with only portrait
     shots is left alone: their reference frame would hold the blurred copy) — one robot from
     different cameras or light looks like one scene; entries are not matched to each other;
   - `graphics.counter` across all entries; an outro with the total over the latest robot;
   - **a target length** (`--target-minutes`, `Options::target_minutes`): the time left after
     the intro, the entries placed so far and the outro (estimated from its words) is shared
     out over the entries still to come — one breathing time after each entry's words, found
     by bisection, each entry held between what its words need (lead, words, 0.8 s; at least
     3 s) and how long its robot is in view in its shots (`Shot::spans`, at most 45 s); the
     current entry's words are measured, the later ones' estimated from their word count. A
     longer entry shows more and longer shots (time a shot cannot fill with its robot in view
     goes to the next shot), a shorter one fewer. Nothing is ever padded with footage where the
     robot is out of view or with held frames, so a target beyond the footage comes out
     shorter: `seconds` in the output is what it came to, next to `target_seconds`;
   - **the music and its beat**: a generated bed (`audio.generate` at `--bpm`: arranged to the
     show's exact length, its energy building through the intro, high through the entries and
     settling for the outro, scaled to the mood's own energy; it ends on a bar with its final
     chord ringing out, so it is not faded; its exact beat map is the grid the cuts land on), or with
     `--music` (`Options::music`) the user's own track: imported, its tempo and beats measured
     by the engine (`media.beats`, the beat tracker of `filmcraft_audio_dsp::remix`), then
     laid from 0 under the whole show — cut to length and faded out when it is longer,
     remixed longer at its own beats (`clip.remix`, joints where the music is most alike) when
     it is shorter, else looped. The cuts land on the beat: the montage cuts every two beats
     (four when it is quick), the intro ends on the first beat after its words, the title's
     impact hits a beat, and each entry ends on the nearest beat within 0.4 s (half a beat at
     most) — its last shot a little shorter (never under what its narration needs) or a
     little longer only while the robot stays in view — so the next entry, its transition
     and year card start on one; the last shot is placed to the beat's own frame. `music` in
     the output says how the track was fitted and how many cuts are within a frame of a beat
     as it finally plays (computed from the remix's pieces when remixed);
   - **the mix** ([`mix.rs`](src/mix.rs)): the voice matched to -16 LUFS (dialogue), the
     shots' sound (ambience, -27) and the music (-24, generated or the user's) ducked under it,
     the effects at -26, and a hard limiter at -1 dBFS on the mix (or the loudness target's).
     By default the mix is **speech first, for small speakers** (`--mix speech`, below): the
     narration EQ'd as far as its measured tone needs, de-essed, compressed and peak-limited on
     its track, the other stems high-passed, the music ducked 18 dB and dipped around 2.5 kHz
     under the voice; `--mix cinematic` keeps the earlier full-range balance (music ducked
     15 dB, effects at -23, the transitions' sounds at -6 dB);
   - **subtitles** (`subtitles.rs`): each narration line as written, split into cues of at most
     two 42-character lines, 1–7 s each, at sentence ends first and then clauses (a small
     dynamic programme over the words), timed in proportion to the characters within the line's
     clip (a cue too short to read stays up into the pause after it, never into the next line);
     on a `Subtitles` caption track of the sequence (shown in the app, not burned in);
   - optionally the YouTube 1080p export (composited on the GPU, the default since 2026-10-21:
     the redaction blur, the blurred fills under portraits, the transitions and the keyframes run
     there, the CPU only for what the GPU does not draw — [docs/gpu.md](../../docs/gpu.md)), with
     `<out>.chapters.txt` (intro, each year, outro —
     YouTube's rules: from 0:00, at least three, none under 10 s) for the description, the
     subtitles as `<out>.srt` and `<out>.vtt` (`captions.export`) to upload with it, and a
     **thumbnail**, `<out>.thumbnail.jpg` (1280×720): the sampled moment where a robot is
     largest in the analysis (`Shot::largest`; a well-known robot or maker —
     `analyze::famous`, the `KNOWN_NAMES` list — counts double, any other named one a third
     more), filling the picture (a portrait shot over its blurred copy), with the title in
     large white capitals outlined in black (on two balanced lines when long) in the half of
     the picture the robot is not in (low, unless the analysis saw it low in the frame) and
     the count ("45 ROBOTS") in yellow in the opposite corner. It is composed in a sequence of its own, "<title> —
     Thumbnail", left in the project to adjust and export again, and written by
     `file.exportFrame` (`showcase::thumbnail` does it on its own); the show's sequence is
     active again afterwards.

   **Spoken vs shown.** Text-to-speech misreads names that are spelled for the eye ("Zaku II",
   "NodeMCU", "Go1", "DP World"); respelling them in the script fixes the voice but puts "Zaku
   Two" in the subtitles. A plan's script keeps the two apart: `pronounce` maps written forms to
   what the voice says (`{"Zaku II": "Zaku Two", "NodeMCU": "Node M C U", "Go1": "Go One"}`;
   whole words, case as written, longest first, applied to every spoken part), and `say` gives
   a part its own spoken version by `"intro"`, `"outro"` or entry number (`{"3": "…"}`). The
   voice reads the spoken text; subtitles (and everything on screen) show the script as written.

   **How each line is said.** The script model writes the lines to be spoken — first person,
   contractions, the when and where worked into a sentence rather than a bare "September 2020,
   at home:" opener — and picks a delivery per line (`conversational`, `warm`, `excited`,
   `amazed`, `curious`, `calm`). A plan's script keeps them in `delivery`, by `"intro"`,
   `"outro"` or entry number (`{"intro": "excited", "7": "amazed"}`), renumbered with their
   entries on review like `say`; a reviewed plan may also choose `dramatic` or `plain`. A part
   without one is spoken `excited` (the intro) or `warm`. The natural cloned voices follow the
   delivery (see `crates/tts/README.md`); other voices read plainly.

   **Every cloned line is heard back.** A cloned voice can garble a line (a VoxCPM reading
   that is fluent noise). After `tts.create`, the line is transcribed (`transcript.generate`,
   then `transcript.inspect {item}`) and read again as another take (`take` 1, 2) when less
   than half of its words come back, the best of three kept; `narration_retakes` counts them,
   and a line no take says well enough is noted. Without a speech model the check is skipped
   (noted once).

   **Malayalam.** A plan's top-level `"language": "ml"` (or `--language ml`) narrates the
   script's Malayalam parts — `lines`, `intro`, `outro` written in Malayalam script — in
   Malayalam: `tts.create` with `language: "ml-IN"`, a cloned voice speaking through IndicF5
   (`crates/tts/README.md`). Parts left in English, `chips` and on-screen labels stay as written.
   The subtitles carry the Malayalam text (the caption track is tagged `ml`). **Every
   Malayalam utterance is heard back** by the engine's strict check (`tts.create` `check:
   "strict"`, the default; `--no-narration-check` or `Options::skip_narration_check` turns it
   off): the Malayalam speech model (`whisper-medium-ml`) hears each utterance, the character
   error rate and the first six syllables of every sentence are scored, and an unclear
   utterance is read again with its own seed, the best of four kept (see
   `crates/tts/README.md`, "The strict check"). The build keeps what was heard line by line in
   `narration_check` (text, heard, `cer`, `startCer`, readings, and per utterance its seed and
   every try), written next to the export as `<out>.narration-check.json`, and notes every line
   never heard clearly. Without the speech model the lines are still read, and that is noted
   once. (Whisper-base, which writes Malayalam in Tamil script, no longer judges Malayalam
   lines: its scores were noise.) A reading far above the voice's pitch is still read again
   as another take.

   **Numbers in Malayalam.** Digits in a Malayalam line are said the way Malayali tech
   creators say them: years, prices and numbers from 100 up in English, in Malayalam script
   ("2021-ൽ" → "ട്വന്റി ട്വന്റി വണ്ണിൽ", "$1,600" → "സിക്സ്റ്റീൻ ഹണ്ട്രഡ് ഡോളർ"), small
   counts in Malayalam words. A plan's top-level `"numbers": "malayalam"` (or `--numbers
   malayalam`) asks for Malayalam words throughout; it goes to `tts.create` as `numbers`. So a
   script can keep its numbers as digits and the subtitles show them as digits. See
   [docs/malayalam-narration.md](../../docs/malayalam-narration.md).

   Narration goes through `tts.create`, which looks in the engine's narration cache
   (`<data dir>/narration-cache`, see `crates/tts/README.md`) before synthesizing: a rebuild with
   the same voice reads only the lines that changed (`narration_cached` in the output counts
   the rest) — with a cloned voice at about 2.6× real time that is minutes per rebuild.
3. **`qa`** (`qa.rs`): the built show is checked by machine — the defects a review by hand found
   in a real build. On by default with `--export` (`--qa` without one, `--no-qa` to skip); the
   findings (time, severity, entry, clip, message) go to `<out>.qa.json`, a contact sheet (one
   rendered frame per entry under its number; the band red for an entry with an error, amber
   for a warning) to `<out>.qa.png`, and both under `qa` in the output.

   | check | finds |
   |---|---|
   | `joins` (`sequence.inspect`) | on V2 (the shots): gaps of a frame or more, gaps holding a frame start (that frame is black), overlaps — errors; clips that do not quite touch (a slowed shot ending between frames) — warnings; on A1 (the shots' sound): neighbours that do not touch, so the fade-in meets silence instead of a crossfade — warnings; time with no clip on any video track — errors (a picture ending before the sound: info) |
   | `black` (`Executor::render`) | frames whose picture is black or nearly (92 % of the frame darker than 5 % luma, so the counter or a lower third over black still counts), sampled every 2 s, at each shot's first frame and at the first frame after it, outside dip-to-black transitions — errors |
   | `sides` (`Executor::render`) | frames where a side of the frame — the left or right pillar, the bar above or below — is black while a picture still shows in the middle (`qa::black_sides`), sampled around every transition (its first frames, either side of the cut, its last frames) and at the first and last frame of each picture shown whole over a blurred copy (a shot with `fit: "full"`, a clip over Auto Reframe's Blur Fill track); only where that side shows something in the frame just before or after the transition or in the middle of the picture shown whole, so letterboxing kept on purpose and titles over black are not findings — errors, with the time and the side |
   | `mix` (`file.exportMedia` to a temporary 32-bit float WAV, metered with `filmcraft-audio-dsp`'s BS.1770 `LoudnessMeter`) | integrated loudness outside −16…−14 LUFS (±0.5 LU) — warning; true peak above −1 dBTP — warning; above 0 dBTP or any sample at full scale — error (clipping, with the time of the first); the median short-term loudness (3 s windows wholly inside narration clips on A2 vs wholly between them; 400 ms windows when lines or pauses are too short) — the voice less than 4 dB louder: warning, not louder at all: error. Then per stem (`audio.analyzeMix`, `qa::mix_findings`), each a warning: the narration (A2) less than 12 LU above the music (A3) while it speaks (median), the Mix's limiter taking more than 2 dB off the peaks, more than 25 % of the mix's energy below 120 Hz (naming the stem that brings most of it), a narration clip whose source is clipped (flat tops). The analysis is `mix_analysis` in the report and its headline numbers are in `loudness` (`speechOverMusicLu`, `speechOverRestLu`, `limiterDb`, `lowShare`, `presenceShare`) |
   | `subject` (`AUTOCUT_ANALYSIS_MODEL`) | the middle frame of every shot, rendered and grounded with the analysis' prompt (`analyze::ground_prompt`) — a shot where the subject is not found: warning; without a reachable model the check is recorded as skipped (model unavailable) |
   | `safe_zones` (`sequence.checkSafeZones`) | a caption, title or lower third reaching into the platform's interface (`Options::platform`; a vertical sequence without one: every platform at once, `vertical`) — warning, with the zone and how much of it is in there; a horizontal sequence without a platform: skipped |

   QA never edits the project and never fails the build: a check that cannot run (no frames
   from a bridge without the UI tools, an export that fails) is recorded as skipped.

```
filmcraft-cli analyze robot --project p.fcproj --out robots.json
filmcraft-cli showcase --report robots.json --title "Every Robot I've Met" --voice clone:me \
    --plan --script-out plan.json            # review the entries and the script…
filmcraft-cli showcase --report robots.json --script-in plan.json --voice clone:me \
    --project p.fcproj --save --export ~/Movies/AutoCut/robots.mp4
                                             # …and robots.qa.json / robots.qa.png, robots.srt /
                                             # robots.vtt and robots.chapters.txt beside it
```

Entries may carry `event` ("GITEX Global 2022, Dubai") and `facts` (checked facts about the
robot): consecutive entries of one event share a chapter — a title card with the event's name,
the event in the lower third and in the YouTube chapters — and the script model weaves one or two
of an entry's facts into its line, told to use those and nothing else.

A reviewed plan may hold `entries` alone (drop, merge or relabel robots, fix a shot's `best`
moment or `seen` times): the model then writes the words for exactly those.

### Per-shot controls

A plan's shot may carry these, all optional ([`shot_edit.rs`](src/shot_edit.rs)); each is checked
before anything is built, and a value that makes no sense (a negative or not-a-number time, `to`
not after `from` or past the clip's end, a speed of 0 or 1000, a window playing more than a minute,
an unknown `fit`, a `focus` outside 0..1, a volume beyond −60…+12 dB) is an `invalid plan:` error
naming the entry and shot.

| shot field | what it does |
|---|---|
| `"from": 12.0, "to": 18.5` | the source window, seconds into the clip, instead of the automatic one ("the part where it sprays the mist"); one end alone anchors the window and the entry picks its length. Never moved onto the beat |
| `"speed": 4` | 0.25–16: the shot plays that fast (`clip.speedDuration`, frames sampled when faster), taking `(to − from) / speed` of the timeline; its own sound is dropped, never fast-forwarded under the narration (slowed shots keep theirs, at its own pitch) |
| `"fit": "full"` / `"fill"` | `full`: the whole picture — a portrait clip at full height in the middle over a blurred (Gaussian Blur, edges repeated), dimmed (45 % over black), 8 % enlarged copy of itself filling the frame, the copy pushing in while the picture stays whole; `fill`: the frame filled (cropped around the subject), whatever `--no-fill` says |
| `"focus": [0.5, 0.3]` | the point (0..1 of the clip, x right, y down) a fill crop keeps in view, instead of the analysis' box: held with a slow tilt, not tracked |
| `"mute": true` | no sound of its own (a robot talking after the narration ends sounds like a garbled voice-over) |
| `"volume": -20` | its own sound that many dB from the level the mix gives the shots (`clip.audioGain` after the mix) |

An entry keeps its shots in order. A shot with both ends takes a fixed length; the entry's other
shots (those anchored by one end, then as many plain ones as the time left calls for) share what
its words still need. When the fixed shots alone are shorter than the words, the shots that may
grow run on, then the clip after the last window, then a fast-forward plays less fast, and a note
says so: the narration always fits. The output's `layout` shows each shot's `source` window,
`speed`, `fixed`, `fit`, `background` (the blurred copy's clip), `focus`, `sound` (its clip on A1,
none when muted or sped) and `volume`.

An entry's `"place": "Busan, South Korea"` goes in its lower third instead of its `event`
("Busan, South Korea · October 2023"; the month alone when the place names the year) and starts
no chapter or card of its own; its `event` still does.

**Places from Google Photos** (`places`). Google strips the location from what its API and
picker hand out, so an entry has no place unless the plan gives one. `filmcraft-cli photos
locations --album "<title>"` reads every album item's place from the Google Photos page
(`crates/screenagent/README.md`) into `[{fileName, date, gmtOffset, place, estimated}]`;
`photos.applyLocations` stores those places with the project's items as their `Location` field
(`Location Estimated: yes` when Google marked it estimated). An entry without a `place` then
takes its first shot's `Location` when the show is built (`places::from_items`, through
`metadata.get`), and `showcase --places-from-locations <file>` fills the plan's entries from the
file itself, matched by the shots' file names (case and a photo or video extension ignored), so
the plan written out (`--plan`, `--script-out`) shows them for review. A place the plan gives is
never replaced.

```json
{"number": 12, "label": "Disinfection robot", "event": "GITEX Global 2022, Dubai",
 "shots": [{"item": 14, "clip_seconds": 13.8, "best": 6.9, "from": 6.0, "to": 11.5},
           {"item": 2417, "clip_seconds": 37.0, "best": 15.5, "from": 2.0, "to": 26.0, "speed": 6},
           {"item": 2422, "clip_seconds": 106.4, "best": 97.1, "fit": "full", "mute": true}]}
```

### Photos: a slow pan and zoom towards the robot

A robot that was only photographed is an entry of its photos (a robot with video is shown in its
video; an entry may still mix them). A photo shot is the plan's shot with `"photo": true`:

```json
{"item": 4977, "name": "IMG_4512.HEIC", "clip_seconds": 20.0, "best": 0.0, "photo": true,
 "subject": [0.55, 0.2, 0.25, 0.5], "hold": 4.0, "move": "in"}
```

| photo field | what it does |
|---|---|
| `"hold": 4.0` | how long it is shown, 1–20 s; without it, its share of the entry's time. When its words need longer, it is held longer (noted), never a second window of it, which would start its move again; it is never slowed, and has no sound |
| `"move": "in"` | `in` (from wide to the robot), `out`, `pan` (across it at a steady zoom), `none`; without it, the other way from the photo before (in after out, out after in; a pan crosses back from the last pan) |
| `"fit": "full"` / `"fill"` | `full`: whole over its blurred copy (the pillars of a portrait clip), the picture breathing at most 5 % towards the robot while the copy pushes in; `fill`: the frame filled, the move aimed at the robot. Without it: a landscape photo fills the frame (the crop keeps three quarters of it or more, as for a 4:3 clip); a portrait or square one fills it when its robot's box fits in what a crop shows (with a margin), else it is whole — a portrait photo of a tall robot, or with no robot found, keeps the whole picture |
| `"focus": [0.6, 0.3]` | the point to aim at instead of the analysis' box (0..1 of the photo) |
| `"subject"` | the analysis' box `[x, y, w, h]` the move ends on (absent: from the largest moment's centre and size, else the middle) |

The move is the engine's `clip.kenBurns` (Clip ▸ Video Options ▸ Ken Burns, for any still): Motion
Position and Scale keyframed from the clip's first frame to its last, eased out of the first and
into the last. The wide end fills the frame centred halfway between the photo's middle and the
robot; the close end is centred on the robot (as far as the photo still covers the frame: no edge
ever shows) and as close as the robot's box filling three quarters of the frame asks, but never
more than 2.5 % closer a second of the hold (at least 4 %, at most 25 %): slow and subtle, and
further on a long hold. A pan travels 2 % of the photo a second along its longest room. The
montage of the cold open takes an entry's video before its photos; photos are left out of
`--match-color`. The privacy pass looks at a photo once (a face found there is blurred for its
whole hold), the contact sheet (`media.contactSheet`, `filmcraft-cli sheet`) shows it as one tile,
and QA does not take a photo's silence for a dip in the shots' sound. The output's `layout` marks
photo shots `photo: true` with their `move`.

Notes (`showcase-plan`, [`plan_patch`](src/plan_patch.rs)) add photo shots and entries of photos:
`{"item": 4977, "hold": 4, "move": "in"}` (an item the project holds as a still is a photo;
`"photo": true` says so without a project) — the plan's analysed shot of it with these over it,
else one aimed at its middle; `from`, `to`, `speed` and `plain` are refused for a photo, and
`hold`/`move` for a video.

### The look: theme, timeline, chips, scenes, transitions, finishing

All optional ([`look.rs`](src/look.rs)): a plan without these fields, and a command without these
flags, builds the show exactly as above. They are top-level fields of the plan (beside `entries`
and `script`; the app's reviewed plan keeps them too), and every one is checked before anything is
built — a field that makes no sense (an unknown theme or field, a colour that is not `#rrggbb`, a
transition id the catalogue does not have, a date that does not exist, chips for an entry that is
not there, years beyond 1900–2200) is an `invalid plan:` error saying which and why.

| plan field | CLI flag | what it does |
|---|---|---|
| `"style": "tech"` or `{"titleFont", "bodyFont", "numberFont", "accent", "text", "panel", "base"}` | `--style tech` (`classic`: today's look) | a theme on the title card, lower thirds, chapter/year cards, spec chips, timeline, counter and outro. `tech`: DIN Condensed titles (in capitals), DIN Alternate numbers, Futura text, electric cyan `#00e5ff`, text `#f4f8fb`, dark translucent panels `#0a1018` at 72 %. An object overrides the `base` theme (`tech` by default): fonts by family name, colours `#rrggbb`, `panel` `#rrggbb`, `#rrggbbaa` or `{"color", "alpha"}`. Families come from `fonts.list` (macOS: the system's, so DIN and Futura are there); one that is not installed is drawn in Inter, noted, and listed in `theme.missing` |
| `"timeline": true`, `"events": [{"name": "GITEX 2022", "from": "2022-10-10", "to": "2022-10-14"}]` | `--timeline` | a thin timeline along the bottom from the first entry to the outro (not over the title or the outro; fading in and out): a tick at each year's start and the years between, event bands (the plan's `events` — dates `YYYY-MM-DD`, `YYYY-MM` or `YYYY` — else each entry `event` from its first date to its last) with their names, and a marker that glides to each entry's date as the entry starts (Motion position keyframes, leaving on the entry's first frame and arriving 0.6 s later, eased out and in); the current year and event are picked out (accent, on a panel). Lower thirds sit a little higher above it |
| `script.chips`: `{"3": ["32.5 kg", "1.6 m/s", "14 kg payload"]}` | | spec chips beside entry 3's lower third (above it when the row would leave the frame): up to three of up to 32 characters, in the number font on a rounded panel, rising and fading in 0.3 s apart from half a second into its narration, leaving together with the lower third; renumbered with their entries on review like `say` |
| entries with `"kind": "scene"` and `"caption": "GITEX 2022 · Dubai World Trade Centre"` | | a scene shot: footage without a robot (a hall, a stage) that opens its event's chapter — its chapter card over it, full frame, a one-line caption (its `caption`, else its label) instead of a name lower third; not counted, never in the thumbnail, no subject check in QA, no words unless its script line has some (an empty line: none, and a scene without words plays 4.5 s with the bed's energy lifted under it and nothing to duck it) |
| `"transitions": true` or `{"within": [ids], "between": [ids], "chapter": [ids]}` | | transitions from the catalogue (`effects.list`), each list in turn: `within` an entry's shots (default none: straight cuts, moved onto the beat), `between` entries (default `cam_punch_in`, `flow_morph`, `gl_crosswarp`, `cam_whip_pan`, `gl_directionalwarp`, `flow_doppler_warp` — Zooms & Camera, Flow & Morph, GL; about a beat long) with a whoosh, into a `chapter` (default `lf_light_leak`, `gl_filmburn`; about two beats) with a riser into the card's impact — the sounds coupled to the transitions (`transitions.addSound`), replacing the whoosh clips; then `transitions.snapToBeats` on the music's beat map puts each midpoint on a beat and makes each a musical length. An empty list means cuts |
| `"loudness": "social"` | `--loudness social\|voice\|broadcast\|<LUFS>` | `audio.normalizeLoudness` on the finished mix (social −14 LUFS, voice −16, broadcast −23) with a −1 dBTP true-peak limiter, instead of the mix's own limiter alone; `loudness` in the output |
| `"mix": "speech"` (default) or `"cinematic"` | `--mix speech\|cinematic` | how the stems are balanced and treated ([`mix.rs`](src/mix.rs), "Mix for small speakers" below); `mix` in the output says what was done (the narration's measured tilt, how much EQ it got, its fader) |
| `"embedCaptions": true` | `--embed-captions` | the export carries the narration's captions as CEA-608/708 closed captions (besides `.srt` / `.vtt`) |
| `"soundfont": "generaluser-gs"` | `--soundfont generaluser-gs\|oscillators\|auto\|<file.sf2>` | the instruments of the generated bed (`audio.generate`'s `soundfont`; Preferences are not changed); one that cannot play is noted and the bed played without it; `music.instruments` in the output |
| `"outro": {"logo", "channel", "handle", "tagline", "cta", "ctaDone", "terminal", "background", "seconds"}` | `--outro-logo f.png --outro-channel C --outro-handle @h --outro-tagline T --outro-cta S --outro-cta-done S --outro-terminal L --outro-seconds 12 --outro-background item[:from[:to]]` (repeatable; the flags go over the plan's fields) | a branded channel outro instead of the count card ([`outro.rs`](src/outro.rs)), below |

The output says what was made: `theme` (fonts as resolved), `timeline` (its scale, bands, the
marker's stops and the clips), `transitions` (id, effect, where, sound), `beat_snap`, `loudness`,
`scenes`, and per entry in `layout` its `lower_third` or `caption` and `chips` clips.

#### The branded outro

`"outro"` (or the `--outro-*` flags) replaces the plain "N ROBOTS and counting" card with the
channel's own, at least `seconds` long (8–30, 12 by default; longer when the spoken outro needs
it), the spoken outro line over it and the generated bed swelling into it instead of settling (a
music file fades out at the end as before; the loudness pass still applies). Every word on it comes
from the plan (a Malayalam cut gives Malayalam fields); a field left out leaves its element out;
it needs a `logo` or a `channel`.

1. **Reveal** (about 4.5 s): the `background` clips (`{"item", "from", "to"}`, in turn; their sound
   left out) blurred and darkened under a near-black purple wash — a gradient when none can be
   placed (noted); the logo (a PNG, imported) on a light badge glowing purple glitches in at the
   centre (held opacity flickers and jitters), scales up past its size and back, a cyan ring
   pulses out and two rings of rainbow halftone dots (the logo's) burst outward and fade; an
   impact lands with it. The `cta` button pops in under it, a cursor glides in and clicks it (a
   ripple), and it turns into `ctaDone`.
2. **End screen** (the rest: at least 5 s, at most 20): the button goes, the logo glides up (a
   whoosh) into a lockup with the `channel` name (title font, capitals, the logo's purple-to-blue)
   and the `handle` (number font, the theme's accent), a ring pulsing as it lands; the `tagline`
   rises in along the bottom and the `terminal` line types itself out under it (a monospaced font
   when one is installed) with a blinking cursor. The last half second fades to black.

YouTube's end screen shows in the last 5–20 s of a video at least 25 s long, with up to four
elements the creator places in YouTube Studio; a video element is about a third of the frame wide
(16:9), the subscribe element a circle about 300 px across on 1920×1080. The layout leaves two
regions clear for the whole end-screen stretch (on 1920×1080, other sizes scaled): **a video tile
left of centre, x 180–900, y 380–785** (720×405, room to enlarge the default tile) and **the
subscribe button right of centre, a 350 px square centred at (1350, 582)**; the lockup stays above
y 360, the tagline and terminal line below y 820. Place the end-screen elements there.

The output's `outro` says what was made: `start`, `end`, `end_screen` (from when to the end the
regions are clear), `regions` (`video`, `subscribe`), `background` (`footage` or `gradient`) and its
clips, and each element (`name`, `clip`, `from`, `to`, and `settled`: where it sits once the end
screen starts). Tests check every element's timing, that each one on screen in the end screen is
inside the frame and clear of both regions, that a rendered frame there has nothing bright in them,
and that QA finds no error in a branded show.

```sh
filmcraft-cli showcase --report robots.json --script-in plan.json --voice clone:me --style tech \
    --outro-logo ~/Movies/AutoCut/branding/h4hacks-logo.png --outro-channel H4hacks \
    --outro-handle @H4hacks --outro-tagline "Hacking, gadgets and robots in English and Malayalam" \
    --outro-cta Subscribe --outro-cta-done Subscribed --outro-terminal "> h4hacks.com" \
    --outro-background 4830:5:25 --outro-seconds 12 --export ~/Movies/AutoCut/robots.mp4
```

```sh
filmcraft-cli showcase --report robots.json --script-in plan.json --voice clone:me \
    --style tech --timeline --loudness social --embed-captions --soundfont generaluser-gs \
    --export ~/Movies/AutoCut/robots.mp4
```

### Mix for small speakers

A TV, a laptop or a phone plays little below about 120 Hz, and words are understood at low volume
from what they have between 1 and 4 kHz. The first finished show, measured with
`audio.analyzeMix` (per stem: band balance, speech over the music while it speaks, the limiters'
gain reduction, clipping), sounded bassy and muffled on a TV because of its narration: the cloned
voice had 37 % of its energy below 120 Hz and 1 % between 1 and 4 kHz, and its peaks made the
Mix's limiter take up to 3.7 dB off the whole mix (4.3 s over 1 dB). The music was not the
problem (the voice already stood 26 LU above it while speaking). So the default mix
([`mix::MixProfile::Speech`](src/mix.rs)) treats each stem on its track, with the engine's own
effects (editable in the Audio Track Mixer):

| stem | track effects | level |
|---|---|---|
| narration (A2) | Parametric EQ: high-pass 80 Hz (24 dB/oct); low shelf −3 dB at 200 Hz and, as far as the voice is dark, up to −3 dB more, −2 dB at 300 Hz, +5 dB at 1.2 kHz, +5 dB at 3 kHz, high shelf +3 dB from 6 kHz (`--mix gentle`: the voice-preserving EQ, below); DeEsser (6 kHz, −10 dB, at most 6 dB); compressor 3:1 from −20 dB, 1 ms attack, 120 ms release, +6 dB; Hard Limiter at −3.5 dB | each line matched to −16 LUFS; the track's fader makes up what the chain took (measured before and after it), so the voice keeps its level against the other stems |
| music (A3) | Parametric EQ: high-pass 40 Hz, low shelf −6 dB at 120 Hz, a band at 2.5 kHz that dips 4 dB while the narration speaks (track automation following the ducking's stretches) | −24 LUFS, ducked 18 dB under the narration |
| shots' sound (A1) | Parametric EQ: high-pass 100 Hz, low shelf −3 dB at 150 Hz | −27 LUFS, ducked 15 dB |
| effects (A4, A5) | Parametric EQ: high-pass 120 Hz (the impacts' sub booms out) | −26 LUFS; the transitions' own sounds at −10 dB |

How much EQ the narration gets is measured: before treating it the mix renders the narration's
track alone (`audio.analyzeMix {"only": ["A2"]}`) and takes its **tilt**, the mean level of its
1, 2 and 4 kHz octaves minus that of its 125 and 250 Hz ones. The cloned voices (IndicF5's
Malayalam and the English clone) measure −18 dB; a typical voice about −10. The EQ's cuts and
lifts move a voice's tilt by 8.8 dB, so a voice at −17.8 dB or darker gets all of them, one at the
target (−9 dB, a touch brighter than typical) or brighter gets only the high-pass and half the low
shelf, and one in between its share (`mix.voice.tiltDb`, `eqAmount` in the output). A bright voice
made brighter only gets harsher and peakier. (`gentle` aims at −10 dB over its own 7.8 dB.)

Measured on the Malayalam final (742 s, `--loudness social`), before → after:

| | before | after |
|---|---|---|
| integrated loudness, true peak | −14.1 LUFS, −1.3 dBTP | −14.0 LUFS, −1.3 dBTP |
| energy below 120 Hz / 1–4 kHz (mix) | 37.8 % / 1.5 % | 15.2 % / 8.4 % |
| the narration's, alone | 36.8 % / 1.0 % | 15.2 % / 8.4 % |
| the Mix's limiter: most, time over 1 dB | 3.7 dB, 4.3 s | 0.8 dB, 0 s |
| the mix before its limiter, true peak | +2.4 dBTP | −0.5 dBTP |
| the narration over the music while it speaks (median / quietest tenth) | 25.9 / 20.2 LU | 30.5 / 25.3 LU |
| the narration over everything else | 22.0 / 15.7 LU | 28.2 / 22.3 LU |
| the impacts and risers (A5) / the transitions' sounds | −22.8 / −20.6 LUFS, 97 % / 22 % below 120 Hz | −31.7 / −24.6 LUFS, 25 % / 22 % |

The narration's own limiter catches its onsets (at most 5.3 dB, 0.8 dB at the 99th percentile,
3.8 s over 1 dB in 536 s of speech), so the Mix's limiter has almost nothing left to do.

**Voice-preserving: `--mix gentle` (2026-10-21).** The author heard the cloned voice "a bit
different" after this mix; on their TV they then compared the two and kept this one, the
clearer, as the default, so the voice-preserving chain is the `gentle` profile. Measured with
`voice.mixCheck` (the narration chain run as the mixer runs it, and the OpenVoice tone colour of the readings against the speaker's own held-out speech, all at one
loudness; `filmcraft-cli voice mix-check`), the first chain moved the Malayalam narration from
0.921 to 0.856 and the English one from 0.861 to 0.759 — further than the clone itself is from
the speaker — and the speaker's genuine speech as far (0.956 → 0.888). Per band, the low-mid
cuts (the low shelf's extra −3 dB, −2 dB at 300 Hz) cost about twice the timbre a lift at
2–3.5 kHz costs for the same rise in the 1–4 kHz share, and the lift at 1.2 kHz (the vowels'
formants) half as much again; the consonants that make words understood on a TV are at
1.4–5.7 kHz. So in `gentle` the low shelf stays at −3 dB, the 300 Hz cut goes, the lift moves
to 2 and 3.5 kHz, and the target tilt is −10 dB:

| ten lines of the show's narration | none | first chain | voice-preserving |
|---|---|---|---|
| Malayalam (IndicF5): similarity to the speaker | 0.921 | 0.856 | **0.867** |
| English (VoxCPM2): similarity to the speaker | 0.861 | 0.759 | **0.787** |
| the speaker's genuine speech through the chain | 0.956 | 0.888 | 0.899 |
| Malayalam: below 120 Hz / 1–4 kHz / the 2 and 4 kHz octaves | 36.9 % / 1.1 % / −20.4 dB | 15.2 % / 8.9 % / −11.0 dB | 19.0 % / 7.4 % / −11.0 dB |

And the Malayalam final remixed (`showcase-remix --loudness social`), first chain → voice-
preserving: −14.0 LUFS and −1.3 dBTP both; the narration over the music while it speaks 30.3 /
25.2 → 30.3 / 25.3 LU (median / quietest tenth), over everything else 28.1 / 22.3 → 28.2 / 22.4;
the mix's 2 and 4 kHz octaves −13.4 / −15.1 → −13.8 / −14.6 dB (its 1 kHz one −14.7 → −17.7);
energy below 120 Hz 15.2 → 18.9 % and at 1–4 kHz 8.4 → 7.2 %; the Mix's limiter at most 0.86 →
0.46 dB (0.43 → 0.01 at the 99th percentile). The narration's own limiter works a little harder
on the brighter onsets (at most 5.3 → 7.1 dB; 3.8 → 6.7 s over 1 dB in 539 s of speech).
Known limits: the transitions' sounds are synthesised on the Mix's input, so they cannot be
high-passed (only made quieter); the EQ is fixed in shape (only its amount follows the voice).

`showcase-remix [--sequence id] [--mix speech|gentle|cinematic] [--loudness social]` ([`mix::remix`])
applies a profile to a show built before: the effects an earlier mix put on the stems' tracks
(and their automation) are taken off, every clip levelled and ducked again, the profile's track
effects put on, the transitions' sounds set to its level, and the mix normalised; running it
twice gives the same mix. `--save` (or `--save-as`) keeps it; then export the sequence.

### Album as the seed

Put the clips for a video into one album in Google Photos (and the photos of robots you never
filmed) — and add to it whenever you film
more — then make the whole video from it with one command:

```sh
filmcraft-cli photos auth                    # once: see crates/gphotos/README.md
ollama pull qwen2.5vl:7b                     # analysis, and the agent that drives the picker
filmcraft-cli showcase --album "Robot showcase" \
    --prompt "a video of every robot I've filmed since 2020" --voice clone:me \
    --project ~/Movies/AutoCut/robots.fcproj --save \
    --export ~/Movies/AutoCut/robots.mp4 --upload --upload-album "Robot showcases"
```

In order:

1. **Checks.** Usage errors (exit 2), then the Google Photos sign-in (read from disk), the upload
   permission (with `--upload`) and the vision model — all before anything is picked.
2. **What the video is of.** The subject — and a date and a title, when the prompt names them —
   is read from `--prompt`: the built-in rules ("every robot since 2020", "a montage of all my
   vintage cars"), else the classifier (`AUTOCUT_LLM_MODEL`); `--subject robot` skips that. A
   prompt nobody can read stops here, before the picker opens.
3. **The album, picked whole.** Google's picker opens in the browser, and the local screen agent
   ([crates/screenagent](../screenagent/README.md)) searches the picker for exactly that title —
   Return on an album's exact title shows the album's items — selects that whole result, then
   presses Done. Its videos and photos are downloaded into the cache (photos as Google's
   full-resolution originals; an item picked before is not downloaded again): a robot that was
   only photographed is shown in its photos ([Photos](#photos-a-slow-pan-and-zoom-towards-the-robot)).
   `--videos-only` (`photos: false` for the jobs and `showcase_analyze`) leaves the photos out,
   and `--since` skips items created before it.
4. **Imported once.** `file.import {reuse: true}`: a file the project already holds is reused,
   never imported twice — run the command again after adding clips to the album and only the new
   ones come in.
5. **Only the album is analysed**, as `analyze` does (`--frames`, `--screen`, `--compare`;
   `--report-out r.json` keeps the report), never the rest of the project — and a clip analysed
   by an earlier run comes from the analysis cache (`--fresh` analyses again).
6. **Then exactly what `showcase --report` does**: entries, voice-over, build, QA, subtitles,
   thumbnail, chapters, export and upload. The title is `--title`, else the prompt's, else
   "Every <Subject> I've Filmed"; `--plan` stops before the build (with `--save` the imported
   clips are kept). `--album` is the album the clips come from, `--upload-album` the one the
   export goes into (`showcase --report … --upload --album T`, its older spelling, still
   works). Upload into the seed album itself and the next run picks the finished video too: the
   CLI warns.

**Why the picker.** Since 2025-03-31 Google lets an app read back through the Library API only
the albums and media items it created itself, so the clips a person adds to an album by hand —
the seed — are reachable only through the Photos Picker, where a person, or the screen agent
standing in for one, selects them. FilmCraft cannot list an album's contents, and the picker
has no Albums view; the agent searches the picker for the album's exact title, whose result is
the album's items, and selects all of it. It needs Screen Recording and Accessibility for the terminal or app running
it.

The same in the app and for agents: `autocut.analyze` / `autocut.makeShowcase` with `album` (and
`subject` or `prompt`) — the Showcase panel's "From a Google Photos album" field,
`showcase_analyze {album, prompt}` over MCP — then review and build as usual; with
`--bridge 127.0.0.1:9876` the CLI runs it in the open app.

### Showcases in the app

The same pipeline runs inside the app (and for the CLI and MCP agents) as engine jobs, so the
review happens where the footage is:

| command | job |
|---|---|
| `autocut.analyze {subject \| prompt, items? \| album?, photos?, since?, frames?, compare?}` | the analysis above (photos too, unless `photos: false`); `since` ("2020", "2020-06", "2020-06-15") keeps the clips filmed on or after it (undated ones are left out); `album` takes the clips from that Google Photos album (picked, imported once, only those analysed: [Album as the seed](#album-as-the-seed)). The report is kept in the session |
| `autocut.showcasePlan {subject?, title?, subtitle?, voice?, mood?, minCoverage?, entries?}` | group the kept report's clips (or take reviewed `entries`) and write the voice-over; the plan is kept |
| `autocut.makeShowcase {subject \| prompt, since?, items? \| album?, title?, voice?, mood?}` | both in one job (Sequence ▸ Make a Showcase…, the `showcase` intent) |
| `autocut.showcaseBuild {plan?, export?, upload?, uploadAlbum?, qa?}` | build the plan (the kept one, or `plan`); with `export`, the YouTube export, `<out>.chapters.txt` and the QA report beside it; with `upload`, a Google Photos upload job for the export (into `uploadAlbum` too) |
| `autocut.showcaseInspect {report?}` / `autocut.showcaseSet {report?, plan?, reportPath?, planPath?}` | read what is kept / keep a report or plan from elsewhere (the CLI's `analyze --out` and `--script-out` files) |

A plan is the CLI's `plan.json` plus the review: per entry `include` (default true) and
`mergePrevious` (its clips are the included entry before it: their shots join, the three longest in
view kept), and the plan's `title`, `voice` (none: no narration), `mood` (and optional `subtitle`,
`bpm`, `width`, `height`, `fps`); `script.lines` follows `entries` one to one
([`host::PlanDoc`](src/host.rs), [`host::resolve`](src/host.rs)).

The engine runs the jobs and keeps what they make; this crate is the host the app, the CLI and the
MCP server install ([`host::engine_host_with_albums`](src/host.rs), Ollama models from the
`AUTOCUT_*` environment, and `filmcraft_automation::photos::album_source()` picking albums). A job runs on its own thread and edits only through engine commands sent over the
session's command port, which the session's owner runs (the UI between frames, a `wait: true` call
while it waits, the MCP server on a pump thread): every edit is an ordinary, undoable command, and
the app stays responsive. A long command (`file.exportMedia` with `wait: true`, an upload) is
started without waiting and polled, and each narration line is synthesized first as a background
`tts.render`, so nothing long runs on the UI thread. One showcase job runs at a time; `jobs.cancel`
stops it (model requests end at once).

**The Showcase panel** (Window ▸ Showcase, beside the Text panel; Sequence ▸ Make a Showcase… opens
its Analyze form): a subject, a date or a request in words → Analyze; then the entries in date
order, each with a thumbnail, its label, description and narration line (all editable), a switch to
leave it out, "Merge with previous", and its shots (each plays in the Source monitor); the title,
voice (cloned voices listed), mood, intro and outro; Write script (the words again, for the entries
as reviewed), Build, and Build and export (with the export path, and optionally the upload); the
running job's progress and Cancel. "Look and channel outro" holds the style (tech, classic or as
planned), the timeline, loudness and embedded captions, and the branded outro: its logo (Choose… or
a path), channel, handle, tagline, button and clicked words, typed line, length and footage
(`item:from:to, …`); the plan's other look fields (a custom style object, transitions, events) go
back as they came. A job started elsewhere (an agent, the prompt) brings the panel forward and its
plan loads there. Every widget has an automation id (`showcase.lookOpen`, `showcase.style`,
`showcase.outroChannel`, …: the panel's module docs list them) and the draft is UI state
(`ui.set {showcase: {look: {outro: {channel: "H4hacks"}}}}`, see
[docs/control-protocol.md](../../docs/control-protocol.md)).

### Review notes into the plan (`showcase-plan`)

A new round of a showcase is a reviewed plan plus notes, keyed by entry **label** (numbers change
as entries come and go): [`plan_patch::apply`](src/plan_patch.rs) (`filmcraft-cli showcase-plan
--base plan.json --patch notes.json --project p.fcproj --out new.json`) changes entries (`line`,
`chips`, `delivery`, `say`, `place`, `event`, `kind`, `description`, `redact`, `unix` as seconds or
a date in the notes' `timezone`), replaces their `shots` (`{item, from, to, speed, fit, mute}`: the
plan's analysed shot of that item, or with `plain` a shot counted in view across its window, its
clip's name and length from the project) or edits them (`shotEdits`), splits one entry into two
(`split`), removes one (`remove`), adds new ones (`add`), and merges the script's `intro`, `outro`,
`pronounce` and the look's fields (`outro`, …). The entries are sorted by date and numbered again;
`script.lines`, `chips`, `delivery` and `say` follow them, a scene keeps no chips, and the robot
count in the outro ("Forty-three", or `{Count}`) is updated. Entries without a line are reported.
[`plan_patch::with_lines`](src/plan_patch.rs) (`--lines script-ml.json`) writes the plan for
another language: `{"intro", "outro", "lines": {"<label>": "…"}}` over it, `language` set, the
English respellings and `say` cleared, failing on any spoken entry without a line. Numbers are
read and written back exactly (serde_json's `float_roundtrip`). Measured 2026-10-20: the v5 round
(its 14 places, 16 changed lines, a split, 4 new robots, the Malayalam lines) as one notes file
reproduces the `plan-v5.json` and `plan-v5-ml.json` the round's Python script wrote, value for
value; a trimmed copy is the test fixture (`tests/data/plan_patch`). The steps of a whole round:
[docs/showcase-runbook.md](../../docs/showcase-runbook.md).

### Malayalam lines in natural spoken Malayalam (`ml_translate`)

[`ml_translate::translate`](src/ml_translate.rs) (`filmcraft-cli ml-script`, the
`autocut.malayalamScript` job) writes the lines file of a plan's English voice-over with the
script model, in the style of [docs/malayalam-style.md](../../docs/malayalam-style.md): the style
as the system prompt, built-in examples and the user's approved pairs (`<data>/autocut/malayalam-
examples.json`, kept by `--remember`, never committed), each line with the facts it must keep.
Every line then gets `filmcraft_tts::ml_script::fix`, the lint and the fact lock; a line with a
warning or a lost or added fact goes back once with what to fix, and the better version is kept.
Ollama is asked for a 16k context (`Ollama::with_context`): Malayalam takes many tokens.
Measured 2026-10-20 on a 52-part plan with `qwen3:8b` (25 min on a shared M3 Max): all parts
written, no lint warning, but the fact lock flagged 19 parts (the hand-written script: 3) and
several unflagged lines changed meaning; qwen3:8b's Malayalam is a rough draft at best, and the
lint and the fact lock are the dependable part (details in the style guide). The run also found a
bug, fixed: cleaning the model's answer dropped a year that started a line, taking it for a list
number.

## Shorts: long video → vertical clips

`autocut.shorts` (job; `shorts.rs`) is the local "Opus Clip" factory. The transcript (Whisper's,
made first when there is none) is cut into sentences (`.` `!` `?`, pauses over 1.2 s, at most 45
words); every run of whole sentences `minSeconds`..`maxSeconds` long is a candidate, scored by the
highlight reel's interest score (`highlight::interest`: content words, the topic, numbers) per
sentence weighted by speaking time, a length term peaking mid-range, speech density and the
opening's hook (a question, a direct address or a bold claim, said briefly). With Ollama the best
36 openings are also rated 0..10 for how well they hook and get a short title (the script model,
else the classifier). The best `count` that do not overlap each become a vertical sequence:
`sequence.autoReframe` of its range (a little lead-in and tail from the silence around it) to
`aspect`, the speaker followed by the subject tracker (`filmcraft_render::subject`); where nobody
is found — a whole clip, or a stretch of a second or more where the subject is lost — the
`fallback`, `blur` by default: the whole picture fitted to the width over a blurred, dimmed,
enlarged copy of itself (the showcase's portrait-shot look; the crop eases out to it and back in,
cuts jump), `letterbox` for black bars; `transcript.createCaptions` (18 characters, one line) and
`captions.setStyle` (76 px white with a 7 px outline, no box) just above the target `platform`'s
bottom zone, shrunk until `sequence.checkSafeZones` finds none in its side zones; a `title` card
(`graphics.newText`, bold paragraph text centred in a box at the top of the safe area);
`file.exportMedia` into `export` with `burnCaptions`; `youtube.upload` (private unless
`youtube.privacy`) with `uploadYoutube`. A step that fails for one short is noted in its `notes`;
the others are still made.

### Platform safe zones

`platform` (`--platform` on `filmcraft-cli shorts`): `youtube` (the default), `reels`, `tiktok` or
`vertical` (safe on all three). The zones are the parts of a 9:16 frame each platform's own
interface covers, as shares of the frame, rounded outwards from the platforms' public guidance
(sources in `crates/render/src/safe_zones.rs`):

| platform | top | bottom | left | right | buttons |
|---|---|---|---|---|---|
| YouTube Shorts | 15 % | 35 % | 5 % | 18 % | |
| Instagram Reels | 14 % | 35 % | 6 % | 6 % | |
| TikTok | 13 % | 35 % | 12 % | 12 % | right 20 % from 40 % down |
| all (`vertical`) | 15 % | 35 % | 12 % | 18 % | right 20 % from 40 % down |

Each short reports `safeZones` (`{platform, ok, overlaps}`) and a note per caption or title still
in a zone. The QA (`qa.rs`, check `safe_zones`) warns on the same, for `Options::platform` or, on
a vertical sequence without one, for all of them. In the app, the monitors' wrench menu and View ▸
Platform Safe Zones draw the zones over the picture (`view.safeZones`).

Demo (`tests/blur_fill_demo.rs`, by hand): synthetic footage whose subject is gone for a stretch
becomes a 1080×1920 short — followed, eased out to the blurred fill and back, captions and title
inside YouTube Shorts' safe zone — with stills of each state and of the zones over the fill:
`FILMCRAFT_DEMO_DIR=target/blur-fill-demo cargo test --release -p filmcraft-autocut --test
blur_fill_demo -- --ignored --nocapture`; then `safe_zones_ui`'s ignored test screenshots it in
the app under the zones (`FILMCRAFT_DEMO_PROJECT=<dir>/demo.fcproj FILMCRAFT_UI_SHOTS=<dir>`).

The showcase reuses the tracker: a filled (portrait or 4:3) shot calls `clip.autoReframe` with
`keepScale` (the push-in stays) seeded from the analysis' box, and keeps the tilt only where its
subject is not found (`Options::follow_subject`, `--no-track`).

## Trailer for a finished show

`autocut.trailer` (job; `trailer.rs`; `filmcraft-cli trailer`, MCP `trailer_make`) cuts a finished
showcase into a vertical trailer that sends viewers to the full video: 9:16 for Instagram Reels by
default (also 1:1 and 4:5), about 34 s (`seconds`, 10–90).

**What it reads.** The show's plan (entries: labels, kinds, dates, the shots with the analysis'
best moment, each entry's `redact` targets, the `outro`'s logo and handle, the `style`) and its
build report (the JSON `filmcraft-cli showcase` prints, or the last build's outcome: where every
shot played — the window of its clip — what the privacy pass hid, the narration's lines), and the
show's sequence for its narration clip. Without a plan the entries are read from the sequence
(`entries_from_sequence`: the clips of its picture track, named by the lower thirds over them, no
dates).

**The cut** (`timing`): every length is a whole number of beats of the music, every cut on a beat
to the frame.

| part | what |
|---|---|
| hook (about 3 s) | the most striking robot (`striking`: a humanoid, then a robot dog, then arms and vehicles; how big it is in its picture; filmed upright — it fills a vertical frame; less for a photo, much less for an entry that hides a sponsor's logo), or `hook` (a label or `#number`; `hookAt`: from that second of its clip — the analysis' best moment is the robot at its largest, not always its face), filling the frame, under a big title: "48 robots" and "2013 → 2025" from the show's robot count and years (`title` replaces it; " · " splits its lines). The show's first narration sentence plays over it (`hookVoice`, on when the show is narrated): its first narration clip, cut at the pause (`media.pauses`) nearest where the sentence's share of the line's text ends — the owner's own voice, nothing synthesized |
| montage | `robots` robots (12–18 by default, as many as fill the length at 1.3–2.4 s each), in date order so the years build up: the earliest and the latest always, `include` ones, and in between the most striking of each stretch of the show (`choose`); one shot per robot ("Pepper, again" is Pepper), none of `exclude`, nor the hook's robot again. The longer shots come first so the cutting quickens. Each shot plays its entry's best moment (the analysis' largest view of the robot, a little before it) inside the window the show played — or where `at` says: `{"<label>": seconds}`, or `{"item": another clip of the entry, "seconds": …}`. A robot the vertical crop holds (the analysis' box at most 0.8 of what a filling crop shows; portrait footage always) fills the frame and `clip.autoReframe` follows it (seeded with that box; `fallback` blur where it is lost); a wider one — a vehicle, an arm, a robot filmed up close in landscape — would be cut to a slice, so it is shown whole in a band (`band`: the picture as wide as puts the robot across 90 % of the frame, from the whole picture up to filling the frame, the robot centred as far as the picture covers the frame, pushing in slowly) over a blurred, dimmed copy of itself. Its name and year sit in a dark panel at the bottom of the platform's safe area |
| end card (about 3 s) | from the music's final hit, a bar: the hook's footage blurred and dimmed, the channel's logo on a light badge popping in, `endText` ("Full video on YouTube") and the handle (`logo`, `channel`; from the plan's `outro` by default), all inside the safe area |

**Sound.** Generated music (`music.suggest` → `audio.generate`, `mood` energetic, `bpm` 124, a fixed
`seed`) arranged to the trailer's exact length: its final chord lands where the end card starts
(a bed `hit + end card` long ends on the bar `hit`: `end_card_length`), with a riser into it, an
impact on it and a whoosh into the montage. Mixed speech first (`mix::mix_down`, the plan's `mix`),
then the music brought up 8 dB from the show's bed level (it carries a trailer; still ducked under
the voice) with a limiter on its track at −3.5 dBFS, and the voice's fader 3.5 dB down (it stands
over 20 LU above the ducked music), then normalised to −14 LUFS, −1 dBTP. Measured on the reel: the
Mix's limiter takes at most 0.6 dB off (9 dB before these levels: the speech mix normalised a
34 s trailer that is mostly music by +8 dB, straight into the voice's peaks).

**Privacy.** Two layers, so no face the show blurred is shown sharp:

1. The show's own blur, carried over (`carry_redaction`): for every shot of footage the show hid
   something in (the report's `privacy`), its blur and mosaic effects — the masks of its privacy
   pass, keyed in media time — are copied from the show's clip and pasted onto the trailer's clip
   and its band's copy with `edit.pasteAttributes` `keepTimes` (the same moments of the same
   media), whatever window and framing the trailer gives it.
2. The privacy pass again on the trailer's clips (`privacy::redact_show`): faces and plates when the
   show hid any (or `redact`), every entry's named targets (the DP World logo on the Promobot stays
   hidden), each robot's own box left sharp — through the redaction cache. It runs with the shots
   filling the frame, before the bands and the reframe (whose blurred copies keep the masks), and
   hides a face at the size the show hid it: `face_threshold` scales `minFaceHeight` by how much
   smaller the trailer shows the picture than the show did (a portrait clip that filled the 16:9
   show was 3.2 times larger there, so its faces count from a third of the size).

A face the show left sharp is not hidden by policy: the trailer hides what the show hid, so look at
the frame sheet for visitors the show itself let through.

**Out.** The sequence "<show> — Reel trailer"; with `export`: the video (Social Vertical 1080×1920:
H.264, AAC 48 kHz, 12 Mb/s, GPU rendering as set; `preset` for another), `<out>.cover.jpg` (the
hook's frame with its title, full size; inside the 3:4 the profile grid shows), `<out>.caption.txt`
(the caption to paste: the title, three robots by name — the hook's and the most striking others —
"Full video on YouTube: @handle, link in bio" — a Reel's caption shows no links — and hashtags from
the robots' kinds, events, places, the channel and the narration's language), `<out>.caption.json`
(the same with `link`, for the profile's bio, and the robots), `<out>.frames.jpg` (a frame of every
shot from the exported file, read in a session of its own) and QA (`<out>.qa.json`, `.qa.png`,
`.privacy.png`: joins, black frames, black sides of the bands, the robot in every shot, privacy, the
platform's safe zones and the mix — the voice measured against the music while it speaks, not
against the louder music after it — plus the trailer's own: cuts off the beat, text or the logo
outside the safe area, and every shot of footage the show hid faces in where the trailer's own pass
hides none: a warning, or a note when the show's blur was carried onto it). The outcome lists every
shot (robot, year, clip window, inside the show's window, band or followed), the clips that got the
show's blur, the cuts on the beat, the safe-zone check (each text's box) and QA's summary.

Tests: `trailer_tests.rs` (the cut on the beat for 80–180 bpm and 10–90 s, the picks, the windows
and `at`, bands, the face thresholds, redaction carried over, the boxes inside every platform's safe
area, the caption, hostile plans and reports) and `tests/trailer_build.rs` (a synthetic show built
from the demo footage gets a 9:16 trailer of the right length, on the beat, with its words in the
safe zone, an end card, and the show's masked blur on its shot; hostile requests fail cleanly).

## Privacy: faces and plates hidden, robots left alone

`privacy.rs` holds AutoCut's side of the engine's automatic blur (`clip.autoRedact`,
`crates/engine/src/redact.rs`; detectors in `crates/vision`):

- `OllamaJudge`, the engine's `VisionJudge` (installed by the desktop app and the CLI): the crops
  of what the detectors found, up to six, tiled into one numbered grid picture; the analysis model
  (`AUTOCUT_ANALYSIS_MODEL`) answers per tile — faces: person / robot / mannequin / statue /
  poster / screen; plates (outlined in red): plate / label / sign / screen. Only people and
  vehicles' plates are hidden; an unclear answer hides. Measured 2026-10-19 on GITEX footage:
  a grid of six takes about 5 s (one picture alone about 3 s); sending six pictures in one
  request took 20 s and called a robot's head a person, the grid did not.
- The showcase's privacy pass (`Options::redact`; `--redact faces,plates`, the plan's `redact`:
  `true`, `"faces,plates"`, a list, or `{kinds, style, strength, minFaceHeight, verify}`): after the
  show is built and before it is exported, `clip.autoRedact` runs over every footage clip and the
  blurred background copy of each shot shown whole (`fit: full`; the engine analyses a copy once
  with its clip), each robot's subject box (the analysis' largest moment, taken square) excluded.
  What happened goes to the build's notes and progress.
- Named targets: the plan's `redact.targets` (`--redact-target`, repeatable) hide logos, brands
  or signs in every footage clip; an entry's or a shot's own `redact` (`{"targets": [...]}`, or
  `false` to leave its footage alone) adds targets for its clips only. `OllamaJudge::find`
  asks for every place each target appears (`{"found": [{"target", "bbox_2d"}]}` in the model's
  resized frame, `analyze::model_frame`); the engine follows them between looks. Measured
  2026-10-19 on the Promobot clips of Robot Showcase v5 (items 70, 68, 69: 130 s, targets "DP
  World logo" and "dpworld.com text", a look a second), scored by eye on 63 frames sampled
  every 1–2.5 s: 52 of 59 legible DP World logos, wordmarks and the dpworld.com floor sign
  hidden (88 %; 9 of them only the wordmark, the emblem above it left), 7 missed (small badges
  on the robot's arm and base, the dpworld.com label on a printed photo), 3 blurs on something
  else (the tablet's header text, its bezel, a wall screen after its picture changed); the
  robot's LED face never blurred; 794 s (131 looks at about 6 s each). Before two fixes the
  same clip scored 73 % on item 70 and missed the floor sign entirely: the model sometimes
  writes a box corner into `target` (now the place is kept, named by its `label`), and a
  tracker that only held a lost box drifted onto the robot's face (now a held box is not kept,
  and a look that does not find it ends it).

- QA of what was hidden: `clip.autoRedact`'s outcome now says, per clip, how much of the picture
  its masks cover (`coverage`: the largest single mask at any keyframe and the most hidden at
  once, as shares of the clip's frame, with when in media and sequence seconds), and the build
  keeps it (`Built::privacy`, [`privacy::Hidden`](src/privacy.rs)). The showcase QA's `privacy`
  check warns about any mask over a fifth of a picture (the real bug of the v5 round: a named
  target's box over a close-up of the Promobot's tablet filled the frame with blur), counts the
  faces, plates and named places hidden in each entry (`privacy` in `<out>.qa.json`), and draws
  a privacy sheet, `<out>.privacy.png`: one frame of each clip with masks, where most of it is
  hidden, under its entry's number.

Measured on Robot Showcase v4 (560 s of footage, 90 clips, mostly 1080p H.264 with some VP9
4:4:4), the whole sequence on an M-series Mac, 2026-10-19: with faces counted as the sequence
shows them and a face score of 0.6, 356 s (detection 141 s, vision check 214 s for 211
tracks; load average 12 falling to 3): 151 faces hidden in 35 clips, 55 robots / screens /
posters left sharp (Pepper, NAO, Ameca, Furhat, Promobot), 163 faces too small to recognise
left alone. An earlier pass at score 0.7 on the source picture: 248 s, 99 faces hidden; it
missed half-turned faces in a crowd at 0.6–0.7. The plate detector's 33 hits were all robot
name badges, signs or machines (no vehicle in the footage): the shape check removed most, the
vision check the last 5.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `AUTOCUT_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `AUTOCUT_LLM_MODEL` | `phi4-mini` | classifier / planner model |
| `AUTOCUT_VLM_MODEL` | `moondream` | frame-captioning model (any Ollama vision model) |
| `AUTOCUT_ANALYSIS_MODEL` | `qwen2.5vl:7b` | clip analysis: a vision model that grounds objects (boxes) |
| `AUTOCUT_SCRIPT_MODEL` | `qwen3:8b` | showcase voice-overs (falls back to `AUTOCUT_LLM_MODEL` when not installed) |
| `AUTOCUT_CONFIDENCE_THRESHOLD` | `0.6` | below this, hand off to the supervisor (the caller) |

## Surfaces

Caption styles: `add_captions` takes `style: tiktok | bold | clean | default` and follows
`transcript.createCaptions` with `captions.setStyle`.

- **CLI:** `filmcraft-cli autocut '<prompt>' [--dry-run] [--frames]` (headless `--demo`/`--project`,
  or `--bridge` to edit in the running app). A hand-off exits 1 with the report on stdout.
  Showcases: `analyze` + `showcase --report`, or `showcase --album "<title>" --prompt "…"` from
  a Google Photos album in one command. Shorts: `filmcraft-cli shorts talk.mp4 --count 3
  --export out/ [--platform youtube|reels|tiktok|vertical] [--fallback blur|letterbox|center]
  [--upload-youtube]`.
- **MCP:** the `autocut_plan` (dry run) and `autocut_edit` tools, on both MCP backends
  (`filmcraft-cli mcp --demo` / `--bridge`). When a report has `needsSupervisor`, the calling
  agent plans that edit itself with the command tools — that is the supervision loop. Showcases:
  `showcase_analyze` (`plan: true` also plans; `prompt` instead of `subject`; `album` for a
  Google Photos album's clips), `showcase_plan`, `showcase_build` (`upload_album`; long jobs with
  progress and cancellation) and `showcase_inspect`.
- **App:** the Showcase panel (above); `filmcraft-cli exec autocut.makeShowcase subject=robot
  wait=true` runs the same job headless.

## Tests

`cargo test -p filmcraft-autocut` is fully offline: scripted models (`llm::FixedModel`,
`llm::FnModel`, and `host::Models::scripted` for whole showcase jobs), the demo project headlessly,
crafted transcripts through `transcript.set`. Showcase jobs (`tests/showcase_jobs.rs`): make, review
(leave out, merge, relabel), write the script again, build; export with the QA report and an upload
job; the prompt starting the job with no model and no open sequence; cancel; bad plans. An album as
the seed (`tests/showcase_album.rs`, `album::FixedAlbum` for the picker, real ProRes clips from our
encoder): only the album's clips are analysed, a clip already in the project is reused and a
second run imports nothing, a prompt names the subject (rules or classifier, a date reaching the
picker), an unreadable prompt picks nothing, an empty album and a host without albums say why, an
album's photo is kept, analysed once and built as a photo (and left out with `photos: false`).
Photos (`tests/showcase_photos.rs`, photos drawn in the test): a robot only photographed is an
entry of its photos; the planned hold is the still clip's length; Motion position and scale are
eased from the first frame to the last and end with the robot whole in frame; consecutive photos
move the other way; a portrait photo whose robot a crop would cut is whole over its blurred copy;
an entry mixes video and a photo; a visitor's face in a photo is hidden by one static mask; QA
finds no dip or black around the photos; hostile holds and moves are refused before anything is
built. Showcase QA
(`tests/showcase_qa.rs`): a deliberately broken sequence on the demo media (a half-second gap,
a black slate, a tone pushed 30 dB over full scale between two narration lines) gets each
finding (the gap, the hole on every track, black in the gap and on the slate, clipping, the
voice quieter than between its lines); clean showcases, with and without narration, get no
errors (and the voice at least 4 dB on top); a scripted grounding model flags exactly the shot
it does not find the subject in. Live checks
(`tests/live_models.rs`) are opt-in: `AUTOCUT_LIVE=1` (the classifier, the VLM, and a complex
prompt handing off with its document).
