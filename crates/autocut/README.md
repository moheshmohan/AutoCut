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
   longest in view (length × coverage), in time order, under the most specific label among its
   clips (a name read on the robot first), dated when it was first filmed. The local LLM
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
     looks vertical; `--no-fill` shows it whole over a blurred, dimmed copy instead; slowed only
     when the words would outrun the footage), each shot's sound faded in and crossfaded; a
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
   - **the music and its beat**: a generated bed (`audio.generate` at `--bpm`), or with
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
   - **the mix**: the voice matched to -16 LUFS (dialogue), the shots' sound (ambience, -27)
     and the music (-24, generated or the user's) ducked under it, effects at -23, and a
     hard limiter at -1 dBFS on the mix;
   - **subtitles** (`subtitles.rs`): each narration line as written, split into cues of at most
     two 42-character lines, 1–7 s each, at sentence ends first and then clauses (a small
     dynamic programme over the words), timed in proportion to the characters within the line's
     clip (a cue too short to read stays up into the pause after it, never into the next line);
     on a `Subtitles` caption track of the sequence (shown in the app, not burned in);
   - optionally the YouTube 1080p export, with `<out>.chapters.txt` (intro, each year, outro —
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
   | `mix` (`file.exportMedia` to a temporary 32-bit float WAV, metered with `filmcraft-audio-dsp`'s BS.1770 `LoudnessMeter`) | integrated loudness outside −16…−14 LUFS (±0.5 LU) — warning; true peak above −1 dBTP — warning; above 0 dBTP or any sample at full scale — error (clipping, with the time of the first); the median short-term loudness (3 s windows wholly inside narration clips on A2 vs wholly between them; 400 ms windows when lines or pauses are too short) — the voice less than 4 dB louder: warning, not louder at all: error |
   | `subject` (`AUTOCUT_ANALYSIS_MODEL`) | the middle frame of every shot, rendered and grounded with the analysis' prompt (`analyze::ground_prompt`) — a shot where the subject is not found: warning; without a reachable model the check is recorded as skipped (model unavailable) |

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

### Album as the seed

Put the clips for a video into one album in Google Photos — and add to it whenever you film
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
   ([crates/screenagent](../screenagent/README.md)) opens the album with exactly that title and
   selects all of it, then presses Done. Its videos are downloaded into the cache (an item picked
   before is not downloaded again). Photos are skipped — a showcase is built from video clips, so
   there is no `--with-photos` — and `--since` skips items created before it.
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
standing in for one, selects them. FilmCraft cannot list an album's contents; the agent finds
the album in the picker by its exact title (its Albums view, or a search for the title) and
selects everything. It needs Screen Recording and Accessibility for the terminal or app running
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
| `autocut.analyze {subject \| prompt, items? \| album?, since?, frames?, compare?}` | the analysis above; `since` ("2020", "2020-06", "2020-06-15") keeps the clips filmed on or after it (undated ones are left out); `album` takes the clips from that Google Photos album (picked, imported once, only those analysed: [Album as the seed](#album-as-the-seed)). The report is kept in the session |
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
running job's progress and Cancel. A job started elsewhere (an agent, the prompt) brings the panel
forward and its plan loads there. Every widget has an automation id and the draft is UI state
(`ui.set {showcase: …}`, see [docs/control-protocol.md](../../docs/control-protocol.md)).

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
  a Google Photos album in one command.
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
picker), an unreadable prompt picks nothing, an empty album and a host without albums say why. Showcase QA
(`tests/showcase_qa.rs`): a deliberately broken sequence on the demo media (a half-second gap,
a black slate, a tone pushed 30 dB over full scale between two narration lines) gets each
finding (the gap, the hole on every track, black in the gap and on the slate, clipping, the
voice quieter than between its lines); clean showcases, with and without narration, get no
errors (and the voice at least 4 dB on top); a scripted grounding model flags exactly the shot
it does not find the subject in. Live checks
(`tests/live_models.rs`) are opt-in: `AUTOCUT_LIVE=1` (the classifier, the VLM, and a complex
prompt handing off with its document).
