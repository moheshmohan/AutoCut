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
| `custom` | nobody: the report comes back with `needsSupervisor` |

Highlight scoring (`highlight.rs`) is classical: sentence/pause segmentation, topic-keyword and
salience scoring, duration shaping — no model needed to pick the cuts.

Transcription over the bridge can outlast the app's 60 s reply limit: a reply timeout from
`transcript.generate` means "still transcribing", and the runner polls for the words (up to 30
minutes) instead of planning without them.

## Showcases: analyse, then build

A showcase is "every robot I have filmed, in order": many clips, one narrated, counted video.

1. **`analyze`** (`analyze.rs`): for each video item, frames are sampled through the engine
   (`media.frame`) and a grounding vision model (Qwen2.5-VL, `AUTOCUT_ANALYSIS_MODEL`) is asked
   frame by frame to *locate* the subject — a box or none. A box says whether, how big and
   where, so the best moment is where the subject is largest. One more request per clip names
   and describes it; a proper name survives only if it is a known robot or maker
   (`sanitize_label`), since a 7B model invents brands. Capture dates come from
   `media.captureDate` (Google Photos picker sidecars, or the file's own metadata).
2. **`showcase`** (`showcase.rs`): clips that show the subject are grouped into entries — the
   same robot is counted once: the same specific name within a week (an expo's other days), or
   the same day with a generic label of the same kind; up to three shots each. The local LLM
   (`AUTOCUT_SCRIPT_MODEL`, falling back to `AUTOCUT_LLM_MODEL`) writes the voice-over from the
   facts, a few entries per request (template lines fill anything it does not deliver, and a
   copied item heading is dropped so it is not read out). The build runs ordinary engine
   commands into a new 16:9 sequence, every cut on a frame:
   - **cold open**: a montage of robots from across the years, cut on the music's beat, the
     title over it, a riser into an impact;
   - **per entry**: the narration (any voice — `clone:<name>` included), then a few seconds of
     the robot's own picture and sound before the next; shots cut where the analysis saw the
     robot (fitted, over a blurred and dimmed copy when the clip is portrait; slowed only when
     the words would outrun the footage), each shot's sound faded in and crossfaded; a
     name/date lower third; a dramatic transition from a rotating set (cross zoom, whip, flash,
     push, glitch, light leak, zoom blur; dip to black at a new year) with a whoosh; the year
     as a title card with an impact;
   - `graphics.counter` across all entries; an outro with the total over the latest robot;
   - **the mix**: the voice matched to -16 LUFS (dialogue), the shots' sound (ambience, -27)
     and a generated music bed (`audio.generate`, -24) ducked under it, effects at -23, and a
     hard limiter at -1 dBFS on the mix;
   - optionally the YouTube 1080p export, with `<out>.chapters.txt` (intro, each year, outro —
     YouTube's rules: from 0:00, at least three, none under 10 s) for the description.

```
filmcraft-cli analyze robot --project p.fcproj --out robots.json
filmcraft-cli showcase --report robots.json --title "Every Robot I've Met" --voice clone:me \
    --plan --script-out plan.json            # review the entries and the script…
filmcraft-cli showcase --report robots.json --script-in plan.json --voice clone:me \
    --project p.fcproj --save --export ~/Movies/AutoCut/robots.mp4
```

A reviewed plan may hold `entries` alone (drop, merge or relabel robots, fix a shot's `best`
moment or `seen` times): the model then writes the words for exactly those.

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
- **MCP:** the `autocut_plan` (dry run) and `autocut_edit` tools, on both MCP backends
  (`filmcraft-cli mcp --demo` / `--bridge`). When a report has `needsSupervisor`, the calling
  agent plans that edit itself with the command tools — that is the supervision loop.

## Tests

`cargo test -p filmcraft-autocut` is fully offline: scripted models (`llm::FixedModel`), the demo
project headlessly, crafted transcripts through `transcript.set`. Live checks
(`tests/live_models.rs`) are opt-in: `AUTOCUT_LIVE=1` (the classifier, the VLM, and a complex
prompt handing off with its document).
