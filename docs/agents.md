# Agents

FilmCraft is built to be driven by AI agents, and largely built by them. Part 1 covers using
FilmCraft from an agent. Part 2 covers developing FilmCraft as an agent.

# Part 1: driving FilmCraft

## 1. MCP server

`filmcraft-cli mcp` serves MCP on stdio. It has two modes:

| Mode | Command | What it drives |
|---|---|---|
| Headless | `filmcraft-cli mcp --demo` or `--project p.fcproj` (neither = empty project) | an in-process engine session, no window |
| Bridge | `filmcraft-cli mcp --bridge 127.0.0.1:9876` | the running desktop app started with `filmcraft --control 9876` |

Headless mode also takes `--data-dir DIR` (user presets). Any other option is rejected (exit 2):
there are no folder-confinement flags such as `--automation-read-root`, so the server can read
and write wherever the user can.

**Claude Code.** The repository's `.mcp.json` registers both servers (`filmcraft` = bridge,
`filmcraft-headless` = demo). Both point at `target/release/filmcraft-cli`, so build it first:

```sh
cargo build --release -p filmcraft-cli -p filmcraft
./target/release/filmcraft --control 9876 &      # only for the bridge server
claude                                           # approve the project MCP servers when asked
```

To register the server by hand, or for another MCP client, use the absolute binary path:

```sh
claude mcp add filmcraft-headless -- /abs/path/filmcraft/target/release/filmcraft-cli mcp --demo
```

```json
{ "mcpServers": { "filmcraft": { "command": "/abs/path/filmcraft-cli", "args": ["mcp", "--bridge", "127.0.0.1:9876"] } } }
```

### From an installed release

The release packages ship `filmcraft-cli` alongside the desktop app, so no build is needed:

| Install | CLI |
|---|---|
| Windows (MSI) | `C:\Program Files\FilmCraft\filmcraft-cli.exe` by default (wherever you installed it otherwise), not on `PATH` |
| Linux (deb, rpm) | `/usr/bin/filmcraft-cli` |
| macOS | the separate `filmcraft-cli-<version>-macos-<arch>.zip` release asset (the `.app` holds only the desktop app) |

```sh
# Windows, default install folder
claude mcp add filmcraft -- "C:\Program Files\FilmCraft\filmcraft-cli.exe" mcp
# Linux, or macOS with the CLI unzipped onto PATH
claude mcp add filmcraft -- filmcraft-cli mcp
```

### Tools

| Tool | Modes | Purpose |
|---|---|---|
| `command_list` | both | every command: id, label, menu, shortcut, params, enabled now (`filter`, `enabled_only`) |
| `command_run` | both | run a command `{id, params}`; edits are undoable |
| `command_batch` | both | run several commands in order `{steps: [{id, params}], stop_on_error}` → `{completed, failed, results}` |
| `doc_inspect` | both | the project tree and the active sequence in one call (`project_inspect` + `sequence_inspect`) |
| `render_preview` | both | same as `render_frame` |
| `project_inspect` | both | bins and items with ids, types, durations; active sequence |
| `sequence_inspect` | both | the active sequence: tracks, clips (`start` / `end` and `sourceIn` / `sourceOut` in ticks, frames, `speed`, `reverse`: the clip plays its `sourceIn` to `sourceOut` stretch backward, `maintainPitch`: its sound keeps its pitch at that speed, `gainDb`), effects, transitions, markers (name and `comment`), playhead, selection |
| `media_import` | both | import files by absolute path (`paths`, an array; optional `bin`, `image_sequence`) |
| `render_frame` | both | PNG of the program frame at `seconds` (headless renders; bridge screenshots the Program monitor). Read-only: the playhead and selection are left as they were |
| `ui_inspect` | bridge | UI state: tool, workspace, panels, zoom, playback, fps |
| `ui_elements` | bridge | on-screen interactive elements with id, label and rect (`prefix` filter) |
| `ui_click` | bridge | click by `id` or `x`,`y`; `button`, `count`, `modifiers` |
| `ui_drag` | bridge | press–move–release between points or element ids |
| `ui_key` | bridge | key or shortcut: `Space`, `Cmd+K`, `Shift+Delete` |
| `ui_type` | bridge | type text into the focused field |
| `ui_screenshot` | bridge | PNG of the window or one `panel` |
| `ui_control` | bridge | call any control-channel method directly (`ui.set`, `ui.scroll`, `ui.timeline.locate`, …) |
| `showcase_analyze` | both | AutoCut showcase, step 1: find every clip that shows a `subject` (of `items`, filmed `since` a date) with the local vision model; `plan: true` goes on to step 2. A long job (`autocut.analyze` / `autocut.makeShowcase`) |
| `showcase_plan` | both | step 2: group the analysed clips into entries and write the voice-over, or write it for reviewed `entries` (`include`, `mergePrevious`, labels); returns the plan to review with the user (`autocut.showcasePlan`) |
| `showcase_build` | both | step 3: build the plan (kept, or `plan`) into a new sequence with ordinary engine commands; `export` (YouTube 1080p, chapters and QA report beside it), `upload` to Google Photos, `qa` (`autocut.showcaseBuild`) |
| `showcase_inspect` | both | what the session keeps: the analysis (a summary, or every clip with `report: true`), the plan, the last build, the latest showcase job |
| `sequence_auto_reframe` | both | a copy of a sequence in another shape (`aspect` 9:16, 1:1, 4:5, 16:9) whose clips follow their subject: the subject tracker (seeded by `subject` boxes or the salient region), a critically damped path, Motion keyframes; `motion` smooth/responsive, `fallback` letterbox/center, a `start_seconds`..`end_seconds` range (`sequence.autoReframe`; in place: `clip.autoReframe` via `command_run`) |
| `auto_redact` | both | hide clearly recognisable faces (at least `min_face_height`, 5 %, of the picture tall) and legible number plates behind a tracked blur: local YuNet / LPD-YuNet detectors, smoothed tracks, one crop per track shown to the local vision model so robots, mannequins, posters, screens and signs stay sharp (`verify`), `exclude` boxes; `targets` names logos, brands or signs to hide too (the vision model's grounding boxes every `target_every` s and at shot changes, the subject tracker between); one Gaussian Blur or Mosaic per clip with a keyframed mask per track; `dry_run` reports only (`clip.autoRedact` with `clips`, else `sequence.autoRedact`; the detectors via `redact.downloadModels` after the user agrees, `redact.status`; details in [crates/engine/src/redact.rs](../crates/engine/src/redact.rs)) |
| `shorts_make` | both | long video → shorts: transcript sentences scored for interest and hook (the local model rates the openings), the best `count` reframed to `aspect`, captioned, an optional title card; `export` (a folder, captions burned in), `upload_youtube` (private by default) (`autocut.shorts`) |
| `template_list` | both | the video templates (21 built-in starters and `<data dir>/Video Templates`): slots, variants, length; `template` for one in full (`template.list` — filters `aspect`, `category`, `kind`, `minMedia` / `maxMedia` — and `template.inspect`; `template.thumbnail` / `template.preview` draw them, `template.media` lists what can fill a slot) |
| `template_apply` | both | a new sequence from a video template: `variant` (`16:9`, `9:16`, `1:1`, `4:5`), `slots` (`{"photos": ["/a.jpg", …], "music": "/song.mp3", "title": "…"}`), `name`; cuts on the music's beats, Elements sounds synthesized, a `captions` slot transcribed, one undo step (`template.apply`; format in [crates/templates/README.md](../crates/templates/README.md)). In the app: Window ▸ Templates, the browser (`ui.set {templates: …}`). Authoring and sharing go through `command_run`: `template.markSlot`, `template.save` (into the user folder; `slots`, `captions`, `variants`, `posterSeconds`, `bundle`), `template.export` (one .zip), `template.install` (`path` or an https `url`, checked before anything is written), `template.qr`, `template.exportOtio` / `sequence.exportOtio` (OpenTimelineIO, slots in `metadata.filmcraft`) |
| `recipe_list` | both | recipes: parameterised chains of engine commands and AutoCut intents (built in: talking-head clean-up, vertical short from a long clip, apply a template and publish privately to YouTube — publishing off unless asked) and the user's own from `<data dir>/Recipes`; `recipe` for one in full with its steps as they would run (`recipe.list`, `recipe.inspect`; format in [crates/engine/src/recipes.rs](../crates/engine/src/recipes.rs)) |
| `recipe_run` | both | run a recipe with `params` as a job (progress, cancel): every step checked before it runs (intents planned by AutoCut's planner, commands checked against the registry and the recipe allowlist), jobs a step starts waited for, a failing step stops the run and says which; `dry_run` plans only (`recipe.run`). Saving and recording go through `command_run`: `recipe.save`, `recipe.delete`, `recipe.record` (`start`, then `stop` with a `name`: the edits made meanwhile become a recipe). In the app: Window ▸ Recipes (`ui.set {recipes: …}`) |

| `feedback_bundle` | both | write one folder to diagnose a problem from (`app.feedbackBundle`; `path`, default `<data dir>/Feedback/FilmCraft Feedback <date>`; `qa_report`): `manifest.json` (app version, build, OS, every file with its size), `system.json`, `history.json` (undo / redo labels, when steps were made / undone / redone, the last 300 commands), `events.json`, `jobs.json`, `perf.json` (the running app's `perf.stats` in bridge mode), `project.json` (`project.inspect`), `preferences.json`, `qa.json` (the last AutoCut QA report), `logs/` (the last MiB of `filmcraft.log`, the two before it and session logs) and `crashes/` (the newest five crash reports). The home folder becomes `~`, e-mail addresses `[email]`, tokens, passwords, API keys, cookies and other secrets `[redacted]`; token stores and the Keychain are never read; at most 8 MiB. Also Help ▸ Save Feedback Bundle… (`help.saveFeedbackBundle`, `menu.help.saveFeedbackBundle`) and `filmcraft-cli feedback [folder]` |

Typical loop: `project_inspect` / `sequence_inspect` → get ids → `command_run` → `render_frame` or
`ui_screenshot` → look at the result → `edit.undo` if it's wrong. When something goes wrong that
you cannot explain, `feedback_bundle` and read its files.

### Conventions

The server follows a few conventions that make it predictable for agents. They are not
FilmCraft-specific: the sibling craft apps' MCP servers can follow the same ones, so an agent that has
driven one app can drive the others.

- **Core tools.** `command_list`, `command_run`, `command_batch`, `doc_inspect` and `render_preview`
  work in both modes. The FilmCraft tools above stay as they are.
- **Titles and annotations.** Every tool has a `title` and the four MCP hints (`readOnlyHint`,
  `destructiveHint`, `idempotentHint`, `openWorldHint`), so a client can let the read-only tools
  (lists, inspects, renders, screenshots) run without asking and ask before edits. When you add a
  tool, set them in its `#[tool(...)]` attribute; a test checks every listed tool.
- **Strict arguments.** An unknown tool argument is a JSON-RPC `-32602` error that names it and the
  accepted ones (`unknown argument "filtr" for command_list; expected: enabled_only, filter`), not a
  silently ignored key. `command_run.params` goes to the command unchanged.
- **Errors.** A failing command is an `isError` result with the engine's message. A line that is not
  JSON gets a `-32700` parse error (id `null`) and the server keeps serving. A command that panics is
  reported as `internal error: …`, and the session stays usable.
- **Resources.** `filmcraft://document` (the project tree and active sequence, as `doc_inspect`) and
  `filmcraft://commands` (the command catalog, as `command_list`), both `application/json`. For MCP
  2026-07-28 clients such as current Claude Code, list and read results carry the `ttlMs` and
  `cacheScope` hints that revision requires.

### Long exports: progress and cancellation

`command_run {"id": "file.exportMedia", "params": {…, "wait": true}}` in headless mode blocks until
the file is written. It uses the MCP progress and cancellation utilities, so a client can show
progress and stop it:

- The export runs as the same background job `jobs.list` shows. The session is locked only for short
  polls, so other requests (`ping`, `doc_inspect`, `jobs.list`, …) are answered while it encodes.
- `file.exportMedia` with `wait: true`, an export job in `jobs.list` and each `export.queue.list`
  item carry `offlineMedia`: the offline items the export drew the placeholder for (see §2b), `[]`
  when none. The export itself still succeeds; deciding what to do about it is up to the caller.
- Every job in `jobs.list`, and every encoding item of `export.queue.list`, carries `etaSeconds`: the
  time left at the job's speed over the last 15 seconds, `null` until a second of progress has been
  measured (a loudness pass, a seek) and once it is done. The app shows it as `37% · 2:05 left` in the
  status bar, the Export queue and the Progress panel.
- With `"_meta": {"progressToken": T}` on the `tools/call`, the server sends
  `notifications/progress` `{progressToken: T, progress: <frames done>, total: <frames>, message:
  <job status>}` at most every 100 ms while frames advance. Without a token it sends none.
- `notifications/cancelled` for the request stops the encode at the next batch (`jobs.cancel`),
  deletes the partial output (the movie, or the frames and caption sidecar written so far) and sends
  no response, as the MCP cancellation utility asks.
- Without `wait` the command returns `{job, path, poll: "jobs.list", cancel: "jobs.cancel"}` at
  once: poll `jobs.list` for the job's `progress`, `etaSeconds`, `finished` and `result`, and stop it
  with `jobs.cancel` (`export.queue.list` lists the Export Queue only, not direct exports, #1544). Other exports (`file.exportFrame`, interchange formats, …) finish quickly and
  ignore the token.
- Bridge mode works the same way: the export runs as an app job (the app stays responsive and shows
  "Exporting… NN%" in its status bar) and the call returns when it is written, however long it
  takes. `filmcraft-cli --bridge … exec file.exportMedia … wait=true` blocks the same way.
- The other long commands run the same way with `wait: true`: `tts.cloneVoice`,
  `tts.downloadVoices`, `tts.downloadCloneModel`, `music.generateNeural` (ACE-Step text to music;
  the answer carries `music`: the placed item, clip and track), `music.downloadNeuralModel` (needs
  `consent: true` once the user agreed to the 6.3 GB download), `music.downloadSoundfont` (GeneralUser
  GS, 32 MB, for the procedural music's instruments: show the user `music.soundfonts` → `licenseText`
  and pass `consent: true` once they agreed), `music.preview` (renders the music `audio.generate`
  would place, for the user to hear first; returns at once when that exact preview is already
  rendered) and `photos.upload` (a file to the user's Google
  Photos library; progress in bytes). `googlephotos_upload {path, description?}` is that upload as
  its own tool; its answer, like the finished job's `jobs.list` entry, carries `upload` (the new
  media item: `mediaItemId`, `productUrl`, …). Cancelling an upload stops it between chunks and
  never deletes the file being uploaded. It needs a one-time `filmcraft-cli photos auth --upload`
  (see [crates/gphotos](../crates/gphotos/README.md)); until then `photos.upload` is disabled and its
  reason says so. With `album` (a title or id) the upload also goes into that album — the one
  FilmCraft made with exactly that title, created when missing — so the user can add the next
  iteration's material to it by hand. Albums are commands too, short calls that answer directly
  (not jobs): `photos.createAlbum {title, description?}` (idempotent by title),
  `photos.albums` (only the albums FilmCraft made: Google shows apps no others) and
  `photos.addToAlbum {album, mediaItemIds}` (only items FilmCraft uploaded). They need
  `filmcraft-cli photos auth --albums` once; until then they are disabled with that reason. To
  read a user-curated album back, use `googlephotos_pick {album: "<title>"}`: the Library API
  never returns items a user added by hand, so the picker's vision agent searches the picker
  for the album's exact title (the result is the album's items) and selects all of it. To make a showcase straight from such an album, `showcase_analyze {album,
  prompt, plan: true}` picks it, imports its videos once (files the project holds are reused)
  and analyses only those clips (`autocut.analyze` / `autocut.makeShowcase` with `album`).
- So do the AutoCut showcase jobs (`autocut.analyze`, `autocut.showcasePlan`, `autocut.makeShowcase`,
  `autocut.showcaseBuild`; the `showcase_*` tools; progress in clips analysed or entries built). Their
  answer, like the finished job's `jobs.list` entry, carries `showcase`: what the job made (the
  analysis' summary, the plan, the build). A job edits only through engine commands, which it sends
  to the session over its command port: the app runs them between frames, the headless server on a
  pump thread, so other requests are answered meanwhile and every edit is an ordinary, undoable one.
  The AutoCut prompt "make a video of every robot since 2020" (`autocut_edit`) starts the same job
  (see [crates/autocut](../crates/autocut/README.md#showcases-in-the-app)).

## 2. Control channel

`filmcraft --control 9876` (or `FILMCRAFT_CONTROL_PORT=9876`) listens on `127.0.0.1` only. Send one
JSON request per line and get one JSON reply per line. Full method table:
[control-protocol.md](control-protocol.md).

```jsonc
{"id":1,"method":"engine.commands"}
{"id":2,"method":"engine.execute","params":{"command":"file.openDemoProject"}}
{"id":3,"method":"engine.execute","params":{"command":"playhead.set","params":{"seconds":2}}}
{"id":4,"method":"engine.execute","params":{"command":"sequence.addEdit"}}
{"id":5,"method":"engine.execute","params":{"command":"effects.apply","params":{"effect":"Gaussian Blur"}}}
{"id":6,"method":"ui.elements","params":{"prefix":"tools."}}
{"id":7,"method":"ui.click","params":{"id":"tools.Razor"}}
{"id":8,"method":"ui.key","params":{"key":"Cmd+Z"}}
{"id":9,"method":"ui.screenshot","params":{"path":"/tmp/timeline.png","panel":"Timeline"}}
```

Replies look like `{"id":4,"ok":true,"result":{"cuts":2}}` or `{"id":7,"ok":false,"error":"…"}`.
Minimal client:

```python
import json, socket
s = socket.create_connection(("127.0.0.1", 9876)); f = s.makefile("rw")
def call(method, **params):
    f.write(json.dumps({"id": 1, "method": method, "params": params}) + "\n"); f.flush()
    return json.loads(f.readline())
print(call("engine.execute", command="sequence.inspect"))
```

Notes:

- Time is in ticks (254 016 000 000 per second). Commands also accept `seconds`, `frame` or
  `timecode`.
- A track parameter (`track`, `audioTrack`) is a track id or a name such as `"V1"` / `"A2"`.
  Leaving it out picks the command's default track; naming a track the sequence does not have is
  an "invalid parameters" error, never a different track.
  In `timeline.place`, `track` is where the picture goes and `audioTrack` where the sound goes;
  an audio track given as `track` places the sound alone, on that track.
  The clip lands on the frame grid: its start and its end (`time + duration`) each snap to the
  nearest frame.
- `sequence.inspect` reports `durationFrames` as the number of frames export renders: a sequence
  that ends partway through a frame counts that frame.
- `engine.execute` also runs UI-only commands (`tool.razor`, `playback.toggle`,
  `window.workspace.color`, `window.panel.<name>`).
- A command that works on the selection is disabled ("no clips selected", "select a clip in the
  Project panel") until something is selected, unless the call names its targets under a key the
  command documents (`clips` / `clip` for timeline clips, `items` / `item` for project items):
  `clip.replaceFromBin {"clips":[12],"item":4}` runs with nothing selected and leaves the
  selection alone. Only the selection condition is answered by the named targets; every other
  one (an open sequence, a filled clipboard, the right kind of clip) still applies, and ids of
  nothing do not count. `command_list` / `engine.commands` keep reporting the selection's state.
- Element ids come from the previous frame. If an element is missing right after a layout change,
  the app retries on later frames before giving up.
- `timeline.move` takes linked partners along while Linked Selection is on: moving a picture
  clip moves its sound by the same offset (pass `"linked": false` to move only the listed clips).
  It returns `{"moved": [ids]}`, every clip that moved.
- Without a window, `filmcraft-cli run script.jsonl` (lines of `{"id":"…","params":{…}}`) runs the
  same commands headlessly. See §2b for the rest of the CLI.

## 2b. Command-line interface

Every command is also one shell call away. Options go anywhere; output is JSON; exit status is 0 on
success, 1 when a command fails and 2 on a usage error. `filmcraft-cli help`, `filmcraft-cli --help`
and `filmcraft-cli -h` print the reference and exit successfully.
If the reader of stdout closes the pipe early (`filmcraft-cli commands | head`), the CLI drops the rest
of its output but still finishes the work, saves included, and the exit status still reports
failures; any other stdout write error is reported and makes the status 1.

```sh
filmcraft-cli commands razor                     # find ids (add --json for machine output)
filmcraft-cli describe timeline.razor            # one command: menu, shortcut, params, enabled now
filmcraft-cli --demo inspect sequence            # project / sequence as JSON
filmcraft-cli --project p.fcproj --save exec timeline.razor seconds=3.5
filmcraft-cli --project p.fcproj exec effects.apply '{"effect":"Gaussian Blur"}'
filmcraft-cli --project p.fcproj exec markers.exportCsv path=review.csv
filmcraft-cli --project p.fcproj --save import a.mov b.wav
filmcraft-cli --project p.fcproj export out.mp4  # format from the extension; waits for the job
filmcraft-cli --project p.fcproj export out --preset "YouTube 1080p Full HD" --start 0 --end 10
filmcraft-cli export --list-presets prores       # built-in + user presets (--data-dir for another library)
filmcraft-cli --project p.fcproj exec export.queue.add preset="Apple ProRes 422 HQ" path=renders/ start=true wait=true
echo '{"id":"file.newBin","params":{"name":"Selects"}}' | filmcraft-cli --project p.fcproj --save run -
filmcraft-cli --bridge 127.0.0.1:9876 exec window.workspace.color   # the running app
filmcraft-cli feedback /tmp/fc-feedback          # a redacted feedback bundle (logs, crashes, history…)
filmcraft-cli --bridge 127.0.0.1:9876 feedback   # the running app's, with its playback and GPU stats
```

`key=value` values are parsed as JSON when they can be (`3.5`, `true`, `[1,2]`), otherwise taken as
strings; dotted keys nest (`color.r=1`). `run` prints one JSON line per command
(`{"line","id","ok","result"|"error"}`) and stops at the first failure unless `--keep-going`.
`--save` writes back to `--project`; `--save-as path` writes elsewhere.

Media that is offline when it is needed (the file is missing, was made offline, or can't be read)
renders as the "media not found" placeholder. `export` and `render` then name each such item on
stderr (`item 8 a.mp4 (missing): /path/a.mp4`) and exit 1 after writing the output, so a moved or
unmounted media folder doesn't pass as a good render. `--allow-offline` accepts the placeholder
frames: the warning stays and the exit status is 0. The export result lists the same items as
`offlineMedia` (`[{item, name, path, status, error}]`, `status` as in `media.status`: `missing`,
`offline` or `unreadable`), and `[]` when every clip it drew was online; check before exporting with
`exec media.status` or `exec media.findMissing`.

`markers.exportCsv {path}` exports all markers in the active sequence, including colours hidden
in the UI and markers outside In/Out, sorted by start then id. The UTF-8 CSV has a header and CRLF
records; quotes, commas and embedded newlines are escaped. Columns are Sequence, Marker ID, Name,
Comment, Kind, Color, Start Timecode, End Timecode, Duration Frames, Start Ticks, Duration Ticks and
Frame Rate (exact numerator/denominator). Color uses the user's display label from Settings ▸
Labels. Timecodes include the sequence's start timecode and
drop-frame setting; End is exclusive, and a point marker's End equals Start. Duration Frames is
rounded down; tick columns preserve sub-frame timing (254016000000 ticks per second).
An empty sequence writes just the header. User text starting with a spreadsheet formula character
(`=`, `+`, `-`, `@`, including after whitespace) gets a leading apostrophe in the exported cell.
The export leaves the project, dirty flag and undo history untouched, works with locked tracks,
and uses the host's file service (atomic
on desktop, a download on web). It exports sequence markers only, not clip/source markers;
reports are limited to 100000 markers and a conservative 16 MiB output budget. Invalid timing or a
failed write reports an error. The menu opens the normal save dialog; cancelling writes nothing.

**A robot-showcase round** (the album's new clips, review notes into the plan with
`showcase-plan`, both languages built with privacy in one `showcase --languages en,ml` run, QA and
its privacy sheet, `sheet` / `frames` to look at clips and results, the upload, the 4K export) is a
handful of commands: [showcase-runbook.md](showcase-runbook.md).

## 3. Verifying UI work

For any UI change, look at the result:

1. `cargo run --release -p filmcraft -- --control 9876`
2. Drive the feature through the control channel or the MCP bridge: commands, then clicks and drags
   by automation id.
3. Assert on `ui.inspect`, `ui.elements` and `sequence.inspect`.
4. Take `ui.screenshot` (the whole window and the panel you changed) and open the PNG.

Screenshots you commit must follow [AGENTS.md](../AGENTS.md) §1: FilmCraft only, openly licensed
media, sidecar and `ATTRIBUTION.md` entry.

# Part 2: developing FilmCraft

## 4. Orientation, in this order

1. [AGENTS.md](../AGENTS.md): absolute rules (assets, clean-room, licences).
2. [CLAUDE.md](../CLAUDE.md): working instructions and non-negotiables.
3. [ROADMAP.md](../ROADMAP.md) for the summary, then [gaps.md](gaps.md) (the ranked shortfalls) first,
   [target-app-parity.md](target-app-parity.md) and [roadmap.md](roadmap.md) (milestones, current focus).
4. [architecture.md](architecture.md), then the README and tests of the crate you'll touch.
5. [contributing.md](contributing.md) (how to add things, gates) and [testing.md](testing.md).

Maintainers also keep planning notes in `plan/` (roadmaps, execution plans, status), tracked in the
repository; only `plan/premiere/`, the behaviour reference screenshots, stays local and gitignored.
Everything you need to contribute is also in the public docs above. Ask a maintainer for a task id.

## 5. Autonomous work loop

1. **Orient.** Pick the next task: the next unchecked task in the maintainer status file, or an open
   [gaps.md](gaps.md) entry. Prefer those gaps (stability, `.prproj`, camera media, measurement,
   hardware, Windows/Linux) over adding more checklist items. Read the relevant
   architecture section and crate README.
2. **Plan tests first.** Write down the acceptance test before writing code.
3. **Implement and test.**
4. **Verify.** Run all gates (`cargo xtask ci`). For UI work, run the app with `--control`, drive it
   and look at the screenshots (§3).
5. **Record.** Update the crate README (behaviour decisions, test results, limitations) and
   the progress docs when a milestone moves (ROADMAP.md, docs/roadmap.md, the relevant parity doc
   and gaps.md, with their timestamps). Commit as `M<n>.<k>: …`, then move on to the next task.

For parallel agents, use one git worktree and one `CARGO_TARGET_DIR` per agent, and keep each crate
with one owner ([contributing.md §5](contributing.md#5-parallel-work-several-agents-or-worktrees)).

Some resources on one machine can't be shared: the **browser** an agent drives with the real mouse
and keyboard, and the **GPU** a large model speaks on. FilmCraft takes an advisory lock on each
(`crates/locks`): the Google Photos picker agent and `photos tidy` hold `chrome`, cloned-voice
narration holds `gpu`, and a busy lock waits, then fails naming the holder. `filmcraft-cli locks`
shows who holds what. A session that drives the browser itself (Claude in Chrome, a script) holds
it for the duration: `filmcraft-cli locks hold chrome --minutes 30 --note "<what>"` in the
background, stopped when done. The OS releases a lock when its process exits, so a crash never
leaves one stale.

## 6. Decision rules

Don't block on choices you can make yourself:

- **Behaviour.** Match the publicly documented and observable behaviour of professional editors
  (help pages, published shortcut lists, using the app). Never inspect application internals
  ([AGENTS.md](../AGENTS.md) §2). If behaviour is undocumented, choose what editors most commonly
  expect and record the choice in the crate README.
- **Ask a human only about:**
  - licences outside the allowlist;
  - spending money;
  - publishing or pushing;
  - personal data;
  - destructive git operations;
  - anything AGENTS.md leaves unclear ("when in doubt, leave it out").

## 7. Definition of Done

A feature is done when:

- [ ] its command(s) are registered with label, menu path, shortcut (if any) and a parameter doc;
- [ ] engine tests cover it, including undo/redo and the disabled case;
- [ ] the UI is wired: menu, panel control and shortcut;
- [ ] every new interactive widget has an automation id;
- [ ] you have driven it through the control channel and reviewed a screenshot;
- [ ] all gates pass (`cargo xtask ci`), and new assets have sidecars;
- [ ] the crate README and the progress docs (ROADMAP.md, docs/gaps.md, the parity docs) are updated where relevant;
- [ ] it is committed with its task id.

### Reporting progress honestly

ROADMAP.md and [target-app-parity.md](target-app-parity.md) keep two numbers: **feature breadth**
(does it exist?) and **ready for real work** (does it hold up?). When you update either:

- say how a number was obtained: *measured* (a test, a diff, a benchmark you ran) or *estimated*;
- count an approximation, a stub or a partly wired setting as such, never as done;
- record known bugs and limitations in the crate README and [gaps.md](gaps.md), not only
  what works;
- benchmarks state the machine, the load average and before/after on the same build;
- don't raise a percentage without evidence a reviewer can rerun.
