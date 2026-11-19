# AutoCut for agents: making videos with the app and its local pipelines

For catalogue-driven edits, consistent revisions, cached RAEA review/master renders and the
standard delivery gate, also read [local-pipeline-upgrades.md](local-pipeline-upgrades.md).

This guide is for an agent (Claude Code, Codex, ChatGPT, or another) asked to make or change a video
with AutoCut, this fork of FilmCraft. It covers which tool fits a request, how to drive it, how to
check the result before the user sees it, and what earlier rounds learned the hard way.

It is a playbook, not the reference. For the reference, see:

- [agents.md](agents.md) Part 1: the MCP tools, the control channel and CLI conventions.
- [control-protocol.md](control-protocol.md): every control method.
- `filmcraft-cli --help`: every subcommand and flag, about 800 lines.
- `filmcraft-cli describe <command id>`: one command's parameters.
- The pipeline docs in [§15](#15-further-reading).

If you change AutoCut's own code, [AGENTS.md](../AGENTS.md) and [CLAUDE.md](../CLAUDE.md) apply as
well: never crash, the quality gates, and the commit rules.

## Contents

0. [Rules](#0-rules)
1. [This Mac](#1-this-mac)
2. [Three ways to drive it](#2-three-ways-to-drive-it)
3. [Which pipeline for which request](#3-which-pipeline-for-which-request)
4. [Getting the media](#4-getting-the-media)
5. [The one-command pipelines](#5-the-one-command-pipelines)
6. [Building an edit by hand](#6-building-an-edit-by-hand)
7. [Checking your work](#7-checking-your-work)
8. [Export](#8-export)
9. [Delivery](#9-delivery)
10. [Narration and Malayalam](#10-narration-and-malayalam)
11. [360° footage](#11-360-footage)
12. [The live app](#12-the-live-app)
13. [Gotchas](#13-gotchas)
14. [The user's taste](#14-the-users-taste)
15. [Further reading](#15-further-reading)

---

## 0. Rules

Read these before you touch anything.

- **Review first, final after approval.**
  - Build a review at 1080p and upload it to the user's Google Photos.
  - Export the 4K final only after the user approves the review.
  - Uploading to the user's own Google Photos library is the normal way to deliver a video they
    asked for.
- **Ask before anything public.**
  - YouTube uploads are private unless the user says otherwise in their own words.
  - Instagram posts, public or unlisted links, and shared albums need the user's yes each time.
- **Never delete for good.**
  - Use `trash`, not `rm`.
  - Keep earlier exports and projects.
  - When you rebuild a project, trash the old copy first.
- **Never read or print secrets.** Secrets include:
  - the Keychain items ("FilmCraft …");
  - `tokens.json`, `client_secret.json` and the Instagram app secret;
  - the control token file.

  The user completes sign-ins themselves: `photos auth`, `youtube auth`, `instagram auth`.
- **Consent each time.**
  - Model downloads and OAuth consent screens need the user's OK.
  - Commands that take `consent: true` get it only after the user agreed.
- **Shared resources.**
  - Run `filmcraft-cli locks` first to see who holds what.
  - Hold the `chrome` lock while you drive Chrome yourself:
    `filmcraft-cli locks hold chrome --minutes 30 --note "<what>"`, in the background; stop it
    when done.
  - The picker, `photos locations` and `insta360 download` take the lock themselves.
  - Never run a picker session while another session is driving Chrome.
  - Never close a Google Photos tab you didn't open.
- **Privacy.**
  - Bystanders' faces and legible number plates get blurred with `redact`, the
    `auto_redact` tool or `--redact`.
  - A showcase's QA refuses a YouTube upload while unblurred faces are listed.
  - The user's own office or family footage is the user's call: say what is visible and let them
    decide.
- **Media never goes into git.** Projects and exports live in `~/Movies/AutoCut/<Video name>/`.
- **Don't edit a project the running app has open.** Two writers to one `.fcproj` overwrite each
  other. Either work headless on your own `.fcproj`, or drive the open project through the bridge
  (§2).
- **Git**, if you commit app changes:
  - Push only to the `autocut` remote (`main` and `main:feat/autocut`), never to `origin`, which is
    upstream.
  - Push only after the full `cargo xtask ci` and with the user's OK.

## 1. This Mac

| What | Where / value |
|---|---|
| CLI | `.mcp.json` and production workflows use `target/release/filmcraft-cli`, installed with the tested AutoCut app. `target/agent-main/release/filmcraft-cli` is the development candidate. The [November upstream/pipeline upgrade](../plan/pipeline-upgrades-2026-11.md) retains the AI features; the build of 19 November 2026 exposes 921 commands. Package and verify the candidate before replacing production; keep rollback copies. Builds still report `0.12.0`, so inspect the packaged commit and command catalogue too. |
| Building your own CLI | `CARGO_TARGET_DIR=target/agent-<name> cargo build --release -p filmcraft-cli`, one target folder per worktree, never shared. |
| The app | `target/macos-package/AutoCut.app`, control port **9876**: `open -n --env FILMCRAFT_CONTROL_PORT=9876 target/macos-package/AutoCut.app` |
| RAEA | `~/.raea/versions/0.1.0/bin/raea` for local custom animations, 3D scenes and effects. [Render and import workflow](raea-workflow.md). |
| Generation tools | The user's own local models, listed in `~/Library/Application Support/FilmCraft/generation/providers.json`: `h3-image` (MiniMax-H3-Image, pictures and edits), `ltx-video` (LTX-2.3, clips up to 10 s), `sa3-sfx` and `sa3-music` (Stable Audio 3 small, up to 30 s). `filmcraft-cli exec generate.providers` says which are installed. Never edit that file or download a model without the user's OK. [generation.md](generation.md). |
| Projects and exports | `~/Movies/AutoCut/<Video name>/`: the `.fcproj`, the exports, and a `build/` folder (scripts, logs, a `README.txt` recipe) |
| Data folder | `~/Library/Application Support/FilmCraft/`: `google-photos/cache` (picked originals with `.json` date sidecars), `scene-cache`, `redact-cache`, `autocut/analysis-cache`, voices, models, control tokens |
| Ollama models | `qwen2.5vl:7b` (picker and screen agents, clip analysis, redaction checks), `qwen3:8b` (scripts, Malayalam), `phi4-mini` (the AutoCut prompt classifier), `moondream` |
| Engine models | Downloaded into the data folder. Check them with `filmcraft-cli scenes status`, `exec redact.status`, `exec transcript.models`, `exec tts.clonedVoices`, `exec music.neuralModels`, `exec depth.status`, `exec upscale.status`, `exec interpolate.status` |
| ffmpeg | Installed through Homebrew solely for development test oracles. Editing/export uses native `probe`, `frames`, `sheet`, `render` and `publish check`; FFmpeg is never linked, bundled or used as the product pipeline. |
| Google Photos | Signed in for picking, upload and albums. The Cloud project is in Testing mode, so the sign-in lapses after 7 days; the user then re-runs `filmcraft-cli photos auth --upload` (or `--albums`). |
| YouTube | `filmcraft-cli youtube channels` lists the connected channels and the default. Uploads from this unaudited project stay private. |
| Instagram | Connected, but Instagram Login can't do Meta's resumable upload. `instagram publish` needs `--video-url` (a public https address). |
| The user's voice | `clone:mohesh-natural`, cloned from the user's own videos with their consent: VoxCPM2 for English, IndicF5 for Malayalam. Version it with `voice versions` / `voice promote` / `voice rollback`. |

Checks before you start:

```bash
filmcraft-cli locks
```

```bash
ollama list
```

```bash
filmcraft-cli photos albums
```

```bash
filmcraft-cli --bridge 127.0.0.1:9876 exec project.inspect
```

The last one shows whether the app is up and whether its project has unsaved changes.

## 2. Three ways to drive it

Every edit is an engine command: an id plus JSON params. The CLI, the MCP server and the control
channel all run the same commands, so pick the transport that fits the job.

| Transport | Use it for | How |
|---|---|---|
| **CLI, headless** | one-off commands and the one-command pipelines (§5) | `filmcraft-cli --project p.fcproj [--save] exec <id> key=value …` or a subcommand (`quick`, `direct`, `export`, …) |
| **MCP server, headless** | long scripted builds: one engine session, many commands, ids that stay valid | `filmcraft-cli --project p.fcproj mcp`, then `tools/call command_run {id, params}`, and finally `file.save` |
| **Bridge to the running app** | when the user wants to watch, or for UI work | `filmcraft-cli --bridge 127.0.0.1:9876 exec <id> …`, `… call ui.screenshot path=/tmp/x.png`, or `filmcraft-cli mcp --bridge 127.0.0.1:9876` |

In Claude Code, `.mcp.json` registers two servers:

- `filmcraft`: the bridge to the running app.
- `filmcraft-headless`: the **built-in demo project**, not your project. For your own project,
  start your own server with `--project`, as in the client below.

### A minimal MCP client (Python)

Every hand-built edit used this client: `build/fc.py` in a video's folder.

```python
"""A tiny client for filmcraft-cli's MCP server: one headless engine, commands one at a time."""
import json, subprocess, itertools

CLI = "/Users/mohesh/Desktop/dev_work/Autocut/target/agent-main/release/filmcraft-cli"

class FC:
    def __init__(self, project):
        self.p = subprocess.Popen([CLI, "--project", project, "mcp"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.ids = itertools.count(1)
        self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "build", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, m):
        self.p.stdin.write(json.dumps(m) + "\n"); self.p.stdin.flush()

    def _rpc(self, method, params):
        i = next(self.ids)
        self._send({"jsonrpc": "2.0", "id": i, "method": method, "params": params})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("mcp server closed")
            m = json.loads(line)
            if m.get("id") == i:          # skip progress notifications
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m["result"]

    def tool(self, name, args):
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        if r.get("isError"):
            raise RuntimeError(f"{name}: {text[:500]}")
        try:
            return json.loads(text)
        except ValueError:
            return text

    def run(self, cmd, params=None):
        return self.tool("command_run", {"id": cmd, "params": params or {}})

    def close(self):
        self.p.stdin.close(); self.p.wait(timeout=60)
```

### Conventions

- **Time.**
  - Time is in ticks: **254 016 000 000 per second**.
  - Most commands also take `seconds`, `frame` or `timecode`.
  - In scripts, snap every time to the frame grid (`round(s * fps) / fps`), then convert with
    `round(s * T)`. Unsnapped times leave one-frame gaps.
- **Ids.**
  - Item ids come from `inspect project`.
  - Clip ids come from `inspect sequence` or from what `timeline.place` returns (`clips[0]`).
  - Tracks are named `"V1"`, `"A2"`, and so on.
- **Parameters.**
  - `filmcraft-cli describe <id>` shows a command's params.
  - `exec effects.list query=<name> detail=true` shows an effect's params (`paramInfo`: id,
    range, default).
  - `exec effects.list kind=VideoTransition` lists the 258 transitions.
- **Long jobs.** Exports, `clip.autoReframe`, music, narration, uploads and analysis take
  `wait: true`. Without it you get a job id; poll `jobs.list` and stop it with `jobs.cancel`.
- **Saving.** Headless edits vanish unless you pass `--save` / `--save-as`, or run `file.save` over
  MCP.
- **Undo.** Every edit is undoable (`edit.undo`). A pipeline's edits are ordinary commands too.
- **Offline media.** A missing file renders as a placeholder, and `export` / `render` then exit 1.
  Fix the path; don't reach for `--allow-offline`.

## 3. Which pipeline for which request

| The user asks for… | Reach for | Notes |
|---|---|---|
| "a reel / short of my trip, these clips, this album" | `direct` (normal and 360° clips, local model directs on the beat) | `--dry-run` shows the plan first; `--revise NOTES --from <out>.report.json` for the next round |
| "a quick Short from this folder" (template feel) | `quick` | a template chosen and filled by the shots' AI hints |
| "every robot / car / X I've filmed" | `showcase` (`--album` or `--report`) | the robot-showcase runbook, [showcase-runbook.md](showcase-runbook.md); a full narrated 16:9 show |
| a trailer of a finished showcase | `trailer` | vertical, cut on the beat, end card |
| a long talking video cut into Shorts | `shorts` | reframed, captioned, hook-scored |
| raw talking takes cleaned up | `rough-cut` | pauses, fillers and retakes out |
| "use template X with these photos" | `template apply` | `template list` first |
| a routine edit in words ("add a cross dissolve to every cut") | `autocut '<prompt>' --dry-run`, then without `--dry-run` | `needsSupervisor` means plan it yourself with commands |
| a repeatable chain | `recipe_run` / `exec recipe.run` | `recipe.record` turns edits into a recipe |
| blur faces and plates | `redact` | `--dry-run` lists tracks first |
| clean up a voice recording file | `audio enhance` / `audio isolate` | DeepFilterNet with `--download`, else DSP |
| a picture, b-roll clip, sound or music that does not exist yet | `exec generate.image` / `generate.video` / `generate.sound` | the user's local tools (§6.9); `wait: true`; one at a time |
| "fill this empty stretch" | `exec generate.fillGap {track, atSeconds}` or `{all: true}` | continues the frame before the gap; `uncovered` says what is left |
| "add a boat / a character / snow to this shot" | `exec generate.augment {instruction, atSeconds, references?, animate?}` | a new picture laid over the clip; it does not track a moving shot |
| a specific vision: mixed stills and clips, hand-timed beats, titles, its own look | **build it by hand** (§6) | full control, rebuildable from a script |

If a pipeline gets you most of the way, use it, then fix the rest with commands on its sequence.
When a manual step repeats across rounds, make it an app feature (§14).

## 4. Getting the media

### Google Photos, through the picker

Google lets apps read only what they uploaded, so the user's own library is reached through
Google's picker. A local vision agent drives the picker: Qwen2.5-VL through Ollama, moving the real
mouse in Chrome. It needs Screen Recording and Accessibility for the terminal.

A day's pictures **and** clips:

```bash
filmcraft-cli --project "P.fcproj" photos pick --auto "October 30 2026" --all --photos --since 2026-10-30 --save
```

Picking from an album the user curated (the best seed for a big video):

```bash
filmcraft-cli --project "P.fcproj" photos pick --album "Exact Album Title" --save
```

Many searches in one session, then re-importing items already downloaded:

```bash
filmcraft-cli --project "P.fcproj" photos pick --queries searches.txt --all --save
```

```bash
filmcraft-cli photos cached
```

```bash
filmcraft-cli --project "P.fcproj" --save photos add <mediaItemId> <mediaItemId>
```

Watch for these:

- **`--save`.** Without it, the imports never reach the `.fcproj`.
- **`--photos`.** Without it, a search is typed as "<query> videos" and only video tiles are
  clicked, which misses a day's photos.
- **`--all`.** It selects every result: date searches by the picker's day groups, others by
  first-tile-then-Shift-click.
- **Size limits.** One session holds up to 20 searches and 2,000 items. The rest goes to
  `<data dir>/google-photos/left-searches.txt`.
- **Dates.** Creation times are UTC. If a late-evening or early-morning item is missing, widen
  `--since` by a day.
- **Tabs.** The agent closes its tab. If tabs pile up, `filmcraft-cli photos tidy` closes only
  picker tabs.
- **Locations.** Google strips them from downloads. `photos locations --album "<title>"` reads
  them from the Info panel with the vision agent, for lower thirds.

### Local files and Insta360

```bash
filmcraft-cli --project "P.fcproj" --save import ~/Movies/a.mov ~/Pictures/b.jpg
```

```bash
filmcraft-cli insta360 list --since 2026-10-01
```

```bash
filmcraft-cli --project "P.fcproj" --save insta360 import VID_093 VID_096
```

### Look before you plan

- **What came in:** `filmcraft-cli --project P.fcproj inspect project` lists the items: id, name,
  type, duration.
- **What it looks like:** `filmcraft-cli --project P.fcproj sheet --items 3,4,5:0-8 --n 6 --out
  sheet.jpg`. Open it and look.
- **Picture size:** `filmcraft-cli probe <file>` gives `video.width` / `video.height`, which
  `inspect` doesn't carry. You need the sizes for framing math (§6.4).
- **Leave out what doesn't belong:** screenshots, earlier exports, another outing's clips. Say what
  you left out.
- **Keep a pristine copy.** Once the picks are in, copy the project to
  `build/<Name> - picks.fcproj`, so every rebuild starts clean.

## 5. The one-command pipelines

Each writes its result into the project (`--save` / `--save-as`). With `--export out.mp4`, each
also writes the video plus side files:

| Side file | What it is |
|---|---|
| `<out>.report.json` | what was chosen and why |
| `<out>.sheet.jpg` | frames with their times |
| `<out>.qa.json` / `<out>.qa.png` | QA: gaps, black frames, loudness, privacy |

Read those files before showing anything.

| Pipeline | Example | What it does |
|---|---|---|
| `direct` | `filmcraft-cli direct ~/Movies/trip --brief "a 30 s reel of our Oman trip" --aspect 9:16 --seconds 30 --export trip.mp4 --save-as trip.fcproj` | Indexes the shots and ranks every moment. The local model then directs the hook, the order, the length in beats and each shot's camera. 360° shots get tiny planets, orbits and whips; normal shots get push-ins and ramps. Cut on generated music's beats (or `--music`), with colour matched, captions and a −14 LUFS mix. `--style subtle\|energetic\|cinematic`, `--sounds`, `--light-effects`, `--redact`. The per-project taste is `exec autocut.taste`. |
| `quick` | `filmcraft-cli quick ~/Movies/expo --topic "Robots at the expo" --seconds 30 --export expo.mp4 --save-as expo.fcproj` | Fills a template from the shots, with the model's text, captions and redaction. [quick-templates.md](quick-templates.md) |
| `showcase` | `filmcraft-cli showcase --album "Robot showcase" --prompt "every robot since 2020" --voice clone:mohesh-natural --export robots.mp4 --project robots.fcproj --save` | Analyses the album's clips, plans and narrates the show, then builds, QA-checks and exports it. `--plan` stops after the plan for review; `--script-in` takes the reviewed plan; `--languages en,ml` builds both languages. [showcase-runbook.md](showcase-runbook.md), [video-pipeline.md](video-pipeline.md) |
| `trailer` | `filmcraft-cli trailer --project robots.fcproj --plan plan.json --report build.json --export trailer.mp4` | A vertical trailer of a finished show. |
| `shorts` | `filmcraft-cli shorts talk.mp4 --count 3 --platform reels --export shorts/` | Transcript-led Shorts, reframed to the speaker, with Bold captions. |
| `rough-cut` | `filmcraft-cli rough-cut take1.mov take2.mov --language en --export clean.mp4 --project clean.fcproj --save` | A clean talking edit, cut between words. |
| `template apply` | `filmcraft-cli --project p.fcproj --save template apply builtin:photo-beat-slideshow --variant 9:16 --slot photos=a.jpg --slot photos=b.jpg --slot music=song.mp3` | A sequence from a template. |
| `autocut` | `filmcraft-cli --project p.fcproj autocut "add a cross dissolve between every clip" --dry-run` | Natural-language routine edits. |
| `redact` | `filmcraft-cli --project p.fcproj redact --kinds faces,plates --dry-run`, then again with `--save` | Tracked blurs. `--target "logo"` hides named things too. |

Showcases and direct edits keep caches by file content: analysis, scenes, redaction, spoken lines.
A rebuild after review notes redoes only what changed, so iterate freely.

## 6. Building an edit by hand

Use this when the user has a specific picture in mind: mixed photos and clips, a beat grid without
music, hand-placed titles, a particular look. The pattern below made the "Halloween at the Office"
reel (24 s, 9:16, approved first time). Its scripts are in
`~/Movies/AutoCut/Halloween Reel/build/` (`build.py`, `fc.py`, `frames.py`, `README.txt`).

### 6.1 Folder and rebuild loop

```text
~/Movies/AutoCut/<Name>/
  <Name>.fcproj                     the built project (rebuilt from scratch each round)
  <Name> - review (1080x1920).mp4   review exports
  <Name> (2160x3840).mp4            the approved final
  build/
    <Name> - picks.fcproj           pristine: the picks only
    build.py  fc.py                 the build script and the MCP client
    README.txt                      the recipe: every command that made it, in order
    *.log                           pick, export and upload logs
```

Each round:

1. `trash "<Name>.fcproj"`
2. `cp "build/<Name> - picks.fcproj" "<Name>.fcproj"`
3. `python3 build/build.py "<Name>.fcproj"`
4. Check the frames (§7).
5. Export (§8).

A build script that logs failures and carries on, instead of dying on the first one, finishes and
then tells you everything that went wrong at once.

### 6.2 Sequence and beat grid

```python
T, FPS = 254016000000, 30
seq = f.run('file.newSequence', {'name': 'My Reel', 'width': 1080, 'height': 1920, 'fps': FPS,
                                 'video': 5, 'audio': 3})['sequence']
B = 0.5                                   # one beat at 120 BPM
fr = lambda s: round(s * FPS) / FPS       # snap to frames
```

Time every shot in beats, even with no music. When the user adds their own track later, most
tempos land on the cuts.

### 6.3 Placing shots

```python
r = f.run('timeline.place', {'item': item, 'track': 'V1', 'time': round(a * T),
                             'duration': round((b - a) * T), 'audio': False})
clip = r['clips'][0]
```

- **A video clip:** add `'sourceIn': round(src * T)`. For a speed change, place
  `duration × speed/100 + one frame` of source, then run
  `clip.speedDuration {clips: [clip], speed: 180}`.
- **Its sound:** `'audio': False` drops it. Leave it on when the clip's own sound matters.

### 6.4 Framing stills in a vertical frame

`clip.kenBurns {clips: [clip], move: "none", fit: "fill"}` fills the frame and turns off
**Scale to Frame Size**. From then on, Motion's `scale` is relative to the picture's own pixels
(100 % = one picture pixel per frame pixel), and `position` is in sequence pixels (centre
`[540, 960]` at 1080×1920).

For an automatic push toward a subject:

```text
clip.kenBurns {clips: [clip], subject: [x, y, w, h], move: "in"}
```

The box is in 0..1 of the picture.

For a window you choose (a slow push from a wide view to a detail), key Motion yourself:

```python
def window(W, H, cx, cy, hf, rot=0, FW=1080, FH=1920):
    """Motion scale (%) and position that show a FW:FH window of a WxH picture: its centre (cx, cy)
    and its height hf, all as shares of the (turned) picture. The window stays inside the picture."""
    if rot in (90, -90):
        W, H = H, W
    s = max(FH / (hf * H), FW / W, FH / H)          # always cover the frame
    hw, hh = FW / s / 2 / W, FH / s / 2 / H
    cx, cy = min(max(cx, hw), 1 - hw), min(max(cy, hh), 1 - hh)
    return s * 100, [FW / 2 - (cx - 0.5) * W * s, FH / 2 - (cy - 0.5) * H * s]

for t, w in [(a, start_window), (b - 1 / FPS, end_window)]:
    sc, pos = window(W, H, *w, rot)
    f.run('effects.setParam', {'clip': clip, 'effect': 'motion', 'param': 'scale', 'value': sc,
                               'time': round(t * T), 'keyframe': True})
    f.run('effects.setParam', {'clip': clip, 'effect': 'motion', 'param': 'position', 'value': pos,
                               'time': round(t * T), 'keyframe': True})
```

How to use it:

- **Sideways photos:** a photo shot from above can come in sideways. Set Motion `rotation` to −90
  or 90, and give `window()` the same `rot`; the window is then in the turned picture's
  coordinates.
- **Finding the subject:** get each picture's size from `probe`. Find the subject by looking at a
  `sheet` or a `media.frame` of the item.
- **Pacing:** a push of 10–20 % over a shot reads as calm, 30 % or more as dramatic. Short
  "punch" shots of one beat can sit on a tight window.

### 6.5 Video in a vertical frame

| Situation | Command |
|---|---|
| The subject moves around the frame | `clip.autoReframe {clips: [c], motion: "smooth", fallback: "center", analysisFps: 10, wait: true}` follows it with the subject tracker |
| A wide clip that shouldn't be cropped | `clip.autoReframe {…, whole: true}` shows it whole over a blurred, dimmed copy |
| A fixed crop on one detail (a gauge, a screen) | `clip.cameraTrack {clip, target: "box", box: [x, y, w, h], at: s, zoom: "close", wait: true}` |
| A whole sequence to another shape | `sequence.autoReframe {aspect: "9:16", fallback: "blur"}` makes a copy |

You can also build "whole over blur" by hand, as the Halloween reel did for a portrait photo:

1. Place a second copy of the shot on V2: `clip.kenBurns fit: "full"`, scale 1080/W·100.
2. Push the V1 copy past the frame and blur it: `gaussian_blur` with `blurriness` 45 and
   `repeat_edge` on.
3. Darken the V1 copy: `lumetri` `exposure` −0.7.

### 6.6 Colour and look

Start with a base grade on the shots. Leave out the blurred fill copies, which keep their own
darkening:

```text
clip.autoColor {clips: [...], strength: 0.6, match: true, look: "teal", lookStrength: 35}
```

`look` is one of `none`, `punchy`, `soft`, `warm`, `cool`, `golden` or `teal`.

Then lay a look over everything with an adjustment clip on a top track:

```python
adj = f.run('timeline.newAdjustmentClip', {'track': 'V3', 'startSeconds': 0, 'seconds': END,
                                           'effects': ['lumetri', 'vignette', 'cr_film_grain'],
                                           'name': 'Look'})
for e, k, v in [('lumetri', 'exposure', -0.6), ('lumetri', 'contrast', 25), ('lumetri', 'temperature', -18),
                ('vignette', 'amount', -50), ('vignette', 'feather', 60), ('cr_film_grain', 'amount', 16)]:
    f.run('effects.setParam', {'clip': adj['clip'], 'effect': e, 'param': k, 'value': v})
```

Other tools for colour:

- `lumetri.applyMatch` matches one shot to a reference.
- `lumetri.applyPreset` / `lumetri.setLook` apply presets and LUTs.

Video effects used often (the full list is `exec effects.list kind=Video`):

- `gaussian_blur`, `lumetri`, `vignette`
- `cr_film_grain`, `cr_chromatic_aberration`, `cr_glitch`, `cr_vhs`, `cr_tech_hud`, `cr_light_sweep`
- `camera_shake`, `rgb_split`, `light_leaks`
- `warp_stabilizer`, `depth_parallax`, `motion`

Apply one with `effects.apply {clips, effect}`, then set it with `effects.setParam`.

### 6.7 Transitions

```text
sequence.applyVideoTransition {clip, effect: "flash", frames: 6, edge: "in"}
```

Use `edge: "out"` on the last clip to fade out.

Ids that worked:

| Feel | Transitions |
|---|---|
| Calm | `cross_dissolve`, `film_dissolve`, `dip_to_black` |
| Punchy | `flash`, `cam_punch_in`, `cam_whip_pan`, `whip`, `glitch`, `glitch_rgb_split` |
| Spooky or retro | `flicker`, `tv_power`, `crt_glitch`, `gl_tvstatic` |
| Light | `light_leak`, `lf_exposure_flash` |

`exec effects.list kind=VideoTransition` lists all 258.

Related commands:

- `transitions.addSound {transitions, kind, gain}` puts a sound under transitions.
- `transitions.snapToBeats` moves transitions onto the music's beats.
- `transitions.applyToAll {preset, duration}` puts one transition on every cut.

For a custom transition or effect, render transparent PNG frames with RAEA and place the
image sequence above the edit. For a custom animation, 3D shot, or full scene, render a
normal MP4 and place it on the picture track. See the [local RAEA workflow](raea-workflow.md).

### 6.8 Titles

```python
r = f.run('graphics.newText', {'text': 'HAPPY\nHALLOWEEN', 'position': [540, 820], 'size': 132,
                               'seconds': 3.4, 'track': 'auto', 'time': round(t * T)})
tc = r['clip']
f.run('graphics.set', {'clip': tc, 'layer': 0, 'props': {
    'font': 'Chalkduster', 'size': 132, 'align': 'center', 'tracking': 10, 'leading': 10,
    'fill_color': '#ff8a1f', 'stroke': True, 'stroke_width': 6, 'stroke_color': '#000000',
    'shadow': True, 'position': [540, 820]}})
f.run('graphics.animateText', {'clip': tc, 'layer': 0,
    'in':  {'kind': 'glitch', 'unit': 'line', 'stagger': 0.1, 'duration': 0.4},
    'out': {'kind': 'fade', 'unit': 'line', 'stagger': 0, 'duration': 0.25},
    'loop': {'kind': 'flicker', 'unit': 'line', 'amount': 60}})
```

- **Position.** For point text, `position` is the first baseline's alignment point. For
  paragraph text, pass `box: [w, h]` and `position` becomes the box's top-left corner.
- **Width.** At 1080 wide, a 96–132 px display font fits about 10–14 characters a line. Break
  lines with `\n`, then **look at the frame at full size** for clipped letters.
- **Animations.**
  - `in` / `out` kinds: `fade`, `typewriter`, `pop`, `slideUp` (and the other slides), `blurIn`,
    `scaleUp`, `drop`, `glitch`.
  - `loop` kinds: `wave`, `bounce`, `shake`, `pulse`, `flicker`.
  - `unit` is `letter`, `word` or `line`.
- **Use `unit: "line"` with `stagger: 0` for `out`.** A letter-staggered fade-out starts early,
  and letters vanish while the title should still read.
- **Fonts.** Any installed font works (macOS ships Chalkduster, DIN Condensed, Futura and others);
  Inter is the fallback. Malayalam in titles needs the craft-fonts build (see CLAUDE.md).
- **Platform safe zones.** `exec sequence.checkSafeZones platform=reels` lists the text that
  reaches into Reels', Shorts' or TikTok's interface.

Ready-made animated text and stickers: `exec elements.list kind=text`, then
`elements.insert {id, seconds, track}`.

### 6.9 Sound, music and voice

**Sound effects** are synthesized locally, with no licence questions:

```python
r = f.run('audio.generate', {'kind': 'thump', 'seconds': at, 'track': 'A2', 'intensity': 0.6, 'seed': 100 + i})
f.run('clipMixer.set', {'clip': r['clip'], 'effect': 'volume', 'param': 'level', 'value': -10})
```

- Kinds: `whoosh`, `impact`, `riser`, `swish`, `chime`, `shutter`, `pop`, `thump`, `drone`,
  `engineRev`, `engineLaunch`, `engineFlyby`.
- A different `seed` gives a different variation.
- Levels that sat well under the user's own music: pop −16 dB, thump −10 dB, impact −12 dB.

**Music.** Skip all of it when the user says they will add their own.

- **Procedural:** `audio.generate {kind: "music", mood, genre, bpm, duration, time/seconds, track}`.
  Preview it with `music.preview`.
- **Neural:** ACE-Step, through `music.generateNeural {prompt, seconds, bpm, key, atSeconds, track, wait: true}`.
  The 6.3 GB model needs the user's consent to download.
- **The user's own track:** import it, then `media.beats {item}` finds its beats.

**Generated pictures, clips and sounds.** When `generate.providers` lists installed tools (the
user's own local models; [generation.md](generation.md)):

- `generate.image {prompt, references?, fromPlayhead?, transparent?, place?, seconds?, wait: true}`.
- `generate.video {prompt, image?: item|path|"playhead", seconds, atSeconds, track, wait: true}`.
- `generate.sound {prompt, kind: "sfx"|"music", seconds, atSeconds, wait: true}`.
- `generate.fillGap {track, atSeconds | all: true, wait: true}` fills an empty stretch of a video track.
- `generate.augment {instruction, atSeconds, animate?, wait: true}` changes what the frame shows.
- Common parameters: `provider` (default: the first installed tool of the kind), `seed`, `dir`
  (default a `Generated` folder beside the project), `place`, `track`, `atSeconds`.
- One generation at a time; each takes the GPU lock itself, so never hold a GPU lease of your
  own around it (it would wait on you for ever). `jobs.cancel {job}` stops one.
- Times on this Mac: a picture 40 s, an edit from a frame or reference about 4 minutes, a 3 s
  clip about 50 s, a sound a few seconds. Over MCP these are long jobs: `command_run` with
  `wait: true` returns when the result has landed.
- The result is imported and placed as one undo step, scaled to fill the frame, on the next
  free track when the asked one filled up. `generate.history` lists what a project holds.
- For a RAEA asset, write the file only: `generate.image {prompt, transparent: true,
  import: false, dir: "<build>/fx/assets", wait: true}`. A cut-out works best as a flat
  illustration on a plain background; look at its edges before you use it.
- Look at every result (`render_frame`, a contact sheet) before it goes into a review build:
  hands, faces and text in generated pictures go wrong.
- Say in the delivery notes what was generated: the platform's altered-content label is the
  user's decision, never yours. Credit the tool the way `generate.providers` → `attribution`
  asks.

**Narration:**

```text
tts.create {text, voice: "clone:mohesh-natural", language: "en-US"|"ml-IN", delivery: "warm", time, track: "A2", check: "strict"}
```

- `delivery` is one of `plain`, `conversational`, `warm`, `excited`, `amazed`, `curious`, `calm`
  or `dramatic`.
- `check: "strict"` hears the line back and retakes it until it is clear.

**Captions:**

1. `transcript.generate`
2. `transcript.createCaptions`
3. `filmcraft-cli captions style bold --platform reels`, or `exec captions.setAnimatedStyle`

**The mix:**

- Ducking: `essentialSound.setType` (dialogue or music), then `essentialSound.generateDucking`.
- Loudness: `audio.normalizeLoudness {target: "social"}` (−14 LUFS). Skip it when the user will
  add music themselves.
- Clean up a clip's voice with `audio.enhanceVoice {clips}`.

### 6.10 Save

`file.save {}` at the end of the script. Then write `build/README.txt` with the recipe: the
commands, in order, that made the video.

## 7. Checking your work

Never show the user something you haven't looked at.

| Check | Command |
|---|---|
| One Program frame | `filmcraft-cli --project P.fcproj render --seconds 5 --out f.png` (MCP: `file.exportFrame {path, format: "png", target: "program", time}`) |
| Frames of any video file (an export) | `filmcraft-cli frames out.mp4 --at 1,5,9,16.5 --scale 0.5 --columns 4 --out sheet.jpg` (fast, labelled) |
| Source clips side by side | `filmcraft-cli --project P.fcproj sheet --items 3,5:0-8 --n 6 --out sheet.jpg` |
| Specs, loudness, true peak for a platform | `filmcraft-cli publish check youtubeShort out.mp4` (also `youtube`, `instagramReel`); `publish fix` writes a copy that passes |
| The mix | `exec audio.analyzeMix` (bands, speech over music, limiting) |
| Text in the platform's interface areas | `exec sequence.checkSafeZones platform=reels` |
| Unblurred faces in the result | `exec sequence.checkPrivacy '{"seconds":[1,5,9]}'` |
| What the voice actually says | import the export and run `media.understandSpeech` on it; catches garbled or mispronounced lines and mid-phrase pauses |
| A pipeline's own QA | `<out>.qa.json`, `<out>.qa.png`, `<out>.sheet.jpg`, `<out>.report.json` |
| Media info | `filmcraft-cli probe out.mp4` |

How to look:

- **Text.** Look at frames with text at full size, or crop the area. A small sheet hides clipped
  letters.
- **Cuts.** Check the frames either side of every cut that looks wrong in a sheet. A white frame
  in the middle of a `flash` transition is expected; check ±0.2 s around it.
- **The 4K final.** Compare it with the approved review at the same times, frames and sound hits,
  before uploading.
- **Large photos in a 4K render.** Check them in the 4K export itself. A 1080p check once missed a
  number plate that the 4K render showed.

## 8. Export

```bash
filmcraft-cli --project "P.fcproj" export "Name - review (1080x1920).mp4" --preset "Social Vertical 1080×1920"
```

```bash
filmcraft-cli --project "P.fcproj" export "Name (2160x3840).mp4" --preset "High Quality 2160p 4K" --settings '{"frameSize":[2160,3840]}'
```

| Deliverable | Preset |
|---|---|
| 16:9 review | `YouTube 1080p Full HD` |
| 16:9 final | `YouTube 2160p 4K Ultra HD` |
| 9:16 review | `Social Vertical 1080×1920` |
| 9:16 final | `High Quality 2160p 4K` with `--settings '{"frameSize":[2160,3840]}'` (renders at 2×) |

`filmcraft-cli export --list-presets [query]` lists every preset.

- **Speed.** A 24 s vertical 4K export took 189 s here (3.8 fps, about 50 Mb/s, 144 MiB);
  1080p is several times faster. Run long exports in the background and log to `build/`.
- **GPU.** Rendering uses the GPU by default (`--gpu-rendering auto`); `off` only to rule the GPU
  out.
- **Offline media** exits 1 with the item named (§2).

## 9. Delivery

- **Google Photos** is the default for reviews and finals:

  ```bash
  filmcraft-cli photos upload "Name - review (1080x1920).mp4" --description "Name — 9:16 reel, review 1 (AutoCut)"
  ```

  - `--album "<title>"` also puts it in an album FilmCraft made (created when missing). That needs
    `photos auth --albums`.
  - The JSON answer has `productUrl`. Give the user that link.
- **YouTube:** `filmcraft-cli youtube upload f.mp4 --title "…"`, private by default, or
  `publish run youtube|youtubeShort f.mp4` with the pre-flight. Only when the user asked; never
  public on your own.
- **Instagram:** `instagram publish f.mp4 --video-url https://…` (see §1). Only when asked.
- **Files to the user directly.** The user often follows along from a phone over Remote Control.
  Files over about 30 MB don't reach the phone, so give the Google Photos link.

## 10. Narration and Malayalam

The user's videos are often narrated in Malayalam with their cloned voice. The details are in
[malayalam-narration.md](malayalam-narration.md) and [malayalam-style.md](malayalam-style.md).
In short:

- **Write spoken Malayalam, not written Malayalam.**
  - Use short sentences and the English tech words people actually say, in Malayalam script.
  - Write years and prices the English way ("2021" is said "ട്വന്റി ട്വന്റി വൺ").
  - `ml-lint` checks a script.
  - `ml-script --plan plan.json --out script-ml.json` drafts one with the local model.
- **Narrate the story and the feeling, not what is on screen.**
  - Keep hard names (product names, model numbers) on screen, not in the voice; that is where the
    voice stumbles.
  - Use questions and exclamations.
- **Use the strict hearing check.** The Malayalam speech model hears every line back and retakes
  unclear ones (`check: "strict"` / `--narration-check`). Afterwards, transcribe the export and
  read it again.
- **Make samples first.** Render a few lines (`tts.preview` or `tts.render`) and let the user
  listen before a full build.
- **Pick the delivery per line.** `warm` suits long lines; `excited` can overact.

## 11. 360° footage

The details are in [360-video.md](360-video.md).

- **Getting it:** `insta360 import` (or `download`, which drives Insta360 Studio and needs the
  `chrome` lock).
- **Framing it:**
  - `clip.reframe360` for a camera, with presets such as `tinyPlanet`.
  - `clip.vrMove` for moves: orbit, barrel roll, whip pan, reveal.
  - `clip.follow360` follows a subject.
  - `clip.stabilize360` steadies the picture.
- **The easy route:** `direct`, which handles 360° clips itself.
- **The user's preference (travel):** show the places, not the person holding the camera; keep to
  one outing's footage, in order.

## 12. The live app

- **Is it up?** `filmcraft-cli --bridge 127.0.0.1:9876 exec project.inspect`. The `dirty` field
  says whether there are unsaved changes.
- **Look at it:** `filmcraft-cli --bridge 127.0.0.1:9876 call ui.screenshot path=/tmp/app.png`.
  `call ui.elements` lists the clickable ids; `call ui.click id=…` clicks one.
- **Quit it safely:** first `exec project.inspect` must show `dirty: false`, then
  `exec app.quit`. `app.quit` works only over the bridge, so headless `commands` doesn't list it.
- **Rebuild and install it**, only when app code changed and the user said to ship:
  1. Package:
     `AUTOCUT_APP_NAME=AutoCut AUTOCUT_BUNDLE_ID=com.moheshmohan.autocut AUTOCUT_APP_ICON=packaging/autocut/AutoCut.icns CARGO_TARGET_DIR=target/agent-main packaging/macos/package.sh --arch aarch64`
  2. Quit the app as above.
  3. Keep the old bundle as `target/macos-package/AutoCut-previous-<date>.app` and put the new one
     in place.
  4. Relaunch it with the `open -n --env …` line from §1.
  5. Copy the packaged CLI to `target/release/filmcraft-cli` for `.mcp.json`.
     A `cargo test --release` in the main checkout (the full gate included) rebuilds that
     file from the working tree, so copy the packaged CLI again after a gate run. The app keeps
     the layout it had when it closed; the playhead comes back at 0 (`playhead.set` restores it).
- **A stale "Recover Unsaved Changes" dialog** at launch isn't in `ui.elements`. Ask the user to
  click it.
- **The Generate panel** (Window ▸ Generate): set its draft with
  `ui.set {"generate": {"mode": "augment", "prompt": "add snow"}}`, read it under
  `ui.inspect` → `ui.generate`, press `generate.run` / `generate.stop`. Its controls are the
  `generate.*` automation ids. For scripted work call the commands; the panel is for the user.
- **`engine.execute` never opens a dialog** (since the 19 November 2026 build). A command sent
  without parameters runs with its defaults or answers what it needs: `sequence.addTracks {}`
  adds one video track, `clip.audioGain {}` answers "need `db`". To open a dialog for the user,
  use `ui.menu.invoke`. Confirmations and the model-download question still ask.

## 13. Gotchas

- **`ci --changed` doesn't run dependents' tests.** Adding an Elements sound broke a count assert
  in `apps/filmcraft-cli/tests/elements.rs` that only the full gate caught, after 1 h 40 min. Run
  the full `cargo xtask ci` before any push.
- **Testing-mode Google sign-ins expire after 7 days.** An upload or pick failing on auth means the
  user re-runs `photos auth`.
- **The picker doesn't enforce its item cap.** It selects past it and then greys out Done. Keep
  searches to a sensible size, or let the CLI split them into sessions.
- **Audio levels in an export.**
  - Measured in an AAC export, sound hits can read about a frame early (encoder priming). Don't
    chase it.
  - A sequence with no music and no loudness pass exports quiet: the Halloween reel, with only
    light hits, measured −26.6 LUFS. That's right when the user adds music; otherwise normalize.
    `publish check` shows the level.
- **Effects and tracked blurs.**
  - A tracked blur (`track.blurObject`, auto redact) now goes under a clip's `camera_shake`. In
    older projects, a shake moved the plate out of its blur.
  - In the UAE, the plate on vans and minibuses is the white rectangle mid-door; the yellow chevron
    band on the bumper is a reflector. `redact` with `minHits 1` catches one-frame sightings.
- **Cropping.**
  - Filmed screens (phones, monitors, chats) lose their legibility when cropped or zoomed: frame
    them whole and static.
  - Auto-reframe may follow the wrong thing (an arm, not a gauge). Use `clip.cameraTrack` with a
    box.
  - Wide 16:9 clips cropped to 9:16 "zoom in too much" (the user's words): show them `whole` over
    a blurred fill.
- **YouTube.** Embedded CEA-608 captions can't carry Malayalam. Upload an `.srt` in YouTube
  Studio instead.
- **Instagram.** Instagram Login can't upload resumably, so posting needs a public `--video-url`.
- **Driving Chrome by hand.** On this Chrome, AppleScript tab ids are text, and a "Leave site?"
  tab closes on a second `close`.

- **Generation.**
  - `generate.providers` first. A tool marked not installed fails at once; say so instead of
    trying another model on your own.
  - MiniMax-H3-Image has no transparent output: `transparent: true` runs a second step (a
    background remover), which can leave a faint halo and fill small enclosed gaps.
  - A gap is filled from one frame, so the clip continues the look, not the motion. A clip
    shorter than the gap is slowed, never below 40 %; read `uncovered` in the answer.
  - The clip tool makes small pictures (480 on the long side by default). They are scaled to
    the frame; `media.upscale` when that shows.
  - A failed tool's last lines are the error; the full log is `<file>.generation.log`.

## 14. The user's taste

These come from the user's feedback on earlier videos. They are defaults, not laws. When a brief
says otherwise, the brief wins; when in doubt, ask or offer a sample.

- **Review, then final.** Reviews are 1080p in Google Photos; the final is 4K after approval.
- **Branded endings.** Use the H4hacks FX from the Google Photos `YouTube fx` album at the end
  of future videos, choosing an outro or end screen that fits the video. Use the logo-complete
  masters and editable projects in `~/Movies/AutoCut/YouTube FX/Logo Updates/` for the revised
  clips; the other pack masters remain in the main folder and `Volume 2/`.
- **YouTube AI labels.** Ask for the user's explicit approval before applying an AI label to a
  video or a Promote campaign. If a disclosure is required, explain the requirement and get
  approval before publishing or launching the campaign.
- **Pacing.** Abrupt cuts get noticed. Crossfades, pacing and mix levels matter more than effects
  count.
- **Transition sounds are chosen per video.**
  - Subtle `swish` / `chime` are the defaults for calm, narrated videos.
  - Loud whooshes and risers sounded "rocket like" in those.
  - Punchy hits suit energetic reels (the driving reel, the Halloween reel).
  - Ask before loud effects, light leaks or strong colour looks on a new kind of video.
- **Voice-over.**
  - Human and warm, with feeling, never a description of the shots.
  - Leave many shots to music and on-screen labels.
  - Family wording: "our son", not "my son".
  - End a series with "this has been my journey, and it will continue".
- **Travel.** Show the places; the user doesn't need to be in frame. Use one outing's footage, in
  filming order. Leave the director's style on auto.
- **"No music, I'll add my own."**
  - Deliver without music.
  - Keep the cuts on a beat grid.
  - Keep effect sounds light and on their own track (A2), so they are easy to drop.
- **Showcases.** Leave toys out and favour event footage. Portrait clips fill the frame unless
  cropping hides the action; then show them whole over a blurred fill.
- **GPU for everything.** CPU only as the fallback.
- **Turn repeated hand work into app features.**
  - When a round needs throwaway scripts or manual steps that will come back, make them a CLI
    subcommand or engine command, plus docs, in the same round.
  - The earlier ones became `sheet`, `frames`, `showcase-plan`, `photos locations` and
    `photos pick --photos`.

## 15. Further reading

| Topic | Doc |
|---|---|
| MCP tools, control channel, CLI conventions | [agents.md](agents.md), [control-protocol.md](control-protocol.md) |
| Album to YouTube, end to end (a maker's build video recipe included) | [video-pipeline.md](video-pipeline.md) |
| Robot showcase rounds | [showcase-runbook.md](showcase-runbook.md) |
| Quick templates | [quick-templates.md](quick-templates.md), [crates/templates/README.md](../crates/templates/README.md) |
| The director, 360° | [360-video.md](360-video.md) |
| Scene index (shots, descriptions, search) | [scenes.md](scenes.md) |
| AI effects (depth, upscale, slow motion, object mask and removal) | [ai-effects.md](ai-effects.md) |
| Captions | [captions.md](captions.md) |
| Malayalam narration and style | [malayalam-narration.md](malayalam-narration.md), [malayalam-style.md](malayalam-style.md) |
| Recording a voice reference | [voice-recording.md](voice-recording.md) |
| Graphics and text | [graphics.md](graphics.md) |
| Custom animations, rendered scenes and transitions | [raea-workflow.md](raea-workflow.md) |
| Generated pictures, clips, sounds; filling gaps; changing a frame | [generation.md](generation.md), plan: [generation-2026-11.md](../plan/generation-2026-11.md) |
| Google Photos setup | [crates/gphotos/README.md](../crates/gphotos/README.md) |
| AutoCut's planner and showcases | [crates/autocut/README.md](../crates/autocut/README.md) |
| Building on the app (rules, gates, layering) | [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md), [architecture.md](architecture.md) |
