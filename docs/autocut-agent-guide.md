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
  - Push only `main` to the `autocut` remote (public: nothing personal in commits), never to
    `origin`, which is upstream.
  - Push only after the full `cargo xtask ci` and with the user's OK.

## 1. This Mac

| What | Where / value |
|---|---|
| CLI | `.mcp.json` and production workflows use `target/release/filmcraft-cli`, installed with the tested AutoCut app. `target/agent-main/release/filmcraft-cli` is the development candidate. The [November upstream/pipeline upgrade](../plan/pipeline-upgrades-2026-11.md) retains the AI features; the 0.14.0 build exposes 954 commands. Package and verify the candidate before replacing production; keep rollback copies. Builds report the workspace version (0.14.0 since 25 November 2026; [releases](autocut-releases.md)); between releases, inspect the packaged commit and command catalogue too. |
| Building your own CLI | `CARGO_TARGET_DIR=target/agent-<name> cargo build --release -p filmcraft-cli`, one target folder per worktree, never shared. |
| The app | `target/macos-package/AutoCut.app`, control port **9877**: `open -n --env FILMCRAFT_CONTROL_PORT=9877 target/macos-package/AutoCut.app` |
| RAEA | `~/.raea/versions/0.1.8/bin/raea` for local custom animations, 3D scenes and effects (0.1.0 kept beside it). [Render and import workflow](raea-workflow.md). |
| FrameAtlas | The footage catalogue beside this checkout (`../FrameAtlas`, `./run.sh serve`, loopback port 7318): the user's Google Photos and Insta360 recordings by what they show (search by meaning), people, places and trips, ideas and the footage each video used. AutoCut reads it for `photos pick --meaning` (§4) and its own MCP tools hand an idea to AutoCut (`idea_to_autocut`). [FrameAtlas README](https://github.com/moheshmohan/FrameAtlas#readme). |
| EffectCraft | Linked into the installed AutoCut app: `effectcraft.templateInfo` and `effectcraft.renderTemplate` (§6.8). A workspace build without `--features effectcraft` answers that it lacks it. |
| Generation tools | The user's own local models, listed in `~/Library/Application Support/FilmCraft/generation/providers.json`: `h3-image` (MiniMax-H3-Image, pictures and edits), `ltx-video` (LTX-2.3, clips up to 10 s; with the Union-Control adapter in `testground/ltxenv/adapters/` it also makes a clip again, `generate.restyle`), `sa3-sfx` and `sa3-music` (Stable Audio 3 small, up to 30 s). `filmcraft-cli exec generate.providers` says which are installed. Never edit that file or download a model without the user's OK. [generation.md](generation.md). |
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
filmcraft-cli --bridge 127.0.0.1:9877 exec project.inspect
```

The last one shows whether the app is up and whether its project has unsaved changes.

## 2. Three ways to drive it

Every edit is an engine command: an id plus JSON params. The CLI, the MCP server and the control
channel all run the same commands, so pick the transport that fits the job.

| Transport | Use it for | How |
|---|---|---|
| **CLI, headless** | one-off commands and the one-command pipelines (§5) | `filmcraft-cli --project p.fcproj [--save] exec <id> key=value …` or a subcommand (`quick`, `direct`, `export`, …) |
| **MCP server, headless** | long scripted builds: one engine session, many commands, ids that stay valid | `filmcraft-cli --project p.fcproj mcp`, then `tools/call command_run {id, params}`, and finally `file.save` |
| **Bridge to the running app** | when the user wants to watch, or for UI work | `filmcraft-cli --bridge 127.0.0.1:9877 exec <id> …`, `… call ui.screenshot path=/tmp/x.png`, or `filmcraft-cli mcp --bridge 127.0.0.1:9877` |

In Claude Code, `.mcp.json` registers two servers:

- `filmcraft`: the bridge to the running app.
- `filmcraft-headless`: the **built-in demo project**, not your project. For your own project,
  start your own server with `--project`, as in the client below.

### A minimal MCP client (Python)

Every hand-built edit used this client: `build/fc.py` in a video's folder.

```python
"""A tiny client for filmcraft-cli's MCP server: one headless engine, commands one at a time."""
import json, os, subprocess, itertools

CLI = os.environ.get("FILMCRAFT_CLI", "filmcraft-cli")      # or the full path to target/release/filmcraft-cli

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
| "the clips of X" from the whole library (birthday cake, a beach day, the red car) | `photos pick --meaning "X"` | FrameAtlas finds them by meaning, the picker ticks exactly those (§4) |
| a RAEA composition brought in as an editable edit | `exec composition.import {source, output, place?, wait: true}` | its footage and sounds become real clips, each drawn layer a clip of its own ([raea-workflow.md](raea-workflow.md)) |
| a picture, b-roll clip, sound or music that does not exist yet | `exec generate.image` / `generate.video` / `generate.sound` | the user's local tools (§6.9); `wait: true`; one at a time |
| "fill this empty stretch" | `exec generate.fillGap {track, atSeconds, from}` or `{all: true}` | `from: "motion"` carries the movement on, `"both"` goes from the frame before to the frame after, `"before"` (default) continues one frame; `uncovered` says what is left |
| "make this shot a bit longer" | `exec generate.extend {atSeconds, seconds}` | continues the movement past the end of a clip; needs 1.4 s of picture before it |
| "make this shot look like a painting / swap the person for this character" | `exec generate.restyle {prompt, clip?, look?, references?}` | the clip made again through an adapter of the clip tool, laid over the old one; 5 s a go, small, faces change ([generation.md](generation.md) §6) |
| a lower third, end card or chapter card from an `.ectemplate` | `exec effectcraft.renderTemplate {template, values, place, wait: true}` | `effectcraft.templateInfo` lists its controls first; ProRes 4444 with alpha, placed above the picture (§6.8) |
| pictures, cut-outs, clips and sounds for a RAEA scene | `exec generate.assets {manifest}` | an `assets.json` beside the composition ([generation.md](generation.md) §5); `dryRun` first |
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

Picking exactly what FrameAtlas finds by meaning, across the whole library (videos; `--photos`
adds photos):

```bash
filmcraft-cli --project "P.fcproj" photos pick --meaning "birthday cake with candles" --photos --max-items 12 --save
```

FrameAtlas (§1; `--catalog-port` if it is not on 7318) returns each item's capture time and kind,
best first (24 by default, at most 200). The agent types each item's day into the picker's search
and ticks only the tiles whose time and kind match, trying up to three days per item; items it
cannot find come back as `left`. It reads the picker's page, so Chrome needs **View ▸ Developer ▸
Allow JavaScript from Apple Events** ticked in the window signed in to Google Photos, and it refuses
without it. It goes without `--auto`, `--queries`, `--album`, `--all`, `--session` and
`--no-browser`. Over MCP, pass the found items to `googlephotos_pick` as `items: [{captured,
kind}]` (from FrameAtlas's `catalog_search_meaning`).

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
- **Chrome in front.** The agent clicks and types only while Chrome is the front app. If another
  app comes to the front, it stops instead of typing into it: keep Chrome in front, then run it
  again.
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

From a folder of recordings (a camera card, or `.insv` copies with their `.lrv`):

```bash
filmcraft-cli --project "P.fcproj" --save insta360 import --all --folder "/Volumes/X5/DCIM/Camera01"
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

### 6.0 Direction and rhythm

Read this before planning shots. It is adapted, in our words, from huashu-motion's director, rhythm
and verification guides ([alchaincyf/huashu-motion](https://github.com/alchaincyf/huashu-motion),
MIT; see `ATTRIBUTION.md`). The user's taste (§14) always wins over it: no whooshes or risers,
silent Shorts, the places shown in a travel vlog.

- **Three directions first** for a video over about 15 s or a new kind of video, unless the user
  already chose a style. They differ in approach or material, not only in colour: **A** fits the
  footage and the story; **B** pushes it (one continuous move, type as the hero, a montage-led
  cut); **C** is unexpected but still about the subject. Never three quiet ones. Show a key still
  or a few seconds of each and let the user pick.
- **The energy curve before the beat grid.** Give each stretch an energy from 0 to 10 and check
  the shape:
  - a hook within 3 s: a clear onset, a movement or the first cut. A hook isn't a frantic open;
  - a climb in steps you can name (the drums come in, the camera starts moving, the cuts double),
    not a slope nobody notices;
  - the densest five seconds about a third of the way in (30–60 % of the length): 5–8 cuts, about
    one per beat, between shots of 1.5–4 s;
  - the best moment on the strongest point of the music; a near-silent stop of 0.5–1.6 s only when
    it earns its place;
  - a resolve on one idea, held for 1.5 s or more.

  A flat line at 3–5 with a hit at the end, or a build that peaks in the last two seconds, reads
  as boring.
- **Something new at least every two bars** (a cut, an element landing, a camera move starting or
  stopping, a word, a hit), except a planned stop.
- **At every section boundary, change at least two** of: shot scale, speed (0.25 s hard moves
  against 2 s glides), ease family, camera (still, travelling, whipping), density (one subject
  against many), colour, and the music's texture. Change all of them at the drop.
- **Eases.** A hard ease-out for energy. Springs only for things that should feel physical,
  damped at 0.9 or more unless the user asked for something playful. Soft 0.6–0.8 s eases on
  everything make a film sleepy.
- **Numbers, not adjectives.** Write motion as numbers ("entrances 12–18 frames, exits shorter, a
  hard ease-out, no overshoot"), not as "snappy" or "smooth".
- **Three notes before anyone watches.** Before showing a cut, write down the three notes a
  demanding director would give, fix them, and only then show it.

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
- **Even pushes:** two linear scale keyframes look fast at the start and slow at the end. Write
  the push with `keyframes.choreograph` (§6.8), which eases scale in log space by default, or
  give the first keyframe `perceptual: true` with `effects.setInterpolation`.

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
  - `pop` eases out without overshoot. For the bounce, ask for it: `'easing': 'Back Out'`.
- **Use `unit: "line"` with `stagger: 0` for `out`.** A letter-staggered fade-out starts early,
  and letters vanish while the title should still read.
- **Fonts.** Any installed font works (macOS ships Chalkduster, DIN Condensed, Futura and others);
  Inter is the fallback. Malayalam in titles needs the craft-fonts build (see CLAUDE.md).
- **Platform safe zones.** `exec sequence.checkSafeZones platform=reels` lists the text that
  reaches into Reels', Shorts' or TikTok's interface.

Ready-made animated text and stickers: `exec elements.list kind=text`, then
`elements.insert {id, seconds, track}`.

**Moves on a timeline.** `keyframes.choreograph` places tweens the way a GSAP timeline does and
writes them as ordinary keyframes, editable in Effect Controls, in one undo step:

```python
f.run('keyframes.choreograph', {'target': {'clips': [logo, title], 'effect': 'motion'},
    'labels': {'drop': 1.2},
    'stagger': 0.08,
    'tweens': [
        {'prop': 'scale', 'from': 80, 'to': 100, 'duration': 0.5, 'ease': 'power2.out'},
        {'prop': 'position', 'to': [540, 760], 'duration': 0.6, 'ease': 'spring', 'position': 'drop'},
        {'prop': 'opacity', 'effect': 'opacity', 'to': 0, 'duration': 0.25, 'position': '+=1.5'}]})
```

- **Positions:**
  - `"<"` / `">"`: with the previous tween's start / at its end.
  - `"+=0.2"`: after the end so far.
  - A label, or `"label+=0.5"`.
  - Percentages like `"<25%"`.
  - Seconds into the clip.
  - An unknown label is an error.
- **Eases:**
  - GSAP's names (`power1`–`power4`, `sine`, `expo`, `circ`, `back`, `elastic`, `bounce`, with
    `.in`, `.out` or `.inOut`).
  - `spring` or `spring(ratio, stiffness, mass)`, `steps(n)`, `wiggle(amplitude, frequency, seed)`.
  - The Bezier modes.
  - `back.out(1.7)`'s argument is ignored, and the result's `notes` says so.
- **Damping:** keep springs at a damping ratio of 0.9 or more (the default) unless the style is
  playful.
- **Scale tweens are perceptual:** a push from 100 to 200 % moves evenly.
- **Check first:** `dryRun: true` returns the keyframes without writing them.

**Cards from EffectCraft templates.** An `.ectemplate` (lower third, end card, chapter card)
is filled with values and rendered in the app, without RAEA:

```python
info = f.run('effectcraft.templateInfo', {'template': '/abs/lower-third.ectemplate'})
# info['controls']: name, type, default, range or options; info['renderable'] / ['refused']
f.run('effectcraft.renderTemplate', {'template': '/abs/lower-third.ectemplate',
    'values': {'Name': 'Sam Rivera', 'Accent': '#ff8a1f'}, 'size': '1080x1920',
    'place': {'seconds': 4.0}, 'wait': True})
```

- **Values** are keyed by control name or id and checked against each control's type:
  colours as `#rrggbb` or `[r, g, b, a]` from 0 to 1, dropdowns as an index, media as an
  absolute path. A wrong value is refused before anything renders.
- **Size** is `"WxH"` in the comp's shape. Pass the sequence's size for a sharp overlay; a
  render of another size is scaled to fill the frame.
- **Output:** ProRes 4444 with alpha (the default) or `format: "png"`, in `<template>
  renders/rNNN` beside the template unless `out` names a new or empty folder; then probed,
  imported and placed above the picture like `motion.render`. Look at the placed frame.
- **Limits:** no sound from the template; 3D model footage is refused; a template made by an old
  EffectCraft may render a little differently from EffectCraft itself.

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

**Fitting the user's track to the film.** `media.beats {item}` also answers `downbeats`, `sections
[{start, end, energy}]`, `drops`, `stops`, `musicEnd`, `ending {kind: cadence|fade|cut, at}` and
`confidence` (with `confidenceParts`). Then remix the placed clip to the film:

```python
r = f.run('clip.remix', {'clip': music, 'seconds': 45, 'ending': 'own', 'events': [
    {'kind': 'burst', 'seconds': 0}, {'kind': 'peak', 'seconds': 20},
    {'kind': 'landing', 'seconds': 40}]})
# r['eventsLanded']: kind, lift, playsAt, offBy, landed; r['joins']: levelChangeDb, excessDb, ok
# r['alternatives']: film lengths that need no join; r['warnings']
```

- **Events** are film times from the clip's start: `burst` (the opening), `peak` (a hero shot or
  title hit), `landing` (the end card), `stop` (a hard cut to silence). Each lands on a real lift
  in the track (a drop, a section start, a downbeat), never an invented one; an event with no
  lift near it is reported with `landed: false`.
- **Endings:** `own` lands the track's own ending on the landing and lets a cadence ring on;
  `button` ends on a strong downbeat with a short tail; `fade` fades out; `source` (the default)
  cuts where the plain remix does. With `own` or `button`, events after the landing are dropped.
- **Read the warnings.** A join that jumps more than 6 dB beyond what the music itself does is
  replaced; if none passes, the best plan is kept and named in `warnings`. Music that ends more
  than 1 s before the film says so: the close would play in silence. Pick another ending or
  an `alternatives` length then.
- **Low `confidenceParts.downbeats`** (under about 0.5) means the bars may be off: check the
  landings by ear, or fit to beats only.
- Direct, Quick Short, Showcase and trailers already fit the music this way and report it under
  `music.fit` (`musicFit` for Quick Short). Shorts stay silent.

**Generated pictures, clips and sounds.** When `generate.providers` lists installed tools (the
user's own local models; [generation.md](generation.md)):

- `generate.image {prompt, references?, fromPlayhead?, transparent?, place?, seconds?, wait: true}`.
- `generate.video {prompt, image?: item|path|"playhead", seconds, atSeconds, track, wait: true}`.
- `generate.sound {prompt, kind: "sfx"|"music", seconds, atSeconds, wait: true}`.
- `generate.fillGap {track, atSeconds | all: true, from?: before|after|both|motion|prompt, wait: true}`
  fills an empty stretch of a video track; `generate.extend {atSeconds, seconds, wait: true}`
  carries a shot on past its end.
- `variations: 2..4` on image, video and sound makes several; the first is placed, and
  `generate.pick {item}` puts another in its place. Show the user the choices (a contact sheet)
  when the look matters.
- `generate.sound {prompt, clip: <clip id>}` sizes a sound to a clip; `generate.image {…, place:
  true, move: "in"}` makes the placed picture move in depth.
- `generate.assets {manifest, dryRun?, only?, force?, wait: true}` builds a scene's asset list.
- `generate.restyle {prompt, clip?, how?, look?, references?, strength?, wait: true}` makes a clip
  again: its shapes and movement stay, the look follows `prompt`. For another character give
  `look` ("Replace the man with the robot from the second picture, in the same place and pose")
  and `references: [the character's picture]`; say in `prompt` what the new clip shows. It
  takes what the program shows over the clip, so hide titles above it first. About 1 min 45 s
  for 3 s, 7 min with `look`. `generate.providers` → `restyles` says which ways are installed.
- `generate.augment {instruction, atSeconds, animate?, wait: true}` changes what the frame shows.
- Common parameters: `provider` (default: the first installed tool of the kind), `seed`, `dir`
  (default a `Generated` folder beside the project), `place`, `track`, `atSeconds`.
- One generation at a time; each takes the GPU lock itself, so never hold a GPU lease of your
  own around it (it would wait on you for ever). `jobs.cancel {job}` stops one.
- Times on this Mac: a picture 40 s, an edit from a full frame about 4 minutes (about 2 at
  768×768), a 3 s clip about 50 s, a gap filled by carrying the movement on about 70 s, a sound
  a few seconds. Over MCP these are long jobs: `command_run` with
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
| Motion: pops, stills, a flat opening, cuts a minute (warnings) | `exec qa.motionEnergy '{"path":"out.mp4","sheet":"motion.png"}'` (or `sequence`; declare intentional `holds`) |
| Text and timeline craft: reading time, size, contrast over the footage, dead stretches, silence (warnings) | `exec sequence.checkCraft` |
| Motion design on the timeline, no render: bouncy eases, entrance and exit lengths, exits longer than entrances, long staggers, every title entering alike, text looping the whole clip, jumps at clip edges (warnings) | `exec sequence.checkMotion` (`style: "playful"`, or `autocut.taste` `motion: "playful"`, allows overshoot; `bands` overrides the frame ranges). Also `delivery.preflight`'s `motionCraft` check (Reflex, waivable) |
| Motion filmstrips: seven frames per animated title, transition and alpha overlay, with jump, mid-move step, dissolve dip, fade past its shot and replayed-entrance detectors (warnings) | `filmcraft-cli frames --motion --project P.fcproj --out strips/` (`--items 12,12:out,t5`, `--scale 0.25`); MCP: `sequence.motionStrips {out, items?, wait: true}`, a long job. The folder must be new or empty |
| Copy: stock phrases, filler, shot-describing narration, device budgets, figures not in the fact ledger (warnings) | `exec copy.lint '{"sequence":12,"publish":{"title":"…","description":"…"}}'`; your phrases in `autocut.taste` `phrases` or `allow` are never flagged; figures are checked against `facts` or `build/facts.json` |
| The delivery gate with tiers | `exec delivery.preflight` reports each check's `tier`: **Floor** (blocks: sources, clicks, mix, privacy, changed narration, Shorts safe zones, screen protection; the new text contrast under 3:1 and figures only warn until `floorBlocks: true`), **Reflex** (fix it, or waive it in `build/decisions.json` with `reason` and `evidence`; more than five waivers is a finding), **Finish** (reported only) |
| Unblurred faces in the result | `exec sequence.checkPrivacy '{"seconds":[1,5,9]}'` |
| What the voice actually says | import the export and run `media.understandSpeech` on it; catches garbled or mispronounced lines and mid-phrase pauses |
| A pipeline's own QA | `<out>.qa.json`, `<out>.qa.png`, `<out>.sheet.jpg`, `<out>.report.json` |
| Media info | `filmcraft-cli probe out.mp4` |

How much to check (adapted from the same huashu-motion verification guide):

- **Deterministic checks first, every time.** QA commands, probes, loudness and safe zones cost no
  tokens and catch the mechanical faults. Run them before looking at anything.
- **Check motion, not stills.** A sheet of stills can't show a pop, a stall or an entrance replayed
  across a cut. Look at the motion filmstrips, and watch the export once at normal speed as well.
- **A budget.** One full inspection, one batch of fixes, then one round that confirms what changed
  and the cuts beside it. Then stop and report what is left, rather than polishing in circles.
- **Fresh eyes.** A second reviewer gets the brief, the export, the chosen direction and the facts
  to check. They don't get the maker's explanation.
- **Say what was checked.** Stills alone don't prove the motion was reviewed, and loudness figures
  don't prove anyone listened. A check that has never caught a deliberately broken input proves
  little.
- **Know when it's done.** Say up front what "done" means (the checks that must pass, the user's
  approval) and stop there.

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

### Review comments and the finish reviewer

- **Comments pinned to frames.** The user clicks **Comment on Frame** on the Program transport,
  clicks or drags a box on the picture, types and presses Enter. Agents add their own with
  `review.comment {seconds, box?, text}`. Every comment lands in `build/review/inbox.jsonl`
  beside the saved project.
- **Waiting for notes.** `review.inbox {wait: true, timeout: 300}` returns when a new open
  comment arrives (a job over MCP; `jobs.cancel` stops it). `review.look {id}` gives the frame
  with its box and what is on screen there (clips, text, captions, the nearest cuts, the
  transition, motion findings). Fix it, then `review.done {id, note}` with what changed.
- **The finish reviewer** ([finish-reviewer.md](finish-reviewer.md)) is a fresh agent, not the
  one that built the video. Give it the packet from `review.packet {}` (frame sheet, filmstrips,
  QA, decisions, comments) and nothing else. Its answer ends on one line: `GATE: ship`,
  `GATE: fix <n>` (at most 8, with at most one `KEEP:`), `GATE: rebuild - <why>` or
  `GATE: recapture - <what>`. Check it with `review.gate {text}`; after two rounds without a
  ship, only the user's own words (`review.gate {user: "…"}`) close the run. It never blocks
  delivery yet.
- **Over MCP and the shell.** The MCP tools `review_comment`, `review_inbox` (`wait: true` runs
  as a job with progress), `review_look`, `review_done`, `review_packet` and `review_gate` take
  the same parameters. From a terminal: `filmcraft-cli review --bridge 127.0.0.1:9877 inbox`,
  `… look c3`, `… done c3 note="raised the title"`, `… gate --file answer.txt round=1`.

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

- **Is it up?** `filmcraft-cli --bridge 127.0.0.1:9877 exec project.inspect`. The `dirty` field
  says whether there are unsaved changes.
- **Look at it:** `filmcraft-cli --bridge 127.0.0.1:9877 call ui.screenshot path=/tmp/app.png`.
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
  - A gap filled `from: "before"` continues the look of one frame, not the motion; `"motion"`
    continues the motion and takes longer. A clip shorter than the gap is slowed, never below
    40 %; read `uncovered` in the answer. Look at both cuts of a filled gap before a review build.
  - An asset list is the record of a scene: change the list and rerun, do not hand-edit files.
    A pose made from a character (`references`) takes about two minutes at 768×768 with
    MiniMax-H3-Image; a cut-out can keep white between an arm and the body, so look at it.
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

### Taste memory: keep consecutive videos different

- **Record feedback as it comes.** `autocut.feedback {about, id, verdict: "dislike"|"like", note}`
  writes to the user's taste profile (`scope: "project"` for one project only). A disliked id is
  retired: never dealt again, and the pipelines' own defaults skip it. The whoosh lesson is
  `{about: "sound", id: "whoosh", verdict: "dislike", note: "rocket-like under narration"}`.
  `autocut.taste` shows `retired`, `liked` and `reasons`; `set` changes or clears them.
- **The pipelines deal for you.** Direct, Quick Short and Showcase deal the choices the user left
  open (look, music mood, title entrance, transition family, ending, register) and record the
  video's fingerprint; their reports list the deals. `guidance: false` turns both off. Explicit
  options always win.
- **By hand, deal before choosing.** `autocut.deal {for: look|titleStyle|transitionFamily|music|
  register|endCard, options?}` answers a `choice`, what it `skipped` and `why` (used in the last
  three videos, retired, or not allowed by taste). At the end, `autocut.fingerprint {record:
  true, pipeline: "hand"}` adds the video to the history.
- **A `sameAsRecent` warning** means four of the six axes (opening, title entrance, transition
  family, colour look, caption style, close) repeat one of the last three videos. Change one or
  two on purpose; it never blocks.
- **After changing a pipeline's choices**, run `filmcraft-cli eval` (briefs in `eval/briefs/`,
  footage drawn in code). It compares runs with and without guidance in
  `target/eval/<when>/summary.md`; `--no-judge` skips the local vision judge.

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
| The footage catalogue (search by meaning, people, places, ideas) | [FrameAtlas README](https://github.com/moheshmohan/FrameAtlas#readme) |
| What changed in each release | [autocut-releases.md](autocut-releases.md) |
| AutoCut's planner and showcases | [crates/autocut/README.md](../crates/autocut/README.md) |
| Building on the app (rules, gates, layering) | [AGENTS.md](../AGENTS.md), [CLAUDE.md](../CLAUDE.md), [architecture.md](architecture.md) |
