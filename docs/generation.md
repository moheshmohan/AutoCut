# Generation in the editor

AutoCut can make pictures, short clips, music and sound effects with the generation models you
have installed on your own computer, import the result as ordinary media and place it on the
timeline. It can also fill an empty stretch of a track with a clip that carries on from the frame
before it (or runs from the frame before to the frame after, or continues the movement), carry a
shot on past its end, change what a frame shows from an instruction ("add a red sailboat"), make
several variations to choose from, and build the pictures, clips and sounds a motion-graphics scene
needs from one list.

Nothing here is a model built into the app, and the app downloads nothing. A **generation tool**
is a program on your computer (an MLX or PyTorch command line you set up yourself), a workflow of
a ComfyUI server running on this computer (§3.2), or Blender rendering a 3D scene without its
window (§3.3). The app runs it, waits for the file it writes, and takes it from there. This is
the same stance as [RAEA](raea-workflow.md): separately installed, kept outside the repository
and the app bundle.

**Installing them is your choice.** Nothing in AutoCut downloads or installs ComfyUI, its custom
nodes or models, Blender or its add-ons. You install what you want, accept its licences, and
describe it in the provider list (§3); AutoCut only runs what that list names.

Built into the app, without any of this: procedural music and sounds (`audio.generate`), neural
music (`music.generateNeural`, ACE-Step) and narration (`tts.create`).

## 1. Commands

Every action is an engine command, so the Generate panel, the CLI, the control channel and MCP
(`command_run`) all do the same thing.

| Command | Does |
|---|---|
| `generate.providers` | The tools of this computer: kind, whether the program is there, licence, what must be credited; where the list is kept; the generators built into the app. |
| `generate.image` | A picture from `prompt`. `references` (project item ids or absolute paths, at most 16) and `fromPlayhead` / `frameSeconds` (the program frame there, as the first reference) turn it into an edit or put a character into a scene. `transparent` asks for a cut-out. `place` puts it on the timeline for `seconds`; `move` (`auto`, `in`, `out`, `left`, `right`, `up`, `down`, `circle`) makes the placed picture move in depth (3D Photo, needs the depth model). `import: false` with `dir` only writes the file (an asset for a motion-graphics scene). |
| `generate.video` | A clip from `prompt`, `seconds` long, optionally starting from `image` (an item id, a path, or `"playhead"`) and ending on `lastImage` (where the tool can). Placed at the playhead (`atSeconds`, `track`) unless `place: false`. `audio: true` keeps the clip's own sound. |
| `generate.sound` | `kind: "sfx"` (default) or `"music"` from `prompt`, `seconds` long, placed on a free audio track. `clip` (a timeline clip id) makes it as long as that clip, starting where it starts. |
| `generate.fillGap` | Fills the gap of `track` (default `V1`) under the playhead (`atSeconds`). `from` says what the clip is made from: `"before"` (default) starts from the last frame before the gap; `"after"` leads into the first frame after it; `"both"` starts on the one and ends on the other; `"motion"` gives the tool the last moments before the gap as a video, so the movement carries on; `"prompt"` follows only `prompt`. `all: true` fills every gap of the track (eight at most). A clip longer than the gap is cut to it; a shorter one is slowed towards it (never below 40 %), and `uncovered` says what is left. |
| `generate.extend` | Carries a shot on: a clip of `seconds` (0.3 to 10, default 2) that continues the movement of the picture just before `atSeconds` (default: the end of the clip under the playhead), placed there on the same track when it is free. |
| `generate.restyle` | Makes a clip again from `prompt`: the clip (`clip`, default the selected one, else the one under the playhead) goes to the tool as a small video, and an adapter of its model (`how`, one of the ways `generate.providers` lists under `restyles`; default the first that is installed) keeps its shapes and movement while the look follows the description. `image` is the picture the new clip starts on; `look` has the picture tool change the clip's first frame by an instruction first (with `references`: a character to put in), and the new clip starts on that. `strength` (0 to 1) is how closely the clip is followed. The new clip is laid over the old one, on a track above it. Up to five seconds in one go; `left` says how much of a longer clip stays as it was. |
| `generate.assets` | Makes the files a scene's `assets.json` names (§5) and leaves the ones that are there. `only` names some, `force` makes them again, `dryRun` only says what would be made, `import` brings them into the project. |
| `generate.pick` | Puts a generated `item` in the place of the variation of the same request that is on the timeline (or of `clip`), as one undo step. |
| `generate.augment` | Takes the program frame at the playhead (`atSeconds`), changes it by `instruction` (plus `references`: a character, a prop), and lays the result over the clip on the next free track, as a still for `seconds` or, with `animate: true`, as a clip made from the changed frame (`motion` says what moves). |
| `generate.history` | What was generated into this project: item, tool, description, seed. |
| `generate.render3d` | A 3D scene rendered by your Blender without its window (§3.3): a `scene` (.blend) or a plain stage for a `text` title, named `texts` and `objects` set, a `camera` track keyed on for a match-move, a frame range or `seconds`, `fps` and size (default the sequence's), `transparent` (default on), `format` `png` or `exr`, `cryptomatte` (EXR). The frames come in as a clip and are placed above what is at the playhead. |
| `generate.checkTool` | Whether a tool can run, without running it: its program (or workflow) is there; a ComfyUI workflow is read and checked (inputs, models, licences) and the server asked whether it answers; Blender says its version. `ok`, `problems`. |

All of them take `provider` (a tool id; default: the first installed one of the kind), `seed`,
`dir` (where the file goes; default a `Generated` folder beside the project) and `wait`.
A request the tool cannot do (a size, length, rate or input its `capabilities` do not allow,
§3.1) is refused before anything runs, saying what the tool can do and the nearest value it
allows.
`generate.image`, `generate.video` and `generate.sound` take `variations` (1 to 4): each gets the
next seed, all are imported, the first is the one placed, and `generate.pick` swaps another in.

- **One at a time.** A generation holds the shared GPU lock, so a second one is refused while one
  runs, and it waits its turn behind another session's model. `jobs.cancel {job}` stops it: the
  program is killed and nothing is changed.
- **One undo step.** Results are imported and placed when the job finishes (each UI frame,
  `jobs.list`, or any `generate.*` command). A track that filled up meanwhile is not overwritten:
  the next free one is used. Placed pictures and clips are scaled to fill the frame.
- **Kept with the media.** The description, tool, seed and size are in the item's metadata
  (Description, Log Note, Generator, Credit).

```sh
CLI=target/release/filmcraft-cli
$CLI exec generate.providers
$CLI --project film.fcproj exec generate.image \
    '{"prompt":"A white lighthouse on a rocky coast at golden hour","place":true,"seconds":3,"wait":true}'
$CLI --project film.fcproj exec generate.fillGap '{"track":"V1","atSeconds":12.5,"wait":true}'
$CLI --project film.fcproj exec generate.augment \
    '{"instruction":"Add a small red sailboat on the sea","atSeconds":4,"wait":true}'
```

## 2. The Generate panel

Window ▸ Generate. Six modes: **Picture**, **Clip**, **Sound**, **Fill gap**, **Augment**,
**Restyle clip**. Pick
the tool, describe what to make, and press Generate; the work runs in the background with a
progress bar and a Stop button, and the result lands at the playhead. Under the tool menu the
panel shows what the model's licence asks to be credited ("Made with …"), its licence line and,
when its entry says, what it can do ("What it can do: …", §3.1). "Generated in this project" lists the last results with a Place button, and a Use button on a
variation that puts it in the place of the one on the timeline.

- Picture, Clip and Sound have **How many to choose from** (1 to 4).
- Picture has a movement menu for the placed picture (still, push in, pull out, drift, circle).
- Clip has **Carry on the movement before the playhead** (`generate.extend`).
- Sound has **As long as the clip selected in the timeline**.
- Fill gap has a menu for what the clip is made from: the frame before, the frame after, both,
  the movement before the gap, or the description only. A choice the tool cannot do says so.
- Restyle clip makes the clip selected in the timeline again (`generate.restyle`): a menu of the
  tool's ways (one whose adapter is missing says so), **Change the first frame first** (an
  instruction for the picture tool, with the media selected in the Project panel as its
  pictures), and **How closely the clip is followed**.

The panel's state is `ui.generate` (`ui.inspect`), settable with
`ui.set {"generate": {"mode": "augment", "prompt": "add snow", "animate": true}}` (also
`variations`, `depthMove`, `forClip`, `fillFrom`, `carryOn`, `restyleHow`, `restyleLook`,
`restyleStrength`); its controls have automation ids
`generate.*` (`ui.elements {"prefix": "generate."}`).

## 3. Setting up the tools

The tools are listed in one file you keep:

- macOS: `~/Library/Application Support/FilmCraft/generation/providers.json`
- elsewhere: `generation/providers.json` in the app's data folder (`generate.providers` prints the path)

No command writes this file. What may be run on your computer is decided by whoever can edit it,
never by a parameter that arrived over the control channel or MCP. A prompt is passed to the
program as one argument, with no shell in between, so it cannot become a command.

[`examples/generation/providers.example.json`](../examples/generation/providers.example.json) is a
starting point for three MLX tools on Apple Silicon; replace `/path/to/…` with where you installed
them. Each entry:

| Field | Meaning |
|---|---|
| `id`, `kind`, `name` | `kind` is `image`, `video`, `music` or `sfx` (or `composition`, below). |
| `command` | The program (an absolute path) and its arguments. Placeholders: `{out}` (required), `{prompt}`, `{negative}`, `{width}`, `{height}`, `{seed}`, `{seconds}`, `{frames}`, `{fps}`, `{image}`, `{lastImage}` (the picture a clip ends on), `{lastFrame}` (the index of the clip's last frame), `{steps}`, `{strength}`. An argument that is exactly `{references}` becomes `perReference` once per reference picture (`{path}` is its file); one that is exactly `{optional}` becomes every group of `optional` whose value the request gave (`image`, `lastImage`, `transparent`, `negative`, `steps`, `strength`, `seconds`). A video tool with a `lastImage` group can end a clip on a picture (`from: "both"`). |
| `env`, `cwd` | Environment variables and the folder to run in. |
| `extension` | The file type written (default `png`, `mp4`, `wav` by kind). |
| `multipleOf`, `defaultSide`, `maxSide` | Sizes the model takes: each side a multiple of this; the long side when none is asked for; the largest side. The shape follows the open sequence. |
| `fps`, `framesStep`, `framesOffset`, `maxSeconds` | Video: the rate it makes, and frame counts it takes (`framesOffset` plus a multiple of `framesStep`; LTX takes 8k + 1). |
| `cutoutCommand`, `cutoutPromptSuffix` | For a picture tool with no transparent output of its own: the program that removes a picture's background (`{image}` in, `{out}` out), run after it when `transparent` is asked, and what to add to the prompt of such a picture (a plain background cuts cleanly). The uncut picture is kept beside the result as `… source <n>.png`. |
| `extendCommand`, `carryFrames`, `extendMultipleOf` | For a video tool that can carry a shot on (`generate.extend`, `from: "motion"`): the program that is given the last `carryFrames` frames of picture (default 33) as a small H.264 video (`{video}`; each side a multiple of `extendMultipleOf`, default 64, in the shape of the sequence) and writes that video with `{frames}` new frames after it to `{out}`. The app places the new part. |
| `restyleCommand`, `restyleOptional`, `restyles`, `restyleEstimateSeconds`, `restyleMaxSeconds` | For a video tool that can make a clip again (`generate.restyle`). `restyleCommand` is the program: it is given the stretch as a small H.264 video (`{video}`, each side a multiple of `extendMultipleOf`, holding exactly `{frames}` frames) and the adapter of the way asked for (`{adapter}`), and writes a clip as long (`{out}`); `{strength}` is how closely to follow it, and an argument that is exactly `{optional}` takes the groups of `restyleOptional` (`image`: the picture to start on). `restyles` names the ways: `{"edges": {"adapter": "/abs/file.safetensors", "args": ["--how", "edges"], "about": "one line", "license": "…"}}`; a way is there when its adapter file is. `restyleMaxSeconds` (default 5) is the longest stretch made in one go. |
| `maxReferences`, `referencePromptPrefix`, `referenceMarker` | Reference pictures: how many, what to put before a prompt that does not mention them itself, and how a prompt mentions one. |
| `estimateSeconds`, `timeoutSeconds` | For the progress bar (the tools say nothing while they work), and when a run is given up on (default an hour). |
| `capabilities` | What the tool can do: sizes, shapes, lengths, frame counts, rates, inputs (§3.1). Optional. |
| `comfyui` | Instead of `command`: a workflow of a local ComfyUI server (§3.2). |
| `blender` | With `kind: "scene"`: Blender, rendering 3D scenes without its window (§3.3). |
| `license`, `attribution`, `source` | Shown in `generate.providers` and the panel. |

A program must write exactly the file `{out}` names and exit with status 0. Its output goes to a
log beside the result; on failure its last lines are the error message.

**The composition renderer.** One more kind of entry names the program that renders motion
graphics written as code for `motion.render`, `motion.stills` and `motion.check`
([raea-workflow.md](raea-workflow.md)). Today that is RAEA, under the id `raea`:

```json
{"id": "raea", "kind": "composition", "name": "RAEA", "command": ["/absolute/path/to/raea"], "timeoutSeconds": 3600}
```

`command` is the program alone (the motion commands build its command line), or leave it out:
RAEA is then the newest `~/.raea/versions/<version>/bin/raea` (the home folder read when the
command runs), else `raea` on PATH. `env`, `timeoutSeconds` (a render; default an hour) and
`estimateSeconds` (the progress bar) apply; generation fields are refused. With none of these,
the motion commands say how to install RAEA. `generate.providers` lists the entry with the
others. EffectCraft (PX4) is not an entry here: it is linked into the AutoCut app, renders
`.ectemplate` packages rather than compositions written as code and runs no program of its own,
so it has its own commands, `effectcraft.templateInfo` and `effectcraft.renderTemplate`
([autocut-features.md](autocut-features.md)). Remotion is never added (its licence asks a company
of four or more for a paid licence).

**Progress.** A program that prints a line `progress 0.42` (or `progress 42%`, or `progress 3/8`)
moves the bar to there; the last such line counts. A program that prints none gets a bar that
follows `estimateSeconds`.

[`examples/generation/ltx_video.py`](../examples/generation/ltx_video.py) is a small runner for
LTX-2 through `ltx-2-mlx` (model folder, size, frame count, rate, seed, optional start picture,
optional last picture). Run it with the Python of the environment that has `ltx_pipelines_mlx`.
[`examples/generation/ltx_extend.py`](../examples/generation/ltx_extend.py) is the `extendCommand`
for the same tool: it carries a video on by a number of new frames.
[`examples/generation/ltx_restyle.py`](../examples/generation/ltx_restyle.py) is the
`restyleCommand`: it draws the lines of the clip (with ffmpeg, which `ltx-2-mlx` needs anyway) and
runs the tool's `ic-lora` command with an adapter (§6).
[`examples/generation/cutout.py`](../examples/generation/cutout.py) removes a picture's background
with `rembg` (U2-Net), for `cutoutCommand`.

### Tested on an M3 Max with 36 GB (19 and 20 November 2026)

| Kind | Tool | Time |
|---|---|---|
| Picture, 1024×576 | MiniMax-H3-Image (MLX, int8) through `h3-image-mlx` | 37 s |
| Picture changed from a frame (one reference) | the same | about 4 minutes |
| Clip, 3 s at 480×272, from a picture | LTX-2.3 (MLX, q4 distilled) | 51 s |
| Gap of 2 s filled from the frame before it | the same | 41 s |
| Sound effect, 6 s | Stable Audio 3 small (MLX) | a few seconds |
| Music, 8 s | the same, music model | a few seconds |
| Cut-out, 768×768 (picture, then background removal) | MiniMax-H3-Image, then rembg | about 50 s |
| Gap of 2 s filled from the frame before it to the frame after it | LTX-2.3 | 42 s |
| Gap of 2 s filled by carrying the movement on | the same, its extend step | 68 s |
| A shot carried on by 2 s | the same | 64 s |
| Two variations of a picture, 768×448 | MiniMax-H3-Image | 58 s |
| A scene's six assets: a character, two poses made from it (one a cut-out), a plate, the plate moving for 3 s, a sound | all three tools | 6 min 36 s |
| A 3 s clip made again as a painting, 576×320 (`generate.restyle`, edges) | LTX-2.3 with the Union-Control adapter | 1 min 43 s |
| A 3 s clip of a walking man made again with a robot in his place (`look` with one reference, then edges) | MiniMax-H3-Image, then LTX-2.3 with the adapter | 7 min 10 s |

The two poses came out as the same character as the first picture. One deleted picture of the
list was made again in 31 s, and the file was the same, byte for byte.

### 3.1 What a tool can do: `capabilities`

An entry can say what its tool makes and takes. `generate.*` then checks every request against
it **before anything is written or run**, and refuses what the tool cannot do with a message
that says what it can and, where one is near, the value it allows (`SDXL cannot make 1000×1000:
it makes widths 512 to 2048 px in steps of 64, heights 512 to 2048 px in steps of 64 (nearest:
1024×1024)`). Where the request left a value open (no size, so the sequence's shape; a gap's or
a clip's length), the nearest allowed value is used instead of a refusal. Every field is
optional; without the block, or without a field, nothing is checked there and the tool behaves
as it always did.

```json
"capabilities": {
  "width": {"min": 256, "max": 768, "step": 32}, "height": {"min": 256, "max": 768, "step": 32},
  "aspects": ["16:9", "9:16", "1:1"],
  "seconds": {"max": 10}, "frames": {"min": 9, "max": 241, "step": 8, "offset": 1}, "fps": [24],
  "firstFrame": true, "lastFrame": true, "audio": false
}
```

| Field | Kinds | Meaning |
|---|---|---|
| `sizes` | image, video, scene | Fixed sizes, `[[1024, 1024], [1344, 768]]` (instead of `width` and `height`). |
| `width`, `height` | image, video, scene | `{min, max, step}`: sides that are multiples of `step` from `min` to `max`. |
| `aspects` | image, video, scene | Shapes it makes, `"16:9"` (2 % either way counts). |
| `seconds` | video, music, sfx, scene | `{min, max}`: the lengths it makes. |
| `frames` | video, scene | `{min, max, step, offset}`: frame counts `offset + k·step` (LTX: step 8, offset 1). The length a clip is asked for is rounded up to one. |
| `fps` | video, scene | The rates it makes; the entry's own `fps` must be one of them. |
| `firstFrame`, `lastFrame` | video | Whether a clip can start or end on a given picture. `true` needs an `{image}` (or `lastImage` option) to put it in; `false` refuses `image`, `lastImage` and the gap fills that need them. |
| `references` | image | The most reference pictures, a frame taken from the timeline included (0: none). |
| `transparent` | image, scene | Whether it makes transparent pictures (its own option or its cut-out step). |
| `negative` | image | Whether it takes a negative prompt. |
| `steps` | image | `{min, max}` sampler steps. |
| `audio` | video | Whether a clip comes with its own sound. |

The block is checked when the list is read: a step of 0, a range whose `min` is above its
`max`, a grid that holds no value, a field the tool's kind is never asked for, an unknown field,
or a claim the entry cannot back (`firstFrame: true` with nothing taking `{image}`) makes the
list invalid, and `generate.providers` says why. `generate.providers` lists each tool's block
and a line in words (`can`), which the Generate panel shows under the tool. The cut-out, extend
and restyle programs keep their own fields; the block describes the main command.

### 3.2 A ComfyUI workflow as a tool

A tool can be a workflow of a [ComfyUI](https://github.com/comfyanonymous/ComfyUI) server running
on this computer. ComfyUI is a separate program (GPL-3.0) that AutoCut only talks to over HTTP; it
is not linked, shipped or read. Build and test the workflow in ComfyUI, then save it in its API
format (Export (API), a JSON object of nodes), and describe it:

```json
{
  "id": "sdxl", "kind": "image", "name": "SDXL base (ComfyUI)",
  "comfyui": {
    "server": "http://127.0.0.1:8188",
    "workflow": "/absolute/path/workflows/sdxl_api.json",
    "inputs": {
      "6.text": "{prompt}",
      "7.text": "{negative}",
      "3.seed": "{seed}",
      "5.width": "{width}", "5.height": "{height}",
      "3.steps": 30
    },
    "output": "9",
    "models": [{"file": "sd_xl_base_1.0.safetensors", "license": "CreativeML-OpenRAIL++-M"}],
    "allowLicenses": ["CreativeML-OpenRAIL++-M"]
  },
  "capabilities": {"width": {"min": 512, "max": 2048, "step": 64}, "height": {"min": 512, "max": 2048, "step": 64}, "negative": true},
  "estimateSeconds": 40, "timeoutSeconds": 900,
  "license": "CreativeML Open RAIL++-M", "attribution": "Stable Diffusion XL"
}
```

| Field | Meaning |
|---|---|
| `server` | The server, on this computer only: `http://127.0.0.1:<port>`, `http://localhost:<port>` or `http://[::1]:<port>` (default `http://127.0.0.1:8188`). Any other host, https or a path makes the list invalid. |
| `workflow` | The API-format workflow (an absolute `.json` path). The editor's own format is refused with a note to export it. |
| `inputs` | `"<node id>.<input>"` → what that input takes: one placeholder alone (`"{seed}"` keeps its type: a number stays a number), text with placeholders (`"{prompt}, film still"`), or a fixed number, text or `true`/`false`. Placeholders: `{prompt}`, `{negative}`, `{width}`, `{height}`, `{seed}`, `{seconds}`, `{frames}`, `{lastFrame}`, `{fps}`, `{steps}`, `{strength}`, `{transparent}`, and the files `{image}`, `{lastImage}`, `{video}`, `{reference1}` … `{reference16}`. A value the request did not give leaves the workflow's own. Each named node and input must be in the workflow. |
| `output` | The node whose saved file is the result (default: the first saved file of the tool's type, `png`, `mp4` or `wav` by kind, or `extension`). |
| `models` | Every model the workflow loads, with its licence: `{"file": "name.safetensors", "license": "SPDX id or the licence's name"}` (a file in a sub-folder matches by its name). |
| `allowLicenses` | Licences you allow for this workflow besides the open ones. |

How a job runs: AutoCut fills a copy of the workflow, uploads the files it needs (a reference
picture, the frame under the playhead) to ComfyUI's `/upload/image` under a name of its own
(no path of your computer is sent), queues it (`POST /prompt`), follows it (`/history/<id>`, and
`/queue` for where it waits: "Waiting in ComfyUI's queue (2 ahead)", "Running in ComfyUI"),
fetches the result (`/view`) and then asks ComfyUI to unload its models and free its memory
(`POST /free`) after every job, whatever became of it, so the GPU lock the job held means the
GPU is free afterwards. It runs under the GPU lock as a job like the others: `jobs.cancel` takes
it out of ComfyUI's queue, or interrupts it when it is the one running (never another
client's), and so does `timeoutSeconds`. A server that stops answering for 30 seconds is given
up on. ComfyUI says how far a job is only on its websocket, which AutoCut does not use, so the
bar follows `estimateSeconds`.

**Models and licences.** AutoCut reads the workflow before each job and refuses to run it when
a node loads a model that `models` does not list (a loader's `*_name`, `model` or `model_path`
input, or any input naming a `.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`, `.sft`,
`.onnx` or `.pkl` file), or when a listed model's licence is not allowed. Allowed are the open
licences (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, 0BSD, ISC, Zlib, Unlicense, CC0-1.0,
CC-BY-3.0/4.0, CC-BY-SA-3.0/4.0, MPL-2.0) and what `allowLicenses` adds after you read it. No
non-commercial or no-derivatives licence (CC BY-NC, CC BY-ND, the FLUX.1 [dev] Non-Commercial
License…) runs unless you name it there, and a monetized channel is commercial use. `inputs`
can never point at a model input: the workflow's models are its own, never a request's.

**Starting ComfyUI** is yours to do, on loopback and offline:

```sh
cd /path/to/ComfyUI
HF_HUB_OFFLINE=1 python main.py --listen 127.0.0.1 --port 8188 --offline --disable-auto-launch
```

`--listen 127.0.0.1` (its default) keeps it to this computer; `--offline` stops its front end
from reaching the internet and leaves out the paid API nodes; `HF_HUB_OFFLINE=1` stops custom
nodes that use the Hugging Face hub from downloading. Download the models you chose beforehand
into its `models/` folders. Then `generate.checkTool {"provider": "sdxl"}` reads the workflow,
checks its inputs, models and licences, asks the server whether it answers, and lists any
problem without running anything.

### 3.3 A 3D scene renderer: Blender without its window

An entry of `kind: "scene"` names [Blender](https://www.blender.org) (GPL, a separate program
AutoCut starts; nothing of it is linked or shipped). `generate.render3d` uses it, and no other
command does:

```json
{
  "id": "blender", "kind": "scene", "name": "Blender (headless)",
  "blender": {"program": "/Applications/Blender.app/Contents/MacOS/Blender", "engine": "eevee"},
  "capabilities": {"fps": [24, 25, 30], "frames": {"max": 1800}},
  "estimateSeconds": 120, "timeoutSeconds": 3600
}
```

| `blender` field | Meaning |
|---|---|
| `program` | Blender's program, an absolute path (refused otherwise). |
| `scene` | The `.blend` rendered when the request names none (absolute). Without either, the script builds a plain stage for a `text` title. |
| `engine`, `samples` | `eevee`, `cycles` or `workbench` (default: the scene's own; EEVEE for the plain stage), and its samples. Cycles uses the GPU (Metal, OptiX, CUDA, HIP or oneAPI) where there is one; `device: "cpu"` says not to. |
| `format` | `png` (default) or `exr` (16-bit half, ZIP). |
| `script` | A script of your own instead of AutoCut's (an absolute `.py`); it gets the same request. |
| `factoryStartup` | `false` starts Blender with your own preferences and add-ons (a scene that needs an add-on); default `true`. |

AutoCut runs

```text
blender --background --factory-startup --disable-autoexec --python-exit-code 3 [scene.blend] \
    --python <data dir>/generation/scene_render.py -- '<request as JSON>'
```

so a scene's own Python (drivers, startup scripts) never runs, and a failing script ends Blender
with an error whose last lines AutoCut shows. The script
([`crates/engine/src/scene_render.py`](../crates/engine/src/scene_render.py)) is AutoCut's own,
written for this from Blender's documented Python API; it is written to the data folder before
a render. It sets the size, rate, frame range, transparency and format, sets the request's texts
and objects, keys the camera track onto the scene's camera, renders, and prints `progress k/n`
after each frame for the bar. The render runs under the GPU lock as a job (`jobs.cancel` stops
Blender; so does `timeoutSeconds`), and the frames come in as an image sequence (one frame: a
still), placed above what is at the playhead when they are transparent.

```sh
# a 3D title on the plain stage: 3 s at the sequence's size and rate, with alpha
$CLI --project film.fcproj exec generate.render3d '{"text": "Chapter Two", "wait": true}'
# a scene of yours, its title text and one object changed, frames 1 to 120 at 24 fps
$CLI --project film.fcproj exec generate.render3d \
    '{"scene": "/abs/scenes/lower_third.blend", "texts": {"Name": "Part Two"}, "objects": {"Logo": {"hide": true}},
      "frameEnd": 120, "fps": 24, "wait": true}'
```

`generate.render3d` takes `scene`, `text` (the plain stage's title; with a scene, the text object
named `Text`), `texts` (`{object name: text}`, at most 32), `objects` (`{name: {location: [x, y,
z], rotation: [x, y, z] in degrees, scale: n or [x, y, z], hide}}`, at most 64), `camera`,
`frameStart` (default 1), `frameEnd` or `seconds` (default 3 s), `fps` (default the tool's, else
the sequence's, moved to the nearest rate its capabilities list), `width`/`height` (default the
sequence's), `transparent` (default on), `format`, `cryptomatte`, `seed`, and the usual `dir`,
`place`, `track`, `atSeconds`, `wait`. Frame numbers stop at 9999 (the frames are named
`frame_0001.png` …). Everything is checked before Blender starts: the files are there, the
names, numbers and camera track are well formed, and the tool's capabilities allow the size,
rate, frame count and transparency.

**A match-move.** `camera` is a camera track, a JSON file in metres on Blender's axes (Z up):

```json
{"sensorWidth": 36,
 "frames": [{"frame": 1, "position": [0.0, -8.0, 1.6], "rotation": [90.0, 0.0, 0.0], "focalLength": 35},
            {"frame": 2, "position": [0.05, -7.98, 1.6], "quaternion": [0.707, 0.707, 0.0, 0.0]}]}
```

Each frame has `position` and a `rotation` (XYZ Euler, degrees, as Blender shows a camera's) or a
`quaternion` (w, x, y, z), and optionally `focalLength` (mm); they are keyed onto the scene's
camera (one is made if the scene has none). A tracked camera exported in this shape (EffectCraft's
tracked-camera export, or any solver's, converted to these axes) lets the render sit in the
footage it was tracked from: render with alpha and lay it over the clip.

**Cryptomatte.** With `"format": "exr", "cryptomatte": true` the object, material and asset
passes are written as a multilayer EXR a frame in a `cryptomatte` folder inside the frames'
folder, for compositing elsewhere; the frames AutoCut imports stay plain RGBA (its EXR reader
takes R, G, B and A).

`generate.checkTool {"provider": "blender"}` asks Blender for its version (`--version`, nothing
is rendered) and checks that the entry's scene and script files are there.

### 3.4 Blender over MCP, for interactive work

For building a scene with an agent before rendering it, the community add-on
[MCP for Blender](https://github.com/ahujasid/blender-mcp) (MIT; `uvx mcp-for-blender`) lets an
MCP client drive a running Blender. It is not part of AutoCut and runs Python inside Blender;
installing it is your choice. When you use it:

1. **Save the .blend first.** Its code tool runs any Python in Blender.
2. **Telemetry off.** Set `DISABLE_TELEMETRY=true` in the MCP server's environment (the `env`
   block of the client's server entry), and leave the add-on's content-collection checkbox off.
3. **Safe mode on.** Set `BLENDER_MCP_SAFE_MODE=1` in the same environment, so the scripts it
   sends are screened.
4. **Its own port, on this computer.** Not 9876 (its default, which another Blender may hold)
   and not 9877 (AutoCut's control channel): for example `BLENDER_PORT=9878` (or `--port 9878`)
   for the server and the same port in the add-on's panel, with its bind address on localhost
   and "Allow Remote Connections" off. Its socket has no sign-in.
5. **Assets CC0 or CC BY only.** Poly Haven's are CC0. From Sketchfab take only models marked
   CC0 or CC BY, and note each CC BY author and link for the video's credits. Turn off the
   integrations you do not use (in the add-on's sidebar); do not use generated 3D models whose
   licence you have not read.
6. **Hold the GPU** while you work, so AutoCut's own generation waits:
   `filmcraft-cli locks hold gpu --minutes 30 --note "Blender over MCP"` in the background, and
   stop it when done, before the render (which takes the lock itself).
7. **Render with AutoCut.** Save the scene and render it with `generate.render3d {"scene":
   "/abs/scene.blend", …}`, so the frames come in as a clip, under the lock, with the request on
   record.

## 4. Licences are yours to accept

The models have their own terms; the app only runs what you installed. At the time of writing:

- **MiniMax-H3-Image**: MiniMax-H3-Image Community License. Not for use in the EU, the UK, South
  Korea or the USA; written permission is needed above US$1M yearly revenue or for a hosted
  service; "MiniMax-H3-Image" must be shown where it is used (the panel does, from `attribution`).
- **LTX-2**: LTX-2 Community License. Companies with US$10M yearly revenue or more need a paid
  licence.
- **Stable Audio 3**: Stability AI Community License.

Read the licence that came with your weights before you publish. A platform may also ask you to
say that a realistic scene was generated or altered (YouTube's altered-content label): that is
your decision when you upload, and the item's metadata records what was generated.

## 5. Assets for RAEA scenes

A generated picture is a good asset for a [RAEA](raea-workflow.md) composition: a background
plate, an illustrated story beat, or a cut-out character that the composition animates. One file
at a time, `generate.image` with `"import": false` and `"dir"` writes it without touching the
project. For a whole scene, keep a list beside the composition and let `generate.assets` build it.

`assets.json`, in the folder the files go to:

```json
{
  "width": 768, "height": 768,
  "assets": [
    {"name": "mascot", "prompt": "A friendly small orange robot mascot, full body, flat vector illustration", "seed": 21},
    {"name": "mascot wave", "prompt": "The same robot, waving", "references": ["mascot"], "transparent": true},
    {"name": "plate", "prompt": "A pastel sky with flat clouds and rolling hills", "width": 1024, "height": 576},
    {"name": "plate moving", "kind": "video", "prompt": "The clouds drift slowly to the right", "image": "plate", "seconds": 3},
    {"name": "swish", "kind": "sfx", "prompt": "a soft short airy swish", "seconds": 1.5}
  ]
}
```

```sh
$CLI exec generate.assets "{\"manifest\":\"$FX/assets/assets.json\",\"dryRun\":true}"   # what would be made
$CLI exec generate.assets "{\"manifest\":\"$FX/assets/assets.json\",\"wait\":true}"     # make what is missing
```

- Each entry becomes `<name>.<ext>` in that folder (`png`, `mp4`, `wav` by the tool). A file that is
  there is kept; delete it, or name it in `only` with `force: true`, to have it made again.
- `kind` is `image` (default), `video`, `sfx` or `music`. `width` / `height` at the top are the
  default size; without any, pictures take the shape of the open sequence.
- **One character across a scene.** `references` names earlier entries of the list (or files): a
  pose is made from the character's first picture, so the character stays the same. A video's
  `image` is its first picture, by name or file.
- **Made the same again.** The seed is in the list; an entry without one gets a seed that follows
  its name, so a deleted file comes back from the same request.
- `transparent` gives a cut-out; the uncut picture is kept as `<name> source.png`.
- At most 32 entries; names are letters, digits, spaces, `-` and `_`. Nothing is imported unless
  `import: true` is given, so the project is not touched.

Refer to the files from the composition as local assets, render with `examples/raea/render.py` as
usual, and import the render. Other uses in a scene:

- **A moving plate**: a `video` entry made from a plate, under the composition's text and graphics.
- **A picture that moves in depth**: place a generated still with `"move": "in"` (3D Photo).
- **Sound for the scene**: an `sfx` entry, or `generate.sound` with `"clip"` set to the imported
  render's clip, which sizes the sound to it.

## 6. Making a clip again

`generate.restyle` gives a clip to the video tool as a small video and has an **adapter** of its
model make it again. What stays and what changes is the adapter's doing:

| Way | Adapter | What it does | State |
|---|---|---|---|
| `edges` | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control), 654 MB, open download | Follows the lines of the clip: shapes and movement stay, the look follows the description | Tested |
| `relight` | [Lightricks/LTX-2.3-22b-IC-LoRA-Relight](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Relight), 327 MB, gated | Keeps the clip, changes its light | Wired in the example list, not tried: the download needs a Hugging Face login and the terms accepted there |

Both are under the LTX-2 Community License. Download an adapter into a folder of your own
(`hf download <repo> --local-dir …`) and name its file in `restyles`.

```sh
# a new look: the shapes and the movement of the selected clip stay
filmcraft-cli exec generate.restyle '{"prompt": "A watercolour painting of a lighthouse on a rocky coast", "wait": true}'
# another character: the picture tool puts it into the first frame, the clip follows from there
filmcraft-cli exec generate.restyle '{"prompt": "A small orange robot walks along a park path toward the camera",
  "look": "Replace the man with the robot from the second picture, in the same place and pose",
  "references": ["/abs/robot.png"], "wait": true}'
```

- **What is made again** is what the program shows over the clip's stretch (titles and layers
  above it included): hide what should not be part of it first.
- **Length.** The tool makes frame counts of its own (LTX: 8k + 1). A clip of up to
  `restyleMaxSeconds` is written for the tool at a slightly lower frame rate, so that such a
  count spans all of it and every new frame shows the moment of the old one; the new clip is
  then slowed by a few percent to the clip's length. A longer clip is made again from its
  start, and `left` says how many seconds stay as they were.
- **A swap follows the old outline.** With `edges` the new character moves inside the lines of
  the old one. A character of about the same build works (the tested robot took the place of a
  walking man); a very different shape will be bent towards the old outline. Lower `strength`
  loosens that, and loosens everything else too.
- **What the example runner does, and why.** On the 4-bit weights the tool's default (a
  half-size pass, then a refine that does not see the control) ignored the lines entirely at
  448×256, whether the adapter was fused into the weights or not. One pass at full size with
  the adapter applied as the model runs (`LTX2_LORA_MODE=unfused`, `--single-stage`) follows
  them. `--two-stage` gives the tool's default back.
- Depth and pose control are the same adapter given another control video; nothing here draws
  those yet (`ltx_restyle.py --how given` takes a control video made elsewhere).

## 7. Limits

- **Not tried with the real programs yet.** The ComfyUI and Blender tools are tested against
  stand-ins (a small HTTP server answering ComfyUI's endpoints, a script that writes frames as
  Blender would); neither ComfyUI nor Blender was run. The first real use should start with
  `generate.checkTool`, then a small picture or a few frames.
- A ComfyUI job's bar follows `estimateSeconds` (its step progress is only on its websocket).
  Uploading a file a workflow takes goes through `/upload/image`, which some custom loader nodes
  may not read from; such a workflow takes its file another way.
- Cryptomatte passes use the compositor's File Output node as Blender 4 has it; a Blender whose
  compositor differs renders the frames and notes in its log that the passes were not written.

- Short clips only: the tested video model makes up to ten seconds at a small size; AutoCut scales
  it to the frame. Use Upscale and Restore (`media.upscale`) and Smooth Slow Motion
  (`media.interpolate`) on the result when it needs more.
- A gap filled `from: "before"` continues the look of one frame, not the motion of the clip
  before it. `from: "motion"` and `generate.extend` continue the motion, from a small copy of the
  picture (the tool's size), and take longer. Both need 1.4 s of picture before the gap.
- A clip that ends on a given picture (`from: "both"`) arrives close to it, not on it exactly:
  look at the cut.
- An augmented frame is a new picture laid over the clip. It does not track what moves under it;
  for a moving shot, animate it (`animate: true`) or keep it short.
- A cut-out made by the background remover can keep a faint halo and fill small enclosed gaps
  (between legs, inside a handle). A flat illustration on a plain background cuts cleanest.
- The tested tools print no progress line, so their bar follows their usual time.
- A clip made again (`generate.restyle`) is small (the tool's size, 576×320 for 16:9) and at
  most five seconds a go; faces and fine detail change. Use it for a look or a character, not
  to touch up a shot.
- Relighting a clip and lip dub are not usable yet: their adapters are gated downloads (a
  Hugging Face login and the terms accepted there). Relight is wired and waits for its file;
  lip dub is not built. Relighting a still works today as an edit (`generate.image` with
  `references`, or `generate.augment`).

## 8. What comes next

[`plan/generation-2026-11.md`](../plan/generation-2026-11.md) lists what is done and what is
open: relight and lip dub (gated downloads), depth and pose control videos for
`generate.restyle`, and the other adapters of the clip model (outpainting to another frame
shape, removing an object from a clip), each a model download that is the user's decision.
