# Generation in the editor

AutoCut can make pictures, short clips, music and sound effects with the generation models you
have installed on your own computer, import the result as ordinary media and place it on the
timeline. It can also fill an empty stretch of a track with a clip that carries on from the frame
before it (or runs from the frame before to the frame after, or continues the movement), carry a
shot on past its end, change what a frame shows from an instruction ("add a red sailboat"), make
several variations to choose from, and build the pictures, clips and sounds a motion-graphics scene
needs from one list.

Nothing here is a model built into the app, and the app downloads nothing. A **generation tool**
is a program on your computer (an MLX, PyTorch or ComfyUI command line you set up yourself). The
app runs it, waits for the file it writes, and takes it from there. This is the same stance as
[RAEA](raea-workflow.md): separately installed, kept outside the repository and the app bundle.

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

All of them take `provider` (a tool id; default: the first installed one of the kind), `seed`,
`dir` (where the file goes; default a `Generated` folder beside the project) and `wait`.
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
panel shows what the model's licence asks to be credited ("Made with …") and its licence line.
"Generated in this project" lists the last results with a Place button, and a Use button on a
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
others. EffectCraft (PX4) is the next composition renderer planned for this list; Remotion is
never added (its licence asks a company of four or more for a paid licence).

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
