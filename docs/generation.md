# Generation in the editor

AutoCut can make pictures, short clips, music and sound effects with the generation models you
have installed on your own computer, import the result as ordinary media and place it on the
timeline. It can also fill an empty stretch of a track with a clip that carries on from the frame
before it, and change what a frame shows from an instruction ("add a red sailboat").

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
| `generate.image` | A picture from `prompt`. `references` (project item ids or absolute paths, at most 16) and `fromPlayhead` / `frameSeconds` (the program frame there, as the first reference) turn it into an edit or put a character into a scene. `transparent` asks for a cut-out. `place` puts it on the timeline for `seconds`. `import: false` with `dir` only writes the file (an asset for a motion-graphics scene). |
| `generate.video` | A clip from `prompt`, `seconds` long, optionally starting from `image` (an item id, a path, or `"playhead"`). Placed at the playhead (`atSeconds`, `track`) unless `place: false`. `audio: true` keeps the clip's own sound. |
| `generate.sound` | `kind: "sfx"` (default) or `"music"` from `prompt`, `seconds` long, placed on a free audio track. |
| `generate.fillGap` | Fills the gap of `track` (default `V1`) under the playhead (`atSeconds`) with a clip that starts from the last frame before it (`from: "before"`, the default), leads into the first frame after it (`"after"`), or follows only `prompt` (`"prompt"`). `all: true` fills every gap of the track (eight at most). A clip longer than the gap is cut to it; a shorter one is slowed towards it (never below 40 %), and `uncovered` says what is left. |
| `generate.augment` | Takes the program frame at the playhead (`atSeconds`), changes it by `instruction` (plus `references`: a character, a prop), and lays the result over the clip on the next free track, as a still for `seconds` or, with `animate: true`, as a clip made from the changed frame (`motion` says what moves). |
| `generate.history` | What was generated into this project: item, tool, description, seed. |

All of them take `provider` (a tool id; default: the first installed one of the kind), `seed`,
`dir` (where the file goes; default a `Generated` folder beside the project) and `wait`.

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

Window ▸ Generate. Five modes: **Picture**, **Clip**, **Sound**, **Fill gap**, **Augment**. Pick
the tool, describe what to make, and press Generate; the work runs in the background with a
progress bar and a Stop button, and the result lands at the playhead. Under the tool menu the
panel shows what the model's licence asks to be credited ("Made with …") and its licence line.
"Generated in this project" lists the last results with a Place button.

The panel's state is `ui.generate` (`ui.inspect`), settable with
`ui.set {"generate": {"mode": "augment", "prompt": "add snow", "animate": true}}`; its controls
have automation ids `generate.*` (`ui.elements {"prefix": "generate."}`).

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
| `id`, `kind`, `name` | `kind` is `image`, `video`, `music` or `sfx`. |
| `command` | The program (an absolute path) and its arguments. Placeholders: `{out}` (required), `{prompt}`, `{negative}`, `{width}`, `{height}`, `{seed}`, `{seconds}`, `{frames}`, `{fps}`, `{image}`, `{steps}`, `{strength}`. An argument that is exactly `{references}` becomes `perReference` once per reference picture (`{path}` is its file); one that is exactly `{optional}` becomes every group of `optional` whose value the request gave (`image`, `transparent`, `negative`, `steps`, `strength`, `seconds`). |
| `env`, `cwd` | Environment variables and the folder to run in. |
| `extension` | The file type written (default `png`, `mp4`, `wav` by kind). |
| `multipleOf`, `defaultSide`, `maxSide` | Sizes the model takes: each side a multiple of this; the long side when none is asked for; the largest side. The shape follows the open sequence. |
| `fps`, `framesStep`, `framesOffset`, `maxSeconds` | Video: the rate it makes, and frame counts it takes (`framesOffset` plus a multiple of `framesStep`; LTX takes 8k + 1). |
| `cutoutCommand`, `cutoutPromptSuffix` | For a picture tool with no transparent output of its own: the program that removes a picture's background (`{image}` in, `{out}` out), run after it when `transparent` is asked, and what to add to the prompt of such a picture (a plain background cuts cleanly). The uncut picture is kept beside the result as `… source <n>.png`. |
| `maxReferences`, `referencePromptPrefix`, `referenceMarker` | Reference pictures: how many, what to put before a prompt that does not mention them itself, and how a prompt mentions one. |
| `estimateSeconds`, `timeoutSeconds` | For the progress bar (the tools say nothing while they work), and when a run is given up on (default an hour). |
| `license`, `attribution`, `source` | Shown in `generate.providers` and the panel. |

A program must write exactly the file `{out}` names and exit with status 0. Its output goes to a
log beside the result; on failure its last lines are the error message.

[`examples/generation/ltx_video.py`](../examples/generation/ltx_video.py) is a small runner for
LTX-2 through `ltx-2-mlx` (model folder, size, frame count, rate, seed, optional start picture).
Run it with the Python of the environment that has `ltx_pipelines_mlx`.
[`examples/generation/cutout.py`](../examples/generation/cutout.py) removes a picture's background
with `rembg` (U2-Net), for `cutoutCommand`.

### Tested on an M3 Max with 36 GB (19 November 2026)

| Kind | Tool | Time |
|---|---|---|
| Picture, 1024×576 | MiniMax-H3-Image (MLX, int8) through `h3-image-mlx` | 37 s |
| Picture changed from a frame (one reference) | the same | about 4 minutes |
| Clip, 3 s at 480×272, from a picture | LTX-2.3 (MLX, q4 distilled) | 51 s |
| Gap of 2 s filled from the frame before it | the same | 41 s |
| Sound effect, 6 s | Stable Audio 3 small (MLX) | a few seconds |
| Music, 8 s | the same, music model | a few seconds |
| Cut-out, 768×768 (picture, then background removal) | MiniMax-H3-Image, then rembg | about 50 s |

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

## 5. Generated pictures in RAEA scenes

A generated picture is a good asset for a [RAEA](raea-workflow.md) composition: a background
plate, an illustrated story beat, or a cut-out character that the composition animates.

```sh
FX="$HOME/Movies/AutoCut/My Video/build/fx"
$CLI exec generate.image "{\"prompt\":\"A friendly cartoon robot mascot waving, full body, flat vector illustration\",
  \"transparent\":true,\"width\":768,\"height\":768,\"import\":false,\"dir\":\"$FX/assets\",\"wait\":true}"
```

The file lands in `$FX/assets` (`Generated Image 1.png`) without touching the project. Refer to
it from the composition as a local asset, render with `examples/raea/render.py` as usual, and
import the render. Keep the prompt and seed (the command's result prints them) beside the
composition so the asset can be made again.

## 6. Limits

- Short clips only: the tested video model makes up to ten seconds at a small size; AutoCut scales
  it to the frame. Use Upscale and Restore (`media.upscale`) and Smooth Slow Motion
  (`media.interpolate`) on the result when it needs more.
- A gap is filled from one frame: the clip continues the look of that frame, not the motion of the
  clip before it.
- An augmented frame is a new picture laid over the clip. It does not track what moves under it;
  for a moving shot, animate it (`animate: true`) or keep it short.
- A cut-out made by the background remover can keep a faint halo and fill small enclosed gaps
  (between legs, inside a handle). A flat illustration on a plain background cuts cleanest.
- The tools do not report progress; the bar follows their usual time.
