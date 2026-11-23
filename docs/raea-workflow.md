# Local RAEA renders in AutoCut

Use [RAEA](https://github.com/shinyobjectz/raea) to render custom animations, full scenes,
3D explainers, motion graphics, and transition overlays on the local GPU. AutoCut imports
the result as timeline media. Keep the editable TSX or HTML alongside its assets and render
in a new revision directory when the design changes.

A route animation, an exploded product view, or an illustrated story beat can be a full
scene. Lower thirds, callouts, and wipes can retain transparency over the original footage.

## Setup

On the editing Mac, the verified `v0.1.8` Apple Silicon release lives at
`~/.raea/versions/0.1.8/bin/raea`, with `v0.1.0` kept beside it at `~/.raea/versions/0.1.0` for
rollback. On another machine, use the matching official
[release](https://github.com/shinyobjectz/raea/releases/tag/v0.1.8) and check its SHA-256 against the
release's `.sha256` file (`raea-0.1.8-macos-arm64.tar.gz`: `64dc7085…de68b`). Unpack it into its own
`~/.raea/versions/<version>/` (`bin/` and `share/`). The
[installation guide](https://raea.io/docs/start/install.md) covers platform support.
Set `RAEA="$HOME/.raea/versions/0.1.8/bin/raea"` on this Mac. Read the installed version's
composition guidance with `"$RAEA" skill` or use the
[first-video guide](https://raea.io/docs/start/first-video.md). The render helper below picks the
newest version under `~/.raea/versions` unless `--raea` names one.

What changed from 0.1.0 for this workflow (checked 23 November 2026 on the sample below):
- The sample's stills render byte for byte the same, and the helper's MP4, 4K and transparent PNG
  handoffs pass unchanged.
- `raea check --quick` adds exposure warnings. A transparent overlay such as the split wipe reads as
  "crushed to black" there; that is expected for an alpha layer, not a fault.
- `raea render` writes `motion.render.json` beside the frames: every text box per frame (text, box,
  size, colour, opacity), which the craft checks can read instead of guessing from pixels.
- HTML compositions render with `"$RAEA" render PAGE.html -o OUT.mp4|DIR [--at A,B] [--alpha]`:
  a root element with `data-composition-id`, `data-width` and `data-height`, and clips shown in
  windows by `data-start` and `data-duration` (seconds). GSAP timelines registered in
  `window.__timelines` drive animation on the page clock.
- GSAP, three.js and React Three Fiber are not bundled. A project gets them from npm with
  `"$RAEA" add gsap` or `"$RAEA" add three@0.185.0 @react-three/fiber@9.8.2`: that is a download,
  so ask first. Leave `raea bug` (it sends a report to RAEA's server) and `raea skills install`
  (it changes agent settings) alone.
- Checked on 0.1.8 with gsap 3.14.2, three 0.185.0 and @react-three/fiber 9.8.2 (18 packages,
  37 MB in the project's `node_modules`): a GSAP page (`<script src="node_modules/gsap/dist/gsap.min.js">`,
  a paused timeline registered under the root's `data-composition-id`), a plain three.js scene in
  `<Three program={fn}>` and an `<R3F>` scene (both `import … from "raea/r3f"`) render stills and
  MP4s with the motion where it should be.
- `--at A,B` names stills differently: an HTML page's by frame number (`frame-00030.png`), a TSX
  composition's by position (`frame-00000.png`, `frame-00001.png`).

RAEA is a separately installed tool for internal production under its
[FSL-1.1-ALv2 license](https://github.com/shinyobjectz/raea/blob/v0.1.0/LICENSE).
Keep the binary, SDK and runtime outside the repository and AutoCut packages. The app's
rendered-media import is engine-independent. Use trusted compositions and local assets.

## Generated assets

A picture made by a local generation tool (`generate.image`, see [generation.md](generation.md))
can be a composition's background plate or, with `"transparent": true`, a cut-out character it
animates. Write it straight into the composition's assets with `"import": false` and
`"dir": "<build>/fx/assets"`, and keep the prompt and seed beside the TSX.

For a whole scene, list its pictures, clips and sounds in `<build>/fx/assets/assets.json` and run
`generate.assets {"manifest": "…/assets.json", "wait": true}`: it makes what is missing and keeps
the rest, a pose can be made from the character's first picture by name (`references`), a plate
can be animated (`kind: "video"`, `image`), and a deleted file comes back from the same seed
([generation.md](generation.md) §5).

## Authoring a composition

These rules keep a render repeatable and readable. They are written in our words; several come from
HyperFrames' composition rules ([heygen-com/hyperframes](https://github.com/heygen-com/hyperframes),
Apache-2.0; see `ATTRIBUTION.md`).

- **Motion comes only from the frame number.** In TSX that's `useCurrentFrame()`; in an HTML page,
  GSAP timelines on the page clock. Never use `Math.random()`, `Date.now()`, timers,
  `requestAnimationFrame` or network fetches, so a re-render matches frame for frame. For variety,
  use a seeded generator (for example mulberry32) with its seed written in the file.
- **HTML timelines are paused and registered.** Create every GSAP timeline with `{paused: true}` and
  register it as `window.__timelines["<composition-id>"]`. Build it synchronously while the page
  loads, never inside `async`, `setTimeout` or a promise. Never play, pause or seek media yourself:
  the renderer owns the clock. Let running text wrap instead of breaking it with `<br>`.
- **Lay out from the composition's size** (`useVideoConfig()`, or `data-width` and `data-height`).
  Keep text inside the platform's safe area, which `sequence.checkSafeZones` checks after import.
  Keep text at least 2.6 % of the frame's short side (28 px at 1080p), and use tabular numerals
  for numbers that change.
- **One focal element per scene, one entrance per element.** Exits are shorter than entrances.
  Damp springs at 0.9 or more unless a playful style was asked for.
- **Simple moves don't need a composition.** A few tweens of scale, position or opacity on clips
  or graphic layers are cheaper as `keyframes.choreograph`. It takes GSAP's position and ease
  names and writes editable keyframes ([autocut-agent-guide.md](autocut-agent-guide.md) §6.8).
  Use RAEA for what keyframes can't draw.
- **Template cards don't need one either.** A lower third or end card from an `.ectemplate`
  renders in the app with `effectcraft.renderTemplate` ([autocut-agent-guide.md](autocut-agent-guide.md)
  §6.8), placed the same way as a RAEA render.
- **Local, licensed assets** beside the composition: generated plates, the user's footage, sounds
  with a known licence. Take fonts from craft-fonts or the composition's own licensed files.
- **Check before rendering.** Run `raea check --quick`, then stills at the start, the middle and the
  end, and only then the full render. `motion.render.json` lists every text box per frame.

## As engine commands (preferred)

The render, its checks, the import and the placement are engine commands, so the app, the CLI, a
recipe and MCP all drive them the same way. They take the machine's GPU lock themselves, and
`motion.render` and `composition.import` are long jobs (`wait: true`, or follow them in `jobs.list`).

| Command | What it does |
|---|---|
| `motion.check {source, composition?}` | `raea check --quick`: findings with `check`, `level`, `says` |
| `motion.stills {source, composition?, at: [frames], output}` | stills and RAEA's render manifest, with the text boxes per frame |
| `motion.render {source, composition?, output, canvas?, size?, preview?, fps?, alpha?, quality?, holds? \| spec?, expected?, import?, place?, wait?}` | check, render into a new revision folder, native probe (size, rate, gaps), the animation preflight with `qa.motionEnergy`, `autocut-import.json` only on success, then import (MP4, or the PNG sequence at `fps`) and place unless told not to. A render that fails the probe or preflight stays on disk and the error says where |
| `composition.import {source (.tsx), composition?, output, alphaCodec?, place?, wait?}` | one owner per element: RAEA's `export layers` draws only the graphics (ProRes 4444 with alpha, or PNG), and the footage and sounds the composition uses come in as real clips with their times, trims, speed, opacity, box and gain, so they stay editable |

```sh
CLI="$PWD/target/release/filmcraft-cli"
"$CLI" --project "$P" --save exec motion.render "{\"source\":\"$FX/motion.tsx\",\"composition\":\"scene\",
  \"output\":\"$FX/Next Stop v2\",\"spec\":\"$FX/scene-qa.json\",\"place\":true,\"wait\":true}"
```

RAEA is found from `providers.json`'s `kind: "composition"` entry (id `raea`, its program path), else
the newest `~/.raea/versions/<version>/bin/raea`, else `raea` on the PATH ([generation.md](generation.md)).
No command parameter picks the program.

Known limits on 0.1.8:
- `raea check --quick` checks the file's first composition even when `--composition` names another,
  so for `motion.tsx` the scene's check reads the wipe.
- `composition.import` takes `.tsx` only (RAEA's layer export has no HTML path). It shows a footage
  box RAEA crops uncropped and lists it under `approximated`; captions come in as the drawn layer,
  not as AutoCut captions.
- The helper's render cache is not in the commands yet: a new revision always renders.

The Python helper below stays for scripts and older builds.

## Render and review

1. Save the composition and its local assets under `~/Movies/AutoCut/<Video>/build/`.
   Match the target canvas and frame rate. Author a portrait variant if the framing or text
   needs to change for a Short. Render at review size first and regenerate approved graphics
   at final size for 4K delivery.
2. Check `filmcraft-cli locks`. While running RAEA, hold the shared GPU lock in a separate
   terminal with `filmcraft-cli locks hold gpu --minutes 30 --note 'RAEA motion graphics'`.
   Wait for the `holding` message, then stop the holder when RAEA finishes. Extend the hold
   for a longer job. Run `"$RAEA" check file.tsx --quick` and inspect representative stills
   using `"$RAEA" render file.tsx --composition ID --at 0,8,17 -o <new-stills-dir>`.
3. Render with the example helper below. It uses AutoCut's native probe, rejects gaps in
   numbered renders, and writes `autocut-import.json` only after a successful handoff. Each
   destination is a new directory. Pass `--raea /absolute/path/to/raea` for a different install.
   A completed render is retained in that directory even if the native probe or QA command fails.
   Inspect those retained bytes before rendering again; a failed handoff creates no import manifest.

For the sample scene, save `scene-qa.json` in the FX build folder with its deliberate final hold:

```json
{"expected":{"duration":508032000000},"holds":[{"start":228614400000,"end":508032000000,"reason":"settled chapter heading"}]}
```

From the repository root, using the original sample in `examples/raea/motion.tsx`:

```sh
CLI="$PWD/target/release/filmcraft-cli"
FX="$HOME/Movies/AutoCut/My Video/build/fx"
mkdir -p "$FX"

# A complete animated scene, imported as a normal MP4 clip.
python3 examples/raea/render.py examples/raea/motion.tsx "$FX/Next Stop v1" \
    --cli "$CLI" --composition scene --spec "$FX/scene-qa.json"

# An 18-frame transparent transition, imported as one image-sequence clip.
python3 examples/raea/render.py examples/raea/motion.tsx "$FX/Split Wipe v1" \
    --cli "$CLI" --composition wipe --alpha --fps 30
```

The helper's `--fps` is the PNG sequence's import rate and must match the composition's
render rate. MP4 carries its own rate. Use a short PNG sequence for a transparent effect;
use MP4 for an opaque scene or a longer animation.

For review/master sizing, lay out from `useVideoConfig().width` and `.height`, as the
sample does. RAEA v0.1.0 ignores output size overrides on fixed-size compositions;
the helper sets the output stage (0.1.8 renders correctly with it too) and rejects an unexpected
rendered size. A 4K scene
uses `--canvas 960x540 --size 3840x2160`; add `--preview` for a 960-pixel review.

## Place on the timeline

Pass the manifest's `command` and `params` to `command_run` in the same project session as
the other edits. The CLI works too. For example, to import the transparent wipe:

```sh
"$CLI" --project "$P" --save exec file.importImageSequence \
    "path=$FX/Split Wipe v1/frames/frame-00000.png" fps=30
```

Check `filmcraft-cli describe file.importImageSequence` for `fps` when using an older build.
Rebuild AutoCut before importing a sequence with an explicit rate if the field is absent.

Use the returned `items[0]` in `timeline.place`. Put a full scene on the picture track,
or put an alpha sequence on a video track above the shots. For the split wipe, put the
cut nine frames after the overlay starts. Frames 8 and 9 fully cover the cut. Check the
first, middle, and last frames in AutoCut with footage underneath and preview both sides
of the cut at speed. Imported sequences keep the specified source rate when the project
reopens. Leave `fps` out to use Settings ▸ Media ▸ Indeterminate Media Timebase.

Keep the project's render directory with its sources. A `.fcproj` references the MP4 or
the numbered PNGs on disk. Save the project after placement and use AutoCut's existing
colour, audio, narration, captions, privacy checks, and export workflow for the final edit.

See [the local pipeline upgrades](local-pipeline-upgrades.md) for FrameAtlas edit plans, screen protection, editable audio continuity, render-cache commands, RAEA review bundles and the checked 4K delivery recipe.
