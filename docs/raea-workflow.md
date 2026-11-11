# Local RAEA renders in AutoCut

Use [RAEA](https://github.com/shinyobjectz/raea) to render custom animations, full scenes,
3D explainers, motion graphics, and transition overlays on the local GPU. AutoCut imports
the result as timeline media. Keep the editable TSX or HTML alongside its assets and render
in a new revision directory when the design changes.

A route animation, an exploded product view, or an illustrated story beat can be a full
scene. Lower thirds, callouts, and wipes can retain transparency over the original footage.

## Setup

On the editing Mac, the verified `v0.1.0` Apple Silicon release lives at
`~/.raea/versions/0.1.0/bin/raea`. On another machine, use the matching official
[release](https://github.com/shinyobjectz/raea/releases/tag/v0.1.0) and verify its SHA-256.
The [installation guide](https://raea.io/docs/start/install.md) covers platform support.
Set `RAEA="$HOME/.raea/versions/0.1.0/bin/raea"` on this Mac. Read the installed version's
composition guidance with `"$RAEA" skill` or use the
[first-video guide](https://raea.io/docs/start/first-video.md).

RAEA is a separately installed tool for internal production under its
[FSL-1.1-ALv2 license](https://github.com/shinyobjectz/raea/blob/v0.1.0/LICENSE).
Keep the binary, SDK and runtime outside the repository and AutoCut packages. The app's
rendered-media import is engine-independent. Use trusted compositions and local assets.

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
the helper sets the output stage and rejects an unexpected rendered size. A 4K scene
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
