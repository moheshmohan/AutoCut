# 360° video

> **Last reviewed:** 2026-10-26 · **Last updated:** 2026-10-26 · **Change:** major (360° support added: Insta360 footage, VR Properties, the 360 Reframe virtual camera, lens alignment, stabilisation, subject following, camera moves, 360° export, the director) · **Target:** Adobe Premiere Pro 2026 (26.5.2)

AutoCut edits 360° footage flat: a **virtual camera** looks into the sphere and makes a normal
16:9 (or 9:16, 1:1…) shot, with keyframes for where it looks, how wide, and how far back it
stands (up to a "tiny planet"). It reads Insta360 recordings straight from the camera or from
the Insta360 cloud (through Insta360 Studio), and stitched equirectangular MP4s with spherical
metadata. It can also export a 360° video with the metadata YouTube reads.

## 1. Getting the footage

| Where | How |
|---|---|
| Insta360 cloud | `filmcraft-cli insta360 download VID_20260913_114003_00_093 …` (or `insta360.download` / the `insta360_download` MCP tool): the local screen agent drives Insta360 Studio's **Cloud Media** (List View → right-click → Download → the dialog's file name checked → OK), waits for the file and imports it. Needs Studio signed in, `ollama pull qwen2.5vl:7b`, Screen Recording and Accessibility for the terminal; holds the `chrome` lock (the mouse) while it drives. |
| Downloaded by hand | In Studio: Cloud Media → right-click → Download. Files land in `~/Downloads/Insta360 Download/<name>-Original/`. Then `filmcraft-cli insta360 import --since 2026-09-13` (or names: `insta360 import VID_093`, or `--all`). |
| The camera's card | Mounted cards (`/Volumes/*/DCIM/Camera01`) are looked in too. |
| Studio exports | A 360° export (equirectangular MP4 with spherical metadata) imports as 360° media with `file.import`. A reframed (flat) Studio export is ordinary video. |

`insta360 list` shows what is on disk: name, when it was shot (camera time), size, its `.lrv`,
and whether it has finished downloading (the camera's trailer is the last thing written).

`insta360 import` imports each recording as 360° media: VR Properties from the file, the
camera's `.lrv` attached as the **proxy** (both lenses side by side, 1664 × 832: scene analysis,
stabilisation and following decode that instead of the two 2.9K HEVC tracks), and the two lenses
**aligned** along their seam (`media.align360`, a few seconds a clip).

## 2. The formats

- **`.insv`** (Insta360 X3/X4/X5): an MP4 with **one HEVC track a lens** (2880 × 2880 each at
  5.7K, 1920 × 1920 in 4K mode, 3840 × 3840 at 8K), an AAC track, then the camera's trailer.
  FilmCraft decodes both tracks and shows them side by side (the selfie lens left, as the
  camera's own `.lrv` and calibration do): one dual-fisheye picture twice as wide as high.
- **`.lrv`**: the camera's low-resolution copy, H.264, both lenses side by side in one track.
- **The trailer** (no public specification; read from the files): the camera model, serial,
  firmware, the lens calibration (each lens's image circle), the recording's start, a 1 kHz
  motion-sensor stream and exposure times. Details: `crates/isobmff/src/insta360.rs`.
- **Lens model**: equidistant fisheye, 97.5° from the axis to the image circle's edge (measured
  on X5 footage: where the two lenses' pictures agree best), image circles from the calibration;
  `media.align360` refines the angle and the lenses' relative turn per clip.
- **Equirectangular MP4s**: Google's Spherical Video V2 (`st3d`, `sv3d` ▸ `proj` ▸ `equi`) and
  V1 (the `uuid` RDF/XML box) are read; partial panoramas (captured view) and over/under or
  side-by-side stereo too (the left eye is shown).

## 3. Editing

| Command | Does |
|---|---|
| `media.vrProperties` | Modify ▸ VR Properties: `auto` (what the file says), `equirectangular` (+ `layout`, `capturedView`), `dualFisheye` (+ `halfFov`), `none`; a layout, view or lens angle alone changes the item's own; with nothing to change, it reports them |
| `media.align360` | aligns the two lenses of dual-fisheye media along their seam |
| `clip.reframe360` | the virtual camera: `yaw` (turn right), `pitch` (look up), `roll`, `fov` (degrees across), `distance` (0 flat … 1 tiny planet), `output` `flat`/`sphere`, `width`/`height` (default: the sequence frame); `at` keyframes them at a clip time; `keys` writes a whole camera move; presets `natural`, `wide`, `tinyPlanet`, `rabbitHole`, `sphere`; `off` |
| `clip.stabilize360` | holds the view steady: the camera's turn measured from the pictures (points tracked on cube faces, a robust rotation fit), then `smooth`/`light`/`strong` (keep the operator's turns, drop the shake) or `lock` (look one way the whole clip) |
| `clip.follow360` | turns the virtual camera to follow a subject (`subject`, `box`, `face`) |
| `clip.vrMove` | a camera move as Bézier keyframes, editable in Effect Controls: `planetIn` (a tiny planet spinning open onto the view), `planetOut`, `orbit` (`degrees`, default a full turn), `barrelRoll`, `whipPan` (the short way to `yaw`/`pitch`, from `fromYaw` for a whip in, with a Directional Blur peak), `reveal` (from the sky, or `from: "ground"`), `pushIn`, `crashZoom`, `drift`; each has its default place and length (`at`, `seconds` move it); a move replaces its parameters' keyframes inside its window only, so moves, a follow and stabilisation add up |
| `clip.hyperlapse360` | a long stretch at `speed` % (frame blending) held steady by `clip.stabilize360` (`strong`) |
| `media.salience360` | where things happen round the sphere around `seconds`: a few panorama frames 0.2 s apart scored by direction — by motion when one direction moves clearly more than the rest, else by detail — the strongest directions (yaw, pitch) at least 45° apart |

The **360 Reframe** effect (Effects ▸ Immersive Video) is the clip's picture: a reframed clip is
a flat clip of the camera's frame for everything after it — Motion, masks, every effect, Auto
Reframe, tracking, export. Its parameters keyframe in Effect Controls like any others.

**Placing** 360° media (`timeline.place`, insert, overwrite, drag) gives the clip its camera at
once: a natural 90° view framing the sequence (in a 2:1 sequence, the whole sphere), Motion at
its defaults; `timeline.place` lists such clips in `reframed360`. Analyses read 360° media as
it is seen:

- scene detection, descriptions and subject boxes (`scenes.*`, `media.frame`, the showcase's
  analysis) see the stitched whole-sphere panorama, so a box's centre is a direction
  (x → yaw −180…180°, y → pitch 90…−90°);
- tracking, masks, redaction and the privacy check (`track.*`, masks, `redact.*`) work in the
  clip's own pixels, the camera's picture at each frame.

## 4. Pipelines

| Pipeline | With a 360° clip |
|---|---|
| Showcase / Quick Short | the shot's camera turned to the plan's `focus` or the analysis' subject (pitch kept within ±45°), steadied with `clip.stabilize360` when the show is stabilized, following the subject with `clip.follow360` (as a flat shot follows with Auto Reframe), and a push-in made by narrowing the camera's field of view (sharp, where scaling would soften); no blurred backdrop: the camera's picture is the frame's shape |
| Templates | a slot's 360° medium gets a camera of the slot's box (the frame without one), looking at the fill's `focus` (a point of the panorama); the slot's own Motion then applies as to any picture filling it |
| Auto Reframe (`sequence.autoReframe`) | the copy keeps each camera's direction and re-shapes it to the new frame, keeping what it sees along its short side (a 16:9 view's height becomes a 9:16 view's width; `cameras360` counts them); its Motion is not moved — `clip.follow360` follows a subject. `clip.autoReframe` and `clip.cameraTrack` leave 360° clips to `clip.follow360` |

## 5. The director

`autocut.direct` (`filmcraft-cli direct`, the MCP tool `autocut_direct`, Sequence ▸ Direct an
Edit…) turns normal and 360° clips into one catchy edit on the beat, with the local model as the
director:

```text
filmcraft-cli direct ~/Downloads/Insta360\ Download/*/VID_*.insv phone-clips/ \
    --brief "a 30 s reel of our weekend" --export ~/Movies/AutoCut/weekend-reel.mp4
filmcraft-cli direct --insta360 VID_093,VID_096 --brief "a 45 s 16:9 recap" --dry-run
```

1. **Media**: files, folders, Insta360 recordings by name (`--insta360`: imported from Studio's
   download folder, else fetched from the cloud by driving Studio), a Google Photos `--album`,
   or the project's.
2. **Analysis**: the scene index (cuts, quality, the camera's move, the describer's subjects with
   their boxes — a 360° clip is indexed as its stitched panorama, so a box is a direction), the
   speech, and for every 360° moment where the action is (`media.salience360`). A shot longer
   than 12 s offers several moments.
3. **The plan**: the brief, the shots' facts (what each shows, energy, quality, a 360° shot's
   *looks*: the subjects and action directions it can point at) and the camera vocabulary go to
   the local model (the script model, `qwen3:8b`, else `phi4-mini`), which answers with a shot
   list: each shot a clip, its length in beats, its camera, where a 360° camera looks (and turns
   to), how it comes in, any English text. Every answer is checked and repaired — unknown clips
   out, a 360° move on a normal clip replaced, lengths at least what the move needs and inside
   the media, a whip pan between two different things, one barrel roll at most, English text
   only, the total brought to the length asked — and `plan.notes` lists each repair. Without a
   model (or with an unusable plan) rules direct: a tiny-planet hook, the story in filming order
   with normal and 360° shots taking turns, each 360° shot's move chosen by what it shows (a whip
   pan between two subjects, a follow of a named one that moves, a reveal or orbit of a wide
   view, one barrel roll at most, a hyperlapse of a long quiet stretch, else a drift or push in),
   and a planet outro. The report's `director` says who directed.
4. **Commands**: the plan compiles to engine commands — `timeline.place`, `clip.reframe360`,
   `clip.stabilize360`, `clip.vrMove`, `clip.follow360`, `clip.hyperlapse360`, `clip.kenBurns`,
   `clip.autoReframe`, `clip.speedRamp`, `graphics.newText`, `audio.generate`… — each shot's own
   run on its clip once it is placed. `--dry-run` prints the plan and its commands.
5. **The build**: generated music first (its tempo `bpm`, mood from the brief or the model), its
   exact beats the grid every cut lands on; between two 360° shots a whip out of one and into the
   next; a soft swish under each whip and roll and a chime under the title — never a whoosh or a
   riser, no light leaks, flashes or colour tints; a cross dissolve into the ending; the English
   title in the platform's safe zone; captions of English speech; each sound levelled (the
   music −16 LUFS, the clips' own sound under it, someone talking ducks it) and −14 LUFS.
   `--export` writes the video, `<out>.sheet.jpg`, `<out>.report.json` (the plan, the
   candidates, who directed, the GPU frames rendered, the fallbacks) and QA. Faces stay sharp
   unless `--redact` asks: in a trip video the people are the subject.

Seen on the Insta360 samples: a 30 s 9:16 reel of four X5 recordings and two phone clips directed
by `qwen3:8b` and built in about 70 s once the scene index was cached (3–4 minutes the first time).

## 6. Rendering

The Program monitor and exports draw the virtual camera on the GPU (Metal on macOS): the frame's
Y′CbCr planes are uploaded as they are and every output pixel samples the ray it sees, the two
lenses cross-faded over ±4° along their seam. The CPU path (`filmcraft_render::vr`) is the
reference and the fallback; the two are tested against each other.

## 7. Export

A sequence of reframed clips exports as any flat video. A **360° export** (Export ▸ Video is VR,
`vr` in `file.exportMedia` and the export settings) writes Google's Spherical Video metadata
that YouTube and 360° players read: V2 (`st3d` mono, `sv3d` ▸ `equi`) in the video sample entry
and V1 (the `uuid` RDF/XML box) in the track. It is on by itself for a **360° sequence** exported
at 2:1 — a 2:1 sequence whose clips show the whole sphere (cameras with `output: sphere`, which a
2:1 sequence gives placed 360° media, or unreframed whole-sphere equirectangular media) — and
`vr: false` turns it off; `vr: true` on another shape is refused (a 360° picture is 2:1). A
sphere camera can still turn the panorama (`yaw`, `pitch`, `roll`): a re-centred or levelled
360° video.

## 8. Limits

- Stitching is a cross-fade over the overlap, not an optical-flow stitch: things near the camera
  (closer than about a metre) show their parallax along the seam.
- Stabilisation is measured from the pictures (the camera's motion-sensor stream is read but its
  axes and scale are not documented); the horizon is the clip's first frame's unless the view is
  levelled by hand (`roll`, `pitch`). A camera held tilted keeps its tilted horizon in the
  director's shots too.
- 360° exports are monoscopic equirectangular; stereoscopic 360° is read (the left eye shown)
  but not written.
- Cube maps and meshes (`cbmp`, `mshp`) are recognised but shown flat; `.insp` photos are not
  imported yet; two-file cameras (ONE X: `_00_` and `_10_` files) are found but only their first
  lens is shown.
