# 360° video

> **Last reviewed:** 2026-10-26 · **Last updated:** 2026-10-26 · **Change:** major (360° support added: Insta360 footage, VR Properties, the 360 Reframe virtual camera, lens alignment, stabilisation, subject following, 360° export) · **Target:** Adobe Premiere Pro 2026 (26.5.2)

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

## 5. Rendering

The Program monitor and exports draw the virtual camera on the GPU (Metal on macOS): the frame's
Y′CbCr planes are uploaded as they are and every output pixel samples the ray it sees, the two
lenses cross-faded over ±4° along their seam. The CPU path (`filmcraft_render::vr`) is the
reference and the fallback; the two are tested against each other.

## 6. Export

A sequence of reframed clips exports as any flat video. A **360° export** (Export ▸ Video is VR,
`vr` in `file.exportMedia` and the export settings) writes Google's Spherical Video metadata
that YouTube and 360° players read: V2 (`st3d` mono, `sv3d` ▸ `equi`) in the video sample entry
and V1 (the `uuid` RDF/XML box) in the track. It is on by itself for a **360° sequence** exported
at 2:1 — a 2:1 sequence whose clips show the whole sphere (cameras with `output: sphere`, which a
2:1 sequence gives placed 360° media, or unreframed whole-sphere equirectangular media) — and
`vr: false` turns it off; `vr: true` on another shape is refused (a 360° picture is 2:1). A
sphere camera can still turn the panorama (`yaw`, `pitch`, `roll`): a re-centred or levelled
360° video.

## 7. Limits

- Stitching is a cross-fade over the overlap, not an optical-flow stitch: things near the camera
  (closer than about a metre) show their parallax along the seam.
- Stabilisation is measured from the pictures (the camera's motion-sensor stream is read but its
  axes and scale are not documented); the horizon is the clip's first frame's unless the view is
  levelled by hand (`roll`, `pitch`).
- 360° exports are monoscopic equirectangular; stereoscopic 360° is read (the left eye shown)
  but not written.
- Cube maps and meshes (`cbmp`, `mshp`) are recognised but shown flat; `.insp` photos are not
  imported yet; two-file cameras (ONE X: `_00_` and `_10_` files) are found but only their first
  lens is shown.
