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
  motion-sensor stream (accelerometer and gyroscope) and each picture's exposure time on the
  sensor's clock. Details: `crates/isobmff/src/insta360.rs`.
- **The horizon**: an Insta360 recording is **levelled automatically** from its motion sensor.
  When it is imported, the gyroscope and accelerometer are fused (a complementary filter run
  forwards and backwards in time, so a turn is followed without lag and a shove is ignored) into
  which way was up, ten times a second from the first picture; the sphere is then turned at every
  frame so that up is up — in the Program monitor and exports (GPU and CPU), the analysis
  panorama, stabilising and following alike. A camera held 40° off upright on an escalator comes
  out with a level horizon. VR Properties ▸ Level Horizon (`media.vrProperties` `level`) turns it
  off and on; the samples are kept in the project (`horizon` in VR Properties, summed up when
  asked: seconds, samples, the largest tilt).
- **Lens model**: equidistant fisheye, 97.5° from the axis to the image circle's edge (measured
  on X5 footage: where the two lenses' pictures agree best), image circles from the calibration;
  `media.align360` refines the angle and the lenses' relative turn per clip.
- **Equirectangular MP4s**: Google's Spherical Video V2 (`st3d`, `sv3d` ▸ `proj` ▸ `equi`) and
  V1 (the `uuid` RDF/XML box) are read; partial panoramas (captured view) and over/under or
  side-by-side stereo too (the left eye is shown).

## 3. Editing

| Command | Does |
|---|---|
| `media.vrProperties` | Modify ▸ VR Properties: `auto` (what the file says), `equirectangular` (+ `layout`, `capturedView`), `dualFisheye` (+ `halfFov`), `none`; `level` levels the horizon by the camera's motion sensor (on by itself for Insta360 recordings) or stops; a layout, view or lens angle alone changes the item's own; with nothing to change, it reports them |
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
once, framing the sequence (in a 2:1 sequence, the whole sphere), Motion at its defaults;
`timeline.place` lists such clips in `reframed360`. A new camera (and the `natural` preset)
sees the frame's **natural view**, the same coverage whatever its shape: 100° across (67° down)
in 16:9, 68° across (100° down) in 9:16, 84° in a square — so a person near the camera fits a
vertical frame and a landscape one alike. Analyses read 360° media as
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
   their boxes — a 360° clip is indexed as its stitched panorama, levelled by its camera's
   motion sensor, so a box is a direction and every shot's horizon is level), the
   speech, and for every 360° moment where the action is (`media.salience360`). A shot longer
   than 12 s offers several moments.
   **Which moments.** Each moment is measured on its own frames (`media.measureMoments`:
   sharpness, exposure, the camera's shake and the faces in it — a long take is not one score)
   and ranked: sharpness, exposure, steadiness and the aesthetic score first, then faces,
   expressions (the describer's words: smiling, laughing, waving — there is no expression
   model), speech worth hearing (English said in it, an exclamation, a word like *wow* or
   *look*) and energy. The unusable are left out (out of focus, too dark, blown out, a shaky
   normal clip), then near-duplicates of a better moment (SigLIP 2 image embeddings at least
   0.93 alike; different rooms of one trip measure 0.6–0.86), as long as enough moments stay to
   fill the edit; the rest keep filming order. The report's `moments` gives every moment's
   scores, whether it was kept and why ("kept: rank 0.81 — sharp, well exposed, faces in view;
   says "wow look at that"", "dropped: a near-duplicate of c4 (0.96 alike), which ranks
   higher"). The model sees each moment's rank; without a model the rules use it too.
3. **The plan**: the rhythm is set first — a hook long enough for a tiny planet to open, body
   shots alternating about 2.4 s and 1.8 s, an ending, every length a whole number of beats
   adding up to the length asked. The local model (the script model, `qwen3:8b`, else
   `phi4-mini`) fills each slot from the brief and the shots' facts (what each shows, its file,
   energy, rank, faces, smiles and speech, a 360° shot's *looks* — the subjects and action directions it can point at —
   and the cameras that clip may use): a clip, its camera, where a 360° camera looks (and turns
   to), any English text. Every pick is checked and repaired — an unknown clip, the same moment
   twice or a third shot of one file in a row is replaced by the rules' next clip in filming
   order; a camera that does not fit (a 360° move on a normal clip, too long for its slot, a
   planet anywhere but the ends, a sky reveal indoors, a second barrel roll, a whip pan without
   two things to turn between) is replaced by one that does — and `plan.notes` lists each repair.
   A reply that is not JSON is asked for once more. Without a model (or with an unusable plan)
   rules direct: a tiny-planet hook, the story in filming order with normal and 360° shots taking
   turns, each 360° shot's move chosen by what it shows (a whip pan between two subjects, a
   follow of a named one that moves, a reveal or orbit of a wide outdoor view, one barrel roll at
   most, a hyperlapse of a long quiet stretch, else a drift or push in), and a planet outro. The
   report's `director` says who directed.
4. **Commands**: the plan compiles to engine commands — `timeline.place`, `clip.reframe360`,
   `clip.stabilize360`, `clip.vrMove`, `clip.follow360`, `clip.hyperlapse360`, `clip.kenBurns`,
   `clip.autoReframe`, `clip.speedRamp`, `graphics.newText`, `audio.generate`… — each shot's own
   run on its clip once it is placed. `--dry-run` prints the plan and its commands.
   **Framing for the frame's shape.** Every 360° shot is framed for the edit's own aspect: the
   camera turns to the subject's box (a person right by the camera, taller than 75°, by the
   upper part — the face) and its field of view is sized so the box takes about 65 % of the
   frame's height and at most 80 % of its width — a 16:9 cut and a 9:16 reel of the same shot
   both show the whole person — never narrower than half the natural view, its long side at most
   110° (past that a little lens distance takes in the rest rather than stretching the edges).
   A named subject under a camera that does not turn by itself (hold, push in, speed ramp) is
   **tracked** (`clip.follow360`); a normal clip of another shape than the frame is cropped with
   its subject tracked (`clip.autoReframe`, keeping the push in).
5. **The build**: generated music first (its tempo `bpm`, mood from the brief or the model), its
   exact beats the grid every cut lands on; **each cut picked from its two shots** — a dissolve
   into the ending and across a jump in time (filmed 3 h or more apart); a whip where the camera
   is already moving fast (between two 360° shots their cameras whip out and in; else a whip-pan
   transition the way the faster shot pans); a match cut where the subject can sit where the last
   one was (a 360° camera turns to put it there), a pan carries on, or two pictures are framed
   alike; a J cut into someone who starts talking as their shot begins (heard half a beat
   early) and an L cut out of a sentence that runs past its shot (`clip.splitEdit`); otherwise a
   plain cut — every one on its beat, each an ordinary edit to change afterwards, and the plan's
   `cut` says why. How lively the cuts are, whether light effects are used and which sounds are
   heard on them is the edit's **style** (`--style auto|subtle|energetic|cinematic`,
   `--light-effects`, `--sounds`, `--sound-level`): subtle makes a few whips with a quiet swish,
   a chime under the title and no light effects; energetic whips more, punches in across time (a
   light leak where allowed) with whooshes and an impact; cinematic dissolves longer, dips to
   black across days, whips at most once. Unless the video or the taste profile names one, the
   director chooses it from the shots it planned (`auto`): a style the brief names; plain cuts
   when many shots have people talking; energetic when many move fast or their energy is high
   (light effects too when most are not at night); cinematic when they are calm and wide or
   filmed over days; subtle otherwise. The report's `style` says which and why, and a revision
   keeps it.
   **A careful colour pass** (`clip.autoColor`, on unless `--no-color`): each shot is rendered
   small on the GPU and measured — its near-neutral greys (the least coloured pixels, even a
   white wall under warm light), its skin and its median brightness — and given a gentle
   Lumetri Basic Correction 70 % of the way (`--color-strength`): white balance as far as the
   greys are sure (less with little grey, half as far in a colourful picture, whose greys pick
   up the colour around them), capped at ±20 temperature and ±10 tint; exposure only outside a
   wide middle band (a dim room may stay dim), at most ±0.5 stops, never pushing highlights past
   white; left alone where there is no grey to judge the light by (a close-up of a face). The cameras are
   brought together by how vivid and contrasty they look (saturation and contrast half way to
   the edit's middle) — never by copying a hue from a reference shot, which once turned a set of
   clips sepia. A look of our own is optional (`--look punchy|soft|warm|cool|golden|teal`, at
   25 % by default; no LUT or preset from anyone else). Each shot is then rendered again with
   its correction and checked: greys more coloured, skin moved off the skin-tone line,
   highlights clipped or shadows crushed, and the correction is halved, then its white balance
   dropped, then dropped altogether — the report says which. The correction is a Lumetri Color
   of its own at the end of the clip's effects, so it is easy to switch off or change, and
   `<out>.color.jpg` shows each shot before and after. The English title in the platform's
   safe zone; captions of English speech; each sound levelled (the
   music −16 LUFS, the clips' own sound under it, someone talking ducks it) and −14 LUFS.
   `--export` writes the video, `<out>.sheet.jpg`, `<out>.report.json` (the plan, the
   candidates, who directed, the GPU frames rendered, the fallbacks) and QA. Faces stay sharp
   unless `--redact` asks: in a trip video the people are the subject.
   **Taste.** What the video's options leave out comes from the taste profile (`autocut.taste`:
   the user's default under the project's own): the cutting style, light effects, sounds, the
   colour pass and look, privacy, and wording rules on every text ("our son", never "my son").
   See [the AutoCut README](../crates/autocut/README.md#taste-profile-how-this-person-likes-their-videos).
6. **Revising**: notes on a finished edit — "1:10 is too long; warmer at the end; fewer barrel
   rolls" — become changes to its saved plan rather than a new edit:

   ```text
   filmcraft-cli direct --revise "1:10 is too long; warmer at the end; no barrel rolls" \
       --from ~/Movies/AutoCut/weekend-reel.report.json
   ```

   The report keeps the plan (each shot's file, moment, camera and beats) and the options it was
   made with; a revision starts from them (new options win; the export becomes
   `<name>-revised.mp4` unless one is given), finds each shot again in the footage, reads the
   common notes by rules and takes them literally, and asks the local model to turn the rest
   into changes — a shot shorter or longer, removed, another
   camera, fewer of a move, a stretch warmer, cooler, brighter or more vivid over the colour pass,
   a text or the title, the cutting style or the look, another moment in a shot's place — each
   checked before it is made (a camera its clip cannot use, or footage too short, is refused and
   said). The rules read times like `1:10`, "too long", "remove", "fewer / no <move>",
   "warmer / cooler / brighter / darker at the end / start / 1:10", "punchier", "calmer",
   `title: …` (given the whole notes, the model added changes nobody asked for: "warmer" came
   back warmer, brighter and more vivid). The cuts are picked again for the changed shots and the
   edit is rebuilt; the report's `changes` lists what changed and what was not understood.

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
- Levelling turns the picture about a level axis only: the camera's heading (where its front
  lens looked) is kept, and stabilisation (`clip.stabilize360`) is still measured from the
  pictures, not the gyroscope. Recordings without a motion-sensor record (other cameras, stitched
  equirectangular exports) keep the horizon they have; `roll` and `pitch` level them by hand.
  The sensor's axes and scale were matched on X5 footage; other Insta360 models are assumed to
  share them.
- 360° exports are monoscopic equirectangular; stereoscopic 360° is read (the left eye shown)
  but not written.
- Cube maps and meshes (`cbmp`, `mshp`) are recognised but shown flat; `.insp` photos are not
  imported yet; two-file cameras (ONE X: `_00_` and `_10_` files) are found but only their first
  lens is shown.
