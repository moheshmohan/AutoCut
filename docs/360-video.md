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
camera's `.lrv` attached as the **proxy** (both lenses side by side, 1664 × 832), and the two lenses
**aligned** along their seam (`media.align360`, a few seconds a clip). A plain `file.import` of an
`.insv` (the director's `direct <files>`, File ▸ Import) attaches the `.lrv` beside it too
(`proxies` in its answer; part of the import's undo step), when its length fits: a proxy may end up
to three frames before its original, as the camera's copies do.

Every 360° **analysis reads the proxy**: the scene index, `media.salience360`,
`media.measureMoments`, `media.frame` (unless the size asked for is sharper than the proxy's
panorama), stabilisation and following decode one H.264 picture of 1664 × 832 instead of two
2.9K HEVC lenses: a tenth of the pixels. Five Istanbul recordings (17 minutes, 30,533 analysed
frames) were indexed, looked round at 15 moments and measured at 25 in 2.2 minutes instead of 13.5
(2,273 s of CPU before, 332 s after), and the direction found first at each moment was within 15°
of one found on the full-size lenses every time. Export, the monitors (unless proxies are on) and lens alignment read the
recording itself. A camera's own dual-fisheye recording is one take, with no cuts in it: the scene
index skips the shot-boundary network (TransNetV2) for it, and the signature detector still marks
a sudden change; an equirectangular 360° file may be an edit and is looked at like any video.

## 2. The formats

- **`.insv`** (Insta360 X3/X4/X5): an MP4 with **one HEVC track a lens** (2880 × 2880 each at
  5.7K, 1920 × 1920 in 4K mode, 3840 × 3840 at 8K), an AAC track, then the camera's trailer.
  FilmCraft decodes both tracks and shows them side by side (the selfie lens left, as the
  camera's own `.lrv` and calibration do): one dual-fisheye picture twice as wide as high.
- **`.lrv`**: the camera's low-resolution copy, H.264, both lenses side by side in one track.
- **`.insp`** (photos): a JPEG of both lenses side by side, the camera's trailer after it. Opened
  as a 360° still (`filmcraft_codecs::insp`): the image circles from the trailer's calibration,
  as for its videos; scaled to the still limit of 8192 × 4096. Imported by File ▸ Import and with
  the recordings (`insta360.import`, lenses aligned); the camera's `.jpg` and `.dng` exports are
  read as flat pictures. The director looks round a 360° photo as it does a 360° video — a tiny
  planet, an orbit, a whip, a reveal, a drift, a push in — never a follow, a hyperlapse or a
  speed ramp (nothing in it moves), and does not steady it; its shot can be any length. A
  showcase turns one to its subject and opens it as a tiny planet or pans slowly across it, by
  turns.
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
- **Steady heading**: the same recording's **heading is steadied** from its gyroscope too. The
  camera's turning about the vertical is added up and smoothed (a Gaussian of half a second, run
  both ways); what is left — the sway of a walk, a step or two a second, 9–18° from side to side
  on X5 footage walked with — is turned back, at most 20°, so a quick turn of the camera is let
  through and every slow one (the way the person walks) kept. It takes out about a third of the
  sway as a view shows it (the time between the sensor and the pictures differs by up to five
  frames from recording to recording, so the samples stay a tenth of a second apart and the quick
  jolts are not chased); `clip.stabilize360`, measuring from the pictures themselves, takes out
  the rest and the bounce of the steps (a walked shot's wobble 6.6 → 2.1 px and 4.7 → 0.2 px in a
  135 px wide view), and Direct an Edit runs it on every 360° shot. VR Properties ▸ Steady
  (`media.vrProperties` `steady`) turns the heading's steadying off and on.
- **The sensor's timing**: the horizon carries a `lag` — how far the sensor's clock runs behind
  the pictures — and is read at the picture's time plus it. `media.syncGyro360` measures it for a
  recording: the levelled pictures' own sway (their turning about the vertical a tenth of a second
  at a time, on the proxy, over the 20 s where the steadying moves most, less its half-second
  smoothing) against the sensor's steadying turn at each lag within ±0.3 s, kept only when the two
  clearly turn against each other (`match` ≥ 0.5). It is not there yet: on five Istanbul
  recordings none was clear. On 078 and 086 the best lags (0 and 139 ms) fall near the ones found
  by eye earlier (0–50 and about 100 ms) but agree only 0.28 and 0.39; on 075 (about 130 ms by
  eye) nothing agrees. Why they agree so little is not known yet (the pictures' turning measured
  on the proxy's half-size frames is one suspect, the steadying turn's cap during the
  operator's own turns another), so no lag is set and the samples stay a tenth of a second
  apart.
- **Lens model**: equidistant fisheye, 97.5° from the axis to the image circle's edge (measured
  on X5 footage: where the two lenses' pictures agree best), image circles from the calibration;
  `media.align360` refines the angle and the lenses' relative turn per clip.
- **Equirectangular MP4s**: Google's Spherical Video V2 (`st3d`, `sv3d` ▸ `proj` ▸ `equi`) and
  V1 (the `uuid` RDF/XML box) are read; partial panoramas (captured view) and over/under or
  side-by-side stereo too (the left eye is shown).

## 3. Editing

| Command | Does |
|---|---|
| `media.vrProperties` | Modify ▸ VR Properties: `auto` (what the file says), `equirectangular` (+ `layout`, `capturedView`), `dualFisheye` (+ `halfFov`), `none`; `level` levels the horizon by the camera's motion sensor and `steady` steadies its heading (both on by themselves for Insta360 recordings) or stop; `seamFlow` stitches near things on a dual fisheye's seam by the lenses' shifts (on by itself) or only cross-fades them; `nadir` (`off`, `mirror`, `smooth`) and `nadirSize` patch the hand and stick at the bottom of the sphere; a layout, view or lens angle alone changes the item's own; with nothing to change, it reports them |
| `media.align360` | aligns the two lenses of dual-fisheye media along their seam |
| `media.seam360` | the seam of a dual-fisheye item at a moment: how much the lenses disagree across it cross-faded and stitched, the largest shift, how long measuring took; a picture of the band unrolled round the seam (lens 0, lens 1, the cross-fade, the stitch) with `path` |
| `clip.reframe360` | the virtual camera: `yaw` (turn right), `pitch` (look up), `roll`, `fov` (degrees across), `distance` (0 flat … 1 tiny planet), `output` `flat`/`sphere`, `width`/`height` (default: the sequence frame); `at` keyframes them at a clip time; `keys` writes a whole camera move; presets `natural`, `wide`, `tinyPlanet`, `rabbitHole`, `sphere`; `off` |
| `clip.stabilize360` | holds the view steady: the camera's turn measured from the pictures (points tracked on cube faces, a robust rotation fit), then `smooth`/`light`/`strong` (keep the operator's turns, drop the shake) or `lock` (look one way the whole clip) |
| `clip.follow360` | turns the virtual camera to follow a subject (`subject`, `box`, `face`); one lost for a few frames is looked for again by its look (`refind`, see below) |
| `clip.aim360` | the camera aimed along `samples` (`[seconds, yaw, pitch]`, as a drag on the Program monitor records them: View ▸ Aim 360° Camera), smoothed over `smooth` seconds (0.5) and kept to the keyframes needed; `merge` joins a drag's calls into one undo step |
| `clip.vrMove` | a camera move as Bézier keyframes, editable in Effect Controls: `planetIn` (a tiny planet spinning open onto the view), `planetOut`, `orbit` (`degrees`, default a full turn), `barrelRoll`, `whipPan` (the short way to `yaw`/`pitch`, from `fromYaw` for a whip in, with a Directional Blur peak), `reveal` (from the sky, or `from: "ground"`), `pushIn`, `crashZoom`, `drift`, `drone` (a high stick read as a drone: rising and pulling back, tipping down to −50° and widening), `thirdPerson` (looking down at the person holding the camera from above and behind, `yaw` their way, for the rest of the clip), `spin` (a fast full turn, blurred), `flip` (a somersault forwards through the ground and the sky, blurred up and down), `wobble` (a playful shake dying away, `amount` its size) — GoPro's CameraFX and Insta360's invisible-stick shots as moves; each has its default place and length (`at`, `seconds` move it); a move replaces its parameters' keyframes inside its window only, so moves, a follow and stabilisation add up |
| `clip.hyperlapse360` | a long stretch at `speed` % (frame blending) held steady by `clip.stabilize360` (`strong`) |
| `media.salience360` | where things happen round the sphere around `seconds`: a few panorama frames 0.2 s apart scored by direction — by motion when one direction moves clearly more than the rest, else by detail — the strongest directions (yaw, pitch) at least 45° apart |
| `media.denoise` | Clip ▸ Modify ▸ Reduce Noise, for any video: each frame merged with its neighbours (`frames` either side, 2), each moved to line up with it first (see Rendering); `strength` 0..1, `off`; `auto` turns it on only for the dark, noisy ones (grain ≥ 1.5 levels and light under 0.3 at the middle frame), `measure` reports both |

**Following through blocking.** A subject `clip.follow360` follows can be hidden — someone walks
in front of it, it goes behind a pillar and comes out elsewhere — and the tracker, which only holds
its last box while it is gone, would leave the camera there. Once it has been lost (held or gone)
for three analysed frames, and every three frames while it stays lost, the subject is looked for
by how it looked in its first confident frames: windows of its size over the tracking camera's
110° picture — and every other time over the views 99° either side — compared by a colour
signature (hue, saturation, lightness binned, the middle weighted), the best grid window moved
onto it in three finer steps, clearly better than the best one elsewhere; with the scene models
downloaded, SigLIP 2's embedding of that one window must agree that it is the same subject
(`refind: "auto"`; `"colour"` without the check, `"siglip"` every window by SigLIP, slower;
`"off"`). The tracker starts again there, the camera turned onto it when it was in a side view;
the report gives `foundAgain` and when, with each match. On a test walk hidden by a passer-by for
12 frames and coming out 70° on — outside the tracking view — the camera ends on it (within 10°;
without looking again it stays more than 30° away). On 20 s of a crowded Istiklal street at night
(086), following a woman in an orange shirt who walks on behind two men, she was seen in 75% of
the analysed frames before and in all of them with `auto` (found again three times, by the colour
of her shirt between them: colour 0.93–0.98, SigLIP agreeing at 0.88–0.96), at 0.4 s more than colour alone; SigLIP on every
window found her only once and took three times as long.

**Aiming by hand.** View ▸ Aim 360° Camera (`view.aim360`) makes a drag on the Program monitor
turn the view of the topmost 360° clip under the playhead: the picture follows the pointer, as in
a 360° player (a drag of the picture's width turns by its field of view). Paused, the drag sets
the view at the playhead (a keyframe where the camera is animated); while the sequence plays,
every turn is recorded — `clip.aim360` gets the samples, unwraps the yaw (no spin the long way
past ±180°), smooths them over half a second when the button is let go (`smooth`, 0–2 s) and keeps
only the keyframes needed to stay within 0.25° of the path, replacing the yaw and pitch keyframes
inside the drag's span. One drag is one undo step; the keyframes stay editable in Effect
Controls. Automation id `program.aim360`.

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
  analysis) see the stitched whole-sphere panorama (of the proxy when there is one), so a box's
  centre is a direction (x → yaw −180…180°, y → pitch 90…−90°);
- tracking, masks, redaction, the privacy check and cut-outs (`track.*`, masks, `redact.*`,
  `clip.cutOut`) work in the clip's own pixels, the camera's picture at each frame.

### People cut out (Shot Lab's ideas)

`clip.cutOut` outlines the person in a clip — flat or 360° — every 0.2 s: the person detector
(D-FINE, with RTMPose's keypoints) finds them, **MobileSAM** (Apache-2.0; `cutout.downloadModel`,
40.7 MB) cuts them out from points on their body and beside them, on the GPU (Metal; about 0.5 s
a frame, 0.7 s with finding the person), and the largest piece is traced into a 96-point outline
kept as the clip's Opacity mask **Person**, switched off. The effects are then ordinary clips,
masks and titles above it, so they render on the GPU and stay editable:

| command | what it makes |
|---|---|
| `clip.cloneTrail` | frozen copies of the person left where they were along their path (Frame Hold with Hold Filters, the mask held too), each until the clip ends |
| `clip.shadowClone` | see-through copies following the person (the same media, later), the person in front |
| `clip.timeFreeze` | the person frozen at `at` while the world goes on: a clean plate (each pixel's median over 24 frames where the person is elsewhere) over the moving person, the frozen copy above; refused when the background behind them is not seen clear and still (they stay in place, or the camera moves with them) |
| `clip.textBehind` | a title behind the person (centred, fitted to the frame) |
| `clip.personInFront` | the person above everything over the clip, so titles there go behind them |
| `clip.personGlow` | a glow of a colour round the person (a copy tinted to it, its mask grown and feathered, Screen) |

A copy in step with the clip (the glow, the person in front) lines up whatever the camera does;
frozen and following copies need a still camera — a 360° clip's virtual camera may not move (its
stabilisation is kept; `clip.stabilize360` `mode: "lock"` holds a walk still). Measured on a 6 s
360° selfie shot from Istanbul: the person found and outlined in 30 of 30 frames in 21 s; the GPU
export with two shadow clones ran at 21 frames a second. The director puts the opening's title
behind its person when a person is its subject (a named look at least 30° tall, or a flat shot's
subject a third of the picture), after the colour pass, in people's videos (not a travel video's
places); `--no-cutouts` switches it off.

### Hiding the hand and stick

The bottom of a 360° camera's sphere shows the hand and stick holding it. VR Properties `nadir`
patches it within `nadirSize` degrees (22) of straight down — the sphere is levelled, so the patch
stays on the ground as the view turns: `mirror` shows the ground round it mirrored in across its
edge (texture that meets it without a seam), `smooth` draws the colour at its edge in to the middle,
`off` shows it as it is. Every view of the clip draws it, on the GPU and the CPU alike (tested
against each other). A tiny planet looks straight down and a drone far down, so `clip.vrMove`
`planetIn` / `planetOut` / `drone` and `clip.reframe360` `preset: "tinyPlanet"` patch it (mirrored)
wherever it was never set — the director's planets get it this way. `clip.nadirLogo` lays one of the user's pictures round the
bottom instead (`size` 24°): a whole-sphere picture, transparent elsewhere, above the clip and seen
through its camera. The patch does not change the analyses' key (`sphere_key`): what is there to
see is the same.

### Sky Swap

`clip.findSky` outlines a clip's sky every 0.2 s — flat or 360° (a levelled sphere has its sky at
the top too): a first guess of cells that are bright, plain, grey or blue and join the top edge,
refined by MobileSAM given points inside it and in the detail below when the segmenter is
downloaded (and when the two agree); every sizeable piece is joined into one 256-point outline, the
clip's Opacity mask **Sky**, switched off. `clip.skySwap` puts the new sky above the clip through
that mask — one of the user's own photos (an item of the project, cut to the clip's shape), or a
sky made here: `clear`, `sunset`, `dusk`, `night` (stars) or `clouds` (a cloud layer drawn in
perspective), a whole 2:1 sphere for a 360° clip seen through the clip's own virtual camera; never
stock pictures — and relights the clip a little towards it (Lumetri temperature and exposure from
the old and new skies' colour and light, `relight` 30 %). On a 6 s 360° shot of a park in Istanbul
the sky was outlined in 30 of 30 frames in 17 s; a sunset warmed the scene by 14 and darkened it a
little. Sky seen through the gaps inside a tree's crown stays as it was.

## 4. Pipelines

| Pipeline | With a 360° clip |
|---|---|
| Showcase / Quick Short | the shot's camera turned to the plan's `focus` or the analysis' subject (pitch kept within ±45°), steadied with `clip.stabilize360` when the show is stabilized, following the subject with `clip.follow360` (as a flat shot follows with Auto Reframe), and a push-in made by narrowing the camera's field of view (sharp, where scaling would soften); no blurred backdrop: the camera's picture is the frame's shape |
| Templates | a slot's 360° medium gets a camera of the slot's box (the frame without one), looking at the fill's `focus` (a point of the panorama); the slot's own Motion then applies as to any picture filling it |
| Auto Reframe (`sequence.autoReframe`) | the copy keeps each camera's direction and re-shapes it to the new frame, keeping what it sees along its short side (a 16:9 view's height becomes a 9:16 view's width; `cameras360` counts them); its Motion is not moved — `clip.follow360` follows a subject. `clip.autoReframe` and `clip.cameraTrack` leave 360° clips to `clip.follow360` |

## 5. The director

**Two angles of one moment.** `media.sameMoments` pairs recordings filmed at the same time — a
phone and a 360° camera at one scene: their capture dates within `slack` (600 s) of overlapping, or
with `dates: false` (wrong clocks) every pair of a dozen items or fewer — and lines each pair up by
its sound (the GCC-PHAT sync of Clip ▸ Synchronize, the search kept within the dates' difference),
keeping only clear matches. The director asks it for its media and, where a held, pushing-in or
drifting shot (not the first or last) was filmed by the other camera too, cuts it half-way to the
other camera's picture of the same instants — twice in an edit at most, the best-ranked moments
first, `angles: false` to stop it. Checked on a street walk from Istanbul and a clip cut from it
5.3 s in: lined up at 5.300 s (confidence 397); the 21 recordings of the weekend, one camera's, paired
none.

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
   **Places or people.** What the edit is about (`--focus auto|places|people`): a brief about a
   trip, travel, a vlog, a holiday, a tour or a city is about **places**, one about a birthday, a
   wedding, a party or the family about **people**; with neither, most moments outdoors and few
   people talking is places. A places edit ranks moments by how they look, how wide and whether
   outdoors (faces, smiles and speech count for little) and plans from outdoor moments, keeping
   indoor ones (a shop, a lift) only as many as the outdoors cannot fill; points each 360° shot
   at what is not a person first, the farthest from the camera's holder first (a crowd or people
   walking are the street; the holder is the person below the lens or right by it; with no
   place clear of them, the view the other way round); covers every part of the trip (moments
   hours apart — a morning walk, the night out, the next day — each with its share of the
   shots, and one recording in two at most, a shot over going to a part that is short or to
   another recording; the shots between the opener and the ending in filming order and the
   ending from the last part — Insta360 recordings are dated by their movie header like any
   MP4); never points a shot at the camera's holder (in 360° they are at its edges often
   enough), and anyone else in at most a third of the shots, where a whip ends counting as well
   as where it starts: past that, a 360° shot turns to its place (a follow drifts, a whip with
   no second place pushes in), and orbits drift, as they sweep past the holder. A voice is heard
   over the music, and captioned in whole sentences, only where the shot shows who speaks; an
   edit has two reveals at most (each starts on the sky). The model is told the same, the
   person looks marked; the plan's notes say what was repaired and the report's `focus` which
   and why.
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
   run on its clip once it is placed. Every 360° shot but a planet or a hyperlapse is steadied
   from its pictures (`clip.stabilize360` `smooth`): a walk sways and bounces however the analysis
   judged it. `--dry-run` prints the plan and its commands.
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
   `<name>-revised.mp4` unless one is given), finds each shot again in the footage — its moment
   by file and time, put back if this run's ranking would leave it out (the footage analysed
   again can make it a near-duplicate), and a 360° shot's look by the direction it had (the
   nearest look within 20°, else that direction itself; the report lists every moment the plan
   uses with its looks) — keeps the travel rules' choices the plan was made with, reads the
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
Y′CbCr planes are uploaded as they are and every output pixel samples the ray it sees. The CPU
path (`filmcraft_render::vr`) is the reference and the fallback; the two are tested against each
other, the stitch below included.

**Stitching the seam.** Where both lenses see, they see from two points a few centimetres apart:
far things agree, but a wall, an arm or a face within a metre or so is seen a few degrees apart
across the seam, and a cross-fade shows it twice. For every frame drawn, the band of ±6° round the
seam is unrolled from both lenses into strips (512 columns round it, 32 rows across), and for each
column the shift of one lens's picture against the other's — up to 5.6° across the seam, where
parallax moves things, and 0.7° along it — whose windows of the two strips correlate best is
found, measured on that frame and the three either side of it together (decoders keep those, and
the shifts then hold steady from one frame to the next whatever order frames are drawn in). A
shift is trusted by how well and how unambiguously it matched (another peak of the search means a
repeating pattern) and how much detail there was; shifts are smoothed round the seam by that
trust, and where there is nothing to go by they fade to none. Drawing then moves each lens's
sample part of the way towards the other's across a ±4.5° cross-fade, so a near thing is seen
once. Measuring takes about 6.5 ms a frame on the CPU; the GPU gets the shifts with the frame and
moves the samples in the shader (`filmcraft_render::vr_seam`).

Measured on 24 frames of eight Istanbul recordings (`media.seam360`): how much the lenses disagree
across the band fell from 0.065 to 0.056 on average, in every frame — 27% in a hotel corridor and
14% in an elevator (walls and arms close by), 4–16% outdoors — and the shifts change by 0.08° a
frame on average (0.4% of the seam by more than 1°), against 0.25° measured one frame at a time.
Lines very close to the camera bend a little where they cross the seam rather than doubling.
`media.vrProperties` `seamFlow: false` cross-fades only, as before.

**Cleaner night shots.** Reduce Noise (`media.denoise`, any video, not only 360°) merges each
frame with two either side (`frames` 1–3), with no model: each neighbour's motion against the
frame is measured on the CPU (luma at an eighth of its size, the best shift within ±24 pixels of
every 64-pixel block by sum of absolute differences, refined to a fraction and the median of the
blocks round it taken: 5 ms at 5.7K), then every pixel of every plane is averaged with the
neighbours' pixels where that motion takes it, each weighted by how well it agrees
(`exp(−(d/σ)²)`, σ the frame's own grain — Immerkær's estimate, the black of a fisheye's surround
and blown highlights left out — times 1.5–4.5 by `strength`, at least 1–4 levels), so grain
averages away while what moves, or what the motion missed, is not smeared. The merge runs on the
GPU (Metal: 20 ms for a 5.7K luma plane, against 104 ms on the CPU; `filmcraft_gpu::denoise`,
tested against the CPU's `filmcraft_render::denoise`), the source's frames as decoded
(`plain_frame`) still feed the seam's stitch, and a 5.7K frame of an export costs 66 ms more
(1.06 s → 1.85 s for 12 frames). On a grainy test picture moving 4 pixels a frame it takes a
third of the grain away or more.

On the X5's own night recordings it has little to do, and says so: measured on eight Istanbul
night recordings (083–090) the camera has already cleaned its frames up (the painted look of faces
and brick in the dark is its own noise reduction), leaving 0.2–0.7 levels of grain — `auto`
(grain ≥ 1.5 levels, light under 0.3) leaves them all alone — and at full strength Reduce Noise
lowered the grain in the dark by 3–7% and the shimmer from frame to frame by 1–4%, a change that
does not show. It is for footage that is grainy when it comes in: a phone's or a small camera's
in the dark. The director runs `auto` on the recordings it uses (`denoise: false`,
`--no-denoise` to skip it) and reports those it turned on in `denoised`.

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

- The seam's stitch moves each lens's picture across the seam by one shift per column: something
  near in front of something far at the same point of the seam takes the near one's shift, and
  things closer than about 30 cm (beyond the 5.6° searched) still show twice.
- Levelling turns the picture about a level axis only: the camera's heading (where its front
  lens looked) is kept; the gyroscope steadies the heading's sway (Steady), and
  `clip.stabilize360` measures the rest from the pictures. The time between the sensor and the
  pictures is not known per recording (see Steady heading). Recordings without a motion-sensor record (other cameras, stitched
  equirectangular exports) keep the horizon they have; `roll` and `pitch` level them by hand.
  The sensor's axes and scale were matched on X5 footage; other Insta360 models are assumed to
  share them.
- 360° exports are monoscopic equirectangular; stereoscopic 360° is read (the left eye shown)
  but not written.
- Cube maps and meshes (`cbmp`, `mshp`) are recognised but shown flat; two-file cameras (ONE X: `_00_` and `_10_` files) are found but only their first
  lens is shown.
