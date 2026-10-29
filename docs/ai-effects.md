# AI effects

FilmCraft's local AI effects for ordinary pictures (the 360° ones are in
[360-video.md](360-video.md)). Every model runs on this computer, is downloaded only when the user
agrees (its name, source, size, licence and training data shown first), is pinned to a revision
and checked against its SHA-256, and is never bundled or committed. Only weights under Apache-2.0,
MIT or BSD licences are used, never non-commercial ones. Each runs on the GPU (Metal on macOS)
with the CPU as the fallback.

**Asking first.** In the desktop app, a command needing a model that is not downloaded — from a
menu, a tool or the control channel — opens **Download a Model** instead of running: the model's
name, what needs it, its size, licence, what it learnt from and where it comes from; nothing is
downloaded until the user chooses Download (the download is a job in Progress; the command is
chosen again when it finishes). Agents and the CLI ask the user themselves, then call the
`*.downloadModel` command (each model's `*.status` gives the same facts, `training` among them).

| Feature | Commands | Model | Licence | Download |
|---|---|---|---|---|
| Photos in depth: the 3D photo, depth blur | `clip.photo3d`, `clip.depthBlur` (Clip ▸ Depth), the effects 3D Photo (Perspective) and Depth Blur (Blur & Sharpen) | Depth Anything V2 Small | Apache-2.0 | 99.2 MB, `depth.downloadModel` |
| Upscale and Restore: photos and videos made larger and cleaner | `media.upscale` (Clip ▸ Modify) | Real-ESRGAN general x4v3 | BSD-3-Clause | 4.9 MB, `upscale.downloadModel` |
| Smooth Slow Motion: frames made between a video's frames | `media.interpolate` (Clip ▸ Modify) | RIFE 4.26 | MIT | 22.9 MB, `interpolate.downloadModel` |
| Remove Object: the thing a mask follows taken out of the clip | `clip.removeObject` (Clip ▸ Cut-outs) | MI-GAN | MIT | 29.5 MB, `removal.downloadModel` |
| Object Mask: a thing masked from a click and followed through the clip | `clip.objectMask` (Clip ▸ Cut-outs; Effect Controls ▸ a mask row's sparkle), `masks.track` with the method Object | MobileSAM | Apache-2.0 | 40.7 MB, `cutout.downloadModel` |

## Photos in depth

A photo is made to move in depth, its near parts further than its far ones, and blurred by how far
each part is, from how near each part of it is: Depth Anything V2 Small (Yang et al. 2024; a
DINOv2 ViT-S/14 encoder and a DPT head, written out on candle in `crates/vision/src/depth.rs`)
looks at the photo once (a fifth of a second on the GPU) and its answer is kept with the user's
data (`cache/depth/`), so a photo is looked at once across sessions and exports.

| Command | Does |
|---|---|
| `clip.photo3d` | the **3D Photo** effect on photo clips, keyframed over each clip: `move` `in` / `out` (the camera moving in or back: the near parts grow, the farthest stays), `left` / `right` / `up` / `down` (round the subject, which stays where it is, the nearer parts going one way and the farther ones the other), `circle` (a slow turn round it), or `auto` (the default: the clip's own Ken Burns move — in, out, or the way it pans — else in); `amount` 0–100 (50: the nearest moves 2.5 % of the width either way, or grows 10 %); `pivot` the depth that stays (`auto`, `far`, `subject` or 0–1) |
| `clip.depthBlur` | **Depth Blur**: `focus` the depth kept sharp (`subject`, the default, `near`, `far` or 0–1), `blur` the farthest blur (1.5 % of the width), `range` the depth of field (0.1), `pull` a focus pull from the other end of the depth to the focus over the first half of the clip |
| `depth.status`, `depth.downloadModel` | whether the network is downloaded (name, size, licence, source); the download, after the user agrees |

The subject's depth is the median inside its box (`subject`: `[x, y, w, h]`, 0–1 of the picture,
what a vision model found), else the 80th percentile of the picture's middle 60 % (the main thing
in front). Both effects are ordinary effects: their parameters (Travel Across, Travel Down, Push
In, Still Depth, Zoom to Fill; Focus Depth, Depth of Field, Blur, Blur the Foreground) keyframe
and show in Effect Controls.

**AutoCut** uses them by itself when the network is downloaded: the director (`autocut.direct`)
and showcases give each flat photo the 3D photo that matches its Ken Burns move (`--no-depth`
switches it off; the report lists the photos as `photos3d`).

**How the 3D photo is drawn** (`crates/render/src/depth.rs`). A part of nearness `n` (0 the
farthest, 1 the nearest) moves by `(n − still depth) · V`, `V` the nearest part's travel plus its
push away from the middle. Each output pixel looks back along `V`, from the nearest depth to the
farthest, a pixel's worth a step (at most 96), for the first surface whose moved place it is, so a
near thing covers what is behind it. The depth is first spread outwards by 0.4 % of its width, so
an outline moves whole with its thing. Zoom to Fill enlarges the picture just enough that the
largest move the keyframes ask for shows no edge (1.05× for the default arc). Push In at P makes
the nearest part 1 / (1 − P) times as large.

**How the depth blur is drawn.** Each part's blur is a disc whose radius grows from nothing inside
the depth of field to the full Blur half a depth beyond it (in front too, unless Blur the
Foreground is off). A pixel gathers the discs that reach it — every pixel within the largest
radius, or 256 taps on a golden-angle spiral beyond 256 pixels' area — each weighted by the area it
stands for over its disc's, so a sharp pixel keeps its colour and a blurred one is its disc's
average; a farther part's disc counts only within the pixel's own, so a sharp thing in front is
not smeared over by the blurred background.

**Measured** (M3 Max, release, 2026-10-28). The network: 249 ms a picture on the GPU (924 × 518),
1.13 s on the CPU. The effects, GPU against the CPU (WGSL in `crates/gpu/src/depthfx.wgsl`, the
same arithmetic; the GPU's pictures within 1e-6 of the CPU's):

| Frame | 3D photo GPU | CPU | Depth blur GPU | CPU |
|---|---|---|---|---|
| 1920 × 1080 | 12 ms | 40 ms | 27 ms | 79 ms |
| 3840 × 2160 | 47 ms | 209 ms | 88 ms | 394 ms |

**Limits.** Photos only: a video's depth, made frame by frame, flickers, and 360° clips have their
own moves; the commands refuse both, saying why (the effects themselves work on any picture). What
a near thing uncovers is the picture behind it stretched — nothing is painted in yet — so the
moves stay small (a few percent of the width). Depth is relative (near to far within the picture,
not metres); glass, mirrors and fine hair are where it is least sure.

## Upscale and Restore

A photo or a video made two, three or four times as large, or restored at its own size, with
its blur, noise and compression's blocks taken out: Real-ESRGAN's compact general model (Wang et
al. 2021; `realesr-general-x4v3`, SRVGGNetCompact — 33 convolutions of 64 channels, ×4) written
out on candle in `crates/vision/src/upscale.rs`, run on the GPU as WGSL passes
(`upscale_gpu.rs`, `esrgan.wgsl`). The network was trained, as its project says, on DF2K (DIV2K,
whose site gives it for academic research, and Flickr2K) and OST with synthetic damage.

| Command | Does |
|---|---|
| `media.upscale` | a job: each item (`items`, else the selected clips' media) made `factor` times as large — 2, 3 or 4, 1 to restore it at its own size, or `auto` (the default: enough for the active sequence's frame, at most 4; items at least two thirds as large as it, or with a copy, are left out, saying why) — with `strength` 0–1 of the network's answer (the rest the picture enlarged smoothly); a photo's copy is a PNG, a video's an H.264 (`format: "prores"`: ProRes 422 HQ), beside the media (or in `folder`), shown in its place once written (one undo step); `off` shows the media as it is |
| `upscale.status`, `upscale.downloadModel` | whether the network is downloaded; the download, after the user agrees |

**In its place.** The copy becomes the item's Interpretation (`upscaled`): the item keeps its
size and its clips their framing, masks and effects. The media pool gives the item's frames from
the copy ([`SharperSource`](../crates/engine/src/media_pool.rs)), and the renderer asks it for as
much detail as a clip shows the picture with, up to the copy's size
(`MediaSource::max_scale`; every other source still gives at most its own size): a 480p clip
filling a 1080p frame is drawn from the copy at 2.25 times its size, in a 4K frame at 4. The CPU
renderer and the frame plan the GPU compositor draws both do (`sharper_tests.rs`). The copy has no
sound; the item's own plays. AutoCut upscales photos much smaller than the frame (their long side
under two thirds of the frame's, as `auto` does) before placing them, when the network is downloaded (the
director and showcases, `--no-upscale`).

**How it runs.** A picture is worked on in tiles of 512 pixels read with a 34-pixel margin (as far
as the 33 convolutions reach), so the tiles join exactly (within one level). A factor of 2, 3 or 1
is the network's ×4 averaged down. The GPU passes are a tiled matrix product of each layer's
pixels by its channels over the input's channels × 9 taps, reading each pixel's neighbourhood in
place; candle's own Metal convolution (an im2col buffer, then a matrix product) runs at about
450 GFLOPS whatever the precision, seven times slower.

**Measured** (M3 Max, release, 2026-10-28). A 640 × 480 picture: 474 ms on the GPU, 4.4 s on
the CPU (candle on Metal 3.0 s); the GPU within 2.3e-6 of the CPU. Six photos of 320 × 180 to
320 × 240 made 4× in 2.0 s. A 3 s 480p clip (90 frames) made a 3416 × 1920 H.264 in 56 s (0.62 s a
frame: a minute of video in about 19 minutes). Against the originals of three of the user's
photos reduced four times (and the same saved as JPEG at quality 30):

| Photo | PSNR bicubic / Real-ESRGAN | after JPEG 30 | Edge strength: original, bicubic, Real-ESRGAN |
|---|---|---|---|
| a skull (1280 × 720) | 31.9 / 29.8 dB | 29.0 / 28.7 dB | 10.2, 2.1, 2.7 |
| a printer's bed (1280 × 960) | 32.8 / 29.8 dB | 29.7 / 28.4 dB | 5.2, 2.2, 4.9 |
| a printer on a shelf (1280 × 852) | 28.3 / 27.0 dB | 26.0 / 25.1 dB | 10.4, 2.8, 6.8 |

So it is not a truer picture — by PSNR it is 1.3–3 dB further from the original than a bicubic
enlargement, as GAN-trained upscalers are — but a much crisper one: edges 1.3–2.4 times as
strong, JPEG's blocks gone, clean lines on hard objects. On the 480p clip in a 1080p frame the
edge strength rose from 2.5 to 5.4; the fine detail's change from frame to frame rose 2.5 times
with it, so the frames are sharper without much shimmer of their own (each frame is made alone).

**Limits.** Fine texture (skin, cast metal, foliage) comes out smoothed, a little painted: for
those `strength` 0.6–0.8 keeps some of the picture's own grain. Faces are not restored as faces
(no face model). The copy is at most 8192 pixels a side for a photo and 7680 × 4320 for a video;
the network takes pictures up to 4096 a side; 360° media and image sequences are left out. A
video's copy is slow (about 19 times its length at 480p ×4) and large (62 MB for those 3 s).

## Smooth Slow Motion

A slowed clip shows frames made between its frames instead of repeating (or blending) them:
RIFE 4.26 (Practical-RIFE, Huang et al.; IFNet's five blocks estimating the motion both ways
coarse to fine, warping both frames and their features to the moment between and mixing them by a
learned mask) written out on candle in `crates/vision/src/interp.rs` and run on the GPU as WGSL
passes (`interp_gpu.rs`, `rife.wgsl`). The training data of the 4.x models is not published (the
paper's model was trained on Vimeo-90K).

| Command | Does |
|---|---|
| `media.interpolate` | a job: for the clips asked for (`clips`, else the selected ones: the part of their media they play, three frames either side) or whole items (`items`), a copy with `factor` frames for each of the media's — 2 to 8, or `auto` (the default: as many as the slowest clip needs for the sequence's frame rate, its speed sampled along it so Time Remapping and speed ramps count; clips that play at their own rate or faster are left out, saying why) — an H.264 (`format: "prores"`: ProRes 422 HQ) beside the media (or in `folder`), shown in its range once written (one undo step); `off` shows the media as it is |
| `interpolate.status`, `interpolate.downloadModel` | whether the network is downloaded; the download, after the user agrees |

**In its place.** The copy, with the media time range it covers, becomes the item's
Interpretation (`interpolated`): the item keeps its frame rate, its clips their speeds and
timing. Inside the range the media pool gives the copy's frames
([`FinerSource`](../crates/engine/src/media_pool.rs)), outside it the media's own, and the
renderer times a clip's frames at the copy's finer rate (`MediaSource::time_detail`): Frame
Sampling shows the made frame nearest each moment; Frame Blending and Optical Flow work between the
made frames. The copy has no sound. AutoCut's director makes its speed ramps' slow parts smooth this
way when the network is downloaded (`--no-smooth`).

**How it runs.** Frames are padded to a multiple of 128 (256 over 2048 pixels, looked at at half
scale) and the blocks look at them at 1/16 to the full size. The GPU passes: each 3 × 3
convolution a tiled matrix product (three tile shapes, so layers of 16 or 32 channels keep their
lanes busy), the transposed convolutions visiting only the four taps that reach a pixel with the
pixel shuffle folded in, and resizes, warps, mixes and copies writing straight into channel
ranges (the blocks' inputs are never concatenated). candle on Metal ran the same network as
hundreds of separate whole-frame operations: 1.1 s a 1080p frame. The copy's frames are made as
the export asks for them, the last pair of media frames kept; nothing is locked while a frame
decodes or the network runs (a decoder that works in parallel lets its thread take another frame:
holding a lock there hung the export, `a_slowed_clip_gets_the_frames_between_its_frames`).

**Measured** (M3 Max, release, 2026-10-29). A 1080p frame made in 183 ms on the GPU (190 ms with
its conversions), 4.7 s on the CPU, 1.1 s through candle on Metal; the GPU within 1.4e-6 of the
CPU; a drawn square moving 16 px is placed halfway (0.32 levels from the true middle frame). The
middle of three real frames made from the outer two, against the real one (`the_middle_of_real_triplets`):

| Clip | Frames apart | RIFE | cross-fade | the frame held |
|---|---|---|---|---|
| a 50 fps 1080p clip, two moments | 1 / 2 | 38.0, 39.2 / 34.1, 35.1 dB | 34.5, 29.7 / 31.4, 27.0 dB | 30.7, 27.2 / 28.9, 24.8 dB |
| a 59.94 fps 1080p phone clip, two moments | 1 / 2 | 39.1, 38.3 / 37.9, 38.5 dB | 32.2, 33.4 / 27.7, 27.2 dB | 27.7, 27.4 / 23.8, 23.2 dB |
| a 640 × 640 clip of variable frame rate (29.8 fps), two moments | 1 / 2 | 27.8, 24.2 / 22.0, 23.6 dB | 23.1, 24.4 / 21.7, 23.8 dB | 20.8, 21.2 / 18.4, 20.7 dB |

On the steady-rate 1080p clips the made frame is 3.5 to 11 dB closer to the real one than a
cross-fade. On the variable-rate clip it is within 0.3 dB of a cross-fade at three of four moments:
its frames are not evenly spaced in time, so the real middle frame is not at the halfway time.

A second of the 50 fps clip slowed to 25 % in a 50 fps sequence: `auto` chose four frames for
each, the copy (1.14 s of media, a few frames either side: 228 frames, 171 of them made) took
32.9 s, 0.19 s a made frame. Before it, the slowed clip showed each frame four times and then
jumped (the change between output frames 0, 0, 0, 4.1, 0, 0, 0 levels); with it, every output
frame moved on alike (2.0 to 2.1 levels each).

**Limits.** Frames are made between the media's frames: motion faster than a few dozen pixels a
frame, things that appear from nowhere and repeating patterns come out smeared or doubled (as with
any flow-based interpolation). A copy is at most 240 frames a second and 18,000 made frames (about
an hour's work): a whole long recording is refused — choose the slowed clips. One copy of an item
at a time (not with Upscale and Restore). Variable-frame-rate phone video is timed as its average
rate.

## Object Mask

A thing in a clip is masked from a click on it and followed through the clip, the mask taking its
shape in every frame it is tracked to (the idea of Premiere 26's Object Mask). MobileSAM (Zhang et
al. 2023; the cut-out model of [`clip.cutOut`](360-video.md), Segment Anything's prompt encoder
and mask decoder behind a TinyViT image encoder) cuts the thing out; a box tracker tells it where
the thing went.

| Command | Does |
|---|---|
| `clip.objectMask` | a job: the thing at `point` (`[x, y]`, 0–1 of the clip's picture; more points on it in `inside`, points off it in `outside`) on the clip's frame `at` (seconds into the clip; the playhead) cut out and outlined (`points`, 128) as a mask on `effect` (Opacity) — a new one, `name` (`Object (n)`), `mode` (`add`: on Opacity the clip shows the thing alone), `inverted` — or the mask `mask` cut again there; then followed `track` `both` (the default: on to the clip's end and back to its start), `forward`, `backward` or `none`, every `every` seconds (0.1; masks interpolate between their outlines). The mask's tracking method becomes Object; one undo step when waited for |
| `masks.track` | a mask whose method is Object (or `method: "object"`: a mask drawn round a thing too) follows the thing it outlines from the playhead, `forward` or `backward` (`frames` at most), replacing its outlines on the way |

**In the app.** Effect Controls ▸ a mask row's sparkle turns the Object Mask tool on (asking for
MobileSAM first when it is not downloaded): a click on the Program monitor masks the thing there
on that frame, Shift-click adds to it, Option-click takes away (the same mask cut again with all
the points), Escape ends the tool. The mask's track buttons then follow it (the tracking method
menu shows Object).

**How a thing is followed.** On the clicked frame the points go to MobileSAM: for one point the
surest of its three answers for an ambiguous prompt (a part, the thing, the thing with what it
stands on), for more its one answer. In each frame after, three guesses at where the thing went —
the box tracker's (the subject tracker of motion tracks: median flow with a template that finds a
hidden thing again), the thing's speed's over its last half second, and where it was — each
move the thing as last seen whole; for each, MobileSAM is given the box round it (grown by a
quarter), its deepest points and points beside it in the box (the box's corners and the middles of
its sides where the thing is not, so it does not take the table with the car), and of its four
answers the one overlapping the shape expected most is kept, if it overlaps by at least 0.45, is
coloured like the thing (64-bin colour histograms, Bhattacharyya ≥ 0.5) and is at most 1.6 times
the size expected. A cut reaching its box's edge is asked for again with a larger box, and that
answer kept when it is more like the thing. Cuts are compared with the thing as last seen *whole*
— a cut at least 85 % of the most it covered lately (that memory letting go by 40 % a second and
taking on at most 50 % a second) — not with the frame before: frame to frame, a thing going behind
a post shrinks away a little each frame and the mask would creep onto the post; the box tracker
restarts only from whole sightings. Frames where nothing passes keep no outline (the mask
interpolates over them); the thing lost for 1.5 s ends that way of the run. The cut's largest
piece and the pieces of at least a fifth of it (the thing either side of a post) are one outline.

**The model's prompts.** candle's `Sam` takes points only, and its prompt encoder drops a
dimension embedding a box; `crates/vision/src/segment.rs` embeds a prompt's points and box corners
itself — each a random-Fourier code of where it is plus its kind's learnt vector, from the same
weights — and runs MobileSAM's own mask decoder on them (`Segmenter::answers`).

**Measured** (M3 Max, debug-optimised test build, 2026-10-29). MobileSAM looks at a picture in
0.53 s on the GPU (Metal through candle) and 3.2 s on the CPU; its four answers to a prompt take
53 ms; a look of the follower is 0.6–0.7 s, so 10 s of video at the default 0.1 s is about a
minute. A striped ball crossing a forest behind a post (640 × 360, 3 s; `mobilesam_follows_a_ball_behind_a_post`):

| | frames outlined | overlap with the ball as it shows (intersection over union) |
|---|---|---|
| Object Mask (clicked at 0.5 s, every frame) | 68 of 72 | mean 0.953, worst 0.63; 0.95–0.98 away from the post, the ball picked up again after it |
| Track Mask, position, scale and rotation, from the click's outline | 39 (lost at the post) | mean 0.61, worst 0.06 |

On the user's clips (`object_mask_on_a_real_clip`, 6 s each, strips of eight frames looked at): a
toy Tesla on a bench while the camera circles it — the mask stays on the car as its outline turns
from side-on to three-quarter front (Track Mask's fixed shape drifts off it), each outline
overlapping the next by 0.91 on average; a pair of pliers' handle as the camera moves — every
frame on the handle (0.88); a toy car on a table as the camera pans — on the car, with the cable
plugged into it taken as part of it (0.88).

**Limits.** 0.6 s a look: long clips take a while (a background job, which can be stopped). Things that touch something of their colour may take it with them; thin parts (cables,
handles) come and go between frames. A thing hidden for more than 1.5 s ends the run; a thing that
reappears elsewhere is not looked for. Frames where nothing passed have no outline of their own.
One mask of a clip is followed at a time (the clip's masks are tracked one job at a time).

## Remove Object

The thing a clip's mask follows (an Object Mask) is taken out of the picture, and the hole
filled — first with what the other frames show there, then, where none does, with what MI-GAN
(Sargsyan et al., Picsart AI Research, ICCV 2023; trained on Places2, whose terms are for
non-commercial research and education) paints. It is written once as a copy of the clip's range
and shown in the media's place, as Smooth Slow Motion's is.

| Command | Does |
|---|---|
| `clip.removeObject` | a job: the clip's mask (`mask`: index or name on `effect`, Opacity; default the selected mask, else its last Object mask) grown by `grow` (a hundredth of the picture's short side, at least 8 px) and softened by `feather` (6 px) is the hole; for the media the clip plays where the mask has outlines (0.3 s either side), a copy with the hole filled (`fill`: `auto`; `model`, MI-GAN in every frame, to compare), an H.264 (`format: "prores"`: ProRes 422 HQ) beside the media (or in `folder`); when written the item shows it in its range and the mask is switched off, one undo step; `off` shows the media as it is |
| `removal.status`, `removal.downloadModel` | whether MI-GAN is downloaded; the download, after the user agrees |

**How the hole is filled.** The background's motion from each frame to the next is measured
once, from feature points in a ring round the hole (the background beside the thing, which moves
like what is behind it even where the far background does not; the whole picture when the ring
has too little detail), and chained. For each pixel of a frame's hole, the frames up to 40 either
side, nearest first, are looked at where that motion takes it: what they show outside their own
holes are its candidates, and their median (when, three or more, they agree) is the pixel — real
background, so it stays put. A neighbour is used only when it lines up: moved by the motion, it
must show what the frame shows in a band round the hole (within 10 levels). What no frame shows,
or a hole the frames show less than 60 % of (a thing standing still on what is behind it), is
painted by MI-GAN — in key frames half a second apart, each painting carried to the frames near
it by the background's motion and the two nearest cross-faded, so the painting does not change
from frame to frame. MI-GAN sees a square round the hole (three times its size, at least 512
pixels) scaled to 512; the filled hole is blended in by the mask's soft edge.

**The network.** MI-GAN's 512 generator as its authors export it to ONNX: an encoder from 512 to
4 pixels and a StyleGAN2-like synthesis back, every layer a depthwise 3 × 3 and a pointwise
convolution, with fixed noise and skips from the encoder. `crates/vision/src/inpaint.rs` writes
it out (its weights and noise read from the file, which runs nothing), a CPU reference of plain
loops and candle matrix products, and `migan.wgsl` runs it on the GPU: a depthwise pass (plain,
strided, or reading its input up by 2 with zeros between the pixels) and a tiled matrix product
for the pointwise layers, each finished by the bias, noise, activation and skip it needs, every
pass and binding made once (the size is fixed). candle could not be used for it: it runs a
grouped convolution as one convolution a group, 512 for each depthwise layer.

**Measured** (M3 Max, release, 2026-10-29). MI-GAN: 40 ms a picture on the GPU, 369 ms on the
CPU, the GPU within 2.6e-6 of the CPU; a drawn scene's disc filled 10.5 levels from the scene
without it (75 from the disc). Removal (`remove_object_tests.rs`), on a movie whose background is
known — a textured picture panning 2 px a frame, a disc crossing it, masked frame by frame:

| | inside the disc's place, from the true background | frame to frame |
|---|---|---|
| before | 77–120 levels | — |
| `auto` (from the other frames) | 0.9–1.1 levels | within 0.2 levels of the background's own move |
| painted in every frame (a stand-in painter: the ring's colour) | 32–40 levels | — |

A still shot with a camera's grain (±8 levels) and a disc that never moves, so all of it is
painted by MI-GAN: the hole changes 1.00 levels from frame to frame in `auto` (key frames), 6.75
painted in every frame. On a 3 s clip of the user's (a toy car on a table, the camera walking
round it; masked by Object Mask in 19 s): the removal took 6.3 s for 90 frames (0.07 s a frame),
the car is gone in every frame; the frames show the table behind it only in slivers, so it is
painted — the hole changing 3.08 levels from frame to frame (the camera's own motion in that),
4.98 painted in every frame.

**Limits.** Shadows and reflections stay unless the mask covers them (`grow` takes in a little
round the thing; the toy car needed 30 px). Where nothing shows what is behind the thing, the fill
is MI-GAN's guess: soft, right for floors, walls, grass and sky, not for detailed scenes. The
background's motion is position, scale and rotation: a camera moving in depth past a near thing
lines up less, and then more is painted. 360° media is refused (remove things from the reframed
picture), and so are items with another copy (Upscale and Restore, Smooth Slow Motion); a second
removal reads the first's copy, so removals stack. At most 7,200 frames a job.

