# AI effects

FilmCraft's local AI effects for ordinary pictures (the 360° ones are in
[360-video.md](360-video.md)). Every model runs on this computer, is downloaded only when the user
agrees (its name, source, size, licence and training data shown first), is pinned to a revision
and checked against its SHA-256, and is never bundled or committed. Only weights under Apache-2.0,
MIT or BSD licences are used, never non-commercial ones. Each runs on the GPU (Metal on macOS)
with the CPU as the fallback.

| Feature | Commands | Model | Licence | Download |
|---|---|---|---|---|
| Photos in depth: the 3D photo, depth blur | `clip.photo3d`, `clip.depthBlur` (Clip ▸ Depth), the effects 3D Photo (Perspective) and Depth Blur (Blur & Sharpen) | Depth Anything V2 Small | Apache-2.0 | 99.2 MB, `depth.downloadModel` |
| Upscale and Restore: photos and videos made larger and cleaner | `media.upscale` (Clip ▸ Modify) | Real-ESRGAN general x4v3 | BSD-3-Clause | 4.9 MB, `upscale.downloadModel` |

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
not metres); glass, mirrors and fine hair are where it is least sure. In the desktop app the
download has no dialog yet: agents and the CLI download it (`depth.downloadModel`).

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
video's copy is slow (about 19 times its length at 480p ×4) and large (62 MB for those 3 s). In
the desktop app the download has no dialog yet: agents and the CLI download it.

