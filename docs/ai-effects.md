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
