# The scene index

> **Last updated:** 2026-11-22 · **Change:** `media.embed` and `media.nearest` (vectors for search by meaning outside a project)

What is in a clip, shot by shot, so an edit can be planned from it and a template can fill
itself: where the cuts are, what each shot shows, how good it looks, what it is good for, and an
embedding to find it by. Everything runs locally (on the GPU on macOS); every model is optional.

| command | does |
|---|---|
| `media.analyzeScenes {items?, detail: "fast"\|"full", describe, embed, aesthetic, fresh, wait}` | a background job (`jobs.list` progress, `jobs.cancel`): index the items (default the Project panel's selection, else every video and photo item); one undo step, "Analyze Scenes" |
| `media.scenes {item, embeddings?}` | an item's index: each shot's start, end and keyframes (media seconds), camera move, quality, description |
| `media.searchScenes {query, limit, items?}` | shots ranked for words: what they look like (SigLIP 2), what their descriptions say, what is said in them (transcripts) |
| `media.bestShots {items?, purpose, count}` | the best shots for a `purpose` — hook, broll, reveal, talking-head, establishing — diverse |
| `media.checkText {windows, aspect?, texts?, asks?, frames?, wait}` | a background job: words against the frames they will be shown over — each window (`{item, from, to, focus?}`, media seconds) looked at in `frames` (1–4) frames, cut to what an edit of frame shape `aspect` (width / height) filled around `focus` shows; SigLIP 2's `cosine` and calibrated `match` of every text with every window, the windows' `similar`ity, and for each ask (`{window, text, about?}`) Qwen3-VL's `yes` / `no` / `unclear`: is the thing the text names (`about`: one at a time, which it answers far more reliably), or everything it names, visible? ([`crates/engine/src/scene_check.rs`](../crates/engine/src/scene_check.rs)) |
| `media.embed {paths?, texts?, aesthetic?, maxSide?, precision?}` | vectors for a catalogue outside a project (FrameAtlas): each picture file's unit-length SigLIP 2 image embedding (up to 256 files, scaled to `maxSide` 160–1024 px first) and, with `aesthetic`, its aesthetic score (about 1–10); each text's embedding (up to 64); base64 of little-endian `f16` (default) or `f32`, and the `calibration` that turns a cosine into SigLIP's match probability. Takes the GPU lock; nothing is downloaded ([`crates/engine/src/embed.rs`](../crates/engine/src/embed.rs)) |
| `media.nearest {index, query? \| vector?, dims?, precision?, limit?, min?}` | the rows of an index file (`n × dims` little-endian `f16` or `f32`, as `media.embed` gives them, one after another) nearest a text `query` or a `vector`, best first: `row`, `cosine`, and for a text its calibrated `match` |
| `scenes.status` | the models: installed, sizes, licences, what `fast` and `full` still need |
| `scenes.downloadModel {model: "fast"\|"full"\|<id>, wait}` | download them (pinned, SHA-256 checked) after the user agreed |

CLI: `filmcraft-cli scenes analyze [media…] [--detail full] [--download] [--sheet DIR]`,
`scenes show <item>`, `scenes search "<query>"`, `scenes best --purpose hook`,
`scenes sheet DIR` (an `index.html` of the keyframes with their captions, `scenes.json`,
`sheet.jpg`), `scenes status`, `scenes download [fast|full|<model>]`. MCP: `scenes_analyze`
(a long job), `scenes_search`. Templates fill themselves from it: `filmcraft-cli quick`
([quick-templates.md](quick-templates.md)).

## What is in a shot

```json
{"shot": 0, "start": 4.0, "end": 10.2, "keyframes": [7.1], "camera": "pan-right",
 "quality": {"sharpness": 0.93, "laplacian": 640.2, "brightness": 0.46, "dark": 0.01, "blown": 0.0,
             "exposure": 0.95, "shake": 0.04, "motion": 0.21, "aesthetic": 5.4, "score": 0.86},
 "description": {"shotType": "wide", "caption": "A humanoid robot stands in a crowded exhibition hall.",
   "setting": "an exhibition hall", "mood": "curious", "energy": "medium", "people": 6,
   "facesProminent": true, "uses": ["b-roll", "establishing"],
   "subjects": [{"name": "humanoid robot", "box": [0.42, 0.18, 0.21, 0.66]}],
   "actions": ["people watch the robot"], "text": ["GITEX"], "detail": "…(full only)"}}
```

- **Cuts**: TransNetV2 on 48×27 frames at the clip's rate (at most 30 a second), a transition
  where its probability passes 0.5 (a dissolve is one transition, cut at its middle), shots of
  at least 0.4 s. Without it, the luma-grid and histogram detector of Scene Edit Detection.
- **Quality**: sharpness from the variance of the Laplacian on a 512-px keyframe; exposure from
  the mean luma and the crushed and blown shares; the camera's move and its shake from the global
  shift between consecutive analysis frames (96 px, block matching); the LAION aesthetic score
  (CLIP ViT-L/14); `score` weighs them all.
- **Description**: Qwen3-VL 2B (`fast`, one keyframe at about 448×448 pixels) or 4B (`full`, two
  or three keyframes at about 550×550, and a paragraph). Asked for JSON and read defensively
  (`filmcraft_scenes::json`): a cut-off answer, a list left open, a stray value — what can be
  read is kept, nothing in an answer can fail the job.
- **Embedding**: SigLIP 2 base on the keyframes (their mean), stored as half floats.

The index is stored with the project (`Project::scenes`) and in `<data dir>/scene-cache` by the
file's content, picture size, length and rate, the detail and the models' pins, so footage is
indexed once (in this project or any other). AutoCut's clip analysis (`autocut.analyze`, the
showcase) reads it when it is there: the shots say where the subject is and which moment shows
it best (sharp, steady, large), and the subject's box; the local vision model then only reads the
text and names the subject, instead of looking at every sampled frame.

## Models

See `crates/scenes/README.md` for the catalogue (ids, files, sizes, licences: TransNetV2 MIT,
Qwen3-VL Apache-2.0, SigLIP 2 Apache-2.0, CLIP and the LAION head MIT) and the implementation.
`fast` is 7.5 GB, `full` 12.2 GB; Qwen3-VL 8B (17.5 GB) is in the catalogue for
`model: qwen3-vl-8b` downloads but not used by default.

## Measured (M3 Max, 2026-10-22)

On ten of the owner's robot clips (213 s of 1080p and 4K phone video; each clip one continuous
shot), the release CLI, Metal, with three other build lanes loading the machine (load average
12–15):

| stage | time | rate |
|---|---|---|
| decoding the analysis frames (6404, VideoToolbox, reduced in Y'CbCr) | 21.7 s | 295 frames/s |
| TransNetV2 | 26.5 s | 242 frames/s |
| keyframes, quality, camera moves | 2.9 s | |
| CLIP aesthetic + SigLIP 2 embeddings (load included) | ~2 s | |
| Qwen3-VL 2B descriptions (load 2.0 s, 2 batches by picture size) | 15.5 s | 0.65 shots/s |
| **all** | **68.7 s** | 3.1× real time |

The describer alone, a batch of 16 keyframes of one size (`scenes_probe --repeat 16`): 41 ms a
decoding step for all 16 (30 ms for 8, 27 ms for 2), vision encoder 0.19 s and prefill 0.14 s a
shot; about 1.5 shots a second with answers of 140 tokens, 1.85 with 80. Batches fill up when a
library has many shots of one size; ten clips in two sizes make batches of five.

**An edited video.** The 34-second vertical trailer AutoCut made of the robot showcase (a title
card, 17 robots cut on the music's beats, an end card): TransNetV2 finds its 19 shots, one for
each robot, the title and the end card; the 2B describer reads each robot's name and year off
the frame ("a robot named PROMOBOT ROBO-C 2021", "the text 'BOSTON DYNAMICS SPOT 2022'"). The
ten clips and the trailer together (29 shots, with the other lanes' builds at a load average of
59): the describer 1.06 shots a second (27 s for 29 shots in 3 batches, plus 3.7 s to load);
decoding, CPU-bound, slowed to 82 frames a second under that load, TransNetV2 kept 236.

**`full` (4B).** Four of the clips, a batch of four with three keyframes each (load average 38):
load 8.2 s, vision 3.3 s, prefill 9.2 s, decoding 17.9 s (986 tokens: 73 ms a step for the
four) — about 0.13 shots a second, an eighth of `fast`. It reads small text the 2B gets wrong
("ARDET" where the 2B read "ABDET") and writes the paragraph.

**Memory** (peak footprint, which on Apple silicon includes the GPU's): 6.7 GB without the
describer (decoding, TransNetV2, CLIP, SigLIP), 18.8 GB with the 2B at batches of 16, 25.6 GB
with the 4B at batches of four. Two things keep it there: the weights are read one tensor at a
time and the fused projections joined on the CPU (joined on the GPU, candle's buffer pool kept
the parts too: 32 GB for the 4B), and a prefill's attention scores are worked out one shot at a
time.

What made it fast enough: the instruction tokens before the picture, worked out once and copied
to every shot; projections fused (q/k/v, gate/up); candle's fused attention for the one new
position; the batch decoding in lockstep; TransNetV2's convolutions as matrix products over
shifted rows (candle's Metal `conv2d` was 8× slower); frames reduced while still Y'CbCr before
the RGBA conversion (3× faster decoding).

## Sample descriptions (fast, 2B)

From the live run (`target/agent-scenes/demo/index.html` in the lane's build folder):

- *Ameca at an exhibition*: "A humanoid robot with a metallic, textured surface stands in a
  crowded exhibition hall, surrounded by people who are observing it with interest." — wide,
  static, uses b-roll / talking-head, score 0.85.
- *A delivery robot*: "A white robot with a transparent dome on its body is shown from a low
  angle, revealing its internal components." — close-up, text read: "AI".
- *A Boston Dynamics Spot*: "A yellow robot dog is walking on a white floor in a modern
  exhibition hall." — wide, static, b-roll.
- *A police robot outdoors*: "A large orange robot with a round body and a large screen on its
  face stands on a paved area, surrounded by people and vehicles." — wide, aesthetic 6.8, score
  1.00 (the best b-roll and hook of the ten).
- *ARDET's quadruped*: "A red robot with six wheels and the word 'ABDET' on its body is on a red
  mat, surrounded by people." (the badge reads ARDET).
- *A floor cleaner*: "A robotic cleaning device with a green and gray body is on a tiled floor."

The 2B model misreads small text, sometimes names the nearest object rather than the robot (a
delivery robot's rear "a white car, showing its exhaust pipes") and sometimes lists words that
are not in the picture as its text ("black shoes"). The same clips with `detail: "full"` (4B):

- *A Boston Dynamics Spot*: "A yellow quadruped robot walks across a showroom floor as people
  observe it." — text: "Boston Dynamics", "COMBINED ROBOTICS SOLUTIONS"; the paragraph notes the
  glowing floor lines and the robot's green LED.
- *ARDET's quadruped*: "A red and gray robotic vehicle with four wheels moves across a red mat
  at an exhibition." — "The robot, labeled 'ARDET' and 'Swiss-Mil', has a mechanical body with
  visible gears and a mounted camera…"
- *A police robot outdoors*: "A bright orange robot with a large reflective screen is the center
  of attention as a young boy approaches it."
