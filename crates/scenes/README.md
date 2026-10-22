# filmcraft-scenes

The scene index (`media.analyzeScenes` in `crates/engine/src/scenes.rs`): what is in a clip, shot
by shot — where the cuts are, what each shot shows, how good it looks, and an embedding to find
it by — so editing can be automated and templates can fill themselves. L2; pure Rust. The index
itself is `filmcraft_project::scenes::SceneIndex`, stored with the project.

## Models

Downloaded by `scenes.downloadModel` after the user agreed, pinned to a revision (40-hex commit),
SHA-256 and size (`catalog`), into `<data dir>/models/<id>/`; never bundled or committed. Only
non-gated weights with permissive licences from the authors' own repositories (checked
2026-10-22):

| id | model | files | size | licence |
|---|---|---|---|---|
| `transnetv2` | TransNetV2 shot boundaries (Souček & Lokoč) | TensorFlow checkpoint `variables.index`, `variables.data-00000-of-00001` (github.com/soCzech/TransNetV2, Git LFS) | 30.5 MB | MIT |
| `qwen3-vl-2b` | Qwen3-VL-2B-Instruct | `config.json`, `tokenizer.json`, `model.safetensors` | 4.26 GB | Apache-2.0 |
| `qwen3-vl-4b` | Qwen3-VL-4B-Instruct (`detail: full`) | two safetensors shards | 8.88 GB | Apache-2.0 |
| `qwen3-vl-8b` | Qwen3-VL-8B-Instruct (`model: qwen3-vl-8b`) | four safetensors shards | 17.5 GB | Apache-2.0 |
| `siglip2-base-224` | SigLIP 2 base, patch 16, 224 px (Google) | `config.json`, `tokenizer.json`, `model.safetensors` | 1.54 GB | Apache-2.0 |
| `clip-vit-l14-aesthetic` | CLIP ViT-L/14 (OpenAI) + LAION aesthetic predictor (`sa_0_4_vit_l_14_linear.pth`) | `model.safetensors`, the head | 1.71 GB | MIT, MIT |

`fast` uses TransNetV2, Qwen3-VL 2B, SigLIP 2 and the aesthetic head (7.5 GB); `full` the 4B
describer instead (12.2 GB). Every model is optional: without TransNetV2 the cuts come from the
luma-grid and histogram detector (`filmcraft_render::scene`); without a describer the shots have
no description; without SigLIP no visual search (word search still works); without the aesthetic
head the quality score leaves it out.

## Modules

- `catalog`: the models above.
- `boundaries`: TransNetV2's 100-frame windows (25 frames of context each side), cut
  probabilities to cuts (a run of frames above the threshold is one transition; the new shot
  starts after its middle), contiguous shots with a minimum length, and keyframes per shot.
- `quality`: sharpness (variance of the 4-neighbour Laplacian on a 512-px luma, log-mapped from
  20…1000), exposure (mean luma, crushed and blown shares), the camera's move and shake (global
  shift between analysis frames by exhaustive block matching of a 96-px luma, mean removed; the
  steady part is the pan or tilt, the jitter about a 5-frame average is shake), and the overall
  score.
- `prompt` and `json`: what Qwen3-VL is asked (the answer is started for it with
  `{"shot_type": "`), and the defensive reading of its answer: the first balanced JSON object,
  else a lenient reader that closes what the token limit cut off, ends a list left open before
  the next key, skips stray values and turns unrepresentable numbers into nothing; every field
  read leniently (a string or a list, a number or a word), capped, vocabulary mapped
  (`ShotType::from_words`, `ShotUse::from_words`). Mutation-fuzzed.
- `search`: query terms (stop words out, plurals folded), the share of them in a shot's
  description and in the words spoken during it, SigLIP's calibrated image–text match
  (`sigmoid(scale · cos + bias)`), combined 0.6 / 0.3 / 0.2.
- `best`: shots for a purpose (hook, b-roll, reveal, talking-head, establishing): quality,
  what the describer said the shot is good for, and what the purpose needs, picked greedily with
  penalties for a third pick from one clip and for a shot that looks like one already picked.
- `model` (feature `model`): the networks on candle, Metal on macOS (`FILMCRAFT_SCENES_CPU=1`
  for the CPU), under the machine's `gpu` lock while a job runs. Weights are read without
  memory-mapping (no `unsafe`): safetensors by a strict header check and one seek per tensor,
  TransNetV2's TensorFlow checkpoint by a strict reader of its table (LevelDB-style blocks,
  `BundleEntryProto`s) and its object graph, which names every variable
  (`TransNet/SDDCNN_1/DDCNN_1/Conv3D_1/conv_spatial/kernel`), so nothing is taken by position.
  - `transnet`: written from the paper and the authors' published architecture (MIT);
    channels last, the (2+1)-D convolutions as matrix products over shifted rows (a 3×3
    spatial kernel is nine shifted copies of the frame against the kernel's rows, edge masks for
    the zero padding; the dilated temporal kernel three shifted copies of the clip), half floats
    on the GPU, batch norm folded. candle's Metal `conv2d` was 8× slower and its `mean` over
    middle axes wrong, so neither is used. Finds the cuts of a synthetic clip exactly (live
    test).
  - `qwen3vl`: Qwen3-VL written for FilmCraft from the published configuration and
    architecture (Hugging Face `transformers`' `qwen3_vl`, `candle-transformers`' port): the
    vision encoder (16-px patches two frames deep, an interpolated 48×48 position table, 2-D
    rotary positions, 2×2 merger, DeepStack features into the first three text layers) and the
    Qwen3 text decoder with *interleaved* multimodal rotary positions (`candle-transformers`'
    0.9.2 port uses 1-D positions for pictures and a cache that cannot be reset, so it is not
    used). Batched greedy decoding: the shots of a batch share a picture size and frame count,
    so every prompt has its tokens in the same places and decodes in lockstep; the system and
    instruction tokens come before the picture, so their keys and values are worked out once and
    copied to every shot of every batch; no 10-gram repeats (greedy decoding loops); an answer
    ends when its JSON object closes. Memory: weights read one tensor at a time as the layers
    take them (the fused q/k/v and gate/up joined on the CPU first), the vision encoder at most
    4096 patches a pass, a prefill's attention one shot at a time.
  - `siglip`: `candle-transformers`' SigLIP with SigLIP 2's 256 k vocabulary; text lower-cased,
    `<eos>`, padded to 64.
  - `aesthetic`: `candle-transformers`' CLIP vision tower, the projection, and the LAION linear
    head (read by candle's pickle reader: data only).

Tokenizers: the `tokenizers` crate without default features, with `fancy-regex` (no `onig`, no
C++ `esaxx`).

## Speed

See `docs/scenes.md` for the measured speed on an M3 Max and sample descriptions.

**SmolVLM2** (Apache-2.0, 500M / 2.2B) was considered as the fast describer and not ported. It
is not in `candle-transformers` 0.9.2 (its `smol` module is SmolLM3), so it would be a third
architecture to write and keep (SigLIP encoder, pixel-shuffle connector, SmolLM2 decoder) and
another 1–4 GB download. What it could save is decoding time, and decoding is not where a batch
spends it: with batching and the shared prompt prefix the 2B Qwen3-VL decodes a batch of 8 at
about the per-step cost of one (the measurements in `docs/scenes.md`), and the remaining time is
the vision encoder and the picture tokens' prefill, which a smaller decoder does not shrink. And
Qwen3-VL is trained to ground objects with boxes in a 0–1000 grid, which the index keeps
(`subjects[].box`); SmolVLM2 documents no grounding output. So Qwen3-VL 2B stays the only fast
describer; SmolVLM2 is worth revisiting if a pass ever has to run on a machine without a GPU.

## Tests

`cargo test -p filmcraft-scenes [--features model]`. With `FILMCRAFT_SCENES_MODELS=<models dir>`
the model tests run against the real weights (TransNetV2 on a synthetic clip with known cuts);
the probe example runs every model on still pictures and reports speed:
`cargo run --release -p filmcraft-scenes --features model --example scenes_probe -- <models dir> <pictures…> [--query "a robot"] [--repeat 16 --max-new 80]`.
