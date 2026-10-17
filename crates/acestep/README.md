# filmcraft-acestep

Neural text to music for FilmCraft: **ACE-Step 1.5** (ACE Studio and StepFun, MIT) as an optional,
downloadable backend next to the procedural music engine (`audio.generate`). Engine commands:
`music.generateNeural`, `music.neuralModels`, `music.downloadNeuralModel`
(`crates/engine/src/music_neural.rs`). Roadmap item D2, first milestone: the turbo DiT, no LM
planner, instrumental music.

- `catalog`: the downloadable package — eight files from `huggingface.co/ACE-Step/Ace-Step1.5`,
  each pinned to revision `19671f40…` and its SHA-256 (6.3 GB). Downloaded by the engine only after
  the user agreed (`consent: true`) to the size, source and licence shown; never bundled or committed.
- `request`: the request (description, 5–240 s, seed, optional tempo 30–300 BPM, key, metre 2/3/4/6,
  shift 1/2/3), checked, and the conditioning text laid out exactly as the official pipeline does.
- `model` (feature `model`): the networks in pure Rust on candle. Off by default and in wasm builds.

## Licences (checked 2026-10-17)

| Part | What is used | Licence | Where it says so |
|---|---|---|---|
| DiT `acestep-v15-turbo` (2.4 B parameters with its encoders, bf16) and `silence_latent.pt` | weights | **MIT** | `license: mit` on the model card of `ACE-Step/Ace-Step1.5`; every other 1.5 DiT (`turbo-shift1/-shift3/-continuous`, `sft`, `base`, the `xl-*`) and the LMs (`acestep-5Hz-lm-0.6B/1.7B/4B`) are MIT too |
| VAE (`vae/`, Oobleck architecture, 48 kHz, hop 1920) | weights | **MIT** | the card of `ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format` (the same VAE): "Developed by ACE-STEP", MIT. It is ACE-Step's own VAE, not Stable Audio Open's (that one has a 2048 hop at 44.1 kHz and a non-open licence) |
| Text encoder `Qwen3-Embedding-0.6B` and its tokenizer, as shipped in the ACE-Step repository | weights, `tokenizer.json` | **Apache-2.0** | `Qwen/Qwen3-Embedding-0.6B` (Qwen team, Alibaba) |
| Official code `github.com/ace-step/ACE-Step-1.5` | read as the reference | **MIT** (`LICENSE`, © 2026 ACEStep) | its `acestep/third_parts/nano-vllm` is MIT too; no GPL file in the tree |
| `modeling_acestep_v15_turbo.py`, `configuration_acestep_v15.py` (shipped with the weights) | read as the reference, run as the oracle | **Apache-2.0** header ("The ACESTEO Team"; the Qwen3 config template) | this is where "Apache-2.0" in some write-ups comes from: code Apache-2.0, weights and repository MIT — both permissive |
| diffusers `AutoencoderOobleck` | architecture read, run as the oracle | Apache-2.0 | |
| `vector-quantize-pytorch` (ResidualFSQ of the audio tokenizer, LM path only) | oracle environment only | MIT | not ported: text-to-music without the LM does not use it |

Nothing NC/ND. Avoid `ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA` (CreativeML OpenRAIL-M). The
model card states the training data is licensed, royalty-free/public-domain and synthetic (MIDI
rendered) music and that generated music may be used commercially. No reference code is copied:
this is a port written from reading it (the clean-room rules in `AGENTS.md` apply to codecs and
formats; ML reference code under MIT/Apache may be read).

## The model and the variant

ACE-Step 1.5 is a hybrid: a **5 Hz language model planner** (Qwen3-based, 0.6/1.7/4 B) turns a request
into metadata, a rewritten caption and *audio codes* (a semantic sketch), and a **diffusion
transformer** turns caption, metadata, lyrics and those codes into 64-channel VAE latents at 25 Hz;
the **VAE** decodes them to 48 kHz stereo. The official pipeline also runs without the planner
("No LM", the recommendation for small GPUs): the DiT then conditions on the caption and metadata alone.

DiT variants (all 2 B, 4.79 GB in bf16, except the 4 B XL family at about 20 GB): `base` (50 steps with
classifier-free guidance, every task), `sft` (50 steps, CFG), and four distilled **turbo** models
(8 steps, no CFG; distilled on shift 1, 3, a joint 1–3, or continuous 1–5). **This port uses
`acestep-v15-turbo`**, the default of the main repository: 8 DiT passes per clip instead of the
100 that 50 CFG steps take (≈ 12× less work), the same download size as the other 2 B models, and the
authors' recommended "best balance" (joint shift 1–3 distillation) — it produces full, mixed music in
eight steps. The XL turbo would cost 4× the memory for higher fidelity; the LM planner another
1.3–8 GB and an autoregressive pass.

The text-to-music path implemented (the official one without the LM, `acestep/core/generation/handler`):

1. **Prompt layout.** `# Instruction\nFill the audio semantic mask based on the given conditions:\n\n#
   Caption\n<description>\n\n# Metas\n- bpm: …\n- timesignature: …\n- keyscale: …\n- duration: N
   seconds\n<|endoftext|>\n` for the text encoder (≤ 256 tokens); `# Languages\nunknown\n\n#
   Lyric\n[Instrumental]<|endoftext|>` for the lyric encoder (≤ 2048 tokens).
2. **Tokenizer** (`model::tokenizer`): Qwen2 byte-level BPE — added tokens cut out first, NFC, the
   Qwen2 pre-tokenizer pattern written out as a scanner (its `\s+(?!\S)` lookahead has no `regex`
   equivalent), GPT-2 byte alphabet, ranked merges; the post-processor appends `<|endoftext|>`.
3. **Text encoder** (`qwen3`): Qwen3-Embedding-0.6B, 28 causal layers (GQA 16/8 heads of 128,
   per-head RMSNorm on q/k, RoPE θ = 10⁶, SiLU MLP), `last_hidden_state`. The lyrics only go through
   its embedding table (rows read from the file, the 151 669-row table is never loaded).
4. **Condition encoder** (`dit`): text → linear 1024→2048; lyric embeddings → linear + 8 bidirectional
   layers (even layers within ±128 positions, odd ones full) + RMSNorm; the timbre encoder (4 layers)
   over the first 750 frames of the silence latent (no reference audio), its first frame's state;
   packed as lyrics · timbre · text.
5. **DiT**: input = [silence latent ‖ chunk mask of ones ‖ noisy latent] (192 channels), patched in
   pairs of frames; t and t − r (r = t) through sinusoidal embeddings (×1000, 256 wide) and MLPs giving
   six AdaLN parameters per layer; 24 layers of AdaLN → RoPE self-attention (sliding window 128 on
   even layers) → gated residual, RMSNorm → cross-attention (q/k RMSNorm, no positions, no mask) →
   residual, AdaLN → SiLU MLP → gated residual; AdaLN out, transposed patch convolution back to 64 × T.
   Cross-attention keys and values are computed once per clip.
6. **Sampler**: shift-3 turbo schedule t = 1, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3; Euler steps
   x ← x − v·Δt, and from t = 0.3 straight to x₀ = x − v·t. Seeded Gaussian noise (SplitMix64 +
   Box–Muller: deterministic per seed, not torch's generator).
7. **VAE** (`vae`): Oobleck decoder — 7-tap conv to 2048 channels, five blocks (Snake, transposed
   conv ×10, ×6, ×4, ×4, ×2 halving channels, three residual units with dilations 1/3/9), Snake, 7-tap
   conv to stereo; weight norm folded, Snake α/β exponentiated. Convolutions are shifted matrix
   products (no im2col buffers); decoding runs in 128-frame windows with 16 frames of context either
   side (the receptive field is ≈ ±9 frames), so memory is bounded and the result equals one pass.
8. Trimmed to the requested length; scaled down only if the peak exceeds full scale (as the
   reference); the engine then normalises the peak to −1 dBFS (the official default).

Hostile input: configs are checked against this architecture with size caps; safetensors headers are
checked before any read (offsets, sizes, dtypes, element caps) and values must be finite; the silence
latent is read from its zip without unpickling (shape found in the pickle, size and byte order
checked); the tokenizer refuses any other configuration; requests are validated (`Request::check`).

## Oracle: agreement with the official reference

`tools/oracle.py` runs the official reference (the modeling file shipped with the weights,
transformers' `Qwen3Model`, diffusers' `AutoencoderOobleck`) on the CPU in float32 with one prompt
(124 BPM, A minor, 24 s, instrumental) and writes every intermediate tensor and the noise; the
env-gated `live_oracle` test feeds each stage the reference's own inputs. Result (max |Δ| relative to
the reference's peak):

| Stage | CPU, f32 | Metal (M3 Max), bf16-stored weights |
|---|---|---|
| tokens (prompt, lyrics, 20 hostile cases, truncation) | identical | identical |
| text encoder layer 0 / output | 1.6e-6 / 3.0e-6 | 4.7e-7 / 3.0e-6 |
| lyric encoder, timbre, packed conditions, condition embedding | ≤ 1.5e-6 | ≤ 3.2e-7 |
| timestep embeddings, patch input, DiT layer 0 / 1 / 24 (pass 1) | ≤ 9.4e-7, 1.9e-6 / 5.5e-6 / 3.0e-5 | ≤ 1.4e-6, 1.4e-6 / 2.2e-6 / 3.8e-5 |
| velocity, each of the 8 passes from the reference's x_t | 4.1e-4 (pass 1), 4.0e-5 … 2.0e-6 | 5.2e-4 (pass 1), 6.4e-5 … 1.9e-6 |
| latents after the whole sampler from the reference's noise | 7.8e-4 (rms 2.2e-4) | 3.3e-4 (rms 8.5e-5) |
| VAE decode (random latents / the music) | 7.5e-6 / 1.1e-5 | 7.6e-6 / 1.1e-5 |

For scale: the reference's own `generate_audio` and a replica of its loop differ by 6.9e-5 on the same
latents (float noise of threaded kernels). On the GPU the linear weights stay bf16 in memory (as
released) and are widened to f32 for each product — exactly, so the arithmetic is the reference's;
an earlier variant that multiplied in bf16 put the first velocity 26 % (max) off the reference.

## Speed and memory (Apple M3 Max, 36 GB, shared with five other build lanes, load average 60–180)

| Run | Music | Time | Real-time factor | Peak memory footprint |
|---|---|---|---|---|
| Metal, first run (shaders compiled) | 30 s | 25.1 s | 0.84 | 7.9 GB |
| Metal | 20 s | 12.8 s | 0.64 | 7.0 GB |
| Metal | 30 s | 14.3 s | 0.48 | 7.5 GB |
| CPU (f32 weights), oracle run | 24 s | 220 s | 9.2 | — |
| CPU (f32 weights), during a workspace build | 10 s | 177 s | 17.7 | 10.4 GB |

Per stage on Metal (30 s): description 0.5 s, DiT load 3.6–4.2 s, eight passes ≈ 3 s, VAE load and
decode ≈ 7 s. On the CPU one DiT pass over 24 s of music takes ≈ 15 s and the VAE ≈ 100 s (its
convolutions are matrix products per tap; a CPU-tuned decoder is future work).

## Samples

`acestep_generate` wrote, from the same build (seeds in the names' order 42, 7, 2026):
EDM 124 BPM A minor 30 s, lo-fi hip hop 85 BPM F major 20 s, cinematic orchestral 30 s. Measured
tempo (onset autocorrelation): 62 BPM (half of 124) and 85 BPM. Like the reference (whose own 24 s
render ends in 4 s of silence), pieces often end a few seconds before the requested length with a
fade.

## Limitations (what is left for the full port)

- **No LM planner**: no audio codes, caption rewriting or metadata completion; the 0.6/1.7/4 B
  planners (Qwen3 with a 5 Hz audio-code vocabulary, constrained decoding, CFG) are the next
  milestone. Without it the structure is the DiT's own and pieces often end a few seconds early.
- **Instrumental only**: lyrics, vocals and the 50 languages need the lyric path with real text (the
  encoder is implemented; the request does not take lyrics yet).
- **Turbo only**: `base`/`sft` (50 steps, CFG, APG guidance) and the XL models are refused; the
  official pipeline's newer DCW sampler correction (on by default there for turbo) is not ported
  (the oracle compares the plain Euler sampler of the modeling file).
- No reference audio (timbre), cover, repaint, extract or lego tasks; no LoRA.
- The noise generator is ours, so a seed does not reproduce the official pipeline's audio.
- The networks are loaded per clip (≈ 4 s on Metal); the UI panel ("Generate soundtrack…", job
  progress, cancel) is not built yet — the commands, CLI and MCP drive it.

## Tests

`cargo test -p filmcraft-acestep --features model`: request checks (hostile numbers, prompts, keys),
catalogue pins, config parsing (released values, other architectures, hostile values), the safetensors
reader (every byte flip and truncation, doctored headers, NaN/Inf in both load paths, table rows), the
silence-latent reader (the released pickle byte for byte, flips and truncations, wrong byte order or
channel count), the tokenizer (the pre-tokenizer pattern, merges, specials, template, truncation, NFC,
hostile text, other configurations refused), attention (every mask, chunked long sequences and the
Metal path against plain loops), rotary embeddings, the patch convolutions and the VAE's
convolutions against their definitions, windowed VAE decoding equal to one pass, a tiny random text
encoder (causal, deterministic), DiT (odd lengths, deterministic) and whole package (deterministic per
seed, every stage cancellable, damaged or mismatched packages refused).

Live, against the oracle:

```sh
# the reference, CPU f32 (pip installs only in a scratch venv; downloads only from huggingface.co/ACE-Step)
python -I crates/acestep/tools/oracle.py <Ace-Step1.5 checkout> <dir with the two .py reference files> oracle24.safetensors 24 42
# the port (package dir with the catalogue's flat file names)
FILMCRAFT_ACESTEP_DIR=<package> ACESTEP_ORACLE=oracle24.safetensors \
  cargo test --release -p filmcraft-acestep --features model live_oracle -- --nocapture --test-threads 1
# the same on the GPU (holds the machine-wide GPU lock)
ACESTEP_ORACLE_DEVICE=metal FILMCRAFT_ACESTEP_DIR=… ACESTEP_ORACLE=… cargo test --release …
# a clip
cargo run --release -p filmcraft-acestep --features model --example acestep_generate -- <package> out.wav 30 42 "calm lo-fi hip hop, soft piano" 85 "F major"
```

## References

- J. Gong, Y. Song, W. Zhao, S. Wang, S. Xu, J. Guo, "ACE-Step 1.5: Pushing the Boundaries of
  Open-Source Music Generation", arXiv:2602.00744 (2026); github.com/ace-step/ACE-Step-1.5 (MIT).
- Flow matching: Y. Lipman et al., ICLR 2023. Oobleck VAE: Stability AI, Stable Audio Open (the
  architecture; diffusers' implementation, Apache-2.0). Snake: L. Ziyin et al., NeurIPS 2020.
  Qwen3 / Qwen3-Embedding: Qwen team (Apache-2.0).
