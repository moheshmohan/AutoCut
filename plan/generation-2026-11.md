# Generation with local tools: plan

Requested 19 November 2026. Source reviewed: the user's `freefield-mac` layer (a local gateway
that launches MLX command-line tools) and its installed engines: MiniMax-H3-Image (int8 MLX
build), LTX-2.3 (q4 distilled), Stable Audio 3 small, rembg (U2-Net) and
Depth-Anything-V2-Small. The upstream Freefield project (MIT, ComfyUI apps for NVIDIA cards) was
read for ideas only; none of its code is in AutoCut.

Rules that stay in force: generation tools are separate installations outside the repository and
the app bundle; a model download needs the user's OK each time; one generation at a time under
the shared GPU lock; every action is an engine command, reachable from the panel, the CLI, the
control channel and MCP.

## Done

- [x] **GEN1: generate in the editor.** `generate.providers`, `generate.image`,
  `generate.video`, `generate.sound`, `generate.fillGap`, `generate.augment`,
  `generate.history`; the Generate panel; MCP long-job handling; the provider file
  (`<data dir>/generation/providers.json`); a two-step cut-out for picture tools with no
  transparent output. Tested live on this Mac (timings in `docs/generation.md` §3).
- [x] **Assets for Render Anything (RAEA), first step.** `generate.image` with
  `"import": false` and `"dir": "<build>/fx/assets"` writes a background plate or a cut-out
  character straight into a composition's assets (`docs/generation.md` §5,
  `docs/raea-workflow.md` "Generated assets").

## Facts that shape the next steps

- MiniMax-H3-Image is not a light model: 33.1 B parameters, about 69 GB of weights plus a
  67 GB text encoder in full precision; the copy on this Mac is a 46 GB int8 build. A picture
  takes 37 to 45 s; an edit from a reference picture takes about 4 minutes.
- Its licence (MiniMax-H3-Image Community License) excludes use in the EU, the UK, South Korea
  and the USA, needs written permission above US$1M yearly revenue or for a hosted service, and
  asks for the model's name to be shown. The panel shows it from the provider's `attribution`.
- H3 has no transparent output; cut-outs go through the background remover and can keep a halo.
- The clip model makes at most ten seconds at a small size.

## To explore and implement

Each item is explored first with a throwaway project and real tools, then built only if the
result is good enough to use in a video.

- [ ] **GEN2: scene asset list for RAEA.** A composition folder carries `assets.json`
  (name, prompt, seed, size, transparent, tool). `generate.assets {manifest}` makes the files
  that are missing and leaves the others, so a scene can be rebuilt from its sources.
  Acceptance: deleting one asset and rerunning remakes exactly that file with the same seed.
- [ ] **GEN3: consistent character across scenes.** Use H3's reference pictures (up to 16) to
  keep one mascot or product the same in several RAEA scenes: a reference sheet made once,
  then each pose generated from it. Measure: how many of ten poses are usable without retouch.
- [ ] **GEN4: moving background plates.** Animate a generated or real still with the clip tool
  (`generate.video` from `image`) and use it under RAEA's text and graphics. Check that the
  loop point and the small source size hold up after `media.upscale`.
- [ ] **GEN5: depth for parallax.** Freefield has Depth-Anything-V2-Small; the engine already
  has depth code (`crates/engine/src/depth.rs`). Find out what that covers before adding a
  tool, then try a 2.5D push-in on a still inside a RAEA scene.
- [ ] **GEN6: sound for a scene.** `generate.sound` sized to a RAEA scene (ambience, a soft
  swish at a transition), placed under the imported render. Follows the per-video sound style.
- [ ] **GEN7: a lighter picture tool as a second provider.** `h3-image-mlx` can also run
  Qwen-Image-2.1-turbo, which has transparent output and fewer steps. Needs a model download
  (ask first) and a licence check. Would make cut-outs one step and edits faster.
- [ ] **GEN8: variations.** Generate several seeds for one prompt and choose in the panel
  (a small grid with Place on each), the part of Freefield's gallery that fits an editor.
- [ ] **GEN9: tool progress.** Let a tool report progress on a line of its output
  (`progress 0.42`) so the bar shows real progress.
- [ ] **GEN10: Freefield apps that need more than one frame.** Character or motion swap,
  relight and video-to-video. On this Mac only if an MLX tool exists for them; the ComfyUI
  apps target NVIDIA cards. Review what is installable before promising any of it.

## Not planned

- Bundling any model or Python runtime with the app.
- A command that writes the provider file (what may run is decided by who can edit the file).
- Freefield's web interface inside the app; the Generate panel covers manual use.
