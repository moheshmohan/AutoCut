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

## Shipped (20 November 2026)

- The first full `cargo xtask ci` on the branch (GEN1 with the upstream ports AV37 to AV49) failed
  one test, `wasm_clock_tests::no_std_instant_in_code_the_web_build_runs`: the generation runner
  named `std::time::Instant` in two places. Both now use `web_time::Instant`, folded into the
  GEN1 commit (`c0e6e6e0`) before anything was pushed.
- The full gate then passed: formatting, Clippy with warnings as errors, 6,388 tests (none
  failed, 89 ignored), layering, asset attributions and the 57 WebAssembly crate checks. The
  WebAssembly check prints one dead-code warning for the runner's `args` field, which the web
  build never reads; it does not fail the gate.
- Pushed to `autocut` `main` and `feat/autocut` (`dc416ef8..835639da`).
- The app and the CLI were packaged from `835639da` and installed. The build before it is kept
  as `target/macos-package/AutoCut-before-GEN1-2026-11-20.app` and
  `target/macos-package/bin-before-GEN1-2026-11-20/`. The CLI lists 921 commands (914 before, plus
  the seven `generate.*`). The app reopened the EXEED project clean with the project file
  unchanged; the playhead came back at 0 and was set back with `playhead.set`.
- In the installed app: `generate.providers` lists the four tools of this Mac as installed, and
  the Generate panel shows its fourteen controls, the tool's credit and its licence line.
- `target/release/filmcraft-cli` is rebuilt by every `cargo test --release` in the main checkout
  (the gate included), so the CLI there before this install was already the gate's build, and no
  copy of the 15 November CLI was kept beyond `filmcraft-cli-before-AV35-2026-11-15`.

## Facts that shape the next steps

- MiniMax-H3-Image is not a light model: 33.1 B parameters, about 69 GB of weights plus a
  67 GB text encoder in full precision; the copy on this Mac is a 46 GB int8 build. A picture
  takes 37 to 45 s; an edit from a reference picture takes about 4 minutes.
- Its licence (MiniMax-H3-Image Community License) excludes use in the EU, the UK, South Korea
  and the USA, needs written permission above US$1M yearly revenue or for a hosted service, and
  asks for the model's name to be shown. The panel shows it from the provider's `attribution`.
- H3 has no transparent output; cut-outs go through the background remover and can keep a halo.
- The clip model makes at most ten seconds at a small size.

## Second round (GEN2, 20 November 2026)

Requested: "Do the remaining tasks, we need to add more features". One commit, GEN2; the item
numbers below are the ones this plan gave them.

- [x] **Scene asset list for RAEA** (was GEN2). `generate.assets {manifest}` reads an `assets.json`
  beside a composition and makes the files that are missing: pictures, cut-outs, clips, sounds.
  `dryRun`, `only`, `force`, `import`. Live: six assets in 6 min 36 s; with one picture deleted,
  exactly that one was made again (31 s) and the file was the same byte for byte.
- [x] **One character across scenes** (was GEN3). An entry's `references` names earlier entries,
  so a pose is made from the character's first picture. Live: a robot mascot and two poses made
  from it; both poses are plainly the same character. Measured on two poses, not the ten the
  first plan asked for. The cut-out pose kept white between the raised arms and the head.
- [x] **Moving background plates** (was GEN4). A `video` entry with `image` naming a plate. Live:
  a 3 s plate at 448×256. Not checked: how the loop point and the small size hold up after
  `media.upscale` under a real composition.
- [x] **Depth for parallax** (was GEN5). The engine already had it (`clip.photo3d`, Depth Anything
  V2 Small, installed). `generate.image … "place": true, "move": "in"` applies it to the placed
  picture. Live: a push-in on a generated picture.
- [x] **Sound for a scene** (was GEN6). `sfx` and `music` entries in an asset list, and
  `generate.sound {"clip": id}`, which sizes the sound to a clip and starts it where the clip
  starts. Live: 3 s of wind for a 3 s clip.
- [x] **Variations** (was GEN8). `variations` 1 to 4 on pictures, clips and sounds; the first is
  placed, `generate.pick {item}` swaps another in as one undo step; a Use button in the panel.
  Live: two pictures in 58 s, the second swapped in.
- [x] **Tool progress** (was GEN9). A tool that prints `progress 0.42`, `progress 42%` or
  `progress 3/8` drives the bar. None of the three tested tools prints such a line, so their
  bars still follow their usual time.
- [x] **Gaps filled better** (new). The clip tool turned out to do more than the first round
  used: `generate.fillGap` `from: "both"` starts on the frame before the gap and ends on the
  frame after it (42 s for 2 s); `from: "motion"` gives the tool the last 33 frames before the
  gap as a small video, so the movement carries on (68 s); `generate.extend` does the same past
  the end of a clip (64 s). Live: all three on a lighthouse clip; the frames across each cut
  match. Not tested: a gap between two very different pictures.
- [x] **What a waited MCP call answers.** Found by the live run: `command_run` with `wait: true`
  did not say what a generation made, and an asset list with nothing left to make came back as
  a failure. Both fixed, with a test over MCP.

## Shipped, second round (20 November 2026)

- Before the commit the lane checks passed (formatting, Clippy on the engine, the interface,
  automation and both binaries, their tests, layering, assets, WebAssembly). The dead-code
  warning the WebAssembly check printed for the runner's `args` field is gone.
- The full `cargo xtask ci` on `57d38069` passed at the first run: formatting, Clippy with
  warnings as errors, 6,397 tests (none failed, 89 ignored), layering, asset attributions and
  the 57 WebAssembly crate checks.
- Pushed to `autocut` `main` and `feat/autocut` (`09993f7d..57d38069`); the fork's "AutoCut
  checks" passed on both.
- The app and the CLI were packaged from `57d38069` and installed. The build before it (GEN1)
  is kept as `target/macos-package/AutoCut-before-GEN2-2026-11-20.app` and
  `target/macos-package/bin-before-GEN2-2026-11-20/` (with the CLI that was in `target/release`
  as `filmcraft-cli-from-target-release`). The CLI lists 924 commands (921 before, plus
  `generate.extend`, `generate.assets` and `generate.pick`). The app reopened the EXEED project
  clean with the project file unchanged; the playhead was set back with `playhead.set`.
- In the installed app: `generate.providers` shows `bothEnds` and `carriesOn` for the clip tool,
  and the Generate panel shows the new controls in each mode (How many to choose from, the
  depth movement, Carry on, For the selected clip, what a gap is filled from).
- The provider file of this Mac (`<data dir>/generation/providers.json`) was extended for the
  clip tool (`lastImage`, `extendCommand`, `carryFrames`, `extendMultipleOf`); the file as it
  was is beside it as `providers.before-GEN2-2026-11-20.json`.

## Open, each waiting on a decision of the user

- [ ] **A lighter picture tool** (was GEN7). `h3-image-mlx` can run Qwen-Image-2.1-turbo, which
  has transparent output. Its weights are tagged for non-commercial research use, the download
  is about 25 GB, and nothing is measured on an M3 Max. Recommendation: do not take it for
  videos that are published; H3 with the background remover stays.
- [ ] **Restyle and pose-driven character swap in video** (part of GEN10). The clip tool's
  `ic-lora` command with the Union-Control adapter (654 MB, LTX-2 Community License) regenerates
  a clip from a control video (edges, depth or pose) and a picture of the new character. Edges
  need no extra model; pose needs a pose estimator. Would become `generate.restyle`. The tool's
  own tests are on its q8 weights; this Mac has q4, where a small adapter may lose strength.
- [ ] **Relight a clip** (part of GEN10). A relight adapter (327 MB, gated: needs a Hugging Face
  login and accepted terms). Relighting a still works today as an edit with MiniMax-H3-Image.
- [ ] **Lip dub** (part of GEN10). An adapter of 2.5 GB, gated, marked experimental by the tool.
- Freefield's own `character-swap` app is a two-photo picture edit, not video; it is what
  `generate.image` with two `references` already does.

## Not planned

- Bundling any model or Python runtime with the app.
- A command that writes the provider file (what may run is decided by who can edit the file).
- Freefield's web interface inside the app; the Generate panel covers manual use.
