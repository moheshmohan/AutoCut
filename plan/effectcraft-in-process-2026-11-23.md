# EffectCraft in-process (PX4): study and decisions, 23 November 2026

A read-only study of EffectCraft (https://github.com/storytold/effectcraft) for decision D3 in
[video-automation-ingest-2026-11-22.md](video-automation-ingest-2026-11-22.md). Nothing was built
in this repository. Paths are in the EffectCraft source unless marked AutoCut.

## Decided (23 November 2026)

| | Decision |
|---|---|
| Fonts | Accept the duplicate. EffectCraft's text crate compiles in seven fonts with `include_bytes!` and no feature switch (`crates/text/src/fonts.rs:21-27`). They are byte-identical (SHA-256) to fonts AutoCut already ships and credits. No font file enters this repository; the binary carries a second copy (about 2.7 MB). `CRAFT_FONTS` is fed in through EffectCraft's `FontSource` hook. |
| App build | The `effectcraft` feature is off by default in the workspace (tests and `cargo xtask wasm` unchanged) and gets its own gate step; `packaging/macos/package.sh` builds the AutoCut app with it on. |

## Pin and verdict

- Pin: `1738360417403d8780aa3a4949bd6f6a49b6588d` (main, 2026-11-23), workspace version 0.24.1,
  edition 2024, rust 1.95. The repository moves fast (309 commits in the 7 days before the pin,
  82 in the crates we link), so the pin moves only deliberately, with the adapter's tests as the gate.
- **Go** for six crates: `effectcraft-render`, `-effects`, `-gpu`, `-expr`, `-text`, `-path`, with
  `time`, `geom`, `color`, `raster`, `keyframe`, `project`, `segment`, `track`, `model` under them.
  They type-check as git dependencies against versions seeded from AutoCut's `Cargo.lock`.
- **No-go** for `effectcraft-engine`, `-media`, `-host`, `-export`, `-interchange`: they depend on
  upstream FilmCraft's `filmcraft-*` 0.20.1 crates from git (EffectCraft `Cargo.toml:111-119`),
  which collide by name with ours, plus symphonia (MPL-2.0) and vectorcraft. Never let them in.

## Dependencies

- Same versions on both sides: wgpu and naga 30.0.1, image, serde, rayon, skrifa 0.44, harfrust
  0.12, kurbo 0.13.1, chrono, half, bytemuck, thiserror, pollster.
- No egui, eframe, winit, rfd, cpal or craft-ui in the closure; no glam, fontdb, rustybuzz,
  swash, cosmic-text or candle.
- craft-libs crates that come along (all MIT OR Apache-2.0, no `unsafe`, no unwrap outside tests):
  `craft-cms`, `craft-exr`, `craft-ecproj-format`, `craft-mesh`, `craft-mime`, `craft-codepage`,
  on three craft-libs revisions.
- Licences across the 329 packages: MIT, Apache-2.0, Zlib, BSD-2/3, ISC, Unicode-3.0,
  Unlicense OR MIT, Apache-2.0 OR BSL-1.0, 0BSD (`r-efi` adds an LGPL option, UEFI targets
  only). No GPL, AGPL, NC or ND.
- About 63 new `Cargo.lock` entries even with the feature off (mostly boa, the JavaScript
  engine for expressions); six crates in two versions, all from boa (hashbrown 0.14, num-bigint
  0.5, phf 0.14 and siblings). Every fresh checkout fetches EffectCraft (~97 MB) and craft-libs
  (~164 MB) once.
- EffectCraft's wgpu dependency turns on wgpu's default features plus
  `fragile-send-sync-non-atomic-wasm` (`Cargo.toml:122`); ours uses `default-features = false`.
  With the feature on, that unifies across the build, so the feature stays native-only.
- Its opt-level 3 profile overrides do not carry over to a dependent: add them to our root
  manifest for raster, render, gpu and effects.

## API

- `.ectemplate`: a stored ZIP (`crates/engine/src/cmd/essential.rs:10-12, 848-898`) with
  `manifest.json` (format `effectcraft-template`, version 1, comp, controls, media, size, rate,
  duration), `project.ecproj`, `media/<item>/<file>` and `poster.png`. Readable with AutoCut's
  `filmcraft_templates::zip::read` (AutoCut `crates/templates/src/zip.rs:122`).
- Filling it happens inside EffectCraft's engine, which we don't link. The adapter does it from
  public functions (about 150 lines):
  - `Project::from_json_repaired(&str) -> Result<(Project, Vec<String>), ProjectError>`
    (`crates/project/src/item.rs:401`);
  - `Value::coerce_json(&self, &serde_json::Value) -> Option<Value>` (`crates/keyframe/src/value.rs:434`);
  - `Override { control, value, keys }`, `with_overrides(&Project, ItemId, &[Override]) -> Option<Project>`
    (`crates/project/src/essential.rs:674, 686`);
  - `Renderer::new(&Project, &dyn FootageSource, RenderOpts)`, `comp_frame(ItemId, Tick) -> Image`,
    `comp_frame_scoped(...) -> Result<Image, E>` (`crates/render/src/pipeline.rs:168, 609, 620`);
    set `expr = Some(&effectcraft_expr::Expressions)` (`crates/render/src/lib.rs:69-110`);
  - `RenderOpts` (`crates/render/src/opts.rs:6`): scale, motion_blur, draft, backend (Cpu/Gpu/Auto), roi.
  We lose the engine's `upgrade_effects` migration, so old templates may differ; a parity test
  against `effectcraft-cli render --template … --values … --out … [--gpu]`
  (`apps/effectcraft-cli/src/main.rs:489`) covers it.
- `FootageSource` (`crates/render/src/footage.rs:20`, `Send + Sync`, called from rayon workers):
  only `frame(item, &Footage, t) -> Option<Arc<Image>>` is required (about 24 more defaulted).
  `Image` is `{width, height, data: Vec<[f32; 4]>}` (`crates/raster/src/lib.rs:78-87`),
  premultiplied, sRGB/Rec.709-encoded (not linear); AutoCut works in premultiplied linear, so the
  adapter converts both ways. Time is EffectCraft's `Tick`, the same 254 016 000 000/s base, in
  the footage's own time. We decode all footage, including media embedded in templates.
- GPU: `Gpu::new(&adapter, device, queue)` (`crates/gpu/src/lib.rs:163`) accepts our device;
  needs compute, not the GL backend, 7+ storage buffers per stage, f32 and Rgba8 storage
  textures; on a shared device forward faults through `device_failure_notifier()`
  (`crates/gpu/src/context.rs:831-866, 1076`).
- Fonts: `add_font_data(Vec<u8>)` (`crates/text/src/fonts.rs:415`) copies; a `FontSource`
  (:68) returning `FaceInfo` with `FaceData::Static(&'static [u8])` (:36-65, built with
  `sfnt::read_faces_bytes`, `crates/text/src/sfnt.rs:90`) passed to `add_source` (:398) is zero
  copy and suits `CRAFT_FONTS` (empty is fine). The first unknown family scans system fonts on the
  calling thread (:641-642), so tests use bundled fonts only.

## Safety

- No `unsafe`; `#![deny(clippy::unwrap_used, …)]` in every crate. Indexing is the remaining panic
  risk (rough counts: effects ~1690, track ~1050, render ~400, raster ~330, gpu ~190, path ~170,
  text ~100). Every adapter entry point (template parse, each frame, footage callbacks) runs under
  `catch_unwind`.
- Out-of-memory can't be caught: project loading clamps to 30 000 px (`crates/project/src/limits.rs:17`),
  but one 30 000² f32 frame is 14.4 GB and `Image::new` allocates unchecked. The adapter caps
  comp size × scale.
- Expressions are bounded (1M loop iterations, recursion 400, nesting 16, 5 s per evaluation,
  `crates/expr/src/lib.rs:45-56`), but boa itself can panic.
- Threads: `gpu-readback-deadlines` per GPU context, font scans, a disk-cache writer if set, one
  boa runtime per thread. Global state: the font database, memory accountant statics, text caches.

## Integration plan

1. An L3 crate `crates/effectcraft` (`filmcraft-effectcraft`), feature `effectcraft` off by
   default, git dependencies pinned to the SHA above under a `cfg(not(target_arch = "wasm32"))`
   table. Without the feature it compiles a stub that answers "not built with EffectCraft".
   Depends on `filmcraft-frame`, `-time`, `-color` and wgpu; a `HostFrames` trait the engine
   (L4) implements, so there are no upward edges. Add its row to `cargo xtask layers`.
2. Engine commands: `effectcraft.templateInfo {template}`, then `effectcraft.renderTemplate
   {template, values, size?, fps?, out?, place?}` (validate values with `coerce_json`, apply
   overrides, cap the size, render with `comp_frame_scoped`, CPU fallback), writing a PNG
   sequence or ProRes 4444 with alpha (AutoCut `crates/prores/src/encode.rs:22`) and reusing
   `motion.render`'s revision folder, probe, import and place (AutoCut
   `crates/engine/src/motion_graphics.rs`); also a `kind: "composition"` provider. Later
   `effectcraft.renderComp` and a live clip kind.
3. Gates: `cargo xtask ci` gains a step that runs clippy and tests with `--features effectcraft`
   on the adapter, engine and CLI. Tests with the feature off cover validation and colour
   helpers; with it on: golden renders of a template built in code, hostile inputs (truncated or
   bit-flipped template, oversized comp, endless expression, deep nesting, missing media, NaN
   values), GPU-vs-CPU parity skipped without a GPU.
4. Stage 1 (EffectCraft's own CLI and stdio MCP) stays a test oracle, not a product path. Its
   control port is opt-in (`--control` / `EFFECTCRAFT_CONTROL_PORT`), so there is no clash with
   AutoCut's 9877.

## Not verified

A full build or any rendered frame; our real workspace resolving the same way (feature
unification with our crates); a wasm32 check of the six crates; frame speed; whether
`with_overrides` on the template comp matches the CLI's output bit for bit.
