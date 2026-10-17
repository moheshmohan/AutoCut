# AutoCut improvement plan — October 2026

> Research-based roadmap for the four asked-for directions: **easy editing templates**, **quick
> Instagram/YouTube publishing**, **more transition effects**, and a **music generation engine** —
> plus the connective tissue they need. Written 2026-10-17 against `autocut/main` @ `c469d440`
> (2012 commits, 45+ crates, ~2,520 tests), after a full codebase inventory and a survey of the
> open-source editing landscape (Kdenlive, Shotcut, OpenShot 4.0, Olive, Blender VSE, video-use,
> OpenStoryline, CapCut/Resolve/Descript/Opus Clip as expectation-setters) and of music-generation
> and shader-transition ecosystems. All sources are listed at the end. This file lives in the
> gitignored `plan/` directory; promote items into `docs/roadmap.md` / `docs/gaps.md` as they are
> scheduled.

## Where we stand (the honest snapshot)

AutoCut is already unusually strong in the places the market is moving:

- **Agent-native**: control channel + MCP server (`crates/automation/src/server.rs`), the
  supervisor-is-the-client model, intent classification → deterministic planners → engine
  commands. Nothing else in open source has this; `video-use` (29k★) and OpenStoryline build
  entire products around a weaker version of it.
- **Transcript-first editing exists** (`crates/edit/src/transcript.rs`): the Descript-style
  "delete a sentence, footage disappears" flow that every 2026 roadmap wants — we already ship it.
- **Transitions**: 84 modern + 21 legacy, data-driven catalog (`crates/project/src/vtransition.rs`)
  with CPU reference implementations (`crates/render/src/transitions/`).
- **Procedural audio + music beds** (`audio.generate`, `crates/audio-dsp/src/synth.rs`), Remix
  beat-consistent re-timing, `media.beats`, the whole showcase pipeline that composes with them.
- **Local ML on candle**: Whisper STT, Kokoro TTS, VoxCPM voice cloning on Metal.

The gaps, mapped to the four asks:

| Ask | Have today | Missing |
|---|---|---|
| Easy editing templates | `.fcgt` graphics templates, effect/export presets, the auto-generated Showcase | Whole-sequence style templates with media slots (vlog / intro / promo / reel starters), a template browser, template authoring & sharing |
| Quick IG/YT publishing | Export presets (incl. YouTube 1080p/4K, Social Vertical 9:16), easy-export intents, chapters/SRT/thumbnail sidecars, Google Photos upload | Any actual upload to YouTube/Instagram/TikTok; OAuth, token storage, upload jobs, platform constraint validation |
| More transitions | 105 shipped, incl. Morph Cut on real optical flow | The GL-Transitions-class shader catalog (zooms, warps, glitch, particles), GPU path beyond dissolves, beat-synced timing, audio coupling |
| Music generation engine | Oscillator-based mood beds, deterministic, seeds | Real instruments (soundfont/MIDI path), exact-length fitting, stems, taxonomy/search, and a license-clean neural text-to-music backend |

**Strategic frame from the research:** 2025–26 split the market in two — incumbents shipping craft
parity with almost no AI (Kdenlive 26.08, Shotcut 26.9, OpenShot 4.0's exception proves the rule),
and an AI-native layer living *outside* real NLEs on ffmpeg+EDL. CapCut's paywall backlash and
Resolve's AI-behind-$295 leave the **local, free, AI-assisted, social-first** quadrant empty. That
is precisely our quadrant: pure-Rust, on-device ML, MCP-native, no account required. Every
workstream below is chosen to occupy it.

---

## Workstream A — Easy editing templates (CapCut-grade, file-based, agent-drivable)

### A1. Template format and engine

A template is an **ordinary project fragment plus slot metadata** — not a new timeline model:

- **Format**: `.fcvideo-template/` directory (or `.zip` for sharing): the sequence JSON (same
  schema as `.fcproj` items), a `template.json` manifest, optional bundled openly-licensed assets
  (fonts via craft-fonts references, looks, generated audio), an `ATTRIBUTION`-style manifest with
  hashes and licences per repo rules (§1: every asset needs a sidecar; `cargo xtask assets` gates).
- **Slots**: typed placeholders — `video`, `photo`, `text`, `logo`, `music` — with order, count,
  constraints (min/max duration, orientation, aspect). Support **slot dedup** (one photo fills
  several slots, VN-style) and **multi-variant slots** (same asset re-framed 16:9 and 9:16).
- **Auto-fit policy per slot** (the iMovie/CapCut contract): media conforms to the slot, never the
  reverse — trim if long, time-stretch/loop if short (we already have Remix for beat-true
  stretching); the user picks *which* portion of the source is used. Effects, transitions and text
  stay attached to the slot, not the media.
- **Beat grid**: templates ship with the music's beats precomputed as Tick markers (from
  `media.beats`); cuts snap to them. If the user swaps the track, re-analyze and quantize
  (onset-strength → tempo → DP peak-picking; implement from the published Ellis/librosa algorithm,
  ISC-licensed reference is safe to read).
- **Aspect variants**: one template, N layouts (16:9 / 9:16 / 1:1 / 4:5) with per-variant position
  hints — the Canva Magic-Resize model. Publishing (Workstream B) picks the variant per platform.
- **Look + style presets**: a template carries its grade (Lumetri preset / LUT), caption style
  token (font/colours/animation), and export preset, so the whole "brand" travels together.

Everything executes as ordinary engine commands, so templates are **applyable by the local planner,
by the MCP supervisor, and by the UI** — the same three faces every other feature has.

### A2. Built-in starter templates (the content, not just the machinery)

Ship ~20 originals (drawn/derived, licence-clean, all original work registered in
`ATTRIBUTION.md`):

- **Reels/Shorts**: talking-head + auto-captions starter; photo-beat slideshow (cuts on beats,
  7–15 slots); product promo 15 s; travel vlog intro/outro; "top 5" listicle; before/after reveal.
- **YouTube**: channel intro 5 s; lower-third + chapter-card kit; end-card with subscribe
  placeholder; tutorial layout (screen + webcam PiP); podcast/long-form conversation starter.
- **Vertical-from-horizontal**: the auto-reframe starter (see E3).

The Showcase panel's review UX (include/merge/relabel) is the precedent for the template apply
flow: pick template → slot picker (media browser / Google Photos recent) → preview → build →
tweak.

### A3. Authoring and sharing

- "Save as Template" command: marks placeholders, strips personal media (replaces with neutral
  generated placeholders — `crates/media` already generates procedural footage), validates slots,
  writes the package. Mirrors CapCut's creator flow without the marketplace.
- Sharing is **plain files**: git repo / URL / QR code (VN model). No server, no account — a
  differentiator, not a compromise.
- **OpenTimelineIO export** of template timelines (Apache-2.0 library; Blender is adding OTIO in
  GSoC 2026 — the interchange wave is starting). OTIO has no slot concept; slots live in our
  manifest layered on top.

### A4. Template-adjacent automation (cheap wins on existing machinery)

- "Save this edit as a recipe": the Showcase planner and AutoCut intents are already deterministic
  programs over engine commands; let users persist a *parameterized* intent chain (e.g. "remove
  silence → captions in my style → my intro/outro → vertical export") as a one-click recipe — the
  OpenStoryline "Style Skills" idea, but native. This is mostly UI + serialization over
  `planner.rs`.

**Estimate:** format+engine 60–100 agent-hours, 20 starter templates 40–80 h, authoring/sharing
30–50 h, recipes 20–40 h.

---

## Workstream B — Quick publishing to YouTube and Instagram

### B1. Platform adapter architecture

One `publish` engine job family (`publish.*` commands, MCP tools, CLI) over per-platform adapters,
following the Google Photos upload precedent (resumable, progress/cancel via `long_job.rs`,
verified against a fake platform in tests):

- `publish.connect` (OAuth), `publish.accounts`, `publish.revoke`,
- `publish.upload { destination, file, title, description, tags, privacy, thumbnail/cover,
  playlist, scheduleAt }` — validates constraints **before** uploading and fails with actionable
  errors (never crashes; hostile-parameter tests for every field).

### B2. Auth (the part desktop apps get wrong)

- OAuth 2.0 authorization-code + **PKCE (S256)**, system browser, **loopback redirect listener**
  on `http://127.0.0.1:{port}` with a "return to app" page — the OBS pattern.
- Refresh tokens in the **macOS Keychain** (and equivalents; `crates/platform` is the natural home
  for the FFI), never plaintext. Per-account revoke UI.
- **YouTube: BYO-client-ID mode.** `youtube.upload` is a sensitive scope; unverified apps cap at
  100 test users and 7-day refresh tokens. Ship without our own client ID: each user creates a GCP
  OAuth client (documented wizard), like `youtubeuploader` does. This is the only clean answer for
  an OSS app and avoids the verification-audit trap entirely.

### B3. YouTube adapter

- **Resumable upload protocol**: initiate → session URI (`Location` header) → chunked PUTs
  (256 KiB multiples) → 308/Range progress; resume after crash by persisting the session URI;
  exponential backoff on 5xx; session-expiry (404) → restart. Our Google Photos uploader already
  implements the same protocol shape (resumable, chunked, cancellable) — reuse the patterns.
- Metadata: title ≤100, description ≤5,000, tags, categoryId, privacy
  (private/unlisted/public), `publishAt` scheduling, madeForKids.
- Thumbnail via `thumbnails.set` (author at 1280×720 — Showcase already composes thumbnails).
- **Shorts**: no special endpoint — vertical/square + ≤3 min auto-classifies; the upload form
  should detect "this looks like a Short" and say so.
- **Quota UX**: since Dec 2025 the practical ceiling is **100 `videos.insert` calls/day** per
  project (plus 10,000 generic units). Surface remaining-quota errors as first-class messages.

### B4. Instagram adapter

Two API surfaces, and the choice matters for a local-first desktop app:

- **Instagram API with Instagram Login** (graph.instagram.com, the recommended modern path):
  container → poll (`status_code` once/min ≤5 min; containers expire in 24 h) → `media_publish`.
  **Constraint: `video_url` must be publicly fetchable — Meta pulls the file.**
- **Graph API with Facebook Login**: supports **resumable direct upload**
  (`rupload.facebook.com`), but requires a professional account linked to a FB Page, and app
  review + business verification for production.

Plan: implement **Facebook-Login + rupload resumable** as the primary path (true local-first
upload), with the Instagram-Login container path as the fallback for accounts without a Page.
Document both honestly in the Connect dialog. Rate limits to surface: **100 published posts + 400
containers per rolling 24 h**. Specs to enforce pre-upload: MP4/MOV, H.264/HEVC closed-GOP 4:2:0,
AAC ≤48 kHz, 23–60 fps, 9:16 recommended (0.01:1–10:1 accepted), ≤1920 px wide, ≤25 Mbps, 3 s–15
min; cover via `cover_url` (JPEG ≤8 MB) or `thumb_offset`; caption ≤2,200 chars / 30 hashtags.

### B5. Destinations UI + one-click flow

- The Export mode **Destinations** column (`export_mode.rs`) grows real rows: Media File (today),
  YouTube, YouTube (Short), Instagram Reels, (later TikTok). Row = variant + metadata form +
  Connect state.
- **One-click from templates/recipes**: template apply → render → publish, per-platform variant
  selection (A1). "Publish" is just another command, so the MCP supervisor can run the whole chain.
- Pre-flight checklist: aspect/duration/size/codec per platform, loudness (−14 LUFS social),
  caption burn-in choice, cover frame — validated before any bytes leave the machine.
- Icons: **draw generic glyphs** (play-triangle-in-rounded-square, camera-outline). YouTube/Meta
  logos are trademarks — repo rule §1.8 admits only the ArtCraft marks.

### B6. TikTok (later)

Content Posting API with CHUNKED upload (direct) or PULL_FROM_URL; unaudited apps are forced to
SELF_ONLY private posts — same BYO-app posture as YouTube. Slot into the same adapter interface
when YouTube + Instagram are proven.

**Estimate:** adapter framework + auth 50–80 h; YouTube 40–70 h; Instagram 60–100 h (two
transports); destinations UI 30–50 h; TikTok 30–50 h.

---

## Workstream C — Transition effects: from 105 to a best-in-class catalog

### C1. Port GL Transitions to WGSL (the big, safe breadth win)

GL Transitions (MIT) is the de-facto interchange standard — 106 shaders, tiny contract:
`progress` + `getFromColor/getToColor` + params. Remotion, ffmpeg-gl-transition and three.js all
consume it, so its names are what users know.

- **Curate ~60** (the iconic set: crosswarp, CrossZoom, DreamyZoom, SimpleZoom, cube, BookFlip,
  InvertedPageCurl, Fold, GridFlip, Swirl, ripple, WaterDrop, luminance_melt, perlin,
  displacement, directionalwarp, pixelize, Mosaic, hexagonalize, chessboard, randomsquares,
  PolkaDotsCurtain, crosshatch, pinwheel, angular, circleopen, StarWipe, kaleidoscope,
  GlitchDisplace, GlitchMemories, TVStatic, FilmBurn, Doorway, StereoViewer, Bounce, butterfly
  wave, polar…).
- **Hand-port to WGSL** (naga's GLSL frontend is a poor fit for WebGL1-era syntax; a curated
  hand-port with golden-image tests is lower risk). Porting guide gotchas: `textureSampleLevel(…,
  0)` inside non-uniform control flow; `atan2`; float `mod` as `x - y*floor(x/y)`; explicit f32
  literals; uniform-buffer alignment. Each ported shader keeps its upstream `// Author:/License:`
  header, gets an `.attribution` sidecar + `ATTRIBUTION.md` row (MIT; adaptation with attribution
  is exactly what MIT permits).
- **Host owns time**: easing (Penner curves + cubic-Bézier, as data) applied to `progress` on the
  CPU at exact Tick time → preview and export are bit-identical, per repo timing rules.
- **Invariants as tests**: `progress=0` ≡ frame A, `progress=1` ≡ frame B (slideshow safety);
  params/ratio fuzzed (hostile-input rule); WGSL validated in a registry that errors, never
  panics.
- This slots straight into the existing data-driven catalog (`vtransition.rs` + a new
  `crates/render/src/transitions/shader.rs` GPU path alongside the CPU reference oracle), and
  publishing the WGSL set back as a standalone MIT crate would be a first-mover community win —
  **no WGSL transition collection exists yet**.

### C2. Generic engines (hundreds of variants for two shaders)

- **Luma-mask wipe engine**: one shader + procedural grayscale masks (radial, linear, barn-door,
  blinds, star, clock, noise, Worley) × feather radius = the entire MLT luma-wipe family **without
  touching GPL luma image files** — generate every mask from scratch. Mask blur gives soft edges.
- **Blend-mode dissolve engine**: any of the 27 existing GPU blend modes as the transition kernel
  (addition, screen, overlay, difference…) — the frei0r insight, nearly free for us.

### C3. The flagship category: flow-based "AI" transitions

We already own the hard part — real Lucas–Kanade optical flow (`crates/render/src/flow.rs`) used
by Morph Cut and optical-flow speed. Build on it:

- **Flow Morph** (CapCut "AI transition" core): bidirectional flow between A's last frame and B's
  first frame → synthesize in-between warps → dissolve at the ends. This is also the engine for
  stylized presets:
  - **Doppler Warp** — flow morph + radial motion streaks (velocity-scaled radial blur);
  - **Hype Ramp** — speed-ramp + blur + brightness pulse peaking at the swap;
  - **Glitch Flow** — flow morph + datamosh block displacement clustered at the cut;
  - **Split Bands** — horizontal bands sliding alternately, reassembling as B.
- **Content-Aware Wipe** (research item): seam-carving (Avidan–Shamir / Grundmann lineage,
  implement from papers) routes the wipe boundary around salient objects. Differentiator; no open
  editor has it.
- Morph Cut upgrade path: region/face tracking to gate the morph window (today it's flow-only).

### C4. Motion blur, beat-sync, audio coupling

- **Analytic motion blur** (8–16 taps along the pixel's zoom/pan trajectory in-shader) for
  zoom/whip transitions; temporal supersampling (K sub-times per shutter) where analytic smears
  look wrong (whip pan).
- **Beat snap**: `media.beats` → "snap transitions to beats" toggle quantizes midpoints to the
  nearest beat, durations to musical fractions; the Showcase already cuts on beats — generalize to
  user timelines.
- **Audio-side transitions as commands, not assets**: whoosh / riser / downlift / duck synthesized
  by the existing `audio-dsp` synth (band-passed noise sweeps + envelopes), sample-accurate on the
  audio graph, optionally attached to any visual transition. Zero new licences, and it's the
  CapCut feel.

### C5. Catalog organization

Ship as new folders in the data-driven catalog (names users recognize from GL Transitions/xfade):
Zooms & Camera (12), 3D & Spatial (11), Light & Film (9), Distortion (14), Glitch & Datamosh
(10), Particles & Patterns (8), plus the existing folders. Target: **~200 named transitions
within the year**, every one CPU-oracled and GPU-fast. Research report §7 carries the full
~100-entry catalog with per-item one-liners and provenance tags.

**Estimate:** GLT port + tests 60–100 h; generic engines 20–40 h; flow presets 60–120 h;
blur/beat/audio coupling 40–80 h.

**Do not**: Shadertoy code (CC BY-NC-SA — banned), MLT luma images (GPL), OBS plugin shaders
(GPL), stock light-leak/film-burn footage (proprietary) — all generated procedurally instead.

---

## Workstream D — Music generation engine

Two-tier hybrid, both tiers licence-clean. The already-shipped `audio.generate` synth (mood/key/
BPM/sections, seeds, deterministic) is tier 0 — extend it, don't replace it.

### D1. Tier 1 — Procedural soundtrack engine (ships first, always available)

The SmartSound/iMovie-trailer model, modernized:

- **Real instruments**: render MIDI through **rustysynth** (MIT, pure-Rust SF2 synth) with
  **GeneralUser GS** (commercial-use-OK for rendered music; verify current release from the
  author, ship with attribution sidecar per §1). Fall back to the existing oscillator voices when
  no soundfont is present.
- **Block-based composition**: sections (intro / verse / build / chorus / bridge / outro) authored
  as per-instrument patterns over a chord/energy map; an arranger searches the block graph for an
  arrangement that hits the **exact target duration with a musical ending** — the expired-patent
  SmartSound trick, deterministic and instant. This kills today's "pad it to N minutes" feel.
- **Stems**: drums / bass / melody / pad as separate layers → auto-ducking under dialogue, energy
  follows the scene (iMovie-trailer style), and agents can mute/rebalance via commands.
- **Taxonomy + search**: mood → genre → tempo → energy → use-case (the Epidemic Sound vocabulary)
  as metadata on every generated track; drives a "find music by feel" browser and gives the
  planner/MCP supervisor a selection vocabulary (`music.suggest { mood, duration, energy }`).
- **Beat grid out of the box**: generated tracks carry exact Tick beat maps → Workstream C's beat
  snap and the Showcase's beat cutting work on them for free.
- Deterministic seeds stay: same request, same track — reproducible projects, testable golden
  mixes.

### D2. Tier 2 — Neural text-to-music backend (optional download, like Kokoro/VoxCPM)

The licensing survey is brutal — almost everything famous is a dead end (MusicGen/MAGNeT/JASCO
weights CC-BY-NC; AudioLDM2 NC-SA; YuE2 weights NC; Stable Audio 3 & MiniMax revenue-gated
community licences; MERT NC). Exactly two families are clean **and** consumer-hardware sized:

- **ACE-Step 1.5 (MIT, code and weights)** — the anchor. LM planner + 2B DiT + VAE, 10 s–10 min,
  <4 GB VRAM class, first-class Apple Silicon (MPS/MLX; GGUF/Metal via acestep.cpp). A pure-Rust
  candle implementation already exists as proof (**ace-step-rs**, MIT — usable as reference and
  upstream contributions). Our Whisper/VoxCPM candle+Metal experience transfers directly; expect
  minutes-per-song on M-series — fine for a background job with progress and cancellation.
- **DiffRhythm (Apache-2.0)** — second backend; simpler stack (one DiT + VAE), full songs with
  vocals, macOS-supported.

Product shape: "Generate soundtrack…" in the Generate Audio panel: prompt/mood/duration/exact-fit
→ job renders → lands as a stem track + beat map, then everything tier-1 does (ducking, Remix
refit, beat cuts) applies. Generated audio is original output — no attribution burden beyond the
model card notice, but record model+seed in project metadata for reproducibility.

### D3. Adjacent audio intelligence (small, high-value)

- **BPM/beat analysis on user-imported music**: `bpm-analyzer` (MIT) or the librosa-style onset →
  autocorrelation → DP algorithm implemented from the papers (aubio is GPL — avoid). Extends
  `media.beats` accuracy.
- **Loudness targets per destination**: −14 LUFS social / −16 voice-forward / −23 broadcast as
  one-command normalizes (ebur128-rs is already the BS.1770 reference we match).
- **Stem separation** (later, optional): Demucs (code MIT; verify weight terms before bundling)
  as a cancellable job — "separate this song, duck the vocals".

**Estimate:** tier 1 (rustysynth + arranger + stems + taxonomy + UI) 80–140 h; tier 2 (candle
ACE-Step 1.5 port + model-download flow + job UX) 120–250 h; adjacent items 20–40 h.

---

## Workstream E — Connective tissue (small items that make A–D land)

1. **Whisper enabled in release builds (G19).** The single highest-leverage line item: auto
   captions, remove-silence/fillers, highlight reels, transcript editing and the Shorts factory
   all route through local STT, and it's off in release today. Resolve model-size/download UX
   (Kokoro's consent-gated download is the pattern) and turn it on. ~15–30 h.
2. **Caption embedding into exports** (608/708 in H.264/HEVC streams; sidecar SRT exists): needed
   for YouTube's caption upload and broadcast parity. 30–50 h.
3. **Auto-reframe / vertical factory**: tracked-subject 9:16 crop from 16:9 sources (Extend Edit
   + Motion keyframes exist; add subject tracking — point tracker first, tracker+flow later),
   plus a "long → shorts" recipe: transcript segmentation (highlight.rs already scores) → top-N
   vertical clips with captions → batch export → publish. This is the Opus-Clip factory, local.
   60–120 h.
4. **Bundled open "Elements" library** (Shotcut 26.8's move): a small set of CC0/original
   stickers, whoosh/impact/riser sounds (or generate them — our synth can), overlay transitions,
   animated text presets — all with attribution sidecars; gives templates (A2) raw material.
   20–40 h + curation.
5. **Adjustment clips + per-track auto-ducking** (Shotcut 26.9's crowd-pleasers; adjustment layers
   exist in the data model — surface them, add the track-duck toggle wired to
   `generateDucking`). 20–40 h.

**Explicitly deferred**: OpenFX/VST plugin hosting (the OSS world is converging on OpenFX, but
FFI plugin loading conflicts with our pure-Rust/no-unsafe stance outside `crates/platform`; our
WGSL-first path covers the need short-term — revisit as a platform-crate boundary decision
later), and any Remotion dependency (source-available, paid for 4+ person orgs — reference only).

---

## Sequencing

| Phase | Contents | Why this order |
|---|---|---|
| **1 (weeks 1–4)** | E1 whisper-on, E4 elements, C1 GLT port starts, D1 procedural engine core | Unblocks everything; D1+C1 are pure local wins with no platform risk |
| **2 (weeks 5–10)** | A1+A2 template format + starters, B1+B2+B3 auth + YouTube, C2 generic engines + C4 beat/audio coupling | Templates + YouTube publish = the "one-click video" loop end to end |
| **3 (weeks 11–16)** | B4+B5 Instagram + destinations UI, A3/A4 authoring + recipes, E2/E3 caption embedding + vertical factory | Completes the social loop; recipes make it repeatable |
| **4 (quarter 2)** | C3 flow-presets + content-aware wipe, D2 neural backend, E5, B6 TikTok | Flagship differentiation; heavier ML work after the loop ships |

Upstream housekeeping: local `main` (autocut) is current, but **upstream FilmCraft `origin/main`
has unreconciled commits** (`8309f21a..09aaf9bf`) — merge or rebase deliberately before large
worklands to avoid a painful late merge.

## Guardrails (from repo rules, restated because this plan touches all of them)

- Never crash: publish jobs, model downloads, template validation and shader compilation all
  return errors; hostile-parameter tests for every new command (URLs, quotas, aspect strings,
  template packages — treat template zips like project files: untrusted input).
- Assets: every bundled asset (soundfont, element, starter-template asset, ported shader) needs
  an `.attribution` sidecar + `ATTRIBUTION.md` row; platform logos are trademarks → generic icons;
  nothing NC/GPL/ND ships (MusicGen weights, Stable Audio, Shadertoy, MLT luma files are all out).
- Clean-room: CapCut/Canva/Premiere behaviours only — never their code, formats reverse-engineered,
  or artwork. ffmpeg stays an external test oracle. Algorithms (beat tracking, seam carving, SmartSound
  fitting) are implemented from papers/expired-patent descriptions.
- Everything is a command: templates, publishing, music generation and transitions all surface as
  engine commands + MCP tools + CLI, so the supervisor can drive the whole pipeline.

## Sources

Landscape: kdenlive.org release index (25.04→26.08.2); shotcut.org blog (26.1–26.9.27);
openshot.org 4.0/4.0.1 announcements; olivevideoeditor.org + olive issue #2395; Blender VSE 2026
devtalk roadmap; LosslessCut + auto-editor repos; browser-use/video-use; FireRedTeam/
FireRed-OpenStoryline; ClipsAI; CapCut pricing/backlash threads; Blackmagic Resolve 20 pages;
Descript/Opus Clip product pages.

Templates/publishing: CapCut template guides + creator program; Canva Beat Sync/Magic Resize;
iMovie trailer docs; Adobe .mogrt media replacement; VN templates; KDE Discuss template-workflow
thread; OpenTimelineIO; mifi/Editly; Remotion licence pages; Google OAuth native-app +
resumable-upload docs + quota calculator (Dec 2025 change); Instagram content-publishing docs
(two API surfaces, container flow, specs, 100/24h limit); TikTok Content Posting API; OBS
account-linking discussions; porjo/youtubeuploader.

Transitions: gl-transitions repo/spec (MIT, 106 shaders); ffmpeg xfade wiki; MLT/frei0r plugin
registries; Kdenlive transitions docs; godotshaders.com licence (CC0 default) + Universal
Transition Shader; three.js WebGPU transition example; Shadertoy terms (NC); Adobe Morph Cut
docs; CapCut AI/optical-flow pages; seam-carving + video-retargeting papers (Avidan–Shamir 2007,
Rubinstein, Grundmann), Deep Video Prior (NeurIPS 2020), ICCV 2025 transition-generation paper;
naga/wgpu docs.

Music: ACE-Step / ACE-Step-1.5 repos + HF cards (MIT); Marenz/ace-step-rs (candle, MIT);
Serveurperso/ACE-Step-1.5-GGUF; DiffRhythm; YuE/YuE2 licence notes; MiniMax-Music3 LICENSE;
Stable Audio 3 announcement + Community License; AudioCraft repo (weights CC-BY-NC); Magenta
RealTime; rustysynth; GeneralUser GS; ebur128-rs; bpm-analyzer; SmartSound; FCP "Magic Music";
Epidemic Sound taxonomy write-ups.
