# AutoCut vs FilmCraft — features we added

> **Last updated:** 2026-11-23 · **documented-through:** bbefc3f11797120154f6955ea034bed8aee99bbf
>
> AutoCut is the private layer built on top of upstream FilmCraft (storytold/filmcraft): a
> local-first, agent-drivable video production pipeline. Everything here runs on the machine —
> scene understanding, speech, voice cloning, music, privacy redaction, publishing — with models
> downloaded only after the user consents, and every capability exposed three ways: engine
> commands, MCP tools, and the `filmcraft-cli`. The proof it holds together end to end is the
> worked example in [video-pipeline.md](video-pipeline.md): a Google Photos album turned into a
> Malayalam-narrated, voice-cloned 4K video published to YouTube, plus an Instagram Reel trailer.

## RAEA — local motion-graphics renders on the timeline

[RAEA](https://github.com/shinyobjectz/raea) renders custom animations, full scenes, 3D
explainers, motion graphics and transition overlays on the local GPU from editable TSX/HTML
compositions; AutoCut imports the results as ordinary timeline media. A route animation or an
illustrated story beat can be a full opaque scene (MP4); lower thirds, callouts and wipes can
keep transparency over footage (numbered PNG sequences).

- **Separately installed, never bundled.** FSL-1.1-ALv2 licensed; the binary, SDK and runtime
  stay outside the repository and the app packages. Verified release pinned by SHA-256
  (`~/.raea/versions/<v>/bin/raea`), composition guidance via `raea skill`.
- **Workflow** ([raea-workflow.md](raea-workflow.md), `examples/raea/render.py` +
  `examples/raea/motion.tsx`): compositions live under `~/Movies/AutoCut/<Video>/build/` with
  their assets; review-size renders first, approved graphics regenerated at final size for 4K.
  The helper runs AutoCut's native probe, rejects gaps in numbered renders, and writes
  `autocut-import.json` only after a successful handoff; a failed handoff retains the rendered
  bytes but creates no import manifest.
- **GPU co-ordination:** `filmcraft-cli locks hold gpu --minutes 30 --note 'RAEA motion
  graphics'` reserves the shared GPU while RAEA runs (the same advisory lock the engine takes
  for its own model work — `crates/locks`: `chrome` and `gpu` file locks, `locks [--json]`).
- **Placement:** the manifest's `command`/`params` go to `command_run` (e.g.
  `file.importImageSequence` for an alpha wipe), then `timeline.place` — a scene on the picture
  track, an alpha sequence on a track above the shots. Sample QA spec files declare expected
  duration and final holds; `media.preflightAnimation` checks holds/text bounds/sound cues.
- **As commands (PX3.2):** `motion.check`, `motion.stills` and `motion.render` do the helper's
  work in the engine (CLI, MCP, control channel), RAEA found through `providers.json` or where it
  installs itself.
- Compositions are authored per revision directory; keep the editable TSX with the render so the
  design can change in a new revision. RAEA v0.1.0 ignores output-size overrides on fixed-size
  compositions — the helper sets the output stage and rejects an unexpected rendered size.

## EffectCraft — templates rendered in process

[EffectCraft](https://github.com/storytold/effectcraft) (MIT OR Apache-2.0), upstream FilmCraft's
compositor, is linked into AutoCut (PX4, decisions D3 and D3a): its renderer, effects, GPU
compositor, expressions and text, as git dependencies pinned to one revision
(`filmcraft_effectcraft::PIN`) behind the cargo feature `effectcraft`. The packaged AutoCut app
is built with it (`packaging/macos/package.sh`); workspace builds, `cargo test --workspace` and
the web build are not, and there every `effectcraft.*` command says the build lacks it.

- **Templates:** an `.ectemplate` (an Essential Graphics template: a ZIP with `manifest.json`,
  `project.ecproj`, its media and a poster) is filled with values and rendered: data-driven lower
  thirds, end cards, chapter cards. `effectcraft.templateInfo {template}` lists the comp (size,
  rate, duration, frames), the controls (name, type, default, slider range or options), the media
  it carries or needs, the poster and whether it can be rendered here.
- **Rendering:** `effectcraft.renderTemplate {template, values?, size?, fps?, range?, out?, format?,
  gpu?, import?, place?, wait?}` checks everything first (each value through its control's own
  coercion, the size against the comp's shape and the caps, the frame count, the media), then
  renders as a job (`jobs.list`, `jobs.cancel`, an MCP long job) under the machine's GPU lock:
  EffectCraft's GPU compositor on a device of its own, its CPU renderer for any frame the GPU
  declines or fails. Output is ProRes 4444 with alpha (`format: "prores4444"`, the default) or a
  PNG sequence with alpha (`"png"`) in a new revision folder (`out`, else `<template> renders/rNNN`
  beside the template), probed by AutoCut, written up in `autocut-import.json`, imported and, with
  `place`, put above the picture as one undo step — `motion.render`'s handoff.
- **Footage and fonts:** the template's media and media values are decoded by AutoCut's own
  decoders (colour converted both ways); EffectCraft's bundled fonts are used, plus the craft-fonts
  when the app is built with `CRAFT_FONTS_DIR`.
- **Not linked:** EffectCraft's engine, media, host, export and interchange crates (they bring
  upstream FilmCraft crates under our names; `cargo xtask layers` refuses them in `Cargo.lock`).
  So the engine's migration of old effects does not run: a template made by an old EffectCraft may
  render differently here. Templates render without their sound, and 3D model footage is refused.
  For a sharp overlay give `size` the sequence's size (a render of another size fills the frame).

## The production pipeline, end to end

`docs/video-pipeline.md` documents the whole chain with commands: **album → pick → analyse →
plan → script → voice → build (1080p review) → QA + privacy → final (4K) → YouTube (private) →
subtitles → publish → trailer (Reel) → Instagram**, with review notes feeding the next revision
round. The worked example ("Every Robot I've Met": 48 robots, Malayalam narration in the
author's cloned voice, 4K on YouTube, 34-second Reel trailer) kept its artifacts
(plans EN/ML, script, YouTube metadata, chapters, SRT/VTT, narration-check, QA sheets) out of the
public repository, as they held the owner's paths, library ids and locations. `docs/showcase-runbook.md` covers the revision round in depth:
review notes → `showcase-plan --patch`, **places** from Google Photos location data
(`photos locations`, `--places-from-locations`), photos-only entries, strict Malayalam
heard-back checks, and the vertical trailer with safe-zone and cuts-on-beat checks.

## Quick Short: a finished Reel or Short in one command

`autocut.quick` (CLI `filmcraft-cli quick <folder|files|project>`, MCP `quick_short`, UI
**Sequence ▸ Quick Short…**) turns an album, folder or selection into a finished vertical video:
index the clips shot by shot, listen to what is said, pick a template and fill it — every slot's
shot, the text, the music, the captions, the look, the privacy blur. Options: `--template`
(`auto` scores all starters), `--aspect` 9:16/1:1/4:5/16:9, `--platform` reels/shorts/tiktok
(text and captions keep inside that platform's UI-safe areas), `--seconds`, `--topic`,
`--language` (en/ml), `--look` tech/vhs/clean, `--music` (own file — cuts follow its beats — or
generated bed), `--redact` faces/plates/off. Writes the video plus contact sheet, report and QA
JSON. Nothing runs without its models: the report says what was used and what was not.

## Templates and the scene index (the machinery under Quick Short)

- **Template format** (`crates/templates`, `Name.fcvideo-template/`): a strict
  `template.json` manifest + a normal `.fcproj` sequence fragment + optional licensed `assets/`.
  Manifest: up to 8 **aspect variants** (per-clip layout hints), ≤64 **slots** (video, photo,
  text, logo, music, captions) with constraints (duration/orientation/aspect/char limits), **fit
  policies** (loop/stretch/hold/remix, portion, fill/fit), a **beat grid** (cuts re-quantize when
  music swaps), look (Lumetri preset), caption style, export preset, poster. **AI hints per
  slot** (purpose: hook/broll/reveal/talking-head/…, query, energy, speech want/avoid,
  minQuality) let the scene index choose footage automatically. Filling swaps media only —
  effects, transitions and markers stay; `template.markSlot`/`template.save` round-trip.
- **Starters:** ~20 built-ins (photo-beat-slideshow, channel-intro, top-3/top-5-listicle,
  talking-head-captions, product-promo-15, travel-vlog intro/outro, before-after-reveal,
  lower-third-chapter-kit, end-card-subscribe, tutorial-pip, podcast-two-speakers, quote-card,
  countdown, event-recap, day-in-my-life, unboxing, qa-session, …) plus the AI-ready shorts set
  (robot-showcase, tech-reveal, event-recap-short, talking-head-tip, gadget-showcase,
  before-after-short, day-in-30s) with vertical safe-area text and Creator looks. Live list:
  `template.list`.
- **Scene index** (`crates/scenes`, `docs/scenes.md`): per-shot cuts (TransNetV2), quality
  (sharpness/exposure/shake/motion + LAION aesthetic head), camera move, and Qwen3-VL
  descriptions (2B fast pass, 4B full-detail on the best shots) with SigLIP 2 embeddings.
  Commands `media.analyzeScenes/scenes/searchScenes/bestShots/checkText` (the last checks
  generated text against the frames before it is placed). Cached by content hash under
  `<data dir>/scene-cache`; `fast` needs ~7.5 GB of models, `full` ~12.2 GB; ~3.1× real time
  end-to-end on M3 Max.
- **Platform safe zones** (`crates/project/src/safe_zones.rs`): measured UI margins for Shorts,
  Reels, TikTok (+ worst-case `vertical`); `view.safeZones`, `sequence.checkSafeZones`; caption
  burn-in respects them.

## Direct: the local model edits on the beat

`filmcraft-cli direct` (engine `autocut.direct`, `crates/autocut/src/director.rs`) directs
normal and 360° clips into one edit: scene/speech index, 360° moments via `media.salience360`,
moments measured and ranked (`media.measureMoments`: sharpness/exposure/shake/faces/speech), a
local-model shot list from the brief (checked and repaired, rules fallback), camera moves
(`clip.vrMove/follow360/hyperlapse360` or flat push-in/follow/speed-ramp), all cut on generated
music beats, with match cuts, whips and J/L cuts (`clip.splitEdit`), styles
`subtle|energetic|cinematic|auto`, auto colour, captions and mix. `revise`+`previous` re-edit;
`dryRun` lists the commands; export writes video + sheet + report + QA.

## Showcase: the long-form generator

The original AutoCut capability, now far deeper: Qwen2.5-VL clip analysis with identity
grouping (one subject across days counts once), chapters by event/year, narration with
researched facts, cloned voice with strict heard-back checks, beat-cut editing, rotating
transitions with whooshes, lower thirds, thumbnail + `chapters.txt` + SRT/VTT sidecars, QA
reports (joins/gaps, black frames, loudness, subject re-grounding) and privacy sheets.
Album-seeded (`--album`, a Google Photos album as the whole input), `--stabilize`,
`--match-color`, `--target-minutes`. In-app Showcase panel, MCP `showcase_analyze/plan/build/
inspect`, CLI `showcase`. Redaction pass runs over every footage clip after build (`--redact
faces,plates`).

## Publishing: YouTube and Instagram from the CLI/MCP

- **YouTube** (`crates/youtube`): resumable `videos.insert` (8 MiB chunks, sessions kept ~6
  days so reruns resume), metadata (title/description/tags/category/privacy), `--publish-at`
  scheduling, `--made-for-kids`, custom thumbnail (`thumbnails.set`), playlist add (found, never
  auto-created), chapters sidecar appended to the description, Shorts hint, 15-minute check
  against `longUploadsStatus`. Quota tracked locally (10,000 units + 100 uploads/day, Pacific
  midnight; Google stays authoritative). Commands `youtube.connect/channels/setDefault/
  disconnect/upload`; CLI `filmcraft-cli youtube auth|channels|default|disconnect|quota|upload`;
  MCP `youtube_upload`; `showcase --upload-youtube`.
- **Instagram** (`crates/instagram`): Reels via the Instagram API with Instagram Login —
  caption, cover (`--cover-at` seconds or `--cover-url`), share-to-feed; direct resumable
  upload attempted first, falling back to a public `--video-url` (what Meta currently requires
  for this app); containers kept ~23 h so reruns resume; processing polled ≤5 min; limits read
  from Instagram's own `content_publishing_limit` before every publish. Commands
  `instagram.connect/account/limits/disconnect/publish`; CLI `filmcraft-cli instagram
  setup|token|auth|account|refresh|limits|disconnect|publish`; MCP `instagram_publish`.
- **Auth** (`crates/oauth`): bring-your-own OAuth client (each user's own Google Desktop
  client JSON), PKCE S256, loopback redirect, system browser, `state` check; tokens in the
  macOS Keychain, one vault per YouTube channel; revocation on disconnect. Instagram uses its
  fixed-redirect flow with short-lived→long-lived token exchange (app secret in the Keychain)
  or a pasted Dashboard token.
- **Shared publish layer** (`crates/engine/src/publish.rs`): `publish.destinations/preflight/
  fixCopy/run` over `youtube`, `youtubeShort`, `instagramReel` — pre-flight checks container,
  codecs, fps, aspect, bitrate, duration, size, `moov` position and loudness (−14 LUFS, −1
  dBTP); `fixCopy` re-encodes or moves `moov`; a privacy finding blocks upload (explicit
  override flag). MCP `publish_run`; recipe `builtin:template-publish-youtube`.

## Music: three generation paths, one interface

1. **Procedural beds** (`crates/audio-dsp::music`): exact-length four-stem arrangements
   (drums/bass/melody/pad), deterministic per seed, 9 moods × 6 genres × 30 use cases × 18
   styles; `music.taxonomy`, `music.suggest`, `music.preview`, `audio.generate {kind:"music"}`.
2. **SoundFont MIDI** (`crates/soundfont`): rustysynth (MIT, pinned) with GeneralUser GS
   (32 MB, consent-downloaded, never bundled); same arrangement and beat map as the oscillator
   path, level-matched; `music.soundfonts/downloadSoundfont/useSoundfont`.
3. **ACE-Step neural** (`crates/acestep`): ACE-Step 1.5 turbo DiT (2.4 B) + Qwen3-Embedding
   text encoder + Oobleck VAE, weights MIT, pure Rust on candle/Metal, 6.3 GB consent-gated
   download pinned to an HF revision; instrumental 5–240 s from a description with seed, tempo,
   key and metre; `music.generateNeural/neuralModels/downloadNeuralModel`. Limitation (honest):
   no LM planner yet, instrumental only.

**Engine sounds**: `audio.generate` kinds whoosh, impact, riser, swish, chime, shutter, pop,
thump, drone, engineRev, engineLaunch, engineFlyby, music — all procedurally synthesized, no
asset licensing. Generated tracks carry beat maps, so cuts and transitions land on beats.

## Voice: cloning, Malayalam, styles, isolation

- **Engines** (`crates/tts`): built-in formant voices (no download); Kokoro-82M (9 US English
  voices); OpenVoice V2 tone-colour cloning; **VoxCPM1.5/2.0** zero-shot natural cloning (the
  default; candle/Metal); **IndicF5** (AI4Bharat, MIT) — Malayalam plus 10 other Indian
  languages, continuing a reference clip; whisper-medium-ml hears Malayalam readings back with
  CER thresholds and re-reads failures.
- **Voice styles**: per-style references (conversational/warm/excited/amazed/curious/calm/
  dramatic) with inline tags `[excited]…[calm]`, TD-PSOLA prosody reshape
  (`audio-dsp::prosody`), and the **style finder** (`voice.findStyles`, CLI `voice styles`):
  finds 6–12 s candidates in the speaker's own recordings, measures prosody (pitch/energy/
  syllable rate/contour), scores against the speaker's own spread, picks the takes both
  Malayalam models heard correctly.
- **Voice lab** (`docs/voice-recording.md`): `voice.findReferences/audition/promote`,
  `voice.versions/rollback`, `voice.prosody` (measures and reshapes any recording),
  `voice.mixCheck`.
- **Narration takes & cache**: `take: N` re-reads a line afresh; bit-exact narration cache
  (2 GB LRU) keyed on voice identity, model pins, reference bytes and check policy.
- **Isolation** (`crates/voice`): DeepFilterNet 3 in pure Rust, consent-gated —
  `audio.isolateVoice/enhanceVoice/enhanceFile/downloadVoiceModel`, DSP Wiener fallback.
- Malayalam script tools: `script.lintMalayalam`, `autocut.malayalamScript`;
  `docs/malayalam-narration.md`, `docs/malayalam-style.md`.

## AI effects (the "render anything" pack)

`docs/ai-effects.md` is the reference. All models run locally on the GPU (CPU fallback),
consent-gated downloads pinned by revision + SHA-256, permissive weights only:

| Feature | Commands | Model (licence) |
|---|---|---|
| Photos in depth — 3D Photo, Depth Blur | `clip.photo3d`, `clip.depthBlur`, `depth.status/downloadModel` | Depth Anything V2 Small (Apache-2.0, 99 MB) |
| Upscale and Restore | `media.upscale` | Real-ESRGAN x4v3 (BSD-3, 4.9 MB) |
| Face fingerprints and people grouping | `media.faces`, `faces.group`, `faces.status/downloadModel` | AuraFace-v1 (Apache-2.0, 261 MB) + YuNet |
| Smooth Slow Motion | `media.interpolate` | RIFE 4.26 (MIT, 23 MB) |
| Remove Object | `clip.removeObject` | MI-GAN (MIT, 30 MB) |
| Object Mask (click → tracked mask) | `clip.objectMask`, `masks.track` method Object | MobileSAM (Apache-2.0, 41 MB; WGSL encoder) |

AutoCut's director and showcases apply the 3D photo move matching each photo's Ken Burns
automatically (`--no-depth` off). The broader vision crate also carries YuNet face and plate
detectors, RTMPose person/pose models (faces in profile), and DFine tracking.

## Generation with the user's local tools

`docs/generation.md` is the reference. Unlike the AI effects above, nothing here is a model
inside the app: a **generation tool** is a program the user installed (an MLX, PyTorch or
ComfyUI command line), listed in `<data dir>/generation/providers.json`. The app runs it
with no shell in between, waits for the file it writes and takes it from there.

| Command | Does |
|---|---|
| `generate.providers` | the tools of this computer, their licences and what must be credited |
| `generate.image` | a picture from a prompt; reference pictures or the program frame make it an edit; `transparent` for a cut-out |
| `generate.video` | a short clip from a prompt, optionally starting from a picture or the playhead frame |
| `generate.sound` | a sound effect or music from a prompt |
| `generate.fillGap` | fills a gap of a track with a clip that continues the frame before it (or leads into the one after) |
| `generate.augment` | changes the frame at the playhead by an instruction and lays the result over the clip, as a still or an animated clip |
| `generate.extend` | carries a shot on past its end: the tool gets the last moments as a small video, so the movement continues |
| `generate.restyle` | makes a clip again in another look through an adapter of the clip tool: shapes and movement stay; `look` has the picture tool change the first frame first, which swaps a character |
| `generate.assets` | builds the files a scene's `assets.json` names (pictures, cut-outs, clips, sounds), keeps the ones that are there, makes a pose from a character by name |
| `generate.pick` | swaps another variation in for the one on the timeline |
| `generate.history` | what was generated into this project: item, tool, description, seed |

`variations` (up to 4) on pictures, clips and sounds; `generate.fillGap` `from` `before`, `after`,
`both` (frame to frame) or `motion` (the movement carries on); a sound sized to a clip (`clip`); a
placed picture that moves in depth (`move`); a tool that prints `progress 0.4` drives the bar.

One generation at a time under the shared GPU lock; `jobs.cancel` kills the tool and changes
nothing; the result is imported and placed as one undo step and never overwrites what is on
the timeline. The description, tool and seed are kept in the item's metadata. The Generate
panel (Window ▸ Generate; modes Picture, Clip, Sound, Fill gap, Augment) drives the same
commands, with automation ids `generate.*` and its draft in `ui.generate`. No command writes
the provider file: what may run on the computer is decided by whoever can edit it.

Tested with MiniMax-H3-Image, LTX-2.3 and Stable Audio 3 small as MLX tools (timings in
`docs/generation.md` §3); their licences are the user's to accept (§4). A generated picture
is also an asset for a RAEA composition (`"import": false`, `"dir"`). Follow-ups are in
`plan/generation-2026-11.md`.

## Privacy: redaction and the pre-upload gate

`clip.autoRedact` / `sequence.autoRedact` put tracked blur/mosaic masks on faces and plates
(detector + `minHits` tracking between looks, masks held before/after, second 2× detector pass
for small faces, the featured subject's box excluded, robot-box faces hidden only after the
vision judge twice confirms they are not people). `sequence.checkPrivacy` re-scans rendered
frames at export; an uncovered face is a finding that **blocks uploads** (`--allow-privacy-
findings` overrides). QA writes `privacy.png` sheets and `unblurredFaces` counts. Detector
models: `redact.downloadModels` (4.4 MB + 35.8 MB with profiles).

## 360° and Insta360

`docs/360-video.md`: `.insv/.insp/.lrv` library scan, import, and Studio-cloud download via the
screen agent (`insta360.*` commands, MCP `insta360_list/_import/_download`); virtual-camera
reframing (`clip.reframe360` with tiny-planet/rabbit-hole presets, `clip.stabilize360`,
`clip.follow360` subject/face tracking, `clip.vrMove` cinematic moves, `clip.hyperlapse360`);
360° export with GPU rendering; `direct` handles 360° clips in the same edit as flat ones.

## Google Photos and the screen agent

OAuth installed-app flow; picker sessions driven by the vision-model agent with many searches
per session, sweeps across date ranges, picker-tab find/reuse/close handling, `photos tidy`,
`photos pick --photos`, `photos pick --meaning` (FrameAtlas finds items by what they look like —
SigLIP 2 vectors through `media.embed`/`media.nearest` — and the agent picks exactly those by the
day and second each was taken; MCP `googlephotos_pick` `items`), analysis caches, and resumable
uploads. The showcase can be seeded from
an album, and location data becomes chapter **places**. MCP `googlephotos_pick/add/upload/
cached`.

## Local pipeline upgrades (delivery-grade plumbing)

`docs/local-pipeline-upgrades.md`: **FrameAtlas edit plans** (`catalog search/retrieve/job/
assemble`, `catalog.importEditPlan` — versioned sequences with provenance and integrity
checks); **screen protection** (`clip.autoColor protectScreens` — screen rects become editable
Lumetri-pass masks); **render cache** (`render.cachePlan/cacheCommit/cacheInspect`, hash-keyed
chapter cache); **RAEA review bundles** (`render.py --preview --cache` +
`media.preflightAnimation`); **checked 4K delivery** (`delivery.preflight/prepare` — privacy,
continuity, narration and mix gates; 4K H.264/AAC at −14 LUFS; `short:true` portrait) via
recipe `builtin:delivery-quality-gate`.

## GPU policy

`docs/gpu.md` (2026-10-21): GPU by default with CPU as tested fallback — export compositing, 45
GPU effects + 11 Creator effects, WGSL face/plate detectors on macOS, Metal candle models
(Whisper, Kokoro, VoxCPM, IndicF5, ACE-Step, Depth Anything), WGSL MobileSAM/MI-GAN/RIFE/
Real-ESRGAN. The advisory `locks` crate (`chrome`, `gpu`) co-ordinates the GPU with external
renderers like RAEA.

## Agent drivability (unchanged in spirit, much wider in surface)

The supervisor is the MCP client; local Ollama models classify and plan; every edit is an
undoable engine command. The intent catalog: remove_silence, remove_fillers, add_captions,
cut_range, keep_range, highlight_reel, enhance_voice, add_voice_over, export_video, slow_motion,
showcase, custom→needsSupervisor (digest + optional VLM frames). MCP now exposes the whole
pipeline: `quick_short`, `shorts_make`, `trailer_make`, `rough_cut`, `autocut_plan/direct/edit/
report`, `scenes_analyze/search`, `template_list/apply`, `showcase_*`, `sequence_auto_reframe`,
`auto_redact`, `speech_understand`, `isolate_voice`, `enhance_voice/audio_file`,
`googlephotos_*`, `insta360_*`, `youtube_upload`, `instagram_publish`, `publish_run`,
`recipe_list/run`, `feedback_bundle`, `review_*`, `ui_*`, plus `command_run/batch/list`. Recipes
(`builtin:template-publish-youtube`, `builtin:delivery-quality-gate`) chain multi-step flows.
The agent guide is `docs/autocut-agent-guide.md`.

## Update 2026-11-11 (first run, c5c16939..bbefc3f1)

First edition of this document, surveying the AutoCut layer through bbefc3f1 (596 commits since
the upstream base): RAEA local motion-graphics workflow; the end-to-end album→YouTube/Instagram
pipeline with its worked example; Quick Short; the template engine (~20 starters + AI-ready
shorts) and the scene index; `direct` (beat-cut auto-editing incl. 360°); showcase deep-dive
(places, review notes, Malayalam heard-back); YouTube + Instagram publishing with BYO-OAuth and
the shared pre-flight/fix layer; three music paths (procedural, SoundFont, ACE-Step) and engine
sounds; the voice stack (VoxCPM/IndicF5/styles/style finder/lab/DFN3 isolation); the AI effects
pack; privacy redaction + the upload-blocking gate; 360°/Insta360; Google Photos picker
sessions; FrameAtlas/screen-protection/render-cache/delivery-gate upgrades; the GPU policy.

## Update 2026-11-19 (dc416ef8..GEN1, AV36 to AV49)

- **GEN1**: generation with the user's local tools (section above): seven `generate.*`
  commands, the Generate panel, MCP long-job handling, the provider file. 921 commands.
- **AV36**: the Insta360 download agent finds recordings in a newest-first Studio list.
- **AV37 to AV49**: upstream FilmCraft `e6df3186..f67d38b3` (175 commits; 18 taken, 157
  skipped, 150 of those the craft-libs migration), recorded in
  `plan/upstream-sync-2026-11-19.md`. For agents the one that matters is AV42:
  `engine.execute` never opens a dialog, so a command sent without parameters runs with
  its defaults or answers what it needs (`docs/control-protocol.md`, "Parameter dialogs").
  Also: FCPXML carries adjust-conform (clips from other apps import fitted to the frame),
  bounded decoding of animated GIF, APNG and WebP, bounded MCP line lengths, colour
  parameters take `#rrggbb`, a text property change applies to every selected text clip,
  the layout in use at exit comes back at launch, Bosnian interface language.

## Update 2026-11-20 (GEN2)

- **GEN2**: the second round of generation (section above, and
  [`docs/generation.md`](generation.md)). `generate.assets` makes the missing files of a
  scene's `assets.json` (pictures, cut-outs, poses made from one character picture, moving
  plates, sounds); `variations` 1 to 4 with `generate.pick` to swap one in; `generate.extend`
  carries a shot on past its end; `generate.fillGap` `from: "both"` joins the frames on each
  side of a gap and `from: "motion"` carries the movement on; `move` gives a placed picture a
  depth movement; `generate.sound {"clip": id}` sizes a sound to a clip; a tool that prints
  `progress …` lines drives the bar; a waited MCP call answers what was made. 924 commands.
- **GEN3**: `generate.restyle` makes a clip again through an adapter of the clip model
  (Union-Control, 654 MB, downloaded on the user's word): the shapes and the movement stay and
  the look follows the description, and with `look` and `references` the picture tool changes
  the first frame first, which puts another character in a person's place. A Restyle clip mode
  in the Generate panel. 925 commands.

## Update 2026-11-22 (upstream sync, AV55 to AV58)

- **AV55 to AV58**: upstream FilmCraft `f67d38b3..009b72ca` (18 commits; 8 taken, 10 skipped,
  8 of those the craft-libs migration), recorded in `plan/upstream-sync-2026-11-22.md`.
- **Slip Monitor** (AV56): dragging a clip with the Slip tool (Y) turns the Program monitor into
  a four-up (the previous clip's last frame, the slipped clip's new first and last frames with
  their source timecodes, the next clip's first frame) with the offset in timecode and frames;
  the Timeline shows the offset at the pointer and the sound's waveform slides. Agents get
  `timeline.slipPreview {clip, delta|deltaFrames}`: what `timeline.slip` would do, without
  editing. 930 commands.
- The Trim Monitor's two pictures at a cut inside one take no longer cancel each other's decode
  (AV55); VA-API reads pictures without a copy on Linux (AV57); a project saved before
  2026-10-21 with portrait photos opens with them the right size and centred (AV58).

## Update 2026-11-23 (PX1, craft QA)

- **PX1.1**: `qa.motionEnergy {path|sequence, fps?, holds?, loop?, sheet?}` measures how a film
  moves on frames reduced to 96×54 grey: single-frame pops, cuts per minute, still stretches of
  1 s or more outside declared holds, motion a second and its contrast, a flat first 5 s, a film
  that barely moves, a black or empty first frame and the loop seam, with a contact sheet at four
  frames a second. Sequences render on the GPU frame renderer where the host has one. It is a leg
  of `media.preflightAnimation` and `delivery.preflight`; every finding is a warning (D7) and never
  blocks. Thresholds and limits are in `docs/local-pipeline-upgrades.md` ("Craft QA"). 933
  commands.
- **PX1.2**: `sequence.checkCraft {sequence?, holds?}` reads a sequence's text and timeline: a
  title or caption on screen for less than its reading time (3 words a second plus 0.5 s), text
  under 2.6 % of the frame's short side, contrast of text against the picture rendered behind it
  over its life (4.5:1, 3:1 for large text; the text layers are turned off for the backdrop, which
  is an approximation the doc spells out), an empty first frame, more than 3 s with nothing
  changing outside declared holds and more than 15 s with no sound. Warnings only (D7). 934
  commands.
- **PX1.3**: `copy.lint {text?, sequence?, publish?: {title?, description?, caption?, tags?},
  script?, facts?, allow?}` checks a video's words against FilmCraft's own lists: 39 stock phrases
  of video copy ("in this video we'll", "let's dive in", "smash that like button") and 27 of
  marketing copy ("game-changer"), 13 fillers and 14 hedges, 23 phrases of narration that
  describes the shot ("here we see", "as you can see"), device budgets per 100 words (exclamation
  marks, questions, em dashes, emoji, all-caps words; warn over the budget, a `fail` level at
  twice plus one, still a warning this round) and figures in digits that are not in the fact
  ledger (`facts`, or `build/facts.json` beside the project). The person's own phrases win (D6):
  the taste profiles' new `phrases` field, wording rules' `say` and `allow` are never flagged. It
  is a `copy` check of `publish.preflight` (warnings only) and `publish.fixCopy` answers the copy
  with its safe fixes (stock greetings and calls to action trimmed from the start and end of a
  description, repeated exclamation marks collapsed). Details in
  `docs/local-pipeline-upgrades.md` ("Craft QA"). 935 commands.
- **PX1.4**: severity tiers in `delivery.preflight`. Every check has a `tier`: **Floor** blocks
  (sources, clicks at audio boundaries, mix, privacy at cuts, changed narration, safe zones,
  screen protection, all blocking exactly as before; and the new `textContrast` under 3:1 from
  `sequence.checkCraft`, `figures` not in the fact ledger from `copy.lint` and `waivers`, which
  warn this round (D7) and block with `floorBlocks: true`), **Reflex** is fixed or waived with a
  reason and evidence in `build/decisions.json` (`{waivers: [{check, code?, time?, reason,
  evidence}]}`; motion, colour, black frames and the new `craft` and `copy` checks; a waiver
  without both, or for a Floor or Finish check, is refused with a finding, and more than five is a
  finding), **Finish** is reported only (motion energy, caption differences, loudness jumps,
  fillers). The report's `summary {floor, reflex, finish, waived}` counts the open findings of
  each tier, beside `blocking` and `floorBlocks`. Details in `docs/local-pipeline-upgrades.md`
  ("Severity tiers"). 935 commands.

## Update 2026-11-23 (PX3.2, motion graphics as commands)

- **PX3.2**: RAEA's render helper as engine commands, also over MCP and the CLI.
  `motion.check {source, composition?}` runs `raea check --quick` and passes its findings through
  (`check`, `level`, `says`, `fix`), with counts per level and what passed. `motion.stills {source,
  composition?, at, output}` renders stills into a new folder and answers their paths with RAEA's
  render manifest: size, rate, length and the text boxes of each frame. `motion.render {source,
  composition?, output, canvas?, size?, preview?, alpha?, fps?, quality?, spec? | expected?,
  holds?, import?, place?, wait?}` is a job under the machine's GPU lock (an MCP long job): the
  check, the render into a new or empty revision folder (one with files in it is refused), AutoCut's
  own probe (the size and rate asked for, no gaps in a PNG sequence and as many frames as RAEA's
  manifest says), `media.preflightAnimation` with `qa.motionEnergy` and the declared holds, then
  `autocut-import.json` in the helper's shape and the import (`file.import`, or
  `file.importImageSequence` at RAEA's rate) and, with `place`, the clip on the timeline as one undo
  step (a transparent render above V1). A render that fails the probe or the preflight stays on
  disk without an import manifest, and the result says where. RAEA is the `kind: "composition"`
  entry `raea` of `providers.json`, else the newest `~/.raea/versions/*/bin/raea`, else PATH
  (`docs/generation.md`). Tested with a stand-in RAEA; run for real with RAEA 0.1.8 on
  `examples/raea/motion.tsx` (the scene with its hold, the 4K master and the split wipe).
  `examples/raea/render.py` still works. 938 commands.
- **PX3.3**: `composition.import {source, composition?, output, alphaCodec?, place?, wait?}` brings
  a RAEA composition in as the edit it is, under one rule: every visible or audible element has
  exactly one owner. A job under the GPU lock runs RAEA 0.1.8's `raea export layers` into a new
  revision folder: what RAEA draws becomes movies of their own (graphics and captions transparent,
  ProRes 4444 or PNG; what lies under the footage opaque; a backing layer behind a picture in
  picture), and `layers.json` names the footage and sounds the composition uses with their times,
  trims, speeds, gains, opacity and boxes. Every index, number and path in it is checked (a drawn
  layer must stay inside the folder and open at the composition's size). The footage and sounds
  come in as real clips (an item already in the project is reused), at their times and trims, at
  their speed, gain and opacity, footage scaled and placed into its box with its own sound on an
  audio track; each drawn layer is one clip on a track of its own, lowest layer lowest. Into a new
  sequence of the composition's size and rate, or with `place` into the open sequence on new
  tracks (nothing there is moved). One undo step. A footage box RAEA crops is shown uncropped and
  listed in `approximated`. RAEA's `reference.mp4` stays beside the layers; on the real run a frame
  of the layered sequence matched it. 939 commands.

## Update 2026-11-23 (PX5, keyframes that move well)

- **PX5.1**: spring, perceptual scale and new eases for keyframes. Four interpolation modes beside
  Premiere's: **Spring** (damping ratio 0.05–10, default 0.9 with no visible overshoot, 0.5–0.7 for
  a declared playful style only; stiffness, mass and an overshoot clamp; it lands on the next
  keyframe exactly), **Steps** (`steps(n)`, n up to 1000), a seeded **Wiggle** (amplitude,
  frequency, seed: the same every render, no state between frames) and **Custom** curves (up to 64
  points, time increasing). A **perceptual** option interpolates scale (any number or point) in
  log space, so a zoom from 100 % to 400 % feels even; zero or mixed-sign ends fall back to linear.
  `effects.setInterpolation {…, interpolation: "spring"|"steps"|"steps(n)"|"wiggle"|"custom",
  spring?, steps?, wiggle?, curve?, perceptual?}` and `effects.setKeyframe` (the numbers alone)
  set them, clamped to safe ranges, with non-finite numbers, unknown keys and curves out of order
  refused; Effect Controls lists the modes and a Perceptual Scale switch in the keyframe menu.
  Preview, export and audio evaluate them through the same keyframe code. Project schema v18 (a
  no-op step: older keyframes load and save unchanged). Details in `docs/architecture.md`
  ("Keyframes"). 939 commands.
- **PX5.2**: `keyframes.choreograph {target: {clip | clips, effect? | effects? | layer? | layers?,
  mask?}, tweens: [{prop, to, from?, duration?, ease?, position?, effect?, perceptual?}], labels?:
  {name: seconds}, stagger?: seconds | {each, from?: "start"|"end"|"center"|index}, defaults?:
  {duration?, ease?}, dryRun?}` places tweens with the relative grammar agents know from GSAP:
  seconds, `"<"`, `">"`, `"<0.2"`, `">-0.1"`, `"+=0.2"`, `"-=0.2"`, labels and `"label+=0.5"`, and
  percentages (`"<25%"`, `"+=50%"`), and writes them as ordinary editable keyframes on the clips'
  effect parameters in one undo step. Eases are GSAP's names (`power1`–`power4`, `sine`, `expo`,
  `circ`, `back`, `elastic`, `bounce` with `.in`/`.out`/`.inOut`, sampled into a 32-point custom
  curve; `none`, `hold`), `spring`/`spring(ζ, k, m)`, `steps(n)`, `wiggle(…)`, the keyframe modes
  and `{spring|steps|wiggle|curve}` objects; the default is `power1.out`, and scale tweens are
  perceptual (log space) unless they say otherwise. Everything is checked: unknown keys, props,
  effects, labels and eases, durations of 0 or less, at most 200 tweens, 64 targets and 64
  labels, and times before or after the clip are refused with nothing written; the result lists
  every keyframe written with its mode, the resolved tweens and the stagger delays. The grammar
  follows GSAP's public documentation of the position parameter; no GSAP code was used. Unlike
  GSAP, an unknown label is an error, not a new label at the end.
- `graphics.animateText`'s Pop now eases with Quart Out (no overshoot) unless `easing: "Back Out"`
  is asked for; the Pop by Letter template and the golden scene ask for Back Out by name and look
  as before. 940 commands.

## Update 2026-11-23 (PX1.5, motion checks)

- **PX1.5**: `sequence.checkMotion {sequence?, style?: "calm"|"playful", bands?: {entrance?:
  [minFrames, maxFrames], exit?: [minFrames, maxFrames], stagger?: seconds}}` reads how a
  sequence's titles, transitions and keyframes move, from the timeline alone (nothing rendered):
  `overshoot` (text-animator and transition easings with Back, Elastic or Bounce, Pop Motion's
  overshoot and Spring Motion's bounce, springs under damping ratio 0.9, custom curves that leave
  0..1 or turn back, as `keyframes.choreograph` writes for `back`, `elastic` and `bounce`), unless a
  playful style is declared (`style: "playful"`, or the taste profile's new `motion: "playful"`);
  `entranceDuration` and `exitDuration` of titles and overlays (a clip with alpha) against bands in
  frames at the sequence's rate (8–30 and 6–24 frames at 30 fps, scaled); `exitOutlastsEntrance`;
  `longStagger` (letters, words or lines taking more than 0.8 s from the first to the last
  starting); `sameEntrance` (three or more titles all entering the same way); `loopingText` (a
  loop quicker than 8 s a cycle moving a title for its whole clip); `edgePop` (a keyframed
  position, scale, rotation or opacity changing by half its move or more between a clip's first
  two or last two frames). Thresholds after impeccable, taste-skill and hallmark's public motion
  guidance, written fresh; starting values to tune. Warnings only (D7), each saying what to
  change. `delivery.preflight` runs it as the `motionCraft` check at the Reflex tier (waivable in
  `build/decisions.json`; `motionCraft: false` skips it). Details in
  `docs/local-pipeline-upgrades.md` ("Craft QA"). 941 commands.
- **PX1.6**: motion filmstrips. `sequence.motionStrips {sequence?, items?: [clipId | {clip, edge?:
  "in"|"out"} | {transition}], out, scale?, wait?}` renders, for each animated title (its entrance
  and its exit), each transition and each overlay with alpha (`motion.render`,
  `composition.import`), seven frames of the sequence (just before, first, three between, last,
  settled after) into one labelled sheet per item in a new or empty folder, on the GPU frame
  renderer where there is one and the CPU otherwise. Every frame of each move (at most 64) is also
  rendered small and read on 96×54 grey grids: `jump` (a step into the first or out of the last
  frame at least 0.02 and 3 × the move's median step), `midStep` (a frame at least 0.01 and 3 × its
  neighbours), `dissolveDip` (a dissolve's mean grey 0.04 under its darker end), `fadeAfterShot`
  (an exit still on screen when the picture under it cuts) and `entranceReplay` (an entrance at an
  edit that brings in what the frame before already showed); the last two compare frames with the
  same frames rendered without the layer. Warnings only (D7), at most 32 items, a job in
  `jobs.list` (`wait: true` runs it in the call). `filmcraft-cli frames --motion --project p.fcproj
  [--sequence N] [--items 12,12:out,t5] [--scale 0.25] --out folder` runs it. 942 commands.

## Update 2026-11-23 (PX4, EffectCraft in process)

- **PX4.1**: the crate `filmcraft-effectcraft` (L3) links EffectCraft's renderer, GPU compositor,
  expressions, text and project model, pinned to one revision, behind the cargo feature
  `effectcraft` (off by default; a stub without it). `effectcraft.templateInfo {template}`
  describes an `.ectemplate`: comp, controls (name, type, default, range, options), media,
  poster, whether it can be rendered here. Everything in a template is capped and checked before
  EffectCraft sees it; parsing and every frame run under `catch_unwind`. `cargo xtask layers`
  refuses EffectCraft's engine, media, host, export and interchange crates and any outside
  `filmcraft-*` crate in `Cargo.lock`; `cargo xtask ci` gains a step with the feature on.
- **PX4.2**: `effectcraft.renderTemplate {template, values?, size?, fps?, range?, out?, format?:
  "prores4444"|"png", gpu?, import?, place?, wait?}` checks the values against each control and
  renders the template with alpha (EffectCraft's GPU compositor under the GPU lock, its CPU
  renderer as the fallback) into a new revision folder, then hands over as `motion.render` does:
  AutoCut's probe, `autocut-import.json`, the import and, with `place`, the clip above the picture
  in one undo step. A long job (`jobs.list`, `jobs.cancel`, MCP progress).
- **PX4.3**: `packaging/macos/package.sh` builds the AutoCut app and CLI (`AUTOCUT_APP_NAME` set)
  with the feature on; FilmCraft builds stay without it. 944 commands.

## Update 2026-11-23 (PX2, music fitted to the film)

- **PX2.1**: `media.beats` also reports the track's structure (`audio_dsp::structure`):
  `meter` and `downbeats` (the bar phase that best explains chord changes, low-band and level
  rises), `sections [{start, end, energy}]` (checkerboard novelty of the beat self-similarity
  plus level changes, on downbeats; energy halves every 10 dB under the loudest section),
  `drops` (a downbeat into the loud part, 6 dB over the two bars before), `stops` (the sound
  falling away 20 dB within 50 ms), `musicStart`/`musicEnd`, `ending {kind: cadence|fade|cut,
  at}` and `confidence` (with `confidenceParts`: beats, downbeats, sections). Tested on
  synthesized tracks with known answers (intro, verse, build, drop, breakdown, drop; cadence,
  fade and cut endings) and on the music engine's own tracks.
- **PX2.2**: `clip.remix` fits music to the film (`audio_dsp::fit`; ideas from huashu-motion's
  fitting rules, written fresh). `events [{kind: burst|peak|landing|stop, time|seconds}]` (film
  time from the clip's start) get a real lift of the track within half a beat — a drop, a
  rising section start, a section boundary, a downbeat, a stop or the track's own ending — or
  are reported as missed (nothing is invented); `ending` (`own`: the track's own ending ends the
  film, or its final hit lands on the landing; `button`: a strong downbeat then a short faded
  tail; `fade`; default as before); `maxStretchPct` (0–25, default 0) time-stretches the whole
  remix at its pitch (WSOLA at playback) to land its end or anchors exactly. Joins go in stable
  stretches that keep the bars whole, with the 20 ms equal-power crossfade, and are rendered and
  checked: a join whose level change is more than 6 dB away from what the music does at its cut
  points (0.5 s RMS either side) is refused and the planner tries again; a plan that still fails
  is kept with a `warnings` entry (warn first, decision D7). The report adds `eventsLanded
  [{kind, at, lift, playsAt, offBy, landed}]`, `joins [{at, out, in, score, levelChangeDb,
  excessDb, stable, keepsBars, ok}]`, `alternatives [{duration, seconds, joins, ending}]` (film
  lengths that need no join), `ending`, `ended`, `stretchPct`, `fadeOutSeconds`, `warnings`. A
  fitted remix works with the whole track from the clip's In point; its fit is stored with the
  remix (a plain remix stores exactly what it did before) and kept when Remix Properties or the
  Remix tool re-plan. Without the new parameters the plan is the Remix of before.
- **PX2.3**: Direct, Quick Short, Showcase and the trailer fit the user's track to the film
  (`autocut::music_fit`), passing their film events and ending to `clip.remix`: **Direct** a
  burst at the opening, a peak at the hero shot (two in an edit over 30 s: the shots with the
  most hook and energy, neither first nor last) and the landing on the closing shot, ending
  `own`, the cut then laid on the track's beats as they play; **Quick Short** a burst on the
  music's first beat, peaks on the title hits and the landing on the end card (the last title in
  the last 40 %), ending `own` (report `musicFit`); **Showcase** a burst on the first beat,
  peaks where the longest scenes play without words (up to four, 15 s apart), the landing on
  the outro, ending `fade` (`music.fit` `"fitted"`, `music.fitted`), for tracks longer and
  shorter than the show; the **trailer** gains `music` (a file instead of generated music): its
  cut planned on the track's beats, a burst at the hook, a peak where the montage starts and its
  own ending landing on the end card (`music.fit`). A track that cannot be fitted falls back to
  what each did before (trimmed and faded, remixed or looped), noted. Joins kept despite a level
  jump become notes; how each event landed stays in the report. The shorts stay silent. A remix
  that starts the track mid-way (an opening burst on a drop) lands the drop on the event exactly
  (the film's beat grid kept) and fades in over 10 ms.

## Update 2026-11-23 (PX10, taste memory)

- **PX10.1**: a taste history and `autocut.fingerprint {sequence?, record?, pipeline?, deals?}`.
  The history is `taste-history.jsonl` in the data directory beside `taste.json`: one JSON
  line per finished video (when, in UTC through chrono; the project file's name only, never its
  folders; the pipeline; the six axes; the deal choices used), newest last, at most 2000 records
  and the newest 4 MiB read, a damaged line skipped with a note. The fingerprint reads the
  timeline alone: `opening` (`titleCard` a title in the first second, `talking` speech in the
  first 2 s, `montage` three cuts in the first 5 s, `still`, else `coldOpen`), `titleEntrance`
  (the titles' most common entrance: a text animator's in-kind, a transition into the title, or
  keyframes — opacity `fade`, a position `slide…`, `scaleUp` — else `none`),
  `transitionFamily` (the most common family of the transitions between pictures, by catalog
  folder: `dissolve`, `whip`, `zoom`, `light`, `slide`, `wipe`, `spatial`, `glitch`,
  `distortion`, `pattern`, `flow`; none: `cut`), `colorLook` (the colour pass's look — the pass
  now names it in `auto_color_look` — a Lumetri preset matched by its parameters, a Creative
  look, a `Look: …` adjustment clip; by time on screen), `captionStyle` (the caption preset,
  `plain` or `none`) and `close` (`logo`, `endCard` a title over nothing, a graphic or blurred
  footage, `title`, `fadeOut`, `cut`), each with its `evidence`. `matches` lists the axes shared
  with each of the three newest records; four or more shared with one is a `sameAsRecent`
  warning (D7: never a block). `record: true` appends it (a file in the data directory, not an
  undo step; the result says so). Tests use a data directory of their own.
- **PX10.2**: `autocut.deal {for, options?, seed?, dryRun?, pipeline?}` and `autocut.feedback
  {about, id, verdict, note?, scope?}`. A deal picks one `look` (our looks and the creator
  looks), `titleStyle` (the `graphics.animateText` in-kinds, `none`), `transitionFamily` (the
  catalog families, `cut`), `music` (the generated music's moods), `register` (the narration
  deliveries) or `endCard` (`endCard`, `logo`, `fadeOut`, `title`, `cut`); `options` narrows the
  catalog, its first the default. It never deals what feedback `retired` or what the taste
  profile rules out (a look other than the profile's; colour looks with the colour pass off; a
  bouncing `drop` unless motion is playful; `light` with light effects off; a family whose
  transitions carry only sounds the profile leaves out — `sounds` without whoosh or riser rules
  out zooms, spins and cubes, while a whip can take the swish and a dissolve needs none), and
  prefers what the three newest videos did not use (liked first; `seed` picks among them), else
  the least recently used, saying so in `why`; the answer is `{for, choice, from, skipped: [{id,
  why: "used in the last 3"|"retired"|"not allowed by taste"}], why}`. Shorts deal no music.
  Unless `dryRun`, the choice is kept for the next `autocut.fingerprint {record: true}`.
  Feedback writes `retired`, `liked` and `reasons` into the taste profile (new optional fields:
  old `taste.json` files read as before; a project's retired choices add to the user's, except
  what the project likes; capped at 16 kinds × 32 ids, 64 reasons): the user's default by
  default, the project's as one undo step. A retired sound is never heard — `allows_sound`, the
  same place the profile's `sounds` act through, and the director drops it from the style's
  sounds unless the video asks for it. `autocut.taste` shows the new fields and `set` replaces
  or clears them.
- **PX10.3**: Direct, Quick Short and Showcase deal what the video and the taste profile leave
  open (`guidance`, on by default; `false` keeps each pipeline's own defaults and records
  nothing), report each deal (`deals`: choice, why, candidates, what was skipped) and record the
  finished video's fingerprint in the taste history (`fingerprint`: axes, evidence, matches,
  warnings; not on a dry run or a revision). Every explicit option still wins, and with an
  empty history each deal gives the pipeline's own default, so a first video is made as
  before. **Direct** deals the `look` (when the colour pass is on), the music's `mood` (when no
  track is given and the brief names none: happy, uplifting, energetic, dreamy, calm), the
  title's entrance (`titleStyle`: its own fade, or a `graphics.animateText` pop, slide up, blur,
  scale up or typewriter), the `transitionFamily` of the join into the ending and across jumps
  in time (`dissolve` is the style's own dissolves, dips and punches; `zoom` a punch in, `whip`
  a whip pan, `light` a light leak unless light effects are off, `cut`) and the `endCard`
  (`cut`, or `fadeOut`: a dip to black out of the last shot) — all four new options of
  `autocut.direct`; a revision keeps its edit's dealt choices. **Quick Short** deals its
  creator look, the template's own first (`tech`, `vhs`, `clean`), so a template's look changes
  only when the last videos used it; a Short is dealt no music. **Showcase** builds deal the
  music's mood when the plan names none (epic first, then uplifting, energetic, happy, dreamy)
  and, when narrated, the register of every part the script gives no delivery (warm first, then
  conversational, curious, calm). Without guidance the defaults still skip what feedback
  retired: Direct's look, mood, title entrance, ending and caption style, Quick Short's caption
  style, the template's look and the template's sounds of a retired kind (taken out before the
  mix), the showcase's mood and register.
- **PX10.4**: the eval harness, `filmcraft-cli eval [--briefs eval/briefs] [--out target/eval]
  [--modes guided,plain] [--only NAME] [--no-export] [--no-judge] [--judge-model NAME]` (a CLI
  subcommand: it needs the engine and AutoCut, which xtask must not link). The fixed briefs in
  `eval/briefs/*.json` (a pipeline, its options, a length, the procedural demo scenes made in
  code as footage) run headless with the guidance (a fresh taste history per run) and without
  it, each export scored by `sequence.checkMotion`, `delivery.preflight` (by tier, and whether
  it blocks), the fingerprints' variety across the briefs (axis spread, axes shared with the
  brief before, pairs sharing four or more) and, when Ollama on this machine has the vision
  model, a judge of the frame sheet (skipped and said so otherwise; never a cloud call).
  `target/eval/<when>/report.json` and `summary.md` compare the two runs and the previous
  report. Minutes, not part of `cargo xtask ci`: run it when a pipeline's choices change
  (docs/testing.md).

## Update 2026-11-24 (PX8, review pointed at frames)

- **PX8.1**: review comments pinned to frames, kept in `build/review/inbox.jsonl` beside the saved
  project (one JSON object a line, at most 2000 comments and 4 MB; not in the project, so no undo
  step). `review.comment {time|seconds|frame|timecode?, sequence?, clip?, box?, text, author?}`
  adds one (ids c1, c2…, never reused); `review.inbox {status?, since?, wait?, timeout?, job?}`
  lists them and, with `wait`, returns when a new open comment arrives (as a job over MCP, the
  bridge and the control channel, so the app keeps answering); `review.look {id, scale?, out?,
  inline?}` draws the comment's box on its frame (`build/review/look-<id>.png`) with what is on
  screen there; `review.done {id, note}` and `review.delete {id}`.
- **PX8.2**: **Comment on Frame** on the Program monitor's transport: click a spot or drag a box
  on the picture, type, Enter. Open comments are drawn on their frames and marked on the scrub
  bar; Previous and Next step through them. Every control has an automation id
  (`program.review.*`).
- **PX8.3**: the finish reviewer's contract (`docs/finish-reviewer.md`): a fresh reviewer gets
  `review.packet {out?, sequence?, frames?, strips?, qa?, preflight?, wait?}` (a brief, a frame
  sheet, motion filmstrips, QA or preflight, the motion check, decisions, comments, the rounds so
  far) and answers on one last line, `GATE: ship | fix <n> [keep 1] | rebuild - <why> |
  recapture - <what>`; `review.gate {text, round?, packet?, dryRun?}` checks and records it (two
  rounds at most), and `review.gate {user, verdict?}` records the user's own words, which close a
  run early.
- **PX8.4**: the same over MCP and the shell. MCP tools `review_comment`, `review_inbox`,
  `review_look`, `review_done`, `review_packet` and `review_gate` (a wait and a packet's
  filmstrips run as long jobs with progress); `filmcraft-cli review (--project p.fcproj |
  --bridge addr) <comment|inbox|look|done|delete|packet|gate> [key=value … | JSON]`, with the
  comment id first for `look`, `done` and `delete` and `gate --file answer.txt` for a reviewer's
  answer.
- **PX8.5**: the agent guide's "Review comments and the finish reviewer" (§9), and the credits
  for the rules adapted from impeccable, hallmark, taste-skill and huashu-motion (ATTRIBUTION.md).

## Update 2026-11-24 (release 0.13.0 and after)

AutoCut's first tagged release, `autocut-v0.13.0`, with the macOS download and notes in
[autocut-releases.md](autocut-releases.md). Besides the PX work above and the upstream and
craft-libs fixes (AV62–AV90, recorded in `plan/upstream-sync-2026-11-24.md`):

- **AV77**: `photos pick --meaning "text"` picks exactly what FrameAtlas finds by meaning, day by
  day in Google's picker (the screen agent's exact-item mode, `screenagent::items`); MCP
  `googlephotos_pick` takes the found items as `items: [{captured, kind}]` (at most 200). It needs
  Chrome's View ▸ Developer ▸ Allow JavaScript from Apple Events. Live-tested on 24 November
  (3 of 3 picked and imported).
- **AV91**: the screen agent clicks and types only while its app is in front; if another app comes
  to the front it stops the run (`NotInFront`) instead of typing into it.
- **AV92**: `packaging/macos/package.sh` keeps the builder's home folder and checkout paths out of
  the binaries (`remap_build_paths` in `packaging/env.sh`; `PACKAGE_KEEP_PATHS=1` skips it).
- **AV93**: `filmcraft-cli insta360 list|import|download --folder F --until D` take their values;
  the folder was read as a recording name and the import fell back to Studio's downloads.
- **AV94**: the agent guide (FrameAtlas, the pick by meaning, Chrome in front, folder imports,
  `composition.import`), the MCP server's instructions (craft and motion checks, delivery tiers,
  keyframes, music fitted to the film, exact-item picks), the README's feature list and the
  Google Photos README describe the 0.13.0 features. 954 commands.

## Update 2026-11-25 (release 0.14.0)

- **AV95**: HEVC Main 4:2:2 10 (8- and 10-bit 4:2:2) decodes with AutoCut's own decoder (craft-libs
  `bb57d422d0`, FilmCraft #1952): intra and inter prediction, transforms, deblocking and SAO;
  frames are labelled C422 (`video::hevc_pixel_data`, shared with HEIF); the picture-size limit
  counts all three planes. Bit-exact with ffmpeg on 30 libx265 streams; 4:2:0 unchanged.
- **AV96**: residuals dequantised as they are read, a bounding-box clear, a DC-only add and
  table-driven sig_coeff contexts (craft-libs `c25999c8a4`): 4–10 % fewer cycles, bit-identical.
- **AV98**: AV1's cross-frame row wavefront (craft-libs `62e09222f1`, FilmCraft #4542): parse
  first, then rebuild row by row while later frames start; 1.6–2.1× with 4–14 threads,
  bit-identical; a failing frame gives waiting frames an error, never a hang; the pool is capped
  at 64 threads.

