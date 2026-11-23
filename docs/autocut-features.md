# AutoCut vs FilmCraft — features we added

> **Last updated:** 2026-11-11 · **documented-through:** bbefc3f11797120154f6955ea034bed8aee99bbf
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
- Compositions are authored per revision directory; keep the editable TSX with the render so the
  design can change in a new revision. RAEA v0.1.0 ignores output-size overrides on fixed-size
  compositions — the helper sets the output stage and rejects an unexpected rendered size.

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
`photos pick --photos`, analysis caches, and resumable uploads. The showcase can be seeded from
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
`recipe_list/run`, `feedback_bundle`, `ui_*`, plus `command_run/batch/list`. Recipes
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
