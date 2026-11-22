# Automating video creation: what to take from 18 open repositories

22 November 2026. Source: a post of 21 November 2026 listing 18 repositories behind "a new
animation every day" (topic → script → motion → score → export), and the article it quotes,
"The Fable 5.2 Motion Playbook" (lessons from 40 agent-made films). Each repository was read on
GitHub on 22 November: README, docs and the LICENSE file itself; source code only where the
licence is permissive (MIT, Apache-2.0); README and user docs only for GPL and non-commercial
ones. Nothing was installed or run.

**Objective:** an agent takes a brief (a topic, an album, an outing) to a delivered video with
no hand steps, and the result does not look generic. Every capability lands as an engine command
and MCP tool, as the rest of AutoCut does.

## 1. Where we stand and what is missing

| Part | Role | Today |
|---|---|---|
| AutoCut | editor, engine, automation surface | 929 commands over CLI, MCP and control channel: Quick Short, Direct, Showcase, trailers, voice and narration, captions, AI effects, `generate.*`, privacy, 360°, delivery and publish pre-flights |
| FrameAtlas | memory of the footage | catalogue of scene descriptions, transcripts, tags and story ideas; edit plans for AutoCut |
| RAEA ("Render Anything") | code-authored motion | TSX compositions, HyperFrames-style HTML with GSAP, three.js and React Three Fiber, drawn on Bevy/wgpu with no Chrome; 0.1.0 installed, 0.1.8 current; driven by a Python helper |

The gaps the repositories expose:

1. **Nothing measures motion.** The gates check loudness, clicks, captions, privacy and sources;
   RAEA review looks at three stills. An unattended build can come out as a slideshow and pass.
2. **Music is cut to the film, not fitted to it.** `media.beats` has tempo and beats, not
   downbeats, sections, drops or how the track ends.
3. **Motion graphics are a helper script, not commands** (`examples/raea/render.py`, no MCP tool).
4. **No compositor.** Particles, 3D, keying, tracking and data-driven titles come from RAEA or
   nowhere.
5. **Keyframes have no spring and no sequencing grammar.** `graphics.animateText` offers
   `Back Out` easing, which overshoots.
6. **No taste memory across videos.** Looks, title entrances and transitions repeat; only
   Malayalam scripts are linted.
7. **No topic-to-explainer path** (narration-driven shot timeline, fact ledger, chapter bar).
8. **Review is chat, not pointed at frames.**

## 2. The 18 repositories

| # | Repository | What it is | Licence | Take | Into | Value | Effort |
|---|---|---|---|---|---|---|---|
| 1 | alchaincyf/huashu-motion | launch films: director rules, music fitted to events, beat map, motion QA, frame-pinned comments | MIT (GSAP, fonts, music library and a character excluded) | ideas and runbook text, credited | PX1, PX2, PX8, PX12 | H | M |
| 2 | Vincentwei1021/anything2explainer | topic → narrated explainer on Remotion | PolyForm Noncommercial | ideas only | PX7 | H | M |
| 3 | getopenscreen/openscreen | screen recordings polished into demos (cursor data, auto zoom) | MIT (links FFmpeg libraries) | external tool; cursor zoom ported later | PX9 | M | S, then M |
| 4 | feitangyuan/onetake | one-take films; rhythm analysis of a reference; continuity oracle | PolyForm Noncommercial | ideas only | PX6 | H | M |
| 5 | lemomo-ai/lemo-opuscar | 99 film looks shot in code | code and data MIT; style text CC BY-NC (own-film use allowed) | reading-time check, look ideas; no text in the repo | PX1, PX11 | M | S–M |
| 6 | storytold/effectcraft | After Effects-class compositor in Rust, by upstream FilmCraft's team; CLI, MCP, template rendering | MIT OR Apache-2.0 | external tool first, then its crates | PX4 | H | S–M, then L |
| 7 | greensock/GSAP | timeline animation for the web | Webflow "Standard No-Charge": free, proprietary, revocable, no competing no-code builders | inside RAEA compositions only; our own sequencing grammar | PX3, PX5 | H | M |
| 8 | mrdoob/three.js | 3D for the web | MIT (some example assets non-commercial) | inside RAEA compositions | PX3 | M | S |
| 9 | ahujasid/mcp-for-blender | Blender driven over MCP | MIT (Blender is GPL; telemetry on by default; default port 9876) | headless provider and a runbook | PX9 | M | M |
| 10 | Comfy-Org/ComfyUI | node-graph engine for diffusion models | GPL-3.0 | separate program over HTTP, never code | PX9 | M (H on CUDA) | M |
| 11 | heygen-com/hyperframes | HTML → video (Puppeteer, FFmpeg), 450 blocks, `check` codes | Apache-2.0 | QA checks ported, credited; optional renderer | PX1, PX3 | H | M |
| 12 | remotion-dev/remotion | videos as React components | Remotion Licence (source-available; paid "Automator" licence for companies of 4+) | no runtime; spring and one-owner ideas | PX3, PX5 | M | S |
| 13 | remotion-dev/skills | agent skills for Remotion | no LICENSE file (Remotion Licence) | ideas rewritten in our words | PX12 | H | S |
| 14 | pbakaus/impeccable | design polish skill and detector, motion capture | Apache-2.0 | motion detectors and reviewer contracts, written fresh | PX1, PX8, PX12 | H | M |
| 15 | Leonxlnx/taste-skill | style skill: copy lint, budgets, finish gate, deal with history | MIT (bundled photos and icons excluded) | copy lint, deal, gate | PX1, PX10 | H | S–M |
| 16 | Nutlope/hallmark | anti-generic-look skill: tiers, waivers, variety ledger, eval | MIT | tiers, ledger, eval harness | PX1, PX10 | H | S–M |
| 17 | nexu-io/open-design | local AI design workspace (Node, Electron) | Apache-2.0 at the top, mixed inside | skip the app; HyperFrames authoring rules, provider capability lists | PX9, PX12 | M | S |
| 18 | FFmpeg/FFmpeg | media libraries and tools | LGPL-2.1+ / GPL-2+ | skip in the product; test oracle only | — | L | — |

### Licence rules applied

- **MIT and Apache-2.0:** ideas and text may be adapted with credit (`ATTRIBUTION.md`; Apache
  NOTICE text kept). Code is still written fresh to our rules: impeccable's motion code uses
  `unwrap`, which our never-crash rule forbids.
- **Non-commercial** (anything2explainer, onetake, lemo-opuscar's style text): ideas from the
  README and docs only, nothing copied. Commercial use includes a monetized channel.
- **Remotion:** no runtime or adapter in AutoCut. A public AutoCut driving it would hand the
  Automator licence to every company of four or more that uses it. Files move between them
  (OTIO, ProRes 4444).
- **GSAP:** never bundled or shipped from our repository; a user installs it into a RAEA
  composition. Our sequencing grammar (PX5) is written from its public documentation.
- **GPL programs** (ComfyUI, FFmpeg, Blender): separate programs over HTTP or a command line,
  never linked; their source is not read.
- **Assets:** no blocks or design systems named after other brands, no non-commercial models or
  assets (some Sketchfab models, three.js example assets, Flux.1-dev, the larger Depth Anything
  V2 models), no bundled music libraries (Content ID risk). Telemetry off in every tool that has
  it.

## 3. The pipeline we are building

| Stage | What runs | Gate |
|---|---|---|
| 0 Brief | topic, album or outing; length, language, platform | — |
| 1 Material | FrameAtlas: what is real, which pictures exist, what is absent; fact sheet; the one signature moment | absent list feeds the script |
| 2 Direction | `autocut.deal` picks look, title style, transition family, music, register and end card against history (PX10); a motion thesis in numbers | plan critique; fingerprint against the last three videos |
| 3 Script | narration and titles | `script.factLedger`, `copy.lint`, `script.lintMalayalam`, a fresh reader |
| 4 Voice and timeline | narration rendered and heard back; `script.timeline` gives shot ranges (PX7) | narration clarity (existing) |
| 5 Edit | Direct, Quick Short or Showcase with `reference` targets (PX6) and fitted music (PX2) | `sequence.checkMotion`, pacing |
| 6 Motion graphics | `motion.render` through RAEA or EffectCraft (PX3, PX4); spring keyframes and `keyframes.choreograph` (PX5) | `motion.check`, motion filmstrips |
| 7 Missing footage | `generate.*` with ComfyUI and Blender providers (PX9) | provider capability check |
| 8 Craft QA | `qa.motionEnergy`, `sequence.checkCraft`, `sequence.checkCarry` (PX1, PX6) | tiers: Floor blocks, Reflex fix or waive |
| 9 Review | 1080p review; finish reviewer; frame-pinned comments (PX8) | ship / fix / rebuild / recapture, two rounds |
| 10 Deliver | 4K final | `delivery.preflight` with the tiers |
| 11 Publish | metadata, thumbnail | `publish.preflight` with copy lint |
| 12 Learn | fingerprint and outcome logged; feedback retires defaults (PX10) | — |

## 4. Work items

Ids are `PX1`–`PX12`; each ships like the generation items: its own commit, tests that fail
before the change, read-only runs on a few real saved projects before thresholds are trusted
(the lesson of the 15 November review), docs, and the full gate before a push.

### PX1 — Craft QA that measures motion and text (do first)

- `qa.motionEnergy {path|sequence, fps?}` (idea from huashu-motion's `om qa`): frames reduced to
  96×54 grey; single-frame pops, cuts per minute, still stretches of 1 s or more (declared holds
  exempt), motion per second and its contrast, a flat opening (first 5 s), a film that barely
  moves, an empty first frame, the loop seam; a contact sheet at 4 frames a second. A leg of
  `media.preflightAnimation` and `delivery.preflight`.
- `sequence.checkCraft {sequence, holds?}` (hyperframes' check codes, lemo-opuscar's reading
  time): a line on screen shorter than its reading time, text under 2.6 % of the frame's short
  side, contrast of text against the footage actually behind it over its whole life (4.5:1, 3:1
  for large text), an empty first frame, more than 3 s with nothing changing, more than 15 s
  with no sound.
- `sequence.checkMotion` (numbers from impeccable, taste-skill and hallmark; timeline only, no
  render): overshooting eases unless a playful style is declared, entrance and exit durations
  against bands in frames, exits not longer than entrances, total stagger, the same entrance on
  every title, looping text, pops at clip edges.
- `filmcraft-cli frames --motion`: seven frames per animated title, transition or RAEA overlay
  (before, first, three middles, last, settled) with pixel detectors: a jump at the start or end,
  a mid-move step, a luminance dip in a dissolve, a layer still fading after its shot, an
  entrance replayed across a cut.
- `copy.lint {text|sequence|publish}` (taste-skill and hallmark): banned and stock phrases
  including video ones ("in this video we'll", "let's dive in", "don't forget to subscribe"),
  device budgets (warn over budget, fail at twice plus one), numbers not in the fact ledger,
  narration that describes the shot. The user's own phrases (`autocut.taste`) override the lists.
  Also a leg of `publish.preflight` and `publish.fixCopy`.
- Severity tiers in `delivery.preflight`: **Floor** blocks (contrast under 3:1, text in a
  platform's UI zone, invented figures, the existing privacy and click checks), **Reflex** is
  fixed or waived with evidence and a reason in `build/decisions.json` (more than five waivers
  fails), **Finish** is reported only. Loudness jumps and caption differences stay warnings.

### PX2 — Music fitted to the film

- `media.beats` adds `downbeats`, `sections [{start, end, energy}]`, `drops`, `musicEnd`,
  `ending` (cadence, fade, cut) and `confidence`.
- `clip.remix` adds film events (`burst`, `peak`, `landing`, `stop`) that a real lift of the
  track lands on, `ending` (own, button, fade), `maxStretchPct`, returns `alternatives
  [{duration, joins, ending}]` (the film lengths that need no join), and refuses a join that jumps
  more than 6 dB in half a second after rendering. Joins only inside stable stretches, with short
  equal-power crossfades. (Ideas from huashu-motion's fitting rules.)
- Direct, Quick Short, Showcase and trailers use it. Shorts stay silent, as the user asked.
- Later: `music.fromScore {abc, marks}`: a score written by the agent in ABC notation, with
  marks that land on scene labels, played on our SoundFont engine.

### PX3 — Motion graphics as commands

- Upgrade RAEA from 0.1.0 to 0.1.8, checking the release's SHA-256; smoke-test one HTML page with
  GSAP, one `<Three program>` scene and one React Three Fiber scene.
- `motion.render {source, composition?, canvas, size, fps, alpha, at?, revision, wait, place?}`,
  `motion.stills` and `motion.check` replace `examples/raea/render.py`: a job under the GPU lock
  that runs `raea check`, renders into a new revision folder, probes the result, imports and
  places it; an MCP long job.
- A `kind: "composition"` entry in `providers.json`: RAEA by default, EffectCraft (PX4),
  HyperFrames as an option (decision D2), never Remotion.
- `composition.import` follows one rule from Remotion's OTIO export: every visible or audible
  element has exactly one owner. Media used inside a composition comes in as real clips; only the
  graphics come in as transparent overlays.

### PX4 — EffectCraft

- **Stage 1, external tool (S–M).** `effectcraft.renderTemplate {template, values, out, size,
  fps}` renders an `.ectemplate` with data to ProRes 4444 or a PNG sequence, then imports and
  places it: data-driven lower thirds, end cards, chapter cards. `effectcraft.renderComp
  {project, comp, range}` for particle, 3D and keyed shots. A selection goes over as FCP7 XML or
  FCPXML (`file.exportFcp7Xml`, `file.exportFcpxml`). A runbook for its MCP beside ours (its
  control port is 9877; check whether its MCP needs automation roots).
- **Stage 2, video object tracking (M).** SAM 2.1 tracking from `effectcraft-sam2` and
  `-segment` for `clip.objectMask` and `masks.track` (same candle and wgpu versions as ours; that
  it builds in our workspace is not verified).
- **Stage 3, in-process compositions (L, decision D3).** Link `effectcraft-render`, `-effects`,
  `-gpu`, `-expr`, `-text` and `-path` with our own frames through its `FootageSource` trait,
  shaped like upstream's planned Dynamic Link (filmcraft#1024): 308 effects, 280 on the GPU, in
  the timeline. Against it: it brings craft-libs crates (our sync rule skips them), embeds Inter
  fonts (the craft-fonts rule), its internal APIs change daily, and our gates do not lint git
  dependencies, so the adapter needs its own hostile-input tests.

### PX5 — Keyframes that move well

- A spring interpolation (damping ratio, stiffness, mass, overshoot clamp) in the keyframe model,
  from damped-oscillator physics. Default damping 0.9 or more (no visible overshoot); 0.5–0.7
  only for a declared playful style.
- A perceptual scale option (scale interpolated in log space).
- `keyframes.choreograph {target, tweens: [{prop, to, duration, ease, position}], labels,
  stagger}`: the relative placement grammar agents already know from GSAP (`"<"`, `">"`,
  `"+=0.2"`, labels), resolved into ordinary editable keyframes. Written from public docs.
- Eases: seeded wiggle, `steps(n)`, custom curves. Revisit `graphics.animateText`'s `Back Out`.

### PX6 — Edit like a reference

Ideas from onetake (non-commercial, so nothing copied).

- `media.measureStyle {item|path, from?, to?}`: stillness ratio, cuts (jump cuts too), shot
  lengths and their spread, bursts, an energy map, how fast moves settle, motion blur, the
  loudness curve, the palette.
- `reference: item` on `autocut.direct` and `autocut.quick`: those numbers become targets, so
  "like the reel I liked" means something measurable.
- `sequence.checkCarry`: a continuity score per cut (salient boxes and colour).
- FrameAtlas keeps the rhythm fingerprints of approved videos.
- Later: `sequence.retimeToNarration`, a word-aligned warp of clips, graphics and keyframes, for
  builds in two languages.

### PX7 — Topic to explainer

Ideas from anything2explainer (non-commercial, so nothing copied).

- `script.timeline {text, voice, language}`: chapter and paragraph markup goes through the
  narration voice and comes back as shot ranges in ticks, subtitle blocks, chapter starts,
  sequence markers and one JSON per shot for compositions.
- `graphics.chapterBar`, and `script.factLedger`: every number or claim on screen must be in a
  sources list.
- A recipe `builtin:topic-explainer`: research → script → voice → timeline → one composition per
  shot (PX3) → craft QA (PX1) → review (PX8) → delivery.
- Rules to check: no still longer than 3 s, about a second of settle before a shot leaves, the
  hero element at least a quarter of the frame height.

### PX8 — Review pointed at frames

- `review.comment {time, clip?, box?, text}` from a click in the Program Monitor (with an
  automation id), `review.inbox {wait, timeout}`, `review.look {id}` (the frame and its
  context) and `review.done {id, note}`; MCP tools; the inbox lives in the build folder.
  (Idea from huashu-motion.)
- A finish-reviewer contract for agents: a fresh context with frame sheets, motion filmstrips,
  the QA JSON and the decisions; it answers ship, fix (at most eight fixes and one "keep"),
  rebuild or recapture; two rounds at most; the run ends on one `GATE:` line, and only the
  user's own words close it early. (impeccable, hallmark, taste-skill.)

### PX9 — More tools behind the provider list

- **ComfyUI** as a provider kind: AutoCut talks HTTP to the local server; the user keeps
  API-format workflow files, and `providers.json` says which node inputs take `{prompt}`,
  `{seed}` and `{image}`; progress from its socket or by polling; `POST /free` after each job so
  the GPU lock means something; started with `--offline`; a model-licence allow-list per
  workflow.
- A `capabilities` block for every provider (sizes, durations, aspects, frame inputs), so
  `generate.*` refuses what a tool cannot do before starting it (idea from open-design).
- **Blender**, headless: `blender --background [scene] --python <our script> -- <json>` writes
  PNG or EXR with alpha (and Cryptomatte) under the GPU lock; EffectCraft's tracked-camera export
  feeds it for match-moves. For interactive work, a runbook for the Blender MCP with telemetry
  off, safe mode on, a port other than 9876 and 9877, and CC0 or CC-BY assets only.
- **OpenScreen** for product demos: record and export with auto zoom as an external tool, and
  give it AutoCut's own click positions (`setPointerTrack`) for tutorials of AutoCut itself.
  Later a native `clip.autoZoomFromCursor` from its `.cursor.json` (MIT).

### PX10 — Taste memory

- `autocut.deal {for: look|titleStyle|transitionFamily|music|register|endCard}` chooses against a
  history so consecutive videos differ, inside what `autocut.taste` allows (the user's
  preferences always win: no whooshes or risers, silent Shorts, and so on).
- `autocut.fingerprint`: six axes per video (opening shape, title entrance, transition family,
  colour look, caption style, close); a warning when four or more match one of the last three.
- Feedback retires defaults, as the whoosh lesson did.
- An eval harness: fixed briefs on owned test footage, scored by PX1 and a vision judge on frame
  sheets, with and without the guidance, run when the pipeline changes. (hallmark's eval idea.)

### PX11 — More film looks

Ideas from lemo-opuscar, shaders written fresh, GPU first: silent film and newsreel, scratched
film, CRT, one-bit dither, false colour, tilt-shift from depth, double exposure; effects for film
damage, gate weave, flicker, dither, false colour and CRT. A "style card" schema for templates in
our own words (essence, colour, type, motion, camera, sound, pitfalls).

### PX12 — Agent runbooks

- Director and rhythm rules, adapted from huashu-motion (MIT, credited): the hook within 3 s, the
  densest five seconds about a third of the way in, every section boundary changing at least two
  of shot scale, speed, ease, camera, density, colour and music texture, three directions shown
  before building, and "three notes a demanding director would give" before anyone watches.
- Composition authoring for RAEA: open-design's HyperFrames rules (Apache-2.0, credited),
  greensock's gsap-skills (MIT, credited), Remotion's skill ideas rewritten: motion only from the
  frame number, no random or clock values, timelines paused and registered, safe areas and text
  minimums, one focal element per scene.
- The Motion Playbook's lessons: High effort rather than Max; numbers instead of adjectives (the
  motion thesis); spring damping ratios (PX5); check motion, not stills (PX1); an explicit
  finish condition (PX8); Blender by script (PX9); teams of agents with fresh reviewers; a
  verification budget (one inspection, one batch of fixes, one confirming round), and
  deterministic checks that cost no tokens.
- Into `docs/autocut-agent-guide.md` and `docs/raea-workflow.md`, with `ATTRIBUTION.md` entries.

## 5. Order

| Phase | Items | Why first |
|---|---|---|
| A | PX1 (`qa.motionEnergy`, `sequence.checkCraft`, `copy.lint`, tiers), PX12, PX3 (RAEA 0.1.8, `motion.render`), PX5 (spring, choreograph) | cheap, and every later item is judged by these checks |
| B | PX2, PX4 stage 1, PX10, PX8, PX1 (`sequence.checkMotion`, filmstrips) | better music, data-driven graphics, variety, review |
| C | PX6, PX7, PX9 | new kinds of video and tools |
| D | PX4 stages 2–3, PX11, `music.fromScore`, `sequence.retimeToNarration`, a HyperFrames provider | after the decisions below |

## 6. Decisions for the user

- **D1** Upgrade RAEA to 0.1.8 (still separately installed, FSL-1.1-ALv2). Recommended.
- **D2** HyperFrames as a second renderer needs Node, Chrome and ffmpeg on the machine, with its
  telemetry off. Recommended: RAEA's own HTML path only, HyperFrames later if RAEA cannot draw
  something.
- **D3** EffectCraft in-process (stage 3) takes craft-libs crates and embedded fonts, against two
  standing rules. Recommended: external first (stages 1–2), decide after.
- **D4** The Blender MCP add-on defaults to port 9876, AutoCut's control port. Recommended:
  give Blender another port, keep ours.
- **D5** Install ComfyUI (a separate GPL program) and choose its models against the allow-list.
- **D6** The style word lists against the user's own phrases: the user's phrases win.
- **D7** New checks as warnings for one round on real projects, then Floor checks start blocking.

## 7. Not taken

FFmpeg in the product (it stays a test oracle); a Remotion runtime or adapter; GSAP in this
repository; the open-design app; HeyGen's cloud, hosted MCP and paid templates; the cloud 3D
generators behind the Blender MCP; bundled music libraries; brand-named blocks and design
systems; any code or text from the non-commercial repositories; impeccable's binary, hooks and
browser extension (downloads, a live mode that runs agents with the sandbox off).

## 8. Not verified

That EffectCraft's crates build as git dependencies here; EffectCraft's CLI GPU flag and whether
its MCP refuses to start without automation roots; whether RAEA accepts HyperFrames' original
`data-*` names and resolves `node_modules` offline, and whether three.js 0.187 runs under its
pinned 0.185; whether HyperFrames' PNG-sequence output needs ffmpeg; the licences of SAM 3,
Depth Anything 3, SeedVR2 and lemo-opuscar's fetched sample libraries; Blender's wording on
add-on licensing. Thresholds proposed beyond those quoted from the repositories are starting
values to tune on real projects.

## Sources

huashu-motion, anything2explainer, openscreen, onetake, lemo-opuscar, effectcraft, GSAP (and
greensock/gsap-skills), three.js, mcp-for-blender, ComfyUI, hyperframes, remotion, remotion
skills, impeccable, taste-skill, hallmark, open-design, FFmpeg — all on github.com, read
22 November 2026; RAEA from shinyobjectz/raea and its skills.
