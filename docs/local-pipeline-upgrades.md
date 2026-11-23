# Local editing pipeline: catalogue, revisions and delivery

Execution and verification are tracked in [the November upgrade plan](../plan/pipeline-upgrades-2026-11.md).
Correctness fixes and regression evidence are recorded in [the full review](../plan/full-review-2026-11-11.md).
All commands below run through the engine, CLI `exec`, MCP `command_run` or desktop bridge.
Media, projects, compositions and generated reports stay outside git.

## Catalogue selections become editable sequences

`filmcraft-cli catalog search '<JSON filters>'` queries FrameAtlas on `127.0.0.1:7318`.
`asset`, `resolve`, `job` and `stats` take the corresponding JSON arguments. `--catalog-port`
changes only the loopback port. The adapter never accepts a remote service URL.

Save a selection like this in the video's build directory:

```json
{
  "name": "Korea expanded cut",
  "sequence": {"width": 3840, "height": 2160, "fps": 30},
  "entries": [
    {"id": "<catalogue asset id>", "scene_key": "<current scene key>"},
    {"id": "<another asset id>", "speech_key": "<current speech key>"},
    {"id": "<dashcam asset id>", "scene_key": "<scene key>", "dashcam": true}
  ]
}
```

For multipart recordings also retain `part_filename`. Never reconstruct a part by ordinal.
A 360 sample is a point citation: choose explicit `sourceIn` and `duration` ticks around it.
Photos default to four seconds; choose `duration` to change their hold. Native import retains
portrait/rotation and 360 interpretation through the existing media and timeline commands.
Use the existing 360 stabilization workflow before approving an edit.

```sh
filmcraft-cli catalog retrieve selection.json '/existing/editing/folder'
# Repeat the same command to resume pending jobs. Completed originals are reused.
filmcraft-cli catalog job '{"id":123}'
filmcraft-cli catalog cancel 123
filmcraft-cli catalog assemble '/path/from/ready/result.edit-plan.json' --save-as '/new/edit.fcproj'
```

Retrieval receipts and the ready edit plan live in the chosen editing folder. They retain asset,
scene/speech keys, source parts, exact trims, verification and source hashes. A stale reference
fails with FrameAtlas's recovery message; an expired picker selection is never replaced with an
unrelated source. Requests for several references in the same original are serialized. A changed
retained file fails integrity verification. Cancellation leaves completed sources and receipts.
A failed or cancelled retrieval is forgotten: the same `catalog retrieve` command asks for it
again, and completed downloads are kept.
Different scene/speech citations of the same verified source filename reuse the retained file,
with each citation freshly resolved for its own trim. Different camera parts remain separate
retrievals. Reuse still verifies the retained file's digest; it never substitutes another asset.
The adapter requests selected originals only and does not annotate or reanalyse the catalogue.

Engine `jobs.list` exposes `cancelRequested` separately from `finished`: a cancellation
request is visible immediately while the worker finishes stopping. Neither field implies
that partial results may be applied to a project that has since been replaced.

`catalog.importEditPlan {plan: …}` accepts the ready version-1 `autocut.catalogEditPlan`.
It assembles a new sequence in one undo step, reuses imported files, retains provenance on the
sequence, and omits source sound for `dashcam` or `mute` entries. Failed assembly rolls back
project edits and history. Related 360 source files remain dependencies of the edit and cache.

## Measured source-aware colour

`clip.autoColor` retains its existing bounded measurement and drift checks. Source transfer and
primaries go through AutoCut's native colour management; HDR to SDR uses its existing BT.2390
mapping, and HDR sequences preserve their working space. Additional parameters:

- `target`: `sequence` (default), `sdr`, `pq` or `hlg`. An explicit target enables auto tone mapping.
  HDR targets retain scene exposure unless an explicit exposure offset is requested.
- `protectScreens`: `{ "<clip id>": [[x,y,width,height], …] }`, up to eight normalized rectangles
  in the original source. These become editable masks on the measured Lumetri pass. They remain
  when the pass is repeated without new rectangles; an explicit empty list clears protection.
  Rectangles for a clip the pass does not grade are refused, naming the clip.
- Each report includes detected source colour information, target and protected regions.

Protection is static until animated. Use `masks.track` or keyframes for a moving device screen.
Subclips resolve their parent source dimensions for placement and screen masks. Measurement uses
original media, independently of the preview proxy toggle.
Inspect it at source scale. A mask protects the new correction; it cannot reconstruct highlights
already clipped in the original or undo another manually applied grade. High 10 export retains
HDR precision, while SDR compositing still starts with RGBA8; selecting a 10-bit container does
not restore absent source precision.

## Audio that survives revisions

Add any missing tracks with `sequence.addTracks` (for example `video:0,audio:3` after a one-audio-track catalogue import). Assign stems with `audio.configureStems`:

```json
{"roles":{"A1":"source","A2":"narration","A3":"music","A4":"effects"},"muteSourceClips":[123]}
```

Roles persist as named tracks and Essential Sound types. Muting a source video identity also
mutes its linked audio. Existing muted clips stay muted. Narration, source and effects remain
editable. Baked music inside an old master cannot be separated by this operation: return to its
saved timeline or original stems before combining revisions.
Assigning or changing a role preserves existing repair, gain, ducking and mute settings.

`sequence.continuousMusic` builds deterministic loops from one imported music item. Supply
`item`, an existing audio `track`, optional `start`/`end`/`sourceIn` ticks, `crossfade` ticks and
`gainDb` (default -25). `replaceTracks` explicitly lists old music tracks to clear, along with the
destination. It keeps their sources in the project and is one undo step. Loop overlaps use
editable equal-power envelopes across two persistent music tracks (A/B), because clips within a single track may not overlap. The partner is reused on later revisions (a partner track deleted since is made again). Reapply mix/ducking policy to both returned track identities after rebuilding the bed.

`sequence.checkContinuity` measures every audio/video boundary, or specified `times` in ticks.
It reports 100 ms RMS differences and sample discontinuities with exact times. `maxJumpDb`
defaults to 6 dB. This local metric supplements `audio.analyzeMix`'s LUFS/true-peak/stem analysis.
Declare intentional breaks as `exemptions: [{start,end,reason}]`; the report retains them.
Each finding carries `click`: a step between neighbouring samples, as opposed to a loudness jump.

## Chapter and composition cache

`render.cachePlan` accepts a sequence and optional `start`/`end` ticks, resolved export `settings`,
external composition parameters and `dependencies` (all local asset/module/font/sound paths).
It returns a key, its identity (the digest of each selected item and of the project's resources,
with the range, settings, source hashes and external parameters) and a verified hit or miss.
An edit of any size has a small identity. Source hashing streams bytes;
large originals do not become a single allocation. A source is read once: its digest is kept in
the cache folder (`source-digests.json`) and used again while the file's identity, size, modified
time and change time are all the same, so a later plan and the commit read only the sources that
changed. `sourceDigests` in the result counts the sources read (`hashed`) and not read (`reused`).
A file changed in the last two seconds is always read. `rehash:true` reads every source whatever
is remembered: use it for sources on a volume that keeps no change time (FAT, exFAT), where bytes
rewritten under the same size and modified time are not seen. Outside macOS and Linux nothing is
remembered. The key includes actual engine source and
pinned dependencies and build features/target/flags, so two development builds at version 0.12.0 cannot share incompatible renders.

The fingerprint retains selected timeline clips, sequence settings, effects, colour, captions,
audio and nested sources. Related 360 parts and explicit composition dependencies are hashed.
A later, unrelated video edit does not invalidate an earlier chapter. Active transitions retain
both handles. Nested sequences and complete audio tracks are conservative dependencies, retaining
processor history, tails, lookahead and ducking. Embedded LUT text, project settings, source
graphics and motion tracks are conservative resources too. Always list every external RAEA import and asset with `--dependency`;
an unlisted dynamic dependency cannot be discovered from a rendered MP4.

After a successful render, `render.cacheCommit {key,path,manifest:<cachePlan result>}` checks
inputs again, copies to a staging directory, verifies the output hash, syncs it and atomically
publishes it under an OS lock, whose file is removed when the writer is done. `render.cacheInspect {key}` verifies output bytes before reuse.
Missing/changed sources fail; missing or corrupt entries miss. Failed copies remain unpublished.
Editing proxies are excluded from original-media render identity; an absent unused proxy does not
block a final export. A cache-hit delivery stages and verifies its bytes before publishing, creates
missing destination directories, and refuses occupied destinations.
Cache is under the configured Media Cache in `render-cache-v1`, separate from saved projects.
It is disposable. Keep originals, `.fcproj`, compositions and delivery manifests outside it.
The cache stores files; the RAEA helper packs transparent frame sequences for cache transport.

## RAEA review bundle

See [raea-workflow.md](raea-workflow.md). `motion.render` does the helper's handoff as an engine
command and an MCP long job (PX3.2): the same revision folders, `canvas`/`size`/`preview` sizing,
probe, preflight (`spec`, or `expected` and `holds` as parameters), `preflight.json`,
`contact-sheet.png` and `autocut-import.json` in the helper's shape, then the import and, with
`place`, the clip on the timeline, all under the GPU lock it takes itself. It also keeps RAEA's
`check.json`, its render manifest and its log beside the render. It does not use the render cache
yet (`--cache` stays with the helper). `composition.import` (PX3.3) brings a composition in as
the edit it is instead of one flattened render: RAEA's `export layers` writes what it draws as
movies of their own and names the footage and sounds it uses in `layers.json`; AutoCut imports
those files as real clips at their times, trims, speeds, gains, opacity and boxes, and each drawn
layer as one clip on its own track (every visible or audible element has one owner). A footage box
RAEA crops is shown uncropped (`approximated` in the result). The helper now supports:

```sh
python3 examples/raea/render.py scene.tsx '/new/review-folder' \
  --cli '/absolute/filmcraft-cli' --composition scene \
  --canvas 960x540 --size 3840x2160 --preview --cache \
  --spec preflight-spec.json --dependency logo.png --dependency sounds.wav
```

Omit `--preview` for the master. Review uses draft quality with a longest side of 960. The canvas
is the reference layout and `--size` is the master output. Responsive TSX must lay out from
`useVideoConfig` dimensions; fixed-size output is rejected if it does not match. Inspect the
contact sheet. HTML uses its own dimensions.

The helper writes a contact sheet and `preflight.json`, then the import manifest only on success.
`media.preflightAnimation` checks duration, dimensions/rate, decode failures, required alpha,
sampled frozen sections, declared text geometry and audible cues, and adds `qa.motionEnergy`
warnings ([Craft QA](#craft-qa-motion-and-text)) that never fail it. A spec can contain:

```json
{
  "expected":{"duration":2032128000000},
  "holds":[{"start":1524096000000,"end":2032128000000,"reason":"intentional final logo hold"}],
  "textBounds":[{"name":"CTA","start":0,"end":2032128000000,"box":[0.1,0.75,0.8,0.15]}],
  "soundCues":[{"name":"impact","time":254016000000}]
}
```

Times are ticks (254,016,000,000 per second). Geometry is normalized. Text checks validate supplied
geometry, not OCR. Sound checks find audio above -60 dBFS in the next 100 ms, not musical meaning.
Motion is sampled at approximately 10 Hz (up to 600 samples); declare intended holds. Skipped
text/sound evidence stays visible. Failed renders remain for diagnosis without an import manifest.
Hold the shared GPU lock while RAEA renders; cache hits still run native preflight.

## Craft QA: motion and text

These checks measure craft, not correctness. Every finding is a warning (`severity: "warn"`)
for this first round on real projects (decision D7 in
[the ingestion plan](../plan/video-automation-ingest-2026-11-22.md)): they never fail
`media.preflightAnimation` or `delivery.preflight` (only with `floorBlocks: true` do the Floor ones
of `delivery.preflight` block, see [Severity tiers](#severity-tiers)). The thresholds below are starting values, to
be tuned once real projects have been measured with them.

`qa.motionEnergy` measures a rendered file (`path`) or a sequence (`sequence`), exactly one of
the two:

```sh
filmcraft-cli --project edit.fcproj exec qa.motionEnergy sequence=12 sheet='/new/build/motion.png'
filmcraft-cli exec qa.motionEnergy path='/new/build/loop.mov' loop=true
```

Each analysed frame is reduced to 96×54 grey (54×96 for a portrait frame). Every frame is
analysed up to 7,200 frames; a longer input is sampled at a lower rate (`sampledDown: true`, the
rate used in `fps`), and `fps` (0.1 to 240) asks for a lower rate. A file is read through the
media pool's decoders (hardware decoders where the host registered them). A sequence is rendered
on the GPU frame renderer when the host has one, else on the CPU renderer, without captions, as a
horizontal delivery is exported; `renderer` in the result counts the frames each drew.

| Code | Warns when | Starting value |
|---|---|---|
| `pop` | one frame differs from the frames on both sides (a flash frame, a stray frame between shots) | both scene steps at least 0.2 and at least 2.5 × the local median + 0.05 |
| `still` | the picture holds for 1 s or more outside the declared holds | mean grey difference to the stretch's first frame under 0.003 of full scale |
| `flatOpening` | the first 5 s neither move nor cut | mean motion under 0.05 a second, no cut |
| `barelyMoves` | the whole film hardly moves and hardly cuts | mean motion under 0.05 a second, fewer than 2 cuts a minute |
| `evenMotion` | motion keeps the same pace every second (films of 10 s or more) | per-second motion varies by less than 0.2 of its mean |
| `blackFirstFrame`, `emptyFirstFrame` | the first frame is black (or transparent), or one flat colour | mean grey under 0.04; grey standard deviation under 0.01 |
| `loopSeam` | with `loop: true`, the step from the last frame back to the first stands out | over 4 × the median step between frames, and over 0.02 |
| `decode` | frames could not be read or rendered and were left out | any |

Motion is the mean grey change between frames about 0.1 s apart, scaled to one second, as a share
of full scale (0 is still); steps across a cut or a pop are left out of it. The result's `measures`
give motion for each second (`motion.perSecond`, one value per `bucketSeconds`, which grows past an
hour so there are at most 3,600), its mean, its contrast (standard deviation over mean) and the
opening's mean, cuts per minute and the cut times, and the loop seam whether or not `loop` is set.
Cuts use the Scene Edit Detection measure at sensitivity 50 with a shortest shot of 0.2 s, so a
fast dissolve or a whip pan can read as a cut. Pops are looked for only when every frame is
analysed; below the source's rate a single frame cannot be told, and `skipped` says so. `holds`
(`{start, end, reason}` in ticks, as in the preflight spec) exempt still stretches only: the part
of a still outside a hold is still reported, and a hold never hides a pop. `sheet` writes a contact
sheet (PNG, or JPEG by its extension) of four frames a second, 16 to a row and at most 600 tiles,
so a film longer than 150 s gets fewer frames a second. Each code lists at most 200 findings;
`counts` has them all.

As a leg, `media.preflightAnimation` runs it on the same render with the spec's `holds` and
`expectMotion`, lists its findings under `warnings` and the whole result under `motionEnergy`, and
does not judge a transparent overlay's first frame (`motionEnergy: false` turns the leg off).
`delivery.preflight` adds a `motionEnergy` check, `warning` or `pass` and never `fail`, rendered
from the originals. Its holds are the declared ones plus the still-picture spans the gate's own
`motion` check already treats as holds, so a photo story does not warn for every photo.
`motionEnergy: false` skips it, recorded as skipped.

`sequence.checkCraft` reads a sequence's text and timeline (`sequence` defaults to the active one;
`holds` as above):

```sh
filmcraft-cli --project edit.fcproj exec sequence.checkCraft \
  'holds=[{"start":0,"end":1016064000000,"reason":"title card"}]'
```

| Code | Warns when | Starting value |
|---|---|---|
| `readingTime` | a title (every text layer of one graphic clip together) or a caption is on screen for less time than it takes to read | 3 words a second plus 0.5 s |
| `smallText` | a text layer, or a visible caption track, is smaller than the floor | font size under 2.6 % of the frame's short side |
| `lowContrast` | text against the picture behind it, at its worst sample, is under the WCAG ratio | 4.5:1; 3:1 for large text (3.9 % of the short side, 3.03 % when bold) |
| `emptyFirstFrame` | nothing is on the picture tracks at the first frame, every clip there starts at opacity 0, or the first frame renders black or one flat colour | mean grey under 0.04, spread under 0.01 |
| `staticTimeline` | more than 3 s with nothing changing on the timeline, outside the declared holds | 3 s |
| `silence` | more than 15 s with no clip on an enabled, unmuted audio track | 15 s |

Reading time counts words separated by spaces, so text in a script written without spaces counts
low. Every caption track is read for reading time, hidden ones too, since they go out as sidecar
subtitles; size and contrast are judged only on visible caption tracks, as they are burned in.
Large text follows WCAG's 18 pt against 12 pt body text (1.5 × the 2.6 % floor) and 14 pt bold.

Contrast is measured against the closest approximation of the picture behind the text that the
renderer allows. The sequence is rendered (on the GPU frame renderer when the host has one) at 3
to 8 times across each text's life, about one every 2 s, with every title's text layers turned
off and no captions. The backdrop therefore holds the footage, shapes and other graphics under the
text; it also leaves out the text of other titles under it, which is rare. Inside the text's box
the 10th and 90th percentile luminance stand for its darkest and brightest parts, and a text
colour between the two counts as 1:1. A background box is laid over the backdrop at its opacity,
the text's own opacity (fill, layer and clip) is mixed in, and a stroke or caption outline at
least 5 % of the font size counts as a backdrop of its own colour (the better ratio is kept).
Shadows, glows, per-character colours, the highlight colour of animated captions and titles inside
nested sequences are not measured. At most 400 frames are rendered: past that each text gets one
sample, and past 400 texts only an even share of them is measured (`checked.contrast` and
`skipped` say how many). `checked.contrast.ratios` lists each measured text's worst ratio, passing
or not, for tuning the thresholds.

The timeline checks read the timeline, not pixels. Something changes when a clip, caption or
transition starts or ends, a movie plays (a still, a frame hold or a colour matte does not), an
animated parameter moves, a title's text animates, or an animated caption or a nested sequence
plays (both taken as always changing). The timeline is sampled every 0.1 s and at every edit
point. `holds` exempt static stretches only. A clip of silence counts as sound: levels are not
measured. `delivery.preflight` runs it as two checks: `textContrast` (contrast under 3:1, Floor)
and `craft` (the rest), see [Severity tiers](#severity-tiers).

`sequence.checkMotion` reads how a sequence moves, from the timeline alone (`sequence` defaults
to the active one; `style` is `calm` or `playful`; `bands` sets the caller's own bands):

```sh
filmcraft-cli --project edit.fcproj exec sequence.checkMotion
filmcraft-cli --project edit.fcproj exec sequence.checkMotion style=playful 'bands={"entrance":[10,36]}'
```

| Code | Warns when | Starting value |
|---|---|---|
| `overshoot` | an ease passes its mark or bounces: a text animator's or transition's Back, Elastic or Bounce easing (or a Custom Bézier with a handle outside 0..1), Pop Motion's overshoot or Spring Motion's bounce, a spring keyframe, a custom curve (`keyframes.choreograph`'s `back`, `elastic`, `bounce`) that leaves 0..1 or turns back; not with a playful style | damping ratio under 0.9; 0.5 % past the mark, 2 % back |
| `entranceDuration` | a title's or overlay's entrance (its text animator's unit, a transition into it, its first keyframed move from its first frame) is outside the band | 8–30 frames at 30 fps, scaled to the rate (0.27–1 s) |
| `exitDuration` | the same for an exit | 6–24 frames at 30 fps (0.2–0.8 s) |
| `exitOutlastsEntrance` | an exit takes longer than the entrance it mirrors (one text layer's two animators, a clip's two transitions, its edge keyframes) | at most the entrance; 60–75 % reads best |
| `longStagger` | a text animator's letters, words or lines take longer from the first to the last starting | 0.8 s (a typewriter is left to its typing speed) |
| `sameEntrance` | three or more titles, all entering the same way | 3 titles |
| `loopingText` | a looping text animator moves a title for its whole clip | quicker than 8 s a cycle |
| `edgePop` | a keyframed position, scale, rotation or opacity changes by half its move or more between a clip's first two or last two frames | 50 % of its keyframed range, and at least 10 points of opacity, 5 % of scale, 5° or 1 % of the short side |

The numbers come from public motion guidance, read on GitHub and written fresh: impeccable's
(Apache-2.0) focal entrances of 500–800 ms and exits faster than entrances; hallmark's (MIT)
reveals of 0.4–0.8 s, exits at 60–75 % of the entrance, a total stagger of at most 500 ms and
ambient loops of 8 s or slower; taste-skill's (MIT) ban of bounce, elastic, back and overshoot.
The bands are wider than the web's, since a video title is read from further away; all are
starting values to tune on real projects. A playful style (`style: "playful"`, or
`autocut.taste {"set": {"motion": "playful"}}`) lets eases overshoot; `style: "calm"` checks them
even when the taste is playful. A title is a graphic clip with an enabled text layer; its entrance
for `sameEntrance` is its text animator's kind, else a transition into it, else its keyframes. An
overlay is a clip of a picture with alpha (`motion.render`, `composition.import`): its transitions
and keyframes are checked, not what moves inside it. Text units are counted from the laid-out
text, letters as shaped clusters, as the renderer animates them. Clips inside nested sequences are
not read. `delivery.preflight` runs it as the `motionCraft` check (Reflex).

Motion filmstrips show and measure those moves as they render. `sequence.motionStrips` (and
`filmcraft-cli frames --motion`) takes every animated title's entrance and exit, every transition
and every overlay with alpha (or the `items` named: a clip id, `{clip, edge}`, `{transition}`), at
most 32, and writes one sheet per item into `out`, a new or empty folder: seven frames (just
before, the first, a quarter, the middle, three quarters, the last, and settled: 0.25 s after an
entrance or a transition, the first frame after an exit), each labelled with its number and time.
It is a job (`jobs.list` shows its progress and, finished, its result under `motionStrips`);
`wait: true` runs it in the call, as the CLI does:

```sh
filmcraft-cli frames --motion --project edit.fcproj --out /new/build/strips
filmcraft-cli frames --motion --project edit.fcproj --items 12:in,t5 --scale 0.25 --out /new/build/strips-2
```

Every frame of each move (at most 64, spread evenly over a longer one) is also rendered small and
reduced to the 96×54 grey grid of `qa.motionEnergy`; a step is the mean grey change between
neighbouring frames:

| Code | Warns when | Starting value |
|---|---|---|
| `jump` | the step into a move's first frame, or out of its last, stands out from the move's own steps | at least 0.02 of full scale and 3 × the median step; not where another clip cuts |
| `midStep` | one step inside the move stands out from the two on each side (a hitch) | at least 0.01 and 3 × their median; only when every frame was read, and not from or to a frame of one flat colour (the black of a dip, whose first step out reads large in sRGB) |
| `dissolveDip` | a dissolve (cross, film, blur, dither dissolve or morph cut) gets darker in the middle than both its ends | mean grey 0.04 under the darker end |
| `fadeAfterShot` | a title's or overlay's exit is still on screen when another clip cuts under it | the layer's share of the frame at the cut at least a tenth of its share when the exit starts |
| `entranceReplay` | an entrance at an edit brings in what the frame before it already showed (one title cut in two, both halves entering) | the frame before within half the layer's share of the settled frame, a frame of the move at least a quarter away |

A layer's share of a frame is the frame against the same frame rendered without the layer's
clip, so the title is told from the footage under it. The sheets and the analysis render on the
GPU frame renderer where the host has one, the CPU reference renderer otherwise, without
captions. Findings are warnings and say what to change.

`copy.lint` reads the words of a video: `text` (taken as narration), `publish` (`title`,
`description`, `caption`, `tags`), `script` (a lines file, a showcase plan or plain text, as
`script.lintMalayalam` reads it) and `sequence` (its titles, collected as `sequence.checkCraft`
collects them, and every caption cue of every caption track, hidden ones too). Give at least one.

```sh
filmcraft-cli exec copy.lint 'publish={"title":"Printer build","description":"Hey guys! We rebuilt the extruder."}'
filmcraft-cli --project edit.fcproj exec copy.lint sequence=12 'facts=["Bought in 2014","600 km"]'
```

| Code | Warns when | Starting value |
|---|---|---|
| `stockPhrase` | a stock phrase of video copy (openers like "in this video we'll" and "let's dive in", greetings, calls to action like "smash that like button") or of marketing copy ("game-changer", "next level") | 39 video and 27 general phrases |
| `filler` | a filler word ("basically", "kind of") or a hedge ("I think", "probably") | 13 fillers, 14 hedges |
| `describesShot` | narration (caption cues, a script, `text`) says what the picture shows ("here we see", "as you can see", "this is a shot of") | 23 phrases |
| `overBudget` | a device is used more than its budget per 100 words; `level: "fail"` from twice the budget plus one | exclamation marks 1, questions 1, em dashes 2, emoji 2, all-caps words 1; a text under 100 words gets the budget of 100 |
| `unledgeredFigure` | a figure written in digits is not in the fact ledger | compared by value ("1,200" is 1200) |

The lists are FilmCraft's own, written for it; a pattern matches whole words, any case, with
straight or curly apostrophes, and hyphens split words ("game-changer" is "game changer"). A
phrase never matches across a sentence or clause break. "Kind of" and "sort of" after a word that
makes them a noun ("a kind of printer") are not hedges. Budgets are counted per unit: the text,
the publish title, description and caption each, the script, all titles together and each caption
track; tags get only the stock phrase and figure checks. A question counts whether or not it is
rhetorical; an all-caps word is one of at least 5 letters, or a short shout word ("WOW", "FREE"),
since shorter ones are usually acronyms, and a line with no lower-case letter at all is a title set
in capitals and is not counted. Emoji are counted by code point ranges, a flag as one. Web
addresses are left out of every count.

The fact ledger is `facts` (a list of texts, numbers or `{text}`), or else `build/facts.json`
beside the saved project (the same list, or `{"facts": [...]}`). Without one, figures are not
judged and `skipped` says so; a ledger file that cannot be read is said too, never taken as
empty. Figures are digits, not number words. Clock times and chapter stamps ("0:45"), codes and
formats where letters touch the digits ("i3", "4K", "1080p", "3D"), hashtags, web addresses and a
number after "part", "episode", "step", "day" and similar words are not figures; "10km", "2x" and
"25%" are.

The person's own phrases win over the lists (decision D6): the taste profiles' `phrases` (set
with `autocut.taste {"set": {"phrases": [...]}}`, the project's and the user's default together),
each wording rule's `say`, and `allow` are never flagged. A list match that overlaps one is
dropped (`allowed.suppressed` counts them), and marks, all-caps words and figures inside one
(with the exclamation or question marks it ends with) are not counted.

Every finding is a warning, the `fail` level of a budget too (D7). Each code lists at most 200
findings; `counts` has them all; at most 8 MB of text is read in one call. `budgets` lists every
unit's count of every device for tuning. `publish.preflight` adds a `copy` check (warn or pass,
never an error) when it is given publish copy, with the lint under `copy`. `publish.fixCopy`,
given the copy, answers it with its safe fixes at once, with or without a file to fix: a whole
sentence at the start or end of a description or caption made only of greetings and calls to
action (and joining words like "so" and "and") is cut, never the last one left, and runs of
exclamation marks become one, in the title too. Nothing is reworded; a sentence with anything
else in it stays. `delivery.preflight` runs it on the active sequence as two checks: `figures`
(figures not in the ledger, Floor) and `copy` (the rest), see [Severity tiers](#severity-tiers).

## Standard checked delivery

Use `recipe.run` with `builtin:delivery-quality-gate`, or call:

```sh
filmcraft-cli --project edit.fcproj exec delivery.preflight \
  'changedNarration=[101,102]' report='/new/build/delivery.qa.json'
filmcraft-cli --project edit.fcproj exec delivery.prepare \
  path='/new/final-4k.mp4' 'qa={"changedNarration":[101,102]}'
```

The gate checks source availability, privacy at/around cuts and regular intervals, audio boundary
continuity, mix clipping, changed narration, subtitle alignment, screen-mask persistence, sampled
motion/highlights, motion energy ([Craft QA](#craft-qa-motion-and-text), warnings only), the
craft and copy checks, waivers and safe zones for Shorts, each in a
[severity tier](#severity-tiers). `tts.checkClip` hears the **retained source audio**,
without generating a replacement; its cache is keyed by actual samples, script and judge. Missing
models or unclear required readings block delivery. It uses the existing local hearers, second
opinion and speaker gate, and never downloads a model implicitly. A clip is judged by its take's
whole speech: a trimmed or razor-split narration clip passes or fails with its take, and `part:true`
in the result says the clip holds only part of it. Forward takes up to 120 seconds and 1,400
characters are supported; split longer narration for precise checks.

`changedNarration` limits rechecking to edited lines; omitted, every native narration is checked.
`privacy`, `narration` and `mix` may be explicitly disabled by project policy, recorded as skipped.
Motion/highlight/black-frame anomalies are review warnings because still photos, dark scenes and
dips can be intentional. At audio boundaries a click blocks delivery; a loudness jump alone is a
review warning, since speech onsets, music hits and sound effects move more than 6 dB in 100 ms
on finished edits. An edit of any length is measured, 2000 boundaries at a time. Caption tracks
that differ from the first in cue count or timing are a review warning too, with each track's
cue count and last cue end: a translation is rarely cut where its source is.
Required failures block export with the report. Visual inspection and
listening still matter; a sampled pass does not prove every frame or pronunciation is perfect.
Source checks traverse subclips and nested sequences with cycle/size limits. Delivery refreshes
loaded originals before measurement and privacy checks. Disabled photos and still backgrounds do
not automatically exempt a frozen video above them; a genuinely static photo interval can be held.

`delivery.prepare` defaults to 4K H.264/AAC, normalized -14 LUFS, and verified cache reuse. The
cache never blocks a delivery: without a cache folder, or when planning or publishing the cache
entry fails, the export goes ahead and `cache.note` in the result and the manifest says why. Horizontal
videos use sidecars; `short:true` chooses portrait 4K with burned captions. Every caption track gets
its own SRT and cue-id manifest. The final file also gets the normal publishing preflight and
`*.delivery.json`. Existing video, subtitle and manifest paths are refused, using the exporter's
resolved filename (including its extension). The one exception is a delivery that stopped after
its export: when a later step fails (a subtitle, the publishing preflight, the manifest), the
manifest is left with `complete:false`, and the same command run again (same path, settings and
checks, on the same edit) finishes it from the master already there, without checking or exporting
again (`resumed:true`). After any other edit it is refused, as an occupied path. A finished
manifest has `complete:true`. QA reports use exclusive creation and numbered revisions
on retry; read the returned `reportPath` rather than assuming the requested path was used.
`captions.export` accepts `overwrite:false` for exclusive creation (its existing default is `true`).
Movie/frame/caption/report exports reject destinations matching the open project or imported source,
including existing native hard-link and symbolic-link aliases. It prepares local files only: Photos/YouTube
uploads continue through their authorized adapters, private by default, with the project's label
policy unchanged.

### Severity tiers

Every check of `delivery.preflight` has a `tier`:

- **Floor** blocks delivery. The checks that blocked before the tiers keep blocking exactly as
  they did; the new Floor checks warn for this first round on real projects (decision D7) and
  block only with `floorBlocks: true`.
- **Reflex** is fixed, or waived with a reason and evidence in `build/decisions.json`.
  It never blocks: an open Reflex finding is a warning.
- **Finish** is reported only.

| Check | Tier | Blocks |
|---|---|---|
| `sources` | Floor | yes, as before (offline media) |
| `audioContinuity` | Floor for clicks, Finish for loudness jumps | a click, as before; a loudness jump never |
| `mix` | Floor | yes, as before (true peak over 0.05 dBTP, or not measured) |
| `privacyAtCuts` | Floor | yes, as before |
| `changedNarration` | Floor | yes, as before |
| `safeZones` (Shorts) | Floor | yes, as before |
| `screenProtection` | Floor | yes, as before |
| `textContrast` (new) | Floor | with `floorBlocks`: text under 3:1 against the picture behind it, from `sequence.checkCraft` |
| `figures` (new) | Floor | with `floorBlocks`: figures in titles and captions not in the fact ledger, from `copy.lint`; skipped without a ledger |
| `waivers` (new) | Floor | with `floorBlocks`: a refused waiver, an unreadable decisions file, or more than five waivers |
| `motion` | Reflex | never |
| `colour` | Reflex | never |
| `blackFrames` | Reflex | never |
| `motionCraft` (new) | Reflex (every `sequence.checkMotion` finding) | never |
| `craft` (new) | Reflex (`readingTime`, `smallText`, `lowContrast` from 3:1 up to the WCAG ratio, `emptyFirstFrame`); Finish (`staticTimeline`, `silence`) | never |
| `copy` (new) | Reflex (`stockPhrase`, `describesShot`, `overBudget` at its fail level); Finish (`filler`, `overBudget` over the budget) | never |
| `motionEnergy` | Finish | never |
| `subtitleAlignment` | Finish | never |

`craft` and `copy` are new legs: `sequence.checkCraft` runs on the active sequence with the
declared holds and the still-picture spans the `motion` check holds (merged, at most 2,000), and
`copy.lint` reads its titles and caption cues with the ledger (`facts`, else `build/facts.json`
beside the project) and the person's phrases (`allow` and the taste profiles). Their findings
carry their own `tier`. `motionCraft` runs `sequence.checkMotion` on the active sequence, every
finding Reflex. `craft: false`, `motionCraft: false` and `copy: false` skip them (and `textContrast` or
`figures` with them), recorded as skipped. A new Floor check that could not be measured (the
craft check failed, the ledger file could not be read) is a warning too, a failure with
`floorBlocks`.

The decisions file is `build/decisions.json` beside the saved project, or the file `decisions`
names:

```json
{"waivers": [
  {"check": "colour", "reason": "a white backdrop on purpose", "evidence": "frame sheet 0:00 to 0:03 reviewed"},
  {"check": "craft", "code": "readingTime", "time": 508032000000, "reason": "the line is also spoken", "evidence": "narration clip 41"}
]}
```

A waiver names a Reflex `check`, optionally a finding's `code` and a `time` (ticks; it matches a
finding whose time, or span, holds it within a frame), and needs both a `reason` and `evidence`
(at most 2,000 characters each). A finding it matches gets `waived: {waiver, reason, evidence}`,
and a check whose findings are all waived is `waived`. A waiver without a reason or evidence, one
for a Floor or Finish check (a Floor check is fixed, never waived; a Finish one has nothing to
waive), one naming no check of the gate, or with a bad `code` or `time`, is refused and is a
`waiverRefused` finding of the `waivers` check. More than five waivers is a `tooManyWaivers`
finding: the Reflex checks are being waived instead of fixed. A file that cannot be read (or a
named one that is not there) is `decisionsUnreadable`; with no file and an unnamed path nothing is
waived and the check passes. `detail.unmatched` lists accepted waivers that matched nothing.

`floorBlocks` (default `false`) is a parameter of `delivery.preflight`, passed in `qa` to
`delivery.prepare`, as the gate's other settings are. When it is true the new Floor checks block
too: `textContrast`, `figures` and `waivers` fail instead of warning. Loudness jumps and caption
differences stay warnings either way.

The report has `summary: {floor, reflex, finish, waived}`: the open findings of each tier (a check
that lists no findings counts as one), and the waived ones, plus `blocking` (the checks that
failed) and `floorBlocks`.

Prepare a portrait sequence or use the existing reframe workflow before Short delivery;
the output-size setting does not re-edit a landscape story. Enable the intended caption
language for burning; other language tracks can remain hidden and still receive sidecar files.

## Upstream additions

- Scrub audio opens on its own worker; a drag reuses its stream and releases it before playback.
  Pending grains are coalesced so a slow device open retains only the latest request.
- Caption export chooses format and caption track; invalid/deleted requested tracks fall back
  to the displayed first track. ASS/SSA remains available.
- MP3 uses FilmCraft's clean-room encoder. The `mp3` format and `MP3 48 kHz 320 kbps` preset are
  available through UI/CLI. It has encoder delay and no gapless tag.
- H.264 `h264Profile: "high10"` and `h264IntraOnly: true` use the built-in encoder. Hardware
  paths decline these settings. Eight-bit formats remain available; export thread caps apply.
- Input, decoder, waveform, graphic and traversal hardening is tracked in the execution plan.
