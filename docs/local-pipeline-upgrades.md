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
large originals do not become a single allocation. The key includes actual engine source and
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
publishes it under an OS lock. `render.cacheInspect {key}` verifies output bytes before reuse.
Missing/changed sources fail; missing or corrupt entries miss. Failed copies remain unpublished.
Editing proxies are excluded from original-media render identity; an absent unused proxy does not
block a final export. A cache-hit delivery stages and verifies its bytes before publishing, creates
missing destination directories, and refuses occupied destinations.
Cache is under the configured Media Cache in `render-cache-v1`, separate from saved projects.
It is disposable. Keep originals, `.fcproj`, compositions and delivery manifests outside it.
The cache stores files; the RAEA helper packs transparent frame sequences for cache transport.

## RAEA review bundle

See [raea-workflow.md](raea-workflow.md). The helper now supports:

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
sampled frozen sections, declared text geometry and audible cues. A spec can contain:

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
motion/highlights and safe zones for Shorts. `tts.checkClip` hears the **retained source audio**,
without generating a replacement; its cache is keyed by actual samples, script and judge. Missing
models or unclear required readings block delivery. It uses the existing local hearers, second
opinion and speaker gate, and never downloads a model implicitly. Forward clips up to 120 seconds
and 1,400 characters are supported; split longer narration for precise checks.

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
resolved filename (including its extension). QA reports use exclusive creation and numbered revisions
on retry; read the returned `reportPath` rather than assuming the requested path was used.
`captions.export` accepts `overwrite:false` for exclusive creation (its existing default is `true`).
Movie/frame/caption/report exports reject destinations matching the open project or imported source,
including existing native hard-link and symbolic-link aliases. It prepares local files only: Photos/YouTube
uploads continue through their authorized adapters, private by default, with the project's label
policy unchanged.

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
