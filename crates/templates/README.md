# filmcraft-templates — video templates (`.fcvideo-template`)

Whole-sequence starters with media slots: a photo-beat slideshow, a channel intro, a "top 3"
countdown. A template is an **ordinary project fragment plus slot metadata**, not a new timeline
model: its sequence is a FilmCraft project file, and the manifest says which of its clips are
placeholders, what fills them, how media conforms to them, how the layout changes per aspect, and
which look, caption style and export preset travel with it.

This crate holds the format, its strict reader, packages (folder or ZIP), the slot auto-fit, the
beat grid, applying a template to a project, the minimal "save as template", and the built-in
starters. It does no file I/O; the engine's `template.*` commands
([`crates/engine/src/video_templates.rs`](../engine/src/video_templates.rs)) read and write files
through the host services, import media, measure beats and run Remix. Layer L3 (it reads sequence
files with `filmcraft-format`).

## Package

A folder `Name.fcvideo-template/`, or the same files in a `.zip` (at the root, or inside one top
folder):

```text
Name.fcvideo-template/
  template.json        the manifest (below)
  sequence.json        a FilmCraft project file (.fcproj schema, any version this build reads)
                       holding the template sequence and its placeholder media
  assets/…             optional: openly licensed files the sequence uses, each listed in
                       template.json `assets` with its SHA-256, author, source and licence
  README.md, LICENSE…  optional documentation
```

Anything else in a package is an error. The placeholder media in `sequence.json` is generated
(`MediaRef::Generator`: the procedural demo scenes, colour mattes) or a listed bundled asset — never
a path to someone's file. Graphics (`Graphic` items with text and shape layers) and adjustment
layers are fine; nested sequences, subclips, multicam, LUTs, transcripts and narrations are not
(format version 1).

## Manifest (`template.json`)

Times are exact ticks (254 016 000 000 a second). Every field is checked, unknown fields are
errors, and every problem is reported at once with its path (`slots[2].fit.short: \`teleport\` is
not one of loop, stretch, hold, remix, error`).

```json
{
  "format": "filmcraft.videoTemplate",
  "formatVersion": 1,
  "id": "builtin:photo-beat-slideshow",
  "name": "Photo Beat Slideshow",
  "description": "…",
  "version": "1.0.0",
  "author": "FilmCraft contributors",
  "license": "MIT OR Apache-2.0",
  "category": "Reels & Shorts",
  "tags": ["slideshow", "beat"],
  "sequenceFile": "sequence.json",
  "sequence": 12,
  "duration": 4064256000000,
  "variants": [
    {"id": "16:9", "name": "Landscape (YouTube)", "width": 1920, "height": 1080, "exportPreset": "YouTube 1080p Full HD"},
    {"id": "9:16", "name": "Vertical (Shorts, Reels)", "width": 1080, "height": 1920, "exportPreset": "Social Vertical 1080×1920",
     "hints": [{"clip": 31, "layers": [{"layer": 1, "position": [0.5, 0.72], "size": 96}]}]}
  ],
  "defaultVariant": "16:9",
  "slots": [
    {"id": "photos", "kind": "photo", "name": "Photos", "prompt": "Up to 8 photos", "order": 0,
     "count": 8, "minCount": 1,
     "placements": [{"clip": 14, "media": 0}, {"clip": 15, "media": 1}],
     "constraints": {"minDuration": 127008000000, "maxDuration": 1270080000000, "orientation": "any"},
     "fit": {"short": "loop", "portion": "start", "frame": "fill", "minSpeed": 0.5},
     "beatSync": true},
    {"id": "music", "kind": "music", "name": "Music", "count": 1, "minCount": 0,
     "placements": [{"clip": 40, "media": 0}], "fit": {"short": "remix", "frame": "none"},
     "default": {"generate": {"mood": "uplifting", "bpm": 120, "seed": 7}}},
    {"id": "title", "kind": "text", "name": "Title", "count": 1, "minCount": 0,
     "placements": [{"clip": 31, "media": 0, "layer": 1}], "constraints": {"maxChars": 40}}
  ],
  "look": {"lumetri": "Golden Dusk"},
  "captionStyle": {"font": "Inter", "size": 64, "outline": 3, "background": false},
  "exportPreset": null,
  "beatGrid": {"bpm": 120, "beats": [0, 127008000000, 254016000000], "mode": "beats"},
  "assets": [{"path": "assets/whoosh.wav", "sha256": "…", "title": "Whoosh", "author": "…", "source": "original work", "license": "CC0-1.0"}]
}
```

| field | meaning |
|---|---|
| `id` | `[a-z0-9:._-]`; `builtin:…` for the starters, `user:…` for saved ones |
| `license` | the template's SPDX licence; NC / ND / proprietary are refused (AGENTS.md §1) |
| `sequence` | the item id of the template sequence in `sequenceFile`; `duration` must match it (to a frame) |
| `variants` | aspect layouts (≤ 8): frame size, export preset, per-clip `hints` |
| `slots` | typed placeholders (≤ 64), see below |
| `look` | `{"lumetri": name}` (`lumetri.presets`) or `{"preset": name}` (`presets.list`), applied to the picture slots |
| `captionStyle` | a caption track style (the project's `CaptionStyle`), set on every caption track |
| `exportPreset` | export preset name when a variant names none |
| `beatGrid` | the music's beats on the timeline (≥ 2, increasing, ≤ 20 000) and how cuts follow a new track |
| `assets` | the attribution manifest of `assets/` |

### Slots

| field | meaning |
|---|---|
| `id`, `kind` | `[a-z0-9_-]`; `video`, `photo`, `text`, `logo` or `music` |
| `count`, `minCount` | how many media the slot takes, and the fewest accepted (0 = optional) |
| `placements` | the clips it fills: `clip` (id in the template sequence), `media` (which of the slot's media; several placements may share one), `layer` (text slots: the text layer's uid) |
| `constraints` | `minDuration` / `maxDuration` (what a placement may become when re-timed), `orientation` (`any`, `landscape`, `portrait`, `square`), `aspect` (`"16:9"`), `strict` (a wrong shape is an error, not a warning), `maxChars` (text) |
| `fit` | `short`: media shorter than its placement `loop`s, `stretch`es (no slower than `minSpeed`, then loops), `hold`s its last frame, is `remix`ed (music) or is an `error`; `portion`: which part of longer media (`start`, `center`, `end`; the caller's `in` point wins); `frame`: `fill` (cover, cropped around the focus point), `fit` (whole), `none` |
| `beatSync` | the edges of this slot's placements are cuts on the beat grid |
| `default` | `{"text": …}` for text slots, `{"generate": {"mood", "bpm", "seed"}}` for music (a bed made with `audio.generate`) |

Placements are clips of the template sequence: video, photo and logo slots on video tracks, music on
an audio track, text slots on graphic clips with a text layer. Two text slots may share one graphic
clip, each naming its layer. **Slot dedup:** one medium fills every placement with its `media` index,
and with fewer media than `count` they are reused in turn (3 photos fill 8 places as 0 1 2 0 1 2 0 1);
a file named twice is imported once.

**Effects, transitions and text stay with the slot.** Filling a placement swaps its media only: the
clip keeps its effects, keyframes (re-based to the new in point), transitions, markers and linked
sound partner (which follows the media, or goes when it has no sound). Motion on a placeholder is
relative to the slot's framing: scale 100 is "exactly filling" (or fitting), and the template's
position (in its design frame) is where the picture's focus lands.

### Variants and hints

The template sequence is designed at its own frame size; each variant re-lays it out (the
Magic-Resize model). Pictures are framed per their slot's `frame` rule around the media's focus
point (`focus` / `focusByVariant` when applying: the same photo can be re-framed for 16:9 and 9:16);
graphic layers keep their place as fractions of the frame and scale with the frame's smaller ratio.
A hint overrides that for one clip: `position` (fractions of the frame), `scale` (a percentage of the
template's own), `frame`, and per graphic layer `position` and text `size` (pixels in that variant).
Full-frame generated placeholders are drawn at the variant's size.

## Beat grid

`beatGrid.beats` are the music's beats as timeline ticks (`media.beats`, or a steady tempo). When the
template is applied its beat-synced cuts snap to them. When the music slot gets a different track,
the engine measures that track's beats (the Remix analysis behind `media.beats`, in exact ticks) and
the cuts are re-quantized: with `"mode": "beats"` each cut keeps its beat number (a slower song makes
a longer video), with `"nearest"` each moves to the nearest beat (the length stays), always within
the placements' `minDuration` / `maxDuration`. The old → new cuts form a piecewise-linear time map
that re-times the whole sequence — titles, transitions, markers, captions and the music clip move
with the placements they belong to — and every time lands on a frame. Time before the music starts
does not move.

## Applying (`apply::apply`)

1. Check the fills (known slots, kinds, counts, text lengths, the media's shape against the
   constraints).
2. Copy the template's items (in a bin of their own) and sequence with fresh ids — applying twice
   never shares an id.
3. Snap or re-quantize the cuts (above).
4. Fill each placement: [`fit`](src/fit.rs) cuts it into pieces of whole frames that add up to
   exactly its length — trimmed from the chosen portion, looped, stretched, held or marked for Remix.
   Each piece carries a `slot:<kind>:<slot>:<placement>` clip marker, so the sequence can be saved as
   a template again.
5. Lay out the variant, set the caption style, record the template, variant and export preset in
   the sequence's metadata.

The engine wraps all of it — imports, beat measuring, the generated bed, Remix, the look — in one
undo step. A required slot left unfilled keeps its placeholder with a warning (a preview of the
template); `requireAll` makes it an error.

## Built-in starters

Original designs made in code ([`builtin.rs`](src/builtin.rs)); placeholders are the procedural demo
scenes and colour mattes, text is Inter (bundled, OFL). Each has a 16:9 and a 9:16 variant.

| id | length | slots |
|---|---|---|
| `builtin:photo-beat-slideshow` | 16 s, 120 bpm | `photos` ×8 (beat-synced, slow push-in), `music`, `title`, `outro` |
| `builtin:channel-intro` | 5 s | `logo` (fit, pop-in), `name`, `tagline` (one graphic, two layers), `background`, `music` |
| `builtin:top-3-listicle` | 18.5 s, 120 bpm | `items` ×3 (#3, #2, #1 = media 2, 1, 0), `labels` ×3, `title`, `music` |

## Saving (`save::save`)

The minimal flow (the full authoring flow is A3): clips flagged with a `slot:` marker
(`template.markSlot`, or left by an earlier apply) become slots with generated placeholders; pieces of
one placement merge back into one clip; other clips using the user's own files are left out with a
warning; captions are left out (their text is the user's), their style is kept. The variant of the
sequence's own size is joined by its rotated twin (16:9 ↔ 9:16).

## Never crash

Packages are untrusted input, like project files: the ZIP reader caps entries (1024), entry size
(128 MiB) and the total inflated (256 MiB, so a bomb stops at the cap), checks every offset and CRC,
and refuses names that leave the package (`..`, absolute paths, drive letters, backslashes),
symbolic links, encryption, multi-disk archives and ZIP64. Manifests are capped at 1 MiB with bounded
counts and string lengths; sequence files at 32 MiB, 4096 items and 8192 clips; every number is range
checked, all time arithmetic saturates, and the tests mutate whole packages byte by byte under
`catch_unwind`.
