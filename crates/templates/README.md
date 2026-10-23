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
| `poster` | the moment (ticks) its thumbnail shows; default a third of the way in |
| `assets` | the attribution manifest of `assets/` |

### Slots

| field | meaning |
|---|---|
| `id`, `kind` | `[a-z0-9_-]`; `video`, `photo`, `text`, `logo`, `music` or `captions` |
| `count`, `minCount` | how many media the slot takes, and the fewest accepted (0 = optional) |
| `placements` | the clips it fills: `clip` (id in the template sequence), `media` (which of the slot's media; several placements may share one), `layer` (text slots: the text layer's uid), `box` (picture slots: `[x, y, width, height]` in fractions of the frame — the picture fills that box and is cropped to it: a collage cell, half of a split screen, a picture-in-picture) |
| `constraints` | `minDuration` / `maxDuration` (what a placement may become when re-timed), `orientation` (`any`, `landscape`, `portrait`, `square`), `aspect` (`"16:9"`), `strict` (a wrong shape is an error, not a warning), `maxChars` (text) |
| `fit` | `short`: media shorter than its placement `loop`s, `stretch`es (no slower than `minSpeed`, then loops), `hold`s its last frame, is `remix`ed (music) or is an `error`; `portion`: which part of longer media (`start`, `center`, `end`; the caller's `in` point wins); `frame`: `fill` (cover, cropped around the focus point), `fit` (whole), `none` |
| `beatSync` | the edges of this slot's placements are cuts on the beat grid |
| `default` | `{"text": …}` for text slots, `{"generate": {"mood", "bpm", "seed", "genre"?}}` for music (a bed made with `audio.generate`; moods `epic`, `tense`, `uplifting`, `calm`, `happy`, `dark`, `dreamy`, `energetic`, `sad`; genres `cinematic`, `electronic`, `pop`, `ambient`, `lofi`, `piano`) |
| `captions` | captions slots: `{"source": [slot ids], "maxChars": 32, "lines": 1}` |
| `ai` | optional hints for filling the slot automatically (`autocut.quick`), below |

### AI hints

`"ai": {…}` on a slot says what an automatic fill should put there. Every field is optional,
checked against the slot's kind (a picture hint on a text slot is an error), and a template
without hints still fills by hand (the fill then goes by the pictures' quality).

| field | slots | meaning |
|---|---|---|
| `purpose` | video, photo, logo | what the shot is for: `hook`, `broll`, `reveal`, `talking-head`, `establishing`, `outro` (the scene index's purposes, `media.bestShots`) |
| `query` | video, photo, logo | words to find the shot by (`media.searchScenes`), ≤ 200 bytes |
| `energy` | video, photo, logo | `low`, `medium` or `high` (the shot description's energy) |
| `speech` | video | `want` (a shot with someone speaking: its sound is the point) or `avoid` (a cutaway) |
| `minQuality` | video, photo, logo | the least picture quality wanted (the scene index's score, 0–1) |
| `chronological` | video, photo, logo | the picks of every slot saying so, in the template's order, keep the order they were filmed in |
| `maxHold` | video, photo, logo | the longest one of its shots may be held when the template is re-timed, seconds (0.5–600; an automatic fill keeps it there with a faster tempo or a shorter video; without it `autocut.quick` holds a shot 5 s at most, or as long as the template made it) |
| `prompt` | text | what the local model writes here (≤ 1000 bytes) |
| `maxChars` | text | the longest text it may write (at most the slot's own `maxChars`) |
| `about` | text | the picture slot whose shots the texts describe, one text per shot (`names` about `robots`) |

```json
{"id": "broll", "kind": "video", "count": 8, "ai": {"purpose": "broll", "energy": "high"}, …}
{"id": "title", "kind": "text", "ai": {"prompt": "The name of what is revealed, 1 to 3 words", "maxChars": 16, "about": "reveal"}, …}
```

**Captions slots** (one per template) caption what is said in the `source` slots' media (video or
music slots). They have no placements: the captions go on the template's first caption track, in its
`captionStyle`. The template's own captions there are a preview: they stay while no source is filled
and go when one is; the engine then transcribes the sources' sound (`transcript.generate`: Whisper or
the host's transcriber) and writes captions of `maxChars` a line from the words in the sources' clips
only. Callers may give the cues themselves (`[{"start", "end", "text", "speaker"?}]`, timeline
ticks of the result). Without speech-to-text the track stays empty and the apply warns.

**Elements sounds** are clips of an item whose metadata `Element` names a sound element
(`sound.whoosh-short`, `sound.pop`… — `elements.list`): in the package that item is generated silence
of the sound's length, sound only; the engine synthesizes the element when the template is applied
and swaps it in (the WAV keeps the metadata, so saving the sequence as a template turns it back into
a stand-in). Stickers and animated text presets of the Elements library are ordinary graphic clips
(their layers, linked to the graphics template they came from), and their text layers can be text
slots.

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
template's own), `frame`, `box` (the picture's box in this layout: two speakers side by side in
16:9 stack in 9:16), and per graphic layer `position` and text `size` (pixels in that variant; an
area text's box keeps its share of the frame and the text wraps anew). A graphic clip's own Motion position and anchor keep their place as
fractions of the frame. Full-frame generated placeholders are drawn at the variant's size.

In a box, Motion's scale is relative to the box (100 = exactly filling or fitting it), the picture's
focus lands as near the box's centre as covering the box allows, Motion's anchor is the picture's
point at the box's centre (so a pop-in scales about the box) and a Crop effect cuts what lies
outside the box.

## Beat grid

`beatGrid.beats` are the music's beats as timeline ticks (`media.beats`, or a steady tempo). When the
template is applied its beat-synced cuts snap to them. When the music slot gets a different track,
the engine measures that track's beats (the Remix analysis behind `media.beats`, in exact ticks; or
takes `template.apply`'s `musicBeats` when the caller knows them, a bed it generated) and the cuts
are re-quantized: with `"mode": "beats"` each cut keeps its beat number (a slower song makes
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
5. Lay out the variant, name the chapter markers after their text slots (a sequence marker whose
   comment is `text:<slot>:<index>`), set the caption style and the captions slot's cues, record
   the template, variant and export preset in the sequence's metadata.

The engine wraps all of it — imports, beat measuring, the generated bed (whose exact beats the cuts
move onto, the length kept), Remix, the Elements sounds, transcribing the captions, the look — in
one undo step. A required slot left unfilled keeps its placeholder with a warning (a preview of the
template); `requireAll` makes it an error.

## Built-in starters

Original designs made in code ([`builtin/`](src/builtin/)), from generic conventions (a lower third,
a countdown, a split screen) without reference to anyone's templates: placeholders are the
procedural demo scenes and colour mattes, titles and stickers come from the Elements library or are
drawn in code with Inter (bundled, OFL), sounds are Elements sounds, transitions come from the
built-in and GL Transitions catalogs, and unfilled music is generated in the app. Every starter has
16:9 and 9:16 layouts; the social ones also 1:1 and 4:5. The template browser shows them by category:
Reels & Shorts, YouTube, Vlog, Business, Social, Events and Tech.

| id | length | slots |
|---|---|---|
| `builtin:photo-beat-slideshow` | 16 s, 120 bpm | `photos` ×8 (beat-synced, slow push-in), `music`, `title`, `outro` |
| `builtin:channel-intro` | 5 s | `logo` (fit, pop-in), `name`, `tagline` (one graphic, two layers), `background`, `music` |
| `builtin:top-3-listicle` | 18.5 s, 120 bpm | `items` ×3 (#3, #2, #1 = media 2, 1, 0), `labels` ×3, `title`, `music` |
| `builtin:talking-head-captions` | 30 s | `speaker` (with its sound), `hook`, `name`, `role`, `cta`, `captions` (of the speaker), `music` (kept low) |
| `builtin:product-promo-15` | 15 s, 120 bpm | `shots` ×4 (beat-synced, GL transitions, whooshes), `logo` (boxed, twice), `product`, `features` ×4, `offer` (on a burst sticker), `cta`, `music` |
| `builtin:travel-vlog-intro` | 10 s, 120 bpm | `clips` ×6 (beat-synced), `hook` (three punch-in words), `place`, `date`, `music` |
| `builtin:travel-vlog-outro` | 10 s | `clips` ×3 (crossfades), `thanks`, `next`, `music` |
| `builtin:top-5-listicle` | 22 s, 120 bpm | `items` ×5 (#5 … #1), `labels` ×5, `title`, `music` |
| `builtin:before-after-reveal` | 10 s | `before`, `after` (full, a wipe, then boxed side by side / stacked), `labels`, `split_labels`, `title`, `music` |
| `builtin:lower-third-chapter-kit` | 40 s | `video`, `name`, `role`, `chapters` ×3 (cards and chapter markers) |
| `builtin:end-card-subscribe` | 12 s | `next` ×2 (boxed in the end card's frames), `background` (blurred), `heading`, `subheading`, `labels`, `music` |
| `builtin:tutorial-pip` | 60 s | `screen` (fit), `webcam` (picture-in-picture), `title`, `steps` ×3, `captions` (of both), `music` |
| `builtin:podcast-two-speakers` | 60 s | `host`, `guest` (side by side; stacked when vertical), `show`, `host_name`, `guest_name`, `captions`, `music` (intro sting) |
| `builtin:photo-collage` | 10 s | `photos` ×4 (boxed in a 2×2 grid, popping in), `title`, `music` |
| `builtin:quote-card` | 8 s | `background`, `quote` (area text), `author`, `music` |
| `builtin:countdown` | 8 s | `background` (blurred), `title`, `subtitle`, `music` — 3, 2, 1, GO with ticks, a riser and a boom |
| `builtin:event-recap` | 16 s, 120 bpm | `clips` ×12 (beat-synced, glitch transitions), `logo`, `event`, `date`, `outro`, `music` |
| `builtin:day-in-my-life` | 18 s, 100 bpm | `moments` ×6 (beat-synced), `title`, `times` ×6, `music` |
| `builtin:unboxing` | 15 s, 120 bpm | `package`, `opening`, `reveal`, `details` ×2 (beat-synced, with their sound), `hook`, `reveal_text`, `verdict`, `music` |
| `builtin:qa-session` | 24 s | `answers` ×3 (with their sound), `questions` ×3 (speech-bubble cards), `captions`, `music` |
| `builtin:robot-showcase` | 26 s, 120 bpm | `intro`, `robots` ×5 (beat-synced, glitch transitions), `title`, `subtitle`, `names` ×5, `facts` ×5 (spec cards), `outro`, `outro_sub`, `music` |
| `builtin:tech-reveal` | 17 s, 120 bpm | `hook`, `broll` ×8 (beat-synced, glitch and pull cuts, zoom pulses), `reveal`, `hook_line`, `title`, `outro`, `outro_sub`, `music`; tech look |
| `builtin:event-recap-short` | 16 s, 120 bpm | `establishing`, `moments` ×10 (beat-synced, filming order, swipes), `event`, `when`, `closing`, `music` |
| `builtin:talking-head-tip` | 24 s | `takes` ×3 (with their sound, filming order), `cutaways` ×2 (over the joins), `hook`, `cta`, `captions` (Bold animated), `music` (kept low) |
| `builtin:gadget-showcase` | 16.5 s, 120 bpm | `hook`, `features` ×3 (spec cards), `hero`, `names` ×3, `facts` ×3, `name`, `question`, `question_sub`, `music`; tech look |
| `builtin:before-after-short` | 10.5 s | `before`, `after` (filming order; full, a swipe, then stacked), `labels`, `title`, `music` |
| `builtin:day-in-30s` | 28.8 s, 100 bpm | `moments` ×12 (beat-synced, filming order, dissolves and a light leak), `title`, `closing`, `music`; tape look |

The last six are the **Shorts and Reels starters made to fill themselves** (`AI_READY_IDS`):
vertical first, every picture and text slot with AI hints, Creator transitions (`crt_*`), cuts
and zoom pulses on the beat, text inside the safe area every platform leaves in the vertical
layout (15–65 % of the height, 12–82 % of the width), and the tech or tape look on an
adjustment clip named `Look: tech` / `Look: vhs` (an automatic fill swaps it for another look).

## Saving (`save::save`)

The authoring flow (`template.save`, Sequence ▸ Save as Video Template…, the Templates panel's
form). Clips flagged with a `slot:` marker (`template.markSlot`, the Timeline's clip menu ▸ Template
Slot, or left by an earlier apply) become slots with neutral generated placeholders (the demo
scenes, colour mattes); pieces of one placement merge back into one clip. A placeholder's Motion is
rewritten the way a template means it — scale relative to the slot's framing of the media (100 =
exactly filling the frame, or the box) — so it shows the way the clip did; a cropped picture is a
picture box (the visible rectangle). Other clips using the user's own files are left out with a
warning unless bundled: a bundled file goes in `assets/` with its SHA-256 and attribution (its
`.attribution` sidecar's, else the author's own work under the template's licence). Captions are
the user's words: with a captions slot (`captions: {source}`, or the base's) their timing stays
with neutral preview text, otherwise they go; their style stays. Elements sounds become stand-ins
again, text slots show the base template's own text again.

What the format carries beyond the timeline comes from the author's options, else from the
template the sequence was made from (its `Template` metadata), else defaults: slot names, prompts,
counts, constraints, fit, beat sync and defaults (`slots`); the captions slot; the layouts
(`variants`: 16:9, 9:16, 1:1, 4:5 at the sequence's short side, the base's sizes, its per-clip
hints moved onto the saved clips; the layout the sequence is in is the design, hint-free); the
poster frame; the look and the export presets. The result goes through the strict manifest reader
and the package cross-checks, so a saved template always reads back. Saving an applied starter
gives back the starter's slots, boxes, captions slot, layouts, hints, poster, look and sounds
(`save_tests.rs` checks all 21).

## Sharing

Templates are plain files ([`video_template_share.rs`](../engine/src/video_template_share.rs)):
`template.export` writes one `.zip`; `template.install` takes a `.zip` from a file or an https link
(a job, through the host's pure-Rust HTTPS fetcher, at most 128 MiB), checks it whole in memory
(the ZIP caps, names leaving the package, the strict manifest, the sequence; built-in ids refused)
before anything is written, and writes it — re-encoded from what was read — into the user folder.
Nothing in a package is ever run. The panel shows a QR code of the install link ([`qr`](src/qr.rs):
an encoder written from ISO/IEC 18004:2015 — byte mode, levels L–H, versions 1–40, the eight masks
and their penalty rules). `template.exportOtio` / `sequence.exportOtio` write OpenTimelineIO
(`filmcraft-interchange`'s writer) with the slots in `metadata.filmcraft` (the timeline's
`template`, each placement clip's `slot`).

## Never crash

Packages are untrusted input, like project files: the ZIP reader caps entries (1024), entry size
(128 MiB) and the total inflated (256 MiB, so a bomb stops at the cap), checks every offset and CRC,
and refuses names that leave the package (`..`, absolute paths, drive letters, backslashes),
symbolic links, encryption, multi-disk archives and ZIP64. Manifests are capped at 1 MiB with bounded
counts and string lengths; sequence files at 32 MiB, 4096 items and 8192 clips; every number is range
checked, all time arithmetic saturates, and the tests mutate whole packages byte by byte under
`catch_unwind`.
