# Captions

FilmCraft keeps captions on **caption tracks** in the sequence, shown in their own area above the
video tracks (as Premiere does). Caption tracks are saved in the `.fcproj` (project format
version 2; version 1 projects open with no caption tracks).

## Model

- **Caption track** (`CaptionTrack`, `crates/project/src/caption.rs`): name, format (`Subtitle`,
  `CEA-608`, `CEA-708`, `Teletext`), language, output eye (shown in the Program monitor and
  burned in on export), lock, sync lock, and a **track style**:
  font size (pixels at 1080 lines, scaled to the frame), text colour, background box and its colour,
  outline width and colour, alignment (left/centre/right), position (top/middle/bottom) and
  margin (a fraction of the frame height, default 0.08; `captions.setStyle` clamps it to 0–0.45),
  line spacing.
- **Caption**: in/out in exact `Tick`s, text (lines separated by `\n`; `<i>`, `<b>`, `<u>` kept
  as written), optional speaker, and the WebVTT cue id and cue settings, kept for round trips.
  Captions on a track never overlap.

## Files

| Format | Import | Export | Notes |
|---|---|---|---|
| SubRip `.srt` | ✓ | ✓ | Forgiving reader (BOM, UTF-16, Windows-1252, CRLF, missing indexes, `,`/`.` ms, missing hours). |
| WebVTT `.vtt` | ✓ | ✓ | Keeps cue ids, cue settings, `<v Speaker>`, STYLE/REGION/NOTE blocks. `line:N%` (with `,start` / `,center` / `,end`), `position:N%` and `align:` are honoured when drawing. |
| Scenarist SCC `.scc` | ✓ | ✓ | CEA-608 at 29.97 fps. Reads pop-on, roll-up and paint-on; writes pop-on with drop-frame (default) or non-drop timecode. |
| MacCaption `.mcc`, EBU STL `.stl`, TTML `.ttml`, DFXP `.dfxp` | ✓ | ✓ | See the crate README. |
| Advanced SubStation Alpha `.ass`, SubStation Alpha `.ssa` | ✓ | `.ass` | The style most lines use becomes the track style (font, size, colour, alignment, margin, outline, opaque box); `\b` `\i` `\u` become tags, `\N` line breaks, the actor the speaker; `\an`, `\pos` and other styles' placement become cue settings; other overrides and drawings are dropped. |

`File ▸ Import` (or dropping a file) recognises caption files by extension and content and adds a
new caption track to the open sequence (a sequence is created when none is open). Cue times are
snapped to the sequence's frames. `File ▸ Export ▸ Captions…` writes the chosen caption track in
the format given by the file extension (`.srt`, `.vtt`, `.scc`, `.mcc`, `.stl`, `.ttml`, `.dfxp`,
`.ass`).

Details of each format and the SCC encoder are in [`crates/captions/README.md`](../crates/captions/README.md).

## Editing

| Command | Menu / shortcut | Params |
|---|---|---|
| `captions.newTrack` | Sequence ▸ Captions ▸ Add New Caption Track… (⌥⌘A) | `format`, `name`, `language` |
| `captions.add` | Sequence ▸ Captions ▸ Add Caption at Playhead (⌥⌘C) | `track`, `text`, `time`, `durationSeconds` (3) |
| `captions.next` / `captions.previous` | Sequence ▸ Captions ▸ Go to Next/Previous Caption Segment (⌥⌘↓/↑) | |
| `captions.showAll` / `captions.hideAll` | Sequence ▸ Captions | |
| `captions.split` | Text panel ✂; Razor tool on a caption (⇧: every caption track) | `caption`, `time` (default: every caption under the playhead) |
| `captions.merge` | Text panel | `captions` (consecutive, one track) |
| `captions.setText` | Text panel text field | `caption`, `text`, `speaker` |
| `captions.setTimes` | Text panel in/out fields | `caption`, `start…`/`end…` (`Time` ticks, `Frame`, `Seconds`, `Timecode`) |
| `captions.trim` | drag a caption edge in the timeline | `caption`, `edge` (`in`/`out`), `delta` or `deltaFrames` |
| `captions.move` | drag a caption in the timeline | `captions`, `delta` or `deltaFrames` |
| `captions.duplicate` | ⌥-drag a caption in the timeline | `captions`, `delta` or `deltaFrames` — copies on the same tracks, overwriting what is there |
| `captions.delete` | Edit ▸ Clear / Ripple Delete with captions selected | `captions`, `ripple` |
| `edit.copy` / `edit.cut` / `edit.paste` / `edit.pasteInsert` | ⌘C / ⌘X / ⌘V / ⇧⌘V with only captions selected | captions paste at the playhead on the tracks they came from, overwriting (Paste Insert pushes later captions on); Paste places whichever of clips or captions was copied last |
| `captions.select`, `captions.goTo` | click / number button | `captions`, `add` / `caption` |
| `captions.setTrack` | caption track header (eye, lock) | `track`, `name`, `format`, `language`, `enabled`, `locked`, `syncLock` |
| `captions.setStyle` | Text panel style strip | `track`, `size`, `color`, `background`, `backgroundColor`, `outline`, `outlineColor`, `align`, `anchor`, `margin`, `lineSpacing`, `reset` |
| `captions.deleteTrack` | | `track` |
| `captions.import` / `captions.export` | File ▸ Export ▸ Captions… | `path`; import: `name`, `language` (default: the file's TTML `xml:lang`, else `en`); export: `format`, `track`, `dropFrame`, `everyFrame` (MCC only: a packet on every 29.97 frame to the sequence's end, as SDI and frame-wrapped MXF carry; a warning when the sequence is at another rate). TTML and DFXP exports write the track's language as `xml:lang`. SCC and MCC time codes count from the sequence's start timecode (its label: 01:00:00:00 at 25 fps is 01:00:00;00 at 29.97), and an imported SCC or MCC file whose captions all come at or after that label counts from it (`fromStartTimecode` in the result) |
| `captions.list` | (query) | `track` — tracks, styles and captions with timecodes (a `track` that names no caption track is an error) |

Tracks are addressed by id or `"C1"`, `"C2"`… (top first). Edits are limited by neighbouring
captions and a one-frame minimum. Sync-locked caption tracks follow insert and extract edits on
the media tracks (a caption across an insert point is split). `Add Edit to All Tracks` (⇧⌘K)
also splits captions.

## Timeline and Text panel

- Caption rows: header with `C1…` badge, lock, output eye, name and format; caption blocks in
  violet showing their text (Arabic and Hebrew in reading order from the block's right edge). Click selects (⌘/⇧ adds), drag moves (the whole selection; ⌥-drag
  copies), edge drag trims, the Razor tool cuts, double-click
  opens the Text panel. Automation ids: `timeline.caption.<id>`,
  `timeline.captionTrack.C1[.locked|.enabled]`.
- **Text panel ▸ Captions**: search, track picker (also creates tracks of each format), add /
  split / merge / delete / export buttons, the list of segments with editable in/out timecodes
  and text (committed when the field loses focus; Arabic and Hebrew show in reading order and are
  edited in logical order while the field has focus), and the track style strip (size, colour, box,
  alignment, position). The list keeps the caption under the playhead in view (centred as the
  playhead reaches it, not while a field is being edited). Automation ids: `text.tab.Captions`, `text.captions.add|split|merge|delete|export|search|track`,
  `text.captions.<id>.in|out|text|goto|row`, `text.captions.style.*`.

## Embedded closed captions

`file.exportMedia` `{"embedCaptions": true}` (Export mode: Captions ▸ **Embed in Output File**,
H.264 and H.265 only; the built-in preset *H.264 1080p with Closed Captions* turns it on) carries
the first enabled caption track inside the video stream as CEA-608 (CC1, field 1) and CEA-708
(service 1) closed captions: ATSC A/53 `cc_data()` in a `user_data_registered_itu_t_t35` SEI
message (country 0xB5, provider 0x0031, `GA94`, type 0x03; ANSI/SCTE 128-1, ATSC A/72) in every
access unit, before its first slice. The pop-on CEA-608 schedule of the SCC writer and the
two-window CEA-708 schedule of the MCC writer (both on the 29.97 fps line 21 grid) move to the
nearest video frame, `cc_count` = 600 / frame rate triplets a frame (20 at 29.97, 25 at 23.976,
10 at 59.94), padded; each packet gets the data of the frame it shows, through B-frame
reordering, so a caption appears on its first frame at 29.97 fps and within one frame at other
rates. The SEI goes into the stream after encoding, so the hardware H.264 / H.265 encoders get it
too. `filmcraft_captions::embed` also reads the captions back (`sample_cc`, `decode`). Sidecar
files (`captionSidecar`) and burn-in stay available.

## Burn-in

The Program monitor shows the visible caption tracks. On export, `file.exportMedia`
`{"burnCaptions": true}` (Export mode: **Burn Captions Into Video**) draws them into every frame.
Captions are set in Inter SemiBold (OFL, bundled) by the text engine (`crates/text`: shaping,
kerning, bidi); italics/bold markup is not yet drawn differently. The captions of a nested sequence
are part of the nest's picture: they are drawn wherever the nest is shown, in the Program monitor
and in every export, whether or not the outer sequence shows or burns in its own.

A track style's `xOffset` (`captions.setStyle`, a share of the frame width) moves the captions
sideways, e.g. onto the middle of a platform's safe span (YouTube Shorts' right margin is wider
than its left).

## Word timing

A caption can know when each of its words is spoken (`Caption::words`, relative to the caption,
so moving a caption moves its words; trims, splits, merges and extracts keep them where they are
spoken). `transcript.createCaptions` fills it from the transcript's word timestamps;
`captions.wordTiming {track, from: "transcript" | "narration"}` times existing captions from the
sequence transcript or from the narration clips' scripts (spread over each clip's speech by the
words' lengths, pauses kept: an estimate). ASS karaoke (`\k`, `\K`, `\kf`, `\ko`) imports into it
and a caption whose words still match its text exports as karaoke. `captions.list` reports
`timedWords`. Without timing, a caption's time is spread over its words by length.

## Animated captions

`captions.setAnimatedStyle {track, preset, options}` draws a track word by word (in the Program
monitor and the burn-in alike, `filmcraft_captions::animated`):

| preset | look |
|---|---|
| `bold` | uppercase heavy sans with a black stroke and a hard shadow, 2–4 words a page, the spoken word in the active colour popping to 1.1× with an overshoot; the page pops in |
| `clean` | sentence case with a soft shadow, the spoken word coloured |
| `glow` | a neon glow around the spoken word |
| `box` | the spoken word on a rounded box of the active colour |
| `karaoke` | the active colour sweeps through each word as it is spoken |
| `twoLine` | classic subtitles: the whole caption on up to two lines, a soft shadow |
| `off` | whole captions again |

Options: `wordsPerPage` (1–8), `activeColor`, `keywordColor` and `keywords` (drawn in it),
`emoji` (`{"word": "🔥"}`, shown after the word; colour emoji are drawn from the system's emoji
font), `uppercase`, `platform` (`youtube`, `reels`, `tiktok`, `vertical`: the block is centred on
the middle of that platform's safe span at its rows, at most as wide as it, and the margin counts
from its safe area), `pop`, `position`, `margin`, `xOffset`, `size`, `font`, `panel` (a backing
panel behind each line, as tall as its ink: `true` is black at 60 %, or a colour `#rrggbbaa`,
`false` none — for words over bright footage: white `box` captions over a white toy car could not
be read; the Captions tab's **Background box** is the same panel), `keepStyle`. A page
breaks at a sentence end or a pause of 0.6 s; a word too wide for the space gets a smaller page of
its own. `sequence.checkSafeZones` checks every page of an animated track. The Text panel's
Captions tab has a live preview tile of each style, the platform and **Time words…**
(`text.captions.animated.*`); the CLI has `filmcraft-cli captions style <preset> … [--panel |
--panel=#rrggbbaa]` and
`captions timing --from transcript|narration`. AutoCut's shorts and trailers use Bold by default
(`captionStyle` picks another look or `plain`; `captionKeywords` lets the local model colour
keywords and add emoji).

**Malayalam and other scripts** are shaped word by word like any other text: characters the
bundled fonts lack fall back to an installed font made for their script (Malayalam Sangam MN /
Malayalam MN on macOS, Noto Sans Malayalam where installed on Linux), so conjuncts and vowel
signs join, and the highlight moves word by word. To have Malayalam everywhere (the web,
machines without such a font), add Noto Sans Malayalam (SIL OFL 1.1) to
[craft-fonts](https://github.com/storytold/craft-fonts) with script `Mlym` and build with
`CRAFT_FONTS_DIR`: craft-fonts faces are tried before system ones. No font files go in this
repository.

Caption Export Settings now offers a caption-track choice as well as the file format. The selected Text-panel track is the initial choice; invalid/deleted requested tracks fall back to the first displayed track. Direct `captions.export {track,path,format}` remains available to agents, including ASS/SSA.
