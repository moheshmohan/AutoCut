# filmcraft-screenagent

Eyes and hands for the automation agent: screenshot a browser window, ground the UI
elements in it with a local vision model, and drive them — click, type, scroll — with hard
budgets so it can never loop or wander. Built to automate the one step Google reserves for
a human: choosing items in the Google Photos picker.

## Why this exists

Google retired the Photos Library API's library-wide read scopes on 2025-03-31. The Picker
API that remains has, by design, **no search, no filtering, no programmatic selection**:
an app opens a picker page and a human clicks. This crate turns that page into an
agent-drivable surface the honest way — by looking at it. The picker's own search field
does search the user's library, so typing into it (through these eyes and hands) is the
only "search Google Photos" any app can still offer.

## How it works

```
capture (screencapture -R)  →  ground (Qwen2.5-VL via Ollama)  →  act (osascript click/type)
        ↑                                                              ↓
        └──────────── verify (pixel-diff: did it land?) ←──────────────┘
```

- `capture.rs` — the target app's front-window bounds (AppleScript) and a region
  screenshot, downscaled for the model. The pixel-per-point scale is derived from the real
  image, so Retina displays and downscaling are handled arithmetically, not by guesswork.
- `act.rs` — input synthesis through `osascript`. Keys (typing, Return, Escape) go through
  System Events, which sends them to whatever app is in front, whatever process the script
  names. So every key and click script first checks, in the same script, that the target app
  is still the front app. If it isn't, the script does nothing and the run stops with
  `ScreenError::NotInFront`: the agent never takes focus back and never types elsewhere (seen
  live: a picker search landed in another app's message box). Clicks and wheel scrolls are real
  CoreGraphics pointer events posted from JXA (`osascript -l JavaScript`): System Events'
  `click at` acts on the accessibility tree and Chrome's web content ignores it, so it
  never selected a tile. Real events move the pointer, as a person's would. Every click is
  clamped into the window rectangle; AppleScript strings are escaped and control characters
  refused, and only integers (and the app's name, quoted as a JS string) reach the JXA scripts, so nothing model-generated becomes
  script.
- `ground.rs` — the model call. Qwen2.5-VL grounds in absolute pixels,
  `{"bbox_2d": [x1, y1, x2, y2]}`, of the image *it* saw: its runtime snaps images to
  28-px patches and up-scales anything under 1024 patches. So the grounder resizes each
  screenshot to that frame itself (`model_frame`, Qwen's `smart_resize`), sends it, and
  maps the reply back. Asking for Qwen2-VL's normalized 0–1000 `[y1, x1, y2, x2]` does not
  work: 2.5 ignores it and answers in its own pixels. Every coordinate is validated before
  it becomes a pixel, and prose ("There are none.") means nothing found. `inventory()`
  lists elements, `find()` locates one by description.
- `verify.rs` — two captures, downsampled, compared by mean-abs-diff *and* by the share of
  clearly changed pixels (a tile's check mark is ~1% of the window, which a mean averages
  away): the agent knows whether a click changed anything, and stops retrying when the
  screen stops changing.
- `agent.rs` — the budgeted loop (`Plan { queries, max_items, videos_only, max_actions: 40,
  timeout }`): type the query in the picker's search, click video tiles (duration badges),
  scroll for more, press Done. A reply that lumps the grid into one box, or stamps one
  label on many boxes, is asked again once with a stricter prompt. Bounded actions, bounded
  time, bounded patience.
  - **Many searches, one session** (`queries` with more than one): the picker keeps its
    selection while the search changes, so the agent empties the search box (Backspace and
    Forward Delete — never Cmd+A, which in a grid could select everything), types each query,
    selects its results on top of what is selected, and reads the header after each: the count
    only grows. A search with no results is skipped; a search that leaves the grid as it was
    selects nothing (a range over the previous results would clear them); a count that falls
    means the search cleared earlier selections, so it and the searches before it are handed
    back for another try (`Summary::retry`). Before each search the room is checked — the
    session's item cap (`max_items` when selecting all; before a range the biggest search so far
    is the guess of its size, while date searches are taken day by day up to the cap) and the
    budget, which grows with the searches and keeps a reserve for Done — and what does not fit
    is left in `Summary::left` for a next session. The picker
    lets date groups and ranges select past the cap, then refuses Done (live: "143 selected",
    "Select up to 40 items", Done greyed out), so date groups are selected day by day up to the
    cap (below). A range cannot be undone (a second range toggles single tiles), so past the cap
    the agent stops rather than press a Done the picker refuses. Done is pressed once.
    `Summary::searches` reports each search (selected, no results, nothing new, at the cap,
    cleared, deferred, too big, split, failed). Losing a grant (Screen Recording, Accessibility) or
    the model ends the session; a search that merely fails does not.
  - **Date groups**: for a date search ("videos October 13 2022", "videos October 2022")
    select-all walks the date headings, paging down, and selects each day of the searched date
    once (`picker::day_key`: a heading still on screen after a scroll is the same day, read with
    or without its year; a neighbouring day or another year's that Google's search shows too is
    another search's) by the round check button left of its heading. The model is asked for the
    headings (`picker::DAY_HEADINGS`), not the buttons: asked for check buttons it boxed tiles'
    selection circles under other headings and labelled them with dates (live), so a click
    selected one tile or unselected another day. The button is clicked on the heading's own row
    only — at the left end of its box, then just left of it (`picker::check_points`: the box
    starts at the button or at the date's text, live, both). The header's "N selected" is read
    after every click — one more model call per click, the price of an exact count: hover alone
    changes the pixels, so only the count tells what a click did. A click that selected nothing
    is a day of one item, which in compact rows has no check button at all (live, not even on
    hover): it is selected by its tile (`picker::lone_tile`, only when no other tile of a bigger
    day sits beside it on its row — one tile would pass for the whole day), else left as a day
    search. A click that lowered the count unselected something selected already and is clicked
    back on: a day an earlier search of the session selected (the model never says a check is
    filled in), or else another day under a misplaced box (live) — that day is then left as a
    day search. A click that took the count past the cap is clicked again at once, which
    unselects the day while it is still on screen — no walk back over the days. A day too big
    for any session alone is skipped (and left as its own search, which reports it `TooBig`);
    any other ends the walk: the search is `Split`, the days before are kept, and it resumes
    from that day in a next session (`videos October 2022 from 2022-10-29`,
    `picker::resume_point`; typed without the resume point, the walk skips the days after it),
    and the session holds no more searches. One search per day would not do: Google's day
    search leaves out items its month view groups under that day (live: four clips just after
    midnight) and shows other years' days. When none of a search fits beside what is selected it
    waits whole (`Deferred`, and the session ends there too); a date search whose days took
    nothing gets the range only when the range guess has room, else it waits the same way. A
    date search whose results show only other dates' days (Google answers a day it has nothing
    for with neighbouring ones, live), or nothing after its resume point, has no results. A first
    page with no date heading is looked at once more (results still rendering), and when no day
    took at all it falls back to the range gesture from the top.
  - **Closing the tab**: when it begins, `drive` names the picker tab in front by its window
    and tab id (Chrome); when it finishes — after Done, or giving up — it closes *that* tab by
    those ids (`tabs::close_driven_picker`), only while its address is still the picker, closing
    once more if it asks "Leave site?". Never "whatever tab is in front then": another session
    was working in Chrome alongside, live, and Cmd+W would have closed its tab. Browsers that do
    not name their tabs get Cmd+W (`tabs::close_front_picker`), only when the front tab's address
    is the picker right before it, then Leave if the browser asks.
  - **Select all** (`select_all`): click the first tile, wheel to the end, Shift+click the
    last. A grid of one is not Shift+clicked (that would toggle the only selection off). The
    count is the header's "N selected" alone — the ask names no number (an example count was
    once answered back from an empty header), a bare number is not a count — and a header
    showing no selection gets the gesture once more from the top, then an error: Done is
    never pressed with nothing selected. Select-all only ever runs inside a search result (an
    album's too), never the whole library. The range starts at the first tile in reading order.
  - **An album** (`album: Some(title)`): Google's picker has no Albums view (that view is only
    in the Google Photos page behind the picker, where a selection does not count), and its
    search suggestions never name an album (live, 2026-10-19: generic suggestions and people).
    But Return on an album's exact title shows a search result that *is* the album's items,
    newest first, under date headings — the grid of any search result (checked against the
    page: 93 tiles for a 92-item album and the app's own export). So the agent types the title
    as it is, presses Return and takes that result grid for the album: the range gesture over
    all of it — every tile, photos too, so its first and last items are the album's ends — the
    header's "N selected", then Done. There is no album card to look for and nothing to "open".
    The guards come from the live failure, in which the model took the typed title in the search
    field for "the album" and for "the album's heading", and the agent range-selected the
    unfiltered library until its budget ran out:
    - the search must visibly have run, and Google answers an album after a delay (live,
      2026-10-24: "akku tesla toy", 52 items, missed by two looks 2.5 s apart): the view moved
      after Return (`verify::moved`, looked at up to 8 times, 20 s) and the grid's first screen —
      its date headings and its tiles — is not the library's as the picker opened; otherwise
      "searching the picker for `<title>` showed nothing new; nothing was selected";
    - nothing inside the search field's box is ever a result, a tile or a heading
      (`picker::in_field`: every grounding answer of an album pick is filtered through it);
    - a runaway is never confirmed: a grid that does not end within the budget, or a count
      above the run's `max_items`, ends the pick without Done (the agent closes the tab, and
      the caller deletes the session).

    **In the page, where Chrome allows it** (`dom.rs`; Chrome ▸ View ▸ Developer ▸ Allow
    JavaScript from Apple Events): the agent reads the picker tab it drives (by its window and
    tab id, only while its address is the picker) instead of its pixels. Each media tile is a
    `[role=checkbox]` labelled `<kind> - <shape> - <date, time>` (`Video - Landscape - Jul 28,
    2025, 7:18:02 PM`, a U+202F before PM; `aria-checked` says whether it is ticked); `Select`
    and the days' `Select all photos from Tue, Jul 29, 2025` are other checkboxes, never clicked.
    The grid is virtual (about 29 checkboxes at a time). So after Return the agent looks at the
    page every second, up to 30 s, until its tiles are not the library's and are the same twice
    in a row (`dom::wait_for_results`); then it walks the grid from the top, a scroll of 4/5 of
    the grid's scroller at a time to its end, clicking every tile not ticked yet, and walks again
    until nothing is left unticked and the header's "N selected" is not below the tiles seen
    (`dom::tick_all`). More tiles than the run's maximum, or a tile that will not tick, end the
    pick without Done. When Chrome refuses the scripts the agent says how to allow them and
    selects by eyes, as above. Each step is one fixed script; nothing from the page, the title or
    the model is ever put into one.

    The picker cannot select by type, so `--videos-only` takes the album whole and its photos
    are skipped when downloading (`filmcraft_gphotos::keep_picked`); the CLI and the album
    source say how many photos were left out.
  - **Exact items** (`items`: each a `Wanted`, when it was taken in UTC and whether it is a
    video): ones a catalogue found, FrameAtlas by meaning say. The picker cannot search by name
    or id, but a date typed into its search shows that day, and each tile's checkbox in its page
    says when it was taken, to the second, in local time. So the agent searches each item's day
    (`items::day_text`: `October 30 2026`), reads that day's whole grid (`dom::grid_labels`) and
    ticks only the tiles that are the items (`dom::tick_labels`, by a hash of the label so only
    numbers go into a script). A tile is an item's when its kind matches and its local time is
    the item's UTC time moved by a real time-zone offset (`items::OFFSETS`); another tile of the
    day can fit that by chance (about one in 2,300), so `items::Matcher` takes the offset the
    pick has seen most (the Mac's own zone is each item's first guess), ticks nothing for an
    item two tiles fit equally, and looks for an item not on its first day on the day next to
    it (three days at most). Items it could not find are `left`, as their times. It needs the
    page: by eyes the times cannot be read, so without it nothing is selected.
  - **Done is checked**: after the press the agent re-captures; the press took when the
    window closed, the button is gone, or "N selected" is gone. Otherwise it presses again
    (at most twice more, from the budget), then reports that the picker did not accept Done.
- `dom.rs` — the picker's page (above): the scripts of each step (`dom::script`), what a media
  tile's label is (`dom::tile_label`), waiting for an album's results and ticking every tile,
  reading a whole grid and ticking given tiles by their labels' hashes (`dom::tick_script`).
- `items.rs` — exact items: their times, the tiles' times, the day searches and which tile is
  which item (`items::Matcher`), as pure, unit-tested functions.
- `picker.rs` — the picker's visual semantics (the tile-enumeration prompts, what a video
  tile looks like, where to click one — its left third, since grounded boxes start where
  the tile starts but can run on into the next — what the confirm button is called, what a
  date search and a date heading are, where a day's check button is, and that the browser's
  Leave button is labelled "Leave" and button-sized), as pure, unit-tested predicates.
- `tabs.rs` — the picker's browser tabs. Chrome's AppleScript dictionary lists every tab
  (window id, tab id, address); a tab is closed or focused only when found by window, id *and*
  address and only when the address is Google's picker (`https://photos.google.com/[u/N/]picker…`
  or `…/integration/picker…`), so nothing else is ever touched. `close_picker_tabs` closes the
  earlier picker tabs before an agent's session opens its own (pointing an open tab at the new
  session was tried: a tab holding a selection asks "Leave site?" while Chrome already reports
  the new address, so the agent met the dialog); `tidy` (`filmcraft-cli photos tidy`) closes
  every picker tab;
  `answer_leave` lists the "Leave site?" dialog's buttons by their own text and clicks only the
  one that reads "Leave" (asked to *find* "the Leave button", qwen2.5vl boxes it right but
  labels it with the ask's own words, live). Chrome takes a second close of a tab that asks
  "Leave site?" as Leave, so `tidy` closes such a tab once more before it needs any eyes. Ids
  are compared as text (Chrome 155 answers `tab id "180477276"`, never equal to a number).
  Every AppleScript call to the browser has a 15 s timeout, so a dialog cannot hang it.

- `locations.rs` — **where an album's items were taken**, read from the Google Photos page
  (Google strips the location from what its API and picker hand out; the owner's page still
  shows it). `find_album` opens the albums page, finds the card with the album's exact title
  (the vision model's box, below the top bar), clicks it and checks the page's title reads as
  asked; `read_album` opens the album's first thumbnail (or, resuming, the item after the last
  one read, by its address), opens the Info panel with its shortcut `i` when the model reads no
  panel, then for each item crops the right part of a full-resolution capture, has the model
  transcribe the Details lines (`PANEL_ASK`: the place's line marked `Location:`), parses them
  (`parse_panel`: date, time, GMT offset, file name, pixel size, place and the line under it,
  "Estimated location"; told apart by shape — a date, a time, an extension, "1920 × 1080", "ƒ/1.8",
  a camera maker — and by order: the line after the time is the file name), and goes on with
  the Right arrow, or the "next" arrow beside the item when the key does not move on, until
  the address stops changing. Nothing in the library is changed: only the Right arrow and `i`
  are pressed (`i` only while an item is open), nothing is typed, and the only clicks are an
  album card, the first thumbnail and the "next" arrow, each checked to lie where such a thing
  sits (below the top bar; the arrow on the right half, mid-height, left of the Info panel), so
  a misplaced box never reaches the toolbar's favourite, delete or share. The reader opens its
  own Chrome tab, names it by window and tab id, touches it only while it shows
  `photos.google.com` and closes it at the end. Bounded: `max_items`, `per_item` (an item that
  does not read is recorded with its `error` and read again on `--resume`), `budget`; a panel
  that still shows the item before is read again. `filmcraft-automation::locations` holds the
  `chrome` lock, remembers album addresses by title (`<data>/google-photos/locations/albums.json`),
  and writes the JSON after every item; `AUTOCUT_LOCATIONS_DEBUG=<dir>` keeps every capture the
  model is shown and every answer.
  What the live runs taught (qwen2.5vl:7b, 2026-10-20):
  - an item in albums or with people has them above its Details, which push the place below the
    window: the panel is wheel-scrolled to its end (one event over its middle) before each look;
  - scrolled, the map shows, and the model took its labels ("Business Bay Metro Station") for
    the place ("Home"): the coloured block at the panel's bottom is cut off (`rows_above_map`:
    the panel is grey, the map is not), and the ask says never to take names from a map;
  - an ask with a way out ("write NONE if there is no panel") was taken for panels read
    perfectly a moment before; the ask has none, and an open panel is told from what is read;
  - asked to copy "Estimated location" when shown, the model wrote it under every place; a
    separate question (`ESTIMATED_ASK`) — is there a circled i after the place? — answered it
    right (Home ⓘ: yes; Elackad: no), and the circled i is taken to mean estimated;
  - long file names wrap ("…_HSR_120.mp" / "4") and are joined again; a place the model makes
    up from Google's notes ("Shared by saranya mohesh", "Storage saver.") is refused.
  Live, on the album "Every Robot I've Met" (99 items) against the main session's hand reading:
  12 items read (8, then 4 more with `--resume` after the last), 11 places and the one item
  without a place as read by hand; the twelfth's invented place is refused by the parser since
  (unit test on that transcription). About 8–15 s an item on an M3 Max (two or three model calls).

No `unsafe` (the workspace forbids it), no browser embedding, no model runtime linkage:
the OS contact is subprocesses (`osascript`, `screencapture`) exactly like
filmcraft-gphotos' `open`, and the model is served by the Ollama server AutoCut already
requires.

## Using it

CLI: `filmcraft-cli photos pick --auto "beach sunset videos" --max-items 5 --bridge …`, or a
whole album: `filmcraft-cli photos pick --album "Robot showcase" [--videos-only] --bridge …`,
exactly what FrameAtlas finds by meaning: `filmcraft-cli photos pick --meaning "kids laughing on
a boat" [--max-items 24] [--photos] --bridge …`, or many searches in one session: `--auto "q1"
--auto "q2" --all`, `--queries file --all`, or a
sweep of a whole history: `--sweep 2017-10..2025-12 [--month-days 10-21]`. `filmcraft-cli
photos tidy` closes picker tabs left in Chrome.

Places: `filmcraft-cli photos locations --album "Every Robot I've Met" [--max-items 5]
[--resume] [--out locations.json] [--project p.fcproj --save]` (engine: `photos.locations`, then
`photos.applyLocations`).

MCP: `googlephotos_pick { query: "beach sunset videos", max_items: 5 }`,
`googlephotos_pick { queries: ["videos October 2022", "videos November 2022"], all: true }` or
`googlephotos_pick { album: "Robot showcase" }`, or the items FrameAtlas's
`catalog_search_meaning` found: `googlephotos_pick { items: [{captured:
"2020-11-15T13:38:02Z", kind: "video"}, …] }` — the result carries an `autopick` report of
what the agent did (`album_shown`, `selected`, `done_pressed`, `searches`, `left`,
`tab_closed`, `note`), or for many searches the `sessions` and the searches `left`.

## One-time prerequisites

1. `ollama pull qwen2.5vl:7b` (or set `AUTOCUT_SCREEN_MODEL`; `AUTOCUT_OLLAMA_HOST` is
   shared with the AutoCut classifier).
2. **Screen Recording** for the process that runs the picker flow (your terminal, or the
   packaged app): System Settings ▸ Privacy & Security ▸ Screen Recording.
3. **Accessibility** for the same process: System Settings ▸ Privacy & Security ▸
   Accessibility.
4. **Automation ▸ Google Chrome** for the same process (macOS asks the first time): it lets the
   agent read the front tab's address before closing it, close the earlier picker tabs and tidy
   picker tabs. Without it the tabs are simply left open.

Missing grants and missing models come back as errors that say exactly this.

## Model choice, and the ones we did not take

- **Qwen2.5-VL (Apache-2.0)** — grounds UI screenshots natively (bounding boxes and points
  as stable JSON), and runs in Ollama, which AutoCut already requires. `qwen2.5vl:7b` is
  the quality/size sweet spot on Apple silicon; `:3b` works for simple grids. This is the
  default.
- **ShowUI-2B (MIT)** — purpose-built GUI grounding and excellent at it, but it is a
  vision-language-action model with a custom action format: serving it needs a Python/MLX
  sidecar, which breaks the pure-Rust, Ollama-only deployment story. Not taken.
- **OmniParser v2 (Microsoft)** — the original detector weights are fine-tuned from
  YOLOv8 and are **AGPL-licensed**, which the project's clean-room rules forbid outright;
  the MIT `icon_detect_v3` weights exist but still need a Python Detectron2/YOLO server.
  Not taken.

No model weights are committed to this repository or shipped with any binary.

## Honest limits

- The agent is only as good as its eyes: on dense mixed grids a 7B model sometimes boxes
  a tile one position off (live: about 1 in 5), which on a mixed grid would pick a photo.
  So when videos are wanted the agent always searches for them — `beach sunset` is typed as
  `beach sunset videos`, no query as `videos` — and Google's search leaves only videos in
  the grid (verified live: "handheld console videos", "dubai skyline videos"). Every click
  is verified by pixel-diff and bounded by budget, so a miss costs a retry, not a runaway.
  It never clicks outside the picker window.
- Tabs are closed only where the address can be read: Chromium browsers (Chrome, Edge, Brave,
  Arc, Vivaldi) and Safari. Elsewhere the finished picker tab stays open; closing the earlier
  picker tabs and `photos tidy` are Chrome-only.
- A search that overlaps an earlier one in the same session can clear part of the earlier
  selection — Google's picker toggles what is clicked again. Date groups are put back (the count
  after each day tells); a range is not: the agent sees the count fall and hands those searches
  back for another session. Avoid overlapping searches in one list.
- The count is only as exact as the model's reading of "N selected": a misread can still take
  one day for another outcome. A day the model never boxes as a heading is not selected, and a
  day left as a day search inherits that search's limits (items near midnight can be missing).
- It moves the real pointer while it works; leave the mouse alone for the minute or two
  it runs.
- It drives whatever app is frontmost after the picker opens — the caller opens the picker
  immediately before spawning the agent, and the agent re-reads the frontmost app itself.
- macOS only for now; other platforms get a clean "unsupported" error.
- Search happens in Google's own picker UI, so it inherits Google's ranking — an exact
  filename or a distinctive subject works best.
- An album pick trusts Google's search: the picker shows no album as such, so a title that is
  no album's still gets ordinary search results, and they are taken for the album (up to the
  run's `max_items`; past it, or past the budget, nothing is confirmed). Give the album's exact
  title, one no other text in the library matches better.

## Testing

`cargo test -p filmcraft-screenagent` is fully offline: parser fixtures (prose-wrapped
JSON, hostile coordinates, NaN, truncated PNGs), AppleScript escaping and injection
attempts, coordinate math, and the agent loop against scripted fake eyes/hands (budgets,
termination, in-bounds guarantees), many-search sessions (cumulative counts, skipped empty
searches, the cap and the budget leaving searches for a next session, a count that falls),
date-group selection against a fake picker with date groups (a day of one item without a check
button, a heading boxed from its text, a day selected already, a misplaced box that hits another
day, a session filled to its cap and the search resumed from the day that did not fit, a
resumed search, a search deferred whole, a day too big alone, a neighbouring month's day, the
range fallback), album picks against a fake picker that behaves as Google's does (suggestions
that never name the album, Return showing its items under date headings, the typed title boxed
in the field as a tile and as a heading, a Return that shows nothing new, a runaway grid that is
never confirmed, an exact album of 92 items selected whole), and the picker tabs against a fake
Chrome (only picker
tabs closed, a tab that asks "Leave site?" closed once more, Leave clicked and nothing else). A live model
round-trip can be run with
`AUTOCUT_SCREEN_LIVE=1`, and a saved window capture can be replayed through the real
grounding path with `AUTOCUT_SCREEN_LIVE_PNG=/path/to/capture.png cargo test -p
filmcraft-screenagent live_inventory -- --nocapture` (prints what the agent would click; with
`AUTOCUT_SCREEN_LIVE_TASK=day_headings`, the date headings it grounds).
`AUTOCUT_SCREENAGENT_TRACE=1` logs every OS invocation to `/tmp/screenagent-cmds.log`.

Places (`locations`): the panel parser on panels laid out as the page shows them and as the
model transcribes them (bullets, `x` for ×, `Location:`), on the album "Every Robot I've Met"
as read by hand (20 items: file names with and without extensions or with spaces, no place,
"Home", estimated places), a description above Details with its own date and time, and hostile
text (huge lines, control characters, 50 000 lines); addresses (accounts, shared albums, look-
alike hosts and AppleScript injection refused); the reader against a fake Google Photos tab: an
album read to its end, the item limit and the budget, a run stopped by its caller, resuming
after the last item read (by address, no thumbnail clicked), an unreadable item recorded and
read again on resume, a panel lagging one item behind, the "next" arrow when the key does not
move on (and never inside the Info panel or the toolbar), an Info panel that will not open, a
lost model, misplaced thumbnails never clicked; the album found on the albums page by its title
(a card that opens another album refused). `AUTOCUT_LOCATIONS_LIVE=<album address> cargo test
-p filmcraft-screenagent live_read_a_few_items -- --ignored --nocapture` reads five items live
(hold the `chrome` lock).
