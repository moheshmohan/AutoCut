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
  System Events scoped to the one target process. Clicks and wheel scrolls are real
  CoreGraphics pointer events posted from JXA (`osascript -l JavaScript`): System Events'
  `click at` acts on the accessibility tree and Chrome's web content ignores it, so it
  never selected a tile. Real events move the pointer, as a person's would. Every click is
  clamped into the window rectangle; AppleScript strings are escaped and control characters
  refused, and only integers reach the JXA scripts, so nothing model-generated becomes
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
    or without its year; a neighbouring day Google's search shows too is another search's) by the
    round check button left of its heading. The header's "N selected" is read after every click —
    one more model call per day, the price of an exact count: hover alone changes the pixels, so
    only the count tells what a click did. A click that selected nothing is a day of one item,
    which in compact rows has no check button at all (live, not even on hover): it is selected by
    its tile (`picker::lone_tile`, only when no other tile follows it on its row — one tile of a
    bigger day would pass for the whole), else left as a day search. A click that lowered the
    count found a day selected already (the model never says a check is filled in) and clicks it
    back on. A click that took the count past the cap is clicked again at once, which unselects
    the day while it is still on screen — no walk back over the days — and that day and the days
    after it (only listed) are left as day searches (`videos October 13 2022`) for a next
    session: the search is `Split`, and the session holds no more searches. When none of a search
    fits beside what is selected it waits whole (`Deferred`, and the session ends there too); a
    date search whose days took nothing gets the range only when the range guess has room, else
    it waits the same way. When its first day alone passes the
    cap its days are left as day searches, and a day search alone past the cap is `TooBig`
    (`Summary::too_big`). A date search whose results show only other dates' days (Google
    answers a day it has nothing for with neighbouring ones, live) has no results. A first page
    with no date heading is looked at once more (results still rendering), and when no day took
    at all it falls back to the range gesture from the top.
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
    never pressed with nothing selected. Select-all only ever runs inside a search result or
    an opened album, never the whole library.
  - **An album** (`album: Some(title)`): the picker's Albums view when one is grounded (a
    small "album" element, never a thumbnail-sized box), else the title typed into the
    picker's search (Google documents no Albums view; users search an album's title); only
    an element labelled with the title is clicked, the opened screen must be headed by it,
    then select-all. No match, or two albums with that title, is an error with nothing
    selected.
  - **Done is checked**: after the press the agent re-captures; the press took when the
    window closed, the button is gone, or "N selected" is gone. Otherwise it presses again
    (at most twice more, from the budget), then reports that the picker did not accept Done.
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

No `unsafe` (the workspace forbids it), no browser embedding, no model runtime linkage:
the OS contact is subprocesses (`osascript`, `screencapture`) exactly like
filmcraft-gphotos' `open`, and the model is served by the Ollama server AutoCut already
requires.

## Using it

CLI: `filmcraft-cli photos pick --auto "beach sunset videos" --max-items 5 --bridge …`, or a
whole album: `filmcraft-cli photos pick --album "Robot showcase" [--videos-only] --bridge …`,
or many searches in one session: `--auto "q1" --auto "q2" --all`, `--queries file --all`, or a
sweep of a whole history: `--sweep 2017-10..2025-12 [--month-days 10-21]`. `filmcraft-cli
photos tidy` closes picker tabs left in Chrome.

MCP: `googlephotos_pick { query: "beach sunset videos", max_items: 5 }`,
`googlephotos_pick { queries: ["videos October 2022", "videos November 2022"], all: true }` or
`googlephotos_pick { album: "Robot showcase" }` — the result carries an `autopick` report of
what the agent did (`album_opened`, `selected`, `done_pressed`, `searches`, `left`,
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
  one day for another outcome. A day the model never boxes as a heading is not selected.
- It moves the real pointer while it works; leave the mouse alone for the minute or two
  it runs.
- It drives whatever app is frontmost after the picker opens — the caller opens the picker
  immediately before spawning the agent, and the agent re-reads the frontmost app itself.
- macOS only for now; other platforms get a clean "unsupported" error.
- Search happens in Google's own picker UI, so it inherits Google's ranking — an exact
  filename or a distinctive subject works best.

## Testing

`cargo test -p filmcraft-screenagent` is fully offline: parser fixtures (prose-wrapped
JSON, hostile coordinates, NaN, truncated PNGs), AppleScript escaping and injection
attempts, coordinate math, and the agent loop against scripted fake eyes/hands (budgets,
termination, in-bounds guarantees), many-search sessions (cumulative counts, skipped empty
searches, the cap and the budget leaving searches for a next session, a count that falls),
date-group selection against a fake picker with date groups (a day of one item without a check
button, a day selected already, a session filled to its cap and the rest left as day searches,
a search deferred whole or too big, a check the model boxed off its button, a neighbouring
month's day, the range fallback), and the picker tabs against a fake Chrome (only picker
tabs closed, a tab that asks "Leave site?" closed once more, Leave clicked and nothing else). A live model
round-trip can be run with
`AUTOCUT_SCREEN_LIVE=1`, and a saved window capture can be replayed through the real
grounding path with `AUTOCUT_SCREEN_LIVE_PNG=/path/to/capture.png cargo test -p
filmcraft-screenagent live_inventory -- --nocapture` (prints what the agent would click).
`AUTOCUT_SCREENAGENT_TRACE=1` logs every OS invocation to `/tmp/screenagent-cmds.log`.
