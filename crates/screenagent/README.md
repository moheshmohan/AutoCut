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
- `agent.rs` — the budgeted loop (`Plan { query, max_items, videos_only, max_actions: 40,
  timeout }`): type the query in the picker's search, click video tiles (duration badges),
  scroll for more, press Done. A reply that lumps the grid into one box, or stamps one
  label on many boxes, is asked again once with a stricter prompt. Bounded actions, bounded
  time, bounded patience.
- `picker.rs` — the picker's visual semantics (the tile-enumeration prompts, what a video
  tile looks like, where to click one — its left third, since grounded boxes start where
  the tile starts but can run on into the next — and what the confirm button is called),
  as pure, unit-tested predicates.

No `unsafe` (the workspace forbids it), no browser embedding, no model runtime linkage:
the OS contact is subprocesses (`osascript`, `screencapture`) exactly like
filmcraft-gphotos' `open`, and the model is served by the Ollama server AutoCut already
requires.

## Using it

CLI: `filmcraft-cli photos pick --auto "beach sunset videos" --max-items 5 --bridge …`

MCP: `googlephotos_pick { query: "beach sunset videos", max_items: 5 }` — the result
carries an `autopick` report of what the agent did.

## One-time prerequisites

1. `ollama pull qwen2.5vl:7b` (or set `AUTOCUT_SCREEN_MODEL`; `AUTOCUT_OLLAMA_HOST` is
   shared with the AutoCut classifier).
2. **Screen Recording** for the process that runs the picker flow (your terminal, or the
   packaged app): System Settings ▸ Privacy & Security ▸ Screen Recording.
3. **Accessibility** for the same process: System Settings ▸ Privacy & Security ▸
   Accessibility.

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
  a tile one position off, so a photo can be picked for a video (live: about 1 in 5).
  Searching for "videos" (or "… videos") leaves only videos in the grid and sidesteps it.
  Every click is verified by pixel-diff and bounded by budget, so a miss costs a retry,
  not a runaway. It never clicks outside the picker window.
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
termination, in-bounds guarantees). A live model round-trip can be run with
`AUTOCUT_SCREEN_LIVE=1`, and a saved window capture can be replayed through the real
grounding path with `AUTOCUT_SCREEN_LIVE_PNG=/path/to/capture.png cargo test -p
filmcraft-screenagent live_inventory -- --nocapture` (prints what the agent would click).
`AUTOCUT_SCREENAGENT_TRACE=1` logs every OS invocation to `/tmp/screenagent-cmds.log`.
