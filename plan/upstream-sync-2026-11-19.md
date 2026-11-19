# Upstream sync — 19 November 2026

Upstream `storytold/filmcraft` (`origin`) is at `f67d38b3`. The last reviewed upstream commit was
`e6df3186` ([upstream-sync-2026-11-15.md](upstream-sync-2026-11-15.md)). There are 175 non-merge
commits in `e6df3186..origin/main`. AutoCut was at `3cc5be1e` (AV36).

The triage and the ports were done in one pass, so this file is the record of both. For the TAKE
rows, "applies" is `git cherry-pick -n` of the commit on the branch as it stood when its turn
came. The CARE rows were not cherry-picked again: their "applies" is what the first triage found
on `3cc5be1e`, and they were ported by hand from the upstream diff.

Totals: **6 TAKE** (cherry-picked, tests moved where the fork keeps them elsewhere), **12 CARE**
(ported by hand), **0 ALREADY**, **157 SKIP**. Of the 157, 150 are upstream's move to the shared
`craft-libs` crates, which this fork does not take: it keeps its codecs, containers and
interchange code in-tree, and no `craft-libs` dependency was added to any `Cargo.toml`.

## What the fork gains

| Area | Commits | What it closes or adds |
|---|---|---|
| Agents driving the app | `d74acd59`, `ac32e783`, `e8cec5fd`, `d9aad8a6`, `04b2b654` (all CARE) | `engine.execute` over the control channel (what the MCP bridge's `command_run` sends) never opens a parameter dialog, an export settings dialog or a file picker: the engine command runs with what was given. `sequence.settings {}` and `voiceover.settingsDialog {}` answer the settings. |
| Memory | `5bd329d0` | Decoded frames of animated GIF, APNG and WebP files share one 1 GiB cache instead of up to 1 GiB a file. |
| Speed | `ad106cd8` (CARE) | The parts of an export job reuse the previous part's GPU frame renderers instead of each building its own. Upstream measured that building them was most of a preview segment's setup on a cut-heavy timeline; it was not measured here. The same path serves proxies and AutoCut's upscale, interpolation and object-removal renders. |
| Files that would not open | `93decb25` | A project whose media has a codec FilmCraft cannot decode (stored as 0×0) opens again; the size check from the last sync refused it. |
| Wrong results | `9ad3e413`, `68d2a03d`, `176d0729`, `0b3f30ed` (CARE) | FCPXML clips import fitted to the frame, as FCPXML means them, and export keeps native size. An FCPXML lane that cannot be used is reported. Narration reads "COVID-19" without "minus". The colour swatch in Effect Controls showed a different colour from the one stored. |
| Editing and UI | `0b3f30ed` (CARE), `92bea956` (CARE), `47ce7b11`, `87fa0e66`, `a6f1af9e` (all CARE) | A `#rrggbb` field beside every colour parameter. A text property changed with several text clips selected applies to all of them (`graphics.set` takes `clips`). The layout in use when the app closes comes back at the next launch. |
| Languages | `01a36c1c` (CARE) | Bosnian interface (upstream's catalog; AutoCut's own strings show in English). |
| Tests only | `76b945f2` | The performance UI tests step at 60 Hz. |
| Written here, not ported | `2f8ff527` (SKIP) | Upstream bounds MCP lines in `craft-jsonrpc`. The fork's own MCP line reader was unbounded, so the same 2 GiB bound was written into it (AV40). |

## Every commit

The 150 `craft-libs` commits are grouped by kind; the commits the brief named are listed in their
row.

| commit | subject | decision | applies | note |
|---|---|---|---|---|
| `93decb25` | format: skip the zero-sized unsupported-codec marker in media bounds | TAKE | clean | AV37 |
| `76b945f2` | #1673: performance test clock aligned with worker yields | TAKE | clean (`tests/perf_ui.rs`) | AV38 |
| `176d0729` | tts-text: a hyphen is minus only at the start of a number | TAKE | product clean | AV39; the test into the inline `mod tests` |
| `68d2a03d` | FCPXML: a connected clip with an unusable lane is reported | TAKE | product clean | AV41; the test added to `tests/fcpxml.rs` by hand |
| `d74acd59` | #1540: `engine.execute` never opens a parameter dialog | CARE (medium) | conflicts (`app.rs`, `control.rs`, `menus.rs`, panels, tests, docs) | AV42 |
| `ac32e783` | #1540: the remaining dialogs and file pickers | CARE | conflicts | AV42 |
| `e8cec5fd` | #1540: the Clip Mismatch Warning; docs | CARE | conflicts | AV42; the fork has no Clip Mismatch Warning, so only the docs |
| `d9aad8a6` | #1540: the dialog test leaves `file.exportFrame` out | CARE | conflicts | AV42 |
| `04b2b654` | #1540: `voiceover.settingsDialog` answers the settings | CARE | conflicts | AV42 |
| `ad106cd8` | export: parts of a job reuse the previous part's frame renderers | CARE (small–medium) | conflicts in `lib.rs`, `registry.rs`, `tests.rs` | AV43; the fork has no `registry.rs` and a per-thread test registry |
| `5bd329d0` | media: bound decoded animated GIF and APNG frames in a shared cache | TAKE | 1 hunk in `engine/src/perf.rs` | AV44 |
| `9ad3e413` | FCPXML: import and export `adjust-conform` (fit by default) | TAKE | module comment, README, test layout | AV45; **changes how other applications' FCPXML imports** |
| `0b3f30ed` | hex field for colour parameters in Effect Controls | CARE (small) | `effect_controls.rs` | AV46; also turns the swatch, Lumetri tints, the transition colour and the eyedropper to sRGB |
| `92bea956` | Properties: a text property change applies to every selected text clip | CARE (small–medium) | 3 files | AV47 |
| `47ce7b11` | #2421: the layout in use at close comes back at the next launch | CARE (medium) | file layout | AV48 |
| `87fa0e66` | #2421: a window closed before its first frame keeps the remembered layout | CARE | file layout | AV48 |
| `a6f1af9e` | #2421: a current workspace that names none keeps the remembered layout | CARE | file layout | AV48 |
| `01a36c1c` | Bosnian (bs) interface language | CARE (small) | `bs.tsv` absent | AV49; the commit only adds strings: the language came in `293d4e90`, before this range and not in the fork, so that wiring was ported too |
| `2f8ff527` | Bound MCP input and output lines | SKIP, equivalent written | needs `craft-jsonrpc` | AV40: the bound is in the fork's own line reader |
| `39f2abc4` | platform tests: `NVENC_SERIAL` in every NVENC test | SKIP | clean, but would not compile on Windows or Linux (the two files are compiled out on macOS) | it uses `crate::common::NVENC_SERIAL`, which the fork does not have; the hang it fixes needs every test in one process (upstream's `tests/it/`), and the fork has one test binary per file, each of these two with one test |
| `eb380e1d` | test: a ruler press while stopped is not counted across the scrub linger | SKIP | file absent | the fork has no `scrub_stop_ui.rs` (the test it changes is not in the fork) |
| `6b86a61e`, `bb7a19fc` | dock tabs: highlight, then underline, the active tab | SKIP | conflict | written for craft-ui's `DockContent` tab strip (`label`, `group_header`, `DockStyle`); the fork draws its own tab strip in `dock_ui.rs` |
| `b6a37d19` | Clippy `expect_used` under `--all-features` and the no-threads build | SKIP | — | named as a skip in the brief; not looked at further |
| `773e67fe` | #4728: pass web-craft-fonts to the release workflow | SKIP | — | release CI |
| 68 commits | `craft-libs` pins and lockfile refreshes (`Pin …`, `Refresh … lockfile`, `update shared … revision`; only `Cargo.toml` / `Cargo.lock`) | SKIP | — | includes `401b0e30` (the bounded TGA budget is in `craft-tga`) and `412c0fae` |
| 56 commits | adoption of shared crates (`use shared …`, `delegate … to craft-*`, `route … through shared timeline`, `crates/dnx` removed): export 10, codecs 10, interchange 10, media 8, DNG 8, MCP and control protocol 4, project format 3, cfb 1, text 1, caption time types 1 | SKIP | conflicts | would delete or bypass the fork's in-tree crates. Includes `7fa6d3b4`, `df1f42df`, `97f8dec2`, `6df14ec6`, and `d896af61` (YCbCr TIFF orientation, in the shared decoder) |
| 26 commits | follow-ups to the adoption (fixes and tests of the adopted code): interchange 11, DNG 5, codecs 4, export 2, media 2, cfb 1, app 1 | SKIP | file absent or conflict | includes the `timeline_model.rs` fixes `038b28d5`, `18ca5c51`, `a5ab4105`, `81c351f8` (the lane report and conform of AV41 and AV45, on upstream's new import path), `39c075a9`, `e88a15a3`, `1ccd567a`, `a9ac57d4`, and `219c376e` (`Report::push_n`, used only by the shared timeline) |

## Results

Integrated on 19 November 2026 on branch `worktree-agent-ab41b8734a5b8b702`, cut from `3cc5be1e`.
Each port is one commit whose message names the upstream commits and says what was adapted.
Nothing was merged from `origin`, and nothing was pushed.

| Commit | Port | Upstream commits |
|---|---|---|
| `b204959a` AV37 | Projects with the unsupported-codec marker open | `93decb25` |
| `3f0a7240` AV38 | Performance UI tests at 60 Hz | `76b945f2` |
| `5a3ffa04` AV39 | A hyphen is minus only at the start of a number | `176d0729` |
| `0f209262` AV40 | MCP lines bounded | equivalent of `2f8ff527`, written for the fork |
| `1f7324fe` AV41 | FCPXML unusable lane reported | `68d2a03d` |
| `7e2e0877` AV42 | `engine.execute` never opens a dialog | `d74acd59`, `ac32e783`, `e8cec5fd`, `d9aad8a6`, `04b2b654` |
| `78aeb715` AV43 | Export parts reuse frame renderers | `ad106cd8` |
| `8efdd61b` AV44 | Animated image frames bounded | `5bd329d0` |
| `d176c70d` AV45 | FCPXML `adjust-conform` | `9ad3e413` |
| `47146560` AV46 | Hex field for colour parameters | `0b3f30ed` |
| `16c0891c` AV47 | Text property to every selected text clip | `92bea956` |
| `57d76f84` AV48 | Layout at close comes back | `47ce7b11`, `87fa0e66`, `a6f1af9e` |
| `6ffa9609` AV49 | Bosnian | `01a36c1c` (with the wiring of `293d4e90`) |

All 18 TAKE and CARE commits are in. None was deferred. `39f2abc4`, which the first triage listed
as applying cleanly, was skipped for a reason found in the code. `eb380e1d` and the pair
`6b86a61e` / `bb7a19fc` were to be taken only if they fitted the fork, and they do not (see the
table).

### Parts of a commit that were not taken

- `e8cec5fd`: its one code line guards the Clip Mismatch Warning, which the fork does not have.
  `ac32e783` names `file.closeAllProjects`, which the fork does not have either.
- `92bea956`: upstream's `graphics.set` returns keyframe counts for its `keyframe` parameter. The
  fork's `graphics.set` has no such parameter, so `set_props_on` returns nothing.
- `9ad3e413`: the module-comment line and README section about standalone `video` elements. They
  document an import the fork does not have.
- `0b3f30ed`, `5bd329d0`, `ad106cd8`, the #2421 and #1540 commits: nothing left out.

### How the ports were adapted

- **AV39**: the test is in the inline `mod tests` of `normalize.rs`. The Malayalam paths are not
  touched.
- **AV40**: `BoundedLines` in `crates/automation/src/server.rs` replaces `tokio::io::Lines` on
  both sides of `serve_io`. A request line over 2 GiB answers `-32600` and the next line is
  served; a reply over the bound is replaced by `-32603`. What has been read of a line is kept
  between reads, because the writer reads in a `select!`.
- **AV42**: the fork has one `menus::invoke` and builds the app in `lib.rs`, so `engine_call`
  is a field there, set in `control.rs` around the one call. AutoCut's own dialogs follow the
  rule: Save as Video Template…, Make a Showcase…, Quick Short… and the `lumetri.exportCube`
  picker. The question before a model download (`ask_for_model`) is a confirmation and still
  asks, like Revert, Recover, Close Project and Quit.
- **AV43**: the idle list is in `export/src/lib.rs` beside the registry. In unit tests it is
  per test thread, like the registry. The fork's `RendererPool` is unchanged. Every AutoCut job
  runs through `guard_job` (by way of `narration::spawn`), which is where a job's end frees the
  kept renderers.
- **AV44**: `perf.stats` keeps the fork's `audioCacheMB` and gains `animatedCacheMB` and
  `animatedBudgetMB`.
- **AV45**: tests are `tests/fcpxml_conform.rs` with the fork's `clip(start, duration, source)`
  helper.
- **AV46**: the two hunks of `effect_controls.rs` are placed in the fork's `param_row` and
  `lumetri_panel`. Before taking the sRGB change, the fork's render code was checked: the
  keyers compare the stored colour with encoded pixels, and `effects.rs`, `creator_fx.rs`,
  `graphics.rs` and the transitions decode it with `srgb_to_linear`. So the stored components
  are sRGB in the fork too, AutoCut's creator effects included.
- **AV47**: `set_props` keeps its signature for its five other callers (graphic templates and
  text effects among them) and calls `set_props_on` with one target.
- **AV48**: the fork saves a workspace as a dock tree plus floating groups, not as a craft-ui
  `Workspace`. `SavedWorkspace::new` takes the fork's `Layout`, `read_saved` returns one (with
  the fork's legacy Effects migration), and the restore sets `ui.dock` and `ui.floating`.
- **AV49**: `bs.tsv` and its sidecar are upstream's files at `f67d38b3`, unchanged, with
  upstream's `ATTRIBUTION.md` row. The wiring follows Italian's in the fork's unsplit
  `i18n/mod.rs`.

### One fork test changed

`templates_ui::save_share_and_install_from_the_panel` opened the Save as Video Template form with
`engine.execute {"id": "template.save"}`. That is the behaviour AV42 removes. The test now opens
the form with `ui.menu.invoke`; the rest of it is as it was. It was the only test of the 106
`filmcraft-ui-egui` suites that relied on `engine.execute` opening a dialog.

### What behaves differently after this sync

- `engine.execute` of a dialog command without parameters runs the engine command. Most answer
  an error that says what is missing. Some have defaults and act: `sequence.addTracks {}` adds
  a video track, `sequence.simplify {}` makes the simplified copy, `sequence.normalizeMixTrack
  {}` normalizes, `autocut.quick {}` starts the Quick Short job. The headless MCP server always
  behaved this way; the bridge to the running app now matches it. `ui.menu.invoke` opens the
  dialog.
- FCPXML written by another application imports its clips fitted to the frame (scale-to-frame
  on); they used to come in at native size.
- A colour picked with the eyedropper is stored as the sampled sRGB bytes, so the same pick
  stores a different (lighter) value than before. Colours already in projects are untouched and
  render as before; the swatch now shows them as they render.
- At exit the app writes the layout in use to `workspaces.json` (`last`).
- Narration: a hyphen after a letter or digit is no longer "minus" ("2019-2020", "F-16").
- New UI command: `app.language.bosnian`. New `perf.stats` keys: `decode.animatedCacheMB`,
  `decode.animatedBudgetMB`. `graphics.set` takes `clips`.

### Checks

All with `CARGO_TARGET_DIR=target/agent-upstream` in this worktree, one cargo at a time. The full
`cargo xtask ci` was not run here: it is the main session's, after the merge.

At `6ffa9609` (AV49, the last code commit), every command exited 0:

| Command | Result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --no-fail-fast -p filmcraft-format -p filmcraft-tts-text -p filmcraft-automation -p filmcraft-interchange -p filmcraft-export -p filmcraft-media` | 39 suites, 594 passed, 0 failed, 1 ignored |
| `cargo test --no-fail-fast -p filmcraft-engine` | 1,265 passed, 0 failed, 8 ignored |
| `cargo test --no-fail-fast -p filmcraft-ui-egui` | 107 suites, 799 passed, 0 failed, 6 ignored |
| `cargo test --no-fail-fast -p filmcraft` | 81 passed, 0 failed |
| `cargo xtask layers` | ok |
| `cargo xtask assets` | ok (93 assets attributed) |
| `cargo xtask wasm` | ok (57 crates) |

- Not run: the tests of crates this work did not touch (`cargo test --workspace` would add
  `filmcraft-cli`, render, codecs, gpu and the rest). Clippy covered the whole workspace.
- The commits before AV49 were not each built and tested alone. AV37 and AV38 were tested as
  they stand. AV39 to AV42 were tested together (their crates' tests, and the whole
  `filmcraft-ui-egui` suite: 106 suites, 787 passed), with AV43's files already in the tree.
  AV43 to AV49 were tested with the later ones' files in the tree. The ports touch different
  crates or different files, so each commit is expected to be green, but only the last one is
  proven so.
- `cargo xtask i18n-report` ran and lists what Bosnian lacks; missing translations show in
  English and fail nothing.
- Command ids: the 914 ids of `filmcraft-cli commands --json` at `3cc5be1e` are all present at
  `6ffa9609`, and none was added (the list holds engine commands; `app.language.bosnian` is a UI
  command). One entry differs: the `params` text of `graphics.set`, which now names `clips`.
- Test names (`#[test]` functions found by `grep`, so `#[tokio::test]` and proptest cases are
  not counted): 6,222 at `3cc5be1e`, 6,249 at `6ffa9609`. One name changed, upstream's own
  rename with the behaviour: `hostile_media_video_dimensions_are_rejected` is now
  `hostile_media_video_dimensions_are_bounded` (`93decb25`). Four `#[tokio::test]` tests were
  added in `crates/automation` (AV40).
- ffmpeg and ffprobe are not installed, so the external-oracle tests skip, as before.

### Still to do before a push

- The one full `cargo xtask ci` on the merged result, then the user's OK. Push only to `autocut`.
- Not done here, because they need a running app, the user's media or another machine: a look
  at the hex field and swatch in the app (the tests drive them headless); render previews on a
  real cut-heavy timeline with the GPU renderer, to see the renderers reused (the test uses a
  stand-in renderer); an FCPXML file from Final Cut Pro; the Bosnian menus on screen.
- Nothing here was built or run on Windows or Linux.

### Follow-ups noticed, not changed

- `cargo xtask assets` does not look at `.tsv` files (`ASSET_EXT` in `xtask/src/main.rs` has no
  `tsv`), so a catalog without a sidecar passes. `bs.tsv` has its sidecar and its row all the
  same; `de.tsv`, `es.tsv`, `ja.tsv` and `zh-cn.tsv` have neither.
- The fork has no Czech, Polish, Swedish or Turkish catalogs; upstream added them before
  `e6df3186`, with Bosnian. They could be taken the way AV49 took Bosnian.
- `graphics.template.export` and `graphics.template.replaceFonts` still open their dialogs over
  `engine.execute` (upstream left them too, and `graphics_templates_ui` relies on it), and
  `help.systemCompatibilityReport` still opens its window as well as answering the report.
  `file.newBin {}` still starts an inline rename. `file.exportFrame` without a `path` now
  exports over `engine.execute`; upstream notes that it writes a file in the working directory,
  which was not tried here.
- Fixes upstream now makes inside `craft-libs` do not reach this repository. The pins in this
  range name some that the fork's in-tree code may want: a bounded TGA decode budget (`craft-tga`
  #1596, `401b0e30`), short lossy JPEG strips and bounded lossy tile allocation in DNG
  (`craft-tiff` / `craft-raw` #1857, #1884), bounded EDL parse diagnostics (`df1f42df`), fallible
  allocations in DNG opcode encoding (`1ccd567a`), a bounded Radiance HDR decoder (`544723c7`).
  The next sync has to triage `storytold/craft-libs` as well, as the last record said.
- `Report::push` counts with `e.count += 1`; upstream's `219c376e` made it saturating while
  adding `push_n`. The node limits of AV25 keep the count far from `u32::MAX`.

The next upstream review starts at `f67d38b3`.
