# AV17 — upstream integration, 8 November 2026

## Scope and rollback

Compare `storytold/filmcraft` (the `origin` remote), freshly fetched at
`08f8075a`, with the last audited upstream point, `8d925e9d`.
There are 746 non-merge commits in this range. AutoCut starts at
`ea52bcff991438a557116322a03359886b4fd3b9`; the rollback branch is
`codex/backup-before-upstream-2026-11-08`. Work happens on
`codex/upstream-sync-2026-11-08`. This sync does not push either remote.

## Plan and acceptance tests (before implementation)

1. Inventory upstream commits, paths, dependencies and the existing fork.
   Capture the 901-command registry, default app/CLI/engine features and 837
   tracked files in the AI, engine, UI and integration areas. Keep unrelated
   untracked files and all user media/projects untouched.
2. Integrate compatible upstream fixes, adapting them to this fork rather than
   replacing whole files. Priorities: Lumetri curve correctness and shadow
   colour, bounded Warp Stabilizer analysis and cached corrections, upload-ready
   MP4s and encoder fallback, caption import/export, narration normalization,
   model validation, exact timeline parameters, safe saves and bounded input.
3. Retain cloned Malayalam/English narration, Whisper, scene understanding,
   automatic editing, privacy tracking, 360/gyro handling, templates, generated
   audio, Google Photos, YouTube, Instagram and the screen agent. Keep the
   existing default neural-model features and GPU backends. Do not introduce
   mandatory model downloads or change existing projects.
4. Defer the upstream-wide egui/docking migration, codec/dependency relocations,
   release/updater/packaging changes and unrelated contributor edits. They need
   separate migration work and are not prerequisites for these fixes. Maintain
   a per-commit disposition ledger; skipped dependencies remain discoverable.
5. Check every original command remains registered and every protected file
   remains present. Compare existing test/function names and review deletions;
   preserve local customizations when resolving conflicts. Add meaningful
   regression coverage where a port needs adaptation.
6. Run focused release tests while iterating, then one full `cargo xtask ci`
   (format, strict Clippy, workspace tests, layers, asset licensing, WASM).
   Reuse `target/agent-main` only in this same checkout. Build both native apps.
7. Smoke-test the rebuilt command registry and AI command schemas, inspect an
   existing finished project without saving it, render representative frames,
   and exercise a small export. Record results and remaining limitations before
   making the single integration commit. No public release or upload is part of
   this task.

## Results

92 upstream commits are integrated (patches, three-way merges or adapted ports).
The [complete disposition ledger](upstream-sync-2026-11-08-commits.csv) records
all 746 commits, including 654 deferred commits and reasons. These counts include
two required regression-test dependencies, `b268f4d0` and `439bc9a5`, the relative
URL implementation `a5bc9332`, and the CI report reuse fix `7e17c704`.

### Changes available in AutoCut

- Colour: non-wrapping Luma/Sat curves, seamless hue curves, near-black shadow
  lift without blue specks, BOM-safe and degenerate-size-safe LUT import, bright
  Luma Key thresholds, bounded Long Shadow/Glint effects.
- Stabilization: block-wise Warp Stabilizer analysis and bounded reuse of
  smoothed corrections/cover scale. The existing 360 and gyro features stay.
- Delivery: fast-start MP4 for disk/memory/reserved outputs, encoder fallback,
  explicit nested bitrate overrides and three platform-specific vertical
  presets with loudness normalization. Existing web hardware encoding and the
  explicit export thread cap stay.
- Editing: effect indices survive additional effects, transition durations fit
  their clips, copied clips keep transitions, duplicate move identities are
  refused, time objects work in explicit placement/move requests, very large
  negative slips are bounded, and settings inspection creates no undo/journal
  entries. Source-size point resolution and recipe recording are retained.
- Captions/narration: BOM and timecode handling, VTT identifiers/arrows, bounded
  caption outline/CEA-708 text work, TTML track languages, small-caption trim
  bounds, English money/decade/fraction readings and Whisper head validation.
  ASS styles, karaoke words and Malayalam voice-style tags are retained.
- Safety: same-second autosave collision retention, short atomic-save temporary
  names, project-save locking, bounded MCP parse-error replies, serialized MCP
  bridge screenshots, bounded transcript work, shortcut-preset protection and
  privacy-safe selection export of source graphics.
- Interchange: relative FCP7/FCPXML/OTIO media URLs decode spaces and percent
  escapes, and relative OTIO exports encode them. Existing import track bounds
  and plain-file-name handling remain in place.
- Development: the translation report reuses the workspace feature selection
  after the full test run. Full workspace tests, doctests and AutoCut-specific
  checks remain enabled.

### Adaptations that protect the fork

- The save lock covers AutoCut subcommands too. Missing `.fcproj` files retain
  AutoCut's automatic creation behavior, with a lock acquired before creation;
  a regression runs four creators concurrently and checks every edit survives.
  Lock failures return errors before editing. This replaces upstream's blanket
  rejection of a missing project; invalid extensions still fail as before.
- Background mask tracking is polled in headless sessions and tied to a
  document epoch. New/demo/open/recovery/close/template replacements advance the
  epoch; AutoCut cutout tracking carries it too, so stale jobs cannot affect the
  next document. Existing tracking and privacy commands are retained.
- New colour helper inputs are bounded and non-finite points are refused. The
  pause-scan bound also covers AutoCut's style-tag parser and marker-span lookup.
- No manifests, lockfile, neural-model defaults, GPU backends, credentials,
  user projects, media or model downloads are changed. The unrelated `.zcode/`
  and `HANDOVER-chatgpt-spectrogram.md` remain untouched.
- The AccessKit 0.27 patch was deferred because this fork uses 0.26.3 with egui
  0.36. Its version assertion was not imported without its dependency migration.
- Duplicate moves retain the fork's `EditError::Other` contract and are checked
  with a set before allocating tracks or touching history. The existing tests
  still require undo/redo, selection, editor state and ID allocation to survive
  a rejected move unchanged.
- Publishing regressions explicitly remux synthetic legacy files with metadata
  last, so fast-start repair coverage survives the new default. A separate test
  requires current native exports to pass without an unnecessary repair. UI
  expectations reflect the new default and six-frame transition limits.

### Validation

Complete for the integration and staged native build. Detailed local evidence is
under `target/upstream-sync-2026-11-08/`.

- Full `cargo xtask ci` passes after the compatibility fixes: formatting, strict
  workspace Clippy, workspace tests/doctests, layers, all 93 attributed assets
  and the 56-crate WASM check. The log reports no failed tests. Focused publishing,
  undo/redo, hardware-preset and OTIO regressions pass too.

- The rebuilt CLI registers all 901 original commands plus `timeline.delete`
  (902 total). Existing AI command schemas are identical. Eight read-only
  catalogues (voices, clones, Malayalam styles, Whisper, privacy, scenes,
  voice isolation and neural music) return exactly the same results as before.
- Four representative frames of the saved Malayalam Chameleon 4K project
  (14, 81.5, 130 and 159 seconds) render byte-identically at half scale. The
  original project SHA-256 is unchanged.
- A one-second native H.264/AAC export has `ftyp, moov, free, mdat` order,
  270×480 at 30 fps, Rec.709 and stereo 48 kHz audio. It imports with no errors;
  three decoded frames and a decoded PCM WAV round-trip succeed (RMS 0.0603).
- FFmpeg/ffprobe are not installed. Optional external-oracle tests therefore
  skip; the export smoke test uses native probing and decoding. Live model
  inference, cloud uploads and platform promotion actions are outside this sync.
- Validation uses the release profile with only link-time optimization disabled
  and codegen units set to 16 for the many test executables; no Cargo profile or
  product optimization setting is changed. The normal `--release --locked`
  native app and CLI build also passes, with the ordinary thin LTO / one codegen
  unit settings. The signed product CLI repeats the same registry, catalogue and
  byte-identical frame checks successfully.
- The signed desktop app opens an independently saved project copy, displays
  the timeline and Program monitor, and shows working scopes and Lumetri curves
  in the Color workspace. The original project file remains unchanged. macOS
  blocked the test app's direct Desktop-file read, so this check used a private
  temporary folder without changing folder permissions.
- No protected file, existing public function, or existing test function was
  removed. All app/CLI/engine manifests, the workspace manifest and lockfile
  remain unchanged.

### Local delivery and rollback

The checked bundle is staged at
`target/upstream-sync-2026-11-08/AutoCut.app`, with its signed CLI beside it.
Local installation uses `target/macos-package/AutoCut.app` and
`target/release/filmcraft-cli`; its final verification is recorded in
`target/upstream-sync-2026-11-08/delivery.json`.

Rollback copies already exist at
`target/macos-package/AutoCut-before-AV17-2026-11-08.app` and
`target/release/filmcraft-cli-before-AV17-2026-11-08`. The source rollback branch
is `codex/backup-before-upstream-2026-11-08`. Before replacing the running app,
the saved EXEED project is confirmed clean with no active jobs; its project,
playhead and UI state are recorded for restoration. No remote is pushed.

### Next sync

Use `08f8075a` as the new audited upstream boundary, and consult this ledger plus
the deferred groups in `upstream-sync-2026-10-29.md`; the selective integration
does not make all upstream history an ancestor of the fork.
