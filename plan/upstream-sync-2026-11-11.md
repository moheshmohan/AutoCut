# AV18 — FilmCraft sync and AutoCut review

Upstream `storytold/filmcraft` advanced from `08f8075a` to `a64b3d37`.
This pass reviews the 23 non-merge commits in that range against AutoCut
`d50a0fb2` and the existing RAEA changes in the working tree.

| Upstream commits | Decision |
|---|---|
| `a371f881`, `c5bad47d`, `b95d1775`, `26274b39` | Port Up/Down stepping to AutoCut's numeric text fields. Preserve its widget and automation structure. |
| `dbb9b5d3`, `3bfdaacd`, `a09cc409`, `31d64d68`, `20befa3e` | Port Lumetri 3D LUT export and its panel menu. Write through host services. Use the displayed clip in the menu and name effects a LUT cannot encode. |
| `66c5dca8` | Port the non-mutating gap check and use it for Clear, Ripple Delete, and the timeline context menu. |
| `f4090d7b` | The corrected comment is already present. |
| `5a19cd32`, `90dba25a`, `78374d9f`, `310fabdc`, `f42735aa` | Defer the H.264 encoder search and pinned bitstream changes as one batch for encoder conformance and performance measurements. |
| `87c8e669`, `f1a06dfe` | Defer the updater changes and dependency revision pending an AutoCut release-path review. |
| `41977bab` | Defer the app-wide panel and terminology rename. Its surface includes saved UI, automation, and translation strings. |
| `a65d1e97`, `0dba3b1e`, `2d536e6a` | Defer the graphics start and adapter ladder together. Exercise recovery on Windows and dual-GPU hardware before adopting. |
| `a64b3d37` | AutoCut already includes natural and cloned voice engines in its default desktop build. The upstream packaging matrix does not apply to this local build. |

The app review also fixes numbered media import at the filesystem root and
rejects sibling numbers outside the maximum sequence window before allocating
frames. This avoids phantom frames from unrelated names and index truncation on
32-bit builds. Tests exercise both cases.

The next upstream review starts at `a64b3d37`; deferred batches remain eligible
for a separately validated port. This range does not make upstream commits
ancestors of the fork.

The native macOS build passed its smoke checks. All 902 existing commands and
their schemas match the installed build. The added command is
`lumetri.exportCube`. Eight voice, transcription, privacy, scene and music
catalogues match. Seven rendered frames from two saved projects are
byte-identical. The source projects' hashes are unchanged. A native H.264/AAC
export re-imported and decoded successfully with fast-start MP4 metadata.
The Color workspace displayed and exported a 17-point LUT from the new menu.
Local Whisper on the CPU transcribed 18 synthetic words and removed three
deliberate pauses.
A transparent image sequence imported at 30 fps, saved, and rendered after
reopening. A missing frame held the previous frame. A distant numbered
sibling was ignored.

The first workspace run found a Clone Trail test fixture collision. Seven
tests requested the same fixture name, and six fixture directories were
created. The shared helper now adds a process-local atomic counter and
creates each directory exclusively.

The full workspace run and corrected engine rerun cover 6,121 passing tests
and 89 explicitly ignored tests. The corrected engine suite has 1,201
passes. The cut-out suite also passed eight parallel runs. Formatting,
strict workspace Clippy, dependency layers, all 93 asset attributions, and
the WebAssembly build passed. External FFmpeg oracles require tools that
were unavailable on this Mac. Live publishing and non-Mac desktop
hardware were outside the verification scope. Local evidence is under
`target/review-2026-11-11/`.
