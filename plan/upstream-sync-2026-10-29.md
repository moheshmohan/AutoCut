# Upstream FilmCraft sync, 2026-10-29

Synced against `origin/main` at `8d925e9d` (2026-10-28). The next sync starts from
`git log --oneline --no-merges 8d925e9d..origin/main`; nothing goes back upstream.

## Taken

- 110 commits cherry-picked with `-x` (`git log --grep 'cherry picked from' main`), conflicts resolved
  so every AutoCut function, command and test stays (892 commands before, 901 after; tests
  2959 → 3066; no public function or file removed).
  Among them `93a845ce` (missing translations are a report in `cargo xtask ci`, not a failure;
  AutoCut keeps its Quick Short check) and `6f919797` (dependency bumps).
- Written again for AutoCut: `ef4a5f83` (the software H.264 encoder takes 4:2:0 through
  `takes_yuv420`, as `a9c3cd31`) and `6829266b` (macOS Edit menu keys in text fields, as
  `1406455e`).
- Adjusted to AutoCut: with `maxThreads` set an export keeps to its cap (no render pool of its own,
  no decode-ahead thread); Clear History forgets the steps but `history.list`'s `log` stays.

## Not taken

Every upstream commit since the previous sync that is not in AutoCut, by group, so any of them can
still be ported.

## Strict-Clippy rewrites (35)
- `0216c8b0` Refactor hevc for strict Clippy checks
- `0792aa48` Refactor cfb for strict Clippy checks
- `08fcd65e` Refactor ac3 for strict Clippy checks
- `0ba02138` Refactor mpeg2v for strict Clippy checks
- `1117dc3e` Refactor matroska for strict Clippy checks
- `11410a33` Refactor vp8 for strict Clippy checks
- `1654de85` Refactor h264enc for strict Clippy checks
- `343ce050` Refactor export for strict Clippy checks
- `3e723263` Refactor interchange for strict Clippy checks
- `3ffd0ebd` Refactor platform for strict Clippy checks
- `5069c311` Refactor h264 for strict Clippy checks
- `650d95df` Refactor project for strict Clippy checks
- `6a5dacb0` Enforce workspace Clippy standards and document lint policy
- `78d77b72` Refactor opus for strict Clippy checks
- `7a532eca` Refactor aac for strict Clippy checks
- `932261b2` Refactor mxf for strict Clippy checks
- `9b9b4109` Refactor tts-text for strict Clippy checks
- `9bd2f834` Refactor ui-egui for strict Clippy checks
- `9c445fe4` Refactor apv for strict Clippy checks
- `9e45a1af` Refactor dnx for strict Clippy checks
- `a286d195` Refactor dng for strict Clippy checks
- `aa3217bd` Refactor mpegts for strict Clippy checks
- `aae601f4` Refactor media for strict Clippy checks
- `ac06ed16` Refactor prores for strict Clippy checks
- `be46e215` Refactor audio-dsp for strict Clippy checks
- `be560f5a` Refactor filmcraft-cli for strict Clippy checks
- `d7356d43` Refactor av1 for strict Clippy checks
- `db860af3` Group ui-egui panel and bench inputs for strict Clippy checks
- `dcce4679` Refactor engine for strict Clippy checks
- `e64f2028` Refactor golden for strict Clippy checks
- `e974b7fe` Refactor gpu for strict Clippy checks
- `ee985b20` Refactor render for strict Clippy checks
- `f0ea988d` Refactor vp9 for strict Clippy checks
- `f317e3cd` Refactor text for strict Clippy checks
- `f3bf3443` Refactor tts for strict Clippy checks

## File splits and test moves (41)
- `01587b2f` refactor(tts-text): move normalize.rs tests to their own file
- `09599eb4` refactor(ui-egui): move i18n.rs tests to their own file
- `0a660de3` refactor(ui-egui): split panels/timeline.rs into layout, clips, waveforms, headers, ruler, scrollbars, snap, menus and interact
- `0f38eb2f` refactor(aac): convert mod.rs files to <module>.rs
- `1564d36d` refactor(interchange): convert mod.rs files to <module>.rs
- `1710e5db` refactor(engine): split settings.rs into prefs, labels, schema, media_cache and commands
- `2c7b7ad2` refactor(tts): convert mod.rs files to <module>.rs
- `3155acdb` refactor(render): convert mod.rs files to <module>.rs
- `34dc5a5c` refactor(ui-egui): split app.rs into app/{gpu,ime,prefs,window,playback,textures,files,input,keys,commands,control,frame}
- `45121234` refactor(tts-text): move phonemize.rs tests to their own file
- `5c4ea7a5` refactor(mxf): convert mod.rs files to <module>.rs
- `5fea1fe3` refactor(time): move lib.rs tests to their own file
- `6905c21d` refactor(engine): split clip_ops.rs into enabled, common, file, edit_menu, subclip, video_options, audio_options and replace
- `6944b708` refactor(h264enc): split slice.rs into macroblock, residual, intra, motion and b_slices
- `6d42ef9b` refactor(codecs): move mp4.rs tests to their own file
- `6eb0e752` refactor(ui-egui): convert mod.rs files to <module>.rs
- `7129d59a` refactor(render): move graphic_clip.rs tests to their own file
- `7515153e` refactor(av1): split tile.rs into palette, tx_size, predict, coeffs and loop_restoration
- `7a00cb23` refactor(hevc): split slicedec.rs into coding_unit, transform_tree and prediction_unit
- `7d4bcbcc` refactor(ui-egui): move capture.rs tests to their own file
- `7dbdf9d4` refactor(ui-egui): split panels/graphics.rs into canvas, handles, overlay, ime, layer_identity, text_edit and properties
- `7de6bc95` refactor(automation): move server.rs tests to their own file
- `8066728a` refactor(engine): move module test files to <module>/tests.rs
- `806fbe8a` style: cargo fmt the moved test files
- `81db9bd3` refactor(platform): convert mod.rs files to <module>.rs
- `8fd11240` refactor(opus): convert mod.rs files to <module>.rs
- `8fd96e1f` refactor(ui-egui): move install_ime_reset_callbacks to ime_reset and EXPORT_SAVE_DIALOGS to native_dialogs
- `943aa3fa` refactor(media): move animated test file to animated/tests.rs
- `a7e8e310` refactor(ui-egui): split panels/export_mode.rs into queue, settings, preview, presets and quick
- `a834d817` refactor(text): move layout.rs tests to their own file
- `b9a59126` refactor(export): split lib.rs into format, output, registry, stats, encoders, convert, still_encode and run
- `ba3079dc` refactor(filmcraft): move control_server.rs tests to their own file
- `bb2f29ee` refactor(speech): convert mod.rs files to <module>.rs
- `c0901eec` refactor(ui-egui): move FilmcraftApp methods to app.rs and host, GPU, audio and raw-input items to hooks, gpu, audio and raw_input
- `c369ccbc` refactor(codecs): move gop.rs tests to their own file
- `c52a31cd` refactor(project): split lib.rs into label, item, track, sequence, settings, view and source
- `d9dff5ed` refactor(filmcraft): move graphics.rs tests to their own file
- `e708e3a3` refactor(audio-dsp): convert mod.rs files to <module>.rs
- `f14ce056` refactor(ui-egui): move audio_recovery_tests to its own file
- `f37e055b` refactor(frame): move lib.rs tests to their own file
- `f6402960` refactor(project): move lib.rs tests to their own file

## egui 0.37 migration (8)
- `1c69ff3c` egui-037: document UI and input integration contracts
- `320cae24` Migrate FilmCraft to egui 0.37 with current editing changes preserved (#3249)
- `5d80a5e8` docking: egui 0.37 fallout
- `7d6e0d41` test(ui-egui): skip tests.rs files in the i18n source scan
- `a46b7fa0` egui-037: fix capture failures, UTF-8 input and hidden timing
- `df5f86ae` egui 0.37: migrate FilmCraft input and rendering integration
- `e447a189` egui037: Document capture and IME lifecycle contracts
- `ec55f2bd` egui037: simplify UI helpers and preserve pointer IME cancellation

## New docking and floating windows (41)
- `0169eba9` UI.9: no move or float leaves the frame tree empty
- `08816eac` UI.9: register every dock divider; any focused divider keeps shortcuts live
- `1673d42d` docking: saved workspaces load fixed split sizes as ratios
- `172ccf34` docking: floating panels open in their own OS windows
- `190f5def` docking: craft-libs 865c9af
- `217374ba` docking: craft-libs 5bbe9f6 (main)
- `218c03fd` UI.9: repin craft-ui; a press on a sequence tab focuses the Timeline
- `26d0df20` UI.9: grips move Tools and the Audio Meters
- `2a19266b` UI.9: draw panels with craft-ui shared docking (Frame profile)
- `2ff95d03` docking: craft-libs 6e7f285
- `34961f28` docking: describe native window kinds, modal blocking and drops
- `359b744e` UI.9: dock tests expect no focus on a divider or grip after a pointer drag
- `364d0ba1` docking: craft-libs 8a503cf
- `36555784` docking: craft-libs' one docking model; Tools and Audio Meters are ordinary panels
- `3b939a1f` UI.9: only docked frames maximize
- `5787e7e5` docking: rustfmt
- `57959eb4` UI.9: Tools and the Audio Meters keep their factory widths
- `583fc2eb` docking: craft-libs d24e4bc (Mac round 3)
- `5963690c` UI.9: follow craft-ui docking at 958c5ad
- `6881a100` UI.9: report docking layout and gesture rects over the control channel
- `7a8a2da2` Update docking docs and drop leftovers
- `7f77bd73` docking: craft-libs 16fb3d4 with shared docking and native windows
- `8466f6d8` UI.9: use craft-ui's tab wheel, middle click, edge dock and widget ids
- `8525f836` UI.9: follow craft-ui docking at 2fa4ad7
- `85d1c820` docking: docs for the split-tree settings and the new divider ids
- `8bbbb600` UI.9: hidden floating windows don't block drops on a maximized Timeline
- `9a31d342` docking: native windows take no input while a modal dialog is open
- `9a590f0f` UI.9: repin craft-ui to craft-libs 352a830
- `a0ee50a6` UI.9: dock resize test checks Space after a pointer resize
- `a2ecd10e` UI.9: shortcuts work while any floating resize edge has focus
- `a31735da` docking: the native-window maximize test reports the main window's outer rect
- `a540c8db` docking: no needless return in the native window input rewrite off macOS
- `a88eed54` Format dock_ui.rs
- `b35b463e` UI.9: tab menus carry FilmCraft's panel items; closing the Timeline closes its sequence
- `bfb181da` docking: input rewrite and file drops in native windows
- `c04497fd` UI.9: window.panel.activate reopens a closed panel
- `c8de47df` docking: video viewers float as normal windows
- `ccb5bd42` docking: Tools and the Audio Meters keep fixed widths
- `d6ac5361` UI.9: a floated panel takes its full size
- `e43097a2` docking: native windows tag headers, block for the quit prompt, take panel drags
- `f23dfeb1` UI.9: floating panels stay out of the macOS title-bar strip

## craft-libs pins (7)
- `1f0ecaa1` craft-libs: pin every crate to 1e6f1278 (craft-libs main with #353)
- `2afdf2e1` craft-libs: pin every crate to 37c907d2 (craft-libs#353: a drop beside a too-narrow frame at its row's edge)
- `3a22804d` craft-libs: pin every crate to b926b5c0 (craft-libs main with #355)
- `5e8c3219` craft-libs: pin every crate to cd25951c (craft-libs#355: native floating windows publish their IME descriptor)
- `61ab6384` craft-libs: pin every crate to 1a14a680 (craft-libs#340 stack, for craft_ui::ime::Composer)
- `e58b48ae` craft-libs: pin every crate to e8b4dde5 (craft-libs#340 stack, for craft_ui::ime::Composer)
- `fa050260` build: pin craft-update to craft-libs main (495e597) (#3322)

## IME composer (4)
- `1fdab222` ime: the Program monitor's text editor publishes through craft_ui::ime::Composer
- `5a6096b5` ime: an edit that ends while composing commits the marked text (#2445)
- `add6bafe` ime: a floating Program monitor window runs its IME check after its last body
- `fe3e59f6` ime: the Program monitor's text editor draws its preedit through craft_ui::ime::Composer

## Self-updater (2)
- `df106680` feat: startup update prompt modal and direct self-updating (#3298)
- `e6d82d4e` Grant network access so update checks can reach GitHub (#3527)

## Windows installer (9)
- `003a2f56` docs(installer): note that the 32-bit all-users folder is Program Files (x86); tidy wording
- `040d23a0` docs(installer): describe the per-user default install location and scope
- `058050f3` fix(installer): address scope and ARM64 review feedback
- `48681c48` fix(installer): resolve the wizard scope and folder correctly
- `8ba3f917` fix(installer): harden install-policy checks and packaging-lint workflow
- `9efa1e87` feat(installer): default to a per-user ArtCraft folder
- `a9bd7705` ci: use storytold/craft-common-actions at the merged installer-policy commit
- `bfee7b1b` test(installer): cover the compiled folder searches, resolver chains and wizard folder replay
- `c7887b74` fix(installer): check the exact architecture-specific machine folder

## CI, release and packaging (11)
- `0a881266` CI: xtask ci in the dev profile, one test binary per crate, nextest (#2860)
- `20f02b9a` release: the git-sources gate moves into craft-common-actions (storytold/craft-common-actions#78) (#3014)
- `30536bc5` Release 0.13.0
- `491af029` ci: require nextest 0.9.150
- `4d089668` packaging: use [[ for bash conditional tests (SonarCloud S7688) (#3003)
- `8d925e9d` Honor CARGO_BUILD_TARGET in FreeBSD packaging for aarch64 release builds (#3534)
- `a2d8e9ce` Release 0.14.0
- `c3b4c4de` Packaging: pin the signing client and appimagetool, and check their hashes (#2610)
- `e45ff583` ci(sonar): don't cancel superseded runs on main, so coverage is published (#3026)
- `e65d94a3` macOS packaging: retry stapler when Apple's CloudKit drops the connection
- `f46a1a84` CI: run the gate through craft-common-actions' shared gate.yml; FreeBSD nightly and on release (#2937)

## SonarCloud clean-ups (5)
- `20e159dd` tools: quit the super-linear regex scans in the spec table extractors (SonarCloud S8786) (#3007)
- `589fc5ad` refactor: use method paths instead of closures (S1612) (#3335)
- `7725f486` refactor: clear S9045 and S107 new-code issues (#3336)
- `a1232e25` test: split fuzz sweeps into helpers (S3776) (#3337)
- `de529c88` chore: resolve SonarCloud maintainability issues (#3329)

## craft-media move (1)
- `fa550ce2` Build: use craft-media for bitstream, isobmff and matroska (#2435)

## craft-fonts pin (1)
- `fbeccbe9` Fonts: pin craft-fonts bfcbf71; CRAFT_FONTS_REQUIRED fails without CRAFT_FONTS_DIR (#2646)

## Translations (1)
- `70ee7b18` i18n(fr): translate the 42 strings French still lacked (#3226)

## Docs only (2)
- `3a5f217b` docs: add structured issue intake and security reporting guidance (#2054)
- `7749c416` docs: gaps.md revision history records the G8 cut drag

## Test-only (1)
- `8ec57c32` test(engine): skip tests.rs and tests/ in the wasm clock source scan

## Linux Wayland (1)
- `7d8dd5ae` Linux: default to XWayland for file drag-and-drop on Wayland desktops (#634)

## Other (1)
- `70375a3e` refactor(ui): use craft-ui's ComboBox for every dropdown (#3445)

