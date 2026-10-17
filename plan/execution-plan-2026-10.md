# AutoCut execution plan — October 2026

How to deliver `improvement-plan-2026-10.md` (the roadmap: templates, publishing, transitions,
music) with parallel agents without last night's friction. Written 2026-10-17 against
`autocut/main` @ `a94ff779` — 38 commits after the roadmap's `c469d440`.

## 1. What changed since the roadmap was written

| Roadmap item | State now | Consequence |
|---|---|---|
| Upstream housekeeping (`origin` unreconciled) | **Done**: upstream `09aaf9bf` merged (`3a843287`); only one CI-only commit was new | Make it a weekly card instead |
| E1 Whisper in release builds (G19) | **Already on for AutoCut**: `apps/filmcraft` defaults to `whisper` + `neural-voices`; gaps.md is upstream's text. The *running* AutoCut.app predates last night | Rebuild the app and verify (0.1); mark G19 resolved for the fork |
| B Publishing | Google Photos upload, albums and album-seeded showcases shipped. OAuth uses the loopback redirect, but **no PKCE** and tokens are **plaintext** `tokens.json`; Testing-mode tokens lapse after 7 days | B2 becomes a shared `oauth` module (PKCE + Keychain) that Google Photos moves onto first, then YouTube |
| E3 Auto-reframe | Started: the showcase fills 16:9 from portrait clips, crop on the analysed subject centre with an eased tilt (static centre, no tracking) | E3 adds tracking (point tracker → flow) and the 9:16 direction on the same code |
| C4 Audio coupling | Showcase already lays synth whooshes/impacts/risers on transitions | Generalise to user timelines as the roadmap says |
| D1 tier 0 | `audio.generate` mood beds shipped | As planned |
| Not in the roadmap, shipped last night | Narration delivery (warm/excited/…), takes, every cloned line heard back (garble + pitch check); analysis cache and `--screen`; multi-search picker, `photos tidy` | Templates (A) and recipes (A4) can rely on these |

## 2. Phase 0 — make parallel work safe (1–2 days, before roadmap Phase 1)

Last night ran 5 agents/sessions at once; the costs were real: a 243 GB `target/`, Chrome tabs
closed under another session, Ollama contention tripling analysis time, worktrees cut from the
wrong base, 20–30 min gates re-run six times, CI minutes burned on every push.

| # | Item | Done when |
|---|---|---|
| 0.1 | Rebuild + relaunch AutoCut.app on :9876 | `packaging/macos/package.sh` (AutoCut env vars); Whisper transcription, Delivery, albums, frame fill work in the app |
| 0.2 | Resource locks: Chrome, Ollama, GPU | `~/Library/Application Support/FilmCraft/locks/{chrome,ollama,gpu}.lock` (holder, pid, start) taken by `photos pick/tidy`, `analyze`, narration synthesis, live tests; a busy lock waits or fails with "busy: <holder>"; CLAUDE.md tells sessions (and Claude in Chrome use) to respect it |
| 0.3 | Worktrees start from `main` | Agent/session worktrees are cut from local `main` (today they start at upstream `c2f65871`); until fixed, every brief says "ff to main first" |
| 0.4 | Fast gate + nightly full gate | `cargo xtask ci --changed` (fmt, clippy, tests of touched crates, layers, assets) for agents; full `cargo xtask ci` once before each push; try `cargo nextest` + `sccache` |
| 0.5 | Fork CI | Upstream-only workflows (FreeBSD schedule — failing, Windows ARM64 — 34 min a push, Nix, release) disabled on the fork; one macOS job on push |
| 0.6 | Disk hygiene | Each agent builds in `target/agent-<lane>`, deleted on merge; weekly `cargo clean --profile dev` (or cargo-sweep); scratch removed per task. (Cleaned today: 173 → 404 GB free) |
| 0.7 | One-command live check | `cargo xtask live` (env-gated, takes the locks): picker multi-search + tidy, upload into an album, a cloned line heard back, analysis-cache hit, a 10 s showcase build — report in `target/live/`, never touches `~/Movies` |
| 0.8 | Where plans live | **Done 2026-10-17**: `plan/` is tracked; only `plan/premiere/` (the Premiere visual reference) stays gitignored |

## 3. Roadmap phases as parallel lanes

Each lane is one worktree session with its own `target/agent-<lane>`, merged daily by the main
session (gates green, cherry-picked in order). Lanes are chosen so they don't share files; the
hot spots — the command registry (`crates/engine/src/commands.rs`), `crates/automation/src/server.rs`,
`crates/project/src/vtransition.rs`, the CLI's `main.rs` — are touched by one lane at a time or
merged in a fixed order.

**Phase 1 (weeks 1–4)**

| Lane | Work | Files | Needs |
|---|---|---|---|
| T — transitions | C1: ~60 GL Transitions hand-ported to WGSL, CPU oracle + golden tests, attribution per shader | `crates/render/src/transitions/shader.rs`, catalog | GPU lock for golden tests |
| M — music | D1: rustysynth + block arranger (exact length, musical ending), stems, taxonomy, beat maps | `crates/audio-dsp`, `audio.generate` | Decision: GeneralUser GS licence check (else oscillator voices first) |
| O — OAuth | B2: shared `oauth` module (PKCE S256, loopback, Keychain via `crates/platform`), Google Photos migrated onto it, token-expiry handling | `crates/gphotos/src/auth.rs` → new crate, `crates/platform` | Keychain FFI review (the one crate allowed `unsafe`) |
| E — elements | E4: CC0/original stickers, generated whoosh/impact/riser set, animated text presets with sidecars | `assets/`, ATTRIBUTION.md | — |

**Phase 2 (weeks 5–10)** — T: C2 luma-mask + blend-mode engines, C4 beat snap; A: A1 template
format + engine, A2 first 10 starters; Y: B3 YouTube adapter on the `oauth` module (resumable
upload, quota UX, thumbnails, Shorts detection); M: D1 UI ("find music by feel").

**Phase 3 (weeks 11–16)** — I: B4 Instagram + B5 destinations UI + pre-flight; A: A3 authoring/
sharing, A4 recipes; V: E2 caption embedding, E3 vertical factory (tracking + long → shorts).

**Phase 4 (quarter 2)** — C3 flow presets + content-aware wipe; D2 ACE-Step 1.5 on candle (MIT);
E5 adjustment clips + per-track ducking; B6 TikTok.

## 4. Decisions for you

1. **YouTube client ID**: BYO (each user brings a GCP client, as the roadmap recommends). For you
   now: reuse the `autocut-photos` GCP project in Testing mode with the YouTube scope?
2. **Instagram path**: Facebook Login + resumable upload needs a professional account linked to a
   Page. Which account, and is a Page acceptable?
3. **Soundfont**: GeneralUser GS needs its current licence confirmed before bundling; otherwise D1
   starts on the oscillator voices.
4. **ACE-Step 1.5 port (120–250 h)**: commit in Q2, or keep tier 1 only?
5. ~~Plans location~~ — decided: `plan/` is tracked.

## 5. Risks

- Merge hot spots (registry, MCP server, catalog) — fixed merge order, one lane at a time.
- Shared machine: GPU/Ollama/Chrome contention — locks (0.2) before more than two lanes run.
- Quotas and tokens: YouTube 100 uploads/day per project; Google Testing-mode tokens lapse weekly
  (B2 + a reminder); Instagram 100 posts / 400 containers per 24 h.
- Licences: shader-by-shader provenance (MIT only), no Shadertoy/MLT lumas/NC weights; soundfont
  terms; platform logos replaced by generic glyphs.
- Never-crash and clean-room rules apply to every lane (roadmap guardrails).

## 6. First week

| Day | Who | What |
|---|---|---|
| 1 | main session | Phase 0: 0.1 app rebuild, 0.5 fork CI, 0.6 hygiene, 0.8 plans location |
| 1–2 | main session | 0.2 locks, 0.4 fast gate, 0.3 worktree base |
| 2 | main session | 0.7 live check |
| 3–5 | lanes T, M, O, E (sessions) | Phase 1 starts; daily merge + full gate + push |

Carried over from last night (separate small sessions when convenient): picker day-selection
precision; `analyze` without Ollama when cached; album seeds skip our own exports; a livelier
voice sample for the clone (record 2–3 min, re-clone, compare); an expressiveness check that
matches listening (the pitch check missed robot 35).
