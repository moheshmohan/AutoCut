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
| 0.1 | Rebuild + relaunch AutoCut.app on :9876 | **Rebuilt and relaunched 2026-10-17** (`fd00c42d`, `packaging/macos/package.sh` with the AutoCut env vars); checking Whisper transcription, Delivery, albums and frame fill in the app is part of 0.7 |
| 0.2 | Resource locks: Chrome, GPU | **Done 2026-10-17** (`crates/locks`): `<data dir>/locks/{chrome,gpu}.lock`, OS file locks naming the holder; the picker agent and `photos tidy` hold the browser, VoxCPM narration the GPU; busy waits then fails naming the holder; `filmcraft-cli locks [hold]`; CLAUDE.md tells sessions to hold `chrome` while driving it. Ollama left out: it queues requests itself |
| 0.3 | Worktrees start from `main` | **Done 2026-10-17**: `.claude/settings.json` `worktree.baseRef: "head"` — Claude Code cut agent worktrees from `origin/HEAD` (upstream on this fork); now from the local HEAD, so start them from the main checkout on `main` |
| 0.4 | Fast gate + nightly full gate | **Done 2026-10-17** (`7aa5581a`): `cargo xtask ci --changed`; full `cargo xtask ci` before each push. Not yet tried: `cargo nextest`, `sccache` |
| 0.5 | Fork CI | **Done 2026-10-17**: heavy upstream workflows disabled on the fork (GitHub settings); `autocut-checks.yml` runs fmt, layers, assets on push |
| 0.6 | Disk hygiene | **Done 2026-10-17**: `cargo xtask sweep [--days N] [--dev] [--dry-run]` (agents' builds and test outputs untouched for N days, incremental caches; skips a profile a build holds); weekly and after each lane merge (docs/contributing.md §5) |
| 0.7 | One-command live check | **Done 2026-10-17**: `cargo xtask live [--only …] [--google] [--picker] [--voice V] [--clip F]` — models, screen grounding, Whisper, a cloned line heard back, analysis-cache hit, a short showcase exported with QA; Google album/upload and the picker opt-in; report in `target/live/<time>/` (docs/testing.md). First run: all 6 default stages pass (voice 100 % heard, 34 s show, QA 0 errors, −14.1 LUFS) |
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

## 4. Decisions (answered 2026-10-17)

1. **YouTube**: reuse the `autocut-photos` GCP project (Testing mode) with the YouTube Data API
   and `youtube.upload` + `youtube.readonly` scopes. Uploads from an unaudited project stay
   private (YouTube's rule since July 2020) — fine as a review step; apply for the API audit when
   the uploader is proven. The adapter must handle **Brand Accounts**: connect several channels
   (each sign-in grants one channel, picked in Google's account chooser), list them, and choose
   the channel per upload; tokens per channel. **Default channel: H4hacks (@H4hacks)**; the
   user's others are Mohesh Mohan (@MoheshMohan), S4Saranya, Life Of 'S'. Done 2026-10-17: the
   YouTube Data API v3 is enabled on `autocut-photos`; no consent-screen scopes are listed (the app
   asks at sign-in, as Google Photos does). If Google refuses a Brand Account at sign-in (Testing
   mode), add its address as a test user — the user has agreed to that.
2. **Instagram**: the **Instagram API with Instagram Login** (no Facebook Page needed; a
   professional — Creator or Business — account, the user's own). A Meta developer app of the
   user's, the user's Instagram account as its tester; set up from the main session in Chrome.
3. **Soundfont**: GeneralUser GS (licence v2.0, permissive; its author notes sample provenance
   is not fully certain) is **downloaded on first use with consent** (licence shown), like the
   voice and speech models — never bundled in the repo; the oscillator voices stay the fallback.
4. **ACE-Step 1.5**: **proceed** — an optional neural text-to-music backend next to the
   procedural engine (D2, MIT weights on candle, downloaded with consent like VoxCPM); started
   2026-10-17 as its own lane.
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
