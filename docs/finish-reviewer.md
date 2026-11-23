# The finish reviewer

The last look at a film before it ships is given to a reviewer that did not make it: an agent
started in a fresh context, with the evidence of the film and nothing else. It answers one of
four verdicts in a fixed form that AutoCut checks and records. This page is the contract: the
brief the reviewer gets, the evidence it reads, the answers it may give, the limits, and the
commands around it. It is stage 9 of the pipeline (review), after the 1080p review build and
before the 4K delivery (`delivery.preflight`, `delivery.prepare`).

The ideas (a reviewer in a fresh context, a bounded list of fixes with one thing to keep, a
final gate line) come from impeccable (Apache-2.0), hallmark (MIT) and taste-skill (MIT); the
rules and text here are written fresh.

## Why a fresh context

The agent that made the edit knows why every shot is there, and so it sees what it meant rather
than what is on screen. A reviewer with only the evidence sees what a viewer sees. So the
reviewer is started anew for each round, is given the packet folder and this brief, and is not
told the plan, the script's intent or the earlier rounds' discussion. What was decided on purpose
reaches it only through `decisions.json`, with the reasons and evidence written there.

## The packet

`review.packet` gathers the evidence into a new folder in the build folder,
`build/review/packet-<n>` beside the saved project (or `out`, a new or empty folder):

| file | what it is |
|---|---|
| `brief.md` | the brief below, with the round number filled in. The reviewer reads it first |
| `comments.json` | the review comments on this sequence: `open` (each needs an answer) and `done` (with the note of what was changed), from `build/review/inbox.jsonl` |
| `frames.jpg` | one frame of each shot (the middle of each stretch between cuts), in order, labelled with its number and timecode, captions on; at most `frames` (default 40, at most 120), and at least eight spread over a film of few shots |
| `strips/` | motion filmstrips (`sequence.motionStrips`): seven frames of every animated title, transition and overlay, and their findings. Missing when nothing moves |
| `qa.json` | the latest quality report: `qa` if given, else the newest `*.qa.json` in `build/` or beside the project, or `delivery.preflight` run now with `preflight: true` (slow: it renders and measures the whole film) |
| `motion.json` | `sequence.checkMotion`: eases, entrance and exit timing, stagger, loops, pops, read from the timeline |
| `decisions.json` | a copy of `build/decisions.json`: the waivers, each with its reason and evidence |
| `sequence.json` | the film's size, rate, length, shot count and cut times, titles (their text and times) and caption count |
| `rounds.json` | the review runs so far |
| `manifest.json` | what is in the folder, what is missing and why, the run and round |

`review.packet {out?, sequence?, frames?, strips?, qa?, preflight?, wait?}` answers the manifest
with `folder`, `brief` and `next`. The filmstrips are a job unless `wait: true` (`job` in the
answer; `jobs.list` shows their sheets and findings under `motionStrips`); over MCP,
`command_run review.packet {wait: true}` is a long call that waits for them. Anything that cannot
be gathered is listed in `missing` with the reason; it does not stop the packet.

## The brief

This is `brief.md` (the engine's `review_finish::BRIEF`):

> You are the finish reviewer of a short film. You did not make it. Judge it from the evidence in
> this folder alone: what you were told about it before this brief does not count.
>
> Read in this order: `comments.json` (every open comment gets an answer: a FIX line, or the reason
> it needs none), `frames.jpg`, `strips/`, `qa.json` and `motion.json`, `decisions.json` (what was
> decided on purpose: do not reopen it), `sequence.json`.

It then gives the four verdicts and the answer's form, below.

## The four answers

At most eight fixes and one keep in an answer, and two rounds at most in a run.

- **ship**: it is ready; nothing left is worth another round.
- **fix**: one to eight changes, the most important first, each on its own line, and at most one
  thing to keep:

  ```text
  FIX 1: 00:00:12:03 (c3) the title enters in four frames: ease it in over half a second
  FIX 2: 41.5s the dissolve dips to grey between two bright shots; cut straight
  KEEP: the music hit on the reveal at 00:00:42:00
  ```

  A FIX line names where (a timecode, seconds like `41.5s`, or a comment id like `c3`) and says
  what to change and how to tell it is right. KEEP names what must not change while fixing.
- **rebuild**: the cut does not work as it is (structure, order, pacing); the GATE line says why.
- **recapture**: material is missing or unusable (a shot, a sound, a take); the GATE line says
  what to shoot, fetch or generate again.

## The GATE line

The answer ends on exactly one gate line, its last line (blank lines after it are fine):

```text
GATE: ship
GATE: fix <n> [keep 1]
GATE: rebuild - <why>
GATE: recapture - <what>
```

- `GATE:` is upper case; markdown around the line (`**GATE:**`, a leading `-` or `>`) is let in.
  The verdict is any case.
- `fix` may give its counts: `GATE: fix 3` or `GATE: fix 3 keep 1`. When given they must match the
  FIX and KEEP lines; when left out the lines are counted.
- After the verdict (and counts), a dash (`-`, `--`, `–`, `—`) or a colon starts the note: required
  for rebuild and recapture, optional otherwise (`GATE: ship - nothing left`).
- FIX and KEEP lines are `FIX n:` / `FIX:` and `KEEP:` at the start of a line (any case, numbered
  or not). Other lines are the reviewer's notes and are not read.

`review.gate {text}` refuses an answer, with the reason, when: there is no `GATE:` line, more than
one, or it is not the last line; the verdict is not one of the four; a fix lists no FIX line or
more than eight; there is more than one KEEP line; the counts on the GATE line do not match the
lines; a ship lists FIX lines; a rebuild or recapture gives no reason; a FIX or KEEP line is
empty; or the answer is over 64 KB. It warns (and records) when FIX lines name no time or
comment, are numbered out of order, or come with a rebuild or recapture (kept as notes).

## Rounds

A review run has at most two rounds. Each round is a fresh reviewer on a new packet:

1. Round 1: `review.packet`, a fresh reviewer, `review.gate {text, round: 1}`.
2. A **ship** closes the run (`closed.by: "gate"`): deliver.
3. A **fix**, **rebuild** or **recapture** in round 1: make the fixes without touching what KEEP
   names, mark the comments they answer done (`review.done`), build the review cut again, then a
   new packet and a fresh reviewer for round 2.
4. Round 2 without a ship closes the run at its limit (`closed.by: "limit"`): the verdict and its
   list go to the user, who decides. A third round is refused (`round: 3`, or any answer while the
   run waits for the user).

Only the user's own words end a run early, or end one that used its two rounds so a new run can
start: `review.gate {user: "<their words, as they said them>", verdict?}` records the words
(`closed.by: "user"`). An agent never closes a run on its own judgement.

`review.gate` records each round in `build/review/rounds.json` (`{format: "autocut.reviewRounds",
version: 1, runs: [{run, started, rounds: [{round, verdict, fixes, keep, summary, warnings, at,
packet}], closed?}]}`; the packet by its folder name only, at most 50 runs kept). It answers
`{recorded, run, round, gate, closed, next, blocksDelivery: false}`. `dryRun: true` checks an
answer without recording it; `review.gate {}` lists the runs and what the next round would be.

## Not yet a block

Decision D7 (warn first): new checks warn for one round of real projects before any blocks. The
gate is recorded and reported, and `blocksDelivery` is always false: `delivery.preflight` does not
read the rounds yet. The parser itself is strict: a malformed answer is refused, so a reviewer
that does not follow the form is asked again rather than recorded wrong.

## Commands

| command | params |
|---|---|
| `review.packet` | `{out?, sequence?, frames?: 1..120 = 40, strips?: bool = true, qa?: path, preflight?: bool = false, wait?: bool = false}` |
| `review.gate` | `{text, round?: 1 \| 2, packet?: path, dryRun?: bool}`, or `{user, verdict?}`, or `{}` |
| `review.comment`, `review.inbox`, `review.look`, `review.done`, `review.delete` | the review comments the reviewer answers; see [local-pipeline-upgrades.md](local-pipeline-upgrades.md#review-pointed-at-frames) |

All are engine commands: MCP `command_run`, `filmcraft-cli exec`, the control channel.
