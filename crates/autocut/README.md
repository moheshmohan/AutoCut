# filmcraft-autocut

AutoCut: prompt-driven automated editing on top of the FilmCraft engine. Ask in plain language
("remove the silences and add captions", "cut 0:10 to 0:25", "make a 60-second highlight reel
about the launch") and it plans and runs ordinary engine commands — every edit lands in History
and undoes like a human's.

## How it works

```
prompt ──► classify (local LLM, Ollama, JSON mode)
             │  routine intent + confident ──► deterministic planner ──► validated steps
             │  custom / low confidence ────► cloud supervisor (Claude, Anthropic API)
             │                                   with the document digest, captioned
             │                                   frames (local VLM) and the real command
             │                                   catalog; one repair round on bad answers
             ▼
        validate against filmcraft-engine's command registry + an edit-family allowlist
             ▼
        execute through the Executor trait ──► headless Session, or the live app (bridge)
```

- **Local first.** Routine intents never touch the network: phi4-mini classifies, the planner
  emits commands like `transcript.removePauses` / `sequence.extract` / `transcript.createCaptions`.
- **Supervised when needed.** Complex prompts go to Claude with the compact document digest
  (`digest.rs`), up to six VLM-captioned frames (`describe.rs`, moondream/SmolVLM class), and the
  allowed command catalog. Its plan must validate before anything runs (`validate.rs`).
- **Nothing exotic.** AutoCut never edits project state directly; it only runs engine commands,
  so automation, the UI and agents share one undoable truth.

## Intents

| Intent | Commands it plans |
|---|---|
| `remove_silence` | `transcript.generate`? → `transcript.removePauses` |
| `remove_fillers` | `transcript.generate`? → `transcript.removeFillers` |
| `add_captions` | `transcript.generate`? → `transcript.createCaptions` |
| `cut_range` | `markers.markIn/markOut` → `sequence.extract`/`lift` → `markers.clearInOut` |
| `keep_range` | tail-then-head `sequence.extract` (order keeps times valid) |
| `highlight_reel` | `transcript.extract` on the complement of the best stretches, last-first |
| `custom` | the supervisor |

Highlight scoring (`highlight.rs`) is classical: sentence/pause segmentation, topic-keyword and
salience scoring, duration shaping — no model needed to pick the cuts.

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `AUTOCUT_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `AUTOCUT_LLM_MODEL` | `phi4-mini` | classifier / planner model |
| `AUTOCUT_VLM_MODEL` | `qwen2.5vl:7b` | frame-captioning model (moondream works well) |
| `AUTOCUT_ANTHROPIC_KEY` / `ANTHROPIC_API_KEY` | — | enables the supervisor |
| `AUTOCUT_SUPERVISOR_MODEL` | `claude-sonnet-4-5` | supervisor model |
| `AUTOCUT_POLICY` | `complex-only` | `never` \| `complex-only` \| `always-review` |
| `AUTOCUT_CONFIDENCE_THRESHOLD` | `0.6` | below this, escalate |

## Surfaces

- **CLI:** `filmcraft-cli autocut '<prompt>' [--dry-run]` (headless `--demo`/`--project`, or
  `--bridge` to edit in the running app).
- **MCP:** the `autocut_plan` (dry run) and `autocut_edit` tools, on both MCP backends
  (`filmcraft-cli mcp --demo` / `--bridge`).

## Tests

`cargo test -p filmcraft-autocut` is fully offline: scripted models (`llm::FixedModel`), the demo
project headlessly, crafted transcripts through `transcript.set`. Live checks
(`tests/live_models.rs`) are opt-in: `AUTOCUT_LIVE=1` (+ `AUTOCUT_SUPERVISOR_LIVE=1` with an API
key for the supervisor round-trip).
