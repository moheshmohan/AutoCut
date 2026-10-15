//! The AutoCut loop: classify, plan, execute, report — and hand off to the supervisor (the
//! MCP client driving the editor) when a prompt is beyond the routine catalog.

use crate::config::Config;
use crate::digest::Digest;
use crate::intent;
use crate::llm::ChatModel;
use crate::ollama::Ollama;
use crate::planner::{self, Plan};
use crate::validate;
use crate::{AutoCutError, Executor, PlannedStep, Report, StepResult};

/// Whether to stop after planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Plan and validate only; nothing runs.
    PlanOnly,
    /// Plan, validate and execute.
    Execute,
}

/// Per-request options.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub mode: Mode,
    /// Caption sampled frames with the local vision model and attach them to the document on a
    /// supervisor hand-off (slower; the supervisor gets eyes).
    pub frames: bool,
}

impl Options {
    pub fn plan_only() -> Self {
        Self { mode: Mode::PlanOnly, frames: false }
    }
    pub fn execute() -> Self {
        Self { mode: Mode::Execute, frames: false }
    }
}

impl From<Mode> for Options {
    fn from(mode: Mode) -> Self {
        Self { mode, frames: false }
    }
}

/// The models this run uses; injectable so tests can script them.
pub struct Runners {
    pub local: Box<dyn ChatModel>,
    /// The local vision model, for frame descriptions the supervisor sees. `None` = no eyes.
    pub vlm: Option<Ollama>,
}

impl Runners {
    /// The models the [`Config`] names.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            local: Box::new(Ollama::new(&cfg.ollama_host, &cfg.llm_model, cfg.http_timeout)),
            vlm: Some(Ollama::new(&cfg.ollama_host, &cfg.vlm_model, cfg.http_timeout)),
        }
    }
}

/// Run one prompt against the target behind `ex`, with the configured models.
pub async fn run(prompt: &str, ex: &dyn Executor, cfg: &Config, mode: Mode) -> Result<Report, AutoCutError> {
    run_with(&Runners::from_config(cfg), prompt, ex, cfg, mode.into()).await
}

/// [`run`] with explicit models (tests, scripting).
pub async fn run_with(runners: &Runners, prompt: &str, ex: &dyn Executor, cfg: &Config, opts: Options) -> Result<Report, AutoCutError> {
    let digest = Digest::read(ex).await?;
    let mut report = Report {
        prompt: prompt.to_string(),
        intent: None,
        confidence: None,
        source: "local",
        needs_supervisor: false,
        document: None,
        steps: Vec::new(),
        results: None,
        undo: None,
        notes: Vec::new(),
    };

    // 1. classify locally; when the local model is unreachable, the caller (an MCP agent) can
    //    still plan the edit itself with the engine's tools, so say exactly that
    let classification = match intent::classify(runners.local.as_ref(), prompt).await {
        Ok(c) => c,
        Err(e) => {
            return Err(AutoCutError::Llm(format!(
                "{e}. Start Ollama (`ollama serve`) and pull the model (`ollama pull {}`) — or, when an agent is driving the editor, let it plan this edit itself with command_list / command_run",
                cfg.llm_model
            )));
        }
    };

    let use_local = classification.intent != "custom" && classification.confidence >= cfg.confidence_threshold;
    report.intent = Some(classification.intent.clone());
    report.confidence = Some(classification.confidence);

    // 2. plan — or hand off
    let steps: Vec<PlannedStep>;
    if use_local {
        match planner::plan(&classification.intent, &classification.params, &digest, ex).await? {
            Plan::Steps { steps: s, notes } => {
                steps = s;
                report.notes.extend(notes);
            }
            Plan::GenerateThen { generate, params } => {
                // transcript-dependent planning (highlight reel): transcribe, re-read, plan again
                if opts.mode == Mode::Execute {
                    run_steps(std::slice::from_ref(&generate), ex, &mut report).await;
                } else {
                    report.steps = vec![generate];
                    report.notes.push("dry run stops before the transcript exists; run it to see the full plan".into());
                    return Ok(report);
                }
                let digest = Digest::read(ex).await?;
                match planner::plan("highlight_reel", &params, &digest, ex).await? {
                    Plan::Steps { steps: s, notes } => {
                        steps = s;
                        report.notes.extend(notes);
                    }
                    Plan::GenerateThen { .. } => return Err(AutoCutError::Other("transcription produced no transcript".into())),
                }
            }
        }
    } else {
        // the supervisor is whoever is on the other side: return the context it needs to plan
        // the edit itself with the engine's tools
        report.source = "supervisor";
        report.needs_supervisor = true;
        report.steps = Vec::new();
        report.notes.push(
            "this prompt is beyond AutoCut's routine catalog; the caller should plan it with the engine's commands (command_list / command_run / command_batch, render_preview to check)".into(),
        );
        let mut doc = digest.compact();
        if opts.frames
            && let Some(vlm) = &runners.vlm
        {
            let times = crate::describe::sample_times(digest.duration_seconds().min(120.0), 6);
            if let Ok(frames) = crate::describe::describe_frames(vlm, ex, &times, 448).await
                && !frames.is_empty()
            {
                doc["frames"] = crate::describe::to_json(&frames);
            }
        }
        report.document = Some(doc);
        return Ok(report);
    }

    validate::validate(&steps).map_err(AutoCutError::Invalid)?;
    report.steps = steps.clone();

    // 3. execute
    if opts.mode == Mode::Execute {
        run_steps(&steps, ex, &mut report).await;
    }
    Ok(report)
}

/// Run the steps in order (stopping at the first failure), filling in the report.
async fn run_steps(steps: &[PlannedStep], ex: &dyn Executor, report: &mut Report) {
    let mut results = Vec::new();
    let mut undo = 0usize;
    for s in steps {
        match ex.execute(&s.id, s.params.clone()).await {
            Ok(v) => {
                results.push(StepResult { id: s.id.clone(), ok: true, result: Some(v), error: None });
                undo += usize::from(is_edit(&s.id));
            }
            Err(e) => {
                results.push(StepResult { id: s.id.clone(), ok: false, result: None, error: Some(e.clone()) });
                report.notes.push(format!("`{}` failed: {e}; stopping", s.id));
                break;
            }
        }
    }
    report.undo = Some(undo);
    report.results = Some(results);
}

/// Whether a command journals an undoable edit (its spec says so).
fn is_edit(id: &str) -> bool {
    filmcraft_engine::commands::find(id).is_some_and(|s| s.journal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SessionExecutor;
    use crate::llm::FixedModel;
    use filmcraft_engine::Session;
    use serde_json::json;

    fn demo() -> SessionExecutor {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        SessionExecutor::from(s)
    }

    async fn duration(ex: &SessionExecutor) -> f64 {
        let ticks = ex.execute("sequence.inspect", json!({})).await.unwrap()["duration"].as_f64().unwrap_or(0.0);
        ticks / filmcraft_time::TICKS_PER_SECOND as f64
    }

    fn cfg() -> Config {
        Config::default()
    }

    fn classified(intent: &str, params: serde_json::Value) -> Runners {
        Runners { local: Box::new(FixedModel::new(vec![json!({"intent": intent, "confidence": 0.95, "params": params})])), vlm: None }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cut_range_cuts_headless() {
        let ex = demo();
        let before = duration(&ex).await;
        let r =
            run_with(&classified("cut_range", json!({"start": 0.0, "end": 2.0})), "cut the first 2 seconds", &ex, &cfg(), Options::execute()).await.unwrap();
        let cut = before - duration(&ex).await;
        assert!((cut - 2.0).abs() < 0.2, "cut {cut}s, expected ~2s");
        assert!(r.undo.unwrap_or(0) >= 1, "{r:?}");
        assert!(r.steps.iter().any(|s| s.id == "sequence.extract"));
        assert!(!r.needs_supervisor);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dry_run_changes_nothing() {
        let ex = demo();
        let before = duration(&ex).await;
        run_with(&classified("cut_range", json!({"start": 0.0, "end": 2.0})), "cut", &ex, &cfg(), Options::plan_only()).await.unwrap();
        assert_eq!(before, duration(&ex).await);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn complex_prompts_hand_off_to_the_caller() {
        let ex = demo();
        let r = run_with(&classified("custom", json!({})), "teleport the clip to mars", &ex, &cfg(), Options::plan_only()).await.unwrap();
        assert!(r.needs_supervisor);
        assert_eq!(r.source, "supervisor");
        assert!(r.steps.is_empty());
        let doc = r.document.clone().expect("the context to plan with");
        assert!(doc["sequence"]["durationSeconds"].as_f64().unwrap_or(0.0) > 0.0, "{doc}");
        assert!(r.notes.iter().any(|n| n.contains("command_run")), "{r:?}");
        assert!(r.results.is_none(), "nothing ran");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn low_confidence_hands_off_too() {
        let ex = demo();
        let r = Runners { local: Box::new(FixedModel::new(vec![json!({"intent": "add_captions", "confidence": 0.2})])), vlm: None };
        let out = run_with(&r, "caption?", &ex, &cfg(), Options::plan_only()).await.unwrap();
        assert!(out.needs_supervisor, "{out:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_sequence_fails_clearly() {
        let ex = SessionExecutor::from(Session::default());
        let e = run_with(&classified("cut_range", json!({"start": 0.0, "end": 1.0})), "x", &ex, &cfg(), Options::plan_only()).await.unwrap_err();
        assert!(matches!(e, AutoCutError::NoSequence), "{e}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unreachable_local_model_names_the_way_out() {
        let ex = demo();
        let mut c = cfg();
        c.ollama_host = "http://127.0.0.1:1".into();
        c.http_timeout = std::time::Duration::from_secs(2);
        let e = run("anything", &ex, &c, Mode::PlanOnly).await.unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("ollama serve") && msg.contains("command_run"), "{msg}");
    }

    #[test]
    fn edit_commands_are_counted_for_undo() {
        assert!(is_edit("sequence.extract"));
        assert!(!is_edit("sequence.inspect"));
    }
}
