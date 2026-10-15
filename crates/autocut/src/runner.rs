//! The AutoCut loop: classify, plan, supervise, execute, report.

use crate::config::{Config, Policy};
use crate::digest::{Digest, registry_digest};
use crate::intent::{self, Classification};
use crate::llm::ChatModel;
use crate::ollama::Ollama;
use crate::planner::{self, Plan};
use crate::supervisor;
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

/// The models this run uses; injectable so tests (and the supervisor-only mode) can script them.
pub struct Runners {
    pub local: Box<dyn ChatModel>,
    pub supervisor: Option<Box<dyn ChatModel>>,
}

impl Runners {
    /// The models the [`Config`] names: the local Ollama model, plus the supervisor when a key
    /// is configured.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            local: Box::new(Ollama::new(&cfg.ollama_host, &cfg.llm_model, cfg.http_timeout)),
            supervisor: cfg.anthropic_key.as_deref().map(|k| Box::new(crate::anthropic::Anthropic::new(&cfg.supervisor_model, k)) as Box<dyn ChatModel>),
        }
    }
}

/// Run one prompt against the target behind `ex`, with the configured models.
pub async fn run(prompt: &str, ex: &dyn Executor, cfg: &Config, mode: Mode) -> Result<Report, AutoCutError> {
    run_with(&Runners::from_config(cfg), prompt, ex, cfg, mode).await
}

/// [`run`] with explicit models (tests, scripting).
pub async fn run_with(runners: &Runners, prompt: &str, ex: &dyn Executor, cfg: &Config, mode: Mode) -> Result<Report, AutoCutError> {
    let digest = Digest::read(ex).await?;
    let mut report =
        Report { prompt: prompt.to_string(), intent: None, confidence: None, source: "local", steps: Vec::new(), results: None, undo: None, notes: Vec::new() };

    // 1. classify locally; fall back to the supervisor when the local model can't (or is down)
    let classification = match intent::classify(runners.local.as_ref(), prompt).await {
        Ok(c) => Some(c),
        Err(e) if runners.supervisor.is_some() && cfg.policy != Policy::Never => {
            report.notes.push(format!("the local model couldn't classify ({e}); asking the supervisor"));
            None
        }
        Err(e) => {
            return Err(AutoCutError::Llm(format!(
                "{e}. Start Ollama (`ollama serve`) and pull the model (`ollama pull {}`), or set ANTHROPIC_API_KEY to let Claude take over",
                cfg.llm_model
            )));
        }
    };

    let use_local = classification.as_ref().is_some_and(|c| c.intent != "custom" && c.confidence >= cfg.confidence_threshold);
    if let Some(c) = &classification {
        report.intent = Some(c.intent.clone());
        report.confidence = Some(c.confidence);
    }

    // 2. plan
    let mut steps: Vec<PlannedStep>;
    if use_local {
        let c = classification.unwrap_or_else(Classification::custom);
        match planner::plan(&c.intent, &c.params, &digest, ex).await? {
            Plan::Steps { steps: s, notes } => {
                steps = s;
                report.notes.extend(notes);
            }
            Plan::GenerateThen { generate, params } => {
                // transcript-dependent planning (highlight reel): transcribe, re-read, plan again
                if mode == Mode::Execute {
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
    } else if cfg.supervisor_available() {
        let sup = runners.supervisor.as_deref().ok_or_else(|| AutoCutError::Llm("no supervisor configured".into()))?;
        let sup = supervisor::plan(sup, prompt, &digest.compact(), &registry_digest()).await?;
        report.source = "supervisor";
        report.notes.extend(sup.notes);
        steps = sup.steps;
    } else {
        return Err(AutoCutError::Llm(
            "no planner could take this prompt: the local model didn't recognize it and no supervisor is configured (set ANTHROPIC_API_KEY)".into(),
        ));
    }

    // 3. review pass
    if cfg.policy == Policy::AlwaysReview
        && report.source == "local"
        && !steps.is_empty()
        && let Some(sup) = runners.supervisor.as_deref()
    {
        match supervisor::review(sup, prompt, &digest.compact(), &steps).await {
            Ok(s) => {
                report.source = "reviewed";
                steps = s.steps;
                report.notes.extend(s.notes);
            }
            Err(e) => report.notes.push(format!("the supervisor review failed and was skipped: {e}")),
        }
    }

    validate::validate(&steps).map_err(AutoCutError::Invalid)?;
    report.steps = steps.clone();

    // 4. execute
    if mode == Mode::Execute {
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

    fn cfg(policy: Policy) -> Config {
        Config { policy, confidence_threshold: 0.6, ..Config::default() }
    }

    fn classified(intent: &str, params: serde_json::Value) -> Runners {
        Runners { local: Box::new(FixedModel::new(vec![json!({"intent": intent, "confidence": 0.95, "params": params})])), supervisor: None }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cut_range_cuts_headless() {
        let ex = demo();
        let before = duration(&ex).await;
        let r = run_with(&classified("cut_range", json!({"start": 0.0, "end": 2.0})), "cut the first 2 seconds", &ex, &cfg(Policy::Never), Mode::Execute)
            .await
            .unwrap();
        let cut = before - duration(&ex).await;
        assert!((cut - 2.0).abs() < 0.2, "cut {cut}s, expected ~2s");
        assert!(r.undo.unwrap_or(0) >= 1, "{r:?}");
        assert!(r.steps.iter().any(|s| s.id == "sequence.extract"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dry_run_changes_nothing() {
        let ex = demo();
        let before = duration(&ex).await;
        run_with(&classified("cut_range", json!({"start": 0.0, "end": 2.0})), "cut", &ex, &cfg(Policy::Never), Mode::PlanOnly).await.unwrap();
        assert_eq!(before, duration(&ex).await);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn custom_escalates_or_fails_cleanly() {
        let ex = demo();
        // no supervisor, custom intent: clear error
        let e = run_with(&classified("custom", json!({})), "teleport the clip to mars", &ex, &cfg(Policy::Never), Mode::PlanOnly).await.unwrap_err();
        assert!(e.to_string().contains("no planner"), "{e}");

        // with a scripted supervisor, its plan runs
        let sup = Runners {
            local: Box::new(FixedModel::new(vec![json!({"intent": "custom", "confidence": 0.1})])),
            supervisor: Some(Box::new(FixedModel::new(vec![
                json!({"steps": [{"id": "sequence.addEditAllTracks", "params": {"seconds": 2.0}, "why": "test"}], "notes": []}),
            ]))),
        };
        let mut c = cfg(Policy::ComplexOnly);
        c.anthropic_key = Some("test-key".into());
        let r = run_with(&sup, "do something odd", &ex, &c, Mode::Execute).await.unwrap();
        assert_eq!(r.source, "supervisor");
        assert_eq!(r.steps[0].id, "sequence.addEditAllTracks");
        assert!(r.results.as_ref().is_some_and(|v| v[0].ok), "{r:?}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn low_confidence_escalates() {
        let ex = demo();
        let sup = Runners {
            local: Box::new(FixedModel::new(vec![json!({"intent": "add_captions", "confidence": 0.2})])),
            supervisor: Some(Box::new(FixedModel::new(vec![json!({"steps": [], "notes": ["no audio to caption"]})]))),
        };
        let mut c = cfg(Policy::ComplexOnly);
        c.anthropic_key = Some("test-key".into());
        let r = run_with(&sup, "caption?", &ex, &c, Mode::PlanOnly).await.unwrap();
        assert_eq!(r.source, "supervisor");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn no_sequence_fails_clearly() {
        let ex = SessionExecutor::from(Session::default());
        let e = run_with(&classified("cut_range", json!({"start": 0.0, "end": 1.0})), "x", &ex, &cfg(Policy::Never), Mode::PlanOnly).await.unwrap_err();
        assert!(matches!(e, AutoCutError::NoSequence), "{e}");
    }

    #[test]
    fn edit_commands_are_counted_for_undo() {
        assert!(is_edit("sequence.extract"));
        assert!(!is_edit("sequence.inspect"));
    }
}
