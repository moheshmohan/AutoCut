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
    // a showcase is a new video: it needs clips, not an open sequence
    let digest = match Digest::read(ex).await {
        Ok(d) => Some(d),
        Err(AutoCutError::NoSequence) => None,
        Err(e) => return Err(e),
    };
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

    // 1. classify locally ("make a video of every robot since 2020" without a model); when the
    //    local model is unreachable, the caller (an MCP agent) can still plan the edit itself with
    //    the engine's tools, so say exactly that
    let classification = match intent::showcase_request(prompt) {
        Some(c) => c,
        None => match intent::classify(runners.local.as_ref(), prompt).await {
            Ok(c) => c,
            Err(e) => {
                return Err(AutoCutError::Llm(format!(
                    "{e}. Start Ollama (`ollama serve`) and pull the model (`ollama pull {}`) — or, when an agent is driving the editor, let it plan this edit itself with command_list / command_run",
                    cfg.llm_model
                )));
            }
        },
    };
    let digest = match digest {
        Some(d) => d,
        None if classification.intent == "showcase" => Digest { project: serde_json::Value::Null, sequence: serde_json::Value::Null, transcript: None },
        None => return Err(AutoCutError::NoSequence),
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
                if opts.mode != Mode::Execute {
                    report.steps = vec![generate];
                    report.notes.push("dry run stops before the transcript exists; run it to see the full plan".into());
                    return Ok(report);
                }
                let digest = transcribe(&generate, ex, &mut report).await?;
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

/// How long to wait for a transcription that outlived the app's reply limit.
const TRANSCRIBE_WAIT: std::time::Duration = std::time::Duration::from_secs(30 * 60);
/// How often to look for its words meanwhile.
const TRANSCRIBE_POLL: std::time::Duration = std::time::Duration::from_millis(if cfg!(test) { 10 } else { 3000 });

/// Run `generate` (`transcript.generate`) and return the re-read document with its words.
/// Over the bridge a long transcription outlasts the app's 60 s reply limit: the call answers
/// "timeout" while the app keeps transcribing, so a timeout means "wait for the words" — not
/// a failure, and not something to plan around before the words exist. Other errors stop.
async fn transcribe(generate: &PlannedStep, ex: &dyn Executor, report: &mut Report) -> Result<Digest, AutoCutError> {
    let started = std::time::Instant::now();
    match ex.execute(&generate.id, generate.params.clone()).await {
        Ok(_) => {}
        Err(e) if is_reply_timeout(&e) => loop {
            tokio::time::sleep(TRANSCRIBE_POLL).await;
            // the app may still be busy: a failed read is another poll, not an error
            if let Ok(d) = Digest::read(ex).await
                && planner::has_words(&d)
            {
                report
                    .notes
                    .push(format!("transcription outlasted the app's reply limit; waited for it to finish ({:.0}s in all)", started.elapsed().as_secs_f64()));
                return Ok(d);
            }
            if started.elapsed() >= TRANSCRIBE_WAIT {
                return Err(AutoCutError::Transcript(format!(
                    "transcription was still running after {} minutes; run the prompt again once the app shows a transcript",
                    TRANSCRIBE_WAIT.as_secs() / 60
                )));
            }
        },
        Err(e) => return Err(AutoCutError::Transcript(format!("transcription failed: {e}"))),
    }
    Digest::read(ex).await
}

/// Whether an executor error is a reply timeout (the app is still working), as the bridge
/// reports them: `app: timeout` (the control server's limit) or `bridge: timeout`.
fn is_reply_timeout(e: &str) -> bool {
    let e = e.to_ascii_lowercase();
    e.contains("timeout") || e.contains("timed out")
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

    /// Plays the app over the bridge: `transcript.generate` answers "app: timeout" (the control
    /// server's reply limit) while transcription goes on; the words appear a few polls later.
    #[derive(Clone)]
    struct SlowTranscriber {
        inner: SessionExecutor,
        polls: std::sync::Arc<std::sync::atomic::AtomicU32>,
        words_after: u32,
        generate_error: &'static str,
        transcript: serde_json::Value,
    }
    impl Executor for SlowTranscriber {
        fn execute(&self, id: &str, params: serde_json::Value) -> crate::ExecFuture {
            let me = self.clone();
            let id = id.to_string();
            Box::pin(async move {
                match id.as_str() {
                    // this build has no Whisper; the fake app does
                    "command.list" if params["filter"] == json!("transcript.generate") => Ok(json!([{"id": "transcript.generate", "enabled": true}])),
                    "transcript.generate" => Err(me.generate_error.to_string()),
                    "transcript.inspect" => {
                        let n = me.polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                        if n == me.words_after {
                            me.inner.execute("transcript.set", me.transcript.clone()).await?;
                        }
                        me.inner.execute(&id, params).await
                    }
                    _ => me.inner.execute(&id, params).await,
                }
            })
        }
    }

    async fn slow_transcriber(generate_error: &'static str) -> SlowTranscriber {
        let inner = demo();
        let seq = inner.execute("sequence.inspect", json!({})).await.unwrap();
        let clip = &seq["video"][0]["items"][0];
        let item = clip["item"].as_u64().unwrap();
        let base = clip["sourceIn"].as_f64().unwrap_or(0.0) / filmcraft_time::TICKS_PER_SECOND as f64;
        let tps = |s: f64| (s * filmcraft_time::TICKS_PER_SECOND as f64).round() as i64;
        // two sentences, far enough apart that a 1 s highlight keeps one and cuts the rest
        let words: Vec<serde_json::Value> = [(0.1, "Hello"), (0.5, "there."), (2.5, "Second"), (2.9, "sentence.")]
            .iter()
            .map(|(t, w)| json!({"text": w, "start": tps(base + t), "end": tps(base + t + 0.35), "speaker": 0}))
            .collect();
        let transcript = json!({"item": item, "transcript": {"language": "en", "speakers": [{"name": "Speaker 1"}], "words": words}});
        SlowTranscriber { inner, polls: Default::default(), words_after: 4, generate_error, transcript }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_transcription_timeout_waits_for_the_words() {
        // regression: the app's reply timeout was reported as "`transcript.generate` failed:
        // app: timeout; stopping" and planning raced the still-running transcription
        let ex = slow_transcriber("app: timeout").await;
        let r = run_with(&classified("highlight_reel", json!({"targetSeconds": 1})), "highlight reel", &ex, &cfg(), Options::execute()).await.unwrap();
        assert!(!r.notes.iter().any(|n| n.contains("stopping")), "{:?}", r.notes);
        assert!(r.notes.iter().any(|n| n.contains("waited for it to finish")), "{:?}", r.notes);
        assert!(ex.polls.load(std::sync::atomic::Ordering::SeqCst) >= 4, "it polled until the words came");
        assert!(r.steps.iter().any(|s| s.id == "sequence.extract"), "{:?}", r.steps);
        assert!(r.steps.iter().any(|s| s.id == "trim.applyDefaultTransition"), "cuts are smoothed: {:?}", r.steps);
        assert!(r.results.as_ref().is_some_and(|res| res.iter().all(|x| x.ok)), "{:?}", r.results);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_transcription_error_stops_with_the_reason() {
        let ex = slow_transcriber("app: the whisper model could not be loaded").await;
        let before = duration(&ex.inner).await;
        let err = run_with(&classified("highlight_reel", json!({})), "highlight reel", &ex, &cfg(), Options::execute()).await.unwrap_err();
        assert!(err.to_string().contains("could not be loaded"), "{err}");
        assert_eq!(duration(&ex.inner).await, before, "nothing was cut");
    }

    #[test]
    fn reply_timeouts_are_recognized() {
        for t in ["app: timeout", "bridge: timeout", "request timed out", "Timeout"] {
            assert!(is_reply_timeout(t), "{t}");
        }
        for f in ["app: no such command", "transcription failed", ""] {
            assert!(!is_reply_timeout(f), "{f}");
        }
    }
}
