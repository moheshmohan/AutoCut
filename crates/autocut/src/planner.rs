//! Deterministic planning: a classified intent plus the document becomes engine commands.
//!
//! Every handler emits ordinary, undoable commands — the same ones a human's menu clicks run —
//! and never invents ids: [`crate::validate`] checks each step against the registry before
//! anything executes.

use serde_json::{Value, json};

use crate::digest::Digest;
use crate::highlight;
use crate::{AutoCutError, Executor, PlannedStep};

/// What planning produced.
pub enum Plan {
    /// A batch ready to run (or show, in a dry run).
    Steps { steps: Vec<PlannedStep>, notes: Vec<String> },
    /// The intent needs a transcript first: run `generate`, then plan again.
    GenerateThen { generate: PlannedStep, params: Value },
}

/// Plan a classified intent (never `custom`).
pub async fn plan(intent: &str, params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    match intent {
        "remove_silence" => remove_silence(params, digest, ex).await,
        "remove_fillers" => remove_fillers(params, digest, ex).await,
        "add_captions" => add_captions(params, digest, ex).await,
        "cut_range" => cut_range(params, digest),
        "keep_range" => keep_range(params, digest),
        "highlight_reel" => highlight_reel(params, digest, ex).await,
        other => Err(AutoCutError::Invalid(format!("`{other}` is not a locally planned intent"))),
    }
}

/// Whether the sequence already has stitched transcript words.
fn has_words(digest: &Digest) -> bool {
    digest.transcript.as_ref().and_then(|t| t["words"].as_array()).is_some_and(|w| !w.is_empty())
}

/// A transcript-driven intent: prepend `transcript.generate` when there are no words yet, and
/// refuse (with a fix) when this build cannot transcribe at all.
async fn with_transcript(digest: &Digest, ex: &dyn Executor, steps: Vec<PlannedStep>) -> Result<Plan, AutoCutError> {
    if has_words(digest) {
        return Ok(Plan::Steps { steps, notes: Vec::new() });
    }
    ensure_can_transcribe(ex).await?;
    let mut out = vec![PlannedStep::new("transcript.generate", json!({}), "transcribe the audio first")];
    out.extend(steps);
    Ok(Plan::Steps { steps: out, notes: Vec::new() })
}

async fn ensure_can_transcribe(ex: &dyn Executor) -> Result<(), AutoCutError> {
    let enabled = ex
        .execute("command.list", json!({"filter": "transcript.generate"}))
        .await
        .ok()
        .and_then(|v| v.as_array().and_then(|a| a.first().cloned()))
        .and_then(|c| c["enabled"].as_bool())
        .unwrap_or(false);
    if enabled {
        return Ok(());
    }
    Err(AutoCutError::Transcript(
        "this sequence has no transcript, and this build cannot transcribe (build with the `whisper` feature, or import one with transcript.set)".into(),
    ))
}

async fn remove_silence(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let mut p = json!({});
    if let Some(min) = params["minSeconds"].as_f64() {
        p["minSeconds"] = json!(min);
    }
    if let Some(keep) = params["keepSeconds"].as_f64() {
        p["keepSeconds"] = json!(keep);
    }
    with_transcript(digest, ex, vec![PlannedStep::new("transcript.removePauses", p, "cut the pauses between speech")]).await
}

async fn remove_fillers(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let mut p = json!({});
    if let Some(f) = params["fillers"].as_array().filter(|a| !a.is_empty()) {
        p["fillers"] = json!(f);
    }
    with_transcript(digest, ex, vec![PlannedStep::new("transcript.removeFillers", p, "cut the filler words")]).await
}

async fn add_captions(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let mut p = json!({});
    if let Some(n) = params["maxChars"].as_u64() {
        p["maxChars"] = json!(n);
    }
    if let Some(l) = params["lines"].as_u64() {
        p["lines"] = json!(l);
    }
    with_transcript(digest, ex, vec![PlannedStep::new("transcript.createCaptions", p, "caption track from the transcript")]).await
}

/// `markers.markIn` + `markers.markOut` + `sequence.extract`/`lift` + `markers.clearInOut`.
fn mark_and_remove(start: f64, end: f64, mode: &str, why: &str) -> Vec<PlannedStep> {
    vec![
        PlannedStep::new("markers.markIn", json!({"seconds": start}), "mark the range start"),
        PlannedStep::new("markers.markOut", json!({"seconds": end}), "mark the range end"),
        PlannedStep::new(if mode == "lift" { "sequence.lift" } else { "sequence.extract" }, json!({}), why),
        PlannedStep::new("markers.clearInOut", json!({}), "clear the marks"),
    ]
}

fn range_params(params: &Value) -> Result<(f64, f64), AutoCutError> {
    let (start, end) = (params["start"].as_f64(), params["end"].as_f64());
    let (Some(start), Some(end)) = (start, end) else {
        return Err(AutoCutError::Invalid("this intent needs `start` and `end` seconds".into()));
    };
    if !(start.is_finite() && end.is_finite() && start >= 0.0 && end > start) {
        return Err(AutoCutError::Invalid(format!("the range {start}..{end}s makes no sense")));
    }
    Ok((start, end))
}

fn check_within(end: f64, digest: &Digest) -> Result<(), AutoCutError> {
    let duration = digest.duration_seconds();
    if duration > 0.0 && end > duration + 0.001 {
        return Err(AutoCutError::Invalid(format!("the range ends at {end}s but the sequence is only {duration:.1}s")));
    }
    Ok(())
}

fn cut_range(params: &Value, digest: &Digest) -> Result<Plan, AutoCutError> {
    let (start, end) = range_params(params)?;
    let mode = params["mode"].as_str().unwrap_or("extract");
    if mode != "extract" && mode != "lift" {
        return Err(AutoCutError::Invalid(format!("mode `{mode}` is not extract or lift")));
    }
    check_within(end, digest)?;
    Ok(Plan::Steps { steps: mark_and_remove(start, end, mode, "remove the marked range"), notes: Vec::new() })
}

/// Ripple-extract the head and tail around the range to keep. The tail goes first: removing it
/// does not shift the head's absolute time.
fn keep_range(params: &Value, digest: &Digest) -> Result<Plan, AutoCutError> {
    let (start, end) = range_params(params)?;
    let duration = digest.duration_seconds();
    if duration <= 0.0 {
        return Err(AutoCutError::Invalid("the sequence duration is unknown".into()));
    }
    if end > duration + 0.001 {
        return Err(AutoCutError::Invalid(format!("the range ends at {end}s but the sequence is only {duration:.1}s")));
    }
    let mut steps = Vec::new();
    if end < duration - 0.001 {
        steps.extend(mark_and_remove(end, duration, "extract", "cut the tail after the kept range"));
    }
    if start > 0.001 {
        steps.extend(mark_and_remove(0.0, start, "extract", "cut the head before the kept range"));
    }
    if steps.is_empty() {
        return Ok(Plan::Steps { steps: Vec::new(), notes: vec!["the range covers the whole sequence; nothing to cut".into()] });
    }
    Ok(Plan::Steps { steps, notes: Vec::new() })
}

async fn highlight_reel(params: &Value, digest: &Digest, ex: &dyn Executor) -> Result<Plan, AutoCutError> {
    let Some(words) = digest.transcript.as_ref().and_then(highlight::parse_words) else {
        ensure_can_transcribe(ex).await?;
        return Ok(Plan::GenerateThen { generate: PlannedStep::new("transcript.generate", json!({}), "transcribe the audio first"), params: params.clone() });
    };
    let target = params["targetSeconds"].as_f64().unwrap_or(60.0).clamp(1.0, 3600.0);
    let topic = params["topic"].as_str().map(str::to_string);
    let (keep, removals, notes) = highlight::plan(&words, target, topic.as_deref());
    if removals.is_empty() {
        return Ok(Plan::Steps {
            steps: Vec::new(),
            notes: notes.into_iter().chain(std::iter::once("nothing to cut: everything is worth keeping".into())).collect(),
        });
    }
    let n = removals.len();
    let steps = removals
        .into_iter()
        .map(|(from, to)| PlannedStep::new("transcript.extract", json!({"from": from, "to": to}), "cut a stretch that is not part of the highlight reel"))
        .collect();
    let mut out = vec![format!("kept {keep} stretch(es) totalling about {target:.0}s of speech; cutting everything else in {n} cut(s)")];
    out.extend(notes);
    Ok(Plan::Steps { steps, notes: out })
}
