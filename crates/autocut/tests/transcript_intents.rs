//! Transcript-driven intents against a real headless session: a scripted classifier plans, the
//! engine executes. No network — the transcript is imported with `transcript.set`.

use filmcraft_autocut::config::Config;
use filmcraft_autocut::llm::FixedModel;
use filmcraft_autocut::runner::{Runners, run_with};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use filmcraft_time::TICKS_PER_SECOND;
use serde_json::{Value, json};

fn tps(seconds: f64) -> i64 {
    (seconds * TICKS_PER_SECOND as f64).round() as i64
}

/// The demo project with a crafted transcript on the first video item: two ~1.4s sentences
/// separated by a 2-second pause, placed inside the clip's actual source range (the sequence
/// transcript only stitches words the clip plays).
fn demo_with_transcript() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let seq = s.execute("sequence.inspect", json!({})).unwrap();
    let clip = &seq["video"][0]["items"][0];
    let item = clip["item"].as_u64().expect("a video clip");
    let tick = |v: &Value| v.as_f64().unwrap_or(0.0) / TICKS_PER_SECOND as f64;
    let (s_in, s_out) = (tick(&clip["sourceIn"]), tick(&clip["sourceOut"]));
    assert!(s_out - s_in >= 4.96, "the demo clip is too short to craft a transcript into: {s_in}..{s_out}");
    let base = s_in + 0.05;
    let words: Vec<Value> =
        [(base, "Alpha"), (base + 0.5, "bravo"), (base + 1.0, "charlie."), (base + 3.45, "Delta"), (base + 3.95, "echo"), (base + 4.45, "foxtrot.")]
            .iter()
            .map(|(t, w)| json!({"text": w, "start": tps(*t), "end": tps(t + 0.45), "speaker": 0}))
            .collect();
    let set = s
        .execute("transcript.set", json!({"item": item, "transcript": {"language": "en", "speakers": [{"name": "Speaker 1"}], "words": words}}))
        .expect("transcript.set");
    assert!(set["words"].as_u64().unwrap_or(0) > 0, "{set}");
    SessionExecutor::from(s)
}

async fn duration(ex: &SessionExecutor) -> f64 {
    let ticks = ex.execute("sequence.inspect", json!({})).await.unwrap()["duration"].as_f64().unwrap_or(0.0);
    ticks / TICKS_PER_SECOND as f64
}

fn classified(intent: &str, params: Value) -> Runners {
    Runners { local: Box::new(FixedModel::new(vec![json!({"intent": intent, "confidence": 0.95, "params": params})])), vlm: None }
}

fn cfg() -> Config {
    Config::default()
}

#[tokio::test(flavor = "multi_thread")]
async fn remove_silence_cuts_the_pause() {
    let ex = demo_with_transcript();
    let before = duration(&ex).await;
    let r =
        run_with(&classified("remove_silence", json!({})), "remove the silences", &ex, &cfg(), filmcraft_autocut::runner::Options::execute()).await.unwrap();
    assert_eq!(r.steps.len(), 1, "{r:?}");
    assert_eq!(r.steps[0].id, "transcript.removePauses");
    let removed = r.results.as_ref().unwrap()[0].result.as_ref().unwrap()["removed"].as_u64().unwrap_or(0);
    assert_eq!(removed, 1, "the 2s pause between the sentences");
    let after = duration(&ex).await;
    let cut = before - after;
    assert!((cut - 1.85).abs() < 0.4, "kept ~0.15s of the 2s pause (frame-snapped): cut {cut:.2}s");
}

#[tokio::test(flavor = "multi_thread")]
async fn remove_fillers_cuts_the_filler_words() {
    let ex = demo_with_transcript();
    let r = run_with(&classified("remove_fillers", json!({})), "remove the ums", &ex, &cfg(), filmcraft_autocut::runner::Options::execute()).await.unwrap();
    assert_eq!(r.steps[0].id, "transcript.removeFillers");
    // the crafted transcript has no fillers, so the engine reports nothing removed
    let removed = r.results.as_ref().unwrap()[0].result.as_ref().unwrap()["removed"].as_u64().unwrap_or(0);
    assert_eq!(removed, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn add_captions_creates_a_caption_track() {
    let ex = demo_with_transcript();
    let r = run_with(&classified("add_captions", json!({})), "add captions", &ex, &cfg(), filmcraft_autocut::runner::Options::execute()).await.unwrap();
    assert_eq!(r.steps[0].id, "transcript.createCaptions");
    let result = r.results.as_ref().unwrap()[0].result.as_ref().cloned().unwrap_or_default();
    assert!(result["captions"].as_u64().unwrap_or(0) >= 1, "caption blocks created: {result}");
    assert!(result["track"].as_u64().is_some(), "a caption track id: {result}");
}

#[tokio::test(flavor = "multi_thread")]
async fn highlight_reel_keeps_the_best_segment() {
    let ex = demo_with_transcript();
    // ~2.7s of speech in two sentences; a 1-second target keeps one and cuts the other
    let r =
        run_with(&classified("highlight_reel", json!({"targetSeconds": 1.0})), "highlight reel", &ex, &cfg(), filmcraft_autocut::runner::Options::execute())
            .await
            .unwrap();
    assert!(r.steps.iter().all(|s| s.id == "transcript.extract"), "{r:?}");
    let results = r.results.as_ref().unwrap();
    assert!(!results.is_empty() && results[0].ok, "the extract ran: {r:?}");
    let after = ex.execute("transcript.inspect", json!({})).await.unwrap();
    let left = after["words"].as_array().map(Vec::len).unwrap_or(0);
    assert_eq!(left, 3, "the kept sentence's words survive: {left}");
}
