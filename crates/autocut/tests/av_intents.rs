//! The voice / narration / export / speed intents against a real headless session: a scripted
//! classifier plans, the engine executes. No network — the built-in TTS voice and CPU exporters
//! are used.

use filmcraft_autocut::config::Config;
use filmcraft_autocut::runner::{Options, Runners, run_with};
use filmcraft_autocut::{Executor as _, SessionExecutor};
use filmcraft_engine::Session;
use serde_json::{Value, json};

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
}

fn classified(intent: &str, params: Value) -> Runners {
    Runners { local: Box::new(filmcraft_autocut::llm::FixedModel::new(vec![json!({"intent": intent, "confidence": 0.95, "params": params})])), vlm: None }
}

async fn run(ex: &SessionExecutor, intent: &str, params: Value) -> filmcraft_autocut::Report {
    run_with(&classified(intent, params), intent, ex, &Config::default(), Options::execute()).await.expect("the run succeeds")
}

fn first_ok(report: &filmcraft_autocut::Report, id: &str) -> Value {
    report.results.as_ref().unwrap().iter().find(|r| r.id == id).and_then(|r| r.result.clone()).unwrap_or_else(|| panic!("`{id}` ran: {:?}", report.results))
}

#[tokio::test(flavor = "multi_thread")]
async fn enhance_voice_treats_every_audio_clip_as_dialogue() {
    let ex = demo();
    let r = run(&ex, "enhance_voice", json!({})).await;
    let ids: Vec<String> = r.steps.iter().map(|s| s.id.clone()).collect();
    assert!(ids.contains(&"essentialSound.setType".to_string()), "{ids:?}");
    assert!(ids.contains(&"essentialSound.applyPreset".to_string()), "{ids:?}");
    assert!(ids.contains(&"essentialSound.autoMatch".to_string()), "{ids:?}");
    let set = first_ok(&r, "essentialSound.setType");
    assert!(set.is_null() || set.as_object().is_some(), "{set}");
    // headless has no selection, so inspect the same explicit clips the planner targeted
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let clips: Vec<u64> = seq["audio"].as_array().unwrap().iter().flat_map(|t| t["items"].as_array().unwrap()).filter_map(|c| c["clip"].as_u64()).collect();
    let inspect = ex.execute("essentialSound.inspect", json!({"clips": clips})).await.unwrap();
    let typed = inspect["clips"].as_array().unwrap().iter().filter(|c| c["type"].as_str().is_some_and(|t| t.eq_ignore_ascii_case("dialogue"))).count();
    assert_eq!(typed, clips.len(), "every audio clip is dialogue");
}

#[tokio::test(flavor = "multi_thread")]
async fn voice_over_narrates_the_script() {
    let ex = demo();
    let r = run(&ex, "add_voice_over", json!({"text": "Welcome to AutoCut."})).await;
    assert_eq!(r.steps[0].id, "tts.create");
    assert_eq!(r.steps[0].params["text"], json!("Welcome to AutoCut."));
    let created = first_ok(&r, "tts.create");
    assert!(created["clip"].as_u64().is_some(), "a narration clip: {created}");
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let narrating = seq["audio"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|t| t["items"].as_array().unwrap())
        .filter(|c| c["name"].as_str().is_some_and(|n| n.starts_with("Narration")))
        .count();
    assert_eq!(narrating, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn voice_over_fit_paces_the_delivery() {
    let ex = demo();
    // fit: the preview measures the speech, the pace covers the ~27s demo sequence
    let r = run(&ex, "add_voice_over", json!({"text": "A short line.", "fit": true})).await;
    let pace = r.steps[0].params["pace"].as_f64();
    let Some(pace) = pace else {
        panic!("fit sets a pace: {:?}", r.steps[0]);
    };
    assert!((0.5..=2.0).contains(&pace), "{pace}");
}

#[tokio::test(flavor = "multi_thread")]
async fn export_video_writes_the_file() {
    let ex = demo();
    let out = std::env::temp_dir().join(format!("autocut-export-{}.wav", std::process::id()));
    let r = run(&ex, "export_video", json!({"destination": "audio", "path": out.to_string_lossy()})).await;
    assert_eq!(r.steps[0].id, "file.exportMedia");
    assert_eq!(r.steps[0].params["preset"], json!("Waveform Audio 48 kHz 24-bit"));
    let exported = first_ok(&r, "file.exportMedia");
    assert!(exported["path"].as_str().unwrap_or("").ends_with(".wav"), "{exported}");
    assert!(out.exists(), "the file was written");
    let _ = std::fs::remove_file(&out);
}

#[tokio::test(flavor = "multi_thread")]
async fn tiktok_export_targets_the_vertical_preset() {
    let ex = demo();
    let r = run(&ex, "export_video", json!({"destination": "tiktok"})).await;
    assert_eq!(r.steps[0].params["preset"], json!("Social Vertical 1080×1920"), "{:?}", r.steps[0]);
    assert_eq!(r.steps[0].id, "file.exportMedia");
}

#[tokio::test(flavor = "multi_thread")]
async fn slow_motion_with_optical_flow_retimes_and_renders() {
    let ex = demo();
    // the first video clip only, so the render is quick
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let clip = seq["video"][0]["items"][0]["clip"].as_u64().unwrap();
    let r = run(&ex, "slow_motion", json!({"factor": 0.5, "clips": [clip]})).await;
    assert_eq!(r.steps[0].id, "clip.speedDuration");
    assert_eq!(r.steps[0].params["speed"], json!(50.0));
    assert_eq!(r.steps[0].params["interpolation"], json!("opticalFlow"));
    assert!(first_ok(&r, "clip.speedDuration").is_null());
    // the retimed clip renders — this exercises the engine's optical-flow interpolation
    let png = ex.render(2.0, 320).await;
    let png = png.expect("a frame at 2s of the slowed clip");
    assert!(png.starts_with(b"\x89PNG") && png.len() > 1_000);
    // and the clip itself got longer: half speed doubles it (the music bed still caps the
    // sequence, so check the clip, not the timeline)
    let after = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let slowed = after["video"][0]["items"][0]["duration"].as_f64().unwrap_or(0.0) / filmcraft_time::TICKS_PER_SECOND as f64;
    let before = seq["video"][0]["items"][0]["duration"].as_f64().unwrap_or(0.0) / filmcraft_time::TICKS_PER_SECOND as f64;
    assert!((slowed / before - 2.0).abs() < 0.05, "half speed doubles the clip: {before:.1}s -> {slowed:.1}s");
}

#[tokio::test(flavor = "multi_thread")]
async fn caption_styles_apply_to_the_new_track() {
    let ex = demo();
    // a transcript is needed; import one on the first video item (as in transcript_intents.rs)
    let seq = ex.execute("sequence.inspect", json!({})).await.unwrap();
    let clip = &seq["video"][0]["items"][0];
    let item = clip["item"].as_u64().unwrap();
    let tick = |v: &Value| v.as_f64().unwrap_or(0.0) / filmcraft_time::TICKS_PER_SECOND as f64;
    let base = tick(&clip["sourceIn"]) + 0.05;
    let tps = |s: f64| (s * filmcraft_time::TICKS_PER_SECOND as f64).round() as i64;
    let words: Vec<Value> = [(base, "Alpha"), (base + 0.5, "bravo"), (base + 1.0, "charlie.")]
        .iter()
        .map(|(t, w)| json!({"text": w, "start": tps(*t), "end": tps(t + 0.45), "speaker": 0}))
        .collect();
    ex.execute("transcript.set", json!({"item": item, "transcript": {"language": "en", "speakers": [{"name": "Speaker 1"}], "words": words}})).await.unwrap();
    let r = run(&ex, "add_captions", json!({"style": "tiktok"})).await;
    let ids: Vec<&str> = r.steps.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, ["transcript.createCaptions", "captions.setStyle"], "{ids:?}");
    assert_eq!(r.steps[1].params["size"].as_f64(), Some(84.0));
    let styled = ex.execute("captions.list", json!({})).await.unwrap();
    let size = styled["tracks"].as_array().and_then(|t| t.first()).and_then(|t| t["style"]["size"].as_f64()).unwrap_or(0.0);
    assert!((size - 84.0).abs() < 1e-6, "the style landed: {size}");
}
