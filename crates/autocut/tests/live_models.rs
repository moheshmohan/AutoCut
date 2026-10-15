//! Live-model checks, opt-in (they need Ollama running and models pulled). Run:
//!
//! ```sh
//! AUTOCUT_LIVE=1 cargo test -p filmcraft-autocut --test live_models -- --nocapture
//! # optional: AUTOCUT_LLM_MODEL=phi4-mini AUTOCUT_VLM_MODEL=moondream
//! ```

use filmcraft_autocut::SessionExecutor;
use filmcraft_autocut::config::Config;
use filmcraft_autocut::describe;
use filmcraft_autocut::intent;
use filmcraft_autocut::ollama::Ollama;
use filmcraft_engine::Session;
use serde_json::json;

fn live() -> bool {
    std::env::var("AUTOCUT_LIVE").ok().as_deref() == Some("1")
}

fn demo() -> SessionExecutor {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    SessionExecutor::from(s)
}

#[tokio::test(flavor = "multi_thread")]
async fn classifier_understands_prompts() {
    if !live() {
        return;
    }
    let cfg = Config::from_env();
    let m = Ollama::new(&cfg.ollama_host, &cfg.llm_model, cfg.http_timeout);
    for (prompt, want) in [
        ("remove all the silences please", "remove_silence"),
        ("add subtitles to the whole thing", "add_captions"),
        ("cut everything between 5 and 12 seconds", "cut_range"),
        ("make it a vertical tiktok", "export_video"),
        ("make it cinematic with sparkles wherever someone smiles", "custom"),
        ("clean up the audio, it sounds noisy", "enhance_voice"),
        ("export this for tiktok", "export_video"),
        ("add a voice over saying welcome to my channel", "add_voice_over"),
        ("smooth slow motion at half speed", "slow_motion"),
    ] {
        let c = intent::classify(&m, prompt).await.unwrap_or_else(|e| panic!("{prompt}: {e}"));
        assert_eq!(c.intent, want, "{prompt} -> {c:?}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn vlm_describes_a_frame() {
    if !live() {
        return;
    }
    let cfg = Config::from_env();
    let ex = demo();
    let vlm = Ollama::new(&cfg.ollama_host, &cfg.vlm_model, cfg.http_timeout);
    let frames = describe::describe_frames(&vlm, &ex, &[1.0], 448).await.unwrap();
    assert!(!frames.is_empty(), "the frame rendered and was described");
    eprintln!("frame at 1.0s: {}", frames[0].1);
    assert!(frames[0].1.len() > 10, "a real sentence: {}", frames[0].1);
}

#[tokio::test(flavor = "multi_thread")]
async fn complex_prompt_hands_off_with_the_document() {
    if !live() {
        return;
    }
    // no API: the supervisor is the MCP caller; the report comes back with the context
    let cfg = Config::from_env();
    let ex = demo();
    let runners = filmcraft_autocut::runner::Runners::from_config(&cfg);
    let r = filmcraft_autocut::runner::run_with(
        &runners,
        "add sparkle effects wherever someone smiles, make it cinematic",
        &ex,
        &cfg,
        filmcraft_autocut::runner::Options::plan_only(),
    )
    .await
    .unwrap();
    assert!(r.needs_supervisor, "{r:?}");
    assert!(r.document.is_some(), "{r:?}");
}
