//! Live-model checks, opt-in (they need Ollama running and models pulled; the supervisor check
//! also needs ANTHROPIC_API_KEY and spends a little credit). Run:
//!
//! ```sh
//! AUTOCUT_LIVE=1 cargo test -p filmcraft-autocut --test live_models -- --nocapture
//! # optional: AUTOCUT_LLM_MODEL=phi4-mini AUTOCUT_VLM_MODEL=moondream
//! # optional: AUTOCUT_SUPERVISOR_LIVE=1 with ANTHROPIC_API_KEY set
//! ```

use filmcraft_autocut::SessionExecutor;
use filmcraft_autocut::config::Config;
use filmcraft_autocut::describe;
use filmcraft_autocut::intent;
use filmcraft_autocut::llm::ChatModel as _;
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
        ("make it a vertical tiktok", "custom"),
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
async fn supervisor_plans_a_complex_prompt() {
    if std::env::var("AUTOCUT_SUPERVISOR_LIVE").ok().as_deref() != Some("1") {
        return;
    }
    let cfg = Config::from_env();
    let Some(key) = cfg.anthropic_key.clone() else { panic!("ANTHROPIC_API_KEY not set") };
    let sup = filmcraft_autocut::anthropic::Anthropic::new(&cfg.supervisor_model, &key);
    let v = sup.complete_json("Answer with {\"ok\": true} as JSON.", "Say ok.", &[]).await.unwrap();
    assert_eq!(v["ok"], json!(true), "the supervisor answers: {v}");
}
