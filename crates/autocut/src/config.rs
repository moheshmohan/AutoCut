//! AutoCut configuration: which local models to use, and when to hand a prompt to the
//! supervisor (the MCP client — e.g. Claude — that is driving the editor).

use std::time::Duration;

/// Everything AutoCut needs to know to reach its local models.
#[derive(Debug, Clone)]
pub struct Config {
    /// Ollama base URL (`AUTOCUT_OLLAMA_HOST`).
    pub ollama_host: String,
    /// Ollama model used to classify prompts (`AUTOCUT_LLM_MODEL`).
    pub llm_model: String,
    /// Ollama vision model for frame descriptions (`AUTOCUT_VLM_MODEL`); moondream is small
    /// and captions reliably, heavier VLMs work too.
    pub vlm_model: String,
    /// Ollama vision model for clip analysis (`AUTOCUT_ANALYSIS_MODEL`): it must ground objects
    /// (boxes), so a Qwen2.5-VL class model; the same one the screen agent uses.
    pub analysis_model: String,
    /// Ollama model that writes voice-overs (`AUTOCUT_SCRIPT_MODEL`): prose for people to hear,
    /// so a larger model than the classifier; callers fall back to `llm_model` when it is not
    /// installed.
    pub script_model: String,
    /// Local classifications below this confidence hand off to the supervisor (0..=1,
    /// `AUTOCUT_CONFIDENCE_THRESHOLD`).
    pub confidence_threshold: f32,
    /// Per-request HTTP timeout for model calls; LLMs are slow, so this is minutes.
    pub http_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_host: "http://127.0.0.1:11434".into(),
            llm_model: "phi4-mini".into(),
            vlm_model: "moondream".into(),
            analysis_model: "qwen2.5vl:7b".into(),
            script_model: "qwen3:8b".into(),
            confidence_threshold: 0.6,
            http_timeout: Duration::from_secs(180),
        }
    }
}

impl Config {
    /// Config from the environment, falling back to defaults per field.
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Some(v) = env("AUTOCUT_OLLAMA_HOST") {
            c.ollama_host = v;
        }
        if let Some(v) = env("AUTOCUT_LLM_MODEL") {
            c.llm_model = v;
        }
        if let Some(v) = env("AUTOCUT_VLM_MODEL") {
            c.vlm_model = v;
        }
        if let Some(v) = env("AUTOCUT_ANALYSIS_MODEL") {
            c.analysis_model = v;
        }
        if let Some(v) = env("AUTOCUT_SCRIPT_MODEL") {
            c.script_model = v;
        }
        if let Some(v) = env("AUTOCUT_CONFIDENCE_THRESHOLD").and_then(|v| v.parse::<f32>().ok()) {
            c.confidence_threshold = v.clamp(0.0, 1.0);
        }
        c
    }
}

/// A set, non-empty environment variable.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_local() {
        let c = Config::default();
        assert!(c.ollama_host.starts_with("http://127.0.0.1"));
        assert_eq!(c.llm_model, "phi4-mini");
        assert_eq!(c.script_model, "qwen3:8b");
        assert!((c.confidence_threshold - 0.6).abs() < 1e-9);
    }
}
