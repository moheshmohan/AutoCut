//! AutoCut configuration: which local models, which cloud supervisor, and when to escalate.

use std::time::Duration;

/// When the cloud supervisor (Anthropic) takes part.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Policy {
    /// Never call the cloud; fail instead when the local planner can't handle a prompt.
    Never,
    /// Escalate to the supervisor when classification is `custom`, low-confidence, or the local
    /// LLM is unreachable. The default.
    #[default]
    ComplexOnly,
    /// Like `ComplexOnly`, but every local plan also gets a supervisor review pass.
    AlwaysReview,
}

impl Policy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "never" => Some(Self::Never),
            "complex-only" | "complexonly" | "complex" => Some(Self::ComplexOnly),
            "always-review" | "alwaysreview" | "always" => Some(Self::AlwaysReview),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Never => "never",
            Self::ComplexOnly => "complex-only",
            Self::AlwaysReview => "always-review",
        }
    }
}

/// Everything AutoCut needs to know to reach its models.
#[derive(Debug, Clone)]
pub struct Config {
    /// Ollama base URL (`AUTOCUT_OLLAMA_HOST`).
    pub ollama_host: String,
    /// Ollama model used to classify prompts and score highlights (`AUTOCUT_LLM_MODEL`).
    pub llm_model: String,
    /// Ollama vision model for frame descriptions (`AUTOCUT_VLM_MODEL`).
    pub vlm_model: String,
    /// Anthropic API key; absent = the supervisor is off (`AUTOCUT_ANTHROPIC_KEY` or
    /// `ANTHROPIC_API_KEY`).
    pub anthropic_key: Option<String>,
    /// Supervisor model (`AUTOCUT_SUPERVISOR_MODEL`).
    pub supervisor_model: String,
    /// When to escalate to the supervisor (`AUTOCUT_POLICY`).
    pub policy: Policy,
    /// Local classifications below this confidence escalate (0..=1).
    pub confidence_threshold: f32,
    /// Per-request HTTP timeout for model calls; LLMs are slow, so this is minutes.
    pub http_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_host: "http://127.0.0.1:11434".into(),
            llm_model: "phi4-mini".into(),
            vlm_model: "qwen2.5vl:7b".into(),
            anthropic_key: None,
            supervisor_model: "claude-sonnet-4-5".into(),
            policy: Policy::default(),
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
        if let Some(v) = env("AUTOCUT_SUPERVISOR_MODEL") {
            c.supervisor_model = v;
        }
        if let Some(p) = env("AUTOCUT_POLICY").as_deref().and_then(Policy::parse) {
            c.policy = p;
        }
        if let Some(v) = env("AUTOCUT_CONFIDENCE_THRESHOLD").and_then(|v| v.parse::<f32>().ok()) {
            c.confidence_threshold = v.clamp(0.0, 1.0);
        }
        c.anthropic_key = env("AUTOCUT_ANTHROPIC_KEY").or_else(|| env("ANTHROPIC_API_KEY")).filter(|k| !k.trim().is_empty());
        c
    }

    /// Whether the supervisor can run (key present and policy allows cloud calls).
    pub fn supervisor_available(&self) -> bool {
        self.anthropic_key.is_some() && self.policy != Policy::Never
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
    fn policy_parses_and_prints() {
        for (s, p) in [("never", Policy::Never), ("complex-only", Policy::ComplexOnly), ("Always-Review", Policy::AlwaysReview)] {
            assert_eq!(Policy::parse(s), Some(p));
            assert_eq!(p.as_str(), s.to_ascii_lowercase());
        }
        assert_eq!(Policy::parse("sometimes"), None);
        assert_eq!(Policy::default(), Policy::ComplexOnly);
    }

    #[test]
    fn supervisor_needs_a_key_and_a_policy() {
        let mut c = Config::default();
        assert!(!c.supervisor_available());
        c.anthropic_key = Some("k".into());
        assert!(c.supervisor_available());
        c.policy = Policy::Never;
        assert!(!c.supervisor_available());
    }
}
