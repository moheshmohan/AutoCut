//! AutoCut: prompt-driven automated editing on top of the FilmCraft engine.
//!
//! The pipeline ([`runner`]): a local small LLM ([`ollama`]) classifies the user's prompt into an
//! [`intent::Intent`] with parameters; deterministic [`planner`] handlers turn routine intents into
//! batches of ordinary engine commands (`transcript.removePauses`, `sequence.extract`, …); anything
//! complex, or low-confidence, escalates to the cloud supervisor ([`anthropic`], Claude), whose
//! output is [`validate`]d against the real command registry before anything runs. Every step
//! executes through the [`Executor`] trait, so the same code drives a headless `Session`, the CLI
//! and the live app over the MCP bridge.
//!
//! Nothing here calls the engine directly: all reads and writes are commands, so AutoCut edits are
//! ordinary undoable operations like a human's.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod anthropic;
pub mod config;
pub mod digest;
pub mod highlight;
pub mod intent;
pub mod llm;
pub mod ollama;
pub mod planner;
pub mod runner;
pub mod supervisor;
pub mod validate;

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_engine::Session;
use serde::Serialize;
use serde_json::Value;

/// A command execution against whatever target AutoCut is attached to: a headless
/// [`Session`] ([`SessionExecutor`]), the CLI's backend, or the live app through the MCP bridge.
pub trait Executor: Send + Sync {
    /// Run one engine command by id; the future resolves to its result value.
    fn execute(&self, id: &str, params: Value) -> ExecFuture;
}

/// The future [`Executor::execute`] returns.
pub type ExecFuture = Pin<Box<dyn Future<Output = Result<Value, String>> + Send + 'static>>;

/// [`Executor`] over a shared headless session (the same one the MCP server and the CLI use).
#[derive(Clone)]
pub struct SessionExecutor(pub Arc<Mutex<Session>>);

impl From<Session> for SessionExecutor {
    fn from(s: Session) -> Self {
        Self(Arc::new(Mutex::new(s)))
    }
}

impl Executor for SessionExecutor {
    fn execute(&self, id: &str, params: Value) -> ExecFuture {
        let s = self.0.clone();
        let id = id.to_string();
        Box::pin(async move {
            // blocking engine work off the async thread, as the MCP server does
            tokio::task::spawn_blocking(move || {
                // a command that panicked earlier poisoned the lock; the session is still usable
                let mut g = s.lock().unwrap_or_else(PoisonError::into_inner);
                g.execute(&id, params).map_err(|e| e.to_string())
            })
            .await
            .map_err(|e| format!("internal error: {e}"))?
        })
    }
}

/// One planned engine command: what to run and why (the `why` is shown in dry runs and reports).
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PlannedStep {
    /// Command id, e.g. `transcript.removePauses`.
    pub id: String,
    /// Parameters object for the command.
    #[serde(default = "empty_object")]
    pub params: Value,
    /// Human-readable reason, one line.
    pub why: String,
}

fn empty_object() -> Value {
    Value::Object(serde_json::Map::new())
}

/// PNG bytes as base64, for vision-model messages.
pub fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

impl PlannedStep {
    pub fn new(id: &str, params: Value, why: impl Into<String>) -> Self {
        Self { id: id.to_string(), params, why: why.into() }
    }
}

/// The outcome of one executed step.
#[derive(Debug, Clone, Serialize)]
pub struct StepResult {
    pub id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Everything AutoCut did (or plans to do) with one prompt.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub prompt: String,
    /// Classified intent id, when the local planner understood the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Local classifier confidence, when classified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Who wrote the steps: the local planner, the cloud supervisor, or local reviewed by cloud.
    pub source: &'static str,
    pub steps: Vec<PlannedStep>,
    /// Set when the run was executed (not a dry run): each step's outcome.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<StepResult>>,
    /// How many undoable edits were made (run `edit.undo` this many times to revert).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub undo: Option<usize>,
    /// Anything worth telling the user (escalations, skipped parts, advice).
    pub notes: Vec<String>,
}

/// AutoCut failures, all human-readable.
#[derive(Debug, thiserror::Error)]
pub enum AutoCutError {
    #[error("{0}")]
    Llm(String),
    #[error("{0}")]
    Executor(String),
    #[error("no sequence is open; open or create one first")]
    NoSequence,
    #[error("transcript: {0}")]
    Transcript(String),
    #[error("invalid plan: {0}")]
    Invalid(String),
    #[error("{0}")]
    Other(String),
}

/// The first `{`-balanced JSON object in `s` (models wrap JSON in prose or code fences despite
/// being asked not to); the whole string when it already parses.
pub fn extract_json(s: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(s.trim()) {
        return Some(v);
    }
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, b) in s.as_bytes()[start..].iter().enumerate() {
        let c = *b;
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return serde_json::from_str::<Value>(&s[start..start + i + 1]).ok();
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_json_from_prose_and_fences() {
        assert_eq!(extract_json(r#"{"a":1}"#), Some(json!({"a": 1})));
        assert_eq!(extract_json("Sure! Here it is:\n```json\n{\"a\": {\"b\": 2}}\n```"), Some(json!({"a": {"b": 2}})));
        assert_eq!(extract_json(r#"the plan is {"a":"with } brace","b":3} hope it helps"#), Some(json!({"a": "with } brace", "b": 3})));
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json(r#"{"unclosed": 1"#), None);
    }

    #[tokio::test]
    async fn session_executor_runs_commands() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", serde_json::json!({})).unwrap();
        let ex = SessionExecutor::from(s);
        let v = ex.execute("sequence.inspect", serde_json::json!({})).await.unwrap();
        assert!(v.get("video").is_some(), "{v}");
        let e = ex.execute("no.such.command", serde_json::json!({})).await.unwrap_err();
        assert!(e.contains("no.such.command"), "{e}");
    }
}
