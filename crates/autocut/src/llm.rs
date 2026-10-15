//! The model interface AutoCut plans with: any chat model that answers JSON, local or cloud.

use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;

use serde_json::Value;

use crate::AutoCutError;

/// The future [`ChatModel::complete_json`] returns. Implementations do their (blocking) HTTP on
/// a blocking thread, so the future is cheap to await from any runtime.
pub type JsonFuture = Pin<Box<dyn Future<Output = Result<Value, AutoCutError>> + Send + 'static>>;

/// A chat model that answers with a JSON object. [`crate::ollama`] and [`crate::anthropic`]
/// implement it; tests plug in [`FixedModel`].
pub trait ChatModel: Send + Sync {
    /// A short id for reports and errors ("ollama:phi4-mini", "anthropic:claude-sonnet-4-5").
    fn id(&self) -> String;
    /// Ask for a JSON object. `images` are PNG bytes shown to vision models (empty for text-only).
    fn complete_json(&self, system: &str, user: &str, images: &[Vec<u8>]) -> JsonFuture;
}

/// A scripted model for tests: returns queued answers, then errors. Records what it was asked.
pub struct FixedModel {
    pub name: String,
    /// (system, user) per call, appended on every `complete_json`.
    pub asked: Mutex<Vec<(String, String)>>,
    answers: Mutex<std::collections::VecDeque<Value>>,
}

impl FixedModel {
    pub fn new(answers: Vec<Value>) -> Self {
        Self { name: "fixed".into(), asked: Mutex::new(Vec::new()), answers: Mutex::new(answers.into()) }
    }

    fn next(&self, system: &str, user: &str) -> Result<Value, AutoCutError> {
        self.asked.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((system.to_string(), user.to_string()));
        self.answers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .ok_or_else(|| AutoCutError::Llm("FixedModel is out of answers".into()))
    }
}

impl ChatModel for FixedModel {
    fn id(&self) -> String {
        self.name.clone()
    }

    fn complete_json(&self, system: &str, user: &str, _images: &[Vec<u8>]) -> JsonFuture {
        let answer = self.next(system, user);
        Box::pin(async move { answer })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn fixed_model_scripts_then_errors() {
        let m = FixedModel::new(vec![json!({"intent": "custom"})]);
        assert_eq!(m.id(), "fixed");
        assert_eq!(m.complete_json("s", "u", &[]).await.unwrap(), json!({"intent": "custom"}));
        assert!(m.complete_json("s", "u", &[]).await.is_err());
        assert_eq!(m.asked.lock().unwrap().len(), 2);
    }
}
