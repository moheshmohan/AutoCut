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

/// What a [`FnModel`] answers with: `(system, user, images)` → the reply.
pub type Answer = dyn Fn(&str, &str, &[Vec<u8>]) -> Result<Value, AutoCutError> + Send + Sync;

/// A model that answers through a function of what it is asked — for tests and fakes where the
/// order of requests is not fixed (a clip analysis asks to locate, read, describe and compare in
/// an order that depends on the media).
pub struct FnModel {
    pub name: String,
    answer: Box<Answer>,
}

impl FnModel {
    pub fn new(name: &str, answer: impl Fn(&str, &str, &[Vec<u8>]) -> Result<Value, AutoCutError> + Send + Sync + 'static) -> Self {
        Self { name: name.to_string(), answer: Box::new(answer) }
    }
}

impl ChatModel for FnModel {
    fn id(&self) -> String {
        self.name.clone()
    }

    fn complete_json(&self, system: &str, user: &str, images: &[Vec<u8>]) -> JsonFuture {
        let answer = (self.answer)(system, user, images);
        Box::pin(async move { answer })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn fn_model_answers_by_prompt() {
        let m = FnModel::new("fake", |system, user, images| Ok(json!({"system": system, "user": user, "images": images.len()})));
        assert_eq!(m.id(), "fake");
        assert_eq!(m.complete_json("s", "u", &[vec![1]]).await.unwrap(), json!({"system": "s", "user": "u", "images": 1}));
    }

    #[tokio::test]
    async fn fixed_model_scripts_then_errors() {
        let m = FixedModel::new(vec![json!({"intent": "custom"})]);
        assert_eq!(m.id(), "fixed");
        assert_eq!(m.complete_json("s", "u", &[]).await.unwrap(), json!({"intent": "custom"}));
        assert!(m.complete_json("s", "u", &[]).await.is_err());
        assert_eq!(m.asked.lock().unwrap().len(), 2);
    }
}
